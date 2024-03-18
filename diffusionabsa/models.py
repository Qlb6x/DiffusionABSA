from collections import namedtuple
import copy
import math
import random
import torch
from torch import nn as nn
from torch.nn import functional as F
# from modeling_bert import BertConfig, BertModel
# from modeling_albert import AlbertModel, AlbertConfig
# from modeling_xlm_roberta import XLMRobertaConfig
# from modeling_roberta import RobertaConfig, RobertaModel
# import util

from diffusionabsa.modeling_albert import AlbertModel, AlbertConfig
from diffusionabsa.modeling_bert import BertConfig, BertModel
from diffusionabsa.modeling_roberta import RobertaConfig, RobertaModel
from diffusionabsa.modeling_xlm_roberta import XLMRobertaConfig
from transformers.modeling_utils import PreTrainedModel
from diffusionabsa import util
import logging

logger = logging.getLogger()

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


# 左边界和右边界的概率表示
class AspectBoundaryPredictor(nn.Module):
    def __init__(self, config, prop_drop=0.1):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.token_embedding_linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.aspect_embedding_linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.boundary_predictor = nn.Linear(self.hidden_size, 1)

    def forward(self, token_embedding, aspect_embedding, token_mask):
        aspect_token_matrix = self.token_embedding_linear(token_embedding).unsqueeze(1) + self.aspect_embedding_linear(aspect_embedding).unsqueeze(2)
        aspect_token_cls = self.boundary_predictor(torch.relu(aspect_token_matrix)).squeeze(-1)
        token_mask = token_mask.unsqueeze(1).expand(-1, aspect_token_cls.size(1), -1)
        aspect_token_cls[~token_mask] = -1e25
        aspect_token_p = F.sigmoid(aspect_token_cls)

        return aspect_token_p


# 实体类别概率
class AspectTypePredictor(nn.Module):
    def __init__(self, config, aspect_type_count):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, aspect_type_count),
        )

    def forward(self, h_cls):
        aspect_logits = self.classifier(h_cls)
        return aspect_logits


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class SpanAttentionLayer(nn.Module):
    def __init__(self, d_model=768, d_ffn=1024, dropout=0.1, activation="relu", n_heads=8, self_attn=True, cross_attn=True):
        super().__init__()

        self.self_attn_bool = self_attn
        self.cross_attn_bool = cross_attn

        if self.cross_attn_bool:
            # cross attention
            self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout1 = nn.Dropout(dropout)
            self.norm1 = nn.LayerNorm(d_model)
        if self.self_attn_bool:
            # self attention
            self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.norm2 = nn.LayerNorm(d_model)
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, pos, src, mask):
        if self.self_attn_bool:
            q = k = self.with_pos_embed(tgt, pos)
            v = tgt
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))[0].transpose(0, 1)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        if self.cross_attn_bool:
            q = self.with_pos_embed(tgt, pos)
            k = v = src
            tgt2 = self.cross_attn(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), key_padding_mask=~mask if mask is not None else None)[0].transpose(0, 1)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class SpanAttention(nn.Module):
    def __init__(self, decoder_layer, synta_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, synta_layers)

    def forward(self, tgt, pos, src, mask):
        output = tgt

        for lid, layer in enumerate(self.layers):
            output = layer(output, pos, src, mask)

        return output


def span_lw_to_lr(x):
    l, w = x.unbind(-1)
    b = [l, l + w]
    return torch.stack(b, dim=-1)


def span_lr_to_lw(x):
    l, r = x.unbind(-1)
    b = [l, r - l]
    x = torch.stack(b, dim=-1)
    return torch.stack(b, dim=-1)


def create_aspect_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end + 1] = 1
    return mask


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # step5
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def constant_beta_schedule(timesteps):
    scale = 1000 / timesteps
    constant = scale * 0.01
    return torch.tensor([constant] * timesteps, dtype=torch.float64)


def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ Get specific token embedding (e.g. [CLS]) """
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    token_h = token_h[flat == token, :]

    return token_h


class DiffusionABSA(PreTrainedModel):
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def __init__(
            self,
            model_type,
            config,
            aspect_type_count,
            lstm_layers=0,
            synta_layers=0,
            timesteps=1000,
            beta_schedule="cosine",
            p2_loss_weight_gamma=0.,
            p2_loss_weight_k=1,
            sampling_timesteps=5,
            num_proposals=100,
            scale=3.0,
            extand_noise_spans='repeat',
            span_renewal=False,
            step_ensemble=False,
            prop_drop=0.1,
            soi_pooling="maxpool+lrconcat",
            pos_type="sine",
            step_embed_type="add",
            sample_dist_type="normal",
            split_epoch=0,
            pool_type="max",
            all_args=None,
            wo_self_attn=False,
            wo_cross_attn=False,
            wo_synatt=False,
    ):
        super().__init__(config)
        self.model_type = model_type
        self.bert_dropout = nn.Dropout(all_args.bert_dropout)
        self.SynFue = SynFueEncoder(opt=all_args)
        self._aspect_type_count = aspect_type_count
        self.pool_type = pool_type
        self.synta_layers = synta_layers
        self.soi_pooling = soi_pooling
        self.pos_type = pos_type
        self.step_embed_type = step_embed_type
        self.sample_dist_type = sample_dist_type
        self.wo_synatt = wo_synatt

        # build backbone
        if model_type == "roberta":
            self.roberta = RobertaModel(config)
            self.model = self.roberta

        if model_type == "bert":
            self.bert = BertModel(config)
            self.model = self.bert
            for name, param in self.bert.named_parameters():
                if "pooler" in name:
                    param.requires_grad = False

        if model_type == "albert":
            self.albert = AlbertModel(config)
            self.model = self.albert

        self.lstm_layers = lstm_layers
        if self.lstm_layers > 0:
            self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size // 2, num_layers=self.lstm_layers, bidirectional=True, dropout=prop_drop, batch_first=True)

        DiffusionABSA._keys_to_ignore_on_save = ["model." + k for k, v in self.model.named_parameters()]
        DiffusionABSA._keys_to_ignore_on_load_missing = ["model." + k for k, v in self.model.named_parameters()]

        # build head
        self.prop_drop = prop_drop
        self.dropout = nn.Dropout(prop_drop)

        if "lrconcat" in self.soi_pooling:
            self.downlinear = nn.Linear(config.hidden_size * 2, config.hidden_size)
            self.affine_start = nn.Linear(config.hidden_size, config.hidden_size)
            self.affine_end = nn.Linear(config.hidden_size, config.hidden_size)

        if "|" in soi_pooling:
            n = len(soi_pooling.split("|"))
            self.soi_pooling_downlinear = nn.Sequential(
                nn.Linear(config.hidden_size * n, config.hidden_size),
                nn.GELU()
            )

        if self.synta_layers > 0:
            if self.pos_type == "sine":
                self.pos_embeddings = nn.Sequential(
                    SinusoidalPositionEmbeddings(config.hidden_size),
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.GELU(),
                    nn.Linear(config.hidden_size, config.hidden_size),
                )

            spanattentionlayer = SpanAttentionLayer(d_model=config.hidden_size, self_attn=not wo_self_attn, cross_attn=not wo_cross_attn)
            self.spanattention = SpanAttention(spanattentionlayer, synta_layers=self.synta_layers)

        self.left_boundary_predictor = AspectBoundaryPredictor(config)
        self.right_boundary_predictor = AspectBoundaryPredictor(config)
        self.aspect_classifier = AspectTypePredictor(config, aspect_type_count)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        if self.step_embed_type == 'scaleshift':
            self.step_scale_shift = nn.Sequential(nn.SiLU(), nn.Linear(config.hidden_size, config.hidden_size * 2))
        self.split_epoch = split_epoch
        self.has_changed = True

        if self.split_epoch > 0:
            self.has_changed = False
            logger.info(f"Freeze bert weights from begining")
            logger.info("Freeze transformer weights")
            if self.model_type == "bert":
                model = self.bert
            if self.model_type == "roberta":
                model = self.roberta
            if self.model_type == "albert":
                model = self.albert
            for name, param in model.named_parameters():
                param.requires_grad = False

        self.init_weights()

        self.num_proposals = num_proposals
        timesteps = timesteps
        sampling_timesteps = sampling_timesteps
        self.objective = 'pred_x0'
        betas = None
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'constant':
            betas = constant_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.extand_noise_spans = extand_noise_spans

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = scale
        self.span_renewal = span_renewal
        self.step_ensemble = step_ensemble

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        # u = posterior_mean_coef2 * xt + posterior_mean_coef1 * x0
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def model_predictions(self, span, h_token, h_token_lstm, h, timestep, token_masks, x_self_cond=None, clip_x_start=False):
        x_span = torch.clamp(span, min=-1 * self.scale, max=self.scale)  # -scale -- +scale
        x_span = ((x_span / self.scale) + 1) / 2  # 0 -- 1
        x_span = span_lw_to_lr(x_span)  # maybe r > 1
        x_span = torch.clamp(x_span, min=0, max=1)
        outputs_logits, outputs_span, left_aspect_token_p, right_aspect_token_p = self.head(x_span, h_token, h_token_lstm, h, timestep, token_masks)

        token_count = token_masks.long().sum(-1, keepdim=True)
        token_count_expanded = token_count.unsqueeze(1).expand(-1, span.size(1), span.size(2))

        x_start = outputs_span  # (batch, num_proposals, 2) predict spans: absolute coordinates (x1, y1, x2, y2)
        x_start = x_start / (token_count_expanded - 1 + 1e-20)
        # x_start = x_start / token_count_expanded
        x_start = span_lr_to_lw(x_start)
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(span, timestep, x_start)

        return ModelPrediction(pred_noise, x_start), outputs_logits, outputs_span, left_aspect_token_p, right_aspect_token_p

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        # step4
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        # 前向扩散过程，由x0 求 xt
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, span, h_token, h_token_lstm, time_cond, token_masks, self_cond=None, clip_denoised=True):
        preds, outputs_class, outputs_coord, left_aspect_token_p, right_aspect_token_p = self.model_predictions(
            span, h_token, h_token_lstm, time_cond, token_masks, self_cond, clip_x_start=clip_denoised
        )
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=span, t=time_cond)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def sample(self, h_token, h_token_lstm, token_masks):
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(h_token, h_token_lstm, token_masks)

    @torch.no_grad()
    def p_sample_loop(self, h_token, h_token_lstm, token_masks):
        batch = token_masks.shape[0]
        shape = (batch, self.num_proposals, 2)
        span = torch.randn(shape, device=device)

        x_start = None

        for t in reversed(range(0, self.num_timesteps)):
            self_cond = x_start if self.self_condition else None
            span, x_start = self.p_sample(span, h_token, h_token_lstm, t, token_masks, self_cond)
        return span

    @torch.no_grad()
    def p_sample(self, span, h_token, h_token_lstm, t, token_masks, x_self_cond=None, clip_denoised=True):
        b, *_, device = *span.shape, span.device
        batched_times = torch.full((span.shape[0],), t, device=span.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(span, h_token, h_token_lstm, batched_times, token_masks, self_cond=x_self_cond, clip_denoised=clip_denoised)
        noise = torch.randn_like(span) if t > 0 else 0.  # no noise if t == 0
        pred = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def ddim_sample(self, h_token, h_token_lstm, h, token_masks, clip_denoised=True):
        batch = token_masks.shape[0]
        shape = (batch, self.num_proposals, 2)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        if self.sample_dist_type == "normal":
            span = torch.randn(shape, device=self.device)
        elif self.sample_dist_type == "uniform":
            span = (2 * torch.rand(shape, device=self.device) - 1) * self.scale

        x_start = None
        step_ensemble_outputs_class = []
        step_ensemble_outputs_coord = []
        step_ensemble_left_aspect_token_p = []
        step_ensemble_right_aspect_token_p = []
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            preds, outputs_class, outputs_coord, left_aspect_token_p, right_aspect_token_p = self.model_predictions(span, h_token, h_token_lstm, h, time_cond, token_masks,
                                                                                                                    self_cond, clip_x_start=clip_denoised)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            if time_next < 0:
                span = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            if self.sample_dist_type == "normal":
                noise = torch.randn_like(span)
            elif self.sample_dist_type == "uniform":
                noise = torch.rand_like(span)

            span = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            if self.span_renewal:  # filter
                score_per_span, boundary_per_span = outputs_class, outputs_coord
                threshold = 0.0
                score_per_span = F.softmax(score_per_span, dim=-1)
                value, _ = torch.max(score_per_span, -1, keepdim=False)
                keep_idx = value > threshold
                keep_idx = keep_idx * (boundary_per_span[:, :, 1] >= boundary_per_span[:, :, 0])
                num_remain = torch.sum(keep_idx)
                span[~keep_idx] = torch.randn(self.num_proposals * span.size(0) - num_remain, 2, device=span.device).double()

            if self.step_ensemble:
                step_ensemble_outputs_class.append(outputs_class)
                step_ensemble_outputs_coord.append(outputs_coord)
                step_ensemble_left_aspect_token_p.append(left_aspect_token_p)
                step_ensemble_right_aspect_token_p.append(right_aspect_token_p)

        output = {'pred_logits': outputs_class, 'pred_spans': outputs_coord, "pred_left": left_aspect_token_p, "pred_right": right_aspect_token_p}
        if self.step_ensemble:
            output = {'pred_logits': torch.cat(step_ensemble_outputs_class, dim=1), 'pred_spans': torch.cat(step_ensemble_outputs_coord, dim=1),
                      "pred_left": torch.cat(step_ensemble_left_aspect_token_p, dim=1), "pred_right": torch.cat(step_ensemble_right_aspect_token_p, dim=1)}
        return output

    def _bert_encoder(self, input_ids, pieces2word):
        """

        :param input_ids: [B, L'],  L' not equal L
        :param pieces2word: [B, L, L']
        :return:
            word_reps: [B, L, H]
            pooler_output: [B, H]
        """
        if self.model_type == 'bert':
            bert_output = self.bert(input_ids=input_ids, attention_mask=input_ids.ne(0).float())
        elif self.model_type == 'roberta':
            bert_output = self.roberta(input_ids=input_ids, attention_mask=input_ids.ne(0).float())

        sequence_output, pooler_output = bert_output[0], bert_output[1]
        bert_embs = self.bert_dropout(sequence_output)

        length = pieces2word.size(1)
        min_value = torch.min(bert_embs).item()

        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)
        return word_reps, pooler_output

    def forward(self,
                encodings: torch.tensor,
                context_masks: torch.tensor,
                token_masks: torch.tensor,
                context2token_masks: torch.tensor,
                pos_encoding: torch.tensor = None,
                seg_encoding: torch.tensor = None,
                aspect_spans: torch.tensor = None,
                aspect_types: torch.tensor = None,
                aspect_masks: torch.tensor = None,
                pos_indices=None,
                graph=None,
                simple_graph=None,
                pieces2word=None,
                meta_doc=None,
                args=None,
                epoch=None):

        h_token, h_token_lstm = self.backbone(encodings,
                                              context_masks,
                                              token_masks,
                                              pos_encoding,
                                              seg_encoding,
                                              context2token_masks)

        word_reps, cls_output = self._bert_encoder(input_ids=encodings, pieces2word=pieces2word)
        h, dep_output = self.SynFue(word_reps=word_reps, simple_graph=simple_graph, graph=graph, pos=pos_indices)
        # DDIM逆扩散过程
        if not self.training:
            results = self.ddim_sample(h_token, h_token_lstm, h, token_masks)
            return results

        # DDPM 扩散过程
        if self.training:
            if not self.has_changed and epoch >= self.split_epoch:
                logger.info(f"Now, update bert weights @ epoch = {epoch}")
                self.has_changed = True
                for name, param in self.named_parameters():
                    param.requires_grad = True
            d_spans, noises, t = self.prepare_targets(aspect_spans, aspect_types, aspect_masks, token_masks, meta_doc=meta_doc)
            t = t.squeeze(-1)
            outputs_class, outputs_span, left_aspect_token_p, right_aspect_token_p = self.head(d_spans, h_token, h_token_lstm, h, t, token_masks)
            output = {'pred_logits': outputs_class, 'pred_spans': outputs_span, 'pred_left': left_aspect_token_p, 'pred_right': right_aspect_token_p}

            return output

    def prepare_diffusion_repeat(self, gt_spans, gt_num):
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 2, device=self.device)

        num_gt = gt_num.item()
        gt_spans = gt_spans[:gt_num]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_spans = torch.as_tensor([[0., 1.]], dtype=torch.float, device=gt_spans.device)
            num_gt = 1

        num_repeat = self.num_proposals // num_gt  # number of repeat except the last gt box in one image
        repeat_tensor = [num_repeat] * (num_gt - self.num_proposals % num_gt) + [num_repeat + 1] * (
                self.num_proposals % num_gt)
        assert sum(repeat_tensor) == self.num_proposals
        random.shuffle(repeat_tensor)
        repeat_tensor = torch.tensor(repeat_tensor, device=self.device)

        gt_spans = (gt_spans * 2. - 1.) * self.scale
        x_start = torch.repeat_interleave(gt_spans, repeat_tensor, dim=0)

        # noise sample 返回xt
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_spans = span_lw_to_lr(x)
        diff_spans = torch.clamp(diff_spans, min=0, max=1)

        return diff_spans, noise, t

    def prepare_diffusion_concat(self, gt_spans, gt_num):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 2, device=self.device)

        num_gt = gt_num.item()
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_spans = torch.as_tensor([[0., 1.]], dtype=torch.float, device=gt_spans.device)
            num_gt = 1

        if num_gt < self.num_proposals:
            box_placeholder = torch.randn(self.num_proposals - num_gt, 2,
                                          device=self.device) / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6
            box_placeholder[:, 1:] = torch.clip(box_placeholder[:, 1:], min=1e-4)
            x_start = torch.cat((gt_spans, box_placeholder), dim=0)
        elif num_gt > self.num_proposals:
            select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_spans[select_mask]
        else:
            x_start = gt_spans

        x_start = (x_start * 2. - 1.) * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_spans = span_lw_to_lr(x)
        diff_spans = torch.clamp(diff_spans, min=0, max=1)

        return diff_spans, noise, t

    def prepare_targets(self, aspect_spans, aspect_types, aspect_masks, token_masks, meta_doc):
        # step2
        diffused_spans = []
        noises = []
        ts = []
        token_count = token_masks.long().sum(-1, keepdim=True)
        for gt_spans, gt_types, aspect_mask, sent_length in zip(aspect_spans, aspect_types, aspect_masks, token_count):
            gt_num = aspect_mask.sum()
            target = {}
            gt_spans = gt_spans / sent_length
            gt_spans = span_lr_to_lw(gt_spans)
            d_spans = d_noise = d_t = None
            if self.extand_noise_spans == "concat":
                d_spans, d_noise, d_t = self.prepare_diffusion_concat(gt_spans, gt_num)
            elif self.extand_noise_spans == "repeat":
                d_spans, d_noise, d_t = self.prepare_diffusion_repeat(gt_spans, gt_num)

            diffused_spans.append(d_spans)
            noises.append(d_noise)
            ts.append(d_t)

        return torch.stack(diffused_spans), torch.stack(noises), torch.stack(ts)

    def backbone(self,
                 encodings: torch.tensor,
                 context_masks: torch.tensor,
                 token_masks: torch.tensor,
                 pos_encoding: torch.tensor = None,
                 seg_encoding: torch.tensor = None,
                 context2token_masks: torch.tensor = None):

        outputs = self.model(
            input_ids=encodings,
            attention_mask=context_masks,
            # token_type_ids=seg_encoding,
            position_ids=pos_encoding,
            output_hidden_states=True)

        h = outputs.hidden_states[-1]
        h_token = util.combine(h, context2token_masks, self.pool_type)

        h_token_lstm = None
        if self.lstm_layers > 0:
            token_count = token_masks.long().sum(-1, keepdim=True)
            h_token_lstm = nn.utils.rnn.pack_padded_sequence(input=h_token, lengths=token_count.squeeze(-1).cpu().tolist(), enforce_sorted=False, batch_first=True)
            h_token_lstm, (_, _) = self.lstm(h_token_lstm)
            h_token_lstm, _ = nn.utils.rnn.pad_packed_sequence(h_token_lstm, batch_first=True)

        return h_token, h_token_lstm

    def head(self,
             span: torch.tensor,
             h_token: torch.tensor,
             h_token_lstm: torch.tensor,
             h: torch.tensor,
             timestep: torch.tensor,
             token_masks: torch.tensor):

        token_count = token_masks.long().sum(-1, keepdim=True)
        token_count_expanded = token_count.unsqueeze(1).expand(-1, span.size(1), span.size(2))

        old_span = span
        span = old_span * (token_count_expanded - 1)

        span_mask = None
        if "pool" in self.soi_pooling:
            span_mask = []
            for tk, sp in zip(token_count, torch.round(span).to(dtype=torch.long)):
                sp_mask = []
                for s in sp:
                    sp_mask.append(create_aspect_mask(*s, tk))
                span_mask.append(torch.stack(sp_mask))
            span_mask = util.padded_stack(span_mask).to(device=h_token.device)

        timestep_embeddings = self.time_mlp(timestep)

        left_aspect_token_p, right_aspect_token_p, aspect_logits = self.left_right_type(h_token, h_token_lstm, h, span_mask, timestep_embeddings, span, token_count, token_masks)
        aspect_left = left_aspect_token_p.argmax(dim=-1)
        aspect_right = right_aspect_token_p.argmax(dim=-1)
        aspect_spans = torch.stack([aspect_left, aspect_right], dim=-1)
        return aspect_logits, aspect_spans, left_aspect_token_p, right_aspect_token_p

    def left_right_type(self, h_token, h_token_lstm, h, span_mask, timestep_embeddings, span, token_count, token_masks):

        N, nr_spans = span.shape[:2]

        if h_token_lstm is None:
            h_token_lstm = h_token

        aspect_spans_pools = []
        if "maxpool" in self.soi_pooling:
            pool_aspect_spans_pool = util.combine(h_token_lstm, span_mask, "max")
            pool_aspect_spans_pool = self.dropout(pool_aspect_spans_pool)
            aspect_spans_pools.append(pool_aspect_spans_pool)

        if "meanpool" in self.soi_pooling:
            pool_aspect_spans_pool = util.combine(h_token_lstm, span_mask, "mean")
            pool_aspect_spans_pool = self.dropout(pool_aspect_spans_pool)
            aspect_spans_pools.append(pool_aspect_spans_pool)

        if "sumpool" in self.soi_pooling:
            pool_aspect_spans_pool = util.combine(h_token_lstm, span_mask, "sum")
            pool_aspect_spans_pool = self.dropout(pool_aspect_spans_pool)
            aspect_spans_pools.append(pool_aspect_spans_pool)

        if "lrconcat" in self.soi_pooling:

            aspect_spans_token_inner = torch.round(span).to(dtype=torch.long)
            aspect_spans_token_inner[:, :, 0][aspect_spans_token_inner[:, :, 0] < 0] = 0
            aspect_spans_token_inner[:, :, 1][aspect_spans_token_inner[:, :, 1] < 0] = 0
            aspect_spans_token_inner[:, :, 0][aspect_spans_token_inner[:, :, 0] >= token_count] = token_count.repeat(1, aspect_spans_token_inner.size(1))[aspect_spans_token_inner[:, :, 0] >= token_count] - 1
            aspect_spans_token_inner[:, :, 1][aspect_spans_token_inner[:, :, 1] >= token_count] = token_count.repeat(1, aspect_spans_token_inner.size(1))[aspect_spans_token_inner[:, :, 1] >= token_count] - 1
            start_end_embedding_inner = util.batch_index(h_token_lstm, aspect_spans_token_inner)

            start_affined = self.dropout(self.affine_start(start_end_embedding_inner[:, :, 0]))
            end_affined = self.dropout(self.affine_end(start_end_embedding_inner[:, :, 1]))

            embed_inner = [start_affined, end_affined]
            lrconcat_aspect_spans_pool = self.dropout(self.downlinear(torch.cat(embed_inner, dim=2)))
            aspect_spans_pools.append(lrconcat_aspect_spans_pool)

        if len(aspect_spans_pools) > 1:
            if "|" in self.soi_pooling:
                aspect_spans_pool = torch.cat(aspect_spans_pools, dim=-1)
                aspect_spans_pool = self.soi_pooling_downlinear(aspect_spans_pool)
            if "+" in self.soi_pooling:
                aspect_spans_pool = torch.stack(aspect_spans_pools, dim=0).sum(dim=0)
        else:
            aspect_spans_pool = aspect_spans_pools[0]

        if self.synta_layers > 0:

            pos = None

            if self.pos_type == "same":
                pos = aspect_spans_pool
            elif self.pos_type == "sine":
                pos = self.pos_embeddings(torch.arange(nr_spans).to(h.device)).repeat(N, 1, 1)

            if self.wo_synatt:
                aspect_spans_pool = aspect_spans_pool
            else:
                aspect_spans_pool = self.spanattention(aspect_spans_pool, pos, h, token_masks)

        if self.step_embed_type == "add":
            aspect_spans_pool = aspect_spans_pool + timestep_embeddings.unsqueeze(1).repeat(1, nr_spans, 1)
        elif self.step_embed_type == "scaleshift":
            aspect_spans_pool = aspect_spans_pool.reshape(N * nr_spans, -1)
            scale_shift = self.step_scale_shift(timestep_embeddings)
            scale_shift = torch.repeat_interleave(scale_shift, nr_spans, dim=0)
            scale, shift = scale_shift.chunk(2, dim=1)
            aspect_spans_pool = aspect_spans_pool * (scale + 1) + shift
            aspect_spans_pool = aspect_spans_pool.view(N, nr_spans, -1)

        left_aspect_token_p = self.left_boundary_predictor(h_token_lstm, aspect_spans_pool, token_masks)
        right_aspect_token_p = self.right_boundary_predictor(h_token_lstm, aspect_spans_pool, token_masks)
        aspect_logits = self.aspect_classifier(aspect_spans_pool)

        return left_aspect_token_p, right_aspect_token_p, aspect_logits


class BertDiffusionABSA(DiffusionABSA):
    config_class = BertConfig
    base_model_prefix = "bert"
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, *args, **kwagrs):
        super().__init__("bert", *args, **kwagrs)


class RobertaDiffusionABSA(DiffusionABSA):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, *args, **kwagrs):
        super().__init__("roberta", *args, **kwagrs)


class XLMRobertaDiffusionABSA(DiffusionABSA):
    config_class = XLMRobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, *args, **kwagrs):
        super().__init__("roberta", *args, **kwagrs)


class AlbertDiffusionABSA(DiffusionABSA):
    config_class = AlbertConfig
    base_model_prefix = "albert"

    def __init__(self, *args, **kwagrs):
        super().__init__("albert", *args, **kwagrs)


_MODELS = {
    'diffusionabsa': BertDiffusionABSA,
    'roberta_diffusionabsa': RobertaDiffusionABSA,
    'xlmroberta_diffusionabsa': XLMRobertaDiffusionABSA,
    'albert_diffusionabsa': AlbertDiffusionABSA
}


def get_model(name):
    return _MODELS[name]


def sinusoidal_position_embedding(batch_size, max_len, output_dim, device):
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
    ids = torch.arange(0, output_dim // 2, dtype=torch.float)
    theta = torch.pow(10000, -2 * ids / output_dim)

    embeddings = position * theta

    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

    embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))

    embeddings = torch.reshape(embeddings, (batch_size, max_len, output_dim))
    embeddings = embeddings.to(device)
    return embeddings


def RoPE(q, k):
    batch_size = q.shape[0]
    max_len = q.shape[1]
    output_dim = q.shape[-1]

    pos_emb = sinusoidal_position_embedding(batch_size, max_len, output_dim, q.device)
    cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)

    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
    q2 = q2.reshape(q.shape)

    q = q * cos_pos + q2 * sin_pos

    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
    k2 = k2.reshape(k.shape)
    k = k * cos_pos + k2 * sin_pos

    return q, k


class LabelAwareGCN(nn.Module):
    """
    Simple GCN layer
    """

    def __init__(self, dep_dim, in_features, out_features, pos_dim=None, bias=True):
        super(LabelAwareGCN, self).__init__()
        self.dep_dim = dep_dim
        self.pos_dim = pos_dim
        self.in_features = in_features  # bert_dim
        self.out_features = out_features

        self.dep_attn = nn.Linear(dep_dim + pos_dim + in_features, out_features)
        self.dep_fc = nn.Linear(dep_dim, out_features)
        self.pos_fc = nn.Linear(pos_dim, out_features)

    def forward(self, text, adj, dep_embed, pos_embed=None):
        """

        :param text: [batch size, seq_len, feat_dim]
        :param adj: [batch size, seq_len, seq_len]
        :param dep_embed: [batch size, seq_len, seq_len, dep_type_dim]
        :param pos_embed: [batch size, seq_len, pos_dim]
        :return: [batch size, seq_len, feat_dim]
        """
        batch_size, seq_len, feat_dim = text.shape

        val_us = text.unsqueeze(dim=2)
        val_us = val_us.repeat(1, 1, seq_len, 1)
        pos_us = pos_embed.unsqueeze(dim=2).repeat(1, 1, seq_len, 1)
        val_sum = torch.cat([val_us, pos_us, dep_embed], dim=-1)

        r = self.dep_attn(val_sum)

        p = torch.sum(r, dim=-1)
        mask = (adj == 0).float() * (-1e30)
        p = p + mask
        p = torch.softmax(p, dim=2)
        p_us = p.unsqueeze(3).repeat(1, 1, 1, feat_dim)

        output = val_us + self.pos_fc(pos_us) + self.dep_fc(dep_embed)
        output = torch.mul(p_us, output)

        output_sum = torch.sum(output, dim=2)

        return r, output_sum, p


class nLaGCN(nn.Module):
    def __init__(self, opt):
        super(nLaGCN, self).__init__()
        self.opt = opt
        self.model = nn.ModuleList([LabelAwareGCN(opt.dep_dim, opt.input_bert_dim, opt.output_bert_dim, opt.pos_dim) for i in range(opt.synta_layers)])

        self.dep_embedding = nn.Embedding(opt.dep_num, opt.dep_dim, padding_idx=0)

    def forward(self, x, simple_graph, graph, pos_embed=None, output_attention=False):

        dep_embed = self.dep_embedding(graph)

        attn_list = []
        for lagcn in self.model:
            r, x, attn = lagcn(x, simple_graph, dep_embed, pos_embed=pos_embed)
            attn_list.append(attn)

        if output_attention is True:
            return x, r, attn_list
        else:
            return x, r


class SynFueEncoder(nn.Module):
    def __init__(self, opt):
        super(SynFueEncoder, self).__init__()
        self.opt = opt
        self.lagcn = nLaGCN(opt)

        self.fc = nn.Linear(opt.input_bert_dim * 2 + opt.pos_dim, opt.output_bert_dim)
        # self.fc = nn.Linear(opt.bert_dim * 2 + opt.pos_dim, opt.bert_dim)
        self.output_dropout = nn.Dropout(opt.output_dropout)

        self.pod_embedding = nn.Embedding(opt.pos_num, opt.pos_dim, padding_idx=0)

    def forward(self, word_reps, simple_graph, graph, pos=None, output_attention=False):
        """

        :param word_reps: [B, L, H]
        :param simple_graph: [B, L, L]
        :param graph: [B, L, L]
        :param pos: [B, L]
        :param output_attention: bool
        :return:
            output: [B, L, H]
            dep_reps: [B, L, H]
            cls_reps: [B, H]
        """
        pos_embed = self.pod_embedding(pos)

        lagcn_output = self.lagcn(word_reps, simple_graph, graph, pos_embed, output_attention)

        pos_output = self.local_attn(word_reps, pos_embed, self.opt.synta_layers, self.opt.w_size)

        output = torch.cat((lagcn_output[0], pos_output, word_reps), dim=-1)

        # output = torch.cat((lagcn_output[0], pos_embed, word_reps), dim=-1)
        output = self.fc(output)
        output = self.output_dropout(output)
        return output, lagcn_output[1]

    def local_attn(self, x, pos_embed, synta_layers, w_size):
        """

        :param x:
        :param pos_embed:
        :return:
        """
        batch_size, seq_len, feat_dim = x.shape
        pos_dim = pos_embed.size(-1)
        output = pos_embed
        for i in range(synta_layers):
            val_sum = torch.cat([x, output], dim=-1)  # [batch size, seq_len, feat_dim+pos_dim]
            attn = torch.matmul(val_sum, val_sum.transpose(1, 2))  # [batch size, seq_len, seq_len]
            # pad size = seq_len + (window_size - 1) // 2 * 2
            pad_size = seq_len + w_size * 2
            # print(torch.zeros((batch_size, seq_len, pad_size), dtype=torch.float).shape)
            mask = torch.zeros((batch_size, seq_len, pad_size), dtype=torch.float).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            for i in range(seq_len):
                mask[:, i, i:i + w_size] = 1.0
            pad_attn = torch.full((batch_size, seq_len, w_size), -1e18).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            attn = torch.cat([pad_attn, attn, pad_attn], dim=-1)
            local_attn = torch.softmax(torch.mul(attn, mask), dim=-1)
            local_attn = local_attn[:, :, w_size:pad_size - w_size]  # [batch size, seq_len, seq_len]
            local_attn = local_attn.unsqueeze(dim=3).repeat(1, 1, 1, pos_dim)
            output = output.unsqueeze(dim=2).repeat(1, 1, seq_len, 1)
            output = torch.sum(torch.mul(output, local_attn), dim=2)  # [batch size, seq_len, pos_dim]
        return output
