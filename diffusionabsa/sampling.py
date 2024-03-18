import torch
from diffusionabsa import util


def create_train_sample(doc, repeat_gt_aspects=100):
    encodings = doc.encoding
    seg_encoding = doc.seg_encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)
    pieces2word = torch.zeros((token_count, context_size), dtype=torch.bool)
    start = 0
    for i, token in enumerate(doc.tokens):
        pieces = list(range(token.span_start, token.span_end + 1))
        pieces2word[i, pieces[0]: pieces[-1] + 1] = 1
        start += len(pieces)

    context2token_masks = []
    for t in doc.tokens:
        context2token_masks.append(create_aspect_mask(*t.span, context_size))

    gt_aspects_spans_token = []
    gt_aspect_types = []
    gt_aspect_masks = []

    for e in doc.aspects:
        gt_aspects_spans_token.append(e.span_token)
        gt_aspect_types.append(e.aspect_type.index)
        gt_aspect_masks.append(1)

    if repeat_gt_aspects != -1:
        if len(doc.aspects) != 0:
            k = repeat_gt_aspects // len(doc.aspects)
            m = repeat_gt_aspects % len(doc.aspects)
            gt_aspects_spans_token = gt_aspects_spans_token * k + gt_aspects_spans_token[:m]
            gt_aspect_types = gt_aspect_types * k + gt_aspect_types[:m]
            gt_aspect_masks = gt_aspect_masks * k + gt_aspect_masks[:m]
            assert len(gt_aspects_spans_token) == len(gt_aspect_types) == len(gt_aspect_masks) == repeat_gt_aspects

    encodings = torch.tensor(encodings, dtype=torch.long)
    seg_encoding = torch.tensor(seg_encoding, dtype=torch.long)

    context_masks = torch.ones(context_size, dtype=torch.bool)

    token_masks = torch.ones(token_count, dtype=torch.bool)

    context2token_masks = torch.stack(context2token_masks)

    if len(gt_aspect_types) > 0:
        gt_aspect_types = torch.tensor(gt_aspect_types, dtype=torch.long)
        gt_aspect_spans_token = torch.tensor(gt_aspects_spans_token, dtype=torch.long)
        gt_aspect_masks = torch.tensor(gt_aspect_masks, dtype=torch.bool)
    else:
        gt_aspect_types = torch.zeros([1], dtype=torch.long)
        gt_aspect_spans_token = torch.zeros([1, 2], dtype=torch.long)
        gt_aspect_masks = torch.zeros([1], dtype=torch.bool)

    # pos dep
    pos_indices = torch.tensor(get_pos(token_count, doc.pos_indices), dtype=torch.long)
    graph = torch.tensor(get_graph(token_count, doc.dep, doc.dep_label_indices), dtype=torch.long)
    simple_graph = torch.tensor(get_simple_graph(token_count, doc.dep), dtype=torch.long)

    return dict(encodings=encodings, context_masks=context_masks, seg_encoding=seg_encoding,
                context2token_masks=context2token_masks, token_masks=token_masks,
                gt_types=gt_aspect_types, gt_spans=gt_aspect_spans_token, aspect_masks=gt_aspect_masks, meta_doc=doc,
                pos_indices=pos_indices, graph=graph, simple_graph=simple_graph, pieces2word=pieces2word,
                )


def create_eval_sample(doc, processor=None):
    encodings = doc.encoding
    seg_encoding = doc.seg_encoding

    token_count = len(doc.tokens)
    context_size = len(encodings)

    pieces2word = torch.zeros((token_count, context_size), dtype=torch.bool)
    start = 0
    for i, token in enumerate(doc.tokens):
        pieces = list(range(token.span_start, token.span_end + 1))
        pieces2word[i, pieces[0]: pieces[-1] + 1] = 1
        start += len(pieces)

    context2token_masks = []
    for t in doc.tokens:
        context2token_masks.append(create_aspect_mask(*t.span, context_size))

    encodings = torch.tensor(encodings, dtype=torch.long)
    seg_encoding = torch.tensor(seg_encoding, dtype=torch.long)

    context_masks = torch.ones(context_size, dtype=torch.bool)

    token_masks = torch.ones(token_count, dtype=torch.bool)

    context2token_masks = torch.stack(context2token_masks)

    pos_indices = torch.tensor(get_pos(token_count, doc.pos_indices), dtype=torch.long)
    graph = torch.tensor(get_graph(token_count, doc.dep, doc.dep_label_indices), dtype=torch.long)
    simple_graph = torch.tensor(get_simple_graph(token_count, doc.dep), dtype=torch.long)

    return dict(encodings=encodings, context_masks=context_masks, seg_encoding=seg_encoding,
                context2token_masks=context2token_masks, token_masks=token_masks, meta_doc=doc,
                pos_indices=pos_indices, graph=graph, simple_graph=simple_graph, pieces2word=pieces2word,
                )


def create_aspect_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end + 1] = 1
    return mask


def collate_fn_padding(batch):
    batch = list(filter(lambda x: x is not None, batch))
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]
        if key.startswith("meta"):
            padded_batch[key] = samples
            continue

        if key.startswith("image_inputs"):
            if batch[0]["image_inputs"] == None:
                padded_batch["image_inputs"] = None
            else:
                padded_batch["image_inputs"] = dict(
                    (k, torch.cat([s["image_inputs"][k] for s in batch], dim=0)) for k in
                    batch[0]["image_inputs"].keys())
            continue

        if batch[0][key] is None:
            padded_batch[key] = None
            continue

        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = util.padded_stack([s[key] for s in batch])

    return padded_batch


def get_graph(seq_len, feature_data, feature2id):
    """
    To create a table t_{i,j} in T. t_{i,j} = r, r is the dependency relation label between word i and word j.
    :param seq_len: token
    :param feature_data: dependency head. Specifically, '0' represents the head word is ROOT
    :param feature2id: dependency label indices
    :return:
    """
    assert len(feature2id) == len(feature_data) == seq_len
    ret = [[0] * seq_len for _ in range(seq_len)]
    for i, item in enumerate(feature_data):
        # the head word is ROOT, so this token only has a self-loop edge
        if int(item) == 0:
            ret[i][i] = 1
            continue
        ret[i][int(item) - 1] = feature2id[i]
        ret[int(item) - 1][i] = feature2id[i]
        ret[i][i] = 1
    return ret


def get_simple_graph(seq_len, feature_data):
    """
    To create a table t_{i,j} in T. t_{i,j} = 1, which means there is an edge between the word i and word j.
    :param seq_len: token
    :param feature_data: dependency head. Specifically, '0' represents the head word is ROOT
    :return:
    """
    assert len(feature_data) == seq_len
    ret = [[0] * seq_len for _ in range(seq_len)]
    for i, item in enumerate(feature_data):
        if int(item) == 0:
            ret[i][i] = 1
            continue
        ret[i][int(item) - 1] = 1
        ret[int(item) - 1][i] = 1
        ret[i][i] = 1
    return ret


def get_pos(seq_len, pos_indices):
    assert len(pos_indices) == seq_len
    ret = [0] * seq_len
    for i, item in enumerate(pos_indices):
        ret[i] = pos_indices[i]
    return ret
