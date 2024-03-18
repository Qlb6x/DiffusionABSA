from .aspects import Token
import json
import os
from typing import List, Tuple, Dict

import torch
from sklearn.metrics import precision_recall_fscore_support as prfs
from transformers import BertTokenizer
# import util
# from aspects import  Document,Dataset,AspectType
# from input_reader import JsonInputReader
from diffusionabsa import util
from diffusionabsa.aspects import Document, Dataset, AspectType
from diffusionabsa.input_reader import JsonInputReader
import jinja2

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class Evaluator:
    def __init__(self, dataset: Dataset, input_reader: JsonInputReader, text_encoder: BertTokenizer, logger, no_overlapping: bool, no_partial_overlapping: bool, no_duplicate: bool, predictions_path: str, examples_path: str, example_count: int, epoch: int, dataset_label: str, cls_threshold=0,
                 boundary_threshold=0, aspect_threshold=0, save_prediction=False):
        self._text_encoder = text_encoder
        self._input_reader = input_reader
        self._dataset = dataset
        self._logger = logger
        self._no_overlapping = no_overlapping
        self._no_partial_overlapping = no_partial_overlapping
        self._no_duplicate = no_duplicate
        self._save_prediction = save_prediction
        self._cls_threshold = cls_threshold
        self._boundary_threshold = boundary_threshold
        self._aspect_threshold = aspect_threshold

        self._epoch = epoch
        self._dataset_label = dataset_label

        self._predictions_path = predictions_path

        self._examples_path = examples_path
        self._example_count = example_count

        # aspects
        self._gt_aspects = []  # ground truth
        self._pred_aspects = []  # prediction
        self._raw_preds = []
        self._raw_raw_preds = []

        self._pseudo_aspect_type = AspectType('Aspect', 1, 'Aspect', 'Aspect')  # for span only evaluation
        self._convert_gt(self._dataset.documents)

    def eval_batch(self, outputs, batch=None):
        aspect_logits = outputs["pred_logits"]
        pred_left = outputs["pred_left"]
        pred_right = outputs["pred_right"]

        batch_size = aspect_logits.shape[0]
        aspect_left = pred_left.argmax(dim=-1)
        aspect_right = pred_right.argmax(dim=-1)
        batch_aspect_left_scores = pred_left.max(dim=-1)[0]
        batch_aspect_right_scores = pred_right.max(dim=-1)[0]

        aspect_prob = aspect_logits.softmax(-1)
        batch_aspect_types = aspect_prob.argmax(dim=-1)

        batch_aspect_scores = aspect_prob.max(dim=-1)[0]

        batch_aspect_mask = (batch_aspect_scores > self._cls_threshold) * (batch_aspect_types != 0)

        batch_aspect_spans = torch.stack([aspect_left, aspect_right], dim=-1)
        # batch_aspect_mask = batch_aspect_mask * (batch_aspect_spans[:,:,0] <= batch_aspect_spans[:,:,1])
        batch_aspect_mask = batch_aspect_mask * (batch_aspect_spans[:, :, 0] <= batch_aspect_spans[:, :, 1]) * ((batch_aspect_left_scores > self._boundary_threshold) | (batch_aspect_right_scores > self._boundary_threshold))

        batch_aspect_mask = batch_aspect_mask * ((batch_aspect_left_scores + batch_aspect_right_scores + batch_aspect_scores) > self._aspect_threshold)

        def roundlist(x):
            return list(map(lambda x: round(x, 2), x))

        for i in range(batch_size):

            doc = batch["meta_doc"][i]
            if self._input_reader.aspect_type_count < 1000 and self._save_prediction:
                decode_aspect = dict(tokens=[t.phrase for t in doc.tokens], pre_aspects=[], gt_aspects=[],
                                     # org_id= doc.doc_id
                                     )

                gt_converted_aspects = []
                for aspect in doc.aspects:
                    aspect = aspect.as_tuple_token()
                    aspect_span = aspect[:2]
                    span_tokens = util.get_span_tokens(doc.tokens, aspect_span)
                    aspect_type = aspect[2].identifier
                    aspect_phrase = str(span_tokens)
                    converted_aspect = dict(type=aspect_type, start=span_tokens[0].index, end=span_tokens[-1].index, phrase=aspect_phrase)
                    gt_converted_aspects.append(converted_aspect)
                decode_aspect["gt_aspects"] = sorted(gt_converted_aspects, key=lambda e: e['start'])

                for j in range(aspect_left.size(1)):
                    span_tokens = str(util.get_span_tokens(doc.tokens, batch_aspect_spans[i][j]))
                    decode_aspect["pre_aspects"].append(
                        dict(aspect_left=aspect_left[i][j].item(), aspect_right=aspect_right[i][j].item(), phrase=span_tokens, aspect_type=self._input_reader.get_aspect_type(batch_aspect_types[i][j].item()).identifier, aspect_prob=roundlist(aspect_prob[i][j].tolist())))
                self._raw_raw_preds.append(decode_aspect)

            # #query
            aspect_mask = batch_aspect_mask[i]

            # #query
            aspect_types = batch_aspect_types[i]
            aspect_spans = batch_aspect_spans[i]
            aspect_scores = batch_aspect_scores[i]

            # #ent
            valid_aspect_types = aspect_types[aspect_mask]
            valid_aspect_spans = aspect_spans[aspect_mask]
            valid_type_scores = aspect_scores[aspect_mask]

            valid_left_scores = batch_aspect_left_scores[i][aspect_mask]
            valid_right_scores = batch_aspect_right_scores[i][aspect_mask]
            valid_aspect_scores = valid_type_scores + valid_left_scores + valid_right_scores

            sample_pred_aspects = self._convert_pred_aspects(valid_aspect_types, valid_aspect_spans, valid_aspect_scores, valid_left_scores, valid_right_scores, valid_type_scores, doc)
            sample_pred_aspects = sorted(sample_pred_aspects, key=lambda x: x[3], reverse=True)

            if self._no_overlapping:
                sample_pred_aspects = self._remove_overlapping(sample_pred_aspects)
            elif self._no_partial_overlapping:
                sample_pred_aspects = self._remove_partial_overlapping(sample_pred_aspects)
            if self._no_duplicate:
                sample_pred_aspects = self._remove_duplicate(sample_pred_aspects)

            self._pred_aspects.append(sample_pred_aspects)

    def _log(self, text):
        if self._logger is not None:
            self._logger.info(text)

    def compute_scores(self):
        self._log("Evaluation")

        self._log("")
        self._log("--- ABSA ---")
        self._log("")

        gt, pred = self._convert_by_setting(self._gt_aspects, self._pred_aspects, include_aspect_types=True)
        absa_eval = self._score(gt, pred, print_results=True)

        self._log("")
        self._log("--- ABSA on Localization ---")
        self._log("")
        gt_wo_type, pred_wo_type = self._convert_by_setting(self._gt_aspects, self._pred_aspects, include_aspect_types=False)
        absa_loc_eval = self._score(gt_wo_type, pred_wo_type, print_results=True)

        self._log("")
        self._log("--- ABSA on Classification ---")
        self._log("")
        absa_cls_eval = self._score(gt, pred, print_results=True, cls_metric=True)

        return absa_eval, absa_loc_eval, absa_cls_eval

    def store_predictions(self):
        predictions = []

        for i, doc in enumerate(self._dataset.documents):
            tokens = doc.tokens
            gt_aspects = self._gt_aspects[i]
            pred_aspects = self._pred_aspects[i]

            gt_converted_aspects = []
            for aspect in gt_aspects:
                aspect_span = aspect[:2]
                span_tokens = util.get_span_tokens(tokens, aspect_span)
                aspect_type = aspect[2].identifier
                aspect_phrase = str(util.get_span_tokens(doc.tokens, aspect_span))
                converted_aspect = dict(type=aspect_type, start=span_tokens[0].index, end=span_tokens[-1].index, phrase=aspect_phrase)
                gt_converted_aspects.append(converted_aspect)
            gt_converted_aspects = sorted(gt_converted_aspects, key=lambda e: e['start'])

            # convert aspects
            pre_converted_aspects = []
            for aspect in pred_aspects:
                aspect_span = aspect[:2]
                # print(aspect_span, tokens)
                # import pdb; pdb.set_trace()
                span_tokens = util.get_span_tokens(tokens, aspect_span)
                aspect_type = aspect[2].identifier
                aspect_phrase = str(util.get_span_tokens(doc.tokens, aspect_span))
                converted_aspect = dict(type=aspect_type, start=span_tokens[0].index, end=span_tokens[-1].index, phrase=aspect_phrase)
                pre_converted_aspects.append(converted_aspect)
            pre_converted_aspects = sorted(pre_converted_aspects, key=lambda e: e['start'])

            doc_predictions = dict(tokens=[t.phrase for t in tokens], pre_aspects=pre_converted_aspects, gt_aspects=gt_converted_aspects)
            predictions.append(doc_predictions)

        # store as json
        label, epoch = self._dataset_label, self._epoch
        with open(self._predictions_path % (label, epoch), 'w') as predictions_file:
            json.dump(predictions, predictions_file)
        with open(self._predictions_path % ("raw_all", epoch), 'w') as predictions_file:
            json.dump(self._raw_preds, predictions_file)
        if len(self._raw_raw_preds) != 0:
            with open(self._predictions_path % ("raw_raw_all", epoch), 'w') as predictions_file:
                json.dump(self._raw_raw_preds, predictions_file)
        # 
        raw_preds_match_gt = []
        raw_preds_not_match_gt = []
        for i, (pre, gt) in enumerate(zip(self._raw_preds, self._gt_aspects)):
            doc = self._dataset.documents[i]

            def is_match(ent):
                for gt_ent in gt:
                    if ent["start"] == gt_ent[0] and ent["end"] == gt_ent[1] and ent["aspect_type"] == gt_ent[2].identifier:
                        return True
                else:
                    return False

            # pre_match_gt = list(filter(is_match, pre))
            # pre_not_match_gt = list(filter(lambda a: not is_match(a), pre))
            # pre_not_match_gt = []
            pre_not_match_gt = dict(tokens=[t.phrase for t in doc.tokens], aspects=[],
                                    # org_id= doc.doc_id
                                    )
            no_dup_pre_match_gt = dict(tokens=[t.phrase for t in doc.tokens], aspects=[],
                                       # org_id= doc.doc_id
                                       )
            pre_match_gt_set = []
            # match gt need dedep; not match gt keep all

            for ent in pre["aspects"]:
                aspect_span = (ent["start"], ent["end"])
                ent["phrase"] = str(util.get_span_tokens(doc.tokens, aspect_span))
                if is_match(ent):
                    if (ent["start"], ent["end"], ent["aspect_type"]) not in pre_match_gt_set:
                        pre_match_gt_set.append((ent["start"], ent["end"], ent["aspect_type"]))
                        no_dup_pre_match_gt["aspects"].append(ent)
                else:
                    pre_not_match_gt["aspects"].append(ent)

            # if len(pre_not_match_gt) > 0:
            #     pre_not_match_gt.insert(0, [t.phrase for t in doc.tokens])
            # if len(no_dup_pre_match_gt) > 0:
            #     no_dup_pre_match_gt.insert(0, [t.phrase for t in doc.tokens])

            raw_preds_not_match_gt.append(pre_not_match_gt)
            # no_dup_pre_match_gt = []
            # pre_match_gt_set = []
            # for ent in pre_match_gt:
            #     if (ent["start"], ent["end"], ent["aspect_type"]) not in pre_match_gt_set:
            #         pre_match_gt_set.append((ent["start"], ent["end"], ent["aspect_type"]))
            #         no_dup_pre_match_gt.append(ent)

            raw_preds_match_gt.append(no_dup_pre_match_gt)
        with open(self._predictions_path % ("match_gt", epoch), 'w') as predictions_file:
            json.dump(raw_preds_match_gt, predictions_file)
        with open(self._predictions_path % ("not_match_gt", epoch), 'w') as predictions_file:
            json.dump(raw_preds_not_match_gt, predictions_file)

    def store_examples(self):
        aspect_examples = []

        for i, doc in enumerate(self._dataset.documents):
            # aspects
            # if len(doc.encoding) > 512:
            #     continue
            aspect_example = self._convert_example(doc, self._gt_aspects[i], self._pred_aspects[i],
                                                   include_aspect_types=True, to_html=self._aspect_to_html)
            aspect_examples.append(aspect_example)

        label, epoch = self._dataset_label, self._epoch

        # aspects
        self._store_examples(aspect_examples[:self._example_count],
                             file_path=self._examples_path % ('aspects', label, epoch),
                             template='aspect_examples.html')

        self._store_examples(sorted(aspect_examples[:self._example_count],
                                    key=lambda k: k['length']),
                             file_path=self._examples_path % ('aspects_sorted', label, epoch),
                             template='aspect_examples.html')

    def _convert_gt(self, docs: List[Document]):
        for doc in docs:
            gt_aspects = doc.aspects
            # if len(doc.encoding) > 512:
            #     continue
            # convert ground truth relations and aspects for precision/recall/f1 evaluation
            sample_gt_aspects = [aspect.as_tuple_token() for aspect in gt_aspects]

            # if self._no_overlapping:
            #     sample_gt_aspects = self._remove_overlapping(sample_gt_aspects)

            self._gt_aspects.append(sample_gt_aspects)

    def _convert_pred_aspects(self, pred_types: torch.tensor, pred_spans: torch.tensor, pred_scores: torch.tensor, left_scores, right_scores, type_scores, doc):
        converted_preds = []

        decode_aspect = dict(tokens=[t.phrase for t in doc.tokens], aspects=[],
                             # org_id= doc.doc_id
                             )
        for i in range(pred_types.shape[0]):
            label_idx = pred_types[i].item()
            aspect_type = self._input_reader.get_aspect_type(label_idx)

            start, end = pred_spans[i].tolist()
            aspect_score = pred_scores[i].item()
            cls_score = type_scores[i].item()
            left_score = left_scores[i].item()
            right_score = right_scores[i].item()

            converted_pred = (start, end, aspect_type, aspect_score)
            converted_preds.append(converted_pred)
            decode_aspect["aspects"].append({"start": start, "end": end, "aspect_type": aspect_type.identifier, "cls_score": round(cls_score, 2), "left_score": round(left_score, 2), "right_score": round(right_score, 2), "aspect_score": round(aspect_score, 2)})
        self._raw_preds.append(decode_aspect)
        return converted_preds

    def _remove_duplicate(self, aspects):
        non_duplicate_aspects = []
        for i, can_aspect in enumerate(aspects):
            find = False
            for j, aspect in enumerate(non_duplicate_aspects):
                if can_aspect[0] == aspect[0] and can_aspect[1] == aspect[1]:
                    find = True
            if not find:
                non_duplicate_aspects.append(can_aspect)
        return non_duplicate_aspects

    def _remove_overlapping(self, aspects):
        non_overlapping_aspects = []
        for i, aspect in enumerate(aspects):
            if not self._is_overlapping(aspect, non_overlapping_aspects):
                non_overlapping_aspects.append(aspect)

        return non_overlapping_aspects

    def _remove_partial_overlapping(self, aspects):
        non_overlapping_aspects = []
        for i, aspect in enumerate(aspects):
            if not self._is_partial_overlapping(aspect, non_overlapping_aspects):
                non_overlapping_aspects.append(aspect)

        return non_overlapping_aspects

    def _is_partial_overlapping(self, e1, aspects):
        for e2 in aspects:
            if self._check_partial_overlap(e1, e2):
                return True

        return False

    def _is_overlapping(self, e1, aspects):
        for e2 in aspects:
            if self._check_overlap(e1, e2):
                return True

        return False

    def _check_overlap(self, e1, e2):
        if e1[1] < e2[0] or e2[1] < e1[0]:
            return False
        else:
            return True

    def _check_partial_overlap(self, e1, e2):
        if (e1[0] < e2[0] and e2[0] <= e1[1] and e1[1] < e2[1]) or (e2[0] < e1[0] and e1[0] <= e2[1] and e2[1] < e1[1]):
            return True
        else:
            return False

    def _convert_by_setting(self, gt: List[List[Tuple]], pred: List[List[Tuple]],
                            include_aspect_types: bool = True, include_score: bool = False):
        assert len(gt) == len(pred)

        # either include or remove aspect types based on setting
        def convert(t):
            if not include_aspect_types:
                # remove aspect type and score for evaluation
                c = [t[0], t[1], self._pseudo_aspect_type]
            else:
                c = list(t[:3])

            if include_score and len(t) > 3:
                # include prediction scores
                c.append(t[3])

            return tuple(c)

        converted_gt, converted_pred = [], []

        for sample_gt, sample_pred in zip(gt, pred):
            converted_gt.append([convert(t) for t in sample_gt])
            converted_pred.append([convert(t) for t in sample_pred])

        return converted_gt, converted_pred

    def _score(self, gt: List[List[Tuple]], pred: List[List[Tuple]], print_results: bool = False, cls_metric=False):
        assert len(gt) == len(pred)
        # import pdb;pdb.set_trace()

        gt_flat = []
        pred_flat = []
        types = set()

        for (sample_gt, sample_pred) in zip(gt, pred):
            union = set()
            if cls_metric:
                union.update(sample_gt)
                loc_gt = list(map(lambda x: (x[0], x[1]), sample_gt))
                sample_loc_true_pred = list(filter(lambda x: (x[0], x[1]) in loc_gt, sample_pred))
                union.update(sample_loc_true_pred)
            else:
                union.update(sample_gt)
                union.update(sample_pred)

            for s in union:
                if s in sample_gt:
                    t = s[2]
                    gt_flat.append(t.index)
                    types.add(t)
                else:
                    gt_flat.append(-1)

                if s in sample_pred:
                    t = s[2]
                    pred_flat.append(t.index)
                    types.add(t)
                else:
                    pred_flat.append(-1)
        metrics = self._compute_metrics(gt_flat, pred_flat, types, print_results)
        return metrics

    def _compute_metrics(self, gt_all, pred_all, types, print_results: bool = False):
        labels = [t.index for t in types]
        per_type = prfs(gt_all, pred_all, labels=labels, average=None)
        micro = prfs(gt_all, pred_all, labels=labels, average='micro')[:-1]
        macro = prfs(gt_all, pred_all, labels=labels, average='macro')[:-1]
        total_support = sum(per_type[-1])

        if print_results:
            self._print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], types)

        return [m * 100 for m in micro + macro]

    def _print_results(self, per_type: List, micro: List, macro: List, types: List):
        columns = ('type', 'precision', 'recall', 'f1-score', 'support')

        row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
        self._log(row_fmt % columns)

        metrics_per_type = []
        for i, t in enumerate(types):
            metrics = []
            for j in range(len(per_type)):
                metrics.append(per_type[j][i])
            metrics_per_type.append(metrics)

        for m, t in zip(metrics_per_type, types):
            self._log(row_fmt % self._get_row(m, t.short_name))

        self._log('')

        # micro
        self._log(row_fmt % self._get_row(micro, 'micro'))

        # macro
        self._log(row_fmt % self._get_row(macro, 'macro'))

    def _get_row(self, data, label):
        row = [label]
        for i in range(len(data) - 1):
            row.append("%.2f" % (data[i] * 100))
        row.append(data[3])
        return tuple(row)

    def _convert_example(self, doc: Document, gt: List[Tuple], pred: List[Tuple],
                         include_aspect_types: bool, to_html):
        # encoding = doc.encoding
        tokens = doc.tokens

        gt, pred = self._convert_by_setting([gt], [pred], include_aspect_types=include_aspect_types, include_score=True)
        gt, pred = gt[0], pred[0]

        # get micro precision/recall/f1 scores
        if gt or pred:
            pred_s = [p[:3] for p in pred]  # remove score
            precision, recall, f1 = self._score([gt], [pred_s])[:3]
        else:
            # corner case: no ground truth and no predictions
            precision, recall, f1 = [100] * 3

        cls_scores = [p[3] for p in pred]
        pred = [p[:3] for p in pred]
        union = set(gt + pred)

        # true positives
        tp = []
        # false negatives
        fn = []
        # false positives
        fp = []

        for s in union:
            type_verbose = s[2].verbose_name

            if s in gt:
                if s in pred:
                    cls_score = cls_scores[pred.index(s)]
                    tp.append((to_html(s, tokens), type_verbose, cls_score))
                else:
                    fn.append((to_html(s, tokens), type_verbose, -1))
            else:
                cls_score = cls_scores[pred.index(s)]
                fp.append((to_html(s, tokens), type_verbose, cls_score))

        tp = sorted(tp, key=lambda p: p[2], reverse=True)
        fp = sorted(fp, key=lambda p: p[2], reverse=True)

        phrases = []
        for token in tokens:
            phrases.append(token.phrase)
        text = " ".join(phrases)

        # text = self._prettify(self._text_encoder.decode(encoding))
        text = self._prettify(text)
        return dict(text=text, tp=tp, fn=fn, fp=fp, precision=precision, recall=recall, f1=f1, length=len(doc.tokens))

    def _aspect_to_html(self, aspect: Tuple, tokens: List[Token]):
        start, end = aspect[:2]
        aspect_type = aspect[2].verbose_name

        tag_start = ' <span class="aspect">'
        tag_start += '<span class="type">%s</span>' % aspect_type

        # ctx_before = self._text_encoder.decode(encoding[:start])
        # e1 = self._text_encoder.decode(encoding[start:end])
        # ctx_after = self._text_encoder.decode(encoding[end:])

        ctx_before = ""
        ctx_after = ""
        e1 = ""
        for i in range(start):
            ctx_before += tokens[i].phrase
            if i != start - 1:
                ctx_before += " "
        for i in range(end + 1, len(tokens)):
            ctx_after += tokens[i].phrase
            if i != (len(tokens) - 1):
                ctx_after += " "
        for i in range(start, end + 1):
            e1 += tokens[i].phrase
            if i != end:
                e1 += " "

        html = ctx_before + tag_start + e1 + '</span> ' + ctx_after
        html = self._prettify(html)

        return html

    def _prettify(self, text: str):
        text = text.replace('_start_', '').replace('_classify_', '').replace('<unk>', '').replace('⁇', '')
        text = text.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')
        return text

    def _store_examples(self, examples: List[Dict], file_path: str, template: str):
        template_path = os.path.join(SCRIPT_PATH, 'templates', template)

        # read template
        with open(os.path.join(SCRIPT_PATH, template_path)) as f:
            template = jinja2.Template(f.read())

        # write to disc
        template.stream(examples=examples).dump(file_path)
