import json
from abc import abstractmethod, ABC
from collections import OrderedDict
from logging import Logger
from typing import List
import numpy as np

from tqdm import tqdm
from transformers import AutoTokenizer

from diffusionabsa.aspects import Dataset, AspectType, Aspect, Document, DistributedIterableDataset


class BaseInputReader(ABC):
    def __init__(self, types_path: str, tokenizer: AutoTokenizer, logger: Logger = None, repeat_gt_aspects=None):
        types = json.load(open(types_path), object_pairs_hook=OrderedDict)  # aspect + relation types

        self._aspect_types = OrderedDict()
        self._idx2aspect_type = OrderedDict()
        self._idx2relation_type = OrderedDict()

        # aspects
        # add 'None' aspect type
        none_aspect_type = AspectType('None', 0, 'None', 'No Aspect')
        self._aspect_types['None'] = none_aspect_type
        self._idx2aspect_type[0] = none_aspect_type

        # specified aspect types
        for i, (key, v) in enumerate(types['aspects'].items()):
            aspect_type = AspectType(key, i + 1, v['short'], v['verbose'])
            self._aspect_types[key] = aspect_type
            self._idx2aspect_type[i + 1] = aspect_type

        self._datasets = dict()

        self._tokenizer = tokenizer
        self._logger = logger
        self._repeat_gt_aspects = repeat_gt_aspects

        self._vocabulary_size = tokenizer.vocab_size
        self._context_size = -1

    @abstractmethod
    def read(self, datasets):
        pass

    def get_dataset(self, label):
        return self._datasets[label]

    def get_aspect_type(self, idx) -> AspectType:
        aspect = self._idx2aspect_type[idx]
        return aspect

    def _calc_context_size(self, datasets):
        sizes = [-1]

        for dataset in datasets:
            if isinstance(dataset, Dataset):
                for doc in dataset.documents:
                    sizes.append(len(doc.encoding))

        context_size = max(sizes)
        return context_size

    def _log(self, text):
        if self._logger is not None:
            self._logger.info(text)

    @property
    def datasets(self):
        return self._datasets

    @property
    def aspect_types(self):
        return self._aspect_types

    @property
    def aspect_type_count(self):
        return len(self._aspect_types)

    @property
    def vocabulary_size(self):
        return self._vocabulary_size

    @property
    def context_size(self):
        return self._context_size

    def __str__(self):
        string = ""
        for dataset in self._datasets.values():
            string += "Dataset: %s\n" % dataset
            string += str(dataset)

        return string

    def __repr__(self):
        return self.__str__()


class JsonInputReader(BaseInputReader):
    def __init__(self, types_path: str, tokenizer: AutoTokenizer, logger: Logger = None, repeat_gt_aspects=None):
        super().__init__(types_path, tokenizer, logger, repeat_gt_aspects)

    def read(self, dataset_paths):
        for dataset_label, dataset_path in dataset_paths.items():
            if dataset_path.endswith(".jsonl"):
                dataset = DistributedIterableDataset(dataset_label, dataset_path, self._aspect_types, tokenizer=self._tokenizer, repeat_gt_aspects=self._repeat_gt_aspects)
                print(dataset[0])
                self._datasets[dataset_label] = dataset
            else:
                dataset = Dataset(dataset_label, dataset_path, self._aspect_types, tokenizer=self._tokenizer, repeat_gt_aspects=self._repeat_gt_aspects)
                self._parse_dataset(dataset_path, dataset, dataset_label)
                self._datasets[dataset_label] = dataset

        self._context_size = self._calc_context_size(self._datasets.values())

    def _parse_dataset(self, dataset_path, dataset, dataset_label):
        documents = json.load(open(dataset_path))
        for document in tqdm(documents, desc="Parse dataset '%s'" % dataset_label):
            self._parse_document(document, dataset)

    def _parse_document(self, doc, dataset: Dataset) -> Document:
        ltokens = None
        rtokens = None

        jtokens = doc['tokens']
        jaspects = doc['aspects']
        jpos_indices = doc['pos_indices']
        jdep = doc['dep']
        jdep_indices = doc['dep_label_indices']
        if "ltokens" in doc:
            ltokens = doc["ltokens"]

        if "rtokens" in doc:
            rtokens = doc["rtokens"]

        doc_tokens, doc_encoding, seg_encoding = self._parse_tokens(jtokens, ltokens, rtokens, dataset)
        if len(doc_encoding) > 512:
            self._log(f"Document {doc['orig_id']} len(doc_encoding) = {len(doc_encoding)} > 512, Ignored!")
            return None

        aspects = self._parse_aspects(jaspects, doc_tokens, dataset)

        document = dataset.create_document(doc_tokens, aspects, doc_encoding, seg_encoding, jpos_indices, jdep, jdep_indices)

        return document

    def _parse_tokens(self, jtokens, ltokens, rtokens, dataset):
        '''
        jtokens : 就是tokens
        ltokens : 是上一句
        '''
        doc_tokens = []
        special_tokens_map = self._tokenizer.special_tokens_map
        # cls_token -> [101]
        doc_encoding = [self._tokenizer.convert_tokens_to_ids(special_tokens_map['cls_token'])]

        seg_encoding = [1]

        if ltokens is not None and len(ltokens) > 0:
            for token_phrase in ltokens:
                token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
                doc_encoding += token_encoding
                seg_encoding += [1] * len(token_encoding)
            doc_encoding += [self._tokenizer.convert_tokens_to_ids(special_tokens_map['sep_token'])]
            seg_encoding += [1]

        for i, token_phrase in enumerate(jtokens):
            token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)

            span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding) - 1)
            # 这里在构造每个span
            token = dataset.create_token(i, span_start, span_end, token_phrase)

            doc_tokens.append(token)
            doc_encoding += token_encoding
            seg_encoding += [1] * len(token_encoding)

        if rtokens is not None and len(rtokens) > 0:
            # 插入一个sep_token -> [102]
            doc_encoding += [self._tokenizer.convert_tokens_to_ids(special_tokens_map['sep_token'])]
            seg_encoding += [1]
            for token_phrase in rtokens:
                token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
                # if len(doc_encoding) + len(token_encoding) > 512:
                #     break
                doc_encoding += token_encoding
                seg_encoding += [1] * len(token_encoding)

        return doc_tokens, doc_encoding, seg_encoding

    def _parse_aspects(self, jaspects, doc_tokens, dataset) -> List[Aspect]:
        aspects = []

        for aspect_idx, jaspect in enumerate(jaspects):
            aspect_type = self._aspect_types[jaspect['type']]
            start, end = jaspect['start'], jaspect['end']

            # create aspect mention  (exclusive)
            tokens = doc_tokens[start:end]
            phrase = " ".join([t.phrase for t in tokens])
            aspect = dataset.create_aspect(aspect_type, tokens, phrase)
            aspects.append(aspect)

        return aspects
