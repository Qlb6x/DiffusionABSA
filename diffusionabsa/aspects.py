from collections import OrderedDict
import json
from typing import List
from torch.utils import data
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import IterableDataset as IterableTorchDataset
from diffusionabsa import sampling
# import sampling
import torch.distributed as dist


class AspectType:
    def __init__(self, identifier, index, short_name, verbose_name):
        self._identifier = identifier
        self._index = index
        self._short_name = short_name
        self._verbose_name = verbose_name

    @property
    def identifier(self):
        return self._identifier

    @property
    def index(self):
        return self._index

    @property
    def short_name(self):
        return self._short_name

    @property
    def verbose_name(self):
        return self._verbose_name

    def __int__(self):
        return self._index

    def __eq__(self, other):
        if isinstance(other, AspectType):
            return self._identifier == other._identifier
        return False

    def __hash__(self):
        return hash(self._identifier)

    def __str__(self) -> str:
        return self._identifier + "=" + self._verbose_name


class Token:
    def __init__(self, tid: int, index: int, span_start: int, span_end: int, phrase: str):
        self._tid = tid  # ID within the corresponding dataset
        self._index = index  # original token index in document

        self._span_start = span_start  # start of token span in document (inclusive)
        self._span_end = span_end  # end of token span in document (inclusive)
        self._phrase = phrase

    @property
    def index(self):
        return self._index

    @property
    def span_start(self):
        return self._span_start

    @property
    def span_end(self):
        return self._span_end

    @property
    def span(self):
        return self._span_start, self._span_end

    @property
    def phrase(self):
        return self._phrase

    def __eq__(self, other):
        if isinstance(other, Token):
            return self._tid == other._tid
        return False

    def __hash__(self):
        return hash(self._tid)

    def __str__(self):
        return self._phrase

    def __repr__(self):
        return self._phrase


class TokenSpan:
    def __init__(self, tokens):
        self._tokens = tokens

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    def __getitem__(self, s):
        if isinstance(s, slice):
            return TokenSpan(self._tokens[s.start:s.stop:s.step])
        else:
            return self._tokens[s]

    def __iter__(self):
        return iter(self._tokens)

    def __len__(self):
        return len(self._tokens)

    def __str__(self) -> str:
        return " ".join([str(t) for t in self._tokens])

    def __repr__(self) -> str:
        return str(self)


class Aspect:
    def __init__(self, eid: int, aspect_type: AspectType, tokens: List[Token], phrase: str):
        self._eid = eid  # ID within the corresponding dataset

        self._aspect_type = aspect_type

        self._tokens = tokens
        self._phrase = phrase

    def as_tuple(self):
        return self.span_start, self.span_end, self._aspect_type

    def as_tuple_token(self):
        return self._tokens[0].index, self._tokens[-1].index, self._aspect_type

    @property
    def aspect_type(self):
        return self._aspect_type

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    @property
    def span_token(self):
        # print(self._tokens[0].index, self._tokens[-1].index)
        return self._tokens[0].index, self._tokens[-1].index

    @property
    def phrase(self):
        return self._phrase

    def __eq__(self, other):
        if isinstance(other, Aspect):
            return self._eid == other._eid
        return False

    def __hash__(self):
        return hash(self._eid)

    def __str__(self):
        return self._phrase + f" -> {self.span_token}-> {self.aspect_type.identifier}"

    def __repr__(self) -> str:
        return str(self)


class Document:
    def __init__(self, doc_id: int, tokens: List[Token], aspects: List[Aspect], encoding: List[int], seg_encoding: List[int], pos_indices, dep, dep_label_indices):
        self._doc_id = doc_id  # ID within the corresponding dataset

        self._tokens = tokens
        self._aspects = aspects

        # byte-pair document encoding including special tokens ([CLS] and [SEP])
        self._encoding = encoding
        self._seg_encoding = seg_encoding
        self._pos_indices = pos_indices
        self._dep = dep
        self._dep_label_indices = dep_label_indices

    @property
    def doc_id(self):
        return self._doc_id

    @property
    def aspects(self):
        return self._aspects

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    @property
    def dep_label_indices(self):
        return self._dep_label_indices

    @property
    def dep(self):
        return self._dep

    @property
    def pos_indices(self):
        return self._pos_indices

    @property
    def encoding(self):
        return self._encoding

    @property
    def char_encoding(self):
        return self._char_encoding

    @property
    def seg_encoding(self):
        return self._seg_encoding

    @encoding.setter
    def encoding(self, value):
        self._encoding = value

    @char_encoding.setter
    def char_encoding(self, value):
        self._char_encoding = value

    @seg_encoding.setter
    def seg_encoding(self, value):
        self._seg_encoding = value

    def __str__(self) -> str:
        raw_document = str(self.tokens)
        raw_aspects = str(self.aspects)
        return raw_document + " => " + raw_aspects

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other):
        if isinstance(other, Document):
            return self._doc_id == other._doc_id
        return False

    def __hash__(self):
        return hash(self._doc_id)


class BatchIterator:
    def __init__(self, aspects, batch_size, order=None, truncate=False):
        self._aspects = aspects
        self._batch_size = batch_size
        self._truncate = truncate
        self._length = len(self._aspects)
        self._order = order

        if order is None:
            self._order = list(range(len(self._aspects)))

        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._truncate and self._i + self._batch_size > self._length:
            raise StopIteration
        elif not self._truncate and self._i >= self._length:
            raise StopIteration
        else:
            aspects = [self._aspects[n] for n in self._order[self._i:self._i + self._batch_size]]
            self._i += self._batch_size
            return aspects


class Dataset(TorchDataset):
    TRAIN_MODE = 'train'
    EVAL_MODE = 'eval'

    def __init__(self, label, dataset_path, aspect_types, tokenizer=None, repeat_gt_aspects=None):
        self._label = label
        self._aspect_types = aspect_types
        self._mode = Dataset.TRAIN_MODE
        self._tokenizer = tokenizer
        self._path = dataset_path

        self._repeat_gt_aspects = repeat_gt_aspects

        self._documents = OrderedDict()
        self._aspects = OrderedDict()

        # current ids
        self._doc_id = 0
        self._eid = 0
        self._tid = 0
        self._iid = 0

    def iterate_documents(self, batch_size, order=None, truncate=False):
        return BatchIterator(self.documents, batch_size, order=order, truncate=truncate)

    def create_token(self, idx, span_start, span_end, phrase) -> Token:
        token = Token(self._tid, idx, span_start, span_end, phrase)
        self._tid += 1
        return token

    def create_document(self, tokens, aspect_mentions, doc_encoding, seg_encoding, jpos_indices, jdep, jdep_indices) -> Document:
        document = Document(self._doc_id, tokens, aspect_mentions, doc_encoding, seg_encoding, jpos_indices, jdep, jdep_indices)
        self._documents[self._doc_id] = document
        self._doc_id += 1

        return document

    def create_aspect(self, aspect_type, tokens, phrase) -> Aspect:
        mention = Aspect(self._eid, aspect_type, tokens, phrase)
        self._aspects[self._eid] = mention
        self._eid += 1
        return mention

    def __len__(self):
        return len(self._documents)

    def __getitem__(self, index: int):
        doc = self._documents[index]

        if self._mode == Dataset.TRAIN_MODE:
            return sampling.create_train_sample(doc, self._repeat_gt_aspects)
        else:
            return sampling.create_eval_sample(doc)

    def switch_mode(self, mode):
        self._mode = mode

    @property
    def label(self):
        return self._label

    @property
    def input_reader(self):
        return self._input_reader

    @property
    def documents(self):
        return list(self._documents.values())

    @property
    def aspects(self):
        return list(self._aspects.values())

    @property
    def document_count(self):
        return len(self._documents)

    @property
    def aspect_count(self):
        return len(self._aspects)


class DistributedIterableDataset(IterableTorchDataset):
    TRAIN_MODE = 'train'
    EVAL_MODE = 'eval'

    def __init__(self, label, path, aspect_types, input_reader, tokenizer=None, repeat_gt_aspects=None):
        self._label = label
        self._path = path
        self._aspect_types = aspect_types
        self._mode = Dataset.TRAIN_MODE
        self._tokenizer = tokenizer
        self._input_reader = input_reader
        self._local_rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        # print(self._local_rank, self._world_size)

        self._repeat_gt_aspects = repeat_gt_aspects

        self.statistic = json.load(open(path.split(".")[0] + "_statistic.json"))

        # current ids
        self._doc_id = 0
        self._eid = 0
        self._tid = 0

    def create_token(self, idx, span_start, span_end, phrase) -> Token:
        token = Token(self._tid, idx, span_start, span_end, phrase)
        self._tid += 1
        return token

    def create_document(self, tokens, aspect_mentions, doc_encoding, seg_encoding) -> Document:
        document = Document(self._doc_id, tokens, aspect_mentions, doc_encoding, seg_encoding)
        self._doc_id += 1
        return document

    def create_aspect(self, aspect_type, tokens, phrase) -> Aspect:
        mention = Aspect(self._eid, aspect_type, tokens, phrase)
        self._eid += 1
        return mention

    def parse_doc(self, path):
        inx = 0
        worker_info = data.get_worker_info()
        num_workers = 1
        worker_id = 0
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        offset = 0
        mod = 1
        if self._local_rank != -1:
            offset = self._local_rank * num_workers + worker_id
            mod = self._world_size * num_workers
        with open(self._path, encoding="utf8") as rf:
            for line in rf:
                if inx % mod == offset:
                    doc = json.loads(line)
                    doc = self._input_reader._parse_document(doc, self)
                    if doc is not None:
                        if self._mode == Dataset.TRAIN_MODE:
                            yield sampling.create_train_sample(doc, self._repeat_gt_aspects)
                        else:
                            yield sampling.create_eval_sample(doc)
                inx += 1  # maybe imblance

    def _get_stream(self, path):
        # return itertools.cycle(self.parse_doc(path))
        return self.parse_doc(path)

    def __iter__(self):
        return self._get_stream(self._path)

    def switch_mode(self, mode):
        self._mode = mode

    @property
    def label(self):
        return self._label

    @property
    def input_reader(self):
        return self._input_reader

    @property
    def document_count(self):
        return self.statistic["document_count"]

    @property
    def aspect_count(self):
        return self.statistic["aspect_count"]
