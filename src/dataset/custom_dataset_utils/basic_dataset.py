import random

import numpy as np
import tqdm
from typing import Any
from torch.utils.data import DataLoader, Dataset

from src.custom_utils.fever_dataloader import *
from src.expred import ExpredConfig, ExpredInput
from src.expred.tokenizer.tokenizer import *


class BasicDataset(Dataset):
    def __init__(self, claims, docs, labels, ann_ids, ):
        self.claims = claims
        self.docs = docs
        self.labels = labels
        self.ann_ids = ann_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.claims[index], self.docs[index], self.labels[index], self.ann_ids[index]


def basic_load_data(fname):
    f = open(fname)
    ids = []
    claims = []
    evidences = []
    docs = []
    labels = []
    for line in f:
        line = json.loads(line)
        ann_id = line['annotation_id']
        evidence = []
        for x in line['evidences']:
            x = x[0]
            evidence.append(x['text'])
        claim = line['query']
        label = line['classification']
        doc = line['docids'][0]
        evidences.append(evidence)
        docs.append(doc)
        ids.append(ann_id)
        claims.append(claim)
        labels.append(label)
    f.close()
    return claims, evidences, labels, docs, ids


def trimmed_load_data(fname, include, k):
    f = open(fname)
    ids = []
    claims = []
    evidences = []
    docs = []
    labels = []
    annotations = get_included_annotations(fname, include, k)
    for line in f:
        line = json.loads(line)
        ann_id = line['annotation_id']
        if ann_id in annotations:
            claim = line['query']
            evidence = []
            for x in line['evidences']:
                x = x[0]
                evidence.append(x['text'])
            label = line['classification']
            doc = line['docids'][0]
            evidences.append(evidence)
            docs.append(doc)
            ids.append(ann_id)
            claims.append(claim)
            labels.append(label)
    f.close()
    return claims, evidences, labels, docs, ids, annotations


def get_included_annotations(fname, include, k):
    f = open(fname)
    include_supported = []
    include_refuted = []
    for line in f:
        line = json.loads(line)
        ann_id = line['annotation_id']
        claim: str = line['query']
        label: str = line['classification']

        for evidence in line['evidences']:
            evidence = evidence[0]
            claim = claim + " " + evidence['text']

        if any(word in claim for word in include):
            if label == 'SUPPORTS':
                include_supported.append(ann_id)
            else:
                include_refuted.append(ann_id)
    f.close()
    print(len(include_supported), len(include_refuted))
    min_len = min(len(include_supported), len(include_refuted))
    ret = []
    if min_len > 0:
        ret.extend(random.sample(include_supported, min_len))
        ret.extend(random.sample(include_refuted, min_len))

    if len(ret) > 25:
        ret = ret[0:24]

    ret.extend(get_random_annotations(fname, k))
    return ret


def get_random_annotations(fname, k):
    f = open(fname)
    annotations = []
    for line in f:
        line = json.loads(line)
        ann_id = line['annotation_id']
        annotations.append(ann_id)
    f.close()
    ret = random.sample(annotations, k)
    return ret


def test_preprocess(data):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tok_ip = np.zeros((len(data), 128), dtype="int32")
    for pos, text in tqdm.tqdm(enumerate(data)):
        tok = tokenizer.tokenize(text[0])
        if len(tok) > 128:
            tok = tok[:127] + ["[SEP]"]
        pad_len = 128 - len(tok)
        tok = tokenizer.convert_tokens_to_ids(tok) + [0] * pad_len

        tok_ip[pos] = tok

    return tok_ip


def collate_fn(batch):
    tokenizer = BertTokenizerWithSpans.from_pretrained('bert-base-uncased')
    expred_config = ExpredConfig(
        pretrained_dataset_name='custom_dataset_utils',
        base_dataset_name='custom_dataset_utils',
        device='cuda',
        load_from_pretrained=True)
    collate_input = batch[0]
    queries = [collate_input[0]]
    docs = []
    for x in collate_input[1]:
        docs.append(x.split())
    labels = [collate_input[2]]
    ann_ids = [collate_input[3]]
    print(collate_input[0], collate_input[2])
    expred_input = ExpredInput(
        queries=queries,
        docs=[docs[0]],
        labels=labels,
        config=expred_config,
        ann_ids=ann_ids,
        span_tokenizer=tokenizer)
    expred_input.preprocess()
    expred_input.to('cuda')

    return expred_input, ann_ids


def basic_create_dataloader(data_dir: str, dataset: str, include, k) -> DataLoader:
    data_dir = os.path.join(data_dir, dataset + ".jsonl")
    claims, evidences, labels, docs, ids, annotations = trimmed_load_data(data_dir, include, k)
    labels = np.array(labels)
    print("Claims: ", len(claims))
    print("Evidence: ", len(evidences))
    print("Labels: ", len(labels))
    print("Ids: ", len(ids))
    train_dataset = BasicDataset(claims, evidences, labels, ids)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1, collate_fn=collate_fn, num_workers=0)
    return train_loader, annotations
