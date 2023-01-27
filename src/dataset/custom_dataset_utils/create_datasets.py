# https://github.com/aditya5558/BERT-FEVER-Task/blob/master/SOTA%20Reimplementation/claim_verification.py

import faiss
import numpy as np
import tqdm
from typing import Any
from torch.utils.data import DataLoader, Dataset

from src.custom_utils.fever_dataloader import *
from src.expred.tokenizer import BertTokenizerWithMapping
from src.expred.tokenizer.tokenizer import *


class EvidenceDataset(Dataset):
    def __init__(self, tok_ip, sent_ip, pos_ip, masks, y):
        self.tok_ip = tok_ip
        self.sent_ip = sent_ip
        self.pos_ip = pos_ip
        self.masks = masks
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        tok_ip_tensor = torch.tensor(self.tok_ip[index]).to('cuda')
        self_ip_tensor = torch.tensor(self.sent_ip[index]).to('cuda')
        pos_ip_tensor = torch.tensor(self.pos_ip[index]).to('cuda')
        masks_ip_tensor = torch.tensor(self.masks[index]).to('cuda')
        y_tensor = torch.tensor(self.y[index]).to('cuda')
        return tok_ip_tensor, self_ip_tensor, pos_ip_tensor, masks_ip_tensor, y_tensor


def load_data(fname):
    label_dict = {}
    label_dict['UNK'] = -1
    label_dict['NOT ENOUGH INFO'] = 0
    label_dict['SUPPORTS'] = 1
    label_dict['REFUTES'] = 2
    f = open(fname)
    data = []
    claim_ids = []
    labels = []
    predicted_evidence = []
    for line in f:
        line = json.loads(line)
        evidence = line['evidences'][0][0]
        sentence = ["[CLS]" + line['query'] + "[SEP]", evidence['docid'] + " " + evidence['text'] + "[SEP]"]
        label = label_dict[line['classification']]
        data.append(sentence)
        labels.append(label)
        claim_ids.append(line['annotation_id'])
        predicted_evidence.append([evidence['docid'], line['annotation_id']])
    f.close()
    return data, labels, claim_ids, predicted_evidence


def preprocess(data):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tok_ip = np.zeros((len(data), 128), dtype="int32")
    sent_ip = np.zeros((len(data), 128), dtype="int8")
    pos_ip = np.zeros((len(data), 128), dtype="int8")
    masks = np.zeros((len(data), 128), dtype="int8")
    for pos, text in tqdm.tqdm(enumerate(data)):
        tok0 = tokenizer.tokenize(text[0])
        tok1 = tokenizer.tokenize(text[1])
        tok = tok0 + tok1
        if len(tok) > 128:
            tok = tok[:127] + ["[SEP]"]
        pad_len = 128 - len(tok)
        tok_len = len(tok)
        tok0_len = len(tok0)
        tok = tokenizer.convert_tokens_to_ids(tok) + [0] * pad_len
        pos_val = range(128)
        sent = [0] * tok0_len + [1] * (tok_len - tok0_len) + [0] * pad_len
        mask = [1] * tok_len + [0] * pad_len

        tok_ip[pos] = tok
        pos_ip[pos] = pos_val
        masks[pos] = mask
        sent_ip[pos] = sent

    masks = masks[:, None, None, :]
    return tok_ip, sent_ip, pos_ip, masks


def collate_fn(batch):
    print("Batch print:", batch)
    # change shit into expredinput
    batch.to('cuda')


def create_custom_dataloader(data_dir: str) -> DataLoader:
    data, labels, ids, predicted_evidence = load_data(data_dir)
    print(len(data), len(labels))
    tok_ip, sent_ip, pos_ip, masks = preprocess(data)
    labels = np.array(labels)

    train_dataset = EvidenceDataset(tok_ip, sent_ip, pos_ip, masks, labels)
    train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, num_workers=0)

    return train_loader


def create_expred_datasets(data_dir: str) -> Tuple[List[Annotation], List[Annotation], List[Annotation]]:
    train, val, test = load_datasets(data_dir)
    tokenizer = BertTokenizerWithMapping.from_pretrained("bert-base-uncased")
    indexed_train, indexed_val, indexed_test = [tokenizer.encode_annotations(data) for data in [train, val, test]]
    # indexed_train.Datasets.add_faiss_index()
    return indexed_train, indexed_val, indexed_test
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    #
    # train_data, dev_data, test_data = get_dataloaders(tokenizer, raw_data, max_length, batch_size, label_name_to_id,
    #                                                   output_dir, cache_dir_train)

    #return train_data, dev_data, test_data



