# imports
import os
import json
import math
import re

from tqdm import tqdm
from collections import Counter
from typing import Tuple

WORD = re.compile(r"\w+")
DATA_DIR = "C:/Users/Evan de Kruif/PycharmProjects/expred/src/dataset/eraser"


def load_data(dataset: str):
    with open(os.path.join(DATA_DIR, dataset + ".jsonl"), 'r') as jsonFile:
        json_data = list(jsonFile)

    claim_list = dict([])
    evidence_list = dict([])

    for list_item in json_data:
        item = json.loads(list_item)
        claim_list[item['annotation_id']] = item['query']
        temp_list = []
        for evidence_item in item['evidences']:
            temp_list.append(evidence_item[0]['text'])
        evidence_list[item['annotation_id']] = temp_list

    return claim_list, evidence_list


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def get_cosine_string(str1, str2):
    vec1 = text_vector(str1)
    vec2 = text_vector(str2)
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_vector(text):
    words = WORD.findall(text)
    return Counter(words)


def get_influences(input_claim: str, k: int, dataset: str, annotations):
    claim_list, evidence_list = load_data(dataset)
    results = []
    vector1 = text_vector(input_claim)
    for item in tqdm(claim_list):
        vector2 = text_vector(claim_list[item])
        cosine = get_cosine(vector1, vector2)
        results.append((cosine, item))
        for item_ev in evidence_list[item]:
            vector2 = text_vector(item_ev)
            cosine = get_cosine(vector1, vector2)
            results.append((cosine, item))

    results.sort()
    results.reverse()

    return results[0:k]
