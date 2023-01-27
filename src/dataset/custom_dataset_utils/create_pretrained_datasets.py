import os
from collections import OrderedDict
from itertools import chain
from typing import Tuple, List

from src.dataset.custom_dataset_utils.fetch_doc import get_wikipedia_page
from src.expred import BertTokenizerWithSpans, annotations_from_jsonl, decorate_with_docs_ids
from src.expred.dataset.dataloader import MTLDataLoader
from src.expred.dataset.eraser_utils import convert_annotations_to_examples, Example, load_documents


def load_eraser_data(data_dir: str, dataset: str, merge_evidences: bool) -> List[Example]:
    train = annotations_from_jsonl(os.path.join(data_dir, dataset + ".jsonl"))

    train = list(map(decorate_with_docs_ids, train))

    docids = set(chain.from_iterable(map(lambda ann: ann.docids, chain(train))))

    removal = []

    for id in docids:
        if not os.path.exists(os.path.join(data_dir,'docs', id)):
            print(id, " does not exist. Fetching.")
            status = get_wikipedia_page(id, data_dir)
            if status == 'failed':
                removal.append(id)

    for f in removal:
        if f in docids:
            docids.remove(f)

    if not removal:
        print("All docs were retrieved successfully.")
    else:
        print("Removed: \n", removal)

    docs = load_documents(data_dir, docids)

    train = convert_annotations_to_examples(train, docs, merge_evidences)

    return train


def create_pretrained_datasets(data_dir: str, dataset: str):
    exclude_annotations = {}

    raw_data_train = load_eraser_data(data_dir, dataset, True)
    raw_data_train = list(filter(lambda example: example.ann_id not in exclude_annotations, raw_data_train))

    label_id_to_name = ['REFUTES', 'SUPPORTS']
    label_name_to_id = OrderedDict({k: v for v, k in enumerate(label_id_to_name)})
    print("label_name_to_id: ", label_name_to_id)

    tokenizer = BertTokenizerWithSpans.from_pretrained('bert-base-uncased')

    max_length = 512
    batch_size = 8

    print("raw_data_train :", raw_data_train.__len__())
    print("label_name_to_id :", label_name_to_id.__len__())

    train_data = MTLDataLoader(
        raw_data_train,
        label_name_to_id,
        tokenizer,
        max_length,
        batch_size,
        shuffle=True,
        num_workers=0,
        cache_fname=None,
    )

    return train_data
