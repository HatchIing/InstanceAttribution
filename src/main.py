import os
from collections import OrderedDict
from datetime import datetime

from src.TracIn.main_tracin import tracin_run
from src.custom_utils.custom_outputs import output_save_results
from src.custom_utils.knn import get_influences
from src.dataset.custom_dataset_utils.basic_dataset import *
from src.expred import (seeding, Expred)
from src.fastif.experiments.influence_helpers import compute_influences_simplified

seeding(1234)
start_time = datetime.now()

claim = "Steve Wozniak designed homes."
evidence = [
        [
            "He primarily designed the 1977 Apple II , known as one of the first highly successful mass-produced microcomputers , while Jobs oversaw the development of its unusual case and Rod Holt developed the unique power supply .".split()]
    ]

k = 300

data_dir = "<SYS_DIR>/PycharmProjects/expred/src/dataset/eraser"
output_dir = "<SYS_DIR>/PycharmProjects/expred/src/custom_utils/outputs"
dataset = "train"

tokenizer = BertTokenizerWithSpans.from_pretrained('bert-base-uncased')

expred_config = ExpredConfig(
    pretrained_dataset_name='custom_dataset_utils',
    base_dataset_name='custom_dataset_utils',
    device='cuda',
    load_from_pretrained=True)

expred = Expred.from_pretrained(expred_config)
expred.eval()

# CREATE THE INPUT
expred_input = ExpredInput(
    queries=[claim],
    docs=evidence,
    labels=['REFUTES'],
    config=expred_config,
    ann_ids=["100030"],
    span_tokenizer=tokenizer)

# PREPROCESSING FOR FIRST INPUT
expred_input.preprocess()
expred_input.to('cuda:0')

inputs = expred_input

# CREATE DATALOADER
train, annotations = basic_create_dataloader(data_dir, dataset, include, k)

# FASTIF
influences1 = compute_influences_simplified(
    k=1,
    faiss_index=None,
    model=expred,
    inputs=inputs,
    train_dataset=train,
    use_parallel=False,
    s_test_damp=5e-3,
    s_test_scale=1e4,
    s_test_num_samples=len(train),
    device_ids=[1],
    precomputed_s_test=None,
    faiss_index_use_mean_features_as_query=False
)

# KNN
influences2 = get_influences(claim, 5, dataset, annotations)

#TRACIN
influences3 = tracin_run(expred, inputs, train)

end_time = datetime.now()

file_date = start_time.__str__().replace(':', '-').split('.')[0]

print("RUNTIME: ", end_time - start_time)

# HANDLE THE RESULTS
output_save_results(output_dir, file_date, annotations, influences1, influences2, influences3, claim)
