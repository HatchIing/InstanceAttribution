import json

from src.custom_utils.knn import get_cosine, get_cosine_string

fetch_list = ['37435',
              '8288',
              '67986',
              '48651',
              '47183']

fname_train = "C:/Users/Evan de Kruif/PycharmProjects/expred/src/dataset/eraser/train.jsonl"
fname_questionnaire = "C:/Users/Evan de Kruif/PycharmProjects/expred/src/dataset/eraser/QUESTIONNAIRE THINGS.txt"


def fetch_formatted_entries(fname, fetch):
    f = open(fname)
    fetched = []
    for line in f:
        line = json.loads(line)
        ann_id = line['annotation_id']
        if ann_id in fetch:
            claim = line['query']
            label = line['classification']
            if label == 'SUPPORTS':
                label = 'TRUE'
            else:
                label = 'FALSE'
            fetched.append(ann_id + ": " + claim + " [" + label + "]")
    f.close()
    results = []
    for f in fetch:
        for x in fetched:
            if f in x:
                results.append(x)

    return results


def fetch_option_annotations(fname):
    lines = []
    annotations = []

    with open(fname) as f:
        for line in f:
            check = line[0]
            if check.isnumeric():
                split = line.split()
                annotations.append(split[0][:-1])
        f.close()

    return annotations


def compare_sim(inputs):
    ft, fknn, tknn = 0, 0, 0
    fastif = inputs[0:5]
    tracin = inputs[5:10]
    knn = inputs[10:15]

    for f, t in zip(fastif, tracin):
        ft += get_cosine_string(f, t)

    ft = ft / 5

    for f, k in zip(fastif, knn):
        fknn += get_cosine_string(f, k)

    fknn = fknn / 5

    for t, knn in zip(tracin, knn):
        tknn += get_cosine_string(t, knn)

    tknn = tknn / 5

    return ft, fknn, tknn


annotation = fetch_option_annotations(fname_questionnaire)
questions = fetch_formatted_entries(fname_train, annotation)

wozniak = questions[0:15]
ripon = questions[15:30]
kesha = questions[30:45]
vietnam = questions[45:60]
shadowhunters = questions[60:75]
illinois = questions[75:90]
kungfupanda = questions[90:105]
bowen = questions[105:120]
camden = questions[120:135]
silver = questions[135:150]

# print(len(kungfupanda))
# print('\n'.join(kungfupanda))

print(compare_sim(wozniak))
print(compare_sim(ripon))
print(compare_sim(kesha))
print(compare_sim(vietnam))
print(compare_sim(shadowhunters))
print(compare_sim(illinois))
print(compare_sim(kungfupanda))
print(compare_sim(bowen))
print(compare_sim(camden))
print(compare_sim(silver))
