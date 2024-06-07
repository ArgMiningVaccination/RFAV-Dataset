import argparse
from glob import glob
import os

from sklearn import metrics

columns = {
    "Reason": 1,
    "Stance": 2,
    "ScientificAuthority": 3,
    "Compressed": 2
}

compression = {
    1: 1,
    2: 1,
    3: 2,
    4: 3,
    5: 3
}

def list_of_strings(arg):
    return arg.split(',')

parser = argparse.ArgumentParser()
parser.add_argument(
    "annotators",
    type=list_of_strings,
    default='',
    help="list of folders dir where the annotators files are",
)

parser.add_argument(
    "category",
    type=str,
    default='Reason',
    choices=['Reason', 'Stance', 'ScientificAuthority', 'Compressed'],
    help="The agreement will be calculated for this category"
)

args = parser.parse_args()

def convert_labels_to_numbers(labels):
    if args.category == "Stance":
        return [int(label) if label != 'O' else 0 for label in labels]
    if args.category == "Compressed":
        return [compression[int(label)] if label != 'O' else 0 for label in labels]
    return [0 if label == "O" else 1 for label in labels]

def crossed_agreement(labels_per_example):
    agreements = {}
    f1_scores = {}
    precision_scores = {}
    recall_scores = {}
    sum_agreements = 0
    sum_f1_scores = 0
    sum_precision_scores = 0
    sum_recall_scores = 0
    avrg = 'macro' if args.category == "Stance" or args.category == "Compressed" else 'binary'
    first_example_len = len(labels_per_example[0])
    if all([len(lbl) == first_example_len for lbl in labels_per_example]):
        for i in range(len(labels_per_example)):
            for j in range(i + 1, len(labels_per_example)):
                annotator_i = convert_labels_to_numbers(labels_per_example[i])
                annotator_j = convert_labels_to_numbers(labels_per_example[j])
                if sum(annotator_j) == 0 and sum(annotator_i) == 0:
                    agreement = 1.0
                    f1_1 = 1.0
                    precision_1 = 1.0
                    recall_1 = 1.0
                    f1_2 = 1.0
                    precision_2 = 1.0
                    recall_2 = 1.0
                else:
                    agreement = metrics.cohen_kappa_score(annotator_i, annotator_j)
                    f1_1 = metrics.f1_score(annotator_i, annotator_j, average=avrg)
                    precision_1 = metrics.precision_score(annotator_i, annotator_j, average=avrg)
                    recall_1 = metrics.recall_score(annotator_i, annotator_j, average=avrg)
                    f1_2 = metrics.f1_score(annotator_j, annotator_i, average=avrg)
                    precision_2 = metrics.precision_score(annotator_j, annotator_i, average=avrg)
                    recall_2 = metrics.recall_score(annotator_j, annotator_i, average=avrg)
                agreements[(i, j)] = agreement
                f1_scores[(i, j)] = f1_1
                precision_scores[(i, j)] = precision_1
                recall_scores[(i, j)] = recall_1
                f1_scores[(j, i)] = f1_2
                precision_scores[(j, i)] = precision_2
                recall_scores[(j, i)] = recall_2
                sum_agreements += agreement
                sum_f1_scores += f1_1 + f1_2
                sum_precision_scores += precision_1 + precision_2
                sum_recall_scores += recall_1 + recall_2
        return agreements, sum_agreements / (len(labels_per_example) * (len(labels_per_example) - 1) / 2), f1_scores, sum_f1_scores / (len(labels_per_example) * (len(labels_per_example) - 1)), precision_scores, sum_precision_scores / (len(labels_per_example) * (len(labels_per_example) - 1)), recall_scores, sum_recall_scores / (len(labels_per_example) * (len(labels_per_example) - 1))
    else:
        return {}, 0, {}, 0, {}, 0, {}, 0

all_labels = {}
for annotator in args.annotators:
    all_labels_per_annotator = {}
    for file in glob(f"{annotator}/*"):
        if ".cnll" in file:
            labels_per_example = []
            for line in open(file):
                labels_per_example.append(line.split(" ")[columns[args.category]])
            all_labels_per_annotator[file.split("/")[-1].replace("ChatGptAnnotation", "")] = labels_per_example
    file_names = sorted(all_labels_per_annotator.keys())
    all_labels[annotator.split("/")[-1]] = all_labels_per_annotator

labels_per_annotator = {}
for file_name in file_names:
    labels_per_example = []
    for annotator in args.annotators:
        annotator_name = annotator.split("/")[-1]
        label_annotator = all_labels[annotator_name][file_name.replace("ChatGptAnnotation", "")]
        labels_per_example.append(label_annotator)
        if annotator_name not in labels_per_annotator:
            labels_per_annotator[annotator_name] = []
        labels_per_annotator[annotator_name].extend(label_annotator)

labels = []
for annotator in args.annotators:
        annotator_name = annotator.split("/")[-1]
        labels.append(labels_per_annotator[annotator_name])


agreements, avrg_agreement, f1_scores, avrg_f1, precision_scores, avrg_precision, recall_scores, avrg_recall = crossed_agreement(labels)
print("Agreements")
print(agreements)
print("\n")
print("Average agreement")
print(avrg_agreement)
print("\n")
print("F1 scores")
print(f1_scores)
print("\n")
print("Average F1 score")
print(avrg_f1)
print("\n")
print("Precision scores")
print(precision_scores)
print("\n")
print("Average precision")
print(avrg_precision)
print("\n")
print("Recall scores")
print(recall_scores)
print("\n")
print("Average recall")
print(avrg_recall)
print("\n")









