import os
import pandas as pd
import utils
from sklearn.metrics import cohen_kappa_score, f1_score

def calculate_cohens_kappa(folder1, folder2):
    examples1, labels1 = utils.process_cnll_files(folder1, "compressed")
    examples2, labels2 = utils.process_cnll_files(folder2, "compressed")

    kappa_scores = []
    f1_scores = []
    for lbl1, lbl2 in zip(labels1, labels2):
        if len(lbl1) != len(lbl2):
            continue
        kappa = cohen_kappa_score(lbl1, lbl2)
        f1 = f1_score(lbl1, lbl2, average='macro')
        kappa_scores.append(kappa)
        f1_scores.append(f1)

    return kappa_scores, f1_scores

folder1 = '../corpus/gpt_annotated_ground_truth/human_annotated'
folder2 = '../corpus/gpt_annotated_ground_truth/gpt_annotated'
kappa_scores, f1_scores = calculate_cohens_kappa(folder1, folder2)

for kappa in kappa_scores:
    print(f"Cohen's kappa for file: {kappa}")
print(sum(kappa_scores) / len(kappa_scores))

for f1 in f1_scores:
    print(f"F1 score for file: {f1}")
print(sum(f1_scores) / len(f1_scores))