#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, re, ast

file_absolute_path = sys.argv[1]
intents_list = ast.literal_eval(sys.argv[2])

wrong_phrases_classifications = []
scores = []
with open(file_absolute_path) as f:
    lines =  [x.strip('\n') for x in f.readlines()]
    for line in lines:
        if line[:6] == 'SCORE:':
            scores.append(line)
        else:
            wrong_phrases_classifications.append(line)

predicted_classifications = []
correct_classifications = []
for classified_phrase in wrong_phrases_classifications:
    data = [x.split() for x in re.split(r"[\[\]]", classified_phrase)]
    if (len(data) > 1):
        predicted_classifications.append(data[1][1])
        correct_classifications.append(data[1][4])

from sklearn.metrics import confusion_matrix

print("EXTRACTED INFO")
print("\nconfusion_matrix")
print(confusion_matrix(correct_classifications, predicted_classifications, labels=intents_list))

accepted_scores = []
total_scores = []
for score in scores:
    data = score.split()
    accepted_scores.append(data[1])
    total_scores.append(data[3])

import numpy as np

a = np.array(accepted_scores, dtype=np.float)
b = np.array(total_scores, dtype=np.float)

means = a / b
print("\nmeans")
print(means)

print("\naverage")
print(np.average(means))

print("\nstd")
print(np.std(means))
