#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import shuffle
import lr_helper

import sys
sys.path.append('data/english/')
from dataset import Dataset

from sklearn.linear_model import LogisticRegression

SUBSETS_COUNT = 100
TRAIN_SUBSETS_COUNT = int(0.8 * SUBSETS_COUNT)
TEST_SUBSETS_COUNT = SUBSETS_COUNT - TRAIN_SUBSETS_COUNT

d = Dataset()
classifier = LogisticRegression()

corpus_features = set([feature for phrase_features in d.default_features.values() for feature in phrase_features])
generated_subsets = [subset for subset in d.get_subsets(SUBSETS_COUNT)]
shuffle(generated_subsets) # randomize subsets

train_set = {phrase for subset in generated_subsets[:TRAIN_SUBSETS_COUNT] for phrase in subset.items()}
X = []
y = []
for phrase, intent in train_set:
    X.append(lr_helper.format_features_for_classifier(d.default_features[phrase], corpus_features))
    y.append(intent)

classifier.fit(X, y)

test_set = {phrase for subset in generated_subsets[-TEST_SUBSETS_COUNT:] for phrase in subset.items()}
accepted = 0
for phrase, intent in test_set:
    predicted = classifier.predict([lr_helper.format_features_for_classifier(d.default_features[phrase], corpus_features)])
    if predicted[0] == intent:
        accepted += 1
    else:
        print(phrase, ' [predicted: ', predicted[0], '; correct: ', intent, ']')

total = len(test_set)
print('SCORE: ', accepted, '/', total, ' (', accepted/(1.0 * total), ')')
