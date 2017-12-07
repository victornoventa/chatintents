#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import shuffle
import numpy as np
import nn_helper

import sys
sys.path.append('data/english/')
from dataset import Dataset

from sklearn.neural_network import MLPClassifier

SUBSETS_COUNT = 100
TRAIN_SUBSETS_COUNT = int(0.8 * SUBSETS_COUNT)
TEST_SUBSETS_COUNT = SUBSETS_COUNT - TRAIN_SUBSETS_COUNT

d = Dataset(w2v_model_name = 'word2vec_size25_window1')
custom_features = d.custom_features(d.full_set, nn_helper.get_tokens)

generated_subsets = [subset for subset in d.get_subsets(SUBSETS_COUNT)]
shuffle(generated_subsets) # randomize subsets

train_set = {phrase for subset in generated_subsets[:TRAIN_SUBSETS_COUNT] for phrase in subset.items()}
X = []
y = []
for phrase, intent in train_set:
    X.append(nn_helper.format_features_for_classifier(custom_features[phrase], d.word_vectors))
    y.append(intent)

classifier = MLPClassifier()
classifier.fit(X, y)

test_set = {phrase for subset in generated_subsets[-TEST_SUBSETS_COUNT:] for phrase in subset.items()}
accepted = 0
for phrase, intent in test_set:
    predicted = classifier.predict([nn_helper.format_features_for_classifier(custom_features[phrase], d.word_vectors)])
    if predicted[0] == intent:
        accepted += 1
    
    print(phrase.encode('utf-8'), ' [predicted: ', predicted[0], '; correct: ', intent, ']')

total = len(test_set)
print('SCORE: ', accepted, '/', total, ' (', accepted/(1.0 * total), ')')
