#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import shuffle
import time
import nb_helper

import sys
sys.path.append('data/portuguese/')
from dataset import Dataset

from sklearn.naive_bayes import BernoulliNB

SUBSETS_COUNT = 50
TRAIN_SUBSETS_COUNT = int(0.8 * SUBSETS_COUNT)
TEST_SUBSETS_COUNT = SUBSETS_COUNT - TRAIN_SUBSETS_COUNT

d = Dataset()
custom_features = d.custom_features(d.full_set, nb_helper.get_unigrams)

corpus_features = set([feature for phrase_features in custom_features.values() for feature in phrase_features])
generated_subsets = [subset for subset in d.get_subsets(SUBSETS_COUNT)]
shuffle(generated_subsets) # randomize subsets

train_set = {phrase for subset in generated_subsets[:TRAIN_SUBSETS_COUNT] for phrase in subset.items()}
X = []
y = []
for phrase, intent in train_set:
    X.append(nb_helper.format_features_for_classifier(custom_features[phrase], corpus_features))
    y.append(intent)

start_time = time.time()

classifier = BernoulliNB()
classifier.fit(X, y)

test_set = {phrase for subset in generated_subsets[-TEST_SUBSETS_COUNT:] for phrase in subset.items()}
accepted = 0
for phrase, intent in test_set:
    predicted = classifier.predict([nb_helper.format_features_for_classifier(custom_features[phrase], corpus_features)])
    if predicted[0] == intent:
        accepted += 1

    print(phrase.encode('utf-8'), ' [predicted: ', predicted[0], '; correct: ', intent, ']')

print('TIME: ', (time.time() - start_time))

total= len(test_set)
print('SCORE: ', accepted, '/', total, ' (', accepted/(1.0 * total), ')')
