#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import helper
from random import shuffle

from sklearn.naive_bayes import MultinomialNB

TEST_SIZE = 120
NGRAMS = 1

data = json.loads(open('intents.json').read().decode('utf-8'))
corpus_features = helper.get_corpus_features(data, NGRAMS)

all_phrases = data.keys()
shuffle(all_phrases)

classifier = MultinomialNB()

train_set = all_phrases[0:TEST_SIZE]
X = []
y = []
for phrase in train_set:
    X.append(helper.extract_features(phrase, corpus_features, NGRAMS))
    y.append(data[phrase])

classifier.fit(X, y)

test_set = all_phrases[TEST_SIZE:]
accepted = total = 0
for phrase in test_set:
    phrase_features = [helper.extract_features(phrase, corpus_features)]
    predicted = classifier.predict(phrase_features)
    print phrase, predicted, data[phrase]
    total += 1
    if predicted[0] == data[phrase]:
        accepted += 1

print total, accepted, accepted/(1.0 * total)
