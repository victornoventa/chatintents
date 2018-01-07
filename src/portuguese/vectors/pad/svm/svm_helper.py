#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

W2V_SIZE = 25
MAX_PHRASE_FEATURES_COUNT = 0

def format_features_for_classifier(phrase_features, vectors):
    features = np.concatenate([vectors[feature] for feature in phrase_features])
    features.resize(W2V_SIZE * MAX_PHRASE_FEATURES_COUNT)
    return features

def get_tokens(phrase):
    tokens = phrase.split()
    tokens_count = len(tokens)
    global MAX_PHRASE_FEATURES_COUNT
    if tokens_count > MAX_PHRASE_FEATURES_COUNT:
        MAX_PHRASE_FEATURES_COUNT = tokens_count
    return tokens
