#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

W2V_SIZE = 50

def format_features_for_classifier(phrase_features, vectors):
    features = np.zeros(W2V_SIZE)
    for feature in phrase_features:
        features = np.add(features, vectors[feature])
    return features

def get_tokens(phrase):
    return phrase.split()
