#!/usr/bin/env python
# -*- coding: utf-8 -*-

def format_features_for_classifier(phrase_features, corpus_features):
    list_of_boolean_features = []
    for feature in corpus_features:
        list_of_boolean_features.append(feature in phrase_features)
    return list_of_boolean_features
