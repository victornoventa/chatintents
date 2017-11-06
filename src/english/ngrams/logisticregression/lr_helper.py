#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nltk.util import ngrams

def format_features_for_classifier(phrase_features, corpus_features):
    list_of_boolean_features = []
    for feature in corpus_features:
        list_of_boolean_features.append(feature in phrase_features)
    return list_of_boolean_features

def get_unigrams(phrase):
    return get_ngrams(phrase, 1)

def get_bigrams(phrase):
    return get_ngrams(phrase, 2)

def get_trigrams(phrase):
    return get_ngrams(phrase, 3)

def get_ngrams(phrase, ngrams_size):
    phrase_ngrams = ngrams(phrase.split(), ngrams_size)

    phrase_ngrams_as_str = []
    for phrase_ngram in phrase_ngrams:
        phrase_ngrams_as_str.append(' '.join(phrase_ngram))

    return phrase_ngrams_as_str
