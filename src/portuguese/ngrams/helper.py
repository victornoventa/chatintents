#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import nltk
from unicodedata import normalize
from sets import Set
from nltk.util import ngrams

def extract_features(phrase, corpus_features, ngrams_size = 1):
    phrase_ngrams = get_ngrams(phrase, ngrams_size)

    features = []
    for ngram in corpus_features:
        features.append(ngram in phrase_ngrams)

    return features

def tokenize(phrase):
    return remove_special_characters(phrase).lower().split()

def remove_special_characters(txt):
    return re.sub('[^a-zA-Z0-9 \\\]', '', normalize('NFKD', txt).encode('ASCII','ignore').decode('ASCII'))

def get_corpus_features(data, ngrams_size = 1):
    stopwords = nltk.corpus.stopwords.words('portuguese')
    features = Set()

    for phrase in data.keys(): # iterate through phrases read from file
        phrase_ngrams = get_ngrams(phrase, ngrams_size)

        for ngram in phrase_ngrams:
            if ngram not in list(features) + stopwords:
                features.add(ngram)

    return features

def get_ngrams(phrase, ngrams_size = 1):
    phrase_ngrams = ngrams(tokenize(phrase), ngrams_size)

    phrase_ngrams_as_str = []
    for phrase_ngram in phrase_ngrams:
        phrase_ngrams_as_str.append(' '.join(phrase_ngram))

    return phrase_ngrams_as_str
