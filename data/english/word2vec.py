#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('data/english/')
from dataset import Dataset

from nltk.util import ngrams
from gensim.models import Word2Vec

def get_tokens(phrase):
    return phrase.split()

d = Dataset()
custom_features = d.custom_features(d.full_set, get_tokens)

model = Word2Vec(custom_features.values(), size=25, min_count=1, workers=2)
model.save('./word2vec.model')
