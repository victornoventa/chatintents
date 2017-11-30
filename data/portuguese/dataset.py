#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, json, math, re, nltk, gensim
from itertools import islice
from unicodedata import normalize

MY_PATH = os.path.dirname(os.path.realpath(__file__))

class Dataset:
    intents = [
        'add',
        'bad',
        'bye',
        'cancel',
        'delete',
        'get',
        'greeting',
        'help',
        'ok',
        'saymyname',
        'thanks',
        'welcome'
    ]

    full_set = {}

    stopwords = nltk.corpus.stopwords.words('portuguese')
    word_vectors = []

    def __init__(self, w2v_model_name = 'word2vec_size25_window1'):
        def load_phrases_and_intents(from_intent_path):
            with open(from_intent_path, encoding='utf-8') as intent_data:
                self.full_set = json.load(intent_data)

        intents_by_phrase_path = MY_PATH + '/intents_by_phrase.json'
        load_phrases_and_intents(intents_by_phrase_path)

        w2v_model_path = MY_PATH + '/' + w2v_model_name + '.model'
        w2v_model = gensim.models.Word2Vec.load(w2v_model_path)
        self.word_vectors = w2v_model.wv
        del w2v_model

    def custom_features(self, custom_set, custom_extractor_function):
        custom_features = {}
        for phrase in custom_set:
            custom_features[phrase] = self.get_features_as_list(custom_extractor_function(phrase))
        return custom_features

    def get_subsets(self, subsets_count = 10):
        it = iter(self.full_set)
        set_len = len(self.full_set)
        chunk_size = int(math.ceil(set_len / subsets_count))

        for i in range(0, set_len, chunk_size):
            yield { k : self.full_set[k] for k in islice(it, chunk_size)}

    def normalizer(self, feature):
        def remove_special_characters(txt):
            return re.sub('[^a-zA-Z0-9 \\\]', '', normalize('NFKD', txt).encode('ASCII','ignore').decode('ASCII'))

        return remove_special_characters(feature).strip().lower()

    def get_features_as_list(self, features):
        return [self.normalizer(feature) for feature in features if self.normalizer(feature) not in self.stopwords + ['']]
