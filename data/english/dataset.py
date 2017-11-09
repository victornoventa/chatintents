#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, json, math, re, nltk, gensim
from itertools import islice
from unicodedata import normalize

MY_PATH = os.path.dirname(os.path.realpath(__file__))

class Dataset:
    intents = [
        'AddToPlaylist',
        'BookRestaurant',
        'GetWeather',
        'PlayMusic',
        'RateBook',
        'SearchCreativeWork',
        'SearchScreeningEvent'
    ]

    full_set = {}
    default_features = {}

    stopwords = nltk.corpus.stopwords.words('english')
    word_vectors = []

    def __init__(self):
        def load_phrases_and_intents(from_intent_path):
            with open(from_intent_path) as intent_data:
                phrases = json.load(intent_data)

            for phrase in phrases[intent]:
                phrase_as_str = get_data_as_str(phrase['data'])

                self.full_set[phrase_as_str] = intent
                self.default_features[phrase_as_str] = get_data_default_features_as_list(phrase['data'])

        def get_data_as_str(data):
            data_as_str = ''
            for feature in data:
                data_as_str = data_as_str + feature['text']
            return data_as_str.strip()

        def get_data_default_features_as_list(data):
            return self.get_features_as_list([feature['text'] for feature in data])

        for intent in self.intents:
            intent_train_path = MY_PATH + '/nlu-benchmark/2017-06-custom-intent-engines/' + intent + '/train_' + intent + '.json'
            load_phrases_and_intents(intent_train_path)

            intent_train_full_path = MY_PATH + '/nlu-benchmark/2017-06-custom-intent-engines/' + intent + '/train_' + intent + '_full.json'
            load_phrases_and_intents(intent_train_full_path)

            intent_validate_path = MY_PATH + '/nlu-benchmark/2017-06-custom-intent-engines/' + intent + '/validate_' + intent + '.json'
            load_phrases_and_intents(intent_validate_path)

        w2v_model_path = MY_PATH + '/word2vec.model'
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
