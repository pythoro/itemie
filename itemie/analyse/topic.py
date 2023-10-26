# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 21:10:58 2023

@author: Reuben
"""


import pandas as pd
import numpy as np
import spacy 
# from spacy import displacy

import gensim
from gensim.corpora import Dictionary
from gensim import models
from gensim.models import LdaModel, CoherenceModel, LsiModel, HdpModel

import pyLDAvis
import pyLDAvis.gensim_models


nlp = spacy.load('en_core_web_sm')


class Topics:
    def __init__(self, lemmatize=True, tfidf=True):
        self._lemmatize = lemmatize
        self._tfidf = tfidf
    
    def setup(self, item, num_topics=5, data=None):
        data = item.values('default') if data is None else data
        text = '\n'.join(data)
        doc = nlp(text)
        texts = self._get_texts(doc)
        dictionary, corpus = self._make_corpus(texts)
        self._num_topics = num_topics
        model = self._make_topic_model(corpus, num_topics, dictionary)
        self._dictionary = dictionary
        self._corpus = corpus
        self._model = model

    def _get_texts(self, doc):
        texts, article = [], []
        for word in doc:
            if word.text != '\n' and not word.is_stop and not word.is_punct\
                                 and not word.like_num and word.text != 'I':
                to_append = word.lemma_ if self._lemmatize else word.text
                article.append(to_append)
            if word.text == '\n':
                texts.append(article)
                article = []
        texts = self._make_bigrams(texts)
        return texts

    def _make_bigrams(self, texts):
        bigram = gensim.models.phrases.Phrases(texts)
        texts = [bigram[line] for line in texts]
        texts = [bigram[line] for line in texts]
        return texts

    def _make_corpus(self, texts):
        dictionary = Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        if self._tfidf:
            tfidf = models.TfidfModel(corpus)
            corpus = tfidf[corpus]
        return dictionary, corpus

    def _make_topic_model(self, corpus, num_topics, dictionary):
        #lsi_model = LsiModel(corpus=corpus_tfidf, num_topics=10, id2word=dictionary)
        #lsi_model.show_topics(num_topics=5)
        
        #hdp_model = HdpModel(corpus=corpus_tfidf, id2word=dictionary)
        #hdp_model.show_topics()[:5]
        
        lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        # lda_model.show_topics()
        return lda_model

    def view(self, fname):
        lda_model =self._model
        corpus = self._corpus
        dictionary = self._dictionary
        prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
        pyLDAvis.save_html(prepared, fname)

    def strengths_arr(self):
        model = self._model
        topics = model.get_document_topics(self._corpus)
        topics_arr = np.array([topic[1] for text in topics for topic in text])
        topics_arr = topics_arr.reshape((-1, self._num_topics))
        return topics_arr
        
    def strongest(self):
        topics_arr = self.strengths_arr()
        return np.argmax(topics_arr, axis=1)
    
