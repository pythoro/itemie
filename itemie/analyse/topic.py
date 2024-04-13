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


def add_stopwords(stopwords):
    for stopword in stopwords:
        lexeme = nlp.vocab[stopword]
        lexeme.is_stop = True

extra_stopwords = ['nan', '$']
add_stopwords(extra_stopwords)


class Topics:
    def __init__(self, lemmatize=True, tfidf=True):
        self._lemmatize = lemmatize
        self._tfidf = tfidf
                
    def setup(self, item, num_topics=5, data=None, random_state=0):
        data = item.values('default') if data is None else data
        lin_data, mapping = item.linearised(data)
        texts = self._get_texts(lin_data)
        dictionary, corpus = self._make_corpus(texts)
        self._num_topics = num_topics
        self._texts = texts
        model = self._make_topic_model(corpus, num_topics, dictionary, random_state)
        self._dictionary = dictionary
        self._corpus = corpus
        self._model = model
        self._mapping = mapping

    @property
    def mapping(self):
        return self._mapping

    def _get_texts(self, lin_data):
        texts = []
        for text in nlp.pipe(lin_data):
            response = []
            for word in text:
                if word.text != '\n' and not word.is_stop and not word.is_punct\
                                 and not word.like_num and word.text != 'I':
                     to_append = word.lemma_ if self._lemmatize else word.text
                     response.append(to_append)
            texts.append(response)
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

    def _make_topic_model(self, corpus, num_topics, dictionary, random_state):
        #lsi_model = LsiModel(corpus=corpus_tfidf, num_topics=10, id2word=dictionary)
        #lsi_model.show_topics(num_topics=5)
        
        #hdp_model = HdpModel(corpus=corpus_tfidf, id2word=dictionary)
        #hdp_model.show_topics()[:5]
        
        lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,
                             random_state=random_state)
        # lda_model.show_topics()
        return lda_model

    def get_prepared(self):
        lda_model = self._model
        corpus = self._corpus
        dictionary = self._dictionary
        prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
        return prepared

    def to_panel(self):
        prepared = self.get_prepared()
        return pyLDAvis.display(prepared)

    def to_html(self, fname=None):
        prepared = self.get_prepared()
        if fname is None:
            return pyLDAvis.prepared_data_to_html(prepared)
        else:
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
    
    def topic_words(self, as_str=False, n=10):
        lst = []
        for i in range(self._num_topics):
            tt = self._model.get_topic_terms(i, n)
            lst.append([self._dictionary[pair[0]] for pair in tt])
        if as_str:
            return [', '.join(tt) for tt in lst]
        return lst
    
    def get_coherence(self, typ='c_v'):
        coherence_model_lda = CoherenceModel(
           model=self._model, texts=self._texts, dictionary=self._dictionary, coherence=typ)
        return coherence_model_lda.get_coherence()