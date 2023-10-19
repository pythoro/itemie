# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:14:10 2023

@author: Reuben
"""

import numpy as np

TEXTBLOB_LOADED = True
try:
    from textblob import TextBlob
except ModuleNotFoundError:
    TEXTBLOB_LOADED = False


class BaseConverter:
    def convert(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Replacer(BaseConverter):
    def __init__(self, keyvals: dict):
        self._keyvals = keyvals

    def convert(self, data) -> np.ndarray:
        dct = self._keyvals
        return np.array([dct[k] for k in data])


class Function(BaseConverter):
    def __init__(self, func):
        self._func = func

    def convert(self, data) -> np.ndarray:
        return np.array([self._func(val) for val in data])


class AutoCorrect(BaseConverter):
    def __init__(self):
        if not TEXTBLOB_LOADED:
            raise ModuleNotFoundError("TextBlob package required.")

    def _autocorrect(self, text: str):
        tb = TextBlob(text)
        return tb.correct()

    def convert(self, data) -> np.ndarray:
        return np.array([self._autocorrect(val) for val in data])


class Pipeline(BaseConverter):
    def __init__(self, *converters: list[BaseConverter]):
        self._converters = converters

    def convert(self, data: np.ndarray) -> np.ndarray:
        transformed = data
        for converter in self._converters:
            transformed = converter.convert(transformed)
        return transformed
