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

    def __str__(self):
        return self._str()

    def _str(self, prefix=""):
        return prefix + self.__class__.__name__

class AsType(BaseConverter):
    def __init__(self, typ):
        self._typ = typ
    
    def convert(self, data: np.ndarray) -> np.ndarray:
        return np.array(data.astype(self._typ))


class Replace(BaseConverter):
    def __init__(self, keyvals: dict, group_other=True, other_val='other'):
        self._keyvals = keyvals
        self._group_other = group_other
        self._other_val = other_val

    def convert(self, data) -> np.ndarray:
        dct = self._keyvals
        out = []
        for k in data:
            if k in dct:
                out.append(dct[k])
            else:
                if self._group_other:
                    out.append(self._other_val)
                else:
                    out.append(k)
        return np.array(out)


class StrReplace(BaseConverter):
    def __init__(self, keyvals: dict):
        self._keyvals = keyvals

    def convert(self, data) -> np.ndarray:
        dct = self._keyvals
        out = []
        for s in data:
            s2 = s
            for substring, replacement in dct.items():
                s2 = (s.replace(substring, replacement)) 
            out.append(s2)
        return np.array(out)


class Function(BaseConverter):
    def __init__(self, func):
        self._func = func

    def convert(self, data) -> np.ndarray:
        return np.array([self._func(val) for val in data])


class VectorisedFunction(BaseConverter):
    def __init__(self, func):
        self._func = func

    def convert(self, data) -> np.ndarray:
        return self._func(data)


class AutoCorrect(BaseConverter):
    def __init__(self):
        if not TEXTBLOB_LOADED:
            raise ModuleNotFoundError("TextBlob package required.")

    def _autocorrect(self, text: str):
        tb = TextBlob(text)
        return str(tb.correct())

    def convert(self, data) -> np.ndarray:
        return np.array([self._autocorrect(val) for val in data])


class Lower(BaseConverter):
    def convert(self, data: np.ndarray) -> np.ndarray:
        return np.array([s.lower() for s in data])


class Splitter(BaseConverter):
    def __init__(self, splits: list[str]):
        self._splits = splits

    def _split(self, text: str):
        s = text
        for split in self._splits:
            s = s.replace(split, "|||")
        lst = s.split("|||")
        stripped = [t.strip() for t in lst]
        return stripped

    def convert(self, data) -> list[list]:
        return [self._split(val) for val in data]


class Pipeline(BaseConverter):
    def __init__(self, *converters: BaseConverter):
        self._converters = converters

    def __str__(self):
        return self._str()

    def _str(self, prefix=""):
        lst = [c._str(prefix) for c in self._converters]
        return prefix + ', '.join(lst)

    def convert(self, data: np.ndarray) -> np.ndarray:
        transformed = data
        for converter in self._converters:
            transformed = converter.convert(transformed)
        return transformed
