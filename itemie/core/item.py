# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 21:26:39 2023

@author: Reuben
"""

import numpy as np
import pandas as pd

from .convert import BaseConverter


class BaseItem:
    """A base item class"""

    def __init__(
        self,
        name: str,
        key: str,
        converter: BaseConverter = None,
        typ: str = "values",
    ):
        self._name = name
        self._key = key
        self.set_converter(converter)
        self._raw = None
        self._typ = typ

    @property
    def name(self):
        return self._name

    @property
    def key(self):
        return self._key

    def set_converter(self, converter):
        self._converter = converter

    def _get_values(self, typ: str = None) -> np.ndarray:
        typ = self._typ if typ is None else typ
        if typ == "raw":
            return self._raw
        elif typ == "values":
            return self._transformed
        else:
            raise ValueError("typ not known: " + str(typ))

    def _fit_hook(self, converted):
        pass

    def _convert(self, series: np.ndarray) -> np.ndarray:
        if self._converter is None:
            converted = series
        else:
            converted = self._converter.convert(series)
        return converted

    def _get_series(self,df: pd.DataFrame) -> np.ndarray:
        return df[self._key]

    def _fit_transform(self, df: pd.DataFrame, call) -> np.ndarray:
        series = self._get_series(df)
        converted = self._convert(series)
        if call == 'fit':
            return self._fit_hook(converted)
        elif call == 'fit_transform':
            self._fit_hook(converted)
        elif call == 'transform':
            pass
        else:
            raise ValueError("unknown call arg.: " + call)
        self._raw = series
        self._transformed = self._transform_hook(converted)
        return self._transformed

    def fit(self, df: pd.DataFrame) -> np.ndarray:
        return self._fit_transform(df, 'fit')

    def _transform_hook(self, transformed):
        return transformed

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self._fit_transform(df, 'transform')

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self._fit_transform(df, 'fit_transform')

    def values(self, typ: str = None) -> np.ndarray:
        return self._get_values(typ=typ)

    def raw(self) -> np.ndarray:
        return self._raw


class NumericItem(BaseItem):
    """A numeric item class"""

    def __init__(
        self,
        name: str,
        key: str,
        converter: BaseConverter = None,
        typ: str = "values",
        reverse_offset: float = None,
    ):
        super().__init__(name=name, key=key, converter=converter, typ=typ)
        self._reverse_offset = reverse_offset

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def sem(self):
        return self._sem

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def size(self):
        return len(self._transformed)

    def stats(self):
        labels = ['mean', 'std', 'min', 'max', 'sem', 'ci95']
        values = [self.mean, self.std, self.min, self.max, self.sem, self.sem * 1.95996]
        return labels, values

    def counts(self, as_int=True, as_percent=False):
        vals, counts = np.unique(self._transformed, return_counts=True)
        n = self.size / 100 if as_percent else 1
        def convert_nan(v):
            if np.isnan(v):
                return 'na'
            if as_int:
                return int(v)
            return v
        return {convert_nan(v): c / n for v, c in zip(vals, counts)}
    
    def _fit_hook(self, converted):
        if self._reverse_offset is not None:
            converted = -converted + self._reverse_offset
        self._mean = np.nanmean(converted)
        self._std = np.nanstd(converted)
        self._max = np.nanmax(converted)
        self._min = np.nanmin(converted)
        self._sem = np.nanstd(converted, ddof=1) / np.sqrt(np.size(converted))

    def _transform_hook(self, transformed):
        if self._reverse_offset is not None:
            transformed = -transformed + self._reverse_offset
        return transformed

    def _get_values(self, typ: str = None) -> np.ndarray:
        typ = self._typ if typ is None else typ
        if typ == "standardised":
            return self.standardised()
        elif typ == "normalised":
            return self.normalised()
        else:
            return super()._get_values(typ=typ)

    def standardised(self) -> np.ndarray:
        return (self._transformed - self._mean) / self._std

    def normalised(self) -> np.ndarray:
        out = self._transformed - self._min
        return out / np.max(out)


class PhraseCount(BaseItem):
    """A numeric item class"""
    
    def __init__(
        self,
        name: str,
        key: str,
        converter: BaseConverter = None,
        typ: str = "values",
        seq: list[str] = None,
    ):
        super().__init__(name=name, key=key, converter=converter, typ=typ)
        self._seq = seq

    @property
    def counts(self):
        return self._counts

    def _get_values(self, typ: str = None) -> np.ndarray:
        typ = self._typ if typ is None else typ
        if typ == 'counts':
            return self.counts
        else:
            return super()._get_values(typ=typ)

    def _count(self, converted):
        if self._seq is not None:
            counts = {k: 0 for k in self._seq}
        else:
            counts = {}
        all_phrases = []
        for obj in converted:
            if isinstance(obj, list):
                all_phrases.extend(obj)
            else:
                all_phrases.append(obj)
        for phrase in all_phrases:
            if phrase in counts:
                counts[phrase] += 1
            else:
                counts[phrase] = 0
        if self._seq is None:
            sorted_keys = list(counts.keys())
            sorted_keys.sort(key=lambda x: str(x))
            counts = {k: counts[k] for k in sorted_keys}
        return counts

    def _transform_hook(self, transformed):
        self._counts = self._count(transformed)
        return transformed

    def as_list(self):
        return list(self._transformed.keys())