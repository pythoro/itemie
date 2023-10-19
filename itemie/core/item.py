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
        typ: str = "raw",
    ):
        self._name = name
        self._key = key
        self._converter = converter  # or convert class
        self._raw = None
        self._typ = typ

    @property
    def name(self):
        return self._name

    @property
    def key(self):
        return self._key

    def _get_values(self, data: np.ndarray, typ: str = None) -> np.ndarray:
        typ = self._typ if typ is None else typ
        if typ == "raw":
            return data
        else:
            raise ValueError("typ not known: " + str(typ))

    def _fit(self, series: np.ndarray) -> np.ndarray:
        if self._converter is None:
            out = series
        else:
            out = self._converter.convert(series)
        return out

    def fit(self, df: pd.DataFrame) -> np.ndarray:
        series = df[self._key]
        return self._fit(series)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.fit(df)
        self._raw = raw
        return raw

    def values(self, typ: str = None) -> np.ndarray:
        return self._get_values(self._raw, typ=typ)

    def __call__(self, typ: str = None) -> np.ndarray:
        return self._get_values(self._raw, typ=typ)

    def raw(self) -> np.ndarray:
        return self._raw


class NumericItem(BaseItem):
    """A numeric item class"""

    @property
    def mean(self):
        return np.mean(self._raw)

    @property
    def std(self):
        return np.std(self._raw)

    def _standardise(self, data: np.ndarray) -> np.ndarray:
        return (data - np.nanmean(data)) / np.nanstd(data)

    def _normalise(self, data: np.ndarray) -> np.ndarray:
        out = data - np.min(data)
        return out / np.max(out)

    def _get_values(self, data: np.ndarray, typ: str = None) -> np.ndarray:
        typ = self._typ if typ is None else typ
        if typ == "standardised":
            return self._standardise(data)
        elif typ == "normalised":
            return self._normalise(data)
        else:
            return super()._get_values(data, typ=typ)

    def standarised(self) -> np.ndarray:
        return self._get_values(self._raw, typ="standardised")

    def normalised(self) -> np.ndarray:
        return self._get_values(self._raw, typ="normalised")
