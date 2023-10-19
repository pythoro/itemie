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
        self._converter = converter  # or convert class
        self._raw = None
        self._typ = typ

    @property
    def name(self):
        return self._name

    @property
    def key(self):
        return self._key

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
        self._raw = series
        self._transformed = converted
        return self._transform_hook(converted)

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

    def __call__(self, typ: str = None) -> np.ndarray:
        return self._get_values(typ=typ)

    def raw(self) -> np.ndarray:
        return self._raw


class NumericItem(BaseItem):
    """A numeric item class"""

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    def _fit_hook(self, corrected):
        self._mean = np.nanmean(corrected)
        self._std = np.nanstd(corrected)
        self._max = np.max(corrected)
        self._min = np.min(corrected)

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
