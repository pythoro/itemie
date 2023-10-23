# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 19:32:05 2023

@author: Reuben
"""

import numpy as np
import pandas as pd

import warnings
from contextlib import suppress


from .item import BaseItem


class BaseGroup:
    def __init__(self, name, items=None):
        self._name = name
        self._items = {}
        self._all_items = {}
        if items is not None:
            self.add(items)

    def __str__(self):
        return self._str()

    def __repr__(self):
        return str(self)

    def _str(self, prefix=''):
        next_prefix = prefix + '  '
        return (
            prefix + self.__class__.__name__
            + " '"
            + self.name
            + "'\n"
            + "\n".join([item._str(next_prefix) for item in self.items])
        )

    @property
    def name(self):
        return self._name

    @property
    def names(self):
        return [item.name for item in self.items]

    @property
    def items(self):
        return list(self._items.values())

    @property
    def size(self):
        return self.items[0].size

    def __getitem__(self, key):
        return self.get(key)

    def get(self, key):
        return self._all_items[key]

    def stats(self, assemble_as="df"):
        labels, _ = self.items[0].stats()
        values = [item.stats()[1] for item in self.items]
        stats = self._assemble(values, assemble_as=assemble_as)
        return stats

    def _get_all_items(self):
        out = {self.name: self}
        out.update(self._all_items)
        return out

    def _add(self, item: BaseItem):
        self._items[item.name] = item
        self._all_items.update(item._get_all_items())

    def add(self, items):
        if isinstance(items, (tuple, list)):
            for item in items:
                self._add(item)
        elif isinstance(items, BaseItem):
            self._add(item)
        else:
            raise ValueError("Unknown type for item: " + str(type(item)))

    def raw_fitted_data(self, typ="df"):
        values = [item.raw_fitted for item in self.items]
        return self._assemble(values, typ)

    def converted_fitted_data(self, typ="df"):
        values = [item.converted_fitted for item in self.items]
        return self._assemble(values, typ)

    def raw_data(self, typ="df"):
        values = [item.raw for item in self.items]
        return self._assemble(values, typ)

    def converted_data(self, typ="df"):
        values = [item.converted for item in self.items]
        return self._assemble(values, typ)

    def _assemble(self, val_list, typ):
        if typ == "dict":
            return self._as_dict(val_list)
        elif typ == "array":
            return self._as_array(val_list)
        elif typ == "df":
            return self._as_df(val_list)

    def _as_dict(self, val_list):
        names = self.names
        return {name: series for name, series in zip(names, val_list)}

    def _as_array(self, val_list):
        return np.array(val_list).T

    def _as_df(self, val_list):
        names = self.names
        data = np.array(val_list).T
        df = pd.DataFrame(data, columns=names)
        df.index.name = "response"
        df.columns.name = "item"
        return df

    def fit(self, df: pd.DataFrame) -> np.ndarray:
        for item in self.items:
            item.fit(df)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        for item in self.items:
            item.transform(df)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        for item in self.items:
            item.fit_transform(df)

    def values(self, typ):
        raise ValueError(
            "Item " + self.name + " — typ not understood:" + str(typ)
        )

    def data_dict(
        self,
        typ: str = "default",
        match_size: bool = True,
    ) -> dict:
        dcts = [
            item.data_dict(typ=typ, match_size=match_size)
            for item in self.items
        ]
        # Now append it's own scale / summary
        with suppress(ValueError):
            values = self.values(typ)
            if match_size and len(values) != self.size:
                pass
            else:
                dcts.append({self.name: values})
        out = {}
        for dct in dcts:
            out.update(dct)
        return out

    def data_df(self, typ="default", match_size=True):
        dct = self.data_dict(typ=typ, match_size=match_size)
        df = pd.DataFrame.from_dict(dct)
        df.index.name = "response"
        df.columns.name = "item"
        return df


class Group(BaseGroup):
    pass


class NumericGroup(BaseGroup):
    def _means(self, values):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmean(values, axis=1)

    @property
    def raw(self):
        return self._means(self.raw_data("array"))

    @property
    def converted(self):
        return self._means(self.converted_data("array"))

    @property
    def standardised(self):
        return self._means([item.standardised for item in self.items])

    @property
    def normalised(self):
        return self._means([item.normalised for item in self.items])

    def means(self, typ: str = None) -> np.ndarray:
        values = self.converted_data("array")
        return np.nanmean(values, axis=1)

    def item_counts(self, as_int=True, as_percent=False):
        lst = []
        for item in self.items:
            dct = {"item": item.name}
            counts = item.counts(as_int, as_percent)
            dct.update(counts)
            lst.append(dct)
        return lst

    def item_means(self):
        return np.array([item.mean for item in self.items])

    def item_stds(self):
        return np.array([item.std for item in self.items])

    def values(self, typ):
        if typ == "raw":
            return self.raw
        elif typ == "converted":
            return self.converted
        elif typ in ["default", "standardised"]:
            return self.standardised
        elif typ == "normalised":
            return self.normalised
        else:
            return super().values(typ)
