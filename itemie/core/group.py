# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 19:32:05 2023

@author: Reuben
"""

import numpy as np
import pandas as pd

from .item import BaseItem


class BaseGroup:
    
    def __init__(self, name, items=None, assemble_as='array'):
        self._name = name
        self._items = {}
        assert assemble_as in ['array', 'dict'], 'invalid assemble_as arg.'
        self._assemble_as = assemble_as
        if items is not None:
            self.add(items)
    
    @property
    def name(self):
        return self._name
    
    @property
    def names(self):
        return [item.name for item in self.items]
    
    @property
    def items(self):
        return list(self._items.values())
    
    def __getitem__(self, key):
        return self._items[key]
    
    def get(self, key):
        return self._items[key]
    
    def stats(self, assemble_as='df'):
        labels, _ = self.items[0].stats()
        values = [item.stats()[1] for item in self.items]
        stats = self._assemble(values, assemble_as=assemble_as, index=labels)
        return stats
    
    def _add(self, item: BaseItem):
        self._items[item.name] = item
    
    def add(self, items):
        if isinstance(items, (tuple, list)):
            for item in items:
                self._add(item)
        elif isinstance(items, BaseItem):
            self._add(item)
        else:
            raise ValueError('Unknown type for item: ' + str(type(item)))
    
    def _assemble(self, values, assemble_as=None, index=None):
        assemble_as = self._assemble_as if assemble_as is None else assemble_as
        if assemble_as == 'array':
            return np.array(values).T
        elif assemble_as == 'dict':
            names = self.names
            return {name: series for name, series in zip(names, values)}
        elif assemble_as == 'df':
            names = self.names
            data = np.array(values).T
            df = pd.DataFrame(data, index=index, columns=names)
            return df
        else:
            raise ValueError('assemble_as argument not understood: ' + 
                             assemble_as)
        
    def fit(self, df: pd.DataFrame) -> np.ndarray:
        values = [item.fit(df) for item in self.items]
        return self._assemble(values)
        
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        values = [item.transform(df) for item in self.items]
        return self._assemble(values)
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        values = [item.fit_transform(df) for item in self.items]
        return self._assemble(values)
    
    def all_values(self, typ=None, assemble_as=None):
        values = [item.values(typ=typ) for item in self.items]
        return self._assemble(values, assemble_as=assemble_as)
    


class NumericGroup(BaseGroup):
    
    def means(self, typ: str=None) -> np.ndarray:
        values = self.values(typ=typ, assemble_as='array')
        return np.mean(values, axis=1)

    def item_counts(self, as_int=True, as_percent=False):
        lst = []
        for item in self.items:
            dct = {'item': item.name}
            counts = item.counts(as_int, as_percent)
            dct.update(counts)
            lst.append(dct)
        return lst

    def item_percents(self, as_int=True):
        lst = []
        for item in self.items:
            dct = {'item': item.name}
            counts = item.counts(as_int)
            dct.update(counts)
            lst.append(dct)
        return lst

    def values(self, typ=None):
        values = self.all_values(typ=typ, assemble_as='array')
        return np.nanmean(values, axis=1)

    def item_means(self):
        return np.array([item.mean for item in self.items])

    def item_stds(self):
        return np.array([item.std for item in self.items])
        