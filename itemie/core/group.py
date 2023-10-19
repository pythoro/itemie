# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 19:32:05 2023

@author: Reuben
"""

import numpy as np
import pandas as pd

from .item import BaseItem


class BaseGroup:
    def __init__(self, items=None, assemble_as='array'):
        self._items = {}
        assert assemble_as in ['array', 'dict'], 'invalid assemble_as arg.'
        self._assemble_as = assemble_as
        if items is not None:
            self.add(items)
    
    @property
    def names(self):
        return [item.name for item in self.items]
    
    @property
    def items(self):
        return list(self._items.values())
    
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
    
    def _assemble(self, values, assemble_as=None):
        assemble_as = self._assemble_as if assemble_as is None else assemble_as
        if assemble_as == 'array':
            return np.array(values).T
        elif assemble_as == 'dict':
            names = self.names
            return {name: series for name, series in zip(names, values)}
        else:
            raise ValueError('assemble_as argument not understood: ' + 
                             assemble_as)
    
    def fit(self, df: pd.DataFrame) -> np.ndarray:
        values = [item.fit(df) for item in self.items]
        return self._assemble(values)
        
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        values = [item.fit_transform(df) for item in self.items]
        return self._assemble(values)
    
    def values(self, typ=None):
        values = [item.values(typ=typ) for item in self.items]
        return self._assemble(values)
    
    def __call__(self, typ=None):
        values = [item.values(typ=typ) for item in self.items]
        return self._assemble(values)


class NumericGroup(BaseGroup):
    def means(self, typ: str=None) -> np.ndarray:
        values = self.values(typ=typ, assemble_as='array')
        return np.mean(values, axis=1)

    def item_means(self):
        return np.array([item.mean for item in self.items])

    def item_stds(self):
        return np.array([item.std for item in self.items])
        