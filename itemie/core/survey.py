# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 18:41:34 2023

@author: Reuben
"""

import pandas as pd
import numpy as np

from .item import BaseItem


class Survey:
    def __init__(self, name):
        self._name = name
        self._items = {}
        
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
    
    def _add(self, item: BaseItem):
        self._items[item.name] = item
        try:
            subitems = item.items
            for subitem in subitems:
                self._add(subitem)
        except AttributeError:
            pass
            
    def add(self, items):
        if isinstance(items, (tuple, list)):
            for item in items:
                self._add(item)
        elif isinstance(items, BaseItem):
            self._add(item)
        else:
            raise ValueError('Unknown type for item: ' + str(type(item)))
            
    def fit(self, df: pd.DataFrame) -> None:
        for item in self.items:
            try:
                item.fit(df)
            except Exception as e:
                raise Exception("Error in item: " + item.name) from e
        
    def transform(self, df: pd.DataFrame) -> None:
        for item in self.items:
            try:
                item.transform(df)
            except Exception as e:
                raise Exception("Error in item: " + item.name) from e
    
    def fit_transform(self, df: pd.DataFrame) -> None:
        for item in self.items:
            try:
                item.fit_transform(df)
            except Exception as e:
                raise Exception("Error in item: " + item.name) from e

    def item_data_dict(self, typ=None):
        values = [item.values(typ=typ) for item in self.items]
        names = self.names
        return {n: v for n, v in zip(names, values)}

    def item_data_df(self, typ=None):
        dct = self.item_data_dict(typ=typ)
        df = pd.DataFrame.from_dict(dct)
        df.index.name = 'response'
        df.columns.name = 'item'
        return df