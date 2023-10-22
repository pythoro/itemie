# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 18:09:17 2023

@author: Reuben
"""

def subitems(cls, prefix, names, keys, converter=None, kwargs=None):
    kwargs = [{}] * len(names) if kwargs is None else kwargs
    items = []
    for name, key, kw in zip(names, keys, kwargs):
        item = cls(name=name, key=prefix + key, converter=converter, **kw)
        items.append(item)
    return items