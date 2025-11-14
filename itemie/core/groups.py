# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 19:32:05 2023

@author: Reuben
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import base
from . import items
from . import group_conversion
from typing import Generator


class Group(base.Object):
    """Class for handling group operations."""

    def __init__(self, data: list[base.Object], name: str, desc: str | None = None):
        self._data = data
        self._name = name
        self._desc = desc

    def __iter__(self) -> Generator[base.Object]:
        # This is a generator function that yields items from self.data
        for item in self._data:
            yield item

    def to_df(self) -> pd.DataFrame:
        """Create a DataFrame from the group of items."""
        dfs = [item.to_df() for item in self._data]
        # Show error if dfs are not all the same length
        lengths = [len(df) for df in dfs]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"All items in {self.name} must have the same length " 
                "to convert to DataFrame."
            )
        df = pd.concat(dfs, axis=1)
        return df

    def to(self, group_converter: group_conversion.GroupConverter) -> Group:
        """Convert all items in the group using a converter function."""
        return group_converter.convert(self)


class Numeric(Group):
    """Class for handling numeric group operations."""

    def __iter__(self) -> Generator[items.Numeric]:
        # This is a generator function that yields items from self.data
        for item in self._data:
            if isinstance(item, items.Numeric):
                yield item
            else:
                raise TypeError("All items must be of type items.Numeric")

    def standardise(self, skip: bool = False) -> Numeric:
        """Standardise the numeric item."""
        if skip:
            return self
        item_list = []
        for item in self._data:
            if isinstance(item, items.Numeric):
                item_list.append(item.standardise())
            else:
                raise TypeError("All items must be of type items.Numeric")
        return Numeric(data=item_list, name=self._name, desc=self._desc)
