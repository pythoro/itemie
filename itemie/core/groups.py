
from __future__ import annotations

import numpy as np
import pandas as pd

from . import base
from . import items
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


class GSeries(Group):
    def __init__(self, data: list[items.Series], name: str, desc: str | None = None):
        if not all(isinstance(item, items.Series) for item in data):
            raise TypeError("All items in GSeries must be of type Series.")
        self._data = data
        self._name = name
        self._desc = desc

    def __iter__(self) -> Generator[items.Series]:
        # This is a generator function that yields items from self.data
        for item in self._data:
            yield item


class GNumeric(Group):
    """Class for handling numeric group operations."""

    def __init__(self, data: list[items.Numeric], name: str, desc: str | None = None):
        if not all(isinstance(item, items.Numeric) for item in data):
            raise TypeError("All items in GNumeric must be of type Numeric.")
        self._data = data
        self._name = name
        self._desc = desc

    def __iter__(self) -> Generator[items.Numeric]:
        # This is a generator function that yields items from self.data
        for item in self._data:
            yield item

    def standardise(self, skip: bool = False) -> GNumeric:
        """Standardise the numeric item."""
        if skip:
            return self
        item_list = [item.standardise() for item in self._data]
        return GNumeric(data=item_list, name=self._name, desc=self._desc)
