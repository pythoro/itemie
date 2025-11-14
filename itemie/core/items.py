
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from . import base, groups

class DataFrame():
    """Class for handling DataFrame operations."""

    def __init__(self, data: pd.DataFrame, name: str, desc: str | None = None):
        self._data = data
        self._name = name
        self._desc = desc

    @classmethod
    def from_csv(cls, file_path: Path, name: str, desc: str) -> DataFrame:
        """Create a DataFrame instance from a CSV file."""
        df = pd.read_csv(file_path)
        return cls(data=df, name=name, desc=desc)
    
    def col(self, column: str, name: str, desc: str | None = None) -> Series:
        """Create an Item from a column."""
        series = self._data[column]
        series.name = name
        return Series(data=series, name=name, desc=desc)

    def to_gseries(self) -> groups.GSeries:
        """Create a GSeries from the DataFrame columns."""
        items_list = [
            Series(data=self._data[col], name=col, desc=None)
            for col in self._data.columns
        ]
        return groups.GSeries(data=items_list, name=self._name, desc=self._desc)

class Series(base.Object):
    """Class for handling single item operations."""

    def apply(self, func) -> Series:
        """Apply a function to the series data."""
        return self._data.apply(func)

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._data)


class ListSeries(base.Object):
    """Class for handling single item operations."""
    pass


class Numeric(base.Object):
    """Class for handling integer item operations."""

    def standardise(self, skip: bool = False) -> Numeric:
        """Standardise the numeric item."""
        if skip:
            return self
        mean = self._data.mean(axis=0)
        std = self._data.std(axis=0)
        standardised_series = (self._data - mean) / std
        return Numeric(data=standardised_series, name=self._name, desc=self._desc)

