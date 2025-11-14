# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 21:26:39 2023

@author: Reuben
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any, Generator

from . import base

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


class Series(base.Object):
    """Class for handling single item operations."""

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._data)


class Numeric(base.Object):
    """Class for handling integer item operations."""

    def standardise(self, skip: bool = False) -> Numeric:
        """Standardise the numeric item."""
        if skip:
            return self
        mean = self._data.mean()
        std = self._data.std()
        standardised_series = (self._data - mean) / std
        return Numeric(data=standardised_series, name=self._name, desc=self._desc)

