
from __future__ import annotations

import pandas as pd
from typing import Any, Generator

from . import conversion

class Object:
    """Class for handling single item operations."""

    def __init__(self, data: Any, name: str, desc: str | None = None):
        self._data = data
        self._name = name
        self._desc = desc

    @property
    def name(self) -> str:
        return self._name

    @property
    def desc(self) -> str | None:
        return self._desc
    
    @property
    def data(self) -> Any:
        return self._data

    def __iter__(self) -> Generator[Object]:
        # This is a generator function that yields items from self.data
        for item in self._data:
            yield item

    def to(self, converter: conversion.Converter) -> Object:
        """Convert the item using a converter function."""
        return converter.convert(self)

    def to_df(self) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement the to_df property.")
