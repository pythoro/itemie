# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:14:10 2023

@author: Reuben
"""

import numpy as np
import pandas as pd
from . import items
from . import base

class Converter:
    """Class for handling conversion operations."""

    def convert(self, object: base.Object) -> base.Object:
        """Convert the input value."""
        raise NotImplementedError("Converter subclasses must implement the convert method.")


class Numeric(Converter):
    """Converter to convert values to integers."""

    def __init__(self, mapping: dict[object, int], missing=None):
        self._mapping = mapping
        self._missing = missing

    def _convert_value(self, value: object) -> int | float | None:
        """Convert a single value using the mapping."""
        return self._mapping.get(value, self._missing)

    def convert(self, item: items.Series) -> items.Numeric:
        """Convert the item to integers using the mapping."""
        converted_series = item.data.apply(self._convert_value)
        return items.Numeric(data=converted_series, name=item.name, desc=item.desc)

