from __future__ import annotations

import numpy as np
import pandas as pd
from . import groups
from . import conversion


class Converter:
    """Class for handling conversion operations."""

    def __call__(self, group: groups.Group) -> groups.Group:
        """Convert the input value."""
        raise NotImplementedError(
            "Converter subclasses must implement the __call__ method."
        )


class ToNumeric(Converter):
    """Converter to convert values to integers."""

    def __init__(self, to_numeric: conversion.ToNumeric):
        self._converter = to_numeric

    def __call__(self, group: groups.Series) -> groups.Numeric:
        """Convert the item to integers using the mapping."""
        converted = [self._converter(value) for value in group.data]
        return groups.Numeric(data=converted, name=group.name, desc=group.desc)
