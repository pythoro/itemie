
from __future__ import annotations

from . import base
from . import items, groups
from typing import Callable

class Converter:
    """Class for handling conversion operations."""

    def __call__(self, object: base.Object) -> base.Object:
        """Convert the input value."""
        raise NotImplementedError(
            "Converter subclasses must implement the __call__ method."
        )

class SeriesToNumeric(Converter):
    """Converter to convert values to integers."""

    def __init__(self, mapping: dict, missing=None):
        self._mapping = mapping
        self._missing = missing

    def __call__(self, item: items.Series) -> items.Numeric:
        """Convert the item to integers using the mapping."""
        converted = item.apply(self._convert_value)
        return items.Numeric(data=converted, name=item.name, desc=item.desc)

    def _convert_value(self, value: object) -> int | float | None:
        """Convert a single value using the mapping."""
        return self._mapping.get(value, self._missing)


class SeriesToSeries(Converter):
    """Converter to convert values to integers."""

    def __init__(self, func: Callable):
        self._func = func

    def __call__(self, item: items.Series) -> items.Series:
        """Convert the item to integers using the mapping."""
        converted = item.apply(self._func)
        return items.Series(data=converted, name=item.name, desc=item.desc)


class SeriesToListSeries(Converter):
    """Converter to convert values to integers."""

    def __init__(self, split_char: str):
        self._split_char = split_char

    def __call__(self, item: items.Series) -> items.ListSeries:
        """Convert the item to integers using the mapping."""
        converted = item.apply(self._convert_value)
        return items.ListSeries(data=converted, name=item.name, desc=item.desc)

    def _convert_value(self, value: str) -> list[str]:
        """Convert a single value using the mapping."""
        return value.split(self._split_char)


class ListSeriesToDataFrame(Converter):
    """Converter to convert list series to a data frame."""

    def __call__(self, item: items.ListSeries) -> items.DataFrame:
        """Convert the item to integers using the mapping."""
        pass


class ListSeriesToGSeries(Converter):
    """Converter to convert list series to a data frame."""

    def __call__(self, item: items.ListSeries) -> groups.GSeries:
        """Convert the item to integers using the mapping."""
        pass
