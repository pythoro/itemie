# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:14:10 2023

@author: Reuben
"""

from __future__ import annotations

from . import items
from . import groups
from . import conversion


class GroupConverter:
    """Class for handling conversion operations."""

    def __call__(self, group: groups.Group) -> groups.Group:
        """Convert the input value."""
        return self.convert(group)

    def convert(self, group: groups.Group) -> groups.Group:
        """Convert the input value."""
        raise NotImplementedError(
            "Converter subclasses must implement the convert method."
        )


class Numeric(GroupConverter):
    """Converter to convert values to integers."""

    def __init__(self, converter: conversion.Numeric):
        self._converter = converter

    def convert(self, group: groups.Group) -> groups.Numeric:
        """Convert the item to integers using the mapping."""
        item_list = []
        for obj in group:
            if isinstance(obj, items.Series):
                item_list.append(self._converter.convert(obj))
            else:
                raise TypeError(
                    f"All items in group {group.name} must be of "
                    "type items.Series"
                )
        return groups.Numeric(data=item_list, name=group.name, desc=group.desc)  # type: ignore
