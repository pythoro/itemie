# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:14:10 2023

@author: Reuben
"""


from . import items
from . import groups
from . import conversion

class GroupConverter:
    """Class for handling conversion operations."""

    def convert(self, group: groups.Group) -> groups.Group:
        """Convert the input value."""
        raise NotImplementedError("Converter subclasses must implement the convert method.")


class Numeric(GroupConverter):
    """Converter to convert values to integers."""

    def __init__(self, converter: conversion.Numeric):
        self._converter = converter

    def convert(self, group: groups.Group) -> groups.Numeric:
        """Convert the item to integers using the mapping."""
        converted_items = [obj.to(self._converter) for obj in group]
        return groups.Numeric(data=converted_items, name=group.name, desc=group.desc) # type: ignore
