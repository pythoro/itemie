# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:26:03 2023

@author: Reuben
"""
import pytest
import numpy as np
import pandas as pd
from pytest import approx

from itemie.core import item


class TestConverter:
    def test_convert(self):
        keyvals = {"apple": 1.0, "pear": 2.0}
        converter = item.Converter(keyvals)
        data = np.array(["apple", "pear", "apple", "apple", "pear"])
        converted = converter.convert(data)
        expected = np.array([1.0, 2.0, 1.0, 1.0, 2.0])
        assert converted == approx(expected)


@pytest.fixture
def item_b():
    data = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [1, 2, 3, 4]])
    df = pd.DataFrame(data, columns=["a", "b", "c", "d"])
    b = item.BaseItem(name="banana", key="b")
    b.fit_transform(df)
    return b


class TestBaseItem:
    def test_fit(self):
        data = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [1, 2, 3, 4]])
        df = pd.DataFrame(data, columns=["a", "b", "c", "d"])
        b = item.BaseItem(name="banana", key="b")
        res = b.fit(df)
        expected = np.array([2, 3, 2])
        print(res)
        assert res.values == approx(expected)

    def test_fit_transform_values(self, item_b):
        res = item_b.values()
        expected = np.array([2, 3, 2])
        print(res)
        assert res.values == approx(expected)

    def test_fit_transform_raw(self, item_b):
        res = item_b.raw()
        expected = np.array([2, 3, 2])
        print(res)
        assert res.values == approx(expected)

    def test_fit_transform_call(self, item_b):
        res = item_b()
        expected = np.array([2, 3, 2])
        print(res)
        assert res.values == approx(expected)


@pytest.fixture
def numericitem_b():
    data = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [1, 2, 3, 4]])
    df = pd.DataFrame(data, columns=["a", "b", "c", "d"])
    b = item.NumericItem(name="banana", key="b", typ="standardised")
    b.fit_transform(df)
    return b


@pytest.fixture
def numericitem_b_raw():
    data = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [1, 2, 3, 4]])
    df = pd.DataFrame(data, columns=["a", "b", "c", "d"])
    b = item.NumericItem(name="banana", key="b", typ="raw")
    b.fit_transform(df)
    return b


class TestNumericItem:
    def test_fit_transform_values(self, numericitem_b):
        res = numericitem_b.values()
        expected = np.array([-0.70710678, 1.41421356, -0.70710678])
        print(res)
        assert res.values == approx(expected)

    def test_fit_transform_normalised(self, numericitem_b):
        res = numericitem_b.normalised()
        expected = np.array([0, 1, 0])
        print(res)
        assert res.values == approx(expected)

    def test_fit_transform_values_arg(self, numericitem_b_raw):
        res = numericitem_b_raw.values(typ="standardised")
        expected = np.array([-0.70710678, 1.41421356, -0.70710678])
        print(res)
        assert res.values == approx(expected)
