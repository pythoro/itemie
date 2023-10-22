# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:01:05 2023

@author: Reuben
"""

import pytest
import numpy as np
import pandas as pd
from pytest import approx

from itemie.core import convert

class TestReplace:
    def test_convert(self):
        keyvals = {"apple": 1.0, "pear": 2.0}
        converter = convert.Replace(keyvals)
        data = np.array(["apple", "pear", "apple", "apple", "pear"])
        converted = converter.convert(data)
        expected = np.array([1.0, 2.0, 1.0, 1.0, 2.0])
        assert converted == approx(expected)
        
        
class TestAutoCorrect:
    def test_convert(self):
        converter = convert.AutoCorrect()
        data = np.array(["apple", "pear", "applee"])
        converted = converter.convert(data)
        expected = np.array(["apple", "pear", "apple"])
        print(converted)
        assert converted == approx(expected)
        
        
class TestPipeline:
    def test_convert(self):
        autocorrect = convert.AutoCorrect()
        keyvals = {"apple": 1.0, "pear": 2.0}
        replacer = convert.Replace(keyvals)
        converter = convert.Pipeline(autocorrect, replacer)
        data = np.array(["apple", "pear", "apple", "applee", "pear"])
        converted = converter.convert(data)
        expected = np.array([1.0, 2.0, 1.0, 1.0, 2.0])
        print(converted)
        assert converted == approx(expected)