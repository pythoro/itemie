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

class TestReplacer:
    def test_convert(self):
        keyvals = {"apple": 1.0, "pear": 2.0}
        converter = convert.Replacer(keyvals)
        data = np.array(["apple", "pear", "apple", "apple", "pear"])
        converted = converter.convert(data)
        expected = np.array([1.0, 2.0, 1.0, 1.0, 2.0])
        assert converted == approx(expected)