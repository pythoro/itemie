import pytest
import pandas as pd
import numpy as np
from itemie.core import items, conversion


class Test_Numeric:
    def test_convert(self):
        data = pd.Series(['a', 'b', 'c', 'a', 'b', 'c'])
        item = items.Series(data=data, name='test_series', desc='desc')

        mapping = {'a': 1, 'b': 2, 'c': 3}
        to_numeric = conversion.Numeric(mapping=mapping, missing=np.nan)

        converted_item = to_numeric.convert(item)

        expected_data = pd.Series([1, 2, 3, 1, 2, 3])
        expected_item = items.Numeric(data=expected_data, name='test_series', desc='desc')

        pd.testing.assert_series_equal(converted_item.data, expected_item.data)
        assert converted_item.name == expected_item.name
        assert converted_item.desc == expected_item.desc

    def test_call(self):
        data = pd.Series(['a', 'b', 'c', 'a', 'b', 'c'])
        item = items.Series(data=data, name='test_series', desc='desc')

        mapping = {'a': 1, 'b': 2, 'c': 3}
        to_numeric = conversion.Numeric(mapping=mapping, missing=np.nan)

        converted_item = to_numeric(item)

        expected_data = pd.Series([1, 2, 3, 1, 2, 3])
        expected_item = items.Numeric(data=expected_data, name='test_series', desc='desc')

        pd.testing.assert_series_equal(converted_item.data, expected_item.data)
        assert converted_item.name == expected_item.name
        assert converted_item.desc == expected_item.desc