import pytest
import pandas as pd

from itemie.core import items

class Test_DataFrame:
    def test_item(self):
        df = pd.DataFrame({
            'col1': ['a', 'b', 'c', 'a', 'b', 'c'],
            'col2': ['x', 'y', 'z', 'x', 'y', 'z']
        })
        idf = items.DataFrame(data=df, name='test_df', desc='desc')
        item = idf.col(column='col1', name='test_series', desc='desc')
        expected_data = pd.Series(['a', 'b', 'c', 'a', 'b', 'c'], name='test_series')
        expected_item = items.Series(data=expected_data, name='test_series', desc='desc
        )
        pd.testing.assert_series_equal(item.data, expected_item.data)
        assert item.name == expected_item.name
        assert item.desc == expected_item.desc
