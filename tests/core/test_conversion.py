import pytest
import pandas as pd
import numpy as np
from itemie.core import items, conversion, groups


class Test_Numeric:

    def test_call(self):
        data = pd.Series(["a", "b", "c", "a", "b", "c"])
        item = items.Series(data=data, name="test_series", desc="desc")

        mapping = {"a": 1, "b": 2, "c": 3}
        to_numeric = conversion.SeriesToNumeric(mapping=mapping, missing=np.nan)

        converted_item = to_numeric(item)

        expected_data = pd.Series([1, 2, 3, 1, 2, 3])
        expected_item = items.Numeric(
            data=expected_data, name="test_series", desc="desc"
        )

        pd.testing.assert_series_equal(converted_item.data, expected_item.data)
        assert converted_item.name == expected_item.name
        assert converted_item.desc == expected_item.desc

    def test_convert_several(self):
        df = pd.DataFrame(
            {
                "col1": ["a", "b", "c", "a", "b", "c"],
                "col2": ["x", "y", "z", "x", "y", "z"],
            }
        )
        mapping = {"a": 1, "b": 2, "c": 3}
        to_numeric = conversion.SeriesToNumeric(mapping=mapping, missing=np.nan)

        idf = items.DataFrame(data=df, name="test_df", desc="desc")
        item1 = idf.col(column="col1", name="series1", desc="desc1")
        item2 = idf.col(column="col2", name="series2", desc="desc2")
        group = groups.GSeries(
            data=[item1, item2], name="test_group", desc="group_desc"
        )

        converted = [to_numeric(item) for item in group]
        converted_group = groups.GNumeric(converted, name=group.name, desc=group.desc)
        assert isinstance(converted_group, groups.Group)

    def test_convert_several_2(self):
        df = pd.DataFrame(
            {
                "col1": ["a", "b", "c", "a", "b", "c"],
                "col2": ["x", "y", "z", "x", "y", "z"],
            }
        )
        mapping = {"a": 1, "b": 2, "c": 3}
        to_numeric = conversion.SeriesToNumeric(mapping=mapping, missing=np.nan)

        idf = items.DataFrame(data=df, name="test_df", desc="desc")
        item1 = idf.col(column="col1", name="series1", desc="desc1")
        item2 = idf.col(column="col2", name="series2", desc="desc2")
        item_list = [item1, item2]
        converted = [to_numeric(item) for item in item_list]
        converted_group = groups.GNumeric(
            converted, name="test_group", desc="test_desc"
        )
        assert isinstance(converted_group, groups.Group)
