# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 21:26:39 2023

@author: Reuben
"""

from pathlib import Path
import numpy as np
import pandas as pd

from .convert import BaseConverter


class BaseItem:
    """A base item class"""

    def __init__(
        self,
        name: str,
        key: str,
        converter: BaseConverter = None,
        text=None,
    ):
        self._name = name
        self._key = key
        self.set_converter(converter)
        self._text = text
        self._raw = None

    def __str__(self):
        return self.__class__.__name__ + " '" + self.name + "'"

    def __repr__(self):
        return str(self)

    def _str(self, prefix=""):
        conv = ""
        if self._converter is not None:
            conv = " (" + self._converter._str() + ")"
        return prefix + self.__class__.__name__ + " '" + self.name + "'" + conv

    @property
    def name(self):
        return self._name

    @property
    def key(self):
        return self._key

    @property
    def text(self):
        return self._text

    @property
    def size(self):
        return len(self.raw)

    @property
    def raw_fitted(self):
        return self._raw_fitted

    @property
    def converted_fitted(self):
        return self._converted_fitted

    @property
    def raw(self):
        return self._raw

    @property
    def converted(self):
        return self._converted

    def set_converter(self, converter):
        self._converter = converter

    def _convert(self, series: np.ndarray) -> np.ndarray:
        if self._converter is None:
            converted = series
        else:
            converted = self._converter.convert(series)
        return converted

    def get_raw(self, df: pd.DataFrame) -> np.ndarray:
        return df[self._key]

    def fit(self, df: pd.DataFrame) -> None:
        raw = self.get_raw(df)
        converted = self._convert(raw)
        self._raw_fitted = raw
        self._converted_fitted = converted
        self._post_fit(converted)

    def transform(self, df: pd.DataFrame) -> None:
        raw = self.get_raw(df)
        converted = self._convert(raw)
        self._raw = raw
        self._converted = converted
        self._post_transform(converted)
        return self._converted

    def fit_transform(self, df: pd.DataFrame) -> None:
        raw = self.get_raw(df)
        converted = self._convert(raw)
        self._raw_fitted = raw
        self._converted_fitted = converted
        self._post_fit(converted)
        self._raw = raw
        self._converted = converted
        self._post_transform(converted)
        return converted

    def _post_fit(self, converted):
        pass

    def _post_transform(self, converted):
        pass

    def values(self, typ="default"):
        """Make the default an instance attribute, so it can be set"""
        if typ == "raw":
            return self.raw
        elif typ in ["default", "converted"]:
            return self.converted
        else:
            raise ValueError(
                "Item " + self.name + " — typ not understood:" + str(typ)
            )

    def data_dict(self, typ="default", match_size=True):
        values = self.values(typ)
        if match_size and len(values) != self.size:
            return {}
        return {self.name: values}

    def data_df(self, typ="default", match_size=True):
        dct = self.data_dict(typ=typ, match_size=match_size)
        df = pd.DataFrame.from_dict(dct)
        df.index.name = "response"
        df.columns.name = "item"
        return df

    def _get_all_items(self):
        return {self.name: self}


class Item(BaseItem):
    pass


class MultiCodedItem(BaseItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._df_wide = None

    def linearised(self, data=None):
        data = self.converted if data is None else data
        if not isinstance(data, list):
            raise ValueError("Data is not a list: " + str(type(data)))
        mapping = [[i] * len(lst) for i, lst in enumerate(data)]
        lin_data = [value for lst in data for value in lst]
        lin_mapping = [value for lst in mapping for value in lst]
        return lin_data, lin_mapping

    def linearised_df(self, data=None):
        lin_data, lin_mapping = self.linearised(data)
        dct = {"index": lin_mapping, "text": lin_data}
        df = pd.DataFrame(dct)
        return df

    def linearised_to_csv(self, folder, filename=None):
        root = Path(folder)
        if filename is None:
            filename = self.name + "_linearised"
        fname = (root / filename).with_suffix(".csv")
        df = self.linearised_df()
        df.to_csv(fname)

    def set_coded(
        self,
        df: pd.DataFrame,
        column_name,
        present_value=1,
        fill_value=0,
        exclude=None,
    ):
        lin_data, lin_mapping = self.linearised()
        if len(lin_mapping) != len(df["index"]):
            raise ValueError("Coding DataFrame is the wrong length for item "
                             + self.name + ".")
        self._df_coded = df
        df2 = df.copy()
        df2["values"] = present_value
        df_wide = pd.pivot_table(
            df2,
            columns=column_name,
            index="index",
            values="values",
            fill_value=fill_value,
        )
        if exclude:
            for excl_name in exclude:
                df_wide = df_wide.drop(excl_name)
        df_wide["count"] = np.sum(df_wide, axis=1)
        df_wide = df_wide.add_prefix(self.name + "_")
        self._df_wide = df_wide

    def set_coded_from_csv(
        self, folder, column_name, present_value=1, fill_value=0, filename=None
    ):
        root = Path(folder)
        if filename is None:
            filename = self.name + "_linearised_coded"
        fname = (root / filename).with_suffix(".csv")
        df = pd.read_csv(fname, index_col=0)
        self.set_coded(
            df,
            column_name=column_name,
            present_value=present_value,
            fill_value=fill_value,
        )

    def data_dict(self, typ="default", match_size=True):
        dct = super().data_dict(typ=typ, match_size=match_size)
        if self._df_wide is not None:
            dct2 = self._df_wide.to_dict("list")
            dct.update(dct2)
        return dct


class NumericItem(BaseItem):
    """A numeric item class"""

    def __init__(
        self,
        name: str,
        key: str,
        converter: BaseConverter = None,
        text=None,
        reverse_offset: float = None,
    ):
        super().__init__(name=name, key=key, converter=converter, text=text)
        self._reverse_offset = reverse_offset

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def sem(self):
        return self._sem

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def size(self):
        return len(self._converted)

    @property
    def standardised(self) -> np.ndarray:
        return (self._converted - self._mean) / self._std

    @property
    def normalised(self) -> np.ndarray:
        out = self._converted - self._min
        return out / np.max(out)

    def stats(self):
        labels = ["mean", "std", "min", "max", "sem", "ci95"]
        values = [
            self.mean,
            self.std,
            self.min,
            self.max,
            self.sem,
            self.sem * 1.95996,
        ]
        return labels, values

    def counts(self, as_int=True, as_percent=False):
        vals, counts = np.unique(self._converted, return_counts=True)
        n = self.size / 100 if as_percent else 1

        def convert_nan(v):
            if np.isnan(v):
                return "na"
            if as_int:
                return int(v)
            return v

        return {convert_nan(v): c / n for v, c in zip(vals, counts)}

    def _post_fit(self, converted):
        if self._reverse_offset is not None:
            converted = -converted + self._reverse_offset
        self._mean = np.nanmean(converted)
        self._std = np.nanstd(converted)
        self._max = np.nanmax(converted)
        self._min = np.nanmin(converted)
        self._sem = np.nanstd(converted, ddof=1) / np.sqrt(np.size(converted))
        return converted

    def _post_transform(self, transformed):
        if self._reverse_offset is not None:
            transformed = -transformed + self._reverse_offset
        return transformed

    def values(self, typ="default"):
        if typ in ["default", "standardised"]:
            return self.standardised
        elif typ == "normalised":
            return self.normalised
        else:
            return super().values(typ)


class PhraseCount(BaseItem):
    """A numeric item class"""

    def __init__(
        self,
        name: str,
        key: str,
        converter: BaseConverter = None,
        text=None,
        seq: list[str] = None,
    ):
        super().__init__(name=name, key=key, converter=converter, text=text)
        self._seq = seq

    @property
    def counts(self):
        return self._counts

    def _count(self, converted):
        if self._seq is not None:
            counts = {k: 0 for k in self._seq}
        else:
            counts = {}
        all_phrases = []
        for obj in converted:
            if isinstance(obj, list):
                all_phrases.extend(obj)
            else:
                all_phrases.append(obj)
        for phrase in all_phrases:
            if phrase in counts:
                counts[phrase] += 1
            else:
                counts[phrase] = 0
        if self._seq is None:
            sorted_keys = list(counts.keys())
            sorted_keys.sort(key=lambda x: str(x))
            counts = {k: counts[k] for k in sorted_keys}
        return counts

    def _post_transform(self, transformed):
        self._counts = self._count(transformed)
        return transformed

    def as_list(self):
        return list(self._counts.keys())

    def values(self, typ="default"):
        if typ in ["default", "counts"]:
            return self.counts
        else:
            return super().values(typ)
