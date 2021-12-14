from typing import Iterable, List

import pandas as pd
from genpipes import declare


@declare.processor()
def drop_columns(stream: Iterable[pd.DataFrame], columns: List[str]) -> Iterable[pd.DataFrame]:
    """
    Drops the specified columns from the dataframe.
    """
    for df in stream:
        df.drop(columns, axis=1, inplace=True)
        yield df


@declare.processor()
def convert_date_to_datetime(stream: Iterable[pd.DataFrame]) -> Iterable[pd.DataFrame]:
    """
    Convert the date column to a datetime column.
    """
    for df in stream:
        df["date"] = pd.to_datetime(df["date"])
        yield df


@declare.processor()
def print_df(stream: Iterable[pd.DataFrame]) -> Iterable[pd.DataFrame]:
    """
    Print the dataframe.
    """
    for df in stream:
        print(df)
        yield df


@declare.processor()
def group_by(stream: Iterable[pd.DataFrame], group_by: List[str]) -> Iterable[pd.DataFrame]:
    """
    Group the data by a given column.
    """
    for df in stream:
        yield df.groupby(group_by, as_index=False).sum()


@declare.processor()
def filter_column(stream: Iterable[pd.DataFrame], column: str, value: int) -> Iterable[pd.DataFrame]:
    for df in stream:
        yield df[df[column] > value]
