from typing import Iterable, List, Tuple

import pandas as pd
from genpipes import declare
from pandas import DataFrame
from src.utils.config_parser import config


@declare.processor()
def drop_columns(stream: Iterable[DataFrame], columns: List[str]) -> Iterable[DataFrame]:
    """
    Drops the specified columns from the dataframe.
    """
    for df in stream:
        df.drop(columns, axis=1, inplace=True)
        yield df


@declare.processor()
def convert_date_to_datetime(stream: Iterable[DataFrame]) -> Iterable[DataFrame]:
    """
    Convert the date column to a datetime column.
    """
    for df in stream:
        df["date"] = pd.to_datetime(df["date"])
        yield df


@declare.processor()
def print_df(stream: Iterable[DataFrame]) -> Iterable[DataFrame]:  # pragma: no cover
    """
    Print the dataframe.
    """

    for df in stream:
        print(df)
        yield df


# pragma: no cover
@declare.processor()
def print_info(stream: Iterable[DataFrame]) -> Iterable[DataFrame]:  # pragma: no cover
    for df in stream:
        print(df.info())
        yield df


@declare.processor()
def group_by_and_keep_category_cols(
    stream: Iterable[DataFrame], group_by: List[str]
) -> Iterable[DataFrame]:  # pragma: no cover
    """
    Group the data by a given column and keep the automatically removed "nuicanse" columns
    https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#automatic-exclusion-of-nuisance-columns
    """
    categories = pd.read_csv(config["data"]["categories_path"].get())
    categories_name = categories[["title", "internal_doc_id"]]
    for df in stream:
        summed_result = df.groupby(group_by, as_index=False).sum()
        merged_result = summed_result.merge(
            categories_name, how="left", left_on="cat_id", right_on="internal_doc_id"
        )
        yield merged_result


@declare.processor()
def filter_column(stream: Iterable[DataFrame], column: str, value: int) -> Iterable[DataFrame]:
    for df in stream:
        yield df[df[column] > value]


@declare.processor()
def pivot_transform(
    stream: Iterable[DataFrame], **xargs
) -> Iterable[DataFrame]:  # pragma: no cover
    """
    Pivot the dataframe.
    """
    for df in stream:
        yield df.pivot(**xargs)


@declare.processor()
def rename(stream: Iterable[DataFrame], **xargs) -> Iterable[DataFrame]:  # pragma: no cover
    for df in stream:
        renamed_df = df.rename(columns={"title": "cat_name"}, inplace=False)
        yield renamed_df


@declare.processor()
def merge(
    stream: Iterable[DataFrame], join_with: DataFrame, **xargs
) -> Iterable[DataFrame]:  # pragma: no cover
    for df in stream:
        joined_df = pd.merge(
            left=df,
            right=join_with,
            how="left",
            left_on="cat_id",
            right_on="internal_doc_id",
            **xargs
        )
        yield joined_df


@declare.processor()
def filter_by_cat_id(
    stream: Iterable[DataFrame], cat_id: int
) -> Iterable[DataFrame]:  # pragma: no cover
    for df in stream:
        yield df[df["cat_id"] == cat_id]


@declare.processor()
def choose_columns(
    stream: Iterable[DataFrame], columns: List["str"]
) -> Iterable[DataFrame]:  # pragma: no cover
    for df in stream:
        yield df[columns]


@declare.processor()
def fill_in_dates(stream: Iterable[DataFrame]) -> Iterable[DataFrame]:  # pragma: no cover
    for df in stream:
        yield df.groupby(pd.Grouper(key="date", freq="D")).sum()


@declare.processor()
def split_into_training_and_test_set(
    stream: Iterable[DataFrame], training_size: float
) -> Iterable[Tuple[DataFrame, DataFrame]]:  # pragma: no cover
    for df in stream:
        training_set = int((df.shape[0] - 1) * training_size)
        training_df = df[:training_set]
        testing_set = df[training_set:]
        yield training_df, testing_set
