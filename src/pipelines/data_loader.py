import logging
from typing import Iterable

import pandas as pd
from genpipes import declare
from pandas import DataFrame


@declare.generator()
def load_and_merge_market_insight_and_categories(
    market_insight_path: str, categories_path: str
) -> DataFrame:  # pragma: no cover
    try:
        df1 = pd.read_csv(market_insight_path)
        df2 = pd.read_csv(categories_path)
        return pd.merge(df1, df2, how="left", left_on="cat_id", right_on="internal_doc_id")
    except Exception as e:
        logging.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )


@declare.generator()
def load_csv_data(path: str) -> DataFrame:  # pragma: no cover
    """
    Loads data from a file.

    :param path:
    :return: pandas dataframe containing the data
    """
    return (pd.read_csv(path), None)


@declare.generator()
def dataframe_to_generator(df: DataFrame) -> Iterable[DataFrame]:
    return df
