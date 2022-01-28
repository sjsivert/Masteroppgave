from typing import Iterable, List

import pandas as pd
from genpipes import declare
from pandas import DataFrame


@declare.generator
def drop_columns(stream: Iterable[DataFrame], columns: List[str]) -> Iterable[DataFrame]:
    """
    Drops the specified columns from the dataframe.
    """
    for df in stream:
        df.drop(columns, axis=1, inplace=True)
        yield df
