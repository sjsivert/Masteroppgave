# fmt: off
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from genpipes import declare
from numpy import ndarray
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.datasets.time_series_dataset import TimeseriesDataset
from src.utils.config_parser import config
from torch.utils.data import DataLoader, Dataset


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
        filtered_df = df[df["cat_id"] == cat_id]
        yield filtered_df


@declare.processor()
def choose_columns(
    stream: Iterable[DataFrame], columns: List["str"]
) -> Iterable[DataFrame]:  # pragma: no cover
    for df in stream:
        chosen_columns = df[columns]
        yield chosen_columns


@declare.processor()
def fill_in_dates(stream: Iterable[DataFrame]) -> Iterable[DataFrame]:  # pragma: no cover
    for df in stream:
        yield df.groupby(pd.Grouper(key="date", freq="D")).sum()

@declare.processor()
def convert_to_np_array(stream: Iterable[DataFrame]) -> Iterable[ndarray]:  # pragma: no cover
    for df in stream:
        yield np.array(df)

@declare.processor()
def scale_data(
    stream: Iterable[ndarray], should_scale: bool = False
) -> (Iterable[ndarray], Optional[MinMaxScaler]):  # pragma: no cover
    for df in stream:
        if (df.size == 0):
            print("Empty dataframe",df)
            raise ValueError("Numpy is empty after earlier filtering steps. Check your category configuration", df)

        if should_scale:
            scaler = MinMaxScaler(feature_range=(0.1, 1))
            scaled_df = scaler.fit_transform(df)
            yield scaled_df, scaler
        else:
            yield df, None

@declare.processor()
def split_into_training_and_test_forecast_window(
        stream: Iterable[Tuple[DataFrame, Optional[StandardScaler]]], forecast_window_size: int, input_window_size: int
) -> Iterable[Tuple[DataFrame, DataFrame, Optional[MinMaxScaler]]]:  # pragma: no cover
    for (df, scaler) in stream:
        # The testing set is the same as the prediction output window
        test_data_split_index = df.shape[0] - forecast_window_size - input_window_size
        training_df = df[:test_data_split_index]
        testing_set = df[test_data_split_index:]

        yield training_df, testing_set, scaler

@declare.processor()
def keras_split_into_training_and_test_set(
        stream: Iterable[Tuple[Tuple[ndarray], Tuple[ndarray], Tuple[ndarray], Optional[StandardScaler]]],
        test_window_size: int,
) -> Iterable[Tuple[DataFrame, DataFrame, Optional[MinMaxScaler]]]:  # pragma: no cover
    for (x, y, scaler) in stream:
        # The testing set is the same as the prediction output window
        x_train = x[:-1]
        y_train = y[:-1]
        x_test = x[-1:]
        y_test = y[-1:]

        yield ((x_train, y_train),  (x_test, y_test), scaler)
@declare.processor()
def save_datasets_to_file(
        stream: Iterable[Tuple[Tuple[ndarray, ndarray], Tuple[ndarray, ndarray], Optional[StandardScaler]]],
) -> Iterable[Tuple[DataFrame, DataFrame, Optional[MinMaxScaler]]]:  # pragma: no cover
    for (training, testing, scaler) in stream:
        # The testing set is the same as the prediction output window
        training_df = pd.DataFrame(training[1][:, 0, 0])
        testing_df = pd.DataFrame(testing[1][0, :, 0])
        training_df.to_csv("./datasets/interim/lstm_training_set.csv")
        testing_df.to_csv("./datasets/interim/lstm_testing_set.csv")

        yield (training, testing, scaler)

@declare.processor()
def split_into_training_and_test_forecast_window_arima(
        stream: Iterable[Tuple[DataFrame, Optional[StandardScaler]]], forecast_window_size: int, input_window_size: int
) -> Iterable[Tuple[DataFrame, DataFrame, Optional[MinMaxScaler]]]:  # pragma: no cover
    for (df, scaler) in stream:
        # The testing set is the same as the prediction output window
        test_data_split_index = df.shape[0] - forecast_window_size
        training_df = df[:test_data_split_index]
        testing_set = df[test_data_split_index:]


        yield training_df, testing_set, scaler
@declare.processor()
def save_datasets_to_file_arima(
        stream: Iterable[Tuple[Tuple[ndarray, ndarray], Tuple[ndarray, ndarray], Optional[StandardScaler]]],
        name: str,
) -> Iterable[Tuple[DataFrame, DataFrame, Optional[MinMaxScaler]]]:  # pragma: no cover
    for (training, validation_set, testing, scaler) in stream:
        print(training.shape)
        training_df = pd.DataFrame(training)
        print(training_df.head())
        testing_df = pd.DataFrame(testing)
        training_df.to_csv(f"./datasets/interim/arima_training_set{name}.csv")
        testing_df.to_csv(f"./datasets/interim/arima_testing_set{name}.csv")

        yield (training, validation_set, testing, scaler)

@declare.processor()
def split_into_training_and_validation_forecast_window_arima(
        stream: Iterable[Tuple[DataFrame, Optional[StandardScaler]]], forecast_window_size: int, input_window_size: int
) -> Iterable[Tuple[DataFrame, DataFrame, DataFrame, Optional[MinMaxScaler]]]:  # pragma: no cover
    for (training_set, testing_set, scaler) in stream:
        # The testing set is the same as the prediction output window
        test_data_split_index = training_set.shape[0] - forecast_window_size
        training_df = training_set[:test_data_split_index]
        validation_set = training_set[test_data_split_index:]

        yield training_df, validation_set, testing_set, scaler

@declare.processor()
def split_into_training_and_validation_forecast_window(
        stream: Iterable[Tuple[DataFrame, Optional[StandardScaler]]], forecast_window_size: int, input_window_size: int
    ) -> Iterable[Tuple[DataFrame, DataFrame, DataFrame, Optional[MinMaxScaler]]]:  # pragma: no cover
    for (training_set, testing_set, scaler) in stream:
        # The testing set is the same as the prediction output window
        test_data_split_index = training_set.shape[0] - forecast_window_size - input_window_size
        training_df = training_set[:test_data_split_index]
        validation_set = training_set[test_data_split_index:]

        yield training_df, validation_set, testing_set, scaler


@declare.processor()
def split_into_training_and_test_set(
    stream: Iterable[Tuple[DataFrame, Optional[StandardScaler]]], training_size: Union[int, float],
) -> Iterable[Tuple[DataFrame, DataFrame, Optional[MinMaxScaler]]]:  # pragma: no cover
    for (df, scaler) in stream:
        if type(training_size) == float:
            # if the training size is a float, we take the percentage of the dataframe
            training_set = int((df.shape[0] - 1) * training_size)
        else:
            # The testing set is the same as the prediction output window
            training_set = df.shape[0] - training_size
        training_df = df[:training_set]
        testing_set = df[training_set:]

        yield training_df, testing_set, scaler


@declare.processor()
def split_into_training_and_validation_set(
    stream: Iterable[Tuple[DataFrame, DataFrame, Optional[StandardScaler]]], training_size: float
) -> Iterable[Tuple[DataFrame, DataFrame, DataFrame, Optional[MinMaxScaler]]]:  # pragma: no cover
    for (training_set, testing_set, scaler) in stream:
        if type(training_size) == float:
            # if the training size is a float, we take the percentage of the dataframe
            training_set_size = int((training_set.shape[0] - 1) * training_size)
        else:
            # The testing set is the same as the prediction output window
            training_set_size = training_set.shape[0] - training_size
        training_df = training_set[:training_set_size]
        validation_df = training_set[training_set_size:]
        yield training_df, validation_df, testing_set, scaler


@declare.processor()
def combine_hits_and_clicks(
    stream: Iterable[DataFrame], hits_scalar: int = 1, clicks_scalar: int = 1
) -> Iterable[DataFrame]:  # pragma: no cover
    for df in stream:
        df["interest"] = hits_scalar * df["hits"] + clicks_scalar * df["clicks"]
        yield df


@declare.processor()
def convert_to_time_series_dataset(
    stream: Iterable[Tuple[DataFrame, DataFrame, DataFrame, Optional[StandardScaler]]],
    input_window_size: int,
    output_window_size: int,
) -> Iterable[Tuple[Dataset, Dataset, Optional[MinMaxScaler]]]:  # pragma: no cover
    for (training_data, validation_data, testing_data, scaler) in stream:
        training_dataset = TimeseriesDataset(
            training_data, seq_len=input_window_size, y_size=output_window_size
        )
        validation_dataset = TimeseriesDataset(
            validation_data, seq_len=input_window_size, y_size=output_window_size
        )
        testing_dataset = TimeseriesDataset(
            validation_data, seq_len=input_window_size, y_size=output_window_size
        )
        x = training_data.__getitem__(1)
        yield training_dataset, validation_dataset, testing_dataset, scaler


@declare.processor()
def convert_to_pytorch_dataloader(
    stream: Iterable[Tuple[Dataset, Dataset, Dataset, Optional[StandardScaler]]], batch_size: int
) -> Iterable[Tuple[DataLoader, DataLoader, Optional[MinMaxScaler]]]:  # pragma: no cover
    for (training_data, validation_data, testing_data, scaler) in stream:
        training_dataloader = DataLoader(
            dataset=training_data, batch_size=batch_size, shuffle=False
        )
        validation_dataloader = DataLoader(
            dataset=validation_data, batch_size=batch_size, shuffle=False
        )
        testing_dataloader = DataLoader(
            dataset=testing_data, batch_size=batch_size, shuffle=False
        )
        yield training_dataloader, validation_dataloader, testing_dataloader, scaler

@declare.processor()
def sliding_window_x_y_generator(
        stream: Iterable[Tuple[ndarray, Optional[StandardScaler]]],
        input_window_size: int,
        output_window_size: int,
) -> Iterable[Tuple[ndarray, ndarray, Optional[StandardScaler]]]:  # pragma: no cover
    for data, scaler in stream:
        X = []
        Y = []
        for i in range(0, len(data) - input_window_size - output_window_size + 1):
            x = data[i:i + input_window_size]
            y = data[i + input_window_size:i + input_window_size + output_window_size]
            X.append(x)
            Y.append(y)

        yield np.array(X), np.array(Y), scaler


@declare.processor()
def simple_time_series_processor(stream: Iterable[DataFrame]) -> Iterable[DataFrame]:
    for df in stream:
        #raw_data = np.expand_dims(df, axis=1)
        raw_data = np.array(df)

        test_size = 0.2

        scaler_test_dataset = MinMaxScaler(feature_range=(-1, 1))
        raw_data_scaled = scaler_test_dataset.fit_transform(raw_data)

        #simple_data_train = raw_data_scaled[: -int(test_size * len(raw_data))]
        #simple_data_test = raw_data_scaled[-int(test_size * len(raw_data)):]
        #simple_data_val = raw_data_scaled[-int(test_size * len(simple_data_train)) :]
        input_window_size = 10
        output_window_size = 7
        test_data_split_index = len(raw_data) - output_window_size - input_window_size
        simple_data_train = raw_data_scaled[:test_data_split_index]
        simple_data_test = raw_data_scaled[test_data_split_index:]
        simple_data_val = simple_data_test

        train_data = TimeseriesDataset(simple_data_train, seq_len=10, y_size=7)
        test_data = TimeseriesDataset(simple_data_val, seq_len=10, y_size=7)
        val_data = TimeseriesDataset(simple_data_test, seq_len=10, y_size=7)

        #train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=False)
        #test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)
        #val_loader = DataLoader(dataset=val_data, batch_size=32, shuffle=False)
        yield train_data, test_data, val_data, scaler_test_dataset
