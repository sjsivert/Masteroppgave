# fmt: off
import math
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from genpipes import declare
from numpy import ndarray
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.datasets.time_series_dataset import TimeseriesDataset
from src.pipelines.date_feature_generator import calculate_season
from src.utils.config_parser import config
from src.utils.visuals import visualize_data_series
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
    for (df, test_data, scaler) in stream:
        yield np.array(df), np.array(test_data), scaler

@declare.processor()
def convert_to_np_array_univariate(stream: Iterable[DataFrame]) -> Iterable[ndarray]:  # pragma: no cover
    for df, test_data in stream:
        # visualize_data_series(
        #     title=f"original_training_data",
        #     data_series=[df],
        #     data_labels=["Targets", ],
        #     colors=["blue",],
        #     x_label="Time",
        #     y_label="Interest",
        # ).savefig("original_training_data.png")
        yield np.array(df), np.array(test_data, dtype=np.float64)

@declare.processor()
def scale_data_dataframe(
    stream: Iterable[DataFrame], should_scale: bool = False
) -> (Iterable[DataFrame], Optional[MinMaxScaler]):  # pragma: no cover
    for (df, test_data) in stream:
        if (df.size == 0):
            print("Empty dataframe",df)
            raise ValueError("Numpy is empty after earlier filtering steps. Check your category configuration", df)

        if should_scale:
            scaler = StandardScaler()
            df["interest"] = scaler.fit_transform(df[["interest"]])
            for col in df.columns:
                if col != "interest":
                    feature_scaler = StandardScaler()
                    df[col] = feature_scaler.fit_transform(df[[col]])
            yield df, scaler
        else:
            yield df, scaler

@declare.processor()
def decrease_variance(
    stream: Iterable[ndarray]
) -> (Iterable[ndarray], Optional[MinMaxScaler]):  # pragma: no cover
    for df, test_data in stream:
        decrease_variance = lambda x: math.log(x+1)
        vector_func = np.vectorize(decrease_variance)
        df_decreased_variance = vector_func(df)
        # visualize_data_series(
        #     title=f"decreased_variance",
        #     data_series=[df_decreased_variance],
        #     data_labels=["Targets", ],
        #     colors=["blue",],
        #     x_label="Time",
        #     y_label="Interest",
        # ).savefig("decrease_variance.png")
        yield df_decreased_variance, test_data

@declare.processor()
def scale_down_outliers(
    stream: Iterable[ndarray]
) -> (Iterable[ndarray], Optional[MinMaxScaler]):  # pragma: no cover
    for df, test_data in stream:
        df = pd.DataFrame(df)
        std = (df.describe().transpose()["std"]).to_numpy()[0]
        mean = (df.describe().transpose()["mean"]).to_numpy()[0]

        minimize_outliers = lambda x: x if (x <= (5 *std)) else  x * 0.3
        vector_func = np.vectorize(minimize_outliers)
        removed_outliers = vector_func(df)

        df.plot().get_figure().savefig("normal.png")
        pd.DataFrame(removed_outliers).plot().get_figure().savefig("outliers.png")
        yield removed_outliers, test_data

@declare.processor()
def scale_data(
    stream: Iterable[ndarray], should_scale: bool = False
) -> (Iterable[ndarray], Optional[MinMaxScaler]):  # pragma: no cover
    for training_data, test_data, not_differenced_data in stream:
        if (training_data.size == 0):
            print("Empty dataframe",training_data)
            raise ValueError("Numpy is empty after earlier filtering steps. Check your category configuration", training_data)

        if should_scale:
            # scaler = MinMaxScaler(feature_range=(0.1, 1))
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(training_data)
            training_data = pd.DataFrame(scaled_data)
            training_data.to_csv("./datasets/interim/scaled_data.csv")
            
            # visualize_data_series(
            #     title=f"scaled_data",
            #     data_series=[scaled_data],
            #     data_labels=["Targets", ],
            #     colors=["blue",],
            #     x_label="Time",
            #     y_label="Interest",
            # ).savefig("scaled_down_data.png")

            yield scaled_data, test_data, not_differenced_data, scaler
        else:
            yield training_data, test_data, not_differenced_data, None

@declare.processor()
def differencing(
        stream: Iterable[Tuple[ndarray, Optional[StandardScaler]]],
) -> Iterable[Tuple[ndarray, Optional[StandardScaler]]]:  # pragma: no cover
    for data, test_data in stream:
        diff = np.ndarray(shape=data.shape)
        for i in range(1, data.shape[0]):
            diff[i] = data[i] - data[i-1]
        
        # visualize_data_series(
        #     title=f"differenced_data",
        #     data_series=[diff],
        #     data_labels=["Targets", ],
        #     colors=["blue",],
        #     x_label="Time",
        #     y_label="Interest",
        # ).savefig("differenced_data.png")
        yield diff, test_data, data
@declare.processor()
def differencing_not(
        stream: Iterable[Tuple[ndarray, Optional[StandardScaler]]],
) -> Iterable[Tuple[ndarray, Optional[StandardScaler]]]:  # pragma: no cover
    for data, test_data in stream:
        yield data, test_data, data

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
    output_size = test_window_size
    for (array) in stream:
        # The testing set is the same as the prediction output window
        training_data = array[:-output_size]
        test_data = array[-output_size:]
        # x_train = x[:-output_size]
        # y_train = y[:-output_size]
        # x_test = x[-1:]
        # y_test = y[-1:]

        # yield ((x_train, y_train),  (x_test, y_test), scaler)
        yield training_data, test_data

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
        training_df = pd.DataFrame(training)
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
    for training_data, test_data, de_differ_data, scaler in stream:
        X = []
        Y = []
        for i in range(0, len(training_data) - input_window_size - output_window_size + 1):
            x = training_data[i:i + input_window_size]
            y = training_data[i + input_window_size:i + input_window_size + output_window_size]
            # Chose only the interest column in the y array
            y_interest = y[:, 0]
            X.append(x)
            Y.append(y_interest)
        x_test = [training_data[-input_window_size:]]
        y_test = [test_data]

        yield (np.array(X), np.array(Y)), (np.array(x_test)[:, :, 0], np.array(y_test)[:, :, 0]), scaler, training_data, de_differ_data


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

@declare.processor()
def generate_feature(
    stream: Iterable[DataFrame],
    function: Callable,
    new_feature_name: str
) -> Iterable[DataFrame]:
    for df in stream:
        if (df.size == 0):
            print("Empty dataframe", df)
            raise ValueError("Numpy is empty after earlier filtering steps. Check your category configuration", df)

        all_dates = df.index.date
        vector_function = np.vectorize(function)
        df[new_feature_name] = vector_function(all_dates)
        yield df
