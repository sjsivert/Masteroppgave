import math

import numpy as np
from numpy import ndarray
from sklearn.preprocessing import StandardScaler


def reverse_sliding_window(data: ndarray) -> ndarray:
    # return all bathces, but only first value in each window. And all features
    return data[:, 0, :]


def reverse_differencing(noise: ndarray, last_observed: int) -> ndarray:
    y = np.ndarray(shape=noise.shape)
    y[0] = noise[0] + last_observed
    for i in range(1, noise.shape[0]):
        y[i] = noise[i] + y[i - 1]
    return y


def reverse_scaling(data: ndarray, scaler: StandardScaler) -> ndarray:
    return scaler.inverse_transform(data)


def reverse_decrease_variance(data: ndarray) -> ndarray:
    increase_variance = lambda x: math.exp(x - 1)
    vector_func = np.vectorize(increase_variance)
    df_decreased_variance = vector_func(data)
    return df_decreased_variance
