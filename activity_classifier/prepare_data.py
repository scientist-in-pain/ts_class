import pandas as pd
import numpy as np

from activity_classifier.config import OBS
from scipy.interpolate import interp1d


def interpolate_data(data, seconds, end_frame_rate):
    """
    Calculates given data frame rate and interpolates data to desired end_frame_rate
    :param data: Recorded data
    :param seconds: Recorded data total duration in seconds
    :param end_frame_rate: Desired frame rate (frames per second)
    :return: Interpolated data per end_frame_rate
    """
    recorded_frame_rate = (len(data) - 1) / seconds
    frame_interval = 1 / end_frame_rate
    frame_labels = list(range(0, len(data)))
    x = np.multiply(frame_labels, 1 / recorded_frame_rate)
    interpolate_func = interp1d(x, data, "linear", fill_value="extrapolate")

    x_new = np.arange(0, seconds + frame_interval, frame_interval)
    y_new = interpolate_func(x_new)
    return pd.Series(y_new)


def prepare_data(data, seconds, end_frame_rate):
    """
    Prepare data for model application, including interpolation.
    :param data: Recorded data
    :param seconds: Recorded data total duration in seconds
    :param end_frame_rate: Desired frame rate
    :return: Input data with additional column containing interpolated data per end_frame_rate
    """
    assert data.isnull().sum().sum() == 0, AssertionError("Data contains empty values, correct and retry")
    assert np.isinf(data).values.sum() == 0, AssertionError("Data contains inf values, correct and retry")
    data[OBS] = data.apply(lambda x: interpolate_data(x, seconds, end_frame_rate))
    return data
