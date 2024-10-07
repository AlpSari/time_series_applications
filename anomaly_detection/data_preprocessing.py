import json
from typing import Optional, Callable, Union, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class DataPreprocessor:
    """Class to store convenience functions for data preprocessing."""
    def __init__(self) -> None:
        pass

    @staticmethod
    def create_NARX_model(
        u: np.ndarray,
        y: np.ndarray,
        n_b: int = 0,
        n_ar: int = 0,
        return_history: bool = False,
        n_hist_u: int = 0,
        n_hist_y: int = 0,
        output_as_tuple: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Creates a NARX (Nonlinear AutoRegressive with eXogenous inputs) model dataset.

        NARX Model is defined as:
            y(k) = f(u(k), u(k-1), ..., u(k-(n_b-1)), y(k-1), y(k-2), ..., y(k-n_ar))

        Target vector: y(k)
        Input vector: [u(k), u(k-1), ..., u(k-(n_b-1)), y(k-1), y(k-2), ..., y(k-n_ar), ]

        Parameters
        ----------
        u : np.ndarray
            The input time series data.
        y : np.ndarray
            The output time series data.
        n_b : int
            The number of exogenous input terms to include. Default is 0.
        n_ar : int
            The number of autoregressive terms to include. Default is 0.
        return_history : bool, optional
            Whether to return the historical input-output data. Default is False.
        n_hist_u : int, optional
            The number of historical input terms to include. Default is 0.
        n_hist_y : int, optional
            The number of historical output terms to include. Default is 0.
        output_as_tuple : bool, optional
            Whether to return the output as a tuple. Default is False.
        Returns
        -------
        - Tuple[np.ndarray, np.ndarray] or Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - x_data: The input data for the NARX model.
            - y_data: The output data for the NARX model.
            - h_data (optional): The historical input-output data (if return_history is True).
        """
        x_data = []
        y_data = []
        h_data = []

        # Define the total number of input and output lags
        if return_history:
            total_u_lags = n_b + n_hist_u
            total_y_lags = n_ar + n_hist_y
        else:
            total_u_lags = n_b
            total_y_lags = n_ar

        # Determine the first valid index based on max lags
        first_idx = max(total_u_lags-1, total_y_lags)

        # Loop to collect x_data (inputs), y_data (targets), and h_data (history, if needed)
        for k in range(first_idx, len(y)):
            # x_data consists of recent input and output lags (NARX terms)
            current_u_lags = u[k + 1 - n_b: k + 1] if n_b > 0 else np.array([])
            current_y_lags = y[k - n_ar: k] if n_ar > 0 else np.array([])
            x_data.append(np.concatenate((current_u_lags, current_y_lags)))

            # y_data is the current target value y(t)
            y_data.append(y[k])

            # If history is requested, build the h_data
            if return_history:
                past_u_hist = u[k + 1 - n_b - n_hist_u: k + 1 - n_b] if n_hist_u > 0 else np.array([])
                past_y_hist = y[k - n_ar - n_hist_y: k - n_ar] if n_hist_y > 0 else np.array([])
                h_data.append(np.concatenate((past_u_hist, past_y_hist)))

        # Convert lists to numpy arrays
        x_data = DataPreprocessor._make_at_least_2d(np.array(x_data))
        y_data = DataPreprocessor._make_at_least_2d(np.array(y_data))

        if return_history:
            h_data = DataPreprocessor._make_at_least_2d(np.array(h_data))
            if output_as_tuple:
                return (x_data, h_data), y_data
            else:
                return x_data, y_data, h_data
        else:
            return x_data, y_data

    @staticmethod
    def create_AE_model(y, n_window=1):
        """
        Creates a dataset for an Autoencoder (AE) model by generating
        overlapping windows of the input data.

        Output data is time series y(k) is
        [
            [y(0), y(1), ..., y(n_window-1)],
            [y(1), y(2), ..., y(n_window)],
            ...
            [y(-n_window), y(-n_window+1), ..., y(-1)]
        ]

        Parameters ----------
        y : array-like
            The input time series data.
        n_window : int, optional
            The size of the window to create overlapping segments. Default is 1.

        Returns
        -------
        numpy.ndarray
            A 2D array where each row is a window of the input data.
        """
        y_data = []
        for k in range(0, len(y)-n_window+1):
            y_data.append(y[k: k+n_window])
        y_data = DataPreprocessor._make_at_least_2d(np.array(y_data))
        return y_data

    @staticmethod
    def _make_at_least_2d(x: np.ndarray) -> np.ndarray:
        return x[:, np.newaxis] if x.ndim == 1 else x


class SimpleDataSet(Dataset):
    """
    A Pytorch Dataset wrapper for numpy array input and outputs.

    Input can be a single numpy array or a tuple of numpy arrays.
    Output is a single numpy array.
    """
    def __init__(
        self,
        x_data: Union[np.ndarray, Tuple[np.ndarray, ...]],
        y_data: np.ndarray,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.y_data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(self.x_data, tuple):
            x = tuple(
                x_part[idx] if x_part.ndim == 1 else x_part[idx, :]
                for x_part in self.x_data
            )
        else:
            x = self.x_data[idx] if self.x_data.ndim == 1 else self.x_data[idx, :]

        y = self.y_data[idx] if self.y_data.ndim == 1 else self.y_data[idx, :]

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y


class DataLoaderNAB:
    """
    Class to interact with the Numenta Anomaly Benchmark(NAB) dataset time series anomaly data.

    NAB repository: https://github.com/numenta/NAB
    """
    @staticmethod
    def load_nab_dataset(root_dir, data_dir, label_json_file, window_json_file):
        """
        Load the NAB dataset from the specified directory and JSON files.
        Args:
            root_dir (str): The root directory where the data is stored.
            data_dir (str): The directory containing the dataset CSV file.
            label_json_file (str): The path to the JSON file containing anomaly labels.
            window_json_file (str): The path to the JSON file containing anomaly windows.
        Returns:
            tuple: A tuple containing:
                - data_ts (pd.DataFrame): The time series data with timestamps and values.
                - anomaly_times (pd.Series): The timestamps of anomalies.
                - anomaly_windows (list of pd.DatetimeIndex): The periods of anomaly windows.
        """
        data_path = root_dir + data_dir
        data_ts = pd.read_csv(
            data_path, parse_dates=["timestamp"], dtype={"value": np.float32}
        )
        # Load labels (point in time)
        with open(label_json_file, "r") as key_file:
            key = json.load(key_file)
            anomaly_times = pd.to_datetime(key[data_dir])
        # Load anomaly window (period)
        with open(window_json_file, "r") as window_file:
            windows = json.load(window_file)

            anomaly_windows = [pd.to_datetime(window) for window in windows[data_dir]]

        return data_ts, anomaly_times, anomaly_windows

    def load_data(data_csv_path, labels_json_path, windows_json_path):
        """
        Helper function to load NAB data in custom format.

        Args:
            data_csv_path (str): Path to the CSV file containing the time series data.
            labels_json_path (str): Path to the JSON file containing anomaly labels.
            windows_json_path (str): Path to the JSON file containing anomaly windows.

        Returns:
            tuple: A tuple containing:
                - df (pd.DataFrame): DataFrame with the time series data.
                - anomaly_times (pd.DatetimeIndex): DatetimeIndex of anomaly times.
                - anomaly_windows (list of pd.DatetimeIndex): List of DatetimeIndex objects for anomaly windows.
        """
        df = pd.read_csv(
            data_csv_path, parse_dates=["timestamp"], dtype={"value": np.float32}
        )
        with open(labels_json_path, "r") as f:
            anomaly_times_dict = json.load(f)
        anomaly_times = pd.to_datetime(anomaly_times_dict["anomaly"])
        with open(windows_json_path, "r") as f:
            anomaly_windows_dict = json.load(f)
        anomaly_windows = [
            pd.to_datetime(window) for window in anomaly_windows_dict["anomaly_windows"]
        ]
        return df, anomaly_times, anomaly_windows

    @staticmethod
    def plot_series(
        df: pd.DataFrame,
        anomaly_times: pd.Series,
        anomaly_windows: List[pd.Series],
        ax=None
    ):
        """Helper function to plot the loaded time series data with anomalies."""
        n_anomalies = sum(df["timestamp"].isin(anomaly_times))

        if ax is None:
            fig, ax = plt.subplots()

        # Plot time series
        ax.plot(df["timestamp"], df["value"], color='blue', label="data")

        # Plot anomaly windows
        for w_idx, window in enumerate(anomaly_windows):
            if w_idx == 0:
                ax.axvspan(window[0], window[1], color="black", alpha=0.3, label="anomaly window")
            else:
                ax.axvspan(window[0], window[1], color="black", alpha=0.3)

        # Plot anomalies
        ax.scatter(
            anomaly_times,
            df.loc[df["timestamp"].isin(anomaly_times), "value"],
            color="black",
            label="anomaly",
        )

        # Styling
        ax.legend()
        ax.set_title(f"Anomalies: {n_anomalies}")

        if ax is None:
            plt.show()

        return ax
