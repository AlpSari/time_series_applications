import json
from typing import Optional, Callable, Union, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class DataPreprocessor:
    def __init__(self) -> None:
        pass

    @staticmethod
    def create_NARX_model(
        u,
        y,
        n_ar,
        n_b,
        return_history=False,
        n_hist_u=0,
        n_hist_y=0,
        output_as_tuple=False,
    ):
        """
        Creates a NARX (Nonlinear AutoRegressive with eXogenous inputs) model dataset.
        Parameters:
        u (array-like): The input time series data.
        y (array-like): The output time series data.
        n_ar (int): The number of autoregressive terms (past outputs) to include.
        n_b (int): The number of exogenous input terms (past inputs) to include.
        n_hist_u (int, optional): The number of historical input terms to include. Default is 0.
        n_hist_y (int, optional): The number of historical output terms to include. Default is 0.
        return_history (bool, optional): Whether to return the historical input-output data. Default is False.
        output_as_tuple (bool, optional): Whether to return the output as a tuple. Default is False.
        Returns:
        tuple: A tuple containing three numpy arrays:
            - x_data: The input data for the NARX model.
            - h_data: The historical input-output data (if return_history is True).
            - y_data: The output data for the NARX model.
        """
        x_data = []
        h_data = []
        y_data = []

        n_u = n_b + n_hist_u
        n_y = n_ar + n_hist_y
        if n_u > n_y:
            first_idx = n_u
        else:
            first_idx = n_y + 1

        for k in range(first_idx, len(y)):
            x_data.append(
                np.concatenate((u[k + 1 - n_b : k + 1], y[k - n_ar : k]))
            )  # system input
            y_data.append(y[k])  # system output
            if return_history:
                h_data.append(
                    np.concatenate(
                        (
                            u[k + 1 - n_b - n_hist_u : k + 1 - n_b],
                            y[k - n_ar - n_hist_y : k - n_ar],
                        )
                    )
                )  # system input-output history

        make_at_least_2d = lambda x: x[:, np.newaxis] if x.ndim == 1 else x
        x_data = make_at_least_2d(np.array(x_data))
        y_data = make_at_least_2d(np.array(y_data))
        if return_history:
            h_data = make_at_least_2d(np.array(h_data))
            if output_as_tuple:
                return (x_data, h_data), y_data
            else:
                return x_data, y_data, h_data
        else:
            return x_data, y_data


class SimpleDataSet(Dataset):
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
    def plot_series(df, anomaly_times: pd.Series, anomaly_windows: List[pd.Series]):
        """Helper function to plot the loaded time series data with anomalies."""
        n_anomalies = sum(df["timestamp"].isin(anomaly_times))
        plt.plot(df["timestamp"], df["value"], label="value")
        for window in anomaly_windows:
            plt.axvspan(window[0], window[1], color="red", alpha=0.3)
        plt.scatter(
            anomaly_times,
            df.loc[df["timestamp"].isin(anomaly_times), "value"],
            color="red",
            label="anomaly",
        )
        plt.legend()
        plt.title(f"Anomalies: {n_anomalies}")
        plt.show()


def main():
    u = np.arange(0, 9)
    y = np.arange(9, 18)
    print(u)
    print(y)

    x, h, y_a = DataPreprocessor.create_NARX_model(
        u, y, n_ar=0, n_b=3, return_history=True, n_hist_u=3, n_hist_y=3
    )
    print(x)
    print(h)
    print(y_a)

    ds1 = SimpleDataSet((x, h), y_a)
    ds2 = SimpleDataSet(x, y_a)

    print(ds1[0:2])
    print(ds2[0:2])


if __name__ == "__main__":
    main()
