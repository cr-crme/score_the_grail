from typing import Self
from pathlib import Path

import xmltodict
import pandas as pd
import numpy as np

from .enums import NormativeData


def _channels_map() -> dict[str, tuple[str]]:
    """
    The channels to analyse in the kinematic data.
    """
    both = ["PelvicObl", "PelvicRot", "PelvicTil"]
    joints = ["HipAbAd", "HipFlex", "HipRot", "KneeFlex", "AnkleFlex", "AnklePron"]
    left = [f"L{angle}" for angle in joints]
    right = [f"R{angle}" for angle in joints]

    return {
        "all": {
            "grail_names": tuple(f"Rotation {name}" for name in [*both, *left, *right]),
            "show_names": tuple(f"{name}" for name in [*both, *left, *right]),
        },
        "split": {
            "show_names": tuple(f"{name}" for name in [*both, *joints]),
            "grail_names_left": tuple(f"Rotation {name}" for name in [*both, *left]),
            "grail_names_right": tuple(f"Rotation {name}" for name in [*both, *right]),
        },
    }


class KinematicData:

    def __init__(self, data: np.ndarray, channel_names: tuple[str], side_names: tuple[str]) -> None:
        if data.ndim != 4:
            raise ValueError("The data must be a 4D array (time, channel, step, side).")

        self._data = data
        self._channel_names = channel_names
        self._side_names = side_names

    def keys(self) -> list:
        """
        The keys of the data.
        """
        return self._channel_names

    def map(self, normative_data: NormativeData = NormativeData.CROUCHGAIT) -> Self:
        """
        Compute the Movement Analysis Profile (root mean square error) for each channel.

        Parameters
        ----------
        normative_data : NormativeData
            The normative data to compare to.

        Returns
        -------
        KinematicData
            The root mean square error for each channel.
        """
        norm = KinematicData.from_normative_data(normative_data)

        if not self.has_side:
            raise ValueError("The data does not contain side information.")

        return ((self.mean_step - norm) ** 2).mean_time.sqrt

    def gps(self, normative_data: NormativeData = NormativeData.CROUCHGAIT) -> Self:
        """
        Compute the GPS (Gait Profile Score) for each channel.

        Parameters
        ----------
        normative_data : NormativeData
            The normative data to compare to.

        Returns
        -------
        KinematicData
            The GPS for each channel.
        """
        map = self.map(normative_data=normative_data)
        return KinematicData(
            data=map._data.mean(axis=1)[:, None, :, :], channel_names=["gps"], side_names=self._side_names
        )

    @property
    def data(self) -> np.ndarray:
        """
        The kinematic data.
        """
        return self._data.squeeze()

    @property
    def channel_names(self) -> tuple[str]:
        """
        The names of the channels.
        """
        return self._channel_names

    @property
    def side_names(self) -> tuple[str]:
        """
        The names of the sides.
        """
        return self._side_names

    @property
    def has_steps(self) -> bool:
        """
        If the data contains multiple steps.
        """
        return self._data.shape[2] > 1

    @property
    def has_side(self) -> bool:
        """
        If the data contains both side information.
        """
        return len(self._side_names) == 2

    @property
    def mean_time(self) -> Self:
        """
        The mean value for each time step.
        """
        return KinematicData(
            data=self._data.mean(axis=0)[None, :, :, :], channel_names=self._channel_names, side_names=self._side_names
        )

    @property
    def mean_step(self) -> Self:
        """
        The mean value for each channel.
        """
        if not self.has_steps:
            # No multiple steps so we return the data as is
            return self

        return KinematicData(
            data=self._data.mean(axis=2)[:, :, None, :], channel_names=self._channel_names, side_names=self._side_names
        )

    @property
    def sqrt(self) -> Self:
        """
        The square root of the data.
        """
        return KinematicData(data=np.sqrt(self._data), channel_names=self._channel_names, side_names=self._side_names)

    @property
    def left(self) -> Self:
        """
        The kinematic data for the left side.
        """
        return self._get_side("left")

    @property
    def right(self) -> Self:
        """
        The kinematic data for the right side.
        """
        return self._get_side("right")

    def _get_side(self, side: str) -> Self:
        """
        Get the data for a specific side.

        Parameters
        ----------
        side : str
            The side to get the data for.

        Returns
        -------
        KinematicData
            The data for the side.
        """
        if side not in self._side_names:
            raise ValueError(f"The side must be in {self._side_names}.")

        # Check if the keys are str or tuple. If they are str, we dont have side information
        if not self.has_side:
            return self

        # Find the indices of the side
        side_index = self._side_names.index(side)

        return KinematicData(
            data=self._data[:, :, :, side_index : side_index + 1], channel_names=self._channel_names, side_names=(side,)
        )

    @classmethod
    def from_csv(cls, file_path: str) -> Self:
        """
        Extract the data from the export of the GOAT software.

        Parameters
        ----------
        file_path : str
            The path to the CSV file containing the kinematic data.

        Returns
        -------
        KinematicData
            The kinematic data extracted from the file.
        """
        try:
            data = pd.read_csv(file_path)[list(_channels_map()["all"]["grail_names"])].to_numpy()[:, :, None, None]
            return cls(
                data=np.concatenate((data, data), axis=3),
                channel_names=_channels_map()["all"]["show_names"],
                side_names=("left", "right"),
            )
        except KeyError:
            raise ValueError("The CSV file does not contain all the expected columns.")

    @classmethod
    def from_normalized_csv(cls, file_path: str) -> Self:
        """
        Extract the data from the export of the GOAT software when normalized is enabled.

        The structure of the file is a bit peculiar. It is as follows:
        - The first column contains if the data is for the left or right side
        - The second column contains the channel name
        - The third column contains the step number
        - The fourth to last columns contain the data for each frame (normalized over 101 frames)
        Each step is stored onto 121 rows

        Parameters
        ----------
        file_path : str
            The path to the CSV file containing the normalized kinematic data.

        Returns
        -------
        KinematicData
            The kinematic data extracted from the file
        """

        raw_data = pd.read_csv(file_path, header=None)
        header_count = 121  # The expected number of elements in the header (supplied over the rows)

        # The last value in the third column necessarily contains the total number of steps
        step_count = int(raw_data.iloc[-1, 2]) + 1  # 0-based index

        # As a sanity check, the total number of rows should be header_count * step_count * 2 (left and right)
        if raw_data.shape[0] != header_count * step_count * 2:
            raise ValueError("The CSV file does not contain the expected number of rows.")

        # Extract a piece of the header from the first rows
        header = raw_data.iloc[:header_count, 1].values

        # Now we can reorganize the data into a DataFrame with the headers as columns and steps separated by dimension
        data = np.array(raw_data.iloc[:, 3:]).T.reshape((101, header_count, step_count, 2), order="F")

        # Find the index of the header that correspond to channels to keep
        left_header_to_keep = list(_channels_map()["split"]["grail_names_left"])
        right_header_to_keep = list(_channels_map()["split"]["grail_names_right"])
        if len(left_header_to_keep) != len(right_header_to_keep):
            raise ValueError("The left and right channels do not have the same length.")
        header_count = len(left_header_to_keep)

        left_indices_to_keep = [list(header).index(value) for value in left_header_to_keep]
        right_indices_to_keep = [list(header).index(value) for value in right_header_to_keep]
        if (len(left_indices_to_keep) != header_count) or (len(right_indices_to_keep) != header_count):
            raise ValueError("The CSV file does not contain all the expected channels.")

        # Remove the extra columns
        left_data = data[:, left_indices_to_keep, :, 0:1]
        right_data = data[:, right_indices_to_keep, :, 1:2]

        return cls(
            data=np.concatenate((left_data, right_data), axis=3),
            channel_names=_channels_map()["split"]["show_names"],
            side_names=("left", "right"),
        )

    @classmethod
    def from_mox(cls, file_path: str) -> Self:
        """
        Extract the data from the MOX file directly output from the GRAIL software.

        Parameters
        ----------
        file_path : str
            The path to the MOX file containing the kinematic data.

        Returns
        -------
        KinematicData
            The kinematic data extracted from the file.
        """
        with open(file_path) as f:
            raw_data = xmltodict.parse(f.read())

        frame_count = int(raw_data["moxie_viewer_datafile"]["viewer_header"]["nr_of_samples"])

        channels_to_keep = list(_channels_map()["all"]["grail_names"])
        header = []
        data = np.ndarray((frame_count, 0))
        for channel in raw_data["moxie_viewer_datafile"]["viewer_data"]["viewer_channel"]:
            if channel["channel_label"] in channels_to_keep:
                header.append(channel["channel_label"])
                data = np.hstack(
                    (data, np.array(str(channel["raw_channel_data"]["channel_data"]).split(" "), dtype=float)[:, None])
                )

        # Reorganise and reshape the data into a 4D array
        data = pd.DataFrame(data, columns=header)[channels_to_keep].to_numpy()
        data = np.concatenate((data[:, :, None, None], data[:, :, None, None]), axis=3)

        try:
            return cls(data=data, channel_names=_channels_map()["all"]["show_names"], side_names=("left", "right"))
        except KeyError:
            raise ValueError("The MOX file does not contain all the expected channels.")

    @classmethod
    def from_normative_data(cls, file: NormativeData) -> Self:
        """
        Create a KinematicData object from the normative data file.

        The data are stored in a "header\n/data1\n/data2\n..." format over rows. The data1 are the actual data. I do not
        know what data2 is for.

        Parameters
        ----------
        file : NormativeData
            The normative data file to load

        Returns
        -------
        KinematicData
            The kinematic data extracted from the file
        """

        # The normative file is next to the current file
        with open(Path(__file__).parent / file.value) as f:
            all_lines = f.readlines()

        # Remove the nextline character
        all_lines = [line.strip() for line in all_lines]

        # Parse the value for each header
        left_header_to_keep = list(_channels_map()["split"]["grail_names_left"])
        right_header_to_keep = list(_channels_map()["split"]["grail_names_right"])
        if len(left_header_to_keep) != len(right_header_to_keep):
            raise ValueError("The left and right channels do not have the same length.")
        header_count = len(left_header_to_keep)

        data = np.ndarray((101, header_count, 1, 2))  # 0 for the steps, 2 for left and right
        left_start = all_lines.index("left")
        right_start = all_lines.index("right")
        if left_start > right_start:
            raise ValueError("The normative data file are expected to have left first.")
        for header_index in range(header_count):
            # Find all indices of the header (0 being the left and 1 the right, this assumes the data is ordered as such)
            left_index = [i for i, line in enumerate(all_lines) if line == left_header_to_keep[header_index]][0]
            right_index = [i for i, line in enumerate(all_lines) if line == right_header_to_keep[header_index]][1]

            # +1 for next line
            data[:, header_index, 0, 0] = np.array(all_lines[left_index + 1].split(" "), dtype=float)
            data[:, header_index, 0, 1] = np.array(all_lines[right_index + 1].split(" "), dtype=float)

        # Recast the data into a DataFrame
        return cls(data=data, channel_names=_channels_map()["split"]["show_names"], side_names=("left", "right"))

    def __getitem__(self, key: str) -> Self:
        if key not in self.keys():
            raise KeyError("The key is not in the data.")

        index = self.keys().index(key)
        return KinematicData(
            data=self._data[:, index : index + 1, :, :], channel_names=(key,), side_names=self._side_names
        )

    def __add__(self, other: Self) -> Self:
        if not isinstance(other, KinematicData):
            raise TypeError("The other object is not a KinematicData object.")

        if self._data.shape != other._data.shape:
            raise ValueError("The data shapes are not the same.")

        return KinematicData(
            data=self._data + other._data, channel_names=self._channel_names, side_names=self._side_names
        )

    def __sub__(self, other: Self) -> Self:
        if not isinstance(other, KinematicData):
            raise TypeError("The other object is not a KinematicData object.")

        if self._data.shape != other._data.shape:
            raise ValueError("The data shapes are not the same.")

        return KinematicData(
            data=self._data - other._data, channel_names=self._channel_names, side_names=self._side_names
        )

    def __pow__(self, other: float) -> Self:
        if not isinstance(other, (int, float)):
            raise TypeError("The other object is not a number.")

        return KinematicData(data=self._data**other, channel_names=self._channel_names, side_names=self._side_names)
