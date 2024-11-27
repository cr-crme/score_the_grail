from typing import Self
from pathlib import Path

import xmltodict
import pandas as pd
import numpy as np

from .enums import NormativeData


def _channels_to_analyse() -> tuple[str]:
    """
    The channels to analyse in the kinematic data.
    """
    return tuple(
        f"Rotation {name}"
        for name in (
            "LAnkleAbAd",
            "LAnkleFlex",
            "LAnklePron",
            "LAnkleRot",
            "LHipAbAd",
            "LHipFlex",
            "LHipRot",
            "LKneeAbAd",
            "LKneeFlex",
            "LKneeRot",
            "PelvicObl",
            "PelvicRot",
            "PelvicTil",
            "RAnkleAbAd",
            "RAnkleFlex",
            "RAnklePron",
            "RAnkleRot",
            "RHipAbAd",
            "RHipFlex",
            "RHipRot",
            "RKneeAbAd",
            "RKneeFlex",
            "RKneeRot",
            "TrunkFlex",
            "TrunkRot",
            "TrunkTilt",
        )
    )


class KinematicData:
    channels_to_analyse = _channels_to_analyse()

    def __init__(self, data: pd.DataFrame):
        self.data: pd.DataFrame = data

    def keys(self) -> list:
        """
        The keys of the data.
        """
        return self.data.keys()

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

        left = ((self.left.mean_step - norm.left) ** 2).mean_time.sqrt.data.to_numpy()[None, :, None]
        right = ((self.right.mean_step - norm.right) ** 2).mean_time.sqrt.data.to_numpy()[None, :, None]

        return KinematicData(
            data=pd.DataFrame(
                np.concatenate((left, right), axis=2).reshape([1, -1]),
                columns=pd.MultiIndex.from_product([KinematicData.channels_to_analyse, ("left", "right")]),
            )
        )

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
        map_left = map.left.data.mean(axis=1).to_numpy()[None, :]
        map_right = map.right.data.mean(axis=1).to_numpy()[None, :]
        return KinematicData(data=pd.DataFrame(np.concatenate([map_left, map_right]).T, columns=["left", "right"]))

    @property
    def has_steps(self) -> bool:
        """
        If the data contains multiple steps.
        """
        return isinstance(self.keys()[0], tuple) and isinstance(self.keys()[0][1], np.int_)

    @property
    def has_side(self) -> bool:
        """
        If the data contains side information.
        """
        keys = self.keys()
        if isinstance(keys[0], tuple):
            keys = keys[-1]
        return keys[-1] in ("left", "right")

    @property
    def mean_time(self) -> Self:
        """
        The mean value for each time step.
        """
        return KinematicData(data=self.data.mean(axis=0))

    @property
    def mean_step(self) -> Self:
        """
        The mean value for each channel.
        """
        keys = self.keys()

        # Check if the keys are str or tuple. If they are str, we dont have multiple steps
        if not self.has_steps:
            # No multiple steps
            return self

        # Find all the indices that shares the angles for both left and right
        headers = KinematicData.channels_to_analyse
        data = np.ndarray((self.data.shape[0], len(headers), 2)) * np.nan
        for header_index, header in enumerate(headers):
            indices_left = [i for i, key in enumerate(keys) if key[0] == header and key[-1] == "left"]
            indices_right = [i for i, key in enumerate(keys) if key[0] == header and key[-1] == "right"]

            # Compute the mean for each step
            data[:, header_index, 0] = self.data[keys[indices_left]].mean(axis=1)
            data[:, header_index, 1] = self.data[keys[indices_right]].mean(axis=1)

        has_left = np.isfinite(data[:, :, 0]).any()
        has_right = np.isfinite(data[:, :, 1]).any()
        if not has_left and not has_right:
            raise ValueError("No data to compute the mean step.")

        if has_left and not has_right:
            return KinematicData(data=pd.DataFrame(data[:, :, 0], columns=headers))
        elif has_right and not has_left:
            return KinematicData(data=pd.DataFrame(data[:, :, 1], columns=headers))
        else:
            labels = pd.MultiIndex.from_product([headers, ("left", "right")])
            return KinematicData(data=pd.DataFrame(data.reshape(101, -1), columns=labels))

    @property
    def sqrt(self) -> Self:
        """
        The square root of the data.
        """
        return KinematicData(data=self.data.apply(np.sqrt))

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
        if side not in ("left", "right"):
            raise ValueError("The side must be either 'left' or 'right'.")

        # Check if the keys are str or tuple. If they are str, we dont have side information
        if not self.has_side:
            return self

        # Find the indices of the side
        keys = self.keys()
        if side in keys:
            side_indices = [i for i, key in enumerate(keys) if key == side]
        else:
            side_indices = [i for i, key in enumerate(keys) if key[-1] == side]

        if self.has_steps:
            return KinematicData(data=self.data[keys[side_indices]])
        else:
            data = self.data[keys[side_indices]].to_numpy()
            channel_header = KinematicData.channels_to_analyse
            if len(channel_header) in data.shape:
                return KinematicData(data=pd.DataFrame(data, columns=channel_header))
            else:
                return KinematicData(data=pd.DataFrame(data, columns=[side]))

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
            return cls(data=pd.read_csv(file_path)[list(KinematicData.channels_to_analyse)])
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

        # Find the index of the header that correspond to channels to keep
        header_to_keep = list(KinematicData.channels_to_analyse)
        index_to_keep = [i for i, name in enumerate(header) if name in header_to_keep]
        if len(index_to_keep) != len(header_to_keep):
            raise ValueError("The CSV file does not contain all the expected channels.")

        # Now we can reorganize the data into a DataFrame with the headers as columns and steps separated by dimension
        data = np.array(raw_data.iloc[:, 3:]).T.reshape((101, header_count, step_count, 2), order="F")

        # Remove the extra columns
        data = data[:, index_to_keep, :, :]

        # Recast the data into a DataFrame
        label_first = header_to_keep
        label_second = tuple(range(step_count))
        label_third = ("left", "right")
        return cls(
            pd.DataFrame(
                data[:, :, :, :].reshape(101, -1),
                columns=pd.MultiIndex.from_product([label_first, label_second, label_third]),
            )
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

        channels_to_keep = list(KinematicData.channels_to_analyse)
        header = []
        data = np.ndarray((frame_count, 0))
        for channel in raw_data["moxie_viewer_datafile"]["viewer_data"]["viewer_channel"]:
            if channel["channel_label"] in channels_to_keep:
                header.append(channel["channel_label"])
                data = np.hstack(
                    (data, np.array(str(channel["raw_channel_data"]["channel_data"]).split(" "), dtype=float)[:, None])
                )

        try:
            return cls(data=pd.DataFrame(data, columns=header)[channels_to_keep])
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
        header_to_keep = list(KinematicData.channels_to_analyse)
        data = np.ndarray((101, len(header_to_keep), 2))  # 2 for left and right
        left_index = all_lines.index("left")
        right_index = all_lines.index("right")
        for header_index, header in enumerate(header_to_keep):
            # Find all indices of the header (one for each side)
            indices = [i for i, line in enumerate(all_lines) if line == header]

            # Sanity check (should be exactly 2 separated by the difference between the left and right index)
            if len(indices) != 2 and indices[0] + abs(right_index - left_index) != indices[1]:
                raise ValueError("The normative data file does not contain all the expected channels.")

            data[:, header_index, 0] = np.array(all_lines[indices[0] + 1].split(" "), dtype=float)
            data[:, header_index, 1] = np.array(all_lines[indices[1] + 1].split(" "), dtype=float)

        # Recast the data into a DataFrame
        label_first = header_to_keep
        label_second = ("left", "right")
        return cls(
            pd.DataFrame(
                data[:, :, :].reshape(101, -1), columns=pd.MultiIndex.from_product([label_first, label_second])
            )
        )

    def __add__(self, other: Self) -> Self:
        if not isinstance(other, KinematicData):
            raise TypeError("The other object is not a KinematicData object.")

        return KinematicData(data=self.data + other.data)

    def __sub__(self, other: Self) -> Self:
        if not isinstance(other, KinematicData):
            raise TypeError("The other object is not a KinematicData object.")

        return KinematicData(data=self.data - other.data)

    def __mul__(self, other: float) -> Self:
        if not isinstance(other, (int, float)):
            raise TypeError("The other object is not a number.")

        return KinematicData(data=self.data * other)

    def __truediv__(self, other: float) -> Self:
        if not isinstance(other, (int, float)):
            raise TypeError("The other object is not a number.")

        return KinematicData(data=self.data / other)

    def __pow__(self, other: float) -> Self:
        if not isinstance(other, (int, float)):
            raise TypeError("The other object is not a number.")

        return KinematicData(data=self.data**other)
