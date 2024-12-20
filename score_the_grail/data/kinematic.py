import io
import os
from typing import Self
from pathlib import Path

import xmltodict
import pandas as pd
import numpy as np

from .enums import NormativeData
from .file_io import FileReader


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
            "normalized_names": tuple(f"Rotation {name}" for name in [*both, *joints]),
        },
        "split": {
            "show_names": tuple(f"{name}" for name in [*both, *joints]),
            "grail_names_left": tuple(f"Rotation {name}" for name in [*both, *left]),
            "grail_names_right": tuple(f"Rotation {name}" for name in [*both, *right]),
        },
    }


class KinematicData:

    def __init__(
        self, data: dict[str, np.ndarray], channel_names: tuple[str], data_std: dict[str, np.ndarray] | None = None
    ) -> None:
        """
        Create a new KinematicData object.

        Parameters
        ----------
        data : dict[str, np.ndarray]
            The kinematic data (time, channel, step) for the sides (str).
        channel_names : tuple[str]
            The names of the channels.
        data_std : dict[str, np.ndarray] | None
            The standard deviation of the kinematic data

        """
        for value in data.values():
            if value.ndim != 3:
                raise ValueError("The data must be a 3D array (time, channel, step).")

        if data_std is None:
            data_std = {side: np.zeros_like(data[side]) for side in data.keys()}
        for side in data.keys():
            if side not in data_std:
                raise ValueError("The standard deviation must have the same sides as the data.")
            if data[side].shape != data_std[side].shape:
                raise ValueError("The data and standard deviation must have the same shape.")

        self._data = data
        self._data_std = data_std
        self._channel_names = channel_names

    def keys(self) -> list:
        """
        The keys of the data.
        """
        return self._channel_names

    def map(self, normative_data: NormativeData = NormativeData.NORMAL) -> Self:
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

    def gps(self, normative_data: NormativeData = NormativeData.NORMAL) -> Self:
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
            data={side: map._data[side].mean(axis=1)[:, None, :] for side in map.side_names},
            channel_names=["gps"],
        )

    @property
    def to_numpy(self) -> np.ndarray:
        """
        The kinematic data.
        """
        if len(self.side_names) > 1:
            raise ValueError(
                "It is not possible to merge the data with more than one side. "
                "Please call '.left' or '.right' before calling '.to_numpy'."
            )
        return self._data[self.side_names[0]].squeeze()

    @property
    def std_to_numpy(self) -> np.ndarray:
        """
        The standard deviation data as numpy array.
        """
        if len(self.side_names) > 1:
            raise ValueError(
                "It is not possible to merge the data with more than one side. "
                "Please call '.left' or '.right' before calling '.to_numpy'."
            )
        return self._data_std[self.side_names[0]].squeeze()

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
        return tuple(self._data.keys())

    @property
    def has_steps(self) -> bool:
        """
        If the data contains multiple steps. This is true as soon as at least one side has more than one step.
        """
        return sum(data.shape[2] > 1 for data in self._data.values()) > 0

    @property
    def has_side(self) -> bool:
        """
        If the data contains both side information.
        """
        return "left" in self.side_names and "right" in self.side_names

    @property
    def mean_time(self) -> Self:
        """
        The mean value for each time step.
        """
        return KinematicData(
            data={side: self._data[side].mean(axis=0)[None, :, :] for side in self.side_names},
            channel_names=self._channel_names,
            data_std={side: self._data_std[side].mean(axis=0)[None, :, :] for side in self.side_names},
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
            data={side: self._data[side].mean(axis=2)[:, :, None] for side in self.side_names},
            channel_names=self._channel_names,
            data_std={side: self._data_std[side].mean(axis=2)[:, :, None] for side in self.side_names},
        )

    @property
    def sqrt(self) -> Self:
        """
        The square root of the data.
        """
        return KinematicData(
            data={side: np.sqrt(self._data[side]) for side in self.side_names},
            channel_names=self._channel_names,
        )

    @property
    def left(self) -> Self:
        """
        The kinematic data for the left side.
        """
        return self.get_side("left")

    @property
    def right(self) -> Self:
        """
        The kinematic data for the right side.
        """
        return self.get_side("right")

    def get_side(self, side: str) -> Self:
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
        if side not in self.side_names:
            raise ValueError(f"The side must be in {self.side_names}.")

        # If we already have only the requested side in the data, no need to do anything apart from returning the data themselves
        if not self.has_side:
            return self

        return KinematicData(
            data={side: self._data[side]}, channel_names=self._channel_names, data_std={side: self._data_std[side]}
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
        data = pd.DataFrame(data, columns=header)[channels_to_keep].to_numpy()[:, :, None]

        try:
            return cls(data={"both": data}, channel_names=_channels_map()["all"]["show_names"])
        except KeyError:
            raise ValueError("The MOX file does not contain all the expected channels.")

    @classmethod
    def from_csv(cls, file_path: str, encryption_key: str | None = None) -> Self:
        """
        Extract the data from the export of the GOAT software.

        Parameters
        ----------
        file_path : str
            The path to the CSV file containing the kinematic data.
        encryption_key : str, optional
            The encryption key to decrypt the data, if None is provided, the data is assumed to be unencrypted.

        Returns
        -------
        KinematicData
            The kinematic data extracted from the file.
        """
        try:
            if encryption_key is not None:
                decrypted_data = FileReader(encryption_key=encryption_key).read_encrypted_file(file_path)
                file_path = io.BytesIO(decrypted_data)

            data = pd.read_csv(file_path)[list(_channels_map()["all"]["grail_names"])].to_numpy()[:, :, None]
            return cls(data={"both": data}, channel_names=_channels_map()["all"]["show_names"])
        except KeyError:
            raise ValueError("The CSV file does not contain all the expected columns.")

    @classmethod
    def from_normalized_csv(cls, file_path: str, encryption_key: str | None = None) -> Self:
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
        encryption_key : str, optional
            The encryption key to decrypt the data, if None is provided, the data is assumed to be unencrypted.

        Returns
        -------
        KinematicData
            The kinematic data extracted from the file
        """

        if encryption_key is not None:
            decrypted_data = FileReader(encryption_key=encryption_key).read_encrypted_file(file_path)
            file_path = io.BytesIO(decrypted_data)

        raw_data = pd.read_csv(file_path, header=None).to_numpy()

        # Find the header from the first step of the left side (assuming all steps follow the same structure)
        header_count = raw_data[(raw_data[:, 0] == "left") & (raw_data[:, 2] == 0), :].shape[0]
        header = list(raw_data[:header_count, 1])

        data = {}
        for side in ["left", "right"]:
            sided_data = raw_data[raw_data[:, 0] == side, :]

            # Find the index of the header that correspond to channels to keep
            header_to_keep = list(_channels_map()["split"][f"grail_names_{side}"])
            try:
                header_indices = [header.index(value) for value in header_to_keep]
            except ValueError:
                raise ValueError("The CSV file does not contain all the expected channels.")

            # The last value in the third column contains the total number of steps, the last value is the number of steps
            step_count = int(sided_data[-1, 2]) + 1  # 0-based index

            # As a sanity check, the total number of rows should be header_count * step_count
            if sided_data.shape[0] != header_count * step_count:
                raise ValueError(f"The CSV file does not contain the expected number of rows for side {side}.")

            # Now we can reorganize the data into the final format with the headers as columns and steps as 3rd dimension
            data[side] = np.array(
                sided_data[:, 3:].T.reshape((101, header_count, step_count), order="F")[:, header_indices, :],
                dtype=float,
            )

        return cls(data=data, channel_names=_channels_map()["split"]["show_names"])

    @classmethod
    def from_normative_csv(cls, file_path: str, std_file_path: str | None, encryption_key: str | None = None) -> Self:
        """
        Extract the data provided by MOTEK for the normative data.

        Parameters
        ----------
        file_path : str
            The path to the CSV file containing the kinematic data.
        std_file_path : str | None
            The path to the CSV file containing the standard deviation of the kinematic data, if available.
        encryption_key : str, optional
            The encryption key to decrypt the data, if None is provided, the data is assumed to be unencrypted.

        Returns
        -------
        KinematicData
            The kinematic data extracted from the file.
        """

        if encryption_key is not None:
            decrypted_data = FileReader(encryption_key=encryption_key).read_encrypted_file(file_path)
            file_path = io.BytesIO(decrypted_data)

        raw_data = pd.read_csv(file_path, header=None).to_numpy()
        # Replace the headers from "Type Dof SideCycle" to "Type SDof" where S is the side (L or R)
        header: list[str] = list(raw_data[0, :])
        generic_header_to_keep = list(_channels_map()["all"]["normalized_names"])
        has_std_data = std_file_path is not None
        if has_std_data:
            if encryption_key is not None:
                decrypted_std_data = FileReader(encryption_key=encryption_key).read_encrypted_file(std_file_path)
                std_file_path = io.BytesIO(decrypted_std_data)

            raw_data_std = pd.read_csv(std_file_path, header=None).to_numpy()
            header_std = list(raw_data_std[0, :])
            if header != header_std:
                raise ValueError("The CSV files do not contain the same headers.")
            # From that point on, we can assume that the headers are the same, so we treat them the same way

        data = {}
        data_std = None
        if has_std_data:
            data_std = {}
        for side in ["left", "right"]:
            # Find the index of the header that correspond to channels to keep for each side
            header_indices = []
            for value in generic_header_to_keep:
                # If the value is in the header, use it already
                if value in header:
                    header_indices.append(header.index(value))
                    continue
                elif f"{value} {side.capitalize()}Cycle" in header:
                    # If the value is has a cycle specific value, separate by them
                    header_indices.append(header.index(f"{value} {side.capitalize()}Cycle"))
                    continue
                else:
                    raise ValueError("The CSV file does not contain all the expected channels.")

            data[side] = np.array(raw_data[1:, header_indices], dtype=float)[:, :, None]
            if has_std_data:
                data_std[side] = np.array(raw_data_std[1:, header_indices], dtype=float)[:, :, None]

        return cls(data=data, channel_names=_channels_map()["split"]["show_names"], data_std=data_std)

    @classmethod
    def from_normative_txt(cls, file_path: str, std_file_path: str | None, encryption_key: str | None = None) -> Self:
        """
        Create a KinematicData object from the txt data file.

        The data are stored in a "header\n/data1\n/data2\n..." format over rows. The data1 are the actual data. I do not
        know what data2 is for.

        Parameters
        ----------
        file_path : str
            The path to the TXT file containing the kinematic data.
        std_file_path : str | None
            The path to the TXT file containing the standard deviation of the kinematic data, if available.
        encryption_key : str, optional
            The encryption key to decrypt the data, if None is provided, the data is assumed to be unencrypted.

        Returns
        -------
        KinematicData
            The kinematic data extracted from the file
        """

        if std_file_path is not None:
            raise NotImplementedError("The standard deviation is not implemented yet for TXT files.")

        # The normative file is next to the current file
        if encryption_key is not None:
            decrypted_data = FileReader(encryption_key=encryption_key).read_encrypted_file(file_path)
            decrypted_text = decrypted_data.decode("utf-8")
            data_stream = io.StringIO(decrypted_text)
            all_lines = data_stream.readlines()
        else:
            with open(file_path) as f:
                all_lines = f.readlines()
        all_lines = [line.strip() for line in all_lines]  # Remove the nextline character

        trial_index = 1  # 1 for the next line after the header index. Should we use mean of all trials?
        data = {}
        for side in ["left", "right"]:
            # Parse the value for the header side
            header_to_keep = list(_channels_map()["split"][f"grail_names_{side}"])
            header_indices = [all_lines.index(header) for header in header_to_keep]

            data[side] = np.ndarray((101, len(header_to_keep), 1))  # 1 for the steps
            for cmp, index in enumerate(header_indices):
                data[side][:, cmp, 0] = np.array(all_lines[index + trial_index].split(" "), dtype=float)

        # Recast the data into a DataFrame
        return cls(data=data, channel_names=_channels_map()["split"]["show_names"])

    def from_mox(cls, file_path: str, encryption_key: str | None = None) -> Self:
        """
        Extract the data from the MOX file directly output from the GRAIL software.

        Parameters
        ----------
        file_path : str
            The path to the MOX file containing the kinematic data.
        encryption_key : str, optional
            The encryption key to decrypt the data, if None is provided, the data is assumed to be unencrypted.

        Returns
        -------
        KinematicData
            The kinematic data extracted from the file.
        """

        if encryption_key is not None:
            decrypted_data = FileReader(encryption_key=encryption_key).read_encrypted_file(file_path)
            decrypted_text = decrypted_data.decode("utf-8")
            data_stream = io.StringIO(decrypted_text)
            raw_data = xmltodict.parse(data_stream.read())
        else:
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
        data = pd.DataFrame(data, columns=header)[channels_to_keep].to_numpy()[:, :, None]

        try:
            return cls(data={"both": data}, channel_names=_channels_map()["all"]["show_names"])
        except KeyError:
            raise ValueError("The MOX file does not contain all the expected channels.")

    @classmethod
    def from_normative_data(cls, file: NormativeData) -> Self:
        """
        Create a KinematicData object generating normative data

        Parameters
        ----------
        file : NormativeData
            The normative data file to load=

        Returns
        -------
        KinematicData
            The kinematic data extracted from the file
        """
        key = os.environ["NORMATIVE_GRAIL_DATA_KEY"] if "NORMATIVE_GRAIL_DATA_KEY" in os.environ else None
        if key is None:
            raise ValueError(
                "The encryption key is not found. Please set the environment variable "
                "NORMATIVE_GRAIL_DATA_KEY with the encryption key. If you do not have one, "
                "please contact the owner of this repo to get one."
            )

        # The normative file is next to the current file
        root_folder = Path(__file__).parent
        file_path = root_folder / file.file_path
        std_file_path = None if file.std_file_path is None else (root_folder / file.std_file_path)
        return file.factory(file_path=file_path, std_file_path=std_file_path, encryption_key=key)

    def __getitem__(self, key: str) -> Self:
        if key not in self.channel_names:
            raise KeyError("The key is not in the data.")

        index = self.channel_names.index(key)
        return KinematicData(
            data={side: self._data[side][:, index : index + 1, :] for side in self.side_names},
            channel_names=(key,),
            data_std={side: self._data_std[side][:, index : index + 1, :] for side in self.side_names},
        )

    def __add__(self, other: Self) -> Self:
        if not isinstance(other, KinematicData):
            raise TypeError("The other object is not a KinematicData object.")

        for side in self.side_names:
            if side not in other.side_names:
                raise ValueError("The data does not contain the same sides.")
            if self._data[side].shape != other._data[side].shape:
                raise ValueError(f"The data of side {side} are not the same shapes.")

        return KinematicData(
            data={side: self._data[side] + other._data[side] for side in self.side_names},
            channel_names=self._channel_names,
        )

    def __sub__(self, other: Self) -> Self:
        if not isinstance(other, KinematicData):
            raise TypeError("The other object is not a KinematicData object.")

        for side in self.side_names:
            if side not in other.side_names:
                raise ValueError("The data does not contain the same sides.")
            if self._data[side].shape != other._data[side].shape:
                raise ValueError(f"The data of side {side} are not the same shapes.")

        return KinematicData(
            data={side: self._data[side] - other._data[side] for side in self.side_names},
            channel_names=self._channel_names,
        )

    def __pow__(self, value: float) -> Self:
        if not isinstance(value, (int, float)):
            raise TypeError("The value object is not a number.")

        return KinematicData(
            data={side: self._data[side] ** value for side in self.side_names},
            channel_names=self._channel_names,
        )
