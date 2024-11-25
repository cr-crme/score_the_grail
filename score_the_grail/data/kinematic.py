from typing import Self

import xmltodict
import pandas as pd
import numpy as np


class KinematicData:
    channel_names = (
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

    def __init__(self, data: pd.DataFrame):
        self.data = data

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
        channels_to_keep = [f"Rotation {name}" for name in KinematicData.channel_names]
        try:
            return cls(data=pd.read_csv(file_path)[channels_to_keep])
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
            The kinematic data extracted from the file.
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
        header_to_keep = [f"Rotation {name}" for name in KinematicData.channel_names]
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

        channels_to_keep = [f"Rotation {name}" for name in KinematicData.channel_names]
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
