from enum import Enum
from typing import Callable


class NormativeData(Enum):
    """Enum for normative data files."""

    NORMAL = "normal"
    CROUCHGAIT = "crouch_gait"

    @property
    def file_path(self) -> str:
        if self == NormativeData.NORMAL:
            return "./normative_normal.csv.encrypted"
        elif self == NormativeData.CROUCHGAIT:
            return "./normative_crouchgait.txt.encrypted"
        else:
            raise ValueError(f"Invalid normative data file: {self}")

    @property
    def std_file_path(self) -> str | None:
        if self == NormativeData.NORMAL:
            return "./normative_normal_std.csv.encrypted"
        elif self == NormativeData.CROUCHGAIT:
            return None
        else:
            raise ValueError(f"Invalid normative data file: {self}")

    @property
    def factory(self) -> Callable:
        from .kinematic import KinematicData

        if self == NormativeData.NORMAL:
            return KinematicData.from_normative_csv
        elif self == NormativeData.CROUCHGAIT:
            return KinematicData.from_normative_txt
        else:
            raise ValueError(f"Invalid normative data file: {self}")
