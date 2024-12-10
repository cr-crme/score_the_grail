from enum import Enum
from typing import Callable


class NormativeData(Enum):
    """Enum for normative data files."""

    NORMAL = "normal"
    CROUCHGAIT = "crouch_gait"

    @property
    def file_path(self) -> str:
        if self == NormativeData.NORMAL:
            return "./normative_normal.csv"
        elif self == NormativeData.CROUCHGAIT:
            return "./normative_crouchgait.txt"
        else:
            raise ValueError(f"Invalid normative data file: {self}")

    @property
    def factory(self) -> Callable:
        from .kinematic import KinematicData

        if self == NormativeData.NORMAL:
            return KinematicData.from_normative_csv
        elif self == NormativeData.CROUCHGAIT:
            return KinematicData.from_txt
        else:
            raise ValueError(f"Invalid normative data file: {self}")
