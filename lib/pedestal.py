from __future__ import annotations

import dataclasses
import glob
import logging
import os
import time
from itertools import product
from pathlib import Path
from typing import TypeAlias

import h5py
import numpy
import numpy.typing
from tqdm import tqdm

logger = logging.getLogger(__name__)

gain_keys = {"G0": 0, "G1": 1, "G2": 3}
PedestalDict: TypeAlias = dict[tuple[int, str], numpy.typing.NDArray]


@dataclasses.dataclass
class Pedestal:
    data: numpy.typing.NDArray
    mask: numpy.typing.NDArray
    variance: numpy.typing.NDArray

    filename: Path
    _raw_data: numpy.typing.NDArray | None = None

    def load_raw_data(self) -> numpy.typing.NDArray:
        """Open the source data file and read the entire array."""
        if self._raw_data is not None:
            return self._raw_data

        with h5py.File(self.filename, "r") as f:
            logger.debug(
                f"Reading {f['data'].nbytes/1000/1000:.0f} MB raw data from {self.filename}"
            )
            start = time.monotonic()
            self._raw_data = f["data"][()]
            logger.debug(f"   ... done in {time.monotonic()-start:.1f} s")

        return self._raw_data


class PedestalFile:
    _data: dict[tuple[int, str], Pedestal]
    path: Path

    modules = ["M420", "M418"]
    gains = ["G0", "G1", "G2"]

    def __init__(self, path: str | os.PathLike):
        """Read a processed pedestal file, and get all the data therein"""
        self.path = Path(path)
        self._data = {}

        logging.debug(f"Reading pedestal file {path}")
        with h5py.File(path, "r") as f:
            for gain in [0, 1, 2]:
                for mid, module in [(0, "M420"), (1, "M418")]:
                    raw_data = f[f"{module}/pedestal_{gain}"][...]
                    logger.debug(
                        f"Read {module} {gain} in {raw_data.nbytes/1000:.0f} KB"
                    )
                    self._data[mid, f"G{gain}"] = Pedestal(
                        data=raw_data,
                        filename=Path(f[f"{module}/pedestal_{gain}"].attrs["filename"]),
                        mask=f[f"{module}/pedestal_{gain}_mask"][...],
                        variance=f[f"{module}/pedestal_{gain}_variance"][...],
                    )

    def __getitem__(self, key: tuple[int | str, str]) -> Pedestal:
        if isinstance(key[0], str):
            return self._data[self.modules.index(key[0]), key[1]]
        return self._data[key]


def pedestal_raw_data(
    directory: str | os.PathLike,
) -> PedestalDict:
    """Read pedestal data from the files in the named directory"""
    result = {}
    total_usage = 0

    for module, gain in tqdm(product((0, 1), ("G0", "G1", "G2")), total=6, leave=False):
        tqdm.write(f"Reading raw data {module} {gain}")
        filename = os.path.join(directory, f"{gain}_{module}_0.h5")
        if not os.path.exists(filename):
            matches = glob.glob(os.path.join(directory, f"*{gain}_{module}_0.h5"))
            assert len(matches) == 1
            filename = matches[0]
        assert os.path.exists(filename)
        with h5py.File(filename, "r") as f:
            result[gain, module] = f["data"][()]
            tqdm.write(f"    Read {result[gain, module].nbytes/1000/1000:.0f} MB")
            total_usage += result[gain, module].nbytes

    print(f"Read raw data total {total_usage/1000/1000/1000:.1f} GB")
    return result


def pedestals(
    raw_data: PedestalDict | str | os.PathLike,
) -> dict[tuple[int, str], tuple[numpy.typing.NDArray, numpy.typing.NDArray]]:
    """Read pedestal data; filter; return mean and variance"""

    if not isinstance(raw_data, dict):
        raw_data = pedestal_raw_data(raw_data)

    result = {}
    for module in 0, 1:
        for gain in "G0", "G1", "G2":
            key = gain_keys[gain]
            data = raw_data[(gain, module)]
            # mask, sum image, sum square image
            m = numpy.zeros(data.shape[1:], dtype=numpy.int32)
            i = numpy.zeros(data.shape[1:], dtype=numpy.float64)
            s = numpy.zeros(data.shape[1:], dtype=numpy.float64)
            for j in tqdm.tqdm(range(data.shape[0])):
                b = numpy.right_shift(data[j], 14) == key
                x = (numpy.bitwise_and(data[j], 0x3FFF) * b).astype(numpy.float64)
                m += b
                i += x
                s += numpy.square(x)
            m[m == 0] = 1
            i = i / m
            s = s / m
            v = s - numpy.square(i)
            result[gain, module] = (i, v)

    return result


def pedestals_mean_iqr(raw_data: PedestalDict | str | os.PathLike):
    """Read pedestal data; sort, select iqr and average"""
    if not isinstance(raw_data, dict):
        raw_data = pedestal_raw_data(raw_data)

    result = {}
    for module in 0, 1:
        for gain in "G0", "G1", "G2":
            print(f"IQR for {module} {gain}")
            key = gain_keys[gain]
            data = raw_data[gain, module][()]
            b = numpy.right_shift(data, 14) == key
            m = numpy.bitwise_and(data, 0x3FFF) * b
            s = numpy.sort(m, axis=0)
            q = data.shape[0] // 4
            result[gain, module] = (numpy.mean(s[q:-q, :, :], axis=0),)

    return result
