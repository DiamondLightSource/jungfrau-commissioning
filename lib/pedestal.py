import os

import numpy
import h5py


def pedestal_raw_data(directory):
    """Read pedestal data from the files in the named directory"""

    result = {}

    gain_keys = {"G0": 0, "G1": 1, "G2": 3}

    for module in 0, 1:
        for gain in "G0", "G1", "G2":
            filename = os.path.join(directory, f"{gain}_{module}_0.h5")
            assert os.path.exists(filename)
            with h5py.File(filename, "r") as f:
                result[(gain, module)] = f["data"][()]
    return result
