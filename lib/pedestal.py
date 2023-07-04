import os
import glob

import numpy
import h5py
from tqdm import tqdm
from itertools import product

gain_keys = {"G0": 0, "G1": 1, "G2": 3}


def pedestal_raw_data(directory):
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


def pedestals(directory):
    """Read pedestal data; filter and average"""

    result = {}

    raw_data = pedestal_raw_data(directory)
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


def pedestals_mean_iqr(directory):
    """Read pedestal data; sort, select iqr and average"""

    result = {}

    raw_data = pedestal_raw_data(directory)
    for module in 0, 1:
        for gain in "G0", "G1", "G2":
            print(f"IQR for {module} {gain}")
            key = gain_keys[gain]
            data = raw_data[(gain, module)][()]
            b = numpy.right_shift(data, 14) == key
            m = numpy.bitwise_and(data, 0x3FFF) * b
            s = numpy.sort(m, axis=0)
            q = data.shape[0] // 4
            result[(gain, module)] = (numpy.mean(s[q:-q, :, :], axis=0),)

    return result
