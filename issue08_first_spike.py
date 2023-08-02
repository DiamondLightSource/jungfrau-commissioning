import sys

import numpy
import h5py
import hdf5plugin

with h5py.File(sys.argv[1]) as f:
    data = f["data"]
    nz, ny, nx = data.shape

    for j in range(nz):
        frame = data[j, :, :]
        frame[frame < 0] = 0
        print(f"{j} {numpy.sum(frame)}")

