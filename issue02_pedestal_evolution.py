import sys

import numpy
from matplotlib import pyplot

from lib import pedestal_filter, pedestals_mean_iqr

d0 = sys.argv[1]

p0 = pedestals_mean_iqr(d0)
r0 = pedestal_filter(d0)

for module in 0, 1:
    for gain in "G0", "G1", "G2":
        ratio = r0[(gain, module)] / p0[(gain, module)][0]
        map = numpy.zeros((ratio.shape[0], 200), dtype=numpy.int32)
        for j in range(ratio.shape[0]):
            h, e = numpy.histogram(ratio[j, :, :], range=(-10,10), bins=200)
            map[j] = h
        pyplot.imshow(map.transpose())
        pyplot.colorbar()
        pyplot.show()
