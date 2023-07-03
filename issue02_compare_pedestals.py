import sys

import numpy
from matplotlib import pyplot

from lib import pedestals, pedestals_mean_iqr

d0 = sys.argv[1]

pmi = pedestals_mean_iqr(d0)
pref = pedestals(d0)

for module in 0, 1:
    for gain in "G0", "G1", "G2":
        diff = pmi[(gain, module)][0] - pref[(gain, module)][0]
        pyplot.imshow(diff, vmin=-100, vmax=100)
        pyplot.colorbar()
        pyplot.show()
