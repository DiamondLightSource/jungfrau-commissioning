import sys

import numpy
from matplotlib import pyplot

from lib import pedestals

d0 = sys.argv[1]

pref = pedestals(d0)

for module in 0, 1:
    for gain in "G0", "G1", "G2":
        h, e = numpy.histogram(pref[(gain, module)][1], bins=5000, range=(0, 5000))
        with open(f"{module}.{gain}.dat", "w") as f:
            for j in range(len(h)):
                f.write(f"{e[j]} {e[j+1]} {h[j]}\n")
        fig, axes = pyplot.subplots(2)
        axes[0].imshow(pref[gain, module][0])
        axes[1].imshow(pref[gain, module][1])
        pyplot.show()
