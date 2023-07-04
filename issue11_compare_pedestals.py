import sys

from matplotlib import pyplot

from lib import pedestals

d0, d1 = sys.argv[1:3]

p0 = pedestals(d0)
p1 = pedestals(d1)

for module in 0, 1:
    for gain in "G0", "G1", "G2":
        diff = p0[(gain, module)][0] - p1[(gain, module)][0]
        pyplot.imshow(diff)
        pyplot.colorbar()
        pyplot.show()
