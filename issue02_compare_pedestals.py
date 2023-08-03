import sys

from matplotlib import pyplot

from lib import pedestals, pedestals_mean_iqr, pedestal_raw_data

d0 = sys.argv[1]

raw_data = pedestal_raw_data(d0)
pmi = pedestals_mean_iqr(raw_data)
pref = pedestals(raw_data)

for module in 0, 1:
    for gain in "G0", "G1", "G2":
        diff = pmi[(gain, module)][0] - pref[(gain, module)][0]
        fig, axes = pyplot.subplots(2)
        axes[0].imshow(pmi[gain, module][0])
        axes[1].imshow(pref[gain, module][0])
        pyplot.show()
