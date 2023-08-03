#!/usr/bin/env python3

from __future__ import annotations

import logging
import os
from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lib import PedestalFile

logger = logging.getLogger(__name__)


def run():
    parser = ArgumentParser()
    parser.add_argument("pedestal", type=Path, help="Source processed pedestal data")
    parser.add_argument(
        "-v", action="store_true", help="Verbose output", dest="verbose"
    )
    args = parser.parse_args()
    logging.basicConfig(
        format="%(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    logging.getLogger("matplotlib").setLevel(logging.INFO)

    # Load this data
    pedestals = PedestalFile(args.pedestal)

    def _bins_autorange(data, nbins: int = 100):
        """
        Automatically calculate a bin range from data.

        Currently uses np.partition to find the 99%-of-data range
        """
        data_r = np.ravel(data)
        ratio_index_high = int(data_r.shape[0] * 0.99)
        ratio_index_low = int(data_r.shape[0] * (1 - 0.99))
        parted = np.partition(data_r, [ratio_index_low, ratio_index_high])
        data_min = parted[ratio_index_low]
        data_max = parted[ratio_index_high]

        # var = np.sqrt(np.var(data))
        # mean = np.mean(data)

        # data_min = np.min(data)
        # data_max = np.max(data)
        # data_noz_min = np.min(data[data > 0])
        # print(data_min, data_noz_min, data_max, upper_95)
        edges = np.linspace(data_min, data_max, nbins + 1)
        return edges

    # embed()
    # np.var

    entries = list(product(reversed(pedestals.modules), pedestals.gains))

    for i, (module, gain) in enumerate(entries):
        plt.subplot(2, 3, i + 1)
        ped = pedestals[module, gain]
        w, b = np.histogram(ped.data, bins=_bins_autorange(ped.data))
        plt.hist(b[:-1], b, weights=w)
        plt.title(f"{module} {gain}")
    plt.suptitle("Calculated Pedestal Means")
    # plt.savefig("pedestal.png")
    plt.subplots_adjust()
    plt.show()

    plt.clf()
    for i, (module, gain) in enumerate(entries):
        plt.subplot(2, 3, i + 1)
        ped = pedestals[module, gain]
        # breakpoint()
        w, b = np.histogram(ped.variance, bins=_bins_autorange(ped.variance))

        plt.hist(b[:-1], b, weights=w)
        plt.title(f"{module} {gain}")
    plt.suptitle("Calculated Pedestal Variance")
    # plt.savefig("pedestal.png")
    plt.subplots_adjust()
    plt.savefig("variance_histogram.png")
    plt.show()
    plt.clf()
    # for i, (module, gain) in enumerate(entries):
    col, lin = os.get_terminal_size()
    plt.figure(figsize=(20, 9))
    # , 8 * col / (lin * 1.5)))

    # plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(pedestals["M420", "G0"].variance, vmax=5000)
    plt.title("M420 (Bottom) Variances")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(pedestals["M418", "G0"].variance, vmax=5000)
    plt.title("M418 (Top) Variances")
    plt.colorbar()
    plt.savefig("variance_image.png")
    plt.show()

    plt.clf()
    for module, gain in entries:
        pedestal = pedestals[module, gain]
        raw = pedestal.load_raw_data()
        # Work out the 95% ranges
        count = np.prod(raw.shape)
        pct_range = 0.99
        idxs = [int(count * (1 - pct_range)), int(count * pct_range)]
        parted = np.partition(raw, kth=idxs, axis=None)
        lo, hi = parted[idxs[0]], parted[idxs[1]]
        print(f"Module {module} {gain}")
        print(f"    {pct_range*100:.0f}%: {lo} → {hi}")
        # breakpoint()
        w, b = np.histogram(
            raw[pedestal.mask is False], bins=np.linspace(lo, hi, 201)
        )  # bins=_bins_threesig(ped.data))
        plt.hist(b[:-1], b, weights=w)
        plt.title(f"{module} {gain}")
        plt.show()
        break


if __name__ == "__main__":
    run()
