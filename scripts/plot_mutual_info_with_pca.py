import argparse
import os
import glob

import numpy as np

import config


def get_name(filename):
    filename = os.path.splitext(filename)[0]
    electrode, unit = filename.split("_")[-2:]
    return "{}{}".format(electrode, unit)


def get_subplot_sizes(n):
    """Make width and height of subplot grid for n plots
    """
    width = int(np.sqrt(n))
    height = width if width ** 2 == n else width + 1
    if width * height < n:
        width += 1
    return width, height


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bird", type=str, default=None,
            help="Bird name (i.e. GreBlu9508M)")
    parser.add_argument("-s", "--site", type=int, default=None,
            help="Recording site number")
    parser.add_argument("-c", "--column", type=str, default=None,
            help="Category column (stim or call_type)")
    parser.add_argument("--savefig", type=bool, default=False,
            help="Save figure to file? (default just shows plot)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    import matplotlib
    if args.savefig: matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = os.path.join(config.OUTPUT_DIR, "plot_mutual_info_with_pca")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    mi_filename = os.path.join(output_dir, "{}_{}_{}_mi".format(args.bird, args.site, args.column))
    acc_filename = os.path.join(output_dir, "{}_{}_{}_acc".format(args.bird, args.site, args.column))

    read_dir = os.path.join(config.OUTPUT_DIR, "mutual_info_with_pca")
    filename_matcher = os.path.join(read_dir, "{}_{}_{}_e*_u*.npy".format(args.bird, args.site, args.column))

    files = glob.glob(filename_matcher)
    if not files:
        raise Exception("Files matching bird {}, site {}, column {}, were not found in {}".format(
            args.bird, args.site, args.column, read_dir))

    width, height = get_subplot_sizes(len(files))

    plt.figure(figsize=(2 * width, height))
    for i, filename in enumerate(sorted(files)):
        data = np.load(filename)[()]
        plt.subplot(width, height, i+1)
        plt.plot(data["dims"], data["normal"]["mi"])
        plt.plot(data["dims"], data["shuffled"]["mi"])
        plt.xlim(1, 40)
        plt.ylim(0, 6.5)
        plt.text(30, 0.3, get_name(filename), fontsize=6)
        plt.yticks([0, 3, 6], ["", 3, 6], fontsize=6)
        plt.xticks([10, 20, 30, 40], [])

    if args.savefig:
        plt.savefig(mi_filename, format="png", dpi=200)
    else:
        plt.show()

    plt.figure(figsize=(2 * width, height))
    for i, filename in enumerate(sorted(files)):
        data = np.load(filename)[()]
        plt.subplot(width, height, i+1)
        plt.plot(data["dims"], np.array(data["normal"]["acc"]) * 100)
        plt.plot(data["dims"], np.array(data["shuffled"]["acc"]) * 100)
        plt.hlines(
                100.0 / len(data["normal"]["categories"][0]),
                0,
                40,
                color="red",
                linestyle=":")
        plt.xlim(1, 40)
        plt.ylim(0, 60)
        plt.text(5, 45, get_name(filename), fontsize=6)
        plt.yticks(
                [0, 10, 20, 30, 40, 50, 60],
                ["0", "", "", "", "", "50%", ""],
                fontsize=6)
        plt.xticks([10, 20, 30, 40], [])

    if args.savefig:
        plt.savefig(acc_filename, format="png", dpi=200)
    else:
        plt.show() 
