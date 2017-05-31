"""
Compare accuracy from decoding using LDA, QDA, RF, and templates
"""

import argparse
import os
import glob

import numpy as np

import config


def get_electrode_and_unit(filename):
    filename = os.path.splitext(filename)[0]
    electrode, unit = filename.split("_")[-2:]
    return electrode, unit


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

    output_dir = os.path.join(config.OUTPUT_DIR, "plot_accuracy_comparisons")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = os.path.join(output_dir, "{}_{}_{}_acc".format(args.bird, args.site, args.column))

    template_read_dir = os.path.join(config.OUTPUT_DIR, "mutual_info_with_pca")
    filename_matcher = os.path.join(template_read_dir, "{}_{}_{}_e*_u*.npy".format(args.bird, args.site, args.column))

    classification_read_dir = os.path.join(config.OUTPUT_DIR, "classification_lda_qda_rf")

    files = glob.glob(filename_matcher)
    if not files:
        raise Exception("Files matching bird {}, site {}, column {}, were not found in {}".format(
            args.bird, args.site, args.column, read_dir))

    width, height = get_subplot_sizes(len(files))

    plt.figure(figsize=(2 * width, 1.2 * height))
    for i, filename in enumerate(sorted(files)):
        electrode, unit = get_electrode_and_unit(filename)

        classification_file = os.path.join(classification_read_dir,
                "{}_{}_{}_{}_{}.npy".format(
                    args.bird, args.site, args.column, electrode, unit
                ))
        if os.path.exists(classification_file):
            discriminant_data = np.load(classification_file)[()]
        else:
            discriminant_data = None

        data = np.load(filename)[()]
        plt.subplot(width, height, i+1)
        if discriminant_data is not None:
            plt.errorbar(discriminant_data["dims"],
                    discriminant_data["lda_scores"],
                    yerr=discriminant_data["lda_std"],
                    label="LDA")
            plt.errorbar(discriminant_data["dims"],
                    discriminant_data["qda_scores"],
                    yerr=discriminant_data["qda_std"],
                    label="QDA")
            plt.errorbar(discriminant_data["dims"],
                    discriminant_data["rf_scores"],
                    yerr=discriminant_data["rf_std"],
                    label="RF")
        plt.plot(data["dims"], np.array(data["normal"]["acc"]) * 100, label="Template")
        plt.hlines(
                100.0 / len(data["normal"]["categories"][0]),
                0,
                40,
                color="red",
                linestyle=":")
        plt.xlim(1, 40)
        plt.ylim(0, 100)
        plt.text(2, 96, "{}{}".format(electrode, unit), 
                fontsize=10, alpha=0.3, verticalalignment="top")
        plt.yticks(
                np.arange(0, 110, 10),
                ["0", "", "", "", "", "50%", "", "", "", "", "100%"],
                fontsize=6)
        plt.xticks([10, 20, 30, 40], [10, 20, 30, 40], fontsize=6)
    plt.tight_layout()

    if args.savefig:
        plt.savefig(output_filename, format="png", dpi=200)
    else:
        plt.show() 
