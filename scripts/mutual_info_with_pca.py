"""Script for computing mutual information as a function of dimensionality

Uses template decoding and confusion matrix to estimate mutual information

Run:
$ python scripts/mutual_info_with_pca.py --bird YelBlu6903F --site 1 --column stim

"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import config
import confusion
from confusion import generate_confusion_matrix
from data_loader import SessionDataLoader
from spikes.filters import gaussian_filter, exp_filter
from spikes.binning import bin_spikes
from template_decoding import (
    compute_distances_to_templates,
    decode,
    get_plotty_lines,
    prepare,
    template_selectors,
)


def decode_after_pca(unit_table, template_column="stim", ndim=2):
    """Generate confusion matrix after running pca on data

    Applies PCA on the psth column of the dataset and generates
    a confusion matrix using nearest template decoding.

    Modifies the input dataframe by adding a new column called "resp"
    that represents the "psth" in a lower dimensional space (PCA)

    Parameters
    ----------
    unit_table : pd.DataFrame (n_datapoints, ... )
        Dataframe with a Series `template_column` indicating the
        category being decoded, and a Series "psth" for the response
        data
    template_column : string (default="stim")
        Name of column containing category labels
    ndim : int (default=2)
        Number of principal components to keep on PCA step

    Returns
    -------
    confusion_matrix : np.ndarray (n_stims, n_stims)
        Confusion matrix normalized so that each row sums to 1
        (i.e. represents P(predicted stim | actual stim))
    """
    if ndim:
        pca = PCA(n_components=ndim)
        # save the reduced representation of responses to "resp" column
        unit_table["resp"] = pca.fit_transform(unit_table["psth"].tolist()).tolist()
    else:
        unit_table["resp"] = unit_table["psth"].tolist()

    selectors, categories = template_selectors(unit_table, "resp")
    distances = compute_distances_to_templates(unit_table, selectors, "resp")

    predicted = decode(unit_table, distances, categories)
    actual = unit_table.index

    return generate_confusion_matrix(actual, predicted, categories, joint=False)


def load_session(bird_name, session_num):
    data_loader = SessionDataLoader(bird_name, session_num)
    table = data_loader.load_table()

    t_arr, spikes = bin_spikes(table["spike_times"],
            min_time=config.MIN_TIME,
            max_time=config.MAX_TIME)
    filter_fn = exp_filter if config.FILTER_TYPE is "exp" else gaussian_filter
    spikes_filtered = filter_fn(spikes, config.FILTER_WIDTH)

    table["binned_spikes"] = spikes.tolist()
    table["psth"] = spikes_filtered.tolist()

    return table


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bird", type=str, default=None,
            help="Bird name (i.e. GreBlu9508M)")
    parser.add_argument("-s", "--site", type=int, default=None,
            help="Recording site number")
    parser.add_argument("-c", "--column", type=str, default=None,
            help="Category column (stim or call_type)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Loading data for {} at Site {}".format(args.bird, args.site))
    table = load_session(args.bird, args.site)
    table = table[table["call_type"] != "None"]
    table = table.copy()

    output_dir = os.path.join(config.OUTPUT_DIR, "mutual_info_with_pca")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # For each unit, save the dims array, mutual information array, and confusion matrix fig
    for unit, unit_table in table.groupby(by="unit", sort=True):
        filename_base = os.path.join(output_dir, "{}_{}_{}_e{}_u{}".format(args.bird, args.site, args.column, *unit))

        unit_table = unit_table.copy()
        unit_table = prepare(unit_table, "stim")

        # Dictionary of data to be saved into .npy file
        unit_data = {}
        unit_data["dims"] = config.DIMS
        if config.DIMS[-1] is None:
            unit_data["dims"][-1] = int((config.MAX_TIME - config.MIN_TIME) * 1e3)
        unit_data["mi"] = []
        unit_data["mi_ctrl"] = []
        unit_data["acc"] = []
        unit_data["acc_ctrl"] = []

        for dim in config.DIMS:
            # dim == 0 signifies to not do the initial dimensionality reduction
            print("Analyzing Unit {}, {} dims".format(unit, dim or "Full"))
            conf = decode_after_pca(unit_table, template_column=args.column, ndim=dim)
            unit_data["mi"].append(confusion.mutual_information(conf))
            unit_data["acc"].append(confusion.accuracy(conf))

            # Do a second trial with shuffled labels to get an upper bound on the information bias
            unit_table["psth"] = unit_table["psth"].sample(frac=1).tolist()
            conf_ctrl = decode_after_pca(unit_table, template_column="shuffled_label", ndim=dim)
            unit_data["mi_ctrl"].append(confusion.mutual_information(conf_ctrl))
            unit_data["acc_ctrl"].append(confusion.accuracy(conf))

        np.save(filename_base, unit_data)

        barriers, labels, label_posititions = get_plotty_lines(unit_table)
        plt.figure(figsize=(11, 10))
        plt.pcolormesh(conf, vmin=0.0, vmax=1.0, cmap="hot")
        plt.hlines(barriers, 0, barriers[-1], color="white", linestyles=":", alpha=0.5)
        plt.vlines(barriers, 0, barriers[-1], color="white", linestyles=":", alpha=0.5)
        plt.xticks(label_posititions, labels)
        plt.yticks(label_posititions, labels)
        plt.colorbar()
        plt.title("Dim {}".format(dim))

        plt.savefig(filename_base, format="png", dpi=200)
