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
    actual : list (n_datapoints)
        Each element is a string label for the datapoint
    predicted : list (n_datapoints)
        Each element is a list of the decoded values
        It is a list because there can be ties
    categories : list (n_categories)
        An ordered list of the categories being decoded
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

    return actual, predicted, categories


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
        unit_data = {
            "unit": unit,
            "dims": [],
            "normal": {
                "labels": [],
                "predicted": [],
                "categories": [],
                "mi": [],
                "acc": [],
            },
            "shuffled": {
                "labels": [],
                "predicted": [],
                "categories": [],
                "mi": [],
                "acc": [],
            },
            "plotting": {
                "category_barriers": [],
                "category_labels": [],
                "label_positions": [],
            }
        }

        unit_data["dims"] = config.DIMS[:]
        if config.DIMS[-1] is None:
            unit_data["dims"][-1] = int((config.MAX_TIME - config.MIN_TIME) * 1e3)

        # dim == 0 signifies to not do the initial dimensionality reduction
        for dim in config.DIMS:
            print("Analyzing Unit {}, {} dims".format(unit, dim or "Full"))
            actual, predicted, categories = decode_after_pca(unit_table, template_column=args.column, ndim=dim)
            conf = generate_confusion_matrix(actual, predicted, categories)
            unit_data["normal"]["labels"].append(actual)
            unit_data["normal"]["predicted"].append(predicted)
            unit_data["normal"]["categories"].append(categories)
            unit_data["normal"]["mi"].append(confusion.mutual_information(conf))
            unit_data["normal"]["acc"].append(confusion.accuracy(conf))

        # Do a second trial with shuffled labels to get an upper bound on the information bias
        for dim in config.DIMS:
            unit_table["psth"] = unit_table["psth"].sample(frac=1).tolist()
            actual, predicted, categories = decode_after_pca(unit_table, template_column=args.column, ndim=dim)
            conf_ctrl = generate_confusion_matrix(actual, predicted, categories)
            unit_data["shuffled"]["labels"].append(actual)
            unit_data["shuffled"]["predicted"].append(predicted)
            unit_data["shuffled"]["categories"].append(categories)
            unit_data["shuffled"]["mi"].append(confusion.mutual_information(conf_ctrl))
            unit_data["shuffled"]["acc"].append(confusion.accuracy(conf_ctrl))

        barriers, labels, label_positions = get_plotty_lines(unit_table)

        unit_data["plotting"]["category_barriers"] = barriers
        unit_data["plotting"]["category_labels"] = labels
        unit_data["plotting"]["label_positions"] = label_positions

        np.save(filename_base, unit_data)
