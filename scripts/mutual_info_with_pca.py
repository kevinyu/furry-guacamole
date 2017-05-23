"""Script for computing mutual information as a function of dimensionality

Uses template decoding and confusion matrix to estimate mutual information

Run:
$ python scripts/mutual_info_with_pca.py --bird YelBlu6903F --site 1 --column stim

"""
import argparse
import os

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
    compute_distance_to_templates,
    compute_templates,
    decode,
    unbias_templates
)


def decode_after_pca(unit_table, template_column="stim", ndim=2):
    if ndim:
        pca = PCA(n_components=ndim)
        # save the reduced representation of responses to "resp" column
        unit_table["resp"] = pca.fit_transform(unit_table["psth"].tolist()).tolist()
    else:
        unit_table["resp"] = unit_table["psth"].tolist()

    templates = compute_templates(unit_table, template_column, "resp")

    if "resp_template" in unit_table: unit_table.drop("resp_template", 1, inplace=True)
    unit_table = unbias_templates(unit_table, templates, template_column, "resp")

    compute_distance_to_templates.clear_cache()
    distances = compute_distance_to_templates(unit_table, template_column, "resp")

    unit_table = unit_table.sort_values(["call_type", "stim"])

    actual = unit_table[template_column]
    predicted = decode(unit_table, distances, template_column)

    return generate_confusion_matrix(actual, predicted, joint=False)


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

    output_dir = os.path.join(config.OUTPUT_DIR, "mutual_info_with_pca")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # For each unit, save the dims array, mutual information array, and confusion matrix fig
    for unit, unit_table in table.groupby(by="unit", sort=True):
        filename_base = os.path.join(output_dir, "{}_{}_{}_e{}_u{}".format(args.bird, args.site, args.column, *unit))

        unit_table = unit_table.copy()
        unit_data = {}
        unit_data["dims"] = config.DIMS
        unit_data["mi"] = []
        unit_data["mi_ctrl"] = []
        for dim in config.DIMS:
            print("Analyzing Unit {}, {} dims".format(unit, dim))
            conf = decode_after_pca(unit_table, template_column=args.column, ndim=dim)
            unit_data["mi"].append(confusion.mutual_information(conf))
            unit_table["shuffled_label"] = unit_table[args.column].sample(frac=1).tolist()
            conf = decode_after_pca(unit_table, template_column="shuffled_label", ndim=dim)
            unit_data["mi_ctrl"].append(confusion.mutual_information(conf))

        np.save(filename_base, unit_data)

        plt.figure(figsize=(9, 8))
        plt.imshow(conf, vmin=0.0, vmax=1.0, cmap="hot")
        plt.colorbar()
        plt.title("Dim {}".format(dim))
        plt.savefig(filename_base, format="png", dpi=200)
