"""Script for computing mutual information as a function of dimensionality

Uses template decoding and confusion matrix to estimate mutual information

Run:
$ python scripts/mutual_info_with_pca.py --bird YelBlu6903F --site 1 --column stim

"""
import argparse
import os

import numpy as np

import config
from data_loader import SessionDataLoader
from lda import discriminate
from spikes.filters import gaussian_filter, exp_filter
from spikes.binning import bin_spikes


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

    X = np.array(table["psth"].tolist())
    Y = np.array(table[args.column])

    output_dir = os.path.join(config.OUTPUT_DIR, "classification_lda_qda_rf")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # For each unit, save the dims array, mutual information array, and confusion matrix fig
    for unit, unit_table in table.groupby(by="unit", sort=True):
        filename_base = os.path.join(output_dir, "{}_{}_{}_e{}_u{}".format(args.bird, args.site, args.column, *unit))

        unit_table = unit_table.copy()

        # Dictionary of data to be saved into .npy file
        unit_data = {}
        unit_data["dims"] = config.DIMS
        if config.DIMS[-1] is None:
            unit_data["dims"][-1] = int((config.MAX_TIME - config.MIN_TIME) * 1e3)
        unit_data["mi"] = []
        unit_data["mi_ctrl"] = []
        unit_data["acc"] = []
        unit_data["acc_ctrl"] = []

        dims = filter(None, config.DIMS)  # exclude the full dimensionality

        unit_data["lda_scores"] = []
        unit_data["qda_scores"] = []
        unit_data["rf_scores"] = []
        unit_data["lda_std"] = []
        unit_data["qda_std"] = []
        unit_data["rf_std"] = []
 
        for i, dim in enumerate(dims):
            print("Analyzing Unit {}, {} dims".format(unit, dim or "Full"))
            result = discriminate(X, Y, folds=10, ndim=dim)

            unit_data["lda_scores"].append(result["lda_acc"] * 100.0)
            unit_data["qda_scores"].append(result["qda_acc"] * 100.0)
            unit_data["rf_scores"].append(result["rf_acc"] * 100.0)
            unit_data["lda_std"].append(result["lda_acc_std"] * 100.0)
            unit_data["qda_std"].append(result["qda_acc_std"] * 100.0)
            unit_data["rf_std"].append(result["rf_acc_std"] * 100.0)

        unit_data["n_classes"] = result["n_classes"]
        unit_data["chance_level"] = result["chancel_level"] * 100.0

        np.save(filename_base, unit_data)