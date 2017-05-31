"""Script for computing mutual information as a function of dimensionality

Uses template decoding and confusion matrix to estimate mutual information

Run:
$ python scripts/mutual_info_with_pca.py --bird YelBlu6903F --site 1 --column stim

"""
import argparse
import os

import numpy as np
from sklearn.decomposition import PCA

import config
from data_loader import SessionDataLoader
from discriminant_analysis import lda, qda, rf, cross_validate
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
    parser.add_argument("-f", "--folds", type=int, default=10,
            help="Number of stratified folds (default 10)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Loading data for {} at Site {}".format(args.bird, args.site))
    table = load_session(args.bird, args.site)
    table = table[table["call_type"] != "None"]
    table = table.copy()

    output_dir = os.path.join(config.OUTPUT_DIR, "classification_lda_qda_rf")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # For each unit, save the dims array, mutual information array, and confusion matrix fig
    for unit, unit_table in table.groupby(by="unit", sort=True):
        filename_base = os.path.join(output_dir, "{}_{}_{}_e{}_u{}".format(args.bird, args.site, args.column, *unit))

        unit_table = unit_table.copy()
        X = np.array(unit_table["psth"].tolist())
        Y = np.array(unit_table[args.column])

        # Dictionary of data to be saved into .npy file
        unit_data = {}

        dims = filter(None, config.DIMS)  # exclude the full dimensionality

        unit_data["dims"] = dims
        unit_data["lda_scores"] = []
        unit_data["qda_scores"] = []
        unit_data["rf_scores"] = []
        unit_data["lda_std"] = []
        unit_data["qda_std"] = []
        unit_data["rf_std"] = []
 
        for i, dim in enumerate(dims):
            print("Analyzing Unit {}, {} dims".format(unit, dim or "Full"))
            pca = PCA(n_components=dim)
            X_reduced = pca.fit_transform(X)

            scores = cross_validate(X_reduced, Y, lda, args.folds)
            unit_data["lda_scores"].append(np.mean(scores) * 100.0)
            unit_data["lda_std"].append(np.std(scores) * 100.0)

            scores = cross_validate(X_reduced, Y, qda, args.folds)
            unit_data["qda_scores"].append(np.mean(scores) * 100.0)
            unit_data["qda_std"].append(np.std(scores) * 100.0)

            scores = cross_validate(X_reduced, Y, lda, args.folds)
            unit_data["rf_scores"].append(np.mean(scores) * 100.0)
            unit_data["rf_std"].append(np.std(scores) * 100.0)

        unit_data["n_classes"] = np.unique(Y).size
        unit_data["chance_level"] = 100.0 / unit_data["n_classes"]

        np.save(filename_base, unit_data)
