import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances


def prepare(dataframe, template_column):
    dataframe.set_index(template_column, inplace=True)
    dataframe[template_column] = dataframe.index
    return dataframe.sort_values(by=["call_type", "stim"], ascending=True)


def template_selectors(dataframe, template_column):
    selectors = []
    categories = dataframe.index.unique()

    in_category = dict(
        (category, np.array(dataframe.index) == category)
        for category in categories
    )

    for row in dataframe.itertuples():
        idx = row[0]
        row_selectors = []
        not_the_same_trial = ~(
            (np.array(dataframe["stim"], dtype="<U5") == getattr(row, "stim")) *
            (np.array(dataframe["trial"], dtype="<U5") == getattr(row, "trial"))
        )
        for category in categories:
            row_selectors.append(in_category[category] * not_the_same_trial)
        selectors.append(row_selectors)

    return np.array(selectors).astype(np.bool), categories


def compute_templates(dataframe, selectors, column, sample_n=None):
    # TODO here is where you could cache the selector result if youre in a hurry
    _cache = {}
    def _select(dataframe, selector):
        key = tuple(selector)
        if key in _cache:
            return _cache[key]
        _cache[key] = dataframe[selector]
        return _cache[key]

    return np.array([
        np.array(
            _select(dataframe, selector).sample(n=sample_n)[column].tolist()
        ).mean(axis=0)
        for selector in selectors
    ])


def compute_distances_to_templates(
        dataframe,
        template_selectors,
        column,
        dist=euclidean_distances):

    sample_n = np.min([np.sum(selector)
        for selectors in template_selectors
        for selector in selectors])

    distance_arr = []
    for row, selectors in zip(dataframe.itertuples(), template_selectors):
        templates = compute_templates(dataframe, selectors, column, sample_n)
        distances = dist(np.array(getattr(row, column))[None, :], templates)
        distance_arr.append(distances[0])

    return np.array(distance_arr)


def decode(dataframe, distances, categories):
    """Compute index of nearest template for each datapoint

    Parameters
    ----------
    dataframe : pd.DataFrame (n_datapoints, ...)
        Dataframe with a Series `column` corresponding to the
        category label over which to decode
    distances : np.ndarray (n_datapoints, n_categories)
        Distance of each datapoint to templates for each category

    Returns
    -------
    predicted : np.ndarray (n_datapoints)
        Decoded indices
    """
    nearest_templates = []
    for dist_arr in distances:
        winners = np.where(dist_arr == np.min(dist_arr))[0]
        nearest_templates.append([categories[i] for i in winners])
    return nearest_templates


def get_plotty_lines(table):
    """Return parameters for plotting the separation between
    call types in the confusion matrix
    """
    group_sizes = (table
        .groupby(by=["call_type", "stim"])
        .size()
        .groupby(by="call_type")
        .count())
    labels = group_sizes.index
    barriers = group_sizes.cumsum()
    label_positions = group_sizes / 2.0 + barriers.shift(1).fillna(0)
    return barriers, labels, label_positions
