import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances


def compute_templates(dataframe, template_column, column):
    """Compute template values for a column (mean) grouping by another column

    Args
    dataframe (pandas.DataFrame): dataset
    template_column (str): name of column to group by
    column (str): name of column to compute templates on

    Returns
    templates (pandas.DataFrame)
    """
    grouped = dataframe.groupby(by=template_column, sort=True)
    keys = grouped.groups.keys()

    means = pd.Series([
        np.mean(grouped.get_group(g)[column].tolist(), axis=0)
        for g in keys
    ], index=keys)
    n_trials = pd.Series([
        len(grouped.get_group(g)[column]) for g in keys
    ], index=keys)

    return pd.DataFrame({
        column: means,
        "n": n_trials
    })


def unbias_templates(dataframe, templates, template_column, column):
    """For each row in the dataframe, subtract itself off the computed template

    Returns a series which is the template value minus the row's value
    (weighted by 1/n_trials)
    """
    new_col_name = "{}_template".format(column)
    if new_col_name in dataframe:
        raise Exception("{0} already exists in dataframe; "
                "drop it before calling this fn using "
                "df.drop('{0}', 1, inplace=True)".format(new_col_name))

    temp_frame = dataframe.join(templates, on=template_column, rsuffix="_template")

    # FIXME: this only subtracts 1 off of each... we want to even out n
    temp_frame["selfless_template"] = (
            np.array(temp_frame[new_col_name].tolist()) -
            np.array(temp_frame[column].tolist()) * (1.0 / temp_frame["n"][:, None])
    ).tolist()

    return temp_frame


def compute_distance_to_templates(dataframe, template_column, rush=True, dist=euclidean_distances):
    """
    Dataframe must come with column "selfless_template"
    """
    if "selfless_template" not in dataframe:
        raise Exception("Column 'selfless_template' not in dataframe; "
                "Use the fns compute_templates and unbias_templates first")

    grouped = dataframe.groupby(by=template_column, sort=True)
    distances = []
    keys = list([(g[0].encode("utf-8"), np.float64) for g in grouped])
    for idx, row in dataframe.iterrows():
        # FIXME: this step is fairly slow; can maybe do this better?
        # If we are "rushing", don't resample every time
        if rush and row[template_column] in compute_distance_to_templates._rush_cache:
            sampled = compute_distance_to_templates._rush_cache[row[template_column]]
        else:
            sampled = grouped.apply(lambda x: x.sample(n=1)).set_index(template_column)["selfless_template"]
            sampled.set_value(row[template_column], row["selfless_template"])
            compute_distance_to_templates._rush_cache[row[template_column]] = sampled
        distances.append(tuple(dist(np.array(row["psth"])[None, :], sampled.tolist())[0]))

    return np.array(distances, dtype=keys)

compute_distance_to_templates._rush_cache = {}
compute_distance_to_templates.clear_cache = lambda: compute_distance_to_templates._rush_cache.clear()


def decode(dataframe, distances, column):
    """Compute index of nearest template for each datapoint

    Parameters
    ----------
    dataframe : pd.DataFrame (n_datapoints, ...)
        Dataframe with a Series `column` corresponding to the
        category label over which to decode
    distances : np.ndarray (n_datapoints, n_categories)
        Distance of each datapoint to templates for each category
    column : string
        Name of column with category label

    Returns
    -------
    predicted : np.ndarray (n_datapoints)
        Decoded values (or indices if `labels` not specified)
    """
    nearest_templates = np.argmin(np.array(distances.tolist()), axis=1)
    results = []
    for idx in nearest_templates:
        results.append(distances.dtype.names[idx])

    return np.array(results)
