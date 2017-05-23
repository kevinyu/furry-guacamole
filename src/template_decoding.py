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

    temp_frame["selfless_template"] = (
            np.array(temp_frame[new_col_name].tolist()) -
            np.array(temp_frame[column].tolist()) * (1.0 / temp_frame["n"][:, None])
    ).tolist()

    return temp_frame


_rush_savings = {}
def compute_distance_to_templates(dataframe, template_column, rush=True, dist=euclidean_distances):
    """
    Dataframe must come with column "selfless_template"
    """
    if "selfless_template" not in dataframe:
        raise Exception("Column 'selfless_template' not in dataframe; "
                "Use the fns compute_templates and unbias_templates first")

    grouped = dataframe.groupby(by=template_column, sort=True)
    distances = []
    for idx, row in dataframe.iterrows():
        # FIXME: this step is fairly slow; can maybe do this better?
        # If we are "rushing", don't resample every time
        if rush and row["stim"] in _rush_savings:
            sampled = _rush_savings[row["stim"]]
        else:
            sampled = grouped.apply(lambda x: x.sample(n=1)).set_index(template_column)["selfless_template"]
            sampled.set_value(row["stim"], row["selfless_template"])
            _rush_savings[row["stim"]] = sampled
        distances.append(dist(np.array(row["psth"])[None, :], sampled.tolist())[0])

    # FIXME: need to make this a structured array so that the columns are labeled by the template label
    return np.array(distances)
