import numpy as np


def generate_confusion_matrix(actual, predicted, joint=True):
    """Compute a confusion matrix from actual values and predicted values
    
    Params:
    actual (np.ndarray, dtype=int)
        Elements [0, N] representing true values
    predicted (np.ndarray, dtype=int)
        Elements [0, N] representing predicted values corresponding to actual
        If element is an iterable, distribute the density across all
    joint (bool, default=True)
        Return joint probabilities if True, conditional if False

    Returns:
    confusion_matrix (np.ndarray, NxN)
        Rows represent correct response, columns represent predicted response

    Example:
        
    >>> a = np.array([0, 0, 1, 1, 2, 2])
    >>> b = np.array([0, [1, 2], 1, 1, 0, 2])
    >>> generate_confusion_matrix(a, b)
    [[0.5 0.25 0.25]
     [0.0 1.0 0.0]
     [0.5 0.0 0.5]]
    """
    labels = actual.unique()
    lookup = dict((val, i) for i, val in enumerate(labels))
    n = len(labels)

    confusion_matrix = np.zeros((n, n))
    for a, p in zip(actual, predicted):
        a_i = lookup[a]
        p_i = lookup[p]
        # FIXME: must handle the case of ties
        try:
            tied = len(p)
        except:
            tied = 1
        confusion_matrix[a_i][p_i] += 1.0 / (np.count_nonzero(actual == a) * tied)

    if joint:
        return confusion_matrix / np.sum(confusion_matrix)
    return confusion_matrix


def mutual_information(confusion_matrix):
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix)
    p_col = np.sum(confusion_matrix, axis=0)[None, :]
    p_row = np.sum(confusion_matrix, axis=1)[:, None]
    pp = np.dot(p_row, p_col)
    nonzero = confusion_matrix != 0.0
    return np.sum(confusion_matrix[nonzero] * np.log2((confusion_matrix[nonzero] / pp[nonzero])))


def accuracy(confusion_matrix):
    return np.sum(np.diag(confusion_matrix / np.sum(confusion_matrix)))
