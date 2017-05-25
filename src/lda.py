"""Linear discriminant analysis
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import cross_validation


def discriminate(X, Y, prior=None, colors=None, folds=10, ndim=32):
    """
    
    If dont specify colors, will not plot
    """
    classes, class_count = np.unique(Y, return_counts=True)

    n_classes = classes.size
    if n_classes < 2:
        raise Exception("Too few classes")

    # if prior is not specified, use a uniform prior
    if prior is None:
        prior = np.ones(n_classes) / float(n_classes)

    lda = LDA(n_components=n_classes - 1, priors=prior, shrinkage=True, solver="eigen")
    qda = QDA(priors=prior)
    rf = RF()

    pca = PCA(n_components=ndim)
    X = pca.fit_transform(X)

    lda_scores = np.zeros(folds)
    qda_scores = np.zeros(folds)
    rf_scores = np.zeros(folds)

    skf = cross_validation.StratifiedKFold(Y, folds)

    for fold_idx, (train, test) in enumerate(skf):
        lda.fit(X[train], Y[train])
        qda.fit(X[train], Y[train])
        rf.fit(X[train], Y[train])

        lda_scores[fold_idx] = lda.score(X[test], Y[test])
        qda_scores[fold_idx] = qda.score(X[test], Y[test])
        rf_scores[fold_idx] = rf.score(X[test], Y[test])

    results = {
        "n_classes": n_classes,
        "chance_level": 1.0 / n_classes,
        "lda_acc": np.mean(lda_scores),
        "qda_acc": np.mean(qda_scores),
        "rf_acc": np.mean(rf_scores),
        "lda_acc_std": np.std(lda_scores),
        "qda_acc_std": np.std(qda_scores),
        "rf_acc_std": np.std(rf_scores),
    }

    return results
