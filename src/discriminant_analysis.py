from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import cross_validation


def lda(n_classes, prior):
    return LDA(
        n_components=n_classes - 1,
        priors=prior,
        shrinkage=True,
        solver="eigen"
    )


def qda(n_classes, prior):
    return QDA(priors=prior)


def rf(n_classes, prior):
    return RF()


def cross_validate(X, Y, discriminator, folds=10):
    skf = cross_validation.StratifiedKFold(Y, folds)

    scores = []
    for train, test in skf:
        X_train, Y_train = X[train], Y[train]

        classes, n_per_class = np.unique(Y_train, return_counts=True)
        n_classes = len(classes)
        prior = n_per_class / len(X_train)

        disc = discriminator(n_classes, prior)
        disc.fit(X_train, Y_train)
        scores.append(disc.score(X[test], Y[test]))

    return scores
