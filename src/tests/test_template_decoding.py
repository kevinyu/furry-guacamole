import unittest

import pandas as pd

from confusion import generate_confusion_matrix
from template_decoding import (
        compute_distances_to_templates,
        decode,
        prepare,
        template_selectors
)


class TestTemplate(unittest.TestCase):

    def setUp(self):
        self.table = pd.DataFrame({
            "call_type": pd.Series(["Ag", "Ag", "DC", "DC", "None", "None"]),
            "stim": pd.Series(["1", "1", "2", "2", "3", "3"]),
            "trial": pd.Series(["1", "2", "1", "2", "1", "2"]),
            "psth": pd.Series([[2], [3], [0], [1], [5], [5]])
        })
        self.table = prepare(self.table, "stim")

    def test_template_selectors(self):
        selectors, categories = template_selectors(self.table, "stim")
        self.assertEqual(selectors.shape, (6, 3, 6))
        self.assertListEqual(list(categories), ["1", "2", "3"])
        print selectors

    def test_compute_distances(self):
        selectors, categories = template_selectors(self.table, "stim")
        distances = compute_distances_to_templates(self.table, selectors, "psth")
        print distances

    def test_decode(self):
        selectors, categories = template_selectors(self.table, "stim")
        distances = compute_distances_to_templates(self.table, selectors, "psth")
        result = decode(self.table, distances, categories)

    def test_confusion(self):
        selectors, categories = template_selectors(self.table, "stim")
        distances = compute_distances_to_templates(self.table, selectors, "psth")
        predicted = decode(self.table, distances, categories)
        actual = self.table.index

        print generate_confusion_matrix(actual, predicted, categories)

