from unittest import TestCase

from model import summarize


class TestSummarize(TestCase):

    def test_mean(self):
        numbers = [1, 2, 3, 4, 5]
        mean = summarize.mean(numbers)
        self.assertTrue(3 == mean)
