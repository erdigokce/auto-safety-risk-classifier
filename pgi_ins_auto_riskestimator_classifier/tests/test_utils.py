from unittest import TestCase

from model import utils


class TestUtils(TestCase):

    def test_print(self):
        s = utils.print([0, 1])
        self.assertTrue(s != "")
