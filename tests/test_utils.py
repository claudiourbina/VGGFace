import os
import unittest
import numpy as np
import tensorflow as tf

from vggface.utils import load_image

class UtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_load_image(self) -> None:
        im = load_image(path=os.path.join(os.path.dirname(__file__), 'assets\\test.jpg'), img_shape=(224, 224, 3))
        self.assertIsInstance(im, np.ndarray)
        self.assertEqual(im.shape, (1, 224, 224, 3))