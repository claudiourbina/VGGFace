import os
import logging
import unittest
import numpy as np
import tensorflow as tf

from vggface.utils import load_image
from vggface.resnet50 import ResNet50

class ResNet50Test(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_resnet50(self) -> None:
        resnet50 = ResNet50(include_top=True, input_shape=(224, 224, 3))
        self.assertIsInstance(resnet50, ResNet50)

    def test_load_image(self) -> None:
        resnet50 = ResNet50(include_top=True, input_shape=(224, 224, 3))
        im = load_image(path=os.path.join(os.path.dirname(__file__), 'assets\\test.jpg'), img_shape=(224, 224, 3))
        self.assertIsInstance(im, np.ndarray)
        self.assertEqual(im.shape, (1, 224, 224, 3))

    def test_resnet50_predict(self) -> None:
        resnet50 = ResNet50(include_top=True, input_shape=(224, 224, 3))
        im = load_image(path=os.path.join(os.path.dirname(__file__), 'assets\\test.jpg'), img_shape=(224, 224, 3))
        pred = resnet50.predict(im, postprocessing=True)
        logging.info(pred)
        self.assertIn('A._J._Buckley', pred[0][0][0])
        self.assertAlmostEqual(pred[0][0][1], 0.91819614, places=3)
