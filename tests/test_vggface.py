import os
import logging
import unittest
import numpy as np
import tensorflow as tf

from vggface import VGGFace
from vggface.utils import load_image


class VGGFaceTest(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_vggface(self) -> None:
        vggface = VGGFace(include_top=True, input_shape=(224, 224, 3))
        self.assertIsInstance(vggface, VGGFace)

    def test_vggface_resnet50_predict(self) -> None:
        vggface = VGGFace(architecture="resnet50", include_top=True, input_shape=(224, 224, 3))
        im = load_image(path=os.path.join(os.path.dirname(__file__), 'assets\\test.jpg'), img_shape=(224, 224, 3))
        pred = vggface.predict(im, postprocessing=True)
        logging.info(pred)
        self.assertIn('A._J._Buckley', pred[0][0][0])
        self.assertAlmostEqual(pred[0][0][1], 0.91819614, places=3)
