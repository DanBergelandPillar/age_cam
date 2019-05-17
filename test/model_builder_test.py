from src.model_builder import InceptionAge
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.losses import mean_squared_error
import unittest

class TestModelBuilder(unittest.TestCase):
    def setUp(self):
        self.inception_model = InceptionAge()

    def test_inputLayerShouldHaveDimensionNonex200x200x3(self):
        input_layer_shape = self.inception_model.layers[0].input_shape
        self.assertEqual(input_layer_shape, (None,200,200,3))

    def test_outputLayerShouldHaveDimensionNonex1(self):
        output_layer_shape = self.inception_model.layers[-1].output_shape
        self.assertEqual(output_layer_shape, (None, 1))

    def test_baseLayersAreNotTrainable(self):
        trainable_layers = 3
        head_cutoff = -1*trainable_layers
        for layer in self.inception_model.layers[:head_cutoff]:
            self.assertFalse(layer.trainable)
        for layer in self.inception_model.layers[head_cutoff:]:
            self.assertTrue(layer.trainable)

    def test_modelHasOptimizer(self):
        self.assertEqual(type(self.inception_model.optimizer), type(Adadelta()))

    def test_modelUsesMeanSquareError(self):
        self.assertEqual(self.inception_model.loss, mean_squared_error)


if '__name__' == '__main__':
    unittest.main()