from src.model_builder import Resnet50Age
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error
import unittest

class TestModelBuilder(unittest.TestCase):
    def setUp(self):
        self.resnet_model = Resnet50Age()

    def test_inputLayerShouldHaveDimensionNonex200x200x3(self):
        input_layer_shape = self.resnet_model.layers[0].input_shape
        self.assertEqual(input_layer_shape, (None,200,200,3))

    def test_outputLayerShouldHaveDimensionNonex1(self):
        output_layer_shape = self.resnet_model.layers[-1].output_shape
        self.assertEqual(output_layer_shape, (None, 1))

    def test_baseLayersAreNotTrainable(self):
        trainable_layers = 3
        head_cutoff = -1*trainable_layers
        for layer in self.resnet_model.layers[:head_cutoff]:
            self.assertFalse(layer.trainable)
        for layer in self.resnet_model.layers[head_cutoff:]:
            self.assertTrue(layer.trainable)

    def test_modelHasOptimizer(self):
        self.assertEqual(type(self.resnet_model.optimizer), type(Adam()))

    def test_modelUsesMeanSquareError(self):
        self.assertEqual(self.resnet_model.loss, mean_squared_error)

if '__name__' == '__main__':
    unittest.main()