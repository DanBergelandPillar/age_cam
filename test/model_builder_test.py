from src.model_builder import InceptionAge
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

if '__name__' == '__main__':
    unittest.main()