import unittest
import numpy as np
from src.image_batch_generator import ImageBatchGenerator

class TestDataGenerator(unittest.TestCase):
    def setUp(self):
        self.target_path = 'test/test_images'
        self.data_gen = ImageBatchGenerator(self.target_path)

    def test_getsImageDirectory(self):
        self.assertEqual(self.data_gen.image_dir, self.target_path)

    def test_callingYieldReturnsPictureDataAndAgeLabel(self):
        input, label = next(self.data_gen.generate())
        self.assertEqual(np.shape(input), (200, 200, 3))
        self.assertEqual(label, 1)



if '__name__' == '__main__':
    unittest.main()