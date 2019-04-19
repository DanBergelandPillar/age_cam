import unittest
from src.image_batch_generator import ImageBatchGenerator

class TestDataGenerator(unittest.TestCase):
    def setUp(self):
        self.target_path = 'test_images'
        self.data_gen = ImageBatchGenerator(self.target_path)

    def test_getsImageDirectory(self):
        self.assertEqual(self.data_gen.image_dir, self.target_path)

    def test_callingYieldReturnsXandY(self):
        input, label = next(self.data_gen.generate())
        self.assertTrue(input is not None)
        self.assertTrue(label is not None)



if '__name__' == '__main__':
    unittest.main()