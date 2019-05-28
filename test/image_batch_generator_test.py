import unittest
import numpy as np
from src.image_batch_generator import ImageBatchGenerator

class TestDataGenerator(unittest.TestCase):
    def setUp(self):
        self.target_path = 'test/test_images'
        self.random_choice = 0
        choicer = lambda filelist: filelist[self.random_choice]
        self.batch_size = 1
        self.data_gen = ImageBatchGenerator(self.target_path, choicer, self.batch_size)

    def test_getsImageDirectory(self):
        self.assertEqual(self.data_gen.image_dir, self.target_path)

    def test_callingYieldReturnsPictureDataAndAgeLabel(self):
        input, label = next(self.data_gen.generate())
        self.assertEqual(np.shape(input), (1, self.batch_size, 200, 200, 3))
        self.assertEqual(label, [[.01]])

    def test_callingYieldAndGettingSecondImageHasLabelOf31(self):
        self.random_choice = 2
        input, label = next(self.data_gen.generate())
        self.assertEqual(label, [[.31]])

    def test_createDataFrame_returnsFilenamesAndLabels(self):
        dataFrame = self.data_gen.createDataFrame()
        for index, row in dataFrame.iterrows():
            self.assertEqual(row['labels'], int(row['filepaths'].split('_')[0]))

if '__name__' == '__main__':
    unittest.main()