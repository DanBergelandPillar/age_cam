from scipy import misc
import random
from os import listdir
from os.path import isfile, join
import tensorflow as tf
import pandas as pd

class ImageBatchGenerator():
    def __init__(self, image_directory, random_choice_generator):
        self.image_dir = image_directory
        self.random_choice_generator = random_choice_generator

    def generate(self):
        path = self.image_dir
        files = [filename for filename in listdir(path) if isfile(join(path, filename))]
    
        while True:
            image_name = self.random_choice_generator(files)
            with open(join(path, image_name), 'rb') as image:
                yield misc.imread(image), int(image_name.split('_')[0])

    def createDataFrame(self):
        path = self.image_dir
        files = [filename for filename in listdir(path) if isfile(join(path, filename))]
        labels = [int(image_name.split('_')[0]) for image_name in files]
        dataFrame = pd.DataFrame({
            'filepaths': files,
            'labels': labels
        })
        return dataFrame