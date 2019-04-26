from scipy import misc
import random
from os import listdir
from os.path import isfile, join

class ImageBatchGenerator():
    def __init__(self, image_directory):
        self.image_dir = image_directory

    def generate(self):
        path = '/Users/daniel.a.bergeland/Documents/ai/demographic_cam/test/test_images'
        files = [f for f in listdir(path) if isfile(join(path, f))]
        image = random.choice(files)
        
        with open(join(path, image), 'rb') as image:
            yield misc.imread(image), 1
