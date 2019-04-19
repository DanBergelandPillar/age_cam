class ImageBatchGenerator():
    def __init__(self, image_directory):
        self.image_dir = image_directory

    def generate(self):
        x = 1
        y = 2
        yield x,y