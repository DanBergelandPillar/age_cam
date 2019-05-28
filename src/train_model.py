import image_batch_generator as ibg
import model_builder
from scipy import misc
import random
import numpy as np

def predict_on_samples(generator):
    print('predicting on random examples...')

    next_batch = next(generator)

    print('Predicting ... ')
    predictions = model.predict(next_batch[0])
    labels = next_batch[1][0]
    print(labels)
    for i in range(len(predictions)):
        print('...   ')
        print('Predicted: ', predictions[i]*100)
        print('Actual: ', labels[i]*100)

## Get data generator
BS = 32
utkface_gen = ibg.ImageBatchGenerator('../UTKFace', random.choice, BS)

## Get our model
model = model_builder.Resnet50Age()

#Show some untrained predictions:
predict_on_samples(utkface_gen.generate())

## Get hyper parameters - Epochs and Steps per epoch
EPOCHS = 3

H = model.fit_generator(utkface_gen.generate(), steps_per_epoch=100,
	epochs=EPOCHS)

predict_on_samples(utkface_gen.generate())

    