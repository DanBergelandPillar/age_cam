from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D
from tensorflow.keras import Model, Sequential
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam

def Resnet50Age():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(200,200,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1)(x)

    for layer in base_model.layers:
        layer.trainable = False

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(loss= mean_squared_error,
            optimizer= Adam())

    return model
