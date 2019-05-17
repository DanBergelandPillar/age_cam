from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adadelta

def InceptionAge():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(200,200,3))
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(1, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(loss= mean_squared_error,
            optimizer= Adadelta())

    return model