#libraries 
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPool2D, Dropout, AveragePooling2D
from tensorflow.keras.models import Model

nb_actions = 4
window_lenght = 4 #number of images we will consider in 
target_input = (84, 84, 1)

def model(nb_actions):
    input = Input(shape=target_input)
    x = input
    for i in range(5):
        x = Conv2D(2**(i + 5), activation="relu")(x)
        x = MaxPool2D()(x)
    x = AveragePooling2D()(x)
    output = Dense(nb_actions, activation='softmax')
    model = Model(inputs=input, outputs=output) 
    return model





