import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPool2D, Dropout, AveragePooling2D
from tensorflow.keras.models import Model

buffer_size = 100 #will take sequence of tuples (s_t, a_t, r_t, s_t+1)
number_episodes = 10000
nb_actions = 4
window_lenght = 4 #number of images we will consider in 
image_shape = (84, 84)
target_input = image_shape + (window_lenght,)

#TODO: try time distributed layer

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


def clone_model(model):
    """
    The target network will need the same configuration as the 
    policy network 
    """
    














