#libraries 
import cv2
import numpy as np


epsilon = 0.95

def preprocess(image):
    #transform an RGB image into gray scale image
    if image.shape[2] == 3:
        #convert image to grey scale
        grey_image = image.mean(axis=2)
        #add dimension to image to match keras layers expectations
        grey_image = grey_image[:,:,np.newaxis]
    return grey_image


def sample(n):
    #sampling experiences from buffer :
    #we can pass in a number or a pourcentage 
    examples = np.random.randint(n)
    return examples

"""
def clone_model(model, custom_objects={}):
    # Requires Keras 1.0.7 since get_config has breaking changes.
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    return clone
"""


def epsilon_greedy(Q, state, n_actions):
    if np.random.uniform() < epsilon :
          #explore 
        action = np.random.randint(n_actions)
    else: #pick the max action
        action = np.argmax(Q[state])
    return action 




