#libraries 
import cv2
import numpy as np


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
    examples = n.random.randint()







