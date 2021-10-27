#!usr/bin/env python3

""""
Discretize initial states to make use of the belleman equation
"""
import numpy as np

low_values = np.array([-1.2, -0.7])
high_values = np.array([0.6, 0.7])

n_states = 40
n_actions = 3  
distance = (high_values - low_values) 
step = distance / n_states


def discretize(state):

    """
    This functions  assigns a discrete number  to a state 
    Help : https://github.com/L42Project/Tutoriels/
    blob/master/Divers/renforcement2/MountainCar_common.py
    """

    state = (state - low_values) / step
    return tuple(state.astype(int))

def pick_action(epsilon, Q, state):
    if np.random.uniform() < epsilon :
          #explore 
        action = np.random.randint(n_actions)
    else: #pick the max action
        action = np.argmax(Q[state])
    return action    

print(discretize([0.0, 0.7]))    





