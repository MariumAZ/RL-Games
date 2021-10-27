#!usr/bin/env python3

""""
Discretize initial states to make use of the belleman equation
"""
import numpy as np

low_values = np.array([-1.2, -0.7])
high_values = np.array([0.6, 0.7])

n_states = 40
distance = (high_values - low_values) 
step = distance / n_states


def discretize(state):
    """
    This functions  assigns a discrete number  to a state 
    """

    state = (state - low_values) / step
    return tuple(state.astype(int))




