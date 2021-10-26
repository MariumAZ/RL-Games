#!usr/bin/env python3

import gym
import numpy as np

episodes = 20000
gamma = 0.9
alpha = 0.1 #learning rate
max_steps = 1000
epsilon = 0.98

#initialize Q table:


env = gym.make("MountainCar-v0")
print(env.action_space)
print(env.observation_space)


def pick_action(epsilon, Q, state):
    if np.random.uniform() < epsilon :
        action = np.random.randint(env.actions_space.n)
    else: #pick the max action
        action = np.argmax(Q[state])
    return action    

q_shape = tuple([env.observation, env.action_space.n])
Q = np.zeros(shape = q_shape)
for _ in range(episodes):
    state = env.reset()
    step = 0 
    action = pick_action(epsilon, Q, state)
    while step < max_steps:
        env.render()
        new_state, reward, done, info = env.step(action)
        q_current = Q[state][action]
        q_target = np.max(Q[new_state])
        q_current = (1 - alpha) * q_current + alpha * (reward + gamma * q_target)
        state = new_state
        if done:
            break
        step += 1











