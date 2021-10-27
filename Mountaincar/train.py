#!usr/bin/env python3

import gym
import numpy as np
import utility

episodes = 250000
#discount rate : how much value we give to the future 
gamma = 0.9
#learning rate
alpha = 0.1 
max_steps = 100
epsilon = 0.98
init_epsilon = epsilon
min_epsilon = 0.1
epsilon_decay = 0.05

#initialize Q table:


env = gym.make("MountainCar-v0")
print(env.action_space)
print(env.observation_space)
print(env.goal_position)

total_rewards = []
rewards_per_episode = []
q_shape = [utility.n_states, utility.n_states,  utility.n_actions]
#Q = np.zeros(shape = tuple(q_shape))
Q = np.random.uniform(low=-1, high=1, size= tuple(q_shape))
for episode in range(episodes):
    state = env.reset()
    state = utility.discretize(state)
    done = False
    while not done:
        #pick an action
        action = utility.pick_action(epsilon, Q, state)
        env.render()
        new_state, reward, done, info = env.step(action)
        new_state = utility.discretize(new_state)
        q_current = Q[state][action]
        #print("q_current", q_current)
        q_target = np.max(Q[new_state])
        q_current = (1 - alpha) * q_current + alpha * (reward + gamma * q_target)
        state = new_state
        print(state)
        if new_state[0] >= env.goal_position :
            reward = 1
        rewards_per_episode.append(reward)
        if done:
            break    
    total_rewards.append(rewards_per_episode)    
    #reduce epsilon decay
    epsilon = (min_epsilon + (init_epsilon - min_epsilon) * np.exp(-epsilon_decay * episode))
    #print(Q)
np.save("Mountaincar_qtable", Q)
env.close()
        
















