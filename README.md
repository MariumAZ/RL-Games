# Mountain Car

A car is on a one-dimensional track, positioned between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.

The car’s state, at any point in time, is given by a vector containing its :

- horizonal position 
- velocity.


The car can take 3 actions : 
- left(0) 
- no push(1) 
- right(2)

The state space represents a 2-dimensional box.
To view the values that our states can take : 


```python
> print(env.observation_space.low)
[-1.2  -0.07]
>print(env.observation_space.high)
[0.6  0.07]

```
The states space in infinite so in order to make use of the Bellman equation we need to discretize the space.
For this we use a function discretize in utils :

```python
def discretize(state):

    """
    This functions  assigns a discrete number  to a state 
    Help : https://github.com/L42Project/Tutoriels/
    blob/master/Divers/renforcement2/MountainCar_common.py
    """

    state = (state - low_values) / step
    return tuple(state.astype(int))
```    
Epsilon greedy was updated  using two methods : 

```python
epsilon = (min_epsilon + (init_epsilon - min_epsilon) * np.exp(-epsilon_decay * episode))
epsilon *= 1 - 3 *(episode / episodes)
```

# Bibliography:

[Getting Started with Reinforcement Learning and Open AI Gym](https://towardsdatascience.com/getting-started-with-reinforcement-learning-and-open-ai-gym-c289aca874f)

[Apprentissage par renforcement 2: équation de Bellman](https://www.youtube.com/watch?v=4Ak6OyehqJc&t=251s)
