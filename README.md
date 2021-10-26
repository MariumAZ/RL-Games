# Mountain Car

A car is on a one-dimensional track, positioned between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.

The car’s state, at any point in time, is given by a vector containing its :

- horizonal position 
- velocity.

The state space represents a 2-dimensional box.
To view the values that our states can take : 


```
> print(env.observation_space.low)
[-1.2  -0.07]
>print(env.observation_space.high)
[0.6  0.07]

```

# Bibliography:

[Getting Started with Reinforcement Learning and Open AI Gym](https://towardsdatascience.com/getting-started-with-reinforcement-learning-and-open-ai-gym-c289aca874f)

[Apprentissage par renforcement 2: équation de Bellman](https://www.youtube.com/watch?v=4Ak6OyehqJc&t=251s)
