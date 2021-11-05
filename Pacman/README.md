# Pacman 


1.Initialize replay memory capacity.
2.Initialize the policy network with random weights.
3.Clone the policy network, and call it the target network.
4.For each episode:
  - Initialize the starting state.
  - For each time step:
    - Select an action : Via exploration or exploitation
    - Execute selected action in an emulator.
    - Observe reward and next state.
    - Store experience in replay memory.
    - Sample random batch from replay memory.
    - Preprocess states from batch.
    - Pass batch of preprocessed states to policy network.
    - Calculate loss between output Q-values and target Q-values.
      Requires a pass to the target network for the next state
    - Gradient descent updates weights in the policy network to minimize loss.
    - After x time steps, weights in the target network are updated to the Initialize replay memory capacity.
