# Report

### Model

The agent in this project learns with a Deep DQN learning algorithm. As an agent explores the environment - through simulation - the DQN algorithm learns the value of choosing specific actions in specific states through observation of final cumulative reward after an entire episode of exploration.

What makes the algorithm "deep" is that a neural network is used to model the value of state-action pairs. In this project, a fully-connected neural network is used with 3 linear layers and 64 nodes in the hidden layers. The input of this network is the current state of the environment, and the output is a vector that is the same size as the action space, which represents the value of taking each action at the input state.

The hyperparameters were set to:
* Replay buffer size = 100000
* Batch size = 64
* Discount factor (Gamma) = 0.99
* Target network soft update parameter (Tau) = 0.001
* Learning rate = 0.00005
* How often to update the network (in timepoints) = 4
* Epsilon start value = 1.0
* Epsilon end value = 0.01
* Epsilon decay rate = 0.995

### Performance

The goal of this model is to gain an average final reward of at least +13 over 100 episodes. The graph below show's the reward of the basic DQN agent at each timepoint and also averaged across 100 episodes.

![basic_agent_training](https://github.com/mbluestone/bananas_DQN/blob/master/img/basic_training.png)

The model seems to be learning fairly quickly: it only takes 399 episodes to learn this environment.

### Ideas for Future Work

An obvious first next step would be to perform hyperparameter optimization to determine the best hyperparameters for training the DQN agent. This could potentially lead to an agent that learns faster.

Another potential next step would be to explore how changes to the neural network model effect learning. Things that could be altered include the number of linear layers in the network and the number of nodes in the hidden layers.
