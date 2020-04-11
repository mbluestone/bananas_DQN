[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Deep DQN for an RL Agent in the Unity Bananas Environment

### Introduction

This was a project for the [Deep Reinforcement Learning Nanodegree Program](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) offered by Udacity. For this project, I trained an agent to navigate and collect bananas in a large, square world offered by [Unity](https://www.unity.com).  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Clone this GitHub repository, place the app file in the repository folder, and unzip (or decompress) the file.

### Requirements
* NumPy
* PyTorch
* unityagents

### Training a DQN Agent

To train your own agent, you must first initialize the Bananas environment:

```
from unityagents import UnityEnvironment
env = UnityEnvironment(file_name="./Banana.app")
```

Get the sizes of the environment state and action spaces:

```
# get the default brain of the environment
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of actions
action_size = brain.vector_action_space_size

# size of the state space
state_size = len(env_info.vector_observations[0])
```

Then, you must initialize an agent with the `Agent` class:

```
from agent import Agent
training_agent = Agent(state_size=state_size,
                       action_size=action_size)
```

To train the agent, just call:
```
_, _ = agent.train(env)
```
A `checkpoint.pth` file will automatically be saved with the model weights in the `checkpoints/` folder.

The `Agent` class takes parameters to use more complicated models:
* Double DQN: `double_dqn=True`
* Dueling DQN: `dueling=True`
* Prioritized Experience Replay: `per=True`
  * Also `per_args` which is a tuple consisting of `a`, `beta`, and `beta_increment`

The names of saved model checkpoint files will be updated based on the type of DQN used (e.g. Double DQN -> `checkpoint_double.pth`)

My model training is completed in `model_training.ipynb`.

### Loading a Pre-Trained DQN Agent

To train your own agent, you must first initialize the Bananas environment and Agent instance like above. Make sure that the parameters of the new Agent instance match those that were used for training. For example, if you're loading a model from `checkpoint_dueling.pth` then make sure that you set `dueling=True` in the Agent instantiation.

Once the environment and agent are set up, you can load the model weights into the `agent` like so:

```
checkpoint_path = 'checkpoints/checkpoint.pth'
agent.qnetwork_local.load_state_dict(torch.load(checkpoint_path))
```
