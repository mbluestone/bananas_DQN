'''
DQN Agent Class 
with Double DQN, Dueling DQN, and Prioritized Experience Replay optionality
'''

import numpy as np
import random
from collections import deque

from model import QNetwork
from utils import ReplayBuffer, PrioritizedReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed=0, double_dqn=False, dueling=False, per=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            double_dqn (bool): whether to implement Double DQN (default=False)
            dueling (bool): whether to implement Dueling DQN
            per (bool): whether to implement Prioritized Experience Replay 
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.double_dqn = double_dqn
        self.per = per
        self.gamma = GAMMA
        
        # output name for checkpoint
        self.output_name = ''
        self.output_name += '_double' if double_dqn else ''
        self.output_name += '_dueling' if dueling else ''
        self.output_name += '_per' if per else ''

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, 
                                       action_size, 
                                       seed, 
                                       dueling=dueling).to(device)
        self.qnetwork_target = QNetwork(state_size, 
                                        action_size, 
                                        seed, 
                                        dueling=dueling).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        if self.per:
            self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed) 
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def train(self,env,n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        """Deep Q-Learning.

        Params
        ======
            env (UnityEnvironment): Bananas environment
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        # get the default brain
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        # list containing scores from each episode
        scores = []   
        # last 100 scores
        scores_window = deque(maxlen=100) 
        # initialize epsilon
        eps = eps_start                    
        for i_episode in range(1, n_episodes+1):
            env_info = env.reset(train_mode=True)[brain_name]
            state = env_info.vector_observations[0]
            score = 0
            for t in range(max_t):
                action = self.act(state, eps)
                env_info = env.step(action)[brain_name]
                # get the next state
                next_state = env_info.vector_observations[0]
                # get the reward
                reward = env_info.rewards[0] 
                # see if episode has finished
                done = env_info.local_done[0]
                self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break 
            # save most recent score
            scores_window.append(score)       
            scores.append(score)
            # decrease epsilon
            eps = max(eps_end, eps_decay*eps) 
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window)>=13.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                torch.save(self.qnetwork_local.state_dict(), f'./checkpoints/checkpoint{self.output_name}.pth')
                break
        return scores
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
        # if using PER, get error
        if self.per:
            self.qnetwork_local.eval()
            self.qnetwork_target.eval()
            with torch.no_grad():
                # if Double DQN
                if self.double_dqn:
                    # Get predicted Q values (for next actions chosen by local model) from target model
                    next_action = self.qnetwork_local(next_state).detach().max(1)[1].unsqueeze(1)
                    Q_target_next = self.qnetwork_target(next_state).gather(1,next_action)

                else:
                    # Get max predicted Q values (for next states) from target model
                    Q_target_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)

                # Compute Q targets for current states 
                Q_target = reward + (self.gamma * Q_target_next * (1 - done))

                # Get expected Q values from local model
                Q_expected = self.qnetwork_local(state).cpu().data.numpy().squeeze()[action]
                
            self.qnetwork_local.train()
            self.qnetwork_target.train()
            
            error = abs(Q_expected - Q_target)
            state = state.cpu().data.numpy()
            next_state = next_state.cpu().data.numpy()
            self.memory.add(error, (state, action, reward, next_state, done))
            
        else:
            self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                self.learn()

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self):
        """Update value parameters using given batch of experience tuples.
        """
        # if using PER
        if self.per:
            states, actions, rewards, next_states, dones, idxs, is_weights = self.memory.sample()
            
        # else normal replay buffer
        else:
            states, actions, rewards, next_states, dones = self.memory.sample()

        # if Double DQN
        if self.double_dqn:
            # Get predicted Q values (for next actions chosen by local model) from target model
            next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            Q_targets_next = self.qnetwork_target(next_states).gather(1,next_actions)
        
        else:
            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute loss
        if self.per:
            loss = (torch.FloatTensor(is_weights)*F.mse_loss(Q_expected, Q_targets)).mean()
        else:
            loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # if PER, update priority
        if self.per:
            errors = torch.abs(Q_expected - Q_targets).data.numpy()
            self.memory.update(idxs, errors)

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
