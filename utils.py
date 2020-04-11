import torch
import numpy as np
import random
from collections import namedtuple, deque
from scipy.special import softmax

'''
Helper classes for DQN Agent
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, sample):
        """Add a new experience to memory."""
        e = self.experience(*sample)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, a, beta, beta_inc):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.p = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.e = 0.01
        self.a = a
        self.beta = beta
        self.beta_increment_per_sampling = beta_inc
    
    def add(self, sample):
        """Add a new experience to memory."""
        e = self.experience(*sample)
        self.memory.append(e)
        if len(self.memory) == 1:
            self.p.append(1.0)
        else:
            self.p.append(max(1.0,max(self.p)))
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        idxs = np.random.choice(range(len(self.memory)), 
                                size=self.batch_size, 
                                replace=False, 
                                p=self.p/sum(self.p))
        
        experiences = [self.memory[i] for i in idxs]
        priorities = [self.p[i] for i in idxs]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        sampling_probabilities = np.array(priorities) / sum(self.p)
        is_weight = np.power(len(self.memory) * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
  
        return states, actions, rewards, next_states, dones, idxs, is_weight

    def update(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = (error + self.e) ** self.a
            self.p[idx]=float(p)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
