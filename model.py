import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, dueling=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.dueling = dueling
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
        if self.dueling:
            self.value_fc2 = nn.Linear(fc1_units, fc2_units)
            self.value_fc3 = nn.Linear(fc2_units, 1)
            self.adv_fc2 = nn.Linear(fc1_units, fc2_units)
            self.adv_fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # first linear layer
        x = F.relu(self.fc1(state))
        
        # if dueling architecture
        if self.dueling:
            
            # get state value
            value = F.relu(self.value_fc2(x))
            value = self.value_fc3(value)
            
            # get action advantages
            adv = F.relu(self.adv_fc2(x))
            adv = self.adv_fc3(adv)
            
            # aggregate
            avg_adv = torch.mean(adv, dim=1, keepdim=True)
            
            return value + adv - avg_adv
        
        # else if not dueling
        else:
            x = F.relu(self.fc2(x))
            return self.fc3(x)
