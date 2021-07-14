import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.distributions import Categorical
import visdom
import numpy as np
vis = visdom.Visdom()
import numpy as np
import itertools

learning_rate = 0.005

input = torch.FloatTensor(torch.ones(3)).reshape(1, -1)
n_doors = 3 # This is the number of actions we can take (open a door)
input_features = 3
door_gains = [90.0, 40.0, 30.0] 

class Policy(nn.Module):
    def __init__(self, input_features):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_features, 1)
        self.fc1.weight.data = torch.FloatTensor([1.0, 1.0, 5.0])

    def forward(self, input):
        prob_values = self.fc1.weight * input
        prob_values = F.softmax(prob_values, dim = 1)
        return prob_values

policy_network = Policy(input_features)
optimizer = optim.Adam(policy_network.parameters(), lr = 0.005)

visdom_options = {
    "width" : 500, 
    "height" : 500,
    "ylabel" : "Probs",
    "rownames" : ["gain_{}_door{}".format(int(gain), index +1) for index, gain in enumerate(door_gains)], 
}
vis.bar([0,0,0], win = "my_window", env = "main", opts = visdom_options)

def update(step):
    probabilities = policy_network.forward(input)
    sampler = Categorical(probs=probabilities)
    action = sampler.sample().item()


    # What we want is to take a larger step towards the gain when the probability provided to that is small 
    # loss = -door_gains[action] * torch.log(probabilities[0, action]) # Working !
    loss = -door_gains[action] * probabilities[0, action] # Not Working ! 

    visdom_options["title"] = "Iteration : {}".format(step + 1)
    vis.bar(probabilities.reshape(-1, 1), win = "my_window", env = "main", opts = visdom_options)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

temp = [update(step) for step in range(4000)]
vis.close(win = "my_window", env = "main")
    