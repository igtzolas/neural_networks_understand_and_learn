'''
Introduction to neural networks and Gradient Descent 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define the architecture of the Neural Network
class QNet(nn.Module):   
    
    def __init__(self):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(3, 3)
        # self.fc2 = nn.Linear(256, n_actions)
    
    def forward(self, input):
        x = self.fc1(input)
        # x = F.relu(x)
        # x = self.fc2(x)
        return x

# Instantiating the network
nn = QNet()
# Getting an optimizer which will be operating to the nn parameters
optimizer = optim.SGD(nn.parameters(), lr = 0.1)

# For simplicity setting initial weights to predefined values 
nn.fc1.weight.data = torch.arange(1,10, dtype=torch.float32).reshape(3,3)
nn.fc1.bias.data = torch.zeros(3)

# Defining a sample input
input = torch.cat((torch.ones(1,3), 2 * torch.ones(1,3)), 0)
result = nn.forward(input)

# Defining a very basic loss function and calculating the gradients
loss = result.sum() 
optimizer.zero_grad()
loss.backward()
optimizer.step()



