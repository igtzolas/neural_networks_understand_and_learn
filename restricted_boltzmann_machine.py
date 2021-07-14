# Implementing the scenario in : https://www.youtube.com/watch?v=Fkw0_aAtwIw

import numpy as np  
import pandas as pd 
import itertools

import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.distributions as distributions

# Analyzing training example : [1, 0, 1] -> Aisha : Present (1), Beto : Absent (0), Cameron : Present (1)
training_examples = [[1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1]]
inputs = ["Aisha", "Beto", "Cameron", "Descartes", "Euler"]
name_to_indices = {name:index for (index,name) in enumerate(inputs)}
input_combinations = sum([list(map(list, itertools.combinations(inputs, i))) for i in range(len(inputs) + 1)], [])

V = visible_inputs = training_tensors = torch.tensor(training_examples).type(torch.float)

number_of_visible_layers = 3
number_of_hidden_layers = 2

# Hyperparameters
learning_rate = lr = 0.005
n_epochs = 1000

class RBM(nn.Module):
    def __init__(self):
        super(RBM, self).__init__()

        self.uniform_sampler = distributions.Uniform(0.0, 1,0)

        self.K = trajectory_length = 5
        self.biases_visible = nn.Parameter(data = torch.zeros((number_of_visible_layers, 1)), requires_grad = True)
        self.biases_hidden = nn.Parameter(data = torch.zeros((number_of_hidden_layers, 1)), requires_grad = True)
        self.weights_visible_hidden = nn.Parameter(data = torch.zeros(number_of_visible_layers, number_of_hidden_layers), requires_grad = True)

    
    def forward(self, visible_example): 

        # Visible layer -> Hidden layer

        # Choose hidden_layer_values based on visible_layer_examples
        # In order to choose the values for the hidden neurons, we take advantage of the fact that p(h(i)|v) is independent of p(h(j)|v) 
        prob_h_i_given_visible_example = torch.sigmoid(self.biases_hidden.T + torch.matmul(visible_example.reshape(1, -1), self.weights_visible_hidden))
        uniform_probs = self.uniform_sampler.sample((prob_h_i_given_visible_example.shape))
        hidden_values_for_example = (prob_h_i_given_visible_example > uniform_probs).clone().detach().requires_grad_(False).type(torch.float)
        # Return the probability of (visible_layer_values, hidden_layer_values).
        # We want to increase this probability and as such we will update the biases and weights to this direction
        energy = - (
            torch.sum(visible_example.reshape(self.biases_visible.shape) * self.biases_visible) +   
            torch.sum(hidden_values_for_example.reshape(self.biases_hidden.shape) * self.biases_hidden) +
            torch.sum(torch.matmul(visible_example.reshape(-1, 1).T, self.weights_visible_hidden) * hidden_values_for_example)
        )
        score = - energy
        prob_h_v = torch.exp(score)
        log_prob_positive = score # = torch.log(torch.exp(score))

        # K consecutive samples until we reach a sample for v and h from sampled hidden layer
        sample_v = True
        hidden_example = hidden_values_for_example
        for k in range(self.K):
            if sample_v:
                prob_v_i_given_visible_example = torch.sigmoid(self.biases_visible.T + torch.matmul(self.weights_visible_hidden, hidden_example.reshape(-1, 1)).T)
                uniform_probs = self.uniform_sampler.sample((prob_v_i_given_visible_example.shape))
                visible_example = (prob_v_i_given_visible_example > uniform_probs).clone().detach().requires_grad_(False).type(torch.float)
                sample_v = False
            else:
                prob_h_i_given_visible_example = torch.sigmoid(self.biases_hidden.T + torch.matmul(visible_example.reshape(1, -1), self.weights_visible_hidden))
                uniform_probs = self.uniform_sampler.sample((prob_h_i_given_visible_example.shape))
                hidden_example = (prob_h_i_given_visible_example > uniform_probs).clone().detach().requires_grad_(False).type(torch.float)   
                sample_v = True

        # We want to increase this probability and as such we will update the biases and weights to this direction
        energy = - (
            torch.sum(visible_example.reshape(self.biases_visible.shape) * self.biases_visible) +   
            torch.sum(hidden_example.reshape(self.biases_hidden.shape) * self.biases_hidden) +
            torch.sum(torch.matmul(visible_example.reshape(-1, 1).T, self.weights_visible_hidden) * hidden_example)
        )
        score = - energy
        prob_h_v = torch.exp(score)
        log_prob_negative = score # = torch.log(torch.exp(score))

        return log_prob_positive - log_prob_negative   
    
    def get_joint_probability(self, visible, hidden):
        energy = - (
            torch.sum(visible.reshape(self.biases_visible.shape) * self.biases_visible) +   
            torch.sum(hidden.reshape(self.biases_hidden.shape) * self.biases_hidden) +
            torch.sum(torch.matmul(visible.reshape(-1, 1).T, self.weights_visible_hidden) * hidden)
        )
        score = - energy
        prob_h_v = torch.exp(score)
        return prob_h_v

rbm = RBM()
optimizer = optim.SGD(rbm.parameters(), lr=learning_rate)

n_epochs = 1000
for epoch in range(n_epochs):
    for visible_example in V:
        log_probability = rbm.forward(visible_example)

        loss = - log_probability

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        print(f"{epoch}/{n_epochs} : Epoch The loss is {loss.item()}")
    
for combination in input_combinations:
    key = "-".join([name for name in combination])
    binary_value = torch.tensor([0 for _ in range(len(inputs))]).type(torch.float)
    for name in combination:
        binary_value[name_to_indices[name]] = 1.0
    v, h = binary_value[:number_of_visible_layers], binary_value[number_of_visible_layers:]
    probability = rbm.get_joint_probability(v, h)
    print(f'{key}({binary_value.tolist()}) : {probability.item()}')


