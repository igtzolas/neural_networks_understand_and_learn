'''
Likelihood Ratio Gradient is a technique to find parameters of a probability distribution when 
we want to minimize/mazimize a function depending on the samples drawn from the distribution.
Another method (more restricted in when it can be applied) is Pathwise Derivative.
'''

import torch
import torch.autograd as autograd
import torch.distributions

import numpy as np
import tqdm

torch.manual_seed(3)

target_mean = torch.Tensor([[5., 5.]]) # 1x2 matrix
tracked_mean = initial_mean = torch.zeros((1, 2), requires_grad = True)

def squared_error(samples_from_target_distribution):
    # samples_from_target_distribution : m x 2
    return torch.pow(torch.sum(torch.torch.pow(samples_from_target_distribution - target_mean, 2), dim = 1), 1./2)

def multivariate_gaussian_sampler(n_dimensions, mean, covariance_matrix):
    d = torch.distributions.MultivariateNormal(torch.zeros(n_dimensions), torch.eye(n_dimensions))
    def inner(n_samples):
        return torch.distributions.MultivariateNormal(mean, covariance_matrix).sample((n_samples, )).squeeze(1)
    return inner

# Likelihood ratio gradient
training_steps = 500
batch_size = 20
step = 0.05
for _ in tqdm.tqdm(range(training_steps)):
    normal_sampler = multivariate_gaussian_sampler(2, torch.zeros(2), torch.eye(2))
    normal_samples = normal_sampler(batch_size) 

    expectation_to_minimize = torch.mean(squared_error(tracked_mean + torch.matmul(torch.cholesky(torch.eye(2), upper = False), normal_samples.transpose(0, 1)).transpose(0, 1)))
    grads = autograd.grad(expectation_to_minimize, [tracked_mean])

    tracked_mean = tracked_mean - step * grads[0] # Since we have minimization we use '-'
    print(tracked_mean)
