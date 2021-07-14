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

# f(z) = (z - [5, 5])^2
target_mean = torch.Tensor([[5., 5.]]) # 1x2 matrix
# The mean of the distribution is what we want to find (the parameters of the distribution)
tracked_mean = initial_mean = torch.zeros((1, 2), requires_grad = True)

# This is the function whose values we want to maximize/minimize
def squared_error(samples_from_target_distribution):
    # samples_from_target_distribution : m x 2
    return torch.pow(torch.sum(torch.torch.pow(samples_from_target_distribution - target_mean, 2), dim = 1), 1./2)

# This is the distribution from which we draw samples to be used as the parameters of the function to minimize/maximize
def multivariate_gaussian_distribution(n_dimensions, mean, covariance_matrix):
    def inner(X):
        probs = np.power(2* np.pi, - n_dimensions/2.) * torch.pow(covariance_matrix.det(), -1./2) * torch.exp(-0.5 * torch.sum((X - mean) * (torch.matmul(covariance_matrix.inverse(), (X - mean).transpose(0, 1))).transpose(0,1), dim = 1)) 
        return probs.reshape(-1, 1)
    return inner

def multivariate_gaussian_sampler(n_dimensions, mean, covariance_matrix):
    d = torch.distributions.MultivariateNormal(torch.zeros(n_dimensions), torch.eye(n_dimensions))
    def inner(n_samples):
        normal_samples = d.sample((n_samples,))
        return mean + torch.matmul(torch.cholesky(covariance_matrix, upper = False), normal_samples.transpose(0, 1)).transpose(0, 1)   
    return inner

# Likelihood ratio gradient
training_steps = 3000
batch_size =10
step = 0.05
for _ in tqdm.tqdm(range(training_steps)):
    # Get samples from the distribution
    with torch.no_grad():
        sampler = multivariate_gaussian_sampler(2, tracked_mean, torch.Tensor([[1., 0],[0, 1.]]))
        samples = sampler(batch_size) 
    distribution = multivariate_gaussian_distribution(2, tracked_mean, torch.Tensor([[1., 0], [0, 1.]]))
    log_probabilities = torch.log(distribution(samples))

    expectation_to_minimize = torch.mean(log_probabilities * squared_error(samples))
    grads = autograd.grad(expectation_to_minimize, [tracked_mean])

    tracked_mean = tracked_mean - step * grads[0] # Since we have minimization we use '-'
