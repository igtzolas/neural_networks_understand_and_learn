'''
How to make use of virtualenv : 
Installation : pip install --upgrade virtualenv

Create a virtual environment, activate it and install the requirements in requirements.txt.
Creation : virtualenv venv
Activation : source venv/bin/activate
Installation of requirements : pip install -r requirements.txt
'''

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.distributions as distributions

###################################################################################################################
# 1st order derivatives
# The derivative of the cdf is the pdf value
normal_distribution = distributions.Normal(loc = 0.0, scale = 1.0)
x = torch.tensor([0.0], requires_grad = True)
y = torch.tensor([0.0], requires_grad = True)
expression = normal_distribution.cdf(x)
gradients = autograd.grad([expression], [x])
assert (gradients[0].item() - normal_distribution.log_prob(x).exp().item()) < 0.001 

###################################################################################################################
# Making use of backward function of an autograd Variable to calculate derivatives
c2 = torch.Tensor([100.0])
c1 = torch.Tensor([2.0])
c0 = torch.Tensor([1.0])

x = torch.Tensor([1.0])
x.requires_grad = True
y = torch.Tensor([1.0])
y.requires_grad = True
result_forward_graph = sum(c2 * x * x  + c1 * x + c0  +  c2 * y * y  + c1 * y + c0)
print(x.grad) # The value for the partial derivative with respect to x has not been updated
print(y.grad) # The value for the partial derivative with respect to y has not been updated
# In order to update the values one has to run backward on the autograd variable which traverses the gradient graph
# and updates the gradients in it
# result_forward_graph.backward()
# print(x.grad) # Result is 202
# print(y.grad) # Result is 202
result_forward_graph.backward(create_graph = True, retain_graph = True)
print(x.grad) # Result is 202
print(y.grad) # Result is 202
result_forward_graph.backward(retain_graph = True)
print(x.grad) # Result is 404
print(y.grad) # Result is 404

###################################################################################################################
# 2nd order derivative
# In ordet to get the values of the 2nd derivatives one has to make use of the grad function 
c2 = torch.Tensor([100.0])
c1 = torch.Tensor([2.0])
c0 = torch.Tensor([1.0])

x = torch.Tensor([1.0])
x.requires_grad = True
y = torch.Tensor([1.0])
y.requires_grad = True
result_forward_graph = sum(c2 * x * x  + c1 * x + c0  +  c2 * y * y  + c1 * y + c0)
first_order_derivative_forward_create_retain_graph = autograd.grad([result_forward_graph], [x, y], create_graph = True, retain_graph = True)
second_order_derivative_forward_graph = autograd.grad(first_order_derivative_forward_create_retain_graph, [x, y], create_graph=True, retain_graph = True)
# # result_forward_graph.backward(retain_graph=True)
# first_order_derivative_forward_create_retain_graph[0].backward(retain_graph = True)
# first_order_derivative_forward_create_retain_graph[1].backward(retain_graph = True)
# print(x.grad) # This is the 2nd order derivative
# print(y.grad) # This is the 2nd order derivative
# first_order_derivative_forward_create_retain_graph[0].backward(retain_graph = True)
# first_order_derivative_forward_create_retain_graph[1].backward(retain_graph = True)
# first_order_derivative_forward_create_retain_graph[0].backward(retain_graph = True)
# first_order_derivative_forward_create_retain_graph[1].backward(retain_graph = True)

import torch.nn as nn
layer = nn.Linear(1, 1, bias=False)
# layer.weight.register_hook(lambda x : first_order_derivative_forward_create_retain_graph[0].backward())

# del layer.weight
# layer.weight = first_order_derivative_forward_create_retain_graph[0].reshape(1, 1)

del layer.weight
setattr(layer, "weight", first_order_derivative_forward_create_retain_graph[0].reshape(1, 1))

result = layer.forward(torch.tensor([2.0]))
'''
A Variable (torch.autograd.Variable) has some attributes : 
var.data
var.grad
var.torch_fn : Points to a node in the backward graph
var.is_leaf
var._version

var1.data = var2.data
Any change in var2 data is reflected also in var1 and vice versa

var1.data = var2.data.clone()
Any change in var2 is NOT reflected also in var1 and vice versa

When var.detach() gets called all properties of a tensor (data, grad, etc) are cloned ! 
'''


