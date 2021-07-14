
import torch
import torch.nn as nn 
import torch.nn.functional as F 

# Small illustration of F.softmax function - Creates a discrete probability distribution
test_tensor = torch.Tensor([0.0, 0.0, 0.0]) # The values are called logits
print(F.softmax(test_tensor)) # softmax function applied on logits
nominator = torch.exp(test_tensor)
denominator = torch.sum(torch.exp(test_tensor))
print(nominator/denominator)


# logsumexp function 
assert torch.log(torch.sum(torch.exp(torch.Tensor([1,2,3])))) == torch.logsumexp(torch.Tensor([1,2,3]), dim = 0)


''' 
A short tutorial on transpose convolutions (upsampling)

INPUT  ------ CONV2D (filters/stencils) ----- Similiraties with filters

Similarities with filters --------- ConvTranspose2D (filters/stencils) --------- Input

'''

##############################################################################################################################################
input = torch.ones((1, 1, 4, 4)) # batch_size x input_channels x x_size x y_size

transpose = nn.ConvTranspose2d(1, 1, 3, stride = 1, padding = 0, bias = False) 
transpose.weight.data = torch.ones(transpose.weight.data.shape)

print(transpose.forward(input))
print(transpose.forward(input).shape)

input = torch.ones((1, 1, 4, 4)) # batch_size x input_channels x x_size x y_size

transpose = nn.ConvTranspose2d(1, 1, 3, stride = 1, padding = 1, bias = False) 
# What the shape of the input is which after applied a conv2d with padding = 0 and stride = 1 will give an output of size 4 =====> 6
transpose.weight.data = torch.ones(transpose.weight.data.shape)

print(transpose.forward(input))
# What the shape of the input is which after applied a conv2d with padding = 1 and stride = 1 will give an output of size 4 =====> 4
print(transpose.forward(input).shape)

##############################################################################################################################################
# Many input channels 
input = torch.ones((1, 16, 4, 4)) # batch_size x input_channels x x_size x y_size

transpose = nn.ConvTranspose2d(16, 1, 3, stride = 1, padding = 0, bias = False) # 16 channels should be mapped to 1 (aggregation of opinions)
transpose.weight.data = torch.ones(transpose.weight.data.shape)

print(transpose.forward(input))

##############################################################################################################################################
# Many output channels
input = torch.ones((1, 1, 4, 4)) # batch_size x input_channels x x_size x y_size

transpose = nn.ConvTranspose2d(1, 4, 3, stride = 1, padding = 0, bias = False) # 16 channels should be mapped to 4 (aggregation of opinions)
transpose.weight.data = torch.ones(transpose.weight.data.shape)

print(transpose.forward(input))

##############################################################################################################################################
# Many input + output channels
input = torch.ones((1, 16, 4, 4)) # batch_size x input_channels x x_size x y_size

transpose = nn.ConvTranspose2d(16, 4, 3, stride = 1, padding = 0, bias = False) # 16 channels should be mapped to 4 (aggregation of opinions)
transpose.weight.data = torch.ones(transpose.weight.data.shape)
transpose.weight.data[:, 1,:, :] = transpose.weight.data[:, 1,:, :] / 16

print(transpose.forward(input))
##############################################################################################################################################
'''
What is the role of padding in transpose convolutions ???
Used to specify the input from which the similarities would have stemmed (think right  to left)
transpose_input_x = math.floor( (transpose_output_x - kernel_size + 2 * padding) / stride ) + 1  

What is the role of strides in transpose convolutions ???
'''

##############################################################################################################################################

b = nn.BatchNorm1d(5) # 5 is the number of features, think number of layers of an MLP - Average feature value across examples in batch
l = nn.LayerNorm(5) # 5 is the number of features as above - Average n-th feature values for each example  

print(b.weight)
print(b.bias)

print(l.weight)
print(l.bias)

sample_input = torch.arange(3).reshape(-1,1) * torch.ones((3, 5)) # Say we have 3 training examples 

print(sample_input.T)
print(b(sample_input).T)
print(l(sample_input).T)

# How does BatchNorm2D work ? 
# Here is an illustration of how this works 
input = torch.randn(1, 10, 4, 4)
input[0,0,:,:] = torch.arange(16).reshape(4,4)
print(bn(input)[0,0,:,:])

input = torch.randn(2, 10, 4, 4)
input[0,0,:,:] = torch.arange(16).reshape(4,4)
input[1,0,:,:] = torch.arange(16, 32).reshape(4,4)
print(bn(input).shape)
print(bn(input)[0,0,:,:])
print(bn(input)[1,0,:,:])
