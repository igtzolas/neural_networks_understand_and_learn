'''
Illustration of MAML Algorithm with Python

MAML Story : 

- We have a person with some characteristics/parameters 
- The person makes many trips in different countries and gathers experiences there
- The person fetches some experiences home from these countries / trips

TRIP <-> TASK 

So, how should the person update his/her characteristics so that to learn to adjust better
in future trips/tasks ? 

Training strategy : 

- Person makes trip to Greece, Italy, Malta.
- Person gathers k training experiences in Greece and k testing experiences in Greece.
  Person gathers k training experiences in Italy and k testing experiences in Italy.
  Person gathers k training experiences in Malta and k testing experiences in Malta.
- While in Greece or Italy or Malta, the person considers how his parameters should be adjusted to tackle
  the particular country's environment
  He takes one day of training and updates his/her parameters.
  He takes one gradient step to update the parameters: theta_greece = theta - step * gradient
- So we have : theta_greece = theta - step * gradient_loss_training_greece(theta)
               theta_italy = theta = step * gradient_loss_training_italy(theta)
               theta_malta = theta = step * gradient_loss_training_malta(theta)

- So, now the person comes back home and has brought in his luggage experiences from the trips
  It is time to update the parameters so that to better cope in his future trips
  theta = theta - step * gradient_of_sum(
      loss_testing_greece(theta_greece) + 
      loss_testing_malta(theta_malta)  
      )
      loss_testing_italy(theta_italy)   +

  The above can be decomposed like the following in case we want to make use of 2nd order derivatives
  theta = theta - step * gradient_of_sum(
      loss_testing_greece(theta - step * gradient_loss_training_greece(theta)) + 
      loss_testing_italy(theta - step * gradient_loss_training_italy(theta))  +
      loss_testing_malta(theta - step * gradient_loss_training_malta(theta))  
      )

  If we want to avoid second order derivatives we can pretend that they are already calculated in the training phase
  theta = theta - step * gradient_of_sum(
      loss_testing_greece(theta - V_Greece) + 
      loss_testing_italy(theta - V_Italy)  +
      loss_testing_malta(theta - V_Malta)  
      )

Can the final updates of theta say something about the connection between tasks?
So, if we had Greece, Russia and China as the trip destination countries, would the 
nature of the update say something about the connection between the trips/tasks ?

REPTILE Algorithm
The reptile algorithm introduces one single idea/element:
Countries are visited one after the other but N days are spent training in each one of them
(N gradient descent steps) until another country is visited.
What is the influence of the step under this algorithm ? 


sudo /usr/bin/nvidia-smi -pm ENABLED
sudo /usr/bin/nvidia-smi -c EXCLUSIVE_PROCESS

sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm

https://discuss.pytorch.org/t/assign-parameters-to-nn-module-and-have-grad-fn-track-it/62677/3
'''

import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.autograd as autograd

import tqdm
import numpy as np
import visdom

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vis = visdom.Visdom()
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# Hyper parameters
learning_rate = 0.001
input_size = 1
K = K_shot_number = number_of_training_examples_per_task = number_of_testing_examples_per_task = 20
# How to train no metalearning
training_tasks = 10000 # This is the number of meta-training tasks
batch_size = 1
# How to train metalearning
meta_training_cycles = 10000
tasks_per_meta_training_cycle = 1
alpha = 0.001
hidden_layer_size = 40

#########################################################################################################################
# Dataset class
class SinusoidalDataset(Dataset):
    def __init__(self, K):
        super(SinusoidalDataset, self).__init__()
        self.K = K


    def __getitem__(self, index):
        amplitude = np.random.uniform(0.1, 5.0)
        phase = np.random.uniform(0, np.pi / 2)
        inputs = x_values = np.random.uniform(-5, 5, 2 * self.K) # K training points, K testing point
        outputs = y_values = amplitude * np.sin(inputs - phase)
        return ((
                FloatTensor(inputs).reshape(2 * self.K, 1)[:self.K, :], 
                FloatTensor(outputs).reshape(2 * self.K, 1)[:self.K, :],
            ),
            ( 
                FloatTensor(inputs).reshape(2 * self.K, 1)[self.K:, :], 
                FloatTensor(outputs).reshape(2 * self.K, 1)[self.K:, :]
            ))

    def __len__(self):
        return batch_size

# Dataset class
class SinusoidalDataset2(Dataset):
    def __init__(self, K):
        super(SinusoidalDataset2, self).__init__()
        self.K = K


    def __getitem__(self, index):
        train_data_x = torch.tensor([])
        test_data_x = torch.tensor([])
        train_data_y = torch.tensor([])
        test_data_y = torch.tensor([])

        for i in range(20):
            amplitude = np.random.uniform(0.1, 5.0)
            phase = np.random.uniform(0, np.pi )
            inputs = x_values = np.random.uniform(-5, 5, 2 * self.K) # K training points, K testing point
            outputs = y_values = amplitude * np.sin(inputs - phase)
            train_data_x = torch.cat((train_data_x, FloatTensor(inputs).reshape(2 * self.K, 1)[:self.K, :]), dim = 0) 
            train_data_y = torch.cat((train_data_y, FloatTensor(outputs).reshape(2 * self.K, 1)[:self.K, :]), dim = 0)
            test_data_x = torch.cat((train_data_x, FloatTensor(inputs).reshape(2 * self.K, 1)[self.K:, :]), dim = 0) 
            test_data_y = torch.cat((train_data_y, FloatTensor(outputs).reshape(2 * self.K, 1)[self.K:, :]), dim = 0)  
        return ((
                train_data_x, 
                train_data_y
            ),
            ( 
                test_data_x,
                test_data_y
            ))

    def __len__(self):
        return batch_size

# Model
class LinearModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        # torch.nn.init.xavier_uniform_(self.fc1.weight)
        # torch.nn.init.xavier_uniform_(self.fc2.weight)
        # torch.nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, input):
        output_fc1 = F.relu(self.fc1(input))
        output_fc2 = F.relu(self.fc2(output_fc1))
        output = self.fc3(output_fc2)
        return output

#########################################################################################################################
line_options = {}
line_options["title"] = "Loss"
line_options["xlabel"] = "#Iterations"
line_options["ylabel"] = "Loss"
line_options_1 = {}
line_options_1["title"] = "Predictions"
line_options_1["xlabel"] = "X Values"
line_options_1["ylabel"] = "Predictions"

dataset_sinusoidal = SinusoidalDataset2(K)
dataloader_sinusoidal = DataLoader(dataset_sinusoidal, batch_size = batch_size, shuffle = True, num_workers = 0)
linear_model_no_metalearning = LinearModel(input_size, hidden_layer_size).to(device)
optimizer_no_metalearning = optim.Adam(linear_model_no_metalearning.parameters(), lr = learning_rate)
# parameter_shapes = [el.shape for el in linear_model.parameters()]
parameter_names, parameter_shapes = zip(*((t[0],t[1].shape)  for t in linear_model_no_metalearning.named_parameters()))
for task_index in tqdm.tqdm(range(training_tasks)): 
    for batch_index, ((x_train, y_train),(x_test, y_test)) in enumerate(dataloader_sinusoidal):
        predicted_x = linear_model_no_metalearning(x_train)

        loss = F.mse_loss(predicted_x, y_train)

        optimizer_no_metalearning.zero_grad()
        loss.backward()
        optimizer_no_metalearning.step()

        # Visualizing the loss while training takes place
        # vis.line(X = [task_index] , Y = [loss.item()], env = "main", win = "Loss Evolution", opts = line_options, update = "append")
        # x_values = torch.arange(-5, 5, 0.05).to(device)
        # vis.line(X = x_values , Y = linear_model_no_metalearning.forward(x_values.reshape(1,-1, 1)).squeeze(0).squeeze(-1), env = "main", win = "Predictions", opts = line_options_1)

# Saving the trained model        
torch.save(linear_model_no_metalearning.state_dict(), "./model_no_metalearning.pth")
# linear_model_no_metalearning.load_state_dict(torch.load("./model_no_metalearning.pth"))

#########################################################################################################################
# How should this be adapted to implement MAML
dataset_sinusoidal = SinusoidalDataset(K)
dataloader_sinusoidal = DataLoader(dataset_sinusoidal, batch_size = batch_size, shuffle = True, num_workers = 0) # batch size is set to 1
linear_model_maml = LinearModel(input_size, hidden_layer_size).to(device)
optimizer_maml = optim.Adam(linear_model_maml.parameters(), lr = learning_rate)

for task_index in tqdm.tqdm(range(meta_training_cycles)):
    # We will be considering 10 tasks/trips per meta_training_cycle 
    for _ in range(tasks_per_meta_training_cycle):
        loss_after_trip_updates = 0     
        for batch_index, ((x_train, y_train),(x_test, y_test)) in enumerate(dataloader_sinusoidal):
            linear_model_with_updated_weights_per_task = LinearModel(input_size, hidden_layer_size).to(device)
            
            predicted_x = linear_model_maml(x_train)
            loss = F.mse_loss(predicted_x, y_train)
            task_gradients = autograd.grad(loss, linear_model_maml.parameters(), create_graph=True, retain_graph=True)
      
            for (param, initial_thetas), gradients in zip(linear_model_maml.named_parameters(), task_gradients):
                param_name_portions = param.split(".")
                for name in param_name_portions[:-1]:
                    module = linear_model_with_updated_weights_per_task.__getattr__(name)
                module.__delattr__(param_name_portions[-1])
                setattr(module, param_name_portions[-1], (initial_thetas - alpha * gradients))

            # linear_model_with_updated_weights_per_task.load_state_dict(state_dict)
            predicted_test_x = linear_model_with_updated_weights_per_task(x_test)
            loss_after_trip_updates += F.mse_loss(predicted_test_x, y_test)
        
        optimizer_maml.zero_grad()
        loss_after_trip_updates.backward()
        optimizer_maml.step()

        # # Visualizing the loss while training takes placeS
        # vis.line(X = [task_index] , Y = [loss.item()], env = "main", win = "Loss Evolution", opts = line_options, update = "append")

torch.save(linear_model_maml.state_dict(), "./model_metalearning.pth")
# linear_model_maml.load_state_dict(torch.load("./model_maml.pth"))

#########################################################################################################################
# How should this be adapted to implement Reptile
dataset_sinusoidal = SinusoidalDataset(K)
dataloader_sinusoidal = DataLoader(dataset_sinusoidal, batch_size = batch_size, shuffle = True, num_workers = 0) # batch size is set to 1
linear_model_reptile = LinearModel(input_size, hidden_layer_size).to(device)
optimizer_reptile = optim.Adam(linear_model_reptile.parameters(), lr = learning_rate)
task_optimization_cycles = 3

for task_index in tqdm.tqdm(range(training_tasks)): 
    for batch_index, ((x_train, y_train),(x_test, y_test)) in enumerate(dataloader_sinusoidal):
        for _ in range(task_optimization_cycles):
            predicted_x = linear_model_reptile(x_train)

            loss = F.mse_loss(predicted_x, y_train)

            optimizer_reptile.zero_grad()
            loss.backward()
            optimizer_reptile.step()

# Saving the trained model        
torch.save(linear_model_reptile.state_dict(), "./model_reptile.pth")
# linear_model_reptile.load_state_dict(torch.load("./model_reptile.pth"))


#########################################################################################################################
# How our models perform in new tasks/trips (provided they are given 10 iterations to update/adjust)
optimizer_no_metalearning.defaults["lr"] = 0.005
optimizer_maml.defaults["lr"] = 0.005
optimizer_reptile.defaults["lr"] = 0.005

dataset_sinusoidal = SinusoidalDataset(10)
dataloader_sinusoidal = DataLoader(dataset_sinusoidal, batch_size = batch_size, shuffle = True, num_workers = 0) # batch size is set to 1
iterations = 20

for batch_index, ((x_train, y_train),(x_test, y_test)) in enumerate(dataloader_sinusoidal):
    for iteration in range(iterations):

        predicted_x_no_metalearning = linear_model_no_metalearning(x_train)
        loss_no_metalearning = F.mse_loss(predicted_x_no_metalearning, y_train)

        predicted_x_maml = linear_model_maml(x_train)
        loss_maml = F.mse_loss(predicted_x_maml, y_train)

        predicted_x_reptile = linear_model_reptile(x_train)
        loss_reptile = F.mse_loss(predicted_x_reptile, y_train)

        optimizer_no_metalearning.zero_grad()
        loss_no_metalearning.backward()
        optimizer_no_metalearning.step()

        optimizer_maml.zero_grad()
        loss_maml.backward()
        optimizer_maml.step()

        optimizer_reptile.zero_grad()
        loss_reptile.backward()
        optimizer_reptile.step()

        predicted_test_x_no_metalearning = linear_model_no_metalearning(x_test)
        predicted_test_x_maml = linear_model_maml(x_test)
        predicted_test_x_reptile = linear_model_reptile(x_test)

        # Visualizing the loss while training takes place
        vis.scatter(torch.stack((x_test.reshape(-1), y_test.reshape(-1)), axis = 1), env = "main", win = "Real Values", opts = {"title" : "Real values"})
        vis.scatter(torch.stack((x_test.reshape(-1), predicted_test_x_no_metalearning.reshape(-1)), axis = 1), env = "main", win = "NO METALEARNING", opts = {"title" : "NO METALEARNING"})
        vis.scatter(torch.stack((x_test.reshape(-1), predicted_test_x_maml.reshape(-1)), axis = 1), env = "main", win = "MAML", opts = {"title" : "MAML"})
        vis.scatter(torch.stack((x_test.reshape(-1), predicted_test_x_reptile.reshape(-1)), axis = 1), env = "main", win = "REPTILE", opts = {"title" : "REPTILE"})
#########################################################################################################################

print("No metalearning : {}".format(F.mse_loss(predicted_x_no_metalearning, y_test)))
print("MAML : {}".format(F.mse_loss(predicted_test_x_maml, y_test)))
print("Reptile : {}".format(F.mse_loss(predicted_test_x_reptile, y_test)))