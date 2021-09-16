import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable 
from torch.utils.data import DataLoader

import networkFiles as NF
from samplesDecision import generateSamples


# General parameters that would get set in other code
layers = 1
image_size = 20
N = image_size
num_nodes = image_size**2

dtype = torch.FloatTensor

model = NF.RecurrentDecision_FixedDecision(num_nodes, layers, num_nodes*5, image_size)
model.type(dtype)
testDict = generateSamples(image_size, 2, layers)


# test_dsetPath = torch.utils.data.TensorDataset(testDict["Environment"], testDict["Predator"], testDict["Range"])
# loader = DataLoader(test_dsetPath, batch_size=32, shuffle=True)

# loss_fn = nn.MSELoss()

# model.eval()
# num_correct, num_samples = 0, 0
# losses = []

# # The accuracy on all pixels and path pixels can be calculated from the image labels
# # Also record the loss
# for env, pred, label in loader:
# 	# Cast the image data to the correct type and wrap it in a Variable. At
# 	# test-time when we do not need to compute gradients, marking the Variable
# 	# as volatile can reduce memory usage and slightly improve speed.
# 	env = Variable(env.type(dtype), requires_grad=False)
# 	pred = Variable(pred.type(dtype), requires_grad=False)
# 	label = Variable(label.type(dtype), requires_grad=False)

# 	# Run the model forward and compare with ground truth.
# 	output = model(env, pred, dtype).type(dtype)
# 	loss = loss_fn(output, label).type(dtype)
# 	preds = output.sign() 

# 	# Compute accuracy on ALL pixels
# 	num_correct += (preds.data[:, :] == label.data[:,:]).sum()
# 	num_samples += pred.size(0) * pred.size(1)


# 	losses.append(loss.data.cpu().numpy())



 

# # Return the fraction of datapoints that were incorrectly classified.
# errorAll = 1.0 -  (float(num_correct) / (num_samples))
# avg_loss = sum(losses)/float(len(losses))

# print(errorAll)

# print(output.shape)
# # Look at the output
# fig1, ax = plt.subplots(2, 3)


# ax[0, 0].imshow(np.reshape(output[0,:].numpy(), (N, N)), cmap='Greys',  interpolation='none')
# ax[0, 0].imshow(np.reshape(env[0,:].numpy(), (N, N)), interpolation='none')
# ax[0, 1].imshow(np.reshape(output[0,:].numpy(), (N, N)), cmap='Greys',  interpolation='none')


# print(output[0,:].numpy())
# print('break')
# print(env[0,:].numpy())
