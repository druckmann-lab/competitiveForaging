import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from matplotlib.colors import colorConverter

import networkFiles as NF
from samplesPropagation import generateSamples
from generateDictionary import loadStateDict
# from weightMask import generateFixedWeightMask_PredPrey


image_size = 20
N = image_size
num_nodes = image_size**2
dtype = torch.FloatTensor


model_name = ['modelBlock_prop07.pth.tar', 'modelBlock_prop10.pth.tar', 'modelBlock_prop12.pth.tar']
layers_list = [7, 10, 12]


for layers in layers_list:
	# Loop over models
	loss_fn = nn.MSELoss()
	testDict = generateSamples(image_size, 10000, layers)

	for i in range(3):

		# Load in the relevant model and extract the propagation weights
		modelBlock_State = torch.load(model_name[i], map_location=torch.device('cpu'))
		modelBlock = loadStateDict(modelBlock_State)
		modelProp = modelBlock[modelBlock['Meta']['BestModel']]["Model"]

		w = modelProp.iteratedLayer_Pred.w.data
		a = modelProp.iteratedLayer_Pred.a.data
		b = modelProp.iteratedLayer_Pred.bias.data

		# Set up model with given number of step sizes using these weights
		model = NF.PropagationOnly_FixedWeights(num_nodes, layers, num_nodes*5, image_size)
		model.type(dtype)

		model.iteratedLayer_Pred.w.data = w
		model.iteratedLayer_Pred.a.data = a
		model.iteratedLayer_Pred.bias.data = b

		model.eval()

		

		# Set up for evaluating the accuracy
		test_dsetPath = torch.utils.data.TensorDataset(testDict["Environment"], testDict["Predator"], testDict["Range"])
		loader = DataLoader(test_dsetPath, batch_size=32, shuffle=False)

		num_correct, num_samples = 0, 0
		losses = []


		# The accuracy on all pixels and path pixels can be calculated from the image labels
		# Also record the loss
		for env, pred, label in loader:
			# Cast the image data to the correct type and wrap it in a Variable. At
			# test-time when we do not need to compute gradients, marking the Variable
			# as volatile can reduce memory usage and slightly improve speed.
			env = Variable(env.type(dtype), requires_grad=False)
			pred = Variable(pred.type(dtype), requires_grad=False)
			label = Variable(label.type(dtype), requires_grad=False)

			# Run the model forward and compare with ground truth.
			output = model(env, pred, dtype).type(dtype)
			loss = loss_fn(output, label).type(dtype)
			preds = output.sign() 

			# Compute accuracy on ALL pixels
			num_correct += (preds.data[:, :] == label.data[:,:]).sum()
			num_samples += pred.size(0) * pred.size(1)


			losses.append(loss.data.cpu().numpy())

		errorAll = 1.0 -  (float(num_correct) / (num_samples))
		avg_loss = sum(losses)/float(len(losses))

		print('Trained with %i layers, tested with %i layers: %.6f' % (layers_list[i], layers, errorAll))

