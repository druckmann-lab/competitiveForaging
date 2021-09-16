import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import collections
from torch.autograd import Variable
from weightMask import generateSquareWeightMask, generateGridWeightMask, generateGridWeightMask_PredPrey, generateFixedWeightMask_PredPrey

## Netowrk Types ##


## This is the important class for this folder
## This implements the propagated and recurrent decision layer fixed

class Fixed_PredPrey(torch.nn.Module):
	def __init__(self, num_pixels, layers, H_decision, imageSize):
		super(Fixed_PredPrey, self).__init__()
		weightMask, diagMask, edgeMask, cornerMask = generateFixedWeightMask_PredPrey(imageSize)
		self.iteratedLayer = FixedPropagationDecision(num_pixels, num_pixels, layers, weightMask, diagMask, edgeMask, cornerMask)
		

	def forward(self, X, pred, prey, cave, dtype):	
		
		prey_range, pred_range, decision = self.iteratedLayer(prey, pred, X, X, cave)
		
		# This block is to fix a problem when only one element is passed in
		# The first dimension is normally the batch, but the repeated layers will squeeze out this dimension
		# Add it back in before concatenating

		# if (pred_range.dim() == 1):
		# 	pred_range.unsqueeze(0)
		
		return prey_range, pred_range, decision




## Classes for Predator-Prey Model


# This version has the grid structure but all the parameters are trainable
class PredPrey_Decision(torch.nn.Module):
	def __init__(self, num_pixels, layers, H_decision, imageSize):
		super(PredPrey_Decision, self).__init__()
		self.outputLayer1 = nn.Linear(3*num_pixels, H_decision)
		self.outputLayer2 = nn.Linear(H_decision, 2)


		self.relu = nn.ReLU()
		

	def forward(self, X, pred, prey, cave, dtype):	

		pred_range = pred
		prey_range = prey

		if (pred_range.dim() == 1):
			pred_range.unsqueeze(0)

		if (prey_range.dim() == 1):
			prey_range.unsqueeze(0)

		if (cave.dim() == 1):
			cave.unsqueeze(0)

		tags = torch.cat((pred_range, prey_range, cave), dim = 1)

		h = self.relu(self.outputLayer1(tags))
		label = self.relu(self.outputLayer2(h))
		
		return label

# This version has the grid structure but all the parameters are trainable
class RecurrentScaledGrid_PredPrey(torch.nn.Module):
	def __init__(self, num_pixels, layers, H_decision, imageSize):
		super(RecurrentScaledGrid_PredPrey, self).__init__()
		weightMask, diagMask = generateGridWeightMask_PredPrey(imageSize)
		self.iteratedLayer_Pred = RepeatedLayersScaledMasked(num_pixels, num_pixels, layers, weightMask, diagMask)
		self.iteratedLayer_Prey = RepeatedLayersScaledMasked(num_pixels, num_pixels, layers, weightMask, diagMask)
		self.outputLayer1 = nn.Linear(3*num_pixels, H_decision)
		self.outputLayer2 = nn.Linear(H_decision, 2)

		self.tanh = nn.Tanh()
		

	def forward(self, X, pred, prey, cave, dtype):	
		
		pred_range = self.iteratedLayer_Pred(pred, X)
		prey_range = self.iteratedLayer_Prey(prey, X)

		tags = torch.cat((pred_range, prey_range, cave), dim = 1)

		h = self.tanh(self.outputLayer1(tags))
		label = self.tanh(self.outputLayer2(h))
		
		return label


# This version was intended to be where the weights are shared for each pixel, i.e. it learns a single w



# This is the version that shares w and b across all pixels
# In it's current form, this code should not work
# Corners and edges should have to learn a different weight than pixels in the center of the image


class RecurrentSharedAll_PredPrey(torch.nn.Module):
	def __init__(self, num_pixels, layers, H_decision, imageSize):
		super(RecurrentSharedAll_PredPrey, self).__init__()
		weightMask, diagMask = generateGridWeightMask_PredPrey(imageSize)
		self.iteratedLayer_Pred = RepeatedLayersShared(num_pixels, num_pixels, layers, weightMask, diagMask)
		self.iteratedLayer_Prey = RepeatedLayersShared(num_pixels, num_pixels, layers, weightMask, diagMask)
		self.outputLayer1 = nn.Linear(3*num_pixels, H_decision)
		self.outputLayer2 = nn.Linear(H_decision, H_decision)
		self.outputLayer3 = nn.Linear(H_decision, 2)

		self.tanh = nn.Tanh()
		

	def forward(self, X, pred, prey, cave, dtype):	
		
		pred_range = self.iteratedLayer_Pred(pred, X)
		prey_range = self.iteratedLayer_Prey(prey, X)

		if (pred_range.dim() == 1):
			pred_range.unsqueeze(0)

		if (prey_range.dim() == 1):
			prey_range.unsqueeze(0)

		if (cave.dim() == 1):
			cave.unsqueeze(0)

		tags = torch.cat((pred_range, prey_range, cave), dim = 1)

		h = self.tanh(self.outputLayer1(tags))
		h = self.tanh(self.outputLayer2(h))
		label = self.tanh(self.outputLayer3(h))
		
		return label



# Forward-engineered version of the network
# Recurrent weights are fixed, decision layer is learned

class RecurrentFixed_PredPrey(torch.nn.Module):
	def __init__(self, num_pixels, layers, H_decision, imageSize):
		super(RecurrentFixed_PredPrey, self).__init__()
		weightMask, diagMask, edgeMask, cornerMask = generateFixedWeightMask_PredPrey(imageSize)
		self.iteratedLayer_Pred = RepeatedLayersFixed(num_pixels, num_pixels, layers, weightMask, diagMask, edgeMask, cornerMask)

		self.tanh = nn.Tanh()
		

	def forward(self, X, pred, dtype):	
		
		pred_range = self.iteratedLayer_Pred(pred, X)
		
		# This block is to fix a problem when only one element is passed in
		# The first dimension is normally the batch, but the repeated layers will squeeze out this dimension
		# Add it back in before concatenating

		# if (pred_range.dim() == 1):
		# 	pred_range.unsqueeze(0)
		
		return pred_range


class RecurrentSharedPixel_PredPrey(torch.nn.Module):
	def __init__(self, num_pixels, layers, H_decision, imageSize):
		super(RecurrentSharedPixel_PredPrey, self).__init__()
		weightMask, diagMask, edgeMask, cornerMask = generateFixedWeightMask_PredPrey(imageSize)
		self.iteratedLayer_Pred = RepeatedLayersSharedPixel(num_pixels, num_pixels, layers, weightMask, diagMask, edgeMask, cornerMask)

		self.tanh = nn.Tanh()
		

	def forward(self, X, pred, dtype):	
		
		pred_range = self.iteratedLayer_Pred(pred, X)
		
		# This block is to fix a problem when only one element is passed in
		# The first dimension is normally the batch, but the repeated layers will squeeze out this dimension
		# Add it back in before concatenating

		# if (pred_range.dim() == 1):
		# 	pred_range.unsqueeze(0)
		
		return pred_range



## Classes used to construct network types ##


# This is the original code from the working masked code for edge-connected pixel task

class RepeatedLayersScaledMasked(torch.nn.Module):
	def __init__(self, D_input, hidden, layers, weightMask, diagMask):
		super(RepeatedLayersScaledMasked, self).__init__()

		self.mask = weightMask
		self.diagMask = diagMask
		self.invertMask = torch.ones((hidden, hidden)).type(torch.cuda.ByteTensor) - self.mask
		self.invertDiag = torch.ones((hidden, hidden)).type(torch.cuda.ByteTensor) - self.diagMask
		self.iteration = layers
		self.hiddenWeight = nn.Linear(hidden, hidden)
		self.inputWeight = nn.Linear(D_input, hidden, bias=False)
		self.tanh = nn.Tanh()
		self.scalar = nn.Parameter(torch.ones(1)*2, requires_grad=True)
		self.hiddenWeight.weight.data[self.invertMask] = 0
		#self.hiddenWeight.weight.data[self.mask] = 0.25

		self.inputWeight.weight.data[:] = 0
		#self.inputWeight.weight.data[self.diagMask] = 1
		#self.hiddenWeight.bias.data[:] = -0.15

		self.hiddenWeight.weight.register_hook(self.backward_hook)
		self.inputWeight.weight.register_hook(self.backward_hook_input)
		

	def forward(self, initial_hidden, input):
		u = initial_hidden.clone()
		for _ in range(0, self.iteration):
			v = self.hiddenWeight(u) + self.inputWeight(input)
			u = self.tanh(v * self.scalar.expand_as(v))
			#u = torch.sign(u)
		return u

	def backward_hook(self, grad):
		out = grad.clone()
		out[self.invertMask] = 0
		return out


	def backward_hook_input(self, grad):
		out = grad.clone()
		out[self.invertDiag] = 0
		return out




# Class below is modified to deal with batched matrix multiply
# Has one w and b that are shared by all the pixels

class RepeatedLayersSharedAll(torch.nn.Module):
	def __init__(self, D_input, hidden, layers, weightMask, diagMask):
		super(RepeatedLayersSharedAll, self).__init__()

		self.mask = weightMask
		self.diagMask = diagMask
		self.invertMask = torch.ones((hidden, hidden)).type(torch.cuda.ByteTensor) - self.mask
		self.invertDiag = torch.ones((hidden, hidden)).type(torch.cuda.ByteTensor) - self.diagMask
		self.iteration = layers

		# For this case we want these to remain fixed
		self.hiddenWeight = nn.Parameter(torch.ones(hidden, hidden), requires_grad=False)
		self.bias = nn.Parameter(torch.ones(hidden, 1), requires_grad=False)
		self.inputWeight = nn.Parameter(torch.ones(D_input, hidden), requires_grad=False)


		self.tanh = nn.Tanh()
		self.scalar = nn.Parameter(torch.ones(1)*2, requires_grad=True)

		# Set up the hidden weights
		self.w = nn.Parameter(torch.ones(1)*2, requires_grad=True)
		self.b = nn.Parameter(torch.ones(1)*2, requires_grad=True)
		self.hiddenWeight.data[self.invertMask] = 0
		self.hiddenWeight.data[self.mask] = 1


		# Set up the input weights
		self.a = nn.Parameter(torch.ones(1)*2, requires_grad=True)
		self.inputWeight.data[:] = 0
		self.inputWeight.data[self.diagMask] = 1
		#self.hiddenWeight.bias.data[:] = -0.15

		# self.hiddenWeight.weight.register_hook(self.backward_hook)
		# self.inputWeight.weight.register_hook(self.backward_hook_input)
		

	def forward(self, initial_hidden, input):
		u = initial_hidden.clone()
		input_expand = input.unsqueeze(-1)
		for _ in range(0, self.iteration):
			u_expand = u.unsqueeze(-1)
			v_expand = torch.matmul((self.w.expand_as(self.hiddenWeight) * self.hiddenWeight), u_expand) + (self.b * self.bias) + \
				torch.matmul((self.a.expand_as(self.inputWeight) * self.inputWeight), input_expand)
			v = v_expand.squeeze()
			u = self.tanh(v * self.scalar.expand_as(v))
			#u = torch.sign(u)
		return u


# This code has one weight learned by each pixel

class RepeatedLayersSharedPixel(torch.nn.Module):
	def __init__(self, D_input, hidden, layers, weightMask, diagMask, edgeMask, cornerMask):
		super(RepeatedLayersSharedPixel, self).__init__()

		self.mask = weightMask
		self.diagMask = diagMask
		self.invertMask = torch.ones((hidden, hidden)).type(torch.ByteTensor) - self.mask
		self.invertDiag = torch.ones((hidden, hidden)).type(torch.ByteTensor) - self.diagMask
		self.iteration = layers

		# For this case we want these to remain fixed
		self.hiddenWeight = nn.Parameter(torch.randn(hidden, hidden), requires_grad=False)
		self.bias = nn.Parameter(torch.ones(hidden, 1), requires_grad=True)
		self.inputWeight = nn.Parameter(torch.randn(D_input, hidden), requires_grad=False)


		self.tanh = nn.Tanh()
		self.scalar = nn.Parameter(torch.ones(1)*2, requires_grad=True)

		# Set up the hidden weights
		self.w = nn.Parameter(torch.randn(hidden, 1), requires_grad=True)
		#self.w.data[:] = 0.25
		#self.w.data[edgeMask] = 0.34
		#self.w.data[cornerMask] = 0.5
		self.hiddenWeight.data[self.invertMask] = 0
		self.hiddenWeight.data[self.mask] = 1


		# Set up the input weights
		self.a = nn.Parameter(torch.randn(hidden, 1), requires_grad=True)
		self.inputWeight.data[:] = 0
		self.inputWeight.data[self.diagMask] = 1
		#self.bias.data[:] = -0.15


		# self.hiddenWeight.weight.register_hook(self.backward_hook)
		# self.inputWeight.weight.register_hook(self.backward_hook_input)
		

	def forward(self, initial_hidden, input):
		u = initial_hidden.clone()
		u_fix = initial_hidden.clone()
		u_fix[u_fix==-1]=0
		u_fix_expand = u_fix.unsqueeze(-1)
		input_expand = input.unsqueeze(-1)
		for _ in range(0, self.iteration):
			u_expand = u.unsqueeze(-1)
			v_expand = u_fix_expand + torch.matmul((self.w.expand_as(self.hiddenWeight) * self.hiddenWeight), u_expand) + (self.bias) + \
				torch.matmul((self.a.expand_as(self.inputWeight) * self.inputWeight), input_expand)
			v = v_expand.squeeze()
			u = self.tanh(v * self.scalar.expand_as(v))
			#u = torch.sign(u)
		return u



# Code for the forward-engineered version of the network
# Nothing in the below is trainable, specify the weights for the propagation

class RepeatedLayersFixed(torch.nn.Module):
	def __init__(self, D_input, hidden, layers, weightMask, diagMask, edgeMask, cornerMask):
		super(RepeatedLayersFixed, self).__init__()

		self.mask = weightMask
		self.diagMask = diagMask
		self.invertMask = torch.ones((hidden, hidden)).type(torch.ByteTensor) - self.mask
		self.invertDiag = torch.ones((hidden, hidden)).type(torch.ByteTensor) - self.diagMask
		self.iteration = layers

		# For this case we want these to remain fixed
		self.hiddenWeight = nn.Parameter(torch.ones(hidden, hidden), requires_grad=False)
		self.bias = nn.Parameter(torch.ones(hidden, 1), requires_grad=True)
		self.inputWeight = nn.Parameter(torch.ones(D_input, hidden), requires_grad=False)


		self.tanh = nn.Tanh()
		self.scalar = nn.Parameter(torch.ones(1)*20, requires_grad=True)

		# Set up the hidden weights
		self.w = nn.Parameter(torch.ones(hidden, 1), requires_grad=True)
		self.w.data[:] = 0.25
		self.w.data[edgeMask] = 0.34
		self.w.data[cornerMask] = 0.5
		self.hiddenWeight.data[self.invertMask] = 0
		self.hiddenWeight.data[self.mask] = 1


		# Set up the input weights
		self.a = nn.Parameter(torch.ones(hidden, 1), requires_grad=True)
		self.inputWeight.data[:] = 0
		self.inputWeight.data[self.diagMask] = 1
		self.bias.data[:] = -0.15
		#self.hiddenWeight.bias.data[:] = -0.15

		# self.hiddenWeight.weight.register_hook(self.backward_hook)
		# self.inputWeight.weight.register_hook(self.backward_hook_input)
		

	def forward(self, initial_hidden, input):
		u = initial_hidden.clone()
		u_fix = initial_hidden.clone()
		u_fix[u_fix==-1]=0
		u_fix_expand = u_fix.unsqueeze(-1)
		input_expand = input.unsqueeze(-1)
		for _ in range(0, self.iteration):
			u_expand = u.unsqueeze(-1)
			v_expand = u_fix_expand + torch.matmul((self.w.expand_as(self.hiddenWeight) * self.hiddenWeight), u_expand) + (self.bias) + \
				torch.matmul((self.a.expand_as(self.inputWeight) * self.inputWeight), input_expand)
			v = v_expand.squeeze()
			u = self.tanh(v * self.scalar.expand_as(v))
			#u = torch.sign(u)
		return u



class FixedPropagationDecision(torch.nn.Module):
	def __init__(self, D_input, hidden, layers, weightMask, diagMask, edgeMask, cornerMask):
		super(FixedPropagationDecision, self).__init__()

		# We have two propagation layers to set up
		# 1 will be the prey
		# 2 will be the predator

		self.mask = weightMask
		self.diagMask = diagMask
		self.invertMask = torch.ones((hidden, hidden)).type(torch.ByteTensor) - self.mask
		self.invertDiag = torch.ones((hidden, hidden)).type(torch.ByteTensor) - self.diagMask
		self.iteration = layers
		self.tanh = nn.Tanh()
		self.sigmoid = nn.Sigmoid()



		# These are the prey parameters
		# For this case we want these to remain fixed
		self.hiddenWeight1 = nn.Parameter(torch.ones(hidden, hidden), requires_grad=False)
		self.bias1 = nn.Parameter(torch.ones(hidden, 1), requires_grad=True)
		self.inputWeight1 = nn.Parameter(torch.ones(D_input, hidden), requires_grad=False)
		self.scalar1 = nn.Parameter(torch.ones(1)*20, requires_grad=True)

		# Set up the hidden weights
		self.w1 = nn.Parameter(torch.ones(hidden, 1), requires_grad=True)
		self.w1.data[:] = 0.25
		self.w1.data[edgeMask] = 0.34
		self.w1.data[cornerMask] = 0.5
		self.hiddenWeight1.data[self.invertMask] = 0
		self.hiddenWeight1.data[self.mask] = 1


		# Set up the input weights
		self.a1 = nn.Parameter(torch.ones(hidden, 1), requires_grad=True)
		self.inputWeight1.data[:] = 0
		self.inputWeight1.data[self.diagMask] = 1
		self.bias1.data[:] = -0.15





		# These are the predator parameters
		self.hiddenWeight2 = nn.Parameter(torch.ones(hidden, hidden), requires_grad=False)
		self.bias2 = nn.Parameter(torch.ones(hidden, 1), requires_grad=True)
		self.inputWeight2 = nn.Parameter(torch.ones(D_input, hidden), requires_grad=False)
		self.scalar2 = nn.Parameter(torch.ones(1)*20, requires_grad=True)

		# Set up the hidden weights
		self.w2 = nn.Parameter(torch.ones(hidden, 1), requires_grad=True)
		self.w2.data[:] = 0.25
		self.w2.data[edgeMask] = 0.34
		self.w2.data[cornerMask] = 0.5
		self.hiddenWeight2.data[self.invertMask] = 0
		self.hiddenWeight2.data[self.mask] = 1


		# Set up the input weights
		self.a2 = nn.Parameter(torch.ones(hidden, 1), requires_grad=True)
		self.inputWeight2.data[:] = 0
		self.inputWeight2.data[self.diagMask] = 1
		self.bias2.data[:] = -0.15

		

	def forward(self, initial_hidden1, initial_hidden2, input1, input2, cave):
		u1 = initial_hidden1.clone()
		u1_fix = initial_hidden1.clone()
		u1_fix[u1_fix==-1]=0
		u1_fix_expand = u1_fix.unsqueeze(-1)
		input1_expand = input1.unsqueeze(-1)


		u2 = initial_hidden2.clone()
		u2_fix = initial_hidden2.clone()
		u2_fix[u2_fix==-1]=0
		u2_fix_expand = u2_fix.unsqueeze(-1)
		input2_expand = input2.unsqueeze(-1)

		batch = list(u1.size())
		batch = batch[0]
		

		decision = torch.zeros(batch, 2)
		cave[cave==-1] = 0
		idx = torch.nonzero(cave)

		

		for _ in range(0, self.iteration):
			u1_expand = u1.unsqueeze(-1)
			v1_expand = u1_fix_expand + torch.matmul((self.w1.expand_as(self.hiddenWeight1) * self.hiddenWeight1), u1_expand) + (self.bias1) + \
				torch.matmul((self.a1.expand_as(self.inputWeight1) * self.inputWeight1), input1_expand)
			v1 = v1_expand.squeeze()
			u1 = self.tanh(v1 * self.scalar1.expand_as(v1))
			#u = torch.sign(u)


			u2_expand = u2.unsqueeze(-1)
			v2_expand = u2_fix_expand + torch.matmul((self.w2.expand_as(self.hiddenWeight2) * self.hiddenWeight2), u2_expand) + (self.bias2) + \
				torch.matmul((self.a2.expand_as(self.inputWeight2) * self.inputWeight2), input2_expand)
			v2 = v2_expand.squeeze()
			u2 = self.tanh(v2 * self.scalar2.expand_as(v2))

			decision[:, 0] = self.sigmoid((-100.0*decision[:, 1] + 20*torch.sum(cave*u2, dim=1)))
			decision[:, 1] = self.sigmoid((-100.0*decision[:, 0] + 20*torch.sum(cave*u1, dim=1)))




			#print(torch.sum(cave*u1))
			# print(u2[idx])
			# print(u1[idx])
			# print(decision)

		return u1, u2, decision
