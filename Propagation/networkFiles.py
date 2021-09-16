import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import collections
from torch.autograd import Variable
from weightMask import generateFixedWeightMask_PredPrey

#### These are the top level network files for the propagation task

class PropagationOnly_SharedPixel(torch.nn.Module):
	def __init__(self, num_pixels, layers, H_decision, imageSize):
		super(PropagationOnly_SharedPixel, self).__init__()
		weightMask, diagMask, edgeMask, cornerMask = generateFixedWeightMask_PredPrey(imageSize)
		self.iteratedLayer_Pred = RangePropgation_SharedPixel(num_pixels, num_pixels, layers, weightMask, diagMask, edgeMask, cornerMask)
		

	def forward(self, X, pred, dtype):	
		
		pred_range = self.iteratedLayer_Pred(pred, X)

		# if (pred_range.dim() == 1):
		# 	pred_range.unsqueeze(0)
		
		return pred_range


class PropagationOnly_FixedWeights(torch.nn.Module):
	def __init__(self, num_pixels, layers, H_decision, imageSize):
		super(PropagationOnly_FixedWeights, self).__init__()
		weightMask, diagMask, edgeMask, cornerMask = generateFixedWeightMask_PredPrey(imageSize)
		self.iteratedLayer_Pred = RangePropgation_FixedWeights(num_pixels, num_pixels, layers, weightMask, diagMask, edgeMask, cornerMask)
		

	def forward(self, X, pred, dtype):	
		
		pred_range = self.iteratedLayer_Pred(pred, X)
		
		return pred_range


#### These are sub-functions for the propagation

class RangePropgation_SharedPixel(torch.nn.Module):
	# In this model, each pixel only has one weight w
	def __init__(self, D_input, hidden, layers, weightMask, diagMask, edgeMask, cornerMask):
		super(RangePropgation_SharedPixel, self).__init__()

		self.mask = weightMask
		self.diagMask = diagMask
		self.invertMask = ~self.mask #torch.ones((hidden, hidden)).type(torch.ByteTensor) - self.mask
		self.invertDiag = ~self.diagMask #torch.ones((hidden, hidden)).type(torch.ByteTensor) - self.diagMask
		self.iteration = layers

		# For this case we want these to remain fixed
		self.hiddenWeight = nn.Parameter(torch.ones(hidden, hidden), requires_grad=False)
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
		return u



class RangePropgation_FixedWeights(torch.nn.Module):
	# Nothing in this model is actually trainable

	def __init__(self, D_input, hidden, layers, weightMask, diagMask, edgeMask, cornerMask):
		super(RangePropgation_FixedWeights, self).__init__()

		self.mask = weightMask
		self.diagMask = diagMask
		self.invertMask = ~self.mask #torch.ones((hidden, hidden)).type(torch.ByteTensor) - self.mask
		self.invertDiag = ~self.diagMask #torch.ones((hidden, hidden)).type(torch.ByteTensor) - self.diagMask
		self.iteration = layers

		# For this case we want these to remain fixed
		self.hiddenWeight = nn.Parameter(torch.ones(hidden, hidden), requires_grad=False)
		self.bias = nn.Parameter(torch.ones(hidden, 1), requires_grad=False)
		self.inputWeight = nn.Parameter(torch.ones(D_input, hidden), requires_grad=False)


		self.tanh = nn.Tanh()
		self.scalar = nn.Parameter(torch.ones(1)*20, requires_grad=False)

		# Set up the hidden weights
		self.w = nn.Parameter(torch.ones(hidden, 1), requires_grad=False)
		self.w.data[:] = 0.25
		self.w.data[edgeMask] = 0.34
		self.w.data[cornerMask] = 0.5
		self.hiddenWeight.data[self.invertMask] = 0
		self.hiddenWeight.data[self.mask] = 1


		# Set up the input weights
		self.a = nn.Parameter(torch.ones(hidden, 1), requires_grad=False)
		self.inputWeight.data[:] = 0
		self.inputWeight.data[self.diagMask] = 1
		self.bias.data[:] = -0.15


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









