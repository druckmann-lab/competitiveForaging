import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import collections
import numpy as np
from torch.autograd import Variable

## Netowrk Types ##

def generateSquareWeightMask(imageSize, boundarySize):
	##################################################################
	# Function GENERATE_SQUARE_WEIGHT_MASK
	# Takes in a nested dictionary of model/layer types
	# Outputs a nested dictionary of optimized hyperparameters
	# Parameters:
	# 		* imageSize: Size of one side of the image
	#       * boundarySize: Size of square grid to generate
	# Outputs
	#		* weightMask: Input for Repeated Layers Masked

	# 
	##################################################################

	numPixels = imageSize**2

	weightMask = np.zeros((numPixels, numPixels))
	diagMask = np.zeros((numPixels, numPixels))

	for k in range(0, numPixels):
		i,j = ind2subImage(k, imageSize)
		pixelMask = np.zeros((imageSize, imageSize))

		if((i > 0) and (i < imageSize - 1) and (j > 0) and (j < imageSize - 1)):
			row_min = np.max((0, i - boundarySize))
			row_max = np.min((imageSize, i + boundarySize + 1))
			col_min = np.max((0, j - boundarySize))
			col_max = np.min((imageSize, j + boundarySize + 1))

			pixelMask[row_min:row_max, col_min:col_max] = 1

		pixelMask[i,j] = 0
		diagMask[k, k] = 1


		weightMask[k, :] = np.reshape(pixelMask, (1, numPixels))
	weightMask = torch.from_numpy(weightMask).type(torch.cuda.ByteTensor)
	diagMask = torch.from_numpy(diagMask).type(torch.cuda.ByteTensor)

	return weightMask, diagMask



def generateGridWeightMask(imageSize):
	numPixels = imageSize**2

	weightMask = np.zeros((numPixels, numPixels))
	diagMask = np.zeros((numPixels, numPixels))

	for k in range(0, numPixels):
		i,j = ind2subImage(k, imageSize)
		pixelMask = np.zeros((imageSize, imageSize))

		if((i > 0) and (i < imageSize - 1) and (j > 0) and (j < imageSize - 1)):
			#pixelMask[i, j] = 1
			pixelMask[i - 1, j] = 1
			pixelMask[i + 1, j] = 1
			pixelMask[i, j + 1] = 1
			pixelMask[i, j - 1] = 1

		diagMask[k, k] = 1

		# if (i > 0):
		# 	pixelMask[i - 1, j] = 1
		# if (i < imageSize - 1):
		# 	pixelMask[i + 1, j] = 1
		# if (j > 0):
		# 	pixelMask[i, j - 1] = 1
		# if (j < imageSize - 1):
		# 	pixelMask[i, j + 1] = 1


		weightMask[k, :] = np.reshape(pixelMask, (1, numPixels))
		#diagMask[k, k] = 1
	weightMask = torch.from_numpy(weightMask).type(torch.cuda.BoolTensor)
	diagMask = torch.from_numpy(diagMask).type(torch.cuda.BoolTensor)


	return weightMask, diagMask


def generateGridWeightMask_PredPrey(imageSize):
	numPixels = imageSize**2

	weightMask = np.zeros((numPixels, numPixels))
	diagMask = np.zeros((numPixels, numPixels))

	for k in range(0, numPixels):
		i,j = ind2subImage(k, imageSize)
		pixelMask = np.zeros((imageSize, imageSize))

		diagMask[k, k] = 1

		if (i > 0):
			pixelMask[i - 1, j] = 1
		if (i < imageSize - 1):
			pixelMask[i + 1, j] = 1
		if (j > 0):
			pixelMask[i, j - 1] = 1
		if (j < imageSize - 1):
			pixelMask[i, j + 1] = 1


		weightMask[k, :] = np.reshape(pixelMask, (1, numPixels))
		#diagMask[k, k] = 1
	weightMask = torch.from_numpy(weightMask).type(torch.cuda.BoolTensor)
	diagMask = torch.from_numpy(diagMask).type(torch.cuda.BoolTensor)


	return weightMask, diagMask


# Important: This diagMask does not include corners/edges as these are passed back as separate masks
def generateFixedWeightMask_PredPrey(imageSize):
	numPixels = imageSize**2

	weightMask = np.zeros((numPixels, numPixels))
	diagMask = np.zeros((numPixels, numPixels))
	cornerMask = np.zeros((numPixels))
	edgeMask = np.zeros((numPixels))

	for k in range(0, numPixels):
		i,j = ind2subImage(k, imageSize)
		pixelMask = np.zeros((imageSize, imageSize))

		# Determine if pixel is edge or corner

		if ((i == 0 or i == imageSize - 1) and (j == 0 or j == imageSize - 1)):
			cornerMask[k] = 1
		elif ((i == 0 or i == imageSize - 1) or (j == 0 or j == imageSize - 1)):
			edgeMask[k] = 1
		
		diagMask[k, k] = 1

		# Make mask of all the neighbors
		if (i > 0):
			pixelMask[i - 1, j] = 1
		if (i < imageSize - 1):
			pixelMask[i + 1, j] = 1
		if (j > 0):
			pixelMask[i, j - 1] = 1
		if (j < imageSize - 1):
			pixelMask[i, j + 1] = 1


		weightMask[k, :] = np.reshape(pixelMask, (1, numPixels))

	weightMask = torch.from_numpy(weightMask).type(torch.BoolTensor)
	diagMask = torch.from_numpy(diagMask).type(torch.BoolTensor)
	cornerMask = torch.from_numpy(cornerMask).type(torch.BoolTensor)
	edgeMask = torch.from_numpy(edgeMask).type(torch.BoolTensor)


	return weightMask, diagMask, edgeMask, cornerMask


def ind2subImage(idx, imageSize):
	i = int(np.floor(idx/imageSize))
	j = idx % imageSize
	return i, j