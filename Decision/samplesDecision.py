import numpy as np
import torch
import scipy.io as sio
#import matplotlib.pyplot as plt

from generate_environment import environment

#from matplotlib.ticker import NullLocator
import argparse

#### IMPORTANT: This code returns the propagated values

def generateSamples(N, numTraining, steps):


	# Parameters for environment generation
	N2 = N**2
	p = 0.15

	# Set up the tensors to store the
	trainEnv = np.zeros((numTraining, N2))
	trainPred = np.zeros((numTraining, N2))
	trainPrey = np.zeros((numTraining, N2))
	trainCave = np.zeros((numTraining, N2))
	trainLabel = np.zeros((numTraining, 2))


	runMax = np.ceil(numTraining*0.55)
	hideMax = np.ceil(numTraining*0.55)

	numRun = 0
	numHide = 0


	j = 0

	while(j < numTraining):
		X, prey, predator, cave = environment(N, p)

		# Label is 0 if cave is closer to prey (also equivalent to non-accessible to predator)
		# Label is 1 if cave is closer to predator OR non-accessible to either


		# Will add this to the environment code later
		# How to randomly choose nonzero: https://stackoverflow.com/questions/27414908/how-to-randomly-select-some-non-zero-elements-from-a-numpy-ndarray

		# All three of these will be N x N images with one non-zero element
		# These should be returned from the environment code



		###############################################################################
		# This is the code that  will become the separate labelling function



		###############################################################################

		# These form the two different stopping conditions
		# Diff tracks when we've filled the whole space
		# Stop tracks when one of the ranges includes the cave
		# Label get set initially to 1, the proper value if the cave is in neither range
		diff = 10
		stop = 0
		label = 1
		classLabel = np.array([0, 0])
		sampName = 'none'


########### Propagation round 1 to determine label

		# Set up prey range
		preyRange = np.zeros((N,N))

		row_prey, col_prey = np.nonzero(prey)
		l = len(row_prey)
		rowNext_prey = []
		colNext_prey = []

		for i in range(0, l):
			preyRange[row_prey[i], col_prey[i]] = 1
			rowNext_prey.append(row_prey[i])
			colNext_prey.append(col_prey[i])

		row_prey = []
		col_prey = []



		# Set up predator range
		predatorRange = np.zeros((N,N))

		row_predator, col_predator = np.nonzero(predator)
		l = len(row_predator)
		rowNext_predator = []
		colNext_predator = []

		for i in range(0, l):
			predatorRange[row_predator[i], col_predator[i]] = 1
			rowNext_predator.append(row_predator[i])
			colNext_predator.append(col_predator[i])

		row_predator = []
		col_predator = []


		# Save location of cave
		row_cave, col_cave = np.nonzero(cave)
		row_cave = row_cave[0]
		col_cave = col_cave[0]




		# Currently this is operating under the assumption that barriers are equal to 1

		q = 0

		while((diff != 0) and (stop == 0)):
			q = q+1
			# Propagate Prey
			preyRange_Old = preyRange[:,:]
			del row_prey[:]
			del col_prey[:]
			row_prey = rowNext_prey[:]
			col_prey = colNext_prey[:]
			l = len(row_prey)
			del rowNext_prey[:]
			del colNext_prey[:]
			for i in range(0, l):
				row_current = row_prey[i]
				col_current = col_prey[i]
				if ((row_current != 0) and (X[(row_current - 1), col_current] == 0) and (preyRange[(row_current - 1), col_current] == 0)):
					preyRange[row_current - 1, col_current] = 1
					rowNext_prey.append(row_current - 1)
					colNext_prey.append(col_current)
				if ((row_current != N-1) and (X[row_current + 1, col_current] == 0) and (preyRange[(row_current + 1), col_current] == 0)):
					preyRange[row_current + 1, col_current] = 1
					rowNext_prey.append(row_current + 1)
					colNext_prey.append(col_current)
				if ((col_current != 0) and (X[row_current, col_current-1] == 0) and (preyRange[row_current, col_current-1] == 0)):
					preyRange[row_current, col_current-1] = 1
					rowNext_prey.append(row_current)
					colNext_prey.append(col_current-1)
				if ((col_current != N-1) and (X[row_current, col_current+1] == 0) and (preyRange[row_current, col_current+1] == 0)):
					preyRange[row_current, col_current+1] = 1
					rowNext_prey.append(row_current)
					colNext_prey.append(col_current+1)


			# Propagate predator
			predatorRange_Old = predatorRange[:,:]
			del row_predator[:]
			del col_predator[:]
			row_predator = rowNext_predator[:]
			col_predator = colNext_predator[:]
			l = len(row_predator)
			del rowNext_predator[:]
			del colNext_predator[:]
			for i in range(0, l):
				row_current = row_predator[i]
				col_current = col_predator[i]
				if ((row_current != 0) and (X[(row_current - 1), col_current] == 0) and (predatorRange[(row_current - 1), col_current] == 0)):
					predatorRange[row_current - 1, col_current] = 1
					rowNext_predator.append(row_current - 1)
					colNext_predator.append(col_current)
				if ((row_current != N-1) and (X[row_current + 1, col_current] == 0) and (predatorRange[(row_current + 1), col_current] == 0)):
					predatorRange[row_current + 1, col_current] = 1
					rowNext_predator.append(row_current + 1)
					colNext_predator.append(col_current)
				if ((col_current != 0) and (X[row_current, col_current-1] == 0) and (predatorRange[row_current, col_current-1] == 0)):
					predatorRange[row_current, col_current-1] = 1
					rowNext_predator.append(row_current)
					colNext_predator.append(col_current-1)
				if ((col_current != N-1) and (X[row_current, col_current+1] == 0) and (predatorRange[row_current, col_current+1] == 0)):
					predatorRange[row_current, col_current+1] = 1
					rowNext_predator.append(row_current)
					colNext_predator.append(col_current+1)


			# Check termination condition
			# It is important to check predator condition first!  If both reach at same time, prey should run
			if (predatorRange[row_cave, col_cave] == 1):
				label = 1
				stop = 1
				classLabel = np.array([1, 0])
				sampName = 'hide'
			elif (preyRange[row_cave, col_cave] == 1):
				label = -1
				stop = 1
				classLabel = np.array([0, 1])
				sampName = 'run'

			diff = len(rowNext_prey) + len(rowNext_predator)

		#print(q)

		# Change everything to +1/-1
		if (sampName == 'hide' and numHide < hideMax) or (sampName == 'run' and numRun < runMax):
			Xvec = np.reshape(X, (1, N2))
			Xvec = 1 - Xvec
			Xvec[Xvec == 0] = -1
			predVec = np.reshape(predator, (1, N2))
			predVec[predVec == 0] = -1
			preyVec = np.reshape(prey, (1, N2))
			preyVec[preyVec == 0] = -1
			caveVec = np.reshape(cave, (1, N2))
			caveVec[caveVec == 0] = -1


			trainEnv[j, :] = Xvec
			trainPred[j, :] = predVec
			trainPrey[j, :] = preyVec
			trainCave[j, :] = caveVec
			trainLabel[j, :] = classLabel


			if (sampName == 'hide'):
				numHide = numHide + 1
			elif (sampName == 'run'):
				numRun = numRun + 1

			j = j+1
			# if (j % 1000 == 0):
			# 	print(j)

	# Once all the samples are generated, return dictionary of samples

	trainEnv = torch.from_numpy(trainEnv)
	trainPred = torch.from_numpy(trainPred)
	trainPrey = torch.from_numpy(trainPrey)
	trainCave = torch.from_numpy(trainCave)
	trainLabel = torch.from_numpy(trainLabel)

	sampleDict = {"Environment": trainEnv, "Predator": trainPred, "Prey": trainPrey, "Cave": trainCave, "Label": trainLabel}

	return sampleDict


	# # Look at the output
	# fig1, ax = plt.subplots(2, 3)
	# ax[0, 0].imshow(X, cmap='Greys',  interpolation='none')
	# ax[0, 1].imshow(prey, cmap='Greys',  interpolation='none')
	# ax[0, 2].imshow(predator, cmap='Greys',  interpolation='none')
	# ax[1, 0].imshow(cave, cmap='Greys',  interpolation='none')
	# ax[1, 1].imshow(preyRange, cmap='Greys',  interpolation='none')
	# ax[1, 2].imshow(predatorRange, cmap='Greys',  interpolation='none')

	# print(label)

	# plt.show()

	# fig1.savefig('sample_label.pdf', bbox_inches = 'tight')

	
		










