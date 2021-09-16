import numpy as np
import torch
import torch.utils.data
#import matplotlib.pyplot as plt

# Sub-functions
def addRectangle(X, N):

	i = np.random.randint(0, N-3)
	j = np.random.randint(0, N-3)

	width = np.random.randint(2, 8)
	height = np.random.randint(2,8)


	row_min = np.max((0, i))
	row_max = np.min((N - 1, i + width))
	col_min = np.max((0, j))
	col_max = np.min((N - 1, j + height))

	X[row_min:row_max, col_min:col_max] = 1


def randExpand(i, j, rowNext, colNext, p, X, N):

	if ((i != 0) and (i != N-1) and (j != 0) and (j != N-1)):
		expand = np.random.binomial(1, p, (8))

		if ((X[i-1, j-1] == 0) and (expand[0] == 1)):
			X[i-1, j-1] = 1
			rowNext.append(i - 1)
			colNext.append(j - 1)
		if ((X[i-1, j] == 0) and (expand[1] == 1)):
			X[i-1, j] = 1
			rowNext.append(i - 1)
			colNext.append(j)
		if ((X[i-1, j+1] == 0) and (expand[2] == 1)):
			X[i-1, j+1] = 1
			rowNext.append(i - 1)
			colNext.append(j+1)
		if ((X[i, j+1] == 0) and (expand[3] == 1)):
			X[i, j+1] = 1
			rowNext.append(i)
			colNext.append(j+1)
		if ((X[i+1, j+1] == 0) and (expand[4] == 1)):
			X[i+1, j+1] = 1
			rowNext.append(i+1)
			colNext.append(j+1)
		if ((X[i+1, j] == 0) and (expand[5] == 1)):
			X[i+1, j] = 1
			rowNext.append(i+1)
			colNext.append(j)
		if ((X[i+1, j-1] == 0) and (expand[6] == 1)):
			X[i+1, j-1] = 1
			rowNext.append(i+1)
			colNext.append(j-1)
		if ((X[i, j-1] == 0) and (expand[6] == 1)):
			X[i, j-1] = 1
			rowNext.append(i)
			colNext.append(j-1)






def environment(N, p):
	# These are variable that will eventually be passed into the function
	#N = 20

	# Start of function
	N2 = N**2
	#p = 0.15

	# Random seed a bunch of rectangles
	nrectangles = np.random.randint(2, 5)
	X = np.zeros((N, N))

	for i in range(nrectangles):
		addRectangle(X, N)

	rowNonzero, colNonzero = np.nonzero(X)
	rowNext = []
	colNext = []

	for i in range(len(rowNonzero)):
		rowNext.append(rowNonzero[i])
		colNext.append(colNonzero[i])


	row = []
	col = []
	diff = 10

	while((diff != 0)):
		del row[:]
		del col[:]
		row = rowNext[:]
		col = colNext[:]
		l = len(row)
		del rowNext[:]
		del colNext[:]

				
		for i in range(0, l):
			row_current = row[i]
			col_current = col[i]

			randExpand(row_current, col_current, rowNext, colNext, p, X, N)


		diff = len(rowNext)

	# Now we want to random place predator, prey and cave
	# Need a list of zeros
	row, col = np.nonzero(X)
	X_invert = np.ones((N, N))
	X_invert[row, col] = 0
	row_zero, col_zero = np.nonzero(X_invert)

	prey = np.zeros((N, N))
	predator = np.zeros((N, N))
	cave = np.zeros((N, N))

	idx_prey = np.random.choice(len(row_zero), 1, replace = False)
	idx_predator = np.random.choice(len(row_zero), 3, replace = False)
	idx_cave = np.random.choice(len(row_zero), 1, replace = False)

	prey[row_zero[idx_prey], col_zero[idx_prey]] = 1
	predator[row_zero[idx_predator], col_zero[idx_predator]] = 1
	cave[row_zero[idx_cave], col_zero[idx_cave]] = 1


	return X, prey, predator, cave



# plt.imshow(X, cmap='RdBu',  interpolation='none')
# plt.show()


# Two functions for generating the environment

