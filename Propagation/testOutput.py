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
from weightMask import generateFixedWeightMask_PredPrey

# Model 4 is good in prop5, prop7, and prop10


model_path = '../../PredPrey_Results/Propagation/ResultBlock/'


model_name = ['modelBlock_prop5_09Mar20.pth.tar', 'modelBlock_prop7_09Mar20.pth.tar', 'modelBlock_prop10_09Mar20.pth.tar']
layers_list = [5, 7, 10]

# # Look at the output
fig1, ax = plt.subplots(3, 3)

# This section is to test setting the colorbar setup
cmap1 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',['white','black'],256)
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',['white','skyblue'],256)
cmap3 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',['white','blue'],256)

cmap2._init() # create the _lut array, with rgba values
cmap3._init()

# create your alpha array and fill the colormap with them.
# here it is progressive, but you can create whathever you want
alphas = np.linspace(0, 0.8, cmap2.N+3)
cmap2._lut[:,-1] = alphas
cmap3._lut[:,-1] = alphas

# General parameters that would get set in other code

image_size = 15
N = image_size
num_nodes = image_size**2

weightMask, diagMask, edgeMask, cornerMask = generateFixedWeightMask_PredPrey(image_size)

centerMask = ~(edgeMask + cornerMask)

dtype = torch.FloatTensor

for i in range(3):

	modelBlock_State = torch.load(model_path + model_name[i], map_location=torch.device('cpu'))
	modelBlock = loadStateDict(modelBlock_State)

	layers = layers_list[i]
	testDict = generateSamples(image_size, 2, layers)


	test_dsetPath = torch.utils.data.TensorDataset(testDict["Environment"], testDict["Predator"], testDict["Range"])
	loader = DataLoader(test_dsetPath, batch_size=32, shuffle=True)

	loss_fn = nn.MSELoss()

	#The good model is 4
	model = modelBlock[4]["Model"]
	model.eval()
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


	w = model.iteratedLayer_Pred.w.data[centerMask]
	a = model.iteratedLayer_Pred.a.data[centerMask]
	b = model.iteratedLayer_Pred.bias.data[centerMask]








	#ax[0, 0].imshow(np.reshape(output[0,:].detach().numpy(), (N, N)), cmap='Greys',  interpolation='none')
	ax[i, 0].imshow(-1*np.reshape(env[0,:].numpy(), (N, N)), cmap=cmap1, origin='lower')
	ax[i, 0].imshow(np.reshape(output[0,:].detach().numpy(), (N, N)), cmap=cmap2, origin='lower')
	ax[i, 0].imshow(np.reshape(pred[0,:].detach().numpy(), (N, N)), cmap=cmap3, origin='lower')

	ax[i, 0].tick_params(
	    axis='y',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    left=False,      # ticks along the bottom edge are off
	    right=False,         # ticks along the top edge are off
	    labelleft=False) # labels along the bottom edge are off

	ax[i, 0].tick_params(
	    axis='x',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    bottom=False,      # ticks along the bottom edge are off
	    top=False,         # ticks along the top edge are off
	    labelbottom=False) # labels along the bottom edge are off

	ax[i, 1].imshow(-1*np.reshape(env[1,:].numpy(), (N, N)), cmap=cmap1, origin='lower')
	ax[i, 1].imshow(np.reshape(output[1,:].detach().numpy(), (N, N)), cmap=cmap2, origin='lower')
	ax[i, 1].imshow(np.reshape(pred[1,:].detach().numpy(), (N, N)), cmap=cmap3, origin='lower')

	ax[i, 1].tick_params(
	    axis='y',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    left=False,      # ticks along the bottom edge are off
	    right=False,         # ticks along the top edge are off
	    labelleft=False) # labels along the bottom edge are off

	ax[i, 1].tick_params(
	    axis='x',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    bottom=False,      # ticks along the bottom edge are off
	    top=False,         # ticks along the top edge are off
	    labelbottom=False) # labels along the bottom edge are off

	xpoints = np.linspace(0, 0.6, 100)
	line1 = 2.0*xpoints - 1.0
	line2 = 4.0*xpoints - 1.0
	line3 = -4.0*xpoints + 1


	ax[i, 2].plot(xpoints, line1, '--', color = 'grey')
	ax[i, 2].plot(xpoints, line2, '--', color = 'grey')
	ax[i, 2].plot(xpoints, line3, '--', color = 'grey')
	ax[i, 2].fill_between(xpoints, np.minimum(line1, line3), np.minimum(line2, line3), color='grey', alpha='0.5')
	ax[i, 2].scatter(w/a, b/a, facecolor='skyblue', edgecolor='k', s=15)
	ax[i, 2].set_xlim(0, 0.6)
	ax[i, 2].set_ylim(-0.5, 0.5)
	ax[i, 2].grid(which='major', linestyle='--', linewidth='0.5')



fig1.savefig("propagation1.pdf", bbox_inches = 'tight',
		pad_inches = 0)






