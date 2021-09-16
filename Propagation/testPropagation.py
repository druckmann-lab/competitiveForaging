import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable 
from torch.utils.data import DataLoader
import matplotlib as mpl
from matplotlib.colors import colorConverter

import networkFiles as NF
from samplesPropagation import generateSamples


# General parameters that would get set in other code
layers = 5
image_size = 20
N = image_size
num_nodes = image_size**2

dtype = torch.FloatTensor

model = NF.PropagationOnly_FixedWeights(num_nodes, layers, num_nodes*5, image_size)
model.type(dtype)
testDict = generateSamples(image_size, 2, layers)


test_dsetPath = torch.utils.data.TensorDataset(testDict["Environment"], testDict["Predator"], testDict["Range"])
loader = DataLoader(test_dsetPath, batch_size=32, shuffle=True)

loss_fn = nn.MSELoss()

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



 

# Return the fraction of datapoints that were incorrectly classified.
errorAll = 1.0 -  (float(num_correct) / (num_samples))
avg_loss = sum(losses)/float(len(losses))

print(errorAll)


# Look at the output
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


# # Look at the output
fig1, ax = plt.subplots(1, 2)

#ax[0, 0].imshow(np.reshape(output[0,:].detach().numpy(), (N, N)), cmap='Greys',  interpolation='none')
ax[1].imshow(-1*np.reshape(env[0,:].numpy(), (N, N)), interpolation='none', cmap=cmap1, origin='lower')
ax[1].imshow(np.reshape(output[0,:].detach().numpy(), (N, N)), interpolation='none', cmap=cmap2, origin='lower')
ax[1].imshow(np.reshape(pred[0,:].detach().numpy(), (N, N)), interpolation='none', cmap=cmap3, origin='lower')

#fig1.colorbar(im)
# # ax[0, 2].imshow(predator, cmap='Greys',  interpolation='none')
# # ax[1, 0].imshow(cave, cmap='Greys',  interpolation='none')
# # ax[1, 1].imshow(preyRange, cmap='Greys',  interpolation='none')
# # ax[1, 2].imshow(predatorRange, cmap='Greys',  interpolation='none')

plt.show()

#fig1.savefig('sample_label.pdf', bbox_inches = 'tight')




