import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable 
from torch.utils.data import DataLoader
import matplotlib as mpl
from matplotlib.colors import colorConverter

import networkFiles as NF
from samplesDecision import generateSamples
from generateDictionary import loadStateDict


# General parameters that would get set in other code
layers = 10
image_size = 15
N = image_size
num_nodes = image_size**2
r = 3

dtype = torch.FloatTensor


### First model is the Correct Decision layer
model = NF.FixedAll_PredPrey(num_nodes, layers, num_nodes*5, image_size)
model.type(dtype)

testDict = generateSamples(image_size, r, layers)


modelBlock_State = torch.load('modelBlock_fixedProp_15Jun20_block1.pth.tar', map_location=torch.device('cpu'))
modelBlock = loadStateDict(modelBlock_State)
modelTrained = modelBlock[4]['Model']


# # for key, val in modelBlock.items():
# # 	if key != 'Meta':
# print(modelBlock[4]['Acc_All'])


# ### Second model uses weights from trained network
# modelTrained = NF.FixedDecision_PredPrey(num_nodes, layers, num_nodes*5, image_size)

# modelBlock_State = torch.load('modelBlock_prop12.pth.tar', map_location=torch.device('cpu'))
# modelBlock = loadStateDict(modelBlock_State)
# modelProp = modelBlock[modelBlock['Meta']['BestModel']]["Model"]

# # Extract w, b, a
# w = modelProp.iteratedLayer_Pred.w.data
# a = modelProp.iteratedLayer_Pred.a.data
# b = modelProp.iteratedLayer_Pred.bias.data

# # Set model paramters
# modelTrained.iteratedLayer.w1.data = w
# modelTrained.iteratedLayer.w2.data = w
# modelTrained.iteratedLayer.a1.data = a
# modelTrained.iteratedLayer.a2.data = a
# modelTrained.iteratedLayer.bias1.data = b
# modelTrained.iteratedLayer.bias2.data = b




test_dsetPath = torch.utils.data.TensorDataset(testDict["Environment"], testDict["Predator"], testDict["Prey"], testDict["Cave"], testDict["Label"])
loader = DataLoader(test_dsetPath, batch_size=32, shuffle=False)

loss_fn = nn.BCELoss()

model.eval()
#modelTrained.eval()
num_correct, num_samples = 0, 0
losses = []

# The accuracy on all pixels and path pixels can be calculated from the image labels
# Also record the loss


# print(output)
# print(label)
# print(loss)

# errorAll = 1.0 -  (float(num_correct) / (num_samples))
# print(errorAll)
# print(num_samples)



cmap1 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',['white','black'],256)
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',['white','skyblue'],256)
cmap3 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',['white','blue'],256)
cmap4 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',['white','pink'],256)
cmap5 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',['white','mediumvioletred'],256)
cmap6 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',['white','goldenrod'],256)

cmap2._init() # create the _lut array, with rgba values
cmap3._init()
cmap4._init()
cmap5._init()
cmap6._init()


# create your alpha array and fill the colormap with them.
# here it is progressive, but you can create whathever you want
alphas = np.linspace(0, 0.8, cmap2.N+3)
cmap2._lut[:,-1] = alphas
cmap3._lut[:,-1] = alphas
cmap4._lut[:,-1] = alphas
cmap5._lut[:,-1] = alphas
cmap6._lut[:,-1] = alphas


# # Look at the output
fig1, ax = plt.subplots(1, 4, figsize = (10, 3))

for q in range(2):

	for env, pred, prey, cave, label in loader:
		# Cast the image data to the correct type and wrap it in a Variable. At
		# test-time when we do not need to compute gradients, marking the Variable
		# as volatile can reduce memory usage and slightly improve speed.
		env = Variable(env.type(dtype), requires_grad=False)
		pred = Variable(pred.type(dtype), requires_grad=False)
		prey = Variable(prey.type(dtype), requires_grad=False)
		cave = Variable(cave.type(dtype), requires_grad=False)
		label = Variable(label.type(dtype), requires_grad=False)

		# Run the model forward and compare with ground truth.
		if q == 0:
			prey_range, pred_range, output, trace = model(env, prey, pred, cave, dtype)
		else:
			prey_range, pred_range, output, trace = modelTrained(env, prey, pred, cave, dtype)
			trace = trace[:,:,0:layers-1]


	if q == 0:
		ax[0].imshow(-1*np.reshape(env[0,:].numpy(), (N, N)), cmap=cmap1, origin='lower')
		ax[0].imshow(np.reshape(pred_range[0,:].detach().numpy(), (N, N)), cmap=cmap2, origin='lower')
		ax[0].imshow(np.reshape(pred[0,:].detach().numpy(), (N, N)), cmap=cmap3, origin='lower')
		ax[0].imshow(np.reshape(cave[0,:].detach().numpy(), (N, N)), cmap=cmap6, origin='lower')
		
		ax[0].tick_params(
		    axis='y',          # changes apply to the x-axis
		    which='both',      # both major and minor ticks are affected
		    left=False,      # ticks along the bottom edge are off
		    right=False,         # ticks along the top edge are off
		    labelleft=False) # labels along the bottom edge are off

		ax[0].tick_params(
		    axis='x',          # changes apply to the x-axis
		    which='both',      # both major and minor ticks are affected
		    bottom=False,      # ticks along the bottom edge are off
		    top=False,         # ticks along the top edge are off
		    labelbottom=False) # labels along the bottom edge are off


		ax[1].imshow(-1*np.reshape(env[0,:].numpy(), (N, N)), cmap=cmap1, origin='lower')
		ax[1].imshow(np.reshape(prey_range[0,:].detach().numpy(), (N, N)), cmap=cmap4, origin='lower')
		ax[1].imshow(np.reshape(prey[0,:].detach().numpy(), (N, N)), cmap=cmap5, origin='lower')
		ax[1].imshow(np.reshape(cave[0,:].detach().numpy(), (N, N)), cmap=cmap6, origin='lower') 
		
		ax[1].tick_params(
		    axis='y',          # changes apply to the x-axis
		    which='both',      # both major and minor ticks are affected
		    left=False,      # ticks along the bottom edge are off
		    right=False,         # ticks along the top edge are off
		    labelleft=False) # labels along the bottom edge are off

		ax[1].tick_params(
		    axis='x',          # changes apply to the x-axis
		    which='both',      # both major and minor ticks are affected
		    bottom=False,      # ticks along the bottom edge are off
		    top=False,         # ticks along the top edge are off
		    labelbottom=False) # labels along the bottom edge are off


	if q == 0 :
		ax[2].step(np.arange(0, layers+1), np.insert(1.5 + np.round(trace[0, 0, :].detach().numpy()), 0, 1.5))
		ax[2].step(np.arange(0, layers+1), np.insert(np.round(trace[0, 1, :].detach().numpy()), 0, 0), color='mediumvioletred')
		ax[2].set_ylim((-0.2, 2.7))
		ax[2].set_xlabel('Time Steps')

		ax[2].tick_params(
		    axis='y',          # changes apply to the x-axis
		    which='both',      # both major and minor ticks are affected
		    left=False,      # ticks along the bottom edge are off
		    right=False,         # ticks along the top edge are off
		    labelleft=False) # labels along the bottom edge are off

		major_ticks = np.arange(0, layers, 5)
		minor_ticks = np.arange(0, layers, 1)

		ax[2].set_xticks(major_ticks)
		ax[2].set_xticks(minor_ticks, minor=True)


		# ax[q, 2].tick_params(axis='x',which='minor',bottom=True)


		ax[2].spines['top'].set_visible(False)
		ax[2].spines['right'].set_visible(False)
		ax[2].spines['bottom'].set_visible(False)
		ax[2].spines['left'].set_visible(False)
	
	if q == 1:
		ax[3].step(np.arange(0, layers+1), np.insert(1.5 + np.concatenate(([0],trace[0, 0, :].detach().numpy())), 0, 1.5))
		ax[3].step(np.arange(0, layers+1), np.insert(np.concatenate(([0], trace[0, 1, :].detach().numpy())), 0, 0), color='mediumvioletred')
		ax[3].set_ylim((-0.2, 2.7))
		ax[3].set_xlabel('Time Steps')

		ax[3].tick_params(
		    axis='y',          # changes apply to the x-axis
		    which='both',      # both major and minor ticks are affected
		    left=False,      # ticks along the bottom edge are off
		    right=False,         # ticks along the top edge are off
		    labelleft=False) # labels along the bottom edge are off

		major_ticks = np.arange(0, layers, 5)
		minor_ticks = np.arange(0, layers, 1)

		ax[3].set_xticks(major_ticks)
		ax[3].set_xticks(minor_ticks, minor=True)


		# ax[q, 2].tick_params(axis='x',which='minor',bottom=True)


		ax[3].spines['top'].set_visible(False)
		ax[3].spines['right'].set_visible(False)
		ax[3].spines['bottom'].set_visible(False)
		ax[3].spines['left'].set_visible(False)

	#ax[q, 2].axis('off')




fig1.savefig("decision.pdf", bbox_inches = 'tight',
		pad_inches = 0)
