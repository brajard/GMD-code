import os
import numpy as np
from dapper.mods import Chronology
from dapper.da_methods import EnKF_N

from utils import setup as setup_lorenz
from utils import simulate_ens, NNPredictor, SetupBuilder, plot_L96_2D

#################
# General Setup #
#################
m = 40  # size of the state space
p = 20  # Number of obs at each time step (50%)
std_m = 0.1  # standard deviation of model noise
std_o = 1.  # standard devation of observational noise

ncycle = 40  # Number of cycles
nepochs_init = 40 # Number of epochs for initializing the weights
nepochs = 20 # Number of epochs during training in a cycle
Texpe = 2000 # Length of the experiment in model time unit

# To reduce the time needed to run the experiment (with less accurate results) uncomment the following lines:
# ncycle = 2
# nepochs_init = 10
# nepochs = 5
# Texpe = 500

datadir = 'example_data'  # Directory where to save the results
######################
# Initial simulation #
######################

# Init simulation:
xx_spinup = simulate_ens(setup_lorenz)

# initial state (discard the first 100 time steps as spinup):
xx_init = xx_spinup[100:]

##################
# reference run  #
##################

# setup builder of the run:
sb = SetupBuilder(t=Chronology(0.05, dkObs=1, T=Texpe, BurnIn=1),
	p=p,
	std_o=std_o,
	data=xx_init)

# Define the setup run for the true simulation:
setup_true = sb.setup()
xtrue, yobs = setup_true.simulate()
# NB: the config can be saved using sb.save(...)

###########################
# Data assimilation setup #
###########################
N = 30  # Size of the ensemble
config = EnKF_N(N=N)  # DA config

##########################
# Machine learning setup #
##########################
# Parameters of the neural network (architecture + training):
param_nn = {'archi': ((24, 5, 'relu', 0.0), (37, 5, 'relu', 0.0)),  # CNN layer setup
	'bilin': True,  # activate bilinear layer
	'batchnorm': True,  # activate batchnorm normalization
	'reg': ('ridge', 1e-4),  # L2 regularization for the last layer
	'weighted': True,  # Using a variance matrix for the loss function
	'finetuning': False,  # Deactivate a finetuning of the last layer after optimization
	'npred': 1,  # Number of forecast time step in the loss function
	'Nepochs': nepochs,  # Number of epochs
	'batch_size': 256,  # Batchsize during the training
	'Ntrain':1500
}
nn = NNPredictor(m, **param_nn)

# uncomment the following line to avoid displays during the neural net training:
# nn._verbfit = 0

###################
# Initial weights #
###################

# interpolate in the observations:
xobs = sb.ytox(yobs)  # Obs in the state space
xinterp = sb.interpolate_obs(xobs)

# Calculate the inverse of variance in the loss fuction
# (1 if there is an obs, 0 otherwise):
weights = np.logical_not(np.isnan(xobs)).astype(float)

# Define a particular machine learning setup for the init
param_first_nn = param_nn.copy()
param_first_nn['npred'] = 4
param_first_nn['Nepochs'] = nepochs_init
first_nn = NNPredictor(m, **param_first_nn)

# Training
first_nn.fit((xinterp, weights))

# Save initial weights
first_nn._smodel.save_weights(os.path.join(datadir, 'weights_init.h5'))

# Load the neural net with initial weight
nn._smodel.load_weights(os.path.join(datadir, 'weights_init.h5'))

############################
# Optimize over the cycles #
############################

for icycle in range(ncycle):
	############
	# DA step  #
	############

	# Surrogate model setup:
	setup = nn.define_setup(setup_true, noise=std_o)
	# NB: The setup_true is use only for the observational setup, not for the model definition

	# Run the assimilation:
	config.assimilate(setup, xtrue, yobs)

	xa = config.stats.mu.a  # Analysis
	weights = 1. / (config.stats.std.a**2 + 0.01)  # Inverse Covariance matrix (only diagonal)
	# +0.01 to avoid crash if the DA has degenerated to var =0

	############
	# ML step  #
	############

	# Fit the neural net
	nn.fit((xa, weights))

# Save the weights
nn._smodel.save_weights(os.path.join(datadir, 'weights_nn.h5'))

########
# plot #
########
# Define surrogate model setup:
setup = nn.define_setup(setup_true)

# Init at the end of the training period:
x0 = xtrue[-1]

# Change the integration time
setup.t.T = 5.
setup_true.t.T = 5.

# simulation
xsim_true      = simulate_ens(setup_true, Xinit=x0)
xsim_surrogate = simulate_ens(setup, Xinit=x0)

# plot
fig = plot_L96_2D(xsim_true, xsim_surrogate, 1.67*setup_true.t.tt, labels=['True','Surrogate'])

# save fig
fig.savefig(os.path.join(datadir, 'simulation.png'))
