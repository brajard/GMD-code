#from DAPPER.mods.Lorenz95.sak08 import setup as setup_lorenz
import os
from dapper.mods import Chronology

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
	'batch_size': 256  # Batchsize during the training
}
nn = NNPredictor(m, **param_nn)

# Load the neural net with  weights
nn._smodel.load_weights(os.path.join(datadir, 'weights_nn.h5'))


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
fig.savefig(os.path.join(datadir, 'simulation.png'))
