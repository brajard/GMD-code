from dapper.mods.Lorenz96 import step
from dapper.mods import HiddenMarkovModel, Id_mat, Id_op, Chronology, ens_compatible  #former TwinSetup
from dapper.tools.progressbar import progbar
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, BatchNormalization, Conv1D, Dropout, Add, Multiply, Concatenate
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') #change to activate a GUI matplotlib backend
from sklearn.linear_model import Ridge
from dapper.tools.randvars import RV, GaussRV
from scipy.interpolate import griddata
from inspect import signature
import pickle
import os
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

######################
# Lorenz Model utils #
######################

# Default time period
t = Chronology(0.05, dkObs=1, T=4**5, BurnIn=20)

# Size
m = 40

# Dict of the model
f = {
	'M'    : m,
	'model': step,
	'noise': 0
	}
# Init of the model
X0 = GaussRV(M=m, C=0.001)

# 0bservation operator
h = {
	'M'    : m,
	'model': Id_op(),
	'jacob': Id_mat(m),
	'noise': 1, # abbrev GaussRV(C=CovMat(eye(m)))
	}


other = {'name': os.path.relpath(__file__,'mods/')}
setup = HiddenMarkovModel(f,h,t,X0,**other)

####################
# Simulation utils #
####################


def simulate_ens(setup,N=1,desc='Simul',squeeze=True,Xinit=None):
	"""Generate a simulation on a ensemble (default=1)
	if squeeze is True and N==1, the output is squeeze on the ens dim
	if Xinit is not None, use Xinit as initial state instead of X0 generator"""
	f,h,chrono,X0 = setup.Dyn, setup.Obs, setup.t, setup.X0
# Init

	xx    = np.zeros((chrono.K+1,N,f.M))
	if Xinit is None:
		xx[0] = X0.sample(N)
	else:
		xx[0] = Xinit

# Loop
	for k,kObs,t,dt in progbar(chrono.ticker,desc):
		xx[k] = f(xx[k-1],t-dt,dt) + np.sqrt(dt)*f.noise.sample(1)

	if N == 1 and squeeze:
		xx = xx[:,0,:]

	return xx


####################
# Neural net utils #
####################

# to make a periodic padding of a tensor
def keras_padding ( v ):
	if isinstance(v, int):
		v = (v, v)
	vleft, vright = v

	def padlayer ( x ):
		leftborder = x[..., -vleft:, :]
		rigthborder = x[..., :vright, :]
		return tf.concat([leftborder, x, rigthborder], axis=-2)

	return padlayer

# Add an artificial feature (to handle the weights in the cost function)
def dummy_feature( x ):
	return tf.concat([x,x],axis=-1)

# Define a several step recursive model
def RecModel(rkmodel,nb_timestep,output_sequence=True):
	loutput = []
	shape = rkmodel.input_shape
	input = Input(batch_shape=shape)
	output_true = rkmodel(input)
	#Trick to change the last layer name
	output = Lambda( lambda x:x, name = rkmodel.name+'_t0')(output_true)
	loutput.append(output)
	for i in range(1,nb_timestep):
		w = Lambda(lambda x: x[...,0:1])(loutput[i-1])
		output_true = rkmodel(w)
		output = Lambda(lambda x: x, name=rkmodel.name + '_t' + str(i))(output_true)
		loutput.append(output)
	if output_sequence:
		return Model(input,loutput)
	else:
		return Model(input,loutput[-1])

# Construct the training dataset
def make_train ( xa, nseq=1, burnin=40 ,weights = None):
	xx = xa[burnin:]
	wtrain = None
	if weights is not None:
		if np.isscalar(weights):
			weights = weights*np.ones_like(xa)
		ww = weights[burnin:]
		assert xx.shape == ww.shape, str(xx.shape) + '!= ' + str(ww.shape)
		wtrain = [np.roll(ww, -i, axis=0)[:-nseq, :, np.newaxis] for i in range(1, nseq + 1)]
	xtrain = xx[:-nseq, :, np.newaxis]
	ytrain_v = [np.roll(xx, -i, axis=0)[:-nseq, :, np.newaxis] for i in range(1, nseq + 1)]
	if weights is not None: # add the weights to the target
		yytrain = [np.concatenate((yi,wi),axis=2) for (yi,wi) in zip(ytrain_v,wtrain)]
	else:
		yytrain = ytrain_v
	if len(yytrain) == 1:
		yytrain = yytrain[0]

	return xtrain, yytrain

#Design cost function
def weighted_mse(y_true,y_pred):
	val_true, weight = y_true[...,0:1], y_true[...,1:]
	sq = tf.math.square(y_pred - val_true) * weight
	return tf.reduce_mean(sq)

#Check if a layer is linear
def islinear(layer):
	d = layer.get_config()
	w = layer.get_weights()
	if not 'trainable' in d or not 'activation' in d:
		return False
	return len(w)>0 and d['trainable'] and d['activation']=='linear'

class stepmodel2():
	def __init__ ( self, model ):
		self.model = model
		self.border = (self.model.input_shape[1] - self.model.output_shape[1]) // 2

	def update_weights ( self, wfile ):
		self.model.load_weights(wfile)

	def __call__ ( self, E, t, dt ):
		if E.ndim == 1:
			E = E[np.newaxis, ...]
		return self.model.predict(E[..., np.newaxis]).squeeze()  # return E

class NNPredictor:
	def __init__ (self,m,archi,
	Ntrain=-1,npred=1,nin=1,
	Nepochs=10,bilin=False,batchnorm=True,
	weighted=True, reg=None,finetuning=True,
	batch_size=128,optimizer='Adagrad',patience=100):
		"""
		Main class to handle neural nets
		:param m: size of the model
		:param archi: architecture in form of a dictionnary of tuples (size, filter size, activation, dropout rate)
		:param Ntrain: Number of data taken as training (the rest is taken as test)
		:param npred: Nummber of forecast steps in the loss function
		:param nin: Number of time steps as input
		:param Nepochs: Number of epochs during traning
		:param bilin: Activate bilinera layer for the first layer
		:param batchnorm: Activate a batchnorm layer in input
		:param weighted: Use the inverse of diagonal covariance in the loss function (identity otherwise)
		:param reg: Regulariation of the last layer
		:param finetuning: Fintune the last layer using a linear regression after optimization
		:param batch_size: Batch size during the training
		:param optimizer: Optimizer used for training
		:param patience: Number of epochs to retain the best test score (has an effect only if Ntrain < size of data)
		"""
		assert nin==1 or npred==1, 'Time seq both in and out not implemented'
		self._m = m
		self._archi = archi
		self._Ntrain = Ntrain
		if np.isnan(npred):
			npred = 1
		self._npred = int(npred)
		self._nin = nin
		self._Nepochs = Nepochs
		self._bilin = bilin
		self._batchnorm = batchnorm
		self._batchnorm = batchnorm
		self._weighted = weighted
		self._batchsize = batch_size
		self._optimizer = optimizer
		self._verbfit = 1
		self._patience = patience
		if reg is None:
			self._reg = 'ridge',0
		else:
			self._reg = reg
		self._finetuning = finetuning
		self._smodel,self._wmodel,self._tmodel = self.buildmodels()

	def buildmodels( self ):
		"""
		buid the neuronel model
		:return: return a tuple containing:
		the short model,
		the model with dummy output (to handle covariance),
		the recurrent model to handle Nf > 1
		All the three models share the same weights
		"""
		border = int(np.sum(np.array([kern//2 for fil,kern,activ,dropout in self.archi])))
		xin = Input(shape=(self.m,self.nin))
		x3 = None
		padlayer = keras_padding(border)
		x = Lambda(padlayer)(xin)
		if self.batchnorm:
			x = BatchNormalization()(x)
		bilintodo = self.bilin
		for nfil,nkern,activ,drop in self.archi:
			if bilintodo: #bilinear layer (only once)
				if drop > 0:  # Add the maxnormvalue
					x1 = Conv1D(nfil, nkern, activation=activ, kernel_constraint=maxnorm(3.))(x)
					x1 = Dropout(rate=drop)(x1)
					x2 = Conv1D(nfil, nkern, activation=activ, kernel_constraint=maxnorm(3.))(x)
					x2 = Dropout(rate=drop)(x2)
				else:
					x1 = Conv1D(nfil, nkern, activation=activ)(x)
					x2 = Conv1D(nfil, nkern, activation=activ)(x)
				x3 = Multiply()([x1,x2])

			if drop>0: #Add the maxnormvalue
				x = Conv1D(nfil,nkern,activation=activ,kernel_constraint=maxnorm(3.))(x)
				x =  Dropout(rate=drop)(x)
			else:
				x = Conv1D(nfil,nkern,activation=activ)(x)

			if bilintodo:
				x = Concatenate()([x, x3])
				bilintodo = False
		if self._reg[1]>0:
			if self._reg[0] == 'ridge':
				dy = Conv1D(1,1,activation='linear',kernel_regularizer=regularizers.l2(self._reg[1]))(x)
			else:
				raise NotImplementedError(self._reg[0],'regularization no implemented')
		else:
			dy = Conv1D(1,1,activation='linear')(x)
		soutput = Add()([xin,dy])
		woutput = Lambda(dummy_feature)(soutput)

		smodel = Model(xin,soutput)
		wmodel = Model(xin,woutput)
		tmodel = RecModel(wmodel,self.npred,output_sequence=True)
		return smodel,wmodel,tmodel

	def fit( self , xa, input=None):
		"""
		Run the training of the neural net
		:param xa: traning data given as a time seriz
		:param input: input training data in case they are different than the target
		"""
		if self.weighted:
			xa,weights = xa
		else:
			xa,weights = xa
			#if not np.isscalar(weights) or not weights == 1:
			#	warnings.warn('Weights ignored and set to one')
			weights = 1

		#Filter nan value

		ok = np.all(np.isfinite(xa), axis=1)
		if not input is None:
			ok = ok & np.all(np.isfinite(input), axis=1)
		n_ok = ok.sum()
		if n_ok < xa.shape[0]:
			n_nan = xa.shape[0]-ok.sum()
			warnings.warn(str(n_nan)+
				' nan values found in training set: '+
				str(n_ok) +'/'+str(xa.shape[0])+' kept')
			xa = xa[ok]
			if not np.isscalar(weights):
				weights = weights[ok]
			if not input is None:
				input = input[ok]

		if xa.shape[0]<self.Ntrain:
			warnings.warn('Ntrain('+ str(self.Ntrain)+') value too large for the dataset (' + str(xa.shape[0]) +')',
				Warning)
			limT = xa.shape[0]
		elif self.Ntrain == -1:
			limT = xa.shape[0]
		else:
			limT = self.Ntrain
		if np.isscalar(weights):
			weights = weights*np.ones_like(xa)
		xtrain, ytrain = make_train(xa[:limT], nseq=self.npred, weights=weights[:limT],burnin=0)
		if not input is None:
			xtrain,_ = make_train(input[:limT], nseq=self.npred, weights=weights[:limT],burnin=0)
		if limT>0:
			xval,yval = make_train(xa[limT:], nseq=self.npred, weights=weights[limT:],burnin=0)
			if not input is None:
				xval,_ = make_train(input[limT:], nseq=self.npred,weights=weights[limT:],burnin=0)
			validation_data = (xval,yval)
		else:
			validation_data = None
		early_stopping = EarlyStopping(monitor='val_loss',
			patience=self.patience, verbose=1, mode='auto',restore_best_weights=True)
		self._tmodel.compile(optimizer=self.optimizer,loss=weighted_mse)
		self.hist = self._tmodel.fit(xtrain, ytrain,
			epochs=self.Nepochs,
			batch_size=self._batchsize,
			verbose=self._verbfit,
			validation_data=validation_data, callbacks=[early_stopping])
		if self.finetuning:
			if self.check_finetuning():
				if self.npred>1: #Only tune using the t+1 target
					self.finetune_layer(xtrain,ytrain[0])
				else:
					self.finetune_layer(xtrain, ytrain)
			else:
				warnings.warn('no finetuning performed (architecture not compatible)')

	def finetune_layer(self,xtrain,ytrain):
		"""
		In case the finetuning is activate, make a linear regression to tune the weights of the last
		layer.
		:param xtrain: input of the neural net
		:param ytrain: target of the neural net
		"""
		weights = np.sqrt(ytrain[:,:,1:]).reshape(-1)
		nn = self._smodel
		newnet = Model(nn.layers[0].input,[nn.layers[-2].input,nn.layers[-2].output])
		ysimul = nn.predict(xtrain)
		dysimul1 = newnet.predict(xtrain)
		yimul1 = xtrain+dysimul1[1] #resnet
		assert np.linalg.norm((ysimul-yimul1)/ysimul) < 1e-6*ysimul.size

		W = nn.layers[-2].get_weights()
		#Size of the weights
		# filter_size(input), number_of_features(input), number_of_features(output)

		nw = W[0].size
		assert W[1].size == 1
		A = dysimul1[0].reshape(-1,nw)
		b = (ytrain[:,:,0:1]-xtrain).reshape(-1)
		if self.weighted:
			A = A * np.broadcast_to(weights[:,np.newaxis],A.shape)
			b = b * weights
		clf = Ridge(alpha=self._reg[1])
		clf.fit(A,b)
		W0 = clf.coef_
		W1 = clf.intercept_
		W[0][0,:,0] = W0
		W[1][0] = W1
		nn.layers[-2].set_weights(W)

	def define_setup( self, setup_ref, noise=0. ):
		"""
		define the DAPPER object setup to simulate and assimilate in the model
		:param setup_ref: reference DAPPER setup used to copy the chronology, the observation operator and the initial step
		:param noise: standard deviation of the model to be added to the forecast
		:return: the DAPPER object setup
		"""
		stepnn = stepmodel2(self._smodel)
		fnn = { 'M': self.m, 'model': stepnn, 'noise': noise, 'nn': self._smodel }
		setup = HiddenMarkovModel(fnn,setup_ref.Obs,setup_ref.t,setup_ref.X0)
		return setup
	def plot_history( self ,normalized=True):
		"""
		plot the history of the traning
		:param normalized: if True normalize both validation/traning loss to 1 fort he first eppoch.
		:return: the matplotlib figure
		"""
		fig,ax = plt.subplots()
		S1 = self.hist.history['loss'][2] if normalized else 1
		S2 = self.hist.history['val_loss'][2] if normalized else 1
		ax.semilogy(np.array(self.hist.history['loss'])[2:]/S1, color='gray', linewidth=2, label='train')
		ax.semilogy(np.array(self.hist.history['val_loss'])[2:]/S2, color='black', linewidth=2, label='test')
		ax.legend()
		return fig
	def ntrainable_weights( self ):
		"""
		:return: the number of trainable weights in the neural net
		"""
		return int(
	np.sum([tf.keras.backend.count_params(p) for p in set(self._smodel.trainable_weights)]))
	def check_finetuning( self ):
		""" Check if finetuning is possible, i.e. last layer is a sum and -2 layer is linear"""
		return isinstance(self._smodel.layers[-1],Add) \
			   and islinear(self._smodel.layers[-2])
	def load_weights( self , *args, **kwargs):
		"""load the weights of the neural net (see keras load_weights function)"""
		return self._smodel.load_weights(*args,**kwargs)
	def save_weights( self, *args, **kwargs ):
		return self._smodel.save_weights(*args,**kwargs)
	@property
	def archi( self ):
		return self._archi
	@archi.setter
	def archi( self,val ):
		self._archi = val
		self._smodel, self._wmodel, self._tmodel = self.buildmodels()

	@property
	def m( self ):
		return self._m
	@m.setter
	def m( self,val ):
		self._m = val
		self._smodel, self._wmodel, self._tmodel = self.buildmodels()

	@property
	def nin (self):
		return self._nin

	@nin.setter
	def nin ( self, val ):
		if val>1 and self.npred>1:
			warnings.warn('nin value has not changed: Time seq both in and out not implemented',Warning)
		else:
			self._nin = val
			self._smodel, self._wmodel, self._tmodel = self.buildmodels()

	@property
	def batchnorm( self ):
		return self._batchnorm
	@batchnorm.setter
	def batchnorm( self,val ):
		self._batchnorm = val
		self._smodel, self._wmodel, self._tmodel = self.buildmodels()
	@property
	def bilin( self ):
		return self._bilin
	@bilin.setter
	def bilin( self,val ):
		self._bilin = val
		self._smodel, self._wmodel, self._tmodel = self.buildmodels()
	@property
	def finetuning( self ):
		return self._finetuning
	@property
	def npred( self ):
		return self._npred
	@npred.setter
	def npred ( self, val ):
		if val>1 and self.nin>1:
			warnings.warn('npred value has not changed: Time seq both in and out not implemented')
		else:
			self._npred = val
			self._smodel, self._wmodel, self._tmodel = self.buildmodels()
	@property
	def weighted( self ):
		return self._weighted
	@property
	def Ntrain( self ):
		return self._Ntrain
	@property
	def patience ( self ):
		return self._patience
	@property
	def optimizer( self ):
		return self._optimizer
	@property
	def batchsize( self ):
		return self._batchsize
	@property
	def Nepochs( self ):
		return self._Nepochs

####################
# Neural net utils #
####################

class SetupBuilder:
	def __init__( self,t=None,fname='./data2/L96_train_51.npz',seed_sample=1,
			std_o=1,p=20,sample='random',seed_obs=2,
			m=40,step=step,data=None):
		"""
		Classe allowing the save and generation of the DAPPER setup object
		:param t: chronology of the setup
		:param fname: file used to produce initial states (unused if data is not None)
		:param seed_sample: seed for random generator for chosing initial state
		:param std_o: standard devation of noise on observation
		:param p: number of observation
		:param sample: 'random'/'regular' type of subsampling observations for each time steps
		:param seed_obs: seed fo rendom generator of noise on observations
		:param m: size of the step
		:param step: forecast function for one time step
		:param data: array in which chosing initial states (replace fname if specified)
		"""
		if t is None:
			self.t = Chronology(0.05,dkObs=1, T=4 ** 6, BurnIn=2)
		else:
			self.t = t
		self.dt = self.t.dt
		self.fname = fname
		self.seed_sample = seed_sample
		self.std_o = std_o
		self.p = p
		self.sample = sample
		self.seed_obs = seed_obs
		self.m = m
		self.step = step
		if data is not None:
			self.data = data
		else:
			self.data = np.load(fname)['xxtest']
		assert self.data.shape[1] == self.m
		self.compute_tinds()
		#TODO: Create data properties and setter

	def sampling( self,N ):
		"""
		sample N initial states
		:param N: size of the ensemble to sample
		:return: the ensemble
		"""
		N0 = self.data.shape[0]
		save_state = np.random.get_state()
		np.random.seed(self.seed_sample)
		idx = np.random.choice(N0, N, replace=True)
		np.random.set_state(save_state)
		E = self.data[idx]
		return E
	def X0( self ):
		"""
		:return: the DAPPER function for initial sampling
		"""
		return RV(self.m, func=self.sampling)
	def compute_tinds( self ):
		"""
		compute the index of observation for each time steps
		"""
		self.tinds = dict()
		save_state = np.random.get_state()
		np.random.seed(self.seed_obs)
		for k, KObs, t_, dt in self.t.ticker:
			if KObs is not None:
				if self.sample == 'random':
					self.tinds[t_] = np.random.choice(self.m, size=self.p, replace=False)
				elif self.sample == 'regular':
					self.tinds[t_] = np.linspace(0, self.m, self.p, endpoint=False, dtype=np.int)
		np.random.set_state(save_state)


	def def_hmod( self ):
		"""
		:return: the observation operator
		"""
		@ens_compatible
		def hmod ( E, t ):
			return E[self.tinds[t]]
		return hmod

	def ytox( self, yy, chrono=None ):
		"""
		create a state field with only observation
		:param yy: observation to consider
		:param chrono: time chronology (if None take the object chronology)
		:return: a array sized as a state space with np.nan where there is no observation
		"""
		if chrono is None:
			chrono = self.t
		Xobs = np.nan * np.ones(shape=(chrono.K + 1, self.m))
		for k, KObs, t_, dt in chrono.ticker:
			if KObs is not None:
				Xobs[k, self.tinds[t_]] = yy[KObs]
		return Xobs

	def maskObs( self ,chrono=None):
		"""
		create a mask of observation (True if an observation is present)
		:param chrono: time chronology (if None take the object chronology)
		:return: a array sized as a state space with True where ther is an observation (False otherwise)
		"""
		if chrono is None:
			chrono = self.t
		MaskObs = np.zeros(shape=(chrono.K + 1,self.m)).astype(bool)
		for k, KObs, t_, dt in chrono.ticker:
			MaskObs[k,self.tinds[t_]] = True
		return MaskObs

	def interpolate_obs( self, Xobs, chunk=1000, dt=20 ):
		"""
		Interpolate observation to the state space using cubic interpolation
		:param Xobs: Observation in a state space sized array (as produced by ytox)
		:param chunk: size of array to interpolate at once (optimization paramters)
		:param dt: border size of each chunk
		:return: the interpolated field
		"""
		GD1 = np.zeros_like(Xobs)
		for i in range(1 + Xobs.shape[0] // chunk):
			start = i * chunk
			end = min((i + 1) * chunk, Xobs.shape[0])
			start1 = max(start - dt, 0)
			end1 = min(end + dt, Xobs.shape[0])

			x = np.arange(0, Xobs.shape[1])
			y = np.arange(0, end1 - start1)

			start0 = start - start1
			end0 = end - start1
			# mask invalid values
			array = np.ma.masked_invalid(Xobs[start1:end1])
			xx, yy = np.meshgrid(x, y)
			# get only the valid values
			x1 = xx[~array.mask]
			y1 = yy[~array.mask]
			newarr = array[~array.mask]

			GD1[start:end, :] = griddata((x1, y1), newarr.ravel(), (xx[start0:end0, :], yy[start0:end0, :]),
				method='cubic', fill_value=0)
		return GD1

	def h_dict( self ):
		"""
		:return: Dictionnary corresponding to the observation operator in the DAPPER format
		"""
		h = { 'M': self.p,
			'model': self.def_hmod(),
			'jacob': Id_mat(self.p),
			'noise': GaussRV(C=self.std_o * np.eye(self.p))}
		return h

	def f_dict( self ):
		"""
		:return: Dictionnary corresponding the the model operator in the DAPPER format
		"""
		fref = { 'M': self.m,
			'model': self.step,
			'noise': 0 }
		return fref
	def setup( self ):
		"""
		:return: the setup object in the DAPPER format
		"""
		return HiddenMarkovModel(self.f_dict(),self.h_dict(),self.t,self.X0())

	def get_params_dict( self ):
		"""
		:return: dictionnary of parameter of the class constructor
		"""
		params = self._get_param_names()
		out = dict()
		for key in params:
			out[key] = getattr(self,key,None)
		return out

	def save( self , fname):
		"""
		save the class to a file
		:param fname: name of the file
		"""
		out = self.get_params_dict()
		with open(fname,'wb') as f:
			pickle.dump(out,f,protocol=0)
	def __str__( self ):
		out = self.get_params_dict()
		return 'SetupBuilder\n'+str(out)
	@classmethod
	def _get_param_names( cls):
		# introspect the constructor arguments to find the model parameters
		# to represent
		init_signature = signature(cls.__init__)
		# Consider the constructor parameters excluding 'self'
		parameters = [p.name for p in init_signature.parameters.values() if p.name != 'self' and p.kind != p.VAR_KEYWORD]
		return parameters

def load_sbuilder(fname):
	"""
	load a setup builder
	:param fname: name of the file
	:return: a instance of the class SetupBuilder
	"""
	with open(fname,'rb') as f:
		out = pickle.load(f)
	return SetupBuilder(**out)


################
# Divers utils #
################
def plot_L96_2D(xx,xxpred,tt,labels,vmin=None,vmax=None,vdelta=None):
	"""
	plot two simulation (xx, xxpred, and the difference xpred-xx)
	:param xx: first simulation to plot (size: size of the space, number of time steps)
	:param xxpred: second simulation to plot (size: size of the space, number of time steps)
	:param tt: chronology (used for x-axis)
	:param labels: list of two labels [first simulation, second simulation)
	:param vmin: minimum value of the first two plots
	:param vmax: minimum value of the first two plots
	:param vdelta: extreme value of the difference plot
	:return: a matplotlib figure
	"""
	if vmin is None:
		vmin,vmax = np.nanmin(xx),np.nanmax(xx)
	if vdelta is None:
		vdelta = np.nanmax(np.abs(xxpred-xx))
	m = xx.shape[1]
	tmin = tt[0]
	tmax = tt[-1]
	fig,ax = plt.subplots(nrows=3,sharex='all')

	divider = [make_axes_locatable(a) for a in ax]

	cax = dict()
	for i in range(3):
		cax [i] = divider[i].append_axes('right', size='5%', pad=0.05)

	delta= dict()
	delta[0] = ax[0].imshow(xx.T,cmap=plt.get_cmap('viridis'),vmin=vmin,vmax =vmax,extent=[tmin,tmax,0,m],aspect='auto')
	delta[1] = ax[1].imshow(xxpred.T,cmap=plt.get_cmap('viridis'),vmin=vmin,vmax=vmax,extent=[tmin,tmax,0,m],aspect='auto')
	delta[2] = ax[2].imshow(xxpred.T- xx.T,cmap=plt.get_cmap('bwr'),
		extent=[tmin,tmax,0,m],aspect='auto',vmin=-vdelta,vmax=vdelta)
	ax[0].set_ylabel(labels[0])
	ax[1].set_ylabel(labels[1])
	ax[2].set_ylabel(labels[1][:2] + ' - ' + labels[0][:2] )
	for i in delta:
		fig.colorbar(delta[i],cax=cax[i],orientation='vertical')
	ax[2].set_xlabel('time')
	return fig

