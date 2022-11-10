# GMD-code
This repository is a set a function implementing the method described in the paper: 
"Combining data assimilation and machine learning to emulate a dynamical model from sparse and noisy observations"
by Julien Brajard, Alberto Carrassi, Marc Bocquet, and Laurent Bertino.

A example script is provided to run the "reference setup" described in the paper.

## Installation

Tested on Linux/MacOS<sup>[1](#myfootnote1)</sup>
1. Prerequisiste: `python3.9+` (suggest setting it up with
[anaconda](https://www.anaconda.com/download)).
2. Create a directory to save results: `mkdir example_data`
3. Install the required python modules: `pip install -r requirements.txt`
4. Run the example file  [example.py](example.py)(you can modify the file to speed up the run): `python example.py`

## Results

The code [example.py](example.py) will run the algorithm described in the paper for the standard setup 
(to run the other setups, you can modify the [example.py](example.py) code). The standard experiment run can take several hours.

The output of the code are saved on the `example_data` directory:
- `weights_init.h5`: initial weights of the neural network
- `weights_nn.h5`: weights of the neural network after optimization
- `simulation.png`: figure showing one simulation of 5 unit time steps (about 8 Lyapunov time steps).

If the algorithm has run at least once, and you have already produced weights saved in `example_data/weights_nn.h5`, you can run the code [plot_simu.py](plot_simu.py) to load the weights, make a simulation and a plot without the long optimization process: `python plot_simu.py`

In the file `simulation.png`, you should obtained the following figure:
![reference simulation](simulation_ref.png)

<a name="myfootnote1">1</a>: For MacOS, the `pythonw` was used after installation through `conda install python.app`

