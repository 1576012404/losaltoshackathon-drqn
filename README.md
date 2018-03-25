# Los Altos Hackathon - DRQN:

NOTICE: The current build of the DRQN is NOT complete. For the CIG environment, the network gains a total reward of 0.0 across all 3,000,000 frames and seems to drift towards the far right corner of the map. Do NOT attempt to run this model until it is fixed.

A Deep Recurrent Q-Network designed for the 2018 Los Altos Hackathon. The Deep Recurrent Q-Network (DRQN) is designed to play in ViZDoom's CIG environment. The network has the following architechture:

1. Single Frame Input.
2. 3 x (Convolutional Layer, ReLU layer, Max-Pooling Layer).
3. Dropout Layer.
4. Recurrent Layer or 2 x (Feed-Forward Layer for Delta Variable Computation).
5. Dropout Layer.
6. Feed-Forward Layer.
7. Prediction. 

The optimization algorithm used is the ADAM optimizer. The network was trained for 10000 episodes, each episode representing either 300 frames or the amount of frames until the network dies.

The DRQN is capable of storing its final parameters for later use in a file called `Models/DRQN_best_params.ckpt-[model number].[file_extension]`. As well, the DRQN outputs a text file containing a list of all its reward values and loss values collected throughout training.

# Prerequisites:
* Python v3.6.4.

* GCC v4.2.1.

* [Tensorflow](https://www.tensorflow.org/) v1.6.0.

* Numpy v1.14.0.

* [ViZDoom](http://vizdoom.cs.put.edu.pl/) v1.1.5.

# Installation:
To install the repository, run `git clone https://github.com/Luthanicus/losaltoshackathon-drqn.git` in a terminal window.
To train the network, run `trainAgent.py` or use a predefined model in the `Models/` folder.
To test the network, run `testAgent.py` using the `Models/DRQN_best_params.ckpt-[model number].[file_extension]` model.

# Training:
Training is estimated to finish after 9.43 hours on 2.9 GHz Intel Core i5 CPU. 

# Built With:
* [Python](https://www.python.org/).

* [Tensorflow](https://www.tensorflow.org/).

* [ViZDoom](http://vizdoom.cs.put.edu.pl/).

# Author:
Alexander McDowell. Code written at the 2018 Los Altos Hackathon.
