# LosAltosHackathon-DRQN
A Deep Recurrent Q-Network designed for the 2018 Los Altos Hackathon. The Deep Recurrent Q-Network (DRQN) is designed to play in ViZDoom's CIG environment. The network has the following architechture:

Single Frame Input -> 3 x (Convolutional Layer, ReLU layer, Max-Pooling Layer) -> Dropout Layer -> Recurrent Layer or 2 x (Feed-Forward Layer for Delta Variable Computation) -> Dropout Layer -> Feed-Forward Layer -> Prediction. 

The optimization algorithm used is the ADAM optimizer. The network was trained for 10000 episodes, each episode representing either 300 frames or the amount of frames until the network dies.

Training can be run via the traing code "trainAgent.py". After this code's completion, the DRQN can store its final parameters for later use in a file called "DRQN_best_params.ckpt". As well, the DRQN outputs a text file containing a list of all its reward values and loss values collected throughout training. The ".ckpt" file can be loaded via the testing code "testAgent.py".

# Prerequisites:
-Python v3.6.4

-GCC v4.2.1

-Tensorflow v1.6.0

-Numpy v1.14.0

-ViZDoom v1.1.5

# Installation:
