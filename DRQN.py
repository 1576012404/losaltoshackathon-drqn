# Los Altos Hackathon 2018.
# The following code describes a Deep Recurrent Q-Network to act within environments that present partial observability.
# The agent has the following architechture:
#   Input -> 3 x (Convolutional Layer, ReLU layer, Max-Pooling Layer) -> Dropout Layer ->
#   Recurrent Layer or 2 x (Feed-Forward Layer for Delta Variable Computation) -> Dropout Layer -> Feed-Forward Layer -> Prediction.
# The optimization algorithm used is the ADAM optimizer.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

# Function to compute the final shape of the input after convolving it three times
def final_shape(D, F, S):
    c1 = math.ceil(((D - F + 1) / S))
    p1 = math.ceil((c1 / S))
    c2 = math.ceil(((p1 - F + 1) / S))
    p2 = math.ceil((c2 / S))
    c3 = math.ceil(((p2 - F + 1) / S))
    p3 = math.ceil((c3  / S))
    return int(p3)

class DRQN_agent():
    def __init__(self, input_shape, num_actions, num_delta_variables, inital_learning_rate, name, delta_bool = True):
        ###########################
        ##### Hyperparameters #####
        ###########################

        self.tfcast_type = tf.float32

        # Storage for input variables
        self.input_shape = input_shape  # Of size (length, width, channels)
        self.num_actions = num_actions
        self.initial_learning_rate = inital_learning_rate
        self.num_delta_variables = num_delta_variables

        # Convolutional Layer hyperparameters
        self.filter_size = 5
        self.num_filters = [16, 32, 64]
        self.stride = 2
        self.poolsize = 2

        self.convolution_shape = final_shape(input_shape[0], self.filter_size, self.stride) * final_shape(input_shape[1], self.filter_size, self.stride) * self.num_filters[2]

        # Recurrent Layer and Feed-Forward Layer hyperparameters
        self.cell_size = 100
        self.hidden_layer = 50
        self.dropout_probability = [0.3, 0.2]

        # Optimization hyperparameters
        self.loss_decay_rate = 0.96
        self.loss_decay_steps = 180

        #####################
        ##### Variables #####
        #####################

        # Placeholder Variables
        self.input = tf.placeholder(shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype = self.tfcast_type, name = name + ".input")
        self.target_vector = tf.placeholder(shape = (self.num_actions, 1), dtype = self.tfcast_type, name = name + ".target_vector")

        # Feature Maps
        self.features1 = tf.Variable(initial_value = np.random.rand(self.filter_size, self.filter_size, input_shape[2], self.num_filters[0]),
                                     dtype = self.tfcast_type,
                                     name = name + ".features1")
        self.features2 = tf.Variable(initial_value = np.random.rand(self.filter_size, self.filter_size, self.num_filters[0], self.num_filters[1]),
                                     dtype = self.tfcast_type,
                                     name = name + ".features2")
        self.features3 = tf.Variable(initial_value = np.random.rand(self.filter_size, self.filter_size, self.num_filters[1], self.num_filters[2]),
                                     dtype = self.tfcast_type,
                                     name = name + ".features3")

        # Delta Variable Computation
        if delta_bool:
            self.dW1 = tf.Variable(initial_value = np.random.uniform(
                                                low = -np.sqrt(6. / (self.convolution_shape + self.hidden_layer)),
                                                high = np.sqrt(6. / (self.convolution_shape + self.hidden_layer)),
                                                size = (self.convolution_shape, self.hidden_layer)),
                                  dtype = self.tfcast_type,
                                  name = name + ".dW1")
            self.dW2 = tf.Variable(initial_value = np.random.uniform(
                                                low = -np.sqrt(6. / (self.hidden_layer + self.num_delta_variables)),
                                                high = np.sqrt(6. / (self.hidden_layer + self.num_delta_variables)),
                                                size = (self.hidden_layer, self.num_delta_variables)),
                                  dtype = self.tfcast_type,
                                  name = name + ".dW2")
            self.db1 = tf.Variable(initial_value = np.zeros(self.hidden_layer), dtype = self.tfcast_type, name = name + ".dW2")
            self.db2 = tf.Variable(initial_value = np.zeros(self.num_delta_variables), dtype = self.tfcast_type, name = name + ".dW2")

        # Recurrent State Variables
        self.h = tf.Variable(initial_value = np.zeros((1, self.cell_size)), dtype = self.tfcast_type, name = name + ".h")

        self.rW = tf.Variable(initial_value = np.random.uniform(
                                            low = -np.sqrt(6. / (self.convolution_shape + self.cell_size)),
                                            high = np.sqrt(6. / (self.convolution_shape + self.cell_size)),
                                            size = (self.convolution_shape, self.cell_size)),
                              dtype = self.tfcast_type,
                              name = name + ".rW")
        self.rU = tf.Variable(initial_value = np.random.uniform(
                                            low = -np.sqrt(6. / (2 * self.cell_size)),
                                            high = np.sqrt(6. / (2 * self.cell_size)),
                                            size = (self.cell_size, self.cell_size)),
                              dtype = self.tfcast_type,
                              name = name + ".rU")
        self.rV = tf.Variable(initial_value = np.random.uniform(
                                            low = -np.sqrt(6. / (2 * self.cell_size)),
                                            high = np.sqrt(6. / (2 * self.cell_size)),
                                            size = (self.cell_size, self.cell_size)),
                              dtype = self.tfcast_type,
                              name = name + ".rV")
        self.rb = tf.Variable(initial_value = np.zeros(self.cell_size), dtype = self.tfcast_type, name = name + ".rb")
        self.rc = tf.Variable(initial_value = np.zeros(self.cell_size), dtype = self.tfcast_type, name = name + ".rc")

        # Feed-Forward Variables
        self.fW = tf.Variable(initial_value = np.random.uniform(
                                            low = -np.sqrt(6. / (self.cell_size + self.num_actions)),
                                            high = np.sqrt(6. / (self.cell_size + self.num_actions)),
                                            size = (self.cell_size, self.num_actions)),
                              dtype = self.tfcast_type,
                              name = name + ".fW")
        self.fb = tf.Variable(initial_value = np.zeros(self.num_actions), dtype = self.tfcast_type, name = name + ".fb")

        # Loss Variables
        self.step_count = tf.Variable(initial_value = 0, dtype = self.tfcast_type, name = name + ".step_count")
        self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate,       # Decay Learning Rate
                                                   self.step_count,
                                                   self.loss_decay_steps,
                                                   self.loss_decay_steps,
                                                   staircase = False,
                                                   name = "learning_rate")

        #################
        ##### Model #####
        #################

        # First Convolutional Unit
        self.conv1 = tf.nn.conv2d(input = tf.reshape(self.input, shape = (1, self.input_shape[0], self.input_shape[1], self.input_shape[2])), filter = self.features1, strides = [1, self.stride, self.stride, 1], padding = "VALID", name = name + ".conv1")
        self.relu1 = tf.nn.relu(self.conv1, name = name + ".relu1")
        self.pool1 = tf.nn.max_pool(self.relu1, ksize = [1, self.poolsize, self.poolsize, 1], strides = [1, self.stride, self.stride, 1], padding = "SAME", name = name + ".pool1")

        # Second Convolutional Unit
        self.conv2 = tf.nn.conv2d(input = self.pool1, filter = self.features2, strides = [1, self.stride, self.stride, 1], padding = "VALID", name = name + ".conv2")
        self.relu2 = tf.nn.relu(self.conv2, name = name + ".relu2")
        self.pool2 = tf.nn.max_pool(self.relu2, ksize = [1, self.poolsize, self.poolsize, 1], strides = [1, self.stride, self.stride, 1], padding = "SAME", name = name + ".pool2")

        # Third Convolutional Unit
        self.conv3 = tf.nn.conv2d(input = self.pool2, filter = self.features3, strides = [1, self.stride, self.stride, 1], padding = "VALID", name = name + ".conv3")
        self.relu3 = tf.nn.relu(self.conv3, name = name + ".relu3")
        self.pool3 = tf.nn.max_pool(self.relu3, ksize = [1, self.poolsize, self.poolsize, 1], strides = [1, self.stride, self.stride, 1], padding = "SAME", name = name + ".pool3")

        # Dropout + Reshaping
        self.drop1 = tf.nn.dropout(self.pool3, self.dropout_probability[0], name = name + ".drop1")
        self.reshaped_input = tf.reshape(self.drop1, shape = [1, -1], name = name + ".reshaped_input")

        # Delta Variable Computation
        if delta_bool:
            self.dv1 = tf.nn.relu(tf.matmul(self.reshaped_input, self.dW1) + self.db1, name = name + ".dv1")
            self.dv2 = tf.tanh(tf.matmul(self.dv1, self.dW2) + self.db2, name = name + ".dv2")
            self.dv = 90 * tf.reshape(self.dv2, [-1], name = name + ".dv")    # Converts dv2 into a vector of values in the range [-90, 90]

        # Recurrent Layer
        self.h = tf.tanh(tf.matmul(self.reshaped_input, self.rW) + tf.matmul(self.h, self.rU) + self.rb, name = name + ".h")
        self.o = tf.nn.softmax(tf.matmul(self.h, self.rV) + self.rc, name = name + ".o")

        # Dropout
        self.drop2 = tf.nn.dropout(self.o, self.dropout_probability[1], name = name + ".drop2")

        # Feed-Forward Layer
        self.output = tf.reshape(tf.matmul(self.drop2, self.fW) + self.fb, shape = [-1, 1], name = name + ".output")
        self.prediction = tf.argmax(self.output, name = name + ".prediction")

        # Back-Propagation
        self.loss = tf.reduce_mean(tf.square(self.target_vector - self.output), name = name + ".loss")
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, name = name + ".AdamOptimizer")
        self.gradients = self.optimizer.compute_gradients(self.loss)
        self.update = self.optimizer.apply_gradients(self.gradients)

        self.parameters = (self.features1, self.features2, self.features3,
                           self.rW, self.rU, self.rV, self.rb, self.rc,
                           self.fW, self.fb)
        if delta_bool: self.parameters + (self.dW1, self.dW2, self.db1, self.db2)

class ExperienceReplay():
    def __init__(self, buffer_size):
        self.buffer = []        # Buffer that contains the memory tuples
        self.buffer_size = buffer_size
    def appendToBuffer(self, memory_tuplet):
        if len(self.buffer) > self.buffer_size:     # If the buffer is at its maximum size
            for i in range(len(self.buffer) - self.buffer_size):
                self.buffer.remove(self.buffer[0])      # Remove the oldest tuplets
        self.buffer.append(memory_tuplet)       # Append the newset memory
    def sample(self, n):
        memories = []
        for i in range(n):
            memory_index = np.random.randint(0, len(self.buffer))       # Randomly sample 'n' random memory tuples
            memories.append(self.buffer[memory_index])
        return memories
