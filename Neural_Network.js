require.config({
  paths: {
    mathjs: '/Users/Lex/node_modules/mathjs';
  }
});

var math = require(['mathjs']);

function random_weights(a, b) {
    return Math.random() * (12.0 / (a + b)) - (6.0 / (a + b));
}

function step(x) {
    arr = 0;
    for (i = 0; i < x.length; i++) {
        for (j = 0; j < x[0].length; j++) {
            if (x[i][j] > 0) {
                arr.push(1);
            } else {
                arr.push(0);
            }
        }
    }

    return arr;
}

function getData(num_points, draw = true) {
    var canvas = document.getElementById("data");
    var ctx = canvas.getContext('2d');

    var x = 0;
    var y = 0;

    var data_pairs = math.matrix(math.zeros([num_points, num_points]));
    var target_pairs = math.matrix(math.zeros([num_points]));

    for (i = 0; i < num_points; i++) {
        x = Math.random() * 100;
        y = Math.random() * 100;

        if (0.01 * Math.pow(x, 2) < y) {
            ctx.fillStyle = 'red';
            target_pairs[i] = [0];
        } else {
            ctx.fillStyle = 'blue';
            target_pairs[i] = [1];
        }

        ctx.fillRect(x, y, 5, 5);

        data_pairs[i] = [x, y];
    }

    return [data_pairs, target_pairs];
}

var neural_network = class {
    constructor(batch_size, input_shape, hidden_layers, targets, inital_learning_rate) {
        // Hyperparameters
        var batch_size = batch_size;
        var input_shape = input_shape;
        var targets = targets;
        var hidden_layers = hidden_layer;
        var initial_learning_rate = initial_learning_rate;

        // Placeholders
        var input = math.matrix(math.zeros([this.batch_size, this.input_shape]));
        var target = math.matrix(math.zeros([this.targets]));

        // Optimization Hyperparameters
        var decay_rate = 0.96;
        var decay_steps = 180;
        var step_count = 0;
        var mu = 0.95;
        var v = 0;
        var prev_v = v;

        // Weights and Biases
        var W1 = math.matrix(math.zeros([this.input_shape[0], this.hidden_layer]));
        var W2 = math.matrix(math.zeros([this.hidden_layer, this.targets]));
        var b1 = math.matrix(math.zeros([this.hidden_layer]));
        var b2 = math.matrix(math.zeros([this.targets]));

        // Learning Rate Decay Function
        var decay_learning_rate = function(x) {
            return initial_learning_rate * Math.pow(this.decay_rate, (x / this.decay_steps));
        };

        // Loss Function
        var loss_function = function(x, y) {
            return math.multiply(x, -Math.log(y));
        };
    }

    forward_propagate(in_data) {
        this.input = in_data;

        var activation = math.max(math.add(math.multiply(this.input, this.W1), this.b1), 1);
        var output = math.max(math.add(math.multiply(this.activation, this.W2), this.b2), 1);

        var maximum = math.max(self.output, 1);
        var predictions = [];

        for (i = 0; i < this.output.length; i++) {
            for (j = 0; j < this.output[1].length; j++) {
                for (k = 0; k < this.maximum.length; k++) {
                    if (this.output[i][j] == this.maximum[k]) {
                        predictions.push(j);
                    }
                }
            }
        }

        return predictions;
    }

    backward_propagate(predictions, target_data) {
        var loss = loss_function(predictions, target_data);

        // WARNING: This code for the gradients is incorrect
        var d1 = step(math.add(math.multiply(this.activation, this.W2), this.b2));
        var d2 = step(math.add(math.multiply(this.input, this.W1), this.b1));
        var da = math.multiply(this.W2, this.d1);
        var dW2 = math.multiply(this.activation, this.d1);
        var db2 = this.d1;
        var dW1 = math.multiply(this.da, math.multiply(this.W1, this.d2));
        var db1 = math.multiply(this.da, this.d2);

        this.v_prev = this.v;
        this.v = math.add(math.multiply(this.mu, this.v), -math.multiply(decay_learning_rate(step_count), this.dW2));
        this.W2 = math.add(math.multiply(-this.mu, this.v_prev) + tf.multiply((1 + this.mu) * this.v));

        this.v_prev = this.v;
        this.v = math.add(math.multiply(this.mu, this.v), -math.multiply(decay_learning_rate(step_count), this.dW1));
        this.W1 = math.add(math.multiply(-this.mu, this.v_prev) + tf.multiply((1 + this.mu) * this.v));

        this.v_prev = this.v;
        this.v = math.add(math.multiply(this.mu, this.v), -math.multiply(decay_learning_rate(step_count), this.db2));
        this.b2 = math.add(math.multiply(-this.mu, this.v_prev) + tf.multiply((1 + this.mu) * this.v));

        this.v_prev = this.v;
        this.v = math.add(math.multiply(this.mu, this.v), -math.multiply(decay_learning_rate(step_count), this.db1));
        this.b1 = math.add(math.multiply(-this.mu, this.v_prev) + tf.multiply((1 + this.mu) * this.v));

        step_count += 1;

        return loss;
    }
}

function train(batch_size, num_steps) {
    var model = new neural_network(batch_size, 2, 100, 2, 0.01);

    var data = getData(500, draw = false);
    var input_data = data[0];
    var inputs = [];
    var target_data = data[1];
    var targets = [];

    var predictions = 0;
    var loss = 0;
    var losses = [];

    for (s = 0; s < num_steps; s++) {
        if ((s + batch_size) > 500) {
            inputs = input_data.slice(s, 2 * s + batch_size - 500).push(input_data.slice(0, 500 - s));
            targets = target_data.slice(s, 2 * s + batch_size - 500).push(target_data.slice(0, 500 - s));
        } else {
            inputs = input_data.slice(s, s + batch_size);
            targets = target_data.slice(s, s + batch_size);
        }

        predictions = model.forward_propagate(input_data[s]);
        loss = model.backward_propagate(predictions, target_data[s]);
        losses.push(loss);

        console.log("Loss value at step " + str(s) + ": " + str(loss));
    }
}

function draw_decision_boundary() {
    console.log("Test");
}
