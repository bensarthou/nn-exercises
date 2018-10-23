import numpy as np


class NNet:

    # ---------------------------  MAIN FUNCTIONS  -------------------------------

    def __init__(self, n_features, n_labels, hidden_layers=[]):
        """
        NNet constructor. The output layer is a softmax layer.
        :param n_features: size of input data, int
        :param n_labels: number of classes to predict, int
        :param hidden_layers: description of hidden layers of the NNet, list.
                              Ex: [{'n_neurons': 70, 'activation': 'sigmoid'},
                                   {'n_neurons': 30, 'activation': 'sigmoid'}]
        """
        # init activation functions of hidden layers
        self.n_hidden_layers = len(hidden_layers)
        self.activations_map = {'sigmoid': self.sigmoid, 'relu':self.relu}
        self.activations_derivatives_map = {'sigmoid': self.deriv_sigmoid, 'relu':self.deriv_relu}
        self.hidden_layers_activations = []
        for layer in hidden_layers:
            if layer['activation'] in {'sigmoid', 'relu'}:
                self.hidden_layers_activations.append(layer['activation'])
            else:
                raise ValueError('Activation functions other than sigmoid or relu have not been implemented yet.')

        # init weights and biases of each layer
        self.n_labels = n_labels
        n_in = [n_features] + [layer['n_neurons'] for layer in hidden_layers]
        n_out = [layer['n_neurons'] for layer in hidden_layers] + [n_labels]
        self.weights = [None] * len(n_in)  # each element is W
        self.biases  = [None] * len(n_in)  # each element is b
        for i_layer in range(self.n_hidden_layers + 1):
            self.weights[i_layer], self.biases[i_layer] = self.init_weights(n_in[i_layer],
                                                                            n_out[i_layer])

        # Regularization
        self.reg_lambda = 0.0
        self.lnorm = 2

    def predict(self, X, return_output=False):
        """
        Predict labels of each element of X using trained NNet.
        :param X: list of samples to predict, ndarray(n_samples, n_features)
        :param return_output: (optionnal) if True, return also full nnet_output
        :return: y: list of predicted labels, ndarray(n_samples,)
        :return: (only if return_output = True) the full output of the nnet,
                    ndarray (n_samples, n_labels)
        """
        # pass through hidden layers
        inputs = X.T
        for h_layer in range(self.n_hidden_layers):
            a = self.pre_activation(self.weights[h_layer], self.biases[h_layer], inputs)
            inputs = self.activations_map[self.hidden_layers_activations[h_layer]](a)

        # pass through output layer
        output = self.softmax(self.pre_activation(self.weights[-1], self.biases[-1], inputs))

        # get best labels
        y = np.argmax(output, axis=0)

        # return best labels (and output if necessary)
        if not return_output:
            return y
        else:
            return y, output.T

    def score(self, X, y, metric="accuracy"):
        """
        Predict labels of each element of X using trained NNet and compare it to real labels y.
        :param X: list of samples to predict, ndarray(n_samples, n_features)
        :param y: list of real labels, ndarray(n_samples,)
        :param metric: the score metric to return, {"accuracy", "logloss"}
        :return: precision: ratio of well predicted labels.
        """
        if metric == "accuracy":
            y_pred = self.predict(X)
            error_rate = self.error_rate(y_pred, y)
            return 1 - error_rate
        elif metric == "logloss":
            _, output = self.predict(X, return_output=True)
            return self.log_loss(output.T, y)

    def train(self, X, y,
              valid_set=None,
              n_epoch=10,
              minibatch_size=128,
              eta_0=1,
              eta_decrease_factor=0,
              valid_interp_iter=50,
              reg_lambda=0.0,
              lnorm=2,
              verbose=True):
        """
        Train the NNet with the train_set given.
        :param X: list of samples to train on, ndarray(n_samples, n_features)
        :param y: list of real labels of samples to train on, ndarray(n_samples,)
        :param valid_set: validation set, (X_valid, y_valid). If valid_set is given, the function will return training logs on this validation_set.
                            valid_set: tuple (X_valid, y_valid)
                            X_valid: validation samples features, ndarray(n_samples_valid, n_features)
                            y_valid: validation samples labels, ndarray(n_samples_valid,)
        :param n_epoch: number of epochs to consider for training, int.
        :param minibatch_size: number of samples to use at each SGD iteration, int.
        :param eta_0: initial value for the SGD update factor, float.
        :param eta_decrease_factor: coefficient used in the decreasing rule of eta : eta_0 / (1 + t * eta_decrease_factor), float.
        :param valid_interp_iter: number of iterations between each check on validation set if this set is given.
        :param reg_lambda: regularization parameter, float
        :param lnorm: order for the regularization norm, int
        :param verbose: if True, display some training data, bool.

        :return: logloss_train: log-loss values of the training batches at each iteration, list.
        :return: error_train: error rate values of the training batches at each iteration, list.
        :return: logloss_valid: log-loss values of the full validation set computed each valid_interp_iter and interpolated at each iteration, list.
        :return: error_valid: error rate values of the full validation set computed each valid_interp_iter and interpolated at each iteration, list.
        """
        # init variables to log and evaluate training
        n_training = X.shape[0]
        iteration_counter = 0
        logloss_train, error_train, logloss_valid, error_valid = [], [], [], []

        # Regularization
        if reg_lambda != 0.0:
            self.reg_lambda = reg_lambda
            self.lnorm = lnorm

        # display info
        if verbose:
            if valid_set:
                print("Initial model : score={:.2f}%, logloss={:.3f}".format(100 * self.score(valid_set[0], valid_set[1]), self.score(valid_set[0], valid_set[1], metric="logloss")))
            else:
                print("Initial model : score={:.2f}%, logloss={:.3f}".format(100 * self.score(valid_set[0], valid_set[1]), self.score(X[0], y[1], metric="logloss")))

        # loop over epochs
        for i_epoch in range(n_epoch):

            # display info
            if verbose:
                print("Training model (epoch {}/{})".format(i_epoch + 1, n_epoch), end="")

            # loop over all mini-batches
            for i_batch in range(0, n_training, minibatch_size):
                iteration_counter += 1

                # get minibatch
                batches_index = np.arange(i_batch, i_batch + minibatch_size) % n_training
                X_batch = X[batches_index, :]
                y_batch = y[batches_index]

                # forward propagation : inference
                inputs, pre_activations, outputs, y_pred = self.forward(X_batch)

                # computation of log-loss and error rate for this training batch
                logloss_train.append(self.log_loss(outputs[-1], y_batch))
                error_train.append(self.error_rate(y_pred, y_batch))

                # computation of log-loss and error rate of the validation set
                if valid_set and (iteration_counter % valid_interp_iter == 0):
                    _, _, outputs_valid, y_pred_valid = self.forward(valid_set[0])
                    logloss_valid.append(self.log_loss(outputs_valid[-1], valid_set[1]))
                    error_valid.append(self.error_rate(y_pred_valid, valid_set[1]))

                # backward propagation : computation of gradient w.r.t the weights and biases
                grad_weights, grad_biases = self.backward(inputs, pre_activations, outputs, y_batch)

                # parameters update
                eta = eta_0 / (1 + eta_decrease_factor * iteration_counter)
                for layer in range(self.n_hidden_layers + 1):
                    self.weights[layer], self.biases[layer] = self.update_weights(self.weights[layer], self.biases[layer],
                                                                                  grad_weights[layer], grad_biases[layer], eta)

            # display info
            if verbose:
                if valid_set:
                    print(" : score={:.2f}%, logloss={:.3f}".format((1 - error_valid[-1]) * 100, logloss_valid[-1]))
                else:
                    print(" : score={:.2f}%, logloss={:.3f}".format((1 - error_train[-1]) * 100, logloss_train[-1]))

        # interpolation of training results on un-tested batches
        if valid_set:
            logloss_valid = np.interp(np.arange(len(logloss_train)), np.arange(len(logloss_valid)) * valid_interp_iter, logloss_valid)
            error_valid = np.interp(np.arange(len(error_train)), np.arange(len(error_valid)) * valid_interp_iter, error_valid)

        # display info
        if verbose:
            if valid_set:
                print("Final model : score={:.2f}%, logloss={:.3f}".format(100 * self.score(valid_set[0], valid_set[1]), self.score(valid_set[0], valid_set[1], metric="logloss")))
            else:
                print("Final model : score={:.2f}%, logloss={:.3f}".format(100 * self.score(valid_set[0], valid_set[1]), self.score(X[0], y[1], metric="logloss")))

        return logloss_train, error_train, logloss_valid, error_valid

    # ------------------------------  TOOLS  ----------------------------------

    def init_weights(self, n_in, n_out):
        """
        Init the weights matrix and biases vector for a given layer.
        :param n_in: size of the input, int
        :param n_out: size of the output, int
        :return: W: the weights, ndarray (n_out x n_in)
        :return: b: the bias, ndarray (n_out x 1)
        """
        W = np.random.normal(0, 1 / np.sqrt(n_in), (n_out, n_in))
        b = np.random.normal(0, 1 / np.sqrt(n_in), (n_out, 1))
        return W, b

    def one_hot_labels_representation(self, labels):
        """
        Compute one-hot representation of labels
        :param labels: labels of the samples, ndarray (1 x minibatch_size)
        :return: the one-hot representation, ndarray (n_labels x minibatch_size)
        """
        one_hot = np.zeros((self.n_labels, labels.size))
        one_hot[labels, np.arange(labels.size)] = 1
        return one_hot

    def log_loss(self, outputs, labels):
        """
        Compute log-loss of current batch
        :param outputs: outputs of the NNet, ndarray (n_labels x minibatch_size)
        :param labels: real labels, ndarray (1 x minibatch_size)
        :param lmbda: float, regularization weight
        :param order: int, order of the norm for the weight regularization

        :return log-loss summed over all outputs, float
        """

        one_hot_labels = self.one_hot_labels_representation(labels)
        reg_term = 0

        # Regularization
        if self.reg_lambda != 0.0:

            # Computing ||Theta||
            norm_theta = 0
            for i_layer in range(self.n_hidden_layers + 1):
                # ||theta|| += ||W_i||^2 + ||b_i||^2
                norm_theta += np.linalg.norm(self.weights[i_layer], self.lnorm)**2 +\
                              np.linalg.norm(self.biases[i_layer], self.lnorm)**2

            reg_term = (self.reg_lambda/2)*norm_theta

        return np.sum(-np.log(outputs) * one_hot_labels) / outputs.shape[1] + reg_term

    def error_rate(self, predicted_labels, labels):
        """
        Compute classification error rate of current batch
        :param predicted_labels: predicted labels, ndarray (1 x minibatch_size)
        :param labels: real labels, ndarray (1 x minibatch_size)
        :return error rate computed over all outputs, float
        """
        n_correct_predictions = np.sum(np.equal(np.atleast_2d(labels), np.atleast_2d(predicted_labels)))
        return 1 - n_correct_predictions / labels.size

    # ---------------------- ACTIVATION FUNCTIONS  -----------------------------

    def softmax(self, a):
        """
        Perform the softmax transformation to the pre-activation values
        :param a: the pre-activation values, ndarray (n_output x minibatch_size)
        :return: the activation values, ndarray (n_output x minibatch_size)
        """
        a = a - np.max(a)
        output = a
        for i_sample in range(a.shape[1]):
            output[:, i_sample] = np.exp(a[:, i_sample]) / np.sum(np.exp(a[:, i_sample]))
        return output

    def sigmoid(self, a):
        """
        Perform the sigmoid transformation of the pre-activation values
        :param a: the pre-activation values, ndarray (n_output x minibatch_size)
        :return: the activation values, ndarray (n_output x minibatch_size)
        """
        return 1 / (1 + np.exp(-a))

    def deriv_sigmoid(self, a):
        """
        Compute the derivative of the sigmoid function.
        :param a: the pre-activation of the hidden layer, ndarray (n_hidden x minibatch_size)
        :return: the gradient w.r.t. to the non linear activation function of the hidden layer, ndarray (n_hidden x minibatch_size)
        """
        return self.sigmoid(a) * (1 - self.sigmoid(a))

    def relu(self, x):
        """
        Perform the rectified linear transformation of the pre-activation values
        :param a: the pre-activation values, ndarray (n_output x minibatch_size)
        :return: the activation values, ndarray (n_output x minibatch_size)
        """
        return x * (x > 0)

    def deriv_relu(self, x):
        """
        Compute the derivative of the rectified linear function.
        :param a: the pre-activation values, ndarray (n_hidden x minibatch_size)
        :return: the gradient w.r.t. to the non linear activation function of the hidden layer, ndarray (n_hidden x minibatch_size)
        """
        return 1. * (x > 0)

    # ----------------------- FORWARD PROPAGATION ------------------------------

    def forward(self, X):
        """
        Predict labels of each element of X, and return pre-activations and outputs of each layer.
        :param X: list of samples to predict, ndarray(n_samples, n_features)
        :return: inputs: inputs at each layer, list (n_layers) of ndarray (n_input x minibatch_size)
        :return: pre_activations: pre-activations at each layer, list (n_layers) of ndarray (n_output x minibatch_size)
        :return: outputs: outputs at each layer, list (n_layers) of ndarray (n_output x minibatch_size)
        :return: y: best predicted labels, ndarray (1 x minibatch_size)
        """
        # init
        inputs = [None] * (self.n_hidden_layers + 1)
        pre_activations = [None] * (self.n_hidden_layers + 1)
        outputs = [None] * (self.n_hidden_layers + 1)

        # pass through hidden layers
        inputs[0] = X.T
        for h_layer in range(self.n_hidden_layers):
            pre_activations[h_layer] = self.pre_activation(self.weights[h_layer], self.biases[h_layer],
                                                           inputs[h_layer])
            outputs[h_layer] = self.activations_map[self.hidden_layers_activations[h_layer]](pre_activations[h_layer])
            inputs[h_layer + 1] = outputs[h_layer]

        # pass through output layer
        pre_activations[-1] = self.pre_activation(self.weights[-1], self.biases[-1], inputs[-1])
        outputs[-1] = self.softmax(pre_activations[-1])

        # get best labels
        y = np.argmax(outputs[-1], axis=0)
        return inputs, pre_activations, outputs, y

    def pre_activation(self, W, b, X):
        """
        Perform the pre activation of a layer given weights and inputs.
        :param W: the weights, ndarray (n_output x n_input)
        :param b: the bias, ndarray (n_output * 1)
        :param X: the input, ndarray (n_input x minibatch_size)
        :return: the transformed values, ndarray (n_output x minibatch_size)
        """
        return np.dot(W, X) + b

    # ----------------------- BACKWARD PROPAGATION ------------------------------

    def backward(self, inputs, pre_activations, outputs, labels):
        """
        Compute and return the gradient of the loss w.r.t the weights and biases of each layer.
        :param inputs: inputs at each layer, list (n_layers) of ndarray (n_input x minibatch_size)
        :param pre_activations: pre-activations at each layer, list (n_layers) of ndarray (n_output x minibatch_size)
        :param outputs: outputs at each layer, list (n_layers) of ndarray (n_output x minibatch_size)
        :param labels: real labels, ndarray (1 x minibatch_size)
        :return: grad_weights: gradient of the loss w.r.t W at each layer, list (n_layers) of ndarray (n_output x n_input)
        :return: grad_biases: gradient of the loss w.r.t b at each layer, list (n_layers) of ndarray (n_output x 1)
        """
        grad_weights = [None] * (self.n_hidden_layers + 1)
        grad_biases  = [None] * (self.n_hidden_layers + 1)

        # pass through output layer
        grad_pre_activation = self.gradient_out(outputs[-1], labels)
        grad_weights[-1], grad_biases[-1] = self.gradient_weights(grad_pre_activation,
                                                                  inputs[-1],
                                                                  -1)

        # pass through hidden layers
        for h_layer in reversed(range(self.n_hidden_layers)):
            grad_pre_activation = self.gradient_hidden(self.weights[h_layer + 1],
                                                       grad_pre_activation,
                                                       pre_activations[h_layer],
                                                       self.activations_derivatives_map[self.hidden_layers_activations[h_layer]])
            grad_weights[h_layer], grad_biases[h_layer] = self.gradient_weights(grad_pre_activation,
                                                                                inputs[h_layer],
                                                                                h_layer)

        return grad_weights, grad_biases

    def gradient_out(self, outputs, labels):
        """
        Compute the gradient w.r.t. the pre-activation values of the output layer.
        :param outputs: the softmax values (the outputs of the nnet), ndarray (n_output x minibatch_size)
        :param labels: real labels, ndarray (1 x minibatch_size)
        :return: the gradient w.r.t. the pre-activation of the output layer, ndarray (n_output x minibatch_size)
        """
        return outputs - self.one_hot_labels_representation(labels)

    def gradient_hidden(self, W_kp1, grad_a_kp1, a_k, activation_function_derivative_k):
        """
        Compute the gradient w.r.t. the pre-activation function of the current layer k.
        :param W_kp1: the weights of the k+1 layer, ndarray (n_output_kp1 x n_input_kp1 = n_output_kp1 x n_output_k)
        :param grad_a_kp1: gradient w.r.t the pre-activation of the k+1 layer (n_output_kp1 x minibatch_size)
        :param a_k: the pre-activation of the layer k (n_output_k x minibatch_size)
        :param activation_function_derivative_k: the derivative of the activation function of layer k, handle
        :return: the gradient w.r.t the pre activation of the layer k (n_output_k x minibatch_size)
        """
        # compute gradient of the pre-activation function of the previous layer w.r.t the activation of current layer
        grad_y_k = np.dot(W_kp1.T, grad_a_kp1)

        # compute gradient of the pre-activation function of the previous layer w.r.t the pre-activation of current layer
        grad_a_k = grad_y_k * activation_function_derivative_k(a_k)
        return grad_a_k

    def gradient_weights(self, grad_a, x, i_layer):
        """
        Compute the gradient w.r.t. the parameters.
        :param grad_a: the gradient of the loss w.r.t the pre-activation, ndarray (n_output x minibatch_size)
        :param x: the input data, ndarray (n_input x minibatch_size)
        :param i_layer: index of the layer which gradient are being computed
        :return: the gradient w.r.t. the parameters, ndarray (n_output x n_input), ndarray (n_output, 1)
        """
        minibatch_size = x.shape[1]
        n_output = grad_a.shape[0]
        n_input = x.shape[0]
        grad_w = np.zeros((n_output, n_input))
        grad_b = np.zeros((n_output, 1))

        # Adding the gradient w.r.t for each example of the minibatch
        for i_sample in range(minibatch_size):
            grad_b += np.transpose(np.atleast_2d(grad_a[:, i_sample]))
            grad_w += np.outer(grad_a[:, i_sample], x[:, i_sample])

        # Regularization
        if self.reg_lambda != 0.0:
            reg_w = self.reg_lambda*self.weights[i_layer]
            reg_b = self.reg_lambda*self.biases[i_layer]

            # grad_theta = dL/dtheta + reg_lambda*theta
            grad_b += reg_b
            grad_w += reg_w

        grad_b /= minibatch_size
        grad_w /= minibatch_size

        return grad_w, grad_b

    def update_weights(self, W, b, grad_w, grad_b, eta):
        """
        Update the parameters with an SGD update rule
        :param W: the weights, ndarray
        :param b: the bias, ndarray
        :param grad_w: the gradient w.r.t. the weights, ndarray
        :param grad_b: the gradient w.r.t. the bias, ndarray
        :param eta: the step-size, float
        :return: the updated parameters, ndarray, ndarray
        """
        return W - eta * grad_w, b - eta * grad_b
