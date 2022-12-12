import numpy as np
import numpy.typing as npt

from model.model_utils import softmax, relu, relu_prime
from typing import Tuple


class NeuralNetwork(object):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_classes: int,
            seed: int = 1
    ):
        """
        Initialize neural network's weights and biases.
        """
        ############################# STUDENT SOLUTION ####################
        # YOUR CODE HERE
        #     TODO:
        #         1) Set a seed so that your model is reproducible
        #         2) Initialize weight matrices and biases with uniform
        #         distribution in the range (-1, 1).
        # INITIALIZING WEIGHTS, BIASES AND OUTPUT MATRICES
        np.random.seed(seed)
        self.W1 = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.W2 = np.random.uniform(-1, 1, (num_classes, hidden_size))
        self.b1 = np.random.uniform(-1, 1)
        self.b2 = np.random.uniform(-1, 1)
        self.Z1 = None
        self.Z2 = None
        self.A0 = None
        self.A1 = None
        self.A2 = None
        pass
        ###################################################################

    def forward(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """
        Forward pass with X as input matrix, returning the model prediction
        Y_hat.
        """
        ######################### STUDENT SOLUTION #########################
        # YOUR CODE HERE
        #     TODO:
        #         1) Perform only a forward pass with X as input.

        # ADDING BIAS TERM IN W AND A0 MATRICES
        self.A0 = X
        A0_add = np.ones((1, len(self.A0[0]))) * self.b1
        W1_add = np.ones((len(self.W1), 1)) * self.b1
        A0_temp = np.vstack((self.A0, A0_add))
        W1_temp = np.hstack((self.W1, W1_add))

        # CALCULATING Z1 AND A1 (USING RELU ACTIVATION FUNCTION)
        self.Z1 = np.dot(W1_temp, A0_temp)
        self.A1 = relu(self.Z1)

        # ADDING BIAS TERM TO A1 AND W2
        A1_add = np.ones((1, len(self.A1[0]))) * self.b2
        W2_add = np.ones((len(self.W2), 1)) * self.b2
        A1_temp = np.vstack((self.A1, A1_add))
        W2_temp = np.hstack((self.W2, W2_add))

        # CALCULATING Z2 AND A2 (USING SOFTMAX ACTIVATION FUNCTION)
        self.Z2 = np.dot(W2_temp, A1_temp)
        self.A2 = softmax(self.Z2)

        # PREDICTED PROBABILITY DISTRIBUTION FOR EACH CLASS
        Yhat = self.A2

        return Yhat
        #####################################################################

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """
        Create a prediction matrix with `self.forward()`
        """
        ######################### STUDENT SOLUTION ###########################
        # YOUR CODE HERE
        #     TODO:
        #         1) Create a prediction matrix of the intent data using
        #         `self.forward()` function. The shape of prediction matrix
        #         should be similar to label matrix produced with
        #         `labels_matrix()`

        # A FORWARD PASS TO CALCULATE PREDICTED DISTRIBUTION
        Yhat = self.forward(X)
        for j in range(len(Yhat[0])):
            temp = []

            # GET THE INDEX OF PREDICTED CLASS
            for i in range(len(Yhat)):
                temp.append(Yhat[i][j])
            pred_class = max(temp)
            ind = temp.index(pred_class)

            # PLACE 1 AT INDEX OF PREDICTED CLASS AND 0 AT OTHERS
            for i in range(len(Yhat)):
                if i == ind:
                    Yhat[i][j] = 1
                else:
                    Yhat[i][j] = 0

        return Yhat
        ######################################################################

    def backward(
            self,
            X: npt.ArrayLike,
            Y: npt.ArrayLike
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """
        Backpropagation algorithm.
        """
        ########################## STUDENT SOLUTION ###########################
        # YOUR CODE HERE
        #     TODO:
        #         1) Perform forward pass, then backpropagation
        #         to get gradient for weight matrices and biases
        #         2) Return the gradient for weight matrices and biases

        # CALCULATE DELTA TERMS
        yhat = self.forward(X)
        del_L = np.subtract(yhat, Y)
        del_1 = np.dot(np.transpose(self.W2), relu_prime(self.Z2) * del_L)

        # CALCULATE GRADIENTS
        W1_gradient = np.dot(del_1, np.transpose(self.A0))
        W2_gradient = np.dot(del_L, np.transpose(self.A1))
        b1_gradient = np.average(del_1)
        b2_gradient = np.average(del_L)

        return W1_gradient, W2_gradient, b1_gradient, b2_gradient
        #######################################################################


def compute_loss(pred: npt.ArrayLike, truth: npt.ArrayLike) -> float:
    """
    Compute the cross entropy loss.
    """
    ########################## STUDENT SOLUTION ###########################
    # YOUR CODE HERE
    #     TODO:
    #         1) Compute the cross entropy loss between your model prediction
    #         and the ground truth.
    Loss = []
    for j in range(len(pred[0])):
        Li = 0
        for i in range(len(pred)):
            Li -= truth[i][j] * np.log(pred[i][j])
        Loss.append(Li)

    cost = np.average(Loss)
    return cost
    #######################################################################
