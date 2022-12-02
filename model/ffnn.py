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
        np.random.seed(seed)
        self.W1 = np.random.uniform(-1, 1, (hidden_size, input_size+1))
        self.W2 = np.random.uniform(-1, 1, (num_classes, hidden_size+1))
        self.b1 = np.random.uniform(-1, 1)
        self.b2 = np.random.uniform(-1, 1)
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

        # ADDING X0 TERM (BIAS TERM) IN X MATRIX
        A0 = np.ones((1, len(X[0])))
        A0 = A0 * self.b1
        for i in range(len(X)):
            A0 = np.vstack((A0, np.array(X[i])))

        # CALCULATING Z1 = W1 * A0
        Z1 = np.dot(self.W1, A0)

        # ACTIVATION FUNCTION
        A1_temp = relu(Z1)

        # ADDING X0 TERM (BIAS TERM) IN A1_temp
        A1 = np.ones((1, len(A1_temp[0])))
        A1 = A1 * self.b2
        for i in range(len(A1_temp)):
            A1 = np.vstack((A1, np.array(A1_temp[i])))

        # CALCULATING Z2 = W2 * A1
        Z2 = np.dot(self.W2, A1)

        # ACTIVATION FUNCTION
        Yhat = softmax(Z2)

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
        Yhat = self.forward(X)
        for j in range(len(Yhat[0])):
            temp = []
            for i in range(len(Yhat)):
                temp.append(Yhat[i][j])
            pred_class = max(temp)
            ind = temp.index(pred_class)

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
        pass
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
    pass
    #######################################################################