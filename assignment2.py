import argparse
import numpy as np

from utils import load_dataset
from model.model_utils import bag_of_words_matrix, labels_matrix, softmax, relu, relu_prime
from model.ffnn import NeuralNetwork
from helper import batch_train, minibatch_train


DATA_PATH = './data/dataset.csv'


def main():
    parser = argparse.ArgumentParser(
        description='Train feedforward neural network'
    )

    parser.add_argument(
        '--minibatch', dest='minibatch',
        help='Train feedforward neural network with mini-batch gradient descent/SGD',
        action='store_true'
    )

    parser.add_argument(
        '--train', dest='train',
        help='Turn on this flag when you are ready to train the model with backpropagation.',
        action='store_true'
    )

    args = parser.parse_args()

    sentences, intent, unique_intent = load_dataset(DATA_PATH)

    ############################ STUDENT SOLUTION ####################
    # YOUR CODE HERE
    #     TODO:
    #         1) Convert the sentences and intent to matrices using
    #         `bag_of_words_matrix()` and `labels_matrix()`.
    #         2) Initiallize the model Class with the appropriate parameters

    X = np.array(bag_of_words_matrix(sentences))
    Y = np.array(labels_matrix((intent, unique_intent)))

    model = NeuralNetwork(input_size=len(X), hidden_size=150, num_classes=len(Y))
    ##################################################################

    if not args.minibatch:
        print("Training FFNN using batch gradient descent...")
        batch_train(X, Y, model, train_flag=args.train)
    else:
        print("Training FFNN using mini-batch gradient descent...")
        minibatch_train(X, Y, model, train_flag=args.train)


if __name__ == "__main__":
    main()
    '''
    QUESTIONS
    1 - BOW MATRIX/L Matrix - 1 AND 0s for every word occurrence in data, like one hot encoding?
    2 - UNK COUNT UPDATE AFTER VxM OR BEFORE? UNK tokens counted in vocab or not
    3 - Difference between U and W2
    4 - any improvements using stochastic/batch GD
    
    A - is order of Y matrix important?
    B - cant use relu/prime together?
    
    -Representation Learning
    '''
