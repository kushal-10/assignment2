from model.ffnn import NeuralNetwork, compute_loss
import numpy as np
import matplotlib.pyplot as plt


def batch_train(X, Y, model, train_flag=False):
    ################################# STUDENT SOLUTION #############################
    # YOUR CODE HERE
    #     TODO:
    #         1) Use your neural network to predict the intent
    #         (without any training) and calculate the accuracy 
    #         of the classifier. Should you be expecting high
    #         numbers yet?
    #         2) if train_flag is true, run the training for 1000 epochs using 
    #         learning rate = 0.005 and use this neural network to predict the 
    #         intent and calculate the accuracy of the classifier
    #         3) Then, plot the cost function for each iteration and
    #         compare the results after training with results before training

    # PRINT INITIAL ACCURACY (WITHOUT TRAINING)
    acc = get_accuracy(model, X, Y)
    print("Initial Accuracy : " + str(acc))

    # TRAINING THE MODEL
    if train_flag:
        learning_rate = 0.005
        cost = []
        for i in range(1000):
            # UPDATE THE WEIGHTS OF THE MODEL
            update_weights(X, Y, model, learning_rate, len(X[0]))

            # CALCULATE COST FOR EACH ITERATION AND STORE
            pred = model.forward(X)
            cost_temp = compute_loss(pred, Y)
            cost.append(cost_temp)

        # PRINT THE ACCURACY AFTER TRAINING
        accuracy = get_accuracy(model, X, Y)
        print("Accuracy after training : " + str(accuracy))

        # PLOT THE GRAPH OF COST V/S ITERATION
        plot(cost, "Batch", "Cost")

    return None
    ###############################################################################


def minibatch_train(X, Y, model, train_flag=False):
    ########################## STUDENT SOLUTION #############################
    # YOUR CODE HERE
    #     TODO:
    #         1) As bonus, train your neural network with batch size = 64
    #         and SGD (batch size = 1) for 1000 epochs using learning rate
    #         = 0.005. Then, plot the cost vs iteration for both cases.

    # FOR MINI BATCH
    # SPLIT DATA INTO BATCHES OF 64 EACH
    X_batches, Y_batches = split_data(X, Y, batch_size=64)

    # GET THE COST FOR EACH ITERATION WHILE UPDATING WEIGHTS
    cost_minibatch = train_batches(X_batches, Y_batches, model, learning_rate=0.005)

    # PRINT THE ACCURACY AFTER TRAINING
    accuracy = get_accuracy(model, X, Y)
    print("Accuracy for minibatch : " + str(accuracy))

    # FOR STOCHASTIC
    # SPLIT DATA INTO BATCHES OF 1 EACH
    X_batches, Y_batches = split_data(X, Y, batch_size=1)

    # GET THE COST FOR EACH ITERATION WHILE UPDATING WEIGHTS
    cost_stochastic = train_batches(X_batches, Y_batches, model, learning_rate=0.005)

    # PRINT THE ACCURACY AFTER TRAINING
    accuracy = get_accuracy(model, X, Y)
    print("Accuracy for stochastic : " + str(accuracy))

    # PLOT THE GRAPH OF COST V/S EPOCH FOR MINIBATCH
    plot(cost_minibatch, "Minibatch", "Cost")

    # PLOT THE GRAPH OF COST V/S EPOCH FOR STOCHASTIC
    plot(cost_stochastic, "Stochastic", "Cost")

    return None
    #########################################################################


def train_batches(X1, Y1, model, learning_rate):
    cost_list = []
    # TRAIN FOR 1000 EPOCHS
    for i in range(1000):
        # FOR INDIVIDUAL BATCH
        for j in range(len(X1)):
            # UPDATE THE WEIGHTS OF THE MODEL
            update_weights(X1[j], Y1[j], model, learning_rate, len(X1[j][0]))

            # CALCULATE COST FOR EACH ITERATION AND STORE
            pred = model.forward(X1[j])
            cost = compute_loss(pred, X1[j])
            cost_list.append(cost)


    return cost_list


def update_weights(D, E, model, learning_rate, l):
    # BACK PROPAGATE AND UPDATE THE WEIGHTS
    (del_W1, del_W2, del_b1, del_b2) = model.backward(D, E)
    model.W1 -= learning_rate * del_W1 / l
    model.W2 -= learning_rate * del_W2 / l
    model.b1 -= learning_rate * del_b1 / l
    model.b2 -= learning_rate * del_b2 / l


def get_accuracy(model, X, Y):
    # GET THE ACCURACY OF MODEL GIVEN X AND Y
    correct = 0
    Yhat = model.predict(X)
    for i in range(len(Y)):
        for j in range(len(Y[0])):
            if Yhat[i][j] == 1 and Y[i][j] == 1:
                correct += 1
    accuracy = float(correct) / len(Y[0])

    return accuracy


def split_data(M1, M2, batch_size):
    # GET THE INDICES FOR SPLITTING
    indices = np.arange(batch_size, M1.shape[1], batch_size)

    # SPLITTING X(M1) AND Y(M2) MATRICES
    M1_batches = np.array_split(M1, indices, axis=1)
    M2_batches = np.array_split(M2, indices, axis=1)

    return M1_batches, M2_batches


def plot(cost, type, value):
    # PLOTTING THE GRAPHS
    x_axis = np.arange(0, len(cost), 1)
    plt.plot(x_axis, cost)
    plt.xlabel("Iterations")
    plt.ylabel(value + " per iteration")
    plt.title(value + " v/s Iteration for " + type)
    plt.show()

    return None
