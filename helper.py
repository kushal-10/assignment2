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

    Yhat = model.predict(X)
    data_size = len(X[0])
    correct = 0
    correct2 = 0
    for i in range(len(Y)):
        for j in range(len(Y[0])):
            if Yhat[i][j] == 1 and Y[i][j] == 1:
                correct += 1

    accuracy = float(correct) / data_size
    print("Initial Accuracy : " + str(accuracy))

    if train_flag:
        learning_rate = 0.005
        cost = []
        for i in range(1000):
            (del_W1, del_W2, del_b1, del_b2) = model.backward(X, Y)
            model.W1 -= learning_rate * del_W1 / data_size
            model.W2 -= learning_rate * del_W2 / data_size
            model.b1 -= learning_rate * del_b1 / data_size
            model.b2 -= learning_rate * del_b2 / data_size

            pred = model.forward(X)
            loss = compute_loss(pred, Y)
            cost.append(loss)

        Yhat_trained = model.predict(X)
        for i in range(len(Y)):
            for j in range(len(Y[0])):
                if Yhat_trained[i][j] == 1 and Y[i][j] == 1:
                    correct2 += 1

        accuracy2 = float(correct2) / data_size
        print("Accuracy after training : " + str(accuracy2))

        x_axis = np.arange(0, len(cost), 1)
        plt.plot(x_axis, cost)
        plt.xlabel("Iterations")
        plt.ylabel("Cost per iteration")
        plt.title("Cost v/s Iteration for Batch Training ")
        plt.show()

    return None
    ###############################################################################


def minibatch_train(X, Y, model, train_flag=False):
    ########################## STUDENT SOLUTION #############################
    # YOUR CODE HERE
    #     TODO:
    #         1) As bonus, train your neural network with batch size = 64
    #         and SGD (batch size = 1) for 1000 epochs using learning rate
    #         = 0.005. Then, plot the cost vs iteration for both cases.

    learning_rate = 0.005
    # FOR BATCH TRAINING FOR 64
    data_size = len(X[0])
    batch_size = 64
    indices = np.arange(batch_size, X.shape[1], batch_size)
    X_batches = np.array_split(X, indices, axis=1)
    Y_batches = np.array_split(Y, indices, axis=1)

    cost_batch = []
    for i in range(1000):
        for j in range(len(X_batches)):
            (del_W1, del_W2, del_b1, del_b2) = model.backward(X_batches[j], Y_batches[j])
            model.W1 -= learning_rate * del_W1 / data_size
            model.W2 -= learning_rate * del_W2 / data_size
            model.b1 -= learning_rate * del_b1 / data_size
            model.b2 -= learning_rate * del_b2 / data_size

        pred_batch = model.forward(X)
        loss_batch = compute_loss(pred_batch, Y)
        cost_batch.append(loss_batch)

    correct = 0
    Yhat_batch = model.predict(X)
    for i in range(len(Y)):
        for j in range(len(Y[0])):
            if Yhat_batch[i][j] == 1 and Y[i][j] == 1:
                correct += 1
    accuracy = float(correct) / data_size
    print("Batch Training Accuracy : " + str(accuracy))

    # FOR STOCHASTIC GD
    data_size = len(X[0])
    batch_size = 1
    indices = np.arange(batch_size, X.shape[1], batch_size)
    X_batches = np.array_split(X, indices, axis=1)
    Y_batches = np.array_split(Y, indices, axis=1)
    # print(len(X_batches))

    cost_stochastic = []
    for i in range(1000):
        for j in range(len(X_batches)):
            (del_W1, del_W2, del_b1, del_b2) = model.backward(X_batches[j], Y_batches[j])
            model.W1 -= learning_rate * del_W1 / data_size
            model.W2 -= learning_rate * del_W2 / data_size
            model.b1 -= learning_rate * del_b1 / data_size
            model.b2 -= learning_rate * del_b2 / data_size

        pred_stochastic = model.forward(X)
        loss_stochastic = compute_loss(pred_stochastic, Y)
        cost_stochastic.append(loss_stochastic)

    correct = 0
    Yhat_batch = model.predict(X)
    for i in range(len(Y)):
        for j in range(len(Y[0])):
            if Yhat_batch[i][j] == 1 and Y[i][j] == 1:
                correct += 1
    accuracy = float(correct) / data_size
    print("Stochastic Training Accuracy : " + str(accuracy))

    x_axis_b = np.arange(0, len(cost_stochastic), 1)
    plt.plot(x_axis_b, cost_stochastic)
    plt.xlabel("Iterations")
    plt.ylabel("Cost per iteration")
    plt.title("Cost v/s Iteration for Mini Batch Gradient Descent ")
    plt.show()

    x_axis_s = np.arange(0, len(cost_batch), 1)
    plt.plot(x_axis_s, cost_batch)
    plt.xlabel("Iterations")
    plt.ylabel("Cost per iteration")
    plt.title("Cost v/s Iteration for Stochastic Gradient Descent ")
    plt.show()

    return None
    #########################################################################
