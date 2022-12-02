from model.ffnn import NeuralNetwork, compute_loss


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
    for i in range(len(Y)):
        for j in range(len(Y[0])):
            if Yhat[i][j] == 1 and Y[i][j] == 1:
                correct += 1

    accuracy = float(correct)/data_size
    print("Accuracy : " + str(accuracy))

    if train_flag:
        pass
    ###############################################################################


def minibatch_train(X, Y, model, train_flag=False):
    ########################## STUDENT SOLUTION #############################
    # YOUR CODE HERE
    #     TODO:
    #         1) As bonus, train your neural network with batch size = 64
    #         and SGD (batch size = 1) for 1000 epochs using learning rate
    #         = 0.005. Then, plot the cost vs iteration for both cases.
    pass
    if train_flag:
        pass
    #########################################################################
