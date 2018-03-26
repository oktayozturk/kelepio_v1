# -*- coding: utf-8 -*-
# !/usr/bin/python

import data_management as dm
import numpy as np
import tensorflow as tf

np.set_printoptions(suppress=True, precision=8, linewidth=150, threshold=10000)
epsilon = 1e-7

def softmax_classier_model(bike_brand, bike_model):

    X_train, Y_train, X_test, Y_test = dm.preprocessData(bike_brand, bike_model, 0.9, polynomial_degree=1)

    tf.reset_default_graph()

    #----------------------- HyperParameters -----------------------------------------------

    n_train, m_train = np.shape(X_train)
    n_test, m_test = np.shape(X_test)
    number_of_output = np.shape(Y_test)[0]

    layers_dims = [n_train, 60, 40, 30, number_of_output]
    epochs = 5000
    batch_size = 64
    batches = int(np.ceil(m_train / batch_size))
    alpha = 0.001
    beta = 0.1
    dropout_threshold = 0.5
    logs_path = "Tensor_logs/"
    sample_size = m_train


    #------------------------- PREDICTIVE MODEL -------------------------------------

    X = tf.placeholder(dtype=tf.float32, shape=[n_train,None], name="X")
    Y = tf.placeholder(dtype=tf.float32, shape=[number_of_output, None], name="Y")
    parameters = initWeights(layers_dims)
    Y_ = feedForward(X, parameters, dropout_threshold)


    loss = computeCost(Y,Y_)
    regularisation = compute_L2_regularization(parameters, beta)
    cost = tf.reduce_mean(loss + regularisation)

    optimizer = tf.train.AdamOptimizer(alpha)
    train = optimizer.minimize(cost, name="Training_OP")

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        # ------------------------------ TRAINING SET ----------------------------------
        training_costs = []

        for i in range(epochs):

            for j in range(batches):

                x_batch, y_batch = getNextBatch(X_train, Y_train, j, batch_size)
                train_data = {X: x_batch, Y:y_batch}
                _, c = sess.run([train, cost], feed_dict=train_data)

            if i % 100 == 0:
                training_costs.append(c)
                alpha = decreaseAlpha(alpha,i)

        x_batch, y_batch = getNextBatch(X_train, Y_train, 0, m_train)
        train_data = {X: x_batch, Y: y_batch}
        pred = sess.run(Y_, feed_dict=train_data)
        label = sess.run(Y, feed_dict=train_data)
        print("Train metrics:")
        computeMetrics(label, pred)

        #------------------------------------ TEST SET ----------------------------------

        test_costs = []

        for i in range(m_test):
            x_batch, y_batch = getNextBatch(X_test, Y_test, i, 2)
            test_data = {X: x_batch, Y: y_batch}

            c = sess.run(loss, feed_dict=test_data)
            test_costs.append(c)


        print("Test metrics:")

        x_batch, y_batch = getNextBatch(X_test, Y_test, 0, m_test)
        test_data = {X: x_batch, Y: y_batch}

        pred = sess.run(Y_, feed_dict=test_data)
        label = sess.run(Y, feed_dict=test_data)
        computeMetrics(label, pred)

        #plotPrices(pred, label)
        plotLearningCurve(training_costs, test_costs)



def getNextBatch(X_train, Y_train, j, batch_size):

    if j == 0:
        x = X_train[:, 0:batch_size ]
        y = Y_train[:, 0:batch_size]

    elif (j * batch_size) < len(X_train[0]):
        x = X_train[:, (j*batch_size):((j+1)*(batch_size))]
        y = Y_train[:, (j*batch_size):((j+1)*(batch_size))]

    else:
        x = X_train[:, (j * batch_size):]
        y = Y_train[:, (j * batch_size):]


    return [x,y]


def plotPrices(pred, label):

    import matplotlib.pyplot as plt

    for i in range(len(pred)):
        print(pred[:,i])
        plt.plot(pred[:,i])
        plt.show()


def compute_L2_regularization(parameters, beta):
    reg = 0

    for param in parameters:
        reg += tf.nn.l2_loss(parameters[param])
    reg = reg * beta

    return reg


def computeMetrics(Y,Y_):
    n, m = np.shape(Y)

    true_positives, false_positives, true_negatives, false_negatives = 0, 0, 0, 0

    for i in range(m):
        if np.argmax(Y[:,i]) == np.argmax(Y_[:,i]):
            true_positives += 1
        else:
            false_positives +=1


    print("True positives: {}".format(true_positives))
    print("False positives: {}".format(false_positives))

    Precision = true_positives / (true_positives + false_positives + epsilon)
    Recall = true_positives / (true_positives + false_negatives + epsilon)
    F1 = (2 * Precision * Recall) / (Precision + Recall + epsilon)
    Accuracy = (100 * true_positives) / (m + epsilon)

    print("Accuracy: % {}".format(Accuracy))
    print("F1 Score: {}".format(F1))


    return [F1, Precision, Recall, Accuracy ]


def computeCost(Y,Y_):

    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.transpose(Y), logits=tf.transpose(Y_)))


def initWeights(layers_dims):

    parameters = {}

    for i,val in enumerate(layers_dims[1:], start=1):

        parameters["W" + str(i)] = tf.Variable(tf.truncated_normal([val, layers_dims[i-1]], stddev=0.1, dtype=tf.float32), name="W"+str(i))
        parameters["b" + str(i)] = tf.Variable(tf.constant(0.1, shape=[val,1]), name="b" + str(i))

    return parameters


def feedForward(X, parameters, keep_prob):

    steps = len(parameters) /2

    activations = {"a0": X}

    for i in range(steps-1):
        z = tf.add(tf.matmul(parameters["W"+str(i+1)], activations["a"+str(i)]), parameters["b"+str(i+1)])
        if keep_prob < 1:
            z = tf.nn.dropout(z, keep_prob=keep_prob)
        activations["a"+str(i+1)] = tf.nn.relu(z, name="a"+str(i+1))

    Y_ = tf.add(tf.matmul(parameters["W" + str(steps)], activations["a"+str(steps-1)]), parameters["b"+str(steps)], name="Y_")

    return Y_


def plotLearningCurve(training_costs, test_costs):
    import matplotlib.pyplot as plt
    import math

    training_costs = [cost if not math.isnan(cost) else 0 for cost in training_costs]
    test_costs = [cost if not math.isnan(cost) else 0 for cost in test_costs]

    #print(training_costs)

    plt.plot(range(len(training_costs)), training_costs, 'c-')
    plt.plot(range(len(test_costs)), test_costs, 'r-')
    plt.title("Learning curves")
    plt.show()


# --------------------- Manually calculated cost --------------------
def computeCostManual(Y, Y_):

    t = np.exp(Y_)
    t = t / (np.sum(t,axis=0) + epsilon)
    m = len(Y)
    costs = []
    for i in range(m):
        costs.append(Y[i] * np.log(t[i]))

    total_cost = (np.sum(costs)/(m + epsilon)) * (-1)

    return total_cost


def decreaseAlpha(alpha,t):
    alpha = tf.train.exponential_decay(alpha, t, 100000, 0.5, staircase=True)

    return alpha


softmax_classier_model("honda","cbf 150")



