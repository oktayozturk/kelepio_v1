# -*- coding: utf-8 -*-
# !/usr/bin/python

import data_management as dm
import numpy as np
import tensorflow as tf

np.set_printoptions(suppress=True, precision=8, linewidth=150, threshold=10000)
epsilon = 1e-7

def linear_regression_model(bike_brand, bike_model):

    X_train, Y_train, X_test, Y_test = dm.preprocessData(bike_brand, bike_model, 0.9, polynomial_degree=2)


    tf.reset_default_graph()

    #----------------------- HyperParameters -----------------------------------------------

    n_train, m_train = np.shape(X_train)
    n_test, m_test = np.shape(X_test)
    number_of_output = np.shape(Y_test)[0]

    layers_dims = [n_train, 40, 30, number_of_output]
    epochs = 1000
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


    rms_error = computeCost(Y,Y_)
    regularisation = compute_L2_regularization(parameters, beta)
    cost = tf.reduce_mean(rms_error + regularisation)

    optimizer = tf.train.AdamOptimizer(alpha)
    train = optimizer.minimize(cost, name="Training_OP")

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        # ------------------------------ TRAINING SET ----------------------------------
        training_costs = []

        for i in range(epochs):

            for j in range(batches+1):

                x_batch, y_batch = getNextBatch(X_train, Y_train, j, batch_size)
                train_data = {X: x_batch, Y:y_batch}
                _, c = sess.run([train, cost], feed_dict=train_data)


            if i % 100 == 0:
                training_costs.append(c)
                alpha = decreaseAlpha(alpha,i)
                print("Cost after {} epochs: {}".format(i,c))

        x_batch, y_batch = getNextBatch(X_train, Y_train, 0, m_train)
        train_data = {X: x_batch, Y: y_batch}
        pred = sess.run(Y_, feed_dict=train_data)
        label = sess.run(Y, feed_dict=train_data)
        print("TRAINING METRICS:")
        computeMetrics(label, pred)

        #------------------------------------ TEST SET ----------------------------------

        test_costs = []

        for i in range(m_test):
            x_batch, y_batch = getNextBatch(X_test, Y_test, i, 1)
            test_data = {X: x_batch, Y: y_batch}

            c = sess.run(rms_error, feed_dict=test_data)
            test_costs.append(c)


        print("TEST/CROSSVALIDATION SET METRICS:")

        x_batch, y_batch = getNextBatch(X_test, Y_test, 0, m_test)
        test_data = {X: x_batch, Y: y_batch}

        pred = sess.run(Y_, feed_dict=test_data)
        label = sess.run(Y, feed_dict=test_data)

        print(np.shape(pred))
        print(np.shape(label))
        print(np.concatenate((np.exp(pred),np.exp(label)), axis=0))

        computeMetrics(label, pred)



        #plotPrices(pred, label)
        plotLearningCurve(training_costs, test_costs)


def mergeCV(x_test, y_test, y_preds):

    print(np.shape(x_test))
    print(np.shape(y_test))
    print(np.shape(y_preds))

    dm.writeCSV(np.concatenate((x_test, y_test, y_preds), axis=0))


def getNextBatch(X_train, Y_train, j, batch_size):

    #if batch_size > np.size(Y_train, axis=1) : batch_size = np.size(Y_train, axis=1)

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

    diff = 0
    for i in range(m):
        diff += np.sqrt((np.exp(Y[:,i]) - np.exp(Y_[:,i]))**2)

    diff = diff / (m + epsilon)


    print("Diff: {} TL".format(diff))



def computeCost(Y,Y_):

    return tf.losses.mean_squared_error(labels=Y, predictions=Y_)


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


def applyPCA(X, reduction):


    n,m = np.shape(X)

    # calculate means vector to calculate scatter matrix
    means_vector = np.zeros([n,1])

    for i in range(n):
        means_vector[i,:] = np.mean(X[i,:])

    #calculate scatter matrix to calculate eigen vector and eigen values
    scatter_matrix = np.zeros([n,n])

    for i in range(m):
        scatter_matrix = scatter_matrix + np.dot((X[:,i].reshape(n,1) - means_vector), (X[:,i].reshape(n,1) - means_vector).T)

    #calculate eigen vector and eigen values
    eigen_values, eigen_vectors = np.linalg.eig(np.array(scatter_matrix, dtype=float))

    #testing OUTPUT values
    # for i in range(len(eigen_values)):
    #
    #     eigv = eigen_vectors[:, i].reshape(1, n).T
    #     np.testing.assert_array_almost_equal(scatter_matrix.dot(eigv), eigen_values[i] * eigv,
    #                                          decimal=6, err_msg='', verbose=True)
    #
    # for ev in eigen_vectors:
    #     np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(len(eigen_values))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # Building PCA matrix
    pca_matrix = []
    for i in range(reduction):
        pca_matrix.append(eig_pairs[i][1].reshape(n, 1))

    w = np.hstack(tuple(pca_matrix))

    return w.T.dot(X)



linear_regression_model("honda","cbr 250 r")



