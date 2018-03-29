# -*- coding: utf-8 -*-
# !/usr/bin/python


import numpy as np
import tensorflow as tf
from data_management import datamanager as dm

np.set_printoptions(suppress=True, precision=8, linewidth=150, threshold=10000)


class linear_regression_model(object):

    epsilon = 1e-7
    epochs = 5000
    batch_size = 64
    dropout_threshold = 0.5
    alpha = 0.001
    beta = 0.1


    def __init__(self, bike_brand, bike_model):

        self.b = dm(bike_brand, bike_model, polynomial_degree=1)

        self.b.clear_uncorrelated_fields()

        tf.reset_default_graph()

        self.X_train, self.Y_train, self.X_test, self.Y_test = self.b.splitDataset(0.9)

        #----------------------- HyperParameters -----------------------------------------------

        self.n_train, self.m_train = np.shape(self.X_train)
        self.n_test, self.m_test = np.shape(self.X_test)
        self.number_of_output = np.shape(self.Y_test)[0]
        self.layers_dims = [self.n_train, 40, 30, self.number_of_output]
        self.batches = int(np.ceil(self.m_train / self.batch_size))
        self.logs_path = "Tensor_logs/"
        self.sample_size = self.m_train


    #------------------------- PREDICTIVE MODEL -------------------------------------


        self.X = tf.placeholder(dtype=tf.float32, shape=[self.n_train,None], name="X")

        self.Y = tf.placeholder(dtype=tf.float32, shape=[self.number_of_output, None], name="Y")

        self.parameters = self.initWeights()

        self.Y_ = self.feedForward()

        self.dryLoss = self.computeUnregularizedLoss()

        self.regularisation = self.compute_L2_regularization()

        self.cost = tf.reduce_mean(self.dryLoss + self.regularisation)

        self.optimizer = tf.train.AdamOptimizer(self.alpha)

        self.train_op = self.optimizer.minimize(self.cost, name="Training_OP")

        self.initializer = tf.global_variables_initializer()



    def train(self):

        with tf.Session() as sess:

            sess.run(self.initializer)

            # ------------------------------ TRAINING SET ----------------------------------
            training_costs = []

            for i in range(self.epochs):

                for j in range(self.batches+1):

                    x_batch, y_batch = self.getNextBatch(j)
                    train_data = {self.X: x_batch, self.Y:y_batch}
                    _, c = sess.run([self.train_op, self.cost], feed_dict=train_data)


                if i % 100 == 0:
                    training_costs.append(c)
                    self.decreaseAlpha(i)
                    print("Cost after {} epochs: {}".format(i,c))

            # x_batch, y_batch = self.X_train, self.Y_train
            # train_data = {self.X: x_batch, self.Y: y_batch}
            # pred = sess.run(self.Y_, feed_dict=train_data)
            # label = sess.run(self.Y, feed_dict=train_data)
            # print("TRAINING METRICS:")
            # self.computeMetrics(label, pred)
            #


    def test(self):

        with tf.Session() as sess:
            sess.run(self.initializer)
        #------------------------------------ TEST SET ----------------------------------

            test_costs = []

            for i in range(self.m_test):
                x_batch, y_batch = self.getNextBatch(self.X_test, self.Y_test, i, 1)
                test_data = {self.X: x_batch, self.Y: y_batch}

                c = sess.run(self.cost, feed_dict=test_data)
                test_costs.append(c)


            # print("TEST/CROSSVALIDATION SET METRICS:")
            #
            # x_batch, y_batch = getNextBatch(X_test, Y_test, 0, m_test)
            # test_data = {X: x_batch, Y: y_batch}
            #
            # pred = sess.run(Y_, feed_dict=test_data)
            # label = sess.run(Y, feed_dict=test_data)
            #
            # print(np.shape(pred))
            # print(np.shape(label))
            # print(np.concatenate((np.exp(pred),np.exp(label)), axis=0))
            #




    def initWeights(self):

        parameters = {}

        for i, val in enumerate(self.layers_dims[1:], start=1):
            parameters["W" + str(i)] = tf.Variable(
                tf.truncated_normal([val, self.layers_dims[i - 1]], stddev=0.1, dtype=tf.float32), name="W" + str(i))
            parameters["b" + str(i)] = tf.Variable(tf.constant(0.1, shape=[val, 1]), name="b" + str(i))

        return parameters


    def feedForward(self):

        steps = len(self.parameters) / 2

        activations = {"a0": self.X}

        for i in range(steps-1):

            z = tf.add(tf.matmul(self.parameters["W"+str(i+1)], activations["a"+str(i)]), self.parameters["b"+str(i+1)])

            if self.dropout_threshold < 1:
                z = tf.nn.dropout(z, keep_prob=self.dropout_threshold)
            activations["a"+str(i+1)] = tf.nn.relu(z, name="a"+str(i+1))

        Y_ = tf.add(tf.matmul(self.parameters["W" + str(steps)], activations["a"+str(steps-1)]), self.parameters["b"+str(steps)], name="Y_")

        return Y_


    def computeUnregularizedLoss(self):

        return tf.losses.mean_squared_error(labels=self.Y, predictions=self.Y_)


    def decreaseAlpha(self, t):

        self.alpha = tf.train.exponential_decay(self.alpha, t, 100000, 0.5, staircase=True)

        return self.alpha


    def getNextBatch(self, j):

        #if batch_size > np.size(Y_train, axis=1) : batch_size = np.size(Y_train, axis=1)

        if j == 0:
            x = self.X_train.iloc[:, 0:self.batch_size ]
            y = self.Y_train.iloc[:, 0:self.batch_size]

        elif (j * self.batch_size) < len(self.X):
            x = self.X_train.iloc[:, (j*self.batch_size):((j+1)*(self.batch_size))]
            y = self.Y_train.iloc[:, (j*self.batch_size):((j+1)*(self.batch_size))]

        else:
            x = self.X_train.iloc[:, (j * self.batch_size):]
            y = self.Y_train.iloc[:, (j * self.batch_size):]


        return [x,y]


    def computeMetrics(self):

        n, m = np.shape(self.Y)

        diff = 0
        for i in range(m):
            diff += np.sqrt((np.exp(self.Y.iloc[:, i]) - np.exp(self.Y_.iloc[:, i])) ** 2)

        diff = diff / (m + self.epsilon)

        print("Diff: {} TL".format(diff))


    def compute_L2_regularization(self):
        reg = 0

        for param in self.parameters:
            reg += tf.nn.l2_loss(self.parameters[param])
        reg = reg * self.beta

        return reg



    @staticmethod
    def plotLearningCurve(training_costs, test_costs):
        import matplotlib.pyplot as plt
        import math

        training_costs = [cost if not math.isnan(cost) else 0 for cost in training_costs]
        test_costs = [cost if not math.isnan(cost) else 0 for cost in test_costs]

        # print(training_costs)

        plt.plot(range(len(training_costs)), training_costs, 'c-')
        plt.plot(range(len(test_costs)), test_costs, 'r-')
        plt.title("Learning curves")
        plt.show()







