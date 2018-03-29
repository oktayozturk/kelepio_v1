# -*- coding: utf-8 -*-
# !/usr/bin/python


import numpy as np
import tensorflow as tf
from data_management import datamanager as dm

np.set_printoptions(suppress=True, precision=8, linewidth=150, threshold=10000)


class linear_regression_model(object):

    # ----------------------- HyperParameters -----------------------------------------------

    epsilon = 1e-7
    epochs = 10000
    batch_size = 64
    dropout_threshold = 1
    alpha = 0.001
    beta = 0.1


    def __init__(self, bike_brand, bike_model):

        # ----------------------- Data Gathering -----------------------------------------------
        self.bike_data_manager = dm(bike_brand, bike_model, polynomial_degree=1)
        self.bike_data_manager.clear_uncorrelated_fields()
        self.X_train, self.Y_train, self.X_test, self.Y_test = self.bike_data_manager.splitDataset(0.9)


        #----------------------- Model Parameters -----------------------------------------------

        self.n_train, self.m_train = np.shape(self.X_train)
        self.n_test, self.m_test = np.shape(self.X_test)
        self.number_of_output = np.shape(self.Y_test)[0]
        self.layers_dims = [self.n_train, 40, 30, self.number_of_output]
        self.batches = int(np.ceil(self.m_train / self.batch_size))
        self.sample_size = self.m_train


        #-------------------------  MODEL GRAPH -------------------------------------
        with tf.name_scope("Variables"):

            self.X = tf.placeholder(dtype=tf.float32, shape=[self.n_train,None], name="X")

            self.Y = tf.placeholder(dtype=tf.float32, shape=[self.number_of_output, None], name="Y")

            self.parameters = self.initWeights()

        with tf.name_scope("Feed_forward"):
            self.Y_ = self.feedForward()

        with tf.name_scope("Cost_computation"):
            self.dryLoss = self.computeUnregularizedLoss()

            self.regularisation = self.compute_L2_regularization()

            self.cost = tf.reduce_mean(self.dryLoss + self.regularisation)

        with tf.name_scope("Optimization"):

            self.optimizer = tf.train.AdamOptimizer(self.alpha)

            self.train_op = self.optimizer.minimize(self.cost, name="Training_OP")

            self.training_costs = []

            self.test_costs = []



            # ----------------------- Tensorflow  Configs-----------------------------------------------
            tf.summary.scalar("Cost", self.cost)
            #tf.summary.scalar("W1", self.parameters["W1"].value())

            self.initializer = tf.global_variables_initializer()

            #tf.reset_default_graph()

            self.summary_op = tf.summary.merge_all()

            self.tensor_logs_path = "./Tensor_logs/"
            self.tf_writer = tf.summary.FileWriter(self.tensor_logs_path, graph=tf.get_default_graph())

            self.tensor_save_path = "./Tensor_models/"
            self.tf_saver = tf.train.Saver()


    def train(self, verbose=True, save=False):

        with tf.Session() as sess:

            sess.run(self.initializer)


            # ------------------------------ TRAINING SET ----------------------------------
            training_costs = []

            for i in range(self.epochs):

                for j in range(self.batches+1):

                    x_batch, y_batch = self.getNextBatch("train", j, self.batch_size)
                    train_data = {self.X: x_batch, self.Y:y_batch}
                    _, c, summary = sess.run([self.train_op, self.cost, self.summary_op], feed_dict=train_data)


                if i % 100 == 0:
                    self.tf_writer.add_summary(summary)
                    self.training_costs.append(c)
                    self.decreaseAlpha(i)
                    if verbose: print("Cost after {} epochs: {}".format(i,c))

            if save: self.tf_saver.save(sess, self.tensor_save_path + "model.ckpt")


    def test(self):

        with tf.Session() as sess:

            try:
                self.tf_saver.restore(sess, self.tensor_save_path + "model.ckpt")
            except:
                sess.run(self.initializer)

        #------------------------------------ TEST SET ----------------------------------
            for i in range(self.m_test):
                x_batch, y_batch = self.getNextBatch("test", i, 1)
                test_data = {self.X: x_batch, self.Y: y_batch}

                c = sess.run(self.cost, feed_dict=test_data)
                self.test_costs.append(c)





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


    def compute_L2_regularization(self):
        reg = 0

        for param in self.parameters:
            reg += tf.nn.l2_loss(self.parameters[param])
        reg = reg * self.beta

        return reg


    def decreaseAlpha(self, t):

        self.alpha = tf.train.exponential_decay(self.alpha, t, 100000, 0.5, staircase=True)

        return self.alpha


    def getNextBatch(self, dataset, j, batch_size):

        if dataset == "train":
            X_series = self.X_train
            Y_series = self.Y_train
        elif (dataset == "test"):
            X_series = self.X_test
            Y_series = self.Y_test
        else:
            pass

        if j == 0:
            x = X_series.iloc[:, 0:batch_size ]
            y = Y_series.iloc[:, 0:batch_size]

        elif (j * batch_size) < len(X_series):
            x = X_series.iloc[:, (j*batch_size):((j+1)*(batch_size))]
            y = Y_series.iloc[:, (j*batch_size):((j+1)*(batch_size))]

        else:
            x = X_series.iloc[:, (j * batch_size):]
            y = Y_series.iloc[:, (j * batch_size):]


        return [x,y]


    def computeMetrics(self):

        n, m = np.shape(self.Y)

        diff = 0
        for i in range(m):
            diff += np.sqrt((np.exp(self.Y.iloc[:, i]) - np.exp(self.Y_.iloc[:, i])) ** 2)

        diff = diff / (m + self.epsilon)

        print("Diff: {} TL".format(diff))


    def shapes(self):

        print("Shape and type of Dataset_raw: {} shaped {}".format(np.shape(self.bike_data_manager.X), type(self.bike_data_manager.X)))
        print("Shape and type of X_Train: {} shaped {}".format(np.shape(self.X_train), type(self.X_train)))
        print("Shape and type of Y_Train: {} shaped {}".format(np.shape(self.Y_train), type(self.Y_train)))
        print("Shape and type of X_Test: {} shaped {}".format(np.shape(self.X_test), type(self.X_test)))
        print("Shape and type of Y_Test: {} shaped {}".format(np.shape(self.Y_test), type(self.Y_test)))



    def plotLearningCurve(self):
        import matplotlib.pyplot as plt
        import math

        training_costs = [cost if not math.isnan(cost) else 0 for cost in self.training_costs]
        test_costs = [cost if not math.isnan(cost) else 0 for cost in self.test_costs]


        plt.plot(range(len(training_costs)), training_costs, 'c-')
        plt.plot(range(len(test_costs)), test_costs, 'r-')
        plt.title("Learning curves")
        plt.show()







