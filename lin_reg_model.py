# -*- coding: utf-8 -*-
# !/usr/bin/python


import numpy as np
import pandas as pd
import tensorflow as tf
from data_management import datamanager as dm

np.set_printoptions(suppress=True, precision=8, linewidth=150, threshold=10000)


class linear_regression_model(object):

    # ----------------------- HyperParameters -----------------------------------------------

    epsilon = 1e-7
    epochs = 10000
    batch_size = 256
    dropout_threshold = 1
    alpha = 0.001
    beta = 0.1



    def __init__(self, bike_brand, bike_model):

        # ----------------------- Data Gathering -----------------------------------------------
        self.brand = bike_brand.replace(" ", "-")
        self.model = bike_model.replace(" ", "-")
        self.prefix = (bike_brand + "-" + bike_model)

        self.dataset = dm(bike_brand, bike_model, polynomial_degree=1, logaritmic_prices=True)
        self.dataset.clear_uncorrelated_fields()
        self.X_train, self.Y_train, self.X_test, self.Y_test = self.dataset.splitDataset(0.9)



        #----------------------- Model Parameters -----------------------------------------------

        self.n_train, self.m_train = np.shape(self.X_train)
        self.n_test, self.m_test = np.shape(self.X_test)
        self.number_of_output = np.shape(self.Y_test)[0]
        self.layers_dims = [self.n_train, 40, 30, self.number_of_output]
        self.batches = int(np.ceil(self.m_train / self.batch_size))
        self.sample_size = self.m_train


        #-------------------------  MODEL GRAPH -------------------------------------
        tf.reset_default_graph()
        with tf.name_scope("Input_Data"):
            self.X = tf.placeholder(dtype=tf.float32, shape=[self.n_train,None], name="X")

        with tf.name_scope("Labels"):
            self.Y = tf.placeholder(dtype=tf.float32, shape=[self.number_of_output, None], name="Y")

        with tf.name_scope("Parameters"):
            self.parameters = self.initWeights()

        with tf.name_scope("Predictions"):
            self.Y_ = self.feedForward()

        with tf.name_scope("Cost_computation"):

            with tf.name_scope("Cost_and_Regularization"):
                self.dryLoss = self.computeUnregularizedLoss()
                self.regularisation = self.compute_L2_regularization()
                self.cost = tf.reduce_mean(self.dryLoss + self.regularisation)


            with tf.name_scope("Optimization"):
                self.optimizer = tf.train.AdamOptimizer(self.alpha)
                self.train_op = self.optimizer.minimize(self.cost, name="Training_OP")
                self.training_costs = []
                self.test_costs = []

        with tf.name_scope("Metrics"):
            self.total_error = tf.reduce_sum(tf.square(tf.subtract(self.Y, tf.reduce_mean(self.Y))))
            self.unexplained_error = tf.reduce_sum(tf.square(tf.subtract(self.Y, self.Y_)))
            self.r_squared_score = tf.subtract(1., tf.div(self.unexplained_error, self.total_error))

        # ----------------------- Tensorflow  Logs and Configs-----------------------------------------------
        tf.summary.scalar("Cost", self.cost)
        tf.summary.histogram("W1", self.parameters["W1"])

        self.initializer = tf.global_variables_initializer()


        self.summary_op = tf.summary.merge_all()

        self.tensor_logs_path = self.build_Path("./Tensor_logs/")
        self.tf_writer = tf.summary.FileWriter(self.tensor_logs_path, graph=tf.get_default_graph())

        self.tensor_save_path = self.build_Path("./Tensor_models/")
        self.tf_saver = tf.train.Saver()



    # ------------------------------ TRAINING AND TEST FUNCS ---------------------------------

    def train(self, verbose=True, save=False):

        with tf.Session() as sess:

            sess.run(self.initializer)

            for i in range(self.epochs):

                for j in range(self.batches+1):

                    x_batch, y_batch = self.getNextBatch("train", j, self.batch_size)
                    train_data = {self.X: x_batch, self.Y:y_batch}
                    _, c, summary = sess.run([self.train_op, self.cost, self.summary_op], feed_dict=train_data)


                if i % 100 == 0:
                    self.training_costs.append(c)
                    self.decreaseAlpha(i)
                    if verbose: print("Cost after {} epochs: {}".format(i,c))
                    self.tf_writer.add_summary(summary, i)


            if save: self.tf_saver.save(sess, self.tensor_save_path + "model.ckpt")


    def test(self, show_prices=False):

        with tf.Session() as sess:

            try:
                self.tf_saver.restore(sess, self.tensor_save_path + "model.ckpt")
            except:
                sess.run(self.initializer)

            #------------------------------------ TEST SET ----------------------------------

            x_batch, y_batch = self.getNextBatch("test", 0, self.batch_size)
            test_data = {self.X: x_batch, self.Y: y_batch}

            c, preds = sess.run([self.cost, self.Y_], feed_dict=test_data)

            self.test_costs.append(c)

            r_squared_score = sess.run(self.r_squared_score,feed_dict=test_data)


            print("Test cost:{}".format(c))
            print("R2 Score: {}".format(r_squared_score))

            if show_prices:
                preds = pd.DataFrame({'preds': preds[0]}, index=self.Y_test.keys())
                results = pd.concat([self.Y_test, preds.T], axis=0, ignore_index=False)
                print(results)


    # ------------------------------ HELPER FUNCS ---------------------------------

    def initWeights(self):

        parameters = {}

        for i, val in enumerate(self.layers_dims[1:], start=1):
            parameters["W" + str(i)] = tf.Variable(
                tf.truncated_normal([val, self.layers_dims[i - 1]], stddev=0.1, dtype=tf.float32), name="W" + str(i))
            parameters["b" + str(i)] = tf.Variable(tf.constant(0.1, shape=[val, 1]), name="b" + str(i))

        return parameters


    def build_Path(self, path):

        import os
        fullpath = str(path + self.brand + "/" + self.model + "/")

        if not os.path.exists(fullpath):
            os.makedirs(fullpath)

        return fullpath


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

        return tf.sqrt(tf.losses.mean_squared_error(labels=self.Y, predictions=self.Y_))


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

        print("Shape and type of Dataset_raw: {} shaped {}".format(np.shape(self.dataset.X), type(self.dataset.X)))
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







