# -*- coding: utf-8 -*-
# !/usr/bin/python

# Importing the libraries


import tensorflow as tf
import numpy as np
import pandas as pd
import datetime as dt
import fetch_bike_list as bikes
import matplotlib.pyplot as plt

#--------------------- Get Bike DATASET and splits into TEST and TRAIN sets -------------------------------------


def PreprocessData(dataset, train_test_split_ratio, polynomial_degree):


        from sklearn.preprocessing import PolynomialFeatures

        categorical_columns = ["color", "city"]
        if "brand" in dataset: categorical_columns.append("brand")
        if "model" in dataset: categorical_columns.append("model")


        scalar_columns = ["year", "km"]
        calculated_columns = ["ad_date"]
        unwanted_columns = ["ad_title", "ad_url","price", "ad_date","bike_id"]
        output_column = ["price"]

        x_scalar = dataset.loc[:,scalar_columns].values
        x_categorical = dataset.loc[:, categorical_columns].values
        x_calculated = dataset.loc[:, calculated_columns].values

        for i in range(len(x_calculated)):
            x_calculated[i] = int((dt.datetime.today() - dt.datetime.strptime(str(x_calculated[i][0]), '%d.%m.%Y')).days)



        y = dataset.loc[:, output_column].values

        drop_columns = np.concatenate((categorical_columns, scalar_columns, unwanted_columns, output_column), axis=0)


        x = dataset.drop(drop_columns , axis=1)
        #print(x)

        from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


        #--------------------------- TRANSFORM CATEGORICAL FEATURES ---------------------------------------------


        for i in range(len(x_categorical.T)):
            le = LabelEncoder()
            col = x_categorical.T[i]
            le.fit(col)
            x_categorical.T[i] = le.transform(col)

        onehotencoder = OneHotEncoder(categorical_features="all", sparse=False)
        x_categorical = onehotencoder.fit_transform(x_categorical)


        # --------------------------- TRANSFORM SCALAR FEATURES ---------------------------------------------


        poly = PolynomialFeatures(degree = polynomial_degree)
        x_scalar = poly.fit_transform(x_scalar)
        x_scalar = StandardScaler().fit_transform(x_scalar)
        x_scalar[:,0] = 1



        x = np.concatenate((x_scalar, x_calculated, x, x_categorical), axis =1)

        #x = np.insert(x, 0,  1,  axis=1)
        #np.savetxt("test.csv", x, delimiter=",")
        #x = np.delete(x,[0,1], axis = 1)
        #print(x)


        # #--------------- Train TEST Split ---------------------------------------
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = train_test_split_ratio, random_state = 1)

        return [x_train, y_train, x_test, y_test ]


def generate_data(how_many, number_of_features, train_test_split_ratio):

    data = np.random.rand(how_many, number_of_features)
    intercept = np.ones(how_many).reshape([how_many, 1])

    data = np.concatenate((intercept, data), axis=1)


    answers = data[:, 0] * data[:, 1]
    answers = np.reshape(answers,[how_many,1])

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(data, answers, test_size=train_test_split_ratio, random_state=1)

    return [x_train, y_train, x_test, y_test]



def test(bike_brand, bike_model):


    #------------------------------ GET DATASETS --------------------------------------
    #X_raw, y_raw, x_test_raw, y_test_raw  = PreprocessData(bikes.FetchAllBikes(), 0.2, 3)
    X_raw, y_raw, x_test_raw, y_test_raw = PreprocessData(bikes.FetchBike(bike_brand, bike_model), 0.2, 1)

    #X_raw, y_raw, x_test_raw, y_test_raw = generate_data(2000, 2, 0.2)

    #print(y_raw[1])

    #np.savetxt("test.csv", X_raw, delimiter=",")

    tf.reset_default_graph()


    #----------------------- VARIABLES -----------------------------------------------
    with tf.name_scope("INPUT"):
        m, n = np.shape(X_raw)
        m_test, n_test = np.shape(x_test_raw)
        number_of_output = 1
        number_of_iterations = 10000
        batch_size = 100
        alpha = 0.1

        print("X shape m: %s n: %s" % (m, n))
        print("Y shape: %s %s" % np.shape(y_raw))

        logs_path = "Tensor_logs/"

        regression_type ="Ridge"

        X = tf.placeholder(dtype=tf.float32, shape=[None,n], name="X")
        Y = tf.placeholder(dtype=tf.float32, shape=[None, number_of_output], name="Y")

    with tf.name_scope("Weights"):
        W = tf.Variable(tf.truncated_normal([number_of_output, n], stddev=0.1), name="W")

    with tf.name_scope("Biases"):
        b = tf.Variable(tf.constant(0.1, shape=[number_of_output]), name="b")


    #tf.summary.histogram("W", W)

    #------------------------- PREDICTIVE MODEL -------------------------------------
    with tf.name_scope("PREDICTIVE_MODEL"):
        Y_ = tf.add(tf.matmul(W, tf.transpose(X)),b , name="Y_")


    #---------------------- LOSS FUNCTION --------------------------------------------

    with tf.name_scope("LOSS_FUNCTION"):


        if regression_type == "Lasso":

            lasso_param = tf.constant(0.9)

            heavyside_step = tf.truediv(1., tf.add(1., tf.exp(tf.multiply(-50., tf.subtract(W, lasso_param)))))

            regularization_param = tf.multiply(heavyside_step, 99.)

            cost = tf.add(tf.reduce_mean(tf.square(Y_ - Y)), regularization_param)[0]

        elif regression_type == "Ridge":

            ridge_param = tf.constant(1.)
            ridge_loss = tf.reduce_mean(tf.square(W))
            cost = tf.expand_dims(tf.add(tf.reduce_mean(tf.square(Y_ - Y)), tf.multiply(ridge_param, ridge_loss)), 0)

        elif regression_type == "Lin_Reg":
            cost = tf.reduce_mean(tf.abs(Y_- Y))

        elif regression_type == "Elastic":
            elastic_param1 = tf.constant(1.)
            elastic_param2 = tf.constant(1.)

            l1_w_loss = tf.reduce_mean(tf.abs(W))
            l2_w_loss = tf.reduce_mean(tf.square(W))

            e1_term = tf.multiply(elastic_param1, l1_w_loss)
            e2_term = tf.multiply(elastic_param2, l2_w_loss)

            cost = tf.expand_dims(tf.add(tf.add(tf.reduce_mean(tf.square(Y_ - Y)), e1_term), e2_term),0)

        elif regression_type == "Demming":

            demming_numerator = tf.abs(Y_ - Y)
            demming_denominator = tf.sqrt(tf.add(tf.square(X),1))
            cost = tf.reduce_mean(tf.truediv(demming_numerator, demming_denominator))


        else:
            print('Cost fonksiyonu tırtladı...')

        tf.summary.scalar("cost", cost)




    #----------------- OPTIMIZER FOR LOSS FUNCTION ----------------------------------
    with tf.name_scope("TRAIN_STEP"):
        optimizer = tf.train.AdamOptimizer(alpha)
        train_op = optimizer.minimize(cost)
        summary_op = tf.summary.merge_all()


    # ------------------- SESSION STARTS HERE ----------------------------------------

    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)

        tensorboard = tf.summary.FileWriter(logs_path, tf.get_default_graph())

        #-------------------------------- TRAIN LOOP --------------------------------------------------------
        train_plot = []

        for i in range(number_of_iterations):
            rand_index = np.random.choice(m, size=batch_size)
            batch_x, batch_y = [[], []]

            for i in rand_index:

                batch_x.append(X_raw[i])
                batch_y.append(y_raw[i])

            train_data = {X: batch_x, Y: batch_y}

            _ = sess.run([train_op], feed_dict=train_data)

            cost_summ = sess.run(cost, feed_dict=train_data)


            train_plot.append(cost_summ)
            print("Train variance: %f" % np.var(train_plot))
            #tensorboard.add_summary(cost_summ, i)


        plt.plot(range(len(train_plot)), train_plot, 'C1' )


        # -------------------------------- TEST LOOP --------------------------------------------------------

        batch_start = 0
        batch_size = 1
        test_plot = []
        for i in range(m_test):

            batch_x, batch_y = x_test_raw[i], y_test_raw[i]

            batch_x = np.reshape(batch_x, [batch_size, n])
            batch_y = np.reshape(batch_y, [batch_size, 1])

            test_data = {X: batch_x, Y: batch_y}


            pred = sess.run(Y_, feed_dict=test_data)
            #print("Y: %f Y_Pred = %f" % (batch_y, pred))
            cost_summ = sess.run(cost, test_data)
            test_plot.append(cost_summ)
            #print("Test Cost: %f" % cost_summ)

            print("Test variance: %f" % np.var(test_plot))

        plt.plot(range(len(test_plot)), test_plot, 'C2')
        plt.show()


def test2():
    X_raw, y_raw, x_test_raw, y_test_raw = PreprocessData(bikes.FetchAllBikes(), 0.1, 3)
    from sklearn import linear_model

    regresor = linear_model.Lasso(alpha=0.0001, fit_intercept=False)

    result = regresor.fit(X_raw,y_raw).score(X_raw, y_raw)
    print(result)

    for i in range(len(x_test_raw)):
        #print([x_test_raw[i]])
        y_ = regresor.predict([x_test_raw[i]])

        print("Y pred: %f  Y: %f" % (y_,y_test_raw[i]) )


#bikes.FetchBike("honda","spacy 110 alfa")

test("honda","cbf 150")
#test2()
#PreprocessData(bikes.FetchAllBikes(), 0.2, 1)