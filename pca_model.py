# -*- coding: utf-8 -*-
# !/usr/bin/python

import data_management as dm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, precision=3, linewidth=150, threshold=10000)
epsilon = 1e-7

def pca_model(bike_brand, bike_model):

    X_train, Y_train, X_test, Y_test = dm.preprocessData(bike_brand, bike_model, 0.9, polynomial_degree=1, categorical_price=False)

    n_train, m_train = np.shape(X_train)
    n_test, m_test = np.shape(X_test)


    #x_batch = np.array([[1, 2, 8], [2, 2, 2], [3, 28, 64], [4, 4, 4]])
    x_batch, y_batch = getNextBatch(X_train, Y_train, 0, m_train)


    x_stack = x_batch
    #x_stack = np.concatenate((x_batch, y_batch), axis=0)


    x = applyPCA(x_stack, reduction=2)
    #
    # plt.scatter(x[0,:],x[1,:], c=y_batch[0,:])
    # plt.show()
    #

    assignment_values, centroids  = applyKMeans(x, number_of_clusters = 3)


    x_t = applyPCA(X_test, reduction=2)

    print(np.shape(X_test))
    print(np.shape(x_t))
    print(np.shape(centroids))

    plt.scatter(x_t[0,:], x_t[1,:], c="b")
    plt.scatter(centroids[0,:], centroids[1,:], c="r")
    plt.show()

    distances = find_Distances(x, centroids)


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


def find_Distances(x, centroids):

    distances = []
    for j in range(np.shape(x)[1]):
        d = []
        for i in range(np.shape(centroids)[1]):
            d.append([np.sum(np.square(np.subtract(x[:,j],centroids[:,i])))])
        distances.append(np.argmax(d))

    return distances


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
    eig_pairs = [(np.abs(eigen_values[i]), np.abs(eigen_vectors[:, i])) for i in range(len(eigen_values))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # Building PCA matrix
    pca_matrix = []
    for i in range(reduction):
        pca_matrix.append(eig_pairs[i][1].reshape(n, 1))

    w = np.hstack(tuple(pca_matrix))

    return w.T.dot(X)


def applyKMeans(x, number_of_clusters):

    iteration_n = 100

    points = tf.constant(x.T, dtype=tf.float64)
    centroids = tf.Variable(tf.slice(tf.random_shuffle(points), [0, 0], [number_of_clusters, -1]))

    points_expanded = tf.expand_dims(points, 0)
    centroids_expanded = tf.expand_dims(centroids, 1)


    distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
    assignments = tf.argmin(distances, 0)

    means = []

    for c in xrange(number_of_clusters):
        means.append(tf.reduce_mean(
            tf.gather(points,
                      tf.reshape(
                          tf.where(
                              tf.equal(assignments, c)
                          ), [1, -1])
                      ), reduction_indices=[1]))

    new_centroids = tf.concat(means, axis=0)

    update_centroids = tf.assign(centroids, new_centroids)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for step in xrange(iteration_n):
            [_, centroid_values, points_values, assignment_values] = sess.run(
                [update_centroids, centroids, points, assignments])


        #print "Centroids:" + "\n", centroid_values

        #print(assignment_values)

    return [assignment_values, centroid_values.T]


def drawKMeans(points_values, assignment_values, centroid_values):

    plt.scatter(points_values[:, 0], points_values[:, 1], c=assignment_values, s=50, alpha=0.5)

    plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'kx', markersize=15)

    plt.show()



pca_model("honda","cbr 250 r")



