# -*- coding: utf-8 -*-
# !/usr/bin/python

import numpy as np
import datetime as dt
import pandas as pd



def preprocessData(bike_brand, bike_model, train_test_split_ratio, polynomial_degree=1, categorical_price=False):

        import scrapper as sc

        dataset_raw = sc.FetchBike(bike_brand, bike_model)

        y, y_group, dataset_raw = extractPricesAndCategories(dataset_raw, sensitivity=20, logaritmic=True)

        index = dataset_raw["bike_id"]

        unwanted_categories = ["ad_url", "ad_title", "ad_date"]
        categorical_columns = dataset_raw.select_dtypes(include=[np.object]).drop(labels=unwanted_categories, axis=1)

        correlations = dataset_raw.corr()["price"].dropna(how="all")

        dataset_raw = dataset_raw.loc[:, correlations.keys()]

        x_bool = dataset_raw.select_dtypes(include=[np.bool])



        scalar_columns = dataset_raw.select_dtypes(include=[np.float64, np.float32, np.int]).drop(["price"], axis=1)
        scalar_columns = polynomizeScalarColumns(scalar_columns, polynomial_degree=polynomial_degree)
        # unutma bunu yapmayı
        x_scalar = normalizeScalarColumns(scalar_columns)


        x_categorical = processCategoricalColumns(categorical_columns)


        #plotGausian(y, bike_brand, bike_model)

        X = mergeX(x_scalar, x_categorical, x_bool, index)


        #x_pca = applyPCA(X,2)

        #X = x_scalar


        if categorical_price:
            x_train, y_train, x_test, y_test = splitDataset(X, y_group, train_test_split_ratio)
        else:
            x_train, y_train, x_test, y_test = splitDataset(X, y, train_test_split_ratio)

        return [x_train, y_train, x_test, y_test]


def writeCSV(dataset, var_name):

    np.savetxt("test_outputs/{}.csv".format(var_name), dataset, delimiter=",")


def extractPricesAndCategories(dataset_raw, sensitivity, logaritmic = True):


    max_value, min_value, mean_value, sigma, m = dataset_raw["price"].describe()[["max", "min", "mean", "std", "count"]]


    price_gap = (max_value - min_value) / sensitivity

    print("Gausian öncesi örnek sayısı: {}".format(m))
    print("Max fiyat: {} TL".format(max_value))
    print("Min fiyat: {} TL".format(min_value))
    print("Fiyat aralığı: {} TL".format(price_gap))
    print("Sigma: {}".format(sigma))
    print("Ortalama fiyat: {} TL".format(mean_value))

    gausian_min = round(mean_value - (2.5 * sigma))
    gausian_max = round(mean_value + (2.5 * sigma))

    print("Gausian MIN fiyat: {}".format(gausian_min))
    print("Gausian MAX fiyat: {}".format(gausian_max))

    deleted_rows = dataset_raw[(dataset_raw["price"] > gausian_max) | (dataset_raw["price"] < gausian_min)]
    dataset_raw = dataset_raw[(dataset_raw["price"] < gausian_max) & (dataset_raw["price"] > gausian_min)]

    print("Silinen örnek sayısı: {}".format(len(deleted_rows)))

    print("Gausian sonrası kalan örnek adeti: {}".format(len(dataset_raw)))

    y = np.array(dataset_raw["price"]).reshape([1,len(dataset_raw)])

    y_group = np.round((y - gausian_min) / price_gap)

    if logaritmic : y = np.log(y)

    return [y, y_group, dataset_raw]



def plotGausian(y, bike_brand, bike_model):

    import matplotlib.pyplot as plt
    import matplotlib.mlab as mlab

    mean = np.mean(y)
    print("Ortalama: %f" % mean)
    sigma = np.std(y)
    print("Standart sapma(sigma): %f" % sigma)

    min_value = mean - (3 * sigma)
    max_value = mean + (3 * sigma)

    x = np.linspace(min_value, max_value, 100)
    plt.plot(x, mlab.normpdf(x, mean, sigma))
    plt.title("%s - %s" % (bike_brand, bike_model))
    plt.show()


def gausianValues(sensitivity, group_index):
    a = 0.1  # length of slope
    b = 0.9
    c = group_index  # max index of slope

    group = []
    for i in range(sensitivity):
        y = float(1 / (1 + np.abs((i - c) / a) ** (2. * b)))
        group.append(y)
    return group


def polynomizeScalarColumns(scalar_columns, polynomial_degree):


    from sklearn.preprocessing import  PolynomialFeatures

    poly = PolynomialFeatures(degree=polynomial_degree)

    x_scalar = poly.fit_transform(scalar_columns)

    target_feature_names = ['x'.join(['{}^{}'.format(pair[0], pair[1]) for pair in tuple if pair[1] != 0]) for tuple in
                            [zip(scalar_columns.columns, p) for p in poly.powers_]]

    output_df = pd.DataFrame(x_scalar, columns=target_feature_names)
    output_df = output_df.drop(output_df.columns[0], axis=1)

    return output_df


def normalizeScalarColumns(scalar_columns):

    return scalar_columns


def processCategoricalColumns(categorical_columns):


    x_categorical = pd.get_dummies(categorical_columns)

    return x_categorical


def processDateColumn(dataset, calculated_columns):

    x_calculated = dataset.loc[:, calculated_columns].values

    for i in range(len(x_calculated)):
        x_calculated[i] = int((dt.datetime.today() - dt.datetime.strptime(str(x_calculated[i][0]), '%d.%m.%Y')).days)


    return x_calculated.T



def mergeX(x_scalar, x_categorical, x_bool, index):


    column_names = np.concatenate((list(x_scalar.keys()), list(x_categorical.keys()), list(x_bool.keys())),
                                  axis=0)

    x_scalar = x_scalar.set_index(index)


    x_categorical = x_categorical.set_index(index)

    x_bool = x_bool.set_index(index)

    # print(column_names)
    # print("------------------------------------------")
    # print(x_scalar)
    # print("------------------------------------------")
    # print(x_categorical)
    # print("------------------------------------------")
    # print(x_bool)

    X = pd.concat([x_scalar, x_categorical, x_bool], axis=1)

    print(X.head())

    #print(X)

    return X


def splitDataset(X, y, train_test_split_ratio):


    #from sklearn.model_selection import train_test_split

    #y = y.T


    n,m = np.shape(X)
    train_size = int(round(m * train_test_split_ratio))

    print("Training adeti: %i" % train_size)
    print("Test adeti: %i" % (m - train_size))

    #x_train, x_test, y_train, y_test= train_test_split(X.T, y.T, test_size=(1-train_test_split_ratio))

    x_train = X[:, 0:train_size]
    y_train = y[:, 0:train_size]

    x_test = X[:,train_size:]
    y_test = y[:,train_size:]


    return [x_train, y_train, x_test, y_test]


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


def applyKMeans(X, number_of_clusters):
    from sklearn.neighbors import NearestNeighbors

    #---IMPORTANT : sklearn uses landscape oriented dataset

    knn = NearestNeighbors(n_neighbors=number_of_clusters, algorithm='brute').fit(X.T)

    d, i = knn.kneighbors(X.T)

    print(np.shape(i))

    # print(np.shape(d.T))
    #
    # for i in range(len(d)):
    #     print(d.T[:,i])


    return 0


def applyKMeansTensorflow(x, number_of_clusters):

    import tensorflow as tf

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

        assignment_values = []
        for step in xrange(iteration_n):
            [_, centroid_values, points_values, assignment_values] = sess.run(
                [update_centroids, centroids, points, assignments])


        val = tf.data.Dataset.from_tensors(assignment_values)


    return val


def drawKMeans(points_values, assignment_values, centroid_values):

    import matplotlib.pyplot as plt

    plt.scatter(points_values[:, 0], points_values[:, 1], c=assignment_values, s=50, alpha=0.5)

    plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'kx', markersize=15)

    plt.show()


X_train, Y_train, X_test, Y_test = preprocessData("honda", "cbf 150", 0.9, polynomial_degree=1)