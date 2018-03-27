# -*- coding: utf-8 -*-
# !/usr/bin/python

import numpy as np
import datetime as dt
import pandas as pd

class datamanager(object):

    def __init__(self, bike_brand, bike_model, polynomial_degree=1):
        import scrapper as sc

        self.bike_brand = bike_brand
        self.bike_model = bike_model
        self.polynomial_degree = polynomial_degree
        self.dataset_raw = sc.FetchBike(self.bike_brand, self.bike_model)

        self.clean_dataset, self.deleted_rows = self.clear_dataset_from_extreme_prices()


        self.index = self.clean_dataset["bike_id"]

        self.y, self.y_group =  self.extract_prices_and_price_categories( number_of_price_groups=20, logaritmic=False)


        self.unwanted_categorical_columns = ["ad_url", "ad_title", "ad_date"]
        self.categorical_columns = self.clean_dataset.select_dtypes(include=[np.object]).drop(labels=self.unwanted_categorical_columns, axis=1).set_index(self.index)

        self.unwanted_scalar_columns = ["price", "bike_id"]
        self.scalar_columns = self.clean_dataset.select_dtypes(include=[np.float64, np.float32, np.int]).drop(labels=self.unwanted_scalar_columns, axis=1).set_index(self.index)

        self.unwanted_bool_columns = []
        self.bool_columns = self.clean_dataset.select_dtypes(include=[np.bool]).drop(labels=self.unwanted_bool_columns, axis=1).set_index(self.index)


        self.x_scalar = self.polynomizeScalarColumns()
        #self.x_scalar = self.normalizeScalarColumns() # unutma bunu yapmayı


        self.x_categorical = self.processCategoricalColumns().set_index(self.index)
        self.x_bool = self.bool_columns.set_index(self.index)


        self.X = self.mergeX()

        #self.correlations = self.X.corr()["price"].dropna(how="all")

        #dataset_raw = self.dataset_raw.loc[:, correlations.keys()]



    def clear_dataset_from_extreme_prices(self):
        max_value, min_value, mean_value, sigma, m = self.dataset_raw["price"].describe()[
            ["max", "min", "mean", "std", "count"]]


        print("Raw dataset örnek sayısı: {}".format(m))
        print("Max fiyat: {} TL".format(max_value))
        print("Min fiyat: {} TL".format(min_value))
        print("Sigma: {}".format(sigma))
        print("Ortalama fiyat: {} TL".format(mean_value))

        gausian_min = round(mean_value - (2.5 * sigma))
        gausian_max = round(mean_value + (2.5 * sigma))

        print("Gausian MIN fiyat: {}".format(gausian_min))
        print("Gausian MAX fiyat: {}".format(gausian_max))

        deleted_rows = self.dataset_raw[
            (self.dataset_raw["price"] > gausian_max) | (self.dataset_raw["price"] < gausian_min)]
        clean_dataset = self.dataset_raw[
            (self.dataset_raw["price"] < gausian_max) & (self.dataset_raw["price"] > gausian_min)]

        print("Silinen örnek sayısı: {}".format(len(deleted_rows)))

        print("Gausian sonrası kalan örnek adeti: {}".format(len(clean_dataset)))


        return [clean_dataset, deleted_rows]


    def extract_prices_and_price_categories(self, number_of_price_groups, logaritmic = True):

        max_value, min_value, mean_value, sigma, m = self.dataset_raw["price"].describe()[["max", "min", "mean", "std", "count"]]

        price_gap = (max_value - min_value) / number_of_price_groups

        y = self.clean_dataset["price"]

        y_group = np.round((y - min_value) / price_gap)

        if logaritmic : y = np.log(y)

        return [y, y_group]


    def splitDataset(self, train_test_split_ratio):

        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test= train_test_split(self.X, self.y.T, test_size=(1-train_test_split_ratio))


        return [x_train, y_train, x_test, y_test]


    def plotGausian(self):

        import matplotlib.pyplot as plt
        import matplotlib.mlab as mlab

        mean = np.mean(self.y)
        print("Ortalama: %f" % mean)
        sigma = np.std(self.y)
        print("Standart sapma(sigma): %f" % sigma)

        min_value = mean - (3 * sigma)
        max_value = mean + (3 * sigma)

        x = np.linspace(min_value, max_value, 100)
        plt.plot(x, mlab.normpdf(x, mean, sigma))
        plt.title("%s - %s" % (self.bike_brand, self.bike_model))
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


    def polynomizeScalarColumns(self):


        from sklearn.preprocessing import  PolynomialFeatures

        poly = PolynomialFeatures(degree=self.polynomial_degree)

        x_scalar = poly.fit_transform(self.scalar_columns)


        target_feature_names = ['x'.join(['{}^{}'.format(pair[0], pair[1]) for pair in tuple if pair[1] != 0]) for tuple in
                                [zip(self.scalar_columns.columns, p) for p in poly.powers_]]

        output_df = pd.DataFrame(x_scalar, columns=target_feature_names)
        output_df = output_df.drop(output_df.columns[0], axis=1)


        return output_df


    def mergeX(self):



        self.x_scalar = self.x_scalar.set_index(self.index)

        self.x_categorical = self.x_categorical.set_index(self.index)

        self.x_bool = self.x_bool.set_index(self.index)


        X = pd.concat([self.x_scalar, self.x_categorical, self.x_bool], axis=1)

        return X


    def normalizeScalarColumns(self):

        return self.scalar_columns


    def processCategoricalColumns(self):

        x_categorical = pd.get_dummies(self.categorical_columns)

        return x_categorical


    def shapes(self):

        print("Clean dataset shape: {}".format(np.shape(self.clean_dataset)))
        print("X shape: {}".format(np.shape(self.X)))
        print("y shape: {}".format(np.shape(self.y)))
        print("y_group shape: {}".format(np.shape(self.y_group)))


    @staticmethod
    def writeCSV(dataset, file_name):

        np.savetxt("test_outputs/{}.csv".format(file_name), X=dataset, delimiter=",")

    @staticmethod
    def processDateColumn(dataset, calculated_columns):

        x_calculated = dataset.loc[:, calculated_columns].values

        for i in range(len(x_calculated)):
            x_calculated[i] = int((dt.datetime.today() - dt.datetime.strptime(str(x_calculated[i][0]), '%d.%m.%Y')).days)


        return x_calculated.T


    @staticmethod
    def find_Distances(x, centroids):

        distances = []
        for j in range(np.shape(x)[1]):
            d = []
            for i in range(np.shape(centroids)[1]):
                d.append([np.sum(np.square(np.subtract(x[:,j],centroids[:,i])))])
            distances.append(np.argmax(d))

        return distances

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def drawKMeans(points_values, assignment_values, centroid_values):

        import matplotlib.pyplot as plt

        plt.scatter(points_values[:, 0], points_values[:, 1], c=assignment_values, s=50, alpha=0.5)

        plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'kx', markersize=15)

        plt.show()


