# -*- coding: utf-8 -*-
# !/usr/bin/python

import numpy as np
import datetime as dt
import pandas as pd


class datamanager:

    number_of_price_groups = 20

    def __str__(self):
        return "Dataset for Brand: {} and Model: {}".format(self.bike_brand, self.bike_model)


    def __init__(self, bike_brand, bike_model, polynomial_degree=1, logaritmic_prices= True):

        import scrapper as sc

        def clear_dataset_from_extreme_prices(self):

            max_value, min_value, mean, sigma, m = self.get_data_info(verbose=True)

            gausian_min = round(mean - (2.9 * sigma))
            gausian_max = round(mean + (2.9 * sigma))

            print("Gausian MIN fiyat: {}".format(gausian_min))
            print("Gausian MAX fiyat: {}".format(gausian_max))

            deleted_rows = self.dataset[
                (self.dataset["price"] > gausian_max) | (self.dataset["price"] < gausian_min)]
            clean_dataset = self.dataset[
                (self.dataset["price"] < gausian_max) & (self.dataset["price"] > gausian_min)]

            print("Silinen örnek sayısı: {}".format(len(deleted_rows)))

            print("Gausian sonrası kalan örnek adeti: {}".format(len(clean_dataset)))

            return [clean_dataset, deleted_rows]

        def extract_prices_and_price_categories(self, number_of_price_groups, logaritmic=logaritmic_prices):

            max_value, min_value, mean_value, sigma, m = self.get_data_info(verbose=False)

            price_gap = (max_value - min_value) / number_of_price_groups

            y = self.dataset["price"].to_frame().set_index(self.index)

            y_group = np.round((y["price"] - min_value) / price_gap)

            if logaritmic: y["price"] = np.log(y["price"])

            return [y, y_group]

        def processCategoricalColumns(self):
            x_categorical = pd.get_dummies(self.categorical_columns).astype(np.bool)

            return x_categorical

        def polynomizeScalarColumns(self):
            from sklearn.preprocessing import PolynomialFeatures

            poly = PolynomialFeatures(degree=self.polynomial_degree)

            x_scalar = poly.fit_transform(self.scalar_columns)

            target_feature_names = ['x'.join(['{}^{}'.format(pair[0], pair[1]) for pair in tuple if pair[1] != 0]) for
                                    tuple in
                                    [zip(self.scalar_columns.columns, p) for p in poly.powers_]]

            output_df = pd.DataFrame(x_scalar, columns=target_feature_names)
            output_df = output_df.drop(output_df.columns[0], axis=1)

            return output_df

        def normalizeScalarColumns(self):
            from sklearn import preprocessing

            x_scaled = pd.DataFrame(preprocessing.scale(self.scalar_columns))
            x_scaled.columns = self.scalar_columns.columns


            return x_scaled

        def mergeX(self):

            self.x_scalar = self.x_scalar.set_index(self.index)

            self.x_categorical = self.x_categorical.set_index(self.index)

            self.x_bool = self.x_bool.set_index(self.index)

            X = pd.concat([self.x_scalar, self.x_categorical, self.x_bool, self.y], axis=1)

            return X


        self.bike_brand = bike_brand
        self.bike_model = bike_model
        self.polynomial_degree = polynomial_degree
        self.dataset = sc.FetchBike(self.bike_brand, self.bike_model)

        self.dataset, self.deleted_rows = clear_dataset_from_extreme_prices(self)


        self.index = self.dataset["bike_id"]

        self.y, self.y_group =  extract_prices_and_price_categories(self, number_of_price_groups=self.number_of_price_groups, logaritmic=False)


        self.unwanted_categorical_columns = ["ad_url", "ad_title", "ad_date"]
        self.categorical_columns = self.dataset.select_dtypes(include=[np.object]).drop(labels=self.unwanted_categorical_columns, axis=1).set_index(self.index)

        self.unwanted_scalar_columns = ["price", "bike_id"]
        self.scalar_columns = self.dataset.select_dtypes(include=[np.float64, np.float32, np.int]).drop(labels=self.unwanted_scalar_columns, axis=1).set_index(self.index)

        self.unwanted_bool_columns = []
        self.bool_columns = self.dataset.select_dtypes(include=[np.bool]).drop(labels=self.unwanted_bool_columns, axis=1).set_index(self.index)


        self.x_categorical = processCategoricalColumns(self).set_index(self.index)
        self.x_bool = self.bool_columns.set_index(self.index)
        self.x_scalar = polynomizeScalarColumns(self)
        self.x_scalar = normalizeScalarColumns(self)

        self.X = mergeX(self)



    #------------------------------------------- HELPER FUNCS ------------------------------------------------

    def shapes(self):

        print("Clean dataset shape: {}".format(np.shape(self.dataset)))
        print("X shape: {}".format(np.shape(self.X)))
        print("y shape: {}".format(np.shape(self.y)))
        print("y_group shape: {}".format(np.shape(self.y_group)))


    def PCA_graph(self, reduction=2, show_graph=True):

        def apply_PCA(X, reduction):

            #drop(["price"], axis=1).
            x_ = X.drop(["price"], axis=1).T
            n,m = np.shape(x_)

            # calculate means vector to calculate scatter matrix
            means_vector = np.zeros([n,1])

            for i in range(n):
                means_vector[i,:] = np.mean(x_.iloc[i,:])

            #calculate scatter matrix to calculate eigen vector and eigen values
            scatter_matrix = np.zeros([n,n])

            for i in range(m):
                scatter_matrix = scatter_matrix + np.dot((x_.iloc[:,i].values.reshape(n,1) - means_vector), (x_.iloc[:,i].values.reshape(n,1) - means_vector).T)

            #calculate eigen vector and eigen values
            eigen_values, eigen_vectors = np.linalg.eig(np.array(scatter_matrix, dtype=float))

            eig_pairs = [(np.abs(eigen_values[i]), np.abs(eigen_vectors[:, i])) for i in range(len(eigen_values))]

            # Sort the (eigenvalue, eigenvector) tuples from high to low
            eig_pairs.sort(key=lambda x: x[0], reverse=True)

            # Building PCA matrix
            pca_matrix = []
            for i in range(reduction):
                pca_matrix.append(eig_pairs[i][1].reshape(n, 1))

            w = np.hstack(tuple(pca_matrix))


            return w.T.dot(x_)

        def draw_PCA_graph_bokeh(pca_matrix, prices):
            from bokeh.plotting import figure, show, output_file

            output_file("test_outputs/test.html")

            if show_graph:
                p = figure(title="Markers", plot_width=400, plot_height=400)
                p.circle(pca_matrix[0], pca_matrix[1], size=10, color=list(self.y.values), alpha=0.5)
                show(p)

        def draw_PCA_graph_matplotlib(pca_matrix, prices):
            import matplotlib.pyplot as plt

            plt.scatter(pca_matrix[0], pca_matrix[1], c=prices)
            plt.show()

        pca_matrix = apply_PCA(self.X, reduction)

        if show_graph: draw_PCA_graph_matplotlib(pca_matrix, self.y["price"])

        return pca_matrix


    def clear_uncorrelated_fields(self):

        self.corrs = self.X.corr()["price"].dropna(how="all")
        self.uncorrs = self.X.corr()["price"].isna().loc[lambda b: b == True]
        self.X = self.X.drop(labels=list(self.uncorrs.index), axis=1)


    def splitDataset(self, train_test_split_ratio):

        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test= train_test_split(self.X.drop(["price"], axis=1), self.y, test_size=(1-train_test_split_ratio))


        return [x_train.T, y_train.T, x_test.T, y_test.T]


    def plotGausianPrices(self):

        import matplotlib.pyplot as plt
        import matplotlib.mlab as mlab

        mean = np.mean(self.y["price"])
        print("Ortalama: %f" % mean)
        self.sigma = np.std(self.y["price"])
        print("Standart sapma(sigma): %f" % self.sigma)

        min_value = mean - (3 * self.sigma)
        max_value = mean + (3 * self.sigma)

        x = np.linspace(min_value, max_value, 100)
        plt.plot(x, mlab.normpdf(x, mean, self.sigma))
        plt.title("%s - %s" % (self.bike_brand, self.bike_model))
        plt.show()


    def get_data_info(self, verbose=False):
        max_value, min_value, mean_value, sigma, m = self.dataset["price"].describe()[
            ["max", "min", "mean", "std", "count"]]

        if verbose:
            print("Raw dataset örnek sayısı: {}".format(m))
            print("Max fiyat: {} TL".format(max_value))
            print("Min fiyat: {} TL".format(min_value))
            print("Sigma: {}".format(sigma))
            print("Ortalama fiyat: {} TL".format(mean_value))

        return [max_value, min_value, mean_value, sigma, m]



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
    def applyKMeans(self, number_of_clusters):
        from sklearn.neighbors import NearestNeighbors

        #---IMPORTANT : sklearn uses landscape oriented dataset

        knn = NearestNeighbors(n_neighbors=number_of_clusters, algorithm='brute').fit(self.X.T)

        d, i = knn.kneighbors(self.X.T)

        print(np.shape(i))


        return 0


    @staticmethod
    def drawKMeans(points_values, assignment_values, centroid_values):

        import matplotlib.pyplot as plt

        plt.scatter(points_values[:, 0], points_values[:, 1], c=assignment_values, s=50, alpha=0.5)

        plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'kx', markersize=15)

        plt.show()


