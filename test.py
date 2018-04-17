# -*- coding: utf-8 -*-
# !/usr/bin/python

import pandas as pd
from scrapper import BikeDataScrapper
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data_management import datamanager as dm
from lin_reg_model import linear_regression_model as lr

sc = BikeDataScrapper("Bajaj")

print(sc.base_Url)
sc.load_Main_Pages()
sc.load_Detail_Pages()
#print(sc.detail_pages)

sc.get_BikeSpec(index=0)


# bike = dm("bmw", "c 600 sport", polynomial_degree=2, logaritmic_prices=True)
# bike.clear_uncorrelated_fields()
# #bike.plotGausianPrices()
#
# print(bike.dataset["abs"].describe())
# print(bike.deleted_rows)
#
# print(bike.X.head())
# pca_data = bike.PCA_graph(2, show_graph=False)
#
# #print(bike.index)
#
# print(pd.DataFrame(pca_data, columns=bike.dataset["price"]))
