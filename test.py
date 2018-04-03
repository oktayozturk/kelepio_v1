# -*- coding: utf-8 -*-
# !/usr/bin/python

import pandas as pd
import scrapper as sc
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data_management import datamanager as dm
from lin_reg_model import linear_regression_model as lr




bike = dm("ktm", "200 duke", polynomial_degree=2, logaritmic_prices=True)
bike.clear_uncorrelated_fields()
#bike.plotGausianPrices()

# print(bike.dataset["price"].describe())
# print(bike.deleted_rows)

print(bike.X.head())

bike.PCA_graph(2, show_graph=True)