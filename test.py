# -*- coding: utf-8 -*-
# !/usr/bin/python

import pandas as pd
import scrapper as sc
import matplotlib.pyplot as plt
import numpy as np
from data_management import datamanager as dm


b = dm("bmw","f 650 gs", polynomial_degree=1)

b.clear_uncorrelated_fields()


X_train, Y_train, X_test, Y_test = b.splitDataset(0.9)





print(type(X_train))
print(type(X_test))
print(type(Y_train))
print(type(Y_test))


print(np.shape(X_train))
print(np.shape(X_test))
print(np.shape(Y_train))
print(np.shape(Y_test))


