# -*- coding: utf-8 -*-
# !/usr/bin/python

import pandas as pd
import scrapper as sc
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data_management import datamanager as dm
from lin_reg_model import linear_regression_model as lr




model = lr("bmw", "r 1200 gs")


model.alpha = 0.01
model.epochs = 1000

model.train(verbose=True, save=True)

print(model.Y_test.values)

model.test()