# -*- coding: utf-8 -*-
# !/usr/bin/python

import pandas as pd
import scrapper as sc
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data_management import datamanager as dm
from lin_reg_model import linear_regression_model as lr




model = lr("apachi", "rtr 150")


model.alpha = 0.03
model.epochs = 10000
model.dropout_threshold = 0.5


model.train(verbose=True, save=True)

model.test(show_prices=True)


