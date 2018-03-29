# -*- coding: utf-8 -*-
# !/usr/bin/python

import pandas as pd
import scrapper as sc
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data_management import datamanager as dm
from lin_reg_model import linear_regression_model as lr


y =  tf.constant([[1000, 1000 ]], dtype=tf.float32)
y_ = tf.constant([[1002, 1002 ]], dtype=tf.float32)

r1 = tf.sqrt(tf.losses.mean_squared_error(y,y_))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run(r1))



#model = lr("bmw", "f 650 gs")




# model.alpha = 0.001
# model.epochs = 20000
#
#
# model.train(verbose=True, save=True)
#
# print(model.Y_test.values)
#
# model.test()

