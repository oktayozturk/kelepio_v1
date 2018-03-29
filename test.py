# -*- coding: utf-8 -*-
# !/usr/bin/python

import pandas as pd
import scrapper as sc
import matplotlib.pyplot as plt
import numpy as np
from data_management import datamanager as dm
from lin_reg_model import linear_regression_model as lr


model = lr("bmw", "f 650 gs")

model.alpha = 0.01
model.epochs = 2000


model.train()

