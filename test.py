# -*- coding: utf-8 -*-
# !/usr/bin/python

import pandas as pd
import scrapper as sc
import matplotlib.pyplot as plt
import numpy as np
from data_management import datamanager as dm


b = dm("ktm","200 duke", polynomial_degree=1)

print(b.X.corr()["price"])
