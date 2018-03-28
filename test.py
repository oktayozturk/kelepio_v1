# -*- coding: utf-8 -*-
# !/usr/bin/python

import pandas as pd
import scrapper as sc
import matplotlib.pyplot as plt
import numpy as np
from data_management import datamanager as dm


b = dm("bmw","f 650 gs", polynomial_degree=1)


b.clear_uncorrelated_fields()

print(b.applyPCA(2))

