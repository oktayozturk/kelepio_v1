# -*- coding: utf-8 -*-
# !/usr/bin/python

import pandas as pd
from scrapper import BikeDataScrapper
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data_management import datamanager as dm
from lin_reg_model import linear_regression_model as lr

sc = BikeDataScrapper()
sc.ls_brands()
sc.set_brand_by_index(101)
sc.ls_models()
sc.set_model_by_index(81)
sc.load_model_specs()
sc.write_specs_to_csv()


#sc.load_model_specs(model_index=0)
#print(sc.model_specs)

#sc.search_models("150")

# sc.load_brand_pages()
# print(sc.brand_pages)
#
# sc.load_model_detail_pages()
# print(sc.model_page_urls)
#
# sc.load_model_specs(model_index=5)
# print(sc.model_specs)



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
