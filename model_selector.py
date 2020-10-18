# -*- coding: utf-8 -*-
from Models import Regression
import pandas as pd
import numpy as np

dataset = pd.read_csv('Data.csv')
X = np.array([[14,41,1020,72], [10,40,1010,90]])
best_model = None

for regression in Regression.__subclasses__():
    model = regression(dataset)
    model.train_regressor()
    if best_model == None or best_model.score() < model.score():
        best_model = model

print('The best model is: ' + best_model.__class__.__name__)
print(best_model.score())
print(best_model.predict(X))