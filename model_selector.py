# -*- coding: utf-8 -*-
from Models import *
import pandas as pd
import numpy as np

dataset = pd.read_csv('Data.csv')
model1 = DecisionTreeRegression(dataset)
model2 = MultipleLinearRegression(dataset)
model3 = PolynomialRegression(dataset)
model4 = SupportVectorRegression(dataset)
model5 = RandomForestRegression(dataset)
X = np.array([[14,41,1020,72], [10,40,1010,90]])
print(model1.score())
print(model2.score())
print(model3.score())
print(model4.score())
print(model5.score())
print(model1.predict(X))
print(model2.predict(X))
print(model3.predict(X))
print(model4.predict(X))
print(model5.predict(X))