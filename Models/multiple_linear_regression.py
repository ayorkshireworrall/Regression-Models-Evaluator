# Multiple Linear Regression

from .regression import Regression

class MultipleLinearRegression(Regression):
    def __init__(self, dataset):
        super().__init__(dataset)
        from sklearn.linear_model import LinearRegression
        self.regressor = LinearRegression()
        