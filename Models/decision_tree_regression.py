# Decision Tree Regression

from .regression import Regression

class DecisionTreeRegression(Regression):    
    
    def __init__(self, dataset):
        super().__init__(dataset)
        from sklearn.tree import DecisionTreeRegressor
        self.regressor = DecisionTreeRegressor(random_state=0)