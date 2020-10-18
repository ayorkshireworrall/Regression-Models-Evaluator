# Random Forest Regression

from .regression import Regression

class RandomForestRegression(Regression):
    def __init__(self, dataset):        
        super().__init__(dataset)
        from sklearn.ensemble import RandomForestRegressor
        self.regressor = RandomForestRegressor(n_estimators=10, random_state=0)
        
    