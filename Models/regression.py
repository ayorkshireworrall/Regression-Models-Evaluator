# -*- coding: utf-8 -*-
class Regression:
    def __init__(self, dataset):
        X, y = self.extract_variables(dataset)
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        
        
    def extract_variables(self, dataset):
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        return X, y
    
    def train_regressor(self):
        self.regressor.fit(self.X_train, self.y_train)
    
    def score(self):
        y_pred = self.regressor.predict(self.X_test)
        from sklearn.metrics import r2_score
        return r2_score(self.y_test, y_pred)
    
    def predict(self, X):
        return self.regressor.predict(X)
