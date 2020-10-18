# Multiple Linear Regression

class MultipleLinearRegression:
    def __init__(self, dataset):
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        from sklearn.model_selection import train_test_split
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        
        from sklearn.linear_model import LinearRegression
        self.regressor = LinearRegression()
        self.regressor.fit(X_train, y_train)
        
    def score(self):
        y_pred = self.regressor.predict(self.X_test)
        
        from sklearn.metrics import r2_score
        return r2_score(self.y_test, y_pred)
    
    def predict(self, X):
        return self.regressor.predict(X)