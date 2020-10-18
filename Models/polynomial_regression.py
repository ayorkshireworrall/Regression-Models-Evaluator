# Polynomial Regression

class PolynomialRegression:
    def __init__(self, dataset):        
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        from sklearn.model_selection import train_test_split
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        self.poly_reg = PolynomialFeatures(degree = 4)
        X_poly = self.poly_reg.fit_transform(X_train)
        self.regressor = LinearRegression()
        self.regressor.fit(X_poly, y_train)
        
    def score(self):
        y_pred = self.regressor.predict(self.poly_reg.transform(self.X_test))
        from sklearn.metrics import r2_score
        return r2_score(self.y_test, y_pred)
        
    def predict(self, X):
        return self.regressor.predict(self.poly_reg.transform(X))