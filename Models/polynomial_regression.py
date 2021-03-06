# Polynomial Regression

from .regression import Regression

class PolynomialRegression(Regression):
    def __init__(self, dataset):        
        X, y = self.extract_variables(dataset)
        
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        
        from sklearn.preprocessing import PolynomialFeatures
        self.poly_reg = PolynomialFeatures(degree = 4)
        from sklearn.linear_model import LinearRegression
        self.regressor = LinearRegression()
        
    def train_regressor(self):
        X_poly = self.poly_reg.fit_transform(self.X_train)
        self.regressor.fit(X_poly, self.y_train)
        
    def score(self):
        y_pred = self.regressor.predict(self.poly_reg.transform(self.X_test))
        from sklearn.metrics import r2_score
        return r2_score(self.y_test, y_pred)
        
    def predict(self, X):
        return self.regressor.predict(self.poly_reg.transform(X))