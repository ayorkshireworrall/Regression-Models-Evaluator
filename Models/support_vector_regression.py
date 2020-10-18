# Support Vector Regression (SVR)

from .regression import Regression

class SupportVectorRegression(Regression):
    def __init__(self, dataset):        
        super().__init__(dataset)
        from sklearn.preprocessing import StandardScaler
        self.sc_X = StandardScaler()
        self.sc_y = StandardScaler()
        self.X_train = self.sc_X.fit_transform(self.X_train)
        self.y_train = self.sc_y.fit_transform(self.y_train)
        
        from sklearn.svm import SVR
        self.regressor = SVR(kernel = 'rbf')

    def score(self):
        y_pred = self.sc_y.inverse_transform(self.regressor.predict(self.sc_X.transform(self.X_test)))
        from sklearn.metrics import r2_score
        return r2_score(self.y_test, y_pred)
        
    def predict(self, X):
        return self.sc_y.inverse_transform(self.regressor.predict(self.sc_X.transform(X)))
    
    def extract_variables(self, dataset):
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        y = y.reshape(len(y),1)
        return X, y