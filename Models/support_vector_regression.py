# Support Vector Regression (SVR)

class SupportVectorRegression:
    def __init__(self, dataset):        
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        y = y.reshape(len(y),1)
        
        from sklearn.model_selection import train_test_split
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

        from sklearn.preprocessing import StandardScaler
        self.sc_X = StandardScaler()
        self.sc_y = StandardScaler()
        X_train = self.sc_X.fit_transform(X_train)
        y_train = self.sc_y.fit_transform(y_train)
        
        from sklearn.svm import SVR
        self.regressor = SVR(kernel = 'rbf')
        self.regressor.fit(X_train, y_train)

    def score(self):
        y_pred = self.sc_y.inverse_transform(self.regressor.predict(self.sc_X.transform(self.X_test)))

        from sklearn.metrics import r2_score
        return r2_score(self.y_test, y_pred)
        
    def predict(self, X):
        return self.sc_y.inverse_transform(self.regressor.predict(self.sc_X.transform(X)))