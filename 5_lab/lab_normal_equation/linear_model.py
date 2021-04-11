import numpy as np



class LinearRegression(object):
    def __init__(self, fit_intercept=True, copy_X=True):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X

        self._coef = None
        self._intercept = None
        self._new_X = None

    def fit(self, X, Y):
        self._coef = np.dot(np.dot(np.linalg.inv(np.dot(X.T ,X)),X.T),Y)

    def predict(self, X):
        self._new_X = X
        self._intercept = np.dot(X,self._coef)

    @property
    def coef(self):
        return self._coef

    @property
    def intercept(self):
        return self._intercept
