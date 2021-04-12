import numpy as np
import pandas as pd

class LinearRegressionGD(object):
    def __init__(self, fit_intercept=True, copy_X=True,
                 alpha=0.0001, epochs=1000000, weight_decay=0.9):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self._alpha = alpha
        self._epochs = epochs

        self._cost_history = []

        self._coef = None
        self._intercept = None
        self._new_X = None
        self._w_history = None
        self._weight_decay = weight_decay

    def cost(self, h, y):
        m = y.size
        predictions = h
        sqErrors = (predictions - y)

        J = (1.0/(2*m)) * sqErrors.T.dot(sqErrors)
        return J

    def hypothesis_function(self, X, theta):
        return X.dot(theta)

    def gradient(self, X, y, theta):
        m = y.size
        cost_history = []
        theta_history = []

        for _ in range(self._epochs):
            predictions = X.dot(theta)

            for i in range(theta.size):
                partial_marginal = X[:,i]
                errors_xi = (predictions - y) * partial_marginal
                theta[i] = theta[i] - self._alpha * (1.0/m) * errors_xi.sum()

            if _ % 1000 == 0:
                theta_history.append(theta)
                cost_history.append(self.cost(self.hypothesis_function(X,theta),y))

        return theta,np.array(cost_history),np.array(theta_history)



    def fit(self, X, y):
        # Write your code

        for epoch in range(self._epochs):
            # 아래 코드를 반드시 활용할 것
            gradient = self.gradient(self._new_X, y, theta).flatten()

            # Write your code

            if epoch % 100 == 0:
                self._w_history.append(theta)
                cost = self.cost(
                    self.hypothesis_function(self._new_X, theta), y)
                self._cost_history.append(cost)
            self._eta0 = self._eta0 * self._weight_decay

        # Write your code

    def predict(self, X):
        pass

    @property
    def coef(self):
        return self._coef

    @property
    def intercept(self):
        return self._intercept

    @property
    def weights_history(self):
        return np.array(self._w_history)

    @property
    def cost_history(self):
        return self._cost_history
