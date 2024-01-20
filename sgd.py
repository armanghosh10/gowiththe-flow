import numpy as np
from sklearn.metrics import mean_squared_error


class StochasticGradientDescent:
    def __init__(self, weights, biases, lr=0.01, epochs=1000, batch_size=1):
        self.learning_rate = lr
        self.num_epochs = epochs
        self.batch_size = batch_size
        self.num_weights = weights
        self.num_biases = biases
        self.w = np.random.randn(1, self.num_weights)
        self.b = np.random.randn(1, self.num_biases)

    def fit(self, X, y):
        for epoch in range(1, self.num_epochs+1):
            temp = X.sample(self.batch_size)
            X_tr = temp.iloc[:, 0:self.num_weights].values
            y_tr = temp.iloc[:, -1].values

            Lw = self.w
            Lb = self.b

            loss = 0
            y_pred = []

            for i in range(self.batch_size):
                Lw = (-2/self.batch_size*X_tr[i]) * \
                    (y_tr[i]-np.dot(X_tr[i], self.w.T)-self.b)
                Lb = (-2/self.batch_size) * \
                    (y_tr[i]-np.dot(X_tr[i], self.w.T)-self.b)

                self.w = self.w - self.learning_rate * Lw
                self.b = self.b - self.learning_rate * Lb

                y_predicted = np.dot(X_tr[i], self.w.T)
                y_pred.append(y_predicted)

        loss = mean_squared_error(y_pred, y_tr)

        epoch += 1
        self.learning_rate = self.learning_rate / 1.05

    def get_parameters(self):
        return self.w, self.b
