import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skds
from sklearn.model_selection import train_test_split

class Perceptron:

    def __init__(self, learning_rate=0.2, n_it=100):
        self.learning_rate = learning_rate
        self.n_it = n_it
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        plt.ion()
        fig, ax = plt.subplots()

        for epoch in range(self.n_it+1):
            for i, x in enumerate(X):
                output = np.dot(x, self.weights) + self.bias
                y_pred = self.unit_step_function(output)

                delta = self.learning_rate * (y[i] - y_pred)
                self.weights += delta * x
                self.bias += delta

            if epoch % 10 == 0:
                print(f'Epoch: {epoch}')

                y_pred = model.predict(X)
                m = -self.weights[0] / self.weights[1]
                b = -self.bias / self.weights[1]

                ax.clear()
                ax.set_title('Binary Classification w/ Perceptron - Training Phase', weight='bold')
                ax.scatter(X[:, 0], X[:, 1], c=y_pred)
                ax.plot(X[:, 0], X[:, 0] * m + b, color='black')

                plt.pause(0.9)

        plt.ioff()
        plt.show()

    def predict(self, X):
        output = np.dot(X, self.weights) + self.bias
        y_pred = self.unit_step_function(output)

        return y_pred

    def unit_step_function(self, x):
        return np.where(x >= 0, 1, 0)
    
    def accuracy_score(self, target, y_pred):
        result = np.sum(target == y_pred)
        
        return result / len(target)

if __name__ == '__main__':
    X, y = skds.make_blobs(n_samples=500, centers=2, cluster_std=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    
    model = Perceptron()

    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    acc = model.accuracy_score(y_test, prediction)

    # Plotting
    plt.scatter(X_test[:, 0], X_test[:, 1], c=prediction)
    plt.suptitle('Binary Classification w/ Perceptron - Making Predictions', weight='bold')
    plt.xlabel('\nAccuracy on test dataset: {:.2f}%'.format(round(acc*100, 2)))

    plt.tight_layout()
    plt.show()