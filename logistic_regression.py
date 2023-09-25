import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from matplotlib.colors import ListedColormap


class LogisticRegression:

    def __init__(self, learning_rate=0.001, n_it=1000):
        self.learning_rate = learning_rate
        self.n_it = n_it
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_it):
            linear_model = np.dot(X, self.weights) + self.bias # mx + b
            y_predicted = self._sigmoid(linear_model)
            # print(f'y_predicted: {y_predicted}')

            # Calculate Derivatives
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted-y))
            db = (1 / n_samples) * np.sum(y_predicted-y)

            # Updating Parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db 

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias # mx + b
        y_predicted = self._sigmoid(linear_model)

        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_class

    def _sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))
    

if __name__ == '__main__':
    dataset = datasets.load_breast_cancer()
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1330)

    def acc(y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

    regressor = LogisticRegression(learning_rate=0.0001, n_it=1000)
    regressor.fit(X_train, y_train)

    p = regressor.predict(X_test)

    # Plotting data
    cmap = ListedColormap(['#FF0000', '#147DF5'])

    fig, axs = plt.subplots(1,2,figsize=(10,7))
    fig.suptitle("Breast Cancer Prediction w/ Logistic Regression", weight='bold')

    axs[0].scatter(X_test[:,0], X_test[:,1], c=y_test, cmap=cmap)
    axs[0].set_title("Original")

    axs[1].scatter(X_test[:,0], X_test[:,1], c=p, cmap=cmap)
    axs[1].set_title("Prediction")
    axs[1].set_xlabel("\nAccuracy: {:.2f}%".format(100*acc(y_test, p)))

    plt.show()