import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns

def MSE(y_true, y_predicted): # Mean Squared Error
    return np.mean(np.square(y_true-y_predicted))

class LinearRegression:
    
    def __init__(self, learning_rate=0.001, n_it=1000):
        self.learning_rate = learning_rate
        self.n_it = n_it # Number of iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initializing the parameters of the gradient descent
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0 

        # Gradient Descent
        for _ in range(self.n_it):
            y_predicted = np.dot(X, self.weights) + self.bias # y = wx + b

            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y)) # Derivative of the weights
            db = (1/n_samples) * np.sum(y_predicted - y) # Derivative of the bias

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
    

if __name__ == '__main__':
    X, y = datasets.make_regression(n_samples=120, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    regressor = LinearRegression(learning_rate=0.02)
    regressor.fit(X_train, y_train)

    result = regressor.predict(X)
    mse = MSE(y, result)

    fig = plt.figure(figsize=(7,5))
    fig.subplots_adjust(bottom=0.2)

    plt.title('Simple Linear Regression', weight='bold')
    plt.xlabel('\nMSE: {:.2f}'.format(mse))

    sns.scatterplot(x=np.ravel(X_train), y=y_train, color='#D00000', label='Training Data')
    sns.scatterplot(x=np.ravel(X_test), y=y_test, color='#F48C06', label='Testing Data')

    sns.lineplot(x=np.ravel(X), y=result, color='#00B4D8', label='Prediction')
    plt.show()