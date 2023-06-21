import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

def euclidean_distance(x, y):
    return np.sqrt(np.sum(np.square(x-y)))

def get_k_distances(X, k):
    return np.argsort(X)[:k]

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # Computing euclidean distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Finding the k nearest neighbours
        k_nearest_neighbours = get_k_distances(distances, self.k)
        k_labels = [self.y_train[i] for i in k_nearest_neighbours]
        # print(f'K labels: {k_labels}')
        
        # Returning the most common label
        most_common = Counter(k_labels).most_common(1)[0][0]
        return most_common

if __name__ == '__main__':
    cmap = ListedColormap(["#D00000", "#FFBA08", "#3F88C5"])
    wine_dataset = load_wine()

    data, labels = wine_dataset.data, wine_dataset.target
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

    model = KNN(k=3)
    
    # Training the model
    model.fit(train_data, train_labels)

    # Making predictions
    predictions = model.predict(test_data)

    # Calculating accuracy
    acc = np.sum(predictions == test_labels) / len(test_labels)
    print(f'Accuracy: {acc*100}%')

    # Plotting results
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    fig.subplots_adjust(bottom=0.25)

    axes[0].scatter(test_data[:,1], test_data[:,2], c=predictions, cmap=cmap)
    axes[0].set_title('Result')
    axes[0].set_xlabel('\nAccuracy: {:.2f}%'.format(acc*100))
    
    axes[1].scatter(test_data[:,1], test_data[:,2], c=test_labels, cmap=cmap)
    axes[1].set_title('Original')
    
    # plt.tight_layout()
    plt.suptitle('KNN - Wine Dataset', y=1, fontweight='bold')
    plt.show()