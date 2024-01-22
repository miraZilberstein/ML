import matplotlib.pyplot as plt
import numpy as np
import numpy.random

# Load MNIST dataset
from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784", as_frame=False)
data = mnist["data"]
labels = mnist["target"]

# Split dataset into training and test sets
idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]

# Function to calculate Euclidean L2 distance
def euclidean_l2_metric(image, train):
    # Efficient vectorized computation of L2 distances
    return np.sum(np.square(np.subtract(image, train)), axis=1)

# k-Nearest Neighbors classifier function
def kNN(train, train_labels, image, k):
    euclidean_distance = euclidean_l2_metric(image, train)
    k_indexes = np.argpartition(euclidean_distance, k)[:k]
    return np.argmax(np.bincount(train_labels[k_indexes].astype(int)))

# Evaluate accuracy of kNN classifier
def evaluate_accuracy(test, test_labels, train, train_labels, k, n):
    to_train = train[:n,]
    to_train_labels = train_labels[:n,]
    test_results = [kNN(to_train, to_train_labels, test[i], k) for i in range(len(test))]
    return np.sum(test_results == test_labels.astype(int)) / len(test)

def main():
    # Evaluate and print accuracy for k=10 and n=1000
    print("Accuracy of the prediction with k=10 and n=1000: ", evaluate_accuracy(test, test_labels, train, train_labels, 10, 1000))

    # Plot accuracy as a function of k (1 to 100)
    plt.figure(1)
    ks = range(1, 101)
    accuracies = [evaluate_accuracy(test, test_labels, train, train_labels, k, 1000) for k in ks]
    plt.plot(ks, accuracies)
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('Accuracy as a function of k')

    # Plot accuracy as a function of training set size (n)
    plt.figure(2)
    ns = range(100, 5001, 100)
    accuracies_n = [evaluate_accuracy(test, test_labels, train, train_labels, 1, n) for n in ns]
    plt.plot(ns, accuracies_n)
    plt.xlabel('n')
    plt.ylabel('Accuracy')
    plt.title('Accuracy as a function of training set size (n)')
    plt.show()

if __name__ == "__main__":
    main()
