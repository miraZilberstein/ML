#################################
# Your name: Mira Zilberstein
#################################


import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt
from scipy.special import softmax

def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels



def SGD_hinge(data, labels, C, eta_0, T):
    w = np.zeros(len(data[0]))
    data_len = len(data)
    for t in range(1, T+1): # t=num of iteration
        i = np.random.randint(data_len)
        y_i,x_i = labels[i], data[i]
        curr = y_i*np.dot(w,x_i)
        eta_0_t = eta_0 / t
        w = (1 - eta_0_t) * w
        if curr < 1:
             w = w + eta_0_t * C * y_i * x_i
    return w


#loss func = ln(1 + e^-dot(yw,x)),
#gradient = (-y_i * x_i*e^-dot(yw,x)) / (1 + e^-dot(yw,x))
def SGD_log(data, labels, eta_0, T,plot_norm = False):
    if plot_norm:
        norms = np.zeros(T)
    w = np.zeros(len(data[0]))
    data_len = len(data)
    for t in range(1, T+1): # t=num of iteration
        i = np.random.randint(data_len)
        y_i,x_i = labels[i], data[i]
        eta_0_t = eta_0 / t
        if plot_norm:
            norms[t-1] = np.linalg.norm(w)
        w = w - eta_0_t*gradient(w,x_i,y_i)
    if plot_norm:
        plt.plot(range(T),norms)
        plt.legend()
        plt.xlabel('iteration')
        plt.ylabel('norm of w')
        plt.show()
    return w


############################################ additional functions for question 1

def accuracy(w,validation_data, validation_labels):
    counter = 0
    n = len(validation_data)
    for i in range(n):
        ytag = -1
        if(np.dot(w,validation_data[i]) >= 0):
            ytag = 1
        if(ytag == validation_labels[i]):
            counter += 1
    return counter/n

#question 1a
def q1a(train_data, train_labels, validation_data, validation_labels):
    T = 1000
    C = 1
    eta_0 =[10 ** (-5 + i) for i in range(10)] #[0.01 + 0.01*i for i in range(200)]
    average_accuracy = np.zeros(len(eta_0))
    for i in range(len(eta_0)):
        curr_average_accuracy = 0
        for run in range(10):
            w = SGD_hinge(train_data, train_labels, C, eta_0[i], T)
            curr_average_accuracy += accuracy(w,validation_data, validation_labels)
        curr_average_accuracy /= 10
        average_accuracy[i] = curr_average_accuracy
    plt.plot(eta_0,average_accuracy)
    plt.legend()
    plt.xlabel('eta_0')
    plt.ylabel('average_accuracy')
    plt.xscale('log')
    plt.show()

#question 1b
def q1b(train_data, train_labels, validation_data, validation_labels):
    T = 1000
    C = [10 ** (-5 + i) for i in range(10)]
    eta_0 = 0.73
    average_accuracy = np.zeros(len(C))
    for i in range(len(C)):
        curr_average_accuracy = 0
        for run in range(10):
            w = SGD_hinge(train_data, train_labels, C[i], eta_0, T)
            curr_average_accuracy += accuracy(w,validation_data, validation_labels)
        curr_average_accuracy /= 10
        average_accuracy[i] = curr_average_accuracy
    plt.plot(C,average_accuracy)
    plt.legend()
    plt.xlabel('C')
    plt.ylabel('average_accuracy')
    plt.xscale('log')
    plt.show()

#question 1c,d
def q1cd(train_data, train_labels, test_data, test_labels):
    T = 2000
    C = 10 ** (-4)
    eta_0 = 0.73
    w = SGD_hinge(train_data, train_labels, C, eta_0, T)
    curr_accuracy = accuracy(w, test_data, test_labels)
    print("the accurancy is: ", curr_accuracy)
    plt.imshow(w.reshape(28,28), interpolation = 'nearest')
    plt.show()



############################################ additional functions for question 2

#returns the gradient for log loss
def gradient(w,x_i,y_i):
    return -1 * y_i * x_i * (softmax([0,-y_i * np.dot(w,x_i)])[1])

#question 2a
def q2a(train_data, train_labels, validation_data, validation_labels):
    T = 1000
    eta_0 =[10 ** (-5 + i) for i in range(10)]
    average_accuracy = np.zeros(len(eta_0))
    for i in range(len(eta_0)):
        curr_average_accuracy = 0
        for run in range(10):
            w = SGD_log(train_data, train_labels,eta_0[i], T)
            curr_average_accuracy += accuracy(w,validation_data, validation_labels)
        curr_average_accuracy /= 10
        average_accuracy[i] = curr_average_accuracy
    plt.plot(eta_0,average_accuracy)
    plt.legend()
    plt.xlabel('eta_0')
    plt.ylabel('average_accuracy')
    plt.xscale('log')
    plt.show()

#question 2b
def q2b(train_data, train_labels, test_data, test_labels):
    T = 2000
    eta_0 = 10 ** (-5)
    w = SGD_log(train_data, train_labels, eta_0, T)
    curr_accuracy = accuracy(w, test_data, test_labels)
    print("the accurancy is: ", curr_accuracy)
    plt.imshow(w.reshape(28, 28), interpolation='nearest')
    plt.show()

############################################ main

def main():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    #q1a(train_data, train_labels, validation_data, validation_labels)
    #q1b(train_data, train_labels, validation_data, validation_labels)
    #q1cd(train_data, train_labels, test_data, test_labels)
    #q2a(train_data, train_labels, validation_data, validation_labels)
    #q2b(train_data, train_labels, test_data, test_labels)
    #SGD_log(train_data, train_labels, 10**(-5), 20000, plot_norm=True) #question 2c

if __name__ == "__main__":
    main()

#################################