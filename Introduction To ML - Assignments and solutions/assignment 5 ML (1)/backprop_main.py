import backprop_data

import backprop_network
import matplotlib.pyplot as plt
import numpy as np

# training_data, test_data = backprop_data.load(train_size=10000, test_size=5000)
# net = backprop_network.Network([784, 40, 10])
# net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

#question b
def qb():
    learning_rate = [0.001, 0.01, 0.1, 1, 10, 100]
    training_data, test_data = backprop_data.load(train_size=10000, test_size=5000)
    net = backprop_network.Network([784, 40, 10])
    epochs = 30
    epochs_array = np.arange(epochs)
    train_acc = []
    train_loss = []
    test_acc = []

    for i in range(len(learning_rate)):
        train_acc_i,train_loss_i,test_acc_i = net.SGD(training_data, epochs=epochs, mini_batch_size=10, learning_rate=learning_rate[i], test_data=test_data)
        train_acc.append(train_acc_i)
        train_loss.append(train_loss_i)
        test_acc.append(test_acc_i)

    n = len(learning_rate)
    for i in range(n):
        plt.plot(epochs_array, train_acc[i], label="rate = {}".format(learning_rate[i]))
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('training accuracy')
    plt.show()

    for i in range(n):
        plt.plot(epochs_array, train_loss[i], label="rate = {}".format(learning_rate[i]))
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('training loss')
    plt.show()

    for i in range(n):
        plt.plot(epochs_array, test_acc[i], label="rate = {}".format(learning_rate[i]))
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('test accuracy')
    plt.show()

#question c
def qc():
    training_data, test_data = backprop_data.load(train_size=50000, test_size=10000)
    net = backprop_network.Network([784, 40, 10])
    net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

#question d
def qd():
    training_data, test_data = backprop_data.load(train_size=50000, test_size=10000)
    net = backprop_network.Network([784,200,40, 10])
    net.SGD(training_data, epochs=200, mini_batch_size=10, learning_rate=0.05, test_data=test_data)








def main():
    #qb()
    #qc()
    #qd()



if __name__ == "__main__":
    main()


#done
