#################################
# Your name: Mira Zilberstein
#################################
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import random




def plot_results(models, titles, X, y, plot_sv=False):
    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(1, len(titles))  # 1, len(list(models)))

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    if len(titles) == 1:
        sub = [sub]
    else:
        sub = sub.flatten()
    for clf, title, ax in zip(models, titles, sub):
        # print(title)
        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        if plot_sv:
            sv = clf.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], c='k', s=60)

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.show()

def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

C_hard = 1000000.0  # SVM regularization parameter
C = 10
n = 100



# Data is labeled by a circle

radius = np.hstack([np.random.random(n), np.random.random(n) + 1.5])
angles = 2 * math.pi * np.random.random(2 * n)
X1 = (radius * np.cos(angles)).reshape((2 * n, 1))
X2 = (radius * np.sin(angles)).reshape((2 * n, 1))

X = np.concatenate([X1,X2],axis=1)
y = np.concatenate([np.ones((n,1)), -np.ones((n,1))], axis=0).reshape([-1])

#question a
def qa():
    model1 = svm.SVC(kernel='linear', C=10.0)
    model1.fit(X, y)
    model2 = svm.SVC(kernel='poly', degree=2, C=10.0,coef0=0) #c0 = 0 means the kernel is homogenous.
    model2.fit(X, y)
    model3 = svm.SVC(kernel='poly', C=10.0,coef0=0) #degree=3 by defult
    model3.fit(X, y)

    plot_results([model1,model2,model3], ["linear","homo poly(degree = 2)","homo poly(degree = 3)"], X, y)

#question b
def qb():
    model1 = svm.SVC(kernel='linear', C=10.0)
    model1.fit(X, y)
    model2 = svm.SVC(kernel='poly', degree=2, C=10.0,coef0=1) #c0 = 1 means the kernel is non-homogenous.
    model2.fit(X, y)
    model3 = svm.SVC(kernel='poly', C=10.0,coef0=1) #degree=3 by defult
    model3.fit(X, y)

    plot_results([model1,model2,model3], ["linear","non-homo poly(degree = 2)","non-homo poly(degree = 3)"], X, y)

#question c
def qc():
    y_after_change = []
    for i in range(len(y)):
        if(y[i] == -1):
            y_after_change.append(random.choices([1,-1], weights=(0.1,0.9))[0])
        else:
            y_after_change.append(y[i])

    model1 = svm.SVC(kernel='poly', degree=2, C=10.0, coef0=1)  # c0 = 1 means the kernel is non-homogenous.
    model1.fit(X, y_after_change)
    model2 = svm.SVC(kernel='rbf',gamma= 10 , C=10.0)
    model2.fit(X, y_after_change)

    plot_results([model1,model2], ["non-homo poly(degree = 2)","rbf(gamma = 10)"], X, y_after_change)

    model3 = svm.SVC(kernel='rbf',gamma= 40 , C=10.0)
    model3.fit(X, y_after_change)
    model4 = svm.SVC(kernel='rbf', gamma=100, C=10.0)
    model4.fit(X, y_after_change)
    model5 = svm.SVC(kernel='rbf', gamma=5, C=10.0)
    model5.fit(X, y_after_change)
    model6 = svm.SVC(kernel='rbf', gamma=1.5, C=10.0)
    model6.fit(X, y_after_change)

    plot_results([model6, model5,model2,model3,model4], ["rbf(gamma = 1.5)", "rbf(gamma = 5)","rbf(gamma = 10)","rbf(gamma = 40)","rbf(gamma = 100)"], X, y_after_change)




def main():
    qa()
    qb()
    qc()

if __name__ == "__main__":
    main()