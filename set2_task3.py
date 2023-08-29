from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import axes3d
matplotlib.use('TkAgg')

def sigmoid(w, X) :
    z = X  @ w.transpose()
    return 1/(1+np.exp(-z))

N_1 = multivariate_normal([0,0], cov=1)
N_2 = multivariate_normal([1,1], cov=1)

bias = np.ones((3,1))
X_1 = N_1.rvs(3)
X_1 = np.hstack((bias, X_1))
X_2 = N_2.rvs(3)
X_2 = np.hstack((bias, X_2))

# Suggested parameters for decision boundary. Here w[0] is the constant w_0
w = np.array([-1, 1, 1])
# w = np.array([-1, 1, 1])

z_1 = sigmoid(X_1, w)
z_2 = sigmoid(X_2, w)

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
ax.scatter(X_1[:,0], X_1[:,1], z_1, color = "green")
ax.scatter(X_2[:,0], X_2[:,1], z_2, color = "red")

plt.show()
