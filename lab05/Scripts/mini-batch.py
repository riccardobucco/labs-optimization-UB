import numpy as np
import matplotlib.pyplot as plt
from math import log

def f(X, y, w, b, lamb):
    tmp = 0
    tmp += lamb/2 * np.dot(w.T, w)
    for i in range(0,X.shape[0]):
        tmp += max(0, 1-y[i]*(np.dot(w.T, X[:,i]) + b))
    return tmp

def getGamma(t):
    return 1.0/t

def stochasticGradientMiniBatch(X, y, get_gamma, start_w, start_b, mbatch_size, lamb, n_iters):
    w = start_w
    b = start_b
    f_values = []
    f_values.append(log(f(X,y,w,b,lamb)))
    for t in range(1, n_iters):
        gamma = get_gamma(t)
        indexes = np.random.choice(range(0,X.shape[1]), mbatch_size)
        x_samples = X[:,indexes]
        y_samples = y[indexes]
        sum_w = 0.0
        sum_b = 0.0
        for k in range(0, mbatch_size):
            if y_samples[k] * (np.dot(w.T, x_samples[:,k]) + b) > 1:
                sum_w += 0
                sum_b += 0
            else:
                sum_w += -y_samples[k]*x_samples[:,k]
                sum_b += -y_samples[k]
        w = w - gamma * (lamb * w + sum_w)
        b = b - gamma * sum_b
        f_values.append(log(f(X,y,w,b,lamb)))
    return w, b, f_values

lamb = 0.001
mbatch_size = 10
n_iters = 200
start_w = np.array([1,1])
start_b = -4.4

X = np.array([[0,1,1,2,2,3,4,0,1,2,3,4,5,5],[1,0,2,0,1,1,0,5,5,4,3,4,5,2]])
y = np.array([-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1])
Y = np.eye(14)*y

w, b, f_values = stochasticGradientMiniBatch(X, y, getGamma, start_w, start_b, mbatch_size, lamb, n_iters)
print(w,b)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(X[0,:7],X[1,:7],color='r')
ax.scatter(X[0,7:],X[1,7:],color='b')
x_vals = np.linspace(0,5,100)
f_x_vals = (-b - w[0]*x_vals)/w[1]
ax.plot(x_vals,f_x_vals)
plt.savefig('../Images/tmp-points.png')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(range(0,len(f_values)),f_values)
plt.savefig('../Images/tmp-error.png')

lamb = 0.001
mbatch_size = 20
n_iters = 200
start_w = np.array([1,1])
start_b = -4.4

X = np.array([[1.3,1.5,1,0.5,0.6,0.1,0.7,0,1,1,2,2,3,4,0,2.5,5,2.9,1.4,1.5,4.8,4.2,2.1,1,0.2,0.5,
1,2,3,4,5,5,3.2,4.5,4.9,2.5,4,1,1.2,0.5,3.5,0.2,1.7,4.5,3.1,2.5,3.5,3.6,4.8,4.1],
[1.5,2.3,2.2,0.5,0.4,2.2,1.8,1.9,0,2,0,2.3,1,0,2.6,1,0.8,0.2,0.8,1.9,0.1,1,0.5,1.1,1.3,3,
5,4,3,4,5,2,4,3,4.2,4.5,2.3,4.3,3.8,4,4.8,4.5,4.9,1.8,5,3.6,3.2,3.4,2.5,4.9]])
y = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
Y = np.eye(50)*y

w, b, f_values = stochasticGradientMiniBatch(X, y, getGamma, start_w, start_b, mbatch_size, lamb, n_iters)
print(w,b)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(X[0,:26],X[1,:26],color='r')
ax.scatter(X[0,26:],X[1,26:],color='b')
x_vals = np.linspace(0,5,100)
f_x_vals = (-b - w[0]*x_vals)/w[1]
ax.plot(x_vals,f_x_vals)
plt.savefig('../Images/tmp-points.png')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(range(0,len(f_values)),f_values)
plt.savefig('../Images/tmp-error.png')