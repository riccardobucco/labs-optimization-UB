import numpy as np
import matplotlib.pyplot as plt

def getGamma(t):
    return 1.0/t

def f(X, y, w, b, lamb):
    tmp = 0
    tmp += lamb/2 * np.dot(w.T, w)
    for i in range(0,X.shape[0]):
        tmp += max(0, 1-y[i]*(np.dot(w.T, X[:,i]) + b))
    return tmp

def stochasticGradient(X, y, get_gamma, start_w, start_b, lamb=0.00001, n_iters=200000):
    w = start_w
    b = start_b
    for t in range(1,n_iters):
        gamma = get_gamma(t)
        i = np.random.randint(0, np.shape(X)[1])
        xt = X[:,i]
        yt = y[i]
        if (yt * (np.dot(w.T, xt) + b)) > 1:
            w = w - gamma * lamb * w
            b = b
        else:
            w = w - gamma * (lamb * w - yt*xt)
            b = b - gamma * (-yt)
        print(t)
        print("gamma: ", gamma)
        print("w: ", w)
        print("b: ", b)
    return w, b

X = np.array([[0,1,1,2,2,3,4,0,1,2,3,4,5,5],[1,0,2,0,1,1,0,5,5,4,3,4,5,2]])
y = np.array([-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1])
Y = np.eye(14)*y
start_w = np.ones(np.shape(X)[0])
start_b = -2.9
w, b = stochasticGradient(X, y, getGamma, start_w, start_b, 0.001, 10)
print(f(X,y,w,b,0.001))

# Create a new figure
fig = plt.figure()
# Split the figure in 1*1 (nrows*ncols) subaxes and create a new suplot positioned at 1 (plot_number)
nrows = 1
ncols = 1
plot_number = 1
ax = fig.add_subplot(nrows, ncols, plot_number)
# Set the labels for the axes
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.scatter(X[0,:7],X[1,:7],color='r')
ax.scatter(X[0,7:],X[1,7:],color='b')
# Plot the line
x_vals = np.linspace(0,5,500)
f_x_vals = (-b - w[0]*x_vals)/w[1]
ax.plot(x_vals,f_x_vals)
# Save the current figure
plt.savefig('../Images/02-lab06-14points-with-line-tmp.png')

X = np.array([[1.3,1.5,1,0.5,0.6,0.1,0.7,0,1,1,2,2,3,4,0,2.5,5,2.9,1.4,1.5,4.8,4.2,2.1,1,0.2,0.5,
1,2,3,4,5,5,3.2,4.5,4.9,2.5,4,1,1.2,0.5,3.5,0.2,1.7,4.5,3.1,2.5,3.5,3.6,4.8,4.1],
[1.5,2.3,2.2,0.5,0.4,2.2,1.8,1.9,0,2,0,2.3,1,0,2.6,1,0.8,0.2,0.8,1.9,0.1,1,0.5,1.1,1.3,3,
5,4,3,4,5,2,4,3,4.2,4.5,2.3,4.3,3.8,4,4.8,4.5,4.9,1.8,5,3.6,3.2,3.4,2.5,4.9]])
y = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
Y = np.eye(50)*y
# Create a new figure
fig = plt.figure()
# Split the figure in 1*1 (nrows*ncols) subaxes and create a new suplot positioned at 1 (plot_number)
nrows = 1
ncols = 1
plot_number = 1
ax = fig.add_subplot(nrows, ncols, plot_number)
# Set the labels for the axes
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.scatter(X[0,:26],X[1,:26],color='r')
ax.scatter(X[0,26:],X[1,26:],color='b')
# Save the current figure
plt.savefig('../Images/02-lab06-50points.png')
start_w = np.ones(np.shape(X)[0])
start_b = -2.9
w, b = stochasticGradient(X, y, getGamma, start_w, start_b, 0.0001, 20)
print(f(X,y,w,b,0.0001))
# Plot the line
x_vals = np.linspace(0,5,500)
f_x_vals = (-b - w[0]*x_vals)/w[1]
ax.plot(x_vals,f_x_vals)
# Save the current figure
plt.savefig('../Images/02-lab06-50points-with-line-tmp.png')