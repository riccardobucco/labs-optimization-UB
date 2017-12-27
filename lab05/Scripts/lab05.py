import matplotlib.pyplot as plt
import numpy as np
from library import general_case_lu

def NLACode_to_SVM(X, Y, K=1000000):
    # j is the index of a support vector to calculate b because as of now
    # I do not know how to figure out which point is a support vector.
    n = X.shape[1]
    m = 2*n
    p = 1
    G = np.dot(np.dot(Y,X.T),np.dot(X,Y))
    g = -np.ones(n)
    A = np.reshape(np.diag(Y),(n,1))
    C = np.c_[np.eye(n),-np.eye(n)]
    d = np.append(np.zeros(n),np.repeat(-K,n))
    b = 0
    x = np.ones(n)
    gam = np.ones(p)
    s = np.ones(m)
    lam = np.ones(m)
    x_results, k_results, t_results = general_case_lu(G,g,C,d,A,b,x,gam,lam,s)
    print(x_results)
    print(k_results)
    print(t_results)
    w_results = np.dot(np.dot(X,Y),x_results)
    tmp = 0.0
    tot = 0
    for j, x_result in enumerate(x_results):
        if x_result >= 0 and x_result<=K:
            tmp += np.diag(Y)[j]-np.dot(w_results.T,X[:,j])
            tot += 1
        else:
            print("NOT USED: ", X[:,j], " ", np.diag(Y)[j], " ALPHA: ", x_result)
    print("TOT: ",tot)
    b_result = tmp/tot # possible division by 0!
    return w_results,b_result

X = np.array([[0,1,0,1],[1,0,3,2]])
Y = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
w_results, b_result = NLACode_to_SVM(X,Y)
print("W:",w_results," b:",b_result)

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
ax.scatter(X[0,:2],X[1,:2],color='r')
ax.scatter(X[0,2:],X[1,2:],color='b')
# Save the current figure
plt.savefig('../Images/01-lab05-4points.png')
# Plot the line
x_vals = np.linspace(0,1,100)
f_x_vals = (-b_result - w_results[0]*x_vals)/w_results[1]
ax.plot(x_vals,f_x_vals)
# Save the current figure
plt.savefig('../Images/01-lab05-4points-with-line.png')

X = np.array([[0,1,1,2,2,3,4,0,1,2,3,4,5,5],[1,0,2,0,1,1,0,5,5,4,3,4,5,2]])
Y = np.eye(14)*np.array([-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1])
w_results, b_result = NLACode_to_SVM(X,Y)
print("W:",w_results," b:",b_result)
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
# Save the current figure
plt.savefig('../Images/01-lab05-14points.png')
# Plot the line
x_vals = np.linspace(0,5,500)
f_x_vals = (-b_result - w_results[0]*x_vals)/w_results[1]
ax.plot(x_vals,f_x_vals)
# Save the current figure
plt.savefig('../Images/01-lab05-14points-with-line.png')

X = np.array([[0,1,1,2,2,2,3,4,0,1,2,3,2,4,5,5],[1,4,2,0,1,3,1,0,4,5,4,2,2,4,5,2]])
Y = np.eye(16)*np.array([-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1])
w_results, b_result = NLACode_to_SVM(X,Y)
print("W:",w_results," b:",b_result)
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
ax.scatter(X[0,:8],X[1,:8],color='r')
ax.scatter(X[0,8:],X[1,8:],color='b')
# Save the current figure
plt.savefig('../Images/01-lab05-16points.png')
# Plot the line
x_vals = np.linspace(0,5,500)
f_x_vals = (-b_result - w_results[0]*x_vals)/w_results[1]
ax.plot(x_vals,f_x_vals)
# Save the current figure
plt.savefig('../Images/01-lab05-16points-with-line-K=1000000.png')

w_results, b_result = NLACode_to_SVM(X,Y, 100)
print("W:",w_results," b:",b_result)
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
ax.scatter(X[0,:8],X[1,:8],color='r')
ax.scatter(X[0,8:],X[1,8:],color='b')
# Plot the line
x_vals = np.linspace(0,5,500)
f_x_vals = (-b_result - w_results[0]*x_vals)/w_results[1]
ax.plot(x_vals,f_x_vals)
# Save the current figure
plt.savefig('../Images/01-lab05-16points-with-line-K=100.png')

X = np.array([[1,1,1,2,2,2,2,3,3.5,3.5,3.5,3.5,4.5,4.5,4.5,5],[4,5,6,4,5,6,7,1,2,3,4,5,3,4,5,6]])
Y = np.eye(16)*np.array([-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1])
w_results, b_result = NLACode_to_SVM(X,Y, 100)
print("W:",w_results," b:",b_result)
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
ax.scatter(X[0,:8],X[1,:8],color='r')
ax.scatter(X[0,8:],X[1,8:],color='b')
# Save the current figure
plt.savefig('../Images/01-lab05-16points-special.png')

# Create a new figure
fig = plt.figure()
# Split the figure in 1*1 (nrows*ncols) subaxes and create a new suplot positioned at 1 (plot_number)
nrows = 1
ncols = 2
plot_number = 1
ax = fig.add_subplot(nrows, ncols, plot_number)
# Set the labels for the axes
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.scatter(X[0,:8],X[1,:8],color='r')
ax.scatter(X[0,8:],X[1,8:],color='b')
# Plot the line
x_vals = np.linspace(1,4,100)
f_x_vals = (-b_result - w_results[0]*x_vals)/w_results[1]
ax.plot(x_vals,f_x_vals)

w_results, b_result = NLACode_to_SVM(X,Y)
plot_number = 2
ax = fig.add_subplot(nrows, ncols, plot_number)
# Set the labels for the axes
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.scatter(X[0,:8],X[1,:8],color='r')
ax.scatter(X[0,8:],X[1,8:],color='b')
# Plot the line
x_vals = np.linspace(1,5,500)
f_x_vals = (-b_result - w_results[0]*x_vals)/w_results[1]
ax.plot(x_vals,f_x_vals)
# Save the current figure
plt.savefig('../Images/01-lab05-16points-special-with-line.png')