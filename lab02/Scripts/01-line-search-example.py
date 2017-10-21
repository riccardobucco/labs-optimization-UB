# Line search example

import numpy as np
import scipy as sp
import scipy.optimize

# function evaluation
def func(x):
    return x[0]**2 * (4 - 2.1 * x[0]**2 + 1/3 * x[0]**4) + x[0]*x[1] + x[1]**2 * (-4 + 4 * x[1]**2)
    
# first order derivatives of the function
def dfunc(x):
    return np.array([2*x[0]**5 - 8.4*x[0]**3 + 8*x[0] + x[1], 16*x[1]**3 - 8*x[1] + x[0]])
    
def gradient_descent(x):
    niter = 0
    while niter <= 1000:  # maximum number of iterations
        pk = - dfunc(x)   # Search direction (observe the minus sign)
        return_values = sp.optimize.line_search(func, dfunc, x, pk)
	alpha = return_values[0]
        temp = x+alpha*pk # Observe the plus sign 
        if np.abs(func(temp)-func(x))>0.001: # convergence condition
            x = temp
        else:
            break
        niter += 1
    return x, niter

starting_points = np.array([[1,0], [0.6, -0.3])
for starting_point in starting_points:
    val, iters = gradient_descent(starting_point)
    print ("x = ", val)
    print ("iterations = ", iters)