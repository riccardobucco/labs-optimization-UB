import numpy as np
import matplotlib.pyplot as plt

X = np.arange(-2, +2, 0.1)
Y = X**3 - 2*X + 2
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.title('Graph of the function $f(x)=x^3-2x+2$ over the interval $[-2, +2]$')
plt.plot(X, Y)
# plt.show()
plt.savefig('../Images/01-one-dimensional-function-example.png')