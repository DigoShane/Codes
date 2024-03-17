#This program is to solve the 1D touching plane problem with visualization

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math

#Defining parameters
XL=-20
XR=50
s1=1000

#defining the function
def h1(x):
 return x**4/4-(50/3)*x**3+(0.01/2+1/8)*(50*x)**2
def f(x):
 return h1(x)-s1.x


#Minimizing the function
w = minimize(f, 10, args=(), method=None)

print(w)

#Plotting the function
x = np.linspace(XL,XR,100)
fig = plt.figure()
plt.plot(x,h1(x), color='red')
plt.plot(x,s1*x+w.fun)
plt.show()
