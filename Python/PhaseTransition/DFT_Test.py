####################################################################################################
# The objective here is to test out how the DFT works,
# We want to compute 
#A(u)=\sum_-N^N -i*(2*pi*n) \tilde{h}(n) exp(2*pi*i*n*u)
#     |->2*(x+pi) -pi<=x<0
#h(u)=|
#     |->2*(x-pi) 0<=x<pi
#where
#\tilde{h}(n)=\int_0^1 h(u) exp(-2*pi*i*n*u)du
#
#Another choice of h(u)=sin(2\pi x/2)
#There is an accompanying Mathematica file where i have done some computations by hand, check 
#"Books/Codes/Python/Examples/DFT_vs_FourierSeries.nb"
#####################################################################################################

import numpy as np
from scipy import special as sp
from scipy import integrate
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import math
import sys #for error messages
from scipy.fft import fft, rfft, irfft
from scipy.fft import fft2, rfft2, irfft2


####################################################################################################
#Defining Variables
####################################################################################################
Ngrid = 21 #Always input odd no. here.
if Ngrid%2 == 0:
 print("Code written for Ngrid odd")
T = 2#2*np.pi
deltax = T/Ngrid 



# Original Fn
def h(x):
 z = [0 for i in range(len(x))]
###--Case 1-- Triangular wave
###  for i in range(len(x)):
###   x1 = x[i]
###   if -np.pi<=x1<0:
###    z[i] = (1+x1/np.pi)
###   if 0<=x1<np.pi:
###    z[i] = (1-x1/np.pi)

###--Case 2-- Sine wave
 z = np.sin(2*np.pi*x/T)
 return z


# Modified function A(x)
def A(x,fz):
 fzr = np.roll(fz,int(Ngrid/2))#to match it with the coeff numbering in Fourier Series
 n = np.linspace(-int(Ngrid/2),int(Ngrid/2),Ngrid) 
 AZ = [0 for i in range(len(x))]
 for ii in range(len(x)):
  u = np.asarray(list(map(lambda nn: nn*(-1j)*2*np.pi/T*np.exp(1j*2*np.pi/T*nn*x[ii]),n)))
  AZ[ii] = np.inner( fzr, u)
 return np.real(AZ)

xprime = np.linspace(0, T-deltax, Ngrid)
Z = [0]*len(xprime)
AZ = [0]*len(xprime)
Z = np.transpose(h(xprime))
FZ= fft(Z)
AZ = np.transpose(A(xprime,FZ/Ngrid))
ZZ= irfft(FZ, len(xprime))



#np.set_printoptions(formatter={'float_kind':'{:f}'.format})
#print(Z)
#np.set_printoptions(formatter={'complex_kind': '{:.8f}'.format})
#print(np.around(FZ,decimals=10))
#print(2*np.pi*xprime/T)
#print(xprime)


# Plot 
fig, ax = plt.subplots()
line1 = ax.plot(xprime, AZ, linewidth=1, label='Numerical')
line2 = ax.plot(xprime, np.asarray(list(map(lambda p: -np.pi*np.cos(np.pi*p)+0.1,xprime))), linewidth=1, label='Analytical + 0.1')
plt.title("A(x) vs x")
plt.legend()
#ax.plot(FZ, linewidth=2)
#plt.title("\tilde{h}(k)")

plt.show()


