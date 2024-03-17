####################################################################################################
# The objective here is to test out how the 2DDFT works,
# The idea is similar to that used in 
# We want to compute 
#A(u)=\sum_-N^N -i*k^{perp}/k^2 \tilde{h}(n) exp(2*pi*i*n*u)
#h(u)=sin(u1)cos(u2)
#where
#\tilde{h}(n)=Fourier coeff
#There is an acompanying MAthematica file where i have done some computations by hand, check 
#"Books/Codes/Python/Examples/2DDFT_vs_FourierSeries.nb"
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
Ngrid1 = 21 #Always input odd no. here.
Ngrid2 = 21 #Always input odd no. here.
if Ngrid1%2 == 0 & Ngrid2%2 ==0:
 print("Code written for Ngrid{1,2} odd")
a = 2
T1 = a
deltax1 = T1/Ngrid1
T2 = a
deltax2 = T2/Ngrid2
B = [[1,0],[0,1]]


# Fn defn in x1 x2 space
def h(x1,x2):
 z = [[0 for i in range(len(x1))] for j in range(len(x2))]
 for ii in range(len(x1)):
  for jj in range(len(x2)):
   z[jj][ii] = np.sin(2*np.pi*x1[ii])*np.cos(2*np.pi*x2[jj])
 #print("--------------------------------------------------------")
 #print("z=h(x1,x2)")
 #print('\n'.join([''.join(['{:55}'.format(item) for item in row]) 
 #      for row in z]))
 return z




# perp
def perp(k):
 return np.asarray([-k[1],k[0]])


# Modified function A(x)
def A(x1,x2,fz):
 fzr = fz[:,np.concatenate((range(int(Ngrid1/2)+1,Ngrid1),range(0,int(Ngrid1/2)+1)))]
 fzr = fzr[np.concatenate((range(int(Ngrid2/2)+1,Ngrid2),range(0,int(Ngrid2/2)+1))),:]
 ### print("--------------------------------------------------------")
 ### print("fzr=FFT(h)~ reaaranged according to FS")
 ### print('\n'.join([''.join(['{:55}'.format(item) for item in row]) 
 ###       for row in fzr]))
 l1 = np.linspace(-int(Ngrid1/2),int(Ngrid1/2),Ngrid1) 
 l2 = np.linspace(-int(Ngrid2/2),int(Ngrid2/2),Ngrid2) 
 TT1 = [[0.0+0.0*1j for i in range(Ngrid1)] for j in range(Ngrid2)]
 TT2 = [[0.0+0.0*1j for i in range(Ngrid1)] for j in range(Ngrid2)]
 for n1 in l1:
  for n2 in l2:
   k1 = 2*np.pi*n1/T1
   k2 = 2*np.pi*n2/T2
   temp = np.matmul(np.transpose(np.linalg.inv(B)),[k1,k2])
   if np.inner(temp,temp) == 0:
    temp1 = [TT1[0][0]-TT1[0][0],TT2[0][0]-TT2[0][0]]
   else: 
    temp1 = -1j*perp(temp)/np.inner(temp,temp)
   TT1[int(n2+int(Ngrid2/2))][int(n1+int(Ngrid1/2))] = temp1[0]
   TT2[int(n2+int(Ngrid2/2))][int(n1+int(Ngrid1/2))] = temp1[1]
 ### print("--------------------------------------------------------")
 ### print("TT1 = Coeff of A1(x) not including h(k)")
 ### print('\n'.join([''.join(['{:25}'.format(item) for item in row]) 
 ###       for row in np.imag(TT1)]))
 ### print("--------------------------------------------------------")
 ### print("TT2 = Coeff of A2(x) not including h(k)")
 ### print('\n'.join([''.join(['{:25}'.format(item) for item in row]) 
 ###       for row in np.imag(TT2)]))
 AZ1 = [[0 for i in range(len(x1))] for j in range(len(x2))]
 AZ2 = [[0 for i in range(len(x1))] for j in range(len(x2))]
 for ii in range(len(x1)):
  for jj in range(len(x2)):
   u1 = np.asarray(list(map(lambda nn: np.exp(1j*2*np.pi*nn*x1[ii]/T1),l1)))
   u2 = np.asarray(list(map(lambda nn: np.exp(1j*2*np.pi*nn*x2[jj]/T2),l2)))
   u  = np.outer(u2,u1)
   AZ1[jj][ii] = np.sum(np.multiply( fzr, np.multiply(TT1,u)))
   AZ2[jj][ii] = np.sum(np.multiply( fzr, np.multiply(TT2,u)))
 v1 = np.asarray(list(map(lambda nn: np.exp(1j*2*np.pi*nn/T1*0.5),l1)))
 v2 = np.asarray(list(map(lambda nn: np.exp(1j*2*np.pi*nn/T2*0.25),l2)))
 v  = np.outer(v1,v2)
 ### print("--------------------------------------------------------")
 ### print("v = Testing the outer product of complex exponentials")
 ### print('\n'.join([''.join(['{:45}'.format(item) for item in row]) 
 ###       for row in v]))
 return np.real(AZ1),np.real(AZ2)

x1prime = np.linspace(0, T1-deltax1, Ngrid1)
x2prime = np.linspace(0, T2-deltax2, Ngrid2)
X, Y = np.meshgrid(x1prime, x2prime)
Z = [[0 for i in range(len(x1prime))] for j in range(len(x2prime))] 
Z = h(x1prime, x2prime)
FZ = fft2(Z)
FZbN= FZ/Ngrid1/Ngrid2
#print('\n'.join([''.join(['{:21}'.format(item) for item in row]) 
#      for row in h(x1prime, x2prime)]))
#print("--------------------------------------------------------")
#print("FZbN = FFT(h)/N1/N2")
#print('\n'.join([''.join(['{:25}'.format(item) for item in row]) 
#      for row in FZbN]))
#ZZ= irfft2(FZ, [len(x1prime),len(x2prime)])
AZ1 = [[0]*len(x1prime)]*len(x2prime)
AZ2 = [[0]*len(x1prime)]*len(x2prime)
AZ1,AZ2 = A(x1prime,x2prime,FZbN)

### print("--------------------------------------------------------")
### print("AZ1 = A1(x1,x2)")
### print('\n'.join([''.join(['{:25}'.format(item) for item in row]) 
###       for row in AZ1]))


#np.set_printoptions(formatter={'float_kind':'{:f}'.format})
#print(Z)
#np.set_printoptions(formatter={'complex_kind': '{:.8f}'.format})
#print(np.around(FZ,decimals=10))
#print(2*np.pi*xprime/T)
#print(xprime)


# Plot 
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
#------------------------------------------------------------------------------------------------
#surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#plt.title("h(x) or u(x)  vs x")
#ax.set_zlim(-1.0, 1.0)
#surf = ax.scatter(X, Y, FZbN, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#plt.title("h(k) vs k")
#ax.set_zlim(-1.0, 1.0)
#surf = ax.plot_surface(X, Y, AZ1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
surf = ax.plot_surface(X, Y, AZ2, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.title("A(x)  vs x")
ax.set_zlim(-0.16, 0.16)

# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.75, aspect=7)



plt.show()


