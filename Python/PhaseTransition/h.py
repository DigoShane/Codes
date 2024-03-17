####################################################################################################
#The objective here is to get an upper bound on the energy at the upper critical field (Hc2). 
#We will use test functions that approximate the Minimizing solution. This will have energy lower than the normal phase, ie, u=0 \curl A=H.
#We will work with the non dimensionalized for of the free energy. The details of the calculation can be foun in the One Note file "Effect of Stress on Transition Temp/SRO magnetism/Magnetic field Calculation".
#We pick a triangular lattice and our unit cell is two adjacent triangular unit cells (makes a parallelogram).
#This is because for performing Fourier Transform, we cannot use Triangular unit cell.
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
from scipy.fft import fft2, rfft2, irfft2


####################################################################################################
#Defining Variables
####################################################################################################
N1 = 3 #int(input("Enter # terms to Truncate to: "))
N2 = 3 #int(input("Enter # terms to Truncate to: "))
K = 2 #float(input("Enter the Ginzburg Landau Parameter: "))
DeltaA = 0.0 #float(input("deviation for Abrikosov sollution: "))
l2 = 0.1 #float(input("enter length of l2: "))
a = (2/K)*np.sqrt(np.pi/np.sqrt(3))+DeltaA
l1 = (a-l2)/2
delta = 1
H = K-0.001
uinf = 1
c1 = np.pi
c2 = np.pi
Ngrid = 20
correction = 10**(-6)
B = np.array(([a,a/2],[0, np.sqrt(3)*a/2]))


####################################################################################################
#Defining test function for h(x)
####################################################################################################
def htest(u1p,u2p):
 z = [[0 for i in range(len(u1p))] for j in range(len(u2p))]
 for i in range(len(u1p)):
  for j in range(len(u2p)):
   u1 = u1p[i]
   u2 = u2p[j]
   x1 = np.matmul([1,0],np.matmul(B,[[u1],[u2]]))[0] #(a*(u1+c1)+a*(u2+c2)/2)/(2*np.pi)
   x2 = np.matmul([0,1],np.matmul(B,[[u1],[u2]]))[0] #(np.sqrt(3)*a*(u2+c2)/2)/(2*np.pi)
   if (x2<=0):
    x2=x2+correction 
   if (x2>=np.sqrt(3)*x1):
    x1=x1+correction 
    x2=x2-correction 
   if (x2>=np.sqrt(3)*a/2):
    x2=x2-correction
   if (x2<=np.sqrt(3)*(x1-2*l1-l2)):
    x1=x1-correction
    x2=x2+correction

   if  (0<=x2) & (x2<=np.sqrt(3)*x1) & (0<=x2) & (x2<=np.sqrt(3)*(l1-x1)):
    z[i][j] = H+(-delta/np.sqrt(3)/l1)*(np.sqrt(3)*x1+x2)
    #print("One")
    #print([i,j,z[i][j]])
   elif (np.sqrt(3)*(l1+l2)/2<=x2) & (x2<=np.sqrt(3)*x1) & (np.sqrt(3)*(l1+l2)/2<=x2) & (x2<=np.sqrt(3)*(2*l1+l2-x1)):
    z[i][j] = H+(-2*delta/np.sqrt(3)/l1)*(np.sqrt(3)*(2*l1+l2)/2-x2)
    #print("Two")
    #print([i,j,z[i][j]])
   elif (np.sqrt(3)*(2*l1+l2-x1)<=x2) & (x2<=np.sqrt(3)*(2*l1+l2)/2) & (np.sqrt(3)*(x1-l1)<=x2) & (x2<=np.sqrt(3)*(2*l1+l2)/2):
    z[i][j] = H+(-delta/np.sqrt(3)/l1)*(np.sqrt(3)*x1-x2)
    #print("Three")
    #print([i,j,z])
   elif (np.sqrt(3)*(3*l1+2*l2-x1)<=x2) & (x2<=np.sqrt(3)*(2*l1+l2)/2) & (np.sqrt(3)*(x1-2*l1-l2)<=x2) & (x2<=np.sqrt(3)*(2*l1+l2)/2):
    z[i][j] = H+(delta/np.sqrt(3)/l1)*(x2+np.sqrt(3)*(x1-4*l1-2*l2))
    #print("Four")
    #print([i,j,z])
   elif (np.sqrt(3)*(2*l1+l2-x1)<=x2) & (x2<=np.sqrt(3)*l1/2) & (np.sqrt(3)*(x1-2*l1-l2)<=x2) & (x2<=np.sqrt(3)*l1/2):
    z[i][j] = H+(-2*delta/np.sqrt(3)/l1)*x2
    #print("Five")
    #print([i,j,z])
   elif (0<=x2) & (x2<=np.sqrt(3)*(x1-l1-l2)) & (0<=x2) & (x2<=np.sqrt(3)*(2*l1+l2-x1)):
    z[i][j] = H+(-delta/np.sqrt(3)/l1)*(x2+np.sqrt(3)*(2*l1+l2-x1))
    #print("Six")
    #print([i,j,z])
   else:
    z[i][j] = H-delta
    #print("Seven")
    #print([i,j,z[i][j]])
    #print([x1,x2,u1,u2])
 return z
 del z



####################################################################################################
#Defining test function for u(x)
####################################################################################################
def utest(u1p,u2p):
 z = [[0 for i in range(len(u1p))] for j in range(len(u2p))]
 for i in range(len(u1p)):
  for j in range(len(u2p)):
   u1 = u1p[i]
   u2 = u2p[j]
   x1 = np.matmul([1,0],np.matmul(B,[[u1],[u2]]))[0] #(a*(u1+c1)+a*(u2+c2)/2)/(2*np.pi)
   x2 = np.matmul([0,1],np.matmul(B,[[u1],[u2]]))[0] #(np.sqrt(3)*a*(u2+c2)/2)/(2*np.pi)
   if (x2<=0):
    x2=x2+correction 
   if (x2>=np.sqrt(3)*x1):
    x1=x1+correction 
    x2=x2-correction 
   if (x2>=np.sqrt(3)*a/2):
    x2=x2-correction
   if (x2<=np.sqrt(3)*(x1-2*l1-l2)):
    x1=x1-correction
    x2=x2+correction

   if  (0<=x2) & (x2<=np.sqrt(3)*x1) & (0<=x2) & (x2<=np.sqrt(3)*(l1-x1)):
    #print("One")
    z[i][j] = (uinf/np.sqrt(3)/l1)*(np.sqrt(3)*x1+x2)
    #print([i,j,z[i][j]])
    #print(z)
   elif (np.sqrt(3)*(l1+l2)/2<=x2) & (x2<=np.sqrt(3)*x1) & (np.sqrt(3)*(l1+l2)/2<=x2) & (x2<=np.sqrt(3)*(2*l1+l2-x1)):
    #print("two")
    z[i][j] = (2*uinf/np.sqrt(3)/l1)*(np.sqrt(3)*(2*l1+l2)/2-x2)
    #print([i,j,z[i][j]])
    #print(z)
   elif (np.sqrt(3)*(2*l1+l2-x1)<=x2) & (x2<=np.sqrt(3)*(2*l1+l2)/2) & (np.sqrt(3)*(x1-l1)<=x2) & (x2<=np.sqrt(3)*(2*l1+l2)/2):
    #print("Three")
    z[i][j] = (-uinf/np.sqrt(3)/l1)*(x2-np.sqrt(3)*x1)
    #print([i,j,z[i][j]])
    #print(z)
   elif (np.sqrt(3)*(3*l1+2*l2-x1)<=x2) & (x2<=np.sqrt(3)*(2*l1+l2)/2) & (np.sqrt(3)*(x1-2*l1-l2)<=x2) & (x2<=np.sqrt(3)*(2*l1+l2)/2):
    #print("Four")
    z[i][j] = (-uinf/np.sqrt(3)/l1)*(x2+np.sqrt(3)*(x1-4*l1-2*l2))
    #print([i,j,z[i][j]])
    #print(z)
   elif (np.sqrt(3)*(2*l1+l2-x1)<=x2) & (x2<=np.sqrt(3)*l1/2) & (np.sqrt(3)*(x1-2*l1-l2)<=x2) & (x2<=np.sqrt(3)*l1/2):
    #print("Five")
    z[i][j] = (2*uinf/np.sqrt(3)/l1)*x2
    #print([i,j,z[i][j]])
    #print(z)
   elif (0<=x2) & (x2<=np.sqrt(3)*(x1-l1-l2)) & (0<=x2) & (x2<=np.sqrt(3)*(2*l1+l2-x1)):
    #print("Six")
    z[i][j] = (uinf/np.sqrt(3)/l1)*(x2+np.sqrt(3)*(2*l1+l2-x1))
    #print([i,j,z[i][j]])
    #print(z)
   else:
    #print("Seven")
    z[i][j] = uinf
    #print([i,j,z[i][j]])
    #print(z)
 return z
 del z



#####################################################################################################
#Printing
#####################################################################################################
#For testing out the utest and htest functions in the transformed space x=B(u+c)
uxprime = np.linspace(0,1, Ngrid)
uyprime = np.linspace(0,1, Ngrid)
Zh = [[0]*len(uxprime)]*len(uyprime)
Zu = [[0]*len(uxprime)]*len(uyprime)
X, Y = np.meshgrid(uxprime, uyprime)
Zh = np.transpose(htest(uxprime, uyprime))
Zu = np.transpose(utest(uxprime, uyprime))



# Plot the surface.
#------------------------------------------------------------------------------------------------
figh, axh = plt.subplots(subplot_kw={"projection": "3d"})
surfh = axh.plot_surface(X, Y, Zh, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.title("h(x) vs x")
# Customize the z axis.
axh.set_zlim(1.0, 3.0)
axh.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
axh.zaxis.set_major_formatter('{x:.02f}')
# Add a color bar which maps values to colors.
figh.colorbar(surfh, shrink=0.75, aspect=7)
#------------------------------------------------------------------------------------------------
figu, axu = plt.subplots(subplot_kw={"projection": "3d"})
surfu = axu.plot_surface(X, Y, Zu, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.title("u(x) vs x")
# Customize the z axis.
axu.set_zlim(0.0, 2.0)
axu.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
axu.zaxis.set_major_formatter('{x:.02f}')
# Add a color bar which maps values to colors.
figu.colorbar(surfu, shrink=0.75, aspect=7)
#------------------------------------------------------------------------------------------------

plt.show()


