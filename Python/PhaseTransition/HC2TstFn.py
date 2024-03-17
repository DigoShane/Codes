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
from scipy import integrate


####################################################################################################
#Defining Variables
####################################################################################################
N1 = 5 #int(input("Enter # terms to Truncate to: "))
N2 = 5 #int(input("Enter # terms to Truncate to: "))
if N1%2 == 0 & N2%2 ==0:
 print("Code written for Ngrid{1,2} odd")
K = 2 #float(input("Enter the Ginzburg Landau Parameter: "))
DeltaA = 0.0 #float(input("deviation for Abrikosov sollution: "))
l2 = 0.1 #float(input("enter length of l2: "))
a = (2/K)*np.sqrt(np.pi/np.sqrt(3))+DeltaA
l1 = (a-l2)/2
delta = 0.5
H = K-0.001
uinf = 1
correction = 10**(-6)

print(a)
print(l1)
print(l2)


#####################################################################################################
#Defining the Transform
#####################################################################################################
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

   ### #------------------------------------------------------------------------------------------------------------------
   ### #Original Test function
   ### #------------------------------------------------------------------------------------------------------------------
   ### if  (0<=x2) & (x2<=np.sqrt(3)*x1) & (0<=x2) & (x2<=np.sqrt(3)*(l1-x1)):
   ###  #print("One")
   ###  z[j][i] = H+(-delta/np.sqrt(3)/l1)*(np.sqrt(3)*x1+x2)
   ### elif (np.sqrt(3)*(l1+l2)/2<=x2) & (x2<=np.sqrt(3)*x1) & (np.sqrt(3)*(l1+l2)/2<=x2) & (x2<=np.sqrt(3)*(2*l1+l2-x1)):
   ###  #print("Two")
   ###  z[j][i] = H+(-2*delta/np.sqrt(3)/l1)*(np.sqrt(3)*(2*l1+l2)/2-x2)
   ### elif (np.sqrt(3)*(2*l1+l2-x1)<=x2) & (x2<=np.sqrt(3)*(2*l1+l2)/2) & (np.sqrt(3)*(x1-l1)<=x2) & (x2<=np.sqrt(3)*(2*l1+l2)/2):
   ###  #print("Three")
   ###  z[j][i] = H+(-delta/np.sqrt(3)/l1)*(np.sqrt(3)*x1-x2)
   ### elif (np.sqrt(3)*(3*l1+2*l2-x1)<=x2) & (x2<=np.sqrt(3)*(2*l1+l2)/2) & (np.sqrt(3)*(x1-2*l1-l2)<=x2) & (x2<=np.sqrt(3)*(2*l1+l2)/2):
   ###  #print("Four")
   ###  z[j][i] = H+(delta/np.sqrt(3)/l1)*(x2+np.sqrt(3)*(x1-4*l1-2*l2))
   ### elif (np.sqrt(3)*(2*l1+l2-x1)<=x2) & (x2<=np.sqrt(3)*l1/2) & (np.sqrt(3)*(x1-2*l1-l2)<=x2) & (x2<=np.sqrt(3)*l1/2):
   ###  #print("Five")
   ###  z[j][i] = H+(-2*delta/np.sqrt(3)/l1)*x2
   ### elif (0<=x2) & (x2<=np.sqrt(3)*(x1-l1-l2)) & (0<=x2) & (x2<=np.sqrt(3)*(2*l1+l2-x1)):
   ###  #print("Six")
   ###  z[j][i] = H+(-delta/np.sqrt(3)/l1)*(x2+np.sqrt(3)*(2*l1+l2-x1))
   ### else:
   ###  #print("Seven")
   ###  zz[j][i] = H-delta

    
   #------------------------------------------------------------------------------------------------------------------
   #Benchmark Test function -1
   #------------------------------------------------------------------------------------------------------------------
   z[j][i] = np.sin(2*np.pi*u1)*np.sin(2*np.pi*u2)
   ##------------------------------------------------------------------------------------------------------------------
   ##Benchmark Test function -2
   ##------------------------------------------------------------------------------------------------------------------
   #z[j][i] = 1+ np.sin(2*np.pi*u1)*np.sin(2*np.pi*u2)

 return z



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

   ### #------------------------------------------------------------------------------------------------------------------
   ### #Original Test function
   ### #------------------------------------------------------------------------------------------------------------------
   ### if  (0<=x2) & (x2<=np.sqrt(3)*x1) & (0<=x2) & (x2<=np.sqrt(3)*(l1-x1)):
   ###  #print("One")
   ###  z[i][j] = (uinf/np.sqrt(3)/l1)*(np.sqrt(3)*x1+x2)
   ### elif (np.sqrt(3)*(l1+l2)/2<=x2) & (x2<=np.sqrt(3)*x1) & (np.sqrt(3)*(l1+l2)/2<=x2) & (x2<=np.sqrt(3)*(2*l1+l2-x1)):
   ###  #print("two")
   ###  z[i][j] = (2*uinf/np.sqrt(3)/l1)*(np.sqrt(3)*(2*l1+l2)/2-x2)
   ### elif (np.sqrt(3)*(2*l1+l2-x1)<=x2) & (x2<=np.sqrt(3)*(2*l1+l2)/2) & (np.sqrt(3)*(x1-l1)<=x2) & (x2<=np.sqrt(3)*(2*l1+l2)/2):
   ###  #print("Three")
   ###  z[i][j] = (-uinf/np.sqrt(3)/l1)*(x2-np.sqrt(3)*x1)
   ### elif (np.sqrt(3)*(3*l1+2*l2-x1)<=x2) & (x2<=np.sqrt(3)*(2*l1+l2)/2) & (np.sqrt(3)*(x1-2*l1-l2)<=x2) & (x2<=np.sqrt(3)*(2*l1+l2)/2):
   ###  #print("Four")
   ###  z[i][j] = (-uinf/np.sqrt(3)/l1)*(x2+np.sqrt(3)*(x1-4*l1-2*l2))
   ### elif (np.sqrt(3)*(2*l1+l2-x1)<=x2) & (x2<=np.sqrt(3)*l1/2) & (np.sqrt(3)*(x1-2*l1-l2)<=x2) & (x2<=np.sqrt(3)*l1/2):
   ###  #print("Five")
   ###  z[i][j] = (2*uinf/np.sqrt(3)/l1)*x2
   ### elif (0<=x2) & (x2<=np.sqrt(3)*(x1-l1-l2)) & (0<=x2) & (x2<=np.sqrt(3)*(2*l1+l2-x1)):
   ###  #print("Six")
   ###  z[i][j] = (uinf/np.sqrt(3)/l1)*(x2+np.sqrt(3)*(2*l1+l2-x1))
   ### else:
   ###  #print("Seven")
   ###  z[i][j] = uinf


   ### #------------------------------------------------------------------------------------------------------------------
   ### #Benchmark Test function -1
   ### #------------------------------------------------------------------------------------------------------------------
   ### z[j][i] = np.cos(2*np.pi*u1)*np.cos(2*np.pi*u2)
   #------------------------------------------------------------------------------------------------------------------
   #Benchmark Test function -2
   #------------------------------------------------------------------------------------------------------------------
   z[j][i] = 1




 return z



# perp
def perp(k):
 return np.asarray([-k[1],k[0]])

####################################################################################################
#Fourier Approximation of A(x)
####################################################################################################
def Atest(u1,u2,fz):
 #Settingup parameters for defining A
 R11 = 1
 R12 = 0
 TildeA1 = 0
 TildeA2 = 0
 R = [[R11,R12],[fz[0][0]+R12,-R11]]
 fzr = fz[:,np.concatenate((range(int(N1/2)+1,N1),range(0,int(N1/2)+1)))]
 fzr = fzr[np.concatenate((range(int(N2/2)+1,N2),range(0,int(N2/2)+1))),:]
 ll1 = np.linspace(-int(N1/2),int(N1/2),N1) 
 ll2 = np.linspace(-int(N2/2),int(N2/2),N2) 
 TT1 = [[0.0+0.0*1j for i in range(N1)] for j in range(N2)]
 TT2 = [[0.0+0.0*1j for i in range(N1)] for j in range(N2)]
 for n1 in ll1:
  for n2 in ll2:
   k1 = 2*np.pi*n1
   k2 = 2*np.pi*n2
   temp = np.matmul(np.transpose(np.linalg.inv(B)),[k1,k2])
   if np.inner(temp,temp) == 0:
    temp1 = [TT1[0][0]-TT1[0][0],TT2[0][0]-TT2[0][0]]
   else: 
    temp1 = -1j*perp(temp)/np.inner(temp,temp)
   TT1[int(n2+int(N2/2))][int(n1+int(N1/2))] = temp1[0]
   TT2[int(n2+int(N2/2))][int(n1+int(N1/2))] = temp1[1]
 AZ1 = [[0 for i in range(len(u1))] for j in range(len(u2))]
 AZ2 = [[0 for i in range(len(u1))] for j in range(len(u2))]
 RB = np.matmul(R,B)
 for ii in range(len(u1)):
  for jj in range(len(u2)):
   v1 = np.asarray(list(map(lambda nn: np.exp(1j*2*np.pi*nn*u1[ii]),ll1)))
   v2 = np.asarray(list(map(lambda nn: np.exp(1j*2*np.pi*nn*u2[jj]),ll2)))
   v  = np.outer(v2,v1)
   azo = TildeA1 + np.matmul([1,0],np.matmul(RB,[u1[ii],u2[jj]]))
   AZ1[jj][ii] = np.sum(np.multiply( fzr, np.multiply(TT1,v)))
   AZ1[jj][ii] = AZ1[jj][ii] + azo
   azo = TildeA2 + np.matmul([0,1],np.matmul(RB,[u1[ii],u2[jj]]))
   AZ2[jj][ii] = np.sum(np.multiply( fzr, np.multiply(TT2,v)))
   AZ2[jj][ii] = AZ2[jj][ii] + azo
 return np.real(AZ1),np.real(AZ2)


#####################################################################################################
# Double integration
#####################################################################################################
def double_Integral(xmin, xmax, ymin, ymax, nx, ny, A):

    dS = ((xmax-xmin)/(nx-1)) * ((ymax-ymin)/(ny-1))

    A_Internal = A[1:-1, 1:-1]

    # sides: up, down, left, right
    (A_u, A_d, A_l, A_r) = (A[0, 1:-1], A[-1, 1:-1], A[1:-1, 0], A[1:-1, -1])

    # corners
    (A_ul, A_ur, A_dl, A_dr) = (A[0, 0], A[0, -1], A[-1, 0], A[-1, -1])

    return dS * (np.sum(A_Internal)\
                + 0.5 * (np.sum(A_u) + np.sum(A_d) + np.sum(A_l) + np.sum(A_r))\
                + 0.25 * (A_ul + A_ur + A_dl + A_dr))


#####################################################################################################
#Computing the Energy per unit cell
#####################################################################################################
def Energy(u1,u2,Zu,Z,AZ1,AZ2):
 #Total energy due to \int (u^2-1)^2/2
 E1 = np.sqrt(3)/4*l1**2
 #Total energy due to \int (\nabla u/k)^2
 E2 = 2*np.sqrt(3)/K**2
 detB = np.linalg.det(B)
 E3 = detB*double_Integral(0, 1, 0, 1, N1, N2, np.multiply( np.multiply(Zu,Zu), np.multiply(AZ1,AZ1) + np.multiply(AZ2,AZ2) ) )
 Hh = H*np.ones((len(u1),len(u2)))
 E4 = detB*double_Integral(0, 1, 0, 1, N1, N2, np.multiply(Hh-Z, Hh-Z))
 return E3 #return this to test the construction of A(x)
 #return E1+E2+E3+E4-detB/2


#####################################################################################################
#Printing
#####################################################################################################
#For testing out the utest and htest functions in the transformed space x=B(u+c)
u1prime = np.linspace(0,1, N1)
u2prime = np.linspace(0,1, N2)
Z = [[0]*len(u1prime)]*len(u2prime)
Zu = [[0]*len(u1prime)]*len(u2prime)
X, Y = np.meshgrid(u1prime, u2prime)
Z = htest(u1prime, u2prime)
Zu = utest(u1prime, u2prime)
#------------------------------------------------------
# FFT of h and calling A(x)
#------------------------------------------------------
FZ = fft2(Z)
FZbN = FZ/N1/N2
print("--------------------------------------------------------")
print("FZbN = FFT(h)/N1/N2")
print('\n'.join([''.join(['{:25}'.format(item) for item in row]) 
      for row in np.trunc(FZbN)]))
ZZ= irfft2(FZ, [len(X),len(Y)])
AZ1 = [[0]*len(u1prime)]*len(u2prime)
AZ2 = [[0]*len(u1prime)]*len(u2prime)
AZ1,AZ2 = Atest(u1prime,u2prime,FZbN)
print("E3=...") 
print(Energy(u1prime, u2prime, Zu, Z , AZ1, AZ2)) 

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
##------------------------------------------------------------------------------------------------
#surf = ax.plot_surface(X, Y, np.asarray(Zu), cmap=cm.coolwarm, linewidth=0, antialiased=False)
#plt.title("h(x) vs x")
##------------------------------------------------------------------------------------------------
#surf = ax.plot_surface(X, Y, np.absolute(FZ), cmap=cm.coolwarm, linewidth=0, antialiased=False)
#plt.title("Inverse fourier transform of original plot")
##------------------------------------------------------------------------------------------------
#surf = ax.plot_surface(X, Y, ZZ, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#plt.title("fourier Transform of inverse fourier transform of original plot")
#------------------------------------------------------------------------------------------------
surf = ax.plot_surface(X, Y, AZ1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.title("AZ1 vs x")
##------------------------------------------------------------------------------------------------
#surf = ax.plot_surface(X, Y, AZ2, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#plt.title("AZ2 vs x")

# Customize the z axis.
ax.set_zlim(-1.0, 2.0)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.75, aspect=7)
plt.show()


