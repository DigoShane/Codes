#The objective here is to test how to perform Matrix matrix and Matrix vector multiplication, with ease in python3.
#We will make use of numpy's tensordot.

import numpy as np
import scipy as scp
from numpy import linalg

#initializing the Material Tensors of phase A
A = np.array([[1,2,4],[2,1,3],[-1,-3,4]])
B = np.array([[2,3,1],[4,1,-2],[-2,-3,1]])
c = np.array([1,5,6])
D = np.zeros((3,3,3,3), dtype=float)
E = np.zeros((3,3), dtype=float)
F = np.zeros((3,3,3), dtype=float)

for i in range(3):
 for j in range(3):
  for k in range(3):
   for l in range(3):
    D[i,j,k,l] = 10**3*i+10**2*j+10*k+l
   F[i,j,k] = 10**2*i+10*j+k

print(A)
print(B)
print()
print('---------------')
print()
print(D)
print()
print('-------------')
print()
print('Testing Matrix MAtri Multiplication, A*B')
print(np.tensordot(A,B,1))
print('Testing Matrix vector Multiplication, A*c')
print(np.tensordot(A,c,1))
print('Testing 4th Order tensor Matrix Multiplication, D:A')
print(np.tensordot(D,A,2))
print('Testing 3th Order tensor vector Multiplication, c.F')
print(np.tensordot(c,F,1))

for i in range(3):
 for j in range(3):
  for k in range(3):
   for l in range(3):
    E[i,j] = E[i,j] + D[i,j,k,l]*A[k,l]

print()
print('Testing E_{ij}=D_{ijkl}A_{kl}')
print(E)

E = np.zeros((3,3), dtype=float)
for i in range(3):
 for j in range(3):
  for k in range(3):
   E[i,j] = E[i,j] + c[k]*F[k,i,j]

print()
print('Testing E_{jk}=c_{i}F_{ijk}')
print(E)
