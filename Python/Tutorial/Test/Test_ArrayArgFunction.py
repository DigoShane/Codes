#The objective here is to test if 
# 1. We can pass arrays as arguments to Functions
# 2. Assign the proper matrices to frame the linear equations

import numpy as np
import scipy as scp
import math

#Listing the material parameters of phase A in SI units
CA11 = 11.0 
CA12 = 12.0 
CA13 = 13.0 
CA22 = 22.0 
CA23 = 23.0 
CA33 = 33.0 
CA44 = 44.0 
CA55 = 55.0 
CA66 = 66.0 
SA123 = 123 
SA231 = 231 
SA312 = 312 
SA132 = 132 
SA213 = 213 
SA321 = 321 
XA11 =  11 
XA22 =  22 
XA33 =  33 


#initializing the Material Tensors of phase A
CA = np.zeros((3,3,3,3), dtype=float)
SA = np.zeros((3,3,3), dtype=float)
XA = np.zeros((3,3), dtype=float) 
F0 = np.zeros((3,2),dtype=float) 
UA = np.array([[1,0,0],[0,2,0],[0,0,2]]) 
pA = np.array([1,0,0]) 
Q = np.zeros((3,3),dtype=float) 
CAQ = np.zeros((3,3,3,3), dtype=float)
SAQ = np.zeros((3,3,3), dtype=float)
XAQ = np.zeros((3,3), dtype=float) 
T = np.zeros((6,6), dtype=float)
bp = np.zeros((6,1), dtype=float)
RHS = np.zeros((6,1), dtype=float)


#Inputting the variables
F0[0,0] = 1
F0[1,1] = 1
Q[0,0] = 1
Q[1,1] = 1
Q[2,2] = 1

#imposing the symmetries
for i in range(3):
 for j in range(3):
  for k in range(3):
   for l in range(3):
    CA[i,j,k,l] = 10**3*i + 10**2*j + 10*k + l
   SA[i,j,k] = 10**2*i + 10*j + k
  XA[i,j] = 10*i + j

def LinSolve(T,RHS,CQ,SQ,XQ,QU,Qp):
 for i in range(3):
  for j in range(3):
   T[i,j] = CQ[i,2,j,2]
   T[i,j+3] = SQ[j,i,2]
   T[i+3,j] = SQ[i,j,2]
   T[i+3,j+3] = XQ[i,j]
  RHS[i]=0
  RHS[i+3]=0
  for k in range(3):
   for l in range(3):
    RHS[i] = RHS[i] +  CQ[i,2,k,l]*QU[k,l]
    RHS[i+3] = RHS[i+3] + SQ[i,k,l]*QU[k,l]
  #print('RHS CQ:QU[',i,']',RHS[i])
  #print('RHS SQ:QU[',i+3,']',RHS[i+3])
  for m in range(3):
   RHS[i] = RHS[i] + Qp[m]*SQ[m,i,2]
   RHS[i+3] = RHS[i+3] + Qp[m]*XQ[i,m]
  #print('RHS Qp.SQ[',i,']',RHS[i])
  #print('RHS Qp.XQ[',i+3,']',RHS[i+3])
  for k in range(3):
   for beta in range(2):
    RHS[i] = RHS[i] - CQ[i,2,k,beta]*F0[k,beta]
    RHS[i+3] = RHS[i+3] - SQ[i,k,beta]*F0[k,beta]
  #print('RHS Qp.SQ[',i,']',RHS[i])
  #print('RHS SQ.F0[',i+3,']',RHS[i+3])
 #Temp = np.zeros((3,1), dtype =float)
 #Temp = np.tensordot(CQ[:,2,:,:],QU[:,:],2)
 #print('CQ_{i3kl}(QU)_{kl}=' , Temp)
 #Temp = np.zeros((3,1), dtype =float)
 #Temp = np.tensordot(Qp[:],SQ[:,:,2],1)
 #print('Qp_mSQ_{mi3}=' , Temp)
 #Temp = np.zeros((3,1), dtype =float)
 #Temp = np.tensordot(CQ[:,2,:,0:2],F0[:,0:2],2)#sDs#
#xDx# When i had CQ[:2,:,0:1]F0[:,0:1] the above calculation was an incomplete calculation, it didn't multiply 
#xDx# with the second column of F0. However 0:1 should mean I am lettting the index run from 0 to 1.
 #print('CQ_{i3k\beta}(F_0)_{k\beta}=' , Temp)
 #Temp = np.zeros((3,1), dtype =float)
 #Temp = np.tensordot(SQ[:,:,:],QU[:,:],2)
 #print('SQ_{ikl}QU_{kl}=', Temp)
 #Temp = np.zeros((3,1), dtype =float)
 #Temp = np.tensordot(XQ,Qp,1)
 #print('Qp_{j}XQ_{ij}=', Temp)
 #Temp = np.zeros((3,1), dtype =float)
 #Temp = np.tensordot(SQ[:,:,0:2],F0[:,0:2],2)
 #print('SQ_{ikb}F0_{kb}=', Temp)
 


LinSolve(T,RHS,CA,SA,XA,np.tensordot(Q,UA,1),np.tensordot(Q,pA,1))

print('---------------------------')
print('T=')
print(T)
print('---------------------------')
print('RHS=')
print(RHS)

print('soln=')
print(np.linalg.solve(T, RHS))
