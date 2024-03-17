#This is a code to compute the equilibrium configuration due to epitaxial strain.
#It has two parts one is computing the equilibrium values and the other for computing the  Claussius clayperon equation.
#The corresponding  equations are laid out in the superconductivity overleaf, Thin Films.tex
#The way to proceed is as follows:-
#1. Enter the constants, like S(Q), C(Q), \chi(Q), U, p, etc
#2. Enter F with b as a variable.
#3. Determine Q from F_0
#4. Solve the linear equations
#5. Evaluate the Claussius Clayperon eqn for phase boundary between phase A and B.
#6. Make all the printing, calculating C from for the MS and MFI into functions and move it to a different file

import numpy as np
import scipy as scp
import math

#Listing the material parameters of phase A in SI units
CA11 = 3.36*(10**11) #\Elastic
CA12 = 1.07*(10**11) #\Elastic
CA13 = CA12 #Elastic
CA22 = CA11 #\Elastic
CA23 = CA13 #\Elastic
CA33 = CA11 #Elastic
CA44 = 1.27*(10**11) #\Elastic
CA55 = CA44 #\Elastic
CA66 = CA44 #\Elastic
SA123 = 123 #SA #\piezo
SA231 = 231 #SA #\piezo
SA312 = 312 #SA #\piezo
SA132 = 132 #SA #\piezo
SA213 = 213 #SA #\piezoelectric Tensor
SA321 = 321 #SA #\piezoelectric tensore
XA11 =  11 #XA #\chi
XA22 =  22 #XA11 #\chi
XA33 =  33 #XA33 #\chi
a0A = 3.902*(10**(-10))  #LAttice constant for Phase A 
a0s = 3.944*(10**(-10))  #Strained Lattice constant.
UA = 

#initializing the Material Tensors of phase A
CA = np.zeros((3,3,3,3), dtype=float)
SA = np.zeros((3,3,3), dtype=float)
XA = np.zeros((3,3), dtype=float) #\chi which is the electrical susceptibility. some defns include the \epsilon_0, some don't
F0 = np.zeros((3,2),dtype=float) #epitaxial strain
Q = np.zeros((3,3),dtype=float) #Rotation to be determined
CAQ = np.zeros((3,3,3,3), dtype=float)
SAQ = np.zeros((3,3,3), dtype=float)
XAQ = np.zeros((3,3), dtype=float) #\chi which is the electrical susceptibility. some defns include the \epsilon_0, some don't
T = np.zeros((6,6), dtype=float)
bp = np.zeros((6,1), dtype=float)
RHS = np.zeros((6,1), dtype=float)

#Inputting the variables
CA[0,0,0,0] = CA11
CA[0,0,1,1] = CA12
CA[0,0,2,2] = CA13
CA[1,1,1,1] = CA22
CA[1,1,2,2] = CA23
CA[2,2,2,2] = CA33
CA[1,2,1,2] = CA44
CA[0,2,0,2] = CA55
CA[0,1,0,1] = CA66
SA[0,1,2] = SA123 
SA[1,2,0] = SA231 
SA[2,0,1] = SA312 
SA[0,2,1] = SA132 
SA[1,0,2] = SA213 
SA[2,1,0] = SA321 
XA[0,0] = XA11
XA[1,1] = XA22
XA[2,2] = XA33
F0[0,0] = (a0s-a0A)/a0A
F0[1,1] = (a0s-a0A)/a0A



#imposing the symmetries
for i in range(3):
 for j in range(3):
  for k in range(3):
   for l in range(3):
    if (CA[i,j,k,l]!=0):
     CA[j,i,k,l]=CA[i,j,k,l]
     CA[i,j,l,k]=CA[i,j,k,l]
     CA[j,i,k,l]=CA[i,j,k,l]
     CA[j,i,l,k]=CA[i,j,k,l]
     CA[k,l,j,i]=CA[i,j,k,l]
     CA[k,l,i,j]=CA[i,j,k,l]
     CA[l,k,i,j]=CA[i,j,k,l]
     CA[l,k,j,i]=CA[i,j,k,l]


#Find Q nearest to F0. The solution based on the min problem is
theta =  math.atan2(F0[0,1]-F0[1,0],F0[0,0]+F0[1,1]) #angle in radians for Q[:,1]
Q[:,0] = [math.cos(theta), math.sin(theta), 0]
Q[:,1] = [-math.sin(theta), math.cos(theta), 0]
Q[:,2] = np.cross(Q[:,0],Q[:,1])
#!!cDc!!print('Q=')
#!!cDc!!print(Q)
#!!cDc!!print(np.linalg.det(Q))

#Find C_{pbrd}(QF)=Q_{pa}Q_{rc}C_{abcd}(F)
for p in range(3):
 for b in range(3):
  for r in range(3):
   for d in range(3):
    for a in range(3):
     for c in range(3):
      CAQ[p,b,r,d] = CAQ[p,b,r,d] + Q[p,a]*Q[r,c]*CA[a,b,c,d]
#!cDc!#Testing if C Transforms properly
#!cDc!for p in range(3):
#!cDc! for b in range(3):
#!cDc!  print('{CA,CAQ}[' + str(p+1) +',' + str(b+1) + ',...]=' + str(np.hstack((CA[p,b,:,:],CAQ[p,b,:,:]))) )

#Find S_{abk}(Q)=Q_{ai}Q_{bj}S_{ijk}(I)
for a in range(3):
 for b in range(3):
  for k in range(3):
   for i in range(3):
    for j in range(3):
     SAQ[a,b,k] = SAQ[a,b,k] + Q[a,i]*Q[b,j]*SA[i,j,k]
#!!cDc!!#Testing if S Transforms properly
#!!cDc!!print(SAQ[0,1,2],SAQ[1,2,0],SAQ[2,0,1],SAQ[0,2,1],SAQ[1,0,2],SAQ[2,1,0])
#!!cDc!!print(SA[0,1,2],SA[1,2,0],SA[2,0,1],SA[0,2,1],SA[1,0,2],SA[2,1,0])

#Find X_{ij}(Q)=Q_{ia}Q_{jb}X_{ab}(I)
for i in range(3):
 for j in range(3):
  for a in range(3):
   for b in range(3):
     XAQ[i,j] = XAQ[i,j] + Q[i,a]*Q[j,b]*XA[a,b]
#!!cDc!!#Testing if S Transforms properly
#!!cDc!!print(np.hstack((XAQ,XA)))


##This section Finds the minimum energy configuration for a slice near well A.
#Assigning the proper Matrices
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
  for m in range(3):
   RHS[i] = RHS[i] + Qp[m]*SQ[m,i,2]
   RHS[i+3] = RHS[i+3] + Qp[m]*XQ[i,m]
  for k in range(3):
   for beta in range(2):
    RHS[i] = RHS[i] - CQ[i,2,k,beta]*F0[k,beta]
    RHS[i+3] = RHS[i+3] - SQ[i,k,beta]*F0[k,beta]


bp = np.linalg.solve(T, RHS)



#!cDc! Testing if C was allocated correctly
#!cDc!#printing the result
#!cDc!for i in range(3):
#!cDc! for j in range(3):
#!cDc!  for k in range(3):
#!cDc!   for l in range(3):
#!cDc!    print("[",i+1,j+1,k+1,l+1,"] ->", CA[i,j,k,l])
