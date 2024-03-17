#This is a Test program for constrained minimization
#Check Appendix of Superconductivity.overlaf ThinFilms.Tex Theorem VI.1

import numpy as np
import scipy as scp
from scipy.optimize import minimize

#initializing the Material Tensors of phase A
F0 = np.zeros((3,2),dtype=float) 

#Inputting the variables
F0[0,0] = 1
F0[1,0] = 2
F0[0,1] = 2
F0[1,1] = 1

#defining the objective function
def Obj(Q):
 return -(np.dot(F0[0:2,0],Q[0:2])+np.dot(F0[0:2,1],Q[3:5]))

#defining the constraint 
def con1(Q):
 return np.dot(Q[0:2],Q[3:5])

def con2(Q):
 return np.dot(Q[0:2],Q[0:2])-1

def con3(Q):
 return np.dot(Q[3:5],Q[3:5])-1


#Initial Guess
Q0 = np.array([1/np.sqrt(2),1/np.sqrt(2),0,-1/np.sqrt(2),1/np.sqrt(2),0]) 

#Printing our objective function
print(Obj(Q0))

#primer for minimization
con1 = {'type':'eq', 'fun':con1}
con2 = {'type':'eq', 'fun':con2}
con3 = {'type':'eq', 'fun':con3}
con = [con1, con2, con3]

#Solution process
sol=minimize(Obj,Q0,method='SLSQP', constraints = con)
print(sol)
print(F0)



