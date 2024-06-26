#This is example 4.



#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Another input style.

There are three different methods to initialize a Mole object.
1. To use function pyscf.M or gto.M as shown by 00-simple_hf.py.
2. As this script did, first create gto.Mole object, assign value to each
   attributes (see pyscf/gto/mole.py file for the details of attributes), then
   call mol.build() to initialize the mol object.
3. First create Mole object, then call .build() function with keyword arguments
   eg

   >>> mol = gto.Mole()
   >>> mol.build(atom='O 0 0 0; H 0 -2.757 2.587; H 0 2.757  2.587', basis='ccpvdz', symmetry=1)

mol.build() should be called to update the Mole object whenever Mole's
attributes are changed in your script. Note to mute the noise produced by
mol.build function, you can execute mol.build(0,0) instead.
'''

from pyscf import gto, scf

mol = gto.Mole()  # initialize molecular object
mol.verbose = 5  # For debugging
mol.output = 'out_Ex-4' # Stating where to store the output.
mol.atom = '''
O        0.000000    0.000000    0.117790
H        0.000000    0.755453   -0.471161
H        0.000000   -0.755453   -0.471161'''  # specying the atoms and their locations.
mol.basis = 'ccpvdz'  # specifying the basis
mol.symmetry = 1    # ??? is 1 the same as true???
mol.build()   # building objects.

mf = scf.RHF(mol)  #Specifying that i want to run Restricted Hartree Fock
mf.kernel() # running.
