#This is an implementation for PBC in Pyscf.

from pyscf.pbc import gto
import numpy as np

cell = gto.Cell()
cell.atom = 'H 0 0 0; H 1 1 1'
cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pade'
cell.a = np.eye(3) * 2
cell.build()




