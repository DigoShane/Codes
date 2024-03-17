from dftpy.ions import Ions
from dftpy.field import DirectField
from dftpy.grid import DirectGrid
from dftpy.functional import LocalPseudo, Functional, TotalFunctional
from dftpy.formats import io
from dftpy.math_utils import ecut2nr
from dftpy.time_data import TimeData
from dftpy.optimization import Optimization
from dftpy.mpi import sprint

#path_pp='../DATA/'
path_pp='./'
file1='Al_lda.oe01.recpot'
PP_list = {'Al': path_pp+file1}


from ase.build import bulk
atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
ions = Ions.from_ase(atoms)
# ions = io.read(posfile)

nr = ecut2nr(ecut=35, lattice=ions.cell)
grid = DirectGrid(lattice=ions.cell, nr=nr)
sprint('The final grid size is ', nr)

PSEUDO = LocalPseudo(grid = grid, ions=ions, PP_list=PP_list)

rho_ini = DirectField(grid=grid)
rho_ini[:] = ions.get_ncharges()/ions.cell.volume

KE = Functional(type='KEDF',name='TFvW')
XC = Functional(type='XC',name='LDA')
HARTREE = Functional(type='HARTREE')

evaluator = TotalFunctional(KE=KE, XC=XC, HARTREE=HARTREE, PSEUDO=PSEUDO)

optimization_options = {'econv' : 1e-6*ions.nat}
opt = Optimization(EnergyEvaluator=evaluator, optimization_options = optimization_options,
        optimization_method = 'TN')

rho = opt.optimize_rho(guess_rho=rho_ini)

energy = evaluator.Energy(rho=rho, ions=ions)
print('Energy (a.u.)', energy)

TimeData.output(lprint=True, sort='cost')


rho.write('rho.xsf', ions=ions)
rho.write('rho.cube', ions=ions)

from dftpy.visualize import view

# %matplotlib widget
view(data=rho)

view(ions=ions)



