#!/usr/bin/python

from kim_python_utils.ase import KIMTest
from ase.atoms import Atoms

class TestTest(KIMTest):
    def _calculate(self):
        self.add_property_instance("atomic-mass")        
        print(self.add_key_to_current_property_instance.__doc__)
        self.add_key_to_current_property_instance("species",self.atoms.get_chemical_symbols()[0])
        self.add_key_to_current_property_instance("mass",self.atoms.get_masses()[0],"amu")

        

atoms = Atoms(['Ar','Ar'],[[0,0,0],[1,1,1]],cell=[[2,0,0],[0,2,0],[0,0,2]])

test = TestTest("LennardJones_Ar",atoms)
test()

