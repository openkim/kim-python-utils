#!/usr/bin/python

from kim_python_utils.ase import KIMTest
from ase.atoms import Atoms

class TestTest(KIMTest):
    def _calculate(self):
        self.add_property_instance("atomic-mass")
        self.add_key_to_current_property_instance("species", self.atoms[0].get_chemical_symbols()[0])
        self.add_key_to_current_property_instance("mass", self.atoms[0].get_masses()[0], "amu")


atoms = Atoms(['Ar'], [[0, 0, 0]], cell=[[1, 0, 0], [0, 2, 0], [0, 0, 2]])
test = TestTest(model_name="LennardJones_Ar", atoms=atoms)
test()
