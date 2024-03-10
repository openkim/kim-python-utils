#!/usr/bin/python

from kim_python_utils.ase import KIMTest
from ase.atoms import Atoms

class TestTest(KIMTest):
    def _calculate(self,structure_index: int):
        """
        example calculate method

        Args:
            structure_index:
                KIM tests can loop over multiple structures (i.e. crystals, molecules, etc.). This indicates which is being used for the current calculation.        
        """
        self._add_property_instance("atomic-mass")
        self._add_key_to_current_property_instance("species", self.atoms[structure_index].get_chemical_symbols()[0])
        self._add_key_to_current_property_instance("mass", self.atoms[structure_index].get_masses()[0], "amu")


atoms = Atoms(['Ar'], [[0, 0, 0]], cell=[[1, 0, 0], [0, 2, 0], [0, 0, 2]])
atoms_list = [atoms,atoms] # not modifying, so no need to copy
test = TestTest(model_name="LennardJones_Ar", atoms=atoms_list)
test()
