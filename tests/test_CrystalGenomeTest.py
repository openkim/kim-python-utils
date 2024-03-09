#!/usr/bin/python

from kim_python_utils.ase import CrystalGenomeTest
from ase.atoms import Atoms

class TestTest(CrystalGenomeTest):
    def _calculate(self):
        pass

atoms = Atoms(['Ar'], [[0, 0, 0]], cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
test = TestTest(model_name="LJ_ElliottAkerson_2015_Universal__MO_959249795837_003", stoichiometric_species=['Ar'], prototype_label='A_cF4_225_a')
input()
