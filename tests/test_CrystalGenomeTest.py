#!/usr/bin/python

from kim_python_utils.ase import CrystalGenomeTest, get_isolated_energy_per_atom
from crystal_genome_util.aflow_util import get_stoich_reduced_list_from_prototype

class TestTest(CrystalGenomeTest):
    def _calculate(self,structure_index: int):
        """
        example calculate method. Just writes the binding-energy and crystal-structure-npt properties assuming the provided
        structure is already at equilibrium

        Args:
            structure_index:
                KIM tests can loop over multiple structures (i.e. crystals, molecules, etc.). This indicates which is being used for the current calculation.        
        """
        # binding energy
        self._add_property_instance("binding-energy-crystal")

        # add common fields
        self._add_common_crystal_genome_keys_to_current_property_instance(structure_index=structure_index, write_stress=False, write_temp=False)

        # calculate potential energy and do the required stuff to figure out per-formula and per-atom, and subtract isolated energy
        potential_energy = self.atoms[structure_index].get_potential_energy()
        potential_energy_per_atom = potential_energy/len(self.atoms[structure_index])
        reduced_stoichiometry = get_stoich_reduced_list_from_prototype(self.prototype_label) # i.e. "AB3\_...." -> [1,3]        
        binding_energy_per_formula = potential_energy_per_atom * sum(reduced_stoichiometry)
        for num_in_formula,species in zip(reduced_stoichiometry,self.stoichiometric_species):
            binding_energy_per_formula -= num_in_formula*get_isolated_energy_per_atom(self.model_name,species)
        binding_energy_per_atom = binding_energy_per_formula/sum(reduced_stoichiometry)

        # add the fields unique to this property
        self._add_key_to_current_property_instance("binding-potential-energy-per-atom",binding_energy_per_atom,"eV")
        self._add_key_to_current_property_instance("binding-potential-energy-per-formula",binding_energy_per_formula,"eV")

        # structure
        self._add_property_instance("crystal-structure-npt")
        
        # nothing to actually calculate
        self._add_common_crystal_genome_keys_to_current_property_instance(structure_index=structure_index, write_stress=True, write_temp=True)


# good test case (combination of material and model) -- it has multiple equilibria, one of which has a short-name, the other one does not
test = TestTest(model_name="MEAM_LAMMPS_KoJimLee_2012_FeP__MO_179420363944_002", stoichiometric_species=['Fe','P'], prototype_label='AB_oP8_62_c_c')
test()

atoms = test.atoms # is a list
# can go the other way too -- instead of giving a prototype to query for, give atoms object(s)
test = TestTest(model_name="MEAM_LAMMPS_KoJimLee_2012_FeP__MO_179420363944_002", atoms=atoms)
test()

