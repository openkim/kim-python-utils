################################################################################
#
#  CDDL HEADER START
#
#  The contents of this file are subject to the terms of the Common Development
#  and Distribution License Version 1.0 (the "License").
#
#  You can obtain a copy of the license at
#  http:# www.opensource.org/licenses/CDDL-1.0.  See the License for the
#  specific language governing permissions and limitations under the License.
#
#  When distributing Covered Code, include this CDDL HEADER in each file and
#  include the License file in a prominent location with the name LICENSE.CDDL.
#  If applicable, add the following below this CDDL HEADER, with the fields
#  enclosed by brackets "[]" replaced with your own identifying information:
#
#  Portions Copyright (c) [yyyy] [name of copyright owner]. All rights reserved.
#
#  CDDL HEADER END
#
#  Copyright (c) 2017-2019, Regents of the University of Minnesota.
#  All rights reserved.
#
#  Contributor(s):
#     Ellad B. Tadmor
#     Daniel S. Karls
#
################################################################################
"""
Helper routines for KIM Tests and Verification Checks

"""
# Python 2-3 compatible code issues
from __future__ import print_function

import itertools
import random

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.calculators.kim.kim import KIM

__version__ = "0.1.0"
__author__ = ["Ellad B. Tadmor", "Daniel S. Karls"]
__all__ = [
    "KIMASEError",
    "atom_outside_cell_along_nonperiodic_dim",
    "check_if_atoms_interacting_energy",
    "check_if_atoms_interacting_force",
    "check_if_atoms_interacting",
    "get_isolated_energy_per_atom",
    "get_model_energy_cutoff",
    "fractional_coords_transformation",
    "perturb_until_all_forces_sizeable",
    "randomize_positions",
    "randomize_species",
    "remove_species_not_supported_by_ASE",
    "rescale_to_get_nonzero_energy",
    "rescale_to_get_nonzero_forces",
]


################################################################################
class KIMASEError(Exception):
    def __init__(self, msg):
        # Call the base class constructor with the parameters it needs
        super(KIMASEError, self).__init__(msg)
        self.msg = msg

    def __str__(self):
        return self.msg


################################################################################
def remove_species_not_supported_by_ASE(species):
    """
    Remove any species from the 'species' list that are not supported by ASE
    """
    supported_species = chemical_symbols[1:]
    return [s for s in species if s in supported_species]


################################################################################
def randomize_species(atoms, species):
    """
    Given an ASE 'atoms' object, set random element for each atom selected
    from the list of available 'species' in a way that ensures all are
    represented with the same probabilities.
    """
    # Indefinitely iterate through species putting them at random
    # unoccupied sites until all atoms are exhausted.
    indices = list(range(len(atoms)))
    num_occupied = 0
    for element in itertools.cycle(species):
        i = random.randint(0, len(indices) - 1)
        atoms[indices[i]].symbol = element
        del indices[i]
        num_occupied += 1
        if num_occupied == len(atoms):
            break


################################################################################
def fractional_coords_transformation(cell):
    """
    Given a set of cell vectors, this will return a transformation matrix T that can be
    used to multiply any arbitrary position to get its fractional coordinates in the
    basis of those cell vectors.
    """
    simulation_cell_volume = np.linalg.det(cell)

    T = (
        np.vstack(
            (
                np.cross(cell[:, 1], cell[:, 2]),
                np.cross(cell[:, 2], cell[:, 0]),
                np.cross(cell[:, 0], cell[:, 1]),
            )
        )
        / simulation_cell_volume
    )

    return T


################################################################################
def atom_outside_cell_along_nonperiodic_dim(T, atom_coord, pbc, tol=1e-12):
    """
    Given a transformation matrix to apply to an atomic position to get its fractional
    position in the basis of some cell vectors and the corresponding boundary
    conditions, determine if the atom is outside of the cell along any non-periodic
    directions (measured using tolerance 'tol').  This is relevant when using an SM from
    a simulator such as LAMMPS that performs spatial decomposition -- atoms that leave
    the box along non-periodic dimensions are likely to become "lost."
    """
    if all(pbc):
        # Skip the actual checks
        return False

    # Calculate fractional coordinate of this atom
    atom_coord_fractional = np.dot(T, atom_coord)
    for dof in range(0, 3):
        if not pbc[dof] and (
            atom_coord_fractional[dof] < -tol or atom_coord_fractional[dof] > 1 + tol
        ):
            return True

    # If we made it to here, this atom's position is OK
    return False


################################################################################
def randomize_positions(atoms, pert_amp):
    """
    Given an ASE 'atoms' object, displace all atomic coordinates by a random amount in
    the range [-pert_amp, pert_amp] along *each* dimension.  Note that all atomic
    coordinates must be inside of the corresponding cell along non-periodic dimensions.
    As each atom is looped over, we continue generating perturbations until the
    displaced position is inside of the cell along any non-periodic directions
    (displacing outside the cell along periodic dimensions is allowed, although it's up
    to the calling function to wrap the positions if they need to be).
    """
    pbc = atoms.get_pbc()
    if all(pbc):
        # Don't need to worry about moving atoms outside of cell
        for at in range(0, len(atoms)):
            atoms[at].position += [
                pert_amp * random.uniform(-1.0, 1.0) for i in range(3)
            ]

    else:
        # Get transformation matrix to get fractional coords
        T = fractional_coords_transformation(atoms.get_cell())

        for at in range(0, len(atoms)):
            # Check if the positional coordinate of this atom is valid to begin
            # with, i.e. it's inside the box along along all non-periodic directions
            if atom_outside_cell_along_nonperiodic_dim(T, atoms[at].position, pbc):
                raise KIMASEError(
                    "ERROR: Determined that atom {} with position {} is outside of the "
                    "simulation cell ({}) along one or more non-periodic directions.  "
                    "In order to prevent atoms from being lost, they must all be "
                    "contained inside of the simulation cell along non-periodic "
                    "dimensions.".format(at, atoms[at].position, atoms.get_cell())
                )

            for dof in range(0, 3):
                coord = atoms[at].position[dof].copy()
                done = False
                while not done:
                    atoms[at].position[dof] += random.uniform(-1.0, 1.0) * pert_amp
                    if not atom_outside_cell_along_nonperiodic_dim(
                        T, atoms[at].position, pbc
                    ):
                        done = True
                    else:
                        atoms[at].position[dof] = coord


################################################################################
def get_isolated_energy_per_atom(model, symbol):
    """
    Construct a non-periodic cell containing a single atom and compute its energy.
    """
    single_atom = Atoms(
        symbol,
        positions=[(0.1, 0.1, 0.1)],
        cell=(20, 20, 20),
        pbc=(False, False, False),
    )
    calc = KIM(model)
    single_atom.set_calculator(calc)
    energy_per_atom = single_atom.get_potential_energy()
    calc.__del__()
    del single_atom
    return energy_per_atom


################################################################################
def rescale_to_get_nonzero_energy(atoms, isolated_energy_per_atom, etol):
    """
    If the given configuration has a potential energy, relative to the sum of the
    isolated energy corresponding to each atom present, smaller in magnitude than 'etol'
    (presumably because the distance between atoms is too large), rescale it making it
    smaller.  The 'isolated_energy_per_atom' arg should be a dict containing an entry
    for each atomic species present in the atoms object (additional entries are ignored).
    """
    num_atoms = len(atoms)
    if num_atoms < 2:
        # If we're only using a single atom, then we need to make sure that the cell is
        # periodic along at least one direction
        if num_atoms == 1:
            pbc = atoms.get_pbc()
            if not any(pbc):
                raise RuntimeError(
                    "ERROR: If only a single atom is present, the cell must "
                    "be periodic along at least one direction."
                )
        else:
            raise RuntimeError(
                "ERROR: Invalid configuration. Must have at least one atom"
            )

    if not isinstance(isolated_energy_per_atom, dict):
        raise ValueError(
            "Argument 'isolated_energy_per_atom' passed to "
            "rescale_to_get_nonzero_energy must be a dict containing and entry "
            "for each atomic species present in the atoms object."
        )

    # Check for any flat directions in the initial configuration and ignore them when
    # determining whether to stop rescaling
    pmin = atoms.get_positions().min(axis=0)  # minimum x,y,z coordinates
    pmax = atoms.get_positions().max(axis=0)  # maximum x,y,z coordinates
    delp = pmax - pmin  # system extent across x, y, z
    flat = [(extent <= np.finfo(extent).tiny) for extent in delp]

    # Compute the "trivial energy", i.e. the energy assuming none of the atoms interact
    # with each other at all
    species_of_each_atom = atoms.get_chemical_symbols()
    energy_trivial = 0.0
    for atom_species in species_of_each_atom:
        energy_trivial += isolated_energy_per_atom[atom_species]

    # Rescale cell and atoms
    cell = atoms.get_cell()
    energy = atoms.get_potential_energy()
    adjusted_energy = energy - energy_trivial
    if abs(adjusted_energy) <= etol:
        pmin = atoms.get_positions().min(axis=0)  # minimum x,y,z coordinates
        pmax = atoms.get_positions().max(axis=0)  # maximum x,y,z coordinates
        extent_along_nonflat_directions = [
            extent for direction, extent in enumerate(delp) if not flat[direction]
        ]
        delpmin = min(extent_along_nonflat_directions)
        while delpmin > np.finfo(delpmin).tiny:
            atoms.positions *= 0.5  # make configuration half the size
            cell *= 0.5  # make cell half the size
            atoms.set_cell(cell)  # need to adjust cell in case it's periodic
            delpmin *= 0.5
            energy = atoms.get_potential_energy()
            adjusted_energy = energy - energy_trivial
            if abs(adjusted_energy) > etol:
                return  # success!

        # Get species and write out error
        raise RuntimeError(
            "ERROR: Unable to scale configuration down to nonzero energy.  This was "
            "determined by computing the total potential energy relative to the sum of "
            "the isolated energy corresponding to each atom present and checking if "
            "the magnitude of the difference was larger than the supplied tolerance of "
            "{} eV.  This may mean that the species present in the cell ({}) do not "
            "have a non-trivial energy interaction for the  potential being used."
            "".format(etol, set(species_of_each_atom))
        )


################################################################################
def check_if_atoms_interacting_energy(model, symbols, etol):
    """
    First, get the energy of a single isolated atom of each species given in 'symbols'.
    Then, construct a dimer consisting of these two species and try to decrease its bond
    length until a discernible difference in the energy (from the sum of the isolated
    energy of each species) is detected.  The 'symbols' arg should be a list or tuple of
    length 2 indicating which species pair to check, e.g. to check if Al interacts with
    Al, one should specify ['Al', 'Al'].
    """
    if not isinstance(symbols, (list, tuple)) or len(symbols) != 2:
        raise ValueError(
            "Argument 'symbols' passed to check_if_atoms_interacting_energy "
            "must be a list of tuple of length 2 indicating the species pair to "
            "check"
        )

    isolated_energy_per_atom = {}
    isolated_energy_per_atom[symbols[0]] = get_isolated_energy_per_atom(
        model, symbols[0]
    )
    isolated_energy_per_atom[symbols[1]] = get_isolated_energy_per_atom(
        model, symbols[1]
    )

    dimer = Atoms(
        symbols,
        positions=[(0.1, 0.1, 0.1), (5.1, 0.1, 0.1)],
        cell=(20, 20, 20),
        pbc=(False, False, False),
    )
    calc = KIM(model)
    dimer.set_calculator(calc)
    try:
        rescale_to_get_nonzero_energy(dimer, isolated_energy_per_atom, etol)
        atoms_interacting = True
        return atoms_interacting
    except:  # noqa: E722
        atoms_interacting = False
        return atoms_interacting
    finally:
        calc.__del__()
        del dimer


################################################################################
def check_if_atoms_interacting_force(model, symbols, ftol):
    """
    Construct a dimer and try to decrease its bond length until the force acting on each
    atom is larger than 'ftol' in magnitude.  The 'symbols' arg should be a list or
    tuple of length 2 indicating which species pair to check, e.g. to check if Al
    interacts with Al, one should specify ['Al', 'Al'].
    """
    if not isinstance(symbols, (list, tuple)) or len(symbols) != 2:
        raise ValueError(
            "Argument 'symbols' passed to check_if_atoms_interacting_force "
            "must be a list of tuple of length 2 indicating the species pair to "
            "check"
        )

    dimer = Atoms(
        symbols,
        positions=[(0.1, 0.1, 0.1), (5.1, 0.1, 0.1)],
        cell=(20, 20, 20),
        pbc=(False, False, False),
    )
    calc = KIM(model)
    dimer.set_calculator(calc)
    try:
        rescale_to_get_nonzero_forces(dimer, ftol)
        atoms_interacting = True
        return atoms_interacting
    except:  # noqa: E722
        atoms_interacting = False
        return atoms_interacting
    finally:
        calc.__del__()
        del dimer


################################################################################
def check_if_atoms_interacting(
    model, symbols, check_energy=True, etol=1e-6, check_force=True, ftol=1e-3
):
    """
    Check to see whether non-trivial energy and/or forces can be detected using the
    current model.  The 'symbols' arg should be a list or tuple of length 2 indicating
    which species pair to check, e.g. to check if Al interacts with Al, one should
    specify ['Al', 'Al'].
    """
    if check_energy and not check_force:
        return check_if_atoms_interacting_energy(model, symbols, etol)
    elif not check_energy and check_force:
        return check_if_atoms_interacting_force(model, symbols, ftol)

    elif check_energy and check_force:
        atoms_interacting_energy = check_if_atoms_interacting_energy(
            model, symbols, etol
        )
        atoms_interacting_force = check_if_atoms_interacting_energy(
            model, symbols, ftol
        )
        return atoms_interacting_energy, atoms_interacting_force


################################################################################
def rescale_to_get_nonzero_forces(atoms, ftol):
    """
    If the given configuration has force components which are all smaller in absolute
    value than 'ftol' (presumably because the distance between atoms is too large),
    rescale it to be smaller until the largest force component in absolute value is
    greater than or equal to 'ftol'.  In a perfect crystal, the crystal is rescaled
    until the atoms on the surface reach the minimum value (internal atoms padded with
    another atoms around them will have zero force).  Note that any periodicity is turned
    off for the rescaling and then restored at the end.
    """
    if len(atoms) < 2:
        raise KIMASEError(
            "ERROR: Invalid configuration. Must have at least 2 atoms. Number of atoms "
            "= {}".format(len(atoms))
        )

    # Check for any flat directions in the initial configuration and ignore them when
    # determining whether to stop rescaling
    pmin = atoms.get_positions().min(axis=0)  # minimum x,y,z coordinates
    pmax = atoms.get_positions().max(axis=0)  # maximum x,y,z coordinates
    delp = pmax - pmin  # system extent across x, y, z
    flat = [(extent <= np.finfo(extent).tiny) for extent in delp]

    # Temporarily turn off any periodicity
    pbc_save = atoms.get_pbc()
    cell = atoms.get_cell()
    atoms.set_pbc([False, False, False])
    # Rescale cell and atoms
    forces = atoms.get_forces()
    fmax = max(abs(forces.min()), abs(forces.max()))  # find max in abs value
    if fmax < ftol:
        pmin = atoms.get_positions().min(axis=0)  # minimum x,y,z coordinates
        pmax = atoms.get_positions().max(axis=0)  # maximum x,y,z coordinates
        extent_along_nonflat_directions = [
            extent for direction, extent in enumerate(delp) if not flat[direction]
        ]
        delpmin = min(extent_along_nonflat_directions)
        while delpmin > np.finfo(delpmin).tiny:
            atoms.positions *= 0.5  # make configuration half the size
            cell *= 0.5  # make cell half the size
            delpmin *= 0.5
            forces = atoms.get_forces()  # get max force
            fmax = max(abs(forces.min()), abs(forces.max()))
            if fmax >= ftol:
                # Restore periodicity
                atoms.set_pbc(pbc_save)
                atoms.set_cell(cell)
                return  # success!
        raise KIMASEError(
            "ERROR: Unable to scale configuration down to nonzero forces."
        )
    else:
        # Restore periodicity
        atoms.set_pbc(pbc_save)


################################################################################
def perturb_until_all_forces_sizeable(
    atoms, pert_amp, minfact=0.1, maxfact=5.0, max_iter=1000
):
    """
    Keep perturbing atoms in the ASE 'atoms' object until all force components on each
    atom have an absolute value of least 'minfact' times the largest (in absolute value)
    component across all force vectors coming in.  Note that all atomic coordinates must
    be inside of the corresponding cell along non-periodic dimensions.  Perturbations
    leading to a force component on any atom that is larger than 'maxfact' times the
    largest force component coming in are rejected.  Perturbations leading to atoms
    outside of the span across x, y, and z of the atomic positions coming in or outside
    of the simulation cell along non-periodic directions are also rejected.  The process
    repeats until max_iter iterations have been reached, at which point an exception is
    raised.
    """
    pbc = atoms.get_pbc()

    # Get transformation matrix to get fractional coords
    T = fractional_coords_transformation(atoms.get_cell())

    # First, ensure that all atomic positions are inside of the cell along non-periodic
    # dimensions
    if not all(pbc):
        for at in range(0, len(atoms)):
            # Check if the positional coordinate of this atom is valid to begin
            # with, i.e. it's inside the box along along all non-periodic directions
            if atom_outside_cell_along_nonperiodic_dim(T, atoms[at].position, pbc):
                raise KIMASEError(
                    "ERROR: Determined that atom {} with position {} is outside of the "
                    "simulation cell ({}) along one or more non-periodic directions.  "
                    "In order to prevent atoms from being lost, they must all be "
                    "contained inside of the simulation cell along non-periodic "
                    "dimensions.".format(at, atoms[at].position, atoms.get_cell())
                )

    forces = atoms.get_forces()
    fmax = max(abs(forces.min()), abs(forces.max()))  # find max in abs value
    pmin = atoms.get_positions().min(axis=0)  # minimum x,y,z coordinates
    pmax = atoms.get_positions().max(axis=0)  # maximum x,y,z coordinates
    saved_posns = atoms.get_positions().copy()
    saved_forces = atoms.get_forces().copy()
    some_forces_too_small = True

    # Counter to enforce max iterations
    iters = 0

    while some_forces_too_small:
        for at in range(0, len(atoms)):
            for dof in range(0, 3):
                if abs(forces[at, dof]) < minfact * fmax:
                    done = False
                    coord = atoms[at].position[dof].copy()
                    while not done:
                        atoms[at].position[dof] += random.uniform(-1.0, 1.0) * pert_amp
                        if (
                            pmin[dof] <= atoms[at].position[dof] <= pmax[dof]
                        ) and not atom_outside_cell_along_nonperiodic_dim(
                            T, atoms[at].position, pbc
                        ):
                            done = True
                        else:
                            atoms[at].position[dof] = coord
        try:
            forces = atoms.get_forces()
            fmax_new = max(abs(forces.min()), abs(forces.max()))
            if fmax_new > maxfact * fmax:
                # forces too large, abort perturbation
                atoms.set_positions(saved_posns)
                forces = saved_forces.copy()
                continue
            fmin_new = min(abs(forces.min()), abs(forces.max()))
            if fmin_new > minfact * fmax:
                some_forces_too_small = False
        except:  # noqa: E722
            # force calculation failed, abort perturbation
            atoms.set_positions(saved_posns)
            continue
        finally:
            iters = iters + 1
            if iters == max_iter:
                raise KIMASEError(
                    "Maximum iterations ({}) exceeded in call to "
                    "function perturb_until_all_forces_sizeable()".format(max_iter)
                )


################################################################################
def get_model_energy_cutoff(
    model,
    symbols,
    xtol,
    etol_coarse,
    etol_fine,
    offset,
    max_bisect_iters=1000,
    max_upper_cutoff_bracket=20.0,
):
    """
    Compute the distance at which energy interactions become non-trival for a given
    model and a species pair it supports.  This is done by constructing a dimer composed
    of these species in a large finite box, increasing the separation if necessary until
    the total potential energy is within 'etol_fine' of the sum of the corresponding
    isolated energies, and then shrinking the separation until the energy differs from
    that value by more than 'etol_coarse'.  Using these two separations to bound the
    search range, bisection is used to refine in order to locate the cutoff.  The
    'symbols' arg should be a list or tuple of length 2 indicating which species pair to
    check, e.g. to get the energy cutoff of Al with Al, one should specify ['Al', 'Al'].

    This function is based on the content of the DimerContinuityC1__VC_303890932454_002
    Verification Check in OpenKIM [1-3].

    [1] Tadmor E. Verification Check of Dimer C1 Continuity v002. OpenKIM; 2018.
        doi:10.25950/43d2c6d5

    [2] Tadmor EB, Elliott RS, Sethna JP, Miller RE, Becker CA. The potential of
        atomistic simulations and the Knowledgebase of Interatomic Models. JOM.
        2011;63(7):17. doi:10.1007/s11837-011-0102-6

    [3] Elliott RS, Tadmor EB. Knowledgebase of Interatomic Models (KIM) Application
        Programming Interface (API). OpenKIM; 2011. doi:10.25950/ff8f563a
    """
    from scipy.optimize import bisect

    def get_dimer_positions(a, large_cell_len):
        """
        Generate positions for a dimer of length 'a' centered in a finite simulation box
        with side length 'large_cell_len'
        """
        half_cell = 0.5 * large_cell_len
        positions = [
            [half_cell - 0.5 * a, half_cell, half_cell],
            [half_cell + 0.5 * a, half_cell, half_cell],
        ]
        return positions

    def energy(a, dimer, large_cell_len, offset):
        dimer.set_positions(get_dimer_positions(a, large_cell_len))
        return dimer.get_potential_energy()

    def energy_cheat(a, dimer, large_cell_len, offset):
        dimer.set_positions(get_dimer_positions(a, large_cell_len))
        return dimer.get_potential_energy() + offset

    if not isinstance(symbols, (list, tuple)) or len(symbols) != 2:
        raise ValueError(
            "Argument 'symbols' passed to check_if_atoms_interacting_energy "
            "must be a list of tuple of length 2 indicating the species pair to "
            "check"
        )

    isolated_energy_per_atom = {}
    isolated_energy_per_atom[symbols[0]] = get_isolated_energy_per_atom(
        model, symbols[0]
    )
    isolated_energy_per_atom[symbols[1]] = get_isolated_energy_per_atom(
        model, symbols[1]
    )
    einf = isolated_energy_per_atom[symbols[0]] + isolated_energy_per_atom[symbols[1]]

    # First, establish the upper bracket cutoff by starting at 'b_init' Angstroms and
    # incrementing by 'db' until
    b_init = 4.0

    # Create finite box of size large_cell_len
    large_cell_len = 50
    dimer = Atoms(
        symbols,
        positions=get_dimer_positions(b_init, large_cell_len),
        cell=(large_cell_len, large_cell_len, large_cell_len),
        pbc=(False, False, False),
    )
    calc = KIM(model)
    dimer.set_calculator(calc)

    db = 2.0
    b = b_init
    still_interacting = True
    while still_interacting:
        b += db
        if b > max_upper_cutoff_bracket:
            if hasattr(calc, "__del__"):
                calc.__del__()

            raise KIMASEError(
                "Exceeded limit on upper bracket when determining cutoff "
                "search range"
            )
        else:
            eb = energy(b, dimer, large_cell_len)
            if abs(eb - einf) < etol_fine:
                still_interacting = False

    a = b
    da = 0.01
    not_interacting = True
    while not_interacting:
        a -= da
        if a < 0:
            if hasattr(calc, "__del__"):
                calc.__del__()

            raise RuntimeError(
                "Failed to determine lower bracket for cutoff search using etol_coarse "
                "= {}.  This may mean that the species pair provided ({}) does not "
                "have a non-trivial energy interaction for the potential being "
                "used.".format(etol_coarse, symbols)
            )
        else:
            ea = energy(a, dimer, large_cell_len)
            if abs(ea - einf) > etol_coarse:
                not_interacting = False

    # NOTE: Some Simulator Models have a history dependence due to them maintaining
    #       charges from the previous energy evaluation to use as an initial guess
    #       for the next charge equilibration.  We therefore have to treat them not
    #       as single-valued functions but as distributions, i.e.  for a given
    #       configuration you might get any of a range of energy values depending on
    #       the history of your previous energy evaluations.  This is particularly
    #       problematic for this step, where we set up a bisection problem in order
    #       to determine the cutoff radius of the model.  Our solution for this
    #       specific case is to make a very crude estimate of the variance of that
    #       distribution with a 10% factor of safety on it.
    eb_new = energy(b, dimer, large_cell_len)
    eb_error = abs(eb_new - eb)

    # compute offset to ensure that energy before and after cutoff have
    # different signs
    if ea < eb:
        offset = -eb + 1.1 * eb_error + np.finfo(float).eps
    else:
        offset = -eb - 1.1 * eb_error - np.finfo(float).eps

    rcut, results = bisect(
        energy_cheat,
        a,
        b,
        args=(dimer, large_cell_len, offset),
        full_output=True,
        xtol=xtol,
        maxiter=max_bisect_iters,
    )

    # General clean-up
    if hasattr(calc, "__del__"):
        calc.__del__()

    if not results.converged:
        raise RuntimeError(
            "Bisection search to find cutoff distance did not converge "
            "within {} iterations with xtol = {}".format(max_bisect_iters, xtol)
        )
    else:
        return rcut


# If called directly, do nothing
if __name__ == "__main__":
    pass
