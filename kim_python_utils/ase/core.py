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
from ase.calculators.calculator import Calculator
from typing import Any, Optional, List, Union
from abc import ABC, abstractmethod
from kim_property import kim_property_create, kim_property_modify, kim_property_dump
import kim_edn
from crystal_genome_util import aflow_util
from kim_query import raw_query
from tempfile import NamedTemporaryFile
import os

__version__ = "0.3.0"
__author__ = ["Ellad B. Tadmor", "Daniel S. Karls", "ilia Nikiforov", "Eric Fuemmeler"]
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
    "KIMTest",
    "CrystalGenomeTest"
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
def randomize_species(atoms, species, seed=None):
    """
    Given an ASE 'atoms' object, set random element for each atom selected
    from the list of available 'species' in a way that ensures all are
    represented with the same probabilities.
    """
    if seed is not None:
        random.seed(seed)

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
def randomize_positions(atoms, pert_amp, seed=None):
    """
    Given an ASE 'atoms' object, displace all atomic coordinates by a random amount in
    the range [-pert_amp, pert_amp] along *each* dimension.  Note that all atomic
    coordinates must be inside of the corresponding cell along non-periodic dimensions.
    As each atom is looped over, we continue generating perturbations until the
    displaced position is inside of the cell along any non-periodic directions
    (displacing outside the cell along periodic dimensions is allowed, although it's up
    to the calling function to wrap the positions if they need to be).
    """
    if seed is not None:
        random.seed(seed)

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
    single_atom.calc = calc
    energy_per_atom = single_atom.get_potential_energy()
    if hasattr(calc, "__del__"):
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
    dimer.calc = calc
    try:
        rescale_to_get_nonzero_energy(dimer, isolated_energy_per_atom, etol)
        atoms_interacting = True
        return atoms_interacting
    except:  # noqa: E722
        atoms_interacting = False
        return atoms_interacting
    finally:
        if hasattr(calc, "__del__"):
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
    dimer.calc = calc
    try:
        rescale_to_get_nonzero_forces(dimer, ftol)
        atoms_interacting = True
        return atoms_interacting
    except:  # noqa: E722
        atoms_interacting = False
        return atoms_interacting
    finally:
        if hasattr(calc, "__del__"):
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
    if np.isnan(forces).any():
        raise RuntimeError("ERROR: Computed forces include at least one nan.")
    fmax = max(abs(forces.min()), abs(forces.max()))  # find max in abs value
    if fmax < ftol:
        pmin = atoms.get_positions().min(axis=0)  # minimum x,y,z coordinates
        pmax = atoms.get_positions().max(axis=0)  # maximum x,y,z coordinates
        extent_along_nonflat_directions = [
            extent for direction, extent in enumerate(delp) if not flat[direction]
        ]
        delpmin = min(extent_along_nonflat_directions)
        while delpmin > np.finfo(delpmin).tiny:
            atoms.positions *= 0.75  # make configuration 3/4 the size
            cell *= 0.75  # make cell 3/4 the size
            delpmin *= 0.75
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
    if np.isnan(forces).any():
        raise RuntimeError("ERROR: Computed forces include at least one nan.")
    fmax = max(abs(forces.min()), abs(forces.max()))  # find max in abs value

    if fmax <= 1e2 * np.finfo(float).eps:
        raise KIMASEError(
            "ERROR: Largest force component on configuration is "
            "less than or equal to 1e2*machine epsilon. Cannot proceed."
        )

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
            if np.isnan(forces).any():
                raise RuntimeError("ERROR: Computed forces include at least one nan.")
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
    xtol=1e-8,
    etol_coarse=1e-6,
    etol_fine=1e-15,
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

    def energy(a, dimer, large_cell_len, einf):
        dimer.set_positions(get_dimer_positions(a, large_cell_len))
        return dimer.get_potential_energy() - einf

    def energy_cheat(a, dimer, large_cell_len, offset, einf):
        dimer.set_positions(get_dimer_positions(a, large_cell_len))
        return (dimer.get_potential_energy() - einf) + offset

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
    dimer.calc = calc

    db = 2.0
    b = b_init - db
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
            eb = energy(b, dimer, large_cell_len, einf)
            if abs(eb) < etol_fine:
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
            ea = energy(a, dimer, large_cell_len, einf)
            if abs(ea) > etol_coarse:
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
    eb_new = energy(b, dimer, large_cell_len, einf)
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
        args=(dimer, large_cell_len, offset, einf),
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


################################################################################
class KIMTest(ABC):
    """
    A KIM test

    Attributes:
        model_name:
            KIM model name to use for calculations
        model:
            ASE calculator to use for calculations        
        atoms:
            List of ASE atoms objects to use as the initial configurations or to build supercells
        filename:            
            Filename to which the EDN property instance will be written
    """

    def __init__(self, model_name: Optional[str] = None, model: Optional[Calculator] = None, 
                 atoms: Optional[Union[Atoms,List[Atoms]]] = None, filename: str = "output/results.edn"):
        """
        Args:
            model_name:
                KIM extended-id of the model. Provide this or `model`
            model:
                ASE calculator to use. Provide this or `model_name`
            atoms:
                List of ASE atoms objects to use as the initial configurations or to build supercells. 
                If a single atoms object is provided, it will be converted to a single-element list
            filename:
                Path to results.edn file to be written. The default provided is the correct path to work in
                the KIM Pipeline or KIM Developer Platform
        """
        if model_name is not None:
            if model is not None:
                raise KIMASEError("Please provide either a KIM model name or an ASE calculator, not both")            
            self.model_name = model_name
            self.model = KIM(model_name)
        elif model is not None:
            self.model = model
        else:
            raise KIMASEError("Please provide either a KIM model name or an ASE calculator")
        
        if isinstance(atoms,List):
            self.atoms = atoms
        else:
            self.atoms = [atoms]
        
        self.property_instances = "[]"
        self.filename = filename

    @abstractmethod
    def _calculate(self, structure_index: int, **kwargs):
        """
        Abstract calculate method

        Args:
            structure_index:
                KIM tests can loop over multiple structures (i.e. crystals, molecules, etc.). This indicates which is being used for the current calculation.
                TODO: Using an index here seems un-Pythonic, any way around it?
        """
        raise NotImplementedError("Subclasses must implement the _calculate method.")

    def _write_to_file(self):
        with open(self.filename, "w") as f:
            kim_property_dump(self.property_instances, f)

    def _validate(self):
        """
        Optional physics validation of properies, to be implemented by each sublass
        """
        pass

    def __call__(self, **kwargs):
        """
        runs test and outputs results
        """
        for i,atoms in enumerate(self.atoms):
            # TODO: this seems like a very un-Pythonic way to do this, but I can't think of another way to give the _calculate
            # function a way to handle multiple initial structures except mandating that the developer always include a loop in _calculate.
            # Just passing atoms to calculate wouldn't work, what if, for example, someone has a Crystal Genome test that works directly
            # with the symmetry-reduced description?

            # still, the most common use case is an ASE calculation with Atoms, so set the calculator here
            atoms.calc = self.model
            
            self._calculate(i, **kwargs)
        self._validate()
        self._write_to_file()

    def _add_property_instance(self, property_name: str):
        """
        Initialize a new property instance to self.property_instances. It will automatically get the an instance-id
        equal to the length of self.property_instances after it is added. It assumed that if you are calling this function,
        you have been only using the simplified property functions in this class and not doing any more advanced editing
        to self.property_instance using kim_property or any other methods.

        Args:
            property_name:
                The property name, e.g. "tag:staff@noreply.openkim.org,2023-02-21:property/binding-energy-crystal" or
                "binding-energy-crystal"
        """
        # DEV NOTE: I like to use the package name when using kim_edn so there's no confusion with json.loads etc.
        property_instances_deserialized = kim_edn.loads(self.property_instances)
        new_instance_index = len(property_instances_deserialized) + 1
        for property_instance in property_instances_deserialized:
            if property_instance["instance-id"] == new_instance_index:
                raise KIMASEError("instance-id that matches the length of self.property_instances already exists.\n"
                                  "Was self.property_instances edited directly instead of using this package?")
        self.property_instances = kim_property_create(new_instance_index, property_name, self.property_instances)

    def _add_key_to_current_property_instance(self, name: str, value: Any, units: Optional[str] = None):
        """
        TODO: Add uncertainty output

        Write a key to the last element of self.property_instances. If the value is an array,
        this function will assume you want to write to the beginning of the array in every dimension.
        This function is intended to write entire keys in one go, and should not be used for modifying
        existing keys.

        WARNING! It is the developer's responsibility to make sure the array shape matches the extent
        specified in the property definition. This method uses kim_property.kim_property_modify, and
        fills the values of array keys as slices through the last dimension. If those slices are incomplete,
        kim_property automatically initializes the other elements in that slice to zero. For example,
        consider writing coordinates to a key with extent [":",3]. The correct way to write a single atom
        would be to provide [[x,y,z]]. If you accidentally provide [[x],[y],[z]], it will fill the
        field with the coordinates [[x,0,0],[y,0,0],[z,0,0]]. This will not raise an error, only exceeding
        the allowed dimesnsions of the key will do so.

        Args:
            name:
                Name of the key, e.g. "cell-cauchy-stress"
            value:
                The value of the key. The function will attempt to convert it to a NumPy array, then
                use the dimensions of the resulting array. Scalars, lists, tuples, and arrays should work.
                Data type of the elements should be str, float, or int
            units:
                The units
        """
        value_arr = np.array(value)
        value_shape = value_arr.shape
        current_instance_index = len(kim_edn.loads(self.property_instances))
        modify_args = ["key", name]
        if len(value_shape) == 0:
            modify_args += ["source-value", value]
        else:
            def recur_dimensions(prev_indices: List[int], sub_value: np.ndarray, modify_args: list):
                sub_shape = sub_value.shape
                assert len(sub_shape) != 0, "Should not have gotten to zero dimensions in the recursive function"
                if len(sub_shape) == 1:
                    # only if we have gotten to a 1-dimensional sub-array do we write stuff
                    modify_args += ["source-value", *prev_indices, "1:%d" % sub_shape[0], *sub_value]
                else:
                    for i in range(sub_shape[0]):
                        prev_indices.append(i + 1)  # convert to 1-based indices
                        recur_dimensions(prev_indices, sub_value[i], modify_args)
                        prev_indices.pop()

            prev_indices = []
            recur_dimensions(prev_indices, value_arr, modify_args)

        if units is not None:
            modify_args += ["source-unit", units]

        self.property_instances = kim_property_modify(self.property_instances, current_instance_index, *modify_args)

################################################################################
class CrystalGenomeTest(KIMTest):
    """
    A Crystal Genome KIM test

    Attributes:
        stoichiometric_species: List[str]
            List of unique species in the crystal
        prototype_label: str
            AFLOW prototype label for the crystal
        parameter_names: Union[List[str],None]
            Names of free parameters of the crystal besides 'a'. May be None if the crystal is cubic with no internal DOF.
            Should have length one less than `parameter_values_angstrom`
        parameter_values_angstrom: List[float]
            List of lists of parameter values, one inner list for each equilibrium crystal structure this test will use.
            The first element in each inner list is the 'a' lattice parameter in angstrom, the rest (if present) are
            in degrees or unitless
        library_prototype_label: List[Union[str,None]]
            List of AFLOW library prototype labels, one for each equilibrium. Entries may be `None`. 
        short_name: List[Union[List[str],None]]
            Lists of human-readable short names (e.g. "FCC") for each equilibrium, if present
        cell_cauchy_stress_eV_angstrom3: List[float]
            Cauchy stress on the cell in eV/angstrom^3 (ASE units) in [xx,yy,zz,yz,xz,xy] format
        temperature_K: float
            The temperature in Kelvin
    """

    def __init__(self,
                 model_name: Optional[str] = None, 
                 model: Optional[Calculator] = None,
                 atoms: Optional[Union[List[Atoms],Atoms]] = None,
                 filename: str = "output/results.edn",
                 stoichiometric_species: Optional[List[str]] = None,
                 prototype_label: Optional[str] = None,
                 parameter_values_angstrom: Optional[Union[List[List[float]],List[float]]] = None,
                 cell_cauchy_stress_eV_angstrom3: List[float] = [0,0,0,0,0,0],
                 temperature_K: float = 0
                 ):
        """
        Args:
            model_name:
                KIM model name to use for calculations
            model:
                ASE calculator to use for calculations     
            atoms:
                List of ASE atoms objects to use as the initial configurations or to build supercells.  (NOT YET IMPLEMENTED)
                If a single atoms object is provided, it will be converted to a single-element list
            filename:
                Path to results.edn file to be written. The default provided is the correct path to work in the KIM Pipeline or KIM Developer Platform
            stoichiometric_species:
                List of unique species in the crystal
            prototype_label:
                AFLOW prototype label for the crystal
            parameter_values_angstrom:
                List of lists of AFLOW prototype parameters for the crystal.
                a (first element, always present) is in angstroms, while the other parameters 
                (present for crystals with more than 1 DOF) are in degrees or unitless. 
                If the provided list is not nested, it will be converted to a 
                If this is omitted, the parameters will be queried for
            cell_cauchy_stress_eV_angstrom3:
                Cauchy stress on the cell in eV/angstrom^3 (ASE units) in [xx,yy,zz,yz,xz,xy] format
            temperature_K:
                The temperature in Kelvin
        """        
        self.stoichiometric_species = stoichiometric_species
        self.prototype_label = prototype_label
        self.cell_cauchy_stress_eV_angstrom3 = cell_cauchy_stress_eV_angstrom3
        self.temperature_K = temperature_K

        super().__init__(model_name,model,atoms,filename)

        # Don't raise errors just because you are getting unexpected combinations of inputs, who knows what a developer might want to do?       
        if atoms is not None: 
            self._update_aflow_designation_from_atoms()     
        elif (stoichiometric_species is not None) and (prototype_label is not None):
            # only run this code if atoms is None, so we don't overwrite an existing atoms object
            if (parameter_values_angstrom is None) and (model_name is not None):
                self._query_aflow_designation_from_label_and_species()
            else:                
                if not isinstance(parameter_values_angstrom[0],list):
                    self.parameter_values_angstrom = [parameter_values_angstrom]
                else:
                    self.parameter_values_angstrom = parameter_values_angstrom                    
                # For now, if this constructor is called to build atoms from a fully provided AFLOW designation, don't assign library prototypes to it
                # TODO: Think about how to handle this
                self.library_prototype_label = [None]*len(self.parameter_values_angstrom)
                self.short_name = [None]*len(self.parameter_values_angstrom)
                self.parameter_names = ["dummy"]*(len(self.parameter_values_angstrom[0])-1) 
                # TODO: Get the list of parameter names from prototype (preferably without re-analyzing atoms object)
            aflow = aflow_util.AFLOW()
            self.atoms = []
            for parameter_values_set_angstrom in self.parameter_values_angstrom:
                self.atoms.append(aflow.build_atoms_from_prototype(stoichiometric_species,prototype_label,parameter_values_set_angstrom))
        
    def _query_aflow_designation_from_label_and_species(self):
        """
        Query for all equilibrium parameter sets for this prototype label and species in the KIM database
        """
        # TODO: Some kind of generalized query interface for all tests, this is very hand-made
        cell_cauchy_stress_Pa = [component*1.6021766e+11 for component in self.cell_cauchy_stress_eV_angstrom3]
        query_result=raw_query(
            query={
                "meta.type":"tr",
                "property-id":"tag:staff@noreply.openkim.org,2023-02-21:property/crystal-structure-npt",
                "meta.subject.extended-id":self.model_name,
                "stoichiometric-species.source-value":{
                    "$size":len(self.stoichiometric_species),
                    "$all":self.stoichiometric_species
                },
                "prototype-label.source-value":self.prototype_label,
                "cell-cauchy-stress.si-value":cell_cauchy_stress_Pa,
                "temperature.si-value":self.temperature_K
            },
            fields={
                "a.si-value":1,
                "parameter-names.source-value":1,
                "parameter-values.source-value":1,
                "library-prototype-label.source-value":1,
                "short-name.source-value":1,
                },
            database="data", limit=0, flat='on') # can't use project because parameter-values won't always exist
        if "parameter-names.source-value" in query_result[0]:
            self.parameter_names = query_result[0]["parameter-names.source-value"] # determined by prototype-label, same for all crystals
        else:
            self.parameter_names = None

        self.parameter_values_angstrom = []
        self.library_prototype_label = []
        self.short_name = []

        for parameter_set in query_result:
            self.parameter_values_angstrom.append([parameter_set["a.si-value"]*1e10])
            if "parameter-values.source-value" in parameter_set: # has params other than a
                self.parameter_values_angstrom[-1] += parameter_set["parameter-values.source-value"]
            if "library-prototype-label.source-value" in parameter_set:
                self.library_prototype_label.append(parameter_set["library-prototype-label.source-value"])
            else:
                self.library_prototype_label.append(None)
            if "short-name.source-value" in parameter_set:
                short_name = parameter_set["library-prototype-label.source-value"]
                if not isinstance(short_name,list): # Necessary because we recently changed the property definition to be a list
                    short_name = [short_name]
                self.short_name.append(short_name)
            else:
                self.short_name.append(None)            

    def _update_aflow_designation_from_atoms(self):
        """
        Update all Crystal Genome crystal description fields from the self.atoms objects. 
        """
        aflow = aflow_util.AFLOW()
        self.parameter_values_angstrom = []
        self.library_prototype_label = []
        self.short_name = []
        for i,atoms in enumerate(self.atoms):
            with NamedTemporaryFile('w',delete=False) as fp: #KDP has python3.8 which is missing the convenient `delete_on_close` option
                atoms.write(fp,format='vasp')
                fp.close()
                with open(fp.name) as f:
                    proto_des = aflow.get_prototype(f.name)
                    libproto,short_name = aflow.get_library_prototype_label_and_shortname(f.name,aflow_util.read_shortnames())
                os.remove(fp.name)

            self.parameter_values_angstrom.append(proto_des["aflow_prototype_params_values"])
            self.library_prototype_label.append(libproto)
            if short_name is None:
                self.short_name.append(None)
            else:
                self.short_name.append([short_name])

            if i == 0:
                self.prototype_label = proto_des["aflow_prototype_label"]
                parameter_names = proto_des["aflow_prototype_params_list"][1:]
                if len(parameter_names) > 1:
                    self.parameter_names = parameter_names
                else:
                    self.parameter_names = None
                self.stoichiometric_species = sorted(list(set(atoms.get_chemical_symbols())))
            else:
                if proto_des["aflow_prototype_label"] != self.prototype_label:
                    raise KIMASEError("It appears that the symmetry (i.e. AFLOW prototype label) of each provided atoms object is not the same!")
                if sorted(list(set(atoms.get_chemical_symbols()))) != self.stoichiometric_species:
                    raise KIMASEError("It appears that the set of species in each provided atoms object is not the same!")

    def _add_common_crystal_genome_keys_to_current_property_instance(self, structure_index: int, write_stress: bool = False, write_temp: bool = False):
        """
        Write common Crystal Genome keys -- prototype description and, optionally, stress and temperature

        Args:
            structure_index:
                Crystal Genome tests may take multiple equilibrium crystal structures of a shared prototype label and species,
                resulting in multiple property instances of the same property(s) possibly being written. This indicates which
                one is being written.
            write_stress:
                Write the `cell-cauchy-stress` key
            write_temp:
                Write the `temperature` key
        """
        self._add_key_to_current_property_instance("prototype-label",self.prototype_label)
        self._add_key_to_current_property_instance("stoichiometric-species",self.stoichiometric_species)
        self._add_key_to_current_property_instance("a",self.parameter_values_angstrom[structure_index][0],"angstrom")
        if self.parameter_names is not None:            
            self._add_key_to_current_property_instance("parameter-names",self.parameter_names)
            self._add_key_to_current_property_instance("parameter-values",self.parameter_values_angstrom[structure_index][1:])
        if self.library_prototype_label[structure_index] is not None:
            self._add_key_to_current_property_instance("library-prototype-label",self.library_prototype_label[structure_index])
        if self.short_name[structure_index] is not None:
            self._add_key_to_current_property_instance("short-name",self.short_name[structure_index])
        
        if write_stress:
            self._add_key_to_current_property_instance("cell-cauchy-stress",self.cell_cauchy_stress_eV_angstrom3,"eV/angstrom^3")
        if write_temp:
            self._add_key_to_current_property_instance("temperature",self.temperature_K,"K")
        
# If called directly, do nothing
if __name__ == "__main__":
    pass
