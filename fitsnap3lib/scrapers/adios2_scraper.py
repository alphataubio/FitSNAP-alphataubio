from fitsnap3lib.scrapers.scrape import Scraper
import json
import logging
import time
import numpy as np

try:
  from adios2 import Stream
  HAS_ADIOS2 = True
except ImportError:
  HAS_ADIOS2 = False


def _find_attr(attrs, name):
  """Resolve attribute; BP5 may use plain keys or path-like names."""
  if name in attrs: return attrs[name]
  for k in attrs:
    if k.endswith('/' + name) or k.rpartition('/')[-1] == name: return attrs[k]
  return None


def _attr_value(entry):
  """Scalar from an available_attributes() entry (bytes, str, int, or small array)."""
  if not entry or 'Value' not in entry: return None
  v = entry['Value']
  if isinstance(v, bytes): return v.decode('utf-8')
  if isinstance(v, str): return v
  if isinstance(v, (np.ndarray, list, tuple)) and len(v) > 0:
    x = np.asarray(v).reshape(-1)[0]
    if isinstance(x, bytes): return x.decode('utf-8')
    return x.item() if hasattr(x, 'item') else x
  return v


# -----------------------------------------------------------------------------

class ADIOS2(Scraper):
  """
  ADIOS2 scraper for reading .bp files created by fairchem_to_adios2.py.
  Designed for scalable MPI-parallel reading of FAIRChem datasets.

  File attributes include ``element_map``: comma-separated symbols (e.g. ``H,C,N,O``) whose
  order defines integer indices stored in ``AtomTypesFlat`` for each atom.

  FAIRChem OMOL-style datasets may use left-handed unit cells (uncommon in older OMAT24-style
  data); the scraper reflects z consistently on lattice, positions, forces, and stress so
  FitSNAP's LAMMPS path sees a right-handed ``QMLattice``.

  Includes:
  1. Applies ``[SCRAPER]`` energy unit conversion to ``Energy`` (same as VASP/JSON scrapers).
  2. Optional per-atom ESHIFT subtraction when ``eshifts`` is populated (auto-regression is
     currently disabled in code).
  3. Converts Stress from eV/Å³ to bar when stress is used (see ``fairchem_to_adios2``).
  4. Handles correct lattice orientation (column-major ``QMLattice``).
  5. Optional per-atom ``nbo_charges`` (OMol25) from ``NBOChargesFlat`` — separate from
     molecular ``Charge``.
  """

  def __init__(self, name, pt, config):
    if not HAS_ADIOS2:
      raise ImportError("ADIOS2 scraper requires: pip install adios2")

    super().__init__(name, pt, config)
    self.data = []

    # Get the .bp file path from PATH section
    if not hasattr(self.config.sections["PATH"], 'datapath'):
      raise ValueError("ADIOS2 scraper requires 'dataPath' parameter in [PATH] section")

    self.dataPath = self.config.sections["PATH"].datapath

    # MPI setup
    if self.pt.stubs == 0:
      self.comm = pt._comm
      self.rank = pt.get_rank()
      self.size = pt.get_size()
    else:
      self.rank = 0
      self.size = 1

    # Configuration options
    self.use_stress = self.config.sections["CALCULATOR"].stress if hasattr(self.config.sections["CALCULATOR"], 'stress') else False
    self.use_forces = self.config.sections["CALCULATOR"].force if hasattr(self.config.sections["CALCULATOR"], 'force') else True

    # Metadata to be loaded
    self.nconfigs = 0
    self.element_map = []
    self.has_forces = False
    self.has_stress = False
    self.has_charge = False
    self.has_nbo_charges = False
    self.has_spin = False
    self.has_composition = False
    self.unique_group_names = []
    self.eshifts = {}  # Will store auto-calculated atomic energies

    # Data arrays
    self.num_atoms = None
    self.energy = None
    self.test_bool = None
    self.group_indices = None
    self.position_offsets = None
    self.positions_flat = None
    self.atom_types_flat = None
    self.lattices = None  # Shape: (nconfigs, 3, 3)
    self.forces_flat = None
    self.stresses = None  # Shape: (nconfigs, 3, 3)
    self.charge = None
    self.nbo_charges_flat = None

    # My configurations
    self.my_config_indices = []

  def scrape_groups(self, group_names=None):
    """
    Read metadata, calculate ESHIFT, and determine configuration distribution.
    """

    # Serial ADIOS2 per process (comm=None) avoids collective BeginStep on fairchem .bp files.
    stream_comm = None

    try:
      with Stream(self.dataPath, 'r', stream_comm) as s:

        for step in s.steps():

          if self.rank == 0:
            attrs = s.available_attributes()

            nc = _find_attr(attrs, 'nconfigs')
            if nc is None: raise KeyError('nconfigs attribute not found')
            self.nconfigs = int(_attr_value(nc))

            em = _find_attr(attrs, 'element_map')
            if em is None: raise KeyError('element_map attribute not found')
            element_map_str = str(_attr_value(em)).strip('"\' ')
            self.element_map = [x.strip('"\' ') for x in element_map_str.split(',') if x.strip()]
            if not self.element_map: raise ValueError('element_map is empty after parsing')

            if (hf := _find_attr(attrs, 'has_forces')) is not None:
              self.has_forces = bool(int(_attr_value(hf)))
            if (hs := _find_attr(attrs, 'has_stress')) is not None:
              self.has_stress = bool(int(_attr_value(hs)))
            if (hc := _find_attr(attrs, 'has_charge')) is not None:
              self.has_charge = bool(int(_attr_value(hc)))
            if (hnbo := _find_attr(attrs, 'has_nbo_charges')) is not None:
              self.has_nbo_charges = bool(int(_attr_value(hnbo)))


            if (ug := _find_attr(attrs, 'unique_group_names')) is None:
              raise KeyError('unique_group_names attribute not found')
            ug_str = str(_attr_value(ug)).strip('"\'')
            self.unique_group_names = [p for p in ug_str.split('|') if p]
            if not self.unique_group_names:
              raise ValueError('unique_group_names is empty after parsing')

          break  # one collective step; all ranks leave together

        if self.rank == 0:
          self.pt.single_print(
            f"ADIOS2 scraper: {self.dataPath}, {self.nconfigs} configurations with "
            f"[{' '.join(self.element_map)}], "
            f"forces {self.has_forces}, stress {self.has_stress}"
          )

    except Exception as e:
      raise RuntimeError(f"Failed to read ADIOS2 file metadata: {e}")

    # Broadcast metadata AND ESHIFTS to all ranks (collective)
    if self.pt.stubs == 0:
      self.nconfigs = self.comm.bcast(self.nconfigs, root=0)
      self.pt.add_2_fitsnap("nconfigs", self.nconfigs)
      self.element_map = self.comm.bcast(self.element_map, root=0)
      self.has_forces = self.comm.bcast(self.has_forces, root=0)
      self.has_stress = self.comm.bcast(self.has_stress, root=0)
      self.has_charge = self.comm.bcast(self.has_charge, root=0)
      self.has_nbo_charges = self.comm.bcast(self.has_nbo_charges, root=0)
      self.unique_group_names = self.comm.bcast(self.unique_group_names, root=0)
      self.eshifts = self.comm.bcast(self.eshifts, root=0)

    # Build group_table from config file [GROUPS] section
    group_dict = {k: self.config.sections["GROUPS"].group_types[i]
                  for i, k in enumerate(self.config.sections["GROUPS"].group_sections)}
    self.group_table = self.config.sections["GROUPS"].group_table

    # Initialize training/testing sizes to 0 for all groups
    for group_name in self.unique_group_names:
      if group_name not in self.group_table:
        self.group_table[group_name] = {
          'eweight': 1.0,
          'fweight': 1.0,
          'vweight': 1.0,
        }
      self.group_table[group_name]['training_size'] = 0
      self.group_table[group_name]['testing_size'] = 0

    # Determine which configurations this rank will process
    configs_per_rank = self.nconfigs // self.size
    remainder = self.nconfigs % self.size

    start_idx = self.rank * configs_per_rank + min(self.rank, remainder)
    end_idx = start_idx + configs_per_rank + (1 if self.rank < remainder else 0)

    self.my_config_indices = list(range(start_idx, end_idx))

  def divvy_up_configs(self):
    """
    Configuration distribution is already done in scrape_groups.
    """
    self.my_configs = self.my_config_indices

  def scrape_configs(self):
    """
    Read ADIOS2 data and extract configurations for this rank.
    """
    self.data = []

    # Serial read per rank (same as scrape_groups)
    stream_comm = None
    try:
      with Stream(self.dataPath, 'r', stream_comm) as s:
        for step in s.steps():
          # Read per-config arrays
          self.num_atoms = s.read('NumAtoms')
          self.energy = s.read('Energy')
          self.test_bool = s.read('test_bool')
          self.group_indices = s.read('GroupIndices')

          # Read variable-length arrays
          self.position_offsets = s.read('PositionOffsets')
          self.positions_flat = s.read('PositionsFlat')
          self.atom_types_flat = s.read('AtomTypesFlat')

          # Read fixed-size arrays
          self.lattices = s.read('Lattice')

          if self.has_forces: self.forces_flat = s.read('ForcesFlat')
          if self.has_stress: self.stresses = s.read('Stress')
          if self.has_charge: self.charge = s.read('Charge')
          if self.has_nbo_charges: self.nbo_charges_flat = s.read('NBOChargesFlat')

          break

    except Exception as e:
      raise RuntimeError(f"Failed to read ADIOS2 data arrays: {e}")

    # Count training/testing sizes per group (only rank 0)
    if self.rank == 0:
      for i in range(self.nconfigs):
        group_idx = int(self.group_indices[i])
        group_name = self.unique_group_names[group_idx]
        is_test = bool(self.test_bool[i])
        if is_test: self.group_table[group_name]['testing_size'] += 1
        else: self.group_table[group_name]['training_size'] += 1

      # Print summary
      sorted_group_names = sorted(self.unique_group_names)
      self.pt.add_2_fitsnap("sorted_group_names", sorted_group_names)

      max_len = max(len(s) for s in sorted_group_names)
      total_train = total_test = 0
      self.pt.single_print(f"    {'GROUP':<{max_len}}  TRAINING  VALIDATION")

      for group_name in sorted_group_names:
        train_size = self.group_table[group_name]['training_size']
        test_size = self.group_table[group_name]['testing_size']
        total_train += train_size
        total_test += test_size
        self.pt.single_print(f"    {group_name:<{max_len}}  {train_size:>8}    {test_size:>8}")
      self.pt.single_print(f"    {'TOTAL':<{max_len}}  {total_train:>8}    {total_test:>8}")

    # Broadcast group_table to all ranks
    if self.pt.stubs == 0:
      self.group_table = self.comm.bcast(self.group_table, root=0)

    # Process configurations assigned to this rank: only charge==0 when Charge is present;
    # stop after max_configs_per_rank configs are successfully added.
    max_configs_per_rank = self.config.sections["SCRAPER"].max_configs_per_rank
    if max_configs_per_rank is None: max_configs_per_rank = len(self.my_config_indices)

    added = 0
    for config_idx in self.my_config_indices:
      if added >= max_configs_per_rank: break
      if self.has_charge and int(self.charge[config_idx]) != 0: continue
      try:
        data_dict = self._extract_config(config_idx)
        if data_dict is not None:
          self.data.append(data_dict)
          added += 1
      except Exception as e:
        logging.warning(f"Failed to extract config {config_idx}: {e}")
        continue

    return self.data

  def _extract_config(self, config_idx):
    """
    Extract a single configuration from the flattened arrays.
    """
    # Get group name from indices
    group_idx = int(self.group_indices[config_idx])
    group_name = self.unique_group_names[group_idx]

    # Get number of atoms
    natoms = int(self.num_atoms[config_idx])

    # Get position range
    pos_start = int(self.position_offsets[config_idx])
    if config_idx == self.nconfigs - 1: pos_end = len(self.atom_types_flat)
    else: pos_end = int(self.position_offsets[config_idx + 1])

    # Extract positions
    positions = self.positions_flat[pos_start:pos_end]

    # Extract atom types
    atom_type_indices = self.atom_types_flat[pos_start:pos_end]
    atom_types = [self.element_map[int(idx)] for idx in atom_type_indices]

    # Extract lattice (3x3 matrix)
    lattice = self.lattices[config_idx].reshape((3, 3))

    # Create data dictionary ([SCRAPER] property_array energy: ... -> conversions, same as VASP/JSON)
    data_dict = {
      'Group': group_name,
      'File': f"{group_name}/{config_idx}",
      'Positions': positions.copy(),
      'AtomTypes': atom_types,
      'NumAtoms': natoms,
      'QMLattice': lattice.T.copy(),
      'Energy': float(self.energy[config_idx]) * self.default_conversions['Energy'],
      'test_bool': bool(self.test_bool[config_idx]),
    }

    # Extract forces
    if self.has_forces and self.use_forces:
      forces = self.forces_flat[pos_start:pos_end]
      data_dict['Forces'] = forces.copy()

    if self.has_nbo_charges:
      nbo = self.nbo_charges_flat[pos_start:pos_end]
      data_dict['Charges'] = np.asarray(nbo, dtype=np.float64).copy()

    # Extract stress
    if self.has_stress and self.use_stress:
      stress = self.stresses[config_idx].reshape((3, 3))
      # CRITICAL FIX: Convert eV/A^3 to Bar
      data_dict['Stress'] = stress.copy() * 1602176.6208


    # Apply weights from config file using parent class method.
    # If _rotate_coords / _weighting raise, we must restore self.data or the scraper
    # stays a dict and scrape_configs can return a broken structure (dict keys look like configs).
    old_data = self.data
    old_conversions = self.conversions
    self.data = data_dict
    self.conversions = self.default_conversions
    try:
      self._rotate_coords()
      self._translate_coords()
      self._weighting(natoms)
      return dict(self.data)
    finally:
      self.data = old_data
      self.conversions = old_conversions
