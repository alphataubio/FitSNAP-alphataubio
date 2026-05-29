from fitsnap3lib.scrapers.scrape import Scraper
import logging
import numpy as np

try:
  from adios2 import Adios, FileReader
  HAS_ADIOS2 = True
except ImportError:
  HAS_ADIOS2 = False
  Adios = FileReader = None


def _read_rra(s, name, start, count, dtype):
  """
  Read a subset in ReadRandomAccess mode using ``read_in_buffer`` (fixed buffer size).

  Avoids ``Stream.read`` / ``_read_var`` which can infer a wrong step extent and allocate
  the full first dimension for large variables (e.g. ``PositionsFlat``).
  """
  v = s.inquire_variable(name)
  if not v:
    raise KeyError(f"Variable {name!r} not found in ADIOS2 file")
  ss = int(v.steps_start())
  nst = int(v.steps())
  if nst < 1:
    raise RuntimeError(f"ADIOS2 variable {name!r} reports steps()={nst}")
  use = min(1, nst)
  start = [int(x) for x in start]
  count = [int(x) for x in count]
  if not start or not count:
    raise ValueError(f"{name!r}: non-empty start and count required for subset read")
  shape = tuple(count)
  buf = np.empty(shape, dtype=np.dtype(dtype))
  s.read_in_buffer(name, buf, start=start, count=count, step_selection=[ss, use])
  return buf


# -----------------------------------------------------------------------------

class ADIOS2(Scraper):
  """
  ADIOS2 scraper for reading .bp files created by fairchem_to_adios2.py.

  MPI distribution uses the usual contiguous split of global config indices
  ``[0, nconfigs)``. ``[SCRAPER] max_configs_per_rank`` caps how many configs each rank
  **loads** from disk (arrays and ``self.data`` stay sized to that cap).

  Array I/O uses explicit ``start`` / ``count`` subsets—never whole-file reads of
  million-row variables.

  File attributes include ``element_map``: comma-separated symbols (e.g. ``H,C,N,O``) whose
  order defines integer indices stored in ``AtomTypesFlat`` for each atom.

  FAIRChem OMOL-style datasets may use left-handed unit cells; the scraper reflects z
  consistently on lattice, positions, forces, and stress so FitSNAP's LAMMPS path sees a
  right-handed ``QMLattice``.

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

    if not hasattr(self.config.sections["PATH"], 'datapath'):
      raise ValueError("ADIOS2 scraper requires 'dataPath' parameter in [PATH] section")

    self.dataPath = self.config.sections["PATH"].datapath

    if self.pt.stubs == 0:
      self.comm = pt._comm
      self.rank = pt.get_rank()
      self.size = pt.get_size()
    else:
      self.rank = 0
      self.size = 1

    self.use_stress = self.config.sections["CALCULATOR"].stress if hasattr(self.config.sections["CALCULATOR"], 'stress') else False
    self.use_forces = self.config.sections["CALCULATOR"].force if hasattr(self.config.sections["CALCULATOR"], 'force') else True

    self.all_nconfigs = 0
    self.element_map = []
    self.has_forces = False
    self.has_stress = False
    self.has_charge = False
    self.has_nbo_charges = False
    self.has_spin = False
    self.has_composition = False
    self.unique_group_names = []

    self.num_atoms = None
    self.energy = None
    self.test_bool = None
    self.group_indices = None
    self.position_offsets = None
    self.positions_flat = None
    self.atom_types_flat = None
    self.lattices = None
    self.forces_flat = None
    self.stresses = None
    self.charge = None
    self.nbo_charges_flat = None

    self.my_config_indices = []
    self._cfg_start = 0
    self._cfg_end = 0
    self._n_load = 0
    self._po_base = 0
    self._flat_lo = 0
    self._total_atoms_file = 0

  # --------------------------------------------------------------------------------------------

  def _open_rra_collective(self):
    """MPI-aware RRA open (``FileReader(io, path, comm)`` passes ``comm`` into ``IO.open``)."""
    if self.pt.stubs == 0:
      adios = Adios(self.comm)
      io = adios.declare_io("fitsnap_adios2_rra")
      return FileReader(io, self.dataPath, self.comm)
    return FileReader(self.dataPath, None)

  # --------------------------------------------------------------------------------------------

  def scrape_groups(self, group_names=None):
    """Read global metadata on rank 0 and broadcast; set per-rank config index range and I/O cap."""

    try:
      if self.rank == 0:
        with FileReader(self.dataPath, None) as s:
          raw_nc = s.read_attribute('nconfigs')
          self.all_nconfigs = int(np.asarray(raw_nc).reshape(-1)[0])
          element_map_str = s.read_attribute('element_map')
          self.element_map = [x.strip('"\' ') for x in str(element_map_str).split(',') if x.strip()]
          if not self.element_map:
            raise ValueError('element_map is empty after parsing')

          self.has_forces = bool(int(np.asarray(s.read_attribute('has_forces')).reshape(-1)[0]))
          self.has_stress = bool(int(np.asarray(s.read_attribute('has_stress')).reshape(-1)[0]))
          available_attributes = s.available_attributes()
          if 'has_charge' in available_attributes:
            self.has_charge = bool(int(np.asarray(s.read_attribute('has_charge')).reshape(-1)[0]))
          else:
            self.has_charge = False
          if 'has_nbo_charges' in available_attributes:
            self.has_nbo_charges = bool(int(np.asarray(s.read_attribute('has_nbo_charges')).reshape(-1)[0]))
          else:
            self.has_nbo_charges = False

          ug_str = str(s.read_attribute('unique_group_names')).strip('"\'')
          self.unique_group_names = [p for p in ug_str.split('|') if p]
          if not self.unique_group_names:
            raise ValueError('unique_group_names is empty after parsing')

        self.pt.single_print(
          f"----------------------------------------------------------------\n"
          f"  ADIOS2 SCRAPER                                                \n"
          f"                                                                \n"
          f"    {self.dataPath}                                             \n"
          f"    {self.all_nconfigs} configurations with [{' '.join(self.element_map)}]\n"
          f"    forces {self.has_forces}, stress {self.has_stress}, charge {self.has_charge}\n"
        )

    except Exception as e:
      raise RuntimeError(f"Failed to read ADIOS2 file metadata: {e}")

    if self.pt.stubs == 0:
      self.all_nconfigs = self.comm.bcast(self.all_nconfigs, root=0)
      self.element_map = self.comm.bcast(self.element_map, root=0)
      self.has_forces = self.comm.bcast(self.has_forces, root=0)
      self.has_stress = self.comm.bcast(self.has_stress, root=0)
      self.has_charge = self.comm.bcast(self.has_charge, root=0)
      self.has_nbo_charges = self.comm.bcast(self.has_nbo_charges, root=0)
      self.unique_group_names = self.comm.bcast(self.unique_group_names, root=0)

    q, r = self.all_nconfigs // self.size, self.all_nconfigs % self.size
    self._cfg_start = self.rank * q + min(self.rank, r)
    self._cfg_end = self._cfg_start + q + (1 if self.rank < r else 0)
    n_local = max(0, self._cfg_end - self._cfg_start)

    max_cap = self.config.sections["SCRAPER"].max_configs_per_rank
    if max_cap is not None: self._n_load = min(n_local, int(max_cap))
    else: self._n_load = n_local
    self.my_config_indices = list(range(self._cfg_start, self._cfg_end))

    group_dict = {k: self.config.sections["GROUPS"].group_types[i]
                  for i, k in enumerate(self.config.sections["GROUPS"].group_sections)}
    self.group_table = self.config.sections["GROUPS"].group_table

    for group_name in self.unique_group_names:
      if group_name not in self.group_table:
        self.group_table[group_name] = {'eweight': 1.0, 'fweight': 100.0, 'vweight': 1e-12}
      self.group_table[group_name]['training_size'] = 0
      self.group_table[group_name]['testing_size'] = 0

  # --------------------------------------------------------------------------------------------

  def divvy_up_configs(self):
    self.my_configs = self.my_config_indices

  # --------------------------------------------------------------------------------------------

  def _merge_group_train_test_counts(self, gi_local, tb_local):
    """Reconstruct global train/test counts per group from this rank's ``GroupIndices`` slice."""
    n_g = len(self.unique_group_names)
    train_ct = np.zeros(n_g, dtype=np.int64)
    test_ct = np.zeros(n_g, dtype=np.int64)
    for i in range(len(gi_local)):
      g = int(gi_local[i])
      if g < 0 or g >= n_g: continue
      if int(tb_local[i]) != 0: test_ct[g] += 1
      else: train_ct[g] += 1
    if self.pt.stubs == 0:
      from mpi4py import MPI
      train_ct = self.comm.allreduce(train_ct, op=MPI.SUM)
      test_ct = self.comm.allreduce(test_ct, op=MPI.SUM)
    for gi, gname in enumerate(self.unique_group_names):
      if gname not in self.group_table:
        self.group_table[gname] = {'eweight': 1.0, 'fweight': 100.0, 'vweight': 1e-12}
      self.group_table[gname]['training_size'] = int(train_ct[gi])
      self.group_table[gname]['testing_size'] = int(test_ct[gi])

  # --------------------------------------------------------------------------------------------

  def scrape_configs(self):
    """Read only this rank's config subset (``start``/``count``); build ``self.data``."""
    self.data = []

    c0 = self._cfg_start
    n_load = self._n_load
    c1 = c0 + n_load
    n_local = max(0, self._cfg_end - self._cfg_start)
    nc = int(self.all_nconfigs)

    try:
      with self._open_rra_collective() as s:

        if nc > 0:
          po_last = _read_rra(s, 'PositionOffsets', [nc - 1], [1], np.int64)
          na_last = _read_rra(s, 'NumAtoms', [nc - 1], [1], np.uint16)
          self._total_atoms_file = int(po_last[0]) + int(na_last[0])
        else:
          self._total_atoms_file = 0

        if n_local > 0:
          gi_hist = _read_rra(s, 'GroupIndices', [c0], [n_local], np.int32)
          tb_hist = _read_rra(s, 'test_bool', [c0], [n_local], np.int32)
        else:
          gi_hist = np.array([], dtype=np.int32)
          tb_hist = np.array([], dtype=np.int32)

        self._merge_group_train_test_counts(gi_hist, tb_hist)

        if n_load == 0:
          self.position_offsets = np.array([], dtype=np.int64)
          self._po_base = c0
          self.group_indices = np.array([], dtype=np.int32)
          self.test_bool = np.array([], dtype=bool)
          self.num_atoms = np.array([], dtype=np.int64)
          self.energy = np.array([], dtype=np.float64)
          self.positions_flat = np.zeros((0, 3), dtype=np.float64)
          self.atom_types_flat = np.array([], dtype=np.int32)
          self.lattices = np.zeros((0, 3, 3), dtype=np.float64)
          self.forces_flat = np.zeros((0, 3), dtype=np.float64) if self.has_forces else None
          self.stresses = np.zeros((0, 3, 3), dtype=np.float64) if self.has_stress else None
          self.charge = np.array([], dtype=np.int64) if self.has_charge else None
          self.nbo_charges_flat = np.array([], dtype=np.float64) if self.has_nbo_charges else None
          self._flat_lo = 0
        else:
          self.group_indices = np.asarray(gi_hist[:n_load], dtype=np.int32, order='C')
          self.test_bool = np.asarray(tb_hist[:n_load], dtype=bool)

          po_count = (n_load + 1) if c1 < nc else n_load
          self.position_offsets = _read_rra(s, 'PositionOffsets', [c0], [po_count], np.int64)
          self._po_base = c0

          self._flat_lo = int(self.position_offsets[0])
          if c1 < nc:
            flat_hi = int(self.position_offsets[n_load])
          else:
            flat_hi = self._total_atoms_file
          n_atom_rows = flat_hi - self._flat_lo
          if n_atom_rows < 0:
            raise RuntimeError("Invalid PositionOffsets slice (flat_hi < flat_lo)")

          self.num_atoms = _read_rra(s, 'NumAtoms', [c0], [n_load], np.uint16)
          self.energy = _read_rra(s, 'Energy', [c0], [n_load], np.float64)
          self.lattices = _read_rra(s, 'Lattice', [c0, 0, 0], [n_load, 3, 3], np.float64)

          self.positions_flat = _read_rra(
            s, 'PositionsFlat', [self._flat_lo, 0], [n_atom_rows, 3], np.float64
          )
          self.atom_types_flat = _read_rra(s, 'AtomTypesFlat', [self._flat_lo], [n_atom_rows], np.int32)

          if self.has_forces:
            self.forces_flat = _read_rra(
              s, 'ForcesFlat', [self._flat_lo, 0], [n_atom_rows, 3], np.float64
            )
          if self.has_stress:
            self.stresses = _read_rra(s, 'Stress', [c0, 0, 0], [n_load, 3, 3], np.float64)
          if self.has_charge:
            self.charge = _read_rra(s, 'Charge', [c0], [n_load], np.int8)
          if self.has_nbo_charges:
            self.nbo_charges_flat = _read_rra(
              s, 'NBOChargesFlat', [self._flat_lo], [n_atom_rows], np.float64
            )

    except Exception as e:
      raise RuntimeError(f"Failed to read ADIOS2 data arrays: {e}")

    sorted_group_names = sorted(self.unique_group_names)
    self.pt.add_2_fitsnap("sorted_group_names", sorted_group_names)

    if self.rank == 0:
      max_len = max(len(s) for s in sorted_group_names)
      total_train = total_test = 0
      self.pt.single_print(f"    {'GROUP':<{max_len}}  TRAINING  VALIDATION")
      for group_name in sorted_group_names:
        train_size = self.group_table[group_name]['training_size']
        test_size = self.group_table[group_name]['testing_size']
        total_train += train_size
        total_test += test_size
        self.pt.single_print(f"    {group_name:<{max_len}}  {train_size:>8}    {test_size:>8}")
      self.pt.single_print(f"    {'TOTAL':<{max_len}}  {total_train:>8}    {total_test:>8}\n")

    for config_idx in range(c0, c1):
      if self.has_charge:
        lk = config_idx - c0
        if int(self.charge[lk]) != 0: continue
      try:
        data_dict = self._extract_config(config_idx)
        if data_dict is not None and np.any(data_dict.get('Charges', 1)): self.data.append(data_dict)
      except Exception as e:
        logging.warning(f"Failed to extract config {config_idx}: {e}")
        continue

    # Check for auto_eshift flag
    if "SCRAPER" in self.config.sections and self.config.sections["SCRAPER"].auto_eshift:
      self.auto_eshift()

    n_loc = len(self.data)
    if self.pt.stubs == 0:
      from mpi4py import MPI
      n_tot = int(self.comm.allreduce(n_loc, op=MPI.SUM))
    else: n_tot = n_loc
    self.pt.add_2_fitsnap("nconfigs", n_tot)

    self.pt.single_print(f"----------------------------------------------------------------\n")

    return self.data

  # --------------------------------------------------------------------------------------------

  def auto_eshift(self):

    # Add the Global Auto ESHIFT Reduction

    num_elements = len(self.element_map)
    element_to_idx = {el: i for i, el in enumerate(self.element_map)}
      
    # Prepare local matrices for Normal Equations (AtA * x = Atb)
    AtA_local = np.zeros((num_elements, num_elements), dtype=np.float64)
    Atb_local = np.zeros(num_elements, dtype=np.float64)
      
    for config in self.data:
      counts = np.zeros(num_elements, dtype=np.float64)
      for atom in config['AtomTypes']: counts[element_to_idx[atom]] += 1.0
      energy = config['Energy']
      # Accumulate AtA and Atb locally
      AtA_local += np.outer(counts, counts)
      Atb_local += counts * energy
        
    # Prepare global matrices
    AtA_global = np.zeros_like(AtA_local)
    Atb_global = np.zeros_like(Atb_local)
      
    # Sum across all MPI ranks
    if self.pt.stubs == 0:
      from mpi4py import MPI
      self.comm.Allreduce(AtA_local, AtA_global, op=MPI.SUM)
      self.comm.Allreduce(Atb_local, Atb_global, op=MPI.SUM)
    else:
      AtA_global = AtA_local
      Atb_global = Atb_local
        
    # Solve the linear system
    try:
      x = np.linalg.solve(AtA_global, Atb_global)
    except np.linalg.LinAlgError:
      # Fallback if matrix is singular
      x, _, _, _ = np.linalg.lstsq(AtA_global, Atb_global, rcond=None)
        
    eshift_dict = {el: shift for el, shift in zip(self.element_map, x)}
      
    # Print globally exact shifts from Rank 0
    if self.rank == 0:
      self.pt.single_print("    AUTO_ESHIFT")
      for el, shift in eshift_dict.items(): self.pt.single_print(f"    {el:4s} {shift:>.6f} eV")
      self.pt.single_print(f"")

    # Apply exact shifts to local configurations
    for config in self.data:
      shift_sum = sum(eshift_dict[atom] for atom in config['AtomTypes'])
      config['Energy'] -= shift_sum

    if "PYACE" in self.config.sections:
      pyace_section = self.config.sections["PYACE"]
      eshift_values = [float(v) for v in eshift_dict.values()]
      pyace_section.bbasis.E0vals = eshift_values
      pyace_section.ctilde_basis.E0vals = eshift_values
      pyace_section.create_coupling_coefficients_yace()


  # --------------------------------------------------------------------------------------------

  def _extract_config(self, config_idx):
    """Extract one config; arrays on ``self`` are the rank-local ``n_load`` slice."""
    local_k = config_idx - self._po_base
    if local_k < 0 or local_k >= len(self.group_indices):
      raise IndexError(f"config_idx {config_idx} outside loaded slice [{self._po_base}, {self._po_base + len(self.group_indices)})")

    group_idx = int(self.group_indices[local_k])
    group_name = self.unique_group_names[group_idx]
    natoms = int(self.num_atoms[local_k])

    pos_start = int(self.position_offsets[local_k])
    if config_idx == self.all_nconfigs - 1: pos_end = self._total_atoms_file
    else: pos_end = int(self.position_offsets[local_k + 1])

    lo = self._flat_lo
    positions = self.positions_flat[pos_start - lo : pos_end - lo]
    atom_type_indices = self.atom_types_flat[pos_start - lo : pos_end - lo]
    atom_types = [self.element_map[int(idx)] for idx in atom_type_indices]

    lattice = self.lattices[local_k].reshape((3, 3))

    data_dict = {
      'Group': group_name,
      'File': f"{group_name}/{config_idx}",
      'Positions': positions.copy(),
      'AtomTypes': atom_types,
      'NumAtoms': natoms,
      'QMLattice': lattice.T.copy(),
      'Energy': float(self.energy[local_k]) * self.default_conversions['Energy'],
      'test_bool': bool(self.test_bool[local_k]),
    }

    if "ESHIFT" in self.config.sections and hasattr(self.config.sections["ESHIFT"], 'eshift'):
      for atom in data_dict["AtomTypes"]:
        data_dict["Energy"] += self.config.sections["ESHIFT"].eshift[atom]

    if self.has_forces and self.use_forces:
      forces = self.forces_flat[pos_start - lo : pos_end - lo]
      data_dict['Forces'] = forces.copy()

    if self.has_nbo_charges:
      nbo = self.nbo_charges_flat[pos_start - lo : pos_end - lo]
      data_dict['Charges'] = np.asarray(nbo, dtype=np.float64).copy()
      #self.pt.all_print(f"*** {data_dict['File']} {data_dict['Charges']}\n")

    if self.has_stress and self.use_stress:
      stress = self.stresses[local_k].reshape((3, 3))
      data_dict['Stress'] = stress.copy() * 1.602176634e6

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

  # --------------------------------------------------------------------------------------------
