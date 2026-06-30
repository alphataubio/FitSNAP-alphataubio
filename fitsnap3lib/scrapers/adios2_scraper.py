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

  MPI distribution splits the global config index range ``[0, nconfigs)`` into contiguous
  segments balanced by total atom count (not config count), so per-rank work—which scales with
  atoms—is evenly distributed. ``[SCRAPER] max_configs_per_rank`` caps how many configs each
  rank **loads** from disk (arrays and ``self.data`` stay sized to that cap).

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

  def _atom_balanced_boundaries(self, position_offsets, total_atoms):
    """
    Partition global config indices ``[0, nconfigs)`` into ``self.size`` *contiguous* segments
    with (near-)equal total atom counts, for better MPI load balancing than an equal config-count
    split—per-config work (array reads, descriptor cost) scales with atoms, not config count.

    ``position_offsets`` is the per-config prefix sum of ``NumAtoms`` (FitSNAP's ``PositionOffsets``),
    so the atom count of segment ``[a, b)`` is ``cum[b] - cum[a]`` where ``cum`` is ``position_offsets``
    extended by ``total_atoms``. Each interior boundary is the first config whose cumulative atom
    count reaches that rank's even share of ``total_atoms``. Keeping segments contiguous means the
    existing ``start``/``count`` block reads (PositionsFlat, ForcesFlat, ...) stay intact.

    Returns an ``int64`` array of length ``size + 1`` with ``boundaries[0] == 0`` and
    ``boundaries[size] == nconfigs``; rank ``r`` owns configs ``[boundaries[r], boundaries[r + 1])``.
    """
    nconfigs = int(self.all_nconfigs)
    size = int(self.size)
    boundaries = np.zeros(size + 1, dtype=np.int64)
    boundaries[size] = nconfigs
    if size > 1 and nconfigs > 0 and total_atoms > 0:
      cum = np.append(np.asarray(position_offsets, dtype=np.int64), np.int64(total_atoms))
      targets = (np.arange(1, size, dtype=np.float64) * float(total_atoms)) / size
      interior = np.searchsorted(cum, targets, side='left').astype(np.int64)
      np.clip(interior, 0, nconfigs, out=interior)
      np.maximum.accumulate(interior, out=interior)  # keep boundaries non-decreasing
      boundaries[1:size] = interior
    return boundaries

  # --------------------------------------------------------------------------------------------

  def scrape_groups(self, group_names=None):
    """Read global metadata on rank 0 and broadcast; set per-rank config index range and I/O cap."""

    boundaries = None
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

          # Per-config atom counts drive an atom-balanced (not equal-count) split. We balance by
          # *training* atoms only: validation configs are weight-zeroed by the solver and charge!=0
          # configs are dropped by the scraper, so neither contributes a nonzero design row. The old
          # all-atom balance counted them, which -- because validation/charged configs cluster at the
          # end of the file -- handed the tail ranks segments that are entirely zero-weight, i.e. the
          # all-zero / rank-deficient design blocks the SLATE QR then chokes on. Feeding the training
          # prefix sum + training total to the (unchanged) contiguous boundary routine balances the
          # rows that actually enter the fit, so every rank gets full-rank training data.
          if self.all_nconfigs > 0:
            po_all = _read_rra(s, 'PositionOffsets', [0], [self.all_nconfigs], np.int64)
            na_last = _read_rra(s, 'NumAtoms', [self.all_nconfigs - 1], [1], np.uint16)
            total_atoms = int(po_all[-1]) + int(na_last[0])
            # NumAtoms per config, reconstructed from the PositionOffsets prefix sum.
            na_all = np.diff(np.append(po_all.astype(np.int64), np.int64(total_atoms)))
            tb_all = _read_rra(s, 'test_bool', [0], [self.all_nconfigs], np.int32)
            train_mask = (tb_all == 0)
            if self.has_charge:
              charge_all = _read_rra(s, 'Charge', [0], [self.all_nconfigs], np.int8)
              train_mask &= (charge_all == 0)
            train_atoms = (na_all * train_mask).astype(np.int64)
            # exclusive prefix sum of training atoms (cumulative training atoms in configs [0, i))
            train_prefix = np.concatenate(([np.int64(0)], np.cumsum(train_atoms)[:-1])).astype(np.int64)
            total_train_atoms = int(train_atoms.sum())
          else:
            po_all = np.array([], dtype=np.int64)
            total_atoms = 0
            train_prefix = np.array([], dtype=np.int64)
            total_train_atoms = 0

        # Reuse the contiguous boundary routine unchanged: it balances cum[boundaries] over a total,
        # so passing the training prefix + training total balances *training* atoms per rank.
        boundaries = self._atom_balanced_boundaries(train_prefix, total_train_atoms)
        cum_ext = np.append(train_prefix, np.int64(total_train_atoms))
        seg_atoms = cum_ext[boundaries[1:]] - cum_ext[boundaries[:-1]]

        self.pt.single_print(
          f"----------------------------------------------------------------\n"
          f"  ADIOS2 SCRAPER                                                \n"
          f"                                                                \n"
          f"    {self.dataPath}                                             \n"
          f"    {self.all_nconfigs} configurations, {total_atoms} atoms with [{' '.join(self.element_map)}]\n"
          f"    forces {self.has_forces}, stress {self.has_stress}, charge {self.has_charge}\n"
          f"    training-atom-balanced across {self.size} rank(s): {int(seg_atoms.min())}-{int(seg_atoms.max())} training atoms/rank\n"
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
      boundaries = self.comm.bcast(boundaries, root=0)

    # Contiguous, atom-balanced segment for this rank (computed on rank 0 from PositionOffsets).
    self._cfg_start = int(boundaries[self.rank])
    self._cfg_end = int(boundaries[self.rank + 1])
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

    self._compute_rdf()

    self.pt.single_print(f"----------------------------------------------------------------\n")

    return self.data

  # --------------------------------------------------------------------------------------------

  @staticmethod
  def _resolve_bond_cutoffs(bonds, elem_a, elem_b, default_rcut=6.0, default_rcut_in=0.0):
    """Look up (rcut_in, rcut) for an element pair from PYACE bonds dict."""
    if not bonds:
      return default_rcut_in, default_rcut
    for key in (
      f"{elem_a} {elem_b}", f"{elem_b} {elem_a}",
      f"{elem_a}{elem_b}", f"{elem_b}{elem_a}",
      "ALL",
    ):
      if key in bonds:
        entry = bonds[key]
        rcut = float(entry.get("rcut", default_rcut))
        rcut_in = float(entry.get("rcut_in", default_rcut_in))
        return rcut_in, rcut
    return default_rcut_in, default_rcut

  @staticmethod
  def _cell_volume_and_periodic(qm_lattice):
    """Return (volume, is_periodic) for column-vector lattice ``QMLattice``."""
    cell = np.asarray(qm_lattice, dtype=np.float64).reshape(3, 3)
    det = float(np.linalg.det(cell))
    if not np.isfinite(det) or abs(det) < 1e-12:
      return 0.0, False
    return abs(det), True

  @staticmethod
  def _expand_periodic_images(positions, qm_lattice, r_max):
    """
    Replicate atoms over periodic images within ``r_max`` of the origin cell.
    Returns (all_positions, atom_ids) where ``atom_ids`` maps each row to the
  original atom index in ``positions``.
    """
    cell = np.asarray(qm_lattice, dtype=np.float64).reshape(3, 3)
    axis_lengths = np.linalg.norm(cell, axis=0)
    nrep = [max(0, int(np.ceil(r_max / max(al, 1e-12)))) for al in axis_lengths]
    offsets = [
      (ix, iy, iz)
      for ix in range(-nrep[0], nrep[0] + 1)
      for iy in range(-nrep[1], nrep[1] + 1)
      for iz in range(-nrep[2], nrep[2] + 1)
    ]
    n_atoms = positions.shape[0]
    n_images = len(offsets)
    all_pos = np.empty((n_atoms * n_images, 3), dtype=np.float64)
    atom_ids = np.empty(n_atoms * n_images, dtype=np.int32)
    row = 0
    for ix, iy, iz in offsets:
      shift = ix * cell[:, 0] + iy * cell[:, 1] + iz * cell[:, 2]
      all_pos[row:row + n_atoms] = positions + shift
      atom_ids[row:row + n_atoms] = np.arange(n_atoms, dtype=np.int32)
      row += n_atoms
    return all_pos, atom_ids

  def _compute_rdf(self, n_bins=50):
    """Accumulate partial RDF g(r) numerators/denominators over loaded configs."""
    elements = list(self.element_map)
    n_elem = len(elements)
    if n_elem == 0 or not self.data:
      return

    element_to_idx = {el: i for i, el in enumerate(elements)}
    bonds = {}
    default_rcut = 6.0
    default_rcut_in = 0.0
    if "PYACE" in self.config.sections:
      pyace = self.config.sections["PYACE"]
      bonds = getattr(pyace, "bonds", {}) or {}
      default_rcut = float(getattr(pyace, "cutoff", default_rcut))

    rcut_in = np.zeros((n_elem, n_elem), dtype=np.float64)
    rcut = np.zeros((n_elem, n_elem), dtype=np.float64)
    for ia, ea in enumerate(elements):
      for ib, eb in enumerate(elements):
        rin, rout = self._resolve_bond_cutoffs(
          bonds, ea, eb, default_rcut=default_rcut, default_rcut_in=default_rcut_in
        )
        rcut_in[ia, ib] = rin
        rcut[ia, ib] = rout

    r_cut_max = float(np.max(rcut)) if rcut.size else default_rcut
    r_max = r_cut_max + 1.0
    if r_cut_max <= 0.0:
      return

    r_edges = np.linspace(0.0, r_max, n_bins + 1)
    counts = np.zeros((n_elem, n_elem, n_bins), dtype=np.float64)
    density = np.zeros((n_elem, n_elem), dtype=np.float64)
    min_dist = 1e-8

    for config in self.data:
      positions = np.asarray(config["Positions"], dtype=np.float64).reshape(-1, 3)
      if positions.shape[0] == 0:
        continue
      elem_idx = np.array(
        [element_to_idx[t] for t in config["AtomTypes"]], dtype=np.int32
      )
      n_per_type = np.bincount(elem_idx, minlength=n_elem).astype(np.float64)

      qm = config.get("QMLattice")
      volume, periodic = self._cell_volume_and_periodic(qm) if qm is not None else (0.0, False)
      if not periodic or volume <= 0.0:
        continue

      neigh_pos, neigh_ids = self._expand_periodic_images(positions, qm, r_max)
      for ia in range(n_elem):
        for ib in range(n_elem):
          if ia == ib:
            density[ia, ib] += n_per_type[ia] * max(n_per_type[ia] - 1.0, 0.0) / volume
          else:
            density[ia, ib] += n_per_type[ia] * n_per_type[ib] / volume

      central_elem = elem_idx
      central_ids = np.arange(positions.shape[0], dtype=np.int32)

      for ia in range(n_elem):
        c_mask = central_elem == ia
        if not np.any(c_mask):
          continue
        cpos = positions[c_mask]
        cids = central_ids[c_mask]

        for ib in range(n_elem):
          n_mask = elem_idx[neigh_ids] == ib
          if not np.any(n_mask):
            continue
          npos = neigh_pos[n_mask]
          nids = neigh_ids[n_mask]

          pair_dists = []
          for c_i, (cp, cid) in enumerate(zip(cpos, cids)):
            diff = npos - cp
            d = np.sqrt(np.sum(diff * diff, axis=1))
            if ia == ib:
              d = d[(nids != cid) | (d > min_dist)]
            else:
              d = d[d > min_dist]
            d = d[d < r_max]
            if d.size:
              pair_dists.append(d)

          if not pair_dists:
            continue
          all_d = np.concatenate(pair_dists)
          hist, _ = np.histogram(all_d, bins=r_edges)
          counts[ia, ib, :] += hist

    if self.pt.stubs == 0:
      from mpi4py import MPI
      counts = self.comm.reduce(counts, op=MPI.SUM, root=0)
      density = self.comm.reduce(density, op=MPI.SUM, root=0)
    else:
      counts = counts
      density = density

    if self.rank != 0:
      return

    shell_vol = (4.0 / 3.0) * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    gr = np.zeros_like(counts)
    for ia in range(n_elem):
      for ib in range(n_elem):
        denom = density[ia, ib] * shell_vol
        with np.errstate(divide="ignore", invalid="ignore"):
          gr[ia, ib, :] = np.where(denom > 0.0, counts[ia, ib, :] / denom, 0.0)

    self.pt.add_2_fitsnap("rdf", {
      "elements": elements,
      "n_bins": n_bins,
      "r_centers": r_centers.astype(np.float64),
      "r_edges": r_edges.astype(np.float64),
      "gr": gr.astype(np.float64),
      "rcut_in": rcut_in,
      "rcut": rcut,
    })

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
      #pyace_section.bbasis.E0vals = eshift_values
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
