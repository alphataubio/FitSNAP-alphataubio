
from fitsnap3lib.calculators.lammps_base import LammpsBase, _extract_compute_np
from fitsnap3lib.calculators.lammps_pace import LammpsPace
import lammps
import os
import numpy as np


# LAMMPS ``units real`` uses kcal/mol (and kcal/mol/Å for forces); OMOL/JSON/VASP training uses eV.
_KCAL_MOL_PER_EV = 23.060549

# ------------------------------------------------------------------------------------------------

class LammpsPyace(LammpsPace):
  """
  Calculator using pyace basis in [PYACE] with LAMMPS compute pace
  """
  
  # --------------------------------------------------------------------------------------------

  def __init__(self, name, pt, config):
    super().__init__(name, pt, config, calculator_section="PYACE")
    self._data = {}
    self._i = 0
    self._row_index = 0
    # Saved on first _initialize_lammps so __del__ can finalize Kokkos explicitly.
    # On macOS, libomp tears down pthread infrastructure before Kokkos's atexit/static
    # destructor fires, leaving OpenMPInternal::~OpenMPInternal() holding an invalid
    # mutex (EINVAL). Calling lammps_kokkos_finalize() here — while Python's GC is
    # still running and libomp is fully alive — avoids that race entirely.
    self._lmp_lib = None

  # --------------------------------------------------------------------------------------------

  def get_width(self):
    """Get width of descriptor vector for PYACE calculator"""
    if self._bzeroflag: return self._ncoeff
    else: return self._ncoeff + self._numtypes

  # --------------------------------------------------------------------------------------------

  def _set_box(self):

    super()._set_box()
    
    # kspace_style is none by default just like in LAMMPS
    self._lmp.command(f"kspace_style {self.config.sections['REFERENCE'].kspace_style}")

  # --------------------------------------------------------------------------------------------

  def _initialize_lammps(self, printlammps=0):

    num_threads = os.getenv("SLURM_CPUS_PER_TASK")
    if num_threads is None: num_threads = os.getenv("OMP_NUM_THREADS", 1)

    # pace/kk parallelises only the cheap post-accumulation loop (320 cols), not
    # ACECTildeEvaluator::compute_atom() which dominates runtime. Real atom-level
    # parallelism would need per-thread evaluator instances.
    # Use kk only when SLURM_CPUS_PER_TASK is set (i.e. on cluster with multi-CPU tasks).
    if os.getenv("SLURM_CPUS_PER_TASK") is not None:
      cmds = ["-screen", "none", "-k", "on", "t", num_threads, "-sf", "kk"]
    else:
      cmds = ["-screen", "none"]
    self._lmp = self.pt.initialize_lammps(self.config.args.lammpslog, printlammps=0, cmds=cmds)
    if self._lmp_lib is None:
      self._lmp_lib = self._lmp.lib   # ctypes handle; survives lammps.close()
    self._lmp.command("newton off")
    if os.getenv("SLURM_CPUS_PER_TASK") is not None:
      self._lmp.command("package kokkos neigh full newton off")

  # --------------------------------------------------------------------------------------------

  def _set_neighbor_list(self):
    self._lmp.command("mass * 1.0e-20")
    self._lmp.command("neighbor 1.0e-20 bin")
    self._lmp.command("neigh_modify one 10000")

  # --------------------------------------------------------------------------------------------
  # everything is handled by LAMMPS compute pace (similar format as compute snap)

  def _set_computes(self):

    if self._bikflag: self._lmp.command("compute pace all pace coupling_coefficients.yace 1 0")
    else: self._lmp.command("compute pace all pace coupling_coefficients.yace 0 0")

    
  # --------------------------------------------------------------------------------------------

  def _collect_lammps(self):
  
    num_atoms = self._data["NumAtoms"]
    n_coeff = self._ncoeff
    energy = self._data["Energy"]
    lmp_atom_ids  = self._extract_atom_ids(num_atoms)
    lmp_pos  = self._extract_atom_positions(num_atoms)
    lmp_types  = self._extract_atom_types(num_atoms)
    lmp_volume = self._lmp.get_thermo("vol")
        
    assert np.all(lmp_atom_ids == 1 + np.arange(num_atoms)), "LAMMPS seems to have lost atoms \nGroup and configuration: {} {}".format(self._data["Group"],self._data["File"])

    # Extract pace data, including reference potential data
    nrows_energy = 1
    bik_rows = 1
    ndim_force = 3
    nrows_force = ndim_force * num_atoms
    ndim_virial = 6
    nrows_virial = ndim_virial
    nrows_pace = nrows_energy + nrows_force + nrows_virial
    ncols_descriptors = self._ncoeff
    ncols_reference = 1
    ncols_pace = ncols_descriptors + ncols_reference
    index = self.shared_index
    dindex = self.distributed_index
    lmp_pace = _extract_compute_np(self._lmp, "pace", 0, 2, (nrows_pace, ncols_pace))
    
    np.set_printoptions(
        precision=4, suppress=False, floatmode='fixed', linewidth=np.inf,
        formatter={'float': '{:.6f}'.format}, threshold = 800, edgeitems=50
    )
    
    #self.pt.single_print(f"\n\n*** i {self._i} num_atoms {num_atoms} n_coeff {n_coeff} lmp_pace\n{lmp_pace}")
    

    if (np.isinf(lmp_pace)).any() or (np.isnan(lmp_pace)).any():
      self.pt.single_print("! WARNING! applying np.nan_to_num()")
      lmp_pace = np.nan_to_num(lmp_pace)
    if (np.isinf(lmp_pace)).any() or (np.isnan(lmp_pace)).any():
      raise ValueError(f"NaN in file {self._data['File']} of group {self._data['Group']}")

    units_real = self.config.sections["REFERENCE"].units.lower() == "real"
    # ``compute pace`` reports the reference column and energy/force-related rows in LAMMPS
    # native energy units (kcal/mol, kcal/mol/Å for real). Training energies and forces are eV.
    if units_real:
        ev_per_kcal_mol = 1.0 / _KCAL_MOL_PER_EV
        r0, r1 = 0, nrows_energy
        lmp_pace[r0:r1, :] *= ev_per_kcal_mol
        r0, r1 = nrows_energy, nrows_energy + nrows_force
        lmp_pace[r0:r1, :] *= ev_per_kcal_mol
        if self.config.sections["CALCULATOR"].stress:
            if not getattr(self.pt, "_pyace_real_stress_units_warned", False):
                if self.pt.stubs == 0 and self.pt.get_rank() == 0:
                    self.pt.single_print(
                        "! WARNING LAMMPSPYACE: [REFERENCE] units=real — energy and force rows of "
                        "compute pace were converted kcal/mol → eV to match training data; stress "
                        "rows are not converted (PACE virial vs QM stress may be inconsistent). "
                        "Prefer units metal for combined energy/force/stress fits."
                    )
                self.pt._pyace_real_stress_units_warned = True

    irow = 0
    bik_rows = 1
    icolref = ncols_descriptors

    # -------------------------------- ENERGY --------------------------------

    if self.config.sections["CALCULATOR"].energy:
      b_sum_temp = lmp_pace[irow, :ncols_descriptors] / num_atoms
      if not self._bzeroflag:
        if self._bikflag: raise NotImplementedError("Per atom energy is not implemented without bzeroflag")
        onehot_atoms = np.zeros(self._numtypes)
        for atom in self._data["AtomTypes"]: onehot_atoms[self._type_mapping[atom] - 1] += 1
        onehot_atoms /= len(self._data["AtomTypes"])
        b_sum_temp = np.concatenate((onehot_atoms, b_sum_temp), axis=0)
      self.pt.shared_arrays['a'].array[index] = b_sum_temp
      ref_energy = lmp_pace[irow, icolref]
      # A, ref, and b are eV / (eV/Å) after optional real→eV conversion above. SLATE scales
      # Energy/Force metrics to kcal for [REFERENCE] units = real (see slate_common).
      self.pt.shared_arrays['b'].array[index] = (energy - ref_energy) / num_atoms
      #self.pt.all_print(f"*** i {self._i} energy {energy:.2f} ref_energy {ref_energy:.2f}")
      self.pt.shared_arrays['w'].array[index] = self._data["eweight"]
      self.pt.fitsnap_dict['Row_Type'][dindex:dindex + bik_rows] = ['Energy'] * nrows_energy
      self.pt.fitsnap_dict['Atom_I'][dindex:dindex + bik_rows] = [int(i) for i in range(nrows_energy)]
      index += nrows_energy
      dindex += nrows_energy

    irow += nrows_energy
    
    # -------------------------------- FORCE --------------------------------

    if self.config.sections["CALCULATOR"].force:
      s = slice(index, index + num_atoms*ndim_force)
      db_atom_temp = lmp_pace[irow:irow + nrows_force, :ncols_descriptors]
      db_atom_temp.shape = (num_atoms * ndim_force, self._ncoeff)
      if not self._bzeroflag:
        onehot_atoms = np.zeros((db_atom_temp.shape[0], self._numtypes))
        db_atom_temp = np.concatenate([onehot_atoms, db_atom_temp], axis=1)
      self.pt.shared_arrays['a'].array[s] = db_atom_temp
      ref_forces = lmp_pace[irow:irow + nrows_force, icolref]
      self.pt.shared_arrays['b'].array[s] = self._data["Forces"].ravel() - ref_forces
      self.pt.shared_arrays['w'].array[s] = self._data["fweight"]
      self.pt.fitsnap_dict['Row_Type'][dindex:dindex + nrows_force] = ['Force'] * nrows_force
      self.pt.fitsnap_dict['Atom_I'][dindex:dindex + nrows_force] = [int(np.floor(i/3)) for i in range(nrows_force)]
      index += nrows_force
      dindex += nrows_force

    irow += nrows_force

    # -------------------------------- STRESS --------------------------------

    if self.config.sections["CALCULATOR"].stress:
      vb_sum_temp = 160.2176565 * lmp_pace[irow:irow + nrows_virial, :ncols_descriptors] / lmp_volume
      vb_sum_temp.shape = (ndim_virial, self._ncoeff)
      if not self._bzeroflag:
        onehot_atoms = np.zeros((np.shape(vb_sum_temp)[0], self._numtypes))
        vb_sum_temp = np.concatenate([onehot_atoms, vb_sum_temp], axis=1)

      self.pt.shared_arrays['a'].array[index:index+ndim_virial] = vb_sum_temp
      ref_stress = lmp_pace[irow:irow + nrows_virial, icolref]
        
      # Convert b vector from eV/Å³ to GPa
      tmp1 = 160.2176565 * self._data["Stress"][[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]].ravel()
      tmp2 = ref_stress/10000
      #self.pt.single_print(f"*** tmp1 {tmp1} tmp2 {tmp2}")
      self.pt.shared_arrays['b'].array[index:index+ndim_virial] = tmp1 - tmp2

      self.pt.shared_arrays['w'].array[index:index+ndim_virial] = self._data["vweight"]
      self.pt.fitsnap_dict['Row_Type'][dindex:dindex + ndim_virial] = ['Stress'] * ndim_virial
      self.pt.fitsnap_dict['Atom_I'][dindex:dindex + ndim_virial] = [int(0)] * ndim_virial
      index += ndim_virial
      dindex += ndim_virial

    length = dindex - self.distributed_index
    self.pt.fitsnap_dict['Groups'][self.distributed_index:dindex] = ['{}'.format(self._data['Group'])] * length
    self.pt.fitsnap_dict['Configs'][self.distributed_index:dindex] = ['{}'.format(self._data['File'])] * length
    self.pt.fitsnap_dict['Testing'][self.distributed_index:dindex] = [bool(self._data['test_bool'])] * length
    self.shared_index = index
    self.distributed_index = dindex
    
    # Log validation data if enabled
    if self.config.sections["OUTFILE"].validation:
        pass
        
    #if self._i > 0: quit()
  
  # --------------------------------------------------------------------------------------------

  def __del__(self):
    """Finalize Kokkos before Python's GC releases C++ static state.

    macOS-specific: libomp registers atexit handlers that destroy pthread
    mutexes early; Kokkos's own atexit/static destructor then fires on an
    already-invalid mutex, calling std::terminate. By calling
    lammps_kokkos_finalize() here we pull Kokkos cleanup forward into the
    Python GC phase, while libomp is still fully initialized.
    No-op on Linux (GNU libgomp handles this ordering correctly).
    """
    try:
      lib = self._lmp_lib
      if lib is not None and hasattr(lib, 'lammps_kokkos_finalize'):
        lib.lammps_kokkos_finalize()
        self._lmp_lib = None
    except Exception:
      pass


