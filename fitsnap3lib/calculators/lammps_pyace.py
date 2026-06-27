
from fitsnap3lib.calculators.lammps_base import LammpsBase, _extract_compute_np
from fitsnap3lib.calculators.lammps_pace import LammpsPace
import os
import numpy as np


# LAMMPS ``units real`` uses kcal/mol (and kcal/mol/Å for forces); OMOL/JSON/VASP training uses eV.
_KCAL_MOL_PER_EV = 23.060549

# ------------------------------------------------------------------------------------------------

class LammpsPyace(LammpsPace):
  """
  Calculator using pyace basis in [PYACE] with LAMMPS compute pace.

  One LAMMPS instance and one ``compute pace`` for the whole config loop so the
  coupling-coefficients YAML is parsed only once.  Each config only swaps atoms
  and the box (delete_atoms + change_box + create_atoms); pair_style/compute
  are not recreated.
  """

  # --------------------------------------------------------------------------------------------

  def __init__(self, name, pt, config):
    super().__init__(name, pt, config, calculator_section="PYACE")
    self._data = {}
    self._i = 0
    self._row_index = 0
    self._lmp_lib = None

    self._initialize_lammps()

  # --------------------------------------------------------------------------------------------

  def __del__(self):
    """Close LAMMPS *before* finalizing Kokkos.

    The LAMMPS instance owns Kokkos Views (VerletKokkos::f_merge_copy, the
    compute pace/kk device buffers, ...). Those must be freed while Kokkos is
    still live. If Kokkos is finalized first, the still-open instance is torn
    down later by Python GC and deallocates those Views *after* Kokkos::finalize,
    which aborts with
      "allocation ... is being deallocated after Kokkos::finalize was called".
    This calculator keeps a single self._lmp open for the whole config loop, so
    it is still alive here. Correct teardown order: close the instance first,
    then finalize Kokkos exactly once.
    """
    try:
      if self._lmp is not None:
        self._lmp.close()
        self._lmp = None
        # Clear ParallelTools' handle too, so a later close_lammps() is a no-op
        # and never double-closes a destroyed instance.
        self.pt._lmp = None
      lib = self._lmp_lib
      if lib is not None and hasattr(lib, 'lammps_kokkos_finalize'):
        lib.lammps_kokkos_finalize()
        self._lmp_lib = None
    except Exception:
      pass

  # --------------------------------------------------------------------------------------------

  def get_width(self):
    if self._bzeroflag: return self._ncoeff
    else: return self._ncoeff + self._numtypes

  # --------------------------------------------------------------------------------------------

  def _initialize_lammps(self, printlammps=0):
    """One-time setup: open LAMMPS, declare pair/compute/neighbor (YAML read here)."""

    if (kokkos := self.config.sections["CALCULATOR"].kokkos):
      num_threads = os.getenv("OMP_NUM_THREADS", 4)
      cmds = ["-screen", "lammps_kk.log", "-k", "on", "t", num_threads, "-sf", "kk"]
    else:
      cmds = ["-screen", "lammps.log"]

    self._lmp = self.pt.initialize_lammps(self.config.args.lammpslog, printlammps=0, cmds=cmds)
    if self._lmp_lib is None: self._lmp_lib = self._lmp.lib

    self._lmp.command("newton off")
    if kokkos: self._lmp.command("package kokkos neigh full newton off")

    reference_section = self.config.sections["REFERENCE"]
    self._lmp.command("echo screen")
    self._lmp.command("units " + reference_section.units)
    self._lmp.command("boundary " + reference_section.boundary)
    self._lmp.command("atom_style " + reference_section.atom_style)
    self._lmp.command("atom_modify map array sort 0 2.0")

    numtypes = len(self.config.sections["PYACE"].elements)
    self._lmp.command("region pybox prism 0 1 0 1 0 1 1 1 1")
    self._lmp.command(f"create_box {numtypes} pybox")

    for line in reference_section.lmp_pairdecl:
      if "pair_coeff" in line:
        lower = " ".join([word.lower() for word in line.split()[:4]])
        leave_alone = " ".join([word for word in line.split()[4:]])
        self._lmp.command(f"{lower} {leave_alone}")
      else:
        self._lmp.command(line.lower())

    self._lmp.command("mass * 1.0e-20")
    self._lmp.command("neighbor 1.0e-20 bin") # bin needed for kokkos not nsq
    self._lmp.command("neigh_modify one 10000")

    flags = f"{int(self._bikflag)} {int(self._dgradflag)} {int(self.config.sections['CALCULATOR'].stress)}"
    self._lmp.command(f"compute pace all pace coupling_coefficients.yace {flags}")

    self._lmp.command(f"kspace_style {reference_section.kspace_style}")

    #self._lmp.command(f"fix 1 all acks2/reaxff 1 0 30 1e-6 omol25_hcno.acks2 maxiter 10000")


  # --------------------------------------------------------------------------------------------

  def process_configs(self, data, i):

    try:
      self._data = data
      self._i = i
      self._prepare_lammps()
      self._run_lammps()
      self._collect_lammps()

    except Exception as e:
      if False and self.config.args.printlammps:
        self._data = data
        self._i = i
        self._initialize_lammps(1)
        self._prepare_lammps()
        self._run_lammps()
        self._collect_lammps()
        self._lmp = self.pt.close_lammps()
      raise e

  # --------------------------------------------------------------------------------------------

  def _create_atoms(self):
    self._create_atoms_helper(type_mapping=self._type_mapping)

  # --------------------------------------------------------------------------------------------

  def _prepare_lammps(self):
    """Swap in a new structure without clear and without re-creating compute pace."""

    self._lmp.command("delete_atoms group all")

    ((ax, bx, cx), (ay, by, cy), (az, bz, cz)) = self._data["Lattice"]
    assert all(abs(c) < 1e-10 for c in (ay, az, bz)), \
      f"Cell not normalized for {self._data['Group']} / {self._data['File']}"

    change_box_cmd = "change_box all triclinic"
    change_box_cmd += f" x final 0 {ax:20.20g}"
    change_box_cmd += f" y final 0 {by:20.20g}"
    change_box_cmd += f" z final 0 {cz:20.20g}"
    change_box_cmd += f" xy final {bx:20.20g}"
    change_box_cmd += f" xz final {cx:20.20g}"
    change_box_cmd += f" yz final {cy:20.20g}"
    self._lmp.command(change_box_cmd)

    self._create_atoms()

    atom_style = self.config.sections["REFERENCE"].atom_style
    if atom_style == "spin": self._create_spins()
    #if atom_style == "charge": self._create_charge()

    #q = self._lmp.extract_atom("q") # , dtype=None, nelem=None, dim=None
    #num_atoms = self._data["NumAtoms"]
    #self.pt.single_print(f"*** BEFORE q {q[:num_atoms]}")


  # --------------------------------------------------------------------------------------------

  def _run_lammps(self):
    # Occasional neighbor stencils (compute pace) need timestamp reset when reusing
    # run 0 at step 0 across configs with different box sizes.
    self._lmp.command("reset_timestep 0")
    self._lmp.command("run 0 post no")

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

    nrows_energy = 1
    bik_rows = 1
    ndim_force = 3
    nrows_force = ndim_force * num_atoms
    ndim_virial = 6 if self.config.sections["CALCULATOR"].stress else 0
    nrows_virial = ndim_virial
    nrows_pace = nrows_energy + nrows_force + nrows_virial
    ncols_descriptors = self._ncoeff
    ncols_reference = 1
    ncols_pace = ncols_descriptors + ncols_reference
    index = self.shared_index
    dindex = self.distributed_index
    lmp_pace = _extract_compute_np(self._lmp, "pace", 0, 2, (nrows_pace, ncols_pace))

    np.set_printoptions(precision=3, suppress=True, floatmode='fixed', linewidth=np.inf)
    np.set_printoptions(formatter={'float': '{:.3f}'.format})
    #q = self._lmp.extract_atom("q") # , dtype=None, nelem=None, dim=None
    #qnp = np.array(q[:num_atoms])
    #self.pt.single_print(f"***\nnbo {self._data['Charges']}\nq {qnp}")


    if (np.isinf(lmp_pace)).any() or (np.isnan(lmp_pace)).any():
      self.pt.single_print("! WARNING! applying np.nan_to_num()")
      lmp_pace = np.nan_to_num(lmp_pace)
    if (np.isinf(lmp_pace)).any() or (np.isnan(lmp_pace)).any():
      raise ValueError(f"NaN in file {self._data['File']} of group {self._data['Group']}")

    units_real = self.config.sections["REFERENCE"].units.lower() == "real"
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
      self.pt.shared_arrays['b'].array[index] = (energy - ref_energy) / num_atoms
      self.pt.shared_arrays['w'].array[index] = self._data["eweight"]
      self.pt.fitsnap_dict['Row_Type'][dindex:dindex + bik_rows] = ['Energy'] * nrows_energy
      self.pt.fitsnap_dict['Atom_I'][dindex:dindex + bik_rows] = [int(i) for i in range(nrows_energy)]
      index += nrows_energy
      dindex += nrows_energy

    irow += nrows_energy

    if self.config.sections["CALCULATOR"].force:
      s = slice(index, index + num_atoms*ndim_force)
      db_atom_temp = lmp_pace[irow:irow + nrows_force, :ncols_descriptors]
      db_atom_temp.shape = (num_atoms * ndim_force, self._ncoeff)
      if not self._bzeroflag:
        onehot_atoms = np.zeros((db_atom_temp.shape[0], self._numtypes))
        db_atom_temp = np.concatenate([onehot_atoms, db_atom_temp], axis=1)
      self.pt.shared_arrays['a'].array[s] = db_atom_temp
      ref_forces = lmp_pace[irow:irow + nrows_force, icolref]
      tmp = self._data["Forces"].ravel()
      self.pt.shared_arrays['b'].array[s] =  tmp - ref_forces



      #self.pt.single_print(f"*** tmp {tmp}\n ref_forces {ref_forces}\n\n")
      self.pt.shared_arrays['w'].array[s] = self._data["fweight"]
      self.pt.fitsnap_dict['Row_Type'][dindex:dindex + nrows_force] = ['Force'] * nrows_force
      self.pt.fitsnap_dict['Atom_I'][dindex:dindex + nrows_force] = [int(np.floor(i/3)) for i in range(nrows_force)]
      index += nrows_force
      dindex += nrows_force

    irow += nrows_force

    if self.config.sections["CALCULATOR"].stress:
      vb_sum_temp = 1.6021765e6 * lmp_pace[irow:irow + nrows_virial, :ncols_descriptors] / lmp_volume
      vb_sum_temp.shape = (ndim_virial, self._ncoeff)
      if not self._bzeroflag:
        onehot_atoms = np.zeros((np.shape(vb_sum_temp)[0], self._numtypes))
        vb_sum_temp = np.concatenate([onehot_atoms, vb_sum_temp], axis=1)

      self.pt.shared_arrays['a'].array[index:index+ndim_virial] = vb_sum_temp
      ref_stress = lmp_pace[irow:irow + nrows_virial, icolref]
      tmp1 = self._data["Stress"][[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]].ravel()
      tmp2 = ref_stress
      #self.pt.single_print(f"***\ntmp1 {tmp1}\ntmp2 {tmp2}")

      self.pt.shared_arrays['b'].array[index:index+ndim_virial] = (tmp1 - tmp2)

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

  # --------------------------------------------------------------------------------------------

