import numpy as np
from fitsnap3lib.calculators.lammps_base import LammpsBase, _extract_compute_np
from fitsnap3lib.io.sections.sections import Section


class LammpsUf3(LammpsBase):
  """
  LAMMPS ``compute ... uf3`` descriptors for UF3 linear ridge fitting.
  """

  def __init__(self, name, pt, config):
    super().__init__(name, pt, config)
    self._data = {}
    self._i = 0
    self._row_index = 0

    self._uf3 = Section.sections["UF3"]
    self.pt.check_lammps()

    self.pt.single_print(
      f"----------------------------------------------------------------\n"
      f"  LAMMPS UF3 CALCULATOR                                         \n"
      f"                                                                \n"
    )


  def get_width(self):
    return self._uf3.ncoeff

  def _pre_box_kokkos_setup2(self):
    self._lmp.command("newton off")
    self._lmp.command("package kokkos neigh full newton off")

  def _initialize_lammps(self, printlammps=0):
    #cmds=["-screen", "none", "-k", "on", "t", "1", "-sf", "kk"]
    cmds=["-screen", "none"]
    self._lmp = self.pt.initialize_lammps(self.config.args.lammpslog, printlammps=0, cmds=cmds)

  def _set_box(self):
    self._set_box_helper(numtypes=self._uf3.numtypes)
    # kspace_style is none by default just like in LAMMPS
    self._lmp.command(f"kspace_style {self.config.sections['REFERENCE'].kspace_style}")

  def _prepare_lammps(self):
    self._set_structure()
    for line in self.config.sections["REFERENCE"].lmp_pairdecl:
      if "pair_coeff" in line:
        lower = " ".join([word.lower() for word in line.split()[:4]])
        leave_alone = " ".join([word for word in line.split()[4:]])
        self._lmp.command(f"{lower} {leave_alone}")
      else:
        self._lmp.command(line.lower())
    self._set_computes()
    self._set_neighbor_list()

  def _set_neighbor_list(self):
    self._lmp.command("mass * 1.0e-20")
    #self._lmp.command("neighbor 1.0e-20 nsq")
    self._lmp.command("neigh_modify one 10000")

  def _create_atoms(self):
    for i, (a_t, (a_x, a_y, a_z)) in enumerate(zip(self._data["AtomTypes"], self._data["Positions"])):
      a_t = self._uf3.type_mapping[a_t]
      self._lmp.command(f"create_atoms {a_t} single {a_x:20.20g} {a_y:20.20g} {a_z:20.20g} remap yes")
    n_atoms = int(self._lmp.get_natoms())
    assert i + 1 == n_atoms, "Atom counts don't match when creating atoms: {}, {}\nGroup and configuration: {} {}".format(i + 1, n_atoms, self._data["Group"],self._data["File"])

  def _create_charge(self):
    for i, q in enumerate(self._data["Charges"]): self._lmp.command(f"set atom {i + 1} charge {q:20.20g} ")
    n_atoms = int(self._lmp.get_natoms())
    assert i + 1 == n_atoms, "Atom counts don't match when assigning charge: {}, {}\nGroup and configuration: {} {}".format(i + 1, n_atoms, self._data["Group"],self._data["File"])

  def _set_computes(self):
    virial_flag = 1 if self.config.sections["CALCULATOR"].stress else 0
    types = " ".join(self._uf3.type)
    self._lmp.command(f"compute uf3 all uf3 {self._uf3.degree} {virial_flag} descriptors.uf3 {types}")

  def _collect_lammps(self):
    num_atoms = self._data["NumAtoms"]
    n_lmp = int(self._lmp.get_natoms())
    if n_lmp != num_atoms:
      raise ValueError(
        f"UF3: LAMMPS natoms ({n_lmp}) != data NumAtoms ({num_atoms}) for "
        f"{self._data['Group']} {self._data['File']}"
      )
    energy = self._data["Energy"]
    lmp_atom_ids = self._extract_atom_ids(num_atoms)
    lmp_volume = self._lmp.get_thermo("vol")

    assert np.all(lmp_atom_ids == 1 + np.arange(num_atoms)), (
      "LAMMPS seems to have lost atoms \nGroup and configuration: {} {}".format(
        self._data["Group"], self._data["File"])
    )

    nrows_energy = 1
    nrows_force = 3 * num_atoms
    nrows_virial = 6 if self.config.sections["CALCULATOR"].stress else 0
    nrows_uf3 = nrows_energy + nrows_force + nrows_virial
    ncols_descriptors = self._uf3.ncoeff
    ncols_uf3 = ncols_descriptors + 1
    index = self.shared_index
    dindex = self.distributed_index

    # Prefer LAMMPS-reported shape (extract_compute) over wrapping a raw pointer with a
    # guessed size: if nrows_uf3 * ncols_uf3 were too large, np.frombuffer could read
    # past the compute buffer and surface bogus NaNs/Inf.
    lmp_uf3 = _extract_compute_np(self._lmp, "uf3", 0, 2, None)
    expected_shape = (nrows_uf3, ncols_uf3)
    if lmp_uf3.shape != expected_shape:
      raise ValueError(
        f"UF3: LAMMPS compute shape {lmp_uf3.shape} != expected {expected_shape} for "
        f"{self._data['File']} (group {self._data['Group']}). "
        "Check CALCULATOR energy/force/stress vs ncoeff and descriptors.uf3."
      )

    np.set_printoptions(
        precision=3, suppress=False, floatmode='fixed', linewidth=np.inf,
        formatter={'float': '{: .3f}'.format}, threshold = 800, edgeitems=50
    )

    if (np.isinf(lmp_uf3)).any() or (np.isnan(lmp_uf3)).any():
      self.pt.all_print(f"\n*** lmp_uf3 {lmp_uf3.shape}\n{lmp_uf3}\n");
      raise ValueError(f"NaN in descriptors of {self._data['File']} from group {self._data['Group']}")

    units_real = self.config.sections["REFERENCE"].units.lower() == "real"
    if units_real:
      ev_per_kcal_mol = 1.0 / 23.060549
      lmp_uf3[0:nrows_energy, :] *= ev_per_kcal_mol
      lmp_uf3[nrows_energy:nrows_energy + nrows_force, :] *= ev_per_kcal_mol

    irow = 0
    icolref = ncols_descriptors

    if self.config.sections["CALCULATOR"].energy:
      if self._uf3.bzeroflag:
        lmp_uf3[irow, :ncols_descriptors] /= num_atoms
      else:
        lmp_uf3[irow, self._uf3.numtypes:ncols_descriptors] /= num_atoms
        for atom in self._data["AtomTypes"]: lmp_uf3[irow, self._uf3.type_mapping[atom] - 1] += 1
        lmp_uf3[irow, :self._uf3.numtypes] /= len(self._data["AtomTypes"])

        #self.pt.single_print(f"*** lmp_uf3[irow,:2] {lmp_uf3[irow,:2]}")

      self.pt.shared_arrays['a'].array[index] = lmp_uf3[irow, :-1]
      ref_energy = lmp_uf3[irow, icolref]
      self.pt.shared_arrays['b'].array[index] = (energy - ref_energy) / num_atoms
      self.pt.shared_arrays['w'].array[index] = self._data["eweight"]
      self.pt.fitsnap_dict['Row_Type'][dindex:dindex + nrows_energy] = ['Energy'] * nrows_energy
      self.pt.fitsnap_dict['Atom_I'][dindex:dindex + nrows_energy] = [int(i) for i in range(nrows_energy)]
      index += nrows_energy
      dindex += nrows_energy

    irow += nrows_energy

    if self.config.sections["CALCULATOR"].force:
      s = slice(index, index + 3*num_atoms)
      db_atom_temp = lmp_uf3[irow:irow + nrows_force, :ncols_descriptors]
      db_atom_temp.shape = (3*num_atoms, self._uf3.ncoeff)
      self.pt.shared_arrays['a'].array[s] = db_atom_temp
      ref_forces = lmp_uf3[irow:irow + nrows_force, icolref]
      self.pt.shared_arrays['b'].array[s] = self._data["Forces"].ravel() - ref_forces
      self.pt.shared_arrays['w'].array[s] = self._data["fweight"]
      self.pt.fitsnap_dict['Row_Type'][dindex:dindex + nrows_force] = ['Force'] * nrows_force
      self.pt.fitsnap_dict['Atom_I'][dindex:dindex + nrows_force] = [int(np.floor(i / 3)) for i in range(nrows_force)]
      index += nrows_force
      dindex += nrows_force

    irow += nrows_force

    if self.config.sections["CALCULATOR"].stress:
      vb_sum_temp = lmp_uf3[irow:irow + nrows_virial, :ncols_descriptors] / lmp_volume
      vb_sum_temp.shape = (nrows_virial, self._uf3.ncoeff)
      self.pt.shared_arrays['a'].array[index:index + nrows_virial] = vb_sum_temp
      ref_stress = lmp_uf3[irow:irow + nrows_virial, icolref]
      tmp1 = self._data["Stress"][[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]].ravel()
      self.pt.shared_arrays['b'].array[index:index + nrows_virial] = tmp1 - ref_stress
      self.pt.shared_arrays['w'].array[index:index + nrows_virial] = self._data["vweight"]
      self.pt.fitsnap_dict['Row_Type'][dindex:dindex + nrows_virial] = ['Stress'] * nrows_virial
      self.pt.fitsnap_dict['Atom_I'][dindex:dindex + nrows_virial] = [int(0)] * nrows_virial
      index += nrows_virial
      dindex += nrows_virial

    length = dindex - self.distributed_index
    self.pt.fitsnap_dict['Groups'][self.distributed_index:dindex] = ['{}'.format(self._data['Group'])] * length
    self.pt.fitsnap_dict['Configs'][self.distributed_index:dindex] = ['{}'.format(self._data['File'])] * length
    self.pt.fitsnap_dict['Testing'][self.distributed_index:dindex] = [bool(self._data['test_bool'])] * length
    self.shared_index = index
    self.distributed_index = dindex
