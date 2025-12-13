
import numpy as np
from fitsnap3lib.calculators.lammps_base import LammpsBase, _extract_compute_np
from fitsnap3lib.calculators.lammps_pace import LammpsPace
import lammps

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
        

    # --------------------------------------------------------------------------------------------

    def get_width(self):
        """Get width of descriptor vector for PYACE calculator"""
        
        if not PYACE_AVAILABLE:
            raise RuntimeError("pyace not available")
        
        if self._bzeroflag:
            return self._ncoeff
        else:
            return self._ncoeff + self._numtypes
        

    # --------------------------------------------------------------------------------------------
    # everything is handled by LAMMPS compute pace (similar format as compute snap)

    def _set_computes(self):

        if self._bikflag:
            self._lmp.command("compute pace all pace coupling_coefficients.yace 1 0")
        else:
            self._lmp.command("compute pace all pace coupling_coefficients.yace 0 0")
 
        
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
            self.pt.single_print('! WARNING! applying np.nan_to_num()')
            lmp_pace = np.nan_to_num(lmp_pace)
        if (np.isinf(lmp_pace)).any() or (np.isnan(lmp_pace)).any():
            raise ValueError('NaN in computed data of file {} in group {}'.format(self._data["File"], self._data["Group"]))

        irow = 0
        bik_rows = 1
        icolref = ncols_descriptors
        
        if self.config.sections["REFERENCE"].units == "real":
            factor = 23.060549   # eV -> kcal/mol
        else:
            factor = 1.0

        if self.config.sections["CALCULATOR"].energy:
        
            b_sum_temp = lmp_pace[irow, :ncols_descriptors] / num_atoms

            if not self._bzeroflag:
                if self._bikflag:
                    raise NotImplementedError("Per atom energy is not implemented without bzeroflag")

                onehot_atoms = np.zeros(self._numtypes)
                for atom in self._data["AtomTypes"]:
                    onehot_atoms[self._type_mapping[atom] - 1] += 1
                onehot_atoms /= len(self._data["AtomTypes"])
                b_sum_temp = np.concatenate((onehot_atoms, b_sum_temp), axis=0)
                
            self.pt.shared_arrays['a'].array[index] = b_sum_temp
            ref_energy = lmp_pace[irow, icolref]
            self.pt.shared_arrays['b'].array[index] = factor*(energy - ref_energy) / num_atoms
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
            self.pt.shared_arrays['b'].array[s] = factor*(self._data["Forces"].ravel() - ref_forces)
            self.pt.shared_arrays['w'].array[s] = self._data["fweight"]
            self.pt.fitsnap_dict['Row_Type'][dindex:dindex + nrows_force] = ['Force'] * nrows_force
            self.pt.fitsnap_dict['Atom_I'][dindex:dindex + nrows_force] = [int(np.floor(i/3)) for i in range(nrows_force)]
            index += nrows_force
            dindex += nrows_force
        irow += nrows_force

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
    


