from fitsnap3lib.scrapers.scrape import Scraper
import numpy as np
import logging

try:
    from adios2 import Stream
    HAS_ADIOS2 = True
except ImportError:
    HAS_ADIOS2 = False

# ------------------------------------------------------------------------------------------------

class ADIOS2(Scraper):
    """
    ADIOS2 scraper for reading .bp files created by fairchem_to_adios2.py.
    Designed for scalable MPI-parallel reading of FAIRChem datasets.
    
    Expected .bp file structure:
    - Attributes: nconfigs, element_map, has_forces, has_stress, unique_group_names
    - Arrays: NumAtoms, Energy, test_bool, GroupIndices
    - Variable-length arrays: PositionOffsets, PositionsFlat, AtomTypesFlat, ForcesFlat
    - Fixed-size arrays: Lattice (nconfigs, 3, 3), Stress (nconfigs, 3, 3)
    
    Note: Weights (eweight, fweight, vweight) are read from config file [GROUPS] section.
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
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
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
        self.unique_group_names = []
        
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
        
        # My configurations
        self.my_config_indices = []

    def scrape_groups(self, group_names=None):
        """
        Read metadata from ADIOS2 file and determine configuration distribution.
        """
        
        # Only rank 0 reads metadata, then broadcasts
        if self.rank == 0:
            try:
                with Stream(self.dataPath, 'r') as s:
                    # Step to make attributes available
                    for step in s.steps():
                        # Read attributes
                        attrs = s.available_attributes()
                        
                        nconfigs_attr = attrs.get('nconfigs')
                        if nconfigs_attr:
                            self.nconfigs = int(nconfigs_attr['Value'])
                        else:
                            raise KeyError('nconfigs attribute not found')
                        
                        element_map_attr = attrs.get('element_map')
                        if element_map_attr:
                            element_map_str = element_map_attr['Value']
                            if isinstance(element_map_str, bytes):
                                element_map_str = element_map_str.decode('utf-8')
                            # Strip any surrounding quotes and whitespace
                            element_map_str = element_map_str.strip('"\' ')
                            self.element_map = element_map_str.split(',')
                            # Strip quotes and whitespace from each element
                            self.element_map = [elem.strip('"\' ') for elem in self.element_map]
                        else:
                            raise KeyError('element_map attribute not found')
                        
                        has_forces_attr = attrs.get('has_forces')
                        if has_forces_attr:
                            self.has_forces = bool(int(has_forces_attr['Value']))
                        else:
                            self.has_forces = False
                        
                        has_stress_attr = attrs.get('has_stress')
                        if has_stress_attr:
                            self.has_stress = bool(int(has_stress_attr['Value']))
                        else:
                            self.has_stress = False
                        
                        unique_group_names_attr = attrs.get('unique_group_names')
                        if unique_group_names_attr:
                            unique_group_names_str = unique_group_names_attr['Value']
                            if isinstance(unique_group_names_str, bytes):
                                unique_group_names_str = unique_group_names_str.decode('utf-8')
                            # Strip any surrounding quotes
                            unique_group_names_str = unique_group_names_str.strip('"\'')
                            self.unique_group_names = unique_group_names_str.split('|')
                        else:
                            raise KeyError('unique_group_names attribute not found')
                        
                        break  # Only need first step
                    
                self.pt.single_print(
                    f"ADIOS2 scraper: {self.dataPath}, {self.nconfigs} configurations with "
                    f"[{', '.join(self.element_map)}], "
                    f"forces {self.has_forces}, stress {self.has_stress}"
                )
                    
            except Exception as e:
                raise RuntimeError(f"Failed to read ADIOS2 file metadata: {e}")
        
        # Broadcast metadata to all ranks
        if self.pt.stubs == 0:
            self.nconfigs = self.comm.bcast(self.nconfigs, root=0)
            self.element_map = self.comm.bcast(self.element_map, root=0)
            self.has_forces = self.comm.bcast(self.has_forces, root=0)
            self.has_stress = self.comm.bcast(self.has_stress, root=0)
            self.unique_group_names = self.comm.bcast(self.unique_group_names, root=0)
        
        # Build group_table from config file [GROUPS] section
        # This gets us the weights but we need to set training/testing sizes
        group_dict = {k: self.config.sections["GROUPS"].group_types[i]
                      for i, k in enumerate(self.config.sections["GROUPS"].group_sections)}
        self.group_table = self.config.sections["GROUPS"].group_table
        
        # Initialize training/testing sizes to 0 for all groups
        for group_name in self.unique_group_names:
            if group_name not in self.group_table:
                # Group exists in BP file but not in config - add with default weights
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
        Just store the local config indices.
        """
        # my_config_indices already set in scrape_groups
        self.my_configs = self.my_config_indices
        
    def scrape_configs(self):
        """
        Read ADIOS2 data and extract configurations for this rank.
        """
        self.data = []
        
        # Read all data arrays (all ranks read the same data, but only process their slice)
        try:
            with Stream(self.dataPath, 'r') as s:
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
                    self.lattices = s.read('Lattice')  # Shape: (nconfigs, 3, 3)
                    
                    if self.has_forces:
                        self.forces_flat = s.read('ForcesFlat')
                    if self.has_stress:
                        self.stresses = s.read('Stress')  # Shape: (nconfigs, 3, 3)
                    
                    break  # Only need first step
                
        except Exception as e:
            raise RuntimeError(f"Failed to read ADIOS2 data arrays: {e}")
        
        # Count training/testing sizes per group (only rank 0)
        if self.rank == 0:
            for i in range(self.nconfigs):
                group_idx = int(self.group_indices[i])
                group_name = self.unique_group_names[group_idx]
                is_test = bool(self.test_bool[i])
                
                if is_test:
                    self.group_table[group_name]['testing_size'] += 1
                else:
                    self.group_table[group_name]['training_size'] += 1
            
            # Print summary
            for group_name in self.unique_group_names:
                train_size = self.group_table[group_name]['training_size']
                test_size = self.group_table[group_name]['testing_size']
                self.pt.single_print(f"  {group_name}: {train_size} training, {test_size} testing")
        
        # Broadcast group_table to all ranks
        if self.pt.stubs == 0:
            self.group_table = self.comm.bcast(self.group_table, root=0)
        
        # Process configurations assigned to this rank
        for config_idx in self.my_config_indices:
            try:
                data_dict = self._extract_config(config_idx)
                if data_dict is not None:
                    self.data.append(data_dict)
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
        # For the last config, use total length instead of next offset
        if config_idx == self.nconfigs - 1:
            pos_end = len(self.atom_types_flat)
        else:
            pos_end = int(self.position_offsets[config_idx + 1])
        
        # Extract positions (already shaped as (natoms, 3) in the file)
        positions = self.positions_flat[pos_start:pos_end]
        
        # Extract atom types
        atom_type_indices = self.atom_types_flat[pos_start:pos_end]
        atom_types = [self.element_map[int(idx)] for idx in atom_type_indices]
        
        # Extract lattice (3x3 matrix)
        lattice = self.lattices[config_idx].reshape((3, 3))
        
        # Create data dictionary
        data_dict = {
            'Group': group_name,
            'File': f"{group_name}/{config_idx}",
            'Positions': positions.copy(),
            'AtomTypes': atom_types,
            'NumAtoms': natoms,
            'Lattice': lattice.copy(),
            'QMLattice': lattice.copy().T,  # Transpose for compatibility with parent class
            'Energy': float(self.energy[config_idx]),
            'test_bool': bool(self.test_bool[config_idx]),
        }
        
        # Extract forces if available (already shaped as (natoms, 3) in the file)
        if self.has_forces and self.use_forces:
            forces = self.forces_flat[pos_start:pos_end]
            data_dict['Forces'] = forces.copy()
        
        # Extract stress if available (3x3 matrix)
        if self.has_stress and self.use_stress:
            stress = self.stresses[config_idx].reshape((3, 3))
            data_dict['Stress'] = stress.copy()
        
        # Apply weights from config file using parent class method
        # We need to temporarily set self.data for parent methods to work
        old_data = self.data
        old_conversions = self.conversions
        self.data = data_dict
        self.conversions = self.default_conversions
        
        # Normalize coordinates for LAMMPS
        self._rotate_coords()
        self._translate_coords()
        
        # Apply weighting
        self._weighting(natoms)
        
        result = self.data
        self.data = old_data
        self.conversions = old_conversions
        
        return result

    def __del__(self):
        """
        Clean up resources on destruction.
        """
        # Stream context manager handles cleanup
        pass
