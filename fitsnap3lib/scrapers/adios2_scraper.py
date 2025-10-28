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
    - Attributes: nconfigs, element_map, has_forces, has_stress, group_names
    - Arrays: NumAtoms, Energy, test_bool, eweight, fweight, vweight
    - Variable-length arrays: PositionOffsets, PositionsFlat, AtomTypesFlat, ForcesFlat
    - Fixed-size arrays: Lattice (nconfigs, 9), Stress (nconfigs, 9)
    """

    def __init__(self, name, pt, config):
        if not HAS_ADIOS2:
            raise ImportError("ADIOS2 scraper requires: pip install adios2")
        
        super().__init__(name, pt, config)
        self.data = []
        
        # Get the .bp file path from SCRAPER section
        if not hasattr(self.config.sections["SCRAPER"], 'bp_file'):
            raise ValueError("ADIOS2 scraper requires 'bp_file' parameter in [SCRAPER] section")
        
        self.bp_file = self.config.sections["SCRAPER"].bp_file
        
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
        self.group_names = []
        
        # Data arrays
        self.num_atoms = None
        self.energy = None
        self.test_bool = None
        self.eweight = None
        self.fweight = None
        self.vweight = None
        self.position_offsets = None
        self.positions_flat = None
        self.atom_types_flat = None
        self.lattices = None  # Shape: (nconfigs, 9)
        self.forces_flat = None
        self.stresses = None  # Shape: (nconfigs, 9)
        
        # My configurations
        self.my_config_indices = []

    def scrape_groups(self, group_names=None):
        """
        Read metadata from ADIOS2 file and determine configuration distribution.
        """
        if self.rank == 0:
            self.pt.single_print(f"Opening ADIOS2 file: {self.bp_file}")
        
        # Only rank 0 reads metadata, then broadcasts
        if self.rank == 0:
            try:
                with Stream(self.bp_file, 'r') as s:
                    # Read attributes
                    self.nconfigs = s.read_attribute('nconfigs')
                    if isinstance(self.nconfigs, np.ndarray):
                        self.nconfigs = int(self.nconfigs[0])
                    else:
                        self.nconfigs = int(self.nconfigs)
                    
                    element_map_str = s.read_attribute('element_map')
                    if isinstance(element_map_str, np.ndarray):
                        element_map_str = str(element_map_str[0])
                    elif isinstance(element_map_str, bytes):
                        element_map_str = element_map_str.decode('utf-8')
                    else:
                        element_map_str = str(element_map_str)
                    self.element_map = element_map_str.split(',')
                    
                    has_forces_val = s.read_attribute('has_forces')
                    if isinstance(has_forces_val, np.ndarray):
                        self.has_forces = bool(has_forces_val[0])
                    else:
                        self.has_forces = bool(has_forces_val)
                    
                    has_stress_val = s.read_attribute('has_stress')
                    if isinstance(has_stress_val, np.ndarray):
                        self.has_stress = bool(has_stress_val[0])
                    else:
                        self.has_stress = bool(has_stress_val)
                    
                    group_names_str = s.read_attribute('group_names')
                    if isinstance(group_names_str, np.ndarray):
                        group_names_str = str(group_names_str[0])
                    elif isinstance(group_names_str, bytes):
                        group_names_str = group_names_str.decode('utf-8')
                    else:
                        group_names_str = str(group_names_str)
                    self.group_names = group_names_str.split('|')
                    
                    self.pt.single_print(f"ADIOS2 file contains {self.nconfigs} configurations")
                    self.pt.single_print(f"Elements: {', '.join(self.element_map)}")
                    self.pt.single_print(f"Has forces: {self.has_forces}")
                    self.pt.single_print(f"Has stress: {self.has_stress}")
                    
            except Exception as e:
                raise RuntimeError(f"Failed to read ADIOS2 file metadata: {e}")
        
        # Broadcast metadata to all ranks
        if self.pt.stubs == 0:
            self.nconfigs = self.comm.bcast(self.nconfigs, root=0)
            self.element_map = self.comm.bcast(self.element_map, root=0)
            self.has_forces = self.comm.bcast(self.has_forces, root=0)
            self.has_stress = self.comm.bcast(self.has_stress, root=0)
            self.group_names = self.comm.bcast(self.group_names, root=0)
        
        # Determine which configurations this rank will process
        configs_per_rank = self.nconfigs // self.size
        remainder = self.nconfigs % self.size
        
        start_idx = self.rank * configs_per_rank + min(self.rank, remainder)
        end_idx = start_idx + configs_per_rank + (1 if self.rank < remainder else 0)
        
        self.my_config_indices = list(range(start_idx, end_idx))
        
        if self.rank == 0:
            self.pt.single_print(f"Each rank will process ~{configs_per_rank} configurations")
        
        # Create group_table from unique groups
        unique_groups = sorted(set(self.group_names))
        self.group_table = {}
        for group in unique_groups:
            self.group_table[group] = {
                'eweight': 1.0,
                'fweight': 1.0,
                'vweight': 1.0,
                'training_size': 0,  # Will be updated in divvy_up_configs
                'testing_size': 0,
            }
        
        # Count training and testing configs per group
        for i, group_name in enumerate(self.group_names):
            # We'll read test_bool later, for now just count all
            self.group_table[group_name]['training_size'] += 1
        
        if self.rank == 0:
            self.pt.single_print(f"Found {len(unique_groups)} unique groups: {unique_groups}")

    def divvy_up_configs(self):
        """
        Configuration distribution is already done in scrape_groups.
        Just store the local config indices.
        """
        # my_config_indices already set in scrape_groups
        self.my_configs = self.my_config_indices
        
        if self.rank == 0:
            total_configs = len(self.my_configs) * self.size
            self.pt.single_print(f"Total configurations to process: {total_configs}")
            self.pt.single_print(f"Rank 0 processing {len(self.my_configs)} configurations")

    def scrape_configs(self):
        """
        Read ADIOS2 data and extract configurations for this rank.
        """
        self.data = []
        
        # Read all data arrays (all ranks read the same data, but only process their slice)
        try:
            with Stream(self.bp_file, 'r') as s:
                # Read per-config arrays
                self.num_atoms = s.read('NumAtoms')
                self.energy = s.read('Energy')
                self.test_bool = s.read('test_bool')
                self.eweight = s.read('eweight')
                self.fweight = s.read('fweight')
                self.vweight = s.read('vweight')
                
                # Read variable-length arrays
                self.position_offsets = s.read('PositionOffsets')
                self.positions_flat = s.read('PositionsFlat')
                self.atom_types_flat = s.read('AtomTypesFlat')
                
                # Read fixed-size arrays
                self.lattices = s.read('Lattice')  # Shape: (nconfigs, 9)
                
                if self.has_forces:
                    self.forces_flat = s.read('ForcesFlat')
                if self.has_stress:
                    self.stresses = s.read('Stress')  # Shape: (nconfigs, 9)
                
        except Exception as e:
            raise RuntimeError(f"Failed to read ADIOS2 data arrays: {e}")
        
        # Process configurations assigned to this rank
        for config_idx in self.my_config_indices:
            try:
                data_dict = self._extract_config(config_idx)
                if data_dict is not None:
                    self.data.append(data_dict)
            except Exception as e:
                logging.warning(f"Failed to extract config {config_idx}: {e}")
                continue
        
        if self.rank == 0:
            self.pt.single_print(f"Rank {self.rank}: Successfully processed {len(self.data)} configurations")
        
        return self.data

    def _extract_config(self, config_idx):
        """
        Extract a single configuration from the flattened arrays.
        """
        # Get group name
        group_name = self.group_names[config_idx]
        
        # Get number of atoms
        natoms = int(self.num_atoms[config_idx])
        
        # Get position range
        pos_start = int(self.position_offsets[config_idx])
        pos_end = int(self.position_offsets[config_idx + 1])
        
        # Extract positions (flattened as [x1, y1, z1, x2, y2, z2, ...])
        positions_1d = self.positions_flat[pos_start * 3 : pos_end * 3]
        positions = positions_1d.reshape((natoms, 3))
        
        # Extract atom types
        atom_type_indices = self.atom_types_flat[pos_start : pos_end]
        atom_types = [self.element_map[int(idx)] for idx in atom_type_indices]
        
        # Extract lattice (3x3 matrix stored as 9 elements)
        lattice = self.lattices[config_idx].reshape((3, 3))
        
        # Create data dictionary
        data_dict = {
            'Group': group_name,
            'File': f"{group_name}/{config_idx}",
            'Positions': positions.copy(),
            'AtomTypes': atom_types,
            'NumAtoms': natoms,
            'Lattice': lattice.copy(),
            'Energy': float(self.energy[config_idx]),
            'test_bool': bool(self.test_bool[config_idx]),
            'eweight': float(self.eweight[config_idx]),
            'fweight': float(self.fweight[config_idx]),
            'vweight': float(self.vweight[config_idx]),
        }
        
        # Extract forces if available
        if self.has_forces and self.use_forces:
            forces_1d = self.forces_flat[pos_start * 3 : pos_end * 3]
            forces = forces_1d.reshape((natoms, 3))
            data_dict['Forces'] = forces.copy()
        
        # Extract stress if available (3x3 matrix stored as 9 elements)
        if self.has_stress and self.use_stress:
            stress = self.stresses[config_idx].reshape((3, 3))
            data_dict['Stress'] = stress.copy()
        
        return data_dict

    def __del__(self):
        """
        Clean up resources on destruction.
        """
        # Stream context manager handles cleanup
        pass
