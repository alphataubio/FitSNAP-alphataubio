from fitsnap3lib.scrapers.scrape import Scraper
import numpy as np
import logging
import random
from os import path
from copy import copy
import warnings

# Suppress pkg_resources deprecation warning from torchtnt
warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*')

# Suppress logging warnings from PyTorch distributed on macos
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

try:
    from ase import Atoms
    from fairchem.core.datasets import AseDBDataset
    HAS_LMDB = True
except ImportError:
    HAS_LMDB = False

# ------------------------------------------------------------------------------------------------

class FAIRChem(Scraper):
    """
    FAIRChem scraper for reading OMat24, OC20, OC22, and other ASE-compatible LMDB datasets.
    Designed for read-only access with MPI parallelization.
    """

    def __init__(self, name, pt, config):
        if not HAS_LMDB:
            raise ImportError("LMDB scraper requires: pip install lmdb fairchem-core")
        
        super().__init__(name, pt, config)
        self.data = []
        
        # Get allowed elements from config - REQUIRED, no defaults
        if "PYACE" in self.config.sections:
            allowed_elements = self.config.sections["PYACE"].elements
        elif "ACE" in self.config.sections:
            allowed_elements = self.config.sections["ACE"].types
        elif "BISPECTRUM" in self.config.sections:
            allowed_elements = self.config.sections["BISPECTRUM"].types
        else:
            raise ValueError(
                "FAIRChem scraper requires element types to be specified in ACE, BISPECTRUM, or PYACE section. "
                "Example: In [ACE] section, add 'types = H O' for water."
            )
        
        self.allowed_elements = set(allowed_elements)
        self.pt.single_print(f"FAIRChem scraper: Allowed elements are {sorted(self.allowed_elements)}")
        self.pt.single_print(f"FAIRChem scraper: Any configs with other elements will be filtered out")
        
        # LMDB datasets will be determined from group names in scrape_groups
        # No single filename needed since we use multiple subdataset paths
        
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
        
        # For handling large datasets, allow subsampling
        self.max_configs_per_rank = getattr(self.config.sections["SCRAPER"], 'max_configs_per_rank', None)
        
        # Always use MDB_NOLOCK for distributed filesystems
        self.use_nolock = True
        
        # Option to skip structures with missing forces/energies
        self.require_energy = getattr(self.config.sections["SCRAPER"], 'require_energy', True)
        self.require_forces = getattr(self.config.sections["SCRAPER"], 'require_forces', self.use_forces)
        
        # Debugging options
        self.verbose = bool(getattr(self.config.sections["SCRAPER"], 'verbose', False))
        
        # Filtering options - applied in-memory after loading
        self.filter_data_id = self._parse_list_filter(
            getattr(self.config.sections["SCRAPER"], 'data_id', 'None'))
        self.filter_charge = self._parse_list_filter(
            getattr(self.config.sections["SCRAPER"], 'charge', 'None'))
        self.filter_composition = self._parse_list_filter(
            getattr(self.config.sections["SCRAPER"], 'composition', 'None'))
        
        if self.rank == 0 and (self.filter_data_id or self.filter_charge or self.filter_composition):
            self.pt.single_print("Filters will be applied in-memory after loading:")
            if self.filter_data_id:
                self.pt.single_print(f"  data_id: {self.filter_data_id}")
            if self.filter_charge:
                self.pt.single_print(f"  charge: {self.filter_charge}")
            if self.filter_composition:
                self.pt.single_print(f"  composition: {self.filter_composition}")

    def _parse_list_filter(self, filter_string):
        """Parse comma-separated filter string into list, or None if not set."""
        if filter_string == 'None' or filter_string is None or filter_string == '':
            return None
        # Split by comma and strip whitespace
        return [item.strip() for item in filter_string.split(',') if item.strip()]
    
    def _get_composition_string(self, atoms):
        """Get composition string like 'H2O' from atoms object."""
        from collections import Counter
        symbols = atoms.get_chemical_symbols()
        composition = Counter(symbols)
        # Sort by element symbol for consistent ordering
        sorted_elements = sorted(composition.items())
        return ''.join(f"{elem}{count}" for elem, count in sorted_elements)
    
    def scrape_groups(self, group_names=None):
        """
        Open LMDB datasets and identify available configurations.
        Each group in [GROUPS] maps to a subdataset directory.
        """
        self.group_metadata = {}
        self.local_configs = []
        
        # Get group table from config
        self.group_table = self.config.sections["GROUPS"].group_table
        
        # Build dataset paths for each group and get individual sizes
        dataset_paths = []
        group_to_path = {}
        self.group_index_ranges = {}  # Track which indices belong to which group
        cumulative_size = 0
        
        for group_name in self.group_table.keys():
            dataset_path = path.join(self.config.sections["PATH"].datapath, group_name)
            dataset_paths.append(dataset_path)
            group_to_path[group_name] = dataset_path
            
            # Get size of this individual dataset
            try:
                individual_dataset = AseDBDataset(config=dict(src=dataset_path))
                dataset_size = len(individual_dataset)
                
                # Track index range for this group
                self.group_index_ranges[group_name] = (cumulative_size, cumulative_size + dataset_size)
                cumulative_size += dataset_size
                
                if self.rank == 0:
                    self.pt.single_print(f"Group '{group_name}': {dataset_size} configurations (indices {self.group_index_ranges[group_name][0]}-{self.group_index_ranges[group_name][1]-1})")
                    
            except Exception as e:
                if self.rank == 0:
                    self.pt.single_print(f"Warning: Could not load dataset for group '{group_name}' at {dataset_path}: {e}")
                continue
            
        try:
            # Load dataset without any database-level filtering
            # Each rank will filter its assigned chunk in memory
            self.dataset = AseDBDataset(config=dict(src=dataset_paths))
            total_dataset_size = len(self.dataset)
            
            if self.rank == 0:
                self.pt.single_print(f"Opened {len(dataset_paths)} LMDB subdatasets with {total_dataset_size} total configurations")
            
            # Determine which configurations this rank will process
            configs_per_rank = total_dataset_size // self.size
            remainder = total_dataset_size % self.size
            
            start_idx = self.rank * configs_per_rank + min(self.rank, remainder)
            end_idx = start_idx + configs_per_rank + (1 if self.rank < remainder else 0)
            
            # Apply max_configs_per_rank limit if specified
            if self.max_configs_per_rank:
                end_idx = min(end_idx, start_idx + self.max_configs_per_rank)
            
            # Store configuration indices for this rank
            self.my_config_indices = list(range(start_idx, end_idx))
            
            if self.rank == 0:
                self.pt.single_print(f"Rank {self.rank} will process configurations {start_idx} to {end_idx-1}")
            
            # Create metadata for each group
            for group_name in self.group_table.keys():
                if group_name in self.group_index_ranges:  # Only if dataset was loaded successfully
                    self.group_metadata[group_name] = {
                        "subset": "train",  # Default to training
                        "eweight": 1.0,
                        "fweight": 1.0,
                        "dataset_path": group_to_path[group_name]
                    }
            
            # Add configurations to local list with group determination
            for idx in self.my_config_indices:
                group_name = self._determine_group_for_index(idx)
                self.local_configs.append((group_name, idx))
                
        except Exception as e:
            raise RuntimeError(f"Failed to open LMDB datasets: {e}")
        
        # Synchronize metadata across ranks
        if self.pt.stubs == 0:
            self.comm.barrier()
            all_meta = self.comm.allgather(self.group_metadata)
            self.group_metadata = {k: v for d in all_meta for k, v in d.items()}

    def divvy_up_configs(self):
        """
        Distribute configurations among MPI ranks.
        For LMDB, this is already done in scrape_groups.
        """
        if self.pt.stubs == 1:
            # For single process, limit to a small number for testing
            self.my_configs = self.local_configs[:min(10, len(self.local_configs))]
        else:
            # Already distributed in scrape_groups
            self.my_configs = self.local_configs
            
        if self.rank == 0:
            self.pt.single_print(f"Total configurations to process across all ranks: {len(self.my_configs) * self.size}")

    def scrape_configs(self):
        """
        Read and process LMDB configurations assigned to this rank.
        Each rank loads its chunk and filters in memory.
        """
        self.data = []
        
        # Track filtering statistics
        filtered_by_elements = 0
        filtered_by_data_id = 0
        filtered_by_charge = 0
        filtered_by_composition = 0
        configs_with_disallowed_elements = []  # Track which configs have bad elements
        
        try:
            for group_name, config_idx in self.my_configs:
                atoms = self._get_atoms_from_index(config_idx)
                
                if atoms is None:
                    continue
                
                # Get chemical symbols for this config
                symbols = atoms.get_chemical_symbols()
                
                # Filter by allowed elements - CHECK THIS FIRST
                disallowed = [s for s in symbols if s not in self.allowed_elements]
                if disallowed:
                    filtered_by_elements += 1
                    if self.verbose and len(configs_with_disallowed_elements) < 5:
                        configs_with_disallowed_elements.append(
                            f"{group_name}/{config_idx}: found {set(disallowed)} (allowed: {self.allowed_elements})"
                        )
                    continue
                
                # Filter by data_id if specified
                if self.filter_data_id is not None:
                    data_id = atoms.info.get('data_id', atoms.info.get('data', {}).get('data_id', None))
                    if data_id not in self.filter_data_id:
                        filtered_by_data_id += 1
                        continue
                
                # Filter by charge if specified
                if self.filter_charge is not None:
                    charge = atoms.info.get('charge', 0)
                    if str(charge) not in self.filter_charge and charge not in [int(c) for c in self.filter_charge]:
                        filtered_by_charge += 1
                        continue
                
                # Filter by composition if specified
                if self.filter_composition is not None:
                    composition_str = self._get_composition_string(atoms)
                    if composition_str not in self.filter_composition:
                        filtered_by_composition += 1
                        continue
                
                # Group name was already determined in scrape_groups
                actual_group_name = group_name
                
                # Extract data from atoms object
                data_dict = self._extract_data_from_atoms(atoms, actual_group_name, config_idx)
                
                if data_dict is not None:
                    # Double-check elements in extracted data as a safety measure
                    if not all(symbol in self.allowed_elements for symbol in data_dict["AtomTypes"]):
                        if self.verbose:
                            bad_elements = [s for s in data_dict["AtomTypes"] if s not in self.allowed_elements]
                            self.pt.single_print(
                                f"WARNING: Config {config_idx} passed filter but has disallowed elements {set(bad_elements)}. "
                                f"This should not happen! Skipping config."
                            )
                        filtered_by_elements += 1
                        continue
                    self.data.append(data_dict)
                elif self.verbose and self.rank == 0:
                    self.pt.single_print(f"Skipped configuration {config_idx} (missing required data)")
                    
        except Exception as e:
            raise RuntimeError(f"Error reading LMDB configurations: {e}")
        
        # Report statistics
        if self.rank == 0:
            self.pt.single_print(f"Rank {self.rank}: Successfully processed {len(self.data)} configurations")
            if filtered_by_elements > 0:
                self.pt.single_print(f"Rank {self.rank}: Filtered {filtered_by_elements} configs by element types")
                if self.verbose and configs_with_disallowed_elements:
                    self.pt.single_print(f"Rank {self.rank}: Examples of filtered configs:")
                    for example in configs_with_disallowed_elements:
                        self.pt.single_print(f"  {example}")
            if filtered_by_data_id > 0:
                self.pt.single_print(f"Rank {self.rank}: Filtered {filtered_by_data_id} configs by data_id")
            if filtered_by_charge > 0:
                self.pt.single_print(f"Rank {self.rank}: Filtered {filtered_by_charge} configs by charge")
            if filtered_by_composition > 0:
                self.pt.single_print(f"Rank {self.rank}: Filtered {filtered_by_composition} configs by composition")
            
        return self.data

    def _get_atoms_from_index(self, idx):
        """
        Get ASE Atoms object from LMDB index.
        """
        try:
            # Use fairchem dataset loader
            return self.dataset.get_atoms(idx)
        except Exception as e:
            logging.warning(f"Failed to read configuration {idx}: {e}")
            return None
    
    def _determine_group_for_index(self, config_idx):
        """
        Determine which group (subdataset) a configuration index belongs to.
        Uses the index ranges we tracked during scrape_groups.
        """
        for group_name, (start_idx, end_idx) in self.group_index_ranges.items():
            if start_idx <= config_idx < end_idx:
                return group_name
        
        # Fallback to first group if no match found
        group_names = list(self.group_metadata.keys())
        if group_names:
            return group_names[0]
        else:
            return "unknown_group"

    def _extract_data_from_atoms(self, atoms, group_name, config_idx):
        """
        Extract FitSNAP-compatible data from ASE Atoms object.
        """
        try:
            # Basic atomic information
            positions = atoms.get_positions()
            cell = atoms.get_cell()
            symbols = atoms.get_chemical_symbols()
            
            # Energy (required)
            energy = None
            if hasattr(atoms, 'get_total_energy'):
                try:
                    energy = atoms.get_total_energy()
                except:
                    energy = atoms.info.get('energy', None)
            else:
                energy = atoms.info.get('energy', None)
                
            # Check if energy is required and missing
            if self.require_energy and energy is None:
                return None
            
            # Forces (optional)
            forces = None
            if self.use_forces:
                if hasattr(atoms, 'get_forces'):
                    try:
                        forces = atoms.get_forces()
                    except:
                        forces = atoms.arrays.get('forces', None)
                else:
                    forces = atoms.arrays.get('forces', None)
                    
                # Check if forces are required and missing
                if self.require_forces and forces is None:
                    return None
            
            # Stress (optional)
            stress = None
            if self.use_stress:
                if hasattr(atoms, 'get_stress'):
                    try:
                        stress = atoms.get_stress()
                    except:
                        stress = atoms.info.get('stress', None)
                else:
                    stress = atoms.info.get('stress', None)
            
            # Create lattice matrix (3x3)
            # ASE returns cell vectors as rows, but FitSNAP/LAMMPS expects them as columns
            # So we need to transpose the cell matrix
            cell_array = cell.array.copy() if hasattr(cell, 'array') else np.array(cell)
            
            # Check if we need to create a box for non-periodic/molecular systems
            # This happens when the cell is not 3x3 OR when it's effectively zero
            if cell_array.shape != (3, 3) or np.allclose(cell_array, 0.0, atol=1e-6):
                # Create a large box for non-periodic systems
                max_coord = np.max(np.abs(positions)) + 10.0
                cell_array = np.diag([max_coord * 2, max_coord * 2, max_coord * 2])
            
            # Transpose to convert from ASE format (rows) to FitSNAP format (columns)
            lattice = cell_array.T
            
            # Create data dictionary compatible with FitSNAP
            data_dict = {
                "Group": group_name,
                "File": f"{group_name}/{config_idx}",
                "Subset": self.group_metadata[group_name]["subset"],
                "Positions": positions,
                "Energy": float(energy) if energy is not None else 0.0,
                "AtomTypes": symbols,
                "NumAtoms": len(symbols),
                "Lattice": lattice,
                "test_bool": False,  # Default to training
                "eweight": self.group_metadata[group_name]["eweight"],
                "fweight": self.group_metadata[group_name]["fweight"] / len(symbols),
            }
            
            # Add forces if available
            if forces is not None:
                data_dict["Forces"] = forces
            
            # Add stress if available
            if stress is not None:
                data_dict["Stress"] = stress
            
            return data_dict
            
        except Exception as e:
            logging.warning(f"Failed to extract data from atoms object {config_idx}: {e}")
            return None

    def __del__(self):
        """
        Clean up resources on destruction.
        """
        # AseDBDataset handles its own cleanup
        pass
