#!/usr/bin/env python3
"""
Convert FAIRChem LMDB datasets to ADIOS2 .bp format with element filtering.

Uses fairchem's AseDBDataset to read LMDB databases (mocks torch if not installed).
Expects LMDB databases to be already extracted in train/ and val/ directories.

Usage:
    python fairchem_to_adios2.py --dataset scratch/omat24 --elements Al Ni --output omat24_AlNi.bp
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm

# Mock torch to allow fairchem imports without actually requiring torch
try:
    import torch
except ImportError:
    # Create a more sophisticated mock that handles nested imports
    from types import ModuleType
    import importlib.abc
    import importlib.machinery
    
    class MockModule(ModuleType):
        """A mock module that returns itself for any attribute access"""
        def __init__(self, name):
            super().__init__(name)
            # Make it look like a real module package
            self.__path__ = []
            self.__file__ = f"<mock {name}>"
        
        def __getattr__(self, name):
            mock = MockModule(f"{self.__name__}.{name}")
            sys.modules[mock.__name__] = mock
            return mock
        
        def __call__(self, *args, **kwargs):
            # If called with a function (as a decorator), return that function
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            # Otherwise return a mock that can be used as a decorator
            return MockModule(f"{self.__name__}()")
        
        def __iter__(self):
            # Return empty iterator to satisfy import system
            return iter([])
        
        def __mro_entries__(self, bases):
            # When used as a base class, return object as the actual base
            return (object,)
        
        def __getitem__(self, item):
            # Support generic type subscripting like Dataset[T_co]
            return MockModule(f"{self.__name__}[{item}]")
    
    class TorchMockFinder(importlib.abc.MetaPathFinder):
        """A meta path finder that creates mock modules for torch imports"""
        def find_spec(self, fullname, path, target=None):
            # Intercept any torch-related imports
            if fullname.startswith('torch') or fullname.startswith('torch_'):
                return importlib.machinery.ModuleSpec(fullname, TorchMockLoader())
            return None
    
    class TorchMockLoader(importlib.abc.Loader):
        """A loader that creates mock modules"""
        def create_module(self, spec):
            return MockModule(spec.name)
        
        def exec_module(self, module):
            # Nothing to execute
            pass
    
    # Install the meta path finder
    sys.meta_path.insert(0, TorchMockFinder())
    
    # Create the base mock modules
    mock_torch = MockModule('torch')
    sys.modules['torch'] = mock_torch

try:
    from adios2 import Stream
    HAS_ADIOS2 = True
except ImportError as e:
    HAS_ADIOS2 = False
    print("="*80, file=sys.stderr)
    print("ERROR: Failed to import ADIOS2", file=sys.stderr)
    print("="*80, file=sys.stderr)
    print(f"\nImport error: {e}\n", file=sys.stderr)
    print("ADIOS2 requires both the C++ libraries AND Python bindings.\n", file=sys.stderr)
    print("Installation instructions:", file=sys.stderr)
    print("\n1. macOS (local development):", file=sys.stderr)
    print("   brew install adios2", file=sys.stderr)
    print("   pip uninstall adios2", file=sys.stderr)
    print("   pip install adios2\n", file=sys.stderr)
    print("2. Linux cluster (HPC):", file=sys.stderr)
    print("   module load adios2", file=sys.stderr)
    print("   # or: module load adios2/2.10.1", file=sys.stderr)
    print("   pip install adios2\n", file=sys.stderr)
    print("3. Conda/Mamba (cross-platform):", file=sys.stderr)
    print("   conda install -c conda-forge adios2 adios2-python", file=sys.stderr)
    print("   # or: mamba install -c conda-forge adios2 adios2-python\n", file=sys.stderr)
    print("Note: On macOS, 'pip install adios2' alone will NOT work.", file=sys.stderr)
    print("      You must install the C++ libraries first via Homebrew or conda.", file=sys.stderr)
    print("="*80, file=sys.stderr)
    sys.exit(1)

try:
    from fairchem.core.datasets import AseDBDataset
    from ase.data import chemical_symbols
    HAS_FAIRCHEM = True
except ImportError as e:
    HAS_FAIRCHEM = False
    print("="*80, file=sys.stderr)
    print("ERROR: Failed to import fairchem", file=sys.stderr)
    print("="*80, file=sys.stderr)
    print(f"\nImport error: {e}\n", file=sys.stderr)
    print("Install: pip install fairchem-core\n", file=sys.stderr)
    print("="*80, file=sys.stderr)
    sys.exit(1)


def process_lmdb_dir(lmdb_path, group_name, allowed_elements, test_bool):
    """
    Process a single LMDB database directory using fairchem's AseDBDataset.
    
    Args:
        lmdb_path: Path to LMDB database directory
        group_name: Name for this group of configurations
        allowed_elements: Set of allowed element symbols
        test_bool: Boolean indicating if this is validation data
    
    Returns:
        Tuple of (list of configs, number filtered)
    """
    # AseDBDataset expects a config dict with 'src' pointing to the database
    dataset = AseDBDataset({"src": str(lmdb_path)})
    
    configs = []
    filtered_count = 0
    
    for i in range(len(dataset)):
    
        atoms = dataset.get_atoms(i)
        
        # Get element symbols
        atom_types = atoms.get_chemical_symbols()
        
        # Filter by elements
        if not set(atom_types).issubset(allowed_elements):
            filtered_count += 1
            continue
        
        # Extract data
        num_atoms = len(atoms)
        positions = atoms.get_positions()
        cell = atoms.get_cell().array
        
        # Energy
        energy = 0.0
        if atoms.calc is not None:
            try:
                energy = float(atoms.get_potential_energy())
            except:
                pass
        
        # Forces (if available)
        forces = None
        if atoms.calc is not None:
            try:
                forces = atoms.get_forces()
            except:
                pass
        
        # Stress (if available)
        stress = None
        if atoms.calc is not None:
            try:
                stress_voigt = atoms.get_stress()
                # Convert Voigt to 3x3
                stress = np.array([
                    [stress_voigt[0], stress_voigt[5], stress_voigt[4]],
                    [stress_voigt[5], stress_voigt[1], stress_voigt[3]],
                    [stress_voigt[4], stress_voigt[3], stress_voigt[2]]
                ])
            except:
                pass
        
        config = {
            'Group': group_name,
            'NumAtoms': num_atoms,
            'Positions': positions,
            'AtomTypes': atom_types,
            'Lattice': cell,
            'Energy': energy,
            'test_bool': test_bool,
            'eweight': 1.0,
            'fweight': 1.0 if forces is not None else 0.0,
            'vweight': 1.0 if stress is not None else 0.0,
        }
        
        if forces is not None:
            config['Forces'] = forces
        if stress is not None:
            config['Stress'] = stress
        
        configs.append(config)
    
    return configs, filtered_count


def process_dataset_path(dataset_root, subset_type, allowed_elements, num_workers=None):
    """
    Process train or val directory containing LMDB database directories.
    
    Args:
        dataset_root: Root path containing train/ and val/ subdirectories
        subset_type: Either 'train' or 'val'
        allowed_elements: Set of allowed element symbols
        num_workers: Number of parallel workers (default: cpu_count)
    
    Returns:
        List of configuration dictionaries
    """
    subset_dir = Path(dataset_root) / subset_type
    if not subset_dir.exists():
        print(f"Warning: {subset_dir} does not exist, skipping", file=sys.stderr)
        return []
    
    # Find all LMDB database directories
    lmdb_dirs = []
    for item in subset_dir.iterdir():
        if item.is_dir():
            # Check if directory contains LMDB files
            has_lmdb = (item / 'data.mdb').exists() or any(item.glob('*.aselmdb'))
            if has_lmdb:
                lmdb_dirs.append(item)
    
    if not lmdb_dirs:
        print(f"Warning: No LMDB databases found in {subset_dir}", file=sys.stderr)
        return []
    
    lmdb_dirs = sorted(lmdb_dirs)
    print(f"\nProcessing {subset_type} subset with {len(lmdb_dirs)} LMDB databases", file=sys.stderr)
    
    all_configs = []
    test_bool = (subset_type == 'val')
    
    # Process each LMDB database
    for lmdb_dir in tqdm(lmdb_dirs, desc=f"  {subset_type} databases", unit="db"):
        group_name = lmdb_dir.name
        
        try:
            configs, filtered = process_lmdb_dir(lmdb_dir, group_name, allowed_elements, test_bool)
            all_configs.extend(configs)
            tqdm.write(f"    {group_name}: kept {len(configs)}, filtered {filtered}")
        except Exception as e:
            tqdm.write(f"    {group_name}: ERROR - {e}")
            continue
    
    return all_configs


def write_adios2_file(configs, output_path, allowed_elements):
    """
    Write configurations to ADIOS2 .bp file.
    
    Data structure:
    - nconfigs: total number of configurations (scalar)
    - allowed_elements: string list of allowed elements (attribute)
    
    For each configuration i (arrays of length nconfigs):
    - Group[i]: group name (string)
    - NumAtoms[i]: number of atoms
    - Energy[i]: total energy
    - test_bool[i]: boolean (0=train, 1=val)
    - eweight[i], fweight[i], vweight[i]: weights
    
    Variable-length arrays (flattened with offsets):
    - PositionOffsets[i]: starting index in PositionsFlat for config i
    - PositionsFlat[3*total_atoms]: flattened positions
    - AtomTypesFlat[total_atoms]: flattened atom type indices
    - ForcesFlat[3*total_atoms]: flattened forces (if available)
    
    Fixed-size arrays (nconfigs, 9):
    - Lattice[nconfigs, 9]: lattice matrices (3x3 per config, flattened to 9)
    - Stress[nconfigs, 9]: stress tensors (3x3 per config, flattened to 9, if available)
    
    Element mapping (attributes):
    - element_map: maps element index to symbol
    """
    
    nconfigs = len(configs)
    print(f"\nWriting {nconfigs} configurations to {output_path}", file=sys.stderr)
    
    # Create element mapping
    element_list = sorted(allowed_elements)
    element_to_idx = {elem: idx for idx, elem in enumerate(element_list)}
    
    # Prepare arrays
    group_names = []
    num_atoms_array = np.zeros(nconfigs, dtype=np.int32)
    energy_array = np.zeros(nconfigs, dtype=np.float64)
    test_bool_array = np.zeros(nconfigs, dtype=np.int32)
    eweight_array = np.zeros(nconfigs, dtype=np.float64)
    fweight_array = np.zeros(nconfigs, dtype=np.float64)
    vweight_array = np.zeros(nconfigs, dtype=np.float64)
    
    # Variable-length data
    positions_list = []
    atom_types_list = []
    forces_list = []
    
    position_offsets = np.zeros(nconfigs + 1, dtype=np.int64)
    
    has_forces = any('Forces' in config for config in configs)
    has_stress = any('Stress' in config for config in configs)
    
    # Fixed-size arrays (nconfigs, 9) for 3x3 matrices
    lattice_array = np.zeros((nconfigs, 9), dtype=np.float64)
    stress_array = np.zeros((nconfigs, 9), dtype=np.float64) if has_stress else None
    
    total_atoms = 0
    
    print("  Flattening arrays...", file=sys.stderr)
    for i, config in enumerate(tqdm(configs, desc="  Preparing data", unit="config")):
        group_names.append(config['Group'])
        num_atoms_array[i] = config['NumAtoms']
        energy_array[i] = config['Energy']
        test_bool_array[i] = 1 if config['test_bool'] else 0
        eweight_array[i] = config['eweight']
        fweight_array[i] = config['fweight']
        vweight_array[i] = config['vweight']
        
        position_offsets[i] = total_atoms
        total_atoms += config['NumAtoms']
        
        # Flatten positions
        positions_list.append(config['Positions'].flatten())
        
        # Convert atom types to indices
        atom_type_indices = [element_to_idx[sym] for sym in config['AtomTypes']]
        atom_types_list.append(np.array(atom_type_indices, dtype=np.int32))
        
        # Flatten forces if available
        if has_forces:
            if 'Forces' in config:
                forces_list.append(config['Forces'].flatten())
            else:
                forces_list.append(np.zeros(config['NumAtoms'] * 3, dtype=np.float64))
        
        # Store lattice (3x3 -> 9) directly in array
        lattice_array[i] = config['Lattice'].flatten()
        
        # Store stress if available (3x3 -> 9) directly in array
        if has_stress:
            if 'Stress' in config:
                stress_array[i] = config['Stress'].flatten()
            else:
                stress_array[i] = np.zeros(9, dtype=np.float64)
    
    position_offsets[-1] = total_atoms
    
    # Concatenate variable-length arrays
    print("  Concatenating arrays...", file=sys.stderr)
    positions_flat = np.concatenate(positions_list)
    atom_types_flat = np.concatenate(atom_types_list)
    
    if has_forces:
        forces_flat = np.concatenate(forces_list)
    
    print(f"  Total atoms across all configs: {total_atoms}", file=sys.stderr)
    print(f"  Has forces: {has_forces}", file=sys.stderr)
    print(f"  Has stress: {has_stress}", file=sys.stderr)
    
    # Write to ADIOS2
    print("  Writing to ADIOS2...", file=sys.stderr)
    with Stream(output_path, 'w') as s:
        # Write metadata attributes
        s.write_attribute('nconfigs', nconfigs)
        s.write_attribute('element_map', ','.join(element_list))
        s.write_attribute('has_forces', 1 if has_forces else 0)
        s.write_attribute('has_stress', 1 if has_stress else 0)
        
        # Write per-config arrays
        s.write('NumAtoms', num_atoms_array)
        s.write('Energy', energy_array)
        s.write('test_bool', test_bool_array)
        s.write('eweight', eweight_array)
        s.write('fweight', fweight_array)
        s.write('vweight', vweight_array)
        
        # Write group names as a single concatenated string with delimiters
        group_string = '|'.join(group_names)
        s.write_attribute('group_names', group_string)
        
        # Write variable-length data
        s.write('PositionOffsets', position_offsets)
        s.write('PositionsFlat', positions_flat)
        s.write('AtomTypesFlat', atom_types_flat)
        
        # Write fixed-size arrays
        s.write('Lattice', lattice_array)
        
        if has_forces:
            s.write('ForcesFlat', forces_flat)
        if has_stress:
            s.write('Stress', stress_array)
    
    print(f"\nSuccessfully wrote {output_path}", file=sys.stderr)
    print(f"  File contains {nconfigs} configurations", file=sys.stderr)
    print(f"  Elements: {', '.join(element_list)}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description='Convert FAIRChem LMDB datasets to ADIOS2 format with element filtering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python fairchem_to_adios2.py --dataset scratch/omat24 --elements Al Ni --output omat24_AlNi.bp
    
Expected directory structure:
    scratch/omat24/
    ├── train/
    │   ├── rattled-300.tar.gz
    │   ├── rattled-500.tar.gz
    │   └── ...
    └── val/
        ├── rattled-300.tar.gz
        ├── rattled-500.tar.gz
        └── ...
        """
    )
    
    parser.add_argument('--dataset', required=True, type=str,
                        help='Path to dataset root directory (contains train/ and val/ subdirs)')
    parser.add_argument('--elements', required=True, nargs='+', type=str,
                        help='Allowed element symbols (e.g., Al Ni)')
    parser.add_argument('--output', required=True, type=str,
                        help='Output ADIOS2 .bp file path')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: cpu_count)')
    
    args = parser.parse_args()
    
    # Validate inputs
    dataset_root = Path(args.dataset)
    if not dataset_root.exists():
        print(f"ERROR: Dataset path does not exist: {dataset_root}", file=sys.stderr)
        sys.exit(1)
    
    allowed_elements = set(args.elements)
    print(f"Allowed elements: {', '.join(sorted(allowed_elements))}", file=sys.stderr)
    
    if args.workers:
        print(f"Using {args.workers} parallel workers", file=sys.stderr)
    else:
        print(f"Using {mp.cpu_count()} parallel workers (all CPUs)", file=sys.stderr)
    
    # Process train and val subsets
    all_configs = []
    
    train_configs = process_dataset_path(dataset_root, 'train', allowed_elements, args.workers)
    all_configs.extend(train_configs)
    print(f"\nCollected {len(train_configs)} training configurations", file=sys.stderr)
    
    val_configs = process_dataset_path(dataset_root, 'val', allowed_elements, args.workers)
    all_configs.extend(val_configs)
    print(f"Collected {len(val_configs)} validation configurations", file=sys.stderr)
    
    if not all_configs:
        print("ERROR: No configurations collected!", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nTotal configurations: {len(all_configs)}", file=sys.stderr)
    
    # Write to ADIOS2
    write_adios2_file(all_configs, args.output, allowed_elements)


if __name__ == '__main__':
    main()
