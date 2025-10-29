#!/usr/bin/env python3
"""
Convert FAIRChem LMDB datasets to ADIOS2 .bp format with element filtering.

Reads LMDB databases directly using the lmdb package and ASE, avoiding torch dependency.
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
    from ase.io import read
    from ase import Atoms
    import lmdb
    import pickle
    HAS_ASE = True
except ImportError as e:
    HAS_ASE = False
    print("="*80, file=sys.stderr)
    print("ERROR: Failed to import required packages", file=sys.stderr)
    print("="*80, file=sys.stderr)
    print(f"\nImport error: {e}\n", file=sys.stderr)
    print("Install required packages: pip install ase lmdb\n", file=sys.stderr)
    print("="*80, file=sys.stderr)
    sys.exit(1)


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
    
    # Find all LMDB database directories (containing data.mdb)
    lmdb_dirs = []
    for item in subset_dir.iterdir():
        if item.is_dir() and (item / 'data.mdb').exists():
            lmdb_dirs.append(item)
    
    if not lmdb_dirs:
        print(f"Warning: No LMDB databases found in {subset_dir}", file=sys.stderr)
        return []
    
    lmdb_dirs = sorted(lmdb_dirs)
    print(f"\nProcessing {subset_type} subset with {len(lmdb_dirs)} LMDB databases", file=sys.stderr)
    
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    all_configs = []
    test_bool = (subset_type == 'val')
    
    # Process each LMDB database with progress bar
    for lmdb_dir in tqdm(lmdb_dirs, desc=f"  {subset_type} databases", unit="db"):
        group_name = lmdb_dir.name
        lmdb_path = str(lmdb_dir)
        
        # Open LMDB to get size
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin() as txn:
            dataset_size = txn.stat()['entries']
        env.close()
        
        # Create chunks for parallel processing
        chunk_size = max(100, dataset_size // (num_workers * 4))  # 4 chunks per worker
        chunks = []
        for i in range(0, dataset_size, chunk_size):
            end_idx = min(i + chunk_size, dataset_size)
            chunks.append((lmdb_path, i, end_idx, group_name, test_bool, allowed_elements))
        
        # Process chunks in parallel
        archive_configs = []
        total_filtered = 0
        
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_config_chunk, chunks),
                total=len(chunks),
                desc=f"    {group_name}",
                unit="chunk",
                leave=False
            ))
        
        # Collect results
        for configs, filtered_count in results:
            archive_configs.extend(configs)
            total_filtered += filtered_count
        
        all_configs.extend(archive_configs)
        
        tqdm.write(f"    {group_name}: kept {len(archive_configs)}, filtered {total_filtered}")
    
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
