#!/usr/bin/env python3
"""
Convert FAIRChem LMDB datasets to ADIOS2 .bp format with element filtering.

Usage:
    python fairchem_to_adios2.py --dataset scratch/omat24 --elements Al Ni --output omat24_AlNi.bp
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import tarfile
import tempfile
import shutil
import multiprocessing as mp
from tqdm import tqdm

try:
    from adios2 import Stream
    HAS_ADIOS2 = True
except ImportError:
    HAS_ADIOS2 = False
    print("ERROR: adios2 not found. Install with: pip install adios2", file=sys.stderr)
    sys.exit(1)

try:
    from fairchem.core.datasets import AseDBDataset
    HAS_FAIRCHEM = True
except ImportError:
    HAS_FAIRCHEM = False
    print("ERROR: fairchem-core not found. Install with: pip install fairchem-core", file=sys.stderr)
    sys.exit(1)


def extract_tar_to_temp(tar_path):
    """Extract .tar.gz archive to temporary directory and return path to LMDB."""
    temp_dir = tempfile.mkdtemp(prefix='fairchem_lmdb_')
    
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(temp_dir)
    
    # Find the LMDB directory (usually just the extracted folder)
    extracted = list(Path(temp_dir).iterdir())
    if len(extracted) == 1 and extracted[0].is_dir():
        lmdb_path = str(extracted[0])
    else:
        lmdb_path = temp_dir
    
    return lmdb_path, temp_dir


def process_config_chunk(args):
    """
    Worker function to process a chunk of configurations.
    
    Args:
        args: Tuple of (lmdb_path, start_idx, end_idx, group_name, test_bool, allowed_elements)
    
    Returns:
        List of configuration dictionaries
    """
    lmdb_path, start_idx, end_idx, group_name, test_bool, allowed_elements = args
    
    # Load dataset in worker
    dataset = AseDBDataset(config=dict(src=lmdb_path))
    
    configs = []
    filtered_count = 0
    
    try:
        for idx in range(start_idx, end_idx):
            try:
                atoms = dataset.get_atoms(idx)
                
                # Get chemical symbols
                symbols = atoms.get_chemical_symbols()
                
                # Filter by allowed elements
                if not all(s in allowed_elements for s in symbols):
                    filtered_count += 1
                    continue
                
                # Extract data
                positions = atoms.get_positions()
                cell = atoms.get_cell()
                
                # Energy
                energy = None
                if hasattr(atoms, 'get_total_energy'):
                    try:
                        energy = atoms.get_total_energy()
                    except:
                        energy = atoms.info.get('energy', None)
                else:
                    energy = atoms.info.get('energy', None)
                
                if energy is None:
                    continue
                
                # Forces
                forces = None
                if hasattr(atoms, 'get_forces'):
                    try:
                        forces = atoms.get_forces()
                    except:
                        forces = atoms.arrays.get('forces', None)
                else:
                    forces = atoms.arrays.get('forces', None)
                
                # Stress
                stress = None
                if hasattr(atoms, 'get_stress'):
                    try:
                        stress = atoms.get_stress(voigt=False)  # Get full 3x3 tensor
                    except:
                        stress = atoms.info.get('stress', None)
                else:
                    stress = atoms.info.get('stress', None)
                
                # Handle cell/lattice
                cell_array = cell.array.copy() if hasattr(cell, 'array') else np.array(cell)
                
                # Check if we need to create a box for non-periodic systems
                if cell_array.shape != (3, 3) or np.allclose(cell_array, 0.0, atol=1e-6):
                    max_coord = np.max(np.abs(positions)) + 10.0
                    cell_array = np.diag([max_coord * 2, max_coord * 2, max_coord * 2])
                
                # Transpose to convert from ASE format (rows) to FitSNAP format (columns)
                lattice = cell_array.T
                
                # Create config dictionary
                config_dict = {
                    'Group': group_name,
                    'Positions': positions.astype(np.float64),
                    'AtomTypes': symbols,
                    'NumAtoms': len(symbols),
                    'Lattice': lattice.astype(np.float64),
                    'Energy': float(energy),
                    'test_bool': test_bool,
                    'eweight': 1.0,
                    'fweight': 1.0 / len(symbols),
                    'vweight': 1.0 / 6.0,
                }
                
                if forces is not None:
                    config_dict['Forces'] = forces.astype(np.float64)
                
                if stress is not None:
                    # Convert stress to 3x3 if it's in Voigt notation
                    if stress.shape == (6,):
                        stress_tensor = np.array([
                            [stress[0], stress[5], stress[4]],
                            [stress[5], stress[1], stress[3]],
                            [stress[4], stress[3], stress[2]]
                        ])
                        config_dict['Stress'] = stress_tensor.astype(np.float64)
                    else:
                        config_dict['Stress'] = stress.astype(np.float64)
                
                configs.append(config_dict)
                
            except Exception as e:
                # Silently skip failed configs
                continue
    
    finally:
        # Clean up dataset
        del dataset
    
    return configs, filtered_count


def process_dataset_path(dataset_root, subset_type, allowed_elements, num_workers=None):
    """
    Process train or val directory containing .tar.gz archives.
    
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
    
    # Find all .tar.gz files
    tar_files = sorted(subset_dir.glob('*.tar.gz'))
    if not tar_files:
        print(f"Warning: No .tar.gz files found in {subset_dir}", file=sys.stderr)
        return []
    
    print(f"\nProcessing {subset_type} subset with {len(tar_files)} archives", file=sys.stderr)
    
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    all_configs = []
    test_bool = (subset_type == 'val')
    
    # Process each archive with progress bar
    for tar_path in tqdm(tar_files, desc=f"  {subset_type} archives", unit="archive"):
        # Extract group name from filename (e.g., 'rattled-300-subsampled.tar.gz' -> 'rattled-300')
        filename = tar_path.stem  # Remove .tar.gz
        if filename.endswith('.tar'):
            filename = filename[:-4]
        
        # Remove common suffixes
        group_name = filename.replace('-subsampled', '').replace('_subsampled', '')
        
        # Extract to temporary directory
        lmdb_path, temp_dir = extract_tar_to_temp(tar_path)
        
        try:
            # Load dataset to get size
            dataset = AseDBDataset(config=dict(src=lmdb_path))
            dataset_size = len(dataset)
            del dataset  # Free before forking
            
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
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
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
    - LatticeFlat[9*nconfigs]: flattened lattice matrices (3x3 per config)
    - StressFlat[9*nconfigs]: flattened stress tensors (3x3 per config, if available)
    
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
    lattices_list = []
    stresses_list = []
    
    position_offsets = np.zeros(nconfigs + 1, dtype=np.int64)
    
    has_forces = any('Forces' in config for config in configs)
    has_stress = any('Stress' in config for config in configs)
    
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
        
        # Flatten lattice (3x3 -> 9)
        lattices_list.append(config['Lattice'].flatten())
        
        # Flatten stress if available
        if has_stress:
            if 'Stress' in config:
                stresses_list.append(config['Stress'].flatten())
            else:
                stresses_list.append(np.zeros(9, dtype=np.float64))
    
    position_offsets[-1] = total_atoms
    
    # Concatenate variable-length arrays
    print("  Concatenating arrays...", file=sys.stderr)
    positions_flat = np.concatenate(positions_list)
    atom_types_flat = np.concatenate(atom_types_list)
    lattices_flat = np.concatenate(lattices_list)
    
    if has_forces:
        forces_flat = np.concatenate(forces_list)
    if has_stress:
        stresses_flat = np.concatenate(stresses_list)
    
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
        s.write('LatticesFlat', lattices_flat)
        
        if has_forces:
            s.write('ForcesFlat', forces_flat)
        if has_stress:
            s.write('StressesFlat', stresses_flat)
    
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
    │   ├── rattled-300-subsampled.tar.gz
    │   ├── rattled-1000-subsampled.tar.gz
    │   └── ...
    └── val/
        ├── rattled-300-subsampled.tar.gz
        ├── rattled-1000-subsampled.tar.gz
        └── ...
        """
    )
    
    parser.add_argument('--dataset', required=True, type=str,
                        help='Path to dataset root directory (contains train/ and val/ subdirs)')
    parser.add_argument('--elements', required=True, nargs='+', type=str,
                        help='Allowed element symbols (e.g., Al Ni)')
    parser.add_argument('--output', required=True, type=str,
                        help='Output ADIOS2 .bp file path')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training data')
    parser.add_argument('--skip-val', action='store_true',
                        help='Skip validation data')
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
    
    if not args.skip_train:
        train_configs = process_dataset_path(dataset_root, 'train', allowed_elements, args.workers)
        all_configs.extend(train_configs)
        print(f"\nCollected {len(train_configs)} training configurations", file=sys.stderr)
    
    if not args.skip_val:
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
