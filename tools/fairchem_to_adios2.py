#!/usr/bin/env python3
"""
Convert FAIRChem LMDB datasets to ADIOS2 .bp format with element filtering.

Uses fairchem's AseDBDataset to read LMDB databases (mocks torch if not installed).
Expects LMDB databases to be already extracted in train/ and val/ directories.

OMOL-style data may omit a usable periodic cell (zeros / singular). The converter then writes
an orthorhombic ``Lattice`` from atomic coordinates plus ``--lattice-pad`` / ``--lattice-min-side``,
which is what FitSNAP/LAMMPS use for box size.

Usage:
    python fairchem_to_adios2.py --dataset scratch/omat24 --elements Al Ni --output omat24_AlNi.bp
    python fairchem_to_adios2.py --dataset scratch/omol25_4M --elements C H O --output omol25_CHO.bp
"""

import os
import sys

# Redirect stderr to devnull to suppress torch warnings
_original_stderr = os.dup(2)
_devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull, 2)

import argparse
import numpy as np
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import warnings

try:
    from adios2 import Stream
    HAS_ADIOS2 = True
except ImportError as e:
    HAS_ADIOS2 = False
    # Restore stderr for error messages
    os.dup2(_original_stderr, 2)
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
    # Restore stderr for error messages
    os.dup2(_original_stderr, 2)
    print("="*80, file=sys.stderr)
    print("ERROR: Failed to import fairchem", file=sys.stderr)
    print("="*80, file=sys.stderr)
    print(f"\nImport error: {e}\n", file=sys.stderr)
    print("Install: pip install fairchem-core\n", file=sys.stderr)
    print("="*80, file=sys.stderr)
    sys.exit(1)

# Restore stderr after imports
os.dup2(_original_stderr, 2)
os.close(_devnull)
os.close(_original_stderr)


# Global variables for worker processes
_worker_dataset = None


def _ensure_lattice_for_adios2(
    cell,
    positions,
    forces=None,
    stress=None,
    *,
    pad_angstrom=10.0,
    min_side_angstrom=8.0,
):
    """
    FitSNAP expects a full-rank, right-handed 3x3 lattice (rows = ASE cell vectors, Å).

    OMOL25 / SPICE-style LMDB entries often carry **no meaningful cell** (zeros or rank-deficient
    ``get_cell()``), so there is nothing to "read" for LAMMPS box size. In that case we **define**
    the simulation box as an axis-aligned cell: edge lengths = max(span + 2*pad, min_side) per
    axis from Cartesian positions, then shift positions so the lower corner is the origin.

    OMAT24-style periodic configs keep their stored cell when it is full-rank; if ``det < 0``
    (left-handed), we apply a z-reflection on lattice rows and matching flips on positions,
    forces, and stress.
    """
    cell = np.asarray(cell, dtype=np.float64).reshape(3, 3).copy()
    pos = np.asarray(positions, dtype=np.float64).copy()
    frc = None if forces is None else np.asarray(forces, dtype=np.float64).copy()
    sig = None if stress is None else np.asarray(stress, dtype=np.float64).reshape(3, 3).copy()

    det = float(np.linalg.det(cell))
    rank = int(np.linalg.matrix_rank(cell, tol=1e-8))
    need_box = (not np.isfinite(det)) or abs(det) < 1e-6 or rank < 3

    if need_box:
        lo = pos.min(axis=0) - pad_angstrom
        hi = pos.max(axis=0) + pad_angstrom
        span = np.maximum(hi - lo, min_side_angstrom)
        cell = np.diag(span.astype(np.float64))
        pos -= lo
        det = float(np.linalg.det(cell))
    elif det < 0:
        drefl = np.diag([1.0, 1.0, -1.0])
        cell = cell @ drefl
        pos[:, 2] *= -1.0
        if frc is not None:
            frc[:, 2] *= -1.0
        if sig is not None:
            sig = drefl @ sig @ drefl.T
        det = float(np.linalg.det(cell))

    if not np.isfinite(det) or det <= 0:
        raise RuntimeError(
            f"_ensure_lattice_for_adios2: invalid cell after fix (det={det}, rank={np.linalg.matrix_rank(cell)})"
        )

    return cell, pos, frc, sig


def _init_worker(lmdb_path):
    """Initialize worker process with dataset."""
    global _worker_dataset
    _worker_dataset = AseDBDataset({"src": str(lmdb_path)})

def _process_chunk(args):
    """
    Process a chunk of configuration indices.
    
    Args:
        args: Tuple of (start_idx, end_idx, group_name, allowed_elements, test_bool,
            lattice_pad, lattice_min_side)
    
    Returns:
        Tuple of (list of configs, number filtered)
    """
    start_idx, end_idx, group_name, allowed_elements, test_bool, lattice_pad, lattice_min_side = args
    
    configs = []
    filtered_count = 0
    
    for i in range(start_idx, end_idx):
        try:
            atoms = _worker_dataset.get_atoms(i)
        except Exception as e:
            continue
        
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

        cell, positions, forces, stress = _ensure_lattice_for_adios2(
            cell,
            positions,
            forces,
            stress,
            pad_angstrom=lattice_pad,
            min_side_angstrom=lattice_min_side,
        )

        config = {
            'Group': group_name,
            'NumAtoms': num_atoms,
            'Positions': positions,
            'AtomTypes': atom_types,
            'Lattice': cell,
            'Energy': energy,
            'test_bool': test_bool,
        }
        
        # OMOL25 dataset
        if hasattr(atoms, 'info') and isinstance(atoms.info, dict) and 'data_id' in atoms.info:
            config['Group'] = atoms.info.get('data_id', '')
            config['Charge'] = atoms.info.get('charge', '')
            config['Spin'] = atoms.info.get('spin', '')
            config['Composition'] = atoms.info.get('composition', '')
        
        if forces is not None:
            config['Forces'] = forces
        if stress is not None:
            config['Stress'] = stress
        
        configs.append(config)
    
    return configs, filtered_count


def _is_lmdb_dir(path: Path) -> bool:
    """True if path looks like an AseDB / LMDB root (data.mdb or *.aselmdb)."""
    if not path.is_dir():
        return False
    if (path / "data.mdb").exists():
        return True
    return any(path.glob("*.aselmdb"))


def _discover_lmdb_roots(subset_dir: Path) -> list[tuple[Path, str]]:
    """
    Find LMDB roots under train/ or val/.

    omat24: each group is a subdirectory of train/val containing an LMDB.
    omol25: train/ and val/ are each a single LMDB directory; logical groups
    come from atoms.info['data_id'] (applied in _process_chunk).
    """
    if not subset_dir.is_dir():
        return []

    if _is_lmdb_dir(subset_dir):
        return [(subset_dir, subset_dir.name)]

    roots: list[tuple[Path, str]] = []
    for item in sorted(subset_dir.iterdir()):
        if item.is_dir() and _is_lmdb_dir(item):
            roots.append((item, item.name))
    return roots


def process_lmdb_dir(
    lmdb_path,
    group_name,
    allowed_elements,
    test_bool,
    num_workers=None,
    lattice_pad=10.0,
    lattice_min_side=8.0,
):
    """
    Process a single LMDB database directory using fairchem's AseDBDataset with multiprocessing.
    
    Args:
        lmdb_path: Path to LMDB database directory
        group_name: Name for this group of configurations
        allowed_elements: Set of allowed element symbols
        test_bool: Boolean indicating if this is validation data
        num_workers: Number of parallel workers (default: cpu_count)
    
    Returns:
        Tuple of (list of configs, number filtered)
    """
    # AseDBDataset expects a config dict with 'src' pointing to the database
    dataset = AseDBDataset({"src": str(lmdb_path)})
    dataset_size = len(dataset)
    
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    # Split dataset into chunks
    chunk_size = max(1, dataset_size // num_workers)
    chunks = []
    for i in range(0, dataset_size, chunk_size):
        start_idx = i
        end_idx = min(i + chunk_size, dataset_size)
        chunks.append(
            (
                start_idx,
                end_idx,
                group_name,
                allowed_elements,
                test_bool,
                lattice_pad,
                lattice_min_side,
            )
        )
    
    # Process chunks in parallel with global progress bar
    all_configs = []
    total_filtered = 0
    
    with mp.Pool(processes=num_workers, initializer=_init_worker, initargs=(lmdb_path,)) as pool:
        with tqdm(total=dataset_size, desc=f"  {group_name}", unit="config") as pbar:
            for configs, filtered in pool.imap_unordered(_process_chunk, chunks):
                all_configs.extend(configs)
                total_filtered += filtered
                pbar.update(len(configs) + filtered)
    
    return all_configs, total_filtered


def process_dataset_path(
    dataset_root,
    subset_type,
    allowed_elements,
    num_workers=None,
    lattice_pad=10.0,
    lattice_min_side=8.0,
):
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

    lmdb_entries = _discover_lmdb_roots(subset_dir)
    if not lmdb_entries:
        print(f"Warning: No LMDB databases found in {subset_dir}", file=sys.stderr)
        return []

    layout = "flat (e.g. omol25)" if len(lmdb_entries) == 1 and lmdb_entries[0][0] == subset_dir else "per-group subdirs (e.g. omat24)"
    print(
        f"\nProcessing {subset_type} subset: {len(lmdb_entries)} LMDB root(s), layout={layout}",
        file=sys.stderr,
    )

    all_configs = []
    test_bool = (subset_type == 'val')

    for lmdb_dir, group_name in lmdb_entries:
        try:
            configs, filtered = process_lmdb_dir(
                lmdb_dir,
                group_name,
                allowed_elements,
                test_bool,
                num_workers,
                lattice_pad=lattice_pad,
                lattice_min_side=lattice_min_side,
            )
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
    - unique_group_names: pipe-delimited string of unique group names (attribute)
    
    For each configuration i (arrays of length nconfigs):
    - GroupIndices[i]: index into unique_group_names array
    - NumAtoms[i]: number of atoms
    - Energy[i]: total energy
    - test_bool[i]: boolean (0=train, 1=val)
    
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
    
    # Variable-length data
    positions_list = []
    atom_types_list = []
    forces_list = []
    
    position_offsets = np.zeros(nconfigs, dtype=np.int64)
    
    has_forces = any('Forces' in config for config in configs)
    has_stress = any('Stress' in config for config in configs)
    
    # Fixed-size arrays (nconfigs, 3, 3) for 3x3 matrices
    lattice_array = np.zeros((nconfigs, 3, 3), dtype=np.float64)
    stress_array = np.zeros((nconfigs, 3, 3), dtype=np.float64) if has_stress else None
    
    total_atoms = 0
    
    print("  Flattening arrays...", file=sys.stderr)
    for i, config in enumerate(tqdm(configs, desc="  Preparing data", unit="config")):
        group_names.append(config['Group'])
        num_atoms_array[i] = config['NumAtoms']
        energy_array[i] = config['Energy']
        test_bool_array[i] = 1 if config['test_bool'] else 0
        
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
        
        # Store lattice (3x3) directly in array
        lattice_array[i] = config['Lattice']
        
        # Store stress if available (3x3) directly in array
        if has_stress:
            if 'Stress' in config:
                stress_array[i] = config['Stress']
            else:
                stress_array[i] = np.zeros(3, 3, dtype=np.float64)
    
    # Concatenate variable-length arrays
    print("  Concatenating arrays...", file=sys.stderr)
    positions_flat = np.concatenate(positions_list)
    atom_types_flat = np.concatenate(atom_types_list)
    
    if has_forces:
        forces_flat = np.concatenate(forces_list)
    
    print(f"  Total atoms across all configs: {total_atoms}", file=sys.stderr)
    print(f"  Has forces: {has_forces}", file=sys.stderr)
    print(f"  Has stress: {has_stress}", file=sys.stderr)

    dets = np.linalg.det(lattice_array)
    if np.any(~np.isfinite(dets)) or np.any(dets <= 0):
        nbad = int(np.sum(~np.isfinite(dets) | (dets <= 0)))
        raise RuntimeError(
            f"Lattice verification failed: {nbad}/{nconfigs} configs have det(Lattice) <= 0 or non-finite"
        )
    print(
        f"  Lattice det check: min={float(np.min(dets)):.6g} max={float(np.max(dets)):.6g} (Å³, all > 0)",
        file=sys.stderr,
    )
    
    # Write to ADIOS2
    print("  Writing to ADIOS2...", file=sys.stderr)
    with Stream(output_path, 'w') as s:
        # Write metadata attributes
        s.write_attribute('nconfigs', nconfigs)
        s.write_attribute('element_map', ','.join(element_list))
        s.write_attribute('has_forces', 1 if has_forces else 0)
        s.write_attribute('has_stress', 1 if has_stress else 0)
        
        # Write per-config arrays
        s.write('NumAtoms', num_atoms_array, count=[nconfigs])
        s.write('Energy', energy_array, count=[nconfigs])
        s.write('test_bool', test_bool_array, count=[nconfigs])
        
        # Create unique group names list and indices array
        unique_groups = sorted(set(group_names))
        group_to_idx = {name: idx for idx, name in enumerate(unique_groups)}
        group_indices = np.array([group_to_idx[name] for name in group_names], dtype=np.int32)
        
        # Write unique group names and indices
        unique_groups_string = '|'.join(unique_groups)
        s.write_attribute('unique_group_names', unique_groups_string)
        s.write('GroupIndices', group_indices, count=[nconfigs])
        
        # Write variable-length data
        s.write('PositionOffsets', position_offsets, count=[nconfigs])
        s.write('PositionsFlat', positions_flat, count=[total_atoms,3])
        s.write('AtomTypesFlat', atom_types_flat, count=[total_atoms])
        
        # Write fixed-size arrays
        s.write('Lattice', lattice_array, count=[nconfigs,3,3])
        
        if has_forces:
            s.write('ForcesFlat', forces_flat, count=[total_atoms,3])
        if has_stress:
            s.write('Stress', stress_array, count=[nconfigs,3,3])
    
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
    
Expected directory structures:

    omat24 (one LMDB per group under train/val):
    scratch/omat24/
    ├── train/
    │   ├── rattled-300/
    │   ├── rattled-500/
    │   └── ...
    └── val/
        └── ...

    omol25 (single LMDB per split; groups from atoms.info['data_id']):
    scratch/omol25_4M/
    ├── train/    # LMDB files live here
    └── val/
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
    parser.add_argument(
        '--lattice-pad',
        type=float,
        default=10.0,
        metavar='ANG',
        help='Vacuum padding (Å) on each side when synthesizing a box from positions (OMOL / missing cell)',
    )
    parser.add_argument(
        '--lattice-min-side',
        type=float,
        default=8.0,
        metavar='ANG',
        help='Minimum orthorhombic edge (Å) when synthesizing a cell from positions',
    )

    args = parser.parse_args()
    
    # Validate inputs
    dataset_root = Path(args.dataset)
    if not dataset_root.exists():
        print(f"ERROR: Dataset path does not exist: {dataset_root}", file=sys.stderr)
        sys.exit(1)
    
    allowed_elements = set(args.elements)
    print(f"Allowed elements: {', '.join(sorted(allowed_elements))}", file=sys.stderr)
    print(
        f"Synthetic LAMMPS box (when cell missing/singular): pad={args.lattice_pad} Å, "
        f"min edge={args.lattice_min_side} Å",
        file=sys.stderr,
    )
    
    if args.workers:
        print(f"Using {args.workers} parallel workers", file=sys.stderr)
    else:
        print(f"Using {mp.cpu_count()} parallel workers (all CPUs)", file=sys.stderr)
    
    # Process train and val subsets
    all_configs = []
    
    train_configs = process_dataset_path(
        dataset_root,
        'train',
        allowed_elements,
        args.workers,
        lattice_pad=args.lattice_pad,
        lattice_min_side=args.lattice_min_side,
    )
    all_configs.extend(train_configs)
    print(f"\nCollected {len(train_configs)} training configurations", file=sys.stderr)
    
    val_configs = process_dataset_path(
        dataset_root,
        'val',
        allowed_elements,
        args.workers,
        lattice_pad=args.lattice_pad,
        lattice_min_side=args.lattice_min_side,
    )
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
