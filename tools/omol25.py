import os
os.environ['NUMEXPR_MAX_THREADS'] = '14'
os.environ['OMP_NUM_THREADS'] = '14'

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchtnt')
warnings.filterwarnings('ignore', category=UserWarning, message='.*pkg_resources.*')
warnings.filterwarnings('ignore', message='.*Redirects are currently not supported.*')

from fairchem.core.datasets import AseDBDataset
import pandas as pd
import sys
import multiprocessing as mp
from tqdm import tqdm

# Mapping of element symbols to atomic numbers for sorting
ATOMIC_NUMBERS = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
    'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109,
    'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
}

def process_chunk(args):
    """Worker function to process a chunk of the dataset"""
    db_path, start_idx, end_idx = args
    dataset = AseDBDataset(config=dict(src=db_path))
    
    data_rows = []
    try:
        for i in range(start_idx, end_idx):
            atoms = dataset.get_atoms(i)
            
            # Get unique elements and sort by atomic number
            unique_elements = sorted(set(atoms.get_chemical_symbols()), 
                                   key=lambda x: ATOMIC_NUMBERS.get(x, 999))
            elements_str = ' '.join(unique_elements)
            
            # Extract data from atoms.info
            data_id = atoms.info.get('data_id', '')
            charge = atoms.info.get('charge', '')
            spin = atoms.info.get('spin', '')
            num_atoms = atoms.info.get('num_atoms', '')
            composition = atoms.info.get('composition', '')
            
            data_rows.append({
                'data_id': data_id,
                'elements': elements_str,
                'charge': charge,
                'spin': spin,
                'num_atoms': num_atoms,
                'composition': composition
            })
    finally:
        del dataset
    
    return data_rows

def process_neutral_val(db_path):
    """Process neutral_val database and create Excel output"""
    
    print(f"Loading database: {db_path}", file=sys.stderr)
    
    # Expand path
    db_path = os.path.expanduser(db_path)
    
    if not os.path.exists(db_path):
        print(f"Error: Database path does not exist: {db_path}", file=sys.stderr)
        return
    
    # Load dataset to get size
    dataset = AseDBDataset(config=dict(src=db_path))
    dataset_size = len(dataset)
    del dataset  # Free memory
    
    print(f"Dataset size: {dataset_size} configurations", file=sys.stderr)
    
    # Setup parallel processing
    num_cores = mp.cpu_count()
    chunk_size = max(1000, dataset_size // num_cores)  # Minimum 1000 per chunk
    
    # Create chunks
    chunks = []
    for i in range(0, dataset_size, chunk_size):
        end_idx = min(i + chunk_size, dataset_size)
        chunks.append((db_path, i, end_idx))
    
    print(f"Processing with {num_cores} cores in {len(chunks)} chunks...", file=sys.stderr)
    
    # Process chunks in parallel with progress bar
    all_data_rows = []
    with mp.Pool(processes=num_cores) as pool:
        with tqdm(total=len(chunks), desc="Processing chunks", unit="chunk") as pbar:
            for chunk_data in pool.imap_unordered(process_chunk, chunks):
                all_data_rows.extend(chunk_data)
                pbar.update(1)
    
    print(f"Processed all {dataset_size} entries", file=sys.stderr)
    
    # Create DataFrame
    df = pd.DataFrame(all_data_rows)
    
    # Group by composition/charge/spin and aggregate
    print("Aggregating by composition/charge/spin...", file=sys.stderr)
    df_grouped = df.groupby(['data_id', 'composition', 'charge', 'spin'], dropna=False).agg({
        'elements': 'first',
        'num_atoms': 'first'
    }).reset_index()
    
    # Add count column
    df_grouped['count'] = df.groupby(['data_id', 'composition', 'charge', 'spin'], dropna=False).size().values
    
    # Reorder columns
    df_grouped = df_grouped[['data_id', 'elements', 'charge', 'spin', 'num_atoms', 'composition', 'count']]
    
    print(f"Reduced to {len(df_grouped)} unique combinations (from {len(df)} total entries)", file=sys.stderr)
    
    # Group by data_id
    data_ids = df_grouped['data_id'].unique()
    print(f"Found {len(data_ids)} unique data_id values", file=sys.stderr)
    
    # Use last part of db_path for output filename
    db_name = os.path.basename(db_path.rstrip('/'))
    output_file = f'{db_name}.xlsx'
    print(f"Saving results to {output_file}...", file=sys.stderr)
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for data_id in sorted(data_ids):
            # Filter data for this data_id
            df_subset = df_grouped[df_grouped['data_id'] == data_id].reset_index(drop=True)
            
            # Use data_id as sheet name (max 31 chars for Excel)
            sheet_name = str(data_id)[:31]
            
            print(f"  Writing sheet '{sheet_name}' with {len(df_subset)} entries", file=sys.stderr)
            df_subset.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Auto-adjust column widths
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)  # Max width of 50
                worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"Results saved to {output_file}", file=sys.stderr)
    print(f"Total unique combinations: {len(df_grouped)}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python omol25.py <db_path>", file=sys.stderr)
        print("Example: python omol25.py ~/scratch/neutral_val", file=sys.stderr)
        sys.exit(1)
    
    db_path = sys.argv[1]
    process_neutral_val(db_path)
