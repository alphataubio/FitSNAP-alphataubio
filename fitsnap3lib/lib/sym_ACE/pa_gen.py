

# GEMINI PRO 3's Simplified Direct Generation Wrapper (pa_gen.py)
# Removed the complex serialization (build_tabulated)
# and remapping (lammps_remap) pipelines.
# https://github.com/FitSNAP/FitSNAP/pull/278#issuecomment-3614308257
# [alphataubio, 2025/12]



import os
import json
import itertools
from fitsnap3lib.lib.sym_ACE.pa_lib import apply_ladder_relationships
from fitsnap3lib.lib.sym_ACE.sym_ACE_settings import lib_path

# Ensure the library path exists for caching
if not os.path.exists(lib_path):
    # Fallback to current directory if lib_path is not configured
    lib_path = os.path.dirname(os.path.abspath(__file__))

def get_cache_filename(rank, nmax, lmax, mumax):
    """Generates a unique filename for the basis set parameters."""
    return os.path.join(lib_path, f"basis_cache_r{rank}_n{nmax}_l{lmax}_mu{mumax}.json")

def build_and_cache_basis(rank, nmax, lmax, mumax, lmin=1, L_R=0):
    """
    The modern 'buildtabulated'.
    Generates the full independent basis set using the pa_lib v2 engine
    and saves it to disk to prevent re-calculation.
    """
    print(f"*** Generating Basis Cache for Rank={rank}, Nmax={nmax}, Lmax={lmax}, Mumax={mumax}...")
    
    all_lammps_labs = []
    
    # Define ranges
    # mu: 0 .. mumax-1
    mus = range(mumax)
    # n: 1 .. nmax
    ns = range(1, nmax + 1)
    # l: lmin .. lmax
    ls = range(lmin, lmax + 1)
    
    # 1. Iterate over unique sorted L-vectors (Canonical Blocks)
    # combinations_with_replacement handles the sorting and uniqueness
    l_combs = itertools.combinations_with_replacement(ls, rank)
    
    for lin in l_combs:
        # Global Parity Check: sum(l) must be even for scalar invariant
        if sum(lin) % 2 != 0: continue
        
        # 2. Iterate over unique sorted n-vectors
        n_combs = itertools.combinations_with_replacement(ns, rank)
        
        for nin in n_combs:
            # 3. Iterate over unique sorted mu-vectors
            mu_combs = itertools.combinations_with_replacement(mus, rank)
            
            for muin in mu_combs:
                
                # Combine n and mu into a composite index for Theory Counting.
                # Group Theory needs to know which atoms are distinguishable.
                # Atoms are identical only if BOTH mu and n match.
                # We encode this as composite_n = mu * 1000 + n
                comp_n = [m * 1000 + n for m, n in zip(muin, nin)]
                
                # Call the SVD Engine (pa_lib_v2)
                # It generates the candidates and selects the independent ones
                independent_labs = apply_ladder_relationships(
                    lin, tuple(comp_n), L_R=L_R
                )
                
                all_lammps_labs.extend(independent_labs)

    # Save to JSON Cache
    cache_file = get_cache_filename(rank, nmax, lmax, mumax)
    cache_data = {
        "params": {"rank": rank, "nmax": nmax, "lmax": lmax, "mumax": mumax},
        "labels": all_lammps_labs
    }
    
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"*** Cached {len(all_lammps_labs)} labels to {cache_file}")
    except IOError as e:
        print(f"*** Warning: Could not write cache file: {e}")

    return all_lammps_labs

def pa_labels_raw(rank, nmax, lmax, mumax, lmin=1, L_R=0, M_R=0):
    """
    Main entry point. Checks cache first, then generates if missing.
    Returns: (list_of_labels, list_of_incompatible_labels)
    """
    if rank < 1: return [], []

    cache_file = get_cache_filename(rank, nmax, lmax, mumax)
    
    # 1. Try Loading from Cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                # Optional: Verify params match
                # if data['params']['lmax'] == lmax ...
                print(f"*** Loaded basis from cache: {cache_file}")
                return data['labels'], []
        except (json.JSONDecodeError, KeyError):
            print("*** Cache corrupted, regenerating...")

    # 2. Cache Miss: Generate and Save
    labels = build_and_cache_basis(rank, nmax, lmax, mumax, lmin, L_R)
    
    return labels, []
