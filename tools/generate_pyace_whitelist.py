#!/usr/bin/env python3
"""
Generate PyACE whitelist using proper unification functions.

Keys are unified (mu, n) patterns from unify_mus_ns_comb.
Values are valid (ls, LS) combinations.
"""

import sys
import os
import pickle
import itertools
from multiprocessing import Pool, cpu_count

pyace_path = os.path.expanduser("~/github/python-ace-alphataubio/src")
sys.path.insert(0, pyace_path)

from pyace import coupling


def unify_to_minimized_indices(seq, shift=0):
    """Unify to minimized ordered sequence of indices."""
    seq_map = {e: i + shift for i, e in enumerate(sorted(set(seq)))}
    return tuple([seq_map[e] for e in seq])


def unify_by_ordering(mus_comb, ns_comb):
    """Unify mus_comb and ns_comb to minimized-indices sequence, combine pairwise and sort."""
    return tuple(sorted(zip(unify_to_minimized_indices(mus_comb), unify_to_minimized_indices(ns_comb))))


def unify_mus_ns_comb(mus_comb, ns_comb):
    """
    Unify mus_comb, ns_comb by unifying to min-inds, combining, sorting and
    unifying the pair one more time to minimized-indices sequence.
    """
    unif_comb = unify_by_ordering(mus_comb, ns_comb)
    return unify_to_minimized_indices(unif_comb)


def generate_restricted_growth_strings(n, max_value=None):
    """
    Generate all restricted growth strings of length n.
    
    A restricted growth string is a sequence where:
    - First element is 0
    - Each element is at most 1 + max of all previous elements
    
    For single-element ACE with nmax, max_value = nmax
    
    Args:
        n: Length of string (body order)
        max_value: Maximum value that can appear (nmax)
    
    Returns:
        Set of tuples representing unique patterns
    """
    if n == 0:
        return {()}
    if n == 1:
        return {(0,)}
    
    patterns = set()
    
    def generate(current, position, max_so_far):
        if position == n:
            patterns.add(tuple(current))
            return
        
        # Can use any value from 0 to min(max_so_far + 1, max_value)
        upper_limit = max_so_far + 1
        if max_value is not None:
            upper_limit = min(upper_limit, max_value)
        
        for val in range(upper_limit + 1):
            current.append(val)
            generate(current, position + 1, max(max_so_far, val))
            current.pop()
    
    generate([0], 1, 0)
    return patterns


def process_unified_key(args):
    """
    Worker function to process one unified key in parallel.
    
    Args:
        args: tuple of (unified_key, lmax, lmin)
    
    Returns:
        tuple: (unified_key, list of valid (ls, LS) pairs)
    """
    unified_key, lmax, lmin = args
    valid_ls_LS = []
    
    # ls length should match unified_key length
    ls_length = len(unified_key)
    
    # Try all l-combinations of the correct length
    for ls_combo in itertools.combinations_with_replacement(range(lmin, lmax + 1), ls_length):
        ls = list(ls_combo)
        
        # Determine LS candidates based on ls_length (body order)
        if ls_length == 1:
            LS_candidates = [[]]
        elif ls_length == 2:
            LS_candidates = [[]]
        elif ls_length == 3:
            LS_candidates = [[L] for L in range(lmax + 1)]
        elif ls_length == 4:
            LS_candidates = [[L, L] for L in range(lmax + 1)]
        elif ls_length == 5:
            LS_candidates = []
            for L1 in range(lmax + 1):
                for L2 in range(lmax + 1):
                    LS_candidates.append([L1, L2, ls[-1]])
        elif ls_length >= 6:
            LS_candidates = []
            for L1 in range(lmax + 1):
                for L2 in range(lmax + 1):
                    for L3 in range(lmax + 1):
                        LS_candidates.append([L1, L2, L3, L3])
        
        # Test each LS with coupling module
        for LS in LS_candidates:
            try:
                ms_cg_list = coupling.generate_ms_cg_list(
                    ls=ls, LS=LS,
                    L=0, M=0,
                    half_basis=True,
                    check_is_even=True
                )
                if len(ms_cg_list) > 0:
                    entry = (ls, LS)
                    if entry not in valid_ls_LS:
                        valid_ls_LS.append(entry)
            except:
                pass
    
    return (unified_key, valid_ls_LS)


def generate_whitelist_for_rank(rank, nmax, lmax, lmin=0, num_cores=None):
    """
    Generate whitelist entries for one rank using multiprocessing.
    
    Returns dict: {unified_key: [(ls, LS), ...]}
    """
    if num_cores is None:
        num_cores = cpu_count()
    
    body_order = rank + 1
    whitelist = {}
    
    print(f"\nRank {rank} ({body_order}-body): nmax={nmax}, lmax={lmax}, lmin={lmin}")
    
    # For single element, generate unified keys directly using restricted growth strings
    print(f"  Generating unique unified keys directly...")
    # Max value in unified key is determined by lmax, not nmax!
    unified_keys = generate_restricted_growth_strings(body_order, max_value=lmax)
    
    print(f"  Found {len(unified_keys)} unique unified keys")
    print(f"  Generating valid (ls, LS) combinations using {num_cores} cores...")
    
    # Prepare arguments for parallel processing
    args_list = [(key, lmax, lmin) for key in unified_keys]
    
    # Process in parallel
    with Pool(num_cores) as pool:
        results = pool.map(process_unified_key, args_list)
    
    # Collect results
    for unified_key, valid_ls_LS in results:
        if len(valid_ls_LS) > 0:
            whitelist[unified_key] = valid_ls_LS
    
    print(f"  Generated {len(whitelist)} whitelist entries")
    return whitelist


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['replicate', 'expand'], default='replicate')
    parser.add_argument('--output', default='mus_ns_uni_to_rawlsLS_np_rank_NEW.pckl')
    parser.add_argument('--cores', type=int, default=None, 
                        help='Number of CPU cores to use (default: all available)')
    args = parser.parse_args()
    
    print("="*80)
    print("GENERATE PYACE WHITELIST")
    print("="*80)
    
    num_cores = args.cores if args.cores else cpu_count()
    print(f"Using {num_cores} CPU cores")
    
    if args.mode == 'replicate':
        print("\nMode: REPLICATE")
        # Original was generated with lmax incrementing up to rank 4, then lmax=1
        configs = [
            (0, 10, 0, 0),   # rank 0: 1-body, lmax=0 (radial only)
            (1, 10, 1, 0),   # rank 1: 2-body, lmax=1
            (2, 10, 2, 0),   # rank 2: 3-body, lmax=2
            (3, 10, 3, 0),   # rank 3: 4-body, lmax=3
            (4, 10, 4, 0),   # rank 4: 5-body, lmax=4
            (5, 10, 1, 0),   # rank 5: 6-body, lmax=1
            (6, 10, 1, 0),   # rank 6: 7-body, lmax=1
            (7, 10, 1, 0),   # rank 7: 8-body, lmax=1
            (8, 10, 1, 0),   # rank 8: 9-body, lmax=1
            (9, 10, 1, 0),   # rank 9: 10-body, lmax=1
            (10, 10, 1, 0),  # rank 10: 11-body, lmax=1
            (11, 10, 1, 0),  # rank 11: 12-body, lmax=1
        ]
    else:
        print("\nMode: EXPAND rank 5 to lmax=6")
        configs = [
            (0, 10, 0, 0),
            (1, 10, 1, 0),
            (2, 10, 2, 0),
            (3, 10, 3, 0),
            (4, 10, 4, 0),
            (5, 10, 6, 1),   # EXPANDED: lmax=6, lmin=1 instead of lmax=1
            (6, 10, 1, 0),
            (7, 10, 1, 0),
            (8, 10, 1, 0),
            (9, 10, 1, 0),
            (10, 10, 1, 0),
            (11, 10, 1, 0),
        ]
    
    whitelist = {}
    for config in configs:
        rank_wl = generate_whitelist_for_rank(*config, num_cores=num_cores)
        whitelist.update(rank_wl)
    
    print(f"\n{'='*80}")
    print(f"Total whitelist entries: {len(whitelist)}")
    
    # Compare with original
    orig_path = os.path.expanduser("~/github/python-ace-alphataubio/src/pyace/data/mus_ns_uni_to_rawlsLS_np_rank.pckl")
    if os.path.exists(orig_path):
        with open(orig_path, 'rb') as f:
            orig = pickle.load(f)
        
        print(f"Original entries: {len(orig)}")
        
        gen_keys = set(whitelist.keys())
        orig_keys = set(orig.keys())
        
        print(f"Common keys: {len(gen_keys & orig_keys)}")
        print(f"Only in generated: {len(gen_keys - orig_keys)}")
        print(f"Only in original: {len(orig_keys - gen_keys)}")
        
        # Check first few keys
        print("\nSpot check:")
        for key in sorted(list(orig.keys())[:5]):
            gen_val = whitelist.get(key, [])
            orig_val = orig[key]
            match = "✓" if gen_val == orig_val else "✗"
            print(f"  {key}: gen={len(gen_val)}, orig={len(orig_val)} {match}")
            
            if gen_val != orig_val and len(gen_val) > 0 and len(orig_val) > 0:
                print(f"    Gen first:  {gen_val[0]}")
                print(f"    Orig first: {orig_val[0]}")
    
    # Save
    with open(args.output, 'wb') as f:
        pickle.dump(whitelist, f)
    
    print(f"\nSaved to: {args.output}")
    print("\nTo expand rank 5:")
    print(f"  python {__file__} --mode expand --cores {num_cores}")
    print("\nTo replace original:")
    print(f"  cp {args.output} ~/github/python-ace-alphataubio/src/pyace/data/mus_ns_uni_to_rawlsLS_np_rank.pckl")


if __name__ == "__main__":
    main()
