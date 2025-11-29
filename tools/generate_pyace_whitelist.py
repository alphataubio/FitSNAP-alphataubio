#!/usr/bin/env python3
"""
Generate PyACE whitelist pickle using original C++ coupling code.

First replicates the current pickle to verify the method works,
then expands rank 5 to lmax=6.

Uses: pyace.coupling module (C++ bindings from ace_coupling_binding.cpp)
"""

import sys
import os
import pickle
import itertools
from collections import defaultdict

# Add python-ace to path
pyace_path = os.path.expanduser("~/github/python-ace-alphataubio/src")
if os.path.exists(pyace_path):
    sys.path.insert(0, pyace_path)

try:
    from pyace import coupling
    print("✓ Successfully imported pyace.coupling")
except ImportError as e:
    print(f"✗ Failed to import pyace.coupling: {e}")
    print("\nInstall python-ace first:")
    print("  cd ~/github/python-ace-alphataubio")
    print("  pip install -e .")
    sys.exit(1)


def generate_LS_combinations(rank, ls, lmax):
    """
    Generate valid LS combinations for a given rank and ls.
    
    LS expansion rules from ace_coupling_binding.cpp:
    - rank 1 (2-body): LS is empty
    - rank 2 (3-body): LS has 1 element, equals ls[-1]
    - rank 3 (4-body): LS has 2 elements, LS[-1] = LS[-2]
    - rank 4 (5-body): LS has 3 elements, LS[-1] = ls[-1]
    - rank 5 (6-body): LS has 4 elements, LS[-1] = LS[-2]
    
    Args:
        rank: ACE rank (1 for 2-body, 2 for 3-body, etc.)
        ls: list of l values
        lmax: maximum L value to try
    
    Returns:
        list of valid LS combinations
    """
    if rank == 1:
        # 2-body: no coupling
        return [[]]
    
    elif rank == 2:
        # 3-body: LS has 1 element
        # LS[0] = ls[-1]
        return [[ls[-1]]]
    
    elif rank == 3:
        # 4-body: LS has 2 elements
        # LS[-1] = LS[-2], so only 1 independent value
        LS_list = []
        for L in range(lmax + 1):
            LS_list.append([L, L])
        return LS_list
    
    elif rank == 4:
        # 5-body: LS has 3 elements  
        # LS[-1] = ls[-1], so 2 independent values
        LS_list = []
        for L1 in range(lmax + 1):
            for L2 in range(lmax + 1):
                LS_list.append([L1, L2, ls[-1]])
        return LS_list
    
    elif rank == 5:
        # 6-body: LS has 4 elements
        # LS[-1] = LS[-2], so 3 independent values
        LS_list = []
        for L1 in range(lmax + 1):
            for L2 in range(lmax + 1):
                for L3 in range(lmax + 1):
                    LS_list.append([L1, L2, L3, L3])
        return LS_list
    
    else:
        raise ValueError(f"Rank {rank} not implemented")


def generate_whitelist_for_rank(rank, nmax, lmax, lmin=0):
    """
    Generate whitelist for a specific rank using PyACE coupling code.
    
    Args:
        rank: ACE rank (1=2-body, 2=3-body, etc.)
        nmax: maximum radial quantum number
        lmax: maximum angular momentum
        lmin: minimum angular momentum
    
    Returns:
        dict: whitelist for this rank
    """
    body_order = rank + 1
    whitelist = {}
    
    print(f"\nRank {rank} ({body_order}-body): lmin={lmin}, lmax={lmax}, nmax={nmax}")
    
    # Generate all l-combinations (sorted)
    l_combinations = list(itertools.combinations_with_replacement(
        range(lmin, lmax + 1),
        body_order
    ))
    
    print(f"  Testing {len(l_combinations)} l-combinations...")
    
    valid_count = 0
    entry_count = 0
    
    for l_tuple in l_combinations:
        ls = list(l_tuple)
        
        # Generate possible LS combinations for this ls
        LS_candidates = generate_LS_combinations(rank, ls, lmax)
        
        valid_LS = []
        for LS in LS_candidates:
            try:
                # Validate this (ls, LS) combination
                if coupling.is_valid_ls_LS(ls, LS):
                    # Generate ms combinations with CG coefficients
                    ms_cg_list = coupling.generate_ms_cg_list(
                        ls=ls,
                        LS=LS,
                        L=0,  # Final L (scalar invariant)
                        M=0,  # Final M
                        half_basis=True,
                        check_is_even=True
                    )
                    
                    # If this generated valid ms combinations, it's a valid LS
                    if len(ms_cg_list) > 0:
                        valid_LS.append(LS)
            except Exception:
                # Invalid combination, skip
                continue
        
        # If we found valid LS combinations, generate entries
        if len(valid_LS) > 0:
            valid_count += 1
            l_key = tuple(sorted(ls))
            
            if l_key not in whitelist:
                whitelist[l_key] = []
            
            # Generate n-combinations
            n_combinations = list(itertools.combinations_with_replacement(
                range(nmax + 1),
                body_order
            ))
            
            # Combine n with valid LS
            for n_tuple in n_combinations:
                n_list = list(n_tuple)
                
                for LS in valid_LS:
                    entry = [n_list, LS]
                    if entry not in whitelist[l_key]:
                        whitelist[l_key].append(entry)
                        entry_count += 1
    
    print(f"  Valid l-tuples: {valid_count}/{len(l_combinations)}")
    print(f"  Total entries: {entry_count}")
    
    return whitelist


def generate_full_whitelist(rank_configs):
    """
    Generate complete whitelist for multiple ranks.
    
    Args:
        rank_configs: list of (rank, nmax, lmax, lmin) tuples
    
    Returns:
        dict: complete whitelist
    """
    full_whitelist = {}
    
    print("="*80)
    print("PYACE WHITELIST GENERATION (using C++ coupling module)")
    print("="*80)
    
    for rank, nmax, lmax, lmin in rank_configs:
        rank_whitelist = generate_whitelist_for_rank(rank, nmax, lmax, lmin)
        
        # Merge into full whitelist
        for l_tuple, entries in rank_whitelist.items():
            if l_tuple in full_whitelist:
                print(f"Warning: l-tuple {l_tuple} already exists")
                # Merge avoiding duplicates
                for entry in entries:
                    if entry not in full_whitelist[l_tuple]:
                        full_whitelist[l_tuple].append(entry)
            else:
                full_whitelist[l_tuple] = entries
    
    return full_whitelist


def print_whitelist_stats(whitelist, name="whitelist"):
    """Print statistics about the whitelist."""
    by_rank = defaultdict(list)
    for l_tuple in whitelist.keys():
        rank = len(l_tuple) - 1
        by_rank[rank].append(l_tuple)
    
    print(f"\n{'='*80}")
    print(f"STATISTICS: {name}")
    print(f"{'='*80}")
    print(f"Total l-tuples: {len(whitelist)}")
    
    print("\nBreakdown by rank:")
    for rank in sorted(by_rank.keys()):
        l_tuples = by_rank[rank]
        total_entries = sum(len(whitelist[lt]) for lt in l_tuples)
        max_l = max(max(lt) for lt in l_tuples)
        print(f"  Rank {rank} ({rank+1}-body): {len(l_tuples)} l-tuples, "
              f"{total_entries} entries, max_l={max_l}")


def compare_whitelists(wl1, wl2, name1="WL1", name2="WL2"):
    """Compare two whitelists."""
    print(f"\n{'='*80}")
    print(f"COMPARISON: {name1} vs {name2}")
    print(f"{'='*80}")
    
    keys1 = set(wl1.keys())
    keys2 = set(wl2.keys())
    
    print(f"\n{name1}: {len(keys1)} l-tuples")
    print(f"{name2}: {len(keys2)} l-tuples")
    print(f"Common: {len(keys1 & keys2)}")
    print(f"Only in {name1}: {len(keys1 - keys2)}")
    print(f"Only in {name2}: {len(keys2 - keys1)}")
    
    # Compare by rank
    by_rank1 = defaultdict(set)
    by_rank2 = defaultdict(set)
    
    for key in keys1:
        by_rank1[len(key)-1].add(key)
    for key in keys2:
        by_rank2[len(key)-1].add(key)
    
    all_ranks = sorted(set(by_rank1.keys()) | set(by_rank2.keys()))
    
    print("\nPer rank:")
    for rank in all_ranks:
        r1 = by_rank1[rank]
        r2 = by_rank2[rank]
        print(f"  Rank {rank}: {name1}={len(r1)}, {name2}={len(r2)}, "
              f"common={len(r1 & r2)}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate PyACE whitelist using C++ coupling module'
    )
    parser.add_argument('--mode', choices=['replicate', 'expand'], default='replicate',
                       help='replicate: match current file, expand: add rank 5 lmax=6')
    parser.add_argument('--output', type=str, default='mus_ns_uni_to_rawlsLS_np_rank_NEW.pckl',
                       help='Output pickle file')
    parser.add_argument('--compare', type=str, default=None,
                       help='Path to original pickle for comparison')
    
    args = parser.parse_args()
    
    if args.mode == 'replicate':
        # Replicate current pickle to verify method
        # Current file has: ranks 1-5 with lmax=[1,2,3,4,1] and lmin=0
        print("\nMODE: Replicate current pickle")
        rank_configs = [
            # (rank, nmax, lmax, lmin)
            (1, 10, 1, 0),  # 2-body, lmax=1
            (2, 10, 2, 0),  # 3-body, lmax=2
            (3, 10, 3, 0),  # 4-body, lmax=3
            (4, 10, 4, 0),  # 5-body, lmax=4
            (5, 10, 1, 0),  # 6-body, lmax=1 (current limitation)
        ]
    else:
        # Expand rank 5 to lmax=6
        print("\nMODE: Expand rank 5 to lmax=6")
        rank_configs = [
            (1, 10, 1, 0),
            (2, 10, 2, 0),
            (3, 10, 3, 0),
            (4, 10, 4, 0),
            (5, 10, 6, 1),  # 6-body with lmax=6, lmin=1 (expanded!)
        ]
    
    # Generate whitelist
    whitelist = generate_full_whitelist(rank_configs)
    
    # Print statistics
    print_whitelist_stats(whitelist, "GENERATED")
    
    # Compare with original if provided
    if args.compare:
        if os.path.exists(args.compare):
            with open(args.compare, 'rb') as f:
                original = pickle.load(f)
            print_whitelist_stats(original, "ORIGINAL")
            compare_whitelists(original, whitelist, "ORIGINAL", "GENERATED")
        else:
            print(f"\nWarning: comparison file not found: {args.compare}")
    
    # Save
    with open(args.output, 'wb') as f:
        pickle.dump(whitelist, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\n{'='*80}")
    print(f"Saved to: {args.output}")
    print(f"{'='*80}")
    
    print("\nTo use this whitelist:")
    print(f"  cp {args.output} ~/github/python-ace-alphataubio/src/pyace/data/mus_ns_uni_to_rawlsLS_np_rank.pckl")


if __name__ == "__main__":
    main()
