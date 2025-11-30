#!/usr/bin/env python3
"""
Generate PyACE whitelist using proper unification functions.

Keys are unified (mu, n) patterns from unify_mus_ns_comb.
Values are valid (ls, LS) combinations satisfying angular momentum coupling.

The whitelist structure matches what pyace expects:
- Key: unified (mu, n) pattern tuple 
- Value: list of (ls, LS) tuples where:
  - ls is a list of angular momentum values
  - LS is a list of intermediate coupling values
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
    
    This matches the pyace/lammps-pyace unification exactly.
    """
    unif_comb = unify_by_ordering(mus_comb, ns_comb)
    return unify_to_minimized_indices(unif_comb)


def generate_unified_keys_for_rank(rank, nmax, num_mu_types=1):
    """
    Generate all unique unified keys for a given rank by enumerating 
    all (mu, n) combinations and applying unify_mus_ns_comb.
    
    Args:
        rank: Body order - 1 (rank 0 = 1-body, rank 1 = 2-body, etc.)
        nmax: Maximum n value
        num_mu_types: Number of distinct mu types (1 for single element)
    
    Returns:
        Set of unique unified key tuples
    """
    body_order = rank + 1
    unified_keys = set()
    
    if body_order == 1:
        # For 1-body (rank 0), there's only one key: (0,)
        unified_keys.add((0,))
        return unified_keys
    
    # Generate all (mu, n) combinations
    mu_range = range(num_mu_types)
    n_range = range(1, nmax + 1)  # n starts from 1 in pyace
    
    # For single element, all mus are 0
    mus_comb = (0,) * rank  # rank = body_order - 1, which is the tail length
    
    for ns_comb in itertools.product(n_range, repeat=rank):
        unified_key = unify_mus_ns_comb(mus_comb, ns_comb)
        unified_keys.add(unified_key)
    
    return unified_keys


def triangle_inequality_holds(l1, l2, L):
    """Check if angular momentum triangle inequality holds."""
    return abs(l1 - l2) <= L <= l1 + l2


def generate_valid_ls_LS_for_rank(rank, lmax, lmin=0):
    """
    Generate all valid (ls, LS) combinations for a given rank.
    
    Constraints from ace_couplings.cpp validate_ls_LS:
    - rank 1: ls = [0] only
    - rank 2: ls = [l, l] (both equal)
    - rank 3: LS = [ls[2]] (1 element)
    - rank 4: LS = [L, L] (2 equal elements)
    - rank 5: LS = [L1, L2, ls[4]] (3 elements)
    - rank 6+: LS[-1] == LS[-2]
    - All: sum(ls) must be even
    
    Also applies triangle inequality at each coupling step.
    """
    body_order = rank + 1
    valid_pairs = []
    
    if body_order == 1:
        # rank 0: Only l=0 allowed
        valid_pairs.append(([0], []))
        return valid_pairs
    
    if body_order == 2:
        # rank 1: ls = [l, l], LS = []
        for l in range(lmin, lmax + 1):
            ls = [l, l]
            # Sum of ls must be even (2*l is always even)
            valid_pairs.append((ls, []))
        return valid_pairs
    
    if body_order == 3:
        # rank 2: ls has 3 elements, LS = [ls[2]]
        # Coupling: l0 + l1 -> L0, L0 + l2 -> 0
        # So L0 = l2, and |l0 - l1| <= l2 <= l0 + l1
        for l0 in range(lmin, lmax + 1):
            for l1 in range(l0, lmax + 1):  # l1 >= l0 for canonical ordering
                for l2 in range(lmin, lmax + 1):
                    # Check sum is even
                    if (l0 + l1 + l2) % 2 != 0:
                        continue
                    # Check triangle inequality for first coupling
                    if not triangle_inequality_holds(l0, l1, l2):
                        continue
                    ls = [l0, l1, l2]
                    LS = [l2]  # L0 = l2 for L=0 invariant
                    # Verify with coupling module
                    try:
                        ms_cg_list = coupling.generate_ms_cg_list(
                            ls=ls, LS=LS,
                            L=0, M=0,
                            half_basis=True,
                            check_is_even=True
                        )
                        if len(ms_cg_list) > 0:
                            valid_pairs.append((ls, LS))
                    except:
                        pass
        return valid_pairs
    
    if body_order == 4:
        # rank 3: ls has 4 elements, LS = [L1, L1] (2 equal elements)
        # Tree: (l0,l1)->L0, (L0,l2)->L1, (L1,l3)->0
        # For L=0: L1 = l3... wait no, let me check
        # Actually for rank 4, LS[-1] == LS[-2], so LS = [L, L]
        for l0 in range(lmin, lmax + 1):
            for l1 in range(l0, lmax + 1):
                for l2 in range(lmin, lmax + 1):
                    for l3 in range(l2, lmax + 1):
                        if (l0 + l1 + l2 + l3) % 2 != 0:
                            continue
                        # Try different L values
                        for L in range(lmax + 1):
                            ls = [l0, l1, l2, l3]
                            LS = [L, L]
                            try:
                                ms_cg_list = coupling.generate_ms_cg_list(
                                    ls=ls, LS=LS,
                                    L=0, M=0,
                                    half_basis=True,
                                    check_is_even=True
                                )
                                if len(ms_cg_list) > 0:
                                    valid_pairs.append((ls, LS))
                            except:
                                pass
        return valid_pairs
    
    if body_order == 5:
        # rank 4: ls has 5 elements, LS = [L1, L2, ls[4]]
        for l0 in range(lmin, lmax + 1):
            for l1 in range(l0, lmax + 1):
                for l2 in range(lmin, lmax + 1):
                    for l3 in range(l2, lmax + 1):
                        for l4 in range(lmin, lmax + 1):
                            if (l0 + l1 + l2 + l3 + l4) % 2 != 0:
                                continue
                            for L1 in range(lmax + 1):
                                for L2 in range(lmax + 1):
                                    ls = [l0, l1, l2, l3, l4]
                                    LS = [L1, L2, l4]  # LS[2] = ls[4]
                                    try:
                                        ms_cg_list = coupling.generate_ms_cg_list(
                                            ls=ls, LS=LS,
                                            L=0, M=0,
                                            half_basis=True,
                                            check_is_even=True
                                        )
                                        if len(ms_cg_list) > 0:
                                            valid_pairs.append((ls, LS))
                                    except:
                                        pass
        return valid_pairs
    
    if body_order >= 6:
        # rank 5+: LS[-1] == LS[-2]
        # This gets complex, using the coupling module to validate
        rankL = body_order - 2
        
        # Generate ls combinations
        for ls_combo in itertools.combinations_with_replacement(range(lmin, lmax + 1), body_order):
            ls = list(ls_combo)
            if sum(ls) % 2 != 0:
                continue
            
            # Generate LS candidates with constraint LS[-1] == LS[-2]
            for LS_base in itertools.product(range(lmax + 1), repeat=rankL - 1):
                LS = list(LS_base) + [LS_base[-1]]  # LS[-1] == LS[-2]
                try:
                    ms_cg_list = coupling.generate_ms_cg_list(
                        ls=ls, LS=LS,
                        L=0, M=0,
                        half_basis=True,
                        check_is_even=True
                    )
                    if len(ms_cg_list) > 0:
                        valid_pairs.append((ls, LS))
                except:
                    pass
        return valid_pairs
    
    return valid_pairs


def generate_whitelist_for_rank(rank, nmax, lmax, lmin=0, num_cores=None):
    """
    Generate whitelist entries for one rank.
    
    Args:
        rank: Rank (body_order - 1)
        nmax: Maximum n value for key generation
        lmax: Maximum l value for (ls, LS) generation
        lmin: Minimum l value
        num_cores: Number of CPU cores (not used in this version)
    
    Returns dict: {unified_key: [(ls, LS), ...]}
    """
    body_order = rank + 1
    whitelist = {}
    
    print(f"\nRank {rank} ({body_order}-body): nmax={nmax}, lmax={lmax}, lmin={lmin}")
    
    # Generate unique unified keys
    print(f"  Generating unified keys...")
    unified_keys = generate_unified_keys_for_rank(rank, nmax)
    print(f"  Found {len(unified_keys)} unique unified keys")
    
    # Generate valid (ls, LS) combinations for this rank
    print(f"  Generating valid (ls, LS) combinations...")
    valid_ls_LS = generate_valid_ls_LS_for_rank(rank, lmax, lmin)
    print(f"  Found {len(valid_ls_LS)} valid (ls, LS) pairs")
    
    # Each unified key maps to the same set of (ls, LS) pairs
    # because the key represents the (mu, n) pattern, not the ls pattern
    for key in unified_keys:
        whitelist[key] = valid_ls_LS
    
    print(f"  Generated {len(whitelist)} whitelist entries")
    return whitelist


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate PyACE whitelist')
    parser.add_argument('--mode', choices=['replicate', 'expand'], default='replicate')
    parser.add_argument('--output', default='mus_ns_uni_to_rawlsLS_np_rank_NEW.pckl')
    parser.add_argument('--cores', type=int, default=None, 
                        help='Number of CPU cores to use (default: all available)')
    parser.add_argument('--max-rank', type=int, default=3,
                        help='Maximum rank to generate (default: 3)')
    parser.add_argument('--nmax', type=int, default=10,
                        help='Maximum n value for key generation (default: 10)')
    parser.add_argument('--lmax', type=int, default=4,
                        help='Maximum l value for ls generation (default: 4)')
    args = parser.parse_args()
    
    print("="*80)
    print("GENERATE PYACE WHITELIST")
    print("="*80)
    
    num_cores = args.cores if args.cores else cpu_count()
    print(f"Using {num_cores} CPU cores")
    print(f"Max rank: {args.max_rank}")
    print(f"nmax: {args.nmax}")
    print(f"lmax: {args.lmax}")
    
    whitelist = {}
    
    # Generate for each rank
    # rank 0 = 1-body, rank 1 = 2-body, etc.
    for rank in range(args.max_rank + 1):
        if rank == 0:
            # 1-body: only l=0
            rank_lmax = 0
        else:
            rank_lmax = args.lmax
        
        rank_wl = generate_whitelist_for_rank(
            rank=rank,
            nmax=args.nmax,
            lmax=rank_lmax,
            lmin=0
        )
        whitelist.update(rank_wl)
    
    print(f"\n{'='*80}")
    print(f"Total whitelist entries: {len(whitelist)}")
    
    # Compare with original lammps-pyace
    orig_path = os.path.expanduser("~/github/lammps-pyace/python/lammps_pyace/unif_mus_ns_to_lsLScomb_dict.pckl")
    if os.path.exists(orig_path):
        with open(orig_path, 'rb') as f:
            orig = pickle.load(f)
        
        print(f"\nComparison with lammps-pyace whitelist:")
        print(f"  Original entries: {len(orig)}")
        
        gen_keys = set(whitelist.keys())
        orig_keys = set(orig.keys())
        
        # Only compare keys that we generated (up to max_rank)
        relevant_orig_keys = {k for k in orig_keys if len(k) <= args.max_rank + 1}
        
        print(f"  Original keys up to rank {args.max_rank}: {len(relevant_orig_keys)}")
        print(f"  Generated keys: {len(gen_keys)}")
        print(f"  Common keys: {len(gen_keys & relevant_orig_keys)}")
        print(f"  Only in generated: {len(gen_keys - relevant_orig_keys)}")
        print(f"  Only in original: {len(relevant_orig_keys - gen_keys)}")
        
        # Spot check values
        print("\nSpot check (ls, LS) counts:")
        for key in sorted(list(relevant_orig_keys)[:10]):
            gen_val = whitelist.get(key, [])
            orig_val = orig[key]
            match = "✓" if len(gen_val) == len(orig_val) else "✗"
            print(f"  {key}: gen={len(gen_val)}, orig={len(orig_val)} {match}")
            
            # Show first few entries for debugging
            if len(gen_val) != len(orig_val) and len(key) <= 2:
                print(f"    Gen first 3:  {gen_val[:3]}")
                print(f"    Orig first 3: {orig_val[:3]}")
    
    # Save
    with open(args.output, 'wb') as f:
        pickle.dump(whitelist, f)
    
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
