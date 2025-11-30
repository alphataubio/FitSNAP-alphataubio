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


def generate_unified_keys_for_rank(rank, nmax, num_mu_types=3):
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
    #mus_comb = (0,) * rank  # rank = body_order - 1, which is the tail length
    
    for mus_comb in itertools.product(mu_range, repeat=rank):
        for ns_comb in itertools.product(n_range, repeat=rank):
            unified_key = unify_mus_ns_comb(mus_comb, ns_comb)
            unified_keys.add(unified_key)
    
    return unified_keys


def triangle_inequality_holds(l1, l2, L):
    """Check if angular momentum triangle inequality holds."""
    return abs(l1 - l2) <= L <= l1 + l2


def get_permutation_group_from_key(key):
    """
    Given a unified key, determine which positions are equivalent.
    
    The key encodes the symmetry: repeated values indicate equivalent positions.
    Returns a list of tuples, where each tuple contains indices that can be permuted.
    
    Example:
        key (0, 0, 0) -> all equivalent -> [(0, 1, 2)]
        key (0, 0, 1) -> first two equivalent -> [(0, 1)]
        key (0, 1, 1) -> last two equivalent -> [(1, 2)]
        key (0, 1, 2) -> all different -> []
    """
    from collections import defaultdict
    # Group positions by their key value
    groups = defaultdict(list)
    for idx, val in enumerate(key):
        groups[val].append(idx)
    
    # Return groups with more than one element (these are equivalent positions)
    return [tuple(indices) for indices in groups.values() if len(indices) > 1]


def is_canonical_ls_for_key(ls, key):
    """
    Check if ls is in canonical form given the symmetry encoded in the key.
    
    For positions that are equivalent (same key value), the ls values
    should be in non-decreasing order to avoid counting permutations.
    
    This is the key relationship between key and (ls, LS):
    - The key determines which positions are equivalent
    - For equivalent positions, we only keep ls in sorted order
    """
    from collections import defaultdict
    
    # Group positions by key value
    groups = defaultdict(list)
    for idx, val in enumerate(key):
        groups[val].append(idx)
    
    # For each group of equivalent positions, ls values must be sorted
    for indices in groups.values():
        if len(indices) > 1:
            ls_values = [ls[i] for i in indices]
            if ls_values != sorted(ls_values):
                return False
    return True


def generate_valid_ls_LS_for_key(key, lmax):
    """
    Generate all valid (ls, LS) combinations for a given unified key.
    
    The key encodes the symmetry pattern of (mu, n) pairs.
    Positions with the same key value are equivalent under permutation.
    
    We enumerate all (ls, LS) combinations that:
    1. Are in canonical form w.r.t. the key's symmetry
    2. Satisfy angular momentum coupling (is_valid_ls_LS)
    3. Produce non-zero Clebsch-Gordan coefficients (generate_ms_cg_list)
    
    Args:
        key: Unified key tuple (e.g., (0, 0, 1) for rank 3)
        lmax: Maximum angular momentum value
    
    Returns:
        List of (ls, LS) tuples
    """
    rank = len(key)
    lmin = 0
    valid_pairs = []
    
    # Handle rank 1 specially: only l=0 is valid
    if rank == 1:
        return [([0], [])]
    
    # Generate all possible ls combinations
    l_range = range(lmin, lmax + 1)
    
    for ls_tuple in itertools.product(l_range, repeat=rank):
        ls = list(ls_tuple)
        
        # Check if sum of ls is even (required for invariant coupling)
        if sum(ls) % 2 != 0:
            continue
        
        # Check if ls is in canonical form for this key's symmetry
        if not is_canonical_ls_for_key(ls, key):
            continue
        
        # For rank 2: ls must have equal values (this is validated by is_valid_ls_LS)
        if rank == 2:
            if ls[0] != ls[1]:
                continue
            # For rank 2, LS is empty
            expanded_ls, expanded_LS = coupling.expand_ls_LS(rank, ls, [])
            if coupling.is_valid_ls_LS(expanded_ls, expanded_LS):
                # Check if it produces valid CG coefficients
                ms_cg = coupling.generate_ms_cg_list(expanded_ls, expanded_LS, half_basis=False)
                if len(ms_cg) > 0:
                    valid_pairs.append((list(expanded_ls), list(expanded_LS)))
            continue
        
        # For rank >= 3: need to enumerate valid LS combinations
        # Number of independent LS values depends on rank
        if rank == 3:
            # LS has 1 element: LS[0] = ls[-1] (determined by expand_ls_LS)
            # So we don't enumerate LS, it's determined by ls
            expanded_ls, expanded_LS = coupling.expand_ls_LS(rank, ls, [])
            if coupling.is_valid_ls_LS(expanded_ls, expanded_LS):
                ms_cg = coupling.generate_ms_cg_list(expanded_ls, expanded_LS, half_basis=False)
                if len(ms_cg) > 0:
                    valid_pairs.append((list(expanded_ls), list(expanded_LS)))
        
        elif rank == 4:
            # LS has 2 elements, 1 independent: LS[0] = LS[1]
            # enumerate LS[0] 
            for L0 in l_range:
                LS = [L0]
                expanded_ls, expanded_LS = coupling.expand_ls_LS(rank, ls, LS)
                # Check triangle inequality for the coupling tree
                # The tree for rank 4: (0,1)->L0, (L0,2)->L1, (L1,3)->0
                # So we need |l0-l1| <= L0 <= l0+l1
                if not triangle_inequality_holds(ls[0], ls[1], L0):
                    continue
                if coupling.is_valid_ls_LS(expanded_ls, expanded_LS):
                    ms_cg = coupling.generate_ms_cg_list(expanded_ls, expanded_LS, half_basis=False)
                    if len(ms_cg) > 0:
                        valid_pairs.append((list(expanded_ls), list(expanded_LS)))
        
        elif rank == 5:
            # LS has 3 elements, 2 independent: LS[2] = ls[-1]
            for L0 in l_range:
                for L1 in l_range:
                    LS = [L0, L1]
                    # Check triangle inequalities
                    if not triangle_inequality_holds(ls[0], ls[1], L0):
                        continue
                    if not triangle_inequality_holds(ls[2], ls[3], L1):
                        continue
                    expanded_ls, expanded_LS = coupling.expand_ls_LS(rank, ls, LS)
                    if coupling.is_valid_ls_LS(expanded_ls, expanded_LS):
                        ms_cg = coupling.generate_ms_cg_list(expanded_ls, expanded_LS, half_basis=False)
                        if len(ms_cg) > 0:
                            valid_pairs.append((list(expanded_ls), list(expanded_LS)))
        
        elif rank == 6:
            # LS has 4 elements, 3 independent: LS[3] = LS[2]
            for L0 in l_range:
                for L1 in l_range:
                    for L2 in l_range:
                        LS = [L0, L1, L2]
                        # Check triangle inequalities
                        if not triangle_inequality_holds(ls[0], ls[1], L0):
                            continue
                        if not triangle_inequality_holds(ls[2], ls[3], L1):
                            continue
                        if not triangle_inequality_holds(L0, L1, L2):
                            continue
                        expanded_ls, expanded_LS = coupling.expand_ls_LS(rank, ls, LS)
                        if coupling.is_valid_ls_LS(expanded_ls, expanded_LS):
                            ms_cg = coupling.generate_ms_cg_list(expanded_ls, expanded_LS, half_basis=False)
                            if len(ms_cg) > 0:
                                valid_pairs.append((list(expanded_ls), list(expanded_LS)))
        else:
            raise NotImplementedError(f"Rank {rank} not yet supported")
    
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
    unified_keys = sorted(generate_unified_keys_for_rank(rank, nmax))
    print(f"  Found {len(unified_keys)} unique unified keys")
    
    # Generate valid (ls, LS) combinations for this rank
    print(f"  Generating valid (ls, LS) combinations...")
    for key in unified_keys:
        whitelist[key] = generate_valid_ls_LS_for_key(key, lmax)
    
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
    parser.add_argument('--lmax', type=int, default=10,
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
        
    # Save
    with open(args.output, 'wb') as f:
        pickle.dump(whitelist, f)
    
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()



























