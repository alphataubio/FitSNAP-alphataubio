#!/usr/bin/env python3

import sys, os, pickle, itertools, re
from collections import defaultdict

from fitsnap3lib.lib.sym_ACE.pa_gen import *


def unify_to_minimized_indices(seq, shift=0):
    seq_map = {e: i + shift for i, e in enumerate(sorted(set(seq)))}
    return tuple([seq_map[e] for e in seq])

def unify_by_ordering(mus_comb, ns_comb):
    return tuple(sorted(zip(unify_to_minimized_indices(mus_comb), unify_to_minimized_indices(ns_comb))))

def unify_mus_ns_comb(mus_comb, ns_comb):
    unif_comb = unify_by_ordering(mus_comb, ns_comb)
    return unify_to_minimized_indices(unif_comb)

# def pa_labels_raw(rank,nmax,lmax,mumax,lmin=1,L_R=0,M_R=0):
# ACE descriptor mu0_mu1,mu2,…,muN,n1,n2,…,nN,l1,l2,…,lN_L1-…-L(N-3)-L(N-2). 

def whitelist_for_rank(rank):

    #                1  2 3 4 5 6 7 8 9
    #lmax_by_rank = [0,10,6,4,2,2,1,1,1] ORIGINAL PYACE WHITELIST
    lmax_by_rank =  [0, 1,2,3,4,5]
    lmax = lmax_by_rank[rank-1]
    
    PA_lammps, not_compat = pa_labels_raw(rank=rank,nmax=rank,lmax=lmax,mumax=2,lmin=0)

    valid_ls_LS = defaultdict(list)
    for label in PA_lammps:
        mu0 = int(label.split('_')[0])
        if mu0 == 0:
            tmp = [int(v) for v in re.split(r'[_ ,-]', label) if v.isdigit()]
            key = unify_mus_ns_comb(tmp[0:rank+1],tmp[rank+1:2*rank+1])
            if rank >= 3:
                ls_LS = (tmp[2*rank+1:-rank+2], tmp[-rank+2:])
            else:
                ls_LS = (tmp[2*rank+1:], [])
            if ls_LS not in valid_ls_LS[key]: valid_ls_LS[key].append(ls_LS)
            
    for k,v in sorted(valid_ls_LS.items()): print(f"*** {k}\t|{len(v)}|\tmax {max(v)}")
    return sorted(valid_ls_LS.items())


def main():

    import argparse
    parser = argparse.ArgumentParser(description='Generate PyACE whitelist')
    parser.add_argument('--output', default='../../lammps-pyace/python/lammps_pyace/pyace_whitelist_pa_rpi.pckl')
    args = parser.parse_args()

    whitelist = {}
    
    print("="*80)
    print("GENERATE PYACE WHITELIST")
    print("="*80)
    
    for rank in range(1, 7):
        rank_wl = whitelist_for_rank(rank)
        whitelist.update(rank_wl)
    
    print(f"\n{'='*80}")
    print(f"Total whitelist entries: {len(whitelist)}")
        
    with open(args.output, 'wb') as f:
        pickle.dump(whitelist, f)
    
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()



























