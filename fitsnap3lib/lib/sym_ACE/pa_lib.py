
from fitsnap3lib.lib.sym_ACE.sym_ACE_settings import *

import numpy as np
import multiprocessing
from sympy.physics.wigner import wigner_3j
from itertools import product, permutations
import warnings
import os
import pickle
from collections import Counter, defaultdict
from functools import partial
import scipy.linalg


# GEMINI PRO 3's "PA-RPI Basis with Group Theory & Wigner Algebra"
# https://github.com/FitSNAP/FitSNAP/pull/278#issuecomment-3608889321
# https://github.com/FitSNAP/FitSNAP/pull/278#issuecomment-3608897109
# [alphataubio, 2025/12]


# =========================================================
# SECTION 1: BASIS GENERATION 
# =========================================================

class BasisGenerator:
    """
    Generates the Over-Complete set of ACE basis labels for arbitrary rank/orders.
    """
    
    @staticmethod
    def get_valid_intermediates(l1, l2):
        """Returns all L satisfying triangle inequality for l1, l2."""
        low = abs(l1 - l2)
        high = l1 + l2
        return range(low, high + 1)

    @staticmethod
    def generate_valid_intermediates_recursive(l_current_layer, L_R):
        """
        Recursively generates all valid intermediate tuples.
        Includes the Root node in the output.
        """
        if len(l_current_layer) == 1:
            if l_current_layer[0] == L_R:
                yield []
            return

        # Pairwise coupling of current layer
        pairs_to_couple = []
        pass_through = []
        
        i = 0
        while i < len(l_current_layer):
            if i + 1 < len(l_current_layer):
                pairs_to_couple.append((l_current_layer[i], l_current_layer[i+1]))
                i += 2
            else:
                pass_through.append(l_current_layer[i])
                i += 1

        ranges = [BasisGenerator.get_valid_intermediates(p[0], p[1]) for p in pairs_to_couple]
        
        for intermediates in product(*ranges):
            l_next_layer = list(intermediates) + pass_through
            for sub_inters in BasisGenerator.generate_valid_intermediates_recursive(l_next_layer, L_R):
                yield list(intermediates) + sub_inters

    @staticmethod
    def get_canonical_labels(nin, lin, muin=None, L_R=0):
        """
        Generates the raw "Over-Complete" list of labels for a specific block.
        """
        rank = len(lin)
        if muin is None: muin = tuple([0] * rank)

        atoms = sorted(list(zip(muin, nin, lin)))
        unique_perms = sorted(list(set(permutations(atoms))))
        
        labels = []
        for p in unique_perms:
            p_mu, p_n, p_l = zip(*p)
            
            # Generate intermediates for this leaf ordering
            valid_L_tuples = list(BasisGenerator.generate_valid_intermediates_recursive(p_l, L_R))
            
            for L_tuple in valid_L_tuples:
                # FIX: Slice to rank-2 to remove Root node and match Fitsnap convention
                # Rank 5 -> 3 intermediates. Tuple has 4. Slice [:3].
                L_clean = L_tuple[:rank-2]
                
                L_str = "-".join(map(str, L_clean))
                
                # Combine integers: mu, n, l
                integers = list(p_mu) + list(p_n) + list(p_l)
                int_str = ",".join(map(str, integers))
                
                # Format: "0_mu,n,l_L" (Assuming mu0=0 for single species basis)
                label = f"0_{int_str}_{L_str}"
                labels.append(label)
                
        return sorted(list(set(labels)))

# =========================================================
# SECTION 2: LEGACY HELPERS (Required by pa_gen.py)
# =========================================================

def get_mu_nu_rank(nu_in):
    if len(nu_in.split('_')) > 1:
        nu = nu_in.split('_')[1]
        nu_splt = nu.split(',')
        return int(len(nu_splt)/3)
    else:
        nu = nu_in
        nu_splt = nu.split(',')
        return int(len(nu_splt)/2)

def get_mu_n_l(nu_in, return_L = False, **kwargs):
    """
    Legacy parser for label strings. Restored for compatibility.
    """
    rank = get_mu_nu_rank(nu_in)
    if len(nu_in.split('_')) > 1:
        if len(nu_in.split('_')) == 2:
            nu = nu_in.split('_')[-1]
            Lstr = ''
        else:
            nu = nu_in.split('_')[1]
            Lstr = nu_in.split('_')[-1]
        mu0 = int(nu_in.split('_')[0])
        nusplt = [int(k) for k in nu.split(',')]
        mu = nusplt[:rank]
        n = nusplt[rank:2*rank]
        l = nusplt[2*rank:]
        
        if len(Lstr) >= 1:
            # Handle empty L strings gracefully
            try:
                L = tuple([int(k) for k in Lstr.split('-')])
            except ValueError:
                L = tuple()
        else:
            L = tuple()
            
        if return_L:
            return mu0 , mu , n , l , L
        else:
            return mu0 , mu , n , l
    else:
        # Fallback for deprecated formats
        nu = nu_in
        mu0 = 0
        mu = [0]*rank
        nusplt = [int(k) for k in nu.split(',')]
        n = nusplt[:rank]
        l = nusplt[rank:2*rank]
        if return_L:
            return mu0, mu, n, l, tuple()
        return mu0,mu,n,l

# =========================================================
# SECTION 3: CORE ALGORITHM (Theory & Wigner)
# =========================================================

CACHE_FILE = f'{lib_path}/wigner_cache.pckl'

class WignerCacheManager:
    def __init__(self):
        self.cache = {}
        self.new_entries = {}
    def load(self):
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "rb") as f: self.cache = pickle.load(f)
            except Exception: self.cache = {}
    def save(self):
        if not self.new_entries: return
        self.cache.update(self.new_entries)
        try:
            with open(CACHE_FILE, "wb") as f: pickle.dump(self.cache, f)
        except Exception: pass

def get_exact_w3j(j1, j2, j3, m1, m2, m3, cache=None):
    key = (j1, j2, j3, m1, m2, m3)
    if cache is not None and key in cache: return cache[key]
    try:
        val = np.float64(wigner_3j(j1, j2, j3, m1, m2, m3))
    except Exception:
        val = np.float64(0.0)
    if cache is not None: cache[key] = val
    return val

def calculate_generalized_wigner_exact(l_leaves, l_inters, m_config, local_cache):
    val = np.float64(1.0)
    current_L = list(l_leaves)
    current_m = list(m_config)
    inters_queue = list(l_inters)
    
    if abs(sum(current_m)) > 1e-9: return np.float64(0.0)

    while len(current_L) > 1:
        new_L = []
        new_m = []
        i = 0
        while i < len(current_L):
            if i + 1 < len(current_L):
                l1, l2 = current_L[i], current_L[i+1]
                m1, m2 = current_m[i], current_m[i+1]
                
                if len(current_L) == 2: 
                    L_res = 0 
                else:
                     if not inters_queue: return np.float64(0.0)
                     L_res = inters_queue.pop(0)
                
                M_res = m1 + m2
                if not (abs(l1 - l2) <= L_res <= l1 + l2): return np.float64(0.0)
                if abs(m1) > l1 or abs(m2) > l2 or abs(M_res) > L_res: return np.float64(0.0)

                w3j = get_exact_w3j(l1, l2, L_res, m1, m2, -M_res, local_cache)
                if abs(w3j) < 1e-15: return np.float64(0.0)
                val *= w3j
                
                new_L.append(L_res)
                new_m.append(M_res)
                i += 2
            else:
                new_L.append(current_L[i])
                new_m.append(current_m[i])
                i += 1
        current_L = new_L
        current_m = new_m
    return val

def _trellis_worker_iterative(prefix, l_leaves):
    rank = len(l_leaves)
    prefix_len = len(prefix)
    prefix_sum = sum(prefix)
    if prefix_len == rank - 1:
        m_last = -prefix_sum
        if abs(m_last) <= l_leaves[-1]: return [prefix + (m_last,)]
        return []

    results = []
    stack = [(prefix_len, prefix_sum, ())]
    remaining_cap = [0] * rank
    s = 0
    for i in range(rank - 1, -1, -1):
        remaining_cap[i] = s
        s += l_leaves[i]

    RELAX = 0 
    while stack:
        idx, curr_sum, path = stack.pop()
        if idx == rank - 1:
            m_last = -curr_sum
            if abs(m_last) <= l_leaves[idx]:
                results.append(prefix + path + (m_last,))
            continue
            
        cap = remaining_cap[idx]
        limit = l_leaves[idx]
        min_m = max(-limit, -cap - curr_sum - RELAX)
        max_m = min(limit, cap - curr_sum + RELAX)
        
        for m in range(min_m, max_m + 1):
            stack.append((idx + 1, curr_sum + m, path + (m,)))
    return results

def generate_valid_m_states_parallel(l_leaves, n_cores):
    rank = len(l_leaves)
    if rank == 0: return [()]
    
    target_tasks = max(200, n_cores * 4) 
    prefixes = [()]
    remaining_cap = [0] * rank
    s = 0
    for i in range(rank - 1, -1, -1):
        remaining_cap[i] = s
        s += l_leaves[i]

    depth = 0
    while len(prefixes) < target_tasks and depth < rank - 1:
        new_prefixes = []
        for p in prefixes:
            curr_sum = sum(p)
            idx = len(p)
            cap = remaining_cap[idx]
            limit = l_leaves[idx]
            min_m = max(-limit, -cap - curr_sum)
            max_m = min(limit, cap - curr_sum)
            for m in range(min_m, max_m + 1):
                new_prefixes.append(p + (m,))
        prefixes = new_prefixes
        depth += 1
        
    pool = multiprocessing.Pool(processes=n_cores)
    try:
        worker_func = partial(_trellis_worker_iterative, l_leaves=l_leaves)
        results_lists = pool.map(worker_func, prefixes)
        full_paths = []
        for lst in results_lists:
            full_paths.extend(lst)
    finally:
        pool.close()
        pool.join()
        
    return full_paths

def get_constrained_permutations(ref_full, curr_full):
    if len(ref_full) != len(curr_full): return []
    ref_indices = defaultdict(list)
    for i, prop in enumerate(ref_full): ref_indices[prop].append(i)
    curr_indices = defaultdict(list)
    for i, prop in enumerate(curr_full): curr_indices[prop].append(i)
    if ref_indices.keys() != curr_indices.keys(): return []

    per_bucket_perms = []
    for prop, c_idxs in curr_indices.items():
        r_idxs = ref_indices[prop]
        bucket_maps = []
        for p in permutations(r_idxs):
            mapping = {c: r for c, r in zip(c_idxs, p)}
            bucket_maps.append(mapping)
        per_bucket_perms.append(bucket_maps)
        
    all_maps = []
    for combined in product(*per_bucket_perms):
        full_map = {}
        for m in combined: full_map.update(m)
        indices = [full_map[k] for k in range(len(curr_full))]
        all_maps.append(indices)
    return all_maps

def parse_label_full(label, rank):
    parts = label.split('_')
    integers_block = parts[1].split(',')
    total_ints = len(integers_block)
    
    # Heuristic for label format
    if total_ints == 3 * rank:
        mu_leaves = [int(x) for x in integers_block[:rank]]
        n_leaves = [int(x) for x in integers_block[rank:2*rank]]
        l_leaves = [int(x) for x in integers_block[2*rank:]]
    elif total_ints == 2 * rank:
        mu_leaves = [0] * rank
        n_leaves = [int(x) for x in integers_block[:rank]]
        l_leaves = [int(x) for x in integers_block[rank:]]
    else:
        l_leaves = [int(x) for x in integers_block[-rank:]]
        n_leaves = [0] * rank
        mu_leaves = [0] * rank

    l_inters = []
    if len(parts) > 2:
        try: l_inters = [int(x) for x in parts[-1].split('-')]
        except ValueError: l_inters = []
    
    # Ensure intermediate list matches what BasisGenerator produces 
    # and what calculate_generalized_wigner expects.
    # Wigner calculator consumes intermediates for N-2 steps.
    # Rank 5 -> 3 steps.
    # Fitsnap label usually has Rank-2 intermediates (e.g. 3).
    # If list has more, slice it.
    if len(l_inters) > rank - 2: l_inters = l_inters[:rank-2]
    return mu_leaves, n_leaves, l_leaves, l_inters

def parse_label_to_tree(label, rank):
    _, _, l, inters = parse_label_full(label, rank)
    return l, inters

class CharacterIntegration:
    @staticmethod
    def chi_l(l, theta):
        if abs(theta) < 1e-9: return 2 * l + 1
        return np.sin((l + 0.5) * theta) / np.sin(0.5 * theta)
    @staticmethod
    def cycle_index_term(l, k, theta):
        h = [0.0] * (k + 1); h[0] = 1.0 
        for m in range(1, k + 1):
            sum_val = 0.0
            for i in range(1, m + 1):
                p_i = CharacterIntegration.chi_l(l, i * theta)
                sum_val += p_i * h[m - i]
            h[m] = sum_val / m
        return h[k]
    @staticmethod
    def get_total_character(lin, nin, theta):
        pairs = list(zip(nin, lin))
        counts = Counter(pairs)
        total_chi = 1.0
        for (n, l), k in counts.items():
            if k == 1: total_chi *= CharacterIntegration.chi_l(l, theta)
            else: total_chi *= CharacterIntegration.cycle_index_term(l, k, theta)
        return total_chi
    @staticmethod
    def count_invariants(lin, nin, integration_points=200):
        thetas, dt = np.linspace(0, np.pi, integration_points, retstep=True)
        haar_measure = 1 - np.cos(thetas)
        chis = np.array([CharacterIntegration.get_total_character(lin, nin, t) for t in thetas])
        integral = np.trapz(chis * haar_measure, dx=dt)
        return int(round(integral / np.pi))

def worker_process_exact(chunk_labels, rank, m_configs, ref_full_sorted, initial_cache):
    local_cache = initial_cache.copy()
    initial_keys = set(local_cache.keys())
    results = []
    for label in chunk_labels:
        mu_leaves, n_leaves, l_leaves, l_inters = parse_label_full(label, rank)
        curr_full = list(zip(n_leaves, l_leaves)) # Permutation based on (n,l) identity only
        
        perm_maps = get_constrained_permutations(ref_full_sorted, curr_full)
        if not perm_maps: continue

        w_vec = []
        is_zero_vec = True
        for m_ref in m_configs:
            val = np.float64(0.0)
            for indices in perm_maps:
                m_curr = [m_ref[i] for i in indices]
                w = calculate_generalized_wigner_exact(l_leaves, l_inters, m_curr, local_cache)
                val += w
            w_vec.append(val)
            if abs(val) > 1e-15: is_zero_vec = False
            
        if not is_zero_vec:
            vec = np.array(w_vec, dtype=np.float64)
            norm = np.linalg.norm(vec)
            if norm > 1e-15:
                results.append((label, vec / norm))
    new_entries = {k: v for k, v in local_cache.items() if k not in initial_keys}
    return results, new_entries

# =========================================================
# SECTION 4: MAIN ENTRY POINT
# =========================================================

def apply_ladder_relationships(lin, nin, combined_labs_legacy=None, parity_span=None, parity_span_labs=None, full_span=None, L_R=0):
    
    n_expected = CharacterIntegration.count_invariants(lin, nin)
    if n_expected == 0: return []

    pairs = list(zip(nin, lin))
    # Optimization: Distinct pairs (assuming full basis generation)
    # But since we are regenerating labels, we must run generation.
    # We can skip SVD if distinct.
    distinct_mode = (len(set(pairs)) == len(pairs))

    rank = len(lin)
    
    # 1. GENERATE COMPLETE CANDIDATE SET (Over-Complete)
    combined_labs = BasisGenerator.get_canonical_labels(nin, lin, L_R=L_R)
    
    if len(combined_labs) == 0: return []
    
    if distinct_mode:
        # If all atoms distinct, no ladder relationships exist. Return all valid trees.
        return combined_labs

    # 2. SETUP VECTOR SPACE
    ref_mu, ref_n, ref_l, _ = parse_label_full(combined_labs[0], rank)
    combined = sorted(list(zip(ref_mu, ref_n, ref_l)))
    ref_n_sorted = [x[1] for x in combined]
    ref_l_sorted = [x[2] for x in combined]
    ref_full_sorted = list(zip(ref_n_sorted, ref_l_sorted))
    
    n_cores = multiprocessing.cpu_count()
    m_configs = generate_valid_m_states_parallel(ref_l_sorted, n_cores)
    if not m_configs: return []

    cache_manager = WignerCacheManager()
    cache_manager.load()

    chunk_size = max(1, len(combined_labs) // (n_cores * 4))
    chunks = [combined_labs[i:i + chunk_size] for i in range(0, len(combined_labs), chunk_size)]
    
    valid_candidates = []
    with multiprocessing.Pool(processes=n_cores) as pool:
        results = pool.starmap(worker_process_exact, 
                               [(chunk, rank, m_configs, ref_full_sorted, cache_manager.cache) 
                                for chunk in chunks])
        for sub_res, sub_cache in results:
            valid_candidates.extend(sub_res)
            cache_manager.new_entries.update(sub_cache)
    cache_manager.save()

    if not valid_candidates:
        if n_expected > 0: warnings.warn(f"All vectors zero for block n={nin}, l={lin}")
        return []

    # 3. ROBUST SVD SELECTION
    labels = [x[0] for x in valid_candidates]
    A = np.column_stack([x[1] for x in valid_candidates])
    
    try:
        U, S, Vt = scipy.linalg.svd(A, full_matrices=False, lapack_driver='gesvd')
    except Exception:
        U, S, Vt = scipy.linalg.svd(A, full_matrices=False)

    max_sv = S[0]
    tolerance = max(1e-14, max_sv * 1e-12)
    rank_eff = np.sum(S > tolerance)
    
    independent_labs_final = []
    basis_matrix = []
    
    # Order-Preserving Gram-Schmidt
    for i, label in enumerate(labels):
        if len(independent_labs_final) == rank_eff: break
        vec = A[:, i]
        if len(basis_matrix) == 0:
            basis_matrix.append(vec)
            independent_labs_final.append(label)
        else:
            vec_ortho = vec.copy()
            B = np.array(basis_matrix)
            for _ in range(2):
                projections = np.dot(B, vec_ortho)
                vec_ortho -= np.dot(projections, B)
            if np.linalg.norm(vec_ortho) > tolerance:
                basis_matrix.append(vec_ortho / np.linalg.norm(vec_ortho))
                independent_labs_final.append(label)

    if len(independent_labs_final) < n_expected:
        warnings.warn(f"Under-complete Basis for n={nin}, l={lin}. Theory: {n_expected}, Found: {len(independent_labs_final)}. RankEff: {rank_eff}")

    return independent_labs_final
    
    
    
    
    
    
    
    
# -------------------------------- LEGACY CODE --------------------------------

from fitsnap3lib.lib.sym_ACE.inter_set import *
from fitsnap3lib.lib.sym_ACE.symmetric_grp_manip import *

def get_highest_coupling_representation(lp,lref):
    rank = len(lp)
    coupling_reps = local_sigma_c_partitions[rank]
    ysgi = Young_Subgroup(rank)
    highest_rep = tuple([1]*rank)
    for rep in coupling_reps:
        ysgi.subgroup_fill(lref,[rep],sigma_c_symmetric=False,semistandard=False)
        test_fills = ysgi.fills.copy()
        if lp not in test_fills:
            pass
        else:
            highest_rep = rep
            break
    return highest_rep

def tree_labels(nin,lin,L_R=0,M_R=0):
    rank = len(lin)
    ysgi = Young_Subgroup(rank)
    if type(lin) != list:
        lin = list(lin)
    if type(nin) != list:
        nin = list(nin)
    # get possible unique l permutations based on degeneracy and coupling tree structure
    ysgi.subgroup_fill(lin, partitions=[local_sigma_c_partitions[rank][-1]], max_orbit = len(local_sigma_c_partitions[rank][-1]), sigma_c_symmetric=False, semistandard=False)
    lperms = ysgi.fills.copy()
    lperms = leaf_filter(lperms)
    if rank not in [4,8,16,32]:
        lperms_tmp = []
        used_hrep = []
        for lperm in lperms:
            hrep = get_highest_coupling_representation(tuple(lperm),tuple(lperms[0]))
            if hrep not in used_hrep:
                used_hrep.append(hrep)
                lperms_tmp.append(lperm)
            else:
                pass
                #print('omitting',lperm)
        lperms = lperms_tmp
        #print ('lperms after subtree filter',lperms)
    original_joint_span = {lp:[] for lp in lperms}
    orb_nls = []

    ls = lperms.copy()
    nps_per_l = {}

    # get n permutations per l permutation
    # this could equivalently be done with a search over S_N
    for lp in ls:
        rank = len(lp)
        original_span_SO3 = tree_l_inters(lp) #RI basis size
        degen_orbit, orbit_inds = get_degen_orb(lp) #PI basis size
        ysgi.subgroup_fill(nin,[degen_orbit],sigma_c_symmetric=False,semistandard=False)
        degen_fills = ysgi.fills.copy()
        maxdegen = max([len(ois) for ois in orbit_inds])
        sequential_degen_orbit, orbit_inds_s = enforce_sorted_orbit(orbit_inds)
        #if rank > 4 and maxdegen > math.ceil(rank/2):
        #    ysgi.subgroup_fill(nin,[degen_orbit],sigma_c_symmetric=False,semistandard=False)
        #else:
        ysgi.subgroup_fill(nin,[sequential_degen_orbit],sigma_c_symmetric=False,semistandard=False)
        nps_per_l[lp] = ysgi.fills.copy()
        original_joint_span[lp] = [(prd[0],lp,prd[1]) for prd in itertools.product(degen_fills,original_span_SO3)]

    nlabs = 0
    labels_per_lperm = {}
    #build all labels (unsorted trees)
    for l in ls:
        #print ('in l loop',l)
        l = list(l)
        subblock = []
        rank = len(l)
        inters = tree_l_inters(list(l),L_R=L_R,M_R=M_R)
        nperms = nps_per_l[tuple(l)]
        muperms = [tuple([0]*rank)]
        for inter in inters:
            if rank <= 5:
                if np.sum([inter[0]] + l[:2]) %2 ==0 and np.sum([inter[1]] + l[2:4]) %2 ==0:
                    for muperm in muperms:
                        for nperm in nperms:
                            if rank == 5:
                                orb_nls.append("0_%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d_%d-%d-%d"% (muperm + nperm+tuple(l) + inter))
                                subblock.append("0_%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d_%d-%d-%d"% (muperm + nperm+tuple(l) + inter))
                            elif rank ==4:
                                orb_nls.append("0_%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d_%d-%d"% (muperm + nperm+tuple(l) + inter))
                                subblock.append("0_%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d_%d-%d"% (muperm + nperm+tuple(l) + inter))
                            nlabs +=1
        labels_per_lperm[tuple(l)] = subblock


    block_sizes = {key:len(val) for key,val in labels_per_lperm.items()}
    all_labs = []
    labels_per_block = {block:[] for block in sorted(list(block_sizes.keys()))}
    counts_per_block = {block:0 for block in sorted(list(block_sizes.keys()))}
    # collect sorted trees only
    for block,labs in labels_per_lperm.items():
        used_ns = []
        used_ids = []
        for nu in labs:
            mu0,mutst,ntst,ltst, L = get_mu_n_l(nu,return_L=True)
            ltree = [(li,ni) for ni,li in zip(ntst,ltst)] #sort first on n
            tree_i =  build_tree(ltree,L,L_R)
            ttup = tree_i.full_tup()
            tid = tree_i.tree_id
            conds = tid not in used_ids # sorting is ensured in construction of trees
            if conds:
                if tuple(ntst) not in used_ns:
                    used_ns.append(tuple(ntst))
                used_ids.append(tid)
                labels_per_block[block].append(nu)
                counts_per_block[block] += 1
                all_labs.append(nu)
            else:
                pass

    #collect labels per l permutation block
    max_labs = []
    max_count = max(list(counts_per_block.values()))
    for block,tree_labs in labels_per_block.items():
        if len(tree_labs) == max_count:
            max_labs.append(tree_labs.copy())
    max_labs = max_labs[0]

    return max_labs,all_labs,labels_per_block,original_joint_span

def combine_blocks(blocks,lin,original_spans,L_R=0):
    # tool to recombine trees from multiple permutations of l
    rank = len(lin)
    ysgi = Young_Subgroup(rank)
    lps = list(blocks.keys())
    blockpairs = [(block1,block2) for block1,block2 in itertools.product(lps,lps) if block1 != block2]
    if len(blockpairs) == 0:
        blockpairs = [(block1,block2) for block1,block2 in itertools.product(lps,lps)]
    block_map = {blockpair:None for blockpair in blockpairs}
    all_map = {blockpair:None for blockpair in blockpairs}
    L_map = {blockpair:None for blockpair in blockpairs}
    raw_perms = [p for p in itertools.permutations(list(range(rank)))]
    Ps = [Permutation(pi) for pi in raw_perms]
    for blockpair in list(block_map.keys()):
        l1i,l2i = blockpair
        is_sigma0 = l1i == lps[0]
        Pl1is = [P for P in Ps if tuple(P(list(l1i))) == l2i]
        Pl1_maxorbit_sizes = [max([len(k) for k in P.full_cyclic_form]) for P in Pl1is]
        maxorbit_all = max(Pl1_maxorbit_sizes)
        maxorbit_ind = Pl1_maxorbit_sizes.index(maxorbit_all)
        if not is_sigma0:
            block_map[blockpair] = Pl1is[maxorbit_ind]
            all_map[blockpair] = Pl1is
        else:
            block_map[blockpair] = Permutation(tuple( [ tuple([ii]) for ii in list(range(rank))]))
            all_map[blockpair] = [Permutation(tuple( [ tuple([ii]) for ii in list(range(rank))]))]
            

    for blockpair in list(block_map.keys()):
        l1i,l2i = blockpair
        inters1 = tree_l_inters(l1i,L_R)
        is_sigma0 = tuple(l1i) == lps[0]
        l1i = list(l1i)
        l2i = list(l2i)
        # intermediates hard coded for ramk 4 and 5 right now
        inters1 = [inter for inter in inters1 if np.sum([inter[0]] + l1i[:2]) %2 ==0 and np.sum([inter[1]] + l1i[2:4]) %2 ==0 ]
        inters2 = tree_l_inters(l2i,L_R)
        inters2 = [inter for inter in inters2 if np.sum([inter[0]] + l2i[:2]) %2 ==0 and np.sum([inter[1]] + l2i[2:4]) %2 ==0 ]
        if not is_sigma0:
            L_map[blockpair] = {L1i:L2i for L1i,L2i in zip(inters1,inters2)  }
        else:
            L_map[blockpair] = {L1i:L1i for L1i,L1i in zip(inters1,inters1)  }
    used_ids = []
    used_nl = []
    combined_labs = []
    super_inters_per_nl = {}
    for lp,nus in blocks.items():
        rank = len(lp)
        degen_orbit, orbit_inds = get_degen_orb(lp)
        maxdegen = max([len(ois) for ois in orbit_inds])
        sequential_degen_orbit, orbit_inds_s = enforce_sorted_orbit(orbit_inds)
        block_pairs = [blockpair for blockpair in list(block_map.keys()) if blockpair[0] == tuple(lp)]
        blockpair = block_pairs[0]
        #perm_map = block_map[block_pairs[0]]
        if rank == 4:
            perms_2_check = [block_map[blockpair]]
        else:
            perms_2_check = [block_map[blockpair]]
            #perms_2_check = all_map[blockpair]
        for nu in nus:
            mu0ii,muii,nii,lii,Lii = get_mu_n_l(nu,return_L=True)
            is_sigma0 = tuple(lii) == lps[0]
            degen_orbit, orbit_inds = get_degen_orb(lp)
            sequential_degen_orbit, orbit_inds_s = enforce_sorted_orbit(orbit_inds)
            nlii = [(niii,liii) for niii,liii in zip (nii,lii)]
            atrees = []
            for perm_map in perms_2_check:
                remapped = perm_map(nlii)
                newnii = [nliii[0] for nliii in remapped] 
                newlii = [nliii[1] for nliii in remapped]
                new_Lii = L_map[blockpair][Lii]
                new_ltree = [(liii,niii) for niii,liii in zip(newnii,newlii)] 
                tree_i =  build_tree(new_ltree,Lii,L_R)
                #tree_i =  build_tree(new_ltree,new_Lii,L_R)
                ttup = tree_i.full_tup()
                tid = tree_i.tree_id
                atrees.append(tid)
            cond1 = not any([tid in used_ids for tid in atrees])
            #cond1 = tid not in used_ids
            if is_sigma0:
                #testing
                #cond2 = (tuple(newnii),tuple(newlii)) not in used_nl
                cond2 = True 
            else:
                #cond2 = (tuple(newnii),tuple(newlii)) not in used_nl        
                cond2 = True 

            if cond1 and cond2:
                combined_labs.append(nu)
                used_ids.append(tid)
                used_nl.append((tuple(newnii),tuple(newlii)))
                try:
                    super_inters_per_nl[(tuple(newnii),tuple(newlii))].append(new_Lii)
                except KeyError:
                    super_inters_per_nl[(tuple(newnii),tuple(newlii))] = [new_Lii]
            else:
                pass
    return combined_labs


seq_degen_map = {
((2,2,1),(5,)):(4,1),

((2,2,1),(4,1)):(4,1),
((2,1,1,1),(4,1)):(3,1,1),

((2,2,1),(2,3)):(2,2,1),
((2,1,1,1),(2,3)):(2,1,1,1),
((1,1,1,1,1),(2,3)):(2,1,1,1),
}


#apply ladder relationships
def apply_ladder_relationships_v1(lin, nin, combined_labs, parity_span, parity_span_labs, full_span, L_R=0):
    N = len(lin)
    uniques = list(set(lin))
    tmp = list(lin).copy()
    tmp.sort(key=Counter(lin).get,reverse=True)
    uniques.sort()
    uniques.sort(key=Counter(tmp).get,reverse=True)
    count_uniques =[lin.count(u) for u in uniques]
    mp = {uniques[i]:i for i in range(len(uniques))}
    mprev = {i:uniques[i] for i in range(len(uniques))}
    mappedl = [mp[t] for t in tmp]
    ysgi = Young_Subgroup(N)

    unique_ns =  list(set(nin))
    tmpn = list(nin).copy()
    tmpn.sort(key=Counter(nin).get,reverse=True)
    unique_ns.sort()
    unique_ns.sort(key=Counter(nin).get,reverse=True)
    count_unique_ns =[nin.count(u) for u in unique_ns]
    mp_n = {unique_ns[i]:i for i in range(len(unique_ns))}
    mprev_n = {i:unique_ns[i] for i in range(len(unique_ns))}
    mappedn = [mp_n[t] for t in tmpn]
    mappedn = tuple(mappedn)
    mappedl = tuple(mappedl)

    max_labs = parity_span_labs.copy()
    #mapldegenrep, maplorbit_inds = get_degen_orb(mappedl)
    #ndegen_rep = list(ndegen_rep)
    #ndegen_rep.sort(key=lambda x: x, reverse =True)
    #ndegen_rep = tuple(ndegen_rep)
    # get partition of S_N that the vector of n are commensurate with
    #  based on degeneracy
    ndegen_rep, n_orbit_inds = get_degen_orb(mappedn)
    origndegen_rep, orign_orbit_inds = get_degen_orb(nin)
    ndegen_rep = list(ndegen_rep)
    ndegen_rep.sort(key=lambda x: x, reverse =True)
    ndegen_rep = tuple(ndegen_rep)
    degen_fam = (mappedl,ndegen_rep)

    all_inters = tree_l_inters(lin)
    even_inters = simple_parity_filt(lin, all_inters, L_R)

    if 0 in lin:
        funcs = combined_labs[:len(full_span)]

    else:
        if degen_fam ==   ((0,0,0,0),(4,)):
            funcs = parity_span_labs[::3] #according to full degen ladder relationship
        elif degen_fam == ((0,0,0,0),(3,1)):
            funcs = parity_span_labs[::3]
        elif degen_fam == ((0,0,0,0),(2,2)):
            funcs = parity_span_labs[:len(parity_span)]
        elif degen_fam == ((0,0,0,0),(2,1,1)):
            funcs = parity_span_labs[:len(parity_span)]
        elif degen_fam == ((0,0,0,0),(1,1,1,1)):
            funcs = combined_labs[:len(full_span)]

        elif degen_fam == ((0,0,0,1),(4,)):
            funcs = []
            recurmax = len(max_labs)/2
            count = 0
            for lab in max_labs:
                mu0ii, muii, nii, lii, Lii = get_mu_n_l(lab,return_L=True)
                lidegen_rep, l_orbit_inds = get_degen_orb(lii)
                ysgi.subgroup_fill(list(nin),[lidegen_rep],sigma_c_symmetric=False,semistandard=False)
                degen_nfills = ysgi.fills.copy()
                if count < recurmax and tuple(nii) in degen_nfills:
                    funcs.append(lab)
                    count += 1
        elif degen_fam == ((0,0,0,1),(3,1)):
            funcs = []
            recurmax = len(max_labs)/2
            count = 0
            for lab in max_labs:
                mu0ii, muii, nii, lii, Lii = get_mu_n_l(lab,return_L=True)
                lidegen_rep, l_orbit_inds = get_degen_orb(lii)
                ysgi.subgroup_fill(list(nin),[lidegen_rep],sigma_c_symmetric=False,semistandard=False)
                degen_nfills = ysgi.fills.copy()
                if count < recurmax and tuple(nii) in degen_nfills:
                    funcs.append(lab)
                    count += 1
        elif degen_fam == ((0,0,0,1),(2,2)):
            funcs = parity_span_labs[:len(parity_span)]
        elif degen_fam == ((0,0,0,1),(2,1,1)):
            funcs = []
            recurmax = len(max_labs)/2
            count = 0
            for lab in max_labs:
                mu0ii, muii, nii, lii, Lii = get_mu_n_l(lab,return_L=True)
                lidegen_rep, l_orbit_inds = get_degen_orb(lii)
                l_sequential_degen_orbit, l_orbit_inds_s = enforce_sorted_orbit(l_orbit_inds)
                # switch to lower symmetry SN representation
                ysgi.subgroup_fill(list(nin),[l_sequential_degen_orbit],sigma_c_symmetric=False,semistandard=False)
                degen_nfills = ysgi.fills.copy()
                if count < recurmax and tuple(nii) in degen_nfills:
                    funcs.append(lab)
                    count += 1
        elif degen_fam == ((0,0,0,1),(1,1,1,1)):
            funcs = combined_labs[:len(full_span)]

        elif degen_fam == ((0,0,1,1),(4,)):
            funcs = parity_span_labs
        elif degen_fam == ((0,0,1,1),(3,1)):
            funcs = parity_span_labs
        elif degen_fam == ((0,0,1,1),(2,2)):
            funcs = combined_labs[:len(parity_span) + len(even_inters[1:])]
        elif degen_fam == ((0,0,1,1),(2,1,1)):
            funcs = combined_labs[:len(parity_span) + (2*len(even_inters[1:]))]
        elif degen_fam == ((0,0,1,1),(1,1,1,1)):
            funcs = combined_labs[:len(full_span)]

        elif degen_fam == ((0,0,1,2),(4,)):
            funcs = parity_span_labs
        elif degen_fam == ((0,0,1,2),(3,1)):
            funcs = combined_labs[:len(parity_span) + len(even_inters[1:])]
        elif degen_fam == ((0,0,1,2),(2,2)):
            funcs = combined_labs[:len(parity_span) + len(all_inters[1:])]
        elif degen_fam == ((0,0,1,2),(2,1,1)):
            funcs = combined_labs[:len(parity_span) + ((len(all_inters) -1)*2) + len(even_inters[1:])]
        elif degen_fam == ((0,0,1,2),(1,1,1,1)):
            funcs = combined_labs[:len(full_span)]

        elif degen_fam[0] == (0,1,2,3):
            funcs = combined_labs[:len(full_span)]



        elif degen_fam == ((0,0,0,0,0),(5,)):
            #print (len(full_span),len(parity_span),len(max_labs),len(even_inters))
            combined_labs.reverse()
            funcs = sorted(combined_labs[::4]) # from rank 5 ladder relationship

        elif degen_fam == ((0,0,0,0,0),(4,1)):
            #print (len(full_span),len(parity_span),len(max_labs),len(even_inters))
            combined_labs.reverse()
            funcs = sorted(combined_labs[:len(parity_span)-4])

        elif degen_fam == ((0,0,0,0,0),(3,2)):
            #print (len(full_span),len(parity_span),len(max_labs),len(even_inters))
            funcs = sorted(combined_labs[:len(parity_span)-3])

        elif degen_fam == ((0,0,0,0,0),(3,1,1)):
            #print (len(full_span),len(parity_span),len(max_labs),len(even_inters))
            combined_labs.reverse()
            funcs = sorted(combined_labs[:len(parity_span)-2])

        elif degen_fam == ((0,0,0,0,0),(2,2,1)):
            #print (len(full_span),len(parity_span),len(max_labs),len(even_inters))
            combined_labs.reverse()
            funcs = sorted(combined_labs[:len(parity_span)])

        elif degen_fam == ((0,0,0,0,0),(2,1,1,1)):
            #print (len(full_span),len(parity_span),len(max_labs),len(even_inters))
            combined_labs.reverse()
            funcs = sorted(combined_labs[:int(len(max_labs)/len(even_inters))])

        elif degen_fam == ((0,0,0,0,0),(1,1,1,1,1)):
            #print (len(full_span),len(parity_span),len(max_labs),len(even_inters))
            combined_labs.reverse()
            #funcs = sorted(combined_labs[:len(full_span)])
            funcs = sorted(combined_labs[:int(len(max_labs)/len(even_inters))])

        elif degen_fam == ((0,0,0,0,1),(5,)):
            #print (len(full_span),len(parity_span),len(max_labs),len(even_inters))
            combined_labs.reverse()
            funcs = sorted(combined_labs[:len(parity_span) - len(max_labs)])

        elif degen_fam == ((0,0,0,0,1),(4,1)):
            #print (len(full_span),len(parity_span),len(max_labs),len(even_inters))
            combined_labs.reverse()
            funcs = sorted(combined_labs[:int(len(parity_span)/2)])
            #or:
            #funcs = sorted(combined_labs[:len(even_inters)])

        elif degen_fam == ((0,0,0,0,1),(3,2)):
            #print (len(full_span),len(parity_span),len(max_labs),len(even_inters))
            combined_labs.reverse()
            funcs = sorted(combined_labs[:len(parity_span) -1])

        elif degen_fam == ((0,0,0,0,1),(3,1,1)):
            #print (len(full_span),len(parity_span),len(max_labs),len(even_inters))
            funcs = combined_labs[:len(parity_span) -1]

        elif degen_fam == ((0,0,0,0,1),(2,2,1)):
            #print (len(full_span),len(parity_span),len(max_labs),len(even_inters))
            funcs = combined_labs[:len(parity_span) + (2*int(len(even_inters)/len(degen_fam[1])))]

        elif degen_fam == ((0,0,0,0,1),(2,1,1,1)):
            #print (len(full_span),len(parity_span),len(max_labs),len(even_inters))
            funcs = combined_labs[:len(parity_span) + (2*len(even_inters))]

        elif degen_fam == ((0,0,0,0,1),(1,1,1,1,1)):
            funcs = combined_labs[:len(full_span)]

        elif degen_fam == ((0,0,0,1,1),(5,)):
            #print (len(full_span),len(parity_span),len(max_labs))
            #funcs = parity_span_labs[:len(parity_span)]
            funcs = [] 
            for lab in parity_span_labs:
                mu0ii, muii, nii, lii, Lii = get_mu_n_l(lab,return_L=True)
                if 0 not in Lii:
                    funcs.append(lab)

        elif degen_fam == ((0,0,0,1,1),(4,1)):
            #print (len(full_span),len(parity_span),len(max_labs))
            funcs = combined_labs[:int(len(parity_span)) - len(degen_fam[1])]

        elif degen_fam == ((0,0,0,1,1),(3,2)):
            funcs = combined_labs[:len(parity_span) -1]

        elif degen_fam == ((0,0,0,1,1),(3,1,1)):
            funcs = combined_labs[:int(len(full_span)/2)]

        elif degen_fam == ((0,0,0,1,1),(2,2,1)):
            funcs = combined_labs[:int(len(max_labs)/2) - len(even_inters)]

        elif degen_fam == ((0,0,0,1,1),(2,1,1,1)):
            funcs = combined_labs[:int(len(max_labs)/2) - (3*len(even_inters))]

        elif degen_fam == ((0,0,0,1,1),(1,1,1,1,1)):
            funcs = combined_labs[:len(full_span)]
            
        else:
            # return funcs undefined if none of the elif clauses match degen_fam
            # --> 461 return funcs
            # UnboundLocalError: cannot access local variable 'funcs' where it is not associated with a value
            raise RuntimeError(f"degen_fam {degen_fam} missing from elif block")

    return funcs

















"""
rank =5
lmax4_strs = generate_l_LR(range(5),rank,L_R=0,M_R=0)
lmax5_strs = generate_l_LR(range(6),rank,L_R=0,M_R=0)
lmax7_strs = generate_l_LR(range(8),rank,L_R=0,M_R=0)
lmax3_strs = generate_l_LR(range(4),rank,L_R=0,M_R=0)
lmax2_strs = generate_l_LR(range(3),rank,L_R=0,M_R=0)
lmax1_strs = generate_l_LR(range(2),rank,L_R=0,M_R=0)
lmax4s = [[int(k) for k in li.split(',')] for li in lmax4_strs]
lmax5s = [[int(k) for k in li.split(',')] for li in lmax5_strs]
lmax7s = [[int(k) for k in li.split(',')] for li in lmax7_strs]
lmax3s = [[int(k) for k in li.split(',')] for li in lmax3_strs]
lmax2s = [[int(k) for k in li.split(',')] for li in lmax2_strs]
lmax1s = [[int(k) for k in li.split(',')] for li in lmax1_strs]
nmax4s = [i for i in itertools.combinations_with_replacement(range(0,4),rank)]
nmax5s = [i for i in itertools.combinations_with_replacement(range(0,5),rank)]
nmax3s = [i for i in itertools.combinations_with_replacement(range(0,3),rank)]

reduced_nmax4s=get_mapped_subset(nmax4s)
reduced_nmax5s=get_mapped_subset(nmax5s)

fs_labs = []
all_nl = []

all_PA_tabulated = []
PA_per_nlblock = {}
#for nin in reduced_nmax4s:
for nin in reduced_nmax4s:
    #for lin in lmax3s:
    #for lin in lmax4s:
    #for lin in lmax7s:
    for lin in lmax2s:
        max_labs,all_labs,labels_per_block,original_spans = tree_labels(nin,lin)
        combined_labs = combine_blocks(labels_per_block,lin,original_spans)
        nl = (nin,lin)
        lspan_perm = list(original_spans.keys())[0]
        parity_span = [p for p in original_spans[lspan_perm] if np.sum(lspan_perm[:2] + p[2][:1]) %2 == 0 and np.sum(lspan_perm[2:4] + p[2][1:2]) %2 == 0]

        PA_labels = apply_ladder_relationships(lin, nin, combined_labs, parity_span, parity_span_labs = max_labs, full_span=original_spans[lspan_perm])
        mustrlst = ['%d']*rank
        nstrlst = ['%d']*rank
        lstrlst = ['%d']*rank
        Lstrlst = ['%d']*(rank-2)
        nl_simple_labs = []
        nlstr = ','.join(nstrlst) % tuple(nin) + '_' + ','.join(lstrlst) % tuple(lin)
        for lab in PA_labels:
            mu0,mu,n,l,L = get_mu_n_l(lab,return_L=True)
            if L != None:
                nlL = (tuple(n),tuple(l),L)
            else:
                nlL = (tuple(n),tuple(l),tuple([]))
            simple_str = ','.join(nstrlst) % tuple(n) + '_' + ','.join(lstrlst) % tuple(l) + '_' + ','.join(Lstrlst) % L
            print (simple_str)
            all_PA_tabulated.append(simple_str)
            nl_simple_labs.append(simple_str)
        PA_per_nlblock[nlstr] = nl_simple_labs

#dct = {'labels':all_PA_tabulated}
dct = {'labels':PA_per_nlblock}
import json
with open('all_labels_r5.json','w') as writejson:
    json.dump(dct,writejson, sort_keys=False, indent=2)
import json
with open('all_labels_r5.json','r') as readjson:
    data = json.load(readjson)
#from_tabulated((0,0,1,1),(1,1,2,2),(2,2,3,3),tabulated_all=data)
possible_mus = list(range(2))
rank = 4
nus = from_tabulated((0,0,0,0),(1,1,1,1),(4,4,4,4),allowed_mus = possible_mus, tabulated_all = data)
lammps_ready,not_compatible = lammps_remap(nus,rank=rank,allowed_mus=possible_mus)

possible_mus = list(range(2))
rank = 5
nus = from_tabulated((0,0,0,0,0),(1,1,1,2,2),(2,2,2,2,2),allowed_mus = possible_mus, tabulated_all = data)
lammps_ready,not_compatible = lammps_remap(nus,rank=rank,allowed_mus=possible_mus)

print ('raw PA-RPI',nus)
print ('lammps ready PA-RPI',lammps_ready)
print ('not compatible with lammps (PA-RPI with a nu vector that cannot be reused)',not_compatible)
"""

