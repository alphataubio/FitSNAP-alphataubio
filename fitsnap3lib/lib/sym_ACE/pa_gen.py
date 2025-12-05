import numpy as np
import multiprocessing
from sympy.physics.wigner import wigner_3j
from itertools import product, permutations, combinations_with_replacement
import warnings
import os
import json
import pickle
from collections import Counter, defaultdict
from functools import partial, lru_cache
import scipy.linalg
from tqdm import tqdm  # Progress bar support
from fitsnap3lib.lib.sym_ACE.sym_ACE_settings import lib_path


# -----------------------------------------------------------------------------
# GEMINI PRO 3's "PA-RPI Basis with Group Theory & Wigner Algebra"
# https://github.com/FitSNAP/FitSNAP/pull/278#issuecomment-3608889321
# https://github.com/FitSNAP/FitSNAP/pull/278#issuecomment-3608897109
# https://github.com/FitSNAP/FitSNAP/pull/278#issuecomment-3614955527
# [alphataubio, 2025/12]
# -----------------------------------------------------------------------------


# Ensure lib_path exists
if not os.path.exists(lib_path):
    lib_path = os.path.dirname(os.path.abspath(__file__))

CACHE_FILE = os.path.join(lib_path, "wigner_cache.pckl")

# =========================================================
# SECTION 1: BASIS GENERATION & PARSING
# =========================================================

class BasisGenerator:
    """Generates the Over-Complete set of ACE basis labels."""
    @staticmethod
    def get_valid_intermediates(l1, l2):
        return range(abs(l1 - l2), l1 + l2 + 1)

    @staticmethod
    def generate_valid_intermediates_recursive(l_current_layer, L_R):
        if len(l_current_layer) == 1:
            if l_current_layer[0] == L_R: yield []
            return
        
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
        rank = len(lin)
        if muin is None: muin = tuple([0] * rank)
        atoms = sorted(list(zip(muin, nin, lin)))
        unique_perms = sorted(list(set(permutations(atoms))))
        
        labels = []
        for p in unique_perms:
            p_mu, p_n, p_l = zip(*p)
            valid_L_tuples = list(BasisGenerator.generate_valid_intermediates_recursive(p_l, L_R))
            
            for L_tuple in valid_L_tuples:
                L_clean = L_tuple[:rank-2] if len(L_tuple) > rank - 2 else L_tuple
                L_str = "-".join(map(str, L_clean))
                integers = list(p_mu) + list(p_n) + list(p_l)
                int_str = ",".join(map(str, integers))
                labels.append(f"0_{int_str}_{L_str}")
        return sorted(list(set(labels)))

def parse_label_full(label, rank):
    parts = label.split('_')
    integers_block = parts[1].split(',')
    total_ints = len(integers_block)
    
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
    if len(parts) > 2 and parts[-1]:
        try: l_inters = [int(x) for x in parts[-1].split('-')]
        except ValueError: l_inters = []
    
    if len(l_inters) > rank - 2: l_inters = l_inters[:rank-2]
    return mu_leaves, n_leaves, l_leaves, l_inters

def get_mu_n_l(nu_in, return_L=False, **kwargs):
    """Legacy helper for backward compatibility."""
    try:
        if len(nu_in.split('_')) > 1:
            integers = nu_in.split('_')[1].split(',')
            rank = len(integers)//3 if len(integers)%3==0 else len(integers)//2
        else:
            rank = len(nu_in.split(',')) // 2
        mu, n, l, L = parse_label_full(nu_in, rank)
        mu0 = int(nu_in.split('_')[0])
        if return_L: return mu0, tuple(mu), tuple(n), tuple(l), tuple(L)
        return mu0, tuple(mu), tuple(n), tuple(l)
    except Exception: return 0, (), (), ()

# =========================================================
# SECTION 2: WIGNER CACHE & MATH
# =========================================================

# Global cache for workers (Copy-on-Write)
_worker_wigner_cache = {}

def _init_pool(shared_cache):
    """Initialize worker process with the wigner cache."""
    global _worker_wigner_cache
    _worker_wigner_cache = shared_cache

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

@lru_cache(maxsize=100000)
def get_exact_w3j(j1, j2, j3, m1, m2, m3):
    """Memory cached exact wigner calculation."""
    try:
        return np.float64(wigner_3j(j1, j2, j3, m1, m2, m3))
    except Exception:
        return np.float64(0.0)

def get_w3j_from_global(j1, j2, j3, m1, m2, m3, local_updates):
    """Check global cache, then compute."""
    key = (j1, j2, j3, m1, m2, m3)
    if key in _worker_wigner_cache:
        return _worker_wigner_cache[key]
    
    val = get_exact_w3j(j1, j2, j3, m1, m2, m3)
    local_updates[key] = val
    return val

def calculate_generalized_wigner_exact(l_leaves, l_inters, m_config, local_updates):
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
                
                if len(current_L) == 2: L_res = 0 
                else:
                     if not inters_queue: return np.float64(0.0)
                     L_res = inters_queue.pop(0)
                
                M_res = m1 + m2
                if not (abs(l1 - l2) <= L_res <= l1 + l2): return np.float64(0.0)
                if abs(m1) > l1 or abs(m2) > l2 or abs(M_res) > L_res: return np.float64(0.0)

                w3j = get_w3j_from_global(l1, l2, L_res, m1, m2, -M_res, local_updates)
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

# =========================================================
# SECTION 4: TRAVERSAL & SELECTION (SERIAL KERNELS)
# =========================================================

def _generate_m_states_iterative(l_leaves):
    """Robust iterative m-space walker."""
    rank = len(l_leaves)
    stack = [(0, 0, ())] # idx, current_sum, path
    
    remaining_cap = [0] * rank
    s = 0
    for i in range(rank - 1, -1, -1):
        remaining_cap[i] = s
        s += l_leaves[i]

    RELAX = 0
    results = []
    
    while stack:
        idx, curr_sum, path = stack.pop()
        
        if idx == rank - 1:
            m_last = -curr_sum
            if abs(m_last) <= l_leaves[idx]:
                results.append(path + (m_last,))
            continue
            
        cap = remaining_cap[idx]
        limit = l_leaves[idx]
        min_m = max(-limit, -cap - curr_sum - RELAX)
        max_m = min(limit, cap - curr_sum + RELAX)
        
        for m in range(min_m, max_m + 1):
            stack.append((idx + 1, curr_sum + m, path + (m,)))
    return results

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

# =========================================================
# SECTION 5: CORE LOGIC (Single Block)
# =========================================================

def apply_ladder_relationships_serial(lin, nin, L_R=0):
    """
    Core Kernel: Solves one block (n, l) serially.
    Returns (independent_labels, new_cache_entries).
    """
    local_new_cache = {}
    
    # 1. Recover mu/n
    real_n, real_mu = [], []
    for x in nin:
        if x >= 1000:
            real_mu.append(x // 1000)
            real_n.append(x % 1000)
        else:
            real_mu.append(0)
            real_n.append(x)
            
    # 2. Theory Count
    n_expected = CharacterIntegration.count_invariants(lin, nin)
    if n_expected == 0: return [], {}

    # 3. Generate Candidates
    combined_labs = BasisGenerator.get_canonical_labels(tuple(real_n), lin, tuple(real_mu), L_R=L_R)
    if not combined_labs: return [], {}

    # Optimization: Distinct pairs
    pairs = list(zip(nin, lin))
    if len(set(pairs)) == len(pairs):
        return combined_labs, {}

    # 4. Setup Vector Space
    rank = len(lin)
    ref_mu, ref_n, ref_l, _ = parse_label_full(combined_labs[0], rank)
    combined = sorted(list(zip(ref_mu, ref_n, ref_l)))
    ref_full_sorted = list(zip([x[0] for x in combined], [x[1] for x in combined], [x[2] for x in combined]))
    ref_l_sorted = [x[2] for x in combined]
    
    m_configs = _generate_m_states_iterative(ref_l_sorted)
    if not m_configs: return [], {}

    # 5. Build Vectors (Serial)
    valid_candidates = []
    
    for label in combined_labs:
        mu_leaves, n_leaves, l_leaves, l_inters = parse_label_full(label, rank)
        curr_full = list(zip(mu_leaves, n_leaves, l_leaves))
        
        perm_maps = get_constrained_permutations(ref_full_sorted, curr_full)
        if not perm_maps: continue

        w_vec = []
        is_zero_vec = True
        
        for m_ref in m_configs:
            val = np.float64(0.0)
            # Symmetrize over valid permutations
            for indices in perm_maps:
                m_curr = [m_ref[i] for i in indices]
                w = calculate_generalized_wigner_exact(l_leaves, l_inters, m_curr, local_new_cache)
                val += w
            w_vec.append(val)
            if abs(val) > 1e-15: is_zero_vec = False
            
        if not is_zero_vec:
            vec = np.array(w_vec, dtype=np.float64)
            norm = np.linalg.norm(vec)
            if norm > 1e-15:
                valid_candidates.append((label, vec / norm))

    if not valid_candidates:
        if n_expected > 0: warnings.warn(f"All vectors zero for block n={nin}, l={lin}")
        return [], local_new_cache

    # 6. SVD Selection
    labels = [x[0] for x in valid_candidates]
    A = np.column_stack([x[1] for x in valid_candidates])
    
    try:
        U, S, Vt = scipy.linalg.svd(A, full_matrices=False)
        max_sv = S[0]
        tolerance = max(1e-14, max_sv * 1e-12)
        rank_eff = np.sum(S > tolerance)
    except Exception:
        rank_eff = min(A.shape)
        tolerance = 1e-13

    independent_labs_final = []
    basis_matrix = []
    
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

    return independent_labs_final, local_new_cache

# =========================================================
# SECTION 6: GENERATION WRAPPER (OUTER LOOP PARALLELISM)
# =========================================================

def _worker_wrapper(args):
    """Unpacks arguments for the pool."""
    lin, nin, L_R = args
    # Calls the serial kernel
    return apply_ladder_relationships_serial(lin, nin, L_R)

def get_cache_filename(rank, nmax, lmax, mumax):
    return os.path.join(lib_path, f"pa_rpi_cache_r{rank}_n{nmax}_l{lmax}_mu{mumax}.json")

def build_and_cache_basis(rank, nmax, lmax, mumax, lmin=1, L_R=0):
    print(f"*** Generating Basis Cache for Rank={rank}, Nmax={nmax}, Lmax={lmax}, Mumax={mumax}...")
    
    mus = range(mumax)
    ns = range(1, nmax + 1)
    ls = range(lmin, lmax + 1)
    
    # 1. Prepare Tasks
    l_combs = [l for l in combinations_with_replacement(ls, rank) if sum(l) % 2 == 0]
    n_combs = list(combinations_with_replacement(ns, rank))
    mu_combs = list(combinations_with_replacement(mus, rank))
    
    tasks = []
    for lin in l_combs:
        for nin in n_combs:
            for muin in mu_combs:
                comp_n = tuple([m * 1000 + n for m, n in zip(muin, nin)])
                tasks.append((lin, comp_n, L_R))
    
    print(f"*** Total Blocks to Process: {len(tasks)}")
    
    # 2. Load Wigner Cache
    wigner_mgr = WignerCacheManager()
    wigner_mgr.load()
    
    # 3. Parallel Execution (Outer Loop)
    n_cores = multiprocessing.cpu_count()
    all_lammps_labs = []
    
    # Chunksize: For 700k tasks, set chunksize to handle batch overhead
    chunksize = max(1, len(tasks) // (n_cores * 10))
    
    # Use imap to guarantee order of results matches order of tasks
    # We pass wigner_mgr.cache via initializer to share it read-only via COW
    with multiprocessing.Pool(processes=n_cores, initializer=_init_pool, initargs=(wigner_mgr.cache,)) as pool:
        # TQDM Wrapped Iterator
        for labels, new_cache in tqdm(pool.imap(_worker_wrapper, tasks, chunksize=chunksize), total=len(tasks), desc="Processing Blocks"):
            if labels:
                all_lammps_labs.extend(labels)
            if new_cache:
                wigner_mgr.new_entries.update(new_cache)
                
    # 4. Save Cache
    wigner_mgr.save()
    
    # 5. Write Result
    cache_file = get_cache_filename(rank, nmax, lmax, mumax)
    cache_data = {
        "params": {"rank": rank, "nmax": nmax, "lmax": lmax, "mumax": mumax},
        "labels": all_lammps_labs
    }
    
    try:
        with open(cache_file, 'w') as f: json.dump(cache_data, f, indent=2)
        print(f"*** Cached {len(all_lammps_labs)} labels to {cache_file}")
    except IOError as e:
        print(f"*** Warning: Could not write cache file: {e}")

    return all_lammps_labs

def pa_labels_raw(rank, nmax, lmax, mumax, lmin=1, L_R=0, M_R=0):
    if rank < 1: return [], []
    cache_file = get_cache_filename(rank, nmax, lmax, mumax)
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                #print(f"*** Loaded basis from cache: {cache_file}")
                return data['labels'], []
        except Exception:
            print("*** Cache corrupted, regenerating...")

    labels = build_and_cache_basis(rank, nmax, lmax, mumax, lmin, L_R)
    return labels, []
