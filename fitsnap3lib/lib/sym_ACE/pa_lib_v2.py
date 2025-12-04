import numpy as np
import multiprocessing
from sympy.physics.wigner import wigner_3j
from itertools import product, permutations
import warnings
import os
import pickle
from collections import Counter, defaultdict
from functools import partial

# ---------------------------------------------------------
# PERSISTENT CACHE MANAGER
# ---------------------------------------------------------

CACHE_FILE = "wigner_cache.pckl"

class WignerCacheManager:
    def __init__(self):
        self.cache = {}
        self.new_entries = {}

    def load(self):
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "rb") as f:
                    self.cache = pickle.load(f)
            except (EOFError, pickle.UnpicklingError):
                self.cache = {}

    def save(self):
        if not self.new_entries: return
        self.cache.update(self.new_entries)
        try:
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception: pass

# ---------------------------------------------------------
# EXACT WIGNER CALCULATION
# ---------------------------------------------------------

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

# ---------------------------------------------------------
# ITERATIVE TRELLIS & PERMUTATIONS
# ---------------------------------------------------------

def _trellis_worker_iterative(prefix, l_leaves):
    rank = len(l_leaves)
    prefix_len = len(prefix)
    prefix_sum = sum(prefix)
    
    if prefix_len == rank - 1:
        m_last = -prefix_sum
        if abs(m_last) <= l_leaves[-1]:
            return [prefix + (m_last,)]
        return []

    results = []
    stack = [(prefix_len, prefix_sum, ())]
    
    remaining_cap = [0] * rank
    s = 0
    for i in range(rank - 1, -1, -1):
        remaining_cap[i] = s
        s += l_leaves[i]

    while stack:
        idx, curr_sum, path = stack.pop()
        
        if idx == rank - 1:
            m_last = -curr_sum
            if abs(m_last) <= l_leaves[idx]:
                results.append(prefix + path + (m_last,))
            continue
            
        cap = remaining_cap[idx]
        limit = l_leaves[idx]
        min_m = max(-limit, -cap - curr_sum)
        max_m = min(limit, cap - curr_sum)
        
        for m in range(min_m, max_m + 1):
            stack.append((idx + 1, curr_sum + m, path + (m,)))
            
    return results

def generate_valid_m_states_parallel(l_leaves, n_cores):
    rank = len(l_leaves)
    if rank == 0: return [()]
    
    target_tasks = n_cores * 16 # Increased for better saturation
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
    """
    Returns valid permutation indices mapping ref -> curr.
    Items are only swappable if (mu, n, l) ALL match.
    
    ref_full: list of (n, l) tuples [canonical]
    curr_full: list of (n, l) tuples [current label]
    """
    if len(ref_full) != len(curr_full): return []

    # Bucket indices by full property tuple (n, l)
    ref_indices = defaultdict(list)
    for i, prop in enumerate(ref_full):
        ref_indices[prop].append(i)
        
    curr_indices = defaultdict(list)
    for i, prop in enumerate(curr_full):
        curr_indices[prop].append(i)
        
    if ref_indices.keys() != curr_indices.keys():
        return []

    # Generate per-bucket permutations
    per_bucket_perms = []
    for prop, c_idxs in curr_indices.items():
        r_idxs = ref_indices[prop]
        bucket_maps = []
        for p in permutations(r_idxs):
            # mapping[curr_idx] = ref_idx
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

# ---------------------------------------------------------
# WORKER PROCESS
# ---------------------------------------------------------

def worker_process_exact(chunk_labels, rank, m_configs, ref_full_sorted, initial_cache):
    local_cache = initial_cache.copy()
    initial_keys = set(local_cache.keys())
    results = []
    
    for label in chunk_labels:
        n_leaves, l_leaves, l_inters = parse_label_full(label, rank)
        curr_full = list(zip(n_leaves, l_leaves))
        
        # Only swap indistinguishable atoms (same n, l)
        perm_maps = get_constrained_permutations(ref_full_sorted, curr_full)
        if not perm_maps: continue

        w_vec = []
        is_zero_vec = True
        
        for m_ref in m_configs:
            val = np.float64(0.0)
            
            # Sum Wigner coefficients over valid permutations
            # (Projecting onto Symmetrized Basis)
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

# ---------------------------------------------------------
# UTILS & THEORY
# ---------------------------------------------------------

def parse_label_full(label, rank):
    """
    Parses full (mu, n, l, L) structure.
    Label format: mu0_mu1...muk,n1...nk,l1...lk_L1...Lk
    """
    parts = label.split('_')
    integers_block = parts[1].split(',')
    
    # Assuming integers_block has 3*rank elements: mu, n, l
    # If 2*rank: n, l
    
    total_ints = len(integers_block)
    
    if total_ints == 2 * rank:
        n_leaves = [int(x) for x in integers_block[:rank]]
        l_leaves = [int(x) for x in integers_block[rank:]]
    elif total_ints == 3 * rank:
        # mu = integers_block[:rank]
        n_leaves = [int(x) for x in integers_block[rank:2*rank]]
        l_leaves = [int(x) for x in integers_block[2*rank:]]
    else:
        # Fallback/Unknown format, assume l is last
        l_leaves = [int(x) for x in integers_block[-rank:]]
        n_leaves = [0] * rank # Dummy

    l_inters = []
    if len(parts) > 2:
        try: l_inters = [int(x) for x in parts[-1].split('-')]
        except ValueError: l_inters = []
    if len(l_inters) > rank - 2: l_inters = l_inters[:rank-2]
    
    return n_leaves, l_leaves, l_inters

def parse_label_to_tree(label, rank):
    # Wrapper for legacy calls (only needing l)
    _, l, inters = parse_label_full(label, rank)
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

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def apply_ladder_relationships(lin, nin, combined_labs, parity_span, parity_span_labs, full_span, L_R=0):
    
    n_expected = CharacterIntegration.count_invariants(lin, nin)
    if n_expected == 0 or not combined_labs: return []

    pairs = list(zip(nin, lin))
    if len(set(pairs)) == len(pairs): return combined_labs[:n_expected]

    rank = len(lin)
    
    # 2. Setup Vector Space
    # Use first label to define canonical space.
    # Note: Ref must capture n and l to define distinguishable particles
    ref_n, ref_l, _ = parse_label_full(combined_labs[0], rank)
    
    # Sort the reference so we have a canonical "Identity" state
    # Sort key: (n, l)
    combined = sorted(list(zip(ref_n, ref_l)))
    ref_n_sorted = [x[0] for x in combined]
    ref_l_sorted = [x[1] for x in combined]
    ref_full_sorted = list(zip(ref_n_sorted, ref_l_sorted))
    
    # Generate m-basis for the sorted l-vector
    n_cores = multiprocessing.cpu_count()
    m_configs = generate_valid_m_states_parallel(ref_l_sorted, n_cores)
    
    if not m_configs: return []

    # 3. Cache
    cache_manager = WignerCacheManager()
    cache_manager.load()

    # 4. Parallel Wigner Calculation
    chunk_size = max(1, len(combined_labs) // (n_cores * 4))
    chunks = [combined_labs[i:i + chunk_size] for i in range(0, len(combined_labs), chunk_size)]
    
    valid_candidates = []
    
    with multiprocessing.Pool(processes=n_cores) as pool:
        # Pass ref_full_sorted so workers know which atoms are distinguishable
        results = pool.starmap(worker_process_exact, 
                               [(chunk, rank, m_configs, ref_full_sorted, cache_manager.cache) 
                                for chunk in chunks])
        
        for sub_res, sub_cache in results:
            valid_candidates.extend(sub_res)
            cache_manager.new_entries.update(sub_cache)

    cache_manager.save()

    # 5. Orthogonalization
    independent_labs = []
    basis_matrix = [] 
    
    for label, w_vec in valid_candidates:
        if len(independent_labs) == n_expected: break
            
        if len(basis_matrix) == 0:
            basis_matrix.append(w_vec)
            independent_labs.append(label)
        else:
            vec_ortho = w_vec.copy()
            B = np.array(basis_matrix, dtype=np.float64)
            projections = np.dot(B, vec_ortho)
            vec_ortho -= np.dot(projections, B)
            
            projections_2 = np.dot(B, vec_ortho)
            vec_ortho -= np.dot(projections_2, B)
            
            ortho_norm = np.linalg.norm(vec_ortho)
            
            if ortho_norm > 1e-13:
                basis_matrix.append(vec_ortho / ortho_norm)
                independent_labs.append(label)

    if len(independent_labs) < n_expected:
        warnings.warn(f"Under-complete Basis for n={nin}, l={lin}. Theory: {n_expected}, Found: {len(independent_labs)}")

    return independent_labs
