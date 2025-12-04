import numpy as np
import multiprocessing
from sympy.physics.wigner import wigner_3j
from itertools import product
import warnings
import os
import pickle
from collections import Counter

# ---------------------------------------------------------
# PERSISTENT CACHE MANAGER
# ---------------------------------------------------------

CACHE_FILE = "wigner_cache.pkl"

class WignerCacheManager:
    """
    Manages a persistent dictionary of Wigner-3j symbols to avoid 
    recomputing expensive exact symbolic values across runs.
    """
    def __init__(self):
        self.cache = {}
        self.new_entries = {}
        self.loaded = False

    def load(self):
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "rb") as f:
                    self.cache = pickle.load(f)
                # print(f"Loaded {len(self.cache)} Wigner-3j symbols from {CACHE_FILE}")
            except (EOFError, pickle.UnpicklingError):
                print("Cache corrupted or empty, starting fresh.")
                self.cache = {}
        self.loaded = True

    def save(self):
        if not self.new_entries:
            return
        
        # Merge new entries into main cache
        self.cache.update(self.new_entries)
        
        try:
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(self.cache, f)
            print(f"Saved {len(self.cache)} Wigner-3j symbols to {CACHE_FILE}")
        except Exception as e:
            print(f"Warning: Could not save Wigner cache: {e}")

# ---------------------------------------------------------
# EXACT WIGNER CALCULATION (DOUBLE PRECISION)
# ---------------------------------------------------------

def get_exact_w3j(j1, j2, j3, m1, m2, m3, cache=None):
    """
    Computes exact Wigner-3j symbol using SymPy and casts to np.float64.
    """
    key = (j1, j2, j3, m1, m2, m3)
    
    # Check Cache
    if cache is not None:
        if key in cache:
            return cache[key]
    
    # Compute Exact
    try:
        # SymPy calculation is exact (rational/sqrt)
        val_sym = wigner_3j(j1, j2, j3, m1, m2, m3)
        # Convert to 64-bit Double Precision
        val = np.float64(val_sym)
    except Exception:
        val = np.float64(0.0)

    # Store in cache
    if cache is not None:
        cache[key] = val
        
    return val

def calculate_generalized_wigner_exact(l_leaves, l_inters, m_config, local_cache):
    """
    Calculates Generalized Wigner Symbol using Exact 3j kernels.
    Returns np.float64.
    """
    val = np.float64(1.0)
    current_L = list(l_leaves)
    current_m = list(m_config)
    inters_queue = list(l_inters)
    
    # Global selection rule check
    if sum(current_m) != 0: return np.float64(0.0)

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
                
                # Strict Triangle Inequality
                if not (abs(l1 - l2) <= L_res <= l1 + l2): return np.float64(0.0)
                if abs(m1) > l1 or abs(m2) > l2 or abs(M_res) > L_res: return np.float64(0.0)

                # EXACT CALL with Caching
                w3j = get_exact_w3j(l1, l2, L_res, m1, m2, -M_res, local_cache)
                
                # Use double precision tolerance
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
# DETERMINISTIC WALKER & UTILS
# ---------------------------------------------------------

def generate_valid_m_states(l_leaves):
    rank = len(l_leaves)
    if rank == 0: yield []; return
    bounds = [list(range(-li, li + 1)) for li in l_leaves]
    
    def backtrack(index, current_sum):
        if index == rank - 1:
            m_last = -current_sum
            if abs(m_last) <= l_leaves[index]: yield (m_last,)
            return
        
        remaining_capacity = sum(l_leaves[index+1:])
        for m in bounds[index]:
            if abs(current_sum + m) <= remaining_capacity:
                for res in backtrack(index + 1, current_sum + m):
                    yield (m,) + res

    for res in backtrack(0, 0):
        yield res

def parse_label_to_tree(label, rank):
    parts = label.split('_')
    integers_block = parts[1].split(',')
    l_leaves = [int(x) for x in integers_block[-rank:]]
    l_inters = []
    if len(parts) > 2:
        inters_str = parts[-1].split('-')
        try: l_inters = [int(x) for x in inters_str]
        except ValueError: l_inters = []
    if len(l_inters) > rank - 2: l_inters = l_inters[:rank-2]
    return l_leaves, l_inters

def get_permutation_map(l_ref, l_curr):
    if tuple(l_ref) == tuple(l_curr): return list(range(len(l_ref)))
    indices = [None] * len(l_ref)
    used = [False] * len(l_ref)
    for k, val in enumerate(l_curr):
        found = False
        for i, ref_val in enumerate(l_ref):
            if ref_val == val and not used[i]:
                indices[k] = i; used[i] = True; found = True; break
        if not found: return list(range(len(l_ref))) 
    return indices

# ---------------------------------------------------------
# THEORY ORACLE
# ---------------------------------------------------------
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
# WORKER
# ---------------------------------------------------------

def worker_process_exact(chunk_labels, rank, m_configs, ref_l_sorted, initial_cache):
    """
    Computes exact Wigner vectors using double precision.
    Returns: (list_of_results, new_cache_entries)
    """
    local_cache = initial_cache.copy()
    initial_keys = set(local_cache.keys())
    
    results = []
    
    for label in chunk_labels:
        l_leaves, l_inters = parse_label_to_tree(label, rank)
        
        try:
            perm_indices = get_permutation_map(ref_l_sorted, l_leaves)
        except ValueError:
            continue

        w_vec = []
        is_zero_vec = True
        
        for m_ref in m_configs:
            m_curr = [m_ref[i] for i in perm_indices]
            val = calculate_generalized_wigner_exact(l_leaves, l_inters, m_curr, local_cache)
            w_vec.append(val)
            if abs(val) > 1e-15: is_zero_vec = False
            
        if not is_zero_vec:
            vec = np.array(w_vec, dtype=np.float64) # Force Double
            norm = np.linalg.norm(vec)
            if norm > 1e-15:
                results.append((label, vec / norm))
    
    new_entries = {k: v for k, v in local_cache.items() if k not in initial_keys}
    
    return results, new_entries

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def apply_ladder_relationships(lin, nin, combined_labs, parity_span, parity_span_labs, full_span, L_R=0):
    
    # 1. Theoretical Count
    n_expected = CharacterIntegration.count_invariants(lin, nin)
    if n_expected == 0 or not combined_labs: return []

    pairs = list(zip(nin, lin))
    if len(set(pairs)) == len(pairs): return combined_labs[:n_expected]

    rank = len(lin)
    
    # 2. Setup Vector Space
    ref_l, _ = parse_label_to_tree(combined_labs[0], rank)
    ref_l_sorted = sorted(ref_l)
    m_configs = list(generate_valid_m_states(ref_l_sorted))
    if not m_configs: return []

    # 3. Cache Management
    cache_manager = WignerCacheManager()
    cache_manager.load()

    # 4. Parallel Execution
    n_cores = multiprocessing.cpu_count()
    chunk_size = max(1, len(combined_labs) // n_cores)
    chunks = [combined_labs[i:i + chunk_size] for i in range(0, len(combined_labs), chunk_size)]
    
    valid_candidates = []
    
    with multiprocessing.Pool(processes=n_cores) as pool:
        results = pool.starmap(worker_process_exact, 
                               [(chunk, rank, m_configs, ref_l_sorted, cache_manager.cache) 
                                for chunk in chunks])
        
        for sub_res, sub_cache in results:
            valid_candidates.extend(sub_res)
            cache_manager.new_entries.update(sub_cache)

    cache_manager.save()

    # 5. Orthogonalization (QR with Double Precision)
    independent_labs = []
    basis_matrix = [] 
    
    for label, w_vec in valid_candidates:
        if len(independent_labs) == n_expected: break
            
        if len(basis_matrix) == 0:
            basis_matrix.append(w_vec)
            independent_labs.append(label)
        else:
            vec_ortho = w_vec.copy()
            B = np.array(basis_matrix, dtype=np.float64) # Force Double
            
            # Vectorized projection
            projections = np.dot(B, vec_ortho)
            vec_ortho -= np.dot(projections, B)
            
            ortho_norm = np.linalg.norm(vec_ortho)
            
            # Tighter tolerance for Exact Inputs
            if ortho_norm > 1e-14:
                basis_matrix.append(vec_ortho / ortho_norm)
                independent_labs.append(label)

    if len(independent_labs) < n_expected:
        warnings.warn(f"Under-complete Basis for n={nin}, l={lin}. Theory: {n_expected}, Found: {len(independent_labs)}")

    return independent_labs
