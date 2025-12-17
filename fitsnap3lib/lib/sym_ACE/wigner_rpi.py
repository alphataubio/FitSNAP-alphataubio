import numpy as np
from sympy.physics.wigner import wigner_3j
from itertools import product, permutations, combinations_with_replacement
import os
import json
from collections import Counter, defaultdict
import scipy.linalg
from tqdm import tqdm
from fitsnap3lib.lib.sym_ACE.sym_ACE_settings import lib_path

# -----------------------------------------------------------------------------
# GEMINI PRO 3's "PA-RPI Basis with Group Theory & Wigner Algebra"
# https://github.com/FitSNAP/FitSNAP/pull/278#issuecomment-3608889321
# https://github.com/FitSNAP/FitSNAP/pull/278#issuecomment-3608897109
# https://github.com/FitSNAP/FitSNAP/pull/278#issuecomment-3614955527
# [alphataubio, 2025/12]
# -----------------------------------------------------------------------------

try:
    from mpi4py import MPI
    _HAS_MPI = True
except ImportError:
    _HAS_MPI = False

# =========================================================
# SECTION 1: GROUP THEORY (MOLIEN-WEYL INTEGRATION)
# =========================================================

class MolienWeylIntegration:
    """Computes the number of independent invariants for verification."""
    @staticmethod
    def chi_l(l, theta):
        if abs(theta) < 1e-9: return 2 * l + 1
        return np.sin((l + 0.5) * theta) / np.sin(0.5 * theta)

    @staticmethod
    def cycle_index_term(l, k, theta):
        h = [0.0] * (k + 1)
        h[0] = 1.0 
        for m in range(1, k + 1):
            sum_val = 0.0
            for i in range(1, m + 1):
                p_i = MolienWeylIntegration.chi_l(l, i * theta)
                sum_val += p_i * h[m - i]
            h[m] = sum_val / m
        return h[k]

    @staticmethod
    def get_total_character(lin, nin, theta):
        pairs = list(zip(nin, lin))
        counts = Counter(pairs)
        total_chi = 1.0
        for (n, l), k in counts.items():
            if k == 1:
                total_chi *= MolienWeylIntegration.chi_l(l, theta)
            else:
                total_chi *= MolienWeylIntegration.cycle_index_term(l, k, theta)
        return total_chi

    @staticmethod
    def count_invariants(lin, nin, integration_points=200):
        thetas, dt = np.linspace(0, np.pi, integration_points, retstep=True)
        haar_measure = (1.0 / np.pi) * (1.0 - np.cos(thetas))
        chis = np.array([MolienWeylIntegration.get_total_character(lin, nin, t) for t in thetas])
        integral = np.trapz(chis * haar_measure, dx=dt)
        return int(round(integral))

# =========================================================
# SECTION 2: ENCAPSULATED WIGNER-RPI CLASS
# =========================================================

class WignerRPI:
    def __init__(self, lmax_by_rank=None, comm=None):
        """
        Well-encapsulated manager for PA-RPI basis and Wigner caches.
        
        Args:
            lmax_by_rank (list): e.g. [0, 4, 4, 3, 2, 1, 1, 1] 
                                index is rank, value is lmax.
        """
        self.comm = comm if comm else (MPI.COMM_WORLD if _HAS_MPI else None)
        self.rank = self.comm.Get_rank() if self.comm else 0
        self.size = self.comm.Get_size() if self.comm else 1
        
        self.lib_path = lib_path if os.path.exists(lib_path) else os.path.dirname(__file__)
        self.wigner_filename = os.path.join(self.lib_path, "wigner_cache.json")
        
        # State tracking
        self.wigner_cache = {}
        self.wigner_local_updates = {}
        self.current_wigner_jmax = -1
        self.rpi_memory_cache = defaultdict(dict)
        self.rpi_loaded_ranks = set()
        
        # Initialize Wigner 3j Cache incrementally
        self._initialize_wigner(lmax_by_rank)

    # --- WIGNER CACHE LOGIC ---

    def _initialize_wigner(self, lmax_req):
        # 1. Load JSON Master (Rank 0)
        if self.rank == 0 and os.path.exists(self.wigner_filename):
            try:
                with open(self.wigner_filename, "r") as f:
                    raw = json.load(f)
                    self.wigner_cache = {eval(k): v for k, v in raw.items()}
                    if self.wigner_cache:
                        self.current_wigner_jmax = max(max(k[0], k[1], k[2]) for k in self.wigner_cache)
            except: pass
        
        if self.comm:
            self.wigner_cache = self.comm.bcast(self.wigner_cache, root=0)
            self.current_wigner_jmax = self.comm.bcast(self.current_wigner_jmax, root=0)

        # 2. Rethink Incremental Strategy: Max intermediate J depends on (Rank//2)*Lmax
        if lmax_req:
            required_jmax = 0
            for r, l in enumerate(lmax_req):
                if r < 2: continue
                # For a scalar invariant of rank r, the coupling tree creates 
                # intermediate j-values up to (r // 2) * lmax.
                required_jmax = max(required_jmax, (r // 2) * l)
            
            if required_jmax > self.current_wigner_jmax:
                self._incremental_wigner_update(required_jmax)

    def _incremental_wigner_update(self, new_jmax):
        old_jmax = self.current_wigner_jmax
        if self.rank == 0:
            print(f"*** Incremental Wigner Update: J={old_jmax} -> {new_jmax}")
        
        delta = self._parallel_compute_wigner_delta(new_jmax, old_jmax)
        self.wigner_cache.update(delta)
        self.current_wigner_jmax = new_jmax
        
        if self.rank == 0:
            self._save_wigner_to_disk()

    def _parallel_compute_wigner_delta(self, target, current):
        new_tasks = []
        if self.rank == 0:
            for j1 in range(target + 1):
                for j2 in range(target + 1):
                    j1_old, j2_old = j1 <= current, j2 <= current
                    for j3 in range(abs(j1 - j2), min(j1 + j2, target) + 1):
                        if j1_old and j2_old and j3 <= current: continue
                        for m1 in range(-j1, j1 + 1):
                            for m2 in range(-j2, j2 + 1):
                                m3 = -(m1 + m2)
                                if abs(m3) <= j3: new_tasks.append((j1, j2, j3, m1, m2, m3))
            chunks = [new_tasks[i::self.size] for i in range(self.size)] if self.size > 1 else [new_tasks]
        else: chunks = None

        local_keys = self.comm.scatter(chunks, root=0) if self.comm else new_tasks
        local_res = {k: float(wigner_3j(*k)) for k in (tqdm(local_keys, desc="Delta Wigner") if self.rank == 0 else local_keys)}
        
        if self.comm:
            all_gathered = self.comm.allgather(local_res)
            merged = {}
            for d in all_gathered: merged.update(d)
            return merged
        return local_res

    # --- PUBLIC BASIS API ---

    def get_rpi_basis_vectors(self, mu_in, n_in, l_in):
        rv = len(l_in)
        # Safety check for intermediate J coverage
        shell_j_req = (rv // 2) * max(l_in)
        if shell_j_req > self.current_wigner_jmax:
            self._incremental_wigner_update(shell_j_req)

        self._load_rpi_cache(rv)
        key = (tuple(mu_in), tuple(n_in), tuple(l_in))
        
        if key in self.rpi_memory_cache[rv]:
            # Return cached results including m_configs
            return self.rpi_memory_cache[rv][key]

        # Calculation logic
        n_expected = MolienWeylIntegration.count_invariants(l_in, n_in)
        if n_expected == 0: return []
        
        combined_labs = _BasisGenerator.get_canonical_labels(n_in, l_in, mu_in)
        # m_configs created once per recalculation
        m_configs = _generate_m_states_iterative(sorted(l_in))
        ref_shell = sorted(list(zip(mu_in, n_in, l_in)))
        
        valid = []
        for lab in combined_labs:
            mu_l, n_l, l_l, l_int = _parse_label_full(lab, rv)
            p_maps = _get_constrained_permutations(ref_shell, list(zip(mu_l, n_l, l_l)))
            
            vec = []
            is_zero = True
            for mr in m_configs:
                val = sum(self._calculate_generalized_wigner(l_l, l_int, [mr[i] for i in p]) for p in p_maps)
                vec.append(val)
                if abs(val) > 1e-15: is_zero = False
            
            if not is_zero:
                v = np.array(vec, dtype=np.float64)
                valid.append({
                    'label': lab, 
                    'vector': (v / np.linalg.norm(v)).tolist(), # JSON compatible
                    'mus': mu_l, 'ns': n_l, 'ls': l_l
                })

        if not valid: return []
        A = np.column_stack([np.array(v['vector']) for v in valid])
        _, S, _ = scipy.linalg.svd(A, full_matrices=False)
        tol = max(1e-14, S[0] * 1e-12)
        
        res, basis_mat = [], []
        for v in valid:
            vec_arr = np.array(v['vector'])
            ortho = vec_arr.copy()
            for b in basis_mat: ortho -= np.dot(ortho, b) * b
            if np.linalg.norm(ortho) > tol:
                v['vector'] = (ortho / np.linalg.norm(ortho)).tolist()
                v['m_configs'] = m_configs # Optimized: attach m-configs to result
                basis_mat.append(np.array(v['vector']))
                res.append(v)
            if len(res) >= n_expected: break
            
        self.rpi_memory_cache[rv][key] = res
        return res

    # --- INTERNAL HELPERS ---

    def _calculate_generalized_wigner(self, l_leaves, l_inters, m_config):
        val, cL, cm, iQ = 1.0, list(l_leaves), list(m_config), list(l_inters)
        if abs(sum(cm)) > 1e-9: return 0.0
        while len(cL) > 1:
            nL, nm = [], []
            for i in range(0, len(cL), 2):
                if i+1 < len(cL):
                    l1, l2, m1, m2 = cL[i], cL[i+1], cm[i], cm[i+1]
                    Lr = 0 if len(cL) == 2 else iQ.pop(0)
                    if not (abs(l1-l2) <= Lr <= l1+l2) or abs(m1+m2) > Lr: return 0.0
                    w = self.wigner_cache.get((l1, l2, Lr, m1, m2, -(m1+m2)), 0.0)
                    if abs(w) < 1e-15: return 0.0
                    val *= w; nL.append(Lr); nm.append(m1+m2)
                else: nL.append(cL[i]); nm.append(cm[i])
            cL, cm = nL, nm
        return val

    def _load_rpi_cache(self, rank_val):
        if rank_val in self.rpi_loaded_ranks: return
        path = os.path.join(self.lib_path, f"wigner_rpi_rank{rank_val}.json")
        loaded = {}
        if self.rank == 0 and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    raw = json.load(f)
                    # Key is string shell key, value is list of funcs
                    loaded = {eval(k): v for k, v in raw.items()}
            except: pass
        if self.comm: loaded = self.comm.bcast(loaded, root=0)
        self.rpi_memory_cache[rank_val].update(loaded)
        self.rpi_loaded_ranks.add(rank_val)

    def _save_rpi_cache(self):
        if self.rank != 0: return
        for rv, cache in self.rpi_memory_cache.items():
            if not cache: continue
            path = os.path.join(self.lib_path, f"wigner_rpi_rank{rv}.json")
            try:
                # Store everything including m-configs
                save_data = {str(k): v for k, v in cache.items()}
                with open(path, "w") as f: json.dump(save_data, f)
            except: pass

    def _save_wigner_to_disk(self):
        if self.rank == 0 and self.wigner_cache:
            try:
                with open(self.wigner_filename, "w") as f:
                    json.dump({str(k): v for k, v in self.wigner_cache.items()}, f)
            except: pass

    def __del__(self):
        """Auto-save caches when manager goes out of scope."""
        self._save_wigner_to_disk()
        self._save_rpi_cache()

# =========================================================
# SECTION 3: SHARED HELPERS
# =========================================================

def _generate_m_states_iterative(l_leaves):
    res, stack, remaining, s = [], [(0, 0, ())], [0] * len(l_leaves), 0
    for i in range(len(l_leaves) - 1, -1, -1): remaining[i] = s; s += l_leaves[i]
    while stack:
        idx, cs, path = stack.pop()
        if idx == len(l_leaves) - 1:
            if abs(-cs) <= l_leaves[idx]: res.append(path + (-cs,))
            continue
        lim, cap = l_leaves[idx], remaining[idx]
        for m in range(max(-lim, -cap-cs), min(lim, cap-cs)+1): stack.append((idx+1, cs+m, path+(m,)))
    return res

def _parse_label_full(label, rank):
    parts = label.split('_')
    ints = [int(x) for x in parts[1].split(',')]
    l_inters = [int(x) for x in parts[-1].split('-')] if len(parts) > 2 and parts[-1] else []
    return ints[:rank], ints[rank:2*rank], ints[2*rank:], l_inters[:rank-2]

def _get_constrained_permutations(ref, curr):
    r_idx, c_idx = defaultdict(list), defaultdict(list)
    for i, p in enumerate(ref): r_idx[p].append(i)
    for i, p in enumerate(curr): c_idx[p].append(i)
    if r_idx.keys() != c_idx.keys(): return []
    all_maps = []
    perms_list = [[{c: r for c, r in zip(c_idx[p], perm)} for perm in permutations(r_idx[p])] for p in c_idx]
    for combined_perms in product(*perms_list):
        fm = {}
        for m in combined_perms: fm.update(m)
        all_maps.append([fm[k] for k in range(len(curr))])
    return all_maps

class _BasisGenerator:
    @staticmethod
    def get_canonical_labels(nin, lin, muin=None, L_R=0):
        rank = len(lin)
        muin = tuple([0] * rank) if muin is None else muin
        unique_perms = sorted(list(set(permutations(sorted(list(zip(muin, nin, lin)))))))
        labels = []
        for p in unique_perms:
            p_mu, p_n, p_l = zip(*p)
            for L_tuple in _BasisGenerator._gen_inters(p_l, L_R):
                integers = list(p_mu) + list(p_n) + list(p_l)
                labels.append(f"0_{','.join(map(str, integers))}_{'-'.join(map(str, L_tuple[:rank-2]))}")
        return sorted(list(set(labels)))
    @staticmethod
    def _gen_inters(l_curr, L_R):
        if len(l_curr) == 1:
            if l_curr[0] == L_R: yield []
            return
        pairs, pass_t, i = [], [], 0
        while i < len(l_curr):
            if i+1 < len(l_curr): pairs.append((l_curr[i], l_curr[i+1])); i += 2
            else: pass_t.append(l_curr[i]); i += 1
        ranges = [range(abs(p[0]-p[1]), p[0]+p[1]+1) for p in pairs]
        for inters in product(*ranges):
            for sub in _BasisGenerator._gen_inters(list(inters) + pass_t, L_R): yield list(inters) + sub
