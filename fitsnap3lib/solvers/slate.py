from fitsnap3lib.solvers.slate_validation import SlateValidation
import sys, os, json, time, signal
import numpy as np
from mpi4py import MPI
from sklearn.mixture import GaussianMixture

try:
    from slate_wrapper import slate_ard_update_cython
except ImportError:
    try:
        slate_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib', 'slate_solver')
        if slate_path not in sys.path:
            sys.path.insert(0, slate_path)
        from slate_wrapper import slate_ridge_augmented_qr_cython, slate_ard_update_cython
    except ImportError as e:
        print(f"Warning: Could not import SLATE ARD functions: {e}")
        slate_ard_update_cython = None
    
# -------------------------------- STOPPING CRITERIA --------------------------------

SIGUSR1_signal_received = False

def handle_SIGUSR1(signum, frame):
  global SIGUSR1_signal_received
  SIGUSR1_signal_received = True

signal.signal(signal.SIGUSR1, handle_SIGUSR1)

def get_slurm_time_left():
  try:
    end = int(os.environ.get("SLURM_JOB_END_TIME", 0))
    now = int(time.time())
    if end and now: return end - now
  except Exception:
    pass
  return float("inf")

def mixed_relative_change(coef_old, coef_new, rtol=1e-2, atol=1e-4):
  if coef_old is None: return False, False, None, None
  abs_change = np.linalg.norm(coef_new - coef_old)
  rel_change = abs_change / (np.linalg.norm(coef_old) + atol)
  if rel_change < rtol: return True, False, rel_change, abs_change
  elif abs_change < atol: return False, True, rel_change, abs_change
  else: return False, False, rel_change, abs_change


# --------------------------------------------------------------------------------------------

class SLATE(SlateValidation):

    # --------------------------------------------------------------------------------------------

    def perform_fit_ard(self):
        """
        Perform ARD (Automatic Relevance Determination) regression using SLATE.
        
        Implements the sklearn ARDRegression algorithm for distributed matrices:
        
        Iteratively updates:
        - sigma_diag = diag(inv(alpha * X.T @ X + diag(lambda)))  [diagonal of covariance]
        - coef = alpha * sigma @ X.T @ y                          [coefficient estimates]
        - lambda = (gamma + 2*lambda_1) / (coef^2 + 2*lambda_2)   [feature precisions]
        - alpha = (m - gamma.sum() + 2*alpha_1) / (SSE + 2*alpha_2) [noise precision]
        
        where gamma = 1 - lambda * sigma_diag
        
        NOTE: SLATE returns only the diagonal of sigma to save memory (n vs n^2 doubles).
        
        Assumes m >> n (many samples, fewer features) so no Woodbury formula needed.
        """
        
        pt = self.pt

        # a, b, w are the pristine node-shared buffers; ARD never mutates them. The weighted
        # active design the solve needs is now built inside slate.cpp as a SLATE-owned 2D matrix,
        # so this path no longer touches the shared aw/bw scratch (still used by the ridge path).
        a = pt.shared_arrays['a'].array  # X: design matrix (node-shared, column-major)
        b = pt.shared_arrays['b'].array  # y: target vector (node-shared)
        w = pt.shared_arrays['w'].array  # weights (node-shared)

        # Get dimensions
        a_start_idx, a_end_idx = pt.fitsnap_dict["sub_a_indices"]
        local_slice = slice(a_start_idx, a_end_idx+1)
        local_w = w[local_slice].copy()
        
        # Hybrid architecture: data is distributed across nodes AND across ranks within a node.
        # m   = total global design rows; lld = node-shared buffer leading dim (rows on this node).
        # row_offset (a_start_idx) / m_local = THIS rank's contiguous slice of the node-shared
        # buffer, so the C++ solver gathers each rank's own rows into the 2D-cyclic working matrix
        # instead of piling every tile onto rank 0.
        m = a.shape[0] * pt._number_of_nodes    # total global design rows
        n = a.shape[1]                          # number of features
        lld = a.shape[0]                        # node-shared buffer leading dim (rows on this node)
        m_local = a_end_idx - a_start_idx + 1   # design rows owned by this rank within the node buffer
        
        assert m>n, f"SLATE ARD: m ({m}) < n ({n}), m > n needed for regression."

        # -------- TRAINING/TESTING SPLIT --------
        if 'Testing' in pt.fitsnap_dict and pt.fitsnap_dict['Testing'] is not None:
            testing_mask = pt.fitsnap_dict['Testing'][local_slice]
            local_m_training = np.sum(~np.array(testing_mask, dtype=bool))
            for i in range(a_end_idx-a_start_idx+1):
                if testing_mask[i]: local_w[i] = 0.0
        else:
            local_m_training = a_end_idx - a_start_idx + 1
            
        # -------- IN-PLACE WEIGHTING/CENTERING/SCALING --------
        eps = np.finfo(np.float64).eps
        
        # Weighted-b stats for the adaptive noise prior (local temporary; no shared bw write).
        bw_local = local_w * b[local_slice]

        # Compute mean/std of weighted b across MPI
        local_sum_bw, local_sum_bw2 = np.sum(bw_local), np.sum(bw_local**2)
        local_values = np.array([local_sum_bw, local_sum_bw2, local_m_training])
        
        # Allreduce to get global stats
        pt._comm.Allreduce(MPI.IN_PLACE, local_values, op=MPI.SUM)
        global_sum_bw, global_sum_bw2, global_m_training = local_values.tolist()
        
        mean_bw = global_sum_bw / m
        var_bw = global_sum_bw2 / m - mean_bw**2

        # Compute adaptive hyperparameters
        ap = 1.0 / (var_bw + eps) 

        self.pt.single_print(
          f"----------------------------------------------------------------\n"
          f"  SLATE ARD SOLVER                                              \n"
        )

        pt.debug_single_print(f"inverse variance in training data: {ap:.6f}, logscale for threshold_lambda: {np.log10(ap):.6f}")
        
        if self.directmethod:
            self.alpha_1 = self.alphabig
            self.alpha_2 = self.alphasmall
            self.lambda_1 = self.lambdabig
            self.lambda_2 = self.lambdasmall
            if self.threshold_lambda_config > 0: self.threshold_lambda = self.threshold_lambda_config
            else: self.threshold_lambda = 10**(int(np.abs(np.log10(ap))) + self.logcut)
            pt.single_print(f"    directmethod: alpha_1={self.alpha_1:.2e}, lambda_1={self.lambda_1:.2e}, threshold_lambda={self.threshold_lambda:.2e}")
            pt.single_print(f"    m {m} n {n} ap {ap:.2g} alpha_1 {self.alpha_1:.2g} alpha_2 {self.alpha_2:.2g} lambda_1 {self.lambda_1:.2g} lambda_2 {self.lambda_2:.2g}")
        else:
            self.alpha_1 = self.scap * ap
            self.alpha_2 = self.scap * ap
            self.lambda_1 = self.scai * ap
            self.lambda_2 = self.scai * ap
            if self.threshold_lambda_config > 0: self.threshold_lambda = self.threshold_lambda_config
            else: self.threshold_lambda = 10**(int(np.abs(np.log10(ap))) + self.logcut)
            pt.debug_single_print(f"automated threshold_lambda will be 10**({self.logcut:.6f} + {np.abs(np.log10(ap)):.3f})={self.threshold_lambda:.2g}")
            pt.single_print(
              f"    m {m} n {n} scap {self.scap:.2g} scai {self.scai:.2g} ap {ap:.2g} \n"
              f"    alpha_1 {self.alpha_1:.2g} alpha_2 {self.alpha_2:.2g} lambda_1 {self.lambda_1:.2g} lambda_2 {self.lambda_2:.2g}\n")

        self.pt.single_print(
          #f"----------------------------------------------------------------\n"
          f"    #   TIME    ALPHA   SSE  COND_NUMBER  GAMMA_SUM  N_ACTIVE  COEF_CHANGE \n"
          f"                                                               (REL) (ABS) \n"
        )

        alpha_ = 1.0
        lambda_ = np.ones(n, dtype=np.float64)
        coef_ = np.zeros(n, dtype=np.float64)
        lambda_mask = np.ones(n, dtype=bool)
        coef_old_ = None

        # Solver-method switch: default to the Cholesky normal-equations path (method=1: distributed
        # herk + 2D Cholesky, ~2x fewer flops, no rank-0 serialization). When the PREVIOUS
        # iteration's condition number exceeds this threshold, fall back to the augmented-QR path
        # (method=0), which keeps kappa(X) instead of squaring it via the normal equations.
        cond_switch_threshold = 1e7
        last_cond = None

        iteration = 1
        start_time_iteration = time.time()
        
        while True:
            # Get active indices
            active_indices = np.where(lambda_mask)[0]
            n_active = len(active_indices)
            
            if n_active == 0:
                pt.debug_single_print(f"ARD: all features pruned at iteration {iteration}")
                break
                        
            # active features this iteration; lambda for them.
            lambda_active = lambda_[active_indices].copy()
            active_idx_i64 = np.ascontiguousarray(active_indices, dtype=np.int64)

            # --- HYBRID SOLVER CALL ---
            # No packing / node barrier: slate.cpp builds the weighted active design (a SLATE-owned
            # 2D-cyclic matrix) directly from the pristine node-shared 'a', reading each rank's own
            # rows via row_offset. 'a' is read-only, so nothing here writes shared memory first.

            # Allocate buffers on all ranks
            sigma_diag = np.zeros(n_active, dtype=np.float64)
            coef_active_ = np.zeros(n_active, dtype=np.float64)
            cond_box = np.zeros(1, dtype=np.float64)
            sse_box = np.zeros(1, dtype=np.float64)

            # Determine Communicator (Single vs Multi-Node)
            #if pt._number_of_nodes > 1: comm = pt._head_group_comm
            #else: comm = pt.MPI.COMM_SELF
            comm = pt._comm

            # Call C++ wrapper: raw a, b (node-shared), per-rank effective weights local_w (testing
            # rows already zeroed), and the active column indices. SSE and cond are computed in C++
            # (Cholesky normal-equations by default, augmented-QR fallback -- see method below).
            # Pick the path from the previous iteration's conditioning.
            method = 0 if (last_cond is not None and last_cond > cond_switch_threshold) else 1

            s_d, c_a, sse_val, cn = slate_ard_update_cython(
                a, b, local_w, active_idx_i64, lambda_active, alpha_,
                m, n_active, lld, a_start_idx, m_local, method, comm, self.config.debug
            )
                
            # Copy results to buffers
            sigma_diag[:] = s_d
            coef_active_[:] = c_a
            cond_box[0] = cn
            sse_box[0] = sse_val
            
            cond_number = cond_box[0]
            sse_ = sse_box[0]
            last_cond = cond_number
            
            # --- UPDATE STATE (All Ranks) ---
            
            # Map active coefficients back to full coefficient vector
            coef_ = np.zeros(n, dtype=np.float64)
            coef_[active_indices] = coef_active_
            
            # SSE is now computed in C++ (slate_ard_update) from the augmented-QR residual,
            # so no second pass over aw is needed here. sse_ was set above from sse_box.
            
            # Update Gamma/Lambda
            gamma_active = 1.0 - lambda_active * sigma_diag
            gamma_ = np.zeros(n, dtype=np.float64)
            gamma_[active_indices] = gamma_active
            
            lambda_[active_indices] = (gamma_active + 2.0 * self.lambda_1) / (coef_active_**2 + 2.0 * self.lambda_2)
            
            # Store history for validation
            if self.validation and pt._rank == 0:
                stream = self.config.sections["OUTFILE"].adios2_stream
                stream.begin_step()
                stream.write("gamma", gamma_, count=[n])
                stream.write("lambda", lambda_, count=[n])
                stream.end_step()
            
            # Update Alpha
            alpha_ = (global_m_training - gamma_active.sum() + 2.0 * self.alpha_1) / (sse_ + 2.0 * self.alpha_2)

            # Prune features
            
            if iteration >= 5 and iteration % 5 == 0:
                log10_lambda = np.log10(lambda_ + 1e-9)
                min, max = np.min(log10_lambda), np.max(log10_lambda)
                #self.threshold_lambda = 10.0**(min + .99*(max-min))
                lambda_mask = lambda_ < self.threshold_lambda
                
            coef_[~lambda_mask] = 0

            # --- LOGGING & CONVERGENCE ---

            end_time_iteration = time.time()
            elapsed_iteration = end_time_iteration - start_time_iteration
            start_time_iteration = end_time_iteration
            
            slurm_time_left = get_slurm_time_left()
            slurm_time_left_str = "" if np.isinf(slurm_time_left) else f" slurm_time_left {slurm_time_left/60:.1f}m"

            coef_rel_converged, coef_abs_converged, coef_rel_change, coef_abs_change = mixed_relative_change(coef_old_, coef_)
            coef_change_str = " " if coef_old_ is None else f" {coef_rel_change:5.2g} {coef_abs_change:5.2g}"
            coef_old_ = np.copy(coef_)

            pt.single_print(f"    {iteration:<3d} {elapsed_iteration/60:.1f}m  {alpha_:>6.3g}  {sse_:>6.3g}  {cond_number:>8.3g}  {gamma_active.sum():>5.2f}  {n_active:6d}{coef_change_str}{slurm_time_left_str}")

            iteration += 1
            
            if iteration > self.max_iter:
                pt.single_print(f"SLATE ARD: stopping... reached max_iter {self.max_iter}")
                break

            if cond_number > 1e15:
                pt.single_print(f"SLATE ARD: stopping... cond_number {cond_number:>8.3g} > 1e15")
                break

            if coef_rel_converged:
                pt.single_print(f"SLATE ARD: stopping... coef_rel_change {coef_rel_change} < {self.rtol}")
                break

            if coef_abs_converged:
                pt.single_print(f"SLATE ARD: stopping... coef_abs_change {coef_abs_change} < {self.atol}")
                break

            if SIGUSR1_signal_received:
                pt.single_print(f"SLATE ARD: stopping... received SIGUSR1 signal")
                break

            if slurm_time_left < 2 * elapsed_iteration:
                pt.single_print(f"SLATE ARD: stopping... slurm_time_left {slurm_time_left/60:.1f} minutes < 2 * elapsed_iteration {elapsed_iteration/60:.1f} minutes")
                break
        
        # Store final solution
        if "PYACE" in self.config.sections:
            pyace_section = self.config.sections["PYACE"]
            if pyace_section.bzeroflag: pyace_section.lambda_mask = lambda_mask
            else: pyace_section.lambda_mask = lambda_mask[pyace_section.numtypes:]

        self.fit = coef_
        
        if self.config.debug and pt._rank == 0:
            active_features = np.sum(lambda_mask)
            pt.single_print(f"\nARD final: {active_features}/{n} features active, "
                          f"alpha={alpha_:.2e}, lambda range=[{np.min(lambda_):.2e}, {np.max(lambda_):.2e}]")
