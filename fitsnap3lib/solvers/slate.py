from fitsnap3lib.solvers.slate_validation import SlateValidation
import numpy as np
import json, os
from time import time
from adios2 import Stream

try:
    from slate_wrapper import slate_ridge_augmented_qr_cython, slate_ard_update_cython
except ImportError:
    try:
        import sys
        import os
        slate_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib', 'slate_solver')
        if slate_path not in sys.path:
            sys.path.insert(0, slate_path)
        from slate_wrapper import slate_ridge_augmented_qr_cython, slate_ard_update_cython
    except ImportError as e:
        print(f"Warning: Could not import SLATE ARD functions: {e}")
        slate_ard_update_cython = None

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

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
        a = pt.shared_arrays['a'].array  # X: design matrix (local portion)
        b = pt.shared_arrays['b'].array  # y: target vector (local portion)
        w = pt.shared_arrays['w'].array  # weights
        
        # Note: a, b, w remain unchanged - only aw, bw get modified
        aw = pt.shared_arrays['aw'].array
        bw = pt.shared_arrays['bw'].array

        # Get dimensions
        a_start_idx, a_end_idx = pt.fitsnap_dict["sub_a_indices"]
        local_slice = slice(a_start_idx, a_end_idx+1)
        local_w = w[local_slice].copy()
        m = aw.shape[0] * pt._number_of_nodes  # global number of samples
        n = aw.shape[1]  # number of features
        lld = aw.shape[0]  # local leading dimension

        # -------- TRAINING/TESTING SPLIT --------
        # Zero out test samples (they won't contribute to the fit)
        if 'Testing' in pt.fitsnap_dict and pt.fitsnap_dict['Testing'] is not None:
            testing_mask = pt.fitsnap_dict['Testing'][local_slice]
            for i in range(a_end_idx-a_start_idx+1):
                if testing_mask[i]:
                    #aw[a_start_idx+i,:] = 0.0
                    #bw[a_start_idx+i] = 0.0
                    #w[a_start_idx+i] = 0.0
                    local_w[i] = 0.0
                    
        # Apply weights to my local slice
        aw[local_slice] = local_w[:, np.newaxis] * a[local_slice]
        bw[local_slice] = local_w * b[local_slice]

        # Initialize ARD parameters
        eps = np.finfo(np.float64).eps
        
        # Compute initial alpha from variance of y (requires MPI reduction)
        # Count training samples (after zeroing out test samples above)
        if 'Testing' in pt.fitsnap_dict and pt.fitsnap_dict['Testing'] is not None:
            testing_mask = pt.fitsnap_dict['Testing'][local_slice]
            local_n_training = np.sum(~np.array(testing_mask, dtype=bool))
        else:
            local_n_training = a_end_idx - a_start_idx + 1
        
        # Compute variance (test samples are already zero'd in bw, so they won't affect mean/variance)
        local_bw = bw[local_slice]
        local_sum_bw = np.sum(local_bw)
        local_sum_bw2 = np.sum(local_bw**2)
        global_sum_bw = pt._comm.allreduce(local_sum_bw, op=MPI.SUM)
        global_sum_bw2 = pt._comm.allreduce(local_sum_bw2, op=MPI.SUM)
        global_n_training = pt._comm.allreduce(local_n_training, op=MPI.SUM)
        mean_bw = global_sum_bw / global_n_training
        var_bw = (global_sum_bw2 / global_n_training) - mean_bw**2
                
        # Compute adaptive hyperparameters (matching legacy ARD)
        ap = 1.0 / (var_bw + eps)  # inverse variance ("alpha prior")
        
        pt.debug_single_print(f"inverse variance in training data: {ap:.6f}, logscale for threshold_lambda: {np.log10(ap):.6f}")
        
        if self.directmethod:
            # Direct method: use specified hyperparameters
            self.alpha_1 = self.alphabig
            self.alpha_2 = self.alphabig
            self.lambda_1 = self.lambdasmall
            self.lambda_2 = self.lambdasmall
            if self.threshold_lambda_config > 0:
                self.threshold_lambda = self.threshold_lambda_config
            else:
                # Auto-compute threshold if not specified
                self.threshold_lambda = 10**(int(np.abs(np.log10(ap))) + self.logcut)
            pt.single_print(f"ARD directmethod: alpha_1={self.alpha_1:.2e}, lambda_1={self.lambda_1:.2e}, threshold_lambda={self.threshold_lambda:.2e}")
        else:
            # Adaptive method: scale by inverse variance
            self.alpha_1 = self.scap * ap
            self.alpha_2 = self.scap * ap
            self.lambda_1 = self.scai * ap
            self.lambda_2 = self.scai * ap
            if self.threshold_lambda_config > 0:
                self.threshold_lambda = self.threshold_lambda_config
            else:
                # Auto-compute threshold: 10^(int(abs(log10(ap))) + logcut)
                self.threshold_lambda = 10**(int(np.abs(np.log10(ap))) + self.logcut)
            pt.debug_single_print(f"automated threshold_lambda will be 10**({self.logcut:.6f} + {np.abs(np.log10(ap)):.3f})")
            pt.debug_single_print(f"ARD adaptive: scap={self.scap:.2e}, scai={self.scai:.2e}, alpha_1={self.alpha_1:.6e}, lambda_1={self.lambda_1:.6e}, threshold_lambda={self.threshold_lambda:.2e}")
        
        alpha_ = 1.0 / (var_bw + eps)
        lambda_ = np.ones(n, dtype=np.float64)
        coef_ = np.zeros(n, dtype=np.float64)
        lambda_mask = np.ones(n, dtype=bool)
        coef_old_ = None
        
        # Initialize gamma and lambda history tracking if validation is enabled
        if self.validation and pt._rank == 0:
        
            # Get rank information from PYACE basis if available
            if "PYACE" in self.config.sections:
                pyace_section = self.config.sections["PYACE"]
                outfile_section = self.config.sections["OUTFILE"]
                outfile_section.adios2_stream.write_attribute('nconfigs', pt.fitsnap_dict["nconfigs"])

                if hasattr(pyace_section, 'ctilde_basis'):
                    basis_ranks = []
                    if not pyace_section.bzeroflag:
                        basis_ranks.extend([0] * len(pyace_section.types))
                    for element_basis_rank1_functions in pyace_section.ctilde_basis.basis_rank1:
                        for basis_rank1_function in element_basis_rank1_functions:
                            basis_ranks.append(int(basis_rank1_function.rank))
                    for element_basis_functions in pyace_section.ctilde_basis.basis:
                        for basis_function in element_basis_functions:
                            basis_ranks.append(int(basis_function.rank))
                    outfile_section.adios2_stream.write_attribute('basis_ranks', basis_ranks)
                    outfile_section.adios2_stream.write_attribute('blist', pyace_section.blist)
        
        pt.debug_single_print(f"ARD: m {m} n {n} var(bw)={var_bw:.9f} alpha={alpha_:.2f}")
        
        np.set_printoptions(
            precision=4, suppress=False, floatmode='fixed', linewidth=np.inf,
            formatter={'float': '{:.3f}'.format}, threshold = 800, edgeitems=5
        )
        
        for key, value in sorted(os.environ.items()):
            if key.startswith("SLURM_"):
                print(f"*** {key} = {value}")
      
        # Iterative procedure of ARDRegression
        start_time_iteration = time()
        for iter_ in range(self.max_iter):
            # Get active indices
            active_indices = np.where(lambda_mask)[0]
            n_active = len(active_indices)
            
            if n_active == 0:
                if self.config.debug and pt._rank == 0:
                    pt.single_print(f"ARD: all features pruned at iteration {iter_}")
                break
                        
            # Pack active columns of a into first columns of aw (both are column-major for SLATE)
            aw[local_slice, :len(active_indices)] = local_w[:, np.newaxis] * a[local_slice, active_indices]

            lambda_active = lambda_[active_indices]
                        
            # Update sigma diagonal and coef using SLATE C++ functions
            # NOTE: Returns diagonal of sigma directly (not full matrix) to save memory
            sigma_diag, coef_active_ = slate_ard_update_cython(
                aw, bw, lambda_active, alpha_, m, n_active, lld, self.config.debug
            )
                        
            # Map active coefficients back to full coefficient vector
            coef_ = np.zeros(n, dtype=np.float64)
            coef_[active_indices] = coef_active_
            
            # Compute SSE (sum of squared errors) across all ranks
            # Use weighted data consistently: residual = bw - aw @ coef
            
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                local_pred = aw[local_slice, :n_active] @ coef_active_
                
            local_residual = bw[local_slice] - local_pred
            local_sse = np.sum(local_residual**2)  # Already weighted, don't apply weights again
            sse_ = pt._comm.allreduce(local_sse, op=MPI.SUM)
                        
            # Update alpha and lambda (on all ranks to stay synchronized)
            # Use sigma diagonal (returned directly from SLATE)
            gamma_active = 1.0 - lambda_active * sigma_diag
            
            # Map gamma back to full feature set
            gamma_ = np.zeros(n, dtype=np.float64)
            gamma_[active_indices] = gamma_active
                        
            lambda_[active_indices] = (gamma_active + 2.0 * self.lambda_1) / (coef_active_**2 + 2.0 * self.lambda_2)
                
            # Store gamma and lambda history if validation enabled
            if self.validation and pt._rank == 0:
                outfile_section.adios2_stream.begin_step()
                outfile_section.adios2_stream.write("gamma", gamma_, count=[n])
                outfile_section.adios2_stream.write("lambda", lambda_, count=[n])
                outfile_section.adios2_stream.end_step()
                        
            alpha_ = (global_n_training - gamma_active.sum() + 2.0 * self.alpha_1) / (sse_ + 2.0 * self.alpha_2)

            if coef_old_ is not None:
                coef_change = np.sum(np.abs(coef_old_ - coef_))
                coef_change_str = f"coef_change {coef_change:g} tol {self.tol:g}"
            else:
                coef_change_str = ""
                
            end_time_iteration = time()
            elapsed_iteration = end_time_iteration - start_time_iteration
            start_time_iteration = end_time_iteration

            # Prune features based on selected method
            if self.pruning_method.lower() == 'gamma':
                # Gamma-based pruning: keep features with gamma > threshold
                
                if iter_ >= 3:
                    lambda_mask = gamma_ > self.threshold_gamma
                
                pt.single_print(f"SLATE ARD #{iter_}: elapsed {elapsed_iteration:.2f} alpha {alpha_:.6f} sse {sse_:.6f} gamma_sum {gamma_active.sum():.6f} n_active {n_active} {coef_change_str}")
                
                if self.config.debug:
                    # Show gamma distribution
                    gamma_nonzero = gamma_[gamma_ > 0]
                    pt.single_print(f"  Gamma pruning: keeping {np.sum(lambda_mask)}/{n} features with gamma > {self.threshold_gamma:.3f}")
                    pt.single_print(f"  Gamma range: [{gamma_[gamma_ > 0].min():.4f}, {gamma_.max():.4f}]")
                    pt.single_print(f"  Gamma stats: mean={gamma_nonzero.mean():.3f}, median={np.median(gamma_nonzero):.3f}")
                    pt.single_print(f"  Gamma > 0.5: {np.sum(gamma_ > 0.5)}, > 0.3: {np.sum(gamma_ > 0.3)}, > 0.1: {np.sum(gamma_ > 0.1)}")
                    
            else:
                # Lambda-based pruning: keep features with lambda < threshold (original method)

                if iter_ >= 3:
                    lambda_mask = lambda_ < self.threshold_lambda
                
                pt.single_print(
                    f"SLATE ARD #{iter_}: elapsed {elapsed_iteration:.2f} alpha {alpha_:.6f} sse {sse_:.6f} "
                    f"gamma_sum {gamma_active.sum():.6f} n_active {n_active} {coef_change_str}\n    "
                    f"lambda range [{lambda_[lambda_ > 0].min():.4e}, {lambda_.max():.4e}], "
                    f"keeping {np.sum(lambda_mask)}/{n} features "
                    f"with lambda < {self.threshold_lambda:.1f}"
                )
            
            coef_[~lambda_mask] = 0
            
            # Check for stopping criteria (convergence or SLURM timeout)
            if coef_old_ is not None:
                #coef_change = np.sum(np.abs(coef_old_ - coef_))
                if coef_change < self.tol:
                    active_features = np.sum(lambda_mask)
                    pt.single_print(f"ARD converged after {iter_} iterations, "
                                      f"{active_features}/{n} features active")
                    break
            
            coef_old_ = np.copy(coef_)
        
        # Store final solution
        if "PYACE" in self.config.sections:
            pyace_section = self.config.sections["PYACE"]
            if pyace_section.bzeroflag:
                pyace_section.lambda_mask = lambda_mask
            else:
                pyace_section.lambda_mask = lambda_mask[pyace_section.numtypes:]

        self.fit = coef_
        
        # Save gamma and lambda history using adios2 if validation enabled
        if self.validation and self.pt._rank == 0:
            self.config.sections["OUTFILE"].adios2_stream.close()
        
        if self.config.debug and pt._rank == 0:
            active_features = np.sum(lambda_mask)
            pt.single_print(f"\nARD final: {active_features}/{n} features active, "
                          f"alpha={alpha_:.2e}, lambda range=[{np.min(lambda_):.2e}, {np.max(lambda_):.2e}]")
            if self.pruning_method.lower() == 'gamma':
                gamma_active_final = gamma_[gamma_ > 0]
                pt.single_print(f"Gamma range: [{gamma_active_final.min():.4f}, {gamma_active_final.max():.4f}], "
                              f"mean={gamma_active_final.mean():.4f}")

