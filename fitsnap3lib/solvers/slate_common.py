from fitsnap3lib.solvers.solver import Solver
from fitsnap3lib.parallel_tools import DistributedList
import numpy as np

# --------------------------------------------------------------------------------------------

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# Import the SLATE module
SLATE_AVAILABLE = False
slate_ridge_augmented_qr_cython = None
slate_ard_update_cython = None

try:
    # Primary import method (after pip install -e .)
    from slate_wrapper import set_openmp_threads, slate_ridge_augmented_qr_cython
    SLATE_AVAILABLE = True
except ImportError as e:
    # Fallback: try direct path import for in-place builds
    try:
        import sys
        import os
        slate_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib', 'slate_solver')
        if slate_path not in sys.path:
            sys.path.insert(0, slate_path)
        from slate_wrapper import slate_ridge_augmented_qr_cython, slate_ard_update_cython
        SLATE_AVAILABLE = True
    except ImportError:
        print(f"Warning: SLATE module import failed: {e}")
        print("To install: cd fitsnap3lib/lib/slate_solver && pip install -e .")
        slate_ridge_augmented_qr_cython = None
        slate_ard_update_cython = None
        SLATE_AVAILABLE = False
        
# --------------------------------------------------------------------------------------------

class SlateCommon(Solver):
    """
    Multi-node Ridge regression solver using SLATE (Software for Linear Algebra Targeting Exascale).
    
    This solver leverages SLATE's distributed matrix operations to solve ridge regression
    problems across multiple nodes efficiently.
    
    Solves: (A^T A + alpha * I) x = A^T b
    """
    
    def __init__(self, name, pt, config):
        super().__init__(name, pt, config, linear=True)
        
        # Check that SLATE is available
        if not SLATE_AVAILABLE:
            error_msg = f"[Rank {self.pt._rank}, Node {self.pt._node_index}] SLATE module not available. Please compile it first."
            pt.single_print(error_msg)
            raise RuntimeError(error_msg)
            
        self.validation = self.config.sections["OUTFILE"].validation
        
        # Get parameters from SLATE section
        if 'SLATE' in self.config.sections:
        
            slate_config = self.config.sections['SLATE']
            
            # Method selection: RIDGE or ARD
            self.method = slate_config.method.upper()
        
            if self.method == 'RIDGE':
                self.alpha = slate_config.alpha
            elif self.method == 'ARD':
                self.max_iter = slate_config.max_iter
                self.rtol = slate_config.rtol
                self.atol = slate_config.atol
                
                # Store config parameters for adaptive hyperparameter computation
                self.directmethod = slate_config.directmethod
                self.scap = slate_config.scap
                self.scai = slate_config.scai
                self.logcut = slate_config.logcut
                self.alphabig = slate_config.alphabig
                self.alphasmall = slate_config.alphasmall
                self.lambdabig = slate_config.lambdabig
                self.lambdasmall = slate_config.lambdasmall
                
                # Pruning method threshold
                self.threshold_lambda_config = slate_config.threshold_lambda
                
                # These will be set in perform_fit_ard based on data variance
                self.alpha_1 = None
                self.alpha_2 = None
                self.lambda_1 = None
                self.lambda_2 = None
                self.threshold_lambda = None
        
    # --------------------------------------------------------------------------------------------

    def perform_fit(self):
    
        fs_dict = self.pt.fitsnap_dict

        if self.validation and self.pt._rank == 0:

            stream = self.config.sections["OUTFILE"].adios2_stream
            stream.write_attribute('nconfigs', fs_dict["nconfigs"])
            stream.write_attribute("sorted_group_names", fs_dict["sorted_group_names"])

            # Get rank information from PYACE basis if available
            if "PYACE" in self.config.sections:
                pyace_section = self.config.sections["PYACE"]
                if hasattr(pyace_section, 'ctilde_basis'):
                    ctilde_basis = pyace_section.ctilde_basis
                    stream.write_attribute('basis_ranks', ctilde_basis.basis_ranks)
                    stream.write_attribute('blist', ctilde_basis.blist)
        
            # Get rank information from PYACE basis if available
            if "UF3" in self.config.sections:
                uf3_section = self.config.sections["UF3"]
                stream.write_attribute('basis_ranks', uf3_section.basis_ranks)
                stream.write_attribute('blist', uf3_section.blist)
        
        if self.method.upper() == "ARD": self.perform_fit_ard()
        else: self.perform_fit_ridge()

    # --------------------------------------------------------------------------------------------

    def perform_fit_ridge(self):
        
        pt = self.pt
        a = pt.shared_arrays['a'].array
        b = pt.shared_arrays['b'].array
        w = pt.shared_arrays['w'].array
        
        # Note: a, b, w remain unchanged - only aw, bw get modified by SLATE
        aw = pt.shared_arrays['aw'].array
        bw = pt.shared_arrays['bw'].array
        
        # Debug output - print all in one statement to avoid tangled output
        # *** DO NOT REMOVE !!! ***
        if self.config.debug:
            np.set_printoptions(precision=4, suppress=True, floatmode='fixed', linewidth=np.inf)
            np.set_printoptions(formatter={'float': '{:.4f}'.format})
            pt.sub_print(f"*** ------------------------\n"
                         f"pt.fitsnap_dict['Testing']\n{pt.fitsnap_dict['Testing']}\n"
                         #f"a\n{a}\n"
                         #f"b {b}\n"
                         f"--------------------------------\n")
        
        pt.sub_barrier()
        
        # -------- LOCAL SLICE OF SHARED ARRAY AND REGULARIZATION ROWS --------

        a_start_idx, a_end_idx = pt.fitsnap_dict["sub_a_indices"]
        aw_start_idx, aw_end_idx = pt.fitsnap_dict["sub_aw_indices"]
        reg_row_idx = pt.fitsnap_dict["reg_row_idx"]
        reg_col_idx = pt.fitsnap_dict["reg_col_idx"]
        reg_num_rows = pt.fitsnap_dict["reg_num_rows"]
        #pt.all_print(f"pt.fitsnap_dict {pt.fitsnap_dict}")
        if self.config.debug:
            pt.all_print(f"*** aw_start_idx {aw_start_idx} aw_end_idx {aw_end_idx} reg_row_idx {reg_row_idx} reg_col_idx {reg_col_idx} reg_num_rows {reg_num_rows}")
        
        # -------- WEIGHTS --------
  
        # Apply weights to my local slice
        local_slice = slice(a_start_idx, a_end_idx+1)
        w_local_slice = slice(aw_start_idx, (aw_end_idx-reg_num_rows+1))
        aw[w_local_slice] = w[local_slice, np.newaxis] * a[local_slice]
        bw[w_local_slice] = w[local_slice] * b[local_slice]

        # -------- TRAINING/TESTING SPLIT --------
        
        if 'Testing' in pt.fitsnap_dict and pt.fitsnap_dict['Testing'] is not None:
            testing_mask = pt.fitsnap_dict['Testing'][local_slice]
            for i in range(a_end_idx-a_start_idx+1):
                if testing_mask[i]:
                    if self.config.debug:
                        pt.all_print(f"*** removing i {i} aw_start_idx+i {aw_start_idx+i}")
                    aw[aw_start_idx+i,:] = 0.0
                    bw[aw_start_idx+i] = 0.0

        # -------- REGULARIZATION ROWS --------

        sqrt_alpha = np.sqrt(self.alpha)
        n = a.shape[1]
        
        # DEBUG: Print regularization info
        if self.config.debug and pt._rank == 0:
            pt.single_print(f"\n=== DEBUG: Regularization ===")
            pt.single_print(f"alpha = {self.alpha}")
            pt.single_print(f"sqrt_alpha = {sqrt_alpha}")
            pt.single_print(f"Number of features (n) = {n}")
            pt.single_print(f"Number of regularization rows = {reg_num_rows}")
            pt.single_print(f"==============================\n")
    
        for i in range(reg_num_rows):
            g = reg_col_idx + i  # global index into the [ridge(n rows) | curvature(ncoeff_2b rows)] reg block
            if g < n:  # ridge rows: diagonal identity scaled by sqrt_alpha
                aw[reg_row_idx+i, g] = sqrt_alpha
            bw[reg_row_idx+i] = 0.0

        # Curvature (second-difference D2) regularization for UF3 2-body coefficients.
        # The augmented matrix layout is: n ridge rows, then ncoeff_2b curvature rows.
        # Column layout in A: [1-body (offset_2b cols)] [2-body (ncoeff_2b cols)] [3-body].
        # offset_2b = numtypes = basis_ranks.count(0); independent of bzeroflag.
        # D2 row c: off-diag neighbours +=1, centre -=2 (halved to -1 at boundaries).
        is_slate_ridge_uf3 = pt.fitsnap_dict.get("is_slate_ridge_uf3", False)
        if is_slate_ridge_uf3:
            sqrt_alpha_curvature = np.sqrt(self.config.sections['SLATE'].alpha_curvature)
            ncoeff_2b = pt.fitsnap_dict.get("ncoeff_2b", 0)
            offset_2b = pt.fitsnap_dict.get("offset_2b", 0)
            n_edges_3b = pt.fitsnap_dict.get("n_edges_3b", 0)
            edges_3b = pt.fitsnap_dict.get("edges_3b", None)
            g_3b_start = n + ncoeff_2b  # 3b curvature rows follow 2b curvature rows
            for i in range(reg_num_rows):
                g = reg_col_idx + i
                if n <= g < n + ncoeff_2b:
                    # 2b second-difference curvature: stencil [1, -2, 1] ([-1,1] at boundaries)
                    c = g - n
                    diag_val = -1.0 if (c == 0 or c == ncoeff_2b - 1) else -2.0
                    if c > 0:
                        aw[reg_row_idx+i, offset_2b + c-1] = sqrt_alpha_curvature
                    aw[reg_row_idx+i, offset_2b + c] = sqrt_alpha_curvature * diag_val
                    if c < ncoeff_2b - 1:
                        aw[reg_row_idx+i, offset_2b + c+1] = sqrt_alpha_curvature
                elif edges_3b is not None and g_3b_start <= g < g_3b_start + n_edges_3b:
                    # 3b graph-Laplacian: one incidence row per edge (+1 at u, -1 at v)
                    # D^T D = L3 (graph Laplacian), so augmented LS gives the correct penalty
                    e = g - g_3b_start
                    col_u = int(edges_3b[e, 0])
                    col_v = int(edges_3b[e, 1])
                    aw[reg_row_idx + i, col_u] =  sqrt_alpha_curvature
                    aw[reg_row_idx + i, col_v] = -sqrt_alpha_curvature

        # -------- SLATE AUGMENTED QR --------
        pt.sub_barrier() # make sure all sub ranks done filling local tiles
        m = aw.shape[0] * self.pt._number_of_nodes # global matrix total rows
        lld = aw.shape[0]  # local leading dimension column-major shared array
                    
        np.set_printoptions(precision=3, suppress=True, floatmode='fixed', linewidth=np.inf)
        if False and self.config.debug:
            pt.sub_print(f"*** SENDING TO SLATE ------------------------\n"
                         f"aw\n{aw}\n"
                         f"bw {bw}\n"
                         f"--------------------------------\n")
                     
        # Determine debug flag from EXTRAS section
        debug_flag = 1 if self.config.debug else 0

        # This rank is head of its node
        if pt._sub_rank == 0:
            # Select the correct communicator (Single Node vs Multi-Node)
            if pt._number_of_nodes > 1: comm = pt._head_group_comm
            else: comm = pt.MPI.COMM_SELF
            # Thread count = Total Ranks on Node (since other ranks are sleeping)
            set_openmp_threads(pt._sub_size, self.config.debug)
            slate_ridge_augmented_qr_cython(aw, bw, m, lld, comm, debug_flag)
            # Broadcast solution from Node 0 to all nodes via head ranks
            pt._head_group_comm.Bcast(bw[:n], root=0)
        
        # sub ranks 1,...,N on each node wait in non-blocking "polite" barrier
        # while sub rank 0 on each node solves in SLATE using openmp intra-node
        # and mpi across nodes.
        pt.polite_barrier(pt._comm)
        self.fit = bw[:n]
                
        # *** DO NOT REMOVE !!! ***
        if self.config.debug:
            pt.all_print(f"*** self.fit ------------------------\n"
                f"{self.fit}\n-------------------------------------------------\n")
            

        
    # --------------------------------------------------------------------------------------------

    def error_analysis(self):
        """
        Scalable error analysis using mpi4py collective operations.
        
        Algorithm:
        1. Each rank computes local group statistics (counts, sums, weighted sums)
        2. Gather local stats to rank 0 using comm.gather()
        3. Rank 0 merges all stats and broadcasts merged data to all ranks
        4. Two-pass R² calculation:
           - Pass 1: Compute global means from merged stats (already done)
           - Pass 2: Each rank computes local SS_tot contributions
           - Gather SS_tot to rank 0 and reduce
        5. Rank 0 computes final metrics (MAE, RMSE, R²) and formats as DataFrame
        
        Uses standard mpi4py collectives (gather, bcast) instead of custom tree reductions.
        """

        pt = self.pt
        if self.fit is None: return
        # a, b, w are unchanged from original data (only aw, bw were modified by SLATE)
        fs_dict = pt.fitsnap_dict
        # Create DataFrame like the legacy solver does
        from pandas import DataFrame
        # Use only local slice (excluding regularization rows) for error analysis
        start_idx, end_idx = pt.fitsnap_dict["sub_a_indices"]
        local_slice = slice(start_idx, end_idx+1)
        local_a = pt.shared_arrays['a'].array[local_slice]
        local_b = pt.shared_arrays['b'].array[local_slice]
        local_w = pt.shared_arrays['w'].array[local_slice]
        df_local = DataFrame(local_a)
        df_local['truths'] = local_b.tolist()
        # Check for numerical issues before prediction
        if np.any(np.isnan(local_a)) or np.any(np.isinf(local_a)):
            self.pt.single_print(f"WARNING: NaN or Inf found in descriptor matrix (local_a)")
            self.pt.single_print(f"  NaN count: {np.sum(np.isnan(local_a))}, Inf count: {np.sum(np.isinf(local_a))}")
        if np.any(np.isnan(self.fit)) or np.any(np.isinf(self.fit)):
            self.pt.single_print(f"WARNING: NaN or Inf found in fit coefficients")
            self.pt.single_print(f"  NaN count: {np.sum(np.isnan(self.fit))}, Inf count: {np.sum(np.isinf(self.fit))}")
        
        # Compute predictions with NaN/Inf handling
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            preds = local_a @ self.fit
        
        # Replace NaN/Inf with 0 for error analysis (or could skip these rows)
        if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
            num_bad = np.sum(np.isnan(preds) | np.isinf(preds))
            self.pt.single_print(f"WARNING: {num_bad} NaN/Inf predictions found, replacing with corresponding truth values for error analysis")
            bad_mask = np.isnan(preds) | np.isinf(preds)
            preds[bad_mask] = local_b[bad_mask]
        
        df_local['preds'] = preds.tolist()
        df_local['weights'] = local_w.tolist()
        
        # Gather predictions and truths to rank 0 for scatterplots
        # ALL RANKS participate in gather to avoid deadlock
        if self.validation:
            all_preds = pt._comm.gather(preds, root=0) #.astype(np.float32)
            all_truths = pt._comm.gather(local_b, root=0)
            all_groups = pt._comm.gather(list(fs_dict['Groups'][local_slice]), root=0)
            all_testing = pt._comm.gather(list(fs_dict['Testing'][local_slice] if 'Testing' in fs_dict else [False]*len(preds)), root=0)
            all_row_types = pt._comm.gather(list(fs_dict['Row_Type'][local_slice] if 'Row_Type' in fs_dict else ['Energy']*len(preds)), root=0)
            
            # Only rank 0 writes to adios2
            if pt._rank == 0:
                outfile_section = self.config.sections["OUTFILE"]
                if hasattr(outfile_section, 'adios2_stream') and outfile_section.adios2_stream is not None:
                    stream = outfile_section.adios2_stream
                    
                    # Flatten gathered data
                    preds_flat = np.concatenate(all_preds)
                    truths_flat = np.concatenate(all_truths)
                    groups_flat = [g for sublist in all_groups for g in sublist]
                    testing_flat = [t for sublist in all_testing for t in sublist]
                    row_types_flat = [r for sublist in all_row_types for r in sublist]
                    
                    # Get sorted group names and unique row types
                    sorted_group_names = fs_dict["sorted_group_names"]
                    unique_row_types = sorted(set(row_types_flat))
                    
                    # Write separate variables for each (row_type, group_idx, testing) combination
                    # Each variable has shape (n_points, 2) with columns [predictions, truths]
                    stream.begin_step()
                    for row_type in unique_row_types:
                        row_type_lower = row_type.lower()
                        for group_idx, group_name in enumerate(sorted_group_names):
                            for testing_flag in [False, True]:
                                # Filter data for this combination
                                mask = [(g == group_name and t == testing_flag and r == row_type) 
                                        for g, t, r in zip(groups_flat, testing_flat, row_types_flat)]
                                mask = np.array(mask)
                                
                                if np.any(mask):
                                    # Get predictions and truths for this subset
                                    subset_preds = preds_flat[mask]
                                    subset_truths = truths_flat[mask]
                                    # Stack into (n_points, 2) array: [predictions, truths]
                                    data_array = np.column_stack([subset_truths, subset_preds]).astype(np.float32)
                                    # Variable name: energy_0_training, forces_1_testing, etc.
                                    testing_str = "testing" if testing_flag else "training"
                                    var_name = f"{row_type_lower}_{group_idx}_{testing_str}"
                                    stream.write(var_name, data_array, count=data_array.shape)
                    
                    stream.end_step()
                    stream.close()
                    outfile_section.adios2_stream = None

        
        # Add metadata columns for local slice
        for key in ['Groups', 'Testing', 'Row_Type']:
            if key in fs_dict and isinstance(fs_dict[key], list):
                local_values = fs_dict[key][local_slice]
                df_local[key] = local_values
            else:
                # Set defaults
                if key == 'Groups': df_local[key] = ['*ALL'] * len(df_local)
                elif key == 'Testing': df_local[key] = [False] * len(df_local)
                elif key == 'Row_Type': df_local[key] = ['Energy'] * len(df_local)

        # Compute local group sums
        local_group_data = self._compute_local_group_sums_from_df(df_local)
                
        # Gather all local group data to rank 0 and merge
        all_group_data = pt._comm.gather(local_group_data, root=0)
        
        if pt._rank == 0:
            # Merge all group data on rank 0
            global_group_data = self._merge_group_data(all_group_data)
            # Add '*ALL' groups by aggregating
            global_group_data_with_all = self._add_all_groups_to_global_data(global_group_data)
            # Broadcast merged data to all ranks for two-pass R² calculation
            global_group_data_with_all = pt._comm.bcast(global_group_data_with_all, root=0)
        else:
            global_group_data_with_all = pt._comm.bcast(None, root=0)
        
        # Two-pass algorithm for exact R² calculation
        final_results = self._compute_final_metrics_twopass(global_group_data_with_all, df_local, pt._comm)
        
        if pt._rank == 0: self._format_results_as_dataframe(final_results)
        else: self.errors = []

    # --------------------------------------------------------------------------------------------

    def _merge_group_data(self, all_group_data):
        """Merge group data from all ranks on rank 0"""
        from collections import defaultdict
        
        merged = defaultdict(lambda: {
            'n': 0,
            'sum_weights': 0.0,
            'sum_truths_weighted': 0.0,
            'sum_ae': 0.0,
            'sum_se': 0.0,
            'sum_truths_unweighted': 0.0,
            'sum_ae_unweighted': 0.0,
            'sum_se_unweighted': 0.0
        })
        
        for local_data in all_group_data:
            for group_key, stats in local_data.items():
                for key in ['n', 'sum_weights', 'sum_truths_weighted', 'sum_ae', 'sum_se',
                           'sum_truths_unweighted', 'sum_ae_unweighted', 'sum_se_unweighted']:
                    merged[group_key][key] += stats[key]
        
        return dict(merged)

    # --------------------------------------------------------------------------------------------

    def _compute_local_group_sums_from_df(self, df_local):
        """Compute partial sums for each group from DataFrame (like legacy solver)"""
        from collections import defaultdict
        
        local_group_data = defaultdict(lambda: {
            'n': 0,
            'sum_weights': 0.0,
            'sum_truths_weighted': 0.0,
            'sum_ae': 0.0,
            'sum_se': 0.0,
            # Add unweighted sums for correct unweighted metrics
            'sum_truths_unweighted': 0.0,
            'sum_ae_unweighted': 0.0,
            'sum_se_unweighted': 0.0
        })
        
        for _, row in df_local.iterrows():
            group_key = (row['Groups'], row['Testing'], row['Row_Type'])
            if self.config.sections["REFERENCE"].units == "metal":
              if row['Row_Type'] != "Stress": factor = 1000.0  # energy and forces eV -> meV
              else: factor = .0001   # stress bar -> GPa
            else: factor = 1.0
            weight = row['weights']
            truth = factor * row['truths']
            pred = factor * row['preds']
            stats = local_group_data[group_key]
            stats['n'] += 1
            # Weighted sums
            stats['sum_weights'] += weight
            stats['sum_truths_weighted'] += weight * truth
            stats['sum_ae'] += weight * abs(truth - pred)
            stats['sum_se'] += weight * (truth - pred)**2
            # Unweighted sums (ignore weights entirely)
            stats['sum_truths_unweighted'] += truth
            stats['sum_ae_unweighted'] += abs(truth - pred)
            stats['sum_se_unweighted'] += (truth - pred)**2
        
        return dict(local_group_data)

    # --------------------------------------------------------------------------------------------

    def _compute_final_metrics_twopass(self, global_group_data, df_local, comm):
        """Two-pass algorithm for exact R² using mpi4py collectives"""
        rank = comm.Get_rank()
        
        # Pass 1: Compute global means (already have from global_group_data)
        global_means_weighted = {}
        global_means_unweighted = {}
        
        for group_key, stats in global_group_data.items():
            # Weighted mean
            if stats['sum_weights'] > 0:
                global_means_weighted[group_key] = stats['sum_truths_weighted'] / stats['sum_weights']
            else:
                global_means_weighted[group_key] = 0.0
            
            # Unweighted mean  
            if stats['n'] > 0:
                global_means_unweighted[group_key] = stats['sum_truths_unweighted'] / stats['n']
            else:
                global_means_unweighted[group_key] = 0.0
        

        # Pass 2: Compute local SS_tot contributions using global means
        local_ss_tot_weighted = {}
        local_ss_tot_unweighted = {}
        
        for _, row in df_local.iterrows():
            group_key = (row['Groups'], row['Testing'], row['Row_Type'])

            # Determine conversion factor (same as in _compute_local_group_sums_from_df)
            if self.config.sections["REFERENCE"].units == "metal":
              if row['Row_Type'] != "Stress": factor = 1000.0  # energy and forces eV -> meV
              else: factor = .0001   # stress bar -> GPa
            else: factor = 1.0

            if group_key in global_means_weighted:
                weight = row['weights']
                truth = factor * row['truths']  # Apply same factor as in Pass 1
                
                # Weighted SS_tot for individual group
                weighted_mean = global_means_weighted[group_key]
                if group_key not in local_ss_tot_weighted: local_ss_tot_weighted[group_key] = 0.0
                local_ss_tot_weighted[group_key] += weight * (truth - weighted_mean)**2
                
                # Unweighted SS_tot for individual group
                unweighted_mean = global_means_unweighted[group_key]
                if group_key not in local_ss_tot_unweighted: local_ss_tot_unweighted[group_key] = 0.0
                local_ss_tot_unweighted[group_key] += (truth - unweighted_mean)**2
                
                # Also contribute to "*ALL" groups (but avoid double-counting when group_key is already '*ALL')
                all_key = ('*ALL',) + group_key[1:]
                
                if all_key in global_means_weighted and group_key != all_key:
                    # Weighted SS_tot for *ALL group
                    all_weighted_mean = global_means_weighted[all_key]
                    if all_key not in local_ss_tot_weighted: local_ss_tot_weighted[all_key] = 0.0
                    local_ss_tot_weighted[all_key] += weight * (truth - all_weighted_mean)**2
                    
                    # Unweighted SS_tot for *ALL group
                    all_unweighted_mean = global_means_unweighted[all_key]
                    if all_key not in local_ss_tot_unweighted: local_ss_tot_unweighted[all_key] = 0.0
                    local_ss_tot_unweighted[all_key] += (truth - all_unweighted_mean)**2
        
        # Gather local SS_tot to rank 0 and reduce
        all_ss_tot_weighted = comm.gather(local_ss_tot_weighted, root=0)
        all_ss_tot_unweighted = comm.gather(local_ss_tot_unweighted, root=0)
        
        # Compute final metrics (only on rank 0)
        if rank == 0:
            # Merge SS_tot from all ranks
            global_ss_tot_weighted = {}
            global_ss_tot_unweighted = {}
            
            # DEBUG: Print SS_tot contributions for stress
            if self.config.debug:
                print("\n=== DEBUG: SS_tot contributions ===")
                for i, local_dict in enumerate(all_ss_tot_weighted):
                    for key, value in local_dict.items():
                        if 'Stress' in str(key): print(f"Rank {i}, {key}: SS_tot_weighted = {value}")
                for i, local_dict in enumerate(all_ss_tot_unweighted):
                    for key, value in local_dict.items():
                        if 'Stress' in str(key): print(f"Rank {i}, {key}: SS_tot_unweighted = {value}")
                print("==================================\n")
            
            for local_dict in all_ss_tot_weighted:
                for key, value in local_dict.items():
                    global_ss_tot_weighted[key] = global_ss_tot_weighted.get(key, 0.0) + value
            
            for local_dict in all_ss_tot_unweighted:
                for key, value in local_dict.items():
                    global_ss_tot_unweighted[key] = global_ss_tot_unweighted.get(key, 0.0) + value
            
            final_results = []
            for group_key, stats in global_group_data.items():
                if stats['n'] > 0:
                    # Weighted metrics
                    weighted_mae = stats['sum_ae'] / stats['sum_weights'] if stats['sum_weights'] > 0 else 0
                    weighted_rmse = np.sqrt(stats['sum_se'] / stats['sum_weights']) if stats['sum_weights'] > 0 else 0
                    
                    ss_tot_weighted = global_ss_tot_weighted.get(group_key, 0.0)
                    weighted_rsq = 1 - (stats['sum_se'] / ss_tot_weighted) if ss_tot_weighted != 0 else 0
                    weighted_rsq = max(0.0, weighted_rsq)  # Clip negative R² to 0
                    
                    # Unweighted metrics
                    unweighted_mae = stats['sum_ae_unweighted'] / stats['n']
                    unweighted_rmse = np.sqrt(stats['sum_se_unweighted'] / stats['n'])
                    
                    ss_tot_unweighted = global_ss_tot_unweighted.get(group_key, 0.0)
                    unweighted_rsq = 1 - (stats['sum_se_unweighted'] / ss_tot_unweighted) if ss_tot_unweighted != 0 else 0
                    unweighted_rsq = max(0.0, unweighted_rsq)  # Clip negative R² to 0
                    
                    final_results.append({
                        'group': group_key,
                        'ncount': stats['n'],
                        'weighted_mae': weighted_mae,
                        'weighted_rmse': weighted_rmse,
                        'weighted_rsq': weighted_rsq,
                        'unweighted_mae': unweighted_mae,
                        'unweighted_rmse': unweighted_rmse,
                        'unweighted_rsq': unweighted_rsq,
                        '_sum_weights': stats['sum_weights'],
                        '_sum_ae': stats['sum_ae'],
                        '_sum_se': stats['sum_se'],
                        '_sum_ss_tot_weighted': ss_tot_weighted,
                        '_sum_ss_tot_unweighted': ss_tot_unweighted
                    })
            
            return final_results
        
        return None

    # --------------------------------------------------------------------------------------------

    def _format_results_as_dataframe(self, results):
        """Convert results to pandas DataFrame format matching solver.py"""
        from pandas import DataFrame, concat
        
        if not results:
            self.errors = DataFrame()
            return
        
        # '*ALL' groups already added by _add_all_groups_to_global_data()
        # Create both weighted and unweighted versions
        formatted_results = []
        
        for result in results:
            group_key = result['group']
            
            # Use the correctly computed weighted and unweighted metrics
            unweighted_mae = result['unweighted_mae']
            unweighted_rmse = result['unweighted_rmse']
            unweighted_rsq = result['unweighted_rsq']
            weighted_mae = result['weighted_mae']
            weighted_rmse = result['weighted_rmse']
            weighted_rsq = result['weighted_rsq']
            
            # Add both versions with proper indexing
            testing_str = 'Testing' if group_key[1] else 'Training'
            
            formatted_results.extend([{
                'Groups': group_key[0],
                'Weighting': 'Unweighted',
                'Testing': testing_str,
                'Row_Type': group_key[2],
                'ncount': result['ncount'],
                'mae': unweighted_mae,
                'rmse': unweighted_rmse,
                'rsq': unweighted_rsq
              },{
                'Groups': group_key[0],
                'Weighting': 'weighted',
                'Testing': testing_str,
                'Row_Type': group_key[2],
                'ncount': result['ncount'],
                'mae': weighted_mae,
                'rmse': weighted_rmse,
                'rsq': weighted_rsq
              }])
        
        # Convert to DataFrame with proper MultiIndex
        df = DataFrame(formatted_results)
        if not df.empty:
            df = df.set_index(['Groups', 'Weighting', 'Testing', 'Row_Type']).sort_index()
            df.index.rename(["Group", "Weighting", "Testing", "Subsystem"], inplace=True)
        self.errors = df
  
    # --------------------------------------------------------------------------------------------

    def _add_all_groups_to_global_data(self, global_group_data):
        """Add '*ALL' groups by aggregating raw sums at the global_group_data level"""
        
        # Organize by aggregation keys
        aggregations = {}
        
        for group_key, stats in global_group_data.items():
            # Skip if already an '*ALL' group
            if group_key[0] == '*ALL': continue

            # Create aggregation key: replace Groups with '*ALL'
            agg_key = ('*ALL',) + group_key[1:]
            
            if agg_key not in aggregations:
                aggregations[agg_key] = {
                    'n': 0,
                    'sum_weights': 0.0,
                    'sum_truths_weighted': 0.0,
                    'sum_ae': 0.0,
                    'sum_se': 0.0,
                    'sum_truths_unweighted': 0.0,
                    'sum_ae_unweighted': 0.0,
                    'sum_se_unweighted': 0.0
                }
            
            # Aggregate the raw sums
            agg = aggregations[agg_key]
            for key in ['n', 'sum_weights', 'sum_truths_weighted', 'sum_ae', 'sum_se',
                       'sum_truths_unweighted', 'sum_ae_unweighted', 'sum_se_unweighted']:
                agg[key] += stats[key]
        
        # Add aggregated groups to global data
        result = global_group_data.copy()
        result.update(aggregations)
        
        return result

    # --------------------------------------------------------------------------------------------

