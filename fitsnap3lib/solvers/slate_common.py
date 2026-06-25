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
slate_error_analysis_cython = None

try:
    # Primary import method (after pip install -e .)
    from slate_wrapper import slate_ridge_augmented_qr_cython, slate_error_analysis_cython
    SLATE_AVAILABLE = True
except ImportError as e:
    # Fallback: try direct path import for in-place builds
    try:
        import sys
        import os
        slate_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib', 'slate_solver')
        if slate_path not in sys.path:
            sys.path.insert(0, slate_path)
        from slate_wrapper import slate_ridge_augmented_qr_cython, slate_ard_update_cython, slate_error_analysis_cython
        SLATE_AVAILABLE = True
    except ImportError:
        print(f"Warning: SLATE module import failed: {e}")
        print("To install: cd fitsnap3lib/lib/slate_solver && pip install -e .")
        slate_ridge_augmented_qr_cython = None
        slate_ard_update_cython = None
        slate_error_analysis_cython = None
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

            if "rdf" in fs_dict:
                rdf = fs_dict["rdf"]
                n_elem = len(rdf["elements"])
                stream.write_attribute("rdf_elements", ",".join(rdf["elements"]))
                stream.write_attribute("rdf_n_bins", int(rdf["n_bins"]))
                stream.write_attribute("rdf_r_centers", np.asarray(rdf["r_centers"], dtype=np.float64))
                stream.write_attribute("rdf_gr", np.asarray(rdf["gr"], dtype=np.float64).reshape(-1))
                stream.write_attribute("rdf_rcut_in", np.asarray(rdf["rcut_in"], dtype=np.float64).reshape(-1))
                stream.write_attribute("rdf_rcut", np.asarray(rdf["rcut"], dtype=np.float64).reshape(-1))

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
            ncoeff_2b_per_pair = pt.fitsnap_dict.get("ncoeff_2b_per_pair", None)
            for i in range(reg_num_rows):
                g = reg_col_idx + i
                if n <= g < n + ncoeff_2b:
                    # 2b second-difference curvature applied PER PAIR independently.
                    # Treating the full ncoeff_2b chain as one couples adjacent pair
                    # types at boundaries: wrong center weight (-2 instead of -1)
                    # AND spurious off-diagonal entries into the adjacent pair's columns.
                    c = g - n  # global 2b index [0, ncoeff_2b)
                    # Walk the pair list to find which pair c belongs to
                    pair_start = 0
                    nc = ncoeff_2b  # fallback if ncoeff_2b_per_pair not available
                    if ncoeff_2b_per_pair:
                        for nc_pair in ncoeff_2b_per_pair:
                            if c < pair_start + nc_pair:
                                nc = nc_pair
                                break
                            pair_start += nc_pair
                    local_c = c - pair_start
                    # Per-pair boundary: half-weight at the two ends of each pair's chain
                    diag_val = -1.0 if (local_c == 0 or local_c == nc - 1) else -2.0
                    col_base = offset_2b + pair_start
                    if local_c > 0:
                        aw[reg_row_idx+i, col_base + local_c - 1] = sqrt_alpha_curvature
                    aw[reg_row_idx+i, col_base + local_c]     = sqrt_alpha_curvature * diag_val
                    if local_c < nc - 1:
                        aw[reg_row_idx+i, col_base + local_c + 1] = sqrt_alpha_curvature
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
        Scalable error analysis. Predictions and per-group error statistics are
        computed in C++/SLATE (slate_error_analysis), which replaces the old
        pandas-DataFrame + iterrows path that copied the entire local design
        matrix and blew up memory.

        The C++ side:
          - computes preds = A @ fit (A aliases the node-shared design buffer),
          - accumulates weighted/unweighted per-group sums,
          - two-pass exact R^2 (global means, then SS_tot) via MPI_Allreduce,
        returning this rank's local predictions plus globally-reduced per-group
        arrays. Python only does the O(n_groups) metric/DataFrame formatting and
        the optional validation scatter. The '*ALL' rollup groups are reduced in
        C++ as well (each row contributes to both its own bin and the '*ALL'
        bin), so the '*ALL' SS_tot is taken about the '*ALL' mean.
        """
        pt = self.pt
        if self.fit is None: return
        if slate_error_analysis_cython is None:
            raise RuntimeError("slate_error_analysis_cython not available; rebuild the SLATE "
                               "extension (cd fitsnap3lib/lib/slate_solver && pip install -e .)")

        fs_dict = pt.fitsnap_dict

        # -------- LOCAL SLICE (data rows only; 'a' has no regularization rows) --------
        start_idx, end_idx = fs_dict["sub_a_indices"]
        m_local = end_idx - start_idx + 1
        local_slice = slice(start_idx, end_idx + 1)

        a_arr = pt.shared_arrays['a'].array   # node-shared, column-major (order='F')
        b_arr = pt.shared_arrays['b'].array
        w_arr = pt.shared_arrays['w'].array
        lld = a_arr.shape[0]                  # node row count = column stride
        n = a_arr.shape[1]                    # number of features

        fit = np.ascontiguousarray(self.fit, dtype=np.float64)
        if np.any(~np.isfinite(fit)):
            self.pt.single_print("WARNING: NaN/Inf in fit coefficients "
                                 f"(NaN={int(np.sum(np.isnan(fit)))}, Inf={int(np.sum(np.isinf(fit)))})")

        # -------- PER-ROW METADATA (this rank's slice) --------
        def meta(key, default):
            if key in fs_dict and isinstance(fs_dict[key], list):
                return list(fs_dict[key][local_slice])
            return [default] * m_local
        groups_local  = meta('Groups', '*ALL')
        testing_local = [bool(t) for t in meta('Testing', False)]
        rowtype_local = meta('Row_Type', 'Energy')

        local_keys     = list(zip(groups_local, testing_local, rowtype_local))
        local_all_keys = [('*ALL', t, r) for (_g, t, r) in local_keys]

        # -------- GLOBAL, RANK-CONSISTENT BIN ORDERING --------
        # Allreduce-by-index requires every rank to agree on bin index <-> key. Union
        # the (specific + '*ALL') keys present on any rank and sort deterministically.
        local_keyset = set(local_keys) | set(local_all_keys)
        global_keyset = set()
        for s in pt._comm.allgather(local_keyset):
            global_keyset |= s
        global_keys = sorted(global_keyset,
                             key=lambda k: (str(k[0]), int(bool(k[1])), str(k[2])))
        key_index = {k: i for i, k in enumerate(global_keys)}
        n_groups = len(global_keys)

        if n_groups == 0:
            if pt._rank == 0:
                from pandas import DataFrame
                self.errors = DataFrame()
            else:
                self.errors = []
            return

        bin_specific = np.fromiter((key_index[k] for k in local_keys),
                                   dtype=np.int32, count=m_local)
        bin_all      = np.fromiter((key_index[k] for k in local_all_keys),
                                   dtype=np.int32, count=m_local)

        # -------- PER-BIN UNIT-CONVERSION FACTOR (depends only on Row_Type) --------
        units = self.config.sections["REFERENCE"].units
        def factor_for(row_type):
            if units == "metal":
                return 0.0001 if row_type == "Stress" else 1000.0   # stress bar->GPa ; E/F eV->meV
            return 1.0
        group_factor = np.array([factor_for(k[2]) for k in global_keys], dtype=np.float64)

        # -------- C++/SLATE: predictions + globally-reduced per-group statistics --------
        debug_flag = 1 if self.config.debug else 0
        (preds, count, sum_w, sum_truth_w, sum_ae_w, sum_se_w,
         sum_truth_u, sum_ae_u, sum_se_u, sstot_w, sstot_u) = slate_error_analysis_cython(
            a_arr, b_arr, w_arr, fit, bin_specific, bin_all, group_factor, n_groups,
            lld, start_idx, m_local, pt._comm, debug_flag)

        # -------- VALIDATION SCATTERPLOT DATA (optional; rank 0 writes adios2) --------
        if self.validation:
            truths_local = np.asarray(b_arr[local_slice], dtype=np.float64)
            self._write_validation_scatter(preds, truths_local,
                                           groups_local, testing_local, rowtype_local)

        # -------- FINAL METRICS + DataFrame (rank 0; O(n_groups)) --------
        if pt._rank == 0:
            self._format_results_from_arrays(global_keys, count, sum_w, sum_truth_w,
                                             sum_ae_w, sum_se_w, sum_truth_u, sum_ae_u,
                                             sum_se_u, sstot_w, sstot_u)
        else:
            self.errors = []

    # --------------------------------------------------------------------------------------------

    def _write_validation_scatter(self, preds, truths, groups_local, testing_local, rowtype_local):
        """
        Gather per-row (truth, pred) to rank 0 and write per-(row_type, group, testing)
        arrays to the adios2 stream for the validation scatterplots. ALL ranks must call
        this (collective gather); only rank 0 writes.
        """
        pt = self.pt
        fs_dict = pt.fitsnap_dict

        all_preds     = pt._comm.gather(preds, root=0)
        all_truths    = pt._comm.gather(truths, root=0)
        all_groups    = pt._comm.gather(list(groups_local), root=0)
        all_testing   = pt._comm.gather(list(testing_local), root=0)
        all_row_types = pt._comm.gather(list(rowtype_local), root=0)

        if pt._rank != 0:
            return

        outfile_section = self.config.sections["OUTFILE"]
        if not (hasattr(outfile_section, 'adios2_stream') and outfile_section.adios2_stream is not None):
            return
        stream = outfile_section.adios2_stream

        # Flatten gathered data
        preds_flat     = np.concatenate(all_preds)
        truths_flat    = np.concatenate(all_truths)
        groups_flat    = [g for sublist in all_groups for g in sublist]
        testing_flat   = [t for sublist in all_testing for t in sublist]
        row_types_flat = [r for sublist in all_row_types for r in sublist]

        # Get sorted group names and unique row types
        sorted_group_names = fs_dict["sorted_group_names"]
        unique_row_types = sorted(set(row_types_flat))

        # Write separate variables for each (row_type, group_idx, testing) combination.
        # Each variable has shape (n_points, 2) with columns [truths, predictions].
        stream.begin_step()
        for row_type in unique_row_types:
            row_type_lower = row_type.lower()
            for group_idx, group_name in enumerate(sorted_group_names):
                for testing_flag in [False, True]:
                    # Filter data for this combination
                    mask = np.array([(g == group_name and t == testing_flag and r == row_type)
                                     for g, t, r in zip(groups_flat, testing_flat, row_types_flat)])
                    if np.any(mask):
                        subset_preds = preds_flat[mask]
                        subset_truths = truths_flat[mask]
                        # Stack into (n_points, 2) array: [truths, predictions]
                        data_array = np.column_stack([subset_truths, subset_preds]).astype(np.float32)
                        # Variable name: energy_0_training, force_1_testing, etc.
                        testing_str = "testing" if testing_flag else "training"
                        var_name = f"{row_type_lower}_{group_idx}_{testing_str}"
                        stream.write(var_name, data_array, count=data_array.shape)

        stream.end_step()
        stream.close()
        outfile_section.adios2_stream = None

    # --------------------------------------------------------------------------------------------

    def _format_results_from_arrays(self, global_keys, count, sum_w, sum_truth_w,
                                    sum_ae_w, sum_se_w, sum_truth_u, sum_ae_u, sum_se_u,
                                    sstot_w, sstot_u):
        """
        Turn the globally-reduced per-group arrays from slate_error_analysis into the
        list-of-dicts that _format_results_as_dataframe consumes. The '*ALL' rollup
        bins are already present in global_keys (and already reduced in C++), so no
        Python-side aggregation is required.
        """
        results = []
        for g, key in enumerate(global_keys):
            if count[g] <= 0:
                continue

            sw = sum_w[g]
            weighted_mae  = sum_ae_w[g] / sw if sw > 0 else 0.0
            weighted_rmse = np.sqrt(sum_se_w[g] / sw) if sw > 0 else 0.0
            weighted_rsq  = (1.0 - sum_se_w[g] / sstot_w[g]) if sstot_w[g] != 0 else 0.0
            weighted_rsq  = max(0.0, weighted_rsq)  # Clip negative R^2 to 0

            unweighted_mae  = sum_ae_u[g] / count[g]
            unweighted_rmse = np.sqrt(sum_se_u[g] / count[g])
            unweighted_rsq  = (1.0 - sum_se_u[g] / sstot_u[g]) if sstot_u[g] != 0 else 0.0
            unweighted_rsq  = max(0.0, unweighted_rsq)  # Clip negative R^2 to 0

            results.append({
                'group': key,
                'ncount': int(count[g]),
                'weighted_mae': weighted_mae,
                'weighted_rmse': weighted_rmse,
                'weighted_rsq': weighted_rsq,
                'unweighted_mae': unweighted_mae,
                'unweighted_rmse': unweighted_rmse,
                'unweighted_rsq': unweighted_rsq,
            })

        self._format_results_as_dataframe(results)

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
