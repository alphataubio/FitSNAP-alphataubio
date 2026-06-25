# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++

import numpy as np
cimport numpy as np
from libc.stdint cimport int32_t, int64_t

# -----------------------------------------------------------------------------
# MPI4PY C-API INTEGRATION
# -----------------------------------------------------------------------------
from mpi4py cimport MPI
from mpi4py.libmpi cimport MPI_Comm

cdef extern from *:
    """
    #include <mpi.h> 
    extern "C" {
        
        void slate_ridge_augmented_qr(double* local_aw, double* local_bw,
                                      int64_t m, int64_t n, int64_t lld, 
                                      MPI_Comm comm, int debug);
                                      
        double slate_ard_update(double* local_aw_active, double* local_bw, 
                                double* local_sigma_diag, double* local_coef_active,
                                double* local_sse,
                                int64_t m, int64_t n_active, int64_t lld,
                                int64_t row_offset, int64_t m_local,
                                double alpha, double* lambda_active, 
                                MPI_Comm comm, int debug);

        void slate_error_analysis(double* local_a, double* local_b, double* local_w, double* fit,
                                  int32_t* bin_specific, int32_t* bin_all, double* group_factor, int n_groups,
                                  int64_t n, int64_t lld, int64_t row_offset, int64_t m_local,
                                  double* out_preds, int64_t* out_count,
                                  double* out_sum_w, double* out_sum_truth_w, double* out_sum_ae_w, double* out_sum_se_w,
                                  double* out_sum_truth_u, double* out_sum_ae_u, double* out_sum_se_u,
                                  double* out_sstot_w, double* out_sstot_u,
                                  MPI_Comm comm, int debug);
    }
    """
    
    void slate_set_openmp_threads(int num_threads, int debug) except +
    
    void slate_ridge_augmented_qr(double* local_aw, double* local_bw, 
                                  int64_t m, int64_t n, int64_t lld, 
                                  MPI_Comm comm, int debug) except +
    
    double slate_ard_update(double* local_aw_active, double* local_bw, 
                           double* local_sigma_diag, double* local_coef_active,
                           double* local_sse,
                           int64_t m, int64_t n_active, int64_t lld,
                           int64_t row_offset, int64_t m_local,
                           double alpha, double* lambda_active, 
                           MPI_Comm comm, int debug) except +

    void slate_error_analysis(double* local_a, double* local_b, double* local_w, double* fit,
                              int32_t* bin_specific, int32_t* bin_all, double* group_factor, int n_groups,
                              int64_t n, int64_t lld, int64_t row_offset, int64_t m_local,
                              double* out_preds, int64_t* out_count,
                              double* out_sum_w, double* out_sum_truth_w, double* out_sum_ae_w, double* out_sum_se_w,
                              double* out_sum_truth_u, double* out_sum_ae_u, double* out_sum_se_u,
                              double* out_sstot_w, double* out_sstot_u,
                              MPI_Comm comm, int debug) except +

# -----------------------------------------------------------------------------
# PYTHON WRAPPERS
# -----------------------------------------------------------------------------

def set_openmp_threads(int num_threads, int debug=0):
    slate_set_openmp_threads(num_threads, debug)

def slate_ridge_augmented_qr_cython(double[::1, :] local_aw, double[::1] local_bw,
                                    int m, int lld, 
                                    MPI.Comm comm_obj, 
                                    int debug=0):
    
    cdef int n = <int>local_aw.shape[1]
    
    # Extract the underlying C MPI_Comm handle safely
    cdef MPI_Comm c_comm = comm_obj.ob_mpi
    
    slate_ridge_augmented_qr(&local_aw[0, 0], &local_bw[0], m, n, lld, c_comm, debug)

def slate_ard_update_cython(double[::1, :] local_aw_active, double[::1] local_bw,
                           double[::1] lambda_active, double alpha,
                           int m, int n_active, int lld, 
                           int row_offset, int m_local,
                           MPI.Comm comm_obj, 
                           int debug=0):
    
    if n_active == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64), 0.0, 1.0
    
    cdef np.ndarray[double, ndim=1] sigma_diag = np.zeros(n_active, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] coef_active = np.zeros(n_active, dtype=np.float64)
    cdef double sse = 0.0
    cdef double cond_number
    
    cdef MPI_Comm c_comm = comm_obj.ob_mpi
    
    cond_number = slate_ard_update(&local_aw_active[0, 0], &local_bw[0], 
                    <double*>np.PyArray_DATA(sigma_diag), <double*>np.PyArray_DATA(coef_active),
                    &sse,
                    m, n_active, lld, row_offset, m_local, alpha, &lambda_active[0], c_comm, debug)
    
    return sigma_diag, coef_active, sse, cond_number


def slate_error_analysis_cython(double[::1, :] local_a, double[::1] local_b, double[::1] local_w,
                                double[::1] fit,
                                int32_t[::1] bin_specific, int32_t[::1] bin_all,
                                double[::1] group_factor, int n_groups,
                                int lld, int row_offset, int m_local,
                                MPI.Comm comm_obj, int debug=0):

    cdef int64_t n = <int64_t>local_a.shape[1]
    cdef MPI_Comm c_comm = comm_obj.ob_mpi

    cdef np.ndarray[double, ndim=1]     preds       = np.zeros(m_local if m_local > 0 else 0, dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim=1] count       = np.zeros(n_groups, dtype=np.int64)
    cdef np.ndarray[double, ndim=1]     sum_w       = np.zeros(n_groups, dtype=np.float64)
    cdef np.ndarray[double, ndim=1]     sum_truth_w = np.zeros(n_groups, dtype=np.float64)
    cdef np.ndarray[double, ndim=1]     sum_ae_w    = np.zeros(n_groups, dtype=np.float64)
    cdef np.ndarray[double, ndim=1]     sum_se_w    = np.zeros(n_groups, dtype=np.float64)
    cdef np.ndarray[double, ndim=1]     sum_truth_u = np.zeros(n_groups, dtype=np.float64)
    cdef np.ndarray[double, ndim=1]     sum_ae_u    = np.zeros(n_groups, dtype=np.float64)
    cdef np.ndarray[double, ndim=1]     sum_se_u    = np.zeros(n_groups, dtype=np.float64)
    cdef np.ndarray[double, ndim=1]     sstot_w     = np.zeros(n_groups, dtype=np.float64)
    cdef np.ndarray[double, ndim=1]     sstot_u     = np.zeros(n_groups, dtype=np.float64)

    # m_local==0 ranks still participate in the collective; pass NULL data pointers.
    cdef double*  preds_ptr = <double*>np.PyArray_DATA(preds) if m_local > 0 else NULL
    cdef int32_t* bs_ptr    = &bin_specific[0] if m_local > 0 else NULL
    cdef int32_t* ba_ptr    = &bin_all[0]      if m_local > 0 else NULL

    slate_error_analysis(&local_a[0, 0], &local_b[0], &local_w[0], &fit[0],
                         bs_ptr, ba_ptr, &group_factor[0], n_groups,
                         n, lld, row_offset, m_local,
                         preds_ptr, <int64_t*>np.PyArray_DATA(count),
                         <double*>np.PyArray_DATA(sum_w),       <double*>np.PyArray_DATA(sum_truth_w),
                         <double*>np.PyArray_DATA(sum_ae_w),    <double*>np.PyArray_DATA(sum_se_w),
                         <double*>np.PyArray_DATA(sum_truth_u), <double*>np.PyArray_DATA(sum_ae_u),
                         <double*>np.PyArray_DATA(sum_se_u),
                         <double*>np.PyArray_DATA(sstot_w),     <double*>np.PyArray_DATA(sstot_u),
                         c_comm, debug)

    return (preds, count, sum_w, sum_truth_w, sum_ae_w, sum_se_w,
            sum_truth_u, sum_ae_u, sum_se_u, sstot_w, sstot_u)
