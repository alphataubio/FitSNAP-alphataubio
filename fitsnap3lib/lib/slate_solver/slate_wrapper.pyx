# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++

import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t

# -----------------------------------------------------------------------------
# MPI4PY C-API INTEGRATION
# -----------------------------------------------------------------------------
from mpi4py cimport MPI
from mpi4py.libmpi cimport MPI_Comm

cdef extern from *:
    """
    #include <mpi.h> 
    extern "C" {
        void slate_set_openmp_threads(int num_threads, int debug);
        
        void slate_ridge_augmented_qr(double* local_aw, double* local_bw,
                                      int64_t m, int64_t n, int64_t lld, 
                                      MPI_Comm comm, int debug);
                                      
        double slate_ard_update(double* local_aw_active, double* local_bw, 
                                double* local_sigma_diag, double* local_coef_active,
                                int64_t m, int64_t n_active, int64_t lld,
                                double alpha, double* lambda_active, 
                                MPI_Comm comm, int debug);
    }
    """
    
    void slate_set_openmp_threads(int num_threads, int debug) except +
    
    void slate_ridge_augmented_qr(double* local_aw, double* local_bw, 
                                  int64_t m, int64_t n, int64_t lld, 
                                  MPI_Comm comm, int debug) except +
    
    double slate_ard_update(double* local_aw_active, double* local_bw, 
                           double* local_sigma_diag, double* local_coef_active,
                           int64_t m, int64_t n_active, int64_t lld,
                           double alpha, double* lambda_active, 
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
                           MPI.Comm comm_obj, 
                           int debug=0):
    
    if n_active == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64), 1.0
    
    cdef np.ndarray[double, ndim=1] sigma_diag = np.zeros(n_active, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] coef_active = np.zeros(n_active, dtype=np.float64)
    cdef double cond_number
    
    cdef MPI_Comm c_comm = comm_obj.ob_mpi
    
    cond_number = slate_ard_update(&local_aw_active[0, 0], &local_bw[0], 
                    <double*>np.PyArray_DATA(sigma_diag), <double*>np.PyArray_DATA(coef_active),
                    m, n_active, lld, alpha, &lambda_active[0], c_comm, debug)
    
    return sigma_diag, coef_active, cond_number
