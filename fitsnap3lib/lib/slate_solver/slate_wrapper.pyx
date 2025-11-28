# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++

import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t
from libc.stddef cimport size_t


cdef extern from *:
    """
    #include <cstdint>
    
    extern "C" {
        void slate_ridge_augmented_qr(double* local_aw, double* local_bw,
                                      int64_t m, int64_t n, int64_t lld, int debug);
        
        double slate_ard_update(double* local_aw_active, double* local_bw, 
                             double* local_sigma_diag, double* local_coef_active,
                             int64_t m, int64_t n_active, int64_t lld,
                             double alpha, double* lambda_active, int debug);
    }
    """

    void slate_ridge_augmented_qr(double* local_aw, double* local_bw, 
                                  int64_t m, int64_t n, int64_t lld, int debug) except +
    
    double slate_ard_update(double* local_aw_active, double* local_bw, 
                         double* local_sigma_diag, double* local_coef_active,
                         int64_t m, int64_t n_active, int64_t lld,
                         double alpha, double* lambda_active, int debug) except +

def slate_ridge_augmented_qr_cython(double[::1, :] local_aw, double[::1] local_bw, int m, int lld, int debug=0):
    cdef int n = <int>local_aw.shape[1]
    slate_ridge_augmented_qr(&local_aw[0, 0], &local_bw[0], m, n, lld, debug)

def slate_ard_update_cython(double[::1, :] local_aw_active, double[::1] local_bw,
                           double[::1] lambda_active, double alpha,
                           int m, int n_active, int lld, int debug=0):
    """
    Perform one iteration of ARD updates: compute sigma diagonal and coefficients.
    
    Parameters:
    -----------
    local_aw_active : (lld, n_active) design matrix with only active features (column-major)
    local_bw : (lld,) target vector (local portion)
    lambda_active : (n_active,) feature precisions for active features only
    alpha : float, noise precision
    m : int, global number of samples
    n_active : int, number of active features
    lld : int, local leading dimension
    debug : int, debug flag
    
    Returns:
    --------
    sigma_diag : (n_active,) diagonal of posterior covariance matrix
    coef_active : (n_active,) coefficients for active features
    cond_number : float, condition number estimate from Cholesky diagonal
    """
    
    if n_active == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64), 1.0
    
    # Allocate output arrays - only diagonal of sigma now!
    cdef np.ndarray[double, ndim=1] sigma_diag = np.zeros(n_active, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] coef_active = np.zeros(n_active, dtype=np.float64)
    cdef double cond_number
    
    # Call the C++ function
    cond_number = slate_ard_update(&local_aw_active[0, 0], &local_bw[0], 
                    <double*>np.PyArray_DATA(sigma_diag), <double*>np.PyArray_DATA(coef_active),
                    m, n_active, lld, alpha, &lambda_active[0], debug)
    
    return sigma_diag, coef_active, cond_number
