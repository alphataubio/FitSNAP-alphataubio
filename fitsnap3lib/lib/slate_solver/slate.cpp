#include <slate/slate.hh>
#include <lapack.hh>
#include <mpi.h>
#include <omp.h>

#include <cstdint>
#include <cmath>
#include <vector>
#include <iostream>
#include <functional>
#include <algorithm> 
#include <cinttypes> 

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

extern "C" {

using slate::func::ij_tuple;

constexpr int64_t ceil_div64(int64_t a, int64_t b) { return (a + b - 1) / b; }

// -----------------------------------------------------------------------------
// Helper: Configure OpenMP (Affinity + Threads)
// -----------------------------------------------------------------------------
void slate_set_openmp_threads(int num_threads, int debug) {
#ifdef __linux__
    cpu_set_t mask;
    CPU_ZERO(&mask);
    int ncpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpus <= 0) ncpus = 64;
    for (int i = 0; i < ncpus; i++) CPU_SET(i, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
#endif
    omp_set_num_threads(num_threads);
    if (debug) std::cerr << "SLATE_CPP: Set OpenMP threads to " << num_threads << std::endl;
}

// -----------------------------------------------------------------------------
// SLATE Ridge Solver
// -----------------------------------------------------------------------------
void slate_ridge_augmented_qr(double* local_aw, double* local_bw, 
                              int64_t m, int64_t n, int64_t lld, 
                              MPI_Comm comm, int debug) {
    
    // -------------------------------- HYBRID MPI/OPENMP --------------------------------
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);
    int num_threads = omp_get_max_threads();
    
    // -------------------------------- TILE SIZE --------------------------------
    // FIXME: find optimal tile size based on cache size
    
    int64_t nb = 256;
    int64_t nt = ceil_div64(n, nb);
    int64_t mb = nb;
    int64_t m_node = m / mpi_size;
    int64_t mt_node = ceil_div64(m_node, mb);
    int64_t mt = mt_node * mpi_size;
    
    std::function<int64_t (int64_t)> tile1 = [](int64_t) { return 1; };
       
    int64_t tile_row_last = mt_node - 1;
    int64_t tile_row_remainder = lld - (mt_node-1)*mb;
    std::function<int64_t (int64_t)> tileMb = [mt_node, tile_row_last, tile_row_remainder, mb](int64_t i) {
        if (i % mt_node == tile_row_last) return tile_row_remainder;
        else return mb;
    };
        
    int64_t tile_col_last = nt - 1;
    int64_t tile_col_remainder = n - (nt-1)*nb;
    std::function<int64_t (int64_t)> tileNb = [tile_col_last, tile_col_remainder, nb](int64_t j) {
        if (j == tile_col_last) return tile_col_remainder;
        else return nb;
    };

    std::function<int (slate::func::ij_tuple)> tileRank = [mt_node](slate::func::ij_tuple ij) {
        int64_t i = std::get<0>(ij);
        return i / mt_node;
    };

    // FIXME: GPU device tiles not implemented yet (placeholder)
    // need to sync to device from fitsnap python shared array on node-local ram
    std::function<int (slate::func::ij_tuple)> tileDevice = [](slate::func::ij_tuple) { return 0; };
    
    if (mpi_rank == 0) {
        std::cerr << "\n---------------- SLATE Ridge Solver ----------------" << std::endl;
        std::cerr << "MPI: " << mpi_size << " rank(s) (one per node), ";
        std::cerr            << num_threads << " OpenMP threads/node" << std::endl;
        std::cerr << "Rank: " << mpi_rank << " lld " << lld << std::endl;
        std::cerr << "Matrix size: " << m << " x " << n << std::endl;
        std::cerr << "Tile size: " << mb << " x " << nb << std::endl;
        std::cerr << "Grid: " << mt << " x " << nt << std::endl;
        std::cerr << "----------------------------------------------------" << std::endl;
    }
    
    try {
        
        // -------------------------------- SLATE MATRICES --------------------------------
        // pointer to fitsnap python shared array in node-local ram
        
        slate::Matrix<double> A(m, n, tileMb, tileNb, tileRank, tileDevice, comm);
        slate::Matrix<double> b(m, 1, tileMb,  tile1, tileRank, tileDevice, comm);
        
        for (int64_t i = 0; i < mt; ++i)
            for (int64_t j = 0; j < nt; ++j)
                if (A.tileIsLocal(i, j)) {
                    const int64_t offset = (i % mt_node) * mb + j * nb * lld;
                    A.tileInsert(i, j, local_aw + offset, lld);
                }

        for (int64_t i = 0; i < mt; ++i)
            if (b.tileIsLocal(i, 0)) {
                const int64_t offset = (i % mt_node) * mb;
                b.tileInsert(i, 0, local_bw + offset, lld);
            }
        
        if (debug) {
            slate::Options opts = {
              {slate::Option::PrintVerbose, 4},
              {slate::Option::PrintPrecision, 3},
              {slate::Option::PrintWidth, 7}
            };
            //slate::print("A", A, opts);
            //slate::print("b", b, opts);
        }
        
        // -------------------------------- LEAST SQUARES (QR) --------------------------------
        
        slate::least_squares_solve(A, b);
        MPI_Barrier(comm);

        if (debug) {
            slate::Options opts = {
              {slate::Option::PrintVerbose, 4},
              {slate::Option::PrintPrecision, 3},
              {slate::Option::PrintWidth, 7}
            };
            slate::print("b (solution)", b, opts);
        }

    } catch (const std::exception& e) {
        std::cerr << "[Rank " << mpi_rank << "] SLATE error: " << e.what() << std::endl;
        MPI_Abort(comm, 1);
    }
}


// -----------------------------------------------------------------------------
// SLATE ARD Update
// -----------------------------------------------------------------------------
double slate_ard_update(double* local_aw_active, double* local_bw, double* local_sigma_diag, double* local_coef_active,
                     int64_t m, int64_t n_active, int64_t lld,
                     double alpha, double* lambda_active, 
                     MPI_Comm comm, int debug) {

    // -------------------------------- HYBRID MPI/OPENMP --------------------------------
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);
    int num_threads = omp_get_max_threads();
    
    // -------------------------------- TILE SIZE --------------------------------
    // FIXME: find optimal tile size based on cache size
    
    int64_t nb = 256;
    int64_t nt = ceil_div64(n_active, nb);
    int64_t mb = nb;
    int64_t m_node = m / mpi_size;
    int64_t mt_node = ceil_div64(m_node, mb);
    int64_t mt = mt_node * mpi_size;
    
    std::function<int64_t (int64_t)> tile1 = [](int64_t) { return 1; };
       
    int64_t tile_row_last = mt_node - 1;
    int64_t tile_row_remainder = lld - (mt_node-1)*mb;
    std::function<int64_t (int64_t)> tileMb = [mt_node, tile_row_last, tile_row_remainder, mb](int64_t i) {
        if (i % mt_node == tile_row_last) return tile_row_remainder;
        else return mb;
    };
        
    int64_t tile_col_last = nt - 1;
    int64_t tile_col_remainder = n_active - (nt-1)*nb;
    std::function<int64_t (int64_t)> tileNb = [tile_col_last, tile_col_remainder, nb](int64_t j) {
        if (j == tile_col_last) return tile_col_remainder;
        else return nb;
    };

    std::function<int (slate::func::ij_tuple)> tileRank = [mt_node](slate::func::ij_tuple ij) {
        int64_t i = std::get<0>(ij);
        return i / mt_node;
    };

    // FIXME: GPU device tiles not implemented yet (placeholder)
    // need to sync to device from fitsnap python shared array on node-local ram
    std::function<int (slate::func::ij_tuple)> tileDevice = [](slate::func::ij_tuple) { return 0; };
    
    if (mpi_rank == 0 && debug) {
        std::fprintf(stderr, "\n=== slate_ard_update ===\n");
        std::fprintf(stderr, "  m=%" PRId64 ", n_active=%" PRId64 ", alpha=%.6e\n", m, n_active, alpha);
    }
    
    if (n_active == 0) return 1.0; // Perfect conditioning for empty matrix
    
    try {
    
        // -------------------------------- SLATE MATRICES --------------------------------
        // pointer to fitsnap python shared array in node-local ram

        slate::Matrix<double> X_active(m, n_active, tileMb, tileNb, tileRank, tileDevice, comm);
        slate::Matrix<double>        y(m,        1, tileMb,  tile1, tileRank, tileDevice, comm);
                
        for (int64_t i = 0; i < mt; ++i)
            for (int64_t j = 0; j < nt; ++j)
                if (X_active.tileIsLocal(i, j)) {
                    const int64_t offset = (i % mt_node) * mb + j * nb * lld;
                    X_active.tileInsert(i, j, local_aw_active + offset, lld);
                }

        for (int64_t i = 0; i < mt; ++i)
            if (y.tileIsLocal(i, 0)) {
                const int64_t offset = (i % mt_node) * mb;
                y.tileInsert(i, 0, local_bw + offset, lld);
            }
        
        MPI_Barrier(comm);
        
        // -------------------------------- FORM NORMAL EQUATIONS --------------------------------
        // C = alpha * X.T @ X + Lambda

        // make sure all tiles for C and y_prime are on rank 0
        std::function<int (slate::func::ij_tuple)> tileRank0 = [](slate::func::ij_tuple ij) { return 0; };

        slate::HermitianMatrix<double> C(slate::Uplo::Lower, n_active, tileNb, tileRank0, tileDevice, comm);
        C.insertLocalTiles();
        
        // Initialize C with diagonal lambda values
        slate::set(0.0, C);
        for (int64_t idx = 0; idx < n_active; ++idx) {
            int64_t tile_i = idx / nb;
            int64_t local_i = idx % nb;
            if (C.tileIsLocal(tile_i, tile_i)) {
                auto tile = C(tile_i, tile_i);
                tile.at(local_i, local_i) = lambda_active[idx];
            }
        }
        
        MPI_Barrier(comm);

        // C = alpha * X.T @ X + C
        auto X_active_T = transpose(X_active);
        slate::herk(alpha, X_active_T, 1.0, C);
        MPI_Barrier(comm);
        
        // Compute y' = alpha * X.T @ y
        slate::Matrix<double> y_prime(n_active, 1, tileNb, tile1, tileRank0, tileDevice, comm);
        y_prime.insertLocalTiles();
        slate::set(0.0, y_prime);
        slate::gemm(alpha, X_active_T, y, 0.0, y_prime);
        MPI_Barrier(comm);

        // -------------------------------- LOCAL SOLVE (Rank 0) --------------------------------
        // Gather C and y_prime to Rank 0 and solve using LAPACK SVD (dgelss)
        
        // Allocate local buffers
        std::vector<double> C_full;
        std::vector<double> y_prime_full;
        
        if (mpi_rank == 0) {
            C_full.resize(n_active * n_active, 0.0);
            y_prime_full.resize(n_active, 0.0);
        }
        
        // Gather C (Symmetric Lower) -> Full General Matrix on Rank 0
        // Since tileRank0 forces all tiles to Rank 0, we just copy tiles to C_full.
        
        // Iterate over global tiles
        int64_t C_nt = C.nt();
        for (int64_t j = 0; j < C_nt; ++j) {
            for (int64_t i = j; i < C_nt; ++i) { // Lower triangular
                if (mpi_rank == 0) {
                    C.tileGetForReading(i, j, slate::HostNum, slate::LayoutConvert::None);
                    auto tile = C(i, j);
                    double* data = tile.data();
                    int64_t stride = tile.stride();
                    
                    int64_t r_offset = 0; for(int k=0; k<i; ++k) r_offset += C.tileNb(k);
                    int64_t c_offset = 0; for(int k=0; k<j; ++k) c_offset += C.tileNb(k);
                    
                    for (int64_t jj = 0; jj < tile.nb(); ++jj) {
                        for (int64_t ii = 0; ii < tile.mb(); ++ii) {
                            if (i == j && ii < jj) continue; // Lower part only
                            
                            double val = data[ii + jj*stride];
                            int64_t r = r_offset + ii;
                            int64_t c = c_offset + jj;
                            
                            // Fill C_full (Column Major n_active x n_active)
                            if (r < n_active && c < n_active) {
                                C_full[r + c*n_active] = val;
                                // Symmetric fill
                                if (r != c) C_full[c + r*n_active] = val; 
                            }
                        }
                    }
                }
            }
        }
        
        // Gather y_prime
        int64_t y_mt = y_prime.mt();
        for (int64_t i = 0; i < y_mt; ++i) {
            if (mpi_rank == 0) {
                y_prime.tileGetForReading(i, 0, slate::HostNum, slate::LayoutConvert::None);
                auto tile = y_prime(i, 0);
                double* data = tile.data();
                
                int64_t r_offset = 0; for(int k=0; k<i; ++k) r_offset += y_prime.tileMb(k);
                
                for (int64_t ii = 0; ii < tile.mb(); ++ii) {
                    int64_t r = r_offset + ii;
                    if (r < n_active) {
                        y_prime_full[r] = data[ii]; // stride 1? Check? Assuming vector tile is contiguous.
                    }
                }
            }
        }
        
        MPI_Barrier(comm);
        
        double cond_number = 0.0;
        
        if (mpi_rank == 0) {
            // Solve C * x = y' using SVD to handle near-singular C
            // C = U S V^T. 
            // x = V S^-1 U^T y'
            // Covariance = V S^-1 V^T
            
            // lapack::gesvd
            std::vector<double> S(n_active);
            std::vector<double> U(n_active * n_active); // U
            std::vector<double> VT(n_active * n_active); // VT
            
            // Work query
            // lapack::gesvd(Job::AllVec, Job::AllVec, ...)
            
            // Note: C_full will be destroyed. Make a copy if needed? No need.
            
            lapack::gesvd(lapack::Job::AllVec, lapack::Job::AllVec, n_active, n_active,
                          C_full.data(), n_active, S.data(), U.data(), n_active, VT.data(), n_active);
                          
            // Compute Condition Number
            double s_max = S[0];
            double s_min = S[n_active-1];
            cond_number = (s_min > 0) ? (s_max / s_min) : 1e16;
            
            // Invert S (with threshold)
            std::vector<double> S_inv(n_active, 0.0);
            double threshold = 1e-14 * s_max;
            for(int i=0; i<n_active; ++i) {
                if (S[i] > threshold) S_inv[i] = 1.0 / S[i];
                else S_inv[i] = 0.0;
            }
            
            // Compute x = V * (S_inv * (U^T * y_prime))
            // 1. tmp = U^T * y_prime
            std::vector<double> tmp(n_active, 0.0);
            blas::gemv(blas::Layout::ColMajor, blas::Op::Trans, n_active, n_active, 
                       1.0, U.data(), n_active, y_prime_full.data(), 1, 0.0, tmp.data(), 1);
                       
            // 2. tmp = S_inv * tmp (elementwise)
            for(int i=0; i<n_active; ++i) tmp[i] *= S_inv[i];
            
            // 3. x = V^T^T * tmp = V * tmp. Note VT is V^T. So V is VT^T.
            // x = VT^T * tmp = (tmp^T * VT)^T ? No.
            // x = V * tmp. V has columns of V. VT has rows of V^T = cols of V.
            // So V is simply VT transposed.
            // x = VT^T * tmp. 
            // Use gemv with Trans on VT.
            blas::gemv(blas::Layout::ColMajor, blas::Op::Trans, n_active, n_active,
                       1.0, VT.data(), n_active, tmp.data(), 1, 0.0, local_coef_active, 1);
                       
            // Compute Covariance Diagonal
            // Sigma = V * diag(S_inv) * V^T
            // Sigma_ii = sum_k V_ik^2 * S_inv_k
            // V_ik is element (i,k) of V.
            // V is transpose of VT. VT_ki.
            // So V_ik = VT_ki.
            // Sigma_ii = sum_k (VT_ki)^2 * S_inv_k
            
            for(int i=0; i<n_active; ++i) {
                double sum = 0.0;
                for(int k=0; k<n_active; ++k) {
                    double v_ik = VT[k + i*n_active]; // VT is col-major n x n. VT(k, i)
                    sum += v_ik * v_ik * S_inv[k];
                }
                local_sigma_diag[i] = sum;
            }
        }
        
        // Broadcast results
        MPI_Bcast(local_coef_active, n_active, MPI_DOUBLE, 0, comm);
        MPI_Bcast(local_sigma_diag, n_active, MPI_DOUBLE, 0, comm);
        MPI_Bcast(&cond_number, 1, MPI_DOUBLE, 0, comm);
        
        return cond_number;
        
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << mpi_rank << "] SLATE ARD error: " << e.what() << std::endl;
        MPI_Abort(comm, 1);
        return std::numeric_limits<double>::infinity(); // Return inf on error
    }
}

} // extern "C"
