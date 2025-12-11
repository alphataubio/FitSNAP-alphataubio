#include <slate/slate.hh>
#include <mpi.h>
#include <omp.h>

#include <cstdint>
#include <cmath>
#include <vector>
#include <iostream>
#include <functional>
#include <algorithm> 
#include <cinttypes> 
#include <limits>

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
// SLATE Ridge Solver (SVD-based)
// -----------------------------------------------------------------------------
// Refactored to use SVD.
// Directly views local_aw/local_bw to avoid copying.
// Returns condition number.
double slate_ridge_augmented_qr(double* local_aw, double* local_bw, 
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
    
    double cond_number = 0.0;

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
        
        // -------------------------------- SVD DECOMPOSITION --------------------------------
        // A = U * Sigma * VH
        // Note: slate::svd may modify A (local_aw) which is acceptable for this workflow
        int64_t min_mn = std::min(m, n);
        std::vector<double> Sigma(min_mn);
        
        slate::Matrix<double> U(m, min_mn, tileMb, tileNb, tileRank, tileDevice, comm);
        slate::Matrix<double> VH(min_mn, n, tileNb, tileNb, tileRank, tileDevice, comm);
        
        U.insertLocalTiles();
        VH.insertLocalTiles();
        
        if (debug) std::cerr << "[SLATE] Computing SVD..." << std::endl;
        slate::svd(A, Sigma, U, VH);
        MPI_Barrier(comm);

        // -------------------------------- CONDITION NUMBER --------------------------------
        double max_sigma = Sigma[0];
        double min_sigma = Sigma[min_mn - 1];
        cond_number = (min_sigma > 1e-16) ? (max_sigma / min_sigma) : std::numeric_limits<double>::infinity();
        
        if (mpi_rank == 0 && debug) {
            std::fprintf(stderr, "[SLATE] Cond Number: %.6e (Max: %.2e, Min: %.2e)\n", cond_number, max_sigma, min_sigma);
        }

        // -------------------------------- SOLVE x = V * inv(Sigma) * U^H * b --------------------------------
        
        // 1. Temp = U^H * b
        // Output T is min_mn x 1.
        slate::Matrix<double> T(min_mn, 1, tileNb, tile1, tileRank, tileDevice, comm);
        T.insertLocalTiles();
        slate::set(0.0, T);
        
        auto UH = slate::conj_transpose(U);
        slate::gemm(1.0, UH, b, 0.0, T);
        MPI_Barrier(comm);
        
        // 2. Scale T by inverse singular values
        int64_t T_mt = T.mt();
        for (int64_t i = 0; i < T_mt; ++i) {
            if (T.tileIsLocal(i, 0)) {
                auto tile = T(i, 0);
                int64_t i_global_start = i * nb; 
                for (int64_t ii = 0; ii < tile.mb(); ++ii) {
                    int64_t global_idx = i_global_start + ii;
                    if (global_idx < min_mn) {
                        double s = Sigma[global_idx];
                        if (s > 1e-15) tile.at(ii, 0) /= s;
                        else tile.at(ii, 0) = 0.0;
                    }
                }
            }
        }
        MPI_Barrier(comm);

        // 3. x = V * T  (where V = VH^H)
        // We write the result x (n x 1) directly into local_bw (first n rows).
        // To do this, we create a matrix X_mat that views the first n rows of local_bw.
        // We use tileMb and tileRank from 'b' to ensure distribution matches 'b'.
        
        slate::Matrix<double> X_mat(n, 1, tileMb, tile1, tileRank, tileDevice, comm);
        for (int64_t i = 0; i < X_mat.mt(); ++i) {
            if (X_mat.tileIsLocal(i, 0)) {
                 // Calculate same offset as b
                 const int64_t offset = (i % mt_node) * mb;
                 // Point to local_bw. 
                 // If the last tile is partial, SLATE handles the bounds, 
                 // we just provide the pointer to the start of the block.
                 X_mat.tileInsert(i, 0, local_bw + offset, lld); 
            }
        }
        
        auto V = slate::conj_transpose(VH);
        // Compute x = V * T, writing to X_mat (which wraps local_bw)
        slate::set(0.0, X_mat);
        slate::gemm(1.0, V, T, 0.0, X_mat);
        MPI_Barrier(comm);

    } catch (const std::exception& e) {
        std::cerr << "[Rank " << mpi_rank << "] SLATE Ridge SVD error: " << e.what() << std::endl;
        MPI_Abort(comm, 1);
    }

    return cond_number;
}


// -----------------------------------------------------------------------------
// SLATE ARD Update (SVD-based)
// -----------------------------------------------------------------------------
// Uses SVD on the constructed Covariance matrix C to avoid Cholesky issues.
double slate_ard_update(double* local_aw_active, double* local_bw, double* local_sigma_diag, double* local_coef_active,
                     int64_t m, int64_t n_active, int64_t lld,
                     double alpha, double* lambda_active, 
                     MPI_Comm comm, int debug) {

    // -------------------------------- HYBRID MPI/OPENMP --------------------------------
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);
    
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
        if (i % mt_node == tile_row_last) return tile_row_remainder; else return mb;
    };
    int64_t tile_col_last = nt - 1;
    int64_t tile_col_remainder = n_active - (nt-1)*nb;
    std::function<int64_t (int64_t)> tileNb = [tile_col_last, tile_col_remainder, nb](int64_t j) {
        if (j == tile_col_last) return tile_col_remainder; else return nb;
    };
    std::function<int (slate::func::ij_tuple)> tileRank = [mt_node](slate::func::ij_tuple ij) {
        int64_t i = std::get<0>(ij);
        return i / mt_node;
    };

    // FIXME: GPU device tiles not implemented yet (placeholder)
    // need to sync to device from fitsnap python shared array on node-local ram
    std::function<int (slate::func::ij_tuple)> tileDevice = [](slate::func::ij_tuple) { return 0; };
    
    if (n_active == 0) return 1.0; 
    
    if (mpi_rank == 0 && debug) {
        std::fprintf(stderr, "\n=== slate_ard_update (SVD) ===\n");
        std::fprintf(stderr, "  m=%" PRId64 ", n=%" PRId64 "\n", m, n_active);
    }
            
    try {
    
        // -------------------------------- SLATE MATRICES --------------------------------
        // pointer to fitsnap python shared array in node-local ram

        slate::Matrix<double> X(m, n_active, tileMb, tileNb, tileRank, tileDevice, comm);
        slate::Matrix<double> y(m,        1, tileMb,  tile1, tileRank, tileDevice, comm);
                
        for (int64_t i = 0; i < mt; ++i)
            for (int64_t j = 0; j < nt; ++j)
                if (X.tileIsLocal(i, j)) {
                    const int64_t offset = (i % mt_node) * mb + j * nb * lld;
                    X.tileInsert(i, j, local_aw_active + offset, lld);
                }
                
        for (int64_t i = 0; i < mt; ++i)
            if (y.tileIsLocal(i, 0)) {
                const int64_t offset = (i % mt_node) * mb;
                y.tileInsert(i, 0, local_bw + offset, lld);
            }
        
        MPI_Barrier(comm);
        
        // -------------------------------- FORM C --------------------------------
        // C = alpha * X^T * X + Lambda
        // We use a General Matrix for C to facilitate SVD
        slate::Matrix<double> C(n_active, n_active, tileNb, tileNb, tileRank, tileDevice, comm);
        C.insertLocalTiles();
        slate::set(0.0, C);
        
        // C = alpha * X^T * X
        auto XT = slate::transpose(X);
        slate::gemm(alpha, XT, X, 0.0, C);
        MPI_Barrier(comm);
        
        // Add Lambda to Diagonal
        for (int64_t i = 0; i < n_active; ++i) {
            int64_t tile_i = i / nb;
            int64_t local_i = i % nb;
            if (C.tileIsLocal(tile_i, tile_i)) {
                C(tile_i, tile_i).at(local_i, local_i) += lambda_active[i];
            }
        }
        MPI_Barrier(comm);

        // -------------------------------- SVD OF C --------------------------------
        // C is symmetric positive definite.
        // C = U * Sigma * VH
        std::vector<double> Sigma(n_active);
        slate::Matrix<double> U(n_active, n_active, tileNb, tileNb, tileRank, tileDevice, comm);
        slate::Matrix<double> VH(n_active, n_active, tileNb, tileNb, tileRank, tileDevice, comm);
        U.insertLocalTiles();
        VH.insertLocalTiles();
        
        slate::svd(C, Sigma, U, VH);
        MPI_Barrier(comm);
        
        // Condition Number
        double max_s = Sigma[0];
        double min_s = Sigma[n_active-1];
        double cond = (min_s > 0.0) ? (max_s / min_s) : std::numeric_limits<double>::infinity();
        
        // -------------------------------- INVERT C --------------------------------
        // C^-1 = VH^H * Sigma^-1 * U^H
        
        // 1. Temp = U^H. Scale rows by Sigma^-1
        slate::Matrix<double> Temp(n_active, n_active, tileNb, tileNb, tileRank, tileDevice, comm);
        Temp.insertLocalTiles();
        auto UH = slate::conj_transpose(U);
        slate::copy(UH, Temp); 
        
        for (int64_t i = 0; i < Temp.mt(); ++i) { 
            for (int64_t j = 0; j < Temp.nt(); ++j) { 
                if (Temp.tileIsLocal(i, j)) {
                    auto tile = Temp(i, j);
                    int64_t i_start = i * nb;
                    for (int64_t ii = 0; ii < tile.mb(); ++ii) {
                        double s = Sigma[i_start + ii];
                        double inv_s = (s > 1e-15) ? 1.0/s : 0.0;
                        for (int64_t jj = 0; jj < tile.nb(); ++jj) {
                            tile.at(ii, jj) *= inv_s;
                        }
                    }
                }
            }
        }
        MPI_Barrier(comm);
        
        // 2. C_inv = VH^H * Temp
        slate::Matrix<double> C_inv(n_active, n_active, tileNb, tileNb, tileRank, tileDevice, comm);
        C_inv.insertLocalTiles();
        slate::set(0.0, C_inv);
        
        auto V = slate::conj_transpose(VH);
        slate::gemm(1.0, V, Temp, 0.0, C_inv);
        MPI_Barrier(comm);

        // -------------------------------- CALC RESULTS --------------------------------
        
        // Extract Diagonal of C_inv
        for (int64_t i = 0; i < n_active; ++i) {
            int64_t tile_i = i / nb;
            int64_t local_i = i % nb;
            if (C_inv.tileIsLocal(tile_i, tile_i)) {
                local_sigma_diag[i] = C_inv(tile_i, tile_i).at(local_i, local_i);
            } else {
                local_sigma_diag[i] = 0.0;
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, local_sigma_diag, n_active, MPI_DOUBLE, MPI_SUM, comm);
        
        // Compute Coef = alpha * C_inv * X^T * y
        // 1. Z = X^T * y
        slate::Matrix<double> Z(n_active, 1, tileNb, tile1, tileRank, tileDevice, comm);
        Z.insertLocalTiles();
        slate::set(0.0, Z);
        slate::gemm(1.0, XT, y, 0.0, Z);
        MPI_Barrier(comm);
        
        // 2. Coef_mat = alpha * C_inv * Z
        slate::Matrix<double> Coef_mat(n_active, 1, tileNb, tile1, tileRank, tileDevice, comm);
        Coef_mat.insertLocalTiles();
        slate::set(0.0, Coef_mat);
        slate::gemm(alpha, C_inv, Z, 0.0, Coef_mat);
        MPI_Barrier(comm);
        
        // Extract Coef
        for (int64_t i = 0; i < n_active; ++i) {
            int64_t tile_i = i / nb;
            int64_t local_i = i % nb;
            if (Coef_mat.tileIsLocal(tile_i, 0)) {
                local_coef_active[i] = Coef_mat(tile_i, 0).at(local_i, 0);
            } else {
                local_coef_active[i] = 0.0;
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, local_coef_active, n_active, MPI_DOUBLE, MPI_SUM, comm);

        if (mpi_rank == 0 && debug) {
            std::fprintf(stderr, "=== slate_ard_update (SVD) COMPLETE ===\n");
        }
        
        return cond;
        
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << mpi_rank << "] SLATE ARD SVD error: " << e.what() << std::endl;
        MPI_Abort(comm, 1);
        return std::numeric_limits<double>::infinity();
    }
}

} // extern "C"
