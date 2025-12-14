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
    
    int64_t mb = 256;
    int64_t nb = 256;
    int64_t nt = ceil_div64(n, nb);
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

    std::function<int (slate::func::ij_tuple)> none;
    
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
        
        slate::Matrix<double> A(m, n, tileMb, tileNb, tileRank, none, comm);
        slate::Matrix<double> b(m, 1, tileMb,  tile1, tileRank, none, comm);
        
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
    
    int64_t mb = 64;
    int64_t nb = 64;
    int64_t nt = ceil_div64(n_active, nb);
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

    std::function<int (slate::func::ij_tuple)> none;
    
    if (mpi_rank == 0 && debug) {
        std::fprintf(stderr, "\n=== slate_ard_update ===\n");
        std::fprintf(stderr, "  m=%" PRId64 ", n_active=%" PRId64 ", alpha=%.6e\n", m, n_active, alpha);
    }
    
    if (n_active == 0) return 1.0; // Perfect conditioning for empty matrix
    
    try {
    
        // -------------------------------- SLATE MATRICES --------------------------------
        // pointer to fitsnap python shared array in node-local ram

        slate::Matrix<double> X_active(m, n_active, tileMb, tileNb, tileRank, none, comm);
        slate::Matrix<double>        y(m,        1, tileMb,  tile1, tileRank, none, comm);
                
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

        // We calculate C distributedly first
        slate::HermitianMatrix<double> C(slate::Uplo::Lower, n_active, tileNb, tileRank, none, comm);
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
        
        // Compute y' = alpha * X.T @ y (Distributed)
        slate::Matrix<double> y_prime(n_active, 1, tileNb, tile1, tileRank, none, comm);
        y_prime.insertLocalTiles();
        slate::set(0.0, y_prime);
        slate::gemm(alpha, X_active_T, y, 0.0, y_prime);
        MPI_Barrier(comm);

        // -------------------------------- SLATE SVD --------------------------------
        // Solve C * x = y' using SVD: C = U S V^H
        // x = V S^-1 U^H y'
        
        // 1. Convert Hermitian C to General Matrix C_gen for slate::svd
        // Workaround: Use slate::multiply with Identity (HEMM) instead of slate::copy
        // because slate::copy<Hermitian, Matrix> is not instantiated in the library.
        
        slate::Matrix<double> C_gen(n_active, n_active, tileNb, tileNb, tileRank, none, comm);
        C_gen.insertLocalTiles();
        
        slate::Matrix<double> Eye(n_active, n_active, tileNb, tileNb, tileRank, none, comm);
        Eye.insertLocalTiles();
        slate::set(0.0, 1.0, Eye); // Off-diag=0.0, Diag=1.0 -> Identity
        
        // C_gen = 1.0 * C * Eye + 0.0 * C_gen
        slate::multiply(1.0, C, Eye, 0.0, C_gen);

        // 2. Compute SVD: C_gen = U * Sigma * VH
        slate::Matrix<double> U(n_active, n_active, tileNb, tileNb, tileRank, none, comm);
        slate::Matrix<double> VH(n_active, n_active, tileNb, tileNb, tileRank, none, comm);
        std::vector<double> Sigma(n_active);
        
        U.insertLocalTiles();
        VH.insertLocalTiles();
        
        slate::svd(C_gen, Sigma, U, VH);

        // 3. Compute condition number
        double s_max = Sigma[0];
        double s_min = Sigma[n_active-1];
        double cond_number = (s_min > 0) ? (s_max / s_min) : 1e16;

        // 4. Invert Sigma (S_inv)
        std::vector<double> Sigma_inv(n_active);
        double threshold = 1e-13 * s_max;
        for(int i=0; i<n_active; ++i) {
             Sigma_inv[i] = (Sigma[i] > threshold) ? (1.0/Sigma[i]) : 0.0;
        }

        // 5. Compute tmp = U^H * y_prime (Using transpose of U, assuming real)
        slate::Matrix<double> tmp(n_active, 1, tileNb, tile1, tileRank, none, comm);
        tmp.insertLocalTiles();
        auto UT = slate::transpose(U);
        slate::multiply(1.0, UT, y_prime, 0.0, tmp);

        // 6. Scale tmp = Sigma_inv * tmp (Row scaling)
        // slate::scale_row_col applies D_r * A * D_c. Here A=tmp. D_r = Sigma_inv.
        std::vector<double> ones(1, 1.0);
        slate::scale_row_col(slate::Equed::Row, Sigma_inv, ones, tmp);

        // 7. Compute x = V * tmp = VH^T * tmp
        slate::Matrix<double> x_sol(n_active, 1, tileNb, tile1, tileRank, none, comm);
        x_sol.insertLocalTiles();
        auto V = slate::transpose(VH);
        slate::multiply(1.0, V, tmp, 0.0, x_sol);

        // -------------------------------- GATHER & EXTRACT --------------------------------
        // Gather solution x and VH (for variance) to Rank 0
        
        std::function<int (slate::func::ij_tuple)> tileRank0 = [](slate::func::ij_tuple ij) { return 0; };
        
        slate::Matrix<double> x_loc(n_active, 1, tileNb, tile1, tileRank0, none, comm);
        x_loc.insertLocalTiles();
        slate::copy(x_sol, x_loc);

        slate::Matrix<double> VH_loc(n_active, n_active, tileNb, tileNb, tileRank0, none, comm);
        VH_loc.insertLocalTiles();
        slate::copy(VH, VH_loc);
        
        MPI_Barrier(comm);

        if (mpi_rank == 0) {
            // Extract coefficients
            for(int64_t i=0; i<n_active; ++i) {
                int64_t tile_i = i / nb;
                int64_t loc_i = i % nb;
                if(x_loc.tileIsLocal(tile_i, 0)) {
                   local_coef_active[i] = x_loc(tile_i, 0).at(loc_i, 0);
                }
            }
            
            // Extract Variance Diagonal
            // Sigma_ii = sum_k (V_ik)^2 * S_inv_k
            // V_ik is element (i,k) of V.
            // VH contains V^T. Element (k, i) of VH is V^T_ki = V_ik.
            // So Cov_ii = sum_k (VH(k, i))^2 * S_inv_k
            
            for(int64_t i=0; i<n_active; ++i) {
                double sum = 0.0;
                for(int64_t k=0; k<n_active; ++k) {
                    // Access VH_loc(k, i)
                    int64_t tile_k = k / nb;
                    int64_t local_k = k % nb;
                    int64_t tile_i = i / nb;
                    int64_t local_i = i % nb;
                    
                    double v_ik = 0.0;
                    if (VH_loc.tileIsLocal(tile_k, tile_i)) {
                        v_ik = VH_loc(tile_k, tile_i).at(local_k, local_i);
                    }
                    sum += v_ik * v_ik * Sigma_inv[k];
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


