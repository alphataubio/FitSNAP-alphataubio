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
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);
    int num_threads = omp_get_max_threads();
    
    // Find optimal tile size
    int64_t nb = 256;
    int64_t nt = ceil_div64(n, nb);
    
    int64_t mb = nb;
    int64_t m_node = m / mpi_size;
    int64_t mt_node = ceil_div64(m_node, mb);
    int64_t mt = mt_node * mpi_size;
    
    if (mpi_rank >= 0) {
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
            //int64_t j = std::get<1>(ij);
            return i / mt_node;
        };

        // FIXME: GPU device tiles not implemented yet (placeholder)
        // need to sync to device from fitsnap python shared array on node-local ram
        std::function<int (slate::func::ij_tuple)> tileDevice = [](slate::func::ij_tuple) { return 0; };
        
        // Create SLATE matrices with tile lambdas
        slate::Matrix<double> A(m, n, tileMb, tileNb, tileRank, tileDevice, comm);
        slate::Matrix<double> b(m, 1, tileMb,  tile1, tileRank, tileDevice, comm);
        
        // Insert A matrix tiles
        for (int64_t i = 0; i < mt; ++i)
            for (int64_t j = 0; j < nt; ++j)
                if (A.tileIsLocal(i, j)) {
                    const int64_t offset = (i % mt_node) * mb + j * nb * lld;
                    A.tileInsert(i, j, local_aw + offset, lld);
                }

        // Insert b vector tiles
        for (int64_t i = 0; i < mt; ++i)
            if (b.tileIsLocal(i, 0)) {
                const int64_t offset = (i % mt_node) * mb;
                b.tileInsert(i, 0, local_bw + offset, lld);
            }
        
        // Debug output
        if (debug) {
            slate::Options opts = {
              {slate::Option::PrintVerbose, 4},
              {slate::Option::PrintPrecision, 3},
              {slate::Option::PrintWidth, 7}
            };
            //slate::print("A", A, opts);
            //slate::print("b", b, opts);
        }
        
        // Least squares solve
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
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);
    int num_threads = omp_get_max_threads();
    
    // Find optimal tile size
    int64_t mb = lld;
    int64_t nb = n_active;
    int64_t nt = 1;
    int64_t mt = mpi_size;
    
    if (mpi_rank == 0 && debug) {
        std::fprintf(stderr, "\n=== slate_ard_update ===\n");
        std::fprintf(stderr, "  m=%" PRId64 ", n_active=%" PRId64 ", alpha=%.6e\n", m, n_active, alpha);
    }
    
    if (n_active == 0) return 1.0; // Perfect conditioning for empty matrix
    
    // Define tile lambdas
    std::function<int64_t (int64_t)> tileMb = [mb](int64_t) { return mb; };
    std::function<int64_t (int64_t)> tileNb = [nb](int64_t) { return nb; };
    std::function<int64_t (int64_t)> tile1  = [  ](int64_t) { return  1; };
    
    
    std::function<int (slate::func::ij_tuple)> tileDevice = [](slate::func::ij_tuple) { return 0; };
    
    std::function<int (slate::func::ij_tuple)> tileRank = [](slate::func::ij_tuple ij) {
        int64_t i = std::get<0>(ij);
        //int64_t j = std::get<1>(ij);
        return i;
    };
    
    if (mpi_rank == 0 && debug) {
        std::fprintf(stderr, "*** MPI: %d rank(s) (one rank per node), %d OpenMP threads/node\n",
                    mpi_size, num_threads );
        std::fprintf(stderr, "*** Matrix %" PRId64 " x %" PRId64 " Tile %" PRId64 " x %" PRId64 " Grid %" PRId64 " x %" PRId64 "\n",
                    m, n_active, mb, nb, mt, nt);
        std::fflush(stderr);
    }
        
    MPI_Barrier(comm);
    
    try {
        // Create matrices - now working with pre-filtered active features only
        
        slate::Matrix<double> X_active(m, n_active, tileMb, tileNb, tileRank, tileDevice, comm);
        slate::Matrix<double>        y(m,        1, tileMb,  tile1, tileRank, tileDevice, comm);
        
        // Insert X_active and y tiles - data already filtered to active columns
        
        for (int64_t i = 0; i < mt; ++i)
            for (int64_t j = 0; j < nt; ++j)
                if (X_active.tileIsLocal(i, j))
                    X_active.tileInsert(i, j, local_aw_active, lld);
        
        for (int64_t i = 0; i < mt; ++i)
            if (y.tileIsLocal(i, 0))
                y.tileInsert(i, 0, local_bw, lld);
        
        MPI_Barrier(comm);
        
        slate::HermitianMatrix<double> C(slate::Uplo::Lower, n_active, tileNb, tileRank, tileDevice, comm);
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

        // Compute Cholesky factorization
        slate::potrf(C);
        MPI_Barrier(comm);
        
        // Compute condition number from Cholesky diagonal: cond(A) ≈ (max(L_ii) / min(L_ii))^2
        // Extract diagonal of Cholesky factor L
        std::vector<double> L_diag(n_active, 0.0);
        for (int64_t i = 0; i < n_active; ++i) {
            int64_t tile_i = i / nb;
            int64_t local_i = i % nb;
            
            if (C.tileIsLocal(tile_i, tile_i)) {
                L_diag[i] = C(tile_i, tile_i).at(local_i, local_i);
            }
        }
        
        // Reduce to get full diagonal on all ranks
        MPI_Allreduce(MPI_IN_PLACE, L_diag.data(), n_active, MPI_DOUBLE, MPI_SUM, comm);
        
        // Find max and min of diagonal (absolute values)
        double max_diag = 0.0;
        double min_diag = std::numeric_limits<double>::infinity();
        for (int64_t i = 0; i < n_active; ++i) {
            double abs_val = std::fabs(L_diag[i]);
            if (abs_val > max_diag) max_diag = abs_val;
            if (abs_val < min_diag) min_diag = abs_val;
        }
        
        // Condition number estimate: cond(A) ≈ (max/min)^2
        double cond_number = (min_diag > 0.0) ? (max_diag / min_diag) * (max_diag / min_diag) : std::numeric_limits<double>::infinity();
        
        if (mpi_rank == 0 && debug) {
            std::fprintf(stderr, "*** Cholesky diagonal: max=%.6e, min=%.6e\n", max_diag, min_diag);
            std::fprintf(stderr, "*** Condition number estimate: %.6e\n", cond_number);
            std::fflush(stderr);
        }
        
        MPI_Barrier(comm);
        slate::potri(C);
        MPI_Barrier(comm);

        // Compute X.T @ y
        slate::Matrix<double> XTy(n_active, 1, tileNb, tile1, tileRank, tileDevice, comm);
        XTy.insertLocalTiles();
        slate::set(0.0, XTy);
        slate::gemm(1.0, X_active_T, y, 0.0, XTy);
        MPI_Barrier(comm);
        
        // Compute coef = alpha * C @ XTy
        slate::Matrix<double> coef_active(n_active, 1, tileNb, tile1, tileRank, tileDevice, comm);
        coef_active.insertLocalTiles();
        slate::set(0.0, coef_active);
        slate::hemm(slate::Side::Left, alpha, C, XTy, 0.0, coef_active);
        MPI_Barrier(comm);
        
        // MODIFIED: Extract only diagonal of sigma instead of full matrix
        // This saves massive memory (n_active instead of n_active^2 doubles)
        
        for (int64_t i = 0; i < n_active; ++i) {
            int64_t tile_i = i / nb;
            int64_t local_i = i % nb;
            
            // Only extract diagonal elements
            if (C.tileIsLocal(tile_i, tile_i)) local_sigma_diag[i] = C(tile_i, tile_i).at(local_i, local_i);
            else local_sigma_diag[i] = 0.0;
            
            if (coef_active.tileIsLocal(tile_i, 0)) local_coef_active[i] = coef_active(tile_i, 0).at(local_i, 0);
            else local_coef_active[i] = 0.0;

        }
        
        MPI_Barrier(comm);
        
        // Reduce diagonal to all ranks (allreduce since we need it everywhere)
        MPI_Allreduce(MPI_IN_PLACE, local_sigma_diag, n_active, MPI_DOUBLE, MPI_SUM, comm);
        
        // Reduce coefficients to all ranks
        MPI_Allreduce(MPI_IN_PLACE, local_coef_active, n_active, MPI_DOUBLE, MPI_SUM, comm);
        
        MPI_Barrier(comm);
        
        if (mpi_rank == 0 && debug) {
            std::fprintf(stderr, "=== slate_ard_update COMPLETE ===\n");
            std::fflush(stderr);
        }
        
        return cond_number;
        
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << mpi_rank << "] SLATE ARD error: " << e.what() << std::endl;
        MPI_Abort(comm, 1);
        return std::numeric_limits<double>::infinity(); // Return inf on error
    }
}


/*
{
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);
    
    if (n_active == 0) return 1.0;

    // 1. Data Distribution
    std::vector<int64_t> all_llds(mpi_size);
    int64_t lld64 = lld;
    MPI_Allgather(&lld64, 1, MPI_INT64_T, all_llds.data(), 1, MPI_INT64_T, comm);
    std::vector<int64_t> row_offsets(mpi_size + 1, 0);
    for(int i = 0; i < mpi_size; ++i) row_offsets[i+1] = row_offsets[i] + all_llds[i];
    int64_t my_offset = row_offsets[mpi_rank];

    // 2. Tile Search
    int64_t mb, nb, nt;
    int64_t nt_start = 1;
    for (nt = nt_start; nt >= 1; --nt) {
        int64_t size = ceil_div64(n_active, nt);
        mb = nb = size;
        if (size*size*sizeof(double) > 16*1024*1024) break; 
    }
    if (nt == 0) nt = 1;
    
    int64_t mt = ceil_div64(m, mb);

    // 3. Lambdas
    std::function<int64_t (int64_t)> tileNb = [nb](int64_t) { return nb; };
    std::function<int64_t (int64_t)> tile1  = [](int64_t) { return 1; };
    std::function<int64_t (int64_t)> tileMb = [m, mb](int64_t i) { 
        return (i * mb + mb > m) ? (m - i*mb) : mb; 
    };
    
    std::function<int (slate::func::ij_tuple)> tileRank = [row_offsets, mb, mpi_size](slate::func::ij_tuple ij) {
        int64_t global_row = std::get<0>(ij) * mb;
        auto it = std::upper_bound(row_offsets.begin(), row_offsets.end(), global_row);
        int rank = std::distance(row_offsets.begin(), it) - 1;
        if (rank < 0) rank = 0;
        if (rank >= mpi_size) rank = mpi_size - 1;
        return rank;
    };
    
    std::function<int (slate::func::ij_tuple)> tileDevice = [](slate::func::ij_tuple) { return 0; };
    
    try {
        slate::Matrix<double> X_active(m, n_active, tileMb, tileNb, tileRank, tileDevice, comm);
        slate::Matrix<double> y(m, 1, tileMb, tile1, tileRank, tileDevice, comm);
        
        X_active.insertLocalTiles();
        y.insertLocalTiles();
        
        // Copy Data
        #pragma omp parallel for collapse(2)
        for (int64_t i = 0; i < mt; ++i) {
            for (int64_t j = 0; j < nt; ++j) {
                if (X_active.tileIsLocal(i, j)) {
                    auto tile = X_active(i, j);
                    double* t_ptr = tile.data();
                    int64_t stride = tile.stride();
                    int64_t h = tile.mb();
                    int64_t w = tile.nb();
                    
                    int64_t global_row = i * mb;
                    int64_t local_row = global_row - my_offset;
                    int64_t col_off = j * nb;
                    
                    if (local_row >= 0 && local_row < lld) {
                        for(int64_t jj=0; jj<w; ++jj) {
                            for(int64_t ii=0; ii<h; ++ii) {
                                t_ptr[ii + jj*stride] = local_aw_active[(col_off+jj)*lld + (local_row+ii)];
                            }
                        }
                    }
                }
            }
        }
        
        #pragma omp parallel for
        for (int64_t i = 0; i < mt; ++i) {
            if (y.tileIsLocal(i, 0)) {
                auto tile = y(i, 0);
                double* t_ptr = tile.data();
                int64_t h = tile.mb();
                int64_t local_row = (i * mb) - my_offset;
                if (local_row >= 0 && local_row < lld) {
                    for(int64_t ii=0; ii<h; ++ii) t_ptr[ii] = local_bw[local_row+ii];
                }
            }
        }

        // --- Math Kernels ---
        slate::HermitianMatrix<double> C(slate::Uplo::Lower, n_active, tileNb, tileRank, tileDevice, comm);
        C.insertLocalTiles();
        slate::set(0.0, C);
        
        for (int64_t idx = 0; idx < n_active; ++idx) {
            int64_t tile_i = idx / nb;
            int64_t local_i = idx % nb;
            if (C.tileIsLocal(tile_i, tile_i)) {
                C(tile_i, tile_i).at(local_i, local_i) = lambda_active[idx];
            }
        }

        auto X_active_T = transpose(X_active);
        slate::herk(alpha, X_active_T, 1.0, C);
        slate::potrf(C);
        
        std::vector<double> L_diag(n_active, 0.0);
        for (int64_t i = 0; i < n_active; ++i) {
            int64_t tile_i = i / nb;
            int64_t local_i = i % nb;
            if (C.tileIsLocal(tile_i, tile_i)) L_diag[i] = C(tile_i, tile_i).at(local_i, local_i);
        }
        MPI_Allreduce(MPI_IN_PLACE, L_diag.data(), n_active, MPI_DOUBLE, MPI_SUM, comm);
        
        double max_d = 0.0, min_d = std::numeric_limits<double>::infinity();
        for(double v : L_diag) {
            if(std::abs(v) > max_d) max_d = std::abs(v);
            if(std::abs(v) < min_d && v != 0.0) min_d = std::abs(v);
        }
        double cond_number = (min_d > 0) ? std::pow(max_d/min_d, 2) : std::numeric_limits<double>::infinity();

        slate::potri(C);

        slate::Matrix<double> XTy(n_active, 1, tileNb, tile1, tileRank, tileDevice, comm);
        XTy.insertLocalTiles();
        slate::set(0.0, XTy);
        slate::gemm(1.0, X_active_T, y, 0.0, XTy);
        
        slate::Matrix<double> coef_active(n_active, 1, tileNb, tile1, tileRank, tileDevice, comm);
        coef_active.insertLocalTiles();
        slate::set(0.0, coef_active);
        slate::hemm(slate::Side::Left, alpha, C, XTy, 0.0, coef_active);
        
        // Extract
        for (int64_t i = 0; i < n_active; ++i) {
            int64_t tile_i = i / nb;
            int64_t local_i = i % nb;
            
            if (C.tileIsLocal(tile_i, tile_i)) local_sigma_diag[i] = C(tile_i, tile_i).at(local_i, local_i);
            if (coef_active.tileIsLocal(tile_i, 0)) local_coef_active[i] = coef_active(tile_i, 0).at(local_i, 0);
        }
        
        MPI_Allreduce(MPI_IN_PLACE, local_sigma_diag, n_active, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(MPI_IN_PLACE, local_coef_active, n_active, MPI_DOUBLE, MPI_SUM, comm);
        
        return cond_number;

    } catch (const std::exception& e) {
        std::cerr << "SLATE ARD error: " << e.what() << std::endl;
        MPI_Abort(comm, 1);
        return 0.0;
    }
}

*/



} // extern "C"
