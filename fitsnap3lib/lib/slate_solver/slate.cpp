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

    // 1. DATA DISTRIBUTION MAP
    // We assume data is distributed in contiguous blocks.
    // To be perfectly robust to minor LLD variations or remainders, we gather LLDs.
    std::vector<int64_t> all_llds(mpi_size);
    int64_t lld64 = lld;
    MPI_Allgather(&lld64, 1, MPI_INT64_T, all_llds.data(), 1, MPI_INT64_T, comm);

    // Create prefix sum to map global row indices to ranks
    std::vector<int64_t> row_offsets(mpi_size + 1, 0);
    for(int i = 0; i < mpi_size; ++i) {
        row_offsets[i+1] = row_offsets[i] + all_llds[i];
    }
    
    // Global Row Offset for this rank
    int64_t my_offset = row_offsets[mpi_rank];

    // 2. TILE SIZE SEARCH
    int64_t nb, nt;
    int64_t mb = 1; 
    int64_t nt_start = 1;

    for (nt = nt_start; nt >= 1; --nt) {
        int64_t size = ceil_div64(n, nt);
        mb = nb = size;
        if (size*size*sizeof(double) > 16*1024*1024) break; 
    }
    if (nt == 0) nt = 1;
    
    int64_t mt = ceil_div64(m, mb);

    if (mpi_rank == 0 && debug) {
        std::fprintf(stderr, "SLATE: Distributed %d nodes. Global M=%" PRId64 " x N=%" PRId64 ". Tile %" PRId64 "\n", 
                     mpi_size, m, n, nb);
    }

    try {
        // --- 3. TILING LAMBDAS ---
        
        std::function<int64_t (int64_t)> tileNb = [nb](int64_t) { return nb; };
        std::function<int64_t (int64_t)> tile1  = [](int64_t) { return 1; };
        
        // Tile Height
        std::function<int64_t (int64_t)> tileMb = [m, mb](int64_t i) {
            return (i * mb + mb > m) ? (m - i * mb) : mb;
        };

        // CUSTOM RANK MAP: Matches FitSNAP 1D Block Distribution
        // "Which rank owns global tile i (starting at global row i*mb)?"
        std::function<int (slate::func::ij_tuple)> tileRank = [row_offsets, mb, mpi_size](slate::func::ij_tuple ij) {
            int64_t global_row_start = std::get<0>(ij) * mb;
            
            // Binary search to find which rank owns this row index
            auto it = std::upper_bound(row_offsets.begin(), row_offsets.end(), global_row_start);
            int rank = std::distance(row_offsets.begin(), it) - 1;
            
            if (rank < 0) rank = 0;
            if (rank >= mpi_size) rank = mpi_size - 1;
            return rank;
        };
        
        std::function<int (slate::func::ij_tuple)> tileDevice = [](slate::func::ij_tuple) { return 0; };

        // --- 4. MATRIX CREATION ---
        slate::Matrix<double> A(m, n, tileMb, tileNb, tileRank, tileDevice, comm);
        slate::Matrix<double> b(m, 1, tileMb, tile1,  tileRank, tileDevice, comm);
        
        A.insertLocalTiles();
        b.insertLocalTiles();

        // --- 5. ZERO-COPY DATA INSERTION ---
        #pragma omp parallel for collapse(2)
        for (int64_t i = 0; i < mt; ++i) {
            for (int64_t j = 0; j < nt; ++j) {
                if (A.tileIsLocal(i, j)) {
                    auto tile = A(i, j);
                    double* t_ptr = tile.data();
                    int64_t stride = tile.stride();
                    int64_t h = tile.mb();
                    int64_t w = tile.nb();
                    
                    int64_t global_row = i * mb;
                    int64_t global_col = j * nb;
                    
                    // Local coords
                    int64_t local_row = global_row - my_offset;

                    if (local_row >= 0 && local_row < lld) {
                        for (int64_t jj = 0; jj < w; ++jj) {
                            for (int64_t ii = 0; ii < h; ++ii) {
                                // local_aw is Col-Major
                                int64_t src = (global_col + jj) * lld + (local_row + ii);
                                t_ptr[ii + jj * stride] = local_aw[src];
                            }
                        }
                    }
                }
            }
        }
            
        // Copy b vector
        #pragma omp parallel for
        for (int64_t i = 0; i < mt; ++i) {
             if (b.tileIsLocal(i, 0)) {
                 auto tile = b(i, 0);
                 double* t_ptr = tile.data();
                 int64_t h = tile.mb();
                 int64_t local_row = (i * mb) - my_offset;
                 
                 if (local_row >= 0 && local_row < lld) {
                     for(int64_t ii=0; ii<h; ++ii) {
                         t_ptr[ii] = local_bw[local_row + ii];
                     }
                 }
             }
        }
        
        // --- 6. SOLVE ---
        slate::least_squares_solve(A, b);
        
        // --- 7. EXTRACT SOLUTION ---
        // Solution X is in the first N rows of b.
        // It will be distributed across ranks based on tileRank.
        // We copy out any pieces we own. 
        // Note: For N small, X often fits entirely on Rank 0's tiles.
        
        for (int64_t i = 0; i < mt; ++i) {
             if (b.tileIsLocal(i, 0)) {
                 auto tile = b(i, 0);
                 double* t_ptr = tile.data();
                 int64_t h = tile.mb();
                 int64_t global_row = i * mb;
                 int64_t local_row = global_row - my_offset;

                 for(int64_t ii=0; ii<h; ++ii) {
                     if (global_row + ii < n && local_row + ii < lld) {
                         local_bw[local_row + ii] = t_ptr[ii];
                     }
                 }
             }
        }

    } catch (const std::exception& e) {
        std::cerr << "SLATE error: " << e.what() << std::endl;
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

} // extern "C"
