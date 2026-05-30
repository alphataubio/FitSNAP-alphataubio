#include <slate/slate.hh>
#include <lapack.hh>
#include <mpi.h>
#include <omp.h>

#include <cstdint>
#include <cmath>
#include <vector>
#include <iomanip>
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
    int ncpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpus <= 0) ncpus = 64;
    // Open full CPU mask on calling thread so OpenMP can spawn workers anywhere
    cpu_set_t full_mask;
    CPU_ZERO(&full_mask);
    for (int i = 0; i < ncpus; i++) CPU_SET(i, &full_mask);
    sched_setaffinity(0, sizeof(full_mask), &full_mask);
#endif // __linux__
    // BLAS must be single-threaded so SLATE task parallelism isn't over-subscribed.
    // Without this, all 192 cores rush into each tile GEMM sequentially instead of
    // running 192 concurrent single-threaded tile tasks.
    setenv("BLIS_NUM_THREADS",    "1", 1);  // AOCL/BLIS
    setenv("BLIS_JC_NT",          "1", 1);
    setenv("BLIS_IC_NT",          "1", 1);
    setenv("BLIS_JR_NT",          "1", 1);
    setenv("BLIS_IR_NT",          "1", 1);
    setenv("OPENBLAS_NUM_THREADS","1", 1);  // fallback
    setenv("MKL_NUM_THREADS",     "1", 1);  // fallback

    omp_set_num_threads(num_threads);
#ifdef __linux__
    // OMP_PROC_BIND was initialized with OMP_NUM_THREADS=1, so its binding
    // table only has 1 place — all threads collapse onto core 0.
    // Fix: call sched_setaffinity *inside* the parallel region to pin each
    // thread to its own core via OS syscall, bypassing OMP_PROC_BIND entirely.
    #pragma omp parallel shared(ncpus)
    {
        int tid = omp_get_thread_num();
        cpu_set_t tmask;
        CPU_ZERO(&tmask);
        CPU_SET(tid % ncpus, &tmask);
        sched_setaffinity(0, sizeof(tmask), &tmask);
    }
#endif // __linux__
    if (debug)
    std::cerr << "SLATE_CPP: Set OpenMP threads to " << num_threads
                         << ", pinned threads 0.." << num_threads-1 << " to cores" << std::endl;
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
    
  int64_t mb = 512;
  int64_t nb = 512;
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

  std::function<int (slate::func::ij_tuple)> tileDevice = [](slate::func::ij_tuple ij) {
    return slate::HostNum;
  };

  if (mpi_rank == 0) {
    const double memory_gb_total = static_cast<double>(m*n*sizeof(double))/static_cast<double>(1024*1024*1024);
    const double memory_gb_per_node = memory_gb_total / mpi_size;
    std::cerr << "----------------------------------------------------------------" << std::endl;
    std::cerr << "  SLATE RIDGE SOLVER" << std::endl;
    std::cerr << std::endl;
    std::cerr << "    MPI: " << mpi_size << " rank(s) (one per node), ";
    std::cerr                << num_threads << " OpenMP threads/node" << std::endl;
    std::cerr << "    Rank: " << mpi_rank << " lld " << lld << std::endl;
    std::cerr << "    Matrix size: " << m << " x " << n << std::endl;
    std::cerr << std::fixed << std::setprecision(2);
    std::cerr << "    Memory: " << memory_gb_total << " GB, " << memory_gb_per_node << " GB/node" << std::endl;
    std::cerr << "    Tile size: " << mb << " x " << nb << std::endl;
    std::cerr << "    Grid: " << mt << " x " << nt << std::endl;
    std::cerr << std::endl;
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
      slate::print("A", A, opts);
      slate::print("b", b, opts);
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
      slate::print("A (solution)", A, opts);
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

// NOTE FOR CLAUDE:
// DO NOT ERASE slate_ard_update() slate_ard_update_v1()
// ONLY MODIFY slate_ard_update_v2() !!!!!!!

double slate_ard_update_v1(double*, double*, double*, double*, int64_t, int64_t, int64_t,
  double, double*, MPI_Comm, int);

double slate_ard_update_v2(double*, double*, double*, double*, int64_t, int64_t, int64_t,
  double, double*, MPI_Comm, int);

double slate_ard_update(
  double* local_aw_active, double* local_bw, double* local_sigma_diag, double* local_coef_active,
  int64_t m, int64_t n_active, int64_t lld, double alpha, double* lambda_active,
  MPI_Comm comm, int debug) {

  //return slate_ard_update_v1(local_aw_active, local_bw, local_sigma_diag, local_coef_active, m, n_active, lld, alpha, lambda_active, comm, debug);

  return slate_ard_update_v2(local_aw_active, local_bw, local_sigma_diag, local_coef_active, m, n_active, lld, alpha, lambda_active, comm, debug);

}

double slate_ard_update_v1(
  double* local_aw_active, double* local_bw, double* local_sigma_diag, double* local_coef_active,
  int64_t m, int64_t n_active, int64_t lld, double alpha, double* lambda_active, MPI_Comm comm, int debug) {

  // -------------------------------- HYBRID MPI/OPENMP --------------------------------
    
  int mpi_rank, mpi_size;
  MPI_Comm_rank(comm, &mpi_rank);
  MPI_Comm_size(comm, &mpi_size);
  int num_threads = omp_get_max_threads();

  // --------------------------------PRINT OPTS --------------------------------

  slate::Options opts = {
    {slate::Option::PrintVerbose, 4},
    {slate::Option::PrintPrecision, 1},
    {slate::Option::PrintWidth, 4}
  };

  // -------------------------------- TILE SIZE --------------------------------
  // FIXME: find optimal tile size based on cache size
    
  int64_t mb = 512;
  int64_t nb = 512;
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

  std::function<int (slate::func::ij_tuple)> tileDevice = [](slate::func::ij_tuple ij) {
    return slate::HostNum;
  };

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

    if (debug) {
      slate::print("X_active", X_active, opts);
      slate::print("y", y, opts);
    }


    // -------------------------------- FORM NORMAL EQUATIONS --------------------------------
    // C = alpha * X.T @ X + Lambda

    // We calculate C distributedly first
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

    if (debug) {
      slate::print("C", C, opts);
    }

    // C = alpha * X.T @ X + C
    auto X_active_T = transpose(X_active);
    slate::herk(alpha, X_active_T, 1.0, C);
    MPI_Barrier(comm);

    if (debug) {
      slate::print("C", C, opts);
    }

    // Compute y' = alpha * X.T @ y (Distributed)
    slate::Matrix<double> y_prime(n_active, 1, tileNb, tile1, tileRank, tileDevice, comm);
    y_prime.insertLocalTiles();
    slate::set(0.0, y_prime);
    slate::gemm(alpha, X_active_T, y, 0.0, y_prime);
    MPI_Barrier(comm);

    if (debug) {
      slate::print("y_prime", y_prime, opts);
    }

    // -------------------------------- SLATE QR SOLVE --------------------------------
    // Solve C * x = y' using QR: C = QR
    // R * x = Q^H * y'
    // x = R^-1 * Q^H * y'
    // Covariance C^-1 = R^-1 * Q^H
        
    // 1. Convert Hermitian C to General Matrix C_gen for slate::qr_factor
    // Workaround: Use slate::multiply with Identity (HEMM) instead of slate::copy
        
    slate::Matrix<double> C_gen(n_active, n_active, tileNb, tileNb, tileRank, tileDevice, comm);
    C_gen.insertLocalTiles();
        
    slate::Matrix<double> Eye(n_active, n_active, tileNb, tileNb, tileRank, tileDevice, comm);
    Eye.insertLocalTiles();
    slate::set(0.0, 1.0, Eye); // Identity
        
    // C_gen = C * I
    slate::multiply(1.0, C, Eye, 0.0, C_gen);

    // 2. Compute QR: C_gen = Q * R
    // T holds the triangular factors for the reflectors
    slate::TriangularFactors<double> T;
    slate::qr_factor(C_gen, T);

    // 3. Compute Condition Number of R (which estimates cond(C))
    // Create a Triangular view of R (Upper part of C_gen)
    auto R = slate::TriangularMatrix<double>(slate::Uplo::Upper, slate::Diag::NonUnit, C_gen);
    double R_norm = slate::norm(slate::Norm::One, R);
        
    slate::Options opts;
    // Compute reciprocal condition number estimate
    double rcond = slate::trcondest(slate::Norm::One, R, R_norm, opts);
    double cond_number = (rcond > 1e-16) ? (1.0 / rcond) : 1e16;

    // 4. Solve for x
    // Compute rhs = Q^H * y_prime
    // Apply Q^H to y_prime in-place
    slate::qr_multiply_by_q(slate::Side::Left, slate::Op::ConjTrans, C_gen, T, y_prime);

    // Solve R * x = rhs
    slate::triangular_solve(1.0, R, y_prime);
        
    // y_prime now holds the solution vector x.

    // 5. Compute Covariance Matrix (Inverse of C) for Variance Diagonal
    // C^-1 = (QR)^-1 = R^-1 Q^H
    // We compute this by solving R * X = Q^H for X.
        
    // Generate Q^H
    // Initialize Q_H as Identity, then apply Q^H to it.
    slate::Matrix<double> Q_H(n_active, n_active, tileNb, tileNb, tileRank, tileDevice, comm);
    Q_H.insertLocalTiles();
    slate::set(0.0, 1.0, Q_H); // Set to Identity
        
    slate::qr_multiply_by_q(slate::Side::Left, slate::Op::ConjTrans, C_gen, T, Q_H);
        
    // Solve R * Cov = Q^H  =>  Cov = R^-1 Q^H
    slate::triangular_solve(1.0, R, Q_H);
        
    // Q_H now holds the full covariance matrix C^-1.

    if (debug) {
      slate::print("Q_H", Q_H, opts);
    }


    // -------------------------------- GATHER & EXTRACT --------------------------------
    // Gather solution x (y_prime) and Covariance (Q_H) to Rank 0
        
    std::function<int (slate::func::ij_tuple)> tileRank0 = [](slate::func::ij_tuple ij) { return 0; };
        
    slate::Matrix<double> x_loc(n_active, 1, tileNb, tile1, tileRank0, tileDevice, comm);
    x_loc.insertLocalTiles();
    slate::copy(y_prime, x_loc);

    slate::Matrix<double> Cov_loc(n_active, n_active, tileNb, tileNb, tileRank0, tileDevice, comm);
    Cov_loc.insertLocalTiles();
    slate::copy(Q_H, Cov_loc);
        
    MPI_Barrier(comm);

    if (mpi_rank == 0) {
      // Extract coefficients from x_loc
      for(int64_t i=0; i<n_active; ++i) {
        int64_t tile_i = i / nb;
        int64_t loc_i = i % nb;
        if(x_loc.tileIsLocal(tile_i, 0))
          local_coef_active[i] = x_loc(tile_i, 0).at(loc_i, 0);
      }
            
      // Extract Variance Diagonal from Cov_loc
      for(int64_t i=0; i<n_active; ++i) {
        int64_t tile_i = i / nb;
        int64_t loc_i = i % nb;
        if(Cov_loc.tileIsLocal(tile_i, tile_i))
          local_sigma_diag[i] = Cov_loc(tile_i, tile_i).at(loc_i, loc_i);
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


double slate_ard_update_v2(double* local_aw_active, double* local_bw, double* local_sigma_diag, double* local_coef_active,
                     int64_t m, int64_t n_active, int64_t lld,
                     double alpha, double* lambda_active,
                     MPI_Comm comm, int debug) {

  // -------------------------------- HYBRID MPI/OPENMP --------------------------------
    
  int mpi_rank, mpi_size;
  MPI_Comm_rank(comm, &mpi_rank);
  MPI_Comm_size(comm, &mpi_size);
  int num_threads = omp_get_max_threads();

  // --------------------------------PRINT OPTS --------------------------------

  slate::Options opts = {
    {slate::Option::PrintVerbose, 4},
    {slate::Option::PrintPrecision, 1},
    {slate::Option::PrintWidth, 4}
  };

  // -------------------------------- TILE SIZE --------------------------------
  // FIXME: find optimal tile size based on cache size
    
  int64_t mb = 512;
  int64_t nb = 512;
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

  std::function<int (slate::func::ij_tuple)> tileDevice = [](slate::func::ij_tuple ij) {
    return slate::HostNum;
  };

  if (mpi_rank == 0 && debug) {
    std::fprintf(stderr, "\n=== slate_ard_update ===\n");
    std::fprintf(stderr, "  m=%" PRId64 ", n_active=%" PRId64 ", alpha=%.6e\n", m, n_active, alpha);
  }
    
  if (n_active == 0) return 1.0; // Perfect conditioning for empty matrix
    
  try {
    
    // -------------------------------- SLATE MATRICES --------------------------------
    // Numerically stable ARD update via AUGMENTED QR (no normal equations).
    //
    // The previous version formed C = alpha*X^T X + diag(lambda) with herk(). Forming the
    // cross product squares the conditioning (cond(C) = kappa(X)^2) AND, once alpha grows
    // large (alpha ~ 1e11 in practice here), rounds the O(1) diag(lambda) prior away
    // underneath alpha*X^T X. Both destroy the sigma_diag / gamma accuracy that ARD
    // relevance determination relies on -- which is why nothing ever pruned (N_ACTIVE
    // stayed pinned at n) and the fit collapsed to plain ridge.
    //
    // Stable formulation: scale the whole least-squares system by 1/sqrt(alpha) so the
    // regularizer sits on the same scale as the (alpha-weighted) data, then QR the
    // augmented operator directly, never forming the cross product:
    //
    //     A_aug = [        X         ]  ((m + n) x n)      rhs:  [ y ]
    //             [ diag(sqrt(l/a))  ]                           [ 0 ]
    //
    //     A_aug = Q R   =>   R^T R = X^T X + diag(lambda/alpha) = C / alpha
    //
    //   W          = (R^T R)^-1 = alpha * C^-1
    //   coef       = W * (X^T y) = alpha * C^-1 * X^T y          (the ARD MAP estimate)
    //   sigma_diag = diag(C^-1)  = diag(W) / alpha
    //
    // cond(R) ~ kappa(X) -- the SQUARE ROOT of the cond(C) the old code put in COND_NUMBER
    // (so e.g. the old 1.8e11 reads as ~4e5 now). The residual/sigma floor improves from
    // kappa(X)^2 * eps_double to kappa(X) * eps_double.
    //
    // IMPORTANT: qr_factor() overwrites its matrix in place. slate.py recomputes the SSE
    // in Python from aw AFTER this call (local_pred = aw[:, :n_active] @ coef), so aw must
    // survive. We therefore COPY the weighted design block into a SLATE-owned A_aug and
    // factor the copy; the aw/bw shared buffers are only ever READ here.
    //   Cost: one extra design-matrix-sized copy (per-rank ~ local aw) while A_aug lives.
    //   Assumes the m >> n regime this solver already documents (n <= rows-per-node), so
    //   the leading n x n of A_aug tiles into square diagonal blocks for the R view.

    // ---- X_alias / y: views onto the FitSNAP shared buffers (READ-ONLY here) ----
    slate::Matrix<double> X_alias(m, n_active, tileMb, tileNb, tileRank, tileDevice, comm);
    for (int64_t i = 0; i < mt; ++i)
      for (int64_t j = 0; j < nt; ++j)
        if (X_alias.tileIsLocal(i, j)) {
          const int64_t offset = (i % mt_node) * mb + j * nb * lld;
          X_alias.tileInsert(i, j, local_aw_active + offset, lld);
        }

    slate::Matrix<double> y(m, 1, tileMb, tile1, tileRank, tileDevice, comm);
    for (int64_t i = 0; i < mt; ++i)
      if (y.tileIsLocal(i, 0)) {
        const int64_t offset = (i % mt_node) * mb;
        y.tileInsert(i, 0, local_bw + offset, lld);
      }

    MPI_Barrier(comm);

    if (debug) {
      slate::print("X_alias", X_alias, opts);
      slate::print("y", y, opts);
    }

    // ---- g = X^T y   (read aw/bw; computed before anything is overwritten) ----
    auto X_alias_T = transpose(X_alias);
    slate::Matrix<double> g(n_active, 1, tileNb, tile1, tileRank, tileDevice, comm);
    g.insertLocalTiles();
    slate::set(0.0, g);
    slate::gemm(1.0, X_alias_T, y, 0.0, g);
    MPI_Barrier(comm);

    // ---- Build SLATE-OWNED A_aug = [ X ; diag(sqrt(lambda/alpha)) ] ----
    // Row tiling: data rows reuse the node-local block-row layout (so the copy below is a
    // local, communication-free tile copy); the n regularizer rows follow, tiled by nb.
    std::function<int64_t (int64_t)> tileMb_aug =
      [mt, mt_node, tile_row_last, tile_row_remainder, mb, nt, n_active, nb](int64_t i) -> int64_t {
        if (i < mt) {
          if (i % mt_node == tile_row_last) return tile_row_remainder;
          else return mb;
        }
        int64_t jt = i - mt;
        if (jt == nt - 1) return n_active - (nt - 1) * nb;
        return nb;
      };
    // Data rows keep their original rank; the (small) regularizer rows go to rank 0.
    std::function<int (slate::func::ij_tuple)> tileRank_aug =
      [mt, mt_node](slate::func::ij_tuple ij) {
        int64_t i = std::get<0>(ij);
        if (i < mt) return (int)(i / mt_node);
        return 0;
      };

    slate::Matrix<double> A_aug(m + n_active, n_active,
                                tileMb_aug, tileNb, tileRank_aug, tileDevice, comm);
    A_aug.insertLocalTiles();   // SLATE owns ALL tiles (data + regularizer)

    // Copy the weighted design block into the owned data block (matching node-local
    // tiling => local copy, no redistribution). aw itself is left untouched.
    auto A_data = A_aug.slice(0, m - 1, 0, n_active - 1);
    slate::copy(X_alias, A_data);

    // Regularizer block: zero, then sqrt(lambda/alpha) on its diagonal.
    auto A_reg = A_aug.slice(m, m + n_active - 1, 0, n_active - 1);
    slate::set(0.0, A_reg);
    const double inv_sqrt_alpha = 1.0 / std::sqrt(alpha);
    for (int64_t idx = 0; idx < n_active; ++idx) {
      int64_t jt  = idx / nb;
      int64_t loc = idx % nb;
      if (A_aug.tileIsLocal(mt + jt, jt)) {
        auto tile = A_aug(mt + jt, jt);
        tile.at(loc, loc) = std::sqrt(lambda_active[idx]) * inv_sqrt_alpha;
      }
    }
    MPI_Barrier(comm);

    if (debug)
      slate::print("A_aug", A_aug, opts);

    // ---- A_aug = Q R   (in place on the COPY; aw stays intact) ----
    slate::TriangularFactors<double> T;
    slate::qr_factor(A_aug, T);
    MPI_Barrier(comm);

    // R = leading n x n upper-triangular factor. In the m >> n regime the leading n rows
    // are full 512-row tiles, so slice() yields square diagonal tiles (mb == nb) and the
    // TriangularMatrix view / triangular_solve are valid.
    auto R_sq = A_aug.slice(0, n_active - 1, 0, n_active - 1);
    auto R = slate::TriangularMatrix<double>(slate::Uplo::Upper, slate::Diag::NonUnit, R_sq);

    // ---- Condition number of the REGULARIZED operator (~kappa(X), not kappa(X)^2) ----
    // NOTE: this is the sqrt() of the cond(C) the old code reported; slate.py's
    // cond_number > 1e15 guard now trips only on a genuinely singular operator.
    slate::Options cond_opts;
    double R_norm = slate::norm(slate::Norm::One, R);
    double rcond  = slate::trcondest(slate::Norm::One, R, R_norm, cond_opts);
    double cond_number = (rcond > 1e-300) ? (1.0 / rcond) : 1e300;

    // ---- W = (R^T R)^-1 = R^-1 R^-T  via two triangular solves on the identity ----
    slate::Matrix<double> W(n_active, n_active, tileNb, tileNb, tileRank, tileDevice, comm);
    W.insertLocalTiles();
    slate::set(0.0, 1.0, W);                  // W = I
    auto R_T = transpose(R);
    slate::triangular_solve(1.0, R_T, W);     // W = R^-T
    slate::triangular_solve(1.0, R, W);       // W = R^-1 R^-T = (R^T R)^-1 = alpha * C^-1
    MPI_Barrier(comm);

    // ---- coef = W g = alpha * C^-1 * X^T y ----
    slate::Matrix<double> coef_vec(n_active, 1, tileNb, tile1, tileRank, tileDevice, comm);
    coef_vec.insertLocalTiles();
    slate::set(0.0, coef_vec);
    slate::gemm(1.0, W, g, 0.0, coef_vec);
    MPI_Barrier(comm);

    if (debug) {
      slate::print("W", W, opts);
      slate::print("coef", coef_vec, opts);
    }

    // -------------------------------- GATHER & EXTRACT --------------------------------
    std::function<int (slate::func::ij_tuple)> tileRank0 =
      [](slate::func::ij_tuple) { return 0; };

    slate::Matrix<double> coef_loc(n_active, 1, tileNb, tile1, tileRank0, tileDevice, comm);
    coef_loc.insertLocalTiles();
    slate::copy(coef_vec, coef_loc);

    slate::Matrix<double> W_loc(n_active, n_active, tileNb, tileNb, tileRank0, tileDevice, comm);
    W_loc.insertLocalTiles();
    slate::copy(W, W_loc);

    MPI_Barrier(comm);

    if (mpi_rank == 0) {
      for (int64_t i = 0; i < n_active; ++i) {
        int64_t tile_i = i / nb;
        int64_t loc_i  = i % nb;
        if (coef_loc.tileIsLocal(tile_i, 0))
          local_coef_active[i] = coef_loc(tile_i, 0).at(loc_i, 0);
      }

      // sigma_diag = diag(C^-1) = diag(W) / alpha
      for (int64_t i = 0; i < n_active; ++i) {
        int64_t tile_i = i / nb;
        int64_t loc_i  = i % nb;
        if (W_loc.tileIsLocal(tile_i, tile_i))
          local_sigma_diag[i] = W_loc(tile_i, tile_i).at(loc_i, loc_i) / alpha;
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


