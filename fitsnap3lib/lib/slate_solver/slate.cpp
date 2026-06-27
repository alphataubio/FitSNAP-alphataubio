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
#include <limits>

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

extern "C" {

using slate::func::ij_tuple;

constexpr int64_t ceil_div64(int64_t a, int64_t b) { return (a + b - 1) / b; }

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

double slate_ard_update(double* local_aw_active, double* local_bw, double* local_sigma_diag, double* local_coef_active,
                     double* local_sse,
                     int64_t m, int64_t n_active, int64_t lld,
                     int64_t row_offset, int64_t m_local,
                     double alpha, double* lambda_active,
                     MPI_Comm comm, int debug) {

  // row_offset : this rank's first row WITHIN the node-shared aw buffer (= sub_a_indices start).
  // m_local    : design rows this rank owns in that buffer (= a_end_idx - a_start_idx + 1).
  // lld        : leading dimension / column stride of the shared buffer = aw.shape[0] (node rows).
  //
  // These decouple "this rank's row count" (m_local) from "the buffer column stride" (lld), which
  // the old one-rank-per-node code conflated. With many ranks per node sharing one buffer, every
  // rank gets the SAME base pointer, so each tile offset must add row_offset to reach its slice.

  // -------------------------------- HYBRID MPI/OPENMP --------------------------------
    
  int mpi_rank, mpi_size;
  MPI_Comm_rank(comm, &mpi_rank);
  MPI_Comm_size(comm, &mpi_size);
  int num_threads = omp_get_max_threads();

  // DIAGNOSTIC: SLATE requires MPI_THREAD_MULTIPLE (its OpenMP threads make MPI calls).
  // If the provided level is lower, threaded qr_factor races -> garbage/NaN in R.
  {
    int provided = 0;
    MPI_Query_thread(&provided);
    if (mpi_rank == 0 && (provided < MPI_THREAD_MULTIPLE || debug))
      std::fprintf(stderr,
        "*** SLATE ARD: MPI thread level provided=%d (need MPI_THREAD_MULTIPLE=%d), omp_threads=%d, ranks=%d\n",
        provided, MPI_THREAD_MULTIPLE, num_threads, mpi_size);
  }

  // --------------------------------PRINT OPTS --------------------------------

  slate::Options opts = {
    {slate::Option::PrintVerbose, 4},
    {slate::Option::PrintPrecision, 1},
    {slate::Option::PrintWidth, 4}
  };

  // -------------------------------- TILE SIZE --------------------------------
  // FIXME: find optimal tile size based on cache size
    
  int64_t mb = 256;
  int64_t nb = 256;
  int64_t nt = ceil_div64(n_active, nb);

  // -------------------------------- GLOBAL BLOCK-ROW MAP --------------------------------
  // Build a tile-row -> (owner rank, tile height) map that EVERY rank agrees on, derived from
  // each rank's true row count. The previous code set mt = mt_node (the "* mpi_size" was
  // commented out), so tileRank = i/mt_node returned 0 for every tile -- the ENTIRE matrix
  // landed on rank 0, which is exactly why only rank 0 was busy. It also assumed
  // m/mpi_size == lld (one rank per node) and offset (i%mt_node)*mb measured from buffer row 0;
  // both are false once 24 ranks per node share one buffer.
  //
  // Each rank owns ceil(m_local/mb) consecutive tile-rows. We Allgather m_local so the
  // tileMb/tileRank closures are byte-identical on every rank (SLATE requires this), and it
  // tolerates non-uniform or zero per-rank row counts.
  std::vector<int64_t> rows_per_rank(mpi_size);
  MPI_Allgather(&m_local, 1, MPI_INT64_T, rows_per_rank.data(), 1, MPI_INT64_T, comm);

  std::vector<int64_t> blk_first_tile(mpi_size);
  int64_t mt = 0, m_total = 0;
  //if (mpi_rank == 0) fprintf(stderr, "*** rows_per_rank");

  for (int r = 0; r < mpi_size; ++r) {
    blk_first_tile[r] = mt;
    mt      += ceil_div64(rows_per_rank[r], mb);
    m_total += rows_per_rank[r];
    //if (mpi_rank == 0) fprintf(stderr, " %lli", rows_per_rank[r]);
  }
  //if (mpi_rank == 0) fprintf(stderr, "\n");

  // Flat lookups (mt ~ m/mb, small): global tile-row -> owner rank, and its height.
  std::vector<int>     tile_owner(mt);
  std::vector<int64_t> tile_height(mt);
  for (int r = 0; r < mpi_size; ++r) {
    int64_t mtr = ceil_div64(rows_per_rank[r], mb);
    for (int64_t il = 0; il < mtr; ++il) {
      int64_t i = blk_first_tile[r] + il;
      tile_owner[i]  = r;
      tile_height[i] = (il == mtr - 1) ? (rows_per_rank[r] - (mtr - 1) * mb) : mb;
    }
  }
  const int64_t my_first_tile = blk_first_tile[mpi_rank];  // global index of this rank's first tile-row

  if (debug && mpi_rank == 0)
    std::fprintf(stderr, "*** SLATE ARD grid: %d rank(s), %" PRId64 " data tile-rows, n_active=%"
                 PRId64 " (nt=%" PRId64 ")\n", mpi_size, mt, n_active, nt);

  std::function<int64_t (int64_t)> tile1 = [](int64_t) { return 1; };

  std::function<int64_t (int64_t)> tileMb =
    [tile_height](int64_t i) { return tile_height[i]; };

  int64_t tile_col_last = nt - 1;
  int64_t tile_col_remainder = n_active - (nt-1)*nb;
  std::function<int64_t (int64_t)> tileNb = [tile_col_last, tile_col_remainder, nb](int64_t j) {
    if (j == tile_col_last) return tile_col_remainder;
    else return nb;
  };

  std::function<int (slate::func::ij_tuple)> tileRank =
    [tile_owner](slate::func::ij_tuple ij) { return tile_owner[std::get<0>(ij)]; };

  std::function<int (slate::func::ij_tuple)> tileDevice = [](slate::func::ij_tuple ij) {
    return slate::HostNum;
  };

  if (mpi_rank == 0 && debug) {
    std::fprintf(stderr, "\n=== slate_ard_update ===\n");
    std::fprintf(stderr, "  m=%" PRId64 ", n_active=%" PRId64 ", alpha=%.6e\n", m, n_active, alpha);
  }
    
  if (n_active == 0) return 1.0; // Perfect conditioning for empty matrix
    
  try {

    // R-view guard: the leading n x n factor R = A_aug.slice(0,n-1,0,n-1) is only well-formed
    // when the first n_active rows are uniformly mb-tiled DATA rows -- i.e. they fit entirely in
    // rank 0's block, n_active <= rows_per_rank[0]. With one rank per node this was the whole
    // node; with 24 ranks per node rank 0's block is ~24x smaller, so this guard is MUCH tighter
    // now and can trip for large active sets. If it does, reduce ranks-per-node for the solve, or
    // switch to an R factor that may span ranks. Fail loudly rather than return a wrong answer.
    if (n_active > rows_per_rank[0]) {
      if (mpi_rank == 0)
        std::fprintf(stderr,
          "slate_ard_update: n_active=%" PRId64 " > rows-on-rank-0=%" PRId64
          "; augmented-QR R view spans a rank boundary (use fewer ranks/node for the solve).\n",
          n_active, rows_per_rank[0]);
      MPI_Abort(comm, 2);
    }

    // ------------------------ AUGMENTED-QR ARD UPDATE (no aw copy; SSE in C++) ------------------------
    // Stable ARD WITHOUT normal equations, WITHOUT copying the m x n design matrix, and with
    // the data-residual SSE computed here (slate.py no longer touches aw after this call).
    //
    // Scale the least-squares system by 1/sqrt(alpha) so diag(lambda) is not rounded away
    // under alpha*X^T X, and stack the regularizer as extra rows:
    //
    //     A_aug = [        X         ] ((m+n) x n)     b_aug = [ y ]      (y := bw)
    //             [ diag(sqrt(l/a))  ]                         [ 0 ]
    //
    //     A_aug = Q R   =>   R^T R = X^T X + diag(l/a) = C / alpha
    //
    // The least-squares minimizer of || b_aug - A_aug c ||^2 is exactly the ARD MAP:
    //     coef = (X^T X + diag(l/a))^-1 X^T y = alpha * C^-1 * X^T y,
    // obtained as coef = R^-1 (Q^T b_aug)[0:n]. Its residual decomposes as
    //     || b_aug - A_aug coef ||^2 = ||y - X coef||^2 + (1/alpha) sum_i l_i coef_i^2
    //                                =      SSE          +        penalty
    // and the QR delivers || b_aug - A_aug coef ||^2 = ||d2||^2, where [d1; d2] = Q^T b_aug
    // and d2 is the trailing m entries. Hence SSE = ||d2||^2 - penalty, computed below and
    // returned via local_sse. (penalty ~ gamma_sum/alpha is <= SSE in practice, so this is
    // not a cancellation-prone subtraction.)
    //
    //   sigma_diag = diag(C^-1) = diag( (R^T R)^-1 ) / alpha       (W := (R^T R)^-1)
    //
    // aw is ALIASED into A_aug's data block and OVERWRITTEN in place by qr_factor(). That is
    // safe: slate.py repacks aw from the unweighted design matrix every iteration and no
    // longer reads it post-call. bw is NOT aliased into b_aug (Q^T overwrites b_aug); since
    // bw persists across iterations its values are COPIED in -- an m-vector copy, never the
    // m x n matrix.
    //   R-view validity (square diagonal tiles) needs nt <= mt_node - 1, i.e.
    //   n_active <~ (rows-per-node - mb); holds with large margin in the m >> n regime. If
    //   violated, SLATE throws (caught below) rather than returning a wrong answer.

    // Augmented row tiling: data rows [0,mt) follow the global block-row map above (so aw
    // aliases tile-for-tile); the n_active regularizer rows [mt,mt+nt) follow, tiled by nb on rank 0.
    std::function<int64_t (int64_t)> tileMb_aug =
      [mt, tile_height, nt, n_active, nb](int64_t i) -> int64_t {
        if (i < mt) return tile_height[i];
        int64_t jt = i - mt;
        if (jt == nt - 1) return n_active - (nt - 1) * nb;
        return nb;
      };
    std::function<int (slate::func::ij_tuple)> tileRank_aug =
      [mt, tile_owner](slate::func::ij_tuple ij) {
        int64_t i = std::get<0>(ij);
        if (i < mt) return tile_owner[i];
        return 0;
      };
    std::function<int (slate::func::ij_tuple)> tileRank0 =
      [](slate::func::ij_tuple) { return 0; };

    // ---- A_aug: data tiles ALIAS aw (no copy); regularizer tiles are SLATE-owned ----
    slate::Matrix<double> A_aug(m_total + n_active, n_active,
                                tileMb_aug, tileNb, tileRank_aug, tileDevice, comm);

    for (int64_t i = 0; i < mt; ++i)            // data block aliases aw (col-major, ld = lld)
      for (int64_t j = 0; j < nt; ++j)
        if (A_aug.tileIsLocal(i, j)) {
          const int64_t il = i - my_first_tile;                       // local tile-row in this rank's block
          const int64_t offset = row_offset + il * mb + j * nb * lld; // reach this rank's slice of the shared buffer
          A_aug.tileInsert(i, j, local_aw_active + offset, lld);
        }

    for (int64_t jt = 0; jt < nt; ++jt)         // regularizer block: SLATE-allocated tiles
      for (int64_t j = 0; j < nt; ++j)
        if (A_aug.tileIsLocal(mt + jt, j))
          A_aug.tileInsert(mt + jt, j);

    auto A_reg = A_aug.slice(m_total, m_total + n_active - 1, 0, n_active - 1);
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

    // ---- b_aug = [ bw ; 0 ]  (bw COPIED in via an aliased view; bw is preserved) ----
    slate::Matrix<double> y_alias(m_total, 1, tileMb, tile1, tileRank, tileDevice, comm);
    for (int64_t i = 0; i < mt; ++i)
      if (y_alias.tileIsLocal(i, 0)) {
        const int64_t il = i - my_first_tile;
        const int64_t offset = row_offset + il * mb;
        y_alias.tileInsert(i, 0, local_bw + offset, lld);
      }

    slate::Matrix<double> b_aug(m_total + n_active, 1, tileMb_aug, tile1, tileRank_aug, tileDevice, comm);
    b_aug.insertLocalTiles();
    slate::set(0.0, b_aug);
    auto b_top = b_aug.slice(0, m_total - 1, 0, 0);
    slate::copy(y_alias, b_top);                // b_aug[0:m_total] = bw ; b_aug[m_total:] = 0

    MPI_Barrier(comm);

    if (debug) {
      slate::print("A_aug", A_aug, opts);
      slate::print("b_aug", b_aug, opts);
    }

    // ---- DIAGNOSTIC: scan the actual input to qr_factor (A_aug data+reg tiles, b_aug)
    //      for non-finite values. Reads SLATE-local tiles, so it sees exactly what QR sees,
    //      including the aliased aw block and the SLATE-owned regularizer block. Prints only
    //      if something is wrong (or under debug), so it is silent on healthy runs. ----
    {
      int64_t bad_A = 0, bad_b = 0;
      double  amax = 0.0;
      for (int64_t i = 0; i < A_aug.mt(); ++i)
        for (int64_t j = 0; j < A_aug.nt(); ++j)
          if (A_aug.tileIsLocal(i, j)) {
            auto t = A_aug(i, j);
            for (int64_t c = 0; c < t.nb(); ++c)
              for (int64_t r = 0; r < t.mb(); ++r) {
                const double v = t.at(r, c);
                if (!std::isfinite(v)) ++bad_A;
                else { const double av = std::fabs(v); if (av > amax) amax = av; }
              }
          }
      for (int64_t i = 0; i < b_aug.mt(); ++i)
        if (b_aug.tileIsLocal(i, 0)) {
          auto t = b_aug(i, 0);
          for (int64_t r = 0; r < t.mb(); ++r)
            if (!std::isfinite(t.at(r, 0))) ++bad_b;
        }
      // Per-rank report so we can see WHICH rank/node owns the bad tiles.
      if (bad_A || bad_b || debug)
        std::fprintf(stderr,
          "*** SLATE ARD pre-QR [rank %d]: non-finite A_aug=%" PRId64 ", b_aug=%" PRId64
          ", max|A_aug|=%.3e, m_local=%" PRId64 ", row_offset=%" PRId64 ", lld=%" PRId64 "\n",
          mpi_rank, bad_A, bad_b, amax, m_local, row_offset, lld);
      int64_t tot_bad_A = 0, tot_bad_b = 0;
      MPI_Reduce(&bad_A, &tot_bad_A, 1, MPI_INT64_T, MPI_SUM, 0, comm);
      MPI_Reduce(&bad_b, &tot_bad_b, 1, MPI_INT64_T, MPI_SUM, 0, comm);
      if (mpi_rank == 0 && (tot_bad_A || tot_bad_b || debug))
        std::fprintf(stderr,
          "*** SLATE ARD pre-QR TOTAL: non-finite A_aug=%" PRId64 ", b_aug=%" PRId64
          " (m_total=%" PRId64 ", mt=%" PRId64 ", n_active=%" PRId64 ", alpha=%.3e)\n",
          tot_bad_A, tot_bad_b, m_total, mt, n_active, alpha);
    }

    // ---- A_aug = Q R  (in place; aliased aw tiles are destroyed -- safe, see above) ----
    slate::TriangularFactors<double> T;
    slate::qr_factor(A_aug, T);
    MPI_Barrier(comm);

    // R = leading n x n upper-triangular factor.
    auto R_sq = A_aug.slice(0, n_active - 1, 0, n_active - 1);
    auto R = slate::TriangularMatrix<double>(slate::Uplo::Upper, slate::Diag::NonUnit, R_sq);

    // ---- DIAGNOSTIC: scan R's diagonal for non-finite / zero pivots. With reg=diag(sqrt(l/a))
    //      every singular value of A_aug is >= min sqrt(l/a) > 0, so on a correct factorization R
    //      CANNOT have a zero/NaN pivot. If this fires while the pre-QR scan was clean, qr_factor
    //      itself produced the bad R (SLATE-level failure on this distribution, not bad input).
    //      R's diagonal sits in tiles (idx/nb, idx/nb): valid because n_active <= rows_per_rank[0]
    //      (guarded above) and those leading data tiles are uniformly mb=nb tall. ----
    {
      int64_t bad_diag = 0, zero_diag = 0, first_bad = -1;
      double  min_abs = std::numeric_limits<double>::infinity();
      for (int64_t idx = 0; idx < n_active; ++idx) {
        const int64_t it = idx / nb, li = idx % nb;
        if (A_aug.tileIsLocal(it, it)) {
          const double d = A_aug(it, it).at(li, li);
          if (!std::isfinite(d)) { ++bad_diag; if (first_bad < 0) first_bad = idx; }
          else {
            if (d == 0.0) { ++zero_diag; if (first_bad < 0) first_bad = idx; }
            const double ad = std::fabs(d); if (ad < min_abs) min_abs = ad;
          }
        }
      }
      int64_t g_bad = 0, g_zero = 0, g_first = -1;
      double  g_min = 0.0;
      MPI_Reduce(&bad_diag,  &g_bad,  1, MPI_INT64_T, MPI_SUM, 0, comm);
      MPI_Reduce(&zero_diag, &g_zero, 1, MPI_INT64_T, MPI_SUM, 0, comm);
      MPI_Reduce(&first_bad, &g_first, 1, MPI_INT64_T, MPI_MAX, 0, comm);
      MPI_Reduce(&min_abs,  &g_min,  1, MPI_DOUBLE,  MPI_MIN, 0, comm);
      if (mpi_rank == 0 && (g_bad || g_zero || debug))
        std::fprintf(stderr,
          "*** SLATE ARD post-QR R-diag: non-finite=%" PRId64 ", zero=%" PRId64
          ", first-bad-index=%" PRId64 ", min|R_ii|=%.3e\n",
          g_bad, g_zero, g_first, g_min);
    }

    // ---- cond(R) ~ kappa(X)  (sqrt of the cond(C) the normal-equations path reported) ----
    slate::Options cond_opts;
    double R_norm = slate::norm(slate::Norm::One, R);
    double rcond  = slate::trcondest(slate::Norm::One, R, R_norm, cond_opts);
    double cond_number = (rcond > 1e-300) ? (1.0 / rcond) : 1e300;

    // ---- Q^T b_aug : tail d2 -> augmented residual ; head d1 -> coef = R^-1 d1 ----
    slate::qr_multiply_by_q(slate::Side::Left, slate::Op::ConjTrans, A_aug, T, b_aug);
    MPI_Barrier(comm);

    // ||d2||^2 (d2 = trailing m rows of Q^T b_aug) = SSE + (1/alpha) sum lambda*coef^2
    auto d2 = b_aug.slice(n_active, m_total + n_active - 1, 0, 0);   // named: slate::norm needs an lvalue
    double d2_norm = slate::norm(slate::Norm::Fro, d2);
    double aug_resid_sq = d2_norm * d2_norm;

    // coef = R^-1 d1  (d1 = leading n rows of Q^T b_aug), in place in b_aug[0:n]
    auto d1 = b_aug.slice(0, n_active - 1, 0, 0);
    slate::triangular_solve(1.0, R, d1);
    MPI_Barrier(comm);

    if (debug)
      slate::print("coef", d1, opts);

    // -------------------------------- sigma_diag (streamed; peak workspace n x nb, not n x n) --------------------------------
    // sigma_diag[i] = diag(C^-1)_i = diag((R^T R)^-1)_i / alpha = || column i of R^-T ||^2 / alpha.
    // Only the n diagonal entries are needed, so the full n x n R^-T is never materialized (that
    // second ~n^2 matrix would sit on top of A_aug's regularizer block -- ~800 MB each at n~1e4).
    // Instead solve R^T Zp = E_p one nb-wide block-column of the identity at a time, accumulate
    // that block's column 2-norms, and reuse Zp. Same n^3/2 solve flops; peak inverse workspace
    // drops from n x n to n x nb (~40 MB). A sum of squares is >= 0, so sigma_diag stays
    // non-negative by construction. Result is global on every rank (one n-length Allreduce), so
    // no rank-0 gather and no sigma_diag Bcast are needed.
    auto R_T = transpose(R);
    std::function<int64_t (int64_t)> tileNb_one = [nb](int64_t) { return nb; };
    slate::Matrix<double> Zp(n_active, nb, tileNb, tileNb_one, tileRank, tileDevice, comm);
    Zp.insertLocalTiles();

    std::vector<double> sigma_acc(n_active, 0.0);
    for (int64_t bj = 0; bj < nt; ++bj) {
      const int64_t col0 = bj * nb;
      const int64_t ncol = tileNb(bj);              // 256, or the remainder on the last block
      slate::set(0.0, Zp);                          // Zp = 0
      for (int64_t c = 0; c < ncol; ++c) {          // Zp[:, 0:ncol] = unit columns e_{col0+c}
        const int64_t gi = col0 + c;
        const int64_t it = gi / nb, li = gi % nb;
        if (Zp.tileIsLocal(it, 0)) Zp(it, 0).at(li, c) = 1.0;
      }
      slate::triangular_solve(1.0, R_T, Zp);        // Zp[:, 0:ncol] = columns col0.. of R^-T
      for (int64_t it = 0; it < nt; ++it)           // accumulate this block's column 2-norms
        if (Zp.tileIsLocal(it, 0)) {
          auto tile = Zp(it, 0);
          const int64_t nrow = tileNb(it);
          for (int64_t c = 0; c < ncol; ++c) {
            double s = 0.0;
            for (int64_t r = 0; r < nrow; ++r) { const double v = tile.at(r, c); s += v * v; }
            sigma_acc[col0 + c] += s;
          }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, sigma_acc.data(), n_active, MPI_DOUBLE, MPI_SUM, comm);
    for (int64_t i = 0; i < n_active; ++i) local_sigma_diag[i] = sigma_acc[i] / alpha;

    // -------------------------------- coef + SSE (gather coef n-vector to rank 0) --------------------------------
    slate::Matrix<double> coef_loc(n_active, 1, tileNb, tile1, tileRank0, tileDevice, comm);
    coef_loc.insertLocalTiles();
    slate::copy(d1, coef_loc);
    MPI_Barrier(comm);

    if (mpi_rank == 0) {
      for (int64_t i = 0; i < n_active; ++i) {
        int64_t tile_i = i / nb;
        int64_t loc_i  = i % nb;
        if (coef_loc.tileIsLocal(tile_i, 0))
          local_coef_active[i] = coef_loc(tile_i, 0).at(loc_i, 0);
      }

      // SSE = ||d2||^2 - (1/alpha) sum_i lambda_i coef_i^2   (clamp tiny negative roundoff)
      double penalty = 0.0;
      for (int64_t i = 0; i < n_active; ++i)
        penalty += lambda_active[i] * local_coef_active[i] * local_coef_active[i];
      penalty /= alpha;
      double sse = aug_resid_sq - penalty;
      *local_sse = (sse > 0.0) ? sse : 0.0;
    }

    // Broadcast rank-0 results (sigma_diag is already global via the Allreduce above)
    MPI_Bcast(local_coef_active, n_active, MPI_DOUBLE, 0, comm);
    MPI_Bcast(local_sse,         1,        MPI_DOUBLE, 0, comm);
    MPI_Bcast(&cond_number,      1,        MPI_DOUBLE, 0, comm);

    return cond_number;
        
  } catch (const std::exception& e) {
    std::cerr << "[Rank " << mpi_rank << "] SLATE ARD error: " << e.what() << std::endl;
    MPI_Abort(comm, 1);
    return std::numeric_limits<double>::infinity(); // Return inf on error
  }
}


// -----------------------------------------------------------------------------
// SLATE Error Analysis: predictions + per-group error statistics
// -----------------------------------------------------------------------------
//
// Replaces the pandas-DataFrame / iterrows path in slate_common.error_analysis().
// The old Python built a DataFrame from the entire m_local x n design matrix (a
// multi-GB copy) and looped over every row in Python, then ran a
// gather/merge/bcast/gather dance for the two-pass R^2. Here, in compiled code:
//   1. preds = A @ fit via ONE distributed SLATE multiply (A ALIASES the
//      node-shared design buffer -- no copy, read-only),
//   2. per-group weighted/unweighted sums over local rows,
//   3. MPI_Allreduce those sums so every rank holds the global means,
//   4. a second pass accumulating per-group SS_tot about those global means,
//   5. MPI_Allreduce SS_tot.
// Each row is folded into TWO bins: its own (group,testing,row_type) bin and the
// rolled-up ('*ALL',testing,row_type) bin, so the '*ALL' SS_tot is taken about
// the '*ALL' mean (which is NOT the sum of the per-group SS_tot). Python is left
// only O(n_groups) bookkeeping (MAE/RMSE/R^2 + DataFrame) and the optional
// validation scatter, for which this rank's local preds are returned in out_preds.
//
// Layout (matches the ridge/ARD solvers): a/b/w are node-shared COLUMN-MAJOR
// buffers; this rank owns rows [row_offset, row_offset+m_local) with column
// stride lld (= a.shape[0], the node row count). fit is the replicated n-vector.
// bin_specific[i]/bin_all[i] in [0,n_groups) are the global bins for local row i;
// group_factor[g] is the unit-conversion factor for bin g (constant within a bin
// since it depends only on Row_Type).
void slate_error_analysis(
    double* local_a, double* local_b, double* local_w, double* fit,
    int32_t* bin_specific, int32_t* bin_all, double* group_factor, int n_groups,
    int64_t n, int64_t lld, int64_t row_offset, int64_t m_local,
    double* out_preds, int64_t* out_count,
    double* out_sum_w, double* out_sum_truth_w, double* out_sum_ae_w, double* out_sum_se_w,
    double* out_sum_truth_u, double* out_sum_ae_u, double* out_sum_se_u,
    double* out_sstot_w, double* out_sstot_u,
    MPI_Comm comm, int debug) {

  int mpi_rank, mpi_size;
  MPI_Comm_rank(comm, &mpi_rank);
  MPI_Comm_size(comm, &mpi_size);

  // -------------------------------- TILE SIZE --------------------------------
  int64_t mb = 256;
  int64_t nb = 256;
  int64_t nt = ceil_div64(n, nb);

  // -------------------------------- GLOBAL BLOCK-ROW MAP --------------------------------
  // Same construction as slate_ard_update: Allgather each rank's true row count so the
  // tileMb/tileRank closures are byte-identical on every rank, and each rank owns a
  // contiguous run of tile-rows that aliases its own slice of the node-shared buffer.
  std::vector<int64_t> rows_per_rank(mpi_size);
  MPI_Allgather(&m_local, 1, MPI_INT64_T, rows_per_rank.data(), 1, MPI_INT64_T, comm);

  std::vector<int64_t> blk_first_tile(mpi_size);
  int64_t mt = 0, m_total = 0;
  for (int r = 0; r < mpi_size; ++r) {
    blk_first_tile[r] = mt;
    mt      += ceil_div64(rows_per_rank[r], mb);
    m_total += rows_per_rank[r];
  }
  std::vector<int>     tile_owner(mt);
  std::vector<int64_t> tile_height(mt);
  for (int r = 0; r < mpi_size; ++r) {
    int64_t mtr = ceil_div64(rows_per_rank[r], mb);
    for (int64_t il = 0; il < mtr; ++il) {
      int64_t i = blk_first_tile[r] + il;
      tile_owner[i]  = r;
      tile_height[i] = (il == mtr - 1) ? (rows_per_rank[r] - (mtr - 1) * mb) : mb;
    }
  }
  const int64_t my_first_tile = blk_first_tile[mpi_rank];

  std::function<int64_t (int64_t)> tile1  = [](int64_t) { return 1; };
  std::function<int64_t (int64_t)> tileMb = [tile_height](int64_t i) { return tile_height[i]; };
  int64_t tile_col_last = nt - 1;
  int64_t tile_col_remainder = n - (nt - 1) * nb;
  std::function<int64_t (int64_t)> tileNb = [tile_col_last, tile_col_remainder, nb](int64_t j) {
    return (j == tile_col_last) ? tile_col_remainder : nb;
  };
  std::function<int (slate::func::ij_tuple)> tileRank =
    [tile_owner](slate::func::ij_tuple ij) { return tile_owner[std::get<0>(ij)]; };
  std::function<int (slate::func::ij_tuple)> tileRank0 =
    [](slate::func::ij_tuple) { return 0; };
  std::function<int (slate::func::ij_tuple)> tileDevice =
    [](slate::func::ij_tuple) { return slate::HostNum; };

  if (debug && mpi_rank == 0)
    std::fprintf(stderr, "*** SLATE error_analysis: %d rank(s), m_total=%" PRId64
                 ", n=%" PRId64 ", n_groups=%d\n", mpi_size, m_total, n, n_groups);

  std::vector<double> preds(m_local > 0 ? m_local : 0, 0.0);

  try {
    // ---------------------------- preds = A @ fit (distributed SLATE multiply) ----------------------------
    // A aliases the node-shared design buffer (read-only; multiply never writes A).
    slate::Matrix<double> A(m_total, n, tileMb, tileNb, tileRank, tileDevice, comm);
    for (int64_t i = 0; i < mt; ++i)
      for (int64_t j = 0; j < nt; ++j)
        if (A.tileIsLocal(i, j)) {
          const int64_t il = i - my_first_tile;
          const int64_t offset = row_offset + il * mb + j * nb * lld;
          A.tileInsert(i, j, local_a + offset, lld);
        }

    // X = fit (n x 1), all tiles owned by rank 0; multiply broadcasts it to every rank.
    slate::Matrix<double> X(n, 1, tileNb, tile1, tileRank0, tileDevice, comm);
    X.insertLocalTiles();
    if (mpi_rank == 0)
      for (int64_t jt = 0; jt < nt; ++jt)
        if (X.tileIsLocal(jt, 0)) {
          auto t = X(jt, 0);
          const int64_t base = jt * nb;
          const int64_t h = tileNb(jt);
          for (int64_t r = 0; r < h; ++r) t.at(r, 0) = fit[base + r];
        }

    // C = preds (m_total x 1), row-distributed like A.
    slate::Matrix<double> C(m_total, 1, tileMb, tile1, tileRank, tileDevice, comm);
    C.insertLocalTiles();
    slate::multiply(1.0, A, X, 0.0, C);   // C = A @ fit
    MPI_Barrier(comm);

    // Pull this rank's contiguous local rows of C into preds[0:m_local].
    const int64_t my_ntiles = ceil_div64(m_local, mb);
    for (int64_t il = 0; il < my_ntiles; ++il) {
      const int64_t i = my_first_tile + il;
      if (C.tileIsLocal(i, 0)) {
        auto t = C(i, 0);
        const int64_t h = tile_height[i];
        const int64_t base = il * mb;
        for (int64_t r = 0; r < h; ++r) preds[base + r] = t.at(r, 0);
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "[Rank " << mpi_rank << "] SLATE error_analysis multiply error: " << e.what() << std::endl;
    MPI_Abort(comm, 1);
  }

  for (int64_t i = 0; i < m_local; ++i) out_preds[i] = preds[i];

  // -------------------------------- PASS 1: per-group sums (local) --------------------------------
  std::vector<int64_t> cnt (n_groups, 0);
  std::vector<double>  sw  (n_groups, 0.0), stw (n_groups, 0.0), sae (n_groups, 0.0), sse (n_groups, 0.0);
  std::vector<double>  stu (n_groups, 0.0), saeu(n_groups, 0.0), sseu(n_groups, 0.0);

  auto accumulate_sums = [&](int g, double w, double truth, double ae, double se) {
    cnt [g] += 1;
    sw  [g] += w;
    stw [g] += w * truth;
    sae [g] += w * ae;
    sse [g] += w * se;
    stu [g] += truth;
    saeu[g] += ae;
    sseu[g] += se;
  };

  for (int64_t i = 0; i < m_local; ++i) {
    const int gs = bin_specific[i];
    if (gs < 0 || gs >= n_groups) continue;
    const int ga = bin_all[i];
    const double f     = group_factor[gs];
    const double w     = local_w[row_offset + i];
    const double truth = f * local_b[row_offset + i];
    double       pred  = f * preds[i];
    if (!std::isfinite(pred)) pred = truth;   // mirror old code: bad pred -> zero error
    const double diff  = truth - pred;
    const double ae    = std::fabs(diff);
    const double se    = diff * diff;
    accumulate_sums(gs, w, truth, ae, se);
    if (ga >= 0 && ga < n_groups && ga != gs) accumulate_sums(ga, w, truth, ae, se);
  }

  MPI_Allreduce(cnt.data(),  out_count,       n_groups, MPI_INT64_T, MPI_SUM, comm);
  MPI_Allreduce(sw.data(),   out_sum_w,       n_groups, MPI_DOUBLE,  MPI_SUM, comm);
  MPI_Allreduce(stw.data(),  out_sum_truth_w, n_groups, MPI_DOUBLE,  MPI_SUM, comm);
  MPI_Allreduce(sae.data(),  out_sum_ae_w,    n_groups, MPI_DOUBLE,  MPI_SUM, comm);
  MPI_Allreduce(sse.data(),  out_sum_se_w,    n_groups, MPI_DOUBLE,  MPI_SUM, comm);
  MPI_Allreduce(stu.data(),  out_sum_truth_u, n_groups, MPI_DOUBLE,  MPI_SUM, comm);
  MPI_Allreduce(saeu.data(), out_sum_ae_u,    n_groups, MPI_DOUBLE,  MPI_SUM, comm);
  MPI_Allreduce(sseu.data(), out_sum_se_u,    n_groups, MPI_DOUBLE,  MPI_SUM, comm);

  // Global means (identical on every rank).
  std::vector<double> mean_w(n_groups, 0.0), mean_u(n_groups, 0.0);
  for (int g = 0; g < n_groups; ++g) {
    mean_w[g] = (out_sum_w[g] > 0.0) ? out_sum_truth_w[g] / out_sum_w[g]           : 0.0;
    mean_u[g] = (out_count[g] > 0)   ? out_sum_truth_u[g] / (double) out_count[g]  : 0.0;
  }

  // -------------------------------- PASS 2: per-group SS_tot about the global mean (local) --------------------------------
  std::vector<double> sstw(n_groups, 0.0), sstu(n_groups, 0.0);
  for (int64_t i = 0; i < m_local; ++i) {
    const int gs = bin_specific[i];
    if (gs < 0 || gs >= n_groups) continue;
    const int ga = bin_all[i];
    const double f     = group_factor[gs];
    const double w     = local_w[row_offset + i];
    const double truth = f * local_b[row_offset + i];
    const double dws = truth - mean_w[gs];
    const double dus = truth - mean_u[gs];
    sstw[gs] += w * dws * dws;
    sstu[gs] += dus * dus;
    if (ga >= 0 && ga < n_groups && ga != gs) {
      const double dwa = truth - mean_w[ga];
      const double dua = truth - mean_u[ga];
      sstw[ga] += w * dwa * dwa;
      sstu[ga] += dua * dua;
    }
  }
  MPI_Allreduce(sstw.data(), out_sstot_w, n_groups, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(sstu.data(), out_sstot_u, n_groups, MPI_DOUBLE, MPI_SUM, comm);
}

} // extern "C"


