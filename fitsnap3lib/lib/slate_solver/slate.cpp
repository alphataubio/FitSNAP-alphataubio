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

constexpr int64_t ceil_div64(int64_t a, int64_t b) { return (a + b - 1) / b; }

namespace {  // ---- internal-linkage helpers shared by the ARD QR / Cholesky paths ----

// Block-row map of the node-shared design across ranks, for a given m-tile height mb. Each rank
// owns ceil(m_local/mb) consecutive mb-sized tile-rows (last one short) aliasing its own slice of
// the node-shared buffer. Byte-identical on every rank (SLATE requires identical tile closures).
struct BlockRowMap {
  std::vector<int>     tile_owner;    // [mt] physical owner rank of each data tile-row
  std::vector<int64_t> tile_height;   // [mt] rows in each data tile-row
  std::vector<int64_t> tile_row0;     // [mt] global first row of each data tile-row
  int64_t mt = 0;                     // total data tile-rows across all ranks
  int64_t m_total = 0;                // total data rows across all ranks
  int64_t my_first_tile = 0;          // this rank's first data tile-row
  int64_t my_ntiles = 0;              // this rank's data tile-row count
};

BlockRowMap build_blockrow_map(int64_t mb, int mpi_rank, int mpi_size,
                               int64_t m_local, const std::vector<int64_t>& rows_per_rank) {
  BlockRowMap M;
  std::vector<int64_t> blk_first_tile(mpi_size);
  for (int r = 0; r < mpi_size; ++r) {
    blk_first_tile[r] = M.mt;
    M.mt      += ceil_div64(rows_per_rank[r], mb);
    M.m_total += rows_per_rank[r];
  }
  M.tile_owner.resize(M.mt);
  M.tile_height.resize(M.mt);
  M.tile_row0.resize(M.mt);
  int64_t row = 0;
  for (int r = 0; r < mpi_size; ++r) {
    int64_t mtr = ceil_div64(rows_per_rank[r], mb);
    for (int64_t il = 0; il < mtr; ++il) {
      int64_t i = blk_first_tile[r] + il;
      M.tile_owner[i]  = r;
      M.tile_height[i] = (il == mtr - 1) ? (rows_per_rank[r] - (mtr - 1) * mb) : mb;
      M.tile_row0[i]   = row;
      row += M.tile_height[i];
    }
  }
  M.my_first_tile = blk_first_tile[mpi_rank];
  M.my_ntiles     = (m_local > 0) ? ceil_div64(m_local, mb) : 0;
  return M;
}

// Fill the data row-tiles [0,mt) of a block-row matrix M_ (m_total x n_active) with the weighted
// active design gathered LOCALLY from the node-shared column-major buffer:
//   M_(row, col) = w_eff[k] * a[ row_offset+k , active_indices[col] ]   (zero MPI, 'a' read-only).
void fill_weighted_design(slate::Matrix<double>& M_, const BlockRowMap& map,
                          int64_t mb, int64_t nb, int64_t nt,
                          int64_t row_offset, int64_t lld,
                          const double* local_a, const double* local_w_eff,
                          const int64_t* active_indices,
                          const std::function<int64_t(int64_t)>& tileNb) {
  for (int64_t il = 0; il < map.my_ntiles; ++il) {
    const int64_t i = map.my_first_tile + il;
    const int64_t h = map.tile_height[i];
    for (int64_t jt = 0; jt < nt; ++jt) {
      if (!M_.tileIsLocal(i, jt)) continue;
      auto tile = M_(i, jt);
      const int64_t col0 = jt * nb;
      const int64_t ncol = tileNb(jt);
      for (int64_t r = 0; r < h; ++r) {
        const int64_t k  = il * mb + r;
        const int64_t br = row_offset + k;
        const double  wk = local_w_eff[k];
        for (int64_t c = 0; c < ncol; ++c)
          tile.at(r, c) = wk * local_a[br + active_indices[col0 + c] * lld];
      }
    }
  }
}

// Fill the data row-tiles [0,mt) of a block-row column vector v with w_eff .* b (local).
void fill_weighted_rhs(slate::Matrix<double>& v, const BlockRowMap& map, int64_t mb,
                       int64_t row_offset, const double* local_b, const double* local_w_eff) {
  for (int64_t il = 0; il < map.my_ntiles; ++il) {
    const int64_t i = map.my_first_tile + il;
    if (!v.tileIsLocal(i, 0)) continue;
    auto tile = v(i, 0);
    const int64_t h = map.tile_height[i];
    for (int64_t r = 0; r < h; ++r) {
      const int64_t k = il * mb + r;
      tile.at(r, 0) = local_w_eff[k] * local_b[row_offset + k];
    }
  }
}

}  // anonymous namespace

extern "C" {

using slate::func::ij_tuple;

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

double slate_ard_update(double* local_a, double* local_b, double* local_w_eff,
                     int64_t* active_indices,
                     double* local_sigma_diag, double* local_coef_active,
                     double* local_sse,
                     int64_t m, int64_t n_active, int64_t lld,
                     int64_t row_offset, int64_t m_local,
                     double alpha, double* lambda_active,
                     int method,
                     MPI_Comm comm, int debug) {

  // INPUT (read-only, NEVER written): local_a is this rank's slice of the node-shared,
  // COLUMN-MAJOR FitSNAP design buffer 'a' -- ALL n columns, column stride lld = a.shape[0]
  // (= node rows). ARD rebuilds the working matrix every iteration from this pristine buffer
  // with the current active set + weights, and error_analysis reads it again afterwards, so it
  // must NOT be mutated. local_b is the matching node-shared target slice; local_w_eff[k] is the
  // effective weight (testing rows pre-zeroed by Python) for this rank's local row k in
  // [0, m_local); active_indices[c] is the GLOBAL column of the c-th active feature.
  //
  // row_offset : this rank's first row WITHIN the node-shared buffer (= sub_a_indices start).
  // m_local    : design rows this rank owns in that buffer (= a_end_idx - a_start_idx + 1).
  // lld        : column stride of the shared buffer = a.shape[0] (node rows).
  //
  // The working matrix 'aw' built below is a SEPARATE, SLATE-owned, block-row matrix -- it is NOT
  // an alias of 'a'. qr_factor destroys aw in place; 'a' is left byte-identical for the next
  // iteration. That is the whole reason aw is materialized here instead of in numpy.

  // -------------------------------- HYBRID MPI/OPENMP --------------------------------

  int mpi_rank, mpi_size;
  MPI_Comm_rank(comm, &mpi_rank);
  MPI_Comm_size(comm, &mpi_size);
  int num_threads = omp_get_max_threads();
  (void) m; (void) num_threads;

  // -------------------------------- PRINT OPTS --------------------------------

  slate::Options opts = {
    {slate::Option::PrintVerbose, 4},
    {slate::Option::PrintPrecision, 1},
    {slate::Option::PrintWidth, 4}
  };

  // -------------------------------- TILE SIZE (n-direction) --------------------------------
  // nb = 256 is L2-cache-optimal on Zen4/5: a 256x256 double tile is 512 KiB = half the 1 MiB
  // private L2, leaving the other half for the streaming operand panel, and it aligns with the
  // AVX-512 8-double lane. nb tiles C and Xw's columns. The m-direction tile mb is chosen PER PATH
  // below: fat for the Cholesky/herk default (mb only sets herk's contraction-loop length and the
  // broadcast granularity -- BLIS blocks the contraction in cache itself), and 256 for the QR
  // fallback (whose R factor needs SQUARE diagonal tiles).
  const int64_t nb = 256;
  const int64_t nt = ceil_div64(n_active, nb);
  if (n_active == 0) return 1.0; // Perfect conditioning for empty matrix

  // Per-rank row counts (identical on all ranks after Allgather) drive the block-row maps below.
  std::vector<int64_t> rows_per_rank(mpi_size);
  MPI_Allgather(&m_local, 1, MPI_INT64_T, rows_per_rank.data(), 1, MPI_INT64_T, comm);

  // Shared closures.
  std::function<int64_t (int64_t)> tile1 = [](int64_t) { return 1; };
  std::function<int (slate::func::ij_tuple)> tileDevice =
    [](slate::func::ij_tuple) { return slate::HostNum; };
  const int64_t tile_col_last = nt - 1;
  const int64_t tile_col_remainder = n_active - (nt - 1) * nb;
  std::function<int64_t (int64_t)> tileNb = [tile_col_last, tile_col_remainder, nb](int64_t j) {
    return (j == tile_col_last) ? tile_col_remainder : nb;
  };

  // ======================= CHOLESKY on the normal equations (default) =======================
  // C = Xw^T Xw + diag(lambda/alpha), Xw = w.X.  Formed by a distributed herk from a BLOCK-ROW Xw
  // (each rank fills its own rows locally, zero MPI) into a 2D-block-cyclic Hermitian C, then
  // chol_factor / chol_solve / chol_inverse run fully 2D-distributed -- no rank-0 serialization.
  // Reports cond in kappa(X) units (sqrt of kappa(C)) so the QR<->Cholesky switch threshold in
  // Python is consistent. 2x fewer flops than QR but squares the condition number, hence Python
  // switches to QR when the previous iteration's cond exceeds the threshold.
  if (method == 1) {

    // Fat m-tile: from the SMALLEST rank's row count (so mb is identical on every rank), a handful
    // of tiles per rank so the 8 threads stay fed while herk's contraction loop drops from
    // thousands of steps (the mb=256 pathology) to a few dozen. mb is NOT a cache tile -- BLIS
    // blocks the contraction internally -- so fat is strictly better here.
    int64_t mb;
    {
      int64_t m_min = *std::min_element(rows_per_rank.begin(), rows_per_rank.end());
      mb = m_min / (2 * (int64_t) num_threads);
      if (mb < 256)  mb = 256;
      if (mb > 8192) mb = 8192;
    }
    BlockRowMap map = build_blockrow_map(mb, mpi_rank, mpi_size, m_local, rows_per_rank);
    const int64_t m_total = map.m_total;

    std::function<int64_t (int64_t)> tileMb_data =
      [tile_height = map.tile_height](int64_t i) { return tile_height[i]; };
    std::function<int (slate::func::ij_tuple)> tileRank_phys =
      [tile_owner = map.tile_owner](slate::func::ij_tuple ij) { return tile_owner[std::get<0>(ij)]; };

    // Near-square 2D process grid for C (the small n x n matrix ONLY -- not the m x n data, so no
    // redistribution storm). This is the canonical scalable dense-Cholesky layout.
    int p = 1, q = mpi_size;
    for (int pp = (int) std::floor(std::sqrt((double) mpi_size)); pp >= 1; --pp)
      if (mpi_size % pp == 0) { p = pp; q = mpi_size / pp; break; }
    auto tileRank2D = slate::func::process_2d_grid(slate::GridOrder::Col, p, q);

    if (mpi_rank == 0 && debug)
      std::fprintf(stderr, "\n=== slate_ard_update (Cholesky/herk, mb=%" PRId64 ", nb=%" PRId64
                   ", C grid %dx%d, m=%" PRId64 ", n_active=%" PRId64 ") ===\n",
                   mb, nb, p, q, m_total, n_active);

    try {
      // Xw = w . X[:,active]  (m_total x n_active), block-row, filled locally from 'a'.
      slate::Matrix<double> Xw(m_total, n_active, tileMb_data, tileNb, tileRank_phys, tileDevice, comm);
      Xw.insertLocalTiles();
      fill_weighted_design(Xw, map, mb, nb, nt, row_offset, lld,
                           local_a, local_w_eff, active_indices, tileNb);

      // bw = w . y  (m_total x 1), block-row.
      slate::Matrix<double> bw(m_total, 1, tileMb_data, tile1, tileRank_phys, tileDevice, comm);
      bw.insertLocalTiles();
      fill_weighted_rhs(bw, map, mb, row_offset, local_b, local_w_eff);

      // C = Xw^T Xw  (herk: C = A A^H with A = Xw^T; broadcasts each block-row of Xw from its
      // owner into the 2D C -- the big m x n data is never redistributed).
      slate::HermitianMatrix<double> C(slate::Uplo::Lower, n_active, tileNb, tileRank2D, tileDevice, comm);
      C.insertLocalTiles();
      auto XwT = slate::transpose(Xw);
      slate::rank_k_update(1.0, XwT, 0.0, C);

      // C += diag(lambda/alpha)  (each diagonal entry on its 2D owner).
      for (int64_t idx = 0; idx < n_active; ++idx) {
        const int64_t it = idx / nb, loc = idx % nb;
        if (C.tileIsLocal(it, it))
          C(it, it).at(loc, loc) += lambda_active[idx] / alpha;
      }

      if (debug) slate::print("C", C, opts);

      // cond(C) via pocondest: need ||C||_1 BEFORE factoring.
      double Cnorm = slate::norm(slate::Norm::One, C);

      int64_t info = slate::chol_factor(C);                 // C -> L (potrf)
      if (info != 0) {
        if (mpi_rank == 0)
          std::fprintf(stderr, "slate_ard_update: chol_factor info=%" PRId64
                       " (normal matrix not SPD -- raise alpha or use QR)\n", info);
        MPI_Abort(comm, 3);
      }

      double rcond = slate::chol_rcondest_using_factor(slate::Norm::One, C, Cnorm);
      double condC = (rcond > 1e-300) ? (1.0 / rcond) : 1e300;
      double cond_number = std::sqrt(condC);                // kappa(X) units (C ~ X^T X)

      // g = Xw^T bw  (n_active x 1, on C's 2D grid).
      slate::Matrix<double> g(n_active, 1, tileNb, tile1, tileRank2D, tileDevice, comm);
      g.insertLocalTiles();
      auto XwT2 = slate::transpose(Xw);
      slate::multiply(1.0, XwT2, bw, 0.0, g);               // gemm

      // Snapshot g (= X^T y projection) before the solve overwrites it -- needed for SSE.
      std::vector<double> g_host(n_active, 0.0);
      for (int64_t it = 0; it < nt; ++it)
        if (g.tileIsLocal(it, 0)) {
          auto tile = g(it, 0);
          const int64_t h = tileNb(it), base = it * nb;
          for (int64_t r = 0; r < h; ++r) g_host[base + r] = tile.at(r, 0);
        }
      MPI_Allreduce(MPI_IN_PLACE, g_host.data(), n_active, MPI_DOUBLE, MPI_SUM, comm);

      // coef = C^-1 g  (potrs; g overwritten with the solution).
      slate::chol_solve_using_factor(C, g);
      std::vector<double> coef_host(n_active, 0.0);
      for (int64_t it = 0; it < nt; ++it)
        if (g.tileIsLocal(it, 0)) {
          auto tile = g(it, 0);
          const int64_t h = tileNb(it), base = it * nb;
          for (int64_t r = 0; r < h; ++r) coef_host[base + r] = tile.at(r, 0);
        }
      MPI_Allreduce(MPI_IN_PLACE, coef_host.data(), n_active, MPI_DOUBLE, MPI_SUM, comm);
      for (int64_t i = 0; i < n_active; ++i) local_coef_active[i] = coef_host[i];

      // sigma_diag = diag(C^-1) / alpha  (potri, then read the diagonal tiles).
      slate::chol_inverse_using_factor(C);                  // C -> C^-1
      std::vector<double> sig_host(n_active, 0.0);
      for (int64_t it = 0; it < nt; ++it)
        if (C.tileIsLocal(it, it)) {
          auto tile = C(it, it);
          const int64_t h = tileNb(it), base = it * nb;
          for (int64_t loc = 0; loc < h; ++loc) sig_host[base + loc] = tile.at(loc, loc);
        }
      MPI_Allreduce(MPI_IN_PLACE, sig_host.data(), n_active, MPI_DOUBLE, MPI_SUM, comm);
      for (int64_t i = 0; i < n_active; ++i) local_sigma_diag[i] = sig_host[i] / alpha;

      // SSE = ||bw||^2 - coef.g - penalty ;  penalty = (1/alpha) sum lambda_i coef_i^2.
      double bw_norm = slate::norm(slate::Norm::Fro, bw);
      double bw_sq = bw_norm * bw_norm;
      double coef_dot_g = 0.0, penalty = 0.0;
      for (int64_t i = 0; i < n_active; ++i) {
        coef_dot_g += coef_host[i] * g_host[i];
        penalty    += lambda_active[i] * coef_host[i] * coef_host[i];
      }
      penalty /= alpha;
      double sse = bw_sq - coef_dot_g - penalty;
      *local_sse = (sse > 0.0) ? sse : 0.0;

      return cond_number;

    } catch (const std::exception& e) {
      std::cerr << "[Rank " << mpi_rank << "] SLATE ARD (Cholesky) error: " << e.what() << std::endl;
      MPI_Abort(comm, 1);
      return std::numeric_limits<double>::infinity();
    }
  }

  // ============================ QR (augmented) fallback ============================
  // Used when the previous iteration's cond exceeded the switch threshold. Needs SQUARE R tiles,
  // so mb == nb == 256 here. Accurate (kappa(R) = kappa(X)) but slower and rank-0-bound in its
  // R-space tail, which is why it is the fallback rather than the default.
  const int64_t mb = 256;
  BlockRowMap map = build_blockrow_map(mb, mpi_rank, mpi_size, m_local, rows_per_rank);
  const int64_t mt = map.mt, m_total = map.m_total;
  const std::vector<int>&     tile_owner  = map.tile_owner;
  const std::vector<int64_t>& tile_height = map.tile_height;
  const std::vector<int64_t>& tile_row0   = map.tile_row0;
  const int64_t my_first_tile = map.my_first_tile;
  const int64_t my_ntiles     = map.my_ntiles;

  std::function<int (slate::func::ij_tuple)> tileRank0 =
    [](slate::func::ij_tuple) { return 0; };

  // Augmented row tiling: data rows [0,mt) keep their mb-tiling; the n_active regularizer rows
  // [mt,mt+nt) follow, tiled by nb.
  std::function<int64_t (int64_t)> tileMb_aug =
    [mt, tile_height, nt, n_active, nb](int64_t i) -> int64_t {
      if (i < mt) return tile_height[i];
      int64_t jt = i - mt;
      return (jt == nt - 1) ? (n_active - (nt - 1) * nb) : nb;
    };
  // Augmented row OWNERSHIP: data rows -> physical owner (local fill); regularizer rows -> rank 0.
  std::function<int (slate::func::ij_tuple)> tileRank_aug =
    [mt, tile_owner](slate::func::ij_tuple ij) -> int {
      int64_t i = std::get<0>(ij);
      return (i < mt) ? tile_owner[i] : 0;
    };

  if (mpi_rank == 0 && debug) {
    std::fprintf(stderr, "\n=== slate_ard_update (augmented-QR fallback, mb=nb=256, %d ranks) ===\n", mpi_size);
    std::fprintf(stderr, "  m=%" PRId64 ", n_active=%" PRId64 ", alpha=%.6e, data tile-rows=%"
                 PRId64 " (nt=%" PRId64 ")\n", m_total, n_active, alpha, mt, nt);
  }
    
  try {

    // R-space locality requirement: with block-row aw, the leading n_active rows (hence the R
    // factor and the whole R-space tail: trcondest, triangular_solve, the R^-T column-norm sweep)
    // live entirely in rank 0's block PROVIDED n_active <= rows-on-rank-0. With m >> n this holds
    // with wide margin (rank 0 owns ~m/num_ranks rows >> n_active). If violated, R would straddle a
    // rank boundary and the R-space ops below would be wrong -- so fail loudly instead.
    if (n_active > rows_per_rank[0]) {
      if (mpi_rank == 0)
        std::fprintf(stderr,
          "slate_ard_update: n_active=%" PRId64 " > rows-on-rank-0=%" PRId64
          "; R-space tiling would be irregular (use fewer ranks/node for the solve).\n",
          n_active, rows_per_rank[0]);
      MPI_Abort(comm, 2);
    }

    // ----------------------------------------------------------------------------------------
    // BUILD  aw = [ w . a[:,active] ; diag(sqrt(lambda/alpha)) ]  as a SLATE-owned block-row matrix.
    //
    //     aw = [        w . X         ]  (m_total  x n_active)    b_aug = [ w . y ]
    //          [ diag(sqrt(lambda/a)) ]  (n_active x n_active)            [   0   ]
    //
    //     aw = Q R   =>   R^T R = X^T X + diag(lambda/alpha) = C / alpha
    //
    // Augmented QR (NOT Cholesky on the normal equations): kappa(R) = kappa(X), so precision is
    // retained. The normal-equations form squares the condition number (kappa(X)^2), which in
    // double precision floors achievable SSE near 1e-8.
    //
    //   coef       = R^-1 (Q^T b_aug)[0:n] = alpha C^-1 X^T y          (ARD MAP estimate)
    //   ||d2||^2   = ||(Q^T b_aug)[n:]||^2 = SSE + (1/alpha) sum_i lambda_i coef_i^2
    //   sigma_diag = diag(C^-1) = ||col_i of R^-T||^2 / alpha
    //
    // 'a' is the INPUT and is never written; qr_factor destroys aw (separate memory), so 'a'
    // survives for the next ARD iteration.
    // ----------------------------------------------------------------------------------------
    slate::Matrix<double> aw(m_total + n_active, n_active,
                             tileMb_aug, tileNb, tileRank_aug, tileDevice, comm);
    aw.insertLocalTiles();

    const double inv_sqrt_alpha = 1.0 / std::sqrt(alpha);

    // ---- data block (rows [0,m_total)): each rank fills its OWN physical row-tiles directly from
    //      the node-shared 'a' -- gather the active columns, scale by the row weight. Purely local,
    //      zero MPI, and 'a' is only read. This is the whole payoff of the block-row layout. ----
    for (int64_t il = 0; il < my_ntiles; ++il) {
      const int64_t i = my_first_tile + il;
      const int64_t h = tile_height[i];
      for (int64_t jt = 0; jt < nt; ++jt) {
        if (!aw.tileIsLocal(i, jt)) continue;
        auto tile = aw(i, jt);
        const int64_t col0 = jt * nb;
        const int64_t ncol = tileNb(jt);
        for (int64_t r = 0; r < h; ++r) {
          const int64_t k  = il * mb + r;                // rank-local row in [0, m_local)
          const int64_t br = row_offset + k;             // row within the node-shared buffer
          const double  wk = local_w_eff[k];
          for (int64_t c = 0; c < ncol; ++c)             // gather active cols, scale by weight
            tile.at(r, c) = wk * local_a[br + active_indices[col0 + c] * lld];
        }
      }
    }

    // ---- regularizer block (rows [m_total, m_total+n_active)): diag(sqrt(lambda/alpha)); these
    //      rows live on rank 0, so only rank 0 has local tiles here. ----
    auto A_reg = aw.slice(m_total, m_total + n_active - 1, 0, n_active - 1);
    slate::set(0.0, A_reg);
    for (int64_t idx = 0; idx < n_active; ++idx) {
      const int64_t it = mt + idx / nb, jc = idx / nb, loc = idx % nb;
      if (aw.tileIsLocal(it, jc))
        aw(it, jc).at(loc, loc) = std::sqrt(lambda_active[idx]) * inv_sqrt_alpha;
    }

    // ---- b_aug = [ w . b ; 0 ] : data rows filled locally like aw, regularizer rows zero. ----
    slate::Matrix<double> b_aug(m_total + n_active, 1, tileMb_aug, tile1, tileRank_aug, tileDevice, comm);
    b_aug.insertLocalTiles();
    slate::set(0.0, b_aug);
    for (int64_t il = 0; il < my_ntiles; ++il) {
      const int64_t i = my_first_tile + il;
      if (!b_aug.tileIsLocal(i, 0)) continue;
      auto tile = b_aug(i, 0);
      const int64_t h = tile_height[i];
      for (int64_t r = 0; r < h; ++r) {
        const int64_t k = il * mb + r;
        tile.at(r, 0) = local_w_eff[k] * local_b[row_offset + k];
      }
    }

    MPI_Barrier(comm);

    if (debug) {
      slate::print("aw", aw, opts);
      slate::print("b_aug", b_aug, opts);
    }

    // ---- aw = Q R  (in place; aw is SLATE scratch, 'a' untouched) ----
    slate::TriangularFactors<double> T;
    slate::qr_factor(aw, T);
    MPI_Barrier(comm);

    // R = leading n_active x n_active upper-triangular factor (lives on rank 0; see guard above).
    auto R_sq = aw.slice(0, n_active - 1, 0, n_active - 1);
    auto R = slate::TriangularMatrix<double>(slate::Uplo::Upper, slate::Diag::NonUnit, R_sq);

    // ---- cond(R) ~ kappa(X)  (sqrt of the cond(C) a normal-equations path would report) ----
    slate::Options cond_opts;
    double R_norm = slate::norm(slate::Norm::One, R);
    double rcond  = slate::trcondest(slate::Norm::One, R, R_norm, cond_opts);
    double cond_number = (rcond > 1e-300) ? (1.0 / rcond) : 1e300;

    // ---- Q^T b_aug : tail d2 -> augmented residual ; head d1 -> coef = R^-1 d1 ----
    slate::qr_multiply_by_q(slate::Side::Left, slate::Op::ConjTrans, aw, T, b_aug);
    MPI_Barrier(comm);

    // ||d2||^2 (d2 = trailing m_total rows of Q^T b_aug) = SSE + (1/alpha) sum lambda*coef^2
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
    // second ~n^2 matrix would sit on top of aw's regularizer block -- ~800 MB each at n~1e4).
    // Instead solve R^T Zp = E_p one nb-wide block-column of the identity at a time, accumulate
    // that block's column 2-norms, and reuse Zp. Same n^3/2 solve flops; peak inverse workspace
    // drops from n x n to n x nb (~40 MB). A sum of squares is >= 0, so sigma_diag stays
    // non-negative by construction. Result is global on every rank (one n-length Allreduce), so
    // no rank-0 gather and no sigma_diag Bcast are needed.
    auto R_T = transpose(R);
    std::function<int64_t (int64_t)> tileNb_one = [nb](int64_t) { return nb; };
    slate::Matrix<double> Zp(n_active, nb, tileNb, tileNb_one, tileRank0, tileDevice, comm);
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

    // -------------------------------- coef (local-tile read + Allreduce; global on all ranks) --------------------------------
    // coef occupies the leading n_active rows of b_aug, 2D-distributed. Each such tile has a
    // unique owner, so reading local tiles and summing recovers the full vector on every rank
    // with no rank-0 gather and no Bcast.
    std::vector<double> coef_buf(n_active, 0.0);
    for (int64_t i = 0; i < mt; ++i) {
      if (tile_row0[i] >= n_active) break;
      if (!b_aug.tileIsLocal(i, 0)) continue;
      auto tile = b_aug(i, 0);
      const int64_t h = tile_height[i];
      for (int64_t r = 0; r < h; ++r) {
        const int64_t gi = tile_row0[i] + r;
        if (gi < n_active) coef_buf[gi] = tile.at(r, 0);
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, coef_buf.data(), n_active, MPI_DOUBLE, MPI_SUM, comm);
    for (int64_t i = 0; i < n_active; ++i) local_coef_active[i] = coef_buf[i];

    // -------------------------------- SSE --------------------------------
    // SSE = ||d2||^2 - (1/alpha) sum_i lambda_i coef_i^2  (penalty <= ||d2||^2; clamp roundoff).
    double penalty = 0.0;
    for (int64_t i = 0; i < n_active; ++i)
      penalty += lambda_active[i] * local_coef_active[i] * local_coef_active[i];
    penalty /= alpha;
    double sse = aug_resid_sq - penalty;
    *local_sse = (sse > 0.0) ? sse : 0.0;

    // cond_number, sigma_diag, coef, sse are all global (collective norms / Allreduce), so every
    // rank returns identical values -- no MPI_Bcast needed.
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


