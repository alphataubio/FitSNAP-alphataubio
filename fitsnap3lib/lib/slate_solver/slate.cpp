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

// active set is monotone => n_active identifies it (see note); per-process cache
static int64_t            g_ard_cached_n = -1;
static std::vector<double> g_ard_cached_Rx;   // n x n col-major upper-tri (replicated)

double slate_ard_update(double* local_aw_active, double* local_bw,
                        double* local_sigma_diag, double* local_coef_active,
                        double* local_sse,
                        int64_t m, int64_t n_active, int64_t lld,
                        int64_t row_offset, int64_t m_local,
                        double alpha, double* lambda_active,
                        MPI_Comm comm, int debug) {
  (void)m;
  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(comm, &mpi_rank);
  MPI_Comm_size(comm, &mpi_size);
  if (n_active == 0) { if (local_sse) *local_sse = 0.0; return 1.0; }

  const int64_t n = n_active, nn = n * n;
  const blas::Layout CM = blas::Layout::ColMajor;
  double* Xp = local_aw_active + row_offset;   // &aw[row_offset,0], col-major, ld=lld
  double* yp = local_bw       + row_offset;    // &bw[row_offset]

  if (debug) {
  double xf2 = 0.0;
  if (m_local > 0)
    for (int64_t j = 0; j < n_active; ++j)
      for (int64_t i = 0; i < m_local; ++i) { double v = (local_aw_active + row_offset)[i + j*lld]; xf2 += v*v; }
  double yn2 = (m_local > 0) ? blas::dot(m_local, local_bw + row_offset, 1, local_bw + row_offset, 1) : 0.0;
  std::vector<long long> off(mpi_size), mlo(mpi_size), ld(mpi_size);
  long long a = row_offset, b = m_local, c = lld;
  MPI_Gather(&a,1,MPI_LONG_LONG,off.data(),1,MPI_LONG_LONG,0,comm);
  MPI_Gather(&b,1,MPI_LONG_LONG,mlo.data(),1,MPI_LONG_LONG,0,comm);
  MPI_Gather(&c,1,MPI_LONG_LONG,ld.data(), 1,MPI_LONG_LONG,0,comm);
  std::fprintf(stderr,"[rank %d] off=%lld m_local=%lld lld=%lld ||Xp||_F^2=%.6e ||yp||^2=%.6e\n",
               mpi_rank,(long long)row_offset,(long long)m_local,(long long)lld,xf2,yn2);
  if (mpi_rank==0) for (int r=0;r<mpi_size;++r)
    std::fprintf(stderr,"  PART rank %2d: off=%lld m_local=%lld end=%lld lld=%lld\n",
                 r,off[r],mlo[r],off[r]+mlo[r],ld[r]);
}

  // ---- h = X^T y  and  ynorm2 = ||y||^2   (gemv reads aw BEFORE any geqrf overwrite) ----
  std::vector<double> h(n, 0.0);
  double ynorm2_local = 0.0;
  if (m_local > 0) {
    blas::gemv(CM, blas::Op::Trans, m_local, n, 1.0, Xp, lld, yp, 1, 0.0, h.data(), 1);
    ynorm2_local = blas::dot(m_local, yp, 1, yp, 1);
  }
  MPI_Allreduce(MPI_IN_PLACE, h.data(), (int)n, MPI_DOUBLE, MPI_SUM, comm);
  double ynorm2 = 0.0;
  MPI_Allreduce(&ynorm2_local, &ynorm2, 1, MPI_DOUBLE, MPI_SUM, comm);

  // ---- R_x : R_x^T R_x = X^T X  via local QR + TSQR tree (R only; cached) ----
  std::vector<double> Rx;
  const bool have_cache = (n == g_ard_cached_n) && ((int64_t)g_ard_cached_Rx.size() == nn);
  if (have_cache) {
    Rx = g_ard_cached_Rx;
  } else {
    std::vector<double> Rloc(nn, 0.0);
    if (m_local > 0) {
      const int64_t k = std::min<int64_t>(m_local, n);
      std::vector<double> tau(k);
      lapack::geqrf(m_local, n, Xp, lld, tau.data());          // R in upper(Xp)
      for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i <= std::min<int64_t>(j, k - 1); ++i)
          Rloc[i + j*n] = Xp[i + j*lld];
    }
    std::vector<double> Rrecv(nn), stack(2*nn);
    for (int64_t mask = 1; mask < mpi_size; mask <<= 1) {
      if (mpi_rank & mask) {
        MPI_Send(Rloc.data(), (int)nn, MPI_DOUBLE, mpi_rank - (int)mask, 0, comm);
        break;
      }
      int src = mpi_rank + (int)mask;
      if (src >= mpi_size) continue;
      MPI_Recv(Rrecv.data(), (int)nn, MPI_DOUBLE, src, 0, comm, MPI_STATUS_IGNORE);
      for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < n; ++i) stack[i     + j*2*n] = Rloc[i  + j*n];
        for (int64_t i = 0; i < n; ++i) stack[(n+i) + j*2*n] = Rrecv[i + j*n];
      }
      std::vector<double> tau(n);
      lapack::geqrf(2*n, n, stack.data(), 2*n, tau.data());
      std::fill(Rloc.begin(), Rloc.end(), 0.0);
      for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i <= j; ++i) Rloc[i + j*n] = stack[i + j*2*n];
    }
    MPI_Bcast(Rloc.data(), (int)nn, MPI_DOUBLE, 0, comm);
    Rx = std::move(Rloc);
    g_ard_cached_n = n;  g_ard_cached_Rx = Rx;
  }

  // ---- final R: QR([ R_x ; diag(sqrt(lambda/alpha)) ]);  R^T R = X^T X + diag(lambda/alpha) ----
  const double inv_sqrt_alpha = 1.0 / std::sqrt(alpha);
  std::vector<double> aug(2*nn, 0.0);
  for (int64_t j = 0; j < n; ++j) {
    for (int64_t i = 0; i <= j; ++i) aug[i + j*2*n] = Rx[i + j*n];
    aug[(n + j) + j*2*n] = std::sqrt(lambda_active[j]) * inv_sqrt_alpha;
  }
  { std::vector<double> tau(n); lapack::geqrf(2*n, n, aug.data(), 2*n, tau.data()); }
  std::vector<double> R(nn, 0.0);
  for (int64_t j = 0; j < n; ++j)
    for (int64_t i = 0; i <= j; ++i) R[i + j*n] = aug[i + j*2*n];

  // ---- R^{-1};  coef = R^{-1} R^{-T} h;  sigma_diag;  cond ----
  std::vector<double> Rinv = R;
  lapack::trtri(lapack::Uplo::Upper, lapack::Diag::NonUnit, n, Rinv.data(), n);

  std::vector<double> t(n, 0.0), coef(n, 0.0);
  blas::gemv(CM, blas::Op::Trans,   n, n, 1.0, Rinv.data(), n, h.data(), 1, 0.0, t.data(),    1); // t = R^{-T} h
  blas::gemv(CM, blas::Op::NoTrans, n, n, 1.0, Rinv.data(), n, t.data(), 1, 0.0, coef.data(), 1); // c = R^{-1} t
  for (int64_t i = 0; i < n; ++i) local_coef_active[i] = coef[i];

  for (int64_t i = 0; i < n; ++i) {                          // sigma_ii = (1/alpha)||row i of R^{-1}||^2
    double s = 0.0;
    for (int64_t j = i; j < n; ++j) { double v = Rinv[i + j*n]; s += v*v; }
    local_sigma_diag[i] = s / alpha;
  }

  double R_inf = 0.0, Ri_inf = 0.0;                          // cond_inf(R) = ||R||_inf ||R^{-1}||_inf
  for (int64_t i = 0; i < n; ++i) {
    double sR = 0.0, sI = 0.0;
    for (int64_t j = i; j < n; ++j) { sR += std::fabs(R[i+j*n]); sI += std::fabs(Rinv[i+j*n]); }
    if (sR > R_inf)  R_inf  = sR;
    if (sI > Ri_inf) Ri_inf = sI;
  }
  double cond_number = R_inf * Ri_inf;

  // ---- SSE = ||y||^2 - ||t||^2 - (1/alpha) sum lambda*coef^2 ----
  double tnorm2 = blas::dot(n, t.data(), 1, t.data(), 1);
  double penalty = 0.0;
  for (int64_t i = 0; i < n; ++i) penalty += lambda_active[i] * coef[i] * coef[i];
  penalty /= alpha;
  double sse = ynorm2 - tnorm2 - penalty;
  if (local_sse) *local_sse = (sse > 0.0) ? sse : 0.0;

  if (debug && mpi_rank == 0)
    std::fprintf(stderr, "*** ARD n=%lld %s cond=%.3e ||y||^2=%.3e ||t||^2=%.3e penalty=%.3e sse=%.3e\n",
                 (long long)n, have_cache?"(cached)":"(recomp)", cond_number, ynorm2, tnorm2, penalty, sse);
  return cond_number;
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


