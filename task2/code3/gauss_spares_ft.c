#define _GNU_SOURCE
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

#ifdef USE_ULFM
#include <mpi-ext.h>
#endif

enum { PH_ELIM = 0, PH_BACK = 1 };

typedef struct {
  int magic;   // 'GUSS' = 0x47555353
  int version; // 1
  int N;
  int phase;
  int step;
} ckpt_hdr;

static void die(const char* msg) {
  fprintf(stderr, "%s\n", msg);
  fflush(stderr);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

static void die_mpi(const char* where, int rc) {
  char err[MPI_MAX_ERROR_STRING];
  int len = 0;
  MPI_Error_string(rc, err, &len);
  fprintf(stderr, "%s: %s\n", where, err);
  fflush(stderr);
  MPI_Abort(MPI_COMM_WORLD, 2);
}

#ifdef USE_ULFM
static int is_ulfm_failure(int rc) {
  if (rc == MPI_SUCCESS) return 0;
  int eclass = MPI_ERR_OTHER;
  MPI_Error_class(rc, &eclass);
  return (eclass == MPIX_ERR_PROC_FAILED || eclass == MPIX_ERR_REVOKED);
}
#endif

// ---------- distribution ----------
static void block_decomp(int N, int P, int r, int* r0, int* rn) {
  int base = N / P, rem = N % P;
  int nloc = base + (r < rem ? 1 : 0);
  int start = r * base + (r < rem ? r : rem);
  *r0 = start;
  *rn = nloc;
}

static int owner_of_row(int N, int P, int row) {
  int base = N / P, rem = N % P;
  int cut = (base + 1) * rem;
  if (row < cut) return row / (base + 1);
  return rem + (row - cut) / base;
}

static void make_counts_displs(int N, int P, int* counts, int* displs) {
  int off = 0;
  for (int r = 0; r < P; r++) {
    int r0, rn;
    block_decomp(N, P, r, &r0, &rn);
    counts[r] = rn * (N + 1);
    displs[r] = off;
    off += counts[r];
  }
}

// ---------- MPI-IO checkpoint (rank0 of ACTIVE only, via MPI_COMM_SELF) ----------
static void ckpt_write_root_self(const char* path, int N, int phase, int step, const float* Afull) {
  MPI_File fh;
  int rc = MPI_File_open(MPI_COMM_SELF, path,
                         MPI_MODE_CREATE | MPI_MODE_WRONLY,
                         MPI_INFO_NULL, &fh);
  if (rc != MPI_SUCCESS) die_mpi("MPI_File_open(write,self)", rc);

  ckpt_hdr h = {0};
  h.magic = 0x47555353;
  h.version = 1;
  h.N = N;
  h.phase = phase;
  h.step = step;

  rc = MPI_File_write_at(fh, 0, &h, (int)sizeof(h), MPI_BYTE, MPI_STATUS_IGNORE);
  if (rc != MPI_SUCCESS) die_mpi("MPI_File_write_at(header)", rc);

  MPI_Offset off = (MPI_Offset)sizeof(h);
  rc = MPI_File_write_at(fh, off, (void*)Afull, N * (N + 1), MPI_FLOAT, MPI_STATUS_IGNORE);
  if (rc != MPI_SUCCESS) die_mpi("MPI_File_write_at(matrix)", rc);

  rc = MPI_File_close(&fh);
  if (rc != MPI_SUCCESS) die_mpi("MPI_File_close(write)", rc);
}

static int ckpt_read_root_self(const char* path, int* N, int* phase, int* step, float* Afull) {
  MPI_File fh;
  int rc = MPI_File_open(MPI_COMM_SELF, path, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
  if (rc != MPI_SUCCESS) return 0;

  ckpt_hdr h = {0};
  rc = MPI_File_read_at(fh, 0, &h, (int)sizeof(h), MPI_BYTE, MPI_STATUS_IGNORE);
  if (rc != MPI_SUCCESS) die_mpi("MPI_File_read_at(header)", rc);

  if (h.magic != 0x47555353 || h.version != 1) die("Checkpoint format error");

  *N = h.N; *phase = h.phase; *step = h.step;

  MPI_Offset off = (MPI_Offset)sizeof(h);
  rc = MPI_File_read_at(fh, off, Afull, h.N * (h.N + 1), MPI_FLOAT, MPI_STATUS_IGNORE);
  if (rc != MPI_SUCCESS) die_mpi("MPI_File_read_at(matrix)", rc);

  rc = MPI_File_close(&fh);
  if (rc != MPI_SUCCESS) die_mpi("MPI_File_close(read)", rc);

  return 1;
}

// ---------- Checkpoint save/load over ACTIVE communicator ----------
static int checkpoint_save(MPI_Comm active, const char* path,
                           int N, int phase, int step,
                           const float* Aloc, int nloc,
                           int* counts, int* displs) {
  int arank;
  MPI_Comm_rank(active, &arank);

  float* Afull = NULL;
  if (arank == 0) {
    Afull = (float*)malloc((size_t)N * (size_t)(N + 1) * sizeof(float));
    if (!Afull) die("malloc Afull failed");
  }

  int rc = MPI_Gatherv((void*)Aloc, nloc * (N + 1), MPI_FLOAT,
                       Afull, counts, displs, MPI_FLOAT, 0, active);
  if (rc != MPI_SUCCESS) { if (Afull) free(Afull); return rc; }

  if (arank == 0) {
    ckpt_write_root_self(path, N, phase, step, Afull);
    free(Afull);
  }

  rc = MPI_Barrier(active);
  return rc;
}

static int checkpoint_load(MPI_Comm active, const char* path,
                           int* N, int* phase, int* step,
                           float* Aloc, int nloc,
                           int* counts, int* displs) {
  int arank;
  MPI_Comm_rank(active, &arank);

  float* Afull = NULL;
  int N0 = *N, ph0 = *phase, st0 = *step;

  if (arank == 0) {
    Afull = (float*)malloc((size_t)(*N) * (size_t)((*N) + 1) * sizeof(float));
    if (!Afull) die("malloc Afull failed(load)");
    int ok = ckpt_read_root_self(path, &N0, &ph0, &st0, Afull);
    if (!ok) die("Checkpoint missing on load");
  }

  int rc = MPI_Bcast(&N0, 1, MPI_INT, 0, active);
  if (rc != MPI_SUCCESS) { if (Afull) free(Afull); return rc; }
  rc = MPI_Bcast(&ph0, 1, MPI_INT, 0, active);
  if (rc != MPI_SUCCESS) { if (Afull) free(Afull); return rc; }
  rc = MPI_Bcast(&st0, 1, MPI_INT, 0, active);
  if (rc != MPI_SUCCESS) { if (Afull) free(Afull); return rc; }

  *N = N0; *phase = ph0; *step = st0;

  rc = MPI_Scatterv(Afull, counts, displs, MPI_FLOAT,
                    Aloc, nloc * (N0 + 1), MPI_FLOAT, 0, active);

  if (arank == 0) free(Afull);
  return rc;
}

// ---------- Gauss on ACTIVE communicator; returns MPI error code if any collective failed ----------
static int gauss_run(MPI_Comm active, int N,
                     int* phase_io, int* step_io,
                     int row0, int nloc, float* A,
                     const char* ckpt_path, int ckpt_period,
                     int* counts, int* displs) {
  int arank, asize;
  MPI_Comm_rank(active, &arank);
  MPI_Comm_size(active, &asize);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Failure injection uses WORLD rank
  int fail_rank = -1, fail_step = -1, fail_phase = -1;
  const char* s;
  if ((s = getenv("FAIL_RANK")))  fail_rank  = atoi(s);
  if ((s = getenv("FAIL_STEP")))  fail_step  = atoi(s);
  if ((s = getenv("FAIL_PHASE"))) fail_phase = atoi(s);

  float* pivot = (float*)malloc((size_t)(N + 1) * sizeof(float));
  if (!pivot) die("malloc pivot failed");

  int phase = *phase_io;
  int step  = *step_io;

  // elimination
  if (phase == PH_ELIM) {
    for (int i = step; i < N - 1; i++) {
      if (world_rank == fail_rank && fail_phase == PH_ELIM && fail_step == i) raise(SIGKILL);

      int owner = owner_of_row(N, asize, i);
      if (arank == owner) {
        int li = i - row0;
        memcpy(pivot, &A[li * (N + 1)], (size_t)(N + 1) * sizeof(float));
      }

      int rc = MPI_Bcast(pivot, N + 1, MPI_FLOAT, owner, active);
      if (rc != MPI_SUCCESS) { free(pivot); return rc; }

      for (int lk = 0; lk < nloc; lk++) {
        int k = row0 + lk;
        if (k <= i) continue;
        float aik = A[lk * (N + 1) + i];
        float pii = pivot[i];
        for (int j = i + 1; j <= N; j++) {
          A[lk * (N + 1) + j] -= aik * pivot[j] / pii;
        }
      }

      if (ckpt_period > 0 && (i % ckpt_period == 0)) {
        rc = checkpoint_save(active, ckpt_path, N, PH_ELIM, i + 1, A, nloc, counts, displs);
        if (rc != MPI_SUCCESS) { free(pivot); return rc; }
      }
    }

    phase = PH_BACK;
    step = N - 2;
    int rc = checkpoint_save(active, ckpt_path, N, phase, step, A, nloc, counts, displs);
    if (rc != MPI_SUCCESS) { free(pivot); return rc; }
  }

  // back substitution
  float* X = (float*)calloc((size_t)N, sizeof(float));
  if (!X) die("calloc X failed");

  int owner_last = owner_of_row(N, asize, N - 1);
  if (arank == owner_last) {
    int ll = (N - 1) - row0;
    X[N - 1] = A[ll * (N + 1) + N] / A[ll * (N + 1) + (N - 1)];
  }

  int rc = MPI_Bcast(&X[N - 1], 1, MPI_FLOAT, owner_last, active);
  if (rc != MPI_SUCCESS) { free(X); free(pivot); return rc; }

  for (int j = step; j >= 0; j--) {
    if (world_rank == fail_rank && fail_phase == PH_BACK && fail_step == j) raise(SIGKILL);

    for (int lk = 0; lk < nloc; lk++) {
      int k = row0 + lk;
      if (k > j) continue;
      A[lk * (N + 1) + N] -= A[lk * (N + 1) + (j + 1)] * X[j + 1];
    }

    int owner = owner_of_row(N, asize, j);
    if (arank == owner) {
      int lj = j - row0;
      X[j] = A[lj * (N + 1) + N] / A[lj * (N + 1) + j];
    }

    rc = MPI_Bcast(&X[j], 1, MPI_FLOAT, owner, active);
    if (rc != MPI_SUCCESS) { free(X); free(pivot); return rc; }

    if (ckpt_period > 0 && (j % ckpt_period == 0)) {
      rc = checkpoint_save(active, ckpt_path, N, PH_BACK, j - 1, A, nloc, counts, displs);
      if (rc != MPI_SUCCESS) { free(X); free(pivot); return rc; }
    }
  }

  if (arank == 0) {
    printf("X=(");
    int m = (N > 9 ? 9 : N);
    for (int i = 0; i < m; i++) printf("%.4g%s", X[i], (i % 10 == 9 ? "\n" : ", "));
    printf("...)\n");
    fflush(stdout);
  }

  free(X);
  free(pivot);
  return MPI_SUCCESS;
}

// Build ACTIVE from WORLD communicator: ranks 0..WORK-1 in WORLD become active
static MPI_Comm build_active_from_world(MPI_Comm world_comm, int WORK) {
  int wrank;
  MPI_Comm_rank(world_comm, &wrank);
  int color = (wrank < WORK) ? 1 : MPI_UNDEFINED;
  MPI_Comm active = MPI_COMM_NULL;
  int rc = MPI_Comm_split(world_comm, color, wrank, &active);
  if (rc != MPI_SUCCESS) die_mpi("MPI_Comm_split(active)", rc);
  return active;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  MPI_Comm world = MPI_COMM_WORLD;
  MPI_Comm_set_errhandler(world, MPI_ERRORS_RETURN);

  int world_rank, world_size;
  MPI_Comm_rank(world, &world_rank);
  MPI_Comm_size(world, &world_size);

  if (argc < 4) {
    if (world_rank == 0) {
      fprintf(stderr, "Usage: %s data.in ckpt_path WORK\n", argv[0]);
    }
    MPI_Finalize();
    return 1;
  }

  const char* in_path = argv[1];
  const char* ckpt_path = argv[2];
  int WORK = atoi(argv[3]);
  if (WORK <= 0 || WORK > world_size) die("Bad WORK value");

  // Build initial ACTIVE set from WORLD
  MPI_Comm active = build_active_from_world(world, WORK);
  if (active != MPI_COMM_NULL) MPI_Comm_set_errhandler(active, MPI_ERRORS_RETURN);

  int N = 0, phase = PH_ELIM, step = 0;
  int ckpt_period = 2;

  // ACTIVE rank0 reads N and broadcasts inside ACTIVE
  if (active != MPI_COMM_NULL) {
    int arank;
    MPI_Comm_rank(active, &arank);
    if (arank == 0) {
      FILE* in = fopen(in_path, "r");
      if (!in) die("Cannot open data.in");
      if (fscanf(in, "%d", &N) != 1) die("Wrong data.in");
      fclose(in);
    }
    int rc = MPI_Bcast(&N, 1, MPI_INT, 0, active);
    if (rc != MPI_SUCCESS) die_mpi("MPI_Bcast(N)", rc);
  }

  // Broadcast N to all WORLD so spares know sizes after rebuilds
  int rc = MPI_Bcast(&N, 1, MPI_INT, 0, world);
  if (rc != MPI_SUCCESS) die_mpi("MPI_Bcast(N,world)", rc);

  // Allocate buffers for ACTIVE only
  float* Aloc = NULL;
  int row0 = 0, nloc = 0;
  int* counts = NULL;
  int* displs = NULL;

  auto void active_alloc_rebuild(void) {
    int asize, arank;
    MPI_Comm_size(active, &asize);
    MPI_Comm_rank(active, &arank);

    block_decomp(N, asize, arank, &row0, &nloc);

    if (Aloc) free(Aloc);
    Aloc = (float*)malloc((size_t)nloc * (size_t)(N + 1) * sizeof(float));
    if (!Aloc) die("malloc Aloc failed");

    if (counts) free(counts);
    if (displs) free(displs);
    counts = (int*)malloc((size_t)asize * sizeof(int));
    displs = (int*)malloc((size_t)asize * sizeof(int));
    if (!counts || !displs) die("malloc counts/displs failed");
    make_counts_displs(N, asize, counts, displs);
  }

  if (active != MPI_COMM_NULL) {
    active_alloc_rebuild();

    // init identity + rhs = 1
    for (int lk = 0; lk < nloc; lk++) {
      int i = row0 + lk;
      for (int j = 0; j <= N; j++) {
        float v = (i == j || j == N) ? 1.f : 0.f;
        Aloc[lk * (N + 1) + j] = v;
      }
    }

    rc = checkpoint_save(active, ckpt_path, N, phase, step, Aloc, nloc, counts, displs);
    if (rc != MPI_SUCCESS) die_mpi("initial checkpoint_save", rc);
  }

  double t0 = MPI_Wtime();

  while (1) {
    int active_rc = MPI_SUCCESS;

    // ACTIVE does compute; SPARES do nothing here
    if (active != MPI_COMM_NULL) {
      active_rc = gauss_run(active, N, &phase, &step, row0, nloc, Aloc,
                            ckpt_path, ckpt_period, counts, displs);
#ifdef USE_ULFM
      if (active_rc != MPI_SUCCESS && is_ulfm_failure(active_rc)) {
        // revoke WORLD to pull everyone into recovery
        MPIX_Comm_revoke(world);
      }
#endif
    }

    // Everyone synchronizes at WORLD barrier.
    // If revoke happened, barrier will return an error and we go to recovery.
    rc = MPI_Barrier(world);
    if (rc == MPI_SUCCESS) {
      // no failures and everyone is fine
      break;
    }

#ifndef USE_ULFM
    die_mpi("WORLD barrier failed but USE_ULFM off", rc);
#else
    // If not a ULFM failure, stop (to avoid infinite loops)
    if (!is_ulfm_failure(rc)) die_mpi("WORLD barrier failed (not ULFM failure)", rc);

    // All survivors must shrink WORLD
    MPI_Comm world2;
    MPIX_Comm_shrink(world, &world2);
    world = world2;
    MPI_Comm_set_errhandler(world, MPI_ERRORS_RETURN);

    // Re-broadcast N inside new WORLD (rank0 in new world = old world rank among survivors)
    // We assume N is stable; broadcast current N.
    int rr;
    MPI_Comm_rank(world, &rr);
    rc = MPI_Bcast(&N, 1, MPI_INT, 0, world);
    if (rc != MPI_SUCCESS) die_mpi("MPI_Bcast(N after shrink)", rc);

    // Rebuild ACTIVE from new WORLD
    if (active != MPI_COMM_NULL) MPI_Comm_free(&active);
    active = build_active_from_world(world, WORK);
    if (active != MPI_COMM_NULL) {
      MPI_Comm_set_errhandler(active, MPI_ERRORS_RETURN);
      active_alloc_rebuild();

      // Load checkpoint and continue
      rc = checkpoint_load(active, ckpt_path, &N, &phase, &step, Aloc, nloc, counts, displs);
      if (rc != MPI_SUCCESS) die_mpi("checkpoint_load after rebuild", rc);
    } else {
      // This rank became spare in new world; just continue to next loop
    }
#endif
  }

  double t1 = MPI_Wtime();
  if (active != MPI_COMM_NULL) {
    int arank;
    MPI_Comm_rank(active, &arank);
    if (arank == 0) printf("Time in seconds=%gs\n", (t1 - t0));
  }

  if (Aloc) free(Aloc);
  if (counts) free(counts);
  if (displs) free(displs);
  if (active != MPI_COMM_NULL) MPI_Comm_free(&active);

  MPI_Finalize();
  return 0;
}
