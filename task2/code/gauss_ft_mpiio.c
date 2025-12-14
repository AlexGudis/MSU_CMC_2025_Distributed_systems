#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

#ifdef USE_ULFM
#include <mpi-ext.h>   // MPIX_...
#endif

// -------------------- util --------------------
static double wtime(void) { return MPI_Wtime(); }

static void die(const char* msg, MPI_Comm comm) {
  int r = -1;
  if (comm != MPI_COMM_NULL) MPI_Comm_rank(comm, &r);
  fprintf(stderr, "[rank %d] %s\n", r, msg);
  fflush(stderr);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

static void die_mpi(const char* where, int rc, MPI_Comm comm) {
  char err[MPI_MAX_ERROR_STRING];
  int len = 0;
  MPI_Error_string(rc, err, &len);
  int r = -1;
  if (comm != MPI_COMM_NULL) MPI_Comm_rank(comm, &r);
  fprintf(stderr, "[rank %d] %s: %s\n", r, where, err);
  fflush(stderr);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

// -------------------- distribution --------------------
// block distribution by rows
static void block_decomp(int N, int P, int r, int* r0, int* rn) {
  int base = N / P, rem = N % P;
  int nloc = base + (r < rem ? 1 : 0);
  int start = r * base + (r < rem ? r : rem);
  *r0 = start;
  *rn = nloc;
}

static int owner_of_row(int N, int P, int row) {
  int base = N / P, rem = N % P;
  int cut = (base + 1) * rem; // rows handled by first rem ranks
  if (row < cut) return row / (base + 1);
  return rem + (row - cut) / base;
}

// -------------------- checkpoint format --------------------
// File layout (bytes):
// [header]
//   int magic = 0x47555353 ('GUSS')
//   int version = 1
//   int N
//   int phase  (0=ELIM, 1=BACK)
//   int step   (current i or j)
// [matrix]
//   float A[ N * (N+1) ] in row-major
//
// Each rank writes its owned rows collectively using MPI_File_write_at_all.

enum { PH_ELIM = 0, PH_BACK = 1 };

typedef struct {
  int magic, version, N, phase, step;
} ckpt_hdr;

static MPI_Offset hdr_size_bytes(void) { return (MPI_Offset)sizeof(ckpt_hdr); }

static MPI_Offset matrix_offset_bytes(int N, int row0) {
  return hdr_size_bytes()
       + (MPI_Offset)row0 * (MPI_Offset)(N + 1) * (MPI_Offset)sizeof(float);
}

static void ckpt_write(MPI_Comm comm, const char* path,
                       int N, int phase, int step,
                       int row0, int nloc, const float* Aloc) {
  int rank;
  MPI_Comm_rank(comm, &rank);

  MPI_File fh;
  int rc = MPI_File_open(comm, path,
                         MPI_MODE_CREATE | MPI_MODE_WRONLY,
                         MPI_INFO_NULL, &fh);
  if (rc != MPI_SUCCESS) die_mpi("MPI_File_open(write)", rc, comm);

  // Rank 0 writes header (non-collective, but that's OK: file is already opened collectively)
  if (rank == 0) {
    ckpt_hdr h;
    h.magic = 0x47555353;
    h.version = 1;
    h.N = N;
    h.phase = phase;
    h.step = step;
    rc = MPI_File_write_at(fh, 0, &h, (int)sizeof(h), MPI_BYTE, MPI_STATUS_IGNORE);
    if (rc != MPI_SUCCESS) die_mpi("MPI_File_write_at(header)", rc, comm);
  }

  // Ensure header is visible before collective write
  rc = MPI_Barrier(comm);
  if (rc != MPI_SUCCESS) die_mpi("MPI_Barrier(before matrix write)", rc, comm);

  MPI_Offset off = matrix_offset_bytes(N, row0);
  rc = MPI_File_write_at_all(fh, off, (void*)Aloc,
                             nloc * (N + 1),
                             MPI_FLOAT, MPI_STATUS_IGNORE);
  if (rc != MPI_SUCCESS) die_mpi("MPI_File_write_at_all(matrix)", rc, comm);

  rc = MPI_File_close(&fh);
  if (rc != MPI_SUCCESS) die_mpi("MPI_File_close(write)", rc, comm);
}

static int ckpt_read(MPI_Comm comm, const char* path,
                     int* N, int* phase, int* step,
                     int row0, int nloc, float* Aloc) {
  int rank;
  MPI_Comm_rank(comm, &rank);

  MPI_File fh;
  int rc = MPI_File_open(comm, path, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
  if (rc != MPI_SUCCESS) return 0; // no checkpoint

  ckpt_hdr h;
  if (rank == 0) {
    rc = MPI_File_read_at(fh, 0, &h, (int)sizeof(h), MPI_BYTE, MPI_STATUS_IGNORE);
    if (rc != MPI_SUCCESS) die_mpi("MPI_File_read_at(header)", rc, comm);
  }
  rc = MPI_Bcast(&h, (int)sizeof(h), MPI_BYTE, 0, comm);
  if (rc != MPI_SUCCESS) die_mpi("MPI_Bcast(header)", rc, comm);

  if (h.magic != 0x47555353 || h.version != 1) {
    MPI_File_close(&fh);
    die("Checkpoint has wrong format", comm);
  }

  *N = h.N; *phase = h.phase; *step = h.step;

  MPI_Offset off = matrix_offset_bytes(h.N, row0);
  rc = MPI_File_read_at_all(fh, off, Aloc,
                            nloc * (h.N + 1),
                            MPI_FLOAT, MPI_STATUS_IGNORE);
  if (rc != MPI_SUCCESS) die_mpi("MPI_File_read_at_all(matrix)", rc, comm);

  rc = MPI_File_close(&fh);
  if (rc != MPI_SUCCESS) die_mpi("MPI_File_close(read)", rc, comm);

  return 1;
}

// -------------------- Gauss (distributed) --------------------
static void gauss_run(MPI_Comm comm, int N,
                      int* phase_io, int* step_io,
                      int row0, int nloc, float* A,
                      const char* ckpt_path, int ckpt_period) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  float* pivot = (float*)malloc((size_t)(N + 1) * sizeof(float));
  if (!pivot) die("malloc pivot failed", comm);

  // failure injection: env FAIL_RANK, FAIL_STEP, FAIL_PHASE (0/1)
  int fail_rank = -1, fail_step = -1, fail_phase = -1;
  const char* s;
  if ((s = getenv("FAIL_RANK")))  fail_rank  = atoi(s);
  if ((s = getenv("FAIL_STEP")))  fail_step  = atoi(s);
  if ((s = getenv("FAIL_PHASE"))) fail_phase = atoi(s);

  int phase = *phase_io;
  int step  = *step_io;

  // -------- elimination --------
  if (phase == PH_ELIM) {
    for (int i = step; i < N - 1; i++) {
      if (rank == fail_rank && fail_phase == PH_ELIM && fail_step == i) {
        raise(SIGKILL);
      }

      int owner = owner_of_row(N, size, i);

      if (rank == owner) {
        int li = i - row0;
        memcpy(pivot, &A[li * (N + 1)], (size_t)(N + 1) * sizeof(float));
      }

      int rc = MPI_Bcast(pivot, N + 1, MPI_FLOAT, owner, comm);
      if (rc != MPI_SUCCESS) die_mpi("MPI_Bcast(pivot)", rc, comm);

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
        ckpt_write(comm, ckpt_path, N, PH_ELIM, i + 1, row0, nloc, A);
      }
    }

    phase = PH_BACK;
    step  = N - 2;
    ckpt_write(comm, ckpt_path, N, phase, step, row0, nloc, A);
  }

  // -------- back substitution --------
  float* X = (float*)calloc((size_t)N, sizeof(float));
  if (!X) die("malloc X failed", comm);

  int owner_last = owner_of_row(N, size, N - 1);
  if (rank == owner_last) {
    int ll = (N - 1) - row0;
    X[N - 1] = A[ll * (N + 1) + N] / A[ll * (N + 1) + (N - 1)];
  }
  int rc = MPI_Bcast(&X[N - 1], 1, MPI_FLOAT, owner_last, comm);
  if (rc != MPI_SUCCESS) die_mpi("MPI_Bcast(X[N-1])", rc, comm);

  for (int j = step; j >= 0; j--) {
    if (rank == fail_rank && fail_phase == PH_BACK && fail_step == j) {
      raise(SIGKILL);
    }

    for (int lk = 0; lk < nloc; lk++) {
      int k = row0 + lk;
      if (k > j) continue;
      A[lk * (N + 1) + N] -= A[lk * (N + 1) + (j + 1)] * X[j + 1];
    }

    int owner = owner_of_row(N, size, j);
    if (rank == owner) {
      int lj = j - row0;
      X[j] = A[lj * (N + 1) + N] / A[lj * (N + 1) + j];
    }
    rc = MPI_Bcast(&X[j], 1, MPI_FLOAT, owner, comm);
    if (rc != MPI_SUCCESS) die_mpi("MPI_Bcast(X[j])", rc, comm);

    if (ckpt_period > 0 && (j % ckpt_period == 0)) {
      ckpt_write(comm, ckpt_path, N, PH_BACK, j - 1, row0, nloc, A);
    }
  }

  if (rank == 0) {
    printf("X=(");
    int m = (N > 9 ? 9 : N);
    for (int i = 0; i < m; i++) {
      printf("%.4g%s", X[i], (i % 10 == 9 ? "\n" : ", "));
    }
    printf("...)\n");
    fflush(stdout);
  }

  free(X);
  free(pivot);
}

// -------------------- recovery: ULFM + spawn --------------------
static MPI_Comm recover_comm_spawn(MPI_Comm comm, int target_size, const char* self_exe) {
#ifndef USE_ULFM
  (void)comm; (void)target_size; (void)self_exe;
  die("Build with -DUSE_ULFM and ULFM-enabled MPI to use scenario (b)", comm);
  return MPI_COMM_NULL;
#else
  // Unblock everyone
  MPIX_Comm_revoke(comm);

  // Get communicator of survivors
  MPI_Comm alive;
  MPIX_Comm_shrink(comm, &alive);

  int alive_size;
  MPI_Comm_size(alive, &alive_size);

  int missing = target_size - alive_size;
  if (missing <= 0) {
    return alive; // already full (or over)
  }

  // Spawn missing processes; pass target_size as argv[1]
  char target_str[32];
  snprintf(target_str, sizeof(target_str), "%d", target_size);
  char* child_argv[] = { (char*)self_exe, target_str, NULL };

  MPI_Comm inter;
  int rc = MPI_Comm_spawn(self_exe, child_argv, missing,
                          MPI_INFO_NULL, 0, alive, &inter, MPI_ERRCODES_IGNORE);
  if (rc != MPI_SUCCESS) die_mpi("MPI_Comm_spawn", rc, alive);

  // Merge alive + spawned into one intracommunicator
  MPI_Comm merged;
  rc = MPI_Intercomm_merge(inter, 0, &merged); // high=0 for parent
  if (rc != MPI_SUCCESS) die_mpi("MPI_Intercomm_merge(parent)", rc, alive);

  MPI_Comm_free(&inter);
  MPI_Comm_free(&alive);

  return merged;
#endif
}

// -------------------- main --------------------
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // Determine if this is a spawned child
  MPI_Comm parent = MPI_COMM_NULL;
  MPI_Comm_get_parent(&parent);

  MPI_Comm comm = MPI_COMM_WORLD;

  // Important: return errors instead of aborting immediately on MPI calls
  MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);

  // target_size is the desired "full" size (initial size of the job)
  int target_size = -1;

  // self executable path for spawn
  // We want an executable name that mpirun can find. Usually "./gauss_ft_mpiio".
  // If you run with absolute path, you can put it here too.
  const char* self_exe = "./gauss_ft_mpiio";

#ifdef USE_ULFM
  if (parent != MPI_COMM_NULL) {
    // Spawned child: argv[1] contains target_size
    if (argc < 2) die("Spawned child: missing target_size argument", comm);
    target_size = atoi(argv[1]);

    MPI_Comm merged;
    int rc = MPI_Intercomm_merge(parent, 1, &merged); // high=1 for child
    if (rc != MPI_SUCCESS) die_mpi("MPI_Intercomm_merge(child)", rc, comm);

    MPI_Comm_free(&parent);
    comm = merged;

    // Set errhandler on the merged communicator too
    MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);
  } else
#endif
  {
    // Normal start
    MPI_Comm_size(comm, &target_size);
  }

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // Checkpoint path: use local FS to avoid ROMIO/FS issues
  const char* ckpt_path = "/tmp/gauss.ckpt";
  int ckpt_period = 2;

  // Read N from data.in on rank 0, then broadcast
  int N = 0;
  if (rank == 0) {
    FILE* in = fopen("data.in", "r");
    if (!in) die("Cannot open data.in", comm);
    if (fscanf(in, "%d", &N) != 1) die("Wrong data.in", comm);
    fclose(in);
  }
  int rc = MPI_Bcast(&N, 1, MPI_INT, 0, comm);
  if (rc != MPI_SUCCESS) die_mpi("MPI_Bcast(N)", rc, comm);

  // Setup local storage
  int row0, nloc;
  block_decomp(N, size, rank, &row0, &nloc);

  float* A = (float*)malloc((size_t)nloc * (size_t)(N + 1) * sizeof(float));
  if (!A) die("malloc A failed", comm);

  // Try load checkpoint; if absent -> init matrix and write initial checkpoint (collectively!)
  int phase = PH_ELIM, step = 0;
  int has = ckpt_read(comm, ckpt_path, &N, &phase, &step, row0, nloc, A);

  if (!has) {
    for (int lk = 0; lk < nloc; lk++) {
      int i = row0 + lk;
      for (int j = 0; j <= N; j++) {
        float v = (i == j || j == N) ? 1.f : 0.f;
        A[lk * (N + 1) + j] = v;
      }
    }
    ckpt_write(comm, ckpt_path, N, phase, step, row0, nloc, A);
  }

  double t0 = wtime();

  // Main compute with failure-handling loop:
  // If any MPI collective fails -> recover communicator, reload checkpoint, continue.
  while (1) {
    // Run computation (may fail due to process death)
    gauss_run(comm, N, &phase, &step, row0, nloc, A, ckpt_path, ckpt_period);

    // If everything is fine, barrier succeeds
    rc = MPI_Barrier(comm);
    if (rc == MPI_SUCCESS) break;

#ifdef USE_ULFM
    // Recover communicator to full size by spawning missing ranks
    MPI_Comm newcomm = recover_comm_spawn(comm, target_size, self_exe);

    // Switch communicator
    comm = newcomm;
    MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);

    // Update rank/size and redistribute rows
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    block_decomp(N, size, rank, &row0, &nloc);

    free(A);
    A = (float*)malloc((size_t)nloc * (size_t)(N + 1) * sizeof(float));
    if (!A) die("malloc A after recover failed", comm);

    int ok = ckpt_read(comm, ckpt_path, &N, &phase, &step, row0, nloc, A);
    if (!ok) die("Checkpoint missing after recovery", comm);
#else
    die("MPI failure detected but USE_ULFM not enabled", comm);
#endif
  }

  double t1 = wtime();
  if (rank == 0) printf("Time in seconds=%gs\n", (t1 - t0));

  free(A);

  // If comm != MPI_COMM_WORLD due to merge, freeing it is polite
  if (comm != MPI_COMM_WORLD) MPI_Comm_free(&comm);

  MPI_Finalize();
  return 0;
}
