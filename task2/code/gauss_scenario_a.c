#include <mpi.h>
#include <mpi-ext.h> 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <stdarg.h>
#include <unistd.h>    // getpid

enum { PH_ELIM = 0, PH_BACK = 1 };

typedef struct {
  int magic;
  int version;
  int N;
  int phase;
  int step;
} ckpt_hdr;

// ----------------  Лрогирование для тестов  локально ----------------
static int LOG = 1; // env GAUSS_LOG=0 выключает логи

static void log_init(void) {
  const char* s = getenv("GAUSS_LOG");
  if (s) LOG = atoi(s);
}

static void log0(MPI_Comm comm, const char* fmt, ...) {
  if (!LOG) return;
  int r = -1;
  if (comm != MPI_COMM_NULL) MPI_Comm_rank(comm, &r);
  double t = MPI_Wtime();
  fprintf(stderr, "[t=%8.3f r=%d] ", t, r);
  va_list ap; va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fprintf(stderr, "\n");
  fflush(stderr);
}

// ---------------- виды ошибок ----------------
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

static int is_ulfm_failure(int rc) {
  if (rc == MPI_SUCCESS) return 0;
  int eclass = MPI_ERR_OTHER;
  MPI_Error_class(rc, &eclass);
  return (eclass == MPIX_ERR_PROC_FAILED || eclass == MPIX_ERR_REVOKED);
}

static int is_proc_failed(int rc) {
  if (rc == MPI_SUCCESS) return 0;
  int eclass = MPI_ERR_OTHER;
  MPI_Error_class(rc, &eclass);
  return (eclass == MPIX_ERR_PROC_FAILED);
}

// условно Надежно revoke: кто первым увидел падение процесса, тот и вызывает revoke
static int maybe_revoke_on_failure(MPI_Comm world, int rc,
                                  int* revoked_flag,
                                  int last_phase, int last_step) {
  if (*revoked_flag) return 0;
  if (!is_proc_failed(rc)) return 0;

  *revoked_flag = 1;
  int wr = -1;
  MPI_Comm_rank(world, &wr);
  log0(world, "PROC_FAILED noticed by world_rank=%d at phase=%d step=%d -> revoke",
       wr, last_phase, last_step);
  MPIX_Comm_revoke(world);
  return 1;
}

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

// Сохраняю чекпоин в файл
static int ckpt_write_parallel(MPI_Comm active, const char* path,
                               int N, int phase, int step,
                               int row0, int nloc,
                               const float* Aloc) {
  MPI_File fh;
  int rc = MPI_File_open(active, path,
                         MPI_MODE_CREATE | MPI_MODE_WRONLY,
                         MPI_INFO_NULL, &fh);
  if (rc != MPI_SUCCESS) return rc;

  int arank;
  MPI_Comm_rank(active, &arank);

  if (arank == 0) {
    ckpt_hdr h = {0};
    h.magic = 0x47555353;
    h.version = 1;
    h.N = N;
    h.phase = phase;
    h.step = step;
    rc = MPI_File_write_at(fh, 0, &h, (int)sizeof(h), MPI_BYTE, MPI_STATUS_IGNORE);
    if (rc != MPI_SUCCESS) { MPI_File_close(&fh); return rc; }
  }

  rc = MPI_Barrier(active);
  if (rc != MPI_SUCCESS) { MPI_File_close(&fh); return rc; }

  MPI_Offset total_bytes =
    (MPI_Offset)sizeof(ckpt_hdr) +
    (MPI_Offset)N * (MPI_Offset)(N + 1) * (MPI_Offset)sizeof(float);

  rc = MPI_File_set_size(fh, total_bytes);
  if (rc != MPI_SUCCESS) { MPI_File_close(&fh); return rc; }

  rc = MPI_Barrier(active);
  if (rc != MPI_SUCCESS) { MPI_File_close(&fh); return rc; }

  MPI_Offset off =
    (MPI_Offset)sizeof(ckpt_hdr) +
    (MPI_Offset)row0 * (MPI_Offset)(N + 1) * (MPI_Offset)sizeof(float);

  int count = nloc * (N + 1);
  rc = MPI_File_write_at_all(fh, off, (void*)Aloc, count, MPI_FLOAT, MPI_STATUS_IGNORE);
  if (rc != MPI_SUCCESS) { MPI_File_close(&fh); return rc; }

  rc = MPI_File_sync(fh);
  if (rc != MPI_SUCCESS) { MPI_File_close(&fh); return rc; }

  rc = MPI_File_close(&fh);
  return rc;
}

// Чтение сохраненного чек-поинта
static int ckpt_read_parallel(MPI_Comm active, const char* path,
                              int* N, int* phase, int* step,
                              int row0, int nloc,
                              float* Aloc) {
  MPI_File fh;
  int rc = MPI_File_open(active, path, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
  if (rc != MPI_SUCCESS) return rc;

  int arank;
  MPI_Comm_rank(active, &arank);

  ckpt_hdr h = {0};
  if (arank == 0) {
    rc = MPI_File_read_at(fh, 0, &h, (int)sizeof(h), MPI_BYTE, MPI_STATUS_IGNORE);
    if (rc != MPI_SUCCESS) { MPI_File_close(&fh); return rc; }
    if (h.magic != 0x47555353 || h.version != 1) {
      MPI_File_close(&fh);
      return MPI_ERR_OTHER;
    }
  }

  rc = MPI_Bcast(&h, (int)sizeof(h), MPI_BYTE, 0, active);
  if (rc != MPI_SUCCESS) { MPI_File_close(&fh); return rc; }

  *N = h.N; *phase = h.phase; *step = h.step;

  rc = MPI_Barrier(active);
  if (rc != MPI_SUCCESS) { MPI_File_close(&fh); return rc; }

  MPI_Offset off =
    (MPI_Offset)sizeof(ckpt_hdr) +
    (MPI_Offset)row0 * (MPI_Offset)(h.N + 1) * (MPI_Offset)sizeof(float);

  int count = nloc * (h.N + 1);
  rc = MPI_File_read_at_all(fh, off, Aloc, count, MPI_FLOAT, MPI_STATUS_IGNORE);

  int rc2 = MPI_File_close(&fh);
  if (rc == MPI_SUCCESS) rc = rc2;
  return rc;
}

// ---------------- фаза ACTIVE (перестроение) ----------------
static void active_alloc_rebuild(MPI_Comm active, int N,
                                 int* row0, int* nloc,
                                 float** Aloc) {
  int asize, arank;
  MPI_Comm_size(active, &asize);
  MPI_Comm_rank(active, &arank);

  block_decomp(N, asize, arank, row0, nloc);

  free(*Aloc);
  *Aloc = (float*)malloc((size_t)(*nloc) * (size_t)(N + 1) * sizeof(float));
  if (!*Aloc) die("malloc Aloc failed");
}

static MPI_Comm build_active_from_world(MPI_Comm world, int WORK) {
  int wrank;
  MPI_Comm_rank(world, &wrank);
  int color = (wrank < WORK) ? 1 : MPI_UNDEFINED;

  MPI_Comm active = MPI_COMM_NULL;
  int rc = MPI_Comm_split(world, color, wrank, &active);
  if (rc != MPI_SUCCESS) die_mpi("MPI_Comm_split(active)", rc);
  return active;
}

// ---------------- Метод Гаусса  ----------------
static int gauss_run(MPI_Comm active, int N,
                     int* phase_io, int* step_io,
                     int row0, int nloc, float* A,
                     const char* ckpt_path, int ckpt_period,
                     int* last_phase, int* last_step) {
  int arank, asize;
  MPI_Comm_rank(active, &arank);
  MPI_Comm_size(active, &asize);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int fail_rank = -1, fail_step = -1, fail_phase = -1;
  const char* s;
  if ((s = getenv("FAIL_RANK")))  fail_rank  = atoi(s);
  if ((s = getenv("FAIL_STEP")))  fail_step  = atoi(s);
  if ((s = getenv("FAIL_PHASE"))) fail_phase = atoi(s);

  float* pivot = (float*)malloc((size_t)(N + 1) * sizeof(float));
  if (!pivot) die("malloc pivot failed");

  int phase = *phase_io;
  int step  = *step_io;

  if (phase == PH_ELIM) {
    for (int i = step; i < N - 1; i++) {
      *last_phase = PH_ELIM;
      *last_step  = i;

      if (world_rank == fail_rank && fail_phase == PH_ELIM && fail_step == i) {
        fprintf(stderr, "KILLING myself: world_rank=%d pid=%d phase=%d step=%d\n",
                world_rank, getpid(), phase, i);
        fflush(stderr);
        raise(SIGKILL);
      }

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
        rc = ckpt_write_parallel(active, ckpt_path, N, PH_ELIM, i + 1, row0, nloc, A);
        if (rc != MPI_SUCCESS) { free(pivot); return rc; }
      }
    }

    phase = PH_BACK;
    step  = N - 2;
    {
      int rc = ckpt_write_parallel(active, ckpt_path, N, phase, step, row0, nloc, A);
      if (rc != MPI_SUCCESS) { free(pivot); return rc; }
    }
  }


  // На обратном шаге чекпоинтов нет из-за сложностей сохранения в файл и реализации
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
    *last_phase = PH_BACK;
    *last_step  = j;

    if (world_rank == fail_rank && fail_phase == PH_BACK && fail_step == j) {
      log0(active, "INJECT_FAIL: world_rank=%d phase=BACK step=%d", world_rank, j);
      raise(SIGKILL);
    }

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

  *phase_io = phase;
  *step_io  = step;
  return MPI_SUCCESS;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  log_init();

  MPI_Comm world = MPI_COMM_WORLD;
  MPI_Comm_set_errhandler(world, MPI_ERRORS_RETURN);

  int world_rank, world_size;
  MPI_Comm_rank(world, &world_rank);
  MPI_Comm_size(world, &world_size);

  if (argc < 4) {
    if (world_rank == 0) fprintf(stderr, "Usage: %s data.in ckpt_path WORK\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  const char* in_path   = argv[1];
  const char* ckpt_path = argv[2];
  int WORK = atoi(argv[3]);
  if (WORK <= 0 || WORK > world_size) die("Bad WORK value");

  if (world_rank == 0) {
    log0(world, "START: world_size=%d WORK=%d SPARES=%d", world_size, WORK, world_size - WORK);
  }

  MPI_Comm active = build_active_from_world(world, WORK);
  if (active != MPI_COMM_NULL) MPI_Comm_set_errhandler(active, MPI_ERRORS_RETURN);

  int N = 0, phase = PH_ELIM, step = 0;
  int ckpt_period = 2;

  if (active != MPI_COMM_NULL) {
    int arank;
    MPI_Comm_rank(active, &arank);
    if (arank == 0) {
      FILE* in = fopen(in_path, "r");
      if (!in) die("Cannot open data.in");
      if (fscanf(in, "%d", &N) != 1) die("Wrong data.in");
      fclose(in);
      log0(active, "READ N=%d", N);
    }
    int rc = MPI_Bcast(&N, 1, MPI_INT, 0, active);
    if (rc != MPI_SUCCESS) die_mpi("MPI_Bcast(N)", rc);
  }

  int rc = MPI_Bcast(&N, 1, MPI_INT, 0, world);
  if (rc != MPI_SUCCESS) die_mpi("MPI_Bcast(N,world)", rc);

  float* Aloc = NULL;
  int row0 = 0, nloc = 0;

  if (active != MPI_COMM_NULL) {
    active_alloc_rebuild(active, N, &row0, &nloc, &Aloc);

    for (int lk = 0; lk < nloc; lk++) {
      int i = row0 + lk;
      for (int j = 0; j <= N; j++) {
        Aloc[lk * (N + 1) + j] = (i == j || j == N) ? 1.f : 0.f;
      }
    }

    rc = ckpt_write_parallel(active, ckpt_path, N, phase, step, row0, nloc, Aloc);
    if (rc != MPI_SUCCESS) die_mpi("initial checkpoint", rc);
  }

  double t0 = MPI_Wtime();
  int last_phase = PH_ELIM;
  int last_step  = 0;

  // лаг для проверки операции revoke на процессе
  int revoked = 0;

  while (1) {
    int active_rc = MPI_SUCCESS;

    if (active != MPI_COMM_NULL) {
      active_rc = gauss_run(active, N, &phase, &step, row0, nloc, Aloc,
                            ckpt_path, ckpt_period, &last_phase, &last_step);

      if (active_rc != MPI_SUCCESS) {
        if (is_ulfm_failure(active_rc)) {
          (void)maybe_revoke_on_failure(world, active_rc, &revoked, last_phase, last_step);
        } else {
          die_mpi("MPI error in ACTIVE (not ULFM)", active_rc);
        }
      }
    }

    rc = MPI_Barrier(world);
    if (rc == MPI_SUCCESS) break;

    if (!is_ulfm_failure(rc)) die_mpi("WORLD barrier failed (not ULFM)", rc);

    (void)maybe_revoke_on_failure(world, rc, &revoked, last_phase, last_step);

    // Создаём мир 2 на выживших и распространяем
    MPI_Comm world2;
    MPIX_Comm_shrink(world, &world2);

    if (world != MPI_COMM_WORLD) MPI_Comm_free(&world);
    world = world2;
    MPI_Comm_set_errhandler(world, MPI_ERRORS_RETURN);

    int new_wr, new_ws;
    MPI_Comm_rank(world, &new_wr);
    MPI_Comm_size(world, &new_ws);

    {
      int* surv_old = NULL;
      if (new_wr == 0) surv_old = (int*)malloc((size_t)new_ws * sizeof(int));

      // сохраняем начальный ранг в мире 1, не очень хорошо, если сбоев будет много (сейчас тестируется один)
      int my_initial = world_rank;
      MPI_Gather(&my_initial, 1, MPI_INT, surv_old, 1, MPI_INT, 0, world);

      if (new_wr == 0) {
        fprintf(stderr, "=== RECOVERY ===\n");
        fprintf(stderr, "Survivors(count=%d)\n", new_ws);
        fprintf(stderr, "New ACTIVE size=%d\n", (WORK < new_ws ? WORK : new_ws));
        fprintf(stderr, "==============\n");
        fflush(stderr);
        free(surv_old);
      }
    }

    rc = MPI_Bcast(&N, 1, MPI_INT, 0, world);
    if (rc != MPI_SUCCESS) die_mpi("MPI_Bcast(N after shrink)", rc);

    // "перестраиваю" активную фазу Гаусса в новом мире 
    if (active != MPI_COMM_NULL) MPI_Comm_free(&active);
    active = build_active_from_world(world, WORK);
    if (active != MPI_COMM_NULL) MPI_Comm_set_errhandler(active, MPI_ERRORS_RETURN);

    if (active != MPI_COMM_NULL) {
      active_alloc_rebuild(active, N, &row0, &nloc, &Aloc);

      rc = ckpt_read_parallel(active, ckpt_path, &N, &phase, &step, row0, nloc, Aloc);
      if (rc != MPI_SUCCESS) die_mpi("checkpoint_load after recovery", rc);
    }

    revoked = 0;
  }

  double t1 = MPI_Wtime();
  if (active != MPI_COMM_NULL) {
    int ar;
    MPI_Comm_rank(active, &ar);
    if (ar == 0) printf("Time in seconds=%gs\n", (t1 - t0));
  }

  free(Aloc);
  if (active != MPI_COMM_NULL) MPI_Comm_free(&active);
  if (world != MPI_COMM_WORLD) MPI_Comm_free(&world);

  MPI_Finalize();
  return 0;
}
