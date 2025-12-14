#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <mpi.h>
#include <mpi-ext.h>  /* ULFM */

#define CKPT_FILE "gauss_ckpt.dat"
#define N_DEFAULT 100

/* Глобальные параметры задачи */
int N;               /* размер матрицы */
int Np;              /* число строк, обрабатываемых процессом */
int my_rank;         /* ранг в текущем коммуникаторе */
int world_size;      /* размер MPI_COMM_WORLD */
MPI_Comm active_comm; /* активный коммуникатор (может меняться при сбоях) */

float *A;            /* локальная часть матрицы A (размер Np x (N+1)) */
float *X;            /* вектор решения (размер N) */

/* ------------------------------------------------------------
 *  MPI-IO: запись контрольной точки
 * ------------------------------------------------------------ */
static int save_checkpoint(MPI_Comm comm) {
    MPI_File fh;
    MPI_Status status;
    int ret;
    MPI_Offset offset;
    char err_str[MPI_MAX_ERROR_STRING];
    int err_len;

    ret = MPI_File_open(comm, CKPT_FILE,
                        MPI_MODE_CREATE | MPI_MODE_WRONLY,
                        MPI_INFO_NULL, &fh);
    if (ret != MPI_SUCCESS) {
        MPI_Error_string(ret, err_str, &err_len);
        fprintf(stderr, "Rank %d: MPI_File_open failed: %s\n", 
                my_rank, err_str);
        return ret;
    }

    /* Каждый процесс записывает свою часть матрицы A */
    offset = (MPI_Offset)my_rank * Np * (N+1) * sizeof(float);
    ret = MPI_File_write_at_all(fh, offset, A, Np*(N+1), MPI_FLOAT, &status);
    if (ret != MPI_SUCCESS) {
        MPI_Error_string(ret, err_str, &err_len);
        fprintf(stderr, "Rank %d: MPI_File_write_at_all failed: %s\n", 
                my_rank, err_str);
        MPI_File_close(&fh);
        return ret;
    }

    /* Только процесс 0 записывает размер N и вектор X */
    if (my_rank == 0) {
        offset = (MPI_Offset)world_size * Np * (N+1) * sizeof(float);
        ret = MPI_File_write_at(fh, offset, &N, 1, MPI_INT, &status);
        if (ret != MPI_SUCCESS) {
            MPI_Error_string(ret, err_str, &err_len);
            fprintf(stderr, "Rank 0: MPI_File_write_at (N) failed: %s\n", err_str);
            MPI_File_close(&fh);
            return ret;
        }
        
        offset += sizeof(int);
        ret = MPI_File_write_at(fh, offset, X, N, MPI_FLOAT, &status);
        if (ret != MPI_SUCCESS) {
            MPI_Error_string(ret, err_str, &err_len);
            fprintf(stderr, "Rank 0: MPI_File_write_at (X) failed: %s\n", err_str);
            MPI_File_close(&fh);
            return ret;
        }
    }

    MPI_File_sync(fh);  /* Синхронизация с диском */
    MPI_File_close(&fh);
    
    /* Синхронизация процессов после записи */
    MPI_Barrier(comm);
    
    return MPI_SUCCESS;
}


/* ------------------------------------------------------------
 *  MPI-IO: чтение контрольной точки
 * ------------------------------------------------------------ */
static int load_checkpoint(MPI_Comm comm) {
    MPI_File fh;
    MPI_Status status;
    int ret, saved_N;
    MPI_Offset offset;
    char err_str[MPI_MAX_ERROR_STRING];
    int err_len;

    ret = MPI_File_open(comm, CKPT_FILE, MPI_MODE_RDONLY,
                        MPI_INFO_NULL, &fh);
    if (ret != MPI_SUCCESS) {
        MPI_Error_string(ret, err_str, &err_len);
        fprintf(stderr, "Rank %d: MPI_File_open (read) failed: %s\n", 
                my_rank, err_str);
        return ret;
    }

    /* Каждый процесс читает свою часть матрицы A */
    offset = (MPI_Offset)my_rank * Np * (N+1) * sizeof(float);
    ret = MPI_File_read_at_all(fh, offset, A, Np*(N+1), MPI_FLOAT, &status);
    if (ret != MPI_SUCCESS) {
        MPI_Error_string(ret, err_str, &err_len);
        fprintf(stderr, "Rank %d: MPI_File_read_at_all failed: %s\n", 
                my_rank, err_str);
        MPI_File_close(&fh);
        return ret;
    }

    /* Процесс 0 читает размер N и вектор X */
    if (my_rank == 0) {
        offset = (MPI_Offset)world_size * Np * (N+1) * sizeof(float);
        ret = MPI_File_read_at(fh, offset, &saved_N, 1, MPI_INT, &status);
        if (ret != MPI_SUCCESS) {
            MPI_Error_string(ret, err_str, &err_len);
            fprintf(stderr, "Rank 0: MPI_File_read_at (N) failed: %s\n", err_str);
            MPI_File_close(&fh);
            return ret;
        }
        
        if (saved_N != N) {
            fprintf(stderr, "Error: saved N (%d) != current N (%d)\n", saved_N, N);
            MPI_File_close(&fh);
            return MPI_ERR_SIZE;
        }
        
        offset += sizeof(int);
        ret = MPI_File_read_at(fh, offset, X, N, MPI_FLOAT, &status);
        if (ret != MPI_SUCCESS) {
            MPI_Error_string(ret, err_str, &err_len);
            fprintf(stderr, "Rank 0: MPI_File_read_at (X) failed: %s\n", err_str);
            MPI_File_close(&fh);
            return ret;
        }
    }

    MPI_File_close(&fh);

    /* Рассылаем X всем процессам */
    MPI_Bcast(X, N, MPI_FLOAT, 0, comm);
    
    return MPI_SUCCESS;
}


/* ------------------------------------------------------------
 *  Инициализация матрицы A (локальная часть)
 * ------------------------------------------------------------ */
static void init_matrix() {
    int i, j, global_i;
    for (i = 0; i < Np; i++) {
        global_i = my_rank * Np + i;
        for (j = 0; j <= N; j++) {
            if (global_i == j || j == N)
                A[i*(N+1) + j] = 1.0f;
            else
                A[i*(N+1) + j] = 0.0f;
        }
    }
}

/* ------------------------------------------------------------
 *  Прямой ход (исключение Гаусса) с периодическими чекпоинтами
 * ------------------------------------------------------------ */
static int gauss_elimination(MPI_Comm comm) {
    int i, j, k, global_i, global_k;
    int step = 0;
    int checkpoint_interval = 10; /* сохраняем каждые 10 шагов */

    for (i = 0; i < N-1; i++) {
        /* Определяем, кому принадлежит строка i */
        int owner = i / Np;
        global_i = i - owner * Np; /* локальный индекс в процессе-владельце */

        /* Рассылаем строку i всем процессам */
        float *row_i = (float*)malloc((N+1) * sizeof(float));
        if (my_rank == owner) {
            memcpy(row_i, &A[global_i*(N+1)], (N+1)*sizeof(float));
        }
        MPI_Bcast(row_i, N+1, MPI_FLOAT, owner, comm);

        /* Локальное исключение */
        for (k = 0; k < Np; k++) {
            global_k = my_rank * Np + k;
            if (global_k > i) {  /* ниже диагонали */
                float factor = A[k*(N+1) + i] / row_i[i];
                for (j = i+1; j <= N; j++) {
                    A[k*(N+1) + j] -= factor * row_i[j];
                }
                A[k*(N+1) + i] = 0.0f; /* для устойчивости */
            }
        }
        free(row_i);

        step++;
        /* Периодическое сохранение контрольной точки */
        if (step % checkpoint_interval == 0) {
            if (my_rank == 0) printf("Checkpoint at step %d\n", i);
            if (save_checkpoint(comm) != MPI_SUCCESS) {
                fprintf(stderr, "Checkpoint failed!\n");
                return 0;
            }
        }

        /* Имитация сбоя (для тестирования) */
        if (getenv("FAIL_RANK") && getenv("FAIL_PHASE") && getenv("FAIL_STEP")) {
            int fail_rank = atoi(getenv("FAIL_RANK"));
            int fail_phase = atoi(getenv("FAIL_PHASE"));
            int fail_step = atoi(getenv("FAIL_STEP"));
            if (fail_phase == 0 && i == fail_step && my_rank == fail_rank) {
                fprintf(stderr, "Rank %d simulating failure at elimination step %d\n",
                        my_rank, i);
                raise(SIGKILL);
            }
        }
    }
    return 1;
}

/* ------------------------------------------------------------
 *  Обратный ход (подстановка)
 * ------------------------------------------------------------ */
static void back_substitution(MPI_Comm comm) {
    int i, j, k, global_i;

    /* X[N-1] */
    if (my_rank == (N-1)/Np) {
        int local_idx = (N-1) % Np;
        X[N-1] = A[local_idx*(N+1) + N] / A[local_idx*(N+1) + (N-1)];
    }
    MPI_Bcast(&X[N-1], 1, MPI_FLOAT, (N-1)/Np, comm);

    for (j = N-2; j >= 0; j--) {
        /* Обновляем правые части */
        for (i = 0; i < Np; i++) {
            global_i = my_rank * Np + i;
            if (global_i <= j) {
                A[i*(N+1) + N] -= A[i*(N+1) + (j+1)] * X[j+1];
            }
        }

        /* Вычисляем X[j] */
        if (my_rank == j/Np) {
            int local_idx = j % Np;
            X[j] = A[local_idx*(N+1) + N] / A[local_idx*(N+1) + j];
        }
        MPI_Bcast(&X[j], 1, MPI_FLOAT, j/Np, comm);
    }
}

/* ------------------------------------------------------------
 *  Обработчик ошибок ULFM
 * ------------------------------------------------------------ */
static void failure_handler(MPI_Comm *comm, int *err, ...) {
    int error_class;
    MPI_Error_class(*err, &error_class);
    if (error_class != MPIX_ERR_PROC_FAILED) {
        MPI_Abort(*comm, *err);
    }
    /* Игнорируем ошибку - восстановление произойдет в основном коде */
}

/* ------------------------------------------------------------
 *  Восстановление после сбоя (сценарий б)
 * ------------------------------------------------------------ */
static int recover_from_failure(MPI_Comm *comm) {
    MPI_Group world_group, active_group, failed_group;
    int i, n_failed, *failed_ranks;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    /* Получаем группу отказавших процессов */
    MPIX_Comm_failure_ack(*comm);
    MPIX_Comm_failure_get_acked(*comm, &failed_group);
    MPI_Group_size(failed_group, &n_failed);

    if (n_failed == 0) {
        MPI_Group_free(&failed_group);
        return 1; /* нет сбоев */
    }

    printf("Rank %d: detected %d failed process(es). Attempting recovery...\n",
           my_rank, n_failed);

    /* Получаем ранги отказавших процессов в текущем коммуникаторе */
    failed_ranks = (int*)malloc(n_failed * sizeof(int));
    for (i = 0; i < n_failed; i++) failed_ranks[i] = i;
    MPI_Comm_group(*comm, &active_group);
    MPI_Group_translate_ranks(failed_group, n_failed, failed_ranks,
                              active_group, failed_ranks);

    /* Отключаем коммуникатор */
    MPIX_Comm_revoke(*comm);

    /* Пытаемся создать новые процессы вместо вышедших из строя */
    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "host", hostname);

    MPI_Comm intercomm;
    int spawn_error;
    MPI_Comm_spawn("./gauss_ft_mpiio", MPI_ARGV_NULL, n_failed,
                   info, 0, *comm, &intercomm, &spawn_error);

    if (spawn_error != MPI_SUCCESS) {
        printf("Rank %d: could not spawn new processes. Shrinking communicator.\n",
               my_rank);
        /* Если не удалось создать новые процессы, просто удаляем отказавшие */
        MPIX_Comm_shrink(*comm, comm);
        MPI_Group_free(&active_group);
        MPI_Group_free(&failed_group);
        free(failed_ranks);
        MPI_Info_free(&info);
        return 1;
    }

    /* Объединяем старые и новые процессы */
    MPI_Comm intracomm;
    MPI_Intercomm_merge(intercomm, 0, &intracomm);

    /* Перераспределяем данные */
    MPI_Comm_free(comm);
    *comm = intracomm;
    MPI_Comm_rank(*comm, &my_rank);
    MPI_Comm_size(*comm, &world_size);

    /* Загружаем контрольную точку и продолжаем работу */
    load_checkpoint(*comm);

    MPI_Group_free(&active_group);
    MPI_Group_free(&failed_group);
    free(failed_ranks);
    MPI_Info_free(&info);
    return 1;
}


int main(int argc, char **argv) {
    double start_time, end_time;
    int recovered = 0;
    FILE *in;

    MPI_Init(&argc, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &active_comm);
    MPI_Comm_rank(active_comm, &my_rank);
    MPI_Comm_size(active_comm, &world_size);

    /* Устанавливаем обработчик ошибок ULFM */
    MPI_Errhandler errh;
    MPI_Comm_create_errhandler(failure_handler, &errh);
    MPI_Comm_set_errhandler(active_comm, errh);

    /* Чтение размера матрицы */
    if (my_rank == 0) {
        in = fopen("data.in", "r");
        if (in == NULL) {
            N = N_DEFAULT;
            printf("Using default N=%d\n", N);
        } else {
            fscanf(in, "%d", &N);
            fclose(in);
        }
        printf("Matrix size N=%d, MPI processes=%d\n", N, world_size);
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, active_comm);

    /* Распределение строк по процессам */
    Np = N / world_size;
    int remainder = N % world_size;
    if (my_rank < remainder) Np++;
    
    if (Np == 0) {
        fprintf(stderr, "Rank %d: No rows to process. N=%d, world_size=%d\n", 
                my_rank, N, world_size);
        MPI_Abort(active_comm, 1);
    }

    /* Выделение памяти */
    A = (float*)malloc(Np * (N+1) * sizeof(float));
    X = (float*)malloc(N * sizeof(float));
    if (!A || !X) {
        fprintf(stderr, "Rank %d: Memory allocation failed!\n", my_rank);
        MPI_Abort(active_comm, 1);
    }

    /* Основной цикл вычислений с возможностью восстановления */
    do {
        if (!recovered) {
            /* Первый запуск - инициализируем матрицу */
            init_matrix();
            if (my_rank == 0) {
                printf("Starting Gaussian elimination...\n");
            }
        } else {
            /* Восстановление - загружаем контрольную точку */
            if (load_checkpoint(active_comm) != MPI_SUCCESS) {
                fprintf(stderr, "Rank %d: Failed to load checkpoint\n", my_rank);
                MPI_Abort(active_comm, 1);
            }
            if (my_rank == 0) {
                printf("Resuming from checkpoint...\n");
            }
        }

        start_time = MPI_Wtime();

        /* Прямой ход */
        if (!gauss_elimination(active_comm)) {
            /* Произошел сбой в процессе исключения */
            if (!recover_from_failure(&active_comm)) {
                fprintf(stderr, "Recovery failed. Aborting.\n");
                break;
            }
            recovered = 1;
            continue;
        }

        /* Обратный ход */
        back_substitution(active_comm);

        /* Финальное сохранение контрольной точки */
        if (my_rank == 0) {
            if (save_checkpoint(active_comm) == MPI_SUCCESS) {
                printf("Final checkpoint saved successfully.\n");
            }
            end_time = MPI_Wtime();
            printf("\nGaussian elimination completed successfully.\n");
            printf("Total time: %.3f seconds\n", end_time - start_time);
            printf("Solution vector X (first 10 elements):\n");
            for (int i = 0; i < (N < 10 ? N : 10); i++) {
                printf("  X[%d] = %.6f\n", i, X[i]);
            }
        }
        break; /* Успешное завершение */

    } while (1);

    /* Освобождение ресурсов */
    free(A);
    free(X);
    MPI_Errhandler_free(&errh);
    MPI_Comm_free(&active_comm);
    MPI_Finalize();
    return 0;
}