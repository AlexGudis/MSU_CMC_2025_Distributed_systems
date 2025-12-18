/* 
 * gauss_ulfm_mpiio_ckpt.c
 *
 * Версия вашего кода с подробными комментариями по:
 *  - назначению каждой функции/блока,
 *  - роли каждого MPI-вызова,
 *  - логике контрольных точек (MPI-IO),
 *  - обработке отказов ULFM (revoke/shrink),
 *  - почему условие/цикл расположен именно в этом месте.
 *
 * Комментарии намеренно подробные — под формат отчёта.
 */

#include <mpi.h>        // Базовый стандарт MPI: коммуникаторы, коллективные операции, MPI-IO, таймер MPI_Wtime и т.д.
#include <mpi-ext.h>    // Расширения MPI (ULFM): MPIX_Comm_revoke, MPIX_Comm_shrink и коды ошибок MPIX_ERR_*.
#include <stdio.h>      // Стандартный ввод/вывод (printf/fprintf/fscanf).
#include <stdlib.h>     // malloc/free/atoi/getenv.
#include <string.h>     // memcpy.
#include <signal.h>     // raise, SIGKILL (для искусственной "инъекции" отказа процесса).
#include <stdarg.h>     // va_list для вариадических функций логирования.
#include <unistd.h>     // getpid (чтобы вывести PID при самоуничтожении процесса).

/* 
 * Фазы алгоритма Гаусса:
 *  PH_ELIM — прямой ход (elimination),
 *  PH_BACK — обратный ход (back substitution).
 * Эти значения записываются в checkpoint, чтобы после сбоя понимать, на какой фазе продолжать.
 */
enum { PH_ELIM = 0, PH_BACK = 1 };

/*
 * Заголовок файла контрольной точки (checkpoint header).
 * Хранит минимальный набор метаданных, чтобы корректно восстановить вычисление:
 *  - magic/version позволяют проверить "это наш файл и правильного формата",
 *  - N — размер системы,
 *  - phase/step — состояние алгоритма (где продолжать).
 */
typedef struct {
  int magic;    // "магическое" число-сигнатура (для проверки целостности/типа файла)
  int version;  // версия формата чекпоинта (на будущее, если формат расширится)
  int N;        // размер СЛАУ
  int phase;    // текущая фаза (PH_ELIM/PH_BACK)
  int step;     // индекс шага, с которого нужно продолжить
} ckpt_hdr;

// ---------------- Логирование (удобно для локальных тестов и отладки) ----------------

/*
 * Флаг включения логов.
 * По умолчанию LOG=1, но можно выключить, выставив переменную окружения GAUSS_LOG=0.
 * Важное для производительности: логи сильно влияют на время, поэтому их удобно отключать при замерах.
 */
static int LOG = 1;

/* 
 * Инициализация логов: читаем переменную окружения.
 * Делать это нужно в начале main после MPI_Init, чтобы корректно использовать MPI_Wtime и rank.
 */
static void log_init(void) {
  const char* s = getenv("GAUSS_LOG");
  if (s) LOG = atoi(s);
}

/*
 * logger — печатает сообщение в stderr с временем и рангом (внутри указанного comm).
 * Почему через stderr: чтобы не смешивать служебные сообщения с основным выводом (stdout).
 * Почему MPI_Wtime: это стандартный MPI-таймер, согласованный по семантике внутри MPI-среды.
 */
static void logger(MPI_Comm comm, const char* fmt, ...) {
  if (!LOG) return;               // Если логи отключены — быстрый выход.
  int r = -1;
  if (comm != MPI_COMM_NULL)      // Если коммутатор валиден — узнаём rank, иначе оставляем -1.
    MPI_Comm_rank(comm, &r);

  double t = MPI_Wtime();         // Время "сейчас" (wall time), удобно для трассировки.
  fprintf(stderr, "[t=%8.3f r=%d] ", t, r);

  // Вариадическое форматирование как у printf.
  va_list ap; 
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);

  fprintf(stderr, "\n");
  fflush(stderr);                 // Сбрасываем буфер, чтобы логи не "залипали" при сбоях.
}

// ---------------- Обработка ошибок ----------------

/*
 * die — печатает сообщение и аварийно завершает MPI-приложение.
 * Почему MPI_Abort: при фатальной ошибке (например, malloc/fopen) продолжать нельзя —
 * надо завершить все процессы, иначе часть зависнет.
 */
static void die(const char* msg) {
  fprintf(stderr, "%s\n", msg);
  fflush(stderr);
  MPI_Abort(MPI_COMM_WORLD, 1);   // Код 1 — условный код завершения.
}

/*
 * die_mpi — версия die, которая для MPI-ошибки печатает её текст.
 * MPI_Error_string переводит rc в человекочитаемое описание.
 */
static void die_mpi(const char* where, int rc) {
  char err[MPI_MAX_ERROR_STRING];
  int len = 0;
  MPI_Error_string(rc, err, &len);
  fprintf(stderr, "%s: %s\n", where, err);
  fflush(stderr);
  MPI_Abort(MPI_COMM_WORLD, 2);   // Код 2 — условно "MPI error".
}

/*
  rc - rank code
 * is_ulfm_failure — определяет, является ли rc ошибкой, связанной со сбоями ULFM.
 * Используется, чтобы отличать:
 *  - "ожидаемые" ошибки отказоустойчивости (PROC_FAILED/REVOKED),
 *  - от "обычных" MPI-ошибок (не должны приводить к восстановлению).
 */
static int is_ulfm_failure(int rc) {
  if (rc == MPI_SUCCESS) return 0;
  int eclass = MPI_ERR_OTHER;
  MPI_Error_class(rc, &eclass);
  return (eclass == MPIX_ERR_PROC_FAILED || eclass == MPIX_ERR_REVOKED);
}

/*
 * is_proc_failed — более строгая проверка: именно "процесс умер" (PROC_FAILED),
 * а не "коммуникатор отозван" (REVOKED).
 * Это важно в maybe_revoke_on_failure: revoke имеет смысл вызывать, когда кто-то впервые увидел PROC_FAILED.
 */
static int is_proc_failed(int rc) {
  if (rc == MPI_SUCCESS) return 0;
  int eclass = MPI_ERR_OTHER;
  MPI_Error_class(rc, &eclass);
  return (eclass == MPIX_ERR_PROC_FAILED);
}

/*
 * maybe_revoke_on_failure — "условно надёжный" паттерн:
 *  - кто первым увидел смерть процесса (PROC_FAILED), тот вызывает MPIX_Comm_revoke(world),
 *  - остальные в дальнейшем получат MPIX_ERR_REVOKED на коллективных операциях и не зависнут.
 *
 * Аргументы last_phase/last_step используются только для диагностики (где примерно произошёл отказ).
 * revoked_flag защищает от повторных revoke (избыточно, но снижает шум).
 */
static int maybe_revoke_on_failure(MPI_Comm world, int rc,
                                  int* revoked_flag,
                                  int last_phase, int last_step) {
  if (*revoked_flag) return 0;        // revoke уже был — ничего не делаем.
  if (!is_proc_failed(rc)) return 0;  // если это не PROC_FAILED — revoke здесь не нужен.

  *revoked_flag = 1;
  int wr = -1;
  MPI_Comm_rank(world, &wr);

  // Логируем, кто заметил сбой и на каком логическом шаге.
  logger(world, "PROC_FAILED noticed by world_rank=%d at phase=%d step=%d -> revoke",
       wr, last_phase, last_step);

  MPIX_Comm_revoke(world);            // Отзываем коммуникатор: "разбудит" зависшие операции в world.
  return 1;
}

// ---------------- Разбиение матрицы по строкам ----------------

/*
 * block_decomp — классическое блочное разбиение N строк на P процессов:
 *  - base = N/P строк каждому,
 *  - rem = N%P первых процессов получают на 1 строку больше. Нужно, чтобы корректно разделить
 *  N строк на P процессов, когда N не делится на P
 *  
 * Возвращает:
 *  - r0 (start) — глобальный индекс первой строки процесса,
 *  - rn (nloc)  — число локальных строк.
 *
 * Почему нужно: каждый процесс хранит и обновляет только свой блок строк, экономя память и обеспечивая параллелизм.
 */
static void block_decomp(int N, int P, int r, int* r0, int* rn) {
  int base = N / P, rem = N % P;              // базовый размер блока и остаток
  int nloc = base + (r < rem ? 1 : 0);        // первые rem рангов получают +1 строку
  int start = r * base + (r < rem ? r : rem); // сдвиг из-за "лишних" строк у рангов < rem
  *r0 = start;
  *rn = nloc;
}

/*
 * owner_of_row — обратная функция к block_decomp:
 * по глобальному индексу строки row определяет, какой ранг (в коммуникаторе active) является владельцем строки.
 *
 * Зачем нужно:
 *  - на шаге elimination pivot-строку i должен рассылать её владелец,
 *  - на шаге back-substitution X[j] вычисляет владелец строки j.
 */
static int owner_of_row(int N, int P, int row) {
  int base = N / P, rem = N % P;
  int cut = (base + 1) * rem;              // граница, где заканчиваются "увеличенные" блоки (base+1)
  if (row < cut) return row / (base + 1);  // row лежит в первых rem блоках
  return rem + (row - cut) / base;         // row лежит в обычных блоках размера base
}

// ---------------- Контрольные точки (checkpoint) через MPI-IO ----------------

/*
 * ckpt_write_parallel — параллельная запись чекпоинта (MPI-IO).
 * В файл записывается:
 *  1) header в начале файла (пишет только rank 0 active),
 *  2) полный массив матрицы A|b построчно (каждый процесс пишет свои строки).
 *
 * Почему MPI-IO:
 *  - позволяет всем процессам писать в один файл согласованно,
 *  - MPI_File_write_at_all — коллективная запись, MPI-IO может оптимизировать (агрегирование).
 *
 * Почему "write_at" (с явными смещениями):
 *  - избегаем общих указателей файла (shared file pointer),
 *  - каждый процесс пишет в "своё" место, не мешая другим.
 * 
 * params:
 * path - Путь к файлу checkpoint.
 * phase - PH_ELIM — прямой ход или PH_BACK — обратный ход.
 * step - Номер шага внутри фазы, с которого продолжать. (например, на elim это номер строки)
 * row0 - Глобальный индекс первой строки, принадлежащей этому процессу.
 * nloc - Количество строк, которые хранит этот процесс.
 * Aloc - Буфер локальной части матрицы, куда будут прочитаны данные.
 */
static int ckpt_write_parallel(MPI_Comm active, const char* path,
                               int N, int phase, int step,
                               int row0, int nloc,
                               const float* Aloc) {
  MPI_File fh;

  // Коллективно открываем файл. CREATE|WRONLY: создаём/перезаписываем содержимое чекпоинта.
  int rc = MPI_File_open(active, path,
                         MPI_MODE_CREATE | MPI_MODE_WRONLY,
                         MPI_INFO_NULL, &fh);
  if (rc != MPI_SUCCESS) return rc;

  int arank;
  MPI_Comm_rank(active, &arank);

  // Header пишет только rank 0 активного коммуникатора.
  // Это удобно: header один на файл, остальным он не нужен при записи.
  if (arank == 0) {
    ckpt_hdr h = {0};
    h.magic = 0x47555353;         // сигнатура "GUSS" (условно)
    h.version = 1;                // версия формата
    h.N = N;
    h.phase = phase;
    h.step = step;

    // Пишем header в начало файла (смещение 0). Пишем как "сырые байты" (MPI_BYTE), чтобы не зависеть от типа MPI.
    rc = MPI_File_write_at(fh, 0, &h, (int)sizeof(h), MPI_BYTE, MPI_STATUS_IGNORE);
    if (rc != MPI_SUCCESS) { MPI_File_close(&fh); return rc; } // Неуспешно записали - завершаемся
  }

  // Барьер нужен, чтобы гарантировать: header уже записан до того, как кто-то начнёт писать матрицу.
  // Грубо говоря здесь все процессы синхронизуются в одной точке
  rc = MPI_Barrier(active);
  if (rc != MPI_SUCCESS) { MPI_File_close(&fh); return rc; } // Кто-то где-то упал по пути - завершаемся

  // Вычисляем общий размер файла: header + N*(N+1)*sizeof(float).
  // Почему важно: MPI_File_set_size фиксирует длину, чтобы избежать гонок при расширении файла.
  MPI_Offset total_bytes =
    (MPI_Offset)sizeof(ckpt_hdr) +
    (MPI_Offset)N * (MPI_Offset)(N + 1) * (MPI_Offset)sizeof(float);

  rc = MPI_File_set_size(fh, total_bytes);
  if (rc != MPI_SUCCESS) { MPI_File_close(&fh); return rc; }

  // Ещё один барьер: чтобы все ранги видели согласованное состояние/размер файла перед коллективной записью.
  rc = MPI_Barrier(active);
  if (rc != MPI_SUCCESS) { MPI_File_close(&fh); return rc; }

  // Смещение, куда писать локальные строки:
  // пропускаем header, затем сдвигаемся на row0 строк, каждая строка имеет (N+1) float.
  MPI_Offset off =
    (MPI_Offset)sizeof(ckpt_hdr) +
    (MPI_Offset)row0 * (MPI_Offset)(N + 1) * (MPI_Offset)sizeof(float);

  // Сколько элементов пишем: nloc строк * (N+1) столбцов (включая правую часть).
  int count = nloc * (N + 1);

  // Коллективная запись всех процессов active в свои смещения.
  // Именно _at_all: коллективная, позволяет MPI-IO применять оптимизации.
  rc = MPI_File_write_at_all(fh, off, (void*)Aloc, count, MPI_FLOAT, MPI_STATUS_IGNORE);
  if (rc != MPI_SUCCESS) { MPI_File_close(&fh); return rc; }

  // Синхронизация данных на диск: повышает "надёжность" чекпоинта.
  // (Цена — дополнительное время, особенно на сетевых FS).
  rc = MPI_File_sync(fh);
  if (rc != MPI_SUCCESS) { MPI_File_close(&fh); return rc; }

  // Закрытие файла — обязательный шаг (освобождает ресурсы MPI-IO).
  rc = MPI_File_close(&fh);
  return rc;
}

/*
 * ckpt_read_parallel — параллельное чтение чекпоинта.
 * Логика симметрична записи:
 *  - rank 0 читает header и проверяет magic/version,
 *  - header рассылается всем через Bcast (чтобы у всех были N/phase/step),
 *  - затем каждый процесс читает свой кусок матрицы по вычисленному смещению.
 */
static int ckpt_read_parallel(MPI_Comm active, const char* path,
                              int* N, int* phase, int* step,
                              int row0, int nloc,
                              float* Aloc) {


  
  MPI_File fh;

  // Коллективно открываем файл только на чтение.
  int rc = MPI_File_open(active, path, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
  if (rc != MPI_SUCCESS) return rc;

  int arank;
  MPI_Comm_rank(active, &arank);

  ckpt_hdr h = {0};

  // Header читает только rank 0 (один раз) — затем рассылаем всем.
  if (arank == 0) {
    rc = MPI_File_read_at(fh, 0, &h, (int)sizeof(h), MPI_BYTE, MPI_STATUS_IGNORE);
    if (rc != MPI_SUCCESS) { MPI_File_close(&fh); return rc; }

    // Проверка валидности: защищает от чтения "чужого" файла/битого формата.
    if (h.magic != 0x47555353 || h.version != 1) {
      MPI_File_close(&fh);
      return MPI_ERR_OTHER;
    }
  }

  // Рассылаем header как байты: так проще, чем делать MPI datatype на структуру.
  rc = MPI_Bcast(&h, (int)sizeof(h), MPI_BYTE, 0, active);
  if (rc != MPI_SUCCESS) { MPI_File_close(&fh); return rc; }

  // Возвращаем метаданные вызывающему коду.
  *N = h.N; *phase = h.phase; *step = h.step;

  // Барьер — для синхронизации перед коллективным чтением (не всегда строго обязателен, но делает поведение стабильнее).
  rc = MPI_Barrier(active);
  if (rc != MPI_SUCCESS) { MPI_File_close(&fh); return rc; }

  // Смещение на начало локальных строк.
  MPI_Offset off =
    (MPI_Offset)sizeof(ckpt_hdr) +
    (MPI_Offset)row0 * (MPI_Offset)(h.N + 1) * (MPI_Offset)sizeof(float);

  int count = nloc * (h.N + 1); // читаем nloc строк по h.N+1 чисел в каждой (из-за столбца b)

  // Коллективное чтение: каждый процесс читает свой диапазон.
  rc = MPI_File_read_at_all(fh, off, Aloc, count, MPI_FLOAT, MPI_STATUS_IGNORE);

  // Закрытие файла. Если чтение успешно, но close нет — ресурсы утекут.
  int rc2 = MPI_File_close(&fh);
  if (rc == MPI_SUCCESS) rc = rc2;
  return rc;
}

// ---------------- Перестроение ACTIVE после восстановления ----------------
/*
 * active_alloc_rebuild — функция, которая пересчитывает локальный диапазон строк (row0,nloc)
 * и (пере)выделяет буфер Aloc под новую декомпозицию.
 *
 * Почему это важно после сбоя:
 *  - размер active может измениться (часть "рабочих" умерла и их место заняли "спейры"),
 *  - значит изменятся row0/nloc для каждого ранга,
 *  - и нужно заново выделить память нужного размера и подготовить буферы под чтение checkpoint.
 */
static void active_alloc_rebuild(MPI_Comm active, int N,
                                 int* row0, int* nloc,
                                 float** Aloc) {
  int asize, arank;
  MPI_Comm_size(active, &asize);
  MPI_Comm_rank(active, &arank);

  // Пересчитываем разбиение строк на новый размер active.
  block_decomp(N, asize, arank, row0, nloc);

  // Освобождаем старый буфер (если был) и выделяем новый.
  free(*Aloc);
  *Aloc = (float*)malloc((size_t)(*nloc) * (size_t)(N + 1) * sizeof(float));
  if (!*Aloc) die("malloc Aloc failed");
}

/*
 * build_active_from_world — реализует сценарий (a) с "резервными" процессами:
 *  - Мир (world) содержит WORK + SPARES процессов.
 *  - В active входят только первые WORK рангов мира (wrank < WORK).
 *  - Остальные получают MPI_COMM_NULL и в вычислениях не участвуют, пока не потребуется восстановление.
 *
 * Почему через MPI_Comm_split:
 *  - это стандартный способ создать подкоммуникатор,
 *  - после shrink ранги перенумеруются, и снова можно взять "первые WORK" выживших.
 * 
 *  По сути эта функция создаёт рабочую группу, которая будет трудиться над Гауссом
 *  в active ранги сохраняют “естественный” порядок (0..WORK-1).
 */
static MPI_Comm build_active_from_world(MPI_Comm world, int WORK) {
  int wrank;
  MPI_Comm_rank(world, &wrank);

  // color определяет, попадает ли процесс в новый коммуникатор:
  //  - 1 => участвует в active,
  //  - MPI_UNDEFINED => не входит (получит MPI_COMM_NULL).
  int color = (wrank < WORK) ? 1 : MPI_UNDEFINED;

  MPI_Comm active = MPI_COMM_NULL;
  int rc = MPI_Comm_split(world, color, wrank, &active);
  if (rc != MPI_SUCCESS) die_mpi("MPI_Comm_split(active)", rc); // Группа не создалась - смерть

  return active;
}

// ---------------- Метод Гаусса (вычислительная часть) ----------------
/*
 * gauss_run — выполняет метод Гаусса с поддержкой:
 *  - продолжения с произвольного (phase, step) (т.е. после восстановления),
 *  - периодических checkpoint на прямом ходе,
 *  - инъекции отказа для тестирования (через FAIL_*).
 *
 * Возвращает:
 *  - MPI_SUCCESS при успехе,
 *  - или MPI-ошибку (в т.ч. ULFM PROC_FAILED/REVOKED), которую main будет обрабатывать.
 *
 * Важные параметры:
 *  - phase_io/step_io: входное и выходное состояние алгоритма (checkpoint state).
 *  - row0/nloc/A: локальный блок строк.
 *  - last_phase/last_step: последние "известные" координаты, полезно логировать при сбое.
 */
static int gauss_run(MPI_Comm active, int N,
                     int* phase_io, int* step_io,
                     int row0, int nloc, float* A,
                     const char* ckpt_path, int ckpt_period,
                     int* last_phase, int* last_step) {
  int arank, asize;
  MPI_Comm_rank(active, &arank);
  MPI_Comm_size(active, &asize);

  // world_rank используется для искусственной инъекции отказа по глобальному рангу MPI_COMM_WORLD.
  // Важно: после shrink world_rank в новом мире изменится, но в данном тестовом коде инъекция рассчитана на один сбой.
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Параметры инъекции сбоя (для тестирования отказоустойчивости).
  int fail_rank = -1, fail_step = -1, fail_phase = -1;
  const char* s;
  if ((s = getenv("FAIL_RANK")))  fail_rank  = atoi(s);
  if ((s = getenv("FAIL_STEP")))  fail_step  = atoi(s);
  if ((s = getenv("FAIL_PHASE"))) fail_phase = atoi(s);

  // Буфер для pivot-строки (размер N+1, включая правую часть).
  float* pivot = (float*)malloc((size_t)(N + 1) * sizeof(float));
  if (!pivot) die("malloc pivot failed");

  // Загружаем состояние вычисления:
  // если мы восстановились из чекпоинта, phase/step будут не начальные.
  int phase = *phase_io;
  int step  = *step_io;

  // ---------------- Прямой ход (elimination) ----------------
  if (phase == PH_ELIM) {
    // Идём по pivot-строкам с текущего шага step.
    for (int i = step; i < N - 1; i++) {
      // Эти значения записываем, чтобы при сбое можно было логировать "где мы были".
      *last_phase = PH_ELIM;
      *last_step  = i;

      // Инъекция отказа: на заданном ранге/фазе/шаге процесс убивает себя SIGKILL.
      // Делается ДО коллективных операций, чтобы проверить обработку PROC_FAILED в bcast/barrier.
      if (world_rank == fail_rank && fail_phase == PH_ELIM && fail_step == i) {
        fprintf(stderr, "KILLING myself: world_rank=%d pid=%d phase=%d step=%d\n",
                world_rank, getpid(), phase, i);
        fflush(stderr);
        raise(SIGKILL); // Процесс мгновенно завершается (симуляция "жёсткого" отказа).
      }

      // Определяем владельца pivot-строки i.
      int owner = owner_of_row(N, asize, i);

      // Владелец копирует свою локальную строку в pivot-буфер.
      // Это нужно, потому что затем pivot будет рассылаться всем через MPI_Bcast.
      if (arank == owner) {
        int li = i - row0; // локальный индекс строки i внутри Aloc
        memcpy(pivot, &A[li * (N + 1)], (size_t)(N + 1) * sizeof(float));
      }

      // Рассылка pivot-строки всем процессам active.
      // Это синхронизационный и коммуникационный "узел" алгоритма:
      // все должны иметь одинаковый pivot, прежде чем обновлять свои строки.
      int rc = MPI_Bcast(pivot, N + 1, MPI_FLOAT, owner, active);
      if (rc != MPI_SUCCESS) { free(pivot); return rc; }

      // Обновление локальных строк k > i (каждый процесс делает это для своих строк параллельно).
      // Здесь и есть основной вычислительный параллелизм: обновления строк независимы после получения pivot.
      for (int lk = 0; lk < nloc; lk++) {
        int k = row0 + lk;
        if (k <= i) continue; // строки выше/на диагонали не трогаем

        float aik = A[lk * (N + 1) + i]; // элемент A[k,i], используемый как множитель A[k][i]
        float pii = pivot[i];            // pivot[i] — диагональный элемент pivot-строки

        // Обновляем элементы справа от pivot-столбца: j=i+1..N (включая RHS при j==N).
        for (int j = i + 1; j <= N; j++) {
          A[lk * (N + 1) + j] -= aik * pivot[j] / pii;
        }
      }

      // Периодический checkpoint: каждые ckpt_period шагов.
      // Почему на прямом ходе: проще сохранять состояние (матрица A|b достаточно), чем на back-phase.
      // Почему сохраняем step=i+1: чтобы после восстановления продолжить со следующего pivot.
      if (ckpt_period > 0 && (i % ckpt_period == 0)) {
        rc = ckpt_write_parallel(active, ckpt_path, N, PH_ELIM, i + 1, row0, nloc, A);
        if (rc != MPI_SUCCESS) { free(pivot); return rc; }
      }
    }

    // Прямой ход завершён => переходим к обратному ходу.
    phase = PH_BACK;
    step  = N - 2; // последний индекс, с которого начинается back-substitution

    // Важный checkpoint на границе фаз. Выступает своего рода синхранизационной точкой. Все процессы завершили фазу ELIM
    // Он нужен, чтобы после сбоя можно было начать back-phase,
    // а не повторять весь elimination (если периодический чекпоинт был давно).
    {
      int rc = ckpt_write_parallel(active, ckpt_path, N, phase, step, row0, nloc, A);
      if (rc != MPI_SUCCESS) { free(pivot); return rc; }
    }
  }

  // ---------------- Обратный ход (back substitution) ----------------
  // В вашем коде чекпоинтов на обратном ходе нет (см. комментарий ниже).
  // Это означает: если сбой произойдёт на back-phase, восстановление откатится к чекпоинту на границе фаз
  // (или к последнему чекпоинту elimination, если граница не была записана).
  float* X = (float*)calloc((size_t)N, sizeof(float));
  if (!X) die("calloc X failed");

  // Сначала вычисляем последний элемент X[N-1] у владельца последней строки.
  int owner_last = owner_of_row(N, asize, N - 1);
  if (arank == owner_last) {
    int ll = (N - 1) - row0;
    X[N - 1] = A[ll * (N + 1) + N] / A[ll * (N + 1) + (N - 1)]; // A[N-1][N] / A[N-1][N-1]
  }

  // Рассылаем X[N-1] всем: он нужен всем для обновления правых частей на следующих шагах.
  int rc = MPI_Bcast(&X[N - 1], 1, MPI_FLOAT, owner_last, active);
  if (rc != MPI_SUCCESS) { free(X); free(pivot); return rc; }

  // Теперь идём вниз по j: X[j] зависит от уже найденного X[j+1], X[j+2]...
  for (int j = step; j >= 0; j--) {
    *last_phase = PH_BACK;
    *last_step  = j;

    // Инъекция сбоя на обратном ходе (тест отказоустойчивости).
    if (world_rank == fail_rank && fail_phase == PH_BACK && fail_step == j) {
      logger(active, "INJECT_FAIL: world_rank=%d phase=BACK step=%d", world_rank, j);
      raise(SIGKILL);
    }

    // Обновляем правую часть у строк k <= j с использованием уже известного X[j+1].
    // (Упрощённый вариант вычислений; важно, что обновление локально и параллельно по строкам.)
    for (int lk = 0; lk < nloc; lk++) {
      int k = row0 + lk;
      if (k > j) continue; // обновляем только строки выше/на j

      A[lk * (N + 1) + N] -= A[lk * (N + 1) + (j + 1)] * X[j + 1];
    }

    // Владелец строки j вычисляет X[j] и затем рассылает.
    int owner = owner_of_row(N, asize, j);
    if (arank == owner) {
      int lj = j - row0;
      X[j] = A[lj * (N + 1) + N] / A[lj * (N + 1) + j];
    }

    rc = MPI_Bcast(&X[j], 1, MPI_FLOAT, owner, active);
    if (rc != MPI_SUCCESS) { free(X); free(pivot); return rc; }
  }

  // Выводим первые элементы решения (только rank 0 active), чтобы не дублировать вывод.
  if (arank == 0) {
    printf("X=(");
    int m = (N > 9 ? 9 : N);
    for (int i = 0; i < m; i++) printf("%.4g%s", X[i], (i % 10 == 9 ? "\n" : ", "));
    printf("...)\n");
    fflush(stdout);
  }

  // Освобождаем временные буферы.
  free(X);
  free(pivot);

  // Обновляем состояние, которое хранит вызывающий код (main).
  *phase_io = phase;
  *step_io  = step;

  return MPI_SUCCESS;
}

// ---------------- main: инициализация, запуск, отказоустойчивый цикл ----------------

int main(int argc, char** argv) {
  // Инициализация MPI. После этого доступны MPI_Comm_rank/size, MPI_Wtime и т.д.
  MPI_Init(&argc, &argv);

  // Инициализация логирования (возможность отключить через GAUSS_LOG).
  log_init();

  // world — текущий "мир" процессов приложения. Стартует как MPI_COMM_WORLD,
  // но после сбоя будет заменён на результат MPIX_Comm_shrink (world2).
  MPI_Comm world = MPI_COMM_WORLD;

  // Важнейшая настройка для ULFM: вместо аварийного завершения MPI по ошибке,
  // коллективные операции вернут код rc, который мы сможем обработать.
  MPI_Comm_set_errhandler(world, MPI_ERRORS_RETURN);

  // Узнаём rank/size в исходном мире.
  int world_rank, world_size;
  MPI_Comm_rank(world, &world_rank);
  MPI_Comm_size(world, &world_size);

  // Проверка аргументов командной строки.
  // Ожидается: входной файл data.in, путь чекпоинта, WORK (число рабочих процессов).
  if (argc < 4) {
    if (world_rank == 0) fprintf(stderr, "Usage: %s data.in ckpt_path WORK\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  const char* in_path   = argv[1];  // путь к файлу с N (в вашем примере)
  const char* ckpt_path = argv[2];  // путь к файлу checkpoint
  int WORK = atoi(argv[3]);         // число рабочих процессов (остальные — spares/резерв)
  if (WORK <= 0 || WORK > world_size) die("Bad WORK value");

  // Логируем конфигурацию: сколько процессов всего, сколько рабочих и сколько резервов.
  if (world_rank == 0) {
    logger(world, "START: world_size=%d WORK=%d SPARES=%d", world_size, WORK, world_size - WORK);
  }

  // Строим active-коммуникатор: только ранги < WORK входят в вычисления.
  // Резервные ранги получают active == MPI_COMM_NULL.
  MPI_Comm active = build_active_from_world(world, WORK);

  // Для active тоже ставим MPI_ERRORS_RETURN, чтобы bcast/barrier возвращали rc при сбое.
  if (active != MPI_COMM_NULL) MPI_Comm_set_errhandler(active, MPI_ERRORS_RETURN);

  // Состояние вычисления, которое будет сохраняться в чекпоинте и восстанавливаться:
  int N = 0, phase = PH_ELIM, step = 0;

  // Период чекпоинтов на elimination. 2 означает "каждые 2 шага".
  // Это компромисс: чаще => меньше потери при откате, но больше I/O накладных расходов.
  int ckpt_period = 2;

  // -------- Чтение N --------
  // Читаем N только на rank 0 active, затем рассылаем по active.
  // Почему не по world сразу: резервы не участвуют в вычислениях и могут вообще не иметь доступа к файлу ввода.
  // Но затем N всё равно надо сообщить всем в world, т.к. после shrink резерв может стать рабочим.
  if (active != MPI_COMM_NULL) {
    int arank;
    MPI_Comm_rank(active, &arank);

    if (arank == 0) {
      FILE* in = fopen(in_path, "r");
      if (!in) die("Cannot open data.in");
      if (fscanf(in, "%d", &N) != 1) die("Wrong data.in");
      fclose(in);
      logger(active, "READ N=%d", N);
    }

    // Рассылаем N всем активным процессам.
    int rc = MPI_Bcast(&N, 1, MPI_INT, 0, active);
    if (rc != MPI_SUCCESS) die_mpi("MPI_Bcast(N)", rc);
  }

  // Теперь рассылаем N по всему world (включая spares).
  // Это важно, потому что при восстановлении состав active может измениться,
  // и бывший spare должен знать N, чтобы выделить правильный буфер и читать checkpoint.
  int rc = MPI_Bcast(&N, 1, MPI_INT, 0, world);
  if (rc != MPI_SUCCESS) die_mpi("MPI_Bcast(N,world)", rc);

  // Локальная часть матрицы A|b (строки данного процесса active).
  float* Aloc = NULL;
  int row0 = 0, nloc = 0;

  // -------- Инициализация данных и первый checkpoint --------
  if (active != MPI_COMM_NULL) {
    // Выделяем память и определяем диапазон строк в active.
    active_alloc_rebuild(active, N, &row0, &nloc, &Aloc);

    // Инициализация матрицы (в вашем коде это демонстрационная "почти единичная" матрица с RHS=1).
    // Для реальной задачи обычно читают матрицу/вектор из файла и распределяют.
    for (int lk = 0; lk < nloc; lk++) {
      int i = row0 + lk;
      for (int j = 0; j <= N; j++) {
        Aloc[lk * (N + 1) + j] = (i == j || j == N) ? 1.f : 0.f;
      }
    }

    // Пишем начальный checkpoint (phase=ELIM, step=0), чтобы было откуда восстанавливаться даже при раннем сбое.
    rc = ckpt_write_parallel(active, ckpt_path, N, phase, step, row0, nloc, Aloc);
    if (rc != MPI_SUCCESS) die_mpi("initial checkpoint", rc);
  }

  // Замер времени всей вычислительной части (включая возможное восстановление).
  double t0 = MPI_Wtime();

  // last_phase/last_step используются для диагностики и логирования во время revoke/recovery.
  int last_phase = PH_ELIM;
  int last_step  = 0;

  // revoked — локальный флаг: был ли уже вызван revoke (чтобы не делать это повторно).
  int revoked = 0;

  // -------- Отказоустойчивый цикл --------
  // Логика:
  //  1) active выполняет gauss_run,
  //  2) все (world) встречаются на барьере,
  //  3) если барьер успешен — можно завершать,
  //  4) если барьер возвращает ULFM-ошибку — делаем shrink, перестраиваем active, читаем checkpoint и продолжаем.
  while (1) {
    int active_rc = MPI_SUCCESS;

    // Рабочие процессы (active != NULL) выполняют вычисление.
    // Резервные процессы просто пропускают вычисление и "ждут" на барьерах world.
    if (active != MPI_COMM_NULL) {
      active_rc = gauss_run(active, N, &phase, &step, row0, nloc, Aloc,
                            ckpt_path, ckpt_period, &last_phase, &last_step);

      // Если gauss_run вернул ошибку, проверяем: это ULFM или "обычная" ошибка.
      if (active_rc != MPI_SUCCESS) {
        if (is_ulfm_failure(active_rc)) {
          // При PROC_FAILED инициируем revoke, чтобы вывести всех из коллективных операций.
          (void)maybe_revoke_on_failure(world, active_rc, &revoked, last_phase, last_step);
        } else {
          // Любая другая MPI-ошибка в вычислениях — считаем фатальной.
          die_mpi("MPI error in ACTIVE (not ULFM)", active_rc);
        }
      }
    }

    // Барьер по world — ключевой "синхронизатор завершения/восстановления":
    //  - если никто не умер, барьер завершится успешно у всех,
    //  - если кто-то умер, у живых вернётся ошибка (PROC_FAILED или REVOKED).
    rc = MPI_Barrier(world);
    if (rc == MPI_SUCCESS) break; // Нормальное завершение (без необходимости восстановления).

    // Если барьер провалился, но это не ULFM-ошибка — считаем ситуацию нештатной.
    if (!is_ulfm_failure(rc)) die_mpi("WORLD barrier failed (not ULFM)", rc);

    // Если это ULFM-ошибка, возможно, ещё не было revoke (например, ошибка проявилась сначала на барьере).
    (void)maybe_revoke_on_failure(world, rc, &revoked, last_phase, last_step);

    // -------- Восстановление: shrink --------
    // MPIX_Comm_shrink создаёт новый коммуникатор, содержащий только "живых" процессов.
    // Все выжившие получают согласованный новый world2.
    MPI_Comm world2;
    MPIX_Comm_shrink(world, &world2);

    // Освобождаем старый world, если он не MPI_COMM_WORLD (MPI_COMM_WORLD освобождать нельзя).
    if (world != MPI_COMM_WORLD) MPI_Comm_free(&world);

    // Переходим на новый world.
    world = world2;
    MPI_Comm_set_errhandler(world, MPI_ERRORS_RETURN);

    int new_wr, new_ws;
    MPI_Comm_rank(world, &new_wr);
    MPI_Comm_size(world, &new_ws);

    // Диагностический блок: собираем информацию о "выживших" (какие initial ranks остались).
    // Это чисто для отчёта/отладки и не влияет на вычисление.
    {
      int* surv_old = NULL;
      if (new_wr == 0) surv_old = (int*)malloc((size_t)new_ws * sizeof(int));

      // Важно: world_rank здесь — ранг исходного MPI_COMM_WORLD (до shrink).
      // В вашем тесте предполагается один сбой; при множественных сбоях нужно хранить историю аккуратнее.
      int my_initial = world_rank;

      // Собираем initial ranks на корне нового мира.
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

    // После shrink всем нужно иметь N. Мы снова рассылаем его по world, чтобы все были согласованы.
    rc = MPI_Bcast(&N, 1, MPI_INT, 0, world);
    if (rc != MPI_SUCCESS) die_mpi("MPI_Bcast(N after shrink)", rc);

    // -------- Перестроение active --------
    // Старый active (если был) освобождаем: состав/ранги могли измениться.
    if (active != MPI_COMM_NULL) MPI_Comm_free(&active);

    // Собираем новый active по правилу "первые WORK рангов нового мира".
    // Это и есть сценарий (a): "запасные" процессы уже запущены, и теперь могут стать рабочими.
    active = build_active_from_world(world, WORK);
    if (active != MPI_COMM_NULL) MPI_Comm_set_errhandler(active, MPI_ERRORS_RETURN);

    // -------- Восстановление данных из checkpoint --------
    if (active != MPI_COMM_NULL) {
      // Пересчитываем row0/nloc и выделяем новый буфер под новый размер active.
      active_alloc_rebuild(active, N, &row0, &nloc, &Aloc);

      // Читаем checkpoint: получаем актуальные N/phase/step и локальные строки матрицы.
      // (N может теоретически совпадать, но мы всё равно читаем header для консистентности.)
      rc = ckpt_read_parallel(active, ckpt_path, &N, &phase, &step, row0, nloc, Aloc);
      if (rc != MPI_SUCCESS) die_mpi("checkpoint_load after recovery", rc);
    }

    // Сбрасываем флаг revoke: для следующего возможного сбоя (если бы тестировали несколько).
    revoked = 0;
  }

  // Конец замера времени.
  double t1 = MPI_Wtime();

  // Вывод времени делает только rank 0 active, чтобы не плодить одинаковые строки.
  if (active != MPI_COMM_NULL) {
    int ar;
    MPI_Comm_rank(active, &ar);
    if (ar == 0) printf("Time in seconds=%gs\n", (t1 - t0));
  }

  // Освобождение ресурсов.
  free(Aloc);
  if (active != MPI_COMM_NULL) MPI_Comm_free(&active);
  if (world != MPI_COMM_WORLD) MPI_Comm_free(&world);

  // Финализация MPI. После этого нельзя делать MPI-вызовы.
  MPI_Finalize();
  return 0;
}