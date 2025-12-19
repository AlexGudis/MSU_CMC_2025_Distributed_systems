#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 8
#define V 64

int max(int a, int b){ return a > b ? a : b; }

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // fprintf(stderr, "rank=%d size=%d\n", rank, size);

    if(size != N*N){
        if(rank == 0) fprintf(stderr, "Run with %d processes\n", N*N);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int dims[2] = {N, N};
    int periods[2] = {0, 0}; // Замкнуть как тор? (Карта Карно из ДМ)
    MPI_Comm grid;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid);

    int coords[2];
    MPI_Cart_coords(grid, rank, 2, coords);
    int y = coords[0], x = coords[1];

    int left, right, up, down;
    MPI_Cart_shift(grid, 1, 1, &left, &right);
    MPI_Cart_shift(grid, 0, 1, &up, &down);

    // Вводим массив A[64]
    int buf[V];
    for(int i=0;i<V;i++) {
        buf[i] = rank + i;
    }

    MPI_Status st;

    // Редукция по строкам: сдвиг влево
    for(int k = N-1; k >= 1; --k){
        if(x == k){
            MPI_Send(buf, V, MPI_INT, left, 1000 + y, grid);
        } else if(x == k-1){
            int tmp[V];
            MPI_Recv(tmp, V, MPI_INT, right, 1000 + y, grid, &st);
            for(int i=0;i<V;i++) buf[i] = max(buf[i], tmp[i]);
        }
    }

    // Редукция по столбцу x=0: сдвиг вверх
    if(x == 0){
        for(int k = N-1; k >= 1; --k){
            if(y == k){
                MPI_Send(buf, V, MPI_INT, up, 2000, grid);
            } else if(y == k-1){
                int tmp[V];
                MPI_Recv(tmp, V, MPI_INT, down, 2000, grid, &st);
                for(int i=0;i<V;i++) buf[i] = max(buf[i], tmp[i]);
            }
        }
    }


    // По столбцу x=0 раздать блоки по 8 элементов лидерам строк (0,y)
    int rowblock[N];     
    int have_rowblock = 0;

    if(x == 0){
        if(y == 0){
            for(int i=0;i<N;i++) rowblock[i] = buf[i];
            have_rowblock = 1;

            int tailCount = (N-1)*N; // 56
            MPI_Send(&buf[N], tailCount, MPI_INT, down, 3000, grid);
        } else {
            int cnt = (N - y)*N;
            int* tail = (int*)malloc(cnt * sizeof(int));
            MPI_Recv(tail, cnt, MPI_INT, up, 3000, grid, &st);

            for(int i=0;i<N;i++) rowblock[i] = tail[i];
            have_rowblock = 1;

            if(y != N-1){
                int rest = cnt - N;
                MPI_Send(&tail[N], rest, MPI_INT, down, 3000, grid);
            }
            free(tail);
        }
    }

    // В каждой строке y лидер (0,y) раздаёт по одному элементу вправо по строке до нужного x
    int my_result = -1;

    if(x == 0){
        if(have_rowblock) my_result = rowblock[0];

        if(N > 1){
            MPI_Send(&rowblock[1], N-1, MPI_INT, right, 4000 + y, grid);
        }
    } else {
        int cnt = N - x;
        int* tail = (int*)malloc(cnt * sizeof(int));
        MPI_Recv(tail, cnt, MPI_INT, left, 4000 + y, grid, &st);

        my_result = tail[0];

        if(x != N-1){
            MPI_Send(&tail[1], cnt-1, MPI_INT, right, 4000 + y, grid);
        }
        free(tail);
    }

    // Проверка
    // printf("rank=%d (x=%d,y=%d) got R[%d]=%d\n", rank, x, y, rank, my_result);
    // fflush(stdout);

    int expected = 63 + rank;
    if (my_result != expected) {
        printf("ERROR rank=%d got=%d expected=%d\n", rank, my_result, expected);
    }

    MPI_Comm_free(&grid);
    MPI_Finalize();

    // printf("done");
    return 0;
}