#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 8
#define V 64

static inline int imax(int a, int b){ return a > b ? a : b; }

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

    // 2D cartesian grid 8x8
    int dims[2] = {N, N};
    int periods[2] = {0, 0};
    MPI_Comm grid;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid);

    int coords[2];
    MPI_Cart_coords(grid, rank, 2, coords);
    int y = coords[0], x = coords[1];

    int left, right, up, down;
    MPI_Cart_shift(grid, 1, 1, &left, &right); // shift in x (dim=1)
    MPI_Cart_shift(grid, 0, 1, &up, &down);    // shift in y (dim=0)

    // Input: A[64] (example init)
    int A[V], buf[V];
    for(int i=0;i<V;i++) A[i] = rank + i;
    for(int i=0;i<V;i++) buf[i] = A[i];

    MPI_Status st;

    // ---------------------------
    // Phase A1: row-reduce to x=0 (shift left)
    // For k = 7..1: nodes with x==k send to left, nodes with x==k-1 recv from right
    for(int k = N-1; k >= 1; --k){
        if(x == k){
            MPI_Send(buf, V, MPI_INT, left, 1000 + y, grid);
        } else if(x == k-1){
            int tmp[V];
            MPI_Recv(tmp, V, MPI_INT, right, 1000 + y, grid, &st);
            for(int i=0;i<V;i++) buf[i] = imax(buf[i], tmp[i]);
        }
    }

    // ---------------------------
    // Phase A2: column-reduce on x=0 to y=0 (shift up)
    if(x == 0){
        for(int k = N-1; k >= 1; --k){
            if(y == k){
                MPI_Send(buf, V, MPI_INT, up, 2000, grid);
            } else if(y == k-1){
                int tmp[V];
                MPI_Recv(tmp, V, MPI_INT, down, 2000, grid, &st);
                for(int i=0;i<V;i++) buf[i] = imax(buf[i], tmp[i]);
            }
        }
    }

    // Now root (0,0) has full reduced vector in buf[64]

    // ---------------------------
    // Phase B1: scatter row-blocks (8 ints) down column x=0
    int rowblock[N];         // 8 ints for this row leader
    int have_rowblock = 0;

    if(x == 0){
        // root starts with whole R in buf[]
        if(y == 0){
            // keep own rowblock [0..7]
            for(int i=0;i<N;i++) rowblock[i] = buf[i];
            have_rowblock = 1;

            // send tail for rows 1..7 (56 ints)
            int tailCount = (N-1)*N; // 56
            MPI_Send(&buf[N], tailCount, MPI_INT, down, 3000, grid);
        } else {
            // receive tail from above: size = (N - y)*N ints
            int cnt = (N - y)*N;
            int* tail = (int*)malloc(cnt * sizeof(int));
            MPI_Recv(tail, cnt, MPI_INT, up, 3000, grid, &st);

            // first 8 ints are my rowblock
            for(int i=0;i<N;i++) rowblock[i] = tail[i];
            have_rowblock = 1;

            // forward remaining (cnt-8) ints further down, if any
            if(y != N-1){
                int rest = cnt - N;
                MPI_Send(&tail[N], rest, MPI_INT, down, 3000, grid);
            }
            free(tail);
        }
    }

    // ---------------------------
    // Phase B2: scatter 1 int along each row from x=0 to x=7
    int my_result = -1;

    if(x == 0){
        // row leader: my element is rowblock[0]
        if(have_rowblock) my_result = rowblock[0];

        // send tail elements [1..7] as decreasing message along the row
        if(N > 1){
            MPI_Send(&rowblock[1], N-1, MPI_INT, right, 4000 + y, grid);
        }
    } else {
        // receive tail of length (N-x) from left
        int cnt = N - x;
        int* tail = (int*)malloc(cnt * sizeof(int));
        MPI_Recv(tail, cnt, MPI_INT, left, 4000 + y, grid, &st);

        // first element is mine
        my_result = tail[0];

        // forward remaining to the right if any
        if(x != N-1){
            MPI_Send(&tail[1], cnt-1, MPI_INT, right, 4000 + y, grid);
        }
        free(tail);
    }

    // Проверка
    // printf("rank=%d (x=%d,y=%d) got R[%d]=%d\n", rank, x, y, rank, my_result);

    int expected = 63 + rank;
    if (my_result != expected) {
        printf("ERROR rank=%d got=%d expected=%d\n", rank, my_result, expected);
    }

    MPI_Comm_free(&grid);
    MPI_Finalize();

    // printf("done");
    return 0;
}
