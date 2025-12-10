#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <mpi.h>


// Печать
void prt1a(char *t1, double *v, int n, char *t2)
{
    int j;
    printf("%s", t1);
    for (j = 0; j < n; j++)
        printf("%.2lf%s", v[j], j % 10 == 9 ? "\n" : ", ");
    printf("%s", t2);
}

int N, myN;
double *A;
#define A(i, j) A[(i) * (myN + 1) + (j)]
double *X;
int proc_num, myrank;

int main(int argc, char **argv)
{
    // Инициализация среды выполнения MPI-программы.
    MPI_Init(&argc, &argv);

    // Определение количества процессов
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

    // MPI_COMM_WORLD — стандартный коммуникатор, представляющий все процессы, запущенные в рамках MPI-программы

    // Определения ранга процесса
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int i, j, k;
    FILE *in;
    in = fopen("data.in", "r");
    if (in == NULL)
    {
        printf("Can not open 'data.in' ");
        exit(1);
    }
    i = fscanf(in, "%d", &N);
    if (i < 1)
    {
        printf("Wrong 'data.in' (N ...)");
        exit(2);
    }

    // Каждый процесс отвечает за свои столбцы
    int j_start = (myrank * (N + 1)) / proc_num;
    int j_end = ((myrank + 1) * (N + 1)) / proc_num;
    myN = j_end - j_start; // кол-во столбцов для процесса

    // create arrays
    A = (double *)malloc((myN + 1) * N * sizeof(double));
    X = (double *)malloc(N * sizeof(double));
    if (myrank == 0)
    {
        printf("----------------------\n");
        printf("GAUSS %dx%d THREADS=%d\n", N, N, proc_num);
    }

    // initialize array A
    for (i = 0; i < N; i++)
        for (j = 0; j < myN; j++)
            if (i == j + j_start || j + j_start == N)
                A(i, j) = 1.f;
            else
                A(i, j) = 0.f;

    double time0 = MPI_Wtime();

    // Конструируем производный тип с помощью структурного способа как наиболее общего 
    MPI_Datatype vectype;
    MPI_Type_vector(N, 1, myN + 1, MPI_DOUBLE, &vectype);
    /*  •N - количество блоков,
        •1 - размер каждого блока,
        •myN - количество элементов, расположенных между двумя соседними блоками
        •MPI_DOUBLE - исходный тип данных,
        •vectype - новый определяемый тип данных
    */
    
    // Фиксируем объявление производного типа данных
    MPI_Type_commit(&vectype);

    // elimination
    for (i = 0; i < N - 1; i++)
    {
        // Находим номер процесса с текущим столбцом
        int index = i;
        int proc_id = 0;
        int j_start = 0;
        int j_end = (N + 1) / proc_num;
        while (j_end <= index)
        {
            proc_id += 1;
            j_start = j_end;
            j_end = ((proc_id + 1) * (N + 1)) / proc_num;
        }
        index -= j_start;
        
        // Передаем главный элемент
        int bcast = A(i, index);
        MPI_Bcast(&bcast, 1, MPI_DOUBLE, proc_id, MPI_COMM_WORLD);
        /* int MPI_Bcast( void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm )
            buffer - адрес начала расположения в памяти рассылаемых данных;
            count - число посылаемых элементов;
            datatype - тип посылаемых элементов;
            root- номер процесса-отправителя;
            comm - коммуникатор.
        */

        // Осуществляем копирование элементов найденного столбца
        for (int i = 0; i < N; i++)
            A(i, myN) = A(i, index);
        MPI_Bcast(&A(0, myN), 1, vectype, proc_id, MPI_COMM_WORLD);

        // Находим зону "ответственности" текущего mpi процесса
        if (myrank < proc_id)
            j_start = myN;
        else if (myrank > proc_id)
            j_start = 0;
        else
            j_start = index + 1;

        // Классик часть из исходного алгоритма
        // Исключаем x_i из ур-я ниже текущего
        for (k = i + 1; k <= N - 1; k++)
            for (j = j_start; j < myN; j++)
                A(k, j) = A(k, j) - A(k, myN) * A(i, j) / bcast;
    }

    // reverse substitution
    // Нужно ли вычислять решение? (Если процесс последний)
    if (myrank == proc_num - 1)
    {
        X[N - 1] = A(N - 1, myN - 1) / A(N - 1, myN - 2);
    }
    for (j = N - 2; j >= 0; j--)
    {
        // Определяем процесс, который сейчас работает над этим столбцом
        int index = j + 1;
        int proc_id = 0;
        int j_start = 0;
        int j_end = (N + 1) / proc_num;
        while (j_end <= index)
        {
            proc_id += 1;
            j_start = j_end;
            j_end = ((proc_id + 1) * (N + 1)) / proc_num;
        }
        index -= j_start;
        
        // Определяем процесс который работает с нашей строкой
        int cent_index = j;
        int proc_id_2 = 0;
        int j_start_2 = 0;
        int j_end_2 = (N + 1) / proc_num;
        while (j_end_2 <= cent_index)
        {
            proc_id_2 += 1;
            j_start_2 = j_end_2;
            j_end_2 = ((proc_id_2 + 1) * (N + 1)) / proc_num;
        }
        cent_index -= j_start_2;

        // Передача главного элемента
        int bcast = A(j, cent_index);
        MPI_Bcast(&bcast, 1, MPI_DOUBLE, proc_id_2, MPI_COMM_WORLD);
        
        // Передаем столбца коэффициентов
        for (int i = 0; i < N; i++)
            A(i, myN) = A(i, index);
        MPI_Bcast(&A(0, myN), 1, vectype, proc_id, MPI_COMM_WORLD);
        
        // Классическая часть алгоритма
        // Если процесс последний, то обновляем свободные члены строк 0 - j и находим x[j]
        if (myrank == proc_num - 1)
        {
            for (k = 0; k <= j; k++)
                A(k, myN - 1) = A(k, myN - 1) - A(k, myN) * X[j + 1];
            X[j] = A(j, myN - 1) / bcast;
        }
    }

    // Аннулируем использование производного типа
    MPI_Type_free(&vectype);

    double time1 = MPI_Wtime();

    if (myrank == proc_num - 1)
    {
        printf("Time in seconds=%gs\n", time1 - time0);
        prt1a("X=(", X, N > 9 ? 9 : N, "...)\n");
    }

    free(A);
    free(X);
    MPI_Finalize();
    return 0;
}



