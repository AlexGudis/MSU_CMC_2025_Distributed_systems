#include <mpi.h>
#include <stdio.h>

int main(int argc,char**argv){
  MPI_Init(&argc,&argv);
  MPI_File fh;
  int rc = MPI_File_open(MPI_COMM_WORLD, "/tmp/mpiio_test.bin",
                         MPI_MODE_CREATE|MPI_MODE_WRONLY,
                         MPI_INFO_NULL, &fh);
  if(rc!=MPI_SUCCESS){
    char err[MPI_MAX_ERROR_STRING]; int len=0;
    MPI_Error_string(rc, err, &len);
    int r; MPI_Comm_rank(MPI_COMM_WORLD,&r);
    fprintf(stderr,"rank %d: %s\n", r, err);
    MPI_Abort(MPI_COMM_WORLD,1);
  }
  MPI_File_close(&fh);
  MPI_Finalize();
  return 0;
}
