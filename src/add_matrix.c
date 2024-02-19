#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void initialize_matrix(int **matrix, int size) {
  srand(time(0));
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      matrix[i][j] = rand() % 100; // Generate random integers between 0 and 99
    }
  }
}

void add_matrices(int **matrix1, int **matrix2, int **result, int size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      result[i][j] = matrix1[i][j] + matrix2[i][j];
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <size>\n", argv[0]);
    return 1;
  }

  int size = atoi(argv[1]);
  int **matrix1 = (int **)malloc(size * sizeof(int *));
  int **matrix2 = (int **)malloc(size * sizeof(int *));
  int **result = (int **)malloc(size * sizeof(int *));
  for (int i = 0; i < size; i++) {
    matrix1[i] = (int *)malloc(size * sizeof(int));
    matrix2[i] = (int *)malloc(size * sizeof(int));
    result[i] = (int *)malloc(size * sizeof(int));
  }

  clock_t start, end;
  double cpu_time_used;

  start = clock();
  initialize_matrix(matrix1, size);
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("%d,Initialization,matrix1,%f\n", size, cpu_time_used);

  start = clock();
  initialize_matrix(matrix2, size);
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("%d,Initialization,matrix2,%f\n", size, cpu_time_used);

  start = clock();
  add_matrices(matrix1, matrix2, result, size);
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("%d,Addition,result,%f\n", size, cpu_time_used);

  for (int i = 0; i < size; i++) {
    free(matrix1[i]);
    free(matrix2[i]);
    free(result[i]);
  }
  free(matrix1);
  free(matrix2);
  free(result);

  return 0;
}
