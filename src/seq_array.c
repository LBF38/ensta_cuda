#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void initialize_array(int *array, int size) {
  srand(time(0));
  for (int i = 0; i < size; i++) {
    array[i] = rand() / (float)RAND_MAX;
  }
}

void add_arrays(int *array1, int *array2, int *result, int size) {
  for (int i = 0; i < size; i++) {
    result[i] = array1[i] + array2[i];
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <size>\n", argv[0]);
    return 1;
  }

  int size = atoi(argv[1]);
  int *array1 = malloc(size * sizeof(int));
  int *array2 = malloc(size * sizeof(int));
  int *result = malloc(size * sizeof(int));

  clock_t start, end;
  double cpu_time_used;

  start = clock();
  initialize_array(array1, size);
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("%d,Initialization,array1,%f\n", size, cpu_time_used);

  start = clock();
  initialize_array(array2, size);
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("%d,Initialization,array2,%f\n", size, cpu_time_used);

  start = clock();
  add_arrays(array1, array2, result, size);
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("%d,Addition,total,%f\n", size, cpu_time_used);

  // Print the result array
  //   for (int i = 0; i < size; i++) {
  //     printf("%d ", result[i]);
  //   }
  //   printf("\n");

  free(array1);
  free(array2);
  free(result);

  return 0;
}
