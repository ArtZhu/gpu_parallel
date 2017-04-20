/* file : parallel_search.h
 * author : Tiane Zhu
 * date : Mar 24, 2017
 *
 * header file for parallel_search.cu
 */

#ifndef PARALLEL_SEARCH_H
#define PARALLEL_SEARCH_H

#include <stdio.h>
#include "../../gpu_utils/util.h"

int verbose = 0;

//#define PRETTY_PRINT 1

typedef int number;
typedef unsigned long long int ull;
#define FMT "%d "
#define DEFAULT_ARRAY_LEN 15
#define DEFAULT_TARGET 19
#define DEFAULT_NUM_THREADS 2

// CPU
number target;
number * host_X;
int X_len;
unsigned int num_threads;

	unsigned int X_size;

// GPU
__device__ number * dev_X;

__global__ void search_main(number * X, int n, number target, int num_threads, ull * dev_ret);

void _init(int argc, char ** argv);
void _init_array(int with_file);

#include "cpu_search.h"

#endif // PARALLEL_SEARCH_H
