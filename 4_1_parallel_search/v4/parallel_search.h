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
	unsigned int c_size;
	unsigned int q_size;

// GPU
__device__ number * dev_X;
int * flags;

__device__ void search(number * X, int n, number target, int * c, int * q, int num_threads, volatile int * dev_ret, int * l, int * r, int * flags, int * found);
__device__ void fix(volatile int * dev_ret, int dev_ret_len, int n, int * ret_value);
__global__ void search_main(number * X, int n, number target, int * c, int * q, int num_threads, volatile int * dev_ret, int * l, int * r, int dev_ret_len, int * ret_value, int * flags);

void _init(int argc, char ** argv);
void _init_array(int with_file);

#include "cpu_search.h"

#endif // PARALLEL_SEARCH_H
