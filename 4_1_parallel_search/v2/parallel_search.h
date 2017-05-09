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
number * dev_X;

/* search kernel */
__global__ void search_main(number * dev_X, int X_len, exec_dim_t * d_child, number target, int num_threads, int * c, int * q);

// IMP FLAGS
__device__ int iter = 0;
__device__ int dev_ret = 0;
__device__ int dev_ret_flag = 0;
//int * dev_flag_ptr, * dev_ret_ptr;
int ret_idx_host, ret_idx_dev;
int host_ret_flag = 0;

// step 1.
			//set the variables
			// just launch 1 thread for this
__device__ void init_search(number * X, int n, int * c, int num_threads);
// step 2. first half
			//compute q, use q to compute c, might set dev_ret 
__global__ void compute(number * X, number target, int * c, int * q, int num_threads, int * dev_ret);
// step 2. second half
			// use c, q, set l, r
__global__ void set_bounds(int * c, int * q, int num_threads);
// step 3. first half
__global__ void set_ret1(number * X, number target, int * c, int * q, int num_threads);
// step 3. second half
__global__ void set_ret2(int * c, int num_threads);

// INIT
void _init(int argc, char ** argv);
void _init_array(int with_file);

#include "cpu_search.h"

#endif // PARALLEL_SEARCH_H
