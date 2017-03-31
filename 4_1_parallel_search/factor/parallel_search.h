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

//#define PRETTY_PRINT 1

typedef int number;
#define FMT "%d "

// GPU
number * dev_X;

/* host gpu search function */
void gpu_search(number target, int X_len, int num_threads);

int * c;
// c										-- c array from 0 to p+1
int * q;
// q										-- q array from 0 to p+1

// IMP FLAGS
__device__ int iter = 0;
__device__ int dev_ret = 0;
__device__ int dev_ret_flag = 0;
//int * dev_flag_ptr, * dev_ret_ptr;

int host_ret_flag;
int ret_idx;


// step 1.
			//set the variables
			// just launch 1 thread for this
__global__ void init_search(number * X, int n, int * c, int num_threads);
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


#endif // PARALLEL_SEARCH_H
