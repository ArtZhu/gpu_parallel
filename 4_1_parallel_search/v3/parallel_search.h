/* file : parallel_search.h
 * author : Tiane Zhu
 * date : Mar 24, 2017
 *
 * header file for parallel_search.cu
 */

#ifndef PARALLEL_SEARCH_H
#define PARALLEL_SEARCH_H

#include <stdio.h>
#include <inttypes.h>
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

void _init(int argc, char ** argv);
void _init_array(int with_file);

/* ATOMIC FLAGS */
// implementation here uses an int for each flag : signaling completion for setting q and c
//			and an int for the number of iterations set by the thread setting r and l.
__device__ int iter_flag;
__device__ int * half_iter_signals;
int * host_half_iter_signals_ptr;

#include "cpu_search.h"

#endif // PARALLEL_SEARCH_H
