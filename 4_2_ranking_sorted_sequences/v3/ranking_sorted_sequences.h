/* filename : ranking_sorted_sequences.h
 * author: Tiane Zhu
 * date : Mar 28, 2017
 *
 *
 *	header file for ranking_sorted_sequences.cu
 */

#ifndef RANKING_SORTED_SEQUENCES_H
#define RANKING_SORTED_SEQUENCES_H

#include "../../gpu_utils/util.h"

#define CPU_PRETTY_PRINT 1

typedef int number;
#define FMT "%d"

__global__ void ranking_main(number * A, int n, number * B, int * ret, int m, int num_threads);
__device__ void ranking(number * A, int n, number * B, int * ret, int m, int num_threads);

#include <stdio.h>
#include "cpu_ranking.h"
#include "parallel_search.h"

#endif

