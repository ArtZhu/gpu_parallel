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

typedef unsigned long long int ull;

// GPU

__device__ ull search_rank;

__device__ void search(number * X, int n, number target, int num_threads, int * dev_ret)
{
	__shared__ ull record;
	int l, r, *ptr;
	ull *ptr_u;

	ptr_u = &record;
	ptr = (int *) ptr_u;

	if(threadIdx.x == 0){
		*ptr = 0;
		*(ptr+1) = n+1;
	}

	__syncthreads();
	l = *ptr;
	r = *(ptr+1);

	//printf("%llx %u %d %d\n", -2L, threadIdx.x, l, r);

	int block_n, start, s, idx;
	while(r - l > 1){

		/*
		if(threadIdx.x == 0 && blockIdx.x == 0){
			printf("%llx\n", record);
			printf("%d %d\n", l, r);
		}
		*/

		block_n = (r - l) / gridDim.x;
		
		s = block_n / blockDim.x;
		s = s > 0 ? s : 1;

		start = l + (blockIdx.x * block_n);

		idx = start + threadIdx.x * s;
		if(idx < r){

			//printf("threadIdx.x : %u\nblock_n : %d\ns : %d\nstart : %d\nidx : %d\n", threadIdx.x, block_n, s, start, idx);

			if(X[idx] <= target && X[idx + s] >= target){
				*ptr = idx;
				*(ptr+1) = idx+s;

				atomicExch(&search_rank, *ptr_u);
			}
		}

		if(threadIdx.x == 0){
			record = atomicCAS(&search_rank, -2L, 0);
		}

		__syncthreads();
		l = *ptr;
		r = *(ptr+1);
	}

	/*
	if(threadIdx.x == 0 && blockIdx.x == 0){
		printf("%llx\n", record);
		printf("%d %d\n", l, r);
	}
	*/
	
	*dev_ret = *((int *) &search_rank);
}

#endif // PARALLEL_SEARCH_H
