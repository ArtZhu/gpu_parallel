/* file : search.cu
 * author : Tiane Zhu
 * date : Apr 5, 2017
 *
 * this program is implementations of the parallel search algorithm
 * 	ALGORITHM 4.1 in 
 * "An Introduction to Parallel Algorithms" - by Joseph Jaja
 *		p146 - ISBN 9-789201-548563
 *
 *
 *	2. dynamic parallelism
 */

#include "search.h"

////////////////////////////////////////////////
// 2. dynamic parallelism
////////////////////////////////////////////////
// c, q must be pre allocated space
//
__device__ class Search {
	// c, q						-- c, q array from 0 to p+1
	int * c;
	int * q;
	__device__ int l;
	__device__ int r;
	dim3 Dg, Db;

	unsigned int num_threads;

	// IMP FLAGS
	__device__ int iter;
	__device__ int dev_ret;
	__device__ int dev_ret_flag;

	public:
		__device__ Search(unsigned int nt, int * C, int * Q) : num_threads(nt), c(C), q(Q){

			unsigned int num_blocks = (1023 + num_threads) / 1024;
			unsigned int threads_per_block = num_threads > 1024 ? 1024 : num_threads;

#ifdef SEARCH_PRINT
			printf("launching %u blocks, %u threads per block.\n", num_blocks, threads_per_block);
#endif

			Dg = {num_blocks, 1, 1};
			Db = {threads_per_block, 1, 1};

			iter = 0;
			dev_ret = 0;
			dev_ret_flag = 0;

		}

		__device__ ~Search() {

		}

		/* return idx */
		__device__ int search(number * dev_x, unsigned int X_len, number target) 
		{
			//1.
			init_search(dev_X, X_len, c, num_threads);
			//2. loop
			do{
				compute<<<Dg, Db>>>(dev_X, target, c, q, num_threads, &dev_ret);
				cudaDeviceSynchronize();
				set_bounds<<<Dg, Db>>>(c, q, num_threads);
				cudaDeviceSynchronize();
			}while(dev_ret < 0 && !dev_ret_flag);
			//3.
			if(dev_ret < 0){
				set_ret1<<<Dg, Db>>>(dev_X, target, c, q, num_threads);
				cudaDeviceSynchronize();
				set_ret2<<<Dg, Db>>>(c, num_threads);
			}
		}

	private :

		// step 1.
		//set the variables
		// first step in search_main kernel
		__device__ void init_search(number * X, int n, int * c, int num_threads)
		{
			l = 0;
			r = n + 1;
			X[0] = INT_MIN;
			X[n + 1] = INT_MAX;
			c[0] = 0;
			c[num_threads + 1] = 1;

			dev_ret = -1; // for thread termination purpose
		}

		//step 2. first half
		//compute q, use q to compute c, might set dev_ret 
		__global__ void compute(number * X, number target, int * c, int * q, int num_threads, int * dev_ret)
		{

			int tid = threadIdx.x + blockIdx.x * blockDim.x;

			tid += 1; // so that idx starts from 1

			if(tid > num_threads) return;

			if(tid == 1){
				iter++;
				dev_ret_flag = (r - l <= num_threads);
			}

			//while(r - l > num_threads)
			// if statement below using dev_ret_flag as bool (replacement)

			//refactored this part because no device sync
			if(tid == 1)
				q[0] = l;
			if(tid == num_threads)
				q[num_threads + 1] = r;

			//
			q[tid] = l + tid * ((r - l) / (num_threads + 1));

			/* this is unnecessary because each thread sets its own element from q and uses it.
			//sync -- use r, l, p, tid, num_threads;
			//		 -- set q
			__syncthreads();
			 */

			if(target == X[q[tid]]){
				*dev_ret = q[tid] - 1; // so that ret idx starts from 0
				// can i return here???
				// no
				//return;
			}
			else{
				if(target > X[q[tid]])
					c[tid] = 0;
				else 
					c[tid] = 1;
			}
		}

		// step 2. second half
		// use c, q, set l, r
		__global__ void set_bounds(int * c, int * q, int num_threads)
		{

			int tid = threadIdx.x + blockIdx.x * blockDim.x;

			if(tid > num_threads) return;

			tid += 1; // so that idx starts from 1

			// if ret has been set, return, a replacement for the "return" in the conditional statement;
			if(dev_ret >= 0){
				return;
			}


			if(c[tid] < c[tid + 1]){
				l = q[tid];
				r = q[tid + 1];
				//printf("iter : %d; tid : %d setting r, l to be %d %d\n", iter, tid, r, l);
				//printf("c[%d] = %d; c[%d + 1] = %d;\n", tid, c[tid], tid, c[tid+1]);
			}

			if(tid == 1 && c[0] < c[1]){
				l = q[0];
				r = q[1];
			}
		}

		// step 3. first half
		__global__ void set_ret1(number * X, number target, int * c, int * q, int num_threads)
		{

			int tid = threadIdx.x + blockIdx.x * blockDim.x;

			tid += 1; // so that idx starts from 1

			if(tid > num_threads) return; //safety

			if(tid > r - l){ 
				//corresponds with the next syncthreads();
				return;
			}

			if(target == X[l+tid]){
				dev_ret = l + tid - 1; // so that ret idx starts from 0
			}
			else if(target > X[l+tid]){
				c[tid] = 0;
			}
			else{
				c[tid] = 1;
			}

			/* after step3 becomes 2 parts, this is unecessary.

			// as long as l + tid and X[l+tid] unique, this is unecessary.
			// just a safety for now

			// sync -- use l, X, tid, target
			//			-- set dev_ret, c
			__syncthreads();
			 */
		}

		// step 3. second half
		__global__ void set_ret2(int * c, int num_threads)
		{

			int tid = threadIdx.x + blockIdx.x * blockDim.x;

			tid += 1; // so that idx starts from 1

			if(tid > num_threads) return;

			if(dev_ret >= 0)
				return;

			if(c[tid-1] < c[tid])
				dev_ret = l + tid - 1 - 1; // so that ret idx starts from 0
		}

}



