/* file : parallel_search.cu
 * author : Tiane Zhu
 * date : Mar 23, 2017
 *
 * this program is an implementation of the parallel search algorithm
 * 	ALGORITHM 4.1 in 
 * "An Introduction to Parallel Algorithms" - by Joseph Jaja
 *		p146 - ISBN 9-789201-548563
 */

#include "parallel_search.h"

///////////////////////////////////////////////////////////
// Input to the algorithm //
// X 										-- strictly ordered array
// y (target) 					-- target
// p (num_threads) 			-- num_processor
// j (tid) 							-- processor idx
///////////////////////////////////////////////////////////
// Output 
// i (ret) 							-- X[i] <= y < x[i+1]
//		[ i is initialized to -1 , since it has only non-neg values
//			i non-neg => i set ]
///////////////////////////////////////////////////////////
/* kernel strictly following algorithm */
// additional inputs
__device__ int l;
// l										
__device__ int r;
// r
// n is the number of elements
inline void gpu_search(number target, int X_len, int num_threads)
{
	//1.
	init_search<<<1, 1>>>(dev_X, X_len, c, num_threads);
	//2. loop
	do{
		compute<<<d->Dg, d->Db>>>(dev_X, target, c, q, num_threads, &dev_ret);
		set_bounds<<<d->Dg, d->Db>>>(c, q, num_threads);
		gerror(cudaMemcpyFromSymbol(&host_ret_flag, dev_ret_flag, sizeof(int), 0, cudaMemcpyDeviceToHost));
		gerror(cudaMemcpyFromSymbol(&ret_idx, dev_ret, sizeof(int), 0, cudaMemcpyDeviceToHost));
	}while(ret_idx < 0 && !host_ret_flag);
	//3.
	if(ret_idx < 0){
		set_ret1<<<d->Dg, d->Db>>>(dev_X, target, c, q, num_threads);
		set_ret2<<<d->Dg, d->Db>>>(c, num_threads);
	}
}

// step 1.
//set the variables
// just launch 1 thread for this
__global__ void init_search(number * X, int n, int * c, int num_threads)
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
#ifdef PRETTY_PRINT
		if(tid == 1)
			printf("dev ret0 : %d\n", dev_ret);
#endif
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

#ifdef PRETTY_PRINT
	printf("dev ret1 : %d\n", dev_ret);
#endif

	if(dev_ret >= 0)
		return;

	if(c[tid-1] < c[tid])
		dev_ret = l + tid - 1 - 1; // so that ret idx starts from 0

#ifdef PRETTY_PRINT
	printf("dev ret2 : %d\n", dev_ret);
#endif
}


//---------- old search kernel -------------------------
/*
{


	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	tid += 1; // so that idx starts from 1


#ifdef PRETTY_PRINT
	if(tid == 1)
		printf("KERNEL : \narray length : %d, target : %d, num_threads : %d\n", n, target, num_threads);
#endif

	//if(tid > n) return; // safety

	//1. use init_search kernel
	//device threadfence?


#ifdef PRETTY_PRINT
	if(tid == 1){
		if(n <= 32){
			for(int i=0; i<n+2; i++) printf("%d ", X[i]); 
		}
		//printf("\n");
		//printf("| q0 | q1 | q2 | q3 | c0 | c1 | c2 | c3 | l  | r  |\n");
		printf("1. r : %d ; l : %d\n", r, l);
	}
#endif

	// OLD sync for step 1. still in this kernel
	//sync -- 
	//		 -- set r, l, X, c, dev_ret, tid
	__syncthreads();

	//2.

	//use compute function here

	// this actually requires device sync
	//sync -- use X, q, target, tid
	//     -- set dev_ret, c
	__syncthreads();

	//use set_bounds function here

	// this actually requires device sync
	//sync -- use dev_ret, q, c, tid
	//		 -- set l, r
	__syncthreads();

#ifdef PRETTY_PRINT
	if(tid == 1){
		printf("r : %d ; l : %d\n", r, l);
		printf("c[%d] = %d, c[%d] = %d, c[%d] = %d\n", 1023, c[1023], 1024, c[1024], 1025, c[1025]);
		//printf("|%4d|%4d|%4d|%4d|%4d|%4d|%4d|%4d|%4d|%4d|\n", q[0], q[1], q[2], q[3], c[0], c[1], c[2], c[3], l, r);
	}
#endif

	//do... while(r - l > num_threads)

		if(set_ret){

			if(tid > r - l){ 
				//corresponds with the next syncthreads();
				__syncthreads();
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

			// sync -- use l, X, tid, target
			//			-- set dev_ret, c
			__syncthreads();

#ifdef PRETTY_PRINT
			printf("dev ret1 : %d\n", dev_ret);
#endif
			if(dev_ret >= 0)
				return;

			if(c[tid-1] < c[tid])
				dev_ret = l + tid - 1 - 1; // so that ret idx starts from 0
#ifdef PRETTY_PRINT
			printf("dev ret2 : %d\n", dev_ret);
#endif
		}
}
*/
