/* filename : ranking_sorted_sequences.cu
 * author: Tiane Zhu
 * date : Mar 26, 2017
 *
 * this file contains an O(1) ranking parallel algorithm
 *
 * this program is an implementation of the ranking sorted sequences
 *		ALGORITHM 4.2 in 
 * "An Introduction to Parallel Algorithms" - by Joseph Jaja
 *			p150 - ISBN 9-789201-548563
 */

////
// Input : 	A = (a1, ... , an)
// 					B = (b1, ... , bm)
//			sqm = sqrt(m)
////
// Output : rank(B : A)
////
// begin
// 		1. If m < 4, then rank the elements of B 
//			 							by applying Alg 4_1 with p = n
//								 then exit
//		2. Concurrently rank b_sqm, b_2sqm, ..., bm in A 
//										by applying Alg 4_1 with p = sqrt(n)
//			 Let 	j[i] = rank(b_isqm : A)
//						j[0] = 0
//		3. For 0 <= i <= sqm - 1,
//					let B_i = ( b_isqm+1, ... , b_(i+1)sqm - 1 )
//					let A_i = ( a_j[i]+1, ... , a_j[i+1] )
//			 If j[i] == j[i+1], then
//					set rank(B_i : A_i) = (0, ... , 0)
//			 else
//					recurse compute rank(B_i : A_i)
//		4. Let 1 <= k <= m be an arbitrary index not multiple of sqm
//			 Let i = floor( k / sqm )
//			 rank(b_k : A) = j[i] + rank(b_k : A_i)
// end
////


#include "ranking_sorted_sequences.h"

int main(){
	int n = 8;
	number A[] = {-5, 0, 3, 4, 17, 18, 24, 28};
	int m = 4;
	number B[] = {1, 2, 15, 21};
	number ret[m];

	number * dev_A, * dev_B, *dev_ret;
	
	cudaMalloc(&dev_A, n * sizeof(number));
	cudaMalloc(&dev_B, m * sizeof(number));
	cudaMalloc(&dev_ret, (m + 2)* sizeof(number));

	cudaMemcpy(dev_A, A, n * sizeof(number), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B, m * sizeof(number), cudaMemcpyHostToDevice);

	int num_threads = 1024;

	ranking_main<<<1, 1024>>>(dev_A, n, dev_B, dev_ret, m, num_threads);

	cudaDeviceSynchronize();

	cudaMemcpy(ret, dev_ret+1, m * sizeof(number), cudaMemcpyDeviceToHost);
	printf("\n GPU RANKING : [ ");
	for(int i=0; i<m; i++)
		printf("%d ", ret[i]);
	printf("]\n");

	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_ret);

	cpu_ranking(A, n, B, ret, m);

	printf("\n CPU RANKING : [ ");
	for(int i=0; i<m; i++)
		printf("%d ", ret[i]);
	printf("]\n");

}

//A  = INT_MIN, a0, a1, ..., an, INT_MAX

__global__ void ranking_main(number * A, int n, number * B, int * ret, int m, int num_threads){
	if(threadIdx.x == 0 && blockIdx.x == 0){
		printf("GPU ranking with %d threads: \n", num_threads);
		dbg_array("A", A, "%d ", n);
		dbg_array("B", B, "%d ", m);
	}
	ranking(A, n, B, ret, m, num_threads);
}

__device__ void ranking(number * A, int n, number * B, int * ret, int m, int num_threads)
{
	if(threadIdx.x == 0 && blockIdx.x == 0){
		printf("ranking %d numbers in %d numbers\n", m, n);
	}
	int i;//, dev_ret;
	number target;

// 		1. If m < 4, then rank the elements of B 
//			 							by applying Alg 4_1 with p = n
//								 then exit
	if(m < 4){
		for(i = 0; i<m; i++){

			target = B[i];
			search(A, n, target, ret + i, num_threads);
			//ret[i] = *((int *) &search_rank);
			if(threadIdx.x == 0){
				printf("search for %d got %d\n", target, ret[i]);
				dbg_array("in", A, "%d ", n);
			}

			__syncthreads();
		}
		return;
	}

//		2. Concurrently rank b_sqm, b_2sqm, ..., bm in A 
//										by applying Alg 4_1 with p = sqrt(n)
//			 Let 	j[i] = rank(b_isqm : A)
//						j[0] = 0

	//j[i] == ret[i*sqm+1]
	int sqm = (int) sqrtf(m);
	if(threadIdx.x == 0 && blockIdx.x == 0){
		ret[0] = 0;
		ret[m+1] = n;
		printf("sqm: %d\n", sqm);
	}

	for(i=1; i<=sqm; i++){
		search(A, n, B[i*sqm-1], ret + i*sqm, num_threads);

		if(threadIdx.x == 0){
			printf("B[%d*%d-1] = %d; ret[%d*%d] = %d\n", i, sqm, B[i*sqm-1], i, sqm, ret[i*sqm]); 
			dbg_array("ret", ret, "%d ", m+2);
		}
	}
	

	//prob need some sync here
	__syncthreads();

//		3. For 0 <= i <= sqm - 1,
//					let B_i = ( b_isqm+1, ... , b_(i+1)sqm - 1 )
//					let A_i = ( a_j[i]+1, ... , a_j[i+1] )
//			 If j[i] == j[i+1], then
//					set rank(B_i : A_i) = (0, ... , 0)
//			 else
//					recurse compute rank(B_i : A_i)
	if(threadIdx.x == 0){
		printf("ret[%d*%d] = %d; ret[(%d+1)*%d] = %d;\n", 0, sqm, ret[0*sqm], 0, sqm, ret[(0+1)*sqm]);
		printf("ret[%d*%d] = %d; ret[(%d+1)*%d] = %d;\n", 1, sqm, ret[1*sqm], 1, sqm, ret[(1+1)*sqm]);
		dbg_array("ret", ret, "%d ", m+2);
	}
	for(i=0; i<sqm; i++){
		if(i<sqm-1 && ret[i*sqm] == ret[(i+1) * sqm]){
			if(threadIdx.x == 0 && blockIdx.x == 0){
				for(int j=1; j<sqm; j++){
					ret[i*sqm+j] = 0;
				}
			}
		}else{
			number * B_i = B + i * sqm;
			number * A_i = A + ret[i * sqm];
			int n_i = ret[(i+1)*sqm] - ret[i*sqm];
			//printf("ranking %d numbers in %d numbers\n", sqm-1, n_i);
			ranking(A_i, n_i, B_i, ret + i*sqm + 1, sqm-1, num_threads);
		}
	}

//		4. Let 1 <= k <= m be an arbitrary index not multiple of sqm
//			 Let i = floor( k / sqm )
//			 rank(b_k : A) = j[i] + rank(b_k : A_i)
// end
	if(threadIdx.x == 0 && blockIdx.x == 0){
		dbg_array("ret", ret, "%d ", m+2);
		int j, k;
		for(j=1; j<sqm; j++){
			for(k=1; k<m/sqm; k++){
				//int idx = k * sqm + j;
				ret[k * sqm + j] += ret[k * sqm];
				if(threadIdx.x == 0)
					printf("ret[%d] += ret[%d]\n", k*sqm+j, k*sqm);
			}
		}
	}
}
