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
	cudaMalloc(&dev_ret, m * sizeof(number));

	cudaMemcpy(dev_A, A, n * sizeof(number), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B, m * sizeof(number), cudaMemcpyHostToDevice);

	int num_threads = 1024;

	ranking<<<1, 1024>>>(dev_A, n, dev_B, dev_ret, m, num_threads);

	cudaDeviceSynchronize();

	cudaMemcpy(ret, dev_ret, m * sizeof(number), cudaMemcpyDeviceToHost);
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

__global__ void ranking(number * A, int n, number * B, int * ret, int m, int num_threads)
{
	int i, dev_ret;
	number target;

// 		1. If m < 4, then rank the elements of B 
//			 							by applying Alg 4_1 with p = n
//								 then exit
	if(m < 4){
		for(i = 0; i<m; i++){

			target = B[i];
			search(A, n, target, num_threads, &dev_ret);
			ret[i] = *((int *) &search_rank);
			if(threadIdx.x == 0){
				printf("%d\n", dev_ret);
			}

			__syncthreads();
		}
		return;
	}

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
}
