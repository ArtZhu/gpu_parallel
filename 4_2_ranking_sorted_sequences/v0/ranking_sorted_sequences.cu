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

	cpu_ranking(A, n, B, ret, m);

	printf("\n CPU RANKING : [ ");
	for(int i=0; i<m; i++)
		printf("%d ", ret[i]);
	printf("]\n");

}

__global__ void correction(int * ret, int m);

//
// int sqm = (int) floor(sqrt(m));
//	
// int num_break_points = (m + (sqm - 1)) / sqm;
//	sqm is also num_threads
//  ret must be [0, x1, x2, x3, x4...] but pointing to x1
__global__ void ranking(number * A, int n, number * B, int * ret, int m, int sqm, int num_break_points)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int idx = tid == sqm - 1 ? m : ( tid + 1 ) * sqm;

#ifdef GPU_PRETTY_PRINT
		printf("A length : %d, B length : %d\n", n, m);
		dbg_array("A", A, FMT, n);
		dbg_array("B", B, FMT, m);
#endif

	//1.
	// this search needs to be __device__ function parallel search
	// 		could use the v2 dynamic parallelism
	//							or  v0 1 block version
	// m < 4 should move to thread launching part as well for dynamic parallelism
	if( m == sqm ){
		ret[idx] = cpu_search(A, n, B[idx]) + 1;
		return;
	}

	//2.
	// originally sqm and num_break_points calculations should be here,
	//			but due to thread launching params are related in this dynamic parallelism version
	//		we moved it outside kernel launch.
#ifdef GPU_PRETTY_PRINT
		printf("square root of m: %d; num_break_points: %d\n", sqm, num_break_points);
#endif

	number B_sqm[100000];
	int J[100000];

	//last one
	if(tid == sqm - 1)
		B_sqm[num_break_points-1] = B[m-1];
	else
		B_sqm[tid] = B[ (tid+1)*sqm - 1 ];

	__threadfence();

#ifdef CPU_PRETTY_PRINT
		printf("B_sqm : [ ");
		for(int x=0; x<num_break_points; x++)
			printf("%d ", B_sqm[x]);
		printf("]\n");
#endif

	//rank B_sqm into J
	if(tid == 0)
		J[0] = 0;

	J[tid+1] = cpu_search(A, n, B_sqm[tid]) + 1; 
			// cpu_search return idx, but we want number of elements, so + 1

	__threadfence();

	if(tid == sqm - 1)
		ret[m-1] = J[tid+1];
	else
		ret[tid * sqm - 1] = J[tid+1];

#ifdef CPU_PRETTY_PRINT
		printf("J: [ ");
		for(int x=0; x<num_break_points+1; x++)
			printf("%d ", J[x]);
		printf("]\n");
#endif


	//3. 
	//
	int A_i_len;
	int B_i_len;
	number * A_i;
	int A_jump;
	number * B_i;
	int B_jump;
	number * ret_i;
	//
#ifdef GPU_PRETTY_PRINT
		printf("ret : [ ");
		for(int x=0; x<m; x++)
			printf("%d ", ret[x]);
		printf("]\n");
#endif

//B[ (tid+1)*sqm - 1 ];
		//ptr jumping
		A_jump = J[tid+1];
		A_i = A + A_jump;
		//
		if(tid == sqm-1) 	A_i_len = m - J[tid];
		else 							A_i_len = J[tid+1] - J[tid];

		B_jump = tid * sqm;
		B_i = B + B_jump;
		ret_i = ret + B_jump;

		//last segment (safety)
		if(tid==sqm-1) 
			B_i_len = m - B_jump - 1;
		else
			B_i_len = sqm - 1;

		if(A_i_len == 0){
			for(int j=0; j<B_i_len; j++) {
				//3.
				ret_i[j] = 0;
				//4.
				if(tid != 0) //the first segment does not have a leading rank
					ret_i[j] += *(ret_i-1); 
			}
		}else{
			int sqm_i = B_i_len;
			int num_break_points_i = 0;
			if(B_i_len >= 4){
				sqm_i = (int) floor(sqrt(B_i_len));
				num_break_points_i = (B_i_len + (sqm_i - 1)) / sqm_i;
			}

			unsigned int num_blocks_i = (1023 + sqm_i) / 1024;
			unsigned int threads_per_block_i = sqm_i > 1024 ? 1024 : sqm_i;
	
			ranking<<<num_blocks_i, threads_per_block_i>>>(A_i, A_i_len, B_i, ret_i, B_i_len, sqm_i, num_break_points_i);

#ifdef GPU_PRETTY_PRINT
		printf("raw   : ret_%d : [ ", i);
		for(int x=0; x<B_i_len; x++)
			printf("%d ", ret_i[x]);
		printf("]\n");
#endif

		}
#ifdef GPU_PRETTY_PRINT
		printf("fixed : ret_%d : [ ", i);
		for(int x=0; x<B_i_len; x++)
			printf("%d ", ret_i[x]);
		printf("]\n");
#endif
	//4. 

		//		4. Let 1 <= k <= m be an arbitrary index not multiple of sqm
		//			 Let i = floor( k / sqm )
		//			 rank(b_k : A) = j[i] + rank(b_k : A_i)

		unsigned int num_blocks_correction = (1023 + m) / 1024;
		unsigned int threads_per_block_correction = m > 1024 ? 1024 : m;
		correction<<<num_blocks_correction, threads_per_block_correction>>>(ret, m);
}

__global__ void correction(int * ret, int m){
	int tid = threadIdx.x + blockIdx.x + blockDim.x;

	int base = *( ret - 1 );

	*( ret + tid ) += base;
}
