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

__global__ void ranking(number * A, int n, number * B, int * ret, int m)
{

#ifdef CPU_PRETTY_PRINT
		printf("A length : %d, B length : %d\n", n, m);
		dbg_array("A", A, FMT, n);
		dbg_array("B", B, FMT, m);
#endif

	//1.
	// this search needs to be __device__ function parallel search
	if( m < 4 ){
		for(int i=0; i<m; i++)
			ret[i] = cpu_search(A, n, B[i]) + 1;
		return;
	}

	//2.
	int sqm = (int) floor(sqrt(m));
	
	int num_break_points = (m + (sqm - 1)) / sqm;

#ifdef CPU_PRETTY_PRINT
		printf("square root of m: %d; num_break_points: %d\n", sqm, num_break_points);
#endif

	number B_sqm[num_break_points];
	int J[num_break_points + 1];

	for( int i=0; i<num_break_points - 1; i++ )
		B_sqm[i] = B[ (i+1)*sqm - 1 ];
	//last one
	B_sqm[num_break_points-1] = B[m-1];

#ifdef CPU_PRETTY_PRINT
		printf("B_sqm : [ ");
		for(int x=0; x<num_break_points; x++)
			printf("%d ", B_sqm[x]);
		printf("]\n");
#endif

	//rank B_sqm into J
	J[0] = 0;
	for(int i=1; i<=num_break_points; i++){
		J[i] = cpu_search(A, n, B_sqm[i-1]) + 1; // cpu_search return idx, but we want number of elements, so + 1
		ret[i * sqm - 1] = J[i];
	}
#ifdef CPU_PRETTY_PRINT
		printf("J: [ ");
		for(int x=0; x<num_break_points+1; x++)
			printf("%d ", J[x]);
		printf("]\n");
#endif

	//3. 
	//
	int A_i_len;
	int B_i_len = sqm - 1;
	number * A_i;
	int A_jump;
	number * B_i;
	int B_jump;
	number * ret_i;
	//
#ifdef CPU_PRETTY_PRINT
		printf("ret : [ ");
		for(int x=0; x<m; x++)
			printf("%d ", ret[x]);
		printf("]\n");
#endif

	for(int i=0; i<sqm; i++){
		//ptr jumping
		A_jump = J[i];
		A_i = A + A_jump;
		//
		if(i == sqm-1) 	A_i_len = m - J[i];
		else 						A_i_len = J[i+1] - J[i];

		B_jump = i*sqm;
		B_i = B + B_jump;
		ret_i = ret + B_jump;

		//last segment (safety)
		if(i==sqm-1) 
			B_i_len = m - B_jump - 1;

		if(A_i_len == 0){
			for(int j=0; j<B_i_len; j++) {
				//3.
				ret_i[j] = 0;
				//4.
				if(i != 0) //the first segment does not have a leading rank
					ret_i[j] += *(ret_i-1); 
			}
		}else{
			cpu_ranking(A_i, A_i_len, B_i, ret_i, B_i_len);

#ifdef CPU_PRETTY_PRINT
		printf("raw   : ret_%d : [ ", i);
		for(int x=0; x<B_i_len; x++)
			printf("%d ", ret_i[x]);
		printf("]\n");
#endif

			for(int j=0; j<B_i_len; j++){
				//4.
				if(i != 0) //the first segment does not have a leading rank
					ret_i[j] += *(ret_i-1); 
			}
		}
#ifdef CPU_PRETTY_PRINT
		printf("fixed : ret_%d : [ ", i);
		for(int x=0; x<B_i_len; x++)
			printf("%d ", ret_i[x]);
		printf("]\n");
#endif
	}

}
