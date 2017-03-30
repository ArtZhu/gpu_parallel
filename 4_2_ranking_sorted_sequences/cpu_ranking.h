/* file : cpu_ranking.h
 * author : Tiane Zhu
 * date : March 28
 *
 *	This file contains a fast linear ranking algorithm
 *				to compare with our gpu parallel ranking
 */

#ifndef CPU_RANKING_H
#define CPU_RANKING_H

//#include "ranking_sorted_sequences.h"
#include "cpu_search.h"
#include <math.h>

// rank(B : A)
// ret must be allocated already
void cpu_ranking(number * A, int n, number * B, number * ret, int m){

#ifdef CPU_PRETTY_PRINT
		printf("A length : %d, B length : %d\n", n, m);
		dbg_array("A", A, FMT, n);
		dbg_array("B", B, FMT, m);
#endif

	//1.
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

#endif
