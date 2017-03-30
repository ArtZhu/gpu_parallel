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

	//1.
	if( m < 4 ){
		for(int i=0; i<m; i++)
			ret[i] = cpu_search(A, n, B[i]);
		return ret;
	}

	//2.
	int sqm = (int) floor(sqrt(m));
	
	int num_break_points = (m + (sqm - 1)) / sqm;

	number B_sqm[num_break_points];
	int * J[num_break_points + 1];

	for( int i=0; i<num_break_points - 1; i++ )
		B_sqm[i] = B[ (i+1)*sqm - 1 ];
	//last one
	B_sqm[num_break_points-1] = B[m];

	//rank B_sqm into J
	J[0] = 0;
	for(int i=1; i<=num_break_points; i++)
		J[i] = cpu_search(A, n, B_sqm[i-1]);
	
	//3. 
	//
	int A_i_len;
	int B_i_len = sqm - 2;
	number B_i[B_i_len];
	number ret[B_i_len];
	//
	for(int i=0; i<sqm; i++){
		for(int j=0; j<seg_i_len; j++){
			B_i[j] = B[ i*sqm+j+1 ];

			A_i_len = J[i+1] - J[i];
			if(A_i_len == 0){
				for(int x=0; x<B_i_len; x++) ret[i] = 0;
			}else{
				number A_i[A_i_len];

				cpu_ranking(A_i, A_i_len, B_i, ret, B_i_len);
			}

		//idx jumping
		}
	}

	//4.

}

#endif
