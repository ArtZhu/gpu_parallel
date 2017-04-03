/* file : cpu_search.h
 * author : Tiane Zhu
 * date : March 24
 *
 *	This file contains a fast cpu_search to compare with our gpu search
 */

#ifndef CPU_SEARCH_H
#define CPU_SEARCH_H

#include "parallel_search.h"

int cpu_search(number * X, int n, number target)
{
	int idx, s;
	
	idx = (n - 1)/2;
	s = n/4;

	while(s >= 1){
		if(X[idx] == target){
			return idx;
		}else if(X[idx] > target){
			idx -= s;	
		}else{
			idx += s;
		}

		//printf("idx = %d; X[idx] = %d; %d %d\n", idx, X[idx], X[idx-s], X[idx+s]);

		s = s/2;
	}

	//for n not perfect power of 2
	while(X[idx] > target){
		idx--;
	}
	while(X[idx+1] <= target){
		idx++;
	}

	return idx;
}

#endif
