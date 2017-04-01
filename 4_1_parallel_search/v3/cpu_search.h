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
	
	idx = n/2;
	s = n/4;

	while(s >= 1){
		if(X[idx] == target){
			return idx;
		}else if(X[idx] > target){
			idx -= s;	
		}else{
			idx += s;
		}

		s = s/2;
	}

	return idx;
}

#endif
