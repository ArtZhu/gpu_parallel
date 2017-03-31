/* file : search.h
 * author : Tiane Zhu
 * date : March 30, 2017
 *
 *	header file for MAIN program
 */

#ifndef SEARCH_H
#define SEARCH_H

#include <stdio.h>
#include "../../gpu_utils/util.h"

int verbose = 0;

#define DEFAULT_ARRAY_LEN 15
#define DEFAULT_TARGET 19
#define DEFAULT_NUM_THREADS 2

#include "parallel_search.h"
#include "cpu_search.h"

// CPU
number target;
unsigned int num_threads;

	unsigned int X_size;
	unsigned int c_size;
	unsigned int q_size;

number * host_X;
int X_len;


// INIT
void _init(int argc, char ** argv);
void _init_array(int with_file);


#endif
