/* utilc.h
 * this files contains 
 *				util header functions 
 * 	that can be compiled with nvcc without -std=c++11 flag
 */

		/* function prototype */

#ifndef TIMERC_P_H
#define TIMERC_P_H

#include <stdio.h>

inline void cstart();
inline void cend(float * cputime);
inline void gstart();
inline void gend(float * gputime);

#endif
//
#ifndef UTILC_P_H
#define UTILC_P_H

#define HD cudaMemcpyHostToDevice
#define DH cudaMemcpyDeviceToHost

#define gerror(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void prep_kernel();

#endif

 /* function implementation */

#ifndef UTILC_H
#define UTILC_H

//------------------------------------------------------------------------
//	GPU Error
//------------------------------------------------------------------------
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

//------------------------------------------------------------------------
//	GPU warmup
//------------------------------------------------------------------------

static __global__ void hello(){ 
	if(threadIdx.x == 0 && blockIdx.x == 0) printf("kernel warmup ...\n"); 
	int x=0; 
	for (int j=0; j<100; j++) x+=j;
}
inline void prep_kernel(){ hello<<<10, 10>>>(); }

//------------------------------------------------------------------------
//	Timer
//------------------------------------------------------------------------
#include "timer_headers/timerc.h"

#endif
