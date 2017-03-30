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

static __global__ void hello(){int x = 1; x++;}
inline void prep_kernel()
{
	hello<<<1, 1>>>();
}


//------------------------------------------------------------------------
//	Timer
//------------------------------------------------------------------------
#include "timer_headers/timerc.h"

#endif
