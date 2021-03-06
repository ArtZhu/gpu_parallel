/* this header file contains timer functions
 *
 * 		for gpu kernel
 * 		for cpu functions
 *
 *		for error
 * 
 * compile with -std=c++11 flag
 */

#ifndef UTIL_H
#define UTIL_H

#include "timer_headers/timer.h"
#include "utilc.h"

template <typename T>
__host__ __device__
void dbg_array(const char * tag, T * ptr, const char * FMT, int len);

		/* function prototype */
		/* UTIL */
#ifndef UTILC_P_H
#define UTILC_P_H

#define HD cudaMemcpyHostToDevice
#define DH cudaMemcpyDeviceToHost

#define gerror(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void prep_kernel();

#endif
		/* TIMER */
#ifndef TIMERC_P_H
#define TIMERC_P_H

#include <stdio.h>

inline void cstart();
inline void cend(float * cputime);
inline void gstart();
inline void gend(float * gputime);

#endif
#ifndef TIMER_P_H
#define TIMER_P_H

#include <stdio.h>
#include "timerc.h"
using namespace std;

//------------------------------------------------------------------------
//	CPU
//------------------------------------------------------------------------
// WITH ret
template <typename Ret_t, typename... Arguments>
void ctime(float * cputime, Ret_t * ret, Ret_t (*fn)(Arguments...), Arguments... args);
// WITHOUT ret
template <typename Ret_t, typename... Arguments>
void ctime(float * cputime, Ret_t (*fn)(Arguments...), Arguments... args);

//------------------------------------------------------------------------
//	GPU
//------------------------------------------------------------------------
// cuda WITH ret
template <typename... Arguments>
void gtime(float * gputime, cudaError_t * ret, cudaError_t (*cuda)(Arguments...), Arguments... args);
// cuda WITHOUT ret
template <typename... Arguments>
void gtime(float * gputime, cudaError_t (*cuda)(Arguments...), Arguments... args);
// kernel
template <typename... Arguments>
void gtime(float * gputime, void (*kernel)(Arguments...), exec_dim_t * d, Arguments... args);





#endif

		/* function imp */
// see included files

//------------------------------------------------------------------------
//	Printer
//------------------------------------------------------------------------
template <typename T>
__host__ __device__
inline void dbg_array(const char * tag, T * ptr, const char * FMT, int len){
	printf("%s : [ ", tag);
	for(int iiiii=0; iiiii<len; iiiii++){
		printf(FMT , ptr[iiiii]);
	}
	printf("]\n");
}

#endif
