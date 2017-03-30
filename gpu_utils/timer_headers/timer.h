/* timer.h
 * this files contains 
 *				timer header functions 
 * 	compile with -std=c++11 flag
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
#ifndef TIMER_P_H
#define TIMER_P_H

typedef struct exec_dim exec_dim_t;
//exec_dim_t * d = &params;

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

		/* function implementation */

#ifndef TIMER_H
#define TIMER_H

/*
 *	Ns is of type size_t 
 *				specifies the number of bytes in shared memory 
 *							that is dynamically allocated per block for this call 
 *								in addition to the statically allocated memory; 
 *		this dynamically allocated memory is used by 
 *			any of the variables declared as an external array 
 *				as mentioned in __shared__; 
 *	Ns is an optional argument which defaults to 0;
 *
 *-----------------------------------------------------------------------
 *
 *	S is of type cudaStream_t 
 *				specifies the associated stream; 
 *	S is an optional argument which defaults to 0.
 *
 */
struct exec_dim{
	dim3 Dg;
	dim3 Db;
	size_t Ns;
	cudaStream_t S;
};

			/* global var */
			static dim3 one(1, 1, 1);
			static exec_dim_t params = {one, one, 0, 0};

/* default exec_dim */
exec_dim_t * d = &params;

//------------------------------------------------------------------------
//	CPU
//------------------------------------------------------------------------

// with return value
template <typename Ret_t, typename... Arguments>
void ctime(float * cputime, Ret_t * ret, Ret_t (*fn)(Arguments...), Arguments... args)
{
	cstart();
	*ret = (*fn)(args...);
	cend(cputime);
}

// without return value
template <typename Ret_t, typename... Arguments>
void ctime(float * cputime, Ret_t (*fn)(Arguments...), Arguments... args)
{
	cstart();
	(*fn)(args...);
	cend(cputime);
}


//------------------------------------------------------------------------
//	GPU
//------------------------------------------------------------------------
//gtime cuda functions WITH return value
template <typename... Arguments>
void gtime(float * gputime, cudaError_t * ret, cudaError_t (*cuda)(Arguments...), Arguments... args)
{
	gstart();
	*ret = cuda(args...);
	gend(gputime);
}

//gtime cuda functions WITHOUT return value
template <typename... Arguments>
void gtime(float * gputime, cudaError_t (*cuda)(Arguments...), Arguments... args)
{
	gstart();
	cuda(args...);
	gend(gputime);
}

//gtime kernel
template <typename... Arguments>
void gtime(float * gputime, void (*kernel)(Arguments...), exec_dim_t * ed, Arguments... args)
{
	gstart();
	kernel <<< ed->Dg , ed->Db, ed->Ns, ed->S >>> (args...);
	gend(gputime);
}

#endif
