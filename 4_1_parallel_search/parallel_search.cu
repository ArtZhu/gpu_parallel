/* file : parallel_search.cu
 * author : Tiane Zhu
 * date : Mar 23, 2017
 *
 * this program is an implementation of the parallel search algorithm
 * 	ALGORITHM 4.1 in 
 * "An Introduction to Parallel Algorithms" - by Joseph Jaja
 *		p146 - ISBN 9-789201-548563
 */

#include "parallel_search.h"

///////////////////////////////////////////////////////////
// Input to the algorithm //
// X 										-- strictly ordered array
// y (target) 					-- target
// p (num_threads) 			-- num_processor
// j (tid) 							-- processor idx
///////////////////////////////////////////////////////////
// Output 
// i (ret) 							-- X[i] <= y < x[i+1]
//		[ i is initialized to -1 , since it has only non-neg values
//			i non-neg => i set ]
///////////////////////////////////////////////////////////
/* kernel strictly following algorithm */
// additional inputs
int * c;
// c										-- c array from 0 to p+1
int * q;
// q										-- q array from 0 to p+1
__device__ int l;
// l										
__device__ int r;
// r
__device__ int iter = 0;
// n is the number of elements
__global__ void search(number * X, int n, number target, int * c, int * q, int num_threads, int * dev_ret){

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	tid += 1; // so that idx starts from 1

#ifdef PRETTY_PRINT
	if(tid == 1)
		printf("KERNEL : \narray length : %d, target : %d, num_threads : %d\n", n, target, num_threads);
#endif

	//if(tid > n) return; // safety

	//1.
	if(tid == 1 && iter == 0){
		l = 0;
		r = n + 1;
		X[0] = INT_MIN;
		X[n + 1] = INT_MAX;
		c[0] = 0;
		c[num_threads + 1] = 1;

		*dev_ret = -1; // for thread termination purpose
	}

	if(tid == 1)
		iter++;

#ifdef PRETTY_PRINT
	if(tid == 1){
		if(n <= 32){
			for(int i=0; i<n+2; i++) printf("%d ", X[i]); 
		}
		//printf("\n");
		//printf("| q0 | q1 | q2 | q3 | c0 | c1 | c2 | c3 | l  | r  |\n");
		printf("1. r : %d ; l : %d\n", r, l);
	}
#endif

	//sync -- 
	//		 -- set r, l, X, c, dev_ret, tid
	__syncthreads();

	//2.

	// if statement below as replacement
	//while(r - l > num_threads){
	int set_ret = (r - l <= num_threads);

	if(tid == 1){
		q[0] = l;
		q[num_threads + 1] = r;
	}

	q[tid] = l + tid * ((r - l) / (num_threads + 1));

	//sync -- use r, l, p, tid, num_threads;
	//		 -- set q
	__syncthreads();

	if(target == X[q[tid]]){
		*dev_ret = q[tid] - 1; // so that ret idx starts from 0
		// can i return here???
		// no
		//return;
	}
	else{
		if(target > X[q[tid]])
			c[tid] = 0;
		else 
			c[tid] = 1;
	}

	//sync -- use X, q, target, tid
	//     -- set dev_ret, c
	__syncthreads();

	// if ret has been set, return, a replacement for the "return" in the conditional statement;
	if(*dev_ret >= 0){
#ifdef PRETTY_PRINT
		if(tid == 1)
			printf("dev ret0 : %d\n", *dev_ret);
#endif
		return;
	}


	if(c[tid] < c[tid + 1]){
		l = q[tid];
		r = q[tid + 1];
		printf("tid : %d setting r, l to be %d %d\n", tid, r, l);
	}

	//sync -- use dev_ret, q, c, tid
	//		 -- set l, r
	__syncthreads();


	if(tid == 1 && c[0] < c[1]){
		l = q[0];
		r = q[1];
	}

	//sync -- use dev_ret, q, c, tid
	//		 -- set l, r
	__syncthreads();


#ifdef PRETTY_PRINT
	if(tid == 1){
		printf("r : %d ; l : %d\n", r, l);
		printf("c[%d] = %d, c[%d] = %d, c[%d] = %d\n", 1023, c[1023], 1024, c[1024], 1025, c[1025]);
		//printf("|%4d|%4d|%4d|%4d|%4d|%4d|%4d|%4d|%4d|%4d|\n", q[0], q[1], q[2], q[3], c[0], c[1], c[2], c[3], l, r);
	}
#endif

	//} //while(r - l > num_threads){
		if(set_ret){

			if(tid > r - l){ 
				//corresponds with the next syncthreads();
				__syncthreads();
				return;
			}

			if(target == X[l+tid]){
				*dev_ret = l + tid - 1; // so that ret idx starts from 0
			}
			else if(target > X[l+tid]){
				c[tid] = 0;
			}
			else{
				c[tid] = 1;
			}

			// sync -- use l, X, tid, target
			//			-- set dev_ret, c
			__syncthreads();

#ifdef PRETTY_PRINT
			printf("dev ret1 : %d\n", *dev_ret);
#endif
			if(*dev_ret >= 0)
				return;

			if(c[tid-1] < c[tid])
				*dev_ret = l + tid - 1 - 1; // so that ret idx starts from 0
#ifdef PRETTY_PRINT
			printf("dev ret2 : %d\n", *dev_ret);
#endif
		}
}

// main
int main(int argc, char * argv[]) 
{
	setbuf(stdout, NULL);
	_init(argc, argv);

	if(verbose)
		printf("finding target : %d in array of length %d\n", target, X_len);

	cudaError_t err_code[10];
	float gputime, cputime;
	int ret_idx, * dev_ret;
	
	cudaSetDevice(0);
	cudaDeviceReset();

	// X_len + 2 for the algorithm element at idx 0 and n + 1 (originally 1, 2, ..., n)
	err_code[0] = cudaMalloc( &dev_X , X_size );
	err_code[1] = cudaMalloc( &c , c_size );
	err_code[2] = cudaMalloc( &q , q_size );
	err_code[3] = cudaMalloc( &dev_ret , sizeof(int) );
	for(int i=0; i<4; i++){ gerror(err_code[i]); }

	gerror(cudaMemcpy(dev_X, host_X, X_size, cudaMemcpyHostToDevice));

	unsigned int num_blocks = (1023 + num_threads) / 1024;
	unsigned int threads_per_block = num_threads > 1024 ? 1024 : num_threads;

	ret_idx = 10086;

	printf("launching %u blocks, %u threads per block.\n", num_blocks, threads_per_block);

	d->Dg = {num_blocks, 1, 1};
	d->Db = {threads_per_block, 1, 1};
	gstart();
	do{
		search<<<d->Dg, d->Db>>>(dev_X, X_len, target, c, q, num_threads, dev_ret);
		gerror(cudaMemcpy(&ret_idx, dev_ret, sizeof(int), cudaMemcpyDeviceToHost));
	}while(ret_idx < 0);
	gend(&gputime);
	printf("gputime : %f ms\n", gputime);
	gerror(cudaGetLastError());
	gerror( cudaDeviceSynchronize() );

	//gerror(cudaMemcpy(&ret_idx, dev_ret, sizeof(int), cudaMemcpyDeviceToHost));
	printf("device idx = %d;\n", ret_idx);

	ret_idx = 10086;

	cstart();
	ret_idx = cpu_search(host_X + 1, X_len, target);
	cend(&cputime);
	printf("cputime : %f ms\n", cputime);
	printf("host idx = %d;\n", ret_idx);

	gerror(cudaFree(dev_X));
	gerror(cudaFree(c));
	gerror(cudaFree(q));
	gerror(cudaFree(dev_ret));
	free(host_X);
}// main

char fname[80];
void _init(int argc, char ** argv)
{ 
	X_len = DEFAULT_ARRAY_LEN;
	num_threads = DEFAULT_NUM_THREADS;
	target = DEFAULT_TARGET;
	fname[0] = 0;

	int len_spec = 0;

	for(int i=1; i<argc; i++){
		switch(*argv[i]){
			case '-':
				switch(argv[i][1]){
					case 'v': 
						verbose = 1;
						break;
					case 'f':
						if(!len_spec){
							strcpy(fname, argv[++i]);
							len_spec = 1;
						}
						break;
					case 't':
						sscanf(argv[++i], "%d", &num_threads);
						break;
					case 'l':
						if(!len_spec){
							sscanf(argv[++i], "%d", &X_len);
							len_spec = 1;
						}
						break;
				}
				break;
			default:
				sscanf(argv[i], FMT, &target);
		}
	}

	X_size = (X_len + 2) * sizeof(number);
	c_size = (num_threads + 2) * sizeof(int);
	q_size = (num_threads + 2) * sizeof(int);

	_init_array(fname[0] != 0);
	
	prep_kernel();
}

void _init_array(int with_file)
{
	host_X = (number *) malloc(X_size);

	//not use file
	if(!with_file){
		for(number i=1; i<X_len+1; i++){
			host_X[i] = 2 * i;
		}
		return;
	}
	
	//use file
	FILE * fp;
	printf("array file : \"%s\"", fname);

	if(!(fp = fopen(fname, "r"))){
		printf(" does not exist.\n");
		exit(1);
	}

	if(fscanf(fp, "%d", &X_len) < 1){
		printf(" stats broken.\n");
		exit(1);
	}

	printf("\n");

	for(int i=0; i<X_len; i++){
		if(fscanf(fp, FMT, host_X + i) != 1){
			printf(" missing the %dth number.\n", i);
			exit(1);
		}
		if(verbose)
			printf(FMT, host_X[i]);
	}
	if(verbose) printf("\n");

}
