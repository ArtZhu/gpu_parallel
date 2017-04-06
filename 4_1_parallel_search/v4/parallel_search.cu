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
volatile int * c;
// c										-- c array from 0 to p+1
volatile int * q;
// q										-- q array from 0 to p+1
volatile __device__ int l;
// l										
volatile __device__ int r;
// r
// n is the number of elements
// this function needs iter_flag to be initialized to -1
__global__ void search(number * X, int n, number target, volatile int * c, volatile int * q, int num_threads, int * dev_ret){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	// ATOMIC
	// for comparison and conditionally setting global iter flag.
	int local_iter = 0;
	int * mutex = half_iter_signals + tid;

	tid += 1; // so that idx starts from 1

	if(tid > n) return; // safety, this needs to be fixed, tid+1 part

	//init signals
	//atomicExch(mutex, 0);
	*mutex = 0;

	//1.
	// come back and add the first atomic flag here.
	if(tid == 1){
		l = 0;
		r = n + 1;
		X[0] = INT_MIN;
		X[n + 1] = INT_MAX;
		c[0] = 0;
		c[num_threads + 1] = 1;


		*dev_ret = -1; // for thread termination purpose
		//init iter_flag
#ifdef PRETTY_PRINT
		printf("INITIAL: r: %d; l: %d\n", r, l);
#endif

#ifdef PRETTY_PRINT
		printf("num threads: %d\n", num_threads);
#endif

		__threadfence();

		atomicExch(&iter_flag, 0);
	}


	//2.
	while(atomicCAS(&iter_flag, 0, 0) != 0); // second arg doesn't matter here

#ifdef PRETTY_PRINT
	if(*mutex != 0) printf("%d mutex not 0\n", tid);
#endif


	while(r - l > num_threads){

		if(tid == 1){
			q[0] = l;
			q[num_threads + 1] = r;
		}


		int qtid = l + tid * ((r - l) / (num_threads + 1));
		q[tid] = qtid;

		if(target == X[qtid]){
			*dev_ret = qtid - 1; // so that ret idx starts from 0
		}
		else{
			if(target > X[qtid])
				c[tid] = 0;
			else 
				c[tid] = 1;
		}

		//mark
		__threadfence();
		//guarantees, tid-1 read the value already.

		if(tid != 1)
			while(atomicCAS(mutex, 0, 1) != 0);

		//guarantees, tid+1 set the value already
		if(tid != num_threads)
			while(atomicCAS(mutex + 1, 1, 0) != 1);


		int iter_inc = 0;
		// whoever sets l,r  should let other threads know that 
		//			next iteration is ready.
		if(c[tid] < c[tid + 1]){
			l = q[tid];
			r = q[tid + 1];

			__threadfence();

			//atomicCAS(&iter_flag, local_iter, local_iter+1);
			iter_inc = 1;
		}

		if(tid == 1 && c[0] < c[1]){
			l = q[0];
			r = q[1];

			__threadfence();

			//atomicCAS(&iter_flag, local_iter, local_iter+1);
			iter_inc = 1;
		}

		atomicCAS(&iter_flag, local_iter, local_iter+iter_inc);

		++local_iter;

		//*iter_flag ok here?
		while(atomicCAS(&iter_flag, -1, local_iter) != local_iter); // second arg doesn't matter here

		if(*dev_ret >= 0)
			return;
		
	}



	if(tid <= r - l){
		if(target == X[l+tid]){
			//*dev_ret = l + tid - 1; // so that ret idx starts from 0
			atomicCAS(dev_ret, -1, l + tid - 1);
		}
		else if(target > X[l+tid]){
			c[tid] = 0;
		}
		else{
			c[tid] = 1;
		}
	}

	//mark
	__threadfence();
	//set flag.
	atomicExch(mutex, 1);

	//guarantees, tid-1 set the flag already
	if(tid != 1)
		while(atomicCAS(mutex - 1, 1, 0) != 1);

#ifdef PRETTY_PRINT
	if(tid == 1)
		printf("dev ret1 : %d\n", *dev_ret);
#endif

	if(*dev_ret >= 0)
		return;

	if(tid <= r - l){

		// problematic part
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

	float gputime, cputime;
	int ret_idx, * dev_ret;
	
	cudaSetDevice(0);
	cudaDeviceReset();

	// X_len + 2 for the algorithm element at idx 0 and n + 1 (originally 1, 2, ..., n)
	gerror(cudaMalloc( &dev_X , X_size ));
	gerror(cudaMalloc( &c , c_size ));
	gerror(cudaMalloc( &q , q_size ));
	gerror(cudaMalloc( &dev_ret , sizeof(int) ));
	gerror(cudaMalloc( &host_half_iter_signals_ptr, num_threads * sizeof(int)));

	gerror(cudaMemcpyToSymbol(half_iter_signals, &host_half_iter_signals_ptr, sizeof(int *), 0, cudaMemcpyHostToDevice));

	//use it as a tmp var
	ret_idx = -1;

	gerror(cudaMemcpyToSymbol(iter_flag, &ret_idx, sizeof(int), 0, cudaMemcpyHostToDevice));

	gerror(cudaMemcpy(dev_X, host_X, X_size, cudaMemcpyHostToDevice));

	unsigned int num_blocks = (1023 + num_threads) / 1024;
	unsigned int threads_per_block = num_threads > 1024 ? 1024 : num_threads;

	ret_idx = 10086;

	printf("launching %u blocks, %u threads per block.\n", num_blocks, threads_per_block);

	d->Dg = {num_blocks, 1, 1};
	d->Db = {threads_per_block, 1, 1};
	gstart();
	search<<<d->Dg, d->Db>>>(dev_X, X_len, target, c, q, num_threads, dev_ret);
	gend(&gputime);
	printf("gputime : %f ms\n", gputime);
	gerror( cudaGetLastError());
	gerror( cudaDeviceSynchronize() );

	gerror(cudaMemcpy(&ret_idx, dev_ret, sizeof(int), cudaMemcpyDeviceToHost));
	printf("device idx = %d;\n", ret_idx);

	ret_idx = 10086;

	cstart();
	ret_idx = cpu_search(host_X + 1, X_len, target);
	cend(&cputime);
	printf("cputime : %f ms\n", cputime);
	printf("host idx = %d;\n", ret_idx);

	gerror(cudaFree(dev_X));
	gerror(cudaFree((void *) c));
	gerror(cudaFree((void *) q));
	gerror(cudaFree(host_half_iter_signals_ptr));
	gerror(cudaFree(dev_ret));
	free(host_X);
}

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
