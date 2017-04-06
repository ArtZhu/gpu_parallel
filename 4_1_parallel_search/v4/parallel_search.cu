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
int * l;
// l	must be allocated to num_blocks size
int * r;
// r  must be allocated to num_blocks size
volatile int * dev_ret;
// dev_ret  must be allocated to num_blocks size
// n is the number of elements
__device__ void search(number * X, int n, number target, int * c, int * q, int num_threads, volatile int * dev_ret, int * l, int * r){
	int tid = threadIdx.x;

	X += n * blockIdx.x;
	l += blockIdx.x;
	r += blockIdx.x;
	dev_ret += blockIdx.x;
	c += blockIdx.x * (blockDim.x + 2);
	q += blockIdx.x * (blockDim.x + 2);

	tid += 1; // so that idx starts from 1

	if(tid > n) return; // safety

	//1.
  // initialize this part outside kernel
	if(tid == 1){
		*l = 0;
		*r = n + 1;
		c[0] = 0;
		c[num_threads + 1] = 1;

		*dev_ret = -2; // for thread termination purpose
	}

#ifdef PRETTY_PRINT
	if(tid == 1)
		printf("%d : %d %d\n", blockIdx.x, *l, *r);
#endif

	//sync
	__syncthreads();

	//2.

	int count = 0;
	while(*r - *l > num_threads){

		if(tid == 1){
			q[0] = *l;
			q[num_threads + 1] = *r;
		}

		q[tid] = *l + tid * ((*r - *l) / (num_threads + 1));

		//sync -- use r, l, p;
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

		//sync -- use X, q, target
		//     -- set l, r, c
		__syncthreads();
		// if ret has been set, return, a replacement for the "return" in the conditional statement;
		if(*dev_ret >= -1){
#ifdef PRETTY_PRINT
		if(tid == 1)
			printf("%d : dev_ret0 %d\n", blockIdx.x, *dev_ret);
#endif
			return;
		}


		if(c[tid] < c[tid + 1]){
			*l = q[tid];
			*r = q[tid + 1];
		}


		if(tid == 1 && c[0] < c[1]){
			*l = q[0];
			*r = q[1];
		}

		//sync -- use q, c, tid
		//		 -- set l, r
		__syncthreads();


#ifdef PRETTY_PRINT
		if(tid == 1)
			printf("iter %d, block %d : %d %d\n", count++, blockIdx.x, *l, *r);
#endif

	}

#ifdef PRETTY_PRINT
		if(tid == 1)
			printf("%d : dev_ret1 %d\n", blockIdx.x, *dev_ret);
#endif
	if(tid > *r - *l) return;

	if(target == X[*l+tid]){
		*dev_ret = *l + tid - 1; // so that ret idx starts from 0
	}
	else if(target > X[*l+tid]){
		c[tid] = 0;
	}
	else{
		c[tid] = 1;
	}

#ifdef PRETTY_PRINT
		if(tid == 1)
			printf("%d : dev_ret2 %d\n", blockIdx.x, *dev_ret);
#endif
	if(*dev_ret >= -1)
		return;

	if(c[tid-1] < c[tid])
		*dev_ret = *l + tid - 1 - 1; // so that ret idx starts from 0

#ifdef PRETTY_PRINT
		if(tid == 1)
			printf("%d : dev_ret3 %d\n", blockIdx.x, *dev_ret);
#endif

}

__device__ void fix(volatile int * dev_ret, int dev_ret_len, int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	//while(dev_ret[tid] == -2) ;

	if(tid < dev_ret_len){
		int idx = dev_ret[tid];
		if(idx != -2 && idx != -1){
			dev_ret[0] = idx + n * tid;
		}
	}
}

__global__ void search_main(number * X, int n, number target, int * c, int * q, int num_threads, volatile int * dev_ret, int * l, int * r, int dev_ret_len)
{
	// doesn't work for non-pow 2
	int tmp_n = n / dev_ret_len;

	/*
	if(threadIdx.x == 0){
		printf("array length : %d\n", n);
		printf("tmp_n : %d\n dev_ret_len = %d\n", tmp_n, dev_ret_len);
	}
	*/

	num_threads = num_threads > 1024 ? 1024 : num_threads;

	search(X, tmp_n, target, c, q, num_threads, dev_ret, l, r);

	/*
	if(blockIdx.x == 0 && threadIdx.x == 0){
		printf("[ ");
		for(int i=0; i<dev_ret_len; i++){
			printf("%d ", dev_ret[i]);
		}
		printf("]\n");
	}
	*/

	
	fix(dev_ret, dev_ret_len, tmp_n);

	__threadfence();

	if(threadIdx.x + blockIdx.x * blockDim.x == 0)
	printf("gpu found : %d\n", *dev_ret);

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

	unsigned int num_blocks = (1023 + num_threads) / 1024;
	unsigned int threads_per_block = num_threads > 1024 ? 1024 : num_threads;

	c_size = (2 * num_blocks + num_threads) * sizeof(int);
	q_size = (2 * num_blocks + num_threads) * sizeof(int);

	// X_len + 2 for the algorithm element at idx 0 and n + 1 (originally 1, 2, ..., n)
	err_code[0] = cudaMalloc( &dev_X , X_size );
	err_code[1] = cudaMalloc( &c , c_size );
	err_code[2] = cudaMalloc( &q , q_size );
	err_code[3] = cudaMalloc( &dev_ret , sizeof(volatile int) * num_blocks);
	err_code[4] = cudaMalloc( &l , sizeof(int) * num_blocks );
	err_code[5] = cudaMalloc( &r , sizeof(int) * num_blocks );
	for(int i=0; i<6; i++){ gerror(err_code[i]); }

	gerror(cudaMemcpy(dev_X, host_X, X_size, cudaMemcpyHostToDevice));

	ret_idx = 10086;

	printf("launching %u blocks, %u threads per block.\n", num_blocks, threads_per_block);

	d->Dg = {num_blocks, 1, 1};
	d->Db = {threads_per_block, 1, 1};
	gstart();
	search_main<<<d->Dg, d->Db>>>(dev_X, X_len, target, c, q, num_threads, dev_ret, l, r, num_blocks);
	gend(&gputime);
	printf("gputime : %f ms\n", gputime);
	gerror(cudaGetLastError());
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
	gerror(cudaFree(c));
	gerror(cudaFree(q));
	gerror(cudaFree(dev_ret));
	gerror(cudaFree(l));
	gerror(cudaFree(r));
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

	_init_array(fname[0] != 0);
	
	prep_kernel();
}

void _init_array(int with_file)
{
	host_X = (number *) malloc(X_size);

	host_X[0] = INT_MIN;
	host_X[X_len+1] = INT_MAX;
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
