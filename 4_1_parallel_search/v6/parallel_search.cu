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

typedef unsigned long long int ull;
__global__ void search_main(number * X, int n, number target, int num_threads, ull * dev_ret)
{
	int l, r, *ptr;
	ull record, *ptr_u;

	record = atomicAdd(dev_ret, 0);
	ptr_u = &record;
	ptr = (int *) ptr_u;
	l = *ptr;
	r = *(ptr+1);

	int block_n, start, s;
	while(r - l <= 1){

		block_n = (r - l) / gridDim.x;
		
		s = block_n / blockDim.x;

		start = l + (blockIdx.x * block_n);

		idx = start + threadIdx.x * s;
		idx = idx > r ? r : idx;

		if(X[idx] <= target && X[idx + s] >= target){
			*ptr = idx;
			*(ptr+1) = idx+s;
			
			atomicExch(dev_ret, *ptr_u);
		}

		record = atomicAdd(dev_ret, 0);
		l = *ptr;
		r = *(ptr+1);
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
	int ret_idx, * dev_ret, * ret_value;
	
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
	err_code[6] = cudaMalloc( &ret_value , sizeof(int) );
	err_code[7] = cudaMalloc( &flags , num_blocks * sizeof(int) );
	for(int i=0; i<8; i++){ gerror(err_code[i]); }

	gerror(cudaMemcpy(dev_X, host_X, X_size, cudaMemcpyHostToDevice));
	gerror(cudaMemset(flags, 0, num_blocks * sizeof(int)));

	cudaDeviceSynchronize();

	ret_idx = 10086;

	printf("launching %u blocks, %u threads per block.\n", num_blocks, threads_per_block);

	d->Dg = {num_blocks, 1, 1};
	d->Db = {threads_per_block, 1, 1};
	d->Ns = 3 * sizeof(int) + 2 * (2 + threads_per_block) * sizeof(int);
	printf("Ns : %d\n", d->Ns);
	gstart();
	search_main<<<d->Dg, d->Db, d->Ns>>>(dev_X, X_len, target, num_threads, dev_ret, num_blocks, ret_value, flags);
	gend(&gputime);
	printf("gputime : %f ms\n", gputime);
	gerror( cudaGetLastError() );
	gerror( cudaDeviceSynchronize() );

	gerror(cudaMemcpy(&ret_idx, ret_value, sizeof(int), cudaMemcpyDeviceToHost));
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
	gerror(cudaFree(ret_value));
	gerror(cudaFree(flags));
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
