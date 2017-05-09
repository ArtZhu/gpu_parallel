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

__global__ void search_main(number * X, int n, number target, int num_threads, ull * dev_ret)
{
	int l, r, *ptr;
	ull record, *ptr_u;

	ptr_u = &record;
	ptr = (int *) ptr_u;

	//record = atomicAdd(dev_ret, 0);
	record = atomicCAS(dev_ret, (ull) -2L, 0);
	l = *ptr;
	r = *(ptr+1);

	//printf("%llx %u %d %d\n", (ull) -2L, threadIdx.x, l, r);

	int block_n, start, s, idx;

	int i=0;
	while(i < 10000 && r - l > 1){
		i++;

		/*
		if(threadIdx.x == 0 && blockIdx.x == 0){
			printf("%llx\n", record);
			printf("%d %d\n", l, r);
		}*/

		block_n = (r - l) / gridDim.x;
		
		s = block_n / blockDim.x;
		s = s > 0 ? s : 1;

		start = l + (blockIdx.x * block_n);

		idx = start + threadIdx.x * s;
		if(idx < r){

			//printf("threadIdx.x : %u\nblock_n : %d\ns : %d\nstart : %d\nidx : %d\n", threadIdx.x, block_n, s, start, idx);
			//printf("threadIdx.x : %u\n idx: %d, X[idx]: %d, X[idx+s]: %d\n", threadIdx.x, idx, X[idx], X[idx+s]);

			if(X[idx] <= target && X[idx + s] >= target){
				*ptr = idx;
				*(ptr+1) = idx+s;

				atomicExch(dev_ret, *ptr_u);
			}

			if(threadIdx.x + blockIdx.x * blockDim.x == num_threads - 1)
				if(X[idx + s] <= target){
					*ptr = idx+s;

					atomicExch(dev_ret, *ptr_u);
				}
				
		}

		record = atomicCAS(dev_ret, (ull) -2L, 0);
		l = *ptr;
		r = *(ptr+1);
	}

	/*
	if(threadIdx.x == 0 && blockIdx.x == 0){
		printf("%llx\n", record);
		printf("%d %d\n", l, r);
	}
	*/
	
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
	int ret_idx_dev, ret_idx_host;
	ull ret_ull, * dev_ret;
	
	cudaSetDevice(0);
	cudaDeviceReset();

	unsigned int num_blocks = (1023 + num_threads) / 1024;
	unsigned int threads_per_block = num_threads > 1024 ? 1024 : num_threads;


	// X_len + 2 for the algorithm element at idx 0 and n + 1 (originally 1, 2, ..., n)
	err_code[0] = cudaMalloc( &dev_X , X_size );
	err_code[1] = cudaMalloc( &dev_ret , sizeof(ull));
	for(int i=0; i<2; i++){ gerror(err_code[i]); }

	int _dev_ret[2];
	_dev_ret[0] = 0; _dev_ret[1] = X_len + 1;

	gerror(cudaMemcpy(dev_ret, _dev_ret, sizeof(ull), cudaMemcpyHostToDevice));

	gerror(cudaMemcpy(dev_X, host_X, X_size, cudaMemcpyHostToDevice));

	cudaDeviceSynchronize();

	ret_idx_dev = 10086;

	//printf("launching %u blocks, %u threads per block.\n", num_blocks, threads_per_block);

	d->Dg = {num_blocks, 1, 1};
	d->Db = {threads_per_block, 1, 1};
	//d->Ns = 3 * sizeof(int) + 2 * (2 + threads_per_block) * sizeof(int);
	//printf("Ns : %lu\n", d->Ns);
	gstart();
	search_main<<<d->Dg, d->Db, d->Ns>>>(dev_X, X_len, target, num_threads, dev_ret);
	gend(&gputime);
	//printf("gputime : %f ms\n", gputime);
	gerror( cudaGetLastError() );
	gerror( cudaDeviceSynchronize() );

	gerror(cudaMemcpy(&ret_ull, dev_ret, sizeof(ull), cudaMemcpyDeviceToHost));
	ret_idx_dev = *((int *) &ret_ull);
	//printf("device idx = %d;\n", ret_idx_dev);

	ret_idx_host = 10086;

	cstart();
	ret_idx_host = cpu_search(host_X + 1, X_len, target);
	cend(&cputime);
	//printf("cputime : %f ms\n", cputime);
	//printf("host idx = %d;\n", ret_idx_host);
	if(ret_idx_dev == ret_idx_host){
		printf("N %f %f\n", gputime, cputime);
	}else{
		printf("E %d %d\n", ret_idx_dev, ret_idx_host);
	}

	gerror(cudaFree(dev_X));
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
					default:
						sscanf(argv[i], "%d", &target);
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
