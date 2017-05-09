#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char * argv[])
{
	FILE * fp = fopen("testcases.txt", "w+");

	int tcases = 10;

	if(argc > 1)
		tcases = atoi(argv[1]);

	int i, target, length, num_threads;

	for(i=0; i<tcases; i++){
		target = rand() % 33554432;

		for(length=32; length<=33554432; length*=2){
			
			for(num_threads=2; num_threads<=33554432; num_threads*=2){
				if(num_threads <= length)
					fprintf(fp, "%d %d %d\n", target, num_threads, length);
			}
		}
	}
}
