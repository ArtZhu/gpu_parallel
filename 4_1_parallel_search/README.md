v1

	contains a first version of the working parallel search algorithm. 

	A mix of cpu checking and gpu execution (multiple kernels, Memcpy back to check result in CPU)

	A mix of kernels and host functions

v2

	dynamic parallelism to first factor the thread launching from host function 
	to pure device execution : 1 kernel launch that spawns children.

	basically the same as v1

v3
	
	global sync(not finished)

v4
	
	thread deals with X array in a block-wise fashion, thread count does not affect the run time greatly

v5
	
	minor improvement of v4

v6
	
	Prof. Bento's 1 global var no-sync idea

v7

	dynamic parallelism : only the thread that found the interval spawn threads.

