v1

	contains a first version of the working parallel search algorithm. 

	A mix of cpu checking and gpu execution (multiple kernels, Memcpy back to check result in CPU)

	A mix of kernels and host functions

v2

	dynamic parallelism to first factor the thread launching from host function 
	to pure device execution : 1 kernel launch that spawns children.

	basically the same as v1
