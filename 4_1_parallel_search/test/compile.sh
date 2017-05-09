ALG="parallel_search"

for y in ../v[1245678]*
do
	cd $y;
	nvcc -std=c++11 -ccbin g++ -I../../common/inc -dc -arch=sm_35 -o "$ALG.o" -c "$ALG.cu";
	/usr/local/cuda-8.0/bin/nvcc -ccbin g++ -arch=sm_35 -o "$ALG" "$ALG.o" -lcudadevrt;
	rm $y/*.o;
	echo $y;
done

cd ../test;

