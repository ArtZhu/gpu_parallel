rm runtimes.txt;
touch runtimes.txt;

i=0;
while read p; do
	let "i += 1";
	echo $i;
	length=${p##* };
	p=${p% *};
	threadnum=${p##* };
	p=${p% *};
	target=${p## *};
	echo "#t $target l $length n $threadnum" >> runtimes.txt;

	#for y in ../v[1245678]* 
	for y in ../v[8]_* 
	do
		echo "${y#\/v*}" >> runtimes.txt;
		timeout 1s $y/parallel_search -l $length -t $threadnum $target >> runtimes.txt;
	done
done <testcases.txt
