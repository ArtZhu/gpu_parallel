/* filename : ranking_sorted_sequences.cu
 * author: Tiane Zhu
 * date : Mar 26, 2017
 *
 * this file contains an O(1) ranking parallel algorithm
 *
 * this program is an implementation of the ranking sorted sequences
 *		ALGORITHM 4.2 in 
 * "An Introduction to Parallel Algorithms" - by Joseph Jaja
 *			p150 - ISBN 9-789201-548563
 */

////
// Input : 	A = (a1, ... , an)
// 					B = (b1, ... , bm)
//			sqm = sqrt(m)
////
// Output : rank(B : A)
////
// begin
// 		1. If m < 4, then rank the elements of B 
//			 							by applying Alg 4_1 with p = n
//								 then exit
//		2. Concurrently rank b_sqm, b_2sqm, ..., bm in A 
//										by applying Alg 4_1 with p = sqrt(n)
//			 Let 	j[i] = rank(b_isqm : A)
//						j[0] = 0
//		3. For 0 <= i <= sqm - 1,
//					let B_i = ( b_isqm+1, ... , b_(i+1)sqm - 1 )
//					let A_i = ( a_j[i]+1, ... , a_j[i+1] )
//			 If j[i] == j[i+1], then
//					set rank(B_i : A_i) = (0, ... , 0)
//			 else
//					recurse compute rank(B_i : A_i)
//		4. Let 1 <= k <= m be an arbitrary index not multiple of sqm
//			 Let i = floor( k / sqm )
//			 rank(b_k : A) = j[i] + rank(b_k : A_i)
// end
///
