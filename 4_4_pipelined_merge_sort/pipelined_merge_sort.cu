/* filename : pipelined_merge_sort.cu
 * author : Tiane Zhu
 * date : Mar 26, 2017
 *
 *
 */

/////
// Input : For each node v of a binary tree, 
//			a sorted list L_s[v] such that 
//					v is full whenever s >= 3 * alt(v)
////
// Output : For each node v, 
//			a sorted list L_{s+1}[v] such that 
//					v is full whenever s >= 3 * alt(v) - 1
////
// Algorithm at stage (s + 1)
//		begin
//			for (all active nodes v) pardo
//				1. Let u and w be the children of v. 
//								Set L_{s+1}'[u] = Sample(L_s[u])
//								and L_{s+1}'[w] = Sample(L_s[w])
//				2. Merge L_{s+1}'[u] and L_{s+1}'[w] into
//								SORTED list L_{s+1}[v]
//		end
////
