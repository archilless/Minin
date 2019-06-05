#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <string>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "kernels.cuh"
#include "saveGPU.cuh"
#include "GMRES.cuh"
//#include "GMRES_with_host_single.cuh"
#include "GMRES_with_host_single_debugged.cuh"
//#include "GMRES_with_host_multiple.cuh"
//#include "GMRES_Batched.cuh"
//#include "greedy_v1.cuh"
//#include "discrete_gradient_v1.cuh"
//#include "discrete_gradient_v2.cuh"
//#include "discrete_gradient_v3.cuh"
//#include "discrete_gradient_multiple_points_v1.cuh"
//#include "discrete_gradient_multiple_points_v2.cuh"
#include "launch_GMRES.cuh"
//#include "double_resolute_by_size.cuh"
//#include "steepest_descent_one_point_one_direction_v1.cuh"
#include "steepest_descent_one_point_two_directions_v1.cuh"	



int main()
{
	printf("IN MAIN\n");
	time_t clock_time = clock();


	//discrete_gradient_numerical_method_v1();
	//discrete_gradient_numerical_method_v2();
	//discrete_gradient_numerical_method_v3();
	//greedy_numerical_method_v1();
	//discrete_gradient_multiple_points_numerical_method_v1();
	//discrete_gradient_multiple_points_numerical_method_v2();
	//launch_GMRES();
	//double_resolute_by_size();
	//steepest_descent_one_point_one_direction_numerical_method_v1();
	steepest_descent_one_point_two_directions_numerical_method_v1();


	printf("Successful exit from CUDA\n");
	printf("Consumption time with OUTPUTTING = %f seconds \n", (float)(clock() - clock_time) / (float)(CLOCKS_PER_SEC));

	return 0;
}



