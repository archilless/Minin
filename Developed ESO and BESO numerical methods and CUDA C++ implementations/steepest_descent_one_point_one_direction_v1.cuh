//#include <sstream>

void steepest_descent_one_point_one_direction_numerical_method_v1()
{
	dim3 blocks(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
	dim3 threads(Q, Q);

	time_t clock_time = clock();
	
	cufftHandle plan;	
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	cufftcall(cufftPlan2d(&plan, 2 * N - 1, 2 * N - 1, CUFFT_C2C));

	bool *dev_mask;
	bool *dev_grad_bool_to_add;
	bool *dev_grad_bool_to_sub;
	bool *h_mask = (bool *)malloc(N * N * sizeof(bool));
	bool *dev_checker_minus_100;
	bool h_checker_minus_100;
	cuComplex *dev_gamma_array;
	cuComplex *dev_solution;
	cuComplex *dev_z;
	cuComplex *dev_new_z_extended;
	float *dev_gradient;
	float *dev_intensity_in_max;
	float *dev_actual_residual;
	float h_intensity_in_max = 0.f;
	float tolerance = 0.001f; 	//Tolerance of GMRES
	unsigned int spread = N/100; 	//Spread of exclusion zone
	unsigned int h_i_line = N / 200;
	unsigned int *dev_index_of_maxumum;
	unsigned int h_index_of_maximum = 5632;
	unsigned int GMRES_n = 0;
	unsigned int steepest_descent = 4594;
	unsigned int *dev_suitable_steepest_grad_index_to_add;


	cudacall(cudaSetDevice(0));

	get_gamma_array((cuComplex **)&dev_gamma_array, (cufftHandle)plan);

	cudacall(cudaMalloc((void**)&dev_mask, N * N * sizeof(bool)));
	cudacall(cudaMalloc((void**)&dev_checker_minus_100, sizeof(bool)));
	cudacall(cudaMalloc((void**)&dev_solution, N * N * sizeof(cuComplex)));
	cudacall(cudaMalloc((void**)&dev_z, N * N * sizeof(cuComplex)));
	cudacall(cudaMalloc((void**)&dev_index_of_maxumum, sizeof(unsigned int)));
	cudacall(cudaMalloc((void**)&dev_suitable_steepest_grad_index_to_add, sizeof(unsigned int)));
	cudacall(cudaMalloc((void**)&dev_gradient, N * N * sizeof(float)));
	cudacall(cudaMalloc((void**)&dev_intensity_in_max, sizeof(float)));


//	init_mask <<< N * N / 512, 512 >>> ((bool *)dev_mask);					//1
//	cudacheckSYN();										//1
//	init_mask_kernel <<< blocks, threads >>> ((bool *)dev_mask);				//2
//	cudacheckSYN();										//2


	std::string line;
	std::ifstream myfile ("/home/linux/Savings/Project/data/discrete_gradient/mask_4593.txt");	//3
	if (myfile.is_open())									//3
	{											//3
		unsigned int index = 0;								//3
		while ( getline (myfile,line) )							//3
		{										//3
			h_mask[index++] = (line == "1");					//3
		}										//3
		myfile.close();									//3
	}											//3
	else fprintf(stderr, "Unable to open file");						//3
	cudacall(cudaMemcpy(dev_mask, h_mask, N * N * sizeof(bool), cudaMemcpyHostToDevice));	//3


	init_x0_kernel <<< blocks, threads >>> ((cuComplex *)dev_solution);
	cudacheckSYN();
	
	cudacall(cudaMemcpy(dev_index_of_maxumum, &h_index_of_maximum, sizeof(unsigned int), cudaMemcpyHostToDevice));

	FFT_GMRES_with_CUDA_extended_with_host_memory_single_debugged((const cuComplex *)dev_gamma_array, (const bool *)dev_mask, (cuComplex *)dev_solution, (float **)&dev_actual_residual, (unsigned int *)&GMRES_n, (cufftHandle)plan, (cublasHandle_t *)&handle, tolerance, (bool)0, 0, 100);
	cudacall(cudaFree((float *)dev_actual_residual));
	printf("%i\t%i\n", GMRES_n, steepest_descent);

	do
	{
		init_x0_kernel <<< blocks, threads >>> ((cuComplex *)dev_solution);
		cudacheckSYN();

		FFT_GMRES_with_CUDA_extended_with_host_memory_single_debugged((const cuComplex *)dev_gamma_array, (const bool *)dev_mask, (cuComplex *)dev_solution, (float **)&dev_actual_residual, (unsigned int *)&GMRES_n, (cufftHandle)plan, (cublasHandle_t *)&handle, tolerance, false, 0, 100);
		cudacall(cudaFree((float *)dev_actual_residual));

		printf("GMRES_n = %i\tsteepest_descent = %i\n", GMRES_n, steepest_descent);

		{
			init_x0_kernel <<< blocks, threads >>> ((cuComplex *)dev_z);

			FFT_GMRES_with_CUDA_extended_with_host_memory_single_debugged((const cuComplex *)dev_gamma_array, (const bool *)dev_mask, (cuComplex *)dev_z, (float **)&dev_actual_residual, (unsigned int *)&GMRES_n, (cufftHandle)plan, (cublasHandle_t *)&handle, tolerance, (bool)1, h_index_of_maximum, 100);
			cudacall(cudaFree((float *)dev_actual_residual));

			G_x_fft_matvec((cuComplex *)dev_gamma_array, (bool *)dev_mask, (cuComplex *)dev_z, (cuComplex **)&dev_new_z_extended, (cufftHandle)plan, (bool)1);

			_2D_to_1D_kernel_to_compute_gradient <<< blocks, threads >>> ((cuComplex *)dev_solution, (cuComplex *)dev_new_z_extended, (float *)dev_gradient, (unsigned int *)dev_index_of_maxumum);
			cudacheckSYN();

			cudacall(cudaFree((cuComplex *)dev_new_z_extended));
		
			adjust_grad_to_mask_kernel <<< blocks, threads >>> ((float *)dev_gradient, (bool *)dev_mask, h_i_line, spread);
		}


		target_function_indexed_v3((float *)dev_gradient, (unsigned int *)dev_suitable_steepest_grad_index_to_add, blocks, threads);//get_max_of_gradient

		minus_100_gradient_checker_kernel <<< 1, 1 >>> ((bool *)dev_checker_minus_100, (unsigned int *)dev_suitable_steepest_grad_index_to_add, (float *)dev_gradient);
		cudacheckSYN();

		cudacall(cudaMemcpy(&h_checker_minus_100, dev_checker_minus_100, sizeof(bool), cudaMemcpyDeviceToHost));

		if (h_checker_minus_100)
		{
			break;
		}
		
		get_suboptimal_mask_kernel <<< 1, 1 >>> ((bool *)dev_mask, (unsigned int *)dev_suitable_steepest_grad_index_to_add);
		cudacheckSYN();

		init_x0_kernel <<< blocks, threads >>> ((cuComplex *)dev_solution);
		cudacheckSYN();

		FFT_GMRES_with_CUDA_extended_with_host_memory_single_debugged((const cuComplex *)dev_gamma_array, (const bool *)dev_mask, (cuComplex *)dev_solution, (float **)&dev_actual_residual, (unsigned int *)&GMRES_n, (cufftHandle)plan, (cublasHandle_t *)&handle, tolerance, (bool)0, 0, 100);
		cudacall(cudaFree((float *)dev_actual_residual));
		printf("%i\t%i\n", GMRES_n, steepest_descent);

		get_intensities_in_maximums_kernel <<< 1, 1 >>> ((unsigned int *)dev_index_of_maxumum, (cuComplex *)dev_solution, (float *)dev_intensity_in_max);
		cudacheckSYN();
		
		cudacall(cudaMemcpy(&h_intensity_in_max, dev_intensity_in_max, sizeof(float), cudaMemcpyDeviceToHost));

		saveGPUrealtxt_prefixed("/media/linux/4db3d51d-3503-451d-aff7-07e3ce95927e/steepest_descent/1", (bool *)dev_mask, (cuComplex *)dev_solution, h_intensity_in_max, (unsigned int)steepest_descent);

		steepest_descent++;
		printf("intensity = %f\n\n\n", h_intensity_in_max);
	}
	while (true); //(steepest_descent < 100);

	cudacall(cudaFree((bool *)dev_mask));
	cudacall(cudaFree((bool *)dev_grad_bool_to_add));
	cudacall(cudaFree((bool *)dev_grad_bool_to_sub));
	cudacall(cudaFree((float *)dev_gradient));
	cudacall(cudaFree((float *)dev_intensity_in_max));
	cudacall(cudaFree((cuComplex *)dev_z));
	cudacall(cudaFree((cuComplex *)dev_solution));
	cudacall(cudaFree((cuComplex *)dev_gamma_array));
	cudacall(cudaFree((unsigned int *)dev_index_of_maxumum));
	cudacall(cudaFree((unsigned int *)dev_suitable_steepest_grad_index_to_add));
	cufftcall(cufftDestroy(plan));
	cublascall(cublasDestroy_v2(handle));
}
