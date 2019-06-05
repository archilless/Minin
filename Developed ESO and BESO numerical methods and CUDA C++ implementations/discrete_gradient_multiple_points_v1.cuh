void discrete_gradient_multiple_points_numerical_method_v1()
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
	bool *h_check_optimal = (bool *)malloc(sizeof(bool));
	bool *h_check_optimal_globally = (bool *)malloc(sizeof(bool));
	bool *h_mask = (bool *)malloc(N * N * sizeof(bool));
	cuComplex *dev_gamma_array;
	cuComplex *dev_solution;
	cuComplex *dev_z;
	cuComplex *dev_new_z_extended;
	cuComplex *h_gamma_array = (cuComplex *)malloc((2 * N - 1) * (2 * N - 1) * sizeof(cuComplex));
	float *dev_gradient;
	float *dev_intensities_in_maxs;
	float *dev_sum_of_max_intensities;
	float *dev_actual_residual;
	float h_sum_of_max_intensities = 0.f;
	float h_sum_of_max_intensities_prev = 0.f;
	float tolerance = 0.001f; 	//Tolerance of GMRES
	unsigned int spread = N/100; 	//Spread of exclusion zone
	unsigned int h_i_line = N / 200;
	unsigned int *dev_indeces_of_maxumums;
	unsigned int h_n_maximums = 1;
	unsigned int *h_indeces_of_maximums = (unsigned int *)malloc(h_n_maximums * sizeof(unsigned int));
	unsigned int GMRES_n = 0;
	unsigned int *dev_check_sum;
	unsigned int h_check_sum = 0;
	unsigned int discrete_gradient = 0;
	unsigned int suboptimizations = 0;


	cudacall(cudaSetDevice(0));

	get_gamma_array((cuComplex **)&dev_gamma_array, (cufftHandle)plan);
	cudacall(cudaMemcpy(h_gamma_array, dev_gamma_array, (2 * N - 1) * (2 * N - 1) * sizeof(cuComplex), cudaMemcpyDeviceToHost));

	cudacall(cudaMalloc((void**)&dev_mask, N * N * sizeof(bool)));
	cudacall(cudaMalloc((void**)&dev_grad_bool_to_add, N * N * sizeof(bool)));
	cudacall(cudaMalloc((void**)&dev_grad_bool_to_sub, N * N * sizeof(bool)));
	cudacall(cudaMalloc((void**)&dev_solution, N * N * sizeof(cuComplex)));
	cudacall(cudaMalloc((void**)&dev_z, N * N * sizeof(cuComplex)));
	cudacall(cudaMalloc((void**)&dev_check_sum, sizeof(unsigned int)));
	cudacall(cudaMalloc((void**)&dev_indeces_of_maxumums, h_n_maximums * sizeof(unsigned int)));
	cudacall(cudaMalloc((void**)&dev_gradient, N * N * sizeof(float)));
	cudacall(cudaMalloc((void**)&dev_intensities_in_maxs, h_n_maximums * sizeof(float)));
	cudacall(cudaMalloc((void**)&dev_sum_of_max_intensities, sizeof(float)));


//	init_mask <<< N * N / 512, 512 >>> ((bool *)dev_mask);
//	cudacheckSYN();

//	init_mask_kernel <<< blocks, threads >>> ((bool *)dev_mask);
//	cudacheckSYN();


	std::string line;
	std::ifstream myfile ("/home/linux/Savings/Project/Test_GMRES/data/cylinder_1024.txt");
	if (myfile.is_open())
	{
		unsigned int index = 0;
		while ( getline (myfile,line) )
		{
			h_mask[index++] = (line == "1");
		}
		myfile.close();
	}
	else fprintf(stderr, "Unable to open file");
	cudacall(cudaMemcpy(dev_mask, h_mask, N * N * sizeof(bool), cudaMemcpyHostToDevice));


	init_x0_kernel <<< blocks, threads >>> ((cuComplex *)dev_solution);
	cudacheckSYN();
	
	set_n_maximums_kernel <<< h_n_maximums, 1 >>>((unsigned int *)dev_indeces_of_maxumums, h_n_maximums, h_i_line);
	cudacheckSYN();

	cudacall(cudaMemcpy(h_indeces_of_maximums, dev_indeces_of_maxumums, h_n_maximums * sizeof(unsigned int), cudaMemcpyDeviceToHost));


	FFT_GMRES_with_CUDA_extended_with_host_memory_single_debugged((const cuComplex *)dev_gamma_array, (const bool *)dev_mask, (cuComplex *)dev_solution, (float **)&dev_actual_residual, (unsigned int *)&GMRES_n, (cufftHandle)plan, (cublasHandle_t *)&handle, tolerance, (bool)0, 0, 100);
	cudacall(cudaFree((float *)dev_actual_residual));
	printf("%i\t%i\n", GMRES_n, discrete_gradient);

	do
	{
		h_check_sum = 0;

		init_x0_kernel <<< blocks, threads >>> ((cuComplex *)dev_solution);
		cudacheckSYN();

		FFT_GMRES_with_CUDA_extended_with_host_memory_single_debugged((const cuComplex *)dev_gamma_array, (const bool *)dev_mask, (cuComplex *)dev_solution, (float **)&dev_actual_residual, (unsigned int *)&GMRES_n, (cufftHandle)plan, (cublasHandle_t *)&handle, tolerance, (bool)0, 0, 100);
		cudacall(cudaFree((float *)dev_actual_residual));
		printf("%i\t%i\n", GMRES_n, discrete_gradient);

		cudacall(cudaMemset((bool *)dev_grad_bool_to_add, true, N * N * sizeof(bool)));
		cudacall(cudaMemset((bool *)dev_grad_bool_to_sub, true, N * N * sizeof(bool)));

		for(unsigned int counter_of_max = 0; counter_of_max < h_n_maximums; counter_of_max++)
		{
			init_x0_kernel <<< blocks, threads >>> ((cuComplex *)dev_z);

			FFT_GMRES_with_CUDA_extended_with_host_memory_single_debugged((const cuComplex *)dev_gamma_array, (const bool *)dev_mask, (cuComplex *)dev_z, (float **)&dev_actual_residual, (unsigned int *)&GMRES_n, (cufftHandle)plan, (cublasHandle_t *)&handle, tolerance, (bool)1, *(h_indeces_of_maximums + counter_of_max), 100);
			cudacall(cudaFree((float *)dev_actual_residual));

			fprintf(stderr, "max = %u\n", *(h_indeces_of_maximums + counter_of_max));

			G_x_fft_matvec((cuComplex *)dev_gamma_array, (bool *)dev_mask, (cuComplex *)dev_z, (cuComplex **)&dev_new_z_extended, (cufftHandle)plan, (bool)1);

			_2D_to_1D_kernel_to_compute_gradient <<< blocks, threads >>> ((cuComplex *)dev_solution, (cuComplex *)dev_new_z_extended, (float *)dev_gradient, (unsigned int *)dev_indeces_of_maxumums + counter_of_max);
			cudacheckSYN();

			cudacall(cudaFree((cuComplex *)dev_new_z_extended));
		
			dev_grad_bool_kernel<<< N * N / 512, 512 >>> ((bool *)dev_grad_bool_to_add, (bool *)dev_grad_bool_to_sub, (float *)dev_gradient);
			cudacheckSYN();
		}

		suboptimizations = 0;
		while (h_check_sum != N * N)//(suboptimizations < 1)//
		{
			cudacall(cudaMemset((unsigned int *)dev_check_sum, 0, sizeof(unsigned int)));
			get_neighbour_indices_for_gradient_multiple_maximums_v1 <<<  blocks, threads >>> ((bool *)dev_mask, (bool *)dev_grad_bool_to_add, (bool *)dev_grad_bool_to_sub, (unsigned int *)dev_check_sum, h_i_line, spread);
			cudacheckSYN();	

			cudacall(cudaMemcpy(&h_check_sum, dev_check_sum, sizeof(unsigned int), cudaMemcpyDeviceToHost));

			fprintf(stderr, "dev_check_sum = %i\n", h_check_sum);
			suboptimizations ++;
		}

		init_x0_kernel <<< blocks, threads >>> ((cuComplex *)dev_solution);
		cudacheckSYN();

		FFT_GMRES_with_CUDA_extended_with_host_memory_single_debugged((const cuComplex *)dev_gamma_array, (const bool *)dev_mask, (cuComplex *)dev_solution, (float **)&dev_actual_residual, (unsigned int *)&GMRES_n, (cufftHandle)plan, (cublasHandle_t *)&handle, tolerance, (bool)0, 0, 100);
		cudacall(cudaFree((float *)dev_actual_residual));
		printf("%i\t%i\n", GMRES_n, discrete_gradient);

		get_intensities_in_maximums_kernel <<< h_n_maximums, 1 >>> ((unsigned int *)dev_indeces_of_maxumums, (cuComplex *)dev_solution, (float *)dev_intensities_in_maxs);
		cudacheckSYN();
		
		cudacall(cudaMemset((unsigned int *)dev_sum_of_max_intensities, 0, sizeof(unsigned int)));

		sum_squares_reduce_float_kernel <<< 1, h_n_maximums >>> ((float *)dev_intensities_in_maxs, (float *)dev_sum_of_max_intensities);
		cudacheckSYN();

		cudacall(cudaMemcpy(&h_sum_of_max_intensities, dev_sum_of_max_intensities, sizeof(float), cudaMemcpyDeviceToHost));

		saveGPUrealtxt_prefixed("/media/linux/4db3d51d-3503-451d-aff7-07e3ce95927e/steepest_descent/111", (bool *)dev_mask, (cuComplex *)dev_solution, h_sum_of_max_intensities, (unsigned int)discrete_gradient);
		h_sum_of_max_intensities_prev = h_sum_of_max_intensities;

		discrete_gradient++;
		printf("suboptimizations = %i\tintensity = %f\n\n\n", suboptimizations, h_sum_of_max_intensities);
	}
	while (true); //((h_check_sum < N * N) || (discrete_gradient < 100));

	cudacall(cudaFree((bool *)dev_mask));
	cudacall(cudaFree((bool *)dev_grad_bool_to_add));
	cudacall(cudaFree((bool *)dev_grad_bool_to_sub));
	cudacall(cudaFree((float *)dev_gradient));
	cudacall(cudaFree((float *)dev_intensities_in_maxs));
	cudacall(cudaFree((float *)dev_sum_of_max_intensities));
	cudacall(cudaFree((cuComplex *)dev_z));
	cudacall(cudaFree((cuComplex *)dev_solution));
	cudacall(cudaFree((cuComplex *)dev_gamma_array));
	cudacall(cudaFree((unsigned int *)dev_check_sum));
	cudacall(cudaFree((unsigned int *)dev_indeces_of_maxumums));
	free((unsigned int *)h_indeces_of_maximums);
	free((bool *)h_check_optimal);
	free((bool *)h_check_optimal_globally);
	free((cuComplex *)h_gamma_array);
	cufftcall(cufftDestroy(plan));
	cublascall(cublasDestroy_v2(handle));
}
