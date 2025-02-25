void discrete_gradient_numerical_method()
{
	dim3 blocks(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
	dim3 threads(Q, Q);

	time_t clock_time = clock();
	bool *h_check_optimal = (bool *)malloc(sizeof(bool));
	bool *h_check_optimal_globally = (bool *)malloc(sizeof(bool));
	
	cufftHandle plan;	
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	cufftcall(cufftPlan2d(&plan, 2 * N - 1, 2 * N - 1, CUFFT_C2C));

	bool *dev_mask;
	float *dev_gradient;
	cuComplex *dev_new_z_extended;
	cuComplex *dev_gamma_array;
	cuComplex *dev_solution;
	cuComplex *dev_z;
	cuComplex *h_gamma_array = (cuComplex *)malloc((2 * N - 1) * (2 * N - 1) * sizeof(cuComplex));
	unsigned int h_index_of_max = 0;
	unsigned int GMRES_n = 0;
	unsigned int *dev_index_of_max;
	unsigned int *dev_check_sum;
	unsigned int h_check_sum = 0;
	unsigned int discrete_gradient = 0;
	unsigned int suboptimizations = 0;

	float *dev_actual_residual;
	float tolerance = 0.001f;
	float h_instensity_max = 0.f;

	cudacall(cudaSetDevice(0));

	get_gamma_array((cuComplex **)&dev_gamma_array, (cufftHandle)plan);	
	
	cudacall(cudaMemcpy(h_gamma_array, dev_gamma_array, (2 * N - 1) * (2 * N - 1) * sizeof(cuComplex), cudaMemcpyDeviceToHost));

	cudacall(cudaMalloc((void**)&dev_mask, N * N * sizeof(bool)));
	init_mask <<< N * N / 512, 512 >>> ((bool *)dev_mask);
	cudacheckSYN();

	cudacall(cudaMalloc((void**)&dev_solution, N * N * sizeof(cuComplex)));
	init_x0_kernel <<< blocks, threads >>> ((cuComplex *)dev_solution);
	cudacheckSYN();

	cudacall(cudaMalloc((void**)&dev_check_sum, sizeof(unsigned int)));
	cudacall(cudaMalloc((void**)&dev_index_of_max, sizeof(unsigned int)));
	cudacall(cudaMalloc((void**)&dev_z, N * N * sizeof(cuComplex)));
	cudacall(cudaMalloc((void**)&dev_gradient, N * N * sizeof(float)));

	FFT_GMRES_with_CUDA((const cuComplex *)dev_gamma_array, (const bool *)dev_mask, (cuComplex *)dev_solution, (float **)&dev_actual_residual, (unsigned int *)&GMRES_n, (cufftHandle)plan, (cublasHandle_t *)&handle, tolerance, (bool)0, 0);
	cudacall(cudaFree((float *)dev_actual_residual));
	printf("%i\t%i\n", GMRES_n, discrete_gradient);

	target_function_indexed((cuComplex *)dev_solution, (unsigned int *)dev_index_of_max, (float *)&h_instensity_max);
	cudacall(cudaMemcpy(&h_index_of_max, dev_index_of_max, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	do
	{

		init_x0_kernel <<< blocks, threads >>> ((cuComplex *)dev_z);
		FFT_GMRES_with_CUDA((const cuComplex *)dev_gamma_array, (const bool *)dev_mask, (cuComplex *)dev_z, (float **)&dev_actual_residual, (unsigned int *)&GMRES_n, (cufftHandle)plan, (cublasHandle_t *)&handle, tolerance, (bool)1, h_index_of_max);
		cudacall(cudaFree((float *)dev_actual_residual));

		G_x_fft_matvec((cuComplex *)dev_gamma_array, (bool *)dev_mask, (cuComplex *)dev_z, (cuComplex **)&dev_new_z_extended, (cufftHandle)plan, (bool)1);

		_2D_to_1D_kernel_to_compute_gradient <<< blocks, threads >>> ((cuComplex *)dev_solution, (cuComplex *)dev_new_z_extended, (float *)dev_gradient, (unsigned int *)dev_index_of_max);
		cudacheckSYN();

		cudacall(cudaFree((cuComplex *)dev_new_z_extended));
	
		suboptimizations = 0;
		while (h_check_sum != N * N)
		{
			cudacall(cudaMemset((unsigned int *)dev_check_sum, 0, sizeof(unsigned int)));
			get_neighbour_indices_for_gradient <<<  blocks, threads >>> ((bool *)dev_mask, (float *)dev_gradient, (unsigned int *)dev_check_sum);
			cudacheckSYN();	
			cudacall(cudaMemcpy(&h_check_sum, dev_check_sum, sizeof(unsigned int), cudaMemcpyDeviceToHost));

			fprintf(stderr, "dev_check_sum = %i\n", h_check_sum);
			suboptimizations ++;
		}

		FFT_GMRES_with_CUDA((const cuComplex *)dev_gamma_array, (const bool *)dev_mask, (cuComplex *)dev_solution, (float **)&dev_actual_residual, (unsigned int *)&GMRES_n, (cufftHandle)plan, (cublasHandle_t *)&handle, tolerance, (bool)0, 0);
		cudacall(cudaFree((float *)dev_actual_residual));
		printf("%i\t%i\n", GMRES_n, discrete_gradient);

		target_function_indexed((cuComplex *)dev_solution, (unsigned int *)dev_index_of_max, (float *)&h_instensity_max);
		cudacall(cudaMemcpy(&h_index_of_max, dev_index_of_max, sizeof(unsigned int), cudaMemcpyDeviceToHost));

		saveGPUrealtxt_discrete_gradient((bool *)dev_mask, (cuComplex *)dev_solution, h_instensity_max, (unsigned int)discrete_gradient++);

		printf("suboptimizations = %i\n\n\n", suboptimizations);
	}
	while(h_check_sum < N * N);


	cudacall(cudaFree((bool *)dev_mask));
	cudacall(cudaFree((cuComplex *)dev_z));
	cudacall(cudaFree((float *)dev_gradient));
	cudacall(cudaFree((cuComplex *)dev_solution));
	cudacall(cudaFree((cuComplex *)dev_gamma_array));
	cudacall(cudaFree((unsigned int *)dev_index_of_max));
	cudacall(cudaFree((unsigned int *)dev_check_sum));
	free((bool *)h_check_optimal);
	free((bool *)h_check_optimal_globally);
	free((cuComplex *)h_gamma_array);
	cufftcall(cufftDestroy(plan));
	cublascall(cublasDestroy_v2(handle));
}
