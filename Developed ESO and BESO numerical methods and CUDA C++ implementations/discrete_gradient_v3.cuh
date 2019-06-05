#include <sstream>

void discrete_gradient_numerical_method_v3()
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
	bool *h_mask = (bool *)malloc(N * N * sizeof(bool));
	bool *dev_dublicated_mask;
	unsigned int *dev_i_threshold;
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
	float tolerance = 0.0000009f; 	//Tolerance of GMRES
	unsigned int spread = N/1000; 	//Spread of exclusion zone
	float h_intensity_max = 0.f;
	float h_intensity_prev = 0.f;	

	cudacall(cudaSetDevice(0));

	get_gamma_array((cuComplex **)&dev_gamma_array, (cufftHandle)plan);


/*
	cudacall(cudaMalloc((void**)&dev_gamma_array, (2 * N - 1) * (2 * N - 1) * sizeof(cuComplex)));
	std::string line;
	std::ifstream greens_FFTed_file ("data/greens_matrix_FFTed_1024.txt");
	if (greens_FFTed_file.is_open())
	{
		unsigned int index = 0;
		while ( getline (greens_FFTed_file,line) )
		{
			std::istringstream in_string_stream(line);

			in_string_stream >> h_gamma_array[index].x >> h_gamma_array[index].y;

			//fprintf(stderr, "%f\n", h_gamma_array[index].x);
			index ++;
		}
		greens_FFTed_file.close();
	}
	else fprintf(stderr, "Unable to open file");
	cudacall(cudaMemcpy(dev_gamma_array, h_gamma_array, (2 * N - 1) * (2 * N - 1) * sizeof(cuComplex), cudaMemcpyHostToDevice));		
*/
	cudacall(cudaMemcpy(h_gamma_array, dev_gamma_array, (2 * N - 1) * (2 * N - 1) * sizeof(cuComplex), cudaMemcpyDeviceToHost));

	cudacall(cudaMalloc((void**)&dev_mask, N * N * sizeof(bool)));
	cudacall(cudaMalloc((void**)&dev_solution, N * N * sizeof(cuComplex)));
	cudacall(cudaMalloc((void**)&dev_check_sum, sizeof(unsigned int)));
	cudacall(cudaMalloc((void**)&dev_index_of_max, sizeof(unsigned int)));
	cudacall(cudaMalloc((void**)&dev_z, N * N * sizeof(cuComplex)));
	cudacall(cudaMalloc((void**)&dev_gradient, N * N * sizeof(float)));
	cudacall(cudaMalloc((void**)&dev_i_threshold, N * N * sizeof(unsigned int)));

//	init_mask <<< N * N / 512, 512 >>> ((bool *)dev_mask);
/*	
	std::string line;
	std::ifstream myfile ("data/lens_1024.txt");
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
*/

	init_mask_kernel <<< blocks, threads >>> ((bool *)dev_mask);
	cudacheckSYN();

	init_x0_kernel <<< blocks, threads >>> ((cuComplex *)dev_solution);
	cudacheckSYN();

	FFT_GMRES_with_CUDA_extended_with_host_memory_single_debugged((const cuComplex *)dev_gamma_array, (const bool *)dev_mask, (cuComplex *)dev_solution, (float **)&dev_actual_residual, (unsigned int *)&GMRES_n, (cufftHandle)plan, (cublasHandle_t *)&handle, tolerance, (bool)0, 0, 100);
	cudacall(cudaFree((float *)dev_actual_residual));
	printf("%i\t%i\n", GMRES_n, discrete_gradient);

	//saveGPUrealtxt_B((const bool *)dev_mask, "data/mask_-1.txt", N * N);
	//saveGPUrealtxt_C((const cuComplex *)dev_solution, "data/solution_-1.txt", N * N);

	get_i_threshold_for_max((bool *)dev_mask, (unsigned int *)dev_i_threshold, (dim3)blocks, (dim3)threads);
	target_function_indexed_v2((unsigned int *)dev_i_threshold, (cuComplex *)dev_solution, (unsigned int *)dev_index_of_max, (float *)&h_intensity_max, blocks, threads);
	cudacall(cudaMemcpy(&h_index_of_max, dev_index_of_max, sizeof(unsigned int), cudaMemcpyDeviceToHost));


	printf("h_intensity_max = %f \n\n\n", h_intensity_max);
	printf("index_of_max = %u\n", h_index_of_max); 

	//saveGPUrealtxt_C((const cuComplex *)dev_solution, "data/solution_-1.txt", N * N);

	do
	{
		h_check_sum = 0;

		init_x0_kernel <<< blocks, threads >>> ((cuComplex *)dev_solution);
		cudacheckSYN();

		FFT_GMRES_with_CUDA_extended_with_host_memory_single_debugged((const cuComplex *)dev_gamma_array, (const bool *)dev_mask, (cuComplex *)dev_solution, (float **)&dev_actual_residual, (unsigned int *)&GMRES_n, (cufftHandle)plan, (cublasHandle_t *)&handle, tolerance, false, 0, 100);
		cudacall(cudaFree((float *)dev_actual_residual));
		printf("%i\t%i\n", GMRES_n, discrete_gradient);

		init_x0_kernel <<< blocks, threads >>> ((cuComplex *)dev_z);
		FFT_GMRES_with_CUDA_extended_with_host_memory_single_debugged((const cuComplex *)dev_gamma_array, (const bool *)dev_mask, (cuComplex *)dev_z, (float **)&dev_actual_residual, (unsigned int *)&GMRES_n, (cufftHandle)plan, (cublasHandle_t *)&handle, tolerance, (bool)1, h_index_of_max, 100);
		cudacall(cudaFree((float *)dev_actual_residual));

		G_x_fft_matvec((cuComplex *)dev_gamma_array, (bool *)dev_mask, (cuComplex *)dev_z, (cuComplex **)&dev_new_z_extended, (cufftHandle)plan, true);

		_2D_to_1D_kernel_to_compute_gradient <<< blocks, threads >>> ((cuComplex *)dev_solution, (cuComplex *)dev_new_z_extended, (float *)dev_gradient, (unsigned int *)dev_index_of_max);
		cudacheckSYN();
		//if (discrete_gradient == 0)
		//{
		//	saveGPUrealtxt_F((const float *)dev_gradient, "data/gradient_0.txt", N * N);
		//}


		cudacall(cudaFree((cuComplex *)dev_new_z_extended));

		suboptimizations = 0;
		while ((suboptimizations < 1) || (discrete_gradient == 0) && (h_check_sum != N * N)) //(h_check_sum != N * N)
		{
			cudacall(cudaMemset((unsigned int *)dev_check_sum, 0, sizeof(unsigned int)));
			get_neighbour_indices_for_gradient_v4 <<<  blocks, threads >>> ((bool *)dev_mask, (float *)dev_gradient, (unsigned int *)dev_check_sum, (unsigned int *)dev_index_of_max, spread);

			cudacheckSYN();	
			cudacall(cudaMemcpy(&h_check_sum, dev_check_sum, sizeof(unsigned int), cudaMemcpyDeviceToHost));

			fprintf(stderr, "dev_check_sum = %i\n", h_check_sum);
			suboptimizations ++;
		}

		FFT_GMRES_with_CUDA_extended_with_host_memory_single_debugged((const cuComplex *)dev_gamma_array, (const bool *)dev_mask, (cuComplex *)dev_solution, (float **)&dev_actual_residual, (unsigned int *)&GMRES_n, (cufftHandle)plan, (cublasHandle_t *)&handle, tolerance, (bool)0, 0, 100);
		cudacall(cudaFree((float *)dev_actual_residual));
		printf("%i\t%i\n", GMRES_n, discrete_gradient);

		get_i_threshold_for_max((bool *)dev_mask, (unsigned int *)dev_i_threshold, (dim3)blocks, (dim3)threads);
		target_function_indexed_v2((unsigned int *)dev_i_threshold, (cuComplex *)dev_solution, (unsigned int *)dev_index_of_max, (float *)&h_intensity_max, blocks, threads);
		cudacall(cudaMemcpy(&h_index_of_max, dev_index_of_max, sizeof(unsigned int), cudaMemcpyDeviceToHost));

		if (true)	//(h_intensity_max > h_intensity_prev)
		{
			saveGPUrealtxt_discrete_gradient((bool *)dev_mask, (cuComplex *)dev_solution, h_intensity_max, (unsigned int)discrete_gradient);
			h_intensity_prev = h_intensity_max;
		}

		discrete_gradient++;
		printf("h_intensity_max = %f \t suboptimizations = %i\n\n\n", h_intensity_max, suboptimizations);
	}
	while((h_check_sum < N * N) || (discrete_gradient < 2000));


	cudacall(cudaFree((bool *)dev_mask));
	cudacall(cudaFree((unsigned int *)dev_i_threshold));
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
