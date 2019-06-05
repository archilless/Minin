#include <sstream>

void discrete_gradient_multiple_points_numerical_method_v2()
{
	dim3 blocks(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
	dim3 threads(Q, Q);

	time_t clock_time = clock();
	
	cufftHandle plan;	
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	cufftcall(cufftPlan2d(&plan, 2 * N - 1, 2 * N - 1, CUFFT_C2C));

	bool h_to_be_optimized = false;
	bool h_to_exit = true;
	bool *dev_to_be_optimized;
	bool *dev_to_exit;
	bool *dev_mask;
	bool *dev_grad_bool_to_add;
	bool *dev_grad_bool_to_sub;
	bool *h_check_optimal = (bool *)malloc(sizeof(bool));
	bool *h_check_optimal_globally = (bool *)malloc(sizeof(bool));
	cuComplex *dev_gamma_array;
	cuComplex *dev_solution;
	cuComplex *dev_z;
	cuComplex *dev_new_z_extended;
	cuComplex *h_gamma_array = (cuComplex *)malloc((2 * N - 1) * (2 * N - 1) * sizeof(cuComplex));
	float *dev_max_of_indicators;
	float *dev_indicators;
	float *dev_gradient;
	float *dev_intensities;
	float *dev_squared_norms_of_indicators;
	float *dev_actual_residual;
	float *dev_required_intensity_distribution;
	float *dev_min_intensity_dist_required;
	float *dev_w;
	float *dev_b;
	float *dev_sum_of_indicators;
	float *h_required_intensity_distribution;
	float *dev_max_intensity_specified_zone;
	float h_sum_of_indicators = 0.f;
	float epsilon_abs_ID = 0.001f;
	float delta_abs_ID = 0.001f;
	float tolerance = 0.00009f; 	//Tolerance of GMRES
	unsigned int spread = N / 100; 	//Spread of exclusion zone
	unsigned int h_i_line = N / 200;
	unsigned int *h_required_intensity_distribution_indeces;
	unsigned int *dev_required_intensity_distribution_indeces;
	unsigned int GMRES_n = 0;
	unsigned int *dev_check_sum;
	unsigned int h_check_sum = 0;
	unsigned int discrete_gradient = 0;
	unsigned int suboptimizations = 0;
	unsigned int h_distribution_size = 0;

	cudacall(cudaSetDevice(0));

	cudacall(cudaMalloc((void**)&dev_gamma_array, (2 * N - 1) * (2 * N - 1) * sizeof(cuComplex)));
//	get_gamma_array((cuComplex **)&dev_gamma_array, (cufftHandle)plan);
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


//	cudacall(cudaMemcpy(h_gamma_array, dev_gamma_array, (2 * N - 1) * (2 * N - 1) * sizeof(cuComplex), cudaMemcpyDeviceToHost));

	cudacall(cudaMalloc((void**)&dev_to_exit, sizeof(bool)));
	cudacall(cudaMalloc((void**)&dev_mask, N * N * sizeof(bool)));
	cudacall(cudaMalloc((void**)&dev_grad_bool_to_add, N * N * sizeof(bool)));
	cudacall(cudaMalloc((void**)&dev_grad_bool_to_sub, N * N * sizeof(bool)));
	cudacall(cudaMalloc((void**)&dev_solution, N * N * sizeof(cuComplex)));
	cudacall(cudaMalloc((void**)&dev_z, N * N * sizeof(cuComplex)));
	cudacall(cudaMalloc((void**)&dev_check_sum, sizeof(unsigned int)));
	cudacall(cudaMalloc((void**)&dev_gradient, N * N * sizeof(float)));
	cudacall(cudaMalloc((void**)&dev_intensities, N * N * sizeof(float)));
	cudacall(cudaMalloc((void**)&dev_squared_norms_of_indicators, 4 * sizeof(float)));
	cudacall(cudaMalloc((void**)&dev_max_intensity_specified_zone, THREADS_PER_BLOCK * THREADS_PER_BLOCK * sizeof(float)));
	cudacall(cudaMalloc((void**)&dev_min_intensity_dist_required , THREADS_PER_BLOCK * THREADS_PER_BLOCK * sizeof(float)));
	cudacall(cudaMalloc((void**)&dev_sum_of_indicators, sizeof(float)));
	cudacall(cudaMalloc((void**)&dev_w, 2 * sizeof(float)));
	cudacall(cudaMalloc((void**)&dev_b, 2 * sizeof(float)));
	cudacall(cudaMalloc((void**)&dev_max_of_indicators, sizeof(float)));
	cudacall(cudaMalloc((void**)&dev_to_be_optimized, sizeof(bool)));

	init_mask <<< N * N / 512, 512 >>> ((bool *)dev_mask);
	cudacheckSYN();

	init_x0_kernel <<< blocks, threads >>> ((cuComplex *)dev_solution);
	cudacheckSYN();
	
	//set area
	std::ifstream myfile("/home/linux/Project/data/given_distribution.txt");
	if (myfile.is_open())
	{
		getline(myfile, line);
		h_distribution_size = std::atoi(line.c_str());
		h_required_intensity_distribution = (float *)malloc(h_distribution_size * sizeof(float));
		h_required_intensity_distribution_indeces = (unsigned int *)malloc(h_distribution_size * sizeof(unsigned int));
		unsigned int index = 0;
		while ( getline(myfile, line) )
		{
			std::string float_part, uns_int_part;
			for (unsigned int ii=0; ii < 14; ii ++)
			{
				if(ii < 8)
				{
					float_part[ii] = line[ii];
				}
				else
				{
					if(ii > 8)
					{
						uns_int_part[ii - 9] = line[ii];
					}
				}
			}
			float_part[8] = uns_int_part[14];
			h_required_intensity_distribution[index] = atof(float_part.c_str());
			h_required_intensity_distribution_indeces[index] = atoi(uns_int_part.c_str());
			index++;
		}
		myfile.close();
	}
	else fprintf(stderr, "Unable to open file");

	cudacall(cudaMalloc((void**)&dev_required_intensity_distribution, h_distribution_size * sizeof(float)));
	cudacall(cudaMalloc((void**)&dev_indicators                     , h_distribution_size * sizeof(float)));
	cudacall(cudaMalloc((void**)&dev_required_intensity_distribution_indeces, h_distribution_size * sizeof(unsigned int)));
	cudacall(cudaMemcpy(dev_required_intensity_distribution, h_required_intensity_distribution, h_distribution_size * sizeof(float), cudaMemcpyHostToDevice));
	cudacall(cudaMemcpy(dev_required_intensity_distribution_indeces, h_required_intensity_distribution_indeces, h_distribution_size * sizeof(float), cudaMemcpyHostToDevice));

	do
	{
		h_check_sum = 0;

		init_x0_kernel <<< blocks, threads >>> ((cuComplex *)dev_solution);
		cudacheckSYN();

		FFT_GMRES_with_CUDA_extended_with_host_memory_single_debugged((const cuComplex *)dev_gamma_array, (const bool *)dev_mask, (cuComplex *)dev_solution, (float **)&dev_actual_residual, (unsigned int *)&GMRES_n, (cufftHandle)plan, (cublasHandle_t *)&handle, tolerance, (bool)0, 0, 1000);
		cudacall(cudaFree((float *)dev_actual_residual));
		printf("%i\t%i\n", GMRES_n, discrete_gradient);

	//Begin: Find optimization ID
		intensity_kernel <<< N * N / 512, 512 >>> ((cuComplex *)dev_solution, (float *)dev_intensities);
		cudacheckSYN();

		get_max_intensity_specified_zone_v1((unsigned int *)dev_required_intensity_distribution_indeces, (float *)dev_intensities, (float *)dev_max_intensity_specified_zone, h_distribution_size, blocks, threads);
		get_min_required_intensity_specified_zone_v1((float *)dev_required_intensity_distribution, (unsigned int *)dev_required_intensity_distribution_indeces, (float *)dev_min_intensity_dist_required, h_distribution_size, blocks, threads);

		init_w_b_kernel <<< 4, 1 >>> ((float *)dev_w, (float *)dev_b, (float *)dev_max_intensity_specified_zone, (float *)dev_min_intensity_dist_required, tolerance);
		cudacheckSYN();

		unsigned int counter_w_b = 0;
		float h_w, h_b;
		do
		{
			sum_squares_reduce_w_b_kernel <<< N * N / 512, 512 >>> ((float *)dev_w, (float *)dev_b, (float *)dev_required_intensity_distribution, (unsigned int *)dev_required_intensity_distribution_indeces, (float *)dev_intensities, (float *)dev_squared_norms_of_indicators, h_distribution_size);
			cudacheckSYN();

			min_norm_reduce_w_b_kernel <<< 1, 4 >>> ((float *)dev_squared_norms_of_indicators, (float *)dev_w, (float *)dev_b, (bool *)dev_to_exit, tolerance);
			cudacheckSYN();
		
			cudacall(cudaMemcpy(&h_to_exit, dev_to_exit, sizeof(bool), cudaMemcpyDeviceToHost));
			cudacall(cudaMemcpy(&h_w, dev_w, sizeof(float), cudaMemcpyDeviceToHost));
			cudacall(cudaMemcpy(&h_b, dev_b, sizeof(float), cudaMemcpyDeviceToHost));

			fprintf(stderr, "Searching w = %f and b = %f...iteration number = %i\n", h_w, h_b, counter_w_b++);
		}
		while((!h_to_exit) || (counter_w_b < 100));

		indicators_kernel <<< N * N / 512, 512 >>> ((float *)dev_w, (float *)dev_b, (float *)dev_required_intensity_distribution, (unsigned int *)dev_required_intensity_distribution_indeces, (float *)dev_intensities,  (float *)dev_indicators, h_distribution_size);
		cudacheckSYN();
//dev_indicators <---> h_distribution_size
		get_max_indicator_v1((float *)dev_indicators, (float *)dev_max_of_indicators, h_distribution_size, blocks, threads);
	//End: Find optimization ID

		cudacall(cudaMemset((bool *)dev_grad_bool_to_add, true, N * N * sizeof(bool)));
		cudacall(cudaMemset((bool *)dev_grad_bool_to_sub, true, N * N * sizeof(bool)));

		for(unsigned int counter_of_required_distribution = 0; counter_of_required_distribution < h_distribution_size; counter_of_required_distribution++) // distribution_size times
		{
			cudacall(cudaMemset((bool *)dev_to_be_optimized, false, sizeof(bool)));

			compare_abs_indicators_kernel <<< 2, 1 >>> ((float *)dev_indicators + counter_of_required_distribution, (float *)dev_max_of_indicators, (bool *)dev_to_be_optimized, epsilon_abs_ID * (discrete_gradient+1), delta_abs_ID);
			cudacheckSYN();

			cudacall(cudaMemcpy(&h_to_be_optimized, dev_to_be_optimized, sizeof(bool), cudaMemcpyDeviceToHost));

			if (h_to_be_optimized)
			{
				init_x0_kernel <<< blocks, threads >>> ((cuComplex *)dev_z);
				cudacheckSYN();

				FFT_GMRES_with_CUDA_extended_with_host_memory_single_debugged((const cuComplex *)dev_gamma_array, (const bool *)dev_mask, (cuComplex *)dev_z, (float **)&dev_actual_residual, (unsigned int *)&GMRES_n, (cufftHandle)plan, (cublasHandle_t *)&handle, tolerance, (bool)1, *(h_required_intensity_distribution_indeces + counter_of_required_distribution), 1000);
				cudacall(cudaFree((float *)dev_actual_residual));

				G_x_fft_matvec((cuComplex *)dev_gamma_array, (bool *)dev_mask, (cuComplex *)dev_z, (cuComplex **)&dev_new_z_extended, (cufftHandle)plan, (bool)1);

				_2D_to_1D_kernel_to_compute_gradient <<< blocks, threads >>> ((cuComplex *)dev_solution, (cuComplex *)dev_new_z_extended, (float *)dev_gradient, (unsigned int *)dev_required_intensity_distribution_indeces + counter_of_required_distribution);
				cudacheckSYN();

				cudacall(cudaFree((cuComplex *)dev_new_z_extended));
		
				dev_grad_bool_indicated_kernel <<< N * N / 512, 512 >>> ((float *)dev_indicators + counter_of_required_distribution, (bool *)dev_grad_bool_to_add, (bool *)dev_grad_bool_to_sub, (float *)dev_gradient);
				cudacheckSYN();
			}

			fprintf(stderr, "required distribution considering = %i\n", counter_of_required_distribution);
		}

		suboptimizations = 0;
		while (h_check_sum != N * N)
		{
			cudacall(cudaMemset((unsigned int *)dev_check_sum, 0, sizeof(unsigned int)));
			get_neighbour_indices_for_gradient_multiple_maximums_v1 <<< blocks, threads >>> ((bool *)dev_mask, (bool *)dev_grad_bool_to_add, (bool *)dev_grad_bool_to_sub, (unsigned int *)dev_check_sum, h_i_line, spread);
			cudacheckSYN();

			cudacall(cudaMemcpy(&h_check_sum, dev_check_sum, sizeof(unsigned int), cudaMemcpyDeviceToHost));

			fprintf(stderr, "dev_check_sum = %i\n", h_check_sum);
			suboptimizations ++;
		}

		init_x0_kernel <<< blocks, threads >>> ((cuComplex *)dev_solution);
		cudacheckSYN();

		FFT_GMRES_with_CUDA_extended_with_host_memory_single_debugged((const cuComplex *)dev_gamma_array, (const bool *)dev_mask, (cuComplex *)dev_solution, (float **)&dev_actual_residual, (unsigned int *)&GMRES_n, (cufftHandle)plan, (cublasHandle_t *)&handle, tolerance, (bool)0, 0, 1000);

		cudacall(cudaFree((float *)dev_actual_residual));
		printf("%i\t%i\n", GMRES_n, discrete_gradient);

//sum_of_indicators -> minimize
		sum_indicators_squared_reduce_kernel <<< N * N / 512, 512 >>> ((float *)dev_indicators, (float *)dev_sum_of_indicators, h_distribution_size);
		cudacheckSYN();
		
		cudacall(cudaMemcpy(&h_sum_of_indicators, dev_sum_of_indicators, sizeof(float), cudaMemcpyDeviceToHost));		

		saveGPUrealtxt_discrete_gradient((bool *)dev_mask, (cuComplex *)dev_solution, h_sum_of_indicators, (unsigned int)discrete_gradient);

		discrete_gradient++;
		printf("suboptimizations = %i\n\n\n", suboptimizations);
	}
	while((h_check_sum < N * N) || (discrete_gradient < 100));

	cudacall(cudaFree((bool *)dev_mask));
	cudacall(cudaFree((bool *)dev_grad_bool_to_add));
	cudacall(cudaFree((bool *)dev_grad_bool_to_sub));
	cudacall(cudaFree((bool *)dev_to_be_optimized));
	cudacall(cudaFree((bool *)dev_to_exit));
	cudacall(cudaFree((float *)dev_w));
	cudacall(cudaFree((float *)dev_b));
	cudacall(cudaFree((float *)dev_min_intensity_dist_required));
	cudacall(cudaFree((float *)dev_required_intensity_distribution));
	cudacall(cudaFree((float *)dev_gradient));
	cudacall(cudaFree((float *)dev_intensities));
	cudacall(cudaFree((float *)dev_max_intensity_specified_zone));
	cudacall(cudaFree((float *)dev_max_of_indicators));
	cudacall(cudaFree((float *)dev_squared_norms_of_indicators));
	cudacall(cudaFree((float *)dev_indicators));
	cudacall(cudaFree((float *)dev_sum_of_indicators));
	cudacall(cudaFree((cuComplex *)dev_z));
	cudacall(cudaFree((cuComplex *)dev_solution));
	cudacall(cudaFree((cuComplex *)dev_gamma_array));
	cudacall(cudaFree((unsigned int *)dev_required_intensity_distribution_indeces));
	cudacall(cudaFree((unsigned int *)dev_check_sum));
	free((bool *)h_check_optimal);
	free((bool *)h_check_optimal_globally);
	free((unsigned int *)h_required_intensity_distribution_indeces);
	free((cuComplex *)h_gamma_array);
	cufftcall(cufftDestroy(plan));
	cublascall(cublasDestroy_v2(handle));
}
