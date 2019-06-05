void additive_find_suboptimal_mask(bool **dev_mask, unsigned int **dev_neighbour_optimal, float **dev_intensity_optimal, float **dev_intensity_global, float **dev_intensity_global_global, cuComplex **dev_solution, cuComplex **dev_gamma_array, cufftHandle *plan, bool **h_mask, cuComplex **h_gamma_array, const float tolerance);

void subtractive_find_suboptimal_mask(bool **dev_mask, unsigned int **dev_neighbour_optimal, float **dev_intensity_optimal, float **dev_intensity_global, float **dev_intensity_global_global, cuComplex **dev_solution, cuComplex **dev_gamma_array, cufftHandle *plan, bool **h_mask, cuComplex **h_gamma_array, const float tolerance);

void greedy_numerical_method_v1()
{
	dim3 blocks(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
	dim3 threads(Q, Q);

	time_t clock_time = clock();
	bool *h_check_optimal_globally = (bool *)malloc(sizeof(bool));
	
	cufftHandle plan;
	cufftcall(cufftPlan2d(&plan, 2 * N - 1, 2 * N - 1, CUFFT_C2C));

	bool *dev_mask;
	bool *dev_old_mask;
	bool *h_mask = (bool *)malloc(N * N * sizeof(bool));
	bool *dev_check_optimal;
	bool *dev_check_optimal_globally;
	cuComplex *dev_gamma_array;
	cuComplex *dev_solution;
	cuComplex *h_gamma_array = (cuComplex *)malloc((2 * N - 1) * (2 * N - 1) * sizeof(cuComplex));
	unsigned int *dev_neighbour_optimal;
	unsigned int optimization_number_additive = 0;
	unsigned int optimization_number_subtractive = 0;
	unsigned int h_number_of_differences = 0;
	unsigned int *dev_number_of_differences;
	
	float *dev_intensity_optimal;
	float *dev_intensity_global;
	float *dev_intensity_global_global;
	float tolerance = 0.00001f;

	cudacall(cudaSetDevice(0));

	get_gamma_array((cuComplex **)&dev_gamma_array, (cufftHandle)plan);	
	
	cudacall(cudaMemcpy(h_gamma_array, dev_gamma_array, (2 * N - 1) * (2 * N - 1) * sizeof(cuComplex), cudaMemcpyDeviceToHost));

	cudacall(cudaMalloc((void**)&dev_mask, N * N * sizeof(bool)));

/*
	if(0)
	{
		init_mask <<< N * N / 512, 512 >>> ((bool *)dev_mask);
		cudacheckSYN();

		cudacall(cudaMemcpy(h_mask, dev_mask, N * N * sizeof(bool), cudaMemcpyDeviceToHost));
	}
	else
	{
		std::string line;
		std::ifstream myfile ("/home/linux/Project/data/lens.txt");
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

		optimization_number_additive = 1482600;
		optimization_number_subtractive = 0;

		cudacall(cudaMemcpy(dev_mask, h_mask, N * N * sizeof(bool), cudaMemcpyHostToDevice));
	}
*/
	init_mask <<< N * N / 512, 512 >>> ((bool *)dev_mask);
	cudacheckSYN();

	cudacall(cudaMalloc((void**)&dev_solution, N * N * sizeof(cuComplex)));
	init_x0_kernel <<< blocks, threads >>> ((cuComplex *)dev_solution);
	cudacheckSYN();


	cudacall(cudaMalloc((void**)&dev_intensity_optimal, sizeof(float)));
	cudacall(cudaMalloc((void**)&dev_intensity_global, sizeof(float)));
	cudacall(cudaMalloc((void**)&dev_intensity_global_global, sizeof(float)));
	cudacall(cudaMalloc((void**)&dev_neighbour_optimal, sizeof(unsigned int)));
	cudacall(cudaMalloc((void**)&dev_check_optimal, sizeof(bool)));
	cudacall(cudaMalloc((void**)&dev_check_optimal_globally, sizeof(bool)));

	cudacall(cudaMemset(dev_intensity_optimal, 0, sizeof(float)));
	cudacall(cudaMemset(dev_intensity_global, 0, sizeof(float)));
	cudacall(cudaMemset(dev_intensity_global_global, 0, sizeof(float)));
	
	do
	{
		unsigned int internal_add_i = 0;
		do
		{
			additive_find_suboptimal_mask((bool **)&dev_mask, (unsigned int **)&dev_neighbour_optimal, (float **)&dev_intensity_optimal, (float **)&dev_intensity_global, (float **)&dev_intensity_global_global, (cuComplex **)&dev_solution, (cuComplex **)&dev_gamma_array, (cufftHandle *)&plan, (bool **)&h_mask, (cuComplex **)&h_gamma_array, tolerance);

			cudacall(cudaMalloc((void**)&dev_check_optimal, sizeof(bool)));

			check_maximum_kernel <<< 1, 1 >>> ((float *)dev_intensity_global, (float *)dev_intensity_optimal, (bool *)dev_check_optimal);
			cudacheckSYN();

			if (optimization_number_additive > 1)
			{
				cudacall(cudaMalloc((void**)&dev_old_mask, N * N * sizeof(bool)));
				cudacall(cudaMemcpy(dev_old_mask, dev_mask, N * N * sizeof(bool), cudaMemcpyDeviceToDevice));
			}
	
			get_suboptimal_mask_kernel <<< 1, 1 >>> ((bool *)dev_mask, (unsigned int *)dev_neighbour_optimal);

			fprintf(stderr, "Finished: (additive) (%u, %u)\n\n", optimization_number_additive, optimization_number_subtractive);

			if (optimization_number_additive > 1)
			{
				cudacall(cudaMalloc((void**)&dev_number_of_differences, sizeof(unsigned int)));
				cudacall(cudaMemset(dev_number_of_differences, 0, sizeof(unsigned int)));

				check_mask_difference_kernel <<< N * N / 512, 512 >>> ((bool *)dev_old_mask, (bool *)dev_mask, (unsigned int *)dev_number_of_differences);
				cudacheckSYN();

				cudacall(cudaMemcpy(&h_number_of_differences, dev_number_of_differences, sizeof(unsigned int), cudaMemcpyDeviceToHost));

				cudacall(cudaFree((bool *)dev_old_mask));
				cudacall(cudaFree((unsigned int *)dev_number_of_differences));
				if (!h_number_of_differences)
				{
					break;
				}
			}
		
			cudacall(cudaMemcpy(h_mask, dev_mask, N * N * sizeof(bool), cudaMemcpyDeviceToHost));

			saveGPUrealtxt((bool *)h_mask, (cuComplex *)dev_solution, (float *)dev_intensity_global, (unsigned int)optimization_number_additive++ + optimization_number_subtractive);

			cudacheckSYN();
			internal_add_i ++;
		}
		while(1);

		unsigned int internal_sub_i = 0;
		do
		{


			subtractive_find_suboptimal_mask((bool **)&dev_mask, (unsigned int **)&dev_neighbour_optimal, (float **)&dev_intensity_optimal, (float **)&dev_intensity_global, (float **)&dev_intensity_global_global, (cuComplex **)&dev_solution, (cuComplex **)&dev_gamma_array, (cufftHandle *)&plan, (bool **)&h_mask, (cuComplex **)&h_gamma_array, tolerance);

			cudacall(cudaMalloc((void**)&dev_check_optimal, sizeof(bool)));

			check_maximum_kernel <<< 1, 1 >>> ((float *)dev_intensity_global, (float *)dev_intensity_optimal, (bool *)dev_check_optimal);
			cudacheckSYN();

			if (optimization_number_additive > 1)
			{
				cudacall(cudaMalloc((void**)&dev_old_mask, N * N * sizeof(bool)));
				cudacall(cudaMemcpy(dev_old_mask, dev_mask, N * N * sizeof(bool), cudaMemcpyDeviceToDevice));
			}
	
			get_suboptimal_mask_kernel <<< 1, 1 >>> ((bool *)dev_mask, (unsigned int *)dev_neighbour_optimal);

			fprintf(stderr, "Finished: (subtractive) (%u, %u)\n\n", optimization_number_additive, optimization_number_subtractive);

			{
				cudacall(cudaMalloc((void**)&dev_number_of_differences, sizeof(unsigned int)));
				cudacall(cudaMemset(dev_number_of_differences, 0, sizeof(unsigned int)));

				check_mask_difference_kernel <<< N * N / 512, 512 >>> ((bool *)dev_old_mask, (bool *)dev_mask, (unsigned int *)dev_number_of_differences);
				cudacheckSYN();

				cudacall(cudaMemcpy(&h_number_of_differences, dev_number_of_differences, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	
				cudacall(cudaFree((bool *)dev_old_mask));
				cudacall(cudaFree((unsigned int *)dev_number_of_differences));

				if ((!h_number_of_differences) || (10 * optimization_number_subtractive > 9 * optimization_number_additive) && (10 * internal_sub_i > 9 * internal_add_i))
				{
					break;
				}
			}

			cudacall(cudaMemcpy(h_mask, dev_mask, N * N * sizeof(bool), cudaMemcpyDeviceToHost));

			saveGPUrealtxt((bool *)h_mask, (cuComplex *)dev_solution, (float *)dev_intensity_global, (unsigned int)optimization_number_subtractive++ + optimization_number_additive);

			cudacheckSYN();

			internal_sub_i ++;
		}
		while(1);

		//check_maximum_kernel <<< 1, 1 >>> ((float *)dev_intensity_global_global, (float *)dev_intensity_global, (bool *)dev_check_optimal_globally);
		//cudacheckSYN();
	
		*h_check_optimal_globally = false;//cudacall(cudaMemcpy(h_check_optimal_globally, dev_check_optimal_globally, sizeof(bool), cudaMemcpyDeviceToHost));
	}
	while(!*h_check_optimal_globally);

	cudacall(cudaFree((float *)dev_intensity_optimal));
	cudacall(cudaFree((float *)dev_intensity_global));
	cudacall(cudaFree((float *)dev_intensity_global_global));
	cudacall(cudaFree((unsigned int *)dev_neighbour_optimal));
	cudacall(cudaFree((bool *)dev_mask));
	cudacall(cudaFree((bool *)dev_check_optimal));
	cudacall(cudaFree((bool *)dev_check_optimal_globally));
	cudacall(cudaFree((cuComplex *)dev_solution));
	cudacall(cudaFree((cuComplex *)dev_gamma_array));
	free((bool *)h_mask);
	free((bool *)h_check_optimal_globally);
	free((cuComplex *)h_gamma_array);
	cufftcall(cufftDestroy(plan));
}




void additive_find_suboptimal_mask(bool **dev_mask, unsigned int **dev_neighbour_optimal, float **dev_intensity_optimal, float **dev_intensity_global, float **dev_intensity_global_global, cuComplex **dev_solution, cuComplex **dev_gamma_array, cufftHandle *plan, bool **h_mask, cuComplex **h_gamma_array, const float tolerance)
{
	dim3 blocks(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
	dim3 threads(Q, Q);

	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));

	unsigned int GMRES_n = 0;
	unsigned int Greedy_i = 0;
	bool *dev_stop;
	bool *stop = (bool *)malloc(sizeof(bool));

	float *dev_actual_residual;
	float *dev_intensity;
	unsigned int *dev_neighbour_index;
	unsigned int *dev_neighbor_indeces;	

	cudacall(cudaMalloc((void**)&dev_intensity, (N * N /512) * sizeof(float)));
	cudacall(cudaMalloc((void**)&dev_neighbor_indeces, N * N * sizeof(unsigned int)));
	cudacall(cudaMalloc((void**)&dev_neighbour_index, sizeof(unsigned int)));
	cudacall(cudaMalloc((void**)&dev_stop, sizeof(bool)));
	
	cudacall(cudaMemset(dev_neighbour_index, 0, sizeof(unsigned int)));
	cudacall(cudaMemset(dev_stop, 0, sizeof(bool)));
	cudacall(cudaMemset(*dev_neighbour_optimal, 0, sizeof(unsigned int)));

	get_neighbour_indices <<< blocks, threads >>> ((unsigned int *)dev_neighbor_indeces, (bool *)*dev_mask);
	cudacheckSYN();

	do
	{
		if ((Greedy_i+1) % 300 == 30)
		{

			unsigned int *h_neighbour_index = (unsigned int *)malloc(sizeof(unsigned int));
			unsigned int *h_neighbour_optimal = (unsigned int *)malloc(sizeof(unsigned int));
			float *h_intensity_global_global = (float *)malloc(sizeof(float));
			float *h_intensity_global = (float *)malloc(sizeof(float));
			float *h_intensity_optimal = (float *)malloc(sizeof(float));

			fprintf(stderr, ".");
			cudacall(cudaMemcpy(h_neighbour_index,   dev_neighbour_index,    sizeof(unsigned int), cudaMemcpyDeviceToHost));
			cudacall(cudaMemcpy(h_neighbour_optimal, *dev_neighbour_optimal, sizeof(unsigned int), cudaMemcpyDeviceToHost));
			cudacall(cudaMemcpy(h_intensity_optimal, *dev_intensity_optimal, sizeof(float),        cudaMemcpyDeviceToHost));
			cudacall(cudaMemcpy(h_intensity_global,  *dev_intensity_global,  sizeof(float),        cudaMemcpyDeviceToHost));
			cudacall(cudaMemcpy(h_intensity_global_global,  *dev_intensity_global_global,  sizeof(float),        cudaMemcpyDeviceToHost));

			fprintf(stderr, "%i\n", *h_neighbour_index);

			cudaDeviceReset();
			cudacall(cudaSetDevice(0));
			cublascall(cublasCreate_v2(&handle));
			cufftcall(cufftPlan2d(plan, 2 * N - 1, 2 * N - 1, CUFFT_C2C));
	
			fprintf(stderr, ".");
			cudacall(cudaMalloc((void**)dev_mask, N * N * sizeof(bool)));
			cudacall(cudaMalloc((void**)dev_gamma_array, (2 * N - 1) * (2 * N - 1) * sizeof(cuComplex)));
			cudacall(cudaMemcpy(*dev_gamma_array, *h_gamma_array, (2 * N - 1) * (2 * N - 1) * sizeof(cuComplex), cudaMemcpyHostToDevice));
			cudacall(cudaMemcpy(*dev_mask, *h_mask, N * N * sizeof(bool), cudaMemcpyHostToDevice));

			cudacall(cudaMalloc((void**)dev_solution, N * N * sizeof(cuComplex)));
			init_x0_kernel <<< blocks, threads >>> ((cuComplex *)*dev_solution);
			cudacheckSYN();

			fprintf(stderr, ".");
			cudacall(cudaMalloc((void**)&dev_intensity       , (N * N / 512) * sizeof(float)));
			cudacall(cudaMalloc((void**)dev_intensity_global ,                 sizeof(float)));
			cudacall(cudaMalloc((void**)dev_intensity_global_global, sizeof(float)));
			cudacall(cudaMalloc((void**)dev_intensity_optimal,                 sizeof(float)));
			cudacall(cudaMalloc((void**)&dev_neighbour_index ,                 sizeof(unsigned int)));
			cudacall(cudaMalloc((void**)dev_neighbour_optimal,                 sizeof(unsigned int)));
			cudacall(cudaMalloc((void**)&dev_neighbor_indeces,         N * N * sizeof(unsigned int)));
			cudacall(cudaMalloc((void**)&dev_stop            ,                 sizeof(bool)));

			get_neighbour_indices <<< blocks, threads >>> ((unsigned int *)dev_neighbor_indeces, (bool *)*dev_mask);
			cudacheckSYN();

			fprintf(stderr, ".");
			cudacall(cudaMemset(*dev_intensity_optimal, 0, sizeof(float)));
			cudacall(cudaMemset(dev_stop, 0, sizeof(bool)));

			cudacall(cudaMemcpy( dev_neighbour_index,   h_neighbour_index,   sizeof(unsigned int), cudaMemcpyHostToDevice));
			cudacall(cudaMemcpy(*dev_neighbour_optimal, h_neighbour_optimal, sizeof(unsigned int), cudaMemcpyHostToDevice));
			cudacall(cudaMemcpy(*dev_intensity_optimal, h_intensity_optimal, sizeof(float),        cudaMemcpyHostToDevice));
			cudacall(cudaMemcpy(*dev_intensity_global,  h_intensity_global,  sizeof(float),        cudaMemcpyHostToDevice));
			cudacall(cudaMemcpy(*dev_intensity_global_global,  h_intensity_global_global,  sizeof(float),        cudaMemcpyHostToDevice));
			fprintf(stderr, ".");

			free((unsigned int *)h_neighbour_optimal);
			free((unsigned int *)h_neighbour_index);
			free((float *)h_intensity_optimal);
			free((float *)h_intensity_global);
			free((float *)h_intensity_global_global);
		}

		get_neighbour_next <<< 1, 1 >>> ((unsigned int *)dev_neighbour_index, (unsigned int *)dev_neighbor_indeces, (bool *)*dev_mask, (bool *)dev_stop);
		cudacheckSYN();

		cudacall(cudaMemcpy(stop, dev_stop, sizeof(bool), cudaMemcpyDeviceToHost));
		if (*stop) break;

		FFT_GMRES_with_CUDA_extended_with_host_memory_single_debugged((const cuComplex *)*dev_gamma_array, (const bool *)*dev_mask, (cuComplex *)*dev_solution, (float **)&dev_actual_residual, (unsigned int *)&GMRES_n, (cufftHandle)*plan, (cublasHandle_t *)&handle, tolerance, (bool)0, 0, 100);

		printf("%i\t%i\n", GMRES_n, Greedy_i++);

//		saveGPUrealtxt_C(*dev_solution, "data/field_lens.txt", N*N);
//		fprintf(stderr, "DELETE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
//		break;

		target_function_v1((cuComplex *)*dev_solution, (float *)dev_intensity);

		target_function_compare <<< 1, 1 >>> ((float *)dev_intensity, (float *)*dev_intensity_optimal, (unsigned int *)dev_neighbour_index, (unsigned int *)*dev_neighbour_optimal);
		cudacheckSYN();

		cudacall(cudaFree((float *)dev_actual_residual));
	}
	while(1);

	cudacall(cudaFree((bool *)dev_stop));
	cudacall(cudaFree((float *)dev_intensity));
	cudacall(cudaFree((unsigned int *)dev_neighbor_indeces));
	cudacall(cudaFree((unsigned int *)dev_neighbour_index));

	free((bool *)stop);
	cublascall(cublasDestroy_v2(handle));
}

void subtractive_find_suboptimal_mask(bool **dev_mask, unsigned int **dev_neighbour_optimal, float **dev_intensity_optimal, float **dev_intensity_global, float **dev_intensity_global_global, cuComplex **dev_solution, cuComplex **dev_gamma_array, cufftHandle *plan, bool **h_mask, cuComplex **h_gamma_array, const float tolerance)
{
	dim3 blocks(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
	dim3 threads(Q, Q);

	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));

	unsigned int GMRES_n = 0;
	unsigned int Greedy_i = 0;
	bool *dev_stop;
	bool *stop = (bool *)malloc(sizeof(bool));

	float *dev_actual_residual;
	float *dev_intensity;
	unsigned int *dev_neighbour_index;

	cudacall(cudaMalloc((void**)&dev_intensity      , (N * N /512) * sizeof(float)       ));
	cudacall(cudaMalloc((void**)&dev_neighbour_index,                sizeof(unsigned int)));
	cudacall(cudaMalloc((void**)&dev_stop           ,                sizeof(bool)        ));
	
	cudacall(cudaMemset(dev_neighbour_index   , 0, sizeof(unsigned int)));
	cudacall(cudaMemset(*dev_neighbour_optimal, 0, sizeof(unsigned int)));
	cudacall(cudaMemset(dev_stop              , 0, sizeof(bool)        ));

	do
	{
		if ((Greedy_i + 1) % 300 == 30)
		{

			unsigned int *h_neighbour_index = (unsigned int *)malloc(sizeof(unsigned int));
			unsigned int *h_neighbour_optimal = (unsigned int *)malloc(sizeof(unsigned int));
			float *h_intensity_global_global = (float *)malloc(sizeof(float));
			float *h_intensity_global = (float *)malloc(sizeof(float));
			float *h_intensity_optimal = (float *)malloc(sizeof(float));

			fprintf(stderr, ".");
			cudacall(cudaMemcpy(h_neighbour_index,    dev_neighbour_index  ,    sizeof(unsigned int), cudaMemcpyDeviceToHost));
			cudacall(cudaMemcpy(h_neighbour_optimal, *dev_neighbour_optimal, sizeof(unsigned int), cudaMemcpyDeviceToHost));
			cudacall(cudaMemcpy(h_intensity_optimal, *dev_intensity_optimal, sizeof(float),        cudaMemcpyDeviceToHost));
			cudacall(cudaMemcpy(h_intensity_global,  *dev_intensity_global ,  sizeof(float),        cudaMemcpyDeviceToHost));
			cudacall(cudaMemcpy(h_intensity_global_global,  *dev_intensity_global_global,  sizeof(float),        cudaMemcpyDeviceToHost));

			fprintf(stderr, "%i\n", *h_neighbour_index);

			cudaDeviceReset();
			cudacall(cudaSetDevice(0));
			cublascall(cublasCreate_v2(&handle));
			cufftcall(cufftPlan2d(plan, 2 * N - 1, 2 * N - 1, CUFFT_C2C));
	
			fprintf(stderr, ".");
			cudacall(cudaMalloc((void**)dev_mask, N * N * sizeof(bool)));
			cudacall(cudaMalloc((void**)dev_gamma_array, (2 * N - 1) * (2 * N - 1) * sizeof(cuComplex)));
			cudacall(cudaMemcpy(*dev_gamma_array, *h_gamma_array, (2 * N - 1) * (2 * N - 1) * sizeof(cuComplex), cudaMemcpyHostToDevice));
			cudacall(cudaMemcpy(*dev_mask, *h_mask, N * N * sizeof(bool), cudaMemcpyHostToDevice));	

			cudacall(cudaMalloc((void**)dev_solution, N * N * sizeof(cuComplex)));
			init_x0_kernel <<< blocks, threads >>> ((cuComplex *)*dev_solution);
			cudacheckSYN();

			fprintf(stderr, ".");
			cudacall(cudaMalloc((void**)&dev_intensity, (N * N / 512) * sizeof(float)));
			cudacall(cudaMalloc((void**)dev_intensity_global, sizeof(float)));
			cudacall(cudaMalloc((void**)dev_intensity_global_global, sizeof(float)));
			cudacall(cudaMalloc((void**)dev_intensity_optimal, sizeof(float)));
			cudacall(cudaMalloc((void**)&dev_neighbour_index, sizeof(unsigned int)));
			cudacall(cudaMalloc((void**)dev_neighbour_optimal, sizeof(unsigned int)));
			cudacall(cudaMalloc((void**)&dev_stop, sizeof(bool)));

			fprintf(stderr, ".");
			cudacall(cudaMemset(*dev_intensity_optimal, 0, sizeof(float)));
			cudacall(cudaMemset(dev_stop, 0, sizeof(bool)));

			cudacall(cudaMemcpy( dev_neighbour_index,   h_neighbour_index,   sizeof(unsigned int), cudaMemcpyHostToDevice));
			cudacall(cudaMemcpy(*dev_neighbour_optimal, h_neighbour_optimal, sizeof(unsigned int), cudaMemcpyHostToDevice));
			cudacall(cudaMemcpy(*dev_intensity_optimal, h_intensity_optimal, sizeof(float),        cudaMemcpyHostToDevice));
			cudacall(cudaMemcpy(*dev_intensity_global,  h_intensity_global,  sizeof(float),        cudaMemcpyHostToDevice));
			cudacall(cudaMemcpy(*dev_intensity_global_global,  h_intensity_global_global,  sizeof(float),        cudaMemcpyHostToDevice));
			fprintf(stderr, ".");

			free((unsigned int *)h_neighbour_optimal);
			free((unsigned int *)h_neighbour_index);
			free((float *)h_intensity_optimal);
			free((float *)h_intensity_global);
			free((float *)h_intensity_global_global);
		}

		get_neighbour_subtractive_next_kernel <<< 1, 1 >>> ((unsigned int *)dev_neighbour_index, (bool *)*dev_mask, (bool *)dev_stop);
		cudacheckSYN();

		cudacall(cudaMemcpy(stop, dev_stop, sizeof(bool), cudaMemcpyDeviceToHost));
		if (*stop) break;
	
		FFT_GMRES_with_CUDA_extended_with_host_memory_single_debugged((const cuComplex *)*dev_gamma_array, (const bool *)*dev_mask, (cuComplex *)*dev_solution, (float **)&dev_actual_residual, (unsigned int *)&GMRES_n, (cufftHandle)*plan, (cublasHandle_t *)&handle, tolerance, (bool)0, 0, 100);

		printf("%i\t%i\n", GMRES_n, Greedy_i++);

		target_function_v1((cuComplex *)*dev_solution, (float *)dev_intensity);

		target_function_compare <<< 1, 1 >>> ((float *)dev_intensity, (float *)*dev_intensity_optimal, (unsigned int *)dev_neighbour_index, (unsigned int *)*dev_neighbour_optimal);
		cudacheckSYN();

		cudacall(cudaFree((float *)dev_actual_residual));
	}
	while(1);

	cudacall(cudaFree((bool *)dev_stop));
	cudacall(cudaFree((float *)dev_intensity));
	cudacall(cudaFree((unsigned int *)dev_neighbour_index));

	free((bool *)stop);
	cublascall(cublasDestroy_v2(handle));
}
