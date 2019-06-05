void launch_GMRES()
{
	dim3 blocks(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
	dim3 threads(Q, Q);
	
	cufftHandle plan;
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	cufftcall(cufftPlan2d(&plan, 2 * N - 1, 2 * N - 1, CUFFT_C2C));

	bool *dev_mask;
	bool *h_mask = (bool *)malloc(N * N * sizeof(bool));
	bool h_res_vs_tol = true;
	cuComplex *dev_gamma_array;
	cuComplex *dev_solution;
	cuComplex *dev_solution_sum_divide;
	float tolerance = 0.00009f;//0.00009f;
	float *dev_actual_residual;
	unsigned int GMRES_n = 0;

	cudacall(cudaSetDevice(0));

	cudacall(cudaMalloc((void**)&dev_mask, N * N * sizeof(bool)));
	cudacall(cudaMalloc((void**)&dev_solution, N * N * sizeof(cuComplex)));
	cudacall(cudaMalloc((void**)&dev_solution_sum_divide, N * N * sizeof(cuComplex)));


	cudacall(cudaMemset(dev_solution_sum_divide, 0.f, N * N * sizeof(cuComplex)));

	std::string line;
	std::ifstream myfile ("data/lens_1024.txt");
	if (myfile.is_open())
	{
		unsigned int index = 0;
		while ( getline (myfile,line) )
		{
			h_mask[index++] = 0;//(line == "1");
		}
		myfile.close();
	}
	else fprintf(stderr, "Unable to open file");
	cudacall(cudaMemcpy(dev_mask, h_mask, N * N * sizeof(bool), cudaMemcpyHostToDevice));

	get_gamma_array((cuComplex **)&dev_gamma_array, (cufftHandle)plan);

	for (unsigned int i = 0; i < 1; i ++)
	{
		init_x0_by_ones_kernel <<< blocks, threads >>> ((cuComplex *)dev_solution);
		cudacheckSYN();
		
		FFT_GMRES_with_CUDA_extended_with_host_memory_single_debugged((const cuComplex *)dev_gamma_array, (const bool *)dev_mask, (cuComplex *)dev_solution, (float **)&dev_actual_residual, (unsigned int *)&GMRES_n, (cufftHandle)plan, (cublasHandle_t *)&handle, tolerance, false, 0, 100);

		add_divide_100_kernel <<< N * N / 512, 512 >>> ((const cuComplex *)dev_solution, (cuComplex *)dev_solution_sum_divide);
		cudacheckSYN();

		fprintf(stderr, "sample_i = %i Finished\n\n\n", i);
	}
	
	fprintf(stderr, "File writing");

	saveGPUrealtxt_C(dev_solution, "data/delete_me_4.txt", N * N);
	//saveGPUrealtxt_C(dev_solution_sum_divide, "data/delete_me_4.txt", N * N);

	cudacall(cudaFree((bool *)dev_mask));
	cudacall(cudaFree((cuComplex *)dev_solution));
	cudacall(cudaFree((cuComplex *)dev_solution_sum_divide));
	cudacall(cudaFree((cuComplex *)dev_gamma_array));
	cufftcall(cufftDestroy(plan));
	free((bool *)h_mask);
}
