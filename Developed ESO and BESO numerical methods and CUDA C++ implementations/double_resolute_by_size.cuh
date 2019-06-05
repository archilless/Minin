#include <sstream>

void double_resolute_by_size()
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
	float tolerance = 0.1f;
	float *dev_actual_residual;
	unsigned int GMRES_n = 0;
	unsigned int N_old = N >> 1;
	cuComplex *dev_gamma_array;
	cuComplex *dev_solution;
	cuComplex *dev_solution_old;
	cuComplex *h_solution = (cuComplex *)malloc(N_old * N_old * sizeof(cuComplex));

	cudacall(cudaSetDevice(0));

	cudacall(cudaMalloc((void**)&dev_mask, N * N * sizeof(bool)));
	cudacall(cudaMalloc((void**)&dev_solution, N * N * sizeof(cuComplex)));
	cudacall(cudaMalloc((void**)&dev_solution_old, N_old * N_old * sizeof(cuComplex)));

	std::string line;
	std::ifstream myfile ("data/delete_me.txt");
	if (myfile.is_open())
	{
		unsigned int index = 0;
		while ( getline (myfile,line) )
		{
			std::istringstream in_string_stream(line);

			in_string_stream >> h_solution[index].x >> h_solution[index].y;
			index ++;
		}
		myfile.close();
	}
	else fprintf(stderr, "Unable to open file");

	std::ifstream myfile2 ("data/lens.txt");
	if (myfile2.is_open())
	{
		unsigned int index = 0;
		while ( getline (myfile2,line) )
		{
			h_mask[index++] = (line == "1");
		}
		myfile2.close();
	}
	else fprintf(stderr, "Unable to open file");
	cudacall(cudaMemcpy(dev_mask, h_mask, N * N * sizeof(bool), cudaMemcpyHostToDevice));

	cudacall(cudaMemcpy(dev_solution_old, h_solution, N_old * N_old * sizeof(cuComplex), cudaMemcpyHostToDevice));

	double_expand_cuComplex_kernel <<< dim3(N_old / Q, N_old / Q), dim3(Q, Q) >>> ((const cuComplex *)dev_solution_old, (cuComplex *)dev_solution, N_old);
	cudacheckSYN();

	get_gamma_array((cuComplex **)&dev_gamma_array, (cufftHandle)plan);

	FFT_GMRES_with_CUDA_extended_with_host_memory_single((const cuComplex *)dev_gamma_array, (const bool *)dev_mask, (cuComplex *)dev_solution, (float **)&dev_actual_residual, (unsigned int *)&GMRES_n, (cufftHandle)plan, (cublasHandle_t *)&handle, tolerance, false, 0, (bool *)&h_res_vs_tol, 1000);

	saveGPUrealtxt_C(dev_solution, "data/delete_me_4.txt", N * N);

	cudacall(cudaFree((bool *)dev_mask));
	cudacall(cudaFree((cuComplex *)dev_solution));
	cudacall(cudaFree((cuComplex *)dev_gamma_array));
	cufftcall(cufftDestroy(plan));
	free((bool *)h_mask);
}
