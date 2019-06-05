void FFT_GMRES_with_CUDA_Batched(const cuComplex *dev_gamma_array, const bool *dev_mask, cuComplex *dev_solution, float **dev_actual_residual, unsigned int *GMRES_n, cufftHandle plan, cublasHandle_t *handle_p, const float tolerance, const bool for_gradient, const unsigned int h_index_of_max, bool *h_res_vs_tol_p, unsigned int maxiter)
{
	unsigned int Batch_size = 469145600 / N / N;
	unsigned int i = 0;
	bool h_res_vs_tol = true;

	while ((h_res_vs_tol) && (i < maxiter/Batch_size))
	{
		FFT_GMRES_with_CUDA((const cuComplex *)dev_gamma_array, (const bool *)dev_mask, (cuComplex *)dev_solution, (float **)dev_actual_residual, (unsigned int *)GMRES_n, (cufftHandle)plan, (cublasHandle_t *)handle_p, tolerance, for_gradient, h_index_of_max, (bool *)&h_res_vs_tol, Batch_size);

		saveGPUrealtxt_discrete_gradient((bool *)dev_mask, (cuComplex *)dev_solution, 0.f, (unsigned int)i);
		i ++;
	}
	fprintf(stderr, "h_res_vs_tol = %i\n", h_res_vs_tol ? 1 : 0);
	fprintf(stderr, "Batch_size = %i\n", Batch_size);
	fprintf(stderr, "i = %i\n", i);

	fprintf(stderr, "i = %i\n", maxiter/Batch_size);
	fprintf(stderr, "condition = %i\n", ((h_res_vs_tol) && (i < maxiter/Batch_size)) ? 1 : 0);

	if (h_res_vs_tol && (maxiter % Batch_size > 0))
	{
		FFT_GMRES_with_CUDA((const cuComplex *)dev_gamma_array, (const bool *)dev_mask, (cuComplex *)dev_solution, (float **)dev_actual_residual, (unsigned int *)GMRES_n, (cufftHandle)plan, (cublasHandle_t *)handle_p, tolerance, for_gradient, h_index_of_max, (bool *)h_res_vs_tol, maxiter % Batch_size);
	}

	*h_res_vs_tol_p = h_res_vs_tol;
}
