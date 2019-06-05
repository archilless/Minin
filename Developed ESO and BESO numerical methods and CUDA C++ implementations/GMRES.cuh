void G_x_fft_matvec(cuComplex *dev_gamma_array, bool *dev_mask, cuComplex *dev_solution, cuComplex **dev_matmul_out_extended, cufftHandle plan, const bool for_gradient);
void w_equal_to_A_v(cuComplex *dev_gamma_array, unsigned int GMRES_i, bool *dev_mask, cuComplex *dev_orthogonal_basis, cuComplex **dev_w, cufftHandle plan, const bool for_gradient);
void get_resized(cuComplex **to_be_resized, dim3 gridsize, dim3 blocksize, unsigned int old_size_i, unsigned int old_size_j, unsigned int new_size_i, unsigned int new_size_j);
void usual_MatMul_CUDA(cublasHandle_t *handle, cuComplex *A, cuComplex *B, cuComplex *C, unsigned int A_size_i, unsigned int A_size_j, unsigned int B_size_j);
void get_new_w_and_H(cuComplex **dev_w, cuComplex *dev_Hj, cuComplex *dev_orthogonal_basis_j);
void get_H_equal_norm_w(cuComplex *dev_Hj, cuComplex *dev_w);
void get_v_equal_w_devided_H(cuComplex *dev_orthogonal_basis_j, cuComplex **dev_w, cuComplex *dev_H_j);
void get_solution();
void get_resized_act_res(float **dev_actual_residual, const unsigned int GMRES_i);


void FFT_GMRES_with_CUDA(const cuComplex *dev_gamma_array, const bool *dev_mask, cuComplex *dev_solution, float **dev_actual_residual, unsigned int *GMRES_n, cufftHandle plan, cublasHandle_t *handle_p, const float tolerance, const bool for_gradient, const unsigned int h_index_of_max, bool *h_res_vs_tol_p, unsigned int maxiter)
{
	dim3 blocks(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
	dim3 blocks_M(THREADS_PER_BLOCK_M, THREADS_PER_BLOCK_M);
	dim3 threads(Q, Q);
	dim3 blocksize_sum_reduce(512);
	dim3 gridsize_sum_reduce(N * N / blocksize_sum_reduce.x);

	bool *dev_res_vs_tol;
	bool h_res_vs_tol = true;
	
	cuComplex *dev_matmul_out_extended;
	cuComplex *dev_residual_vec;
	cuComplex *dev_orthogonal_basis;
	cuComplex *dev_w;
	cuComplex *dev_resized;
	cuComplex *dev_HH;
	dev_HH = NULL;
	cuComplex *dev_Jtotal;
	cuComplex *dev_H_;
	cuComplex *dev_Htemp;
	cuComplex *dev_cc;
	cuComplex *dev_Givens_rotation;

	unsigned int GMRES_i = 0;
//========================================= BEGIN: get_residual_vector =======================================================
	G_x_fft_matvec((cuComplex *)dev_gamma_array, (bool *)dev_mask, (cuComplex *)dev_solution, (cuComplex **)&dev_matmul_out_extended, (cufftHandle)plan, for_gradient);

	cudacall(cudaMalloc((void**)&dev_residual_vec, N * N * sizeof(cuComplex)));

	if (for_gradient)
	{
		_2D_to_1D_compared_for_gradient_Kernel <<< blocks, threads >>> ((bool *)dev_mask, (cuComplex *)dev_solution, (cuComplex*)dev_matmul_out_extended, (cuComplex*)dev_residual_vec, h_index_of_max);
	}
	else
	{
		_2D_to_1D_compared_Kernel <<< blocks, threads >>> ((cuComplex *)dev_solution, (cuComplex*)dev_matmul_out_extended, (cuComplex*)dev_residual_vec);
	}	
	cudacheckSYN();

	cudacall(cudaFree((cuComplex *)dev_matmul_out_extended));
//========================================== END: get_residual_vector =========================================================
	cudacall(cudaMalloc((void**)dev_actual_residual, 10 * sizeof(float)));

	cudacall(cudaMemset(*dev_actual_residual, 0, sizeof(float)));

	sum_squares_reduce_kernel <<< gridsize_sum_reduce, blocksize_sum_reduce >>> ((cuComplex *)dev_residual_vec, (float *)*dev_actual_residual);
	cudacheckSYN();

	sqrt_float_kernel <<< 1, 1 >>> ((float *)*dev_actual_residual);
	cudacheckSYN();
//============================================BEGIN:residual_normalization_kernel=======================================================
	cudacall(cudaMalloc((void**)&dev_orthogonal_basis, N * N * sizeof(cuComplex)));

	residual_normalization_kernel <<< gridsize_sum_reduce, blocksize_sum_reduce >>> ((cuComplex *)dev_residual_vec, (float *)*dev_actual_residual, (cuComplex *)dev_orthogonal_basis);
	cudacheckSYN();

	cudacall(cudaFree((cuComplex *)dev_residual_vec));
//============================================= END:residual_normalization_kernel ==================================================
//============================================= Begin: Condition to iterate ==========================================================	
	cudacall(cudaMalloc((void**)&dev_res_vs_tol, sizeof(bool)));

	residual_vs_tolerance_kernel <<< 1, 1 >>> ((float *)*dev_actual_residual, (bool *)dev_res_vs_tol, tolerance);
	cudacheckSYN();

	cudacall(cudaMemcpy(&h_res_vs_tol, dev_res_vs_tol, sizeof(bool), cudaMemcpyDeviceToHost));

//=============================================== End: Condition to iterate ===========================================================
	if (1)//(h_res_vs_tol)
	{
		cudacall(cudaMalloc((void**)&dev_H_,     2 * sizeof(cuComplex)));
		cudacall(cudaMalloc((void**)&dev_w , N * N * sizeof(cuComplex)));

		w_equal_to_A_v((cuComplex *)dev_gamma_array, (unsigned int) 0, (bool *)dev_mask, (cuComplex *)dev_orthogonal_basis, (cuComplex **)&dev_w, (cufftHandle) plan, for_gradient);

		get_new_w_and_H((cuComplex **)&dev_w, (cuComplex *)dev_H_, (cuComplex *)dev_orthogonal_basis);//Fill Hessenberg m.

		get_H_equal_norm_w((cuComplex *)(dev_H_+ 1), (cuComplex *)dev_w);
	//============================================== BEGIN: Fill Orthogonal Basis matrix ============================================
		get_resized((cuComplex **)&dev_orthogonal_basis, dim3(2, N * N / 512), dim3(1, 512), (unsigned int)1, (unsigned int)N * N, (unsigned int)2, (unsigned int)N * N);

		get_v_equal_w_devided_H((cuComplex *)(dev_orthogonal_basis + N * N), (cuComplex **)&dev_w, (cuComplex *)(dev_H_ + 1));
	//============================================== END: Orthogonal Basis matrix  ==================================================
	//============================================== Begin: Least Squares Step =========================================================
		cudacall(cudaMalloc((void**)&dev_Htemp, 2 * sizeof(cuComplex)));

		dublicate_kernel <<< dim3(2, 1), dim3(1, 1) >>> ((cuComplex *)dev_H_, (cuComplex *)dev_Htemp);
		cudacheckSYN();
	//================================================ END: Least Squares Step =========================================================
	//============================================= BEGIN: Create Givens_Rotation_Matrix ========================================
		cudacall(cudaMalloc((void**)&dev_Givens_rotation, 4 * sizeof(cuComplex)));

		create_Givens_rotation_matrix_kernel <<< dim3(2, 2), dim3(1, 1) >>> ((cuComplex *)dev_Givens_rotation, (cuComplex *)dev_Htemp);
		cudacheckSYN();
	//=============================================== END: Create Givens_Rotation_Matrix ========================================
//		cuComplex *solution_sample = (cuComplex *) malloc(sizeof(cuComplex));
//		cudacall(cudaMemcpy(solution_sample, dev_Givens_rotation, sizeof(cuComplex), cudaMemcpyDeviceToHost));
//		if (isnan(solution_sample->x) || isnan(solution_sample->y)) 
//		{//saveGPUrealtxt_C(dev_Htemp, "check.txt", 2 * 2);//fprintf(stderr, "NAN\n");
//			cuComplex *h_Htemp = (cuComplex *) malloc( N*N * sizeof(cuComplex));
//			cudacall(cudaMemcpy(h_Htemp, dev_w, N*N * sizeof(cuComplex), cudaMemcpyDeviceToHost));
//			//for (int iii=0; iii<N*N; iii++) fprintf(stderr, "%f + %f * j\n ", h_Htemp[iii].x, h_Htemp[iii].y); 
//		}
	//================================================== BEGIN: Jtotal = J*Jtotal =================================================
		cudacall(cudaMalloc((void**)&dev_Jtotal, N * N * sizeof(cuComplex)));		
		dublicate_kernel <<< blocks, threads >>> ((cuComplex *)dev_Givens_rotation, (cuComplex *)dev_Jtotal);
		cudacheckSYN();
	//==================================================== END: Jtotal = J*Jtotal =================================================
	//===================================================== BEGIN: Update residual ======================================================
		next_residual_kernel <<< 1, 1 >>> ((cuComplex *)(dev_Jtotal + 2), (float *)*dev_actual_residual, (float *)*dev_actual_residual + 1);
		cudacheckSYN();

		residual_vs_tolerance_kernel <<< 1, 1 >>> ((float *)*dev_actual_residual + 1, (bool *)dev_res_vs_tol, tolerance);
		cudacheckSYN();

		cudacall(cudaMemcpy(&h_res_vs_tol, dev_res_vs_tol, sizeof(bool), cudaMemcpyDeviceToHost));
	//======================================================= END: Update residual ======================================================
		fprintf(stderr, "GMRES_i = %i\n", GMRES_i);
		GMRES_i ++;

		for(GMRES_i = 1; ((h_res_vs_tol) && (GMRES_i < maxiter)); GMRES_i ++)
		{

			fprintf(stderr, "GMRES_i = %i\n", GMRES_i);

			get_resized((cuComplex **)&dev_H_, dim3(GMRES_i + 2, GMRES_i + 1), dim3(1, 1), (unsigned int)GMRES_i + 1, (unsigned int)GMRES_i, (unsigned int)GMRES_i + 2, (unsigned int)GMRES_i + 1);

			w_equal_to_A_v((cuComplex *)dev_gamma_array, (unsigned int) GMRES_i, (bool *)dev_mask, (cuComplex *)dev_orthogonal_basis, (cuComplex **)&dev_w, (cufftHandle) plan, for_gradient);

			for(unsigned int j = 0; j < GMRES_i + 1; j++)
			{
				get_new_w_and_H((cuComplex **)&dev_w, (cuComplex *)(dev_H_ + j * (GMRES_i + 1) + GMRES_i), (cuComplex *)(dev_orthogonal_basis+j * N * N));//Fill Hessenberg matrix
			}

			get_H_equal_norm_w((cuComplex *)(dev_H_+(GMRES_i + 1) * (GMRES_i + 1) + GMRES_i), (cuComplex *)dev_w);//Fill Hessenberg matrix
		//============================================== BEGIN: Fill Orthogonal Basis m.============================================
			get_resized((cuComplex **)&dev_orthogonal_basis, dim3(GMRES_i + 2, N * N / 512), dim3(1, 512), (unsigned int)GMRES_i + 1, (unsigned int)N * N, (unsigned int)GMRES_i + 2, (unsigned int)N * N);
			get_v_equal_w_devided_H((cuComplex *)(dev_orthogonal_basis + N * N * (GMRES_i + 1)), (cuComplex **)&dev_w, (cuComplex *)(dev_H_ + (GMRES_i + 1) * (GMRES_i + 1) + GMRES_i));
		//===============================    END: Fill Orthogonal Basis m.  ===========================================
		//============================================== Begin: Least Squares Step =========================================================
		//========================================== BEGIN:(Jtotal)resize_kernel ==========================================
			cudacall(cudaMalloc((void**)&dev_resized, (GMRES_i + 2) * (GMRES_i + 2) * sizeof(cuComplex)));
	
			Jtotal_resize_kernel <<< dim3(GMRES_i + 2, GMRES_i + 2), dim3(1, 1) >>> ((cuComplex *)dev_Jtotal, GMRES_i + 1, (cuComplex *)dev_resized);
			cudacheckSYN();

			cudacall(cudaFree((cuComplex *)dev_Jtotal));

			dev_Jtotal = dev_resized;
		//====================================== END: (Jtotal) resize_kernel ============================================
		//================================ BEGIN: MATMUL (H_temp=Jtotal * H) ==============================================
			cudacall(cudaFree((cuComplex *)dev_Htemp));

			cudacall(cudaMalloc((void**)&dev_Htemp, (GMRES_i + 2) * (GMRES_i + 1) * sizeof(cuComplex)));

			usual_MatMul_CUDA((cublasHandle_t *)handle_p, (cuComplex *)dev_Jtotal, (cuComplex *)dev_H_, (cuComplex *)dev_Htemp, (unsigned int)(GMRES_i + 2), (unsigned int)(GMRES_i + 2), (unsigned int)(GMRES_i + 1));
			cudacheckSYN();
		//================================== END: MATMUL (H_temp=Jtotal * H) ===============================================
		//================================================ END: Least Squares Step =========================================================
		//============================================= BEGIN: Create Givens_Rotation_Matrix ========================================
			cudacall(cudaFree((cuComplex *)dev_Givens_rotation));

			cudacall(cudaMalloc((void**)&dev_Givens_rotation, (GMRES_i + 2) * (GMRES_i + 2) * sizeof(cuComplex)));

			create_Givens_rotation_matrix_kernel <<< dim3(GMRES_i + 2, GMRES_i + 2), dim3(1, 1) >>> ((cuComplex *)dev_Givens_rotation, (cuComplex *)dev_Htemp);
			cudacheckSYN();
		//=============================================== END: Create Givens_Rotation_Matrix ========================================
		//================================================== BEGIN: Jtotal = J*Jtotal =================================================
			usual_MatMul_CUDA((cublasHandle_t *)handle_p, (cuComplex *)dev_Givens_rotation, (cuComplex *)dev_Jtotal, (cuComplex *)dev_Jtotal, (unsigned int)(GMRES_i + 2), (unsigned int)(GMRES_i + 2), (unsigned int)(GMRES_i + 2));
			cudacheckSYN();
		//==================================================== END: Jtotal = J*Jtotal =================================================
		//===================================================== BEGIN: Update residual ======================================================
			if ((GMRES_i > 8) && ((GMRES_i + 1) % 10 < 1))
				get_resized_act_res(dev_actual_residual, GMRES_i);

			next_residual_kernel <<< 1, 1 >>> ((cuComplex *)(dev_Jtotal + (GMRES_i + 2) * (GMRES_i + 1)), (float *)*dev_actual_residual, (float *)*dev_actual_residual + GMRES_i + 1);
			cudacheckSYN();

			residual_vs_tolerance_kernel <<< 1, 1 >>> ((float *)((*dev_actual_residual) + GMRES_i + 1), (bool *)dev_res_vs_tol, tolerance);
			cudacheckSYN();

			cudacall(cudaMemcpy(&h_res_vs_tol, dev_res_vs_tol, sizeof(bool), cudaMemcpyDeviceToHost));
			fprintf(stderr, "checking status = %i\n", h_res_vs_tol ? 1 : 0);
		//======================================================= END: Update residual ======================================================
		}
	//================================================= BEGIN: Free after loop ==================================================================
		cudacall(cudaFree((bool *)dev_res_vs_tol));
		cudacall(cudaFree((cuComplex *)dev_Htemp));
		cudacall(cudaFree((cuComplex *)dev_Givens_rotation));
		cudacall(cudaFree((cuComplex *)dev_w));
	//=================================================== END: Free after loop ==================================================================
	//================================================== BEGIN: HH = (Jtotal*H)_resized ==========================================================
		cudacall(cudaFree((cuComplex *)dev_HH));
		cudacall(cudaMalloc((void**)&dev_HH, GMRES_i * GMRES_i * sizeof(cuComplex)));

		usual_MatMul_CUDA((cublasHandle_t *)handle_p, (cuComplex *)dev_Jtotal, (cuComplex *)dev_H_, (cuComplex *)dev_HH,(unsigned int)GMRES_i,(unsigned int)(GMRES_i + 1),(unsigned int)GMRES_i);
		cudacheckSYN();

		cudacall(cudaFree((cuComplex *)dev_H_));
	//===================================================== END: HH = (Jtotal*H)_resized ==========================================================
	//================================================= BEGIN: cc = Jtotal * norm_res_vec =========================================================
		cudacall(cudaMalloc((void**)&dev_cc, (GMRES_i + 1) * sizeof(cuComplex)));

		get_cc_kernel <<< GMRES_i, 1 >>> ((cuComplex *)dev_cc, (cuComplex *)dev_Jtotal, (float *)*dev_actual_residual);
		cudacheckSYN();

		cudacall(cudaFree((cuComplex *)dev_Jtotal));
	//=================================================== END: cc = Jtotal * norm_res_vec =========================================================
		if (GMRES_i > 0)
		{
			if (GMRES_i < 2)
			{
				get_new_solution_kernel <<< 1, 1 >>> ((cuComplex *)dev_cc, (cuComplex *)dev_HH);	
				cudacheckSYN();

				get_solution_kernel <<< gridsize_sum_reduce, blocksize_sum_reduce >>> ((cuComplex *)dev_solution, (cuComplex *)dev_cc, (cuComplex *)dev_orthogonal_basis);
				cudacheckSYN();
			}
			else
			{
			//============================================ BEGIN: Find solution to the LES(cc_new) for HH*cc_new=cc ============================================
				int *dev_PivotArray;
				int *dev_InfoArray;
				int *InfoArray;

				InfoArray = (int *)malloc(sizeof(int));

				cudacall(cudaMalloc((void**)&dev_PivotArray, GMRES_i * sizeof(int)));
				cudacall(cudaMalloc((void**)&dev_InfoArray, sizeof(int)));

				cuComplex alpha;
				alpha.x = 1.f;
				alpha.y = 0.f;
				/**********************************/
				/* COMPUTING THE LU DECOMPOSITION */
				/**********************************/
				cuComplex **HH_p = (cuComplex **)malloc(sizeof(cuComplex *));
				cuComplex **dev_HH_p;
				cudacall(cudaMalloc(&dev_HH_p, sizeof(cuComplex *)));

				(*HH_p) = dev_HH;

				cudacall(cudaMemcpy(dev_HH_p, HH_p, sizeof(cuComplex *), cudaMemcpyHostToDevice));

				cublascall(cublasCgetrfBatched(*handle_p, GMRES_i, dev_HH_p, GMRES_i, dev_PivotArray, dev_InfoArray, 1));
				cudacheckSYN();

				cudacall(cudaMemcpy((int *)InfoArray, (int *)dev_InfoArray, sizeof(int), cudaMemcpyDeviceToHost));

				cudacall(cudaFree((int *)dev_InfoArray));
		
				if (*InfoArray) cudacall((cudaError_t) 1);

				free(InfoArray);
				/*********************************************************/
				/*           INVERT UPPER AND LOWER TRIANGULAR MATRICES  */
				/*********************************************************/
				rearrange_kernel <<< 1, GMRES_i >>> ((cuComplex *)dev_cc, (int *)dev_PivotArray);
				cudacheckSYN();

				cudacall(cudaFree((int *)dev_PivotArray));

				cublasCtrsm(*handle_p, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, GMRES_i, 1, (const cuComplex *)&alpha, dev_HH, GMRES_i, dev_cc, GMRES_i);
				cublasCtrsm(*handle_p, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_UNIT, GMRES_i, 1, (const cuComplex *)&alpha, dev_HH, GMRES_i, dev_cc, GMRES_i);
		
				cudacall(cudaFree((cuComplex *)dev_HH_p));
				free(HH_p);
			//============================================ END: Find solution to the LES(cc_new) for HH*cc_new=cc ===========================================
			//============================================ BEGIN: x = x0 + V * cc ===========================================
				for(unsigned int j = 0; j < GMRES_i; j++)
				{
					add_kernel <<< gridsize_sum_reduce, blocksize_sum_reduce >>> ((cuComplex *)dev_solution, (cuComplex *)dev_orthogonal_basis + j * N * N, (cuComplex *)dev_cc + j);
					cudacheckSYN();
				}
			}
		}

		cudacall(cudaFree((cuComplex *)dev_HH));
		cudacall(cudaFree((cuComplex *)dev_cc));
	}
	else
	{
		
		cudacall(cudaFree((bool *)dev_res_vs_tol));
	}
	cudacall(cudaFree((cuComplex *)dev_orthogonal_basis));
	(*GMRES_n) = GMRES_i;
	(*h_res_vs_tol_p) = h_res_vs_tol;
}

void usual_MatMul_CUDA(cublasHandle_t *handle, cuComplex *A, cuComplex *B, cuComplex *C, unsigned int n, unsigned int k, unsigned int m)
{
	unsigned int lda = k, ldb = m;
	unsigned int ldc = (lda > ldb) ? ldb : lda;
	cuComplex alf;
	alf.x = 1.f;
	alf.y = 0.f;
	cuComplex bet;
	bet.x = 0.f;
	bet.y = 0.f;
	const cuComplex *alpha = &alf;
	const cuComplex *beta = &bet;

	cublascall(cublasCgemm3m(*handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, B, ldb, A, lda, beta, C, ldc));
}

void G_x_fft_matvec(cuComplex *dev_gamma_array, bool *dev_mask, cuComplex *dev_solution, cuComplex **dev_matmul_out_extended, cufftHandle plan, const bool for_gradient)
{
	dim3 blocks_M(THREADS_PER_BLOCK_M, THREADS_PER_BLOCK_M);
	dim3 threads(Q, Q);

	cudacall(cudaMalloc((void**)dev_matmul_out_extended, (2 * N - 1) * (2 * N - 1) * sizeof(cuComplex)));

	if (for_gradient)
	{		
		extend_by_zeros_for_gradient_kernel <<< blocks_M, threads >>> ((cuComplex *)dev_solution, (cuComplex *)(*dev_matmul_out_extended));
	}
	else
	{
		extend_by_zeros_kernel <<< blocks_M, threads >>> ((bool *)dev_mask, (cuComplex *)dev_solution, (cuComplex *)(*dev_matmul_out_extended));
	}	
	cudacheckSYN();

	cufftcall(cufftExecC2C(plan, (cuComplex *)(*dev_matmul_out_extended), (cuComplex *)(*dev_matmul_out_extended), CUFFT_FORWARD));
	cudacheckSYN();

	MatMul_ElemWise_Kernel <<< blocks_M, threads >>> ((cuComplex *)dev_gamma_array, (cuComplex *)(*dev_matmul_out_extended));	
	cudacheckSYN();

	cufftcall(cufftExecC2C(plan, (cuComplex *)(*dev_matmul_out_extended), (cuComplex *)(*dev_matmul_out_extended), CUFFT_INVERSE));	
	cudacheckSYN();
}



void w_equal_to_A_v(cuComplex *dev_gamma_array, unsigned int GMRES_i, bool *dev_mask, cuComplex *dev_orthogonal_basis, cuComplex **dev_w, cufftHandle plan, const bool for_gradient)
{
	dim3 blocks(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
	dim3 threads(Q, Q);
	cuComplex *dev_w_extended;

	G_x_fft_matvec((cuComplex *)dev_gamma_array, (bool *)dev_mask, (cuComplex *)(dev_orthogonal_basis + GMRES_i * N * N), (cuComplex **)&dev_w_extended, (cufftHandle) plan, for_gradient);

	if (for_gradient)
	{
		_2D_to_1D_kernel_for_gradient <<< blocks, threads >>> ((bool *)dev_mask, (cuComplex*)(dev_orthogonal_basis + GMRES_i * N * N), (cuComplex *)dev_w_extended, (cuComplex *)(*dev_w));
	}
	else
	{
		_2D_to_1D_kernel <<< blocks, threads >>> ((cuComplex*)(dev_orthogonal_basis + GMRES_i * N * N), (cuComplex *)dev_w_extended, (cuComplex *)(*dev_w));
	}
	cudacheckSYN();

	cudacall(cudaFree((cuComplex *)dev_w_extended));
}



void get_resized(cuComplex **to_be_resized, dim3 gridsize, dim3 blocksize, unsigned int old_size_i, unsigned int old_size_j, unsigned int new_size_i, unsigned int new_size_j)
{
	cuComplex *dev_resized;	

	cudacall(cudaMalloc((void**)&dev_resized, new_size_i * new_size_j * sizeof(cuComplex)));
	
	resize_kernel <<< gridsize, blocksize >>> ((cuComplex *)(*to_be_resized), (unsigned int)old_size_i, (unsigned int)old_size_j, (unsigned int)new_size_j, (cuComplex *)dev_resized);
	cudacheckSYN();

	cudacall(cudaFree((cuComplex *)(*to_be_resized)));

	(*to_be_resized) = dev_resized;
}

void get_new_w_and_H(cuComplex **dev_w, cuComplex *dev_Hj, cuComplex *dev_orthogonal_basis_j)
{
	dim3 blocksize(512);
	dim3 gridsize(N * N / 512);

	inner_product_float_kernel <<< gridsize, blocksize >>> ((cuComplex *)dev_orthogonal_basis_j, (cuComplex *)(*dev_w), (cuComplex *)dev_Hj);
	cudacheckSYN();

	weight_subtract_kernel <<< gridsize, blocksize >>> ((cuComplex *)(*dev_w), (cuComplex *)dev_Hj, (cuComplex *)dev_orthogonal_basis_j);
	cudacheckSYN();
}

void get_H_equal_norm_w(cuComplex *dev_Hj, cuComplex *dev_w)
{
	dim3 blocksize(512);
	dim3 gridsize(N * N / 512);

	cudacall(cudaMemset(dev_Hj, 0, sizeof(cuComplex)));

	sum_squares_reduce_real_kernel <<< gridsize, blocksize >>> ((cuComplex *)dev_w, (cuComplex *)dev_Hj);
	cudacheckSYN();

	sqrt_real_kernel <<< 1, 1 >>> ((cuComplex *)dev_Hj);
	cudacheckSYN();
}

void get_v_equal_w_devided_H(cuComplex *dev_orthogonal_basis_j, cuComplex **dev_w, cuComplex *dev_H_j)
{
	dim3 blocksize(512);
	dim3 gridsize(N * N / 512);

	weight_divide_kernel <<< gridsize, blocksize >>> ((cuComplex *)dev_orthogonal_basis_j, (cuComplex *)(*dev_w), (cuComplex *)dev_H_j);
	cudacheckSYN();
}


void get_gamma_array(cuComplex **dev_gamma_array, cufftHandle plan)
{	
	dim3 blocks(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
	dim3 threads(Q, Q);
	cudacall(cudaMalloc((void**)dev_gamma_array, (2 * N - 1) * (2 * N - 1) * sizeof(cuComplex)));

	Green_matrix_create_Kernel <<< blocks, threads>>> ((cuComplex*)(*dev_gamma_array));
	cudacheckSYN();

	cufftcall(cufftExecC2C(plan, (cuComplex *)(*dev_gamma_array), (cuComplex *)(*dev_gamma_array), CUFFT_FORWARD));
	cudacheckSYN();
}

void get_resized_act_res(float **dev_actual_residual, const unsigned int GMRES_i)
{	
	float *dev_resized;

	cudacall(cudaMalloc((void**)&dev_resized, (GMRES_i + 12) * sizeof(float)));

	dublicate_float_kernel <<< GMRES_i + 1, 1 >>> ((float *)dev_resized, (float *)*dev_actual_residual);
	cudacheckSYN();

	cudacall(cudaFree((float *)*dev_actual_residual));

	*dev_actual_residual = dev_resized;
}

//======================================================================================
void get_exclusion_zone_v1(bool *dev_exclusion_zone, const bool *dev_mask, const unsigned int spread, const dim3 blocks, const dim3 threads)
{
	cudacall(cudaMemset(dev_exclusion_zone, false, N * N * sizeof(bool)));
	fill_exclusion_zone_kernel_v1 <<< blocks, threads >>> ((bool *)dev_exclusion_zone, (bool *)dev_mask, spread);
	cudacheckSYN();
}

void get_exclusion_zone_v2(bool *dev_exclusion_zone, const bool *dev_mask, unsigned int *dev_index_of_max, const unsigned int spread, const dim3 blocks, const dim3 threads)
{
	cudacall(cudaMemset(dev_exclusion_zone, false, N * N * sizeof(bool)));
	fill_exclusion_zone_kernel_v2 <<< blocks, threads >>> ((bool *)dev_exclusion_zone, (bool *)dev_mask, (unsigned int *)dev_index_of_max, spread);
	cudacheckSYN();
}


//exclusion zone (v.3)
void get_i_threshold_for_max(bool *dev_mask, unsigned int *dev_i_indeces, dim3 blocks, dim3 threads)
{
	unsigned int s = blocks.x / Q;
 
	min_reduce_first_kernel <<< blocks, threads >>> ((bool *)dev_mask, (unsigned int *)dev_i_indeces);
	cudacheckSYN();
	while(s > 1)
	{
		blocks.x = blocks.y = s;
		
		min_reduce_second_kernel <<< blocks, threads >>> ((unsigned int *)dev_i_indeces);
		cudacheckSYN();
		
		s = s / Q;
	}
	min_reduce_second_kernel <<< dim3(1, 1), blocks >>> ((unsigned int *)dev_i_indeces);
	cudacheckSYN();
}

 

//v.1 out of 3 versions
void target_function_v1(cuComplex *dev_solution, float *dev_intensity)
{
	unsigned int g = N * N >> 9;
	unsigned int s = g >> 9;

 
	max_reduce_first_kernel_v1 <<< g, 512 >>> ((cuComplex *)dev_solution, (float *)dev_intensity);
	cudacheckSYN();
	while(s > 1)
	{
		g = s;
		
		max_reduce_second_kernel_v1 <<< g, 512 >>> ((float *)dev_intensity);
		cudacheckSYN();
		
		s = s >> 9;
	}
	max_reduce_second_kernel_v1 <<< 1, g >>> ((float *)dev_intensity);
	cudacheckSYN();
}

//v2!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//v.2 out of 3 versions
void target_function_v2(bool *dev_exclusion_zone, cuComplex *dev_solution, float *dev_intensity)
{
	unsigned int g = N * N >> 9;
	unsigned int s = g >> 9;
//v2!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	max_reduce_first_kernel_v2 <<< g, 512 >>> ((bool *)dev_exclusion_zone, (cuComplex *)dev_solution, (float *)dev_intensity);
	cudacheckSYN();
	while(s > 1)
	{
		g = s;
		
//v2???????????????????????????????????????????????????
		max_reduce_second_kernel_v2 <<< g, 512 >>> ((float *)dev_intensity);
		cudacheckSYN();
		
		s = s >> 9;
	}
	max_reduce_second_kernel_v2 <<< 1, g >>> ((float *)dev_intensity);
	cudacheckSYN();
}


//v.3 out of 3 versions
void target_function_v3(unsigned int *dev_i_threshold, cuComplex *dev_solution, float *dev_intensity)
{
	unsigned int g = N * N >> 9;
	unsigned int s = g >> 9;
	max_reduce_first_kernel_v3 <<< g, 512 >>> ((unsigned int *)dev_i_threshold, (cuComplex *)dev_solution, (float *)dev_intensity);
	cudacheckSYN();
	while(s > 1)
	{
		g = s;
		
		max_reduce_second_kernel_v3 <<< g, 512 >>> ((float *)dev_intensity);
		cudacheckSYN();
		
		s = s >> 9;
	}
	max_reduce_second_kernel_v3 <<< 1, g >>> ((float *)dev_intensity);
	cudacheckSYN();
}


void get_max_intensity_specified_zone_v1(unsigned int *dev_required_intensity_distribution_indeces, float *dev_intensities, float *dev_max_intensities, const unsigned int h_distribution_size, dim3 blocks, dim3 threads)
{
	unsigned int s = blocks.x / Q;

	max_reduce_first_specified_zone_kernel_v1 <<< blocks, threads >>>((unsigned int *)dev_required_intensity_distribution_indeces, (float *)dev_intensities, (float *)dev_max_intensities, h_distribution_size);
	cudacheckSYN();
	while(s > 1)
	{
		blocks.x = blocks.y = s;
		
		max_reduce_second_specified_zone_kernel_v1 <<< blocks, threads >>> ((float *)dev_max_intensities);
		cudacheckSYN();
		
		s = s / Q;
	}
	max_reduce_second_specified_zone_kernel_v1 <<< dim3(1, 1), blocks >>> ((float *)dev_max_intensities);
	cudacheckSYN();
}


void get_max_indicator_v1(float *dev_indicators, float *dev_max_indicators, const unsigned int h_distribution_size, dim3 blocks, dim3 threads)
{
	unsigned int s = blocks.x / Q;

	max_reduce_first_indicators_kernel_v1 <<< blocks, threads >>> ((float *)dev_indicators, (float *)dev_max_indicators, h_distribution_size);
	cudacheckSYN();
	while(s > 1)
	{
		blocks.x = blocks.y = s;
		
		max_reduce_second_specified_zone_kernel_v1 <<< blocks, threads >>> ((float *)dev_max_indicators);
		cudacheckSYN();
		
		s = s / Q;
	}
	max_reduce_second_specified_zone_kernel_v1 <<< dim3(1, 1), blocks >>> ((float *)dev_max_indicators);
	cudacheckSYN();
}


void get_min_required_intensity_specified_zone_v1(float *dev_required_intensity_distribution, unsigned int *dev_required_intensity_distribution_indeces, float *dev_required, const unsigned int h_distribution_size, dim3 blocks, dim3 threads)
{
	unsigned int s = blocks.x / Q;

	min_reduce_first_specified_zone_kernel_v1 <<< blocks, threads >>>((float *)dev_required_intensity_distribution, (float *)dev_required, h_distribution_size);
	cudacheckSYN();
	while(s > 1)
	{
		blocks.x = blocks.y = s;
		
		min_reduce_second_specified_zone_kernel_v1 <<< blocks, threads >>> ((float *)dev_required);
		cudacheckSYN();
		
		s = s / Q;
	}
	min_reduce_second_specified_zone_kernel_v1 <<< dim3(1, 1), blocks >>> ((float *)dev_required);
	cudacheckSYN();
}


void find_min_i_index_of_mask(bool *dev_mask, unsigned int *dev_i_indeces, dim3 blocks, dim3 threads)
{
	unsigned int s = blocks.x / Q;
 
	min_reduce_first_kernel <<< blocks, threads >>> ((bool *)dev_mask, (unsigned int *)dev_i_indeces);
	cudacheckSYN();
	while(s > 1)
	{
		blocks.x = blocks.y = s;
		
		min_reduce_second_kernel <<< blocks, threads >>> ((unsigned int *)dev_i_indeces);
		cudacheckSYN();
		
		s = s / Q;
	}
	min_reduce_second_kernel <<< dim3(1, 1), blocks >>> ((unsigned int *)dev_i_indeces);
	cudacheckSYN();
}



//v.1 out of 3 versions
void target_function_indexed_v1(bool *dev_exclusion_zone, cuComplex *dev_solution, unsigned int *dev_index_of_max, float *h_instensity_max)
{
	unsigned int *dev_indeces;
	float *dev_intensities;
	unsigned int g = N * N >> 9;
	unsigned int s = g >> 9;

	cudacall(cudaMalloc((void**)&dev_indeces, (N * N / 512) * sizeof(unsigned int)));
	cudacall(cudaMalloc((void**)&dev_intensities, (N * N / 512) * sizeof(unsigned int)));

	max_reduce_first_indexed_kernel_v1 <<< g, 512 >>> ((bool *)dev_exclusion_zone, (cuComplex *)dev_solution, (float *)dev_intensities, (unsigned int  *)dev_indeces);
	cudacheckSYN();
	while(s > 1)
	{
		g = s;
		
		max_reduce_second_indexed_kernel_v1 <<< g, 512 >>> ((float *)dev_intensities, (unsigned int *)dev_indeces);
		cudacheckSYN();
		
		s = s >> 9;
	}
	max_reduce_second_indexed_kernel_v1 <<< 1, g >>> ((float *)dev_intensities, (unsigned int *)dev_indeces);
	cudacheckSYN();
	
	dublicate_unsigned_int_kernel <<< 1, 1 >>> ((unsigned int *)dev_index_of_max, (unsigned int *)dev_indeces);
	cudacheckSYN();

	cudacall(cudaMemcpy(h_instensity_max, dev_intensities, sizeof(float), cudaMemcpyDeviceToHost));
	
	cudacall(cudaFree((unsigned int *)dev_indeces));
	cudacall(cudaFree((float *)dev_intensities));
}


//v.2 out of 3 versions
void target_function_indexed_v2(unsigned int *dev_i_threshold, cuComplex *dev_solution, unsigned int *dev_index_of_max, float *h_instensity_max, dim3 blocks, dim3 threads)
{
	unsigned int *dev_indeces;
	float *dev_intensities;
	unsigned int s = blocks.x / threads.x;

	cudacall(cudaMalloc((void**)&dev_indeces, (N * N / 512) * sizeof(unsigned int)));
	cudacall(cudaMalloc((void**)&dev_intensities, (N * N / 512) * sizeof(unsigned int)));

	max_reduce_first_indexed_kernel_v2 <<< blocks, threads >>> ((unsigned int *)dev_i_threshold, (cuComplex *)dev_solution, (float *)dev_intensities, (unsigned int  *)dev_indeces);
	cudacheckSYN();

	while(s > 1)
	{
		blocks.x = blocks.y = s;
		
		max_reduce_second_indexed_kernel_v2 <<< blocks, threads >>> ((float *)dev_intensities, (unsigned int *)dev_indeces);
		cudacheckSYN();
		
		s = s / threads.x;
	}
	max_reduce_second_indexed_kernel_v2 <<< dim3(1, 1), blocks >>> ((float *)dev_intensities, (unsigned int *)dev_indeces);
	cudacheckSYN();
	
	dublicate_unsigned_int_kernel <<< 1, 1 >>> ((unsigned int *)dev_index_of_max, (unsigned int *)dev_indeces);
	cudacheckSYN();

	cudacall(cudaMemcpy(h_instensity_max, dev_intensities, sizeof(float), cudaMemcpyDeviceToHost));
	
	cudacall(cudaFree((unsigned int *)dev_indeces));
	cudacall(cudaFree((float *)dev_intensities));
}


//v.3 out of 3 versions
void target_function_indexed_v3(float *dev_gradient, unsigned int *dev_suitable_steepest_grad_index_to_add, dim3 blocks, dim3 threads)
{
	unsigned int *dev_indeces;
	float *dev_intensities;
	unsigned int s = blocks.x / threads.x;

	cudacall(cudaMalloc((void**)&dev_indeces, (N * N / 512) * sizeof(unsigned int)));

	max_reduce_first_indexed_kernel_v5 <<< blocks, threads >>> ((float *)dev_gradient, (unsigned int *)dev_indeces);
	cudacheckSYN();

	while(s > 1)
	{
		blocks.x = blocks.y = s;
		
		max_reduce_second_indexed_kernel_v2 <<< blocks, threads >>> ((float *)dev_gradient, (unsigned int *)dev_indeces);
		cudacheckSYN();
		
		s = s / threads.x;
	}
	max_reduce_second_indexed_kernel_v2 <<< dim3(1, 1), blocks >>> ((float *)dev_gradient, (unsigned int *)dev_indeces);
	cudacheckSYN();

	cudacall(cudaMemcpy(dev_suitable_steepest_grad_index_to_add, dev_indeces, sizeof(unsigned int), cudaMemcpyDeviceToDevice));
	
	cudacall(cudaFree((unsigned int *)dev_indeces));
}
