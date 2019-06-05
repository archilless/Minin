int cuSolve_LES(cuComplex *d_A, cuComplex *d_B, const int m, cusolverDnHandle_t cusolverH);

void FFT_GMRES_with_CUDA_extended_with_host_memory_single_debugged(const cuComplex *dev_gamma_array, const bool *dev_mask, cuComplex *dev_solution, float **dev_actual_residual, unsigned int *GMRES_n, cufftHandle plan, cublasHandle_t *handle_p, const float tolerance, const bool for_gradient, const unsigned int h_index_of_max, unsigned int maxiter)
{
	dim3 blocks(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
	dim3 blocks_M(THREADS_PER_BLOCK_M, THREADS_PER_BLOCK_M);
	dim3 threads(Q, Q);
	dim3 blocksize_sum_reduce(512);
	dim3 gridsize_sum_reduce(N * N / blocksize_sum_reduce.x);

	bool *dev_res_vs_tol;
	bool *res_vs_tol;
	
	cuComplex *dev_matmul_out_extended;
	cuComplex *dev_residual_vec;
	cuComplex *dev_orthogonal_vec;
	cuComplex *dev_orthogonal_basis_first;
	cuComplex *h_orthogonal_basis = (cuComplex *)malloc(100 * N * N * sizeof(cuComplex));
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
	unsigned int Batch_size = 105;//104857600 / N / N;
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

//	fprintf(stderr, "in GMRES\n");
	cudacall(cudaFree((cuComplex *)dev_matmul_out_extended));
//========================================== END: get_residual_vector =========================================================
	cudacall(cudaMalloc((void**)dev_actual_residual, 10 * sizeof(float)));

	cudacall(cudaMemset(*dev_actual_residual, 0, sizeof(float)));

	sum_squares_reduce_kernel <<< gridsize_sum_reduce, blocksize_sum_reduce >>> ((cuComplex *)dev_residual_vec, (float *)*dev_actual_residual);
	cudacheckSYN();

	sqrt_float_kernel <<< 1, 1 >>> ((float *)*dev_actual_residual);
	cudacheckSYN();
//============================================BEGIN:residual_normalization_kernel=======================================================
	cudacall(cudaMalloc((void**)&dev_orthogonal_vec, N * N * sizeof(cuComplex)));
	cudacall(cudaMalloc((void**)&dev_orthogonal_basis_first, Batch_size * N * N * sizeof(cuComplex)));

	residual_normalization_kernel <<< gridsize_sum_reduce, blocksize_sum_reduce >>> ((cuComplex *)dev_residual_vec, (float *)*dev_actual_residual, (cuComplex *)dev_orthogonal_vec);
	cudacheckSYN();

	cudacall(cudaFree((cuComplex *)dev_residual_vec));
//============================================= END:residual_normalization_kernel ==================================================
//============================================= Begin: Condition to iterate ==========================================================	
	cudacall(cudaMalloc((void**)&dev_res_vs_tol, sizeof(bool)));
	res_vs_tol = (bool *)malloc(sizeof(bool));

	residual_vs_tolerance_kernel <<< 1, 1 >>> ((float *)*dev_actual_residual, (bool *)dev_res_vs_tol, tolerance);
	cudacheckSYN();

	cudacall(cudaMemcpy(res_vs_tol, dev_res_vs_tol, sizeof(bool), cudaMemcpyDeviceToHost));
//=============================================== End: Condition to iterate ===========================================================
	if (*res_vs_tol)
	{
		cudacall(cudaMalloc((void**)&dev_H_,     2 * sizeof(cuComplex)));
		cudacall(cudaMalloc((void**)&dev_w , N * N * sizeof(cuComplex)));
		cuComplex *dev_w_extended;

		G_x_fft_matvec((cuComplex *)dev_gamma_array, (bool *)dev_mask, (cuComplex *)dev_orthogonal_vec, (cuComplex **)&dev_w_extended, (cufftHandle) plan, for_gradient);

		if (for_gradient)
		{
			_2D_to_1D_kernel_for_gradient <<< blocks, threads >>> ((bool *)dev_mask, (cuComplex*)dev_orthogonal_vec, (cuComplex *)dev_w_extended, (cuComplex *)dev_w);
		}
		else
		{
			_2D_to_1D_kernel <<< blocks, threads >>> ((cuComplex*)dev_orthogonal_vec, (cuComplex *)dev_w_extended, (cuComplex *)dev_w);
		}
		cudacheckSYN();

		cudacall(cudaFree((cuComplex *)dev_w_extended));

//		get_new_w_and_H((cuComplex **)&dev_w, (cuComplex *)dev_H_, (cuComplex *)dev_orthogonal_vec);//Fill Hessenberg m.
		dim3 blocksize(512);
		dim3 gridsize(N * N / 512);

//		inner_product_float_kernel <<< gridsize, blocksize >>> ((cuComplex *)dev_orthogonal_vec, (cuComplex *)dev_w, (cuComplex *)dev_H_);
//		cudacheckSYN();

		cublasCdotc((cublasHandle_t) *handle_p, N * N, (const cuComplex *)dev_orthogonal_vec, 1, (const cuComplex *)dev_w, 1, (cuComplex *)dev_H_);
		cudacheckSYN();

		weight_subtract_kernel <<< gridsize, blocksize >>> ((cuComplex *)dev_w, (cuComplex *)dev_H_, (cuComplex *)dev_orthogonal_vec);
		cudacheckSYN();

		get_H_equal_norm_w((cuComplex *)(dev_H_+ 1), (cuComplex *)dev_w);
	//============================================== BEGIN: Fill Orthogonal Basis matrix ============================================	
		dublicate_kernel <<< N * N / 512, 512 >>> ((cuComplex *)dev_orthogonal_vec, (cuComplex *)dev_orthogonal_basis_first);
		cudacheckSYN();

		get_v_equal_w_devided_H((cuComplex *)dev_orthogonal_vec, (cuComplex **)&dev_w, (cuComplex *)(dev_H_ + 1));

		dublicate_kernel <<< N * N / 512, 512 >>> ((cuComplex *)dev_orthogonal_vec, (cuComplex *)dev_orthogonal_basis_first + N * N);
		cudacheckSYN();
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
	//================================================== BEGIN: Jtotal = J*Jtotal =================================================
		cudacall(cudaMalloc((void**)&dev_Jtotal, N * N * sizeof(cuComplex)));		
		dublicate_kernel <<< blocks, threads >>> ((cuComplex *)dev_Givens_rotation, (cuComplex *)dev_Jtotal);
		cudacheckSYN();
	//==================================================== END: Jtotal = J*Jtotal =================================================
	//===================================================== BEGIN: Update residual ======================================================
		next_residual_kernel <<< 1, 1 >>> ((cuComplex *)dev_Jtotal + 2, (float *)*dev_actual_residual, (float *)*dev_actual_residual + 1);
		cudacheckSYN();

		residual_vs_tolerance_kernel <<< 1, 1 >>> ((float *)*dev_actual_residual + 1, (bool *)dev_res_vs_tol, tolerance);
		cudacheckSYN();

		cudacall(cudaMemcpy(res_vs_tol, dev_res_vs_tol, sizeof(bool), cudaMemcpyDeviceToHost));
	//======================================================= END: Update residual ======================================================
		GMRES_i ++;

		for(GMRES_i = 1; ((*res_vs_tol) && (GMRES_i < maxiter)); GMRES_i ++) //
		{
//			fprintf(stderr, "GMRES_i = %i\n", GMRES_i);

			get_resized((cuComplex **)&dev_H_, dim3(GMRES_i + 2, GMRES_i + 1), dim3(1, 1), (unsigned int)GMRES_i + 1, (unsigned int)GMRES_i, (unsigned int)GMRES_i + 2, (unsigned int)GMRES_i + 1);
			w_equal_to_A_v((cuComplex *)dev_gamma_array, (unsigned int) 0, (bool *)dev_mask, (cuComplex *)dev_orthogonal_vec, (cuComplex **)&dev_w, (cufftHandle) plan, for_gradient);

			for(unsigned int j = 0; j < GMRES_i + 1; j++)
			{
				if (j + 1 > Batch_size)
				{
					cudacall(cudaMemcpy(dev_orthogonal_vec, h_orthogonal_basis + (j - Batch_size) * N * N, N * N * sizeof(cuComplex), cudaMemcpyHostToDevice));
					get_new_w_and_H((cuComplex **)&dev_w, (cuComplex *)(dev_H_ + j * (GMRES_i + 1) + GMRES_i), (cuComplex *)dev_orthogonal_vec);
				}
				else
				{
					get_new_w_and_H((cuComplex **)&dev_w, (cuComplex *)(dev_H_ + j * (GMRES_i + 1) + GMRES_i), (cuComplex *)dev_orthogonal_basis_first + j * N * N);
				}
			}

			get_H_equal_norm_w((cuComplex *)(dev_H_+(GMRES_i + 1) * (GMRES_i + 1) + GMRES_i), (cuComplex *)dev_w);//Fill Hessenberg matrix
		//============================================== BEGIN: Fill Orthogonal Basis m.============================================
			get_v_equal_w_devided_H((cuComplex *)dev_orthogonal_vec, (cuComplex **)&dev_w, (cuComplex *)(dev_H_ + (GMRES_i + 1) * (GMRES_i + 1) + GMRES_i));

			if (GMRES_i + 2 > Batch_size)
			{
				if ((GMRES_i > Batch_size) && ((GMRES_i + 1 - Batch_size) % 100 == 0))
				{
					h_orthogonal_basis = (cuComplex *)realloc(h_orthogonal_basis, sizeof(cuComplex) * N * N * 100 * ((GMRES_i + 1 - Batch_size) / 100 + 1));
				}
				cudacall(cudaMemcpy(h_orthogonal_basis + (GMRES_i + 1 - Batch_size) * N * N, dev_orthogonal_vec, N * N * sizeof(cuComplex), cudaMemcpyDeviceToHost));
			}
			else
			{
//				fprintf(stderr, "D\n");
				dublicate_kernel <<< N * N / 512, 512 >>> ((cuComplex *)dev_orthogonal_vec, (cuComplex *)dev_orthogonal_basis_first + (GMRES_i + 1) * N * N);
				cudacheckSYN();

//				fprintf(stderr, "E\n");
			}
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
			{
				get_resized_act_res(dev_actual_residual, GMRES_i);
			}

			next_residual_kernel <<< 1, 1 >>> ((cuComplex *)(dev_Jtotal + (GMRES_i + 2) * (GMRES_i + 1)), (float *)*dev_actual_residual, (float *)*dev_actual_residual + GMRES_i + 1);
			cudacheckSYN();

			residual_vs_tolerance_kernel <<< 1, 1 >>> ((float *)((*dev_actual_residual) + GMRES_i + 1), (bool *)dev_res_vs_tol, tolerance);
			cudacheckSYN();

			cudacall(cudaMemcpy(res_vs_tol, dev_res_vs_tol, sizeof(bool), cudaMemcpyDeviceToHost));
		//======================================================= END: Update residual ======================================================
		}
	//================================================= BEGIN: Free after loop ==================================================================
		free(res_vs_tol);
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

		transpose_kernel <<< blocks, threads >>> ((cuComplex *)dev_HH, GMRES_i);
		cudacheckSYN();
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

				cudacall(cudaMemcpy(dev_orthogonal_vec, h_orthogonal_basis, N * N * sizeof(cuComplex), cudaMemcpyHostToDevice));

				get_solution_kernel <<< gridsize_sum_reduce, blocksize_sum_reduce >>> ((cuComplex *)dev_solution, (cuComplex *)dev_cc, (cuComplex *)dev_orthogonal_vec);
				cudacheckSYN();
			}
			else
			{
			//============================================ BEGIN: Find solution to the LES(cc_new) for HH*cc_new=cc ============================================
				cudaStream_t stream = NULL;
				cusolverDnHandle_t cusolverH = NULL;

				cusolvercall(cusolverDnCreate(&cusolverH));

				cudacall(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

				cusolvercall(cusolverDnSetStream(cusolverH, stream));

				cuSolve_LES((cuComplex *)dev_HH, (cuComplex *)dev_cc, GMRES_i, cusolverH);



				//if (cusolverH   ) cusolverDnDestroy(cusolverH);


			//============================================ END: Find solution to the LES(cc_new) for HH*cc_new=cc ===========================================
			//============================================ BEGIN: x = x0 + V * cc ===========================================
				for(unsigned int j = 0; j < GMRES_i; j++)
				{
					if (j + 2 > Batch_size)
					{
						cudacall(cudaMemcpy(dev_orthogonal_vec, h_orthogonal_basis + (j + 1 - Batch_size) * N * N, N * N * sizeof(cuComplex), cudaMemcpyHostToDevice));
						add_kernel <<< gridsize_sum_reduce, blocksize_sum_reduce >>> ((cuComplex *)dev_solution, (cuComplex *)dev_orthogonal_vec, (cuComplex *)dev_cc + j);
					}
					else
					{
						add_kernel <<< gridsize_sum_reduce, blocksize_sum_reduce >>> ((cuComplex *)dev_solution, (cuComplex *)dev_orthogonal_basis_first + j * N * N, (cuComplex *)dev_cc + j);
					}
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
		free((bool *)res_vs_tol);
	}
	cudacall(cudaFree((cuComplex *)dev_orthogonal_vec));
	cudacall(cudaFree((cuComplex *)dev_orthogonal_basis_first));
	free((cuComplex *)h_orthogonal_basis);
	(*GMRES_n) = GMRES_i;
}



int cuSolve_LES(cuComplex *d_A, cuComplex *d_B, const int m, cusolverDnHandle_t cusolverH)
{
    int *d_Ipiv = NULL; /* pivoting sequence */
    int *d_info = NULL; /* error info */
    int  lwork = 0;     /* size of workspace */
    cuComplex *d_work = NULL; /* device workspace for getrf */

    const int lda = m;
    const int ldb = m;

    int info = 0;     /* host copy of error info */
    cuComplex LU[lda*m]; /* L and U */


/* step 3: query working space of getrf */
    cudacall(cudaMalloc ((void**)&d_Ipiv, sizeof(int) * m));
    cudacall(cudaMalloc ((void**)&d_info, sizeof(int)));


    cusolvercall(cusolverDnCgetrf_bufferSize(
        cusolverH,
        m,
        m,
        d_A,
        lda,
        &lwork));

    cudacall(cudaMalloc((void**)&d_work, sizeof(cuComplex)*lwork));

/* step 4: LU factorization */
    cusolvercall(cusolverDnCgetrf(
	    cusolverH,
	    m,
	    m,
	    d_A,
	    lda,
	    d_work,
	    d_Ipiv,
	    d_info));

    cudacall(cudaDeviceSynchronize());

    cudacall(cudaMemcpy(LU   , d_A   , sizeof(cuComplex)*lda*m, cudaMemcpyDeviceToHost));
    cudacall(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 > info ){
        printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }
 
    cusolvercall(cusolverDnCgetrs(
            cusolverH,
            CUBLAS_OP_N,
            m,
            1, /* nrhs */
            d_A,
            lda,
            d_Ipiv,
            d_B,
            ldb,
            d_info));


    cudacall(cudaDeviceSynchronize());


    if (d_Ipiv ) cudaFree(d_Ipiv);
    if (d_info ) cudaFree(d_info);
    if (d_work ) cudaFree(d_work);

    if (cusolverH   ) cusolverDnDestroy(cusolverH);

    return 0;
}

