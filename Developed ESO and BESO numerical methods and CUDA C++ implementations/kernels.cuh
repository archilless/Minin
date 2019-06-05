#include <cufft.h>
#include <cublas_v2.h>
#include <fstream>
#include <iomanip>
#include <curand_kernel.h>
#define N 1024L
#define THREADS_PER_BLOCK N / Q
#define THREADS_PER_BLOCK_M THREADS_PER_BLOCK * 2

//#pragma once
//#ifdef __INTELLISENSE__
//void __syncthreads();
//#endif

#define WAVE_NUMBER 2*3.14f/(N/10.f)
#define Q 32
#define E0 1
#define ALPHA 3.14*0/180
#define EPSILON 2.25f
#define CHI (EPSILON-1)*WAVE_NUMBER*WAVE_NUMBER
#define PRECISION_TO_SAVE_DATA_TO_FILE 10
#define cudacall(call)                                                                                                          \
    do                                                                                                                          \
    {                                                                                                                           \
        cudaError_t err = (call);                                                                                               \
        if(cudaSuccess != err)                                                                                                  \
        {                                                                                                                       \
            fprintf(stderr,"CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
            cudaDeviceReset();                                                                                                  \
            exit(EXIT_FAILURE);                                                                                                 \
        }                                                                                                                       \
    }                                                                                                                           \
    while (0)

#define cudacheckSYN()                                                                                                          \
    do                                                                                                                          \
    {                                                                                                                           \
        cudaError_t err = cudaGetLastError();                                                                                   \
        if(cudaSuccess != err)                                                                                                  \
        {                                                                                                                       \
            fprintf(stderr,"GetL Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
            cudaDeviceReset();                                                                                                  \
            exit(EXIT_FAILURE);                                                                                                 \
        }                                                                                                                       \
        err = cudaDeviceSynchronize();                                                                                          \
        if(cudaSuccess != err)                                                                                                  \
        {                                                                                                                       \
            fprintf(stderr,"DevSyn ERR:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
            cudaDeviceReset();                                                                                                  \
            exit(EXIT_FAILURE);                                                                                                 \
        }                                                                                                                       \
    }                                                                                                                           \
    while (0)

#define cublascall(call)                                                                                        \
    do                                                                                                          \
    {                                                                                                           \
        cublasStatus_t status = (call);                                                                         \
        if(CUBLAS_STATUS_SUCCESS != status)                                                                     \
        {                                                                                                       \
            fprintf(stderr,"CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);     \
            cudaDeviceReset();                                                                                  \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
                                                                                                                \
    }                                                                                                           \
    while(0)

#define cufftcall(call)                                                                                         \
    do                                                                                                          \
    {                                                                                                           \
        cufftResult_t status = (call);                                                                          \
        if(CUFFT_SUCCESS != status)                                                                             \
        {                                                                                                       \
            fprintf(stderr,"CUFFT Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);      \
            cudaDeviceReset();                                                                                  \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
                                                                                                                \
    }                                                                                                           \
    while(0)

#define cusolvercall(call)                                                                                      \
    do                                                                                                          \
    {                                                                                                           \
        cusolverStatus_t status = (call);                                                                       \
        if(CUSOLVER_STATUS_SUCCESS != status)                                                                   \
        {                                                                                                       \
            fprintf(stderr,"CUSOLVER Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);   \
            cudaDeviceReset();                                                                                  \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
                                                                                                                \
    }                                                                                                           \
    while(0)


__global__ void MatMul_ElemWise_Kernel(cuComplex *bttb_sur, cuComplex *vec2D)
{
	unsigned int i = Q * blockIdx.x + threadIdx.x;
	unsigned int j = Q * blockIdx.y + threadIdx.y;
	unsigned int size_limit = (N << 1) - 1;

	if (( i < size_limit ) && ( j < size_limit ))
	{
		unsigned int index = size_limit * i + j;
		cuComplex curr_bttb = bttb_sur[index];
		cuComplex curr_out_mul = vec2D[index];
		vec2D[index].x = (curr_bttb.x * curr_out_mul.x - curr_out_mul.y * curr_bttb.y) / ((2 * N - 1)*(2 * N - 1));
		vec2D[index].y = (curr_out_mul.x * curr_bttb.y + curr_out_mul.y * curr_bttb.x) / ((2 * N - 1)*(2 * N - 1));
	}
}

__device__ __forceinline__ cuComplex my_cexpf(cuComplex z) //FOR ONLY z.x = 0.f;
{
	cuComplex res;
	sincosf(z.y, &res.y, &res.x);
	res.x *= E0;
	res.y *= E0;
	return res;
}


__global__ void _2D_to_1D_kernel(cuComplex *input_mul, cuComplex *_2D_in, cuComplex *_1D_out)
{ 
	unsigned int i = Q * blockIdx.x + threadIdx.x;
	unsigned int j = Q * blockIdx.y + threadIdx.y;
	unsigned int Ni = N * i;
	unsigned int _1D_index = Ni + j;
	unsigned int _2D_index = _1D_index + Ni - i;
	cuComplex current = input_mul[_1D_index];
	cuComplex new_arg = _2D_in[_2D_index];
	
	current.x -= new_arg.x;
	current.y -= new_arg.y;
	_1D_out[_1D_index] = current;
}

__global__ void _2D_to_1D_kernel_for_gradient(bool *dev_mask, cuComplex *input_mul, cuComplex *_2D_in, cuComplex *_1D_out)
{ 
	unsigned int i = Q * blockIdx.x + threadIdx.x;
	unsigned int j = Q * blockIdx.y + threadIdx.y;
	unsigned int Ni = N * i;
	unsigned int _1D_index = Ni + j;
	unsigned int _2D_index = _1D_index + Ni - i;
	cuComplex current = input_mul[_1D_index];
	cuComplex new_arg = _2D_in[_2D_index];

	if (dev_mask[_1D_index])
	{
		current.x -= new_arg.x;
		current.y -= new_arg.y;
	}
	_1D_out[_1D_index] = current;
}


__global__ void _2D_to_1D_kernel_to_compute_gradient(cuComplex *dev_solution, cuComplex *dev_new_z_extended, float *dev_gradient, unsigned int *dev_index_of_max)
{ 
	unsigned int i = Q * blockIdx.x + threadIdx.x;
	unsigned int j = Q * blockIdx.y + threadIdx.y;
	unsigned int Ni = N * i;
	unsigned int _1D_index = Ni + j;
	unsigned int _2D_index = _1D_index + Ni - i;
	cuComplex dev_new_z = dev_new_z_extended[_2D_index];
	cuComplex dev_x = dev_solution[_1D_index];
	dev_new_z.x = CHI * (dev_new_z.x * dev_x.x - dev_new_z.y * dev_x.y);
	dev_new_z.y = CHI * (dev_new_z.x * dev_x.y + dev_new_z.y * dev_x.x);

	dev_new_z.x = dev_new_z.x * dev_solution[*dev_index_of_max].x + dev_new_z.y * dev_solution[*dev_index_of_max].y;

	dev_gradient[_1D_index] = 2 * dev_new_z.x;
}


__global__ void _2D_to_1D_compared_Kernel(cuComplex *input_mul, cuComplex *_2D_in, cuComplex *residual)
{ 
	unsigned int i = Q * blockIdx.x + threadIdx.x;
	unsigned int j = Q * blockIdx.y + threadIdx.y;
	unsigned int size_limit = (N << 1) - 1;


	if ((i < size_limit) && (j < size_limit))
	{
		unsigned int Ni = N * i;
		unsigned int _1D_index = Ni + j;
		unsigned int _2D_index = _1D_index + Ni - i;
		cuComplex current_2D = _2D_in[_2D_index];
		cuComplex arg_old = input_mul[_1D_index];
		cuComplex Input_Field;

		Input_Field.y = - WAVE_NUMBER * (i * cos(ALPHA) + j * sin(ALPHA));
		Input_Field = my_cexpf(Input_Field);
		float sigma = 400.f;
		//Input_Field.x = Input_Field.x * exp(-pow((float)((float)(j) - 512.f), 2.f)/pow(sigma, 2.f))/(sigma * sqrt(7.28f));
		//Input_Field.y = Input_Field.y * exp(-pow((float)((float)(j) - 512.f), 2.f)/pow(sigma, 2.f))/(sigma * sqrt(7.28f));
		//Input_Field.x = Input_Field.x * (exp(-pow((float)((float)(j) - 341.f), 2.f)/pow(sigma, 2.f)) + exp(-pow((float)((float)(j) - 682.f), 2.f)/pow(sigma, 2.f)))/(sigma * sqrt(7.28f));
		//Input_Field.y = Input_Field.y * (exp(-pow((float)((float)(j) - 341.f), 2.f)/pow(sigma, 2.f)) + exp(-pow((float)((float)(j) - 682.f), 2.f)/pow(sigma, 2.f)))/(sigma * sqrt(7.28f));
		current_2D.x += Input_Field.x - arg_old.x;
		current_2D.y += Input_Field.y - arg_old.y;
		residual[_1D_index] = current_2D;
	}
}



__global__ void _2D_to_1D_compared_for_gradient_Kernel(bool *dev_mask, cuComplex *input_mul, cuComplex *_2D_in, cuComplex *residual, const unsigned int h_index_of_max)
{ 
	unsigned int i = Q * blockIdx.x + threadIdx.x;
	unsigned int j = Q * blockIdx.y + threadIdx.y;
	unsigned int size_limit = (N << 1) - 1;


	if ((i < size_limit) && (j < size_limit))
	{
		unsigned int Ni = N * i;
		unsigned int _1D_index = Ni + j;
		unsigned int _2D_index = _1D_index + Ni - i;

		cuComplex current_2D = _2D_in[_2D_index];
		cuComplex arg_old = input_mul[_1D_index];

		if (dev_mask[_1D_index])
		{
			current_2D.x += (h_index_of_max == _1D_index) ? 1.f - arg_old.x : - arg_old.x;
			current_2D.y += - arg_old.y;
		}
		else
		{
			current_2D.x =  (h_index_of_max == _1D_index) ? 1.f - arg_old.x : - arg_old.x;
			current_2D.y = - arg_old.y;
		}

		residual[_1D_index] = current_2D;
	}
}


__global__ void sum_squares_reduce_kernel(cuComplex *g_idata, float *g_odata)
{

	__shared__ float sdata[512];
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int g = 0;

	sdata[threadIdx.x] = pow((float)g_idata[i].x, 2.f) + pow((float)g_idata[i].y, 2.f);
	__syncthreads();

	for (unsigned int s = 1; s < blockDim.x; s = g)
	{	
		g = (s << 1);
		unsigned int index = g * threadIdx.x;
		
		if (index + s < blockDim.x)
		{
			sdata[index] += sdata[index + s];
			__syncthreads();
		}
		else break;
	}

	if (threadIdx.x == 0){
		atomicAdd((float *)&((*g_odata)), sdata[0]);
	}
}


__global__ void sum_squares_reduce_float_kernel(float *dev_intensities_in_maxs, float *dev_sum_of_max_intensities)
{

	__shared__ float sdata[512];
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int g = 0;

	sdata[threadIdx.x] = dev_intensities_in_maxs[i];
	__syncthreads();

	for (unsigned int s = 1; s < blockDim.x; s = g)
	{	
		g = (s << 1);
		unsigned int index = g * threadIdx.x;
		
		if (index + s < blockDim.x)
		{
			sdata[index] += sdata[index + s];
			__syncthreads();
		}
		else break;
	}

	if (threadIdx.x == 0){
		atomicAdd((float *)&((*dev_sum_of_max_intensities)), sdata[0]);
	}
}


__global__ void sum_squares_reduce_real_kernel(cuComplex *g_idata, cuComplex *g_odata)
{
	__shared__ float sdata[512];
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int g = 0;

	sdata[threadIdx.x] = pow((float)g_idata[i].x, 2.f) + pow((float)g_idata[i].y, 2.f);
	__syncthreads();
	for (unsigned int s = 1; s < blockDim.x; s = g)
	{	
		g = (s << 1);
		unsigned int index = g * threadIdx.x;
		
		if (index + s < blockDim.x)
		{
			atomicAdd((float *)&sdata[index], sdata[index + s]);
		}
		else break;
		__syncthreads();
	}

	if (threadIdx.x == 0){
		atomicAdd((float *) &(g_odata->x), (float)sdata[0]);
	}
}


__global__ void sum_indicators_squared_reduce_kernel(float *dev_indicators, float *sum_indicators_squared, const unsigned int h_distribution_size)
{
	__shared__ float sdata[512];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int g = 0;

	sdata[threadIdx.x] = i < h_distribution_size ? pow(dev_indicators[i], 2.f) : 0.f;
	__syncthreads();
	for (unsigned int s = 1; s < blockDim.x; s = g)
	{	
		g = (s << 1);
		unsigned int index = g * threadIdx.x;
		
		if (index + s < blockDim.x)
		{
			atomicAdd((float *)&sdata[index], sdata[index + s]);
		}
		else break;
		__syncthreads();
	}

	if (threadIdx.x == 0){
		atomicAdd((float *)sum_indicators_squared, (float)sdata[0]);
	}
}



__global__ void sum_squares_reduce_w_b_kernel(float *dev_w, float *dev_b, float *dev_required, unsigned int *dev_required_indeces, float *dev_intensities, float *dev_squared_norms_of_indicators, const unsigned int h_distribution_size)
{
	unsigned int size_limit = 2048;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int g = 0;
	__shared__ float sdata[2048];

	if (i < h_distribution_size)
	{
		for(unsigned int i_w = 0; i_w < 2; i_w ++)
		{
			for(unsigned int i_b = 0; i_b < 2; i_b ++)
			{
				sdata[threadIdx.x + (i_w * 2 + i_b) * 512] =  pow((float)(dev_w[i_w] * dev_required[i] + dev_b[i_b] - dev_intensities[dev_required_indeces[i]]), 2.f);
			}
		}
	}
	else
	{
		for(unsigned int shift = 0; shift < size_limit; shift += 512)
		{		
			sdata[threadIdx.x + shift] = 0.f;
		}
	}
	__syncthreads();

	for (unsigned int s = 1; s < blockDim.x; s = g)
	{	
		g = (s << 1);
		unsigned int index = g * threadIdx.x;
		
		if (index + s < blockDim.x)
		{
			for(unsigned int shift = 0; shift < size_limit; shift += 512)
			{
				sdata[index + shift] += sdata[index + shift + s];
			}
			__syncthreads();
		}
		else break;
	}

	if (threadIdx.x == 0)
	{
		unsigned int index_shift = 0;
		for(unsigned int shift = 0; shift < size_limit; shift += 512)
		{
			atomicAdd((float *)&(dev_squared_norms_of_indicators[index_shift++]), sdata[shift]);
		}
	}
}


__global__ void residual_normalization_kernel(cuComplex *residual_vec, float *norm_res_vec, cuComplex *dev_orthogonal_basis)
{
	unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
	cuComplex current = residual_vec[index];
	current.x = current.x / (*norm_res_vec);
	current.y = current.y / (*norm_res_vec);
	dev_orthogonal_basis[index] = current;
}

__global__ void extend_by_zeros_kernel(bool *mask, cuComplex *usual, cuComplex *extended)
{
	unsigned int i = Q * blockIdx.x + threadIdx.x;
	unsigned int j = Q * blockIdx.y + threadIdx.y;
	unsigned int size_limit = (N << 1) - 1;
	cuComplex current;

	if ((i <  size_limit) && (j < size_limit ))
	{	
		unsigned int Ni = N * i;
		unsigned int index = Ni + j;
		unsigned int index_extended = index + Ni - i;
		if ((i < N) && (j < N) && (mask[index]))
		{
			current.x = CHI * usual[index].x;
			current.y = CHI * usual[index].y;
		}
		else
		{
			current.x = current.y = 0.f;
		}
		extended[index_extended] = current;
	}
}


__global__ void extend_by_zeros_for_gradient_kernel(cuComplex *usual, cuComplex *extended)
{
	unsigned int i = Q * blockIdx.x + threadIdx.x;
	unsigned int j = Q * blockIdx.y + threadIdx.y;
	unsigned int size_limit = (N << 1) - 1;
	cuComplex current;

	if ((i <  size_limit) && (j < size_limit ))
	{	
		unsigned int Ni = N * i;
		unsigned int index = Ni + j;
		unsigned int index_extended = index + Ni - i;
		if ((i < N) && (j < N))
		{
			current.x = CHI * usual[index].x;
			current.y = CHI * usual[index].y;
		}
		else
		{
			current.x = current.y = 0.f;
		}
		extended[index_extended] = current;
	}
}



__global__ void inner_product_float_kernel(cuComplex *vj, cuComplex *weight, cuComplex *Hjk)
{
	__shared__ cuComplex sdata[512];
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int g = 0;

	sdata[threadIdx.x].x = weight[i].x * vj[i].x  + weight[i].y * vj[i].y;
	sdata[threadIdx.x].y = weight[i].y * vj[i].x  - weight[i].x * vj[i].y;
	__syncthreads();

	for (unsigned int s = 1; s < blockDim.x; s = g)
	{	
		g = (s << 1);
		unsigned int index = g * threadIdx.x;
		
		if (index + s < blockDim.x)
		{
			atomicAdd((float *)&sdata[index].x, sdata[index + s].x);
			atomicAdd((float *)&sdata[index].y, sdata[index + s].y);
			__syncthreads();
		}
		else break;
	}

	if (threadIdx.x == 0){
		atomicAdd((float *)&(Hjk->x), (float) sdata[0].x);
		atomicAdd((float *)&(Hjk->y), (float) sdata[0].y);
	}
}


__global__ void resize_kernel(cuComplex *data, unsigned int current_size_i, unsigned int current_size_j, unsigned int new_size_j, cuComplex *resized_data)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	cuComplex zero_complex;
	zero_complex.x = 0.f;
	zero_complex.y = 0.f;
	resized_data[new_size_j * i + j] = ((i < current_size_i) && (j < current_size_j)) ? data[current_size_j * i + j] : zero_complex;
}


__global__ void Jtotal_resize_kernel(cuComplex *data, unsigned int current_size_ij, cuComplex *resized_data)
{
	unsigned int i = blockIdx.x;
	unsigned int j = blockIdx.y;
	unsigned int index_new = gridDim.y * i + j;

	if ((i < current_size_ij) && (j < current_size_ij))
	{
		int index_cur = current_size_ij * i + j;
		resized_data[index_new] = data[index_cur];
	}
	else
	{
		if ((i == gridDim.x - 1) && (i == j))
		{
			resized_data[index_new].x = 1.f;
			resized_data[index_new].y = 0.f;
		}
		else
		{
			resized_data[index_new].x = 0.f;
			resized_data[index_new].y = 0.f;
		}
	}
}


__global__ void weight_subtract_kernel(cuComplex *weight, cuComplex *Hjk, cuComplex *vj)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	cuComplex current = weight[i];
	cuComplex current_vj = vj[i];
	
	current.x -= (Hjk->x) * current_vj.x - (Hjk->y) * current_vj.y;
	current.y -= (Hjk->y) * current_vj.x + (Hjk->x) * current_vj.y;
	weight[i] = current;
}

__global__ void weight_divide_kernel(cuComplex *orthogonal_vector, cuComplex *weight, cuComplex *Hjk)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	cuComplex current;

	current = weight[i];
	current.x /= Hjk->x;
	current.y /= Hjk->x;
	orthogonal_vector[i] = current;
}


__global__ void sqrt_float_kernel(float *value)
{
	(*value) = sqrt((*value));
}

__global__ void sqrt_real_kernel(cuComplex *value) 
{
	value->x = sqrt(value->x);
	value->y = 0.f;
}

__global__ void residual_vs_tolerance_kernel(float *residual, bool *res_vs_tol, const float tolerance)
{
	(*res_vs_tol) = ((*residual) > tolerance);
}

__global__ void dublicate_kernel(cuComplex *old_complex, cuComplex *new_complex)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	new_complex[i] = old_complex[i];
}

__global__ void dublicate_mask_kernel(bool *old_bool, bool *new_bool)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	new_bool[i] = old_bool[i];
}

__global__ void check_mask_difference_kernel(bool *old_bool, bool *new_bool, unsigned int *number_of_differences)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((new_bool[i]) && (!(old_bool[i])) || (!(new_bool[i])) && (old_bool[i]))
	{
		atomicInc((unsigned int *)number_of_differences, (unsigned int)1);
	}
}

__global__ void transpose_kernel(cuComplex *dev_matrix, const unsigned int matrix_size)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;

	__shared__ cuComplex s_matrix[1024];

	if ((i < matrix_size) && (j < matrix_size))
	{
		unsigned int index_N = i * matrix_size + j;
		unsigned int index_T = i + j * matrix_size;
		unsigned int thread_UID = blockDim.y * threadIdx.x + threadIdx.y;


		s_matrix[thread_UID] = dev_matrix[index_N];

		__syncthreads();

		dev_matrix[index_T] = s_matrix[thread_UID];
	}
	else
	{
		__syncthreads();
	}
}


__global__ void init_mask(bool *mask)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	mask[index] =( (index > (N * N >> 1) - (N >> 1)  - 5 ) && (index < (N * N >> 1) - (N >> 1) + 3 ) || (index > (N * N >> 1) + (N >> 1)  - 5 ) && (index < (N * N >> 1) + (N >> 1) + 3 ) );
}

__global__ void init_mask_kernel(bool *mask)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int index = i * N + j;

	mask[index] = (i > N / 2 - 3) && (i < N / 2 + 2) && (j > N / 2 - 3) && (j < N / 2 + 2);
}

__global__ void create_Givens_rotation_matrix_kernel(cuComplex *dev_Givens_rotation, cuComplex *Htemp)
{
	unsigned int i = blockIdx.x;
	unsigned int j = blockIdx.y;
	unsigned int index = i * gridDim.y + j;
	if ((i < gridDim.x - 2) && (i == j))
	{
		dev_Givens_rotation[index].x = 1.f;
		dev_Givens_rotation[index].y = 0.f;
	}
	else
	{
		if ((i == gridDim.x - 2) && (j == gridDim.y - 2))
		{	
			unsigned int ind1 = index - i;
			unsigned int ind2 = index + 1;
			float denominator = sqrt(pow((float)Htemp[ind1].x, 2.f) + pow((float)Htemp[ind2].x, 2.f) + pow((float)Htemp[ind1].y, 2.f) + pow((float)Htemp[ind2].y, 2.f));
			dev_Givens_rotation[index].x = Htemp[ind1].x / denominator;
			dev_Givens_rotation[index].y = Htemp[ind1].y / denominator;
		}
		else
		{	
			if ((i == gridDim.x - 2) && (j == gridDim.y - 1))
			{
				unsigned int ind2 = index - j;
				float denominator = sqrt(pow((float)Htemp[index].x, 2.f) + pow((float)Htemp[ind2].x, 2.f) + pow((float)Htemp[index].y, 2.f) + pow((float)Htemp[ind2].y, 2.f));
				dev_Givens_rotation[index].x = Htemp[index].x / denominator;
				dev_Givens_rotation[index].y = Htemp[index].y / denominator;
			}
			else
			{
				if ((i == gridDim.x - 1) && (j == gridDim.y - 2))
				{
					unsigned int ind1 = index - i;
					unsigned int ind2 = ind1  - i;
					float denominator = sqrt(pow((float)Htemp[ind1].x, 2.f) + pow((float)Htemp[ind2].x, 2.f) + pow((float)Htemp[ind1].y, 2.f) + pow((float)Htemp[ind2].y, 2.f));
					dev_Givens_rotation[index].x = - Htemp[ind1].x / denominator;
					dev_Givens_rotation[index].y =   Htemp[ind1].y / denominator;
				}
				else
				{
					if ((i == gridDim.x - 1) && (j == gridDim.y - 1))
					{
						unsigned int ind2 = index - gridDim.x;
						unsigned int ind1 = ind2  - i;
						float denominator = sqrt(pow((float)Htemp[ind1].x, 2.f) + pow((float)Htemp[ind2].x, 2.f) + pow((float)Htemp[ind1].y, 2.f) + pow((float)Htemp[ind2].y, 2.f));
						dev_Givens_rotation[index].x = Htemp[ind1].x / denominator;
						dev_Givens_rotation[index].y = Htemp[ind1].y / denominator;	
					}
					else
					{
						dev_Givens_rotation[index].x = 0.f;
						dev_Givens_rotation[index].y = 0.f;
					}
				}
			}
		}
	}
}


__global__ void get_cc_kernel(cuComplex *cc, cuComplex *Jtotal, float *old_norm_res_vec)
{	
	unsigned int index = blockIdx.x * (gridDim.x + 1);

	cc[blockIdx.x].x = Jtotal[index].x * (*old_norm_res_vec);
	cc[blockIdx.x].y = Jtotal[index].y * (*old_norm_res_vec);
}


__global__ void next_residual_kernel(cuComplex *Jtotal_ij, float *norm_residual, float *actual_residual)
{
	*actual_residual =(*norm_residual) * sqrt( (pow((float)(Jtotal_ij->x), 2.0f) + pow((float)(Jtotal_ij->y), 2.0f)));
}


__global__ void rearrange_kernel(cuComplex *vec, int *pivotArray)
{
	__shared__ cuComplex sdata[1024];
	sdata[threadIdx.x] = vec[pivotArray[threadIdx.x] - 1];
	sdata[pivotArray[threadIdx.x] - 1] = vec[threadIdx.x];
	__syncthreads();
	vec[threadIdx.x] = sdata[threadIdx.x];
}

__global__ void get_new_solution_kernel(cuComplex *dev_cc, cuComplex *dev_HH)
{
	float dominant = pow((float)(dev_HH->x), 2.f) + pow((float)(dev_HH->y), 2.f);
	cuComplex current;
	current.x = (dev_cc->x * dev_HH->x + dev_cc->y * dev_HH->y) / dominant;
	current.y = (dev_cc->y * dev_HH->x - dev_cc->x * dev_HH->y) / dominant;
	(*dev_cc) = current;
}

__global__ void get_solution_kernel(cuComplex *dev_solution, cuComplex *dev_cc, cuComplex *dev_orthogonal_basis)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	cuComplex current = dev_orthogonal_basis[index];
	atomicAdd((float *)&(dev_solution[index].x), current.x * dev_cc->x - current.y * dev_cc->y);
	atomicAdd((float *)&(dev_solution[index].y), current.x * dev_cc->y + current.y * dev_cc->x);
}

__global__ void add_kernel(cuComplex *dev_solution, cuComplex *dev_add_x, cuComplex *mul)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	atomicAdd((float *)&(dev_solution[index].x), mul->x * dev_add_x[index].x - mul->y * dev_add_x[index].y);
	atomicAdd((float *)&(dev_solution[index].y), mul->y * dev_add_x[index].x + mul->x * dev_add_x[index].y);
}

__global__ void init_x0_kernel(cuComplex *input)
{
	unsigned int i = Q * blockIdx.x + threadIdx.x;
	unsigned int j = Q * blockIdx.y + threadIdx.y;
	cuComplex Input_Field;

	Input_Field.y = - WAVE_NUMBER * (i * cos(ALPHA) + j * sin(ALPHA));
	Input_Field = my_cexpf(Input_Field);
	float sigma = 400.f;
	//Input_Field.x = Input_Field.x * exp(-pow((float)((float)(j) - 512.f), 2.f)/pow(sigma, 2.f))/(sigma * sqrt(7.28f));
	//Input_Field.y = Input_Field.y * exp(-pow((float)((float)(j) - 512.f), 2.f)/pow(sigma, 2.f))/(sigma * sqrt(7.28f));
	//Input_Field.x = Input_Field.x * (exp(-pow((float)((float)(j) - 341.f), 2.f)/pow(sigma, 2.f)) + exp(-pow((float)((float)(j) - 682.f), 2.f)/pow(sigma, 2.f)))/(sigma * sqrt(7.28f));
	//Input_Field.y = Input_Field.y * (exp(-pow((float)((float)(j) - 341.f), 2.f)/pow(sigma, 2.f)) + exp(-pow((float)((float)(j) - 682.f), 2.f)/pow(sigma, 2.f)))/(sigma * sqrt(7.28f));
	input[i * N + j] = Input_Field;
}

__global__ void init_x0_by_ones_kernel(cuComplex *input)
{
	unsigned int i = Q * blockIdx.x + threadIdx.x;
	unsigned int j = Q * blockIdx.y + threadIdx.y;
	cuComplex Input_Field;

	Input_Field.x = 1.f;
	Input_Field.y = 1.f;
	input[i * N + j] = Input_Field;
}


__global__ void init_random_x0_kernel(cuComplex *input)
{
	unsigned int i = Q * blockIdx.x + threadIdx.x;
	unsigned int j = Q * blockIdx.y + threadIdx.y;
	unsigned int index = i * N + j;
	cuComplex Input_Field;
	curandState state;

	curand_init((unsigned long long)clock() + index, 0, 0, &state);

	Input_Field.y = - WAVE_NUMBER * (i * cos(ALPHA) + j * sin(ALPHA));
	Input_Field = my_cexpf(Input_Field);
	Input_Field.x = Input_Field.x * 0.5f + curand_uniform(&state) * 0.5f;
	Input_Field.y = Input_Field.y * 0.5f + curand_uniform(&state) * 0.5f;
	input[index] = Input_Field;
}


__global__ void dublicate_float_kernel(float *dev_resized, float *dev_actual_residual)
{
	dev_resized[blockIdx.x] = dev_actual_residual[blockIdx.x];
}

__global__ void dublicate_unsigned_int_kernel(unsigned int *dev_resized, unsigned int *dev_actual_residual)
{
	dev_resized[blockIdx.x] = dev_actual_residual[blockIdx.x];
}

__global__ void Green_matrix_create_Kernel(cuComplex *out)
{
	int i = Q * blockIdx.x + threadIdx.x;
	int j = Q * blockIdx.y + threadIdx.y;
	cuComplex current;
	int index = i * (2 * N - 1) + j;
	float kr_ij = WAVE_NUMBER * sqrt(pow((float)(i) - 0.5f, 2.f) + pow((float)(j), 2.f));

	current.x = -0.25f * y0(kr_ij);
	current.y =  0.25f * j0(kr_ij);
	out[index] = current;
	if ((i > 0) || (j > 0)) {
		out[(2 * N - 1) * (2 * N - 1 - i) + j] = out[(2 * N - 1) * i + (2 * N - 1 - j)] = out[(2 * N - 1) * (2 * N - 1 - i) + (2 * N - 1 - j)] = current;
	}
}


//v.1 out of 2 verisons
__global__ void fill_exclusion_zone_kernel_v1(bool *dev_exclusion_zone, bool *dev_mask, const unsigned int spread)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int index = i * N + j;

	if (dev_mask[index])
	{
		for (unsigned int ii = (i < spread) ? 0 : i - spread; ii < ((i + spread + 1 < N) ? i + spread + 1 : N); ii++)
		{
			for (unsigned int jj = (j < spread) ? 0 : j - spread; jj < ((j + spread + 1 < N) ? j + spread + 1 : N); jj++)
			{
				dev_exclusion_zone[ii * N + jj] = true;
			}
		}
	}
}


//v.2 out of 2 verisons
__global__ void fill_exclusion_zone_kernel_v2(bool *dev_exclusion_zone, bool *dev_mask, unsigned int *dev_index_of_max, const unsigned int spread)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int index = i * N + j;	
	unsigned int i_diff;
	unsigned int j_diff;

	if (dev_mask[index])
	{
		unsigned int i_of_max = *dev_index_of_max / N;
		unsigned int j_of_max = *dev_index_of_max % N;
		for (unsigned int ii = (i < spread) ? 0 : i - spread; ii < ((i + spread + 1 < N) ? i + spread + 1 : N); ii++)
		{
			for (unsigned int jj = (j < spread) ? 0 : j - spread; jj < ((j + spread + 1 < N) ? j + spread + 1 : N); jj++)
			{				
				i_diff = ii < i_of_max ? i_of_max - ii : ii - i_of_max;
				j_diff = jj < j_of_max ? j_of_max - jj : jj - j_of_max;
				if (pow((float)(i_diff), 2.f) + pow((float)(j_diff), 2.f) < pow((float)(spread + 1), 2.f))
				{
					dev_exclusion_zone[ii * N + jj] = true;
				}
			}
		}
	}
}


//replacement of exclusion_zone (first kernel) (v.3)
__global__ void min_reduce_first_kernel(bool *dev_mask, unsigned int *dev_i_indeces)
{
	__shared__ unsigned int sh_i_indeces[1024];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int index_big = i * gridDim.y * blockDim.y + j;
	unsigned int g = 1;
	unsigned int thread_UID = blockDim.y * threadIdx.x + threadIdx.y;
	unsigned int limit_local = blockDim.x * blockDim.y;
	
	sh_i_indeces[thread_UID] = dev_mask[index_big] ? i : N;

	__syncthreads();
	for (unsigned int s = 1; s < limit_local ; s = g)
	{	
		g = (s << 1);
		unsigned int index = g * thread_UID;
		
		if (index + s < limit_local)
		{
			sh_i_indeces[index] = sh_i_indeces[index] > sh_i_indeces[index + s] ? sh_i_indeces[index + s] : sh_i_indeces[index];
			__syncthreads();
		}
		else break;
	}
	if (thread_UID == 0) dev_i_indeces[blockIdx.x * gridDim.y + blockIdx.y] = sh_i_indeces[0];
}

//replacement of exclusion_zone (second kernel) (v.3)
__global__ void min_reduce_second_kernel(unsigned int *dev_i_indeces)
{
	__shared__ unsigned int sh_i_indeces[1024];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int index_big = i * gridDim.y * blockDim.y + j;
	unsigned int g = 1;
	unsigned int thread_UID = blockDim.y * threadIdx.x + threadIdx.y;
	unsigned int limit_local = blockDim.x * blockDim.y;
	
	sh_i_indeces[thread_UID] = dev_i_indeces[index_big];

	__syncthreads();
	for (unsigned int s = 1; s < limit_local; s = g)
	{	
		g = (s << 1);
		unsigned int index = g * thread_UID;
		
		if (index + s < limit_local)
		{
			sh_i_indeces[index] = sh_i_indeces[index] > sh_i_indeces[index + s] ? sh_i_indeces[index + s] : sh_i_indeces[index];
			__syncthreads();
		}
		else break;
	}
	if (thread_UID == 0) dev_i_indeces[blockIdx.x * gridDim.y + blockIdx.y] = sh_i_indeces[0];
}

/*
__global__ void check_max_isolation_kernel(bool *dev_mask, const unsigned int *dev_index_of_max, const unsigned int spread, const unsigned int check_number, bool *max_is_not_isolated)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int i_of_max = *dev_index_of_max / N;
	unsigned int j_of_max = *dev_index_of_max % N;
	unsigned int i_diff = i < i_of_max ? i_of_max - i : i - i_of_max;
	unsigned int j_diff = j < j_of_max ? j_of_max - j : j - j_of_max;

	if (pow((float)(i_diff), 2.f) + pow((float)(j_diff), 2.f) == pow((float)(spread + 1 + check_number), 2.f))
	{
		if (!dev_mask[i * N + j])
		{
			*max_is_not_isolated = true;
		}
	}
}
*/


//v.1 out of 2 versions (max reduce first kernel)
__global__ void max_reduce_first_kernel_v1(cuComplex *g_idata, float *dev_intensity)
{
	__shared__ float sdata[512];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int g = 0;

	sdata[threadIdx.x] = sqrt(pow((float)g_idata[i].x, 2.f) + pow((float)g_idata[i].y, 2.f));
	__syncthreads();
	for (unsigned int s = 1; s < blockDim.x; s = g)
	{	
		g = (s << 1);
		int index = g * threadIdx.x;
		
		if (index + s < blockDim.x)
		{
			sdata[index] = sdata[index] > sdata[index + s] ? sdata[index] : sdata[index + s];
			__syncthreads();
		}
		else break;
	}
	if (threadIdx.x == 0) dev_intensity[blockIdx.x] = sdata[0];
}

//v.1 out of 3 versions (max reduce second kernel)
__global__ void max_reduce_second_kernel_v1(float *dev_intensity)
{
	__shared__ float sdata[512];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int g = 0;

	sdata[threadIdx.x] = dev_intensity[i];
	__syncthreads();
	for (unsigned int s = 1; s < blockDim.x; s = g)
	{	
		g = (s << 1);
		int index = g * threadIdx.x;
	
		if (index + s < blockDim.x)
		{
			sdata[index] = sdata[index] > sdata[index + s] ? sdata[index] : sdata[index + s];
			__syncthreads();
		}
		else break;
	}
	if (threadIdx.x == 0) dev_intensity[blockIdx.x] = sdata[0];
}


//v.2 out of 3 versions (max reduce first kernel)
__global__ void max_reduce_first_kernel_v2(bool *dev_exclusion_zone, cuComplex *g_idata, float *dev_intensity)
{
	__shared__ float sdata[512];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int g = 0;

	sdata[threadIdx.x] = dev_exclusion_zone[i] ? 0.f : sqrt(pow((float)g_idata[i].x, 2.f) + pow((float)g_idata[i].y, 2.f));
	__syncthreads();
	for (unsigned int s = 1; s < blockDim.x; s = g)
	{	
		g = (s << 1);
		unsigned int index = g * threadIdx.x;
		
		if (index + s < blockDim.x)
		{
			sdata[index] = sdata[index] > sdata[index + s] ? sdata[index] : sdata[index + s];
			__syncthreads();
		}
		else break;
	}
	if (threadIdx.x == 0) dev_intensity[blockIdx.x] = sdata[0];
}

//v.2 out of 3 versions (max reduce second kernel)
__global__ void max_reduce_second_kernel_v2(float *dev_intensity)
{
	__shared__ float sdata[512];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int g = 0;

	sdata[threadIdx.x] = dev_intensity[i];
	__syncthreads();
	for (unsigned int s = 1; s < blockDim.x; s = g)
	{	
		g = (s << 1);
		unsigned int index = g * threadIdx.x;
	
		if (index + s < blockDim.x)
		{
			sdata[index] = sdata[index] > sdata[index + s] ? sdata[index] : sdata[index + s];
			__syncthreads();
		}
		else break;
	}
	if (threadIdx.x == 0) dev_intensity[blockIdx.x] = sdata[0];
}


//v.3 out of 3 versions (max reduce first kernel)
__global__ void max_reduce_first_kernel_v3(unsigned int *dev_i_threshold, cuComplex *g_idata, float *dev_intensity)
{
	__shared__ float sdata[1024];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int index_big = i * gridDim.y * blockDim.y + j;
	unsigned int g = 0;
	unsigned int thread_UID = blockDim.y * threadIdx.x + threadIdx.y;
	unsigned int limit_local = blockDim.x * blockDim.y;

	sdata[thread_UID] = (i > *dev_i_threshold) ? 0.f : sqrt(pow((float)g_idata[index_big].x, 2.f) + pow((float)g_idata[index_big].y, 2.f));
	__syncthreads();
	for (unsigned int s = 1; s < limit_local; s = g)
	{	
		g = (s << 1);
		unsigned int index = g * thread_UID;
		
		if (index + s < limit_local)
		{
			sdata[index] = sdata[index] > sdata[index + s] ? sdata[index] : sdata[index + s];
			__syncthreads();
		}
		else break;
	}
	if (thread_UID == 0) dev_intensity[blockIdx.x * gridDim.y + blockIdx.y] = sdata[0];
}

//v.3 out of 3 versions (max reduce second kernel)
__global__ void max_reduce_second_kernel_v3(float *dev_intensity)
{
	__shared__ float sdata[1024];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int index_big = i * gridDim.y * blockDim.y + j;
	unsigned int g = 0;
	unsigned int thread_UID = blockDim.y * threadIdx.x + threadIdx.y;
	unsigned int limit_local = blockDim.x * blockDim.y;

	sdata[thread_UID] = dev_intensity[index_big];
	__syncthreads();
	for (unsigned int s = 1; s < limit_local; s = g)
	{	
		g = (s << 1);
		unsigned int index = g * thread_UID;
	
		if (index + s < limit_local)
		{
			sdata[index] = sdata[index] > sdata[index + s] ? sdata[index] : sdata[index + s];
			__syncthreads();
		}
		else break;
	}
	if (thread_UID == 0) dev_intensity[blockIdx.x * gridDim.y + blockIdx.y] = sdata[0];
}


__global__ void min_norm_reduce_w_b_kernel(float *dev_squared_norms_of_indicators, float *dev_w, float *dev_b, bool *dev_to_exit, const float tolerance)
{
	__shared__ float sdata[4];
	__shared__ unsigned int s_IDs[4];
	unsigned int g = 0;
	bool comparison;

	sdata[threadIdx.x] = dev_squared_norms_of_indicators[threadIdx.x];
	s_IDs[threadIdx.x] = threadIdx.x;
	__syncthreads();

	for (unsigned int s = 1; s < 4; s = g)
	{	
		g = (s << 1);
		unsigned int index = g * threadIdx.x;
	
		if (index + s < 4)
		{
			comparison = sdata[index] < sdata[index + s];
			sdata[index] = comparison ? sdata[index] : sdata[index + s];
			s_IDs[index] = comparison ? s_IDs[index] : s_IDs[index + s];
			__syncthreads();
		}
		else break;
	}

	switch(threadIdx.x)
	{
		case 0:
		{
			unsigned int i_w_best = s_IDs[0] >> 1;
			float new_w_distance = (dev_w[1] - dev_w[0]) / 4.f;
			if(i_w_best)
			{
				dev_w[0] = dev_w[1] + new_w_distance;
				dev_w[1] = dev_w[1] - new_w_distance;
			}
			else
			{
				dev_w[1] = dev_w[0] - new_w_distance;
				dev_w[0] = dev_w[0] + new_w_distance;				
			}
			break;
		}
		case 1:
		{
			unsigned int i_b_best = s_IDs[0] % 2;
			float new_b_distance = (dev_b[1] - dev_b[0]) / 4.f;
			if(i_b_best)
			{
				dev_b[0] = dev_b[1] + new_b_distance;
				dev_b[1] = dev_b[1] - new_b_distance;
			}
			else
			{
				dev_b[1] = dev_b[0] - new_b_distance;
				dev_b[0] = dev_b[0] + new_b_distance;				
			}
			break;
		}
		case 2:
		{
			*dev_to_exit = (dev_b[1] - dev_b[0] < tolerance) && (dev_w[1] - dev_w[0]< tolerance);
		}
	}
}


//v.1 (of first kernel) out of 2 versions
__global__ void max_reduce_first_indexed_kernel_v1(bool *dev_exclusion_zone, cuComplex *g_idata, float *dev_intensity, unsigned int *dev_indeces)
{
	__shared__ float sh_intensities[512];
	__shared__ unsigned int sh_indeces[512];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int g = 0;
	bool comparison;

	sh_intensities[threadIdx.x] = dev_exclusion_zone[i] ? 0.f : sqrt(pow((float)g_idata[i].x, 2.f) + pow((float)g_idata[i].y, 2.f));
	sh_indeces[threadIdx.x] = i;
	__syncthreads();
	for (unsigned int s = 1; s < blockDim.x; s = g)
	{	
		g = (s << 1);
		unsigned int index = g * threadIdx.x;
		
		if (index + s < blockDim.x)
		{
			comparison = sh_intensities[index] > sh_intensities[index + s];
			sh_indeces[index] =     comparison ? sh_indeces[index]     : sh_indeces[index + s];
			sh_intensities[index] = comparison ? sh_intensities[index] : sh_intensities[index + s];
			__syncthreads();
		}
		else break;
	}
	if (threadIdx.x == 0)
	{
		dev_intensity[blockIdx.x] = sh_intensities[0];
		dev_indeces[blockIdx.x] = sh_indeces[0];
	}
}


//v.1 (of second kernel) out of 2 versions
__global__ void max_reduce_second_indexed_kernel_v1(float *dev_intensity, unsigned int *dev_indeces)
{
	__shared__ float sh_intensities[512];
	__shared__ unsigned int sh_indeces[512];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int g = 0;
	bool comparison;

	sh_intensities[threadIdx.x] = dev_intensity[i];
	sh_indeces[threadIdx.x] = dev_indeces[i];
	__syncthreads();
	for (unsigned int s = 1; s < blockDim.x; s = g)
	{	
		g = (s << 1);
		unsigned int index = g * threadIdx.x;
	
		if (index + s < blockDim.x)
		{
			comparison = sh_intensities[index] > sh_intensities[index + s];
			sh_indeces[index] =     comparison ? sh_indeces[index]     : sh_indeces[index + s];
			sh_intensities[index] = comparison ? sh_intensities[index] : sh_intensities[index + s];
			__syncthreads();
		}
		else break;
	}
	if (threadIdx.x == 0)
	{
		dev_intensity[blockIdx.x] = sh_intensities[0];
		dev_indeces[blockIdx.x] = sh_indeces[0];
	}
}



//v.2 (of first kernel) out of 2 versions
__global__ void max_reduce_first_indexed_kernel_v2(unsigned int *dev_exclusion_zone, cuComplex *g_idata, float *dev_intensity, unsigned int *dev_indeces)
{
	__shared__ float sh_intensities[1024];
	__shared__ unsigned int sh_indeces[1024];
	bool comparison;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int index_big = i * gridDim.y * blockDim.y + j;
	unsigned int g = 0;
	unsigned int thread_UID = blockDim.y * threadIdx.x + threadIdx.y;
	unsigned int limit_local = blockDim.x * blockDim.y;


	sh_intensities[thread_UID] = dev_exclusion_zone[index_big] ? 0.f : sqrt(pow((float)g_idata[index_big].x, 2.f) + pow((float)g_idata[index_big].y, 2.f));
	sh_indeces[thread_UID] = index_big;
	__syncthreads();
	for (unsigned int s = 1; s < limit_local; s = g)
	{	
		g = (s << 1);
		unsigned int index = g * thread_UID;
		
		if (index + s < limit_local)
		{
			comparison = sh_intensities[index] > sh_intensities[index + s];
			sh_indeces[index] =     comparison ? sh_indeces[index]     : sh_indeces[index + s];
			sh_intensities[index] = comparison ? sh_intensities[index] : sh_intensities[index + s];
			__syncthreads();
		}
		else break;
	}
	if (thread_UID == 0)
	{
		unsigned int block_UID = blockIdx.x * gridDim.y + blockIdx.y;
		dev_intensity[block_UID] = sh_intensities[0];
		dev_indeces[block_UID] = sh_indeces[0];
	}
}


//v.2 (of second kernel) out of 2 versions
__global__ void max_reduce_second_indexed_kernel_v2(float *dev_intensity, unsigned int *dev_indeces)
{
	__shared__ float sh_intensities[1024];
	__shared__ unsigned int sh_indeces[1024];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int index_big = i * gridDim.y * blockDim.y + j;
	unsigned int g = 0;
	unsigned int thread_UID = blockDim.y * threadIdx.x + threadIdx.y;
	unsigned int limit_local = blockDim.x * blockDim.y;
	bool comparison;

	sh_intensities[thread_UID] = dev_intensity[index_big];
	sh_indeces[thread_UID] = dev_indeces[index_big];
	__syncthreads();
	for (unsigned int s = 1; s < limit_local; s = g)
	{	
		g = (s << 1);
		unsigned int index = g * thread_UID;
	
		if (index + s < limit_local)
		{
			comparison = sh_intensities[index] > sh_intensities[index + s];
			sh_indeces[index] =     comparison ? sh_indeces[index]     : sh_indeces[index + s];
			sh_intensities[index] = comparison ? sh_intensities[index] : sh_intensities[index + s];
			__syncthreads();
		}
		else break;
	}
	if (thread_UID == 0)
	{
		unsigned int block_UID = blockIdx.x * gridDim.y + blockIdx.y;
		dev_intensity[block_UID] = sh_intensities[0];
		dev_indeces[block_UID] = sh_indeces[0];
	}
}


//(v.3)
__global__ void max_reduce_first_specified_zone_kernel_v1(unsigned int *dev_required_intensity_distribution_indeces, float *dev_intensity, float *dev_max_intensities, const unsigned int h_distribution_size)
{
	__shared__ float sh_intensities[1024];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int index_big = i * gridDim.y * blockDim.y + j;
	unsigned int g = 0;
	unsigned int thread_UID = blockDim.y * threadIdx.x + threadIdx.y;
	unsigned int limit_local = blockDim.x * blockDim.y;

	if (index_big < h_distribution_size)
	{
		index_big = dev_required_intensity_distribution_indeces[index_big];	
		sh_intensities[thread_UID] = dev_intensity[index_big];
	}
	else
	{
		sh_intensities[thread_UID] = 0.f;
	}

	__syncthreads();
	for (unsigned int s = 1; s < limit_local; s = g)
	{	
		g = (s << 1);
		unsigned int index = g * thread_UID;
		
		if (index + s < limit_local)
		{
			sh_intensities[index] = sh_intensities[index] > sh_intensities[index + s] ? sh_intensities[index] : sh_intensities[index + s];
			__syncthreads();
		}
		else break;
	}
	if (thread_UID == 0)
	{
		dev_max_intensities[blockIdx.x * gridDim.y + blockIdx.y] = sh_intensities[0];
	}
}

//(v.3) + (v.4)
__global__ void max_reduce_second_specified_zone_kernel_v1(float *dev_max_intensities)
{
	__shared__ float sh_intensities[1024];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int index_big = i * gridDim.y * blockDim.y + j;
	unsigned int g = 0;
	unsigned int thread_UID = blockDim.y * threadIdx.x + threadIdx.y;
	unsigned int limit_local = blockDim.x * blockDim.y;

	sh_intensities[thread_UID] = dev_max_intensities[index_big];
	__syncthreads();

	for (unsigned int s = 1; s < limit_local; s = g)
	{	
		g = (s << 1);
		unsigned int index = g * thread_UID;
	
		if (index + s < limit_local)
		{
			sh_intensities[index] = sh_intensities[index] > sh_intensities[index + s] ? sh_intensities[index] : sh_intensities[index + s];
			__syncthreads();
		}
		else break;
	}
	if (thread_UID == 0)
	{
		dev_max_intensities[blockIdx.x * gridDim.y + blockIdx.y] = sh_intensities[0];
	}
}

//(v.4)
__global__ void max_reduce_first_indicators_kernel_v1(float *dev_indicators, float *dev_max_of_indicators, const unsigned int h_distribution_size)
{
	__shared__ float sh_intensities[1024];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int index_big = i * gridDim.y * blockDim.y + j;
	unsigned int g = 0;
	unsigned int thread_UID = blockDim.y * threadIdx.x + threadIdx.y;
	unsigned int limit_local = blockDim.x * blockDim.y;

	sh_intensities[thread_UID] = index_big < h_distribution_size ? fabs(dev_indicators[index_big]) : 0.f;
	__syncthreads();

	for (unsigned int s = 1; s < limit_local; s = g)
	{	
		g = (s << 1);
		unsigned int index = g * thread_UID;
		
		if (index + s < limit_local)
		{
			sh_intensities[index] = sh_intensities[index] > sh_intensities[index + s] ? sh_intensities[index] : sh_intensities[index + s];
			__syncthreads();
		}
		else break;
	}
	if (thread_UID == 0)
	{
		dev_max_of_indicators[blockIdx.x * gridDim.y + blockIdx.y] = sh_intensities[0];
	}
}


__global__ void max_reduce_first_indexed_kernel_v5(float *dev_gradient, unsigned int *dev_indeces)
{
	__shared__ float sh_grads[1024];
	__shared__ unsigned int sh_indeces[1024];
	bool comparison;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int index_big = i * gridDim.y * blockDim.y + j;
	unsigned int g = 0;
	unsigned int thread_UID = blockDim.y * threadIdx.x + threadIdx.y;
	unsigned int limit_local = blockDim.x * blockDim.y;


	sh_grads[thread_UID] = dev_gradient[index_big];
	sh_indeces[thread_UID] = index_big;
	__syncthreads();
	for (unsigned int s = 1; s < limit_local; s = g)
	{	
		g = (s << 1);
		unsigned int index = g * thread_UID;
		
		if (index + s < limit_local)
		{
			comparison = sh_grads[index] > sh_grads[index + s];
			sh_indeces[index] =     comparison ? sh_indeces[index]	: sh_indeces[index + s];
			sh_grads[index] = 	comparison ? sh_grads[index] 	: sh_grads[index + s];
			__syncthreads();
		}
		else break;
	}
	if (thread_UID == 0)
	{
		unsigned int block_UID = blockIdx.x * gridDim.y + blockIdx.y;
		dev_gradient[block_UID] = sh_grads[0];
		dev_indeces[block_UID] = sh_indeces[0];
	}
}


//(v.1)
__global__ void min_reduce_first_specified_zone_kernel_v1(float *dev_required_intensity_distribution, float *dev_required_min, const unsigned int h_distribution_size)
{
	__shared__ float sh_requiredments[1024];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int index_big = i * gridDim.y * blockDim.y + j;
	unsigned int g = 0;
	unsigned int thread_UID = blockDim.y * threadIdx.x + threadIdx.y;
	unsigned int limit_local = blockDim.x * blockDim.y;

	sh_requiredments[thread_UID] = index_big < h_distribution_size ? dev_required_intensity_distribution[index_big] : 1.f;
	__syncthreads();

	for (unsigned int s = 1; s < limit_local; s = g)
	{	
		g = (s << 1);
		unsigned int index = g * thread_UID;
		
		if (index + s < limit_local)
		{
			sh_requiredments[index] = sh_requiredments[index] < sh_requiredments[index + s] ? sh_requiredments[index] : sh_requiredments[index + s];
			__syncthreads();
		}
		else break;
	}
	if (thread_UID == 0)
	{
		dev_required_min[blockIdx.x * gridDim.y + blockIdx.y] = sh_requiredments[0];
	}
}

//(v.1)
__global__ void min_reduce_second_specified_zone_kernel_v1(float *dev_required_min)
{
	__shared__ float sh_requiredments[1024];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int index_big = i * gridDim.y * blockDim.y + j;
	unsigned int g = 0;
	unsigned int thread_UID = blockDim.y * threadIdx.x + threadIdx.y;
	unsigned int limit_local = blockDim.x * blockDim.y;

	sh_requiredments[thread_UID] = dev_required_min[index_big];
	__syncthreads();
	for (unsigned int s = 1; s < limit_local; s = g)
	{	
		g = (s << 1);
		unsigned int index = g * thread_UID;
	
		if (index + s < limit_local)
		{
			sh_requiredments[index] = sh_requiredments[index] < sh_requiredments[index + s] ? sh_requiredments[index] : sh_requiredments[index + s];
			__syncthreads();
		}
		else break;
	}
	if (thread_UID == 0)
	{
		dev_required_min[blockIdx.x * gridDim.y + blockIdx.y] = sh_requiredments[0];
	}
}



__global__ void get_neighbour_indices(unsigned int *dev_neighbor_indeces, bool *dev_mask)
{
	unsigned int i = blockIdx.x * Q + threadIdx.x;
	unsigned int j = blockIdx.y * Q + threadIdx.y;
	unsigned int index = i * N + j;	

	dev_neighbor_indeces[index] = ((!dev_mask[index]) && (((j + 1 < N) && dev_mask[index + 1]) || ((j  > 0) && dev_mask[index - 1]) || ((i + 1 < N) && dev_mask[index + N]) || ((i  > 0) && dev_mask[index - N]))) ? index : 0;
}


//v.1 out of 4 versions
__global__ void get_neighbour_indices_for_gradient_v1(bool *dev_mask, float *dev_gradient, unsigned int *check_sum)
{
	unsigned int i = blockIdx.x * Q + threadIdx.x;
	unsigned int j = blockIdx.y * Q + threadIdx.y;
	unsigned int index = i * N + j;	

	if (((j + 1 < N) && dev_mask[index + 1]) || ((j  > 0) && dev_mask[index - 1]) || ((i + 1 < N) && dev_mask[index + N]) || ((i  > 0) && dev_mask[index - N]))
	{
		if ((!dev_mask[index]) && (dev_gradient[index]>0))
		{
			dev_mask[index] = true;
		}
		else
		{
			if ((dev_mask[index]) && (dev_gradient[index]<0))
			{
				dev_mask[index] = false;
			}
			else
			{
				atomicAdd((unsigned int *)check_sum, (unsigned int)1);
			}
		}
	}
	else
	{
		atomicAdd((unsigned int *)check_sum, (unsigned int)1);
	}
}

//v.2 out of 4 versions
__global__ void get_neighbour_indices_for_gradient_v2(bool *dev_mask, float *dev_gradient, unsigned int *check_sum, unsigned int *dev_index_of_max, const unsigned int spread)
{
	unsigned int i = blockIdx.x * Q + threadIdx.x;
	unsigned int j = blockIdx.y * Q + threadIdx.y;
	unsigned int index = i * N + j;	

	if (((j + 1 < N) && dev_mask[index + 1]) || ((j  > 0) && dev_mask[index - 1]) || ((i + 1 < N) && dev_mask[index + N]) || ((i  > 0) && dev_mask[index - N]))
	{
		unsigned int i_of_max = *dev_index_of_max / N;
		unsigned int j_of_max = *dev_index_of_max % N;
		if ((i_of_max < i + spread + 1) && (i < i_of_max + spread + 1) && (j_of_max < j + spread + 1) && (j < j_of_max + spread + 1))
		{
			dev_mask[index] = false;
			atomicAdd((unsigned int *)check_sum, (unsigned int)1);
		}
		else
		{

			if ((!dev_mask[index]) && (dev_gradient[index]>0))
			{
				dev_mask[index] = true;
			}
			else
			{
				if ((dev_mask[index]) && (dev_gradient[index]<0))
				{
					dev_mask[index] = false;
				}
				else
				{
					atomicAdd((unsigned int *)check_sum, (unsigned int)1);
				}
			}
		}
	}
	else
	{
		atomicAdd((unsigned int *)check_sum, (unsigned int)1);
	}
}

//v.3 out of 4 versions
__global__ void get_neighbour_indices_for_gradient_v3(bool *dev_mask, float *dev_gradient, unsigned int *check_sum, unsigned int *dev_index_of_max, const unsigned int spread)
{
	unsigned int i = blockIdx.x * Q + threadIdx.x;
	unsigned int j = blockIdx.y * Q + threadIdx.y;
	unsigned int index = i * N + j;	

	if (((j + 1 < N) && dev_mask[index + 1]) || ((j  > 0) && dev_mask[index - 1]) || ((i + 1 < N) && dev_mask[index + N]) || ((i  > 0) && dev_mask[index - N]))
	{

		unsigned int i_of_max = *dev_index_of_max / N;
		unsigned int j_of_max = *dev_index_of_max % N;

		unsigned int i_diff = i < i_of_max ? i_of_max - i : i - i_of_max;
		unsigned int j_diff = j < j_of_max ? j_of_max - j : j - j_of_max;

		if (pow((float)(i_diff), 2.f) + pow((float)(j_diff), 2.f) < pow((float)(spread + 1), 2.f))
		{
			dev_mask[index] = false;
			atomicAdd((unsigned int *)check_sum, (unsigned int)1);
		}
		else
		{

			if ((!dev_mask[index]) && (dev_gradient[index]>0))
			{
				dev_mask[index] = true;
			}
			else
			{
				if ((dev_mask[index]) && (dev_gradient[index]<0))
				{
					dev_mask[index] = false;
				}
				else
				{
					atomicAdd((unsigned int *)check_sum, (unsigned int)1);
				}
			}
		}
	}
	else
	{
		atomicAdd((unsigned int *)check_sum, (unsigned int)1);
	}
}


//v.4 out of 4 versions
__global__ void get_neighbour_indices_for_gradient_v4(bool *dev_mask, float *dev_gradient, unsigned int *check_sum, unsigned int *dev_index_of_max, const unsigned int spread)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.x + threadIdx.y;
	unsigned int index = i * N + j;	

	if (((j + 1 < N) && dev_mask[index + 1]) || ((j  > 0) && dev_mask[index - 1]) || ((i + 1 < N) && dev_mask[index + N]) || ((i  > 0) && dev_mask[index - N]))
	{

		unsigned int i_of_max = *dev_index_of_max / N;

		if (i < i_of_max + spread + 1)
		{
			dev_mask[index] = false;
			atomicAdd((unsigned int *)check_sum, (unsigned int)1);
		}
		else
		{

			if ((!dev_mask[index]) && (dev_gradient[index]>0))
			{
				dev_mask[index] = true;
			}
			else
			{
				if ((dev_mask[index]) && (dev_gradient[index]<0))
				{
					dev_mask[index] = false;
				}
				else
				{
					atomicAdd((unsigned int *)check_sum, (unsigned int)1);
				}
			}
		}
	}
	else
	{
		atomicAdd((unsigned int *)check_sum, (unsigned int)1);
	}
}


__global__ void get_neighbour_indices_for_gradient_multiple_maximums_v1(bool *dev_mask, bool *dev_grad_bool_to_add, bool *dev_grad_bool_to_sub, unsigned int *check_sum, const unsigned int i_line, const unsigned int spread)
{
	unsigned int i = blockIdx.x * Q + threadIdx.x;
	unsigned int j = blockIdx.y * Q + threadIdx.y;
	unsigned int index = i * N + j;

	if (((j + 1 < N) && dev_mask[index + 1]) || ((j  > 0) && dev_mask[index - 1]) || ((i + 1 < N) && dev_mask[index + N]) || ((i  > 0) && dev_mask[index - N]))
	{
		if (i < i_line + spread + 1)
		{
			dev_mask[index] = false;
			atomicAdd((unsigned int *)check_sum, (unsigned int)1);
		}
		else
		{

			if ((!dev_mask[index]) && (dev_grad_bool_to_add[index]))
			{
				dev_mask[index] = true;
			}
			else
			{
				if ((dev_mask[index]) && (dev_grad_bool_to_sub[index]))
				{
					dev_mask[index] = false;
				}
				else
				{
					atomicAdd((unsigned int *)check_sum, (unsigned int)1);
				}
			}
		}
	}
	else
	{
		atomicAdd((unsigned int *)check_sum, (unsigned int)1);
	}
}


__global__ void get_neighbour_next(unsigned int *dev_neighbor, unsigned int *dev_neighbor_indeces, bool *dev_mask, bool *dev_stop)
{
	if (!*dev_stop)
	{
		unsigned int current = *dev_neighbor;
		if (current) 
		{
			dev_mask[current] = false;
			current++;
		}
		while ((current < N * N) && (!dev_neighbor_indeces[current]))
			current ++;
		if (current >  N*N - 1)
		{
			*dev_stop = true;
		}
		else
		{
			dev_mask[current] = true;
			*dev_neighbor = current;
		}
	}
}

__global__ void get_neighbour_subtractive_next_kernel(unsigned int *dev_neighbor, bool *dev_mask, bool *dev_stop)
{
	if (!*dev_stop)
	{
		unsigned int current = *dev_neighbor;
		if (current) 
		{
			dev_mask[current] = true;
			current++;
		}
		while ((current < N * N) && (!dev_mask[current]))
			current ++;
		if (current >  N * N - 1)
		{
			*dev_stop = true;
		}
		else
		{
			dev_mask[current] = false;
			*dev_neighbor = current;
		}
	}
}

__global__ void target_function_compare(float *another_intensity, float *optimal_intensity, unsigned int *curr_neighbour, unsigned int *optimal_neighbour)
{
	bool comparison = (*another_intensity >= *optimal_intensity);
	*optimal_neighbour = comparison ? *curr_neighbour : *optimal_neighbour;
	*optimal_intensity = comparison ? *another_intensity : *optimal_intensity;	
}

__global__ void check_maximum_kernel(float *maxumim_global, float *maximum_optimal, bool *check_optimal)
{
	*check_optimal = (*maximum_optimal >= *maxumim_global);
	if (*check_optimal) *maxumim_global = *maximum_optimal;
}

__global__ void get_suboptimal_mask_kernel(bool *dev_mask, unsigned int *neighbour_optimal)
{
	dev_mask[*neighbour_optimal] = true;
}

__global__ void get_suboptimal_mask_subtractive_kernel(bool *dev_mask, unsigned int *neighbour_optimal)
{
	dev_mask[*neighbour_optimal] = false;
}


/*
__global__ void get_index_of_max_kernel(float *intensity_maximum, cuComplex *dev_solution, unsigned int *index_of_max)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	cuComplex current = dev_solution[index];
	if (*intensity_maximum == sqrt(pow((float)current.x, 2.f) + pow((float)current.y, 2.f)))
	{
		*index_of_max = index;
		printf("SOM\n\n");
	}
}
*/ //Does not work properly

__global__ void get_y_kernel(bool *dev_y, unsigned int *index_of_max)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	dev_y[index] = (*index_of_max == index);
}


__global__ void set_n_maximums_kernel(unsigned int *dev_indeces_positions, const unsigned int n_maximums, const unsigned int i_line)
{
	dev_indeces_positions[blockIdx.x] = i_line * N + (blockIdx.x + 1) * N/(n_maximums + 1);
}

//(v.1)
__global__ void dev_grad_bool_kernel(bool *dev_grad_bool_to_add, bool *dev_grad_bool_to_sub, float *dev_gradient)
{
	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (dev_gradient[index] < 0)
	{
		dev_grad_bool_to_add[index] = false;
	}
	else
	{
		dev_grad_bool_to_sub[index] = false;
	}
}


//(v.2)
__global__ void dev_grad_bool_indicated_kernel(float *dev_indicators_ij, bool *dev_grad_bool_to_add, bool *dev_grad_bool_to_sub, float *dev_gradient)
{
	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (*dev_indicators_ij > 0)
	{
		if (dev_gradient[index] < 0)
		{
			dev_grad_bool_to_add[index] = false;
		}
		else
		{
			dev_grad_bool_to_sub[index] = false;
		}
	}
	else
	{		
		if (dev_gradient[index] < 0)
		{
			dev_grad_bool_to_sub[index] = false;
		}
		else
		{
			dev_grad_bool_to_add[index] = false;
		}
	}
}

__global__ void get_intensities_in_maximums_kernel(unsigned int *dev_indeces_positions, cuComplex *dev_solution, float *dev_intensities_in_maxs)
{
	unsigned int subindex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int index = dev_indeces_positions[subindex];
	dev_intensities_in_maxs[subindex] = sqrt(pow((float)dev_solution[index].x, 2.f) + pow((float)dev_solution[index].y, 2.f));
}


__global__ void init_w_b_kernel(float *dev_w, float *dev_b, float *dev_max_intensity_specified_zone, float *dev_min_intensity_dist_required, const float tolerance)
{
	switch(blockIdx.x)
	{
		case 0:
		{
			if  (*dev_min_intensity_dist_required < tolerance)
			{
				dev_w[0] = *dev_max_intensity_specified_zone * 25.f;
			}
			else
			{
				dev_w[0] = *dev_max_intensity_specified_zone / *dev_min_intensity_dist_required * 0.25f;					
			}
			break;
		}
		case 1:
		{
			if  (*dev_min_intensity_dist_required < tolerance)
			{
				dev_w[1] = *dev_max_intensity_specified_zone * 75.f;
			}
			else
			{
				dev_w[1] = *dev_max_intensity_specified_zone / *dev_min_intensity_dist_required * 0.75f;					
			}
			break;
		}
		case 2:
		{
			dev_b[0] = *dev_max_intensity_specified_zone * 0.25f;
			break;
		}
		case 3:
		{
			dev_b[1] = *dev_max_intensity_specified_zone * 0.75f;
		}

	}
}

__global__ void intensity_kernel(cuComplex *dev_solution, float *dev_intensities)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	dev_intensities[index] = sqrt(pow((float)dev_solution[index].x, 2.f) + pow((float)dev_solution[index].y, 2.f));
}



__global__ void indicators_kernel(float *dev_w, float *dev_b, float *dev_required, unsigned int *required_indeces, float *dev_intensities,  float *dev_indicators, const unsigned int h_distribution_size)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;	
	if (index < h_distribution_size)
	{
		dev_indicators[index] = *dev_w * dev_required[required_indeces[index]] + *dev_b - dev_intensities[index];
	}
}


//use ((|ID| < |ID_max| * (1 - epsilon_abs_ID)) || (|ID_max| < delta_abs_ID)) notation
__global__ void compare_abs_indicators_kernel(float *dev_indicators_ij, float *dev_max_of_indicators, bool *dev_to_be_optimized, const float epsilon_abs_ID, const float delta_abs_ID)
{
	switch(blockIdx.x)
	{
		case 0:
		{
			if (fabs(*dev_indicators_ij) > fabs(*dev_max_of_indicators) * (1.f - epsilon_abs_ID))
			{
				*dev_to_be_optimized = true;
			}
			break;
		}
		case 1:
		{
			if (fabs(*dev_max_of_indicators) < delta_abs_ID)
			{
				*dev_to_be_optimized = true;
			}			
		}
	}
}

__global__ void double_expand_cuComplex_kernel(const cuComplex *input_old, cuComplex *output_new, const unsigned int N_old)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int index_old = i * N_old + j;
	unsigned int N_new = N_old << 1;
	unsigned int index_new = (N_new * i + j) << 1;

	output_new[index_new] = output_new[index_new + 1] = output_new[index_new + N_new] = output_new[index_new + 1 + N_new] = input_old[index_old];	
}


__global__ void double_expand_float_kernel(const float *input_old, float *output_new, const unsigned int N_old)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int index_old = i * N_old + j;
	unsigned int N_new = N_old << 1;
	unsigned int index_new = (N_new * i + j) << 1;

	output_new[index_new] = output_new[index_new + 1] = output_new[index_new + N_new] = output_new[index_new + 1 + N_new] = input_old[index_old];
}

__global__ void double_expand_bool_kernel(const bool *input_old, bool *output_new, const unsigned int N_old)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int index_old = i * N_old + j;
	unsigned int N_new = N_old << 1;
	unsigned int index_new = (N_new * i + j) << 1;

	output_new[index_new] = output_new[index_new + 1] = output_new[index_new + N_new] = output_new[index_new + 1 + N_new] = input_old[index_old];	
}


__global__ void add_divide_100_kernel(const cuComplex *input, cuComplex *output)
{
	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
	output[index].x += input[index].x / 20.f;
	output[index].y += input[index].y / 20.f;
}

__global__ void adjust_grad_to_mask_kernel(float *dev_gradient, bool *dev_mask, const unsigned int h_i_line, const unsigned int spread)
{
	unsigned int i = blockIdx.x * Q + threadIdx.x;
	unsigned int j = blockIdx.y * Q + threadIdx.y;
	unsigned int index = i * N + j;

	if (((j + 1 < N) && dev_mask[index + 1]) || ((j  > 0) && dev_mask[index - 1]) || ((i + 1 < N) && dev_mask[index + N]) || ((i  > 0) && dev_mask[index - N]))
	{
		if (i < h_i_line + spread + 1)
		{
			dev_gradient[index] = -100.f;
		}
		else
		{

			if ((!dev_mask[index]) && (dev_gradient[index] > 0.f))
			{
				//nothing to do
			}
			else
			{
				dev_gradient[index] = -100.f;
			}
		}
	}
	else
	{
		dev_gradient[index] = -100.f;
	}

}

__global__ void adjust_grad_to_mask_to_subtract_kernel(float *dev_gradient, bool *dev_mask, const unsigned int h_i_line, const unsigned int spread)
{
	unsigned int i = blockIdx.x * Q + threadIdx.x;
	unsigned int j = blockIdx.y * Q + threadIdx.y;
	unsigned int index = i * N + j;

	if (i < h_i_line + spread + 1)
	{
		dev_gradient[index] = -100.f;
	}
	else
	{

		if ((dev_mask[index]) && (dev_gradient[index] < 0.f))
		{
			dev_gradient[index] = -dev_gradient[index];
		}
		else
		{
			dev_gradient[index] = -100.f;
		}
	}
}


__global__ void minus_100_gradient_checker_kernel(bool *checker, unsigned int *suitable_index, float *gradient)
{
	*checker = gradient[*suitable_index] < -10.f ? true : false;
}
