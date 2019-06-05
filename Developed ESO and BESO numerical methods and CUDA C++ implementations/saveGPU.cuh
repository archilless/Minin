void saveGPUrealtxt_C(const cuComplex * d_in, const char *filename, const int M) {

	cuComplex *h_in = (cuComplex *)malloc(M * sizeof(cuComplex));

	cudacall(cudaMemcpy(h_in, d_in, M * sizeof(cuComplex), cudaMemcpyDeviceToHost));
	std::ofstream outfile;
	outfile.open(filename);	
	for (int i = 0; i < M; i++) outfile << std::setprecision(PRECISION_TO_SAVE_DATA_TO_FILE) << h_in[i].x << " " << h_in[i].y << "\n";
	outfile.close();
}

void saveCPUrealtxt_C(const cuComplex * h_in, const char *filename, const int M) {
	std::ofstream outfile;
	outfile.open(filename);	
	for (int i = 0; i < M; i++) outfile << std::setprecision(PRECISION_TO_SAVE_DATA_TO_FILE) << h_in[i].x << " " << h_in[i].y << "\n";
	outfile.close();
}


void saveGPUrealtxt_F(const float *d_inx, const char *filename, const int M) {

	float *h_inx = (float *)malloc(M * sizeof(float));

	cudacall(cudaMemcpy(h_inx, d_inx, M * sizeof(float), cudaMemcpyDeviceToHost));
	std::ofstream outfile;
	outfile.open(filename);	
	for (int i = 0; i < M; i++) outfile << std::setprecision(PRECISION_TO_SAVE_DATA_TO_FILE) << h_inx[i] << "\n";
	outfile.close();
}

void saveGPUrealtxt_I(const unsigned int *d_inx, const char *filename, const int M) {

	unsigned int *h_inx = (unsigned int *)malloc(M * sizeof(unsigned int));

	cudacall(cudaMemcpy(h_inx, d_inx, M * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	std::ofstream outfile;
	outfile.open(filename);	
	for (int i = 0; i < M; i++) outfile << std::setprecision(PRECISION_TO_SAVE_DATA_TO_FILE) << h_inx[i] << "\n";
	outfile.close();
}


void saveGPUrealtxt_B(const bool *d_inx, const char *filename, const int M) 
{
	bool *h_inx = (bool *)malloc(M * sizeof(bool));

	cudacall(cudaMemcpy(h_inx, d_inx, M * sizeof(bool), cudaMemcpyDeviceToHost));
	std::ofstream outfile;
	outfile.open(filename);	
	for (int i = 0; i < M; i++) outfile << std::setprecision(PRECISION_TO_SAVE_DATA_TO_FILE) << (h_inx[i] ? 1 : 0) << "\n";
	outfile.close();
}

void saveCPUrealtxt_F(const float *h_inx, const char *filename, const int M)
{
	std::ofstream outfile;
	outfile.open(filename);	
	for (int i = 0; i < M; i++) outfile << std::setprecision(PRECISION_TO_SAVE_DATA_TO_FILE) << h_inx[i] << "\n";
	outfile.close();
}


void saveCPUrealtxt_B(const bool *h_inx, const char *filename, const int M)
{
	std::ofstream outfile;
	outfile.open(filename);	
	for (int i = 0; i < M; i++) outfile << std::setprecision(PRECISION_TO_SAVE_DATA_TO_FILE) << (h_inx[i] ? 1 : 0) << "\n";
	outfile.close();
}

void saveCPUrealtxt_I(const unsigned int *h_inx, const char *filename, const int M)
{
	std::ofstream outfile;
	outfile.open(filename);	
	for (int i = 0; i < M; i++) outfile << std::setprecision(PRECISION_TO_SAVE_DATA_TO_FILE) << h_inx[i] << "\n";
	outfile.close();
}

void saveGPUrealtxt(bool *h_mask, cuComplex *dev_solution, float *dev_intensity_global, unsigned int optimization_number)
{
	char buffer[1024];

	sprintf(buffer, "data/greedy/mask_%i.txt", optimization_number);
	saveCPUrealtxt_B(h_mask, buffer, N * N);

	sprintf(buffer, "data/greedy/field_%i.txt", optimization_number);
	saveGPUrealtxt_C((cuComplex *)dev_solution, buffer, N * N);

	sprintf(buffer, "data/greedy/intensity_maximum_%i.txt", optimization_number);
	saveGPUrealtxt_F(dev_intensity_global, buffer, 1);
}

void saveGPUrealtxt_prefixed(const char *prefix, bool *dev_mask, cuComplex *dev_solution, float h_intensity_max, unsigned int optimization_number)
{
	char buffer[1024];	

	sprintf(buffer, "%s/mask_%i.txt", prefix, optimization_number);
	saveGPUrealtxt_B(dev_mask, buffer, N * N);

	sprintf(buffer, "%s/field_%i.txt", prefix, optimization_number);
	saveGPUrealtxt_C((cuComplex *)dev_solution, buffer, N * N);

	sprintf(buffer, "%s/intensity_maximum_%i.txt", prefix, optimization_number);
	saveCPUrealtxt_F(&h_intensity_max, buffer, 1);
}
