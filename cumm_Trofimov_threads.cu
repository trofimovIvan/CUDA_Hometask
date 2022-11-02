#include <stdio.h>
#include <sys/time.h>
#include <random>

#define BLOCK_SIZE 32
#define N 2048
#define NSTREAM 2

__global__ void MatrixMultiplication_global (double *dA, double *dB, double *dC){
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	double sum = 0;
	int ia = N*(BLOCK_SIZE * by + ty); // номер строки А
	int ib = BLOCK_SIZE*bx + tx; // номер столбца В
	int ic = ia + ib; // номер элемента С
	// вычисление элемента С
	for (int k=0; k<N; k++) sum += dA[ia + k]*dB[ib + k*N];
	dC[ic] = sum;
}

int main()
{
	int numBytes = N*N*sizeof(double);
	double *dA, *dB, *dC, *hA, *hB, *hC;
	// задание сетки нитей и блоков:
	dim3 threads (BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks (N/threads.x, N/threads.y);	
	cudaEvent_t start, stop;
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

        dim3 gridDim(ceilf(N/(float)BLOCK_SIZE), ceilf(N/(float)BLOCK_SIZE), 1);
	cudaEventCreate( &start);
	cudaEventCreate( &stop);
	float time;
	//создаем cuda-потоки
	cudaStream_t stream[NSTREAM];
	for (int i =0; i<NSTREAM; i++) cudaStreamCreate(&stream[i]);

   	dim3 gridDim_s(ceilf(N/(float)BLOCK_SIZE/NSTREAM));
	//выделение памяти на GPU:
	cudaMalloc( (void**) &dA, numBytes );
	cudaMalloc( (void**) &dB, numBytes );
	cudaMalloc( (void**) &dC, numBytes );

	//выделение памяти на HOST и заполнение матриц pinned
	cudaMallocHost( (void**) &hA, numBytes);	
	cudaMallocHost( (void**) &hB, numBytes);
	cudaMallocHost( (void**) &hC, numBytes);
	
	for (int i = 0; i< N; i++){
		for (int j = 0; j < N; j++){
			hA[j + i*N] = (double) rand() / RAND_MAX;
			hB[j + i*N] = (double) rand() / RAND_MAX;
			hC[j + i*N] = 0;
			}
	}
	//копирование матриц на GPU
	cudaEventRecord(start, 0);
	
    for (int i = 0; i < NSTREAM; ++i) {
	    cudaMemcpyAsync(
                dA + (int)i * N / NSTREAM,
                hA + (int)i * N / NSTREAM,
                sizeof(double) * N / NSTREAM,
                cudaMemcpyHostToDevice,
                stream[i]);
        
        cudaMemcpyAsync(
                dB + (int)i * N / NSTREAM,
                hB + (int)i * N / NSTREAM,
                sizeof(double) * N / NSTREAM,
                cudaMemcpyHostToDevice,
                stream[i]);
    }
    for (int i = 0; i < NSTREAM; ++i) {
	    MatrixMultiplication_global <<<gridDim_s, blockDim, 0, stream[i]>>>
                (dA + (int)i * N / NSTREAM, dB + (int)i * N / NSTREAM,
                 dC + (int)i * N / NSTREAM);
    }
    for (int i = 0; i < NSTREAM; ++i) {
	    cudaMemcpyAsync(
                hC + (int)i * N / NSTREAM, 
                dC + (int)i * N / NSTREAM,
                sizeof(double) * N / NSTREAM,
                cudaMemcpyDeviceToHost,
                stream[i]) << '\n';
    }
    cudaDeviceSynchronize();
    for (int i = 0; i < NSTREAM; ++i) cudaStreamDestroy(stream[i]);
    
 
	//cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime (&time, start, stop);
	printf("GPU time pinned async = %3.1f ms \n", time);
	//Освобождение памяти
	for (int i =0; i<NSTREAM; i++) cudaStreamDestroy(stream[i]);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	cudaFree(hA);
	cudaFree(hB);
	cudaFree(hC);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}	
	
