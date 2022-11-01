#include <stdio.h>
#include <sys/time.h>
#include <random>

#define BLOCK_SIZE 32
#define N 2048

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
	cudaEventCreate( &start);
	cudaEventCreate( &stop);
	float time;

	//выделение памяти на GPU:
	cudaMalloc( (void**) &dA, numBytes );
	cudaMalloc( (void**) &dB, numBytes );
	cudaMalloc( (void**) &dC, numBytes );

	//выделение памяти на HOST и заполнение матриц pinned
	cudaHostAlloc( (void**) &hA, numBytes, cudaHostAllocDefault);	
	cudaHostAlloc( (void**) &hB, numBytes, cudaHostAllocDefault);
	cudaHostAlloc( (void**) &hC, numBytes, cudaHostAllocDefault);
	
	for (int i = 0; i< N; i++){
		for (int j = 0; j < N; j++){
			hA[j + i*N] = (double) rand() / RAND_MAX;
			hB[j + i*N] = (double) rand() / RAND_MAX;
			hC[j + i*N] = 0;
			}
	}

	//копирование матриц на GPU
	cudaEventRecord(start, 0);
	cudaMemcpy(dA, hA, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, numBytes, cudaMemcpyHostToDevice);
	// Засекаем время и умножаем
	MatrixMultiplication_global <<<blocks, threads >>> (dA, dB, dC);
	cudaThreadSynchronize();
	cudaEventSynchronize(stop);

	//копирование с GPU на HOST
	cudaMemcpy(hC, dC, numBytes, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime (&time, start, stop);
	printf("GPU time pinned = %f ms \n", time);
	//Освобождение памяти
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
	
