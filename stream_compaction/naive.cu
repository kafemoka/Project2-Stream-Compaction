#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {

// TODO: __global__

__global__ void naive_scan_step(int d, int *x, int *x_next) {
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	int offset = powf(2, d - 1);
	if (i >= offset) {
		x_next[i] = x[i - offset] + x[i];
	}
	else {
		x_next[i] = x[i];
	}
}

__global__ void parallel_copy(int *data, int *copy) {
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	copy[i] = data[i];
}

__global__ void parallel_shift(int *inclusive, int *exclusive) {
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	if (i == 0) {
		exclusive[i] = 0;
		return;
	}
	exclusive[i] = inclusive[i - 1];
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // copy everything in idata over to the GPU
	dim3 dimBlock(n);
	dim3 dimGrid(1);
	int *dev_x;
	int *dev_x_next;
	int *dev_exclusive;
	cudaMalloc((void**)&dev_x, sizeof(int) * n);
	cudaMalloc((void**)&dev_x_next, sizeof(int) * n);
	cudaMalloc((void**)&dev_exclusive, sizeof(int) * n);

	cudaMemcpy(dev_x, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

	// run steps.
	int logn = ilog2ceil(n);
	for (int d = 1; d <= logn; d++) {
		naive_scan_step <<<dimGrid, dimBlock >>>(d, dev_x, dev_x_next);
		parallel_copy <<<dimGrid, dimBlock >>>(dev_x_next, dev_x);
	}

	parallel_shift << <dimGrid, dimBlock >> >(dev_x, dev_exclusive);
	cudaFree(dev_x);
	cudaFree(dev_x_next);

	cudaMemcpy(odata, dev_exclusive, sizeof(int) * n, cudaMemcpyDeviceToHost);
	cudaFree(dev_exclusive);
}

}
}
