#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {

// TODO: __global__

__global__ void naive_scan_step(int d, int *x_1, int *x_2) {
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	int offset = powf(2, d - 1);
	if (i >= offset) {
		x_2[i] = x_1[i - offset] + x_1[i];
	}
	else {
		x_2[i] = x_1[i];
	}
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // copy everything in idata over to the GPU
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int tpb = deviceProp.maxThreadsPerBlock;
	int blockWidth = fmin(n, tpb);
	int blocks = 1;
	if (blockWidth != n) {
		blocks = n / tpb;
		if (n % tpb) {
			blocks++;
		}
	}

	dim3 dimBlock(blockWidth);
	dim3 dimGrid(blocks);

	int *dev_x;
	int *dev_x_next;
	cudaMalloc((void**)&dev_x, sizeof(int) * n);
	cudaMalloc((void**)&dev_x_next, sizeof(int) * n);

	cudaMemcpy(dev_x, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_x_next, dev_x, sizeof(int) * n, cudaMemcpyDeviceToDevice);

	// run steps.
	// no need to pad with 0s to get a power of 2 array here,
	// this can be an "unbalanced" binary tree of ops.
	int logn = ilog2ceil(n);
	for (int d = 1; d <= logn; d++) {
		naive_scan_step <<<dimGrid, dimBlock >>>(d, dev_x, dev_x_next);
		int *temp = dev_x_next;
		dev_x_next = dev_x;
		dev_x = temp;
	}

	cudaMemcpy(odata + 1, dev_x, sizeof(int) * (n - 1), cudaMemcpyDeviceToHost);
	odata[0] = 0;
	
	cudaFree(dev_x);
	cudaFree(dev_x_next);
}

}
}
