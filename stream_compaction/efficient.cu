#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

// TODO: __global__

__global__ void upsweep_step(int d, int *x) {
	int k = threadIdx.x + (blockIdx.x * blockDim.x);
	if (k % (int) powf(2, d + 1)) {
		return;
	}
	x[k + (int) powf(2, d + 1) - 1] += x[k + (int) powf(2, d) - 1];
}

__global__ void downsweep_step(int d, int *x) {
	int k = threadIdx.x + (blockIdx.x * blockDim.x);
	if (k % (int)powf(2, d + 1)) {
		return;
	}
	int t = x[k + (int) powf(2, d) - 1];
	x[k + (int) powf(2, d) - 1] = x[k + (int) powf(2, d + 1) - 1];
	x[k + (int) powf(2, d + 1) - 1] += t;
}

__global__ void fill_by_value(int val, int *x) {
	int k = threadIdx.x + (blockIdx.x * blockDim.x);
	x[k] = val;
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {

	// copy everything in idata over to the GPU.
	// we'll need to pad the device memory with 0s to get a power of 2 array size.
	int logn = ilog2ceil(n);
	int pow2 = (int)pow(2, logn);

	dim3 dimBlock(pow2);
	dim3 dimGrid(1);
	int *dev_x;
	cudaMalloc((void**)&dev_x, sizeof(int) * pow2);
	fill_by_value <<<dimGrid, dimBlock >>>(0, dev_x);

	cudaMemcpy(dev_x, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

    // Up Sweep
	for (int d = 0; d < logn; d++) {
		upsweep_step <<<dimGrid, dimBlock>>>(d, dev_x);
	}

	//debug: peek at the array after upsweep
	//int peek[8];
	//cudaMemcpy(&peek, dev_x, sizeof(int) * 8, cudaMemcpyDeviceToHost);

	// Down-Sweep
	int zero[1];
	zero[0] = 0;
	cudaMemcpy(&dev_x[pow2 - 1], zero, sizeof(int) * 1, cudaMemcpyHostToDevice);
	for (int d = logn - 1; d >= 0; d--) {
		downsweep_step <<<dimGrid, dimBlock>>>(d, dev_x);
	}

	cudaMemcpy(odata, dev_x, sizeof(int) * n, cudaMemcpyDeviceToHost);
	cudaFree(dev_x);
}

__global__ void temporary_array(int *x, int *temp) {
	int k = threadIdx.x + (blockIdx.x * blockDim.x);
	if (x[k] != 0) {
		temp[k] = 1;
	}
	else {
		temp[k] = 0;
	}
}

__global__ void scatter(int *x, int *trueFalse, int* scan, int *out) {
	int k = threadIdx.x + (blockIdx.x * blockDim.x);
	if (trueFalse[k]) {
		out[scan[k]] = x[k];
	}
}

/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int compact(int n, int *odata, const int *idata) {
	dim3 dimBlock(n);
	dim3 dimGrid(1);
	int *dev_x;
	int *dev_tmp;
	cudaMalloc((void**)&dev_x, sizeof(int) * n);
	cudaMalloc((void**)&dev_tmp, sizeof(int) * n);

	// copy everything in idata over to the GPU.
	cudaMemcpy(dev_x, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

    // Step 1: compute temporary true/false array
	temporary_array << <dimGrid, dimBlock >> >(dev_x, dev_tmp);

	// Step 2: run efficient scan on the tmp array
	// TODO: expose the CUDA relevant portions of efficient scan so we don't have to shunt around
	int *trueArray = new int[n];
	int *scanArray = new int[n];
	cudaMemcpy(trueArray, dev_tmp, sizeof(int) * n, cudaMemcpyDeviceToHost);
	scan(n, scanArray, trueArray);

	// Step 3: scatter
	int *dev_scatter;
	cudaMalloc((void**)&dev_scatter, sizeof(int) * n);

	int *dev_scan;
	cudaMalloc((void**)&dev_scan, sizeof(int) * n);
	cudaMemcpy(dev_scan, scanArray, sizeof(int) * n, cudaMemcpyHostToDevice);

	scatter << <dimGrid, dimBlock >> >(dev_x, dev_tmp, dev_scan, dev_scatter);

	cudaMemcpy(odata, dev_scatter, sizeof(int) * n, cudaMemcpyDeviceToHost);

	int return_value = scanArray[n - 1];

	delete trueArray;
	delete scanArray;
	cudaFree(dev_x);
	cudaFree(dev_tmp);
	cudaFree(dev_scan);
	cudaFree(dev_scatter);

	return return_value;
}

}
}
