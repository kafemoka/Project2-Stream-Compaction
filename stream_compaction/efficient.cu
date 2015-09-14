#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

// TODO: __global__

__global__ void upsweep_step(int d, int *x, int *out) {
	int k = threadIdx.x + (blockIdx.x * blockDim.x);
	if (k % (int) powf(2, d + 1)) {
		return;
	}
	int out_index = k + (int)powf(2, d + 1) - 1;
	out[out_index] = x[out_index] + x[k + (int)powf(2, d) - 1];
}

__global__ void downsweep_step(int d, int *x, int *out) {
	int k = threadIdx.x + (blockIdx.x * blockDim.x);
	if (k % (int)powf(2, d + 1)) {
		return;
	}
	int left_index = k + (int)powf(2, d) - 1;
	int right_index = k + (int)powf(2, d + 1) - 1;
	out[left_index] = x[right_index];
	out[right_index] = x[right_index] + x[left_index];
}

__global__ void fill_by_value(int val, int *x) {
	int k = threadIdx.x + (blockIdx.x * blockDim.x);
	x[k] = val;
}

static void setup_dimms(dim3 &dimBlock, dim3 &dimGrid, int n) {
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int tpb = deviceProp.maxThreadsPerBlock;
	int blockWidth = fmin(n, tpb);
	int blocks = 1;
	if (blockWidth != n) {
		blocks = n / tpb;
		if (n % tpb) {
			blocks ++;
		}
	}

	dimBlock = dim3(blockWidth);
	dimGrid = dim3(blocks);
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {

	// copy everything in idata over to the GPU.
	// we'll need to pad the device memory with 0s to get a power of 2 array size.
	int logn = ilog2ceil(n);
	int pow2 = (int)pow(2, logn);

	dim3 dimBlock;
	dim3 dimGrid;
	setup_dimms(dimBlock, dimGrid, pow2);

	int *dev_x;
	cudaMalloc((void**)&dev_x, sizeof(int) * pow2);
	fill_by_value <<<dimGrid, dimBlock >>>(0, dev_x);

	cudaMemcpy(dev_x, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

	// up sweep and down sweep
	up_sweep_down_sweep(pow2, dev_x);

	cudaMemcpy(odata, dev_x, sizeof(int) * n, cudaMemcpyDeviceToHost);
	cudaFree(dev_x);
}

// exposed up sweep and down sweep. expects powers of two!
void up_sweep_down_sweep(int n, int *dev_data1) {
	int logn = ilog2ceil(n);

	dim3 dimBlock;
	dim3 dimGrid;
	setup_dimms(dimBlock, dimGrid, n);

	int *dev_data2;
	cudaMalloc((void**)&dev_data2, sizeof(int) * n);
	cudaMemcpy(dev_data2, dev_data1, sizeof(int) * n, cudaMemcpyDeviceToDevice);

	// Up Sweep
	for (int d = 0; d < logn; d++) {
		upsweep_step <<<dimGrid, dimBlock >>>(d, dev_data1, dev_data2);
		cudaMemcpy(dev_data1, dev_data2, sizeof(int) * n, cudaMemcpyDeviceToDevice);
	}

	//debug: peek at the array after upsweep
	//int peek1[8];
	//int peek2[8];
	//cudaMemcpy(&peek1, dev_data1, sizeof(int) * 8, cudaMemcpyDeviceToHost);
	//cudaMemcpy(&peek2, dev_data2, sizeof(int) * 8, cudaMemcpyDeviceToHost);

	// Down-Sweep
	int zero[1];
	zero[0] = 0;
	cudaMemcpy(&dev_data1[n - 1], zero, sizeof(int) * 1, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_data2, dev_data1, sizeof(int) * n, cudaMemcpyDeviceToDevice);
	for (int d = logn - 1; d >= 0; d--) {
		downsweep_step << <dimGrid, dimBlock >> >(d, dev_data1, dev_data2);
		cudaMemcpy(dev_data1, dev_data2, sizeof(int) * n, cudaMemcpyDeviceToDevice);
	}

	cudaFree(dev_data2);
}

__global__ void temporary_array(int *x, int *temp) {
	int k = threadIdx.x + (blockIdx.x * blockDim.x);
	temp[k] = (x[k] != 0);
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
	int logn = ilog2ceil(n);
	int pow2 = (int)pow(2, logn);

	dim3 dimBlock;
	dim3 dimGrid;
	setup_dimms(dimBlock, dimGrid, pow2);

	int *dev_x;
	int *dev_tmp;
	int *dev_scatter;
	int *dev_scan;

	cudaMalloc((void**)&dev_x, sizeof(int) * pow2);
	cudaMalloc((void**)&dev_tmp, sizeof(int) * pow2);
	cudaMalloc((void**)&dev_scan, sizeof(int) * pow2);
	cudaMalloc((void**)&dev_scatter, sizeof(int) * pow2);

	// 0 pad up to a power of 2 array length.
	// copy everything in idata over to the GPU.
	fill_by_value << <dimGrid, dimBlock >> >(0, dev_x);
	cudaMemcpy(dev_x, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

    // Step 1: compute temporary true/false array
	temporary_array <<<dimGrid, dimBlock >>>(dev_x, dev_tmp);

	// Step 2: run efficient scan on the tmp array
	cudaMemcpy(dev_scan, dev_tmp, sizeof(int) * pow2, cudaMemcpyDeviceToDevice);
	up_sweep_down_sweep(pow2, dev_scan);

	// Step 3: scatter
	scatter <<<dimGrid, dimBlock >>>(dev_x, dev_tmp, dev_scan, dev_scatter);

	cudaMemcpy(odata, dev_scatter, sizeof(int) * n, cudaMemcpyDeviceToHost);

	int last_index;
	cudaMemcpy(&last_index, dev_scan + (n - 1), sizeof(int),
		cudaMemcpyDeviceToHost);

	int last_true_false;
	cudaMemcpy(&last_true_false, dev_tmp + (n - 1), sizeof(int),
		cudaMemcpyDeviceToHost);

	cudaFree(dev_x);
	cudaFree(dev_tmp);
	cudaFree(dev_scan);
	cudaFree(dev_scatter);

	return last_index + last_true_false;
}

}
}
