#include <cstdio>
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
    // Implement exclusive serial scan on CPU
	odata[0] = 0;
	for (int i = 1; i < n; i++) {
		odata[i] = odata[i - 1] + idata[i - 1];
	}
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
    // remove all 0s from the array of ints
	int odataIndex = 0;
	for (int i = 0; i < n; i++) {
		if (idata[i] == 0) {
			continue;
		}
		odata[odataIndex] = idata[i];
		odataIndex++;
	}
	return odataIndex;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
    // Step 1: Compute temporary values in odata
	int *trueArray = new int[n];
	for (int i = 0; i < n; i++) {
		if (idata[i] == 0) {
			trueArray[i] = 0;
		}
		else {
			trueArray[i] = 1;
		}
	}
	// Step 2: Run exclusive scan on temporary array
	int *trueScan = new int[n];
	scan(n, trueScan, trueArray);

	// Step 3: Scatter
	for (int i = 0; i < n; i++) {
		if (trueArray[i]) {
			odata[trueScan[i]] = idata[i];
		}
	}
	delete trueArray;
	int numRemaining = trueScan[n - 1];
	delete trueScan;
	return numRemaining;
}

}
}
