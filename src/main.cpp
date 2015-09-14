/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include "testing_helpers.hpp"

int main(int argc, char* argv[]) {
    const int SIZE = 1 << 15;
    const int NPOT = SIZE - 3;
    int a[SIZE], b[SIZE], c[SIZE];

	// the case in the slides, as a smaller test.
	int small[8];
	for (int i = 0; i < 8; i++) {
		small[i] = i;
	}
	int smallScan[8] = { 0, 0, 1, 3, 6, 10, 15, 21 };
	int smallCompact[7] = { 1, 2, 3, 4, 5, 6, 7 };

	// set "true" for timed tests
	// also set BENCHMARK in common to 1
	if (false) {
		genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
		a[SIZE - 1] = 0;
		printf("array size: %i\n", SIZE);
		int count;

		zeroArray(SIZE, b);
		printDesc("cpu scan, power-of-two");
		StreamCompaction::CPU::scan(SIZE, b, a);
		printf("\n");

		zeroArray(SIZE, c);
		printDesc("naive scan, power-of-two");
		StreamCompaction::Naive::scan(SIZE, c, a);
		printf("\n");

		zeroArray(SIZE, c);
		printDesc("work-efficient scan, power-of-two");
		StreamCompaction::Efficient::scan(SIZE, c, a);
		printf("\n");

		zeroArray(SIZE, c);
		printDesc("thrust scan, power-of-two");
		StreamCompaction::Thrust::scan(SIZE, c, a);
		printf("\n");

		zeroArray(SIZE, b);
		printDesc("cpu compact without scan, power-of-two");
		count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
		printf("\n");

		zeroArray(SIZE, c);
		printDesc("cpu compact with scan");
		count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
		printf("\n");

		zeroArray(SIZE, c);
		printDesc("work-efficient compact, power-of-two");
		count = StreamCompaction::Efficient::compact(SIZE, c, a);
		printf("\n");

		printf("benchmark tests done\n");
		return 0;
	}

    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printArray(NPOT, b, true);
    printCmpResult(NPOT, b, c);

	zeroArray(SIZE, c);
	printDesc("small cpu scan test.");
	StreamCompaction::CPU::scan(8, c, small);
	printCmpResult(8, smallScan, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    //printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

	zeroArray(SIZE, c);
	printDesc("small naive scan test.");
	StreamCompaction::Naive::scan(8, c, small);
	printCmpResult(8, smallScan, c);

	zeroArray(SIZE, c);
	printDesc("small naive scan test, non-power-of-two.");
	StreamCompaction::Naive::scan(7, c, small);
	printCmpResult(7, smallScan, c);

	zeroArray(SIZE, c);
	printDesc("small work efficient scan test.");
	StreamCompaction::Efficient::scan(8, c, small);
	printCmpResult(8, smallScan, c);

	zeroArray(SIZE, c);
	printDesc("small work efficient scan test, non-power-of-two.");
	StreamCompaction::Efficient::scan(7, c, small);
	printCmpResult(7, smallScan, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

	zeroArray(SIZE, c);
	printDesc("small thrust scan.");
	StreamCompaction::Thrust::scan(8, c, small);
	printCmpResult(8, smallScan, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

	zeroArray(SIZE, c);
	printDesc("small cpu compact without scan, power-of-two");
	count = StreamCompaction::CPU::compactWithoutScan(8, c, small);
	printCmpLenResult(count, 7, smallCompact, c);

	zeroArray(SIZE, c);
	printDesc("small cpu compact without scan, non-power-of-two");
	count = StreamCompaction::CPU::compactWithoutScan(7, c, small);
	printCmpLenResult(count, 6, smallCompact, c);

    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

	zeroArray(SIZE, c);
	printDesc("small cpu compact with scan, power-of-two");
	count = StreamCompaction::CPU::compactWithScan(8, c, small);
	printCmpLenResult(count, 7, smallCompact, c);

	zeroArray(SIZE, c);
	printDesc("small cpu compact with scan, non-power-of-two");
	count = StreamCompaction::CPU::compactWithScan(7, c, small);
	printCmpLenResult(count, 6, smallCompact, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

	zeroArray(SIZE, c);
	printDesc("small work-efficient compact with scan, power-of-two");
	count = StreamCompaction::Efficient::compact(8, c, small);
	printCmpLenResult(count, 7, smallCompact, c);

	zeroArray(SIZE, c);
	printDesc("small work-efficient compact with scan, non-power-of-two");
	count = StreamCompaction::Efficient::compact(7, c, small);
	printCmpLenResult(count, 6, smallCompact, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);
	printf("done\n");
}
