/****************************************************************************
 *
 * omp-loop.c - Loop-carried dependences
 *
 * Copyright (C) 2018--2021 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ****************************************************************************/

/**
 * Implementation: Filippo Sanzani
 */

/***
% HPC - Loop-carried dependences
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-10-17

The file [omp-loop.c](omp-loop.c) contains a set of serial functions
with loops that iterate over arrays or matrices. The goal of this
exercise is to apply the loop parallelization techniques seen during
the class (or according to the hint provided below) to create a
parallel version.

The `main()` function checks for correctness of the results comparing
the output of the serial and parallel versions. Note that such fact
check is not (and can not be) exhaustive.

To compile:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-loop.c -o omp-loop

To execute:

        ./omp-loop

## Files

- [omp-loop.c](omp-loop.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

/* Three small functions used below; you should not need to know what
   these functions do */
int f(int a, int b, int c) { return (a+b+c)/3; }
int g(int a, int b) { return (a+b)/2; }
int h(int a) { return (a > 10 ? 2*a : a-1); }

/****************************************************************************/

/**
 * Shift the elements of array |a| of length |n| one position to the
 * right; the rightmost element of |a| becomes the leftmost element of
 * the shifted array.
 */
void vec_shift_right_seq(int *a, int n)
{
    int i;
    int tmp = a[n-1];
    for (i=n-1; i>0; i--) {
        a[i] = a[i-1];
    }
    a[0] = tmp;
}

void vec_shift_right_par1(int *a, int n)
{
    /* [TODO] This function should be a parallel version of
       vec_shift_right_seq(). It is not possible to remove the
       loop-carried dependency by aligning loop iterations. However,
       one solution is to use a temporary array b[] and split the loop
       into two loops: the first copies all elements of a[] in the
       shifted position of b[] (i.e., a[i] goes to b[i+1]; the
       rightmost element of a[] goes into b[0]). The second loop
       copies b[] into a[]. Both loops can be trivially
       parallelized. */

    int b[n];

    // step 1
    int tmp = a[n-1];
#pragma omp parallel for default(none) shared(a, b, n)
    for (int i=n-1; i>0; i--) {
        b[i] = a[i-1];
    }
    b[0] = tmp;

#pragma omp parallel for default(none) shared(a, b, n)
    for (int i=0; i<n; i++) {
        a[i] = b[i];
    }
}

void vec_shift_right_par2(int *a, int n)
{
    /* A different solution to shift a vector without using a
       temporary array: partition a[] into P blocks (P=size of the
       pool of threads). Each process saves the rightmost element of
       its block, then shifts the block on position right. When all
       threads are done (barrier synchronization), each thread fills
       the _leftmost_ element of its block with the _rightmost_
       element saved by its left neighbor.

       Example, with P=4 threads:

       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
       |a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|
       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
       \----------/\----------/\----------/\----------/
            P0          P1          P2          P3

       Each thread stores the rightmost element into a shared array
       rightmost[]:

       +-+-+-+-+
       |f|l|r|x|   rightmost[]
       +-+-+-+-+

       Each thread shifts right its portion

       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
       |?|a|c|b|d|e|?|g|h|i|j|k|?|m|n|o|p|q|?|s|t|u|v|w|
       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
       \----------/\----------/\----------/\----------/
            P0          P1          P2          P3

       Each thread fills the leftmost element with the correct value
       from rightmost[]

       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
       |x|a|c|b|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|
       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
       \----------/\----------/\----------/\----------/
            P0          P1          P2          P3

    */
    const int num_threads = 2;
    int rightmost[num_threads];

#pragma omp parallel default(none) shared(num_threads, a, n, rightmost) num_threads(num_threads)
    {
        const int my_id = omp_get_thread_num();
        const int lower_bound = (n * my_id) / num_threads;
        const int upper_bound = (n * (my_id + 1)) / num_threads;
        rightmost[my_id] = a[upper_bound - 1];

        for (int i=upper_bound - 1; i>lower_bound; i--) {
            a[i] = a[i-1];
        }
    }

    for (int i=0; i<num_threads; i++) {
        a[(n * i) / num_threads] = rightmost[(i+1) % num_threads];
    }
}

/****************************************************************************/

/* This function converts 2D indexes into a linear index; n is the
   number of columns of the matrix being indexed. The proper solution
   would be to use C99-style casts:

   int (*AA)[n] = (int (*)[n])A;

   and then write AA[i][j]. Unfortunately, this triggers a bug in gcc
   5.4.0+OpenMP (works with gcc 8.2.0+OpenMP)
*/
int IDX(int i, int j, int n)
{
    return i*n + j;
}

/* A is a nxn matrix */
void test1_seq(int *A, int n)
{
    int i, j;
    for (i=1; i<n; i++) {
        for (j=1; j<n-1; j++) {
            /*
              A[i][j] = f(A[i-1][j-1], A[i-1][j], A[i-1, j+1])
            */
            A[IDX(i,j,n)] = f(A[IDX(i-1,j-1,n)], A[IDX(i-1,j,n)], A[IDX(i-1,j+1,n)]);
        }
    }
}

void test1_par(int *A, int n)
{
    /* [TODO] This function should be a parallel vesion of
       test1_seq(). Suggestion: start by drawing the dependences among
       the elements of matrix A[][] as they are computed.  Then,
       observe that one of the loops (which one?) can be parallelized
       as is with a #pragma opm parallel for directive. There is no
       need to modify the code, nor to exchange loops. */
    for (int i=1; i<n; i++) {
#pragma omp parallel for default(none) shared(A, n, i)
        for (int j=1; j<n-1; j++) {
            /*
              A[i][j] = f(A[i-1][j-1], A[i-1][j], A[i-1, j+1])
            */
            A[IDX(i,j,n)] = f(A[IDX(i-1,j-1,n)], A[IDX(i-1,j,n)], A[IDX(i-1,j+1,n)]);
        }
    }
}

/****************************************************************************/

void test2_seq(int *A, int n)
{
    int i, j;
    for (i=1; i<n; i++) {
        for (j=1; j<n; j++) {
            /*
              A[i][j] = g(A[i,j-1], A[i-1,j-1])
             */
            A[IDX(i,j,n)] = g(A[IDX(i,j-1,n)], A[IDX(i-1,j-1,n)]);
        }
    }
}

void test2_par(int *A, int n)
{
    /* [TODO] This function should be a parallel version of
       test2_seq(). Suggestion: start by drawing the dependences
       among the elements of matrix A[][] as they are
       computed. Observe that it is not possible to put a "parallel
       for" directive on either loop.

       However, you can exchange the loops (why?), i.e., the loops
       can be rewritten as

       for (j=1; j<n; j++) {
         for (i=1; i<n; i+) {
           ....
         }
       }

       preserving the correctness of the computation. Now, one of the
       loops can be parallelized (which one?) */
}

/****************************************************************************/

void test3_seq(int *A, int n)
{
    int i, j;
    for (i=1; i<n; i++) {
        for (j=1; j<n; j++) {
            /*
              A[i][j] = f(A[i][j-1], A[i-1][j-1], A[i-1][j])
             */
            A[IDX(i,j,n)] = f(A[IDX(i,j-1,n)], A[IDX(i-1,j-1,n)], A[IDX(i-1,j,n)]);
        }
    }
}

void test3_par(int *A, int n)
{
    /* [TODO] This function should be a parallel version of
       test3_seq(). Neither loops can be trivially parallelized;
       exchanging loops (moving the inner loop outside) does not work
       either.

       This is the same example shown on the slides, and can be solved
       by sweeping the matrix "diagonally".

       There is a caveat: the code on the slides sweeps the _whole_
       matrix; in other words, variables i and j will assume all
       values starting from 0. The code of test3_seq() only process
       indexes where i>0 and j>0, so you need to add an "if" statement
       to skip the case where i==0 or j==0. */
}

/**
 ** The code below does not need to be modified
 **/

void fill(int *a, int n)
{
    a[0] = 31;
    for (int i=1; i<n; i++) {
        a[i] = (a[i-1] * 33 + 1) % 65535;
    }
}

int array_equal(int *a, int *b, int n)
{
    for (int i=0; i<n; i++) {
        if (a[i] != b[i]) { return 0; }
    }
    return 1;
}

int main( void )
{
    const int N = 10;
    int *a1, *b1, *c1, *a2, *b2, *c2;

    /* Allocate enough space for all tests */
    a1 = (int*)malloc(N*N*sizeof(int)); assert(a1 != NULL);
    b1 = (int*)malloc(N*sizeof(int)); assert(b1 != NULL);
    c1 = (int*)malloc(N*sizeof(int)); assert(c1 != NULL);

    a2 = (int*)malloc(N*N*sizeof(int)); assert(a2 != NULL);
    b2 = (int*)malloc(N*sizeof(int)); assert(b2 != NULL);
    c2 = (int*)malloc(N*sizeof(int)); assert(c2 != NULL);

    printf("vec_shift_right_par1()\t"); fflush(stdout);
    fill(a1, N);
    vec_shift_right_seq(a1, N);
    fill(a2, N);
    vec_shift_right_par1(a2, N);
    if ( array_equal(a1, a2, N) ) {
        printf("OK\n");
    } else {
        printf("FAILED\n");
    }

    printf("vec_shift_right_par2()\t"); fflush(stdout);
    fill(a2, N);
    vec_shift_right_par2(a2, N);
    if ( array_equal(a1, a2, N) ) {
        printf("OK\n");
    } else {
        printf("FAILED\n");
    }

    /* test1 */
    printf("test1_par()\t\t"); fflush(stdout);
    fill(a1, N*N);
    test1_seq(a1, N);
    fill(a2, N*N);
    test1_par(a2, N);
    if (array_equal(a1, a2, N*N)) {
        printf("OK\n");
    } else {
        printf("FAILED\n");
    }

    /* test2 */
    printf("test2_par()\t\t"); fflush(stdout);
    fill(a1, N*N);
    test2_seq(a1, N);
    fill(a2, N*N);
    test2_par(a2, N);
    if (array_equal(a1, a2, N*N)) {
        printf("OK\n");
    } else {
        printf("FAILED\n");
    }

    /* test3 */
    printf("test3_par()\t\t"); fflush(stdout);
    fill(a1, N*N);
    test3_seq(a1, N);
    fill(a2, N*N);
    test3_par(a2, N);
    if (array_equal(a1, a2, N*N)) {
        printf("OK\n");
    } else {
        printf("FAILED\n");
    }

    free(a1);
    free(b1);
    free(c1);
    free(a2);
    free(b2);
    free(c2);

    return EXIT_SUCCESS;
}
