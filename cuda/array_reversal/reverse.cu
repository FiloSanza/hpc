/****************************************************************************
 *
 * cuda-reverse.cu - Array reversal with CUDA
 *
 * Copyright (C) 2017--2022 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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

/***
% HPC - Array reversal with CUDA
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-11-15

Write a program that reverses an array of length $n$, i.e., swaps the
content of position $0$ and $n-1$, then position $1$ and $n-2$ and so
on. Specifically, write two versions of such a program: the first
version reverses an input array `in[]` into a different output array
`out[]`, so that the input is not modified. The second version
reverses an array `in[]` "in place" using at most $O(1)$ additional
storage.

The file [cuda-reverse.cu](cuda-reverse.cu) provides a CPU-based
implementation of `reverse()` and `inplace_reverse()` functions.  You
are required to modify the functions to make use of the GPU.

**Hint:** `reverse()` can be easily transformed into a kernel executed
by $n$ CUDA threads (one for each element of the array). Each thread
copies an element from `in[]` to the correct position of `out[]`.  Use
one-dimensional _thread blocks_, since that makes easy to map threads
to array elements. The `inplace_reverse()` function can be transformed
into a kernel as well, but in this case only $\lfloor n/2 \rfloor$
CUDA threads are required (note the rounding): each thread swaps an
element from the first half of `in[]` with the appropriate element
from the second half. Make sure that the program works also when the
input length $n$ is odd.

To copmile:

        nvcc cuda-reverse.cu -o cuda-reverse

To execute:

        ./cuda-reverse [n]

Example:

        ./cuda-reverse

## Files

- [cuda-reverse.cu](cuda-reverse.cu)
- [hpc.h](hpc.h)

***/
#include "hpc.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>

__global__ void kernel_reverse(int *in, int *out, int n)
{
    const int in_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int out_idx = n - in_idx - 1; 
    out[out_idx] = in[in_idx];
}

void reverse( int *in, int *out, int n )
{
    int *d_in, *d_out;
    int SIZE = n * sizeof(int);
    
    const int BLKSIZE = 1024;
    const int BLKNUM  = (n + BLKSIZE - 1) / BLKSIZE;

    cudaSafeCall( cudaMalloc((void**)&d_in , SIZE) );
    cudaSafeCall( cudaMalloc((void**)&d_out, SIZE) );

    cudaSafeCall( cudaMemcpy(d_in, in, SIZE, cudaMemcpyHostToDevice) );
    
    kernel_reverse<<<BLKNUM, BLKSIZE>>>(d_in, d_out, n);
    cudaCheckError();

    cudaSafeCall( cudaMemcpy(out, d_out, SIZE, cudaMemcpyDeviceToHost) );

    cudaSafeCall( cudaFree(d_in ) );
    cudaSafeCall( cudaFree(d_out) );
}

__global__ void kernel_inplace_reverse(int *in, int n)
{
    const int in_idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (in_idx >= n / 2) {
        return;
    }

    const int out_idx = n - in_idx - 1;
    const int tmp = in[in_idx];
    in[in_idx] = in[out_idx];
    in[out_idx] = tmp;
}

void inplace_reverse( int *in, int n )
{
    int *d_in;
    int SIZE = n * sizeof(int);
    
    const int BLKSIZE = 1024;
    const int BLKNUM  = ((n/2) + BLKSIZE - 1) / BLKSIZE;

    cudaSafeCall( cudaMalloc((void**)&d_in , SIZE) );
    cudaSafeCall( cudaMemcpy(d_in, in, SIZE, cudaMemcpyHostToDevice) );
    
    kernel_inplace_reverse<<<BLKNUM, BLKSIZE>>>(d_in, n);
    cudaCheckError();

    cudaSafeCall( cudaMemcpy(in, d_in, SIZE, cudaMemcpyDeviceToHost) );

    cudaSafeCall( cudaFree(d_in) );
}

void fill( int *x, int n )
{
    int i;
    for (i=0; i<n; i++) {
        x[i] = i;
    }
}

int check( const int *x, int n )
{
    int i;
    for (i=0; i<n; i++) {
        if (x[i] != n - 1 - i) {
            fprintf(stderr, "Test FAILED: x[%d]=%d, expected %d\n", i, x[i], n-1-i);
            return 0;
        }
    }
    printf("Test OK\n");
    return 1;
}

int main( int argc, char* argv[] )
{
    int *in, *out;
    int n = 1024*1024;
    const int MAX_N = 512*1024*1024;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( n > MAX_N ) {
        fprintf(stderr, "FATAL: input too large (maximum allowed length is %d)\n", MAX_N);
        return EXIT_FAILURE;
    }

    const size_t SIZE = n * sizeof(*in);

    /* Allocate in[] and out[] */
    in = (int*)malloc(SIZE);
    assert(in != NULL);
    out = (int*)malloc(SIZE);
    assert(out != NULL);
    fill(in, n);

    printf("Reverse %d elements... ", n);
    reverse(in, out, n);
    check(out, n);

    printf("In-place reverse %d elements... ", n);
    inplace_reverse(in, n);
    check(in, n);

    /* Cleanup */
    free(in);
    free(out);

    return EXIT_SUCCESS;
}
