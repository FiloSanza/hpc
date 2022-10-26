/****************************************************************************
 *
 * mpi-sum.c - Sum-reduction of an array
 *
 * Copyright (C) 2018--2022 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Sum-reduction of an array
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-10-24

The file [mpi-sum.c](mpi-sum.c) contains a serial implementation of an
MPI program that computes the sum of an array of length $N$; indeed,
the program performsa a _sum-reduction_ of the array. In the version
provided, process 0 performs all computations, and therefore is not a
true parallel program. Modify the program so that all processes
contribute to the computation according to the following steps (see
Figure 1).

![Figure 1: Computing the sum-reduction using MPI](mpi-sum.png)

1. The master process (rank 0) creates and initializes the input array
   `master_array[]`.

2. The master distributes `master_array[]` array among the $P$
   processes (including itself) using `MPI_Scatter()`. You may
   initially assume that $N$ is an integer multiple of $P$; how would
   you modify your program to work with arbitrary values of $N$?

3. Each process computes the sum-reduction of its portion.

4. Each process $p > 0$ sends its own local sum to the master using
   `MPI_Send()`; the master receives the local sums using `MPI_Recv()`
   and accumulates them.

We will see in the next lexture how step 4 can be realized more
efficiently with the MPI reduction operation.

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-sum.c -o mpi-sum

To execute:

        mpirun -n P ./mpi-sun N

Example:

        mpirun -n 4 ./mpi-sum 10000

## Files

- [mpi-sum.c](mpi-sum.c)

***/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/* Compute the sum of all elements of array `v` of length `n` */
float sum(float *v, int n)
{
    float sum = 0;
    int i;

    for (i=0; i<n; i++) {
        sum += v[i];
    }
    return sum;
}

/* Fill the array array `v` of length `n`; return the sum of the
   content of `v` */
float fill(float *v, int n)
{
    const float vals[] = {1, -1, 2, -2, 0};
    const int NVALS = sizeof(vals)/sizeof(vals[0]);
    int i;

    for (i=0; i<n; i++) {
        v[i] = vals[i % NVALS];
    }
    switch(i % NVALS) {
    case 1: return 1; break;
    case 3: return 2; break;
    default: return 0;
    }
}

int main( int argc, char *argv[] )
{
    int my_rank, comm_sz;
    float *master_array = NULL, s = 0, expected, *my_array = NULL, my_s;
    int n = 10000;

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    /* The master initializes the array */
    if ( 0 == my_rank ) {
        master_array = (float*)malloc( n * sizeof(float) );
        assert(master_array != NULL);
        expected = fill(master_array, n);
    }

    // Scatter the array
    const int scatter_sz = n / comm_sz;
    my_array = (float*)malloc(scatter_sz * sizeof(float)); assert(my_array != NULL);
    MPI_Scatter(
        master_array,       // Src buffer        
        scatter_sz,         // N elements sent
        MPI_FLOAT,          // Type sent
        my_array,           // Recv buffer
        scatter_sz,         // N elements recv
        MPI_FLOAT,          // Type recv
        0,                  // Src
        MPI_COMM_WORLD      // Comm
    );

    my_s = sum(my_array, scatter_sz);

    free(my_array);

    if (my_rank == 0 && (n % comm_sz != 0)) {
        my_s += sum(master_array + (scatter_sz * comm_sz), n % comm_sz);
    }

    if (my_rank == 0) {
        float recv_s[comm_sz - 1];
        MPI_Request requests[comm_sz - 1];
        s += my_s;

        for (int i=0; i<comm_sz-1; i++) {
            MPI_Irecv(
                &recv_s[i],         // Buffer
                1,                  // N of elements
                MPI_FLOAT,          // Type of elements
                i + 1,              // Src
                MPI_ANY_TAG,        // Tag
                MPI_COMM_WORLD,     // Comm
                &requests[i]        // Request
            );
        }

        MPI_Waitall(comm_sz - 1, requests, MPI_STATUS_IGNORE);

        for (int i=0; i<comm_sz-1; i++) {
            s += recv_s[i];
        }

    } else {
        MPI_Send(
            &my_s,              // Buffer
            1,                  // N of elements
            MPI_FLOAT,          // Type
            0,                  // Dest
            0,                  // Tag
            MPI_COMM_WORLD      // Comm
        );
    }

    free(master_array);

    if (0 == my_rank) {
        printf("Sum=%f, expected=%f\n", s, expected);
        if (s == expected) {
            printf("Test OK\n");
        } else {
            printf("Test FAILED\n");
        }
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
