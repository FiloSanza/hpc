/****************************************************************************
 *
 * mpi-dot.c - Dot product
 *
 * Copyright (C) 2016--2021 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Dot product
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-11-01

The file [mpi-dot.c](mpi-dot.c) contains a MPI program that computes
the dot product between two arrays `a[]` and `b[]` of length $n$. The
dot product $s$ of two arrays `a[]` and `b[]` is defined as:

$$
s = \sum_{i = 0}^{n-1} a[i] \times b[i]
$$

In the provided program, the master performs the whole computation and
is therefore not parallel. The goal of this exercise is to write a
parallel version. Assume that, at the beginning of the program, `a[]`
and `b[]` are known only to the master. Therefore, they must be
distributed across the processes. Each process computes the scalar
product of the assigned portions of the arrays; the master then uses
`MPI_Reduce()` to sum the partial results and compute $s$.

You may initially assume that $n$ is an exact multiple of the number
of MPI processes $P$; then, relax this assumption and modify the
program so that it works with any array length $n$. The simpler
solution is to distribute the arrays using `MPI_Scatter()` and let the
master take care of any excess data. Another possibility is to use
`MPI_Scatterv()` to distribute the input unevenly across the
processes.

Compile with:

        mpicc -std=c99 -Wall -Wpedantic mpi-dot.c -o mpi-dot -lm

Execute with:

        mpirun -n P ./mpi-dot [n]

Example:

        mpirun -n 4 ./mpi-dot 1000

## Files

- [mpi-dot.c](mpi-dot.c)

***/
#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* for fabs() */
#include <assert.h>
#include <mpi.h>

/*
 * Compute sum { x[i] * y[i] }, i=0, ... n-1
 */
double dot( const double* x, const double* y, int n )
{
    double s = 0.0;
    int i;
    for (i=0; i<n; i++) {
        s += x[i] * y[i];
    }
    return s;
}

int main( int argc, char* argv[] )
{
    double *x = NULL, *y = NULL, *my_x = NULL, *my_y = NULL;
    double my_result = 0.0, result = 0.0;
    int i, n = 1000;
    int my_rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( 0 == my_rank ) {
        /* The master allocates the vectors */
        x = (double*)malloc( n * sizeof(*x) ); assert(x != NULL);
        y = (double*)malloc( n * sizeof(*y) ); assert(y != NULL);
        for ( i=0; i<n; i++ ) {
            x[i] = i + 1.0;
            y[i] = 1.0 / x[i];
        }
    }

    /* Compute how many elements each processor will have, distribute them evenly. */
    int sendcounts[comm_sz], displs[comm_sz];
    for (int i=0; i<comm_sz; i++) {
        sendcounts[i] = n / comm_sz + (i < (n % comm_sz) ? 1 : 0);
        displs[i] = (i > 0) ? displs[i-1] + sendcounts[i-1] : 0;
    }

    my_x = (double*)malloc( sendcounts[my_rank] * sizeof(*my_x) ); assert(my_x != NULL);
    my_y = (double*)malloc( sendcounts[my_rank] * sizeof(*my_y) ); assert(my_y != NULL);

    MPI_Request requests[2];
    MPI_Iscatterv(
        x,                      // Buffer
        sendcounts,             // Number of elements for each process
        displs,                 // Offsets
        MPI_DOUBLE,             // Send Type
        my_x,                   // Recv buffer
        sendcounts[my_rank],    // Recv count
        MPI_DOUBLE,             // Recv Type
        0,                      // Root
        MPI_COMM_WORLD,         // Comm
        &requests[0]            // Request
    );

    MPI_Iscatterv(
        y,                      // Buffer
        sendcounts,             // Number of elements for each process
        displs,                 // Offsets
        MPI_DOUBLE,             // Send Type
        my_y,                   // Recv buffer
        sendcounts[my_rank],    // Recv count
        MPI_DOUBLE,             // Recv Type
        0,                      // Root
        MPI_COMM_WORLD,         // Comm
        &requests[1]            // Request
    );

    MPI_Waitall(2, requests, MPI_STATUS_IGNORE);

    my_result = dot(my_x, my_y, sendcounts[my_rank]);

    MPI_Reduce(
        &my_result,         // Source Buffer
        &result,            // Recv Buffer
        1,                  // Count
        MPI_DOUBLE,         // Type
        MPI_SUM,            // Op   
        0,                  // Root
        MPI_COMM_WORLD      // Comm
    );

    free(my_x);
    free(my_y);

    if (0 == my_rank) {
        printf("Dot product: %f\n", result);
        if ( fabs(result - n) < 1e-5 ) {
            printf("Check OK\n");
        } else {
            printf("Check failed: got %f, expected %f\n", result, (double)n);
        }
    }

    free(x); /* if x == NULL, does nothing */
    free(y);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
