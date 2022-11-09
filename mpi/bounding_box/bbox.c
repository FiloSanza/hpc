/****************************************************************************
 *
 * mpi-bbox.c - Bounding box of a set of rectangles
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
% HPC - Bounding box of a set of rectangles
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-11-09

Write a parallel program that computes the _bounding box_ of a set of
rectangles. The bounding box is the rectangle of minimal area that
contains all the given rectangles (see Figure 1).

![Figure 1: Bounding box of a set of rectangles](mpi-bbox.png)

The progra reads the coordinates of the rectangles from a test
file. The first row contains the number $N$ of rectangles; $N$ lines
follow, each consisting of four space-separated values ​​`x1[i] y1[i]
x2[i] y2[i]` of type `float`. These values are the coordinates of the
opposite corners of each rectangle: (`x1[i], y1[i]`) is the top left,
while (`x2[i], y2[i]`) is the bottom right corner.

You are provided with a serial implementation [mpi-bbox.c](mpi-bbox.c)
where process 0 performs the entire computation. The purpose of this
exercise is to parallelize the program so that $P$ MPI processes
cooperate for determining the bounding box. Only process 0 can read
the input and write the output.

The parallel program should operated according to the following steps:

1. Process 0 reads the data from the input file; you can initially
   assume that the number of rectangles $N$ is a multiple of the
   number $P$ of MPI processes.

2. Process 0 broadcasts $N$ to all processes using `MPI_Bcast()`.  The
   input coordinates are scattered across the processes, so that each
   one receives the data of $N/P$ rectangles

3. Each process computes the bounding box of the rectangles assigned
   to it.

4. The master uses `MPI_Reduce()` to compute the coordinates of the
   corners of the bounding box using the operators of `MPI_MIN` and`
   MPI_MAX` reduction operators.

To generate additional random inputs you can use
[bbox-gen.c](bbox-gen.c); usage instructions are at the beginning of
the source code.

When you have a working program, try to relax the assumption that $N$
is multiple of $P$.

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-bbox.c -o mpi-bbox -lm

To execute:

        mpirun -n P ./mpi-bbox FILE

Example:

        mpirun -n 4 ./mpi-bbox bbox-1000.in

## Files

- [mpi-bbox.c](mpi-bbox.c)
- [bbox-gen.c](bbox-gen.c) (this program generates random inputs for `mpi-bbox.c`)
- [bbox-1000.in](bbox-1000.in)
- [bbox-10000.in](bbox-10000.in)
- [bbox-100000.in](bbox-100000.in)

***/
#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* for fminf() */
#include <assert.h>
#include <mpi.h>

/* Compute the bounding box of |n| rectangles whose opposite vertices
   have coordinates (|x1[i]|, |y1[i]|), (|x2[i]|, |y2[i]|). The
   opposite corners of the bounding box will be stored in (|xb1|,
   |yb1|), (|xb2|, |yb2|) */
void bbox( const float *x1, const float *y1, const float* x2, const float *y2,
           int n,
           float *xb1, float *yb1, float *xb2, float *yb2 )
{
    int i;
    assert(n > 0);
    *xb1 = x1[0];
    *yb1 = y1[0];
    *xb2 = x2[0];
    *yb2 = y2[0];
    for (i=1; i<n; i++) {
        *xb1 = fminf( *xb1, x1[i] );
        *yb1 = fmaxf( *yb1, y1[i] );
        *xb2 = fmaxf( *xb2, x2[i] );
        *yb2 = fminf( *yb2, y2[i] );
    }
}

int main( int argc, char* argv[] )
{
    float *x1, *y1, *x2, *y2;
    float xb1, yb1, xb2, yb2;
    int N;
    int my_rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if ( (0 == my_rank) && (argc != 2) ) {
        printf("Usage: %s [inputfile]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    x1 = y1 = x2 = y2 = NULL;

    /* [TODO] This is not a true parallel version since the master
       does everything */
    if ( 0 == my_rank ) {
        FILE *in = fopen(argv[1], "r");
        int i;
        if ( in == NULL ) {
            fprintf(stderr, "Cannot open %s for reading\n", argv[1]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (1 != fscanf(in, "%d", &N)) {
            fprintf(stderr, "FATAL: cannot read number of boxes\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        /* Remove the following check if your implementation supports
           every value of N */
        if (N % comm_sz) {
            fprintf(stderr, "The number of rectangles (%d) must be a multiple of the communicator size (%d)\n", N, comm_sz);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        x1 = (float*)malloc(N * sizeof(*x1)); assert(x1 != NULL);
        y1 = (float*)malloc(N * sizeof(*y1)); assert(y1 != NULL);
        x2 = (float*)malloc(N * sizeof(*x2)); assert(x2 != NULL);
        y2 = (float*)malloc(N * sizeof(*y2)); assert(y2 != NULL);
        for (i=0; i<N; i++) {
            if (4 != fscanf(in, "%f %f %f %f", &x1[i], &y1[i], &x2[i], &y2[i])) {
                fprintf(stderr, "FATAL: cannot read box %d\n", i);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            assert(x1[i] < x2[i]);
            assert(y1[i] > y2[i]);
        }
        fclose(in);
    }

    MPI_Bcast(
        &N,         // Buffer
        1,          // Count
        MPI_INT,    // Type
        0,
        MPI_COMM_WORLD
    );

    const int local_n = N / comm_sz;
    float* local_x1 = (float*)malloc(local_n * sizeof(float)); assert(local_x1);
    float* local_x2 = (float*)malloc(local_n * sizeof(float)); assert(local_x2);
    float* local_y1 = (float*)malloc(local_n * sizeof(float)); assert(local_y1);
    float* local_y2 = (float*)malloc(local_n * sizeof(float)); assert(local_y2);
    float local_x[2], local_y[2];

    MPI_Request requests[4];
    MPI_Iscatter(
        x1,             // Sendbuff
        local_n,        // Num of elements
        MPI_FLOAT,      // Send Type
        local_x1,       // Recv buffer
        local_n,        // Num of elements
        MPI_FLOAT,      // Recv type
        0,
        MPI_COMM_WORLD,
        &requests[0]
    );

    MPI_Iscatter(
        x2,             // Sendbuff
        local_n,        // Num of elements
        MPI_FLOAT,      // Send Type
        local_x2,       // Recv buffer
        local_n,        // Num of elements
        MPI_FLOAT,      // Recv type
        0,
        MPI_COMM_WORLD,
        &requests[1]
    );

    MPI_Iscatter(
        y1,             // Sendbuff
        local_n,        // Num of elements
        MPI_FLOAT,      // Send Type
        local_y1,       // Recv buffer
        local_n,        // Num of elements
        MPI_FLOAT,      // Recv type
        0,
        MPI_COMM_WORLD,
        &requests[2]
    );

    MPI_Iscatter(
        y2,             // Sendbuff
        local_n,        // Num of elements
        MPI_FLOAT,      // Send Type
        local_y2,       // Recv buffer
        local_n,        // Num of elements
        MPI_FLOAT,      // Recv type
        0,
        MPI_COMM_WORLD,
        &requests[3]
    );

    MPI_Waitall(4, requests, MPI_STATUS_IGNORE);

    /* Compute the bounding box */
    bbox(
        local_x1,
        local_y1, 
        local_x2, 
        local_y2, 
        local_n, 
        &local_x[0], 
        &local_y[0], 
        &local_x[1], 
        &local_y[1]
    );

    float min_x = fminf(local_x[0], local_x[1]);
    float max_x = fmaxf(local_x[0], local_x[1]);
    float min_y = fminf(local_y[0], local_y[1]);
    float max_y = fmaxf(local_y[0], local_y[1]);

    MPI_Ireduce(
        &min_x,                 // Send buffer
        &xb1,                   // Recv buffer
        1,                      // Count
        MPI_FLOAT,              // Type
        MPI_MIN,                // Op
        0,                      // Root
        MPI_COMM_WORLD,
        &requests[0]
    );

    MPI_Ireduce(
        &max_x,                 // Send buffer
        &xb2,                   // Recv buffer
        1,                      // Count
        MPI_FLOAT,              // Type
        MPI_MAX,                // Op
        0,                      // Root
        MPI_COMM_WORLD,
        &requests[1]
    );

    MPI_Ireduce(
        &min_y,                 // Send buffer
        &yb1,                   // Recv buffer
        1,                      // Count
        MPI_FLOAT,              // Type
        MPI_MIN,                // Op
        0,                      // Root
        MPI_COMM_WORLD,
        &requests[2]
    );

    MPI_Ireduce(
        &max_y,                 // Send buffer
        &yb2,                   // Recv buffer
        1,                      // Count
        MPI_FLOAT,              // Type
        MPI_MAX,                // Op
        0,                      // Root
        MPI_COMM_WORLD,
        &requests[3]
    );

    MPI_Waitall(4, requests, MPI_STATUS_IGNORE);

    if (my_rank == 0) {
        /* Print bounding box */
        printf("bbox: %f %f %f %f\n", xb1, yb1, xb2, yb2);
    }

    /* Free the memory */
    free(x1);
    free(y1);
    free(x2);
    free(y2);
    free(local_x1);
    free(local_y1);
    free(local_x2);
    free(local_y2);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
