/****************************************************************************
 *
 * mpi-lookup.c - Parallel linear search
 *
 * Copyright (C) 2021, 2022 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Parallel linear search
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-11-09

Write an MPI program that finds the positions of all occurrences of a
given `key` in an unsorted integer array `v[]`. For example, if `v[] =
{1, 3, -2, 3, 4, 3, 3, 5, -10}` and `key = 3`, the program must
build the result array

        {1, 3, 5, 6}

whose elements correspond to the positions (indices) in `v[]` where
the value `key` is found. Assume that:

- The array `v[]` is initially defined in the local memory of process
  0 only;

- All processes know the value `key`, which is a compile-time
  constant;

- At the end, the result array must reside in the local memory of
  process 0.

![Figure 1: Communication scheme](mpi-lookup.png)

The program should operate as shown in Figure 1; `comm_sz` is the
number of MPI processes, and `my_rank` the rank of the process running
the code:

1. The master distributed `v[]` evenly across the processes. Assume
   that `n` is an exact multiple of `comm_sz`. Each process stores the
   local chunk in the `local_v[]` array of size `n / comm_sz`.

2. Each process computes the number `local_nf` of occurrences of `key`
   within `local_v[]`.

3. Each process creates a local array `local_result[]` of length
   `local_nf`, and fills it with the indices of the occurrences of
   `key` in `local_v[]`. **Warning**: indexes must refer to the
   **global** array `v[]` array, not `local_v[]`.

4. The processes use `MPI_Gather()` to concatenate the values of
   `local_nf` in an array `recvcounts[]` of length `comm_sz` owned by
   process 0.

5. Process 0 computes the _exclusive scane_ of `recvcounts[]`, and
   stores the result in a separate array `displs[]`. Only the master
   needs to know the content of `displs[]`, that is used at the next
   step.

6. All processes use `MPI_Gatherv()` to concatenate the arrays
   `local_result[]` to process 0. Process 0 uses the `displs[]` array
   from the previous step; all other processes do not need the
   displacements array, so they can pass a `NULL` reference.

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-lookup.c -o mpi-lookup -lm

To execute:

        mpirun -n P ./mpi-lookup [N]

Example:

        mpirun -n 4 ./mpi-lookup

## Files

- [mpi-lookup.c](mpi-lookup.c)

***/
#include <stdio.h>
#include <stdlib.h> /* for rand() */
#include <time.h>   /* for time() */
#include <assert.h>
#include <mpi.h>

void fill(int *v, int n)
{
    int i;
    for (i=0; i<n; i++) {
        v[i] = (rand() % 100);
    }
}

int main( int argc, char *argv[] )
{
    int my_rank, comm_sz;
    int n = 1000;       /* lunghezza array di input */
    int *v = NULL;      /* array di input */
    int *result = NULL; /* array degli indici delle occorrenze */
    int nf = 0;         /* numero di occorrenze trovate */
    const int KEY = 42; /* valore da cercare */

    MPI_Init( &argc, &argv );
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (argc > 1)
        n = atoi(argv[1]);

    if (my_rank == 0) {
        if ((n % comm_sz) != 0) {
            fprintf(stderr, "FATAL: array size (%d) must be a multiple of %d\n", n, comm_sz);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        /* The master initializes `v[]` */
        v = (int*)malloc(n * sizeof(*v)); assert(v != NULL);
        fill(v, n);
    }

    int *local_v = NULL;        /* local portion of `v[]` */
    int local_nf = 0;           /* n. of occurrences of `KEY` in `local_v[]` */
    int *displs = NULL;         /* `displs[]` used by `MPI_Gatherv()` */
    int *recvcounts = NULL;     /* `recvcounts[]` used by `MPI_Gatherv()` */
    int *local_result = NULL;   /* array of positions of `KEY` in `local_v[]` */

    /**
     ** Step 1: distribute `v[]` across all MPI processes
     **/
    const int local_size = n / comm_sz;
    const int base = my_rank * n / comm_sz;
    local_v = (int*)malloc(sizeof(int) * local_size); assert(local_v != NULL);
    MPI_Scatter(
        v,
        local_size,
        MPI_INT,
        local_v,
        local_size,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    /**
     ** Step 2: each process computes the number of occurrences of
     ** `KEY` in `local_v[]`
     **/
    
    for (int i=0; i<local_size; i++) {
        local_nf += local_v[i] == KEY;
    }

    /**
     ** Step 3: each process allocates an array `local_result[]` where
     ** the positions (indexes) of `KEY` in `local_v[]` are stored.
     ** It is essential that the positions refer to `v[]`, not to
     ** `local_v[]`.
     **/
    local_result = (int*)malloc(sizeof(int) * local_nf); assert(local_result != NULL);

    int idx = 0;
    for (int i=0; i<local_size; i++) {
        if (local_v[i] == KEY) {
            local_result[idx] = i + base;
            idx++;
        }
    }

    /**
     ** Step 4: Process 0 gathers all values `local_nf` into a local
     ** array `recvcounts[]` of size `comm_sz`
     **/

    if (my_rank == 0) {
        displs = (int*)malloc( comm_sz * sizeof(*displs) ); assert(displs != NULL);
        recvcounts = (int*)malloc( comm_sz * sizeof(*recvcounts) ); assert(recvcounts != NULL);
    }

    MPI_Gather(
        &local_nf,
        1,
        MPI_INT,
        recvcounts,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    /**
     ** Step 5: process 0 performs an exclusive scan of `recvcounts[]`
     ** and stores the result into a new array `displs[]`. Then,
     ** another array `result[]` is created, big enough to store all
     ** occurrences received from all processes.
     **/
    if (my_rank == 0) {
        displs[0] = 0;

        for (int i=1; i<comm_sz; i++) {
            displs[i]  = displs[i-1] + recvcounts[i-1];
        }

        nf = (displs[comm_sz - 1] + recvcounts[comm_sz - 1]);
        result = (int*)malloc(sizeof(int) * nf); assert(result != NULL);
    }

    /**
     ** Step 6: process 0 gathers `local_result[]` into `result[]`
     **/
    
    MPI_Gatherv(
        local_result,
        local_nf,
        MPI_INT,
        result,
        recvcounts,
        displs,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    free(displs);
    free(recvcounts);
    free(local_v);
    free(local_result);

    if (my_rank == 0) {
        printf("There are %d occurrences of %d\n", nf, KEY);
        printf("Positions: ");
        for (int i=0; i<nf; i++) {
            printf("%d ", result[i]);
            if (v[result[i]] != KEY) {
                fprintf(stderr, "\nFATAL: v[%d]=%d, expected %d\n", result[i], v[result[i]], KEY);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }
        printf("\n");
        free(v);
        free(result);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
