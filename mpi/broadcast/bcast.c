/****************************************************************************
 *
 * mpi-my-bcast.c - Broadcast using point-to-point communications
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
% HPC - Broadcast using point-to-point communication
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-10-24

The purpose of this exercise is to implement the function

        void my_Bcast(int *v)

which realizes a _broadcast_ communication, where the value `*v` that
resides on the local memory of process 0 is sent to all other
processes. In practice, this function should behave as

```C
MPI_Bcast(v,             \/\* buffer   \*\/
          1,             \/\* count    \*\/
          MPI_INT,       \/\* datatype \*\/
          0,             \/\* root     \*\/
          MPI_COMM_WORLD \/\* comm     \*\/
          );
```

> **Note**. `MPI_Bcast()` must always be preferred to any home-made
> solution. The purpose of this exercise is to learn how `MPI_Bcast()`
> might be implemented, although the actual implementations are
> architecture-dependent.

To implement `my_Bcast()`, each process determines its own rank $p$
and the number $P$ of active MPI processes.

Then, process 0:

- sends `*v` to processes $(2p + 1)$ and $(2p + 2)$, provided that
  they exist.

Any other process $p>0$:

- receives an integer from $(p - 1)/2$ and stores it in `*v`;

- sends `*v` to processes $(2p + 1)$ and $(2p + 2)$, provided that
  they exist.

For example, with $P = 15$ you get the communication scheme
illustrated in Figure 1; arrows indicate point-to-point
communications, numbers indicate the rank of processes. The procedure
described above works correctly for any $P$.

![Figure 1: Broadcast communication with $P = 15$ processes](mpi-my-bcast.png)

The file [mpi-my-bcast.c](mpi-my-bcast.c) contains the skeleton of the
`my_Bcast()` function. Complete the implementation using
point-to-point send/receive operations.

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-my-bcast.c -o mpi-my-bcast

To execute:

        mpirun -n 4 ./mpi-my-bcast

## Files

- [mpi-my-bcast.c](mpi-my-bcast.c)

***/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define LEFT(x) ((2 * (x) + 1))
#define RIGHT(x) ((2 * (x) + 2))
#define PARENT(x) (((x - 1) / 2))

void my_Bcast(int *v)
{
    int my_rank, comm_sz;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    if (my_rank > 0) {
        MPI_Recv(
            v,                 // Buffer
            1,                  // Count
            MPI_INT,            // Datatype
            PARENT(my_rank),    // Source process
            MPI_ANY_TAG,        // Tag
            MPI_COMM_WORLD,     // Comm
            MPI_STATUS_IGNORE   // Status
        );
    }

    const int left = (LEFT(my_rank) < comm_sz) ? LEFT(my_rank) : MPI_PROC_NULL;
    const int right = (RIGHT(my_rank) < comm_sz) ? RIGHT(my_rank) : MPI_PROC_NULL;
    MPI_Send(
        v,                  // Buffer
        1,                  // Count
        MPI_INT,            // Datatype
        left,               // Dest
        0,                  // Tag
        MPI_COMM_WORLD      // Comm
    );
    MPI_Send(
        v,                  // Buffer
        1,                  // Count
        MPI_INT,            // Datatype
        right,               // Dest
        0,                  // Tag
        MPI_COMM_WORLD      // Comm
    );
}


int main( int argc, char *argv[] )
{
    const int SENDVAL = 999; /* valore che viene inviato agli altri processi */
    int my_rank;
    int v;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if ( 0 == my_rank ) {
        v = SENDVAL; /* only process 0 sets the value to be sent */
        printf("Sending %d\n", v);
    } else {
        v = -1; /* all other processes set v to -1; if everything goes well, the value will be overwritten with the value received from the master */
    }


    my_Bcast(&v);

    if ( v == 999 ) {
        printf("OK\n");
    } else {
        printf("ERROR\n");
    }
    printf("Process %d received %d\n", my_rank, v);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
