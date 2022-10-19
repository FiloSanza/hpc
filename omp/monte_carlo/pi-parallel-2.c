/****************************************************************************
 *
 * omp-pi.c - Monte Carlo approximation of PI
 *
 * Copyright (C) 2017--2021 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Monte Carlo approximation of $\pi$
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-08-09

The file [omp-pi.c](omp-pi.c) contains a serial program for computing
the approximate value of $\pi$ using a Monte Carlo algorithm. Monte
Carlo algorithms use pseudorandom numbers to compute an approximation
of some quantity of interest.

![Figure 1: Monte Carlo computation of the value of $\pi$](pi_Monte_Carlo.png)

The idea is quite simple (see Figure 1). We generate $N$ random points
uniformly distributed inside the square with corners at $(-1, -1)$ and
$(1, 1)$. Let $x$ be the number of points that lie inside the circle
inscribed in the square; then, the ratio $x / N$ is an approximation
of the ratio between the area of the circle and the area of the
square. Since the area of the circle is $\pi$ and the area of the
square is $4$, we have $x/N \approx \pi / 4$ which yelds $\pi \approx
4x / N$. This estimate becomes more accurate as the number of points
$N$ increases.

Modify the serial program to make use of shared-memory parallelism
with OpenMP. Start with a version that uses the `omp parallel`
construct. Let $P$ be the number of OpenMP threads; then, the program
operates as follows:

1. The user specifies the number $N$ of points to generate as a
   command-line parameter, and the number $P$ of OpenMP threads using
   the `OMP_NUM_THREADS` environment variable.

2. Thread $p$ generates $N/P$ points using the provided function
   `generate_points()`, and stores the result in `inside[p]` where
   `inside[]` is an integer array of length $P$. The array must be
   declared outside the parallel region since it must be shared across
   all OpenMP threads.

3. At the end of the parallel region, the master (thread 0) computes
   the sum of the values in the `inside[]` array, and from that value
   the approximation of $\pi$.

You may initially assume that the number of points $N$ is an integer
multiple of $P$; when you get a working program, relax this assumption
to make the computation correct for any value of $N$.

Compile with:

        gcc -std=c99 -fopenmp -Wall -Wpedantic omp-pi.c -o omp-pi -lm

Run with:

        ./omp-pi [N]

For example, to compute the approximate value of $\pi$ using $P=4$
OpenMP threads and $N=20000$ points:

        OMP_NUM_THREADS=4 ./omp-pi 20000

## File2

- [omp-pi.c](omp-pi.c)

***/

/* The rand_r() function is available only if _XOPEN_SOURCE=600 */
#define _XOPEN_SOURCE 600
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* for fabs */

/* Generate `n` random points within the square with corners (-1, -1),
   (1, 1); return the number of points that fall inside the circle
   centered ad the origin with radius 1. */
unsigned int generate_points( unsigned int n )
{
    unsigned int i, n_inside = 0;
    /* The C function rand() is _NOT_ thread-safe, since it uses a
       global (shared) seed. Therefore, it can not be used inside an
       parallel region. We use rand_r() with an explicit per-thread
       seed. However, this means that in general the result computed
       by this program will depend on the number of threads used, and
       not only on the number of points that are generated. */
    unsigned int my_seed = 17 + 19*omp_get_thread_num();
    for (i=0; i<n; i++) {
        /* Generate two random values in the range [-1, 1] */
        const double x = (2.0 * rand_r(&my_seed)/(double)RAND_MAX) - 1.0;
        const double y = (2.0 * rand_r(&my_seed)/(double)RAND_MAX) - 1.0;
        if ( x*x + y*y <= 1.0 ) {
            n_inside++;
        }
    }
    return n_inside;
}

int main( int argc, char *argv[] )
{
    unsigned int n_points = 10000;
    const double PI_EXACT = 3.14159265358979323846;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n_points]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n_points = atol(argv[1]);
    }

    printf("Generating %u points...\n", n_points);
    const double tstart = omp_get_wtime();
    const int num_threads = omp_get_max_threads();

    unsigned int n_inside[num_threads];

#pragma omp parallel num_threads(num_threads) default(none) shared(n_points, num_threads, n_inside)
    {
        const int my_id = omp_get_thread_num();
        const int my_points = (n_points * (my_id + 1) / num_threads) + (n_points * my_id)/num_threads;
        const unsigned int my_inside = generate_points(my_points);

        n_inside[my_id] = my_inside;
    }

    unsigned int tot_inside = 0;
    for (int i=0; i<num_threads; i++) {
        tot_inside += n_inside[i];
    }

    const double elapsed = omp_get_wtime() - tstart;
    const double pi_approx = 4.0 * tot_inside / (double)n_points;
    printf("PI approximation %f, exact %f, error %f%%\n", pi_approx, PI_EXACT, 100.0*fabs(pi_approx - PI_EXACT)/PI_EXACT);
    printf("Elapsed time: %f\n", elapsed);

    return EXIT_SUCCESS;
}
