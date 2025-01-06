
#include <stdio.h>
#include <time.h>
#include <intrin.h>
#include <limits>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

// rook is 12, bishop is 9: keep in mind we only care about blocker locations in our mask, not possible move locations

#define TBL_BIT_PAD         0
#define MAX_MASK_BITS_NPAD  12
#define MAX_MASK_BITS       (MAX_MASK_BITS_NPAD + TBL_BIT_PAD)

#define MAX_TRIES           2000

//#define MAX_TRIES       2000 // similar number of cases to the 24 cpu cores doing 3000000 tries

#define NUM_SQ              64

#define SEED                9396939

#define DEVNUM              0

#define LOOPS_PER_CHECK     0x100

__host__ __device__ void genMaskedBoards(uint64_t mask, uint32_t numbits, uint64_t* cases)
{
    uint8_t shifts[MAX_MASK_BITS];
    uint64_t i, j, s;

    // build a shift table for the bit indexes of the mask
    j = 0;
    for (i = 0; i < 64; i++) {
        s = (1ll << i);

        if (s & mask) {
            shifts[j] = (uint8_t)(i - j);
            j += 1;
        }
    }
    // j should always be numbits

    // fill out the cases
    for (i = 0; i < (1ull << numbits); i++) {
        s = 0;
        for (j = 0; j < numbits; j++) {
            s |= ((i & (1ull << j)) << shifts[j]);
        }

        // s is our case to check
        cases[i] = s;
    }

    // count should always be 1<<numbits
}

// use all just to search for one
extern __shared__ uint64_t s_cases[];
__global__ void findMagicOne(uint64_t mask, const uint64_t* cases, uint32_t numbits, uint64_t* out_magic)
{
    curandStateXORWOW_t state;
    uint32_t bitnpad, shft;
    uint32_t i, j, jend;
    uint64_t magic, val;
    int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
    uint8_t used[1ull << (MAX_MASK_BITS - 3)];

    jend = 1ull << numbits;

    for (i = threadIdx.x; i < (1ull << numbits); i += blockDim.x) {
        s_cases[i] = cases[i];
    }

    // seed the prng unique to the thread
    //curand_init(SEED, idx, 0, &state); // is unique sequence enough? Or do I need to offset by max tries?
    curand_init(SEED, 0, MAX_TRIES * idx, &state);

    bitnpad = numbits + TBL_BIT_PAD;
    //memset(used, 0, (1ull << bitnpad) * sizeof(uint8_t));

    shft = (64 - bitnpad);


    // synch the threads because our shared memory init
    __syncthreads();

    // now with our cases to check, start pulling random numbers and checking them against all cases
    for (i = 1; i < MAX_TRIES; i++) {
        magic = curand(&state);
        magic |= ((uint64_t)curand(&state)) << 32;

        memset(used, 0, 1ull << (bitnpad - 3));

        for (j = 0; j < jend; j++) {
            val = s_cases[j];
            val = magic * val;
            val = val >> shft;

            // if we fail due to a collision, exit early
            // this is terrible for trying to keep warps together
            // also the used index depends on the cases index, so big stalls here
            if (used[val >> 3] & (1 << (val & 0x7))) {
                magic = 0;
                break; // removing this break doesn't change perf much
            }

            //used[val] = i;
            used[val >> 3] |= (1 << (val & 0x7));
        }

        // if it worked exit early
        if ((magic != 0) && (*out_magic == 0)) {
            // try to atomically set the value
            atomicCAS(out_magic, 0ll, magic);

            return;
        } else {
            // check if another thread solved it
            if (((i & (LOOPS_PER_CHECK - 1)) == 0) &&
                (*out_magic != 0)) {
                return;
            }
        }
    }
    // sad, we didn't find one
    return;
}

void printBitBoard(uint64_t bb)
{
    int32_t r;
    int32_t f;

    for (f = (7 << 3); f >= 0; f -= 8) {
        for (r = 0; r < 8; r++) {
            printf("%c", (bb & (1ll << (uint64_t)(f | r))) ? 'X' : '.');
        }
        printf("\n");
    }
}

cudaError_t initCuda()
{
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(DEVNUM);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
        goto Error;
    }

Error:

    return cudaStatus;
}

uint64_t rook_mask(uint32_t sq) {
    uint32_t i;
    uint64_t mask = 0;
    uint32_t rank = sq & 7;
    uint32_t file_sh = sq & (~7);

    // mask is just squares we need to check for collision, not squares we could move into

    for (i = rank + 1; i <= 6; i++) {
        mask |= (1ll << (uint64_t)(i + file_sh));
    }
    if (rank > 1) {
        for (i = rank - 1; i >= 1; i--) {
            mask |= (1ll << (uint64_t)(i + file_sh));
        }
    }

    for (i = file_sh + 8; i <= (6 << 3); i += 8) {
        mask |= (1ll << (uint64_t)(rank + i));
    }

    if (file_sh > 8) {
        for (i = file_sh - 8; i >= 8; i -= 8) {
            mask |= (1ll << (uint64_t)(rank + i));
        }
    }

    return mask;
}

uint64_t bishop_mask(uint32_t sq) {
    int32_t r;
    int32_t f;
    int32_t file_sh_off;
    uint64_t mask = 0;
    int32_t rank = sq & 7;
    int32_t file_sh = sq & (~7);

    // mask is just squares we need to check for collision, not squares we could move into

    for (r = 1; r <= 6; r++) {
        if (r == rank) {
            continue;
        }

        file_sh_off = (r - rank) << 3;

        f = file_sh - file_sh_off;
        if ((f <= (6 << 3)) && (f >= 8)) {
            mask |= (1ll << (uint64_t)(r + f));
        }

        f = file_sh + file_sh_off;
        if ((f <= (6 << 3)) && (f >= 8)) {
            mask |= (1ll << (uint64_t)(r + f));
        }
    }

    return mask;
}

void genMasksAndAnswers(uint64_t** masks, size_t* masks_len)
{
    // generate all the masks and answers
    // just masks for now, while we get it working
    size_t num_masks = NUM_SQ * 2;
    uint32_t i;
    uint64_t m;
    uint64_t* mptr;

    mptr = (uint64_t*)malloc(sizeof(uint64_t) * num_masks);

    for (i = 0; i < NUM_SQ; i++) {
        // rook
        m = rook_mask(i);
        mptr[i] = m;


        // bishop
        m = bishop_mask(i);
        mptr[i + NUM_SQ] = m;
    }

    *masks_len = num_masks;
    *masks = mptr;
}

cudaError_t doOneBoard(uint64_t mask, uint64_t* dev_magic, uint64_t* dev_cases, uint64_t* out_magic, int threadsPerBlock, int blocksPerGrid)
{
    cudaError_t cudaStatus = cudaSuccess;
    uint64_t zero = 0x0;
    uint32_t numbits;
    uint64_t cases[(1ull << MAX_MASK_BITS)];

    cudaStatus = cudaMemcpy(dev_magic, &zero, sizeof(zero), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to copy zero to dev_magic: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }


    // generate and copy over the cases
    numbits = (uint32_t)__popcnt64(mask);
    genMaskedBoards(mask, numbits, cases);

    cudaStatus = cudaMemcpy(dev_cases, cases, (1ull << numbits) * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to copy cases to dev_cases: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // call the kernel
    findMagicOne<<< blocksPerGrid, threadsPerBlock, sizeof(uint64_t) * (1ull << numbits) >>> (mask, (const uint64_t*)dev_cases, numbits, dev_magic);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // get the result
    cudaStatus = cudaMemcpy(out_magic, dev_magic, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Unable to copy out answer!\n");
        goto Error;
    }

Error:

    return cudaStatus;
}


int loopAllBoards() {
    cudaError_t cudaStatus;
    uint64_t* masks;
    uint64_t* dev_magic;
    uint64_t* dev_cases;
    uint64_t* magics;
    size_t i, count;
    size_t masks_len;
    int blocksPerGrid;
    int threadsPerBlock;
    cudaDeviceProp prop;
    clock_t c1, c2;
    double time_spent;
    uint64_t tottries;

    // generate the masks (TODO and answers)
    genMasksAndAnswers(&masks, &masks_len);
    magics = (uint64_t*)malloc(sizeof(uint64_t) * masks_len);

    cudaStatus = initCuda();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda initializationfailed!\n");
        return 1;
    }

    // get the ideal BPG / TPB
    threadsPerBlock = 128;

    cudaStatus = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerGrid, findMagicOne, threadsPerBlock, sizeof(uint64_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error calculating max blocks!\n");
        return 1;
    }

    cudaStatus = cudaGetDeviceProperties(&prop, DEVNUM);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error getting device properties!\n");
        return 1;
    }

    printf("%d multiprocessors detected, and %d recommended blocks per\n", prop.multiProcessorCount, blocksPerGrid);

    blocksPerGrid = prop.multiProcessorCount * blocksPerGrid;


    // alloc memory for out_magic and cases
    cudaMalloc(&dev_magic, sizeof(uint64_t));
    cudaMalloc(&dev_cases, sizeof(uint64_t) * (1ull << MAX_MASK_BITS));

    printf("Starting loop with %d BPG %d TPB, %d max tries (%d padding bits)\n", blocksPerGrid, threadsPerBlock, MAX_TRIES, TBL_BIT_PAD);

    tottries = ((uint64_t)blocksPerGrid) * ((uint64_t)threadsPerBlock) * (MAX_TRIES);
    printf("That's %llx tries\n", tottries);

    //for (i = 0; i < masks_len; i++) {
    //DEBUG
    for (i = 0; i <= 1; i++) {
        printf("Starting search %zu (%llx)\n", i, masks[i]);
        c1 = clock();

        cudaStatus = doOneBoard(masks[i], dev_magic, dev_cases, &magics[i], threadsPerBlock, blocksPerGrid);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Error doing board %zu\n", i);
            break;
        }

        c2 = clock();
        time_spent = (double)(c2 - c1) / CLOCKS_PER_SEC;
        printf("Search took %f sec (%f per mtry if 0)\n", time_spent, time_spent / (((double)tottries) / 1000000.0));

        printf("%zu = %llx\n", i, magics[i]);

    }

    count = 0;
    for (i = 0; i < masks_len; i++) {
        if (magics[i] != 0) {
            count += 1;
        }
    }
    printf("%zu / %zu found\n", count, masks_len);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!\n");
        return 1;
    }

    return 0;
}


int main()
{
    // Questions:
    // Is Nvidia nsight supposed to take so long to analyze? If I lower the max iter to like 100 is that still going to get a reasonable view?
    // How does this compare to a CPU for per-test time?
    // The wiki says "it takes less than a second to find magic numbers for rooks and bishops for all squares"?
    // Why is this so slow?
    
    return loopAllBoards();
    //return doAllBoards();
}
