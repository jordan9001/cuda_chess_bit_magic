
#include <stdio.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

// rook is 12, bishop is 9: keep in mind we only care about blocker locations in our mask, not possible move locations

#define TBL_BIT_PAD     1
#define MAX_MASK_BITS   (12 + TBL_BIT_PAD)

#define MAX_TRIES       100000
//#define MAX_TRIES       3000

#define NUM_SQ          64

#define SEED            9396939

#define DEVNUM          0

#define LOOPS_PER_CHECK 0x100

__device__ void genMaskedBoards(uint64_t mask, uint32_t numbits, uint64_t* cases)
{
    uint8_t shifts[MAX_MASK_BITS];
    uint64_t i, j, s;

    // build a shift table for the bit indexes of the mask
    j = 0;
    for (i = 0; i < 64; i++) {
        s = (1ll << i);

        if (s & mask) {
            shifts[j] = i;
            j += 1;
        }
    }
    // j should always be numbits

    // fill out the cases
    for (i = 0; i < (1ll << numbits); i++) {
        s = 0;
        for (j = 0; j < numbits; j++) {
            s |= ((i & (1ll << j)) << shifts[j]);
        }

        // s is our case to check
        cases[i] = s;
    }

    // count should always be 1<<numbits
}

// should be able to run all the masks at the same time
__global__ void findMagic(uint64_t* masks, uint64_t* out_magic)
{
    curandStateXORWOW_t state;
    uint64_t cases[1 << MAX_MASK_BITS];
    uint32_t used[1 << MAX_MASK_BITS];
    uint32_t numbits;
    uint32_t i, j;
    uint64_t magic, val;
    uint64_t mask;

    mask = masks[threadIdx.x];

    // seed the prng
    // fine if all threads have a identical state
    curand_init(SEED, 0, 0, &state);

    // get numbits from the mask
    numbits = (uint32_t)__popcll(mask);
    genMaskedBoards(mask, numbits, cases);

    memset(used, -1, sizeof(used));

    // now with our cases to check, start pulling random numbers and checking them against all cases
    for (i = 0; i < MAX_TRIES; i++) {
        magic = curand(&state);
        magic |= ((uint64_t)curand(&state)) << 32;

        //memset(used, 0, sizeof(used));

#pragma unroll 8
        for (j = 0; j < (1 << numbits); j++) {
            val = (magic * cases[j]) >> (64 - (numbits + TBL_BIT_PAD));

            // do we need to allow a bigger table? Or can we get it? Just try I guess

            // do we let all tables be the (1 << MAX_MASK_BITS) size with the same shift?
            // doing a tab = rook_moves[sq]; tab[((board_occ & rook_masks[sq]) * rook_magic[sq]) >> rook_shift[sq]]
            // shift from popcount of the mask? Or from a table, whatever


            // if we fail due to a collision, exit early
            if (used[val] == i) {
                magic = 0;
                break;
            }
            used[val] = i;
        }

        // if it worked exit early
        if (magic != 0) {
            out_magic[threadIdx.x] = magic;
            return; // could have waaaaay better usage if we queued tries all over instead of just quick exiting threads
        }
    }

    // sad, we didn't find one
}

// use all just to search for one
__global__ void findMagicOne(uint64_t mask, uint64_t* out_magic)
{
    curandStateXORWOW_t state;
    uint64_t cases[1 << MAX_MASK_BITS];
    uint32_t used[1 << MAX_MASK_BITS] = { 0 };
    uint32_t numbits;
    uint32_t i, j;
    uint64_t magic, val;
    int idx = (blockDim.x * blockIdx.x) + threadIdx.x;

    // seed the prng unique to the thread
    //curand_init(SEED, idx, 0, &state); // is unique sequence enough? Or do I need to offset by max tries?
    curand_init(SEED, 0, MAX_TRIES * idx, &state);

    // get numbits from the mask
    numbits = __popcll(mask);
    genMaskedBoards(mask, numbits, cases);


    // now with our cases to check, start pulling random numbers and checking them against all cases
    for (i = 1; i < MAX_TRIES; i++) {
        magic = curand(&state);
        magic |= ((uint64_t)curand(&state)) << 32;

        for (j = 0; j < (1ull << numbits); j++) {
            val = (magic * cases[j]) >> (64 - (numbits + TBL_BIT_PAD));

            // if we fail due to a collision, exit early
            if (used[val] == i) {
                magic = 0;
                break;
            }

            used[val] = i;
        }

        // if it worked exit early
        if (magic != 0) {
            // try to atomically set the value
            atomicCAS(out_magic, 0ll, magic);
            return;
        } else {
            // check if another thread solved it
            if (((i & (LOOPS_PER_CHECK - 1)) == 0) && (*out_magic != 0)) {
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

int doAllBoards()
{
    size_t i;
    uint64_t m;
    int got = 0;
    int res = -1;
    cudaError_t cudaStatus;
    uint64_t* masks;
    uint64_t* dev_masks;
    uint64_t* dev_magics;
    uint64_t* magics;
    size_t masks_len;
    clock_t c1, c2;
    double time_spent;

    // generate the masks (TODO and answers)
    genMasksAndAnswers(&masks, &masks_len);

    cudaStatus = initCuda();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda initializationfailed!");
        return 1;
    }

    // alloc memory for mask and out_magic
    cudaMalloc(&dev_masks, masks_len * sizeof(uint64_t));
    cudaMalloc(&dev_magics, masks_len * sizeof(uint64_t));

    // copy masks in
    cudaMemcpy(dev_masks, masks, masks_len * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemset(dev_magics, 0, masks_len * sizeof(uint64_t));

    // call the kernel
    int threadsPerBlock = (int)masks_len;
    int blocksPerGrid = 1;

    printf("Starting search\n");
    c1 = clock();
    findMagic<<<blocksPerGrid, threadsPerBlock>>>(dev_masks, dev_magics);
    
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
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    c2 = clock();
    time_spent = (double)(c2 - c1) / CLOCKS_PER_SEC;
    printf("Search took %f sec\n", time_spent);

    // get the results table
    magics = (uint64_t*)malloc(masks_len * sizeof(uint64_t));
    cudaMemcpy(magics, dev_magics, masks_len * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    for (i = 0; i < masks_len; i++) {
        m = magics[i];
        if (m != 0) {
            got += 1;
        }

        printf("%zu %llx\n", i % NUM_SQ, m);
    }

    printf("Got %d / %zu\n", got, masks_len);

    res = 0;

Error:
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!\n");
        return 1;
    }

    return res;
}

cudaError_t doOneBoard(uint64_t mask, uint64_t* dev_magic, uint64_t* out_magic, int threadsPerBlock, int blocksPerGrid)
{
    cudaError_t cudaStatus = cudaSuccess;
    uint64_t zero = 0x0;

    cudaMemcpy(dev_magic, &zero, sizeof(zero), cudaMemcpyHostToDevice);

    // call the kernel
    findMagicOne<<<blocksPerGrid, threadsPerBlock>>> (mask, dev_magic);

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
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
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
    threadsPerBlock = 128; // TODO profile this, see if occupancy with this

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

    // alloc memory for out_magic
    cudaMalloc(&dev_magic, sizeof(uint64_t));

    printf("Starting loop with %d BPG %d TPB, %d max tries (%d padding bits)\n", blocksPerGrid, threadsPerBlock, MAX_TRIES, TBL_BIT_PAD);

    tottries = ((uint64_t)blocksPerGrid) * ((uint64_t)threadsPerBlock) * (MAX_TRIES);
    printf("That's %llx tries\n", tottries);

    for (i = 0; i < masks_len; i++) {
    //DEBUG
    //for (i = 9; i <= 9; i++) {
        printf("Starting search %zu (%llx)\n", i, masks[i]);
        c1 = clock();

        cudaStatus = doOneBoard(masks[i], dev_magic, &magics[i], threadsPerBlock, blocksPerGrid);
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
