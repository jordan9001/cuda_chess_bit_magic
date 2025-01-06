// cudachess_bitboard_cpu.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <random>
#include <thread>
#include <vector>
#include <stdio.h>
#include <time.h>
#include <intrin.h>


// rook is 12, bishop is 9: keep in mind we only care about blocker locations in our mask, not possible move locations

#define TBL_BIT_PAD     0
#define MAX_MASK_BITS   (12 + TBL_BIT_PAD)

#define MAX_TRIES       3000000
//#define MAX_TRIES       3000

#define NUM_SQ          64

#define SEED            9396939

#define DEVNUM          0

#define LOOPS_PER_CHECK 0x100

void genMaskedBoards(uint64_t mask, uint32_t numbits, uint64_t* cases)
{
    uint8_t shifts[MAX_MASK_BITS];
    uint64_t i, j, s;

    // build a shift table for the bit indexes of the mask
    j = 0;
    for (i = 0; i < 64; i++) {
        s = (1ull << i);

        if (s & mask) {
            shifts[j] = (uint8_t)i - j;
            j += 1ull;
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
        //printf("................ - %016llx\n", mask);
        //printf("%016llx = %016llx\n", i, s);
    }

    // count should always be 1<<numbits
}

// use all just to search for one
void findMagicOne(uint64_t mask, uint64_t* out_magic, int idx)
{
    
    uint64_t cases[1 << MAX_MASK_BITS];
    uint32_t used[1 << MAX_MASK_BITS] = { 0 };
    uint32_t numbits;
    uint32_t i, j;
    uint64_t magic, val;

    // seed the prng unique to the thread
    std::seed_seq seed{ idx + SEED };
    std::mt19937_64 state(seed);

    // get numbits from the mask
    numbits = (uint32_t)__popcnt64(mask);
    genMaskedBoards(mask, numbits, cases);

    // now with our cases to check, start pulling random numbers and checking them against all cases
    for (i = 1; i < MAX_TRIES; i++) {
        magic = state();

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
            uint64_t expected = 0ll;
            _InterlockedCompareExchange64((__int64*)out_magic, (__int64)magic, 0ll);
            return;
        }
        else {
            // check if another thread solved it
            if (*out_magic != 0) {
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

int doOneBoard(uint64_t mask, uint64_t* out_magic, int threadsPerBlock)
{
    int i;
    *out_magic = 0;

    // spin up threads if needed
    if (threadsPerBlock > 1) {
        std::vector<std::thread> tv(threadsPerBlock);
        for (i = 0; i < threadsPerBlock; i++) {
            tv[i] = std::thread(findMagicOne, mask, out_magic, i);
        }
        for (i = 0; i < threadsPerBlock; i++) {
            tv[i].join();
        }
    }
    else {
        //launch just one
        findMagicOne(mask, out_magic, 0);
    }


    return 0;
}


int loopAllBoards() {
    uint64_t* masks;
    uint64_t* magics;
    size_t i;
    size_t masks_len;
    size_t count = 0;
    int threadsPerBlock;
    clock_t c1, c2;
    double time_spent;
    uint64_t tottries;
    int res = 0;

    // generate the masks (TODO and answers)
    genMasksAndAnswers(&masks, &masks_len);
    magics = (uint64_t*)malloc(sizeof(uint64_t) * masks_len);

    threadsPerBlock = 24;
    //DEBUG
    //threadsPerBlock = 1;

    printf("Starting loop with %d TPB, %d max tries (%d padding bits)\n", threadsPerBlock, MAX_TRIES, TBL_BIT_PAD);

    tottries = ((uint64_t)threadsPerBlock) * (MAX_TRIES);
    printf("That's %llx tries\n", tottries);

    //for (i = 0; i < masks_len; i++) {
    //DEBUG
    for (i = 0; i <= 1; i++) {
        printf("Starting search %zu (%llx)\n", i, masks[i]);
        c1 = clock();

        res = doOneBoard(masks[i], &magics[i], threadsPerBlock);
        if (res) {
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

    return res;
}

int main()
{
    return loopAllBoards();
}
