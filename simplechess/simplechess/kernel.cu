
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdint.h>

#include <array>

constexpr int32_t NUMSQ = 64;
constexpr int32_t WIDTH = 8;


enum Piece {
    w_pawn = 0x0,
    w_knight,
    w_king,
    w_bishop,
    w_rook,
    w_queen,

    b_pawn = 0x8,
    b_knight,
    b_king,
    b_bishop,
    b_rook,
    b_queen,

    empty = 0xf,
};

struct BoardFlags {
    uint8_t enpassant_sq : 6;
    uint8_t b_turn       : 1;
    uint8_t w_can_oo     : 1;
    uint8_t w_can_ooo    : 1;
    uint8_t b_can_oo     : 1;
    uint8_t b_can_ooo    : 1;
};

struct Board {
    uint8_t layout[NUMSQ >> 1]; // 4 bits per piece
    uint64_t bit_occupied;
    uint64_t bit_white_occupied;
    BoardFlags flags;
    int32_t parent_idx; // for linking our move tree
};

//TODO Where should we keep this?
// Maybe shared memory and each warp gets a different ring buffer and starting position?
struct RingBuf {
    int32_t start; // need to keep around processed ones to have a tree to walk back?
    int32_t next; // cursor for consuming work
    int32_t end; // cursor for producing work
    int32_t cap; // maximum index
    Board* buf;
};

struct MagicInfoStarter {
    uint64_t mask;
    uint64_t num;
    int32_t shift;
};

struct MagicInfo {
    uint64_t mask;
    uint64_t num;
    int32_t shift;
    const uint64_t* table;
};

constexpr MagicInfoStarter starter(uint64_t mask, int32_t padding,  uint64_t magic) {
    // get hamming weight of mask and add padding
    int32_t bits = 0;
    
    for (uint32_t i = 0; i < 64; i++) {
        if (mask & (1ull << i)) {
            bits += 1;
        }
    }

    bits += padding;

    return MagicInfoStarter{ mask, magic, 64 - bits };
}

constexpr MagicInfoStarter bishop_starter[NUMSQ] = {
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),

    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),

    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),

    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),


    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),

    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),

    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),

    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
};

constexpr MagicInfoStarter rook_starter[NUMSQ] = {
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),

    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),

    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),

    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),


    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),

    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),

    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),

    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x101010101017e, 0, 0xe880006cc0017918),
};

constexpr size_t max_magic_lut_mem(const MagicInfoStarter* starter)
{
    size_t amt = 0;
    // given the shifts, count up the amount of memory needed
    for (int i = 0; i < NUMSQ; i++) {
        size_t bits = 64 - starter[i].shift;
        amt += 1ull << bits;
    }

    return amt;
}

constexpr uint64_t lut_mem_bishop[max_magic_lut_mem(bishop_starter)];
constexpr uint64_t lut_mem_rook[max_magic_lut_mem(rook_starter)];

constexpr auto make_magic(const MagicInfoStarter* starter, const uint64_t* lut_mem_buf)
{
    MagicInfo mi_arr[NUMSQ] = {};
    int32_t i = 0;
    for (i = 0; i < NUMSQ; i++) {
        mi_arr[i] = MagicInfo{ starter[i].mask, starter[i].num, starter[i].shift, lut_mem_buf};

        // go and fill out the answers



        size_t bits = 64 - starter[i].shift;
        lut_mem_buf += (1ull << bits);
    }
    return mi_arr;
}

constexpr auto lut_magic_bishop = make_magic(bishop_starter, lut_mem_bishop);

constexpr auto lut_magic_rook = make_magic(rook_starter, lut_mem_rook);

//TODO constexpr things my LUTs are too large to compute, so I need to generate a .h file to include I think


__device__ __host__ void evaluate() {
    // given a position, evaluate it's guessed value
    //TODO

    // get piece count balance
    //TODO
    
    // rate pieces by position and enemy count
    //TODO
}

__device__ __host__ void expand(Board* board_state) {
    int32_t sq;
    int32_t i;
    uint8_t piece;
    uint8_t cpiece;
    uint64_t moves;
    uint64_t move_mask;
    int32_t movesq;
    uint8_t b_turn;
    MagicInfo* magic;
    Board* out_board;
    // given a position, explore it to the desired depth

    b_turn = board_state->flags.b_turn;

    for (sq = 0; sq < NUMSQ; sq++) {
        cpiece = board_state->layout[sq >> 1]; //TODO could do at least 2 at a time

        // branchless shift if in odd square
        cpiece = (cpiece >> ((sq & 1) << 2)) & 0xf;

        if ((cpiece == empty) || (b_turn != (cpiece >> 3))) {
            continue; // big divergance here
        }

        // scrub the piece's color
        piece = (cpiece & 0x7);
        
        if (piece == w_bishop || piece == w_rook || piece == w_queen) {
            moves = 0;

            // do both rook and bishop moves for queen

            if (piece != w_rook) {
                magic = &lut_magic_bishop[sq];
                moves |= magic->table[((board_state->bit_occupied) * magic->num) >> magic->shift];
            }

            if (piece != w_bishop) {
                magic = &lut_magic_rook[sq];
                moves |= magic->table[((board_state->bit_occupied) * magic->num) >> magic->shift];
            }

        } else if (piece == w_pawn) {
            // handle all the pawn nonsense
            if (/*TODO ignore for now?*/ 0 && (sq < (2 * WIDTH) || sq >= (6 * WIDTH))) {
                // handle enpassant, double move, and promotion
                //TODO
                
                // don't fall through for promotion, need to replace our own piece here
                //TODO
                
                // double move needs updated flags
                //TODO
            }
            else {
                // normal pawn moves
                moves = 0;
                movesq = (cpiece == b_pawn) ? sq - WIDTH : sq + WIDTH;
                move_mask = 1ull << movesq;

                // only move forward if unoccupied at all
                moves |= (move_mask ^ (board_state->bit_occupied & move_mask));

                // only move diagonal if occupied by the enemy
                if ((movesq & (WIDTH - 1)) != 0) {
                    move_mask >>= 1;
                    moves |= (cpiece == b_pawn) ?
                        (move_mask & board_state->bit_white_occupied) :
                        ((board_state->bit_occupied & move_mask) ^ board_state->bit_white_occupied);
                }

                if ((movesq & (WIDTH - 1)) != (WIDTH - 1)) {
                    move_mask <<= 2;
                    moves |= (cpiece == b_pawn) ?
                        (move_mask & board_state->bit_white_occupied) :
                        ((board_state->bit_occupied & move_mask) ^ board_state->bit_white_occupied);
                }
                
            }
        } else {
            // knights and kings are simple lookups
            moves = lut_nk_moves[piece - w_knight][sq];

            // except oo and ooo
            // those need new flags as well
            //TODO
        }

        // now fork for each move
        for (movesq = 0; movesq < NUMSQ; movesq++) {
            move_mask = 1ull << movesq;
            if (move_mask & moves) {
                // obtain an out_board in our ring_buffer
                //TODO
                
                //TODO heuristics to search further levels deep for captures

                // copy over board
                *out_board = *board_state;

                // empty our current position
                out_board->layout[sq >> 1] &= (0xf << ((sq & 1) << 2));

                // empty the dest square
                out_board->layout[movesq >> 1] &= (0xf << ((movesq & 1) << 2));

                // or in our piece
                out_board->layout[movesq >> 1] |= (cpiece << ((movesq & 1) << 2));

                // switch turns
                out_board->flags.b_turn = !b_turn;

                //TODO update flags where needed somehow? Need tables for xoring flags?

            }
        }
    }
}

int main()
{
    printf("Done\n");

    return 0;
}

