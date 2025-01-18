
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdint.h>

#include <stdexcept>

constexpr int32_t WIDTH_SHIFT = 3;
constexpr int32_t WIDTH = (1 << WIDTH_SHIFT);
constexpr int32_t HEIGHT = WIDTH;
constexpr int32_t NUMSQ = (WIDTH * HEIGHT);



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

__device__ __host__ void evaluate() {
    // given a position, evaluate it's guessed value
    //TODO

    // get piece count balance
    //TODO
    
    // rate pieces by position and enemy count
    //TODO
}

__device__ __host__ void expand(Board* board_state, LUTs* luts) {
    int32_t sq;
    uint8_t piece;
    uint8_t cpiece;
    uint64_t moves;
    uint64_t move_mask;
    int32_t movesq;
    uint8_t b_turn;
    const MagicInfo* magic;
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
                magic = &luts->lut_magic_bishop.arr[sq];
                moves |= magic->table[((board_state->bit_occupied) * magic->num) >> magic->shift];
            }

            if (piece != w_bishop) {
                magic = &luts->lut_magic_rook.arr[sq];
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
        } else if (piece == w_king) {
            // knights and kings are simple lookups
            moves = luts->lut_nk_moves[piece - w_knight].arr[sq];

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

