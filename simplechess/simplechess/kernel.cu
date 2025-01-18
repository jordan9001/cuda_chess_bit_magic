
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

    constexpr MagicInfo() : mask(0), num(0), shift(0), table(nullptr) {};

    constexpr MagicInfo(uint64_t m, uint64_t n, int32_t s, const uint64_t* t)
        : mask(m), num(n), shift(s), table(t) {};
};

template<size_t N>
struct MagicInfoArray {
    uint64_t lut_mem[N];
    MagicInfo arr[NUMSQ];

    constexpr MagicInfoArray() : arr{}, lut_mem{} {};

    constexpr static MagicInfoArray make_magic_gen(const MagicInfoStarter* starter, uint64_t(*get_ans)(int32_t, uint64_t))
    {
        MagicInfoArray mi_arr{};
        int32_t memidx = 0;
        int32_t sq = 0;

        int8_t shifts[64] = { -1 };

        for (int32_t f = 0; f < HEIGHT; f++) {
            for (int32_t r = 0; r < WIDTH; r++) {
                sq = r + (f * WIDTH);

                uint64_t mask = starter[sq].mask;
                uint64_t magic = starter[sq].num;
                int32_t shiftamt = starter[sq].shift;

                mi_arr.arr[sq] = MagicInfo{ mask, magic, shiftamt, &mi_arr.lut_mem[memidx] };

                size_t mem_sz = (1ull << (64 - shiftamt));

                // For each possible permutation of the masked bits, generate the reachable moves (including captures)
                // first build our shift table for the mask
                uint64_t popcount = 0;
                for (uint64_t bit_i = 0; bit_i < 64; bit_i++) {
                    uint64_t s = (1ull << bit_i);

                    if (s & mask) {
                        shifts[popcount] = (int8_t)(bit_i - popcount);
                        popcount++;
                    }
                }

                // now for each case from 0 to (1 << popcount) we can generate an answer

                for (uint64_t poscase = 0; poscase < (1ull << popcount); poscase++) {
                    uint64_t maskcase = 0;
                    for (uint64_t bit_i = 0; bit_i < popcount; bit_i++) {
                        maskcase |= ((poscase & (1ull << bit_i)) << shifts[bit_i]);
                    }

                    // take our case and make an index from it
                    uint64_t mem_idx = (magic * maskcase) >> shiftamt;

                    if (mem_idx >= mem_sz) {
                        throw std::logic_error("Magic Index too big!");
                    }

                    // fill out the allowed moves with this mask
                    mi_arr.lut_mem[memidx + mem_idx] = get_ans(sq, maskcase);
                }

                memidx += (int32_t)mem_sz;
            }
        }
        return mi_arr;
    }

    constexpr static MagicInfoArray make_magic_bishop(const MagicInfoStarter* starter)
    {
        return make_magic_gen(starter, get_bishop_moves);
    }

    constexpr static MagicInfoArray make_magic_rook(const MagicInfoStarter* starter)
    {
        return make_magic_gen(starter, get_rook_moves);
    }
};

constexpr MagicInfoStarter starter(uint64_t mask, int32_t padding, uint64_t magic) {
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

//TODO I forgot about indexes that map to the same moveset because of shadowed pieces
// so these could all be smaller
constexpr MagicInfoStarter bishop_starter[NUMSQ] = {
    starter(0x40201008040200, 0, 0x9fe0a01d1d0700b2),
    starter(0x402010080400, 0, 0x8c08b87803a2d30f),
    starter(0x4020100a00, 0, 0x1a18080b03b9230e),
    starter(0x40221400, 0, 0x79680c8f0e74b778),
    starter(0x2442800, 0, 0xb18d104185a62863),
    starter(0x204085000, 0, 0x7db0108405c7842),
    starter(0x20408102000, 0, 0xd656c23030385fa4),
    starter(0x2040810204000, 0, 0x6a630242d00c1018),
    starter(0x20100804020000, 0, 0xae5c4828a5041c02),
    starter(0x40201008040000, 0, 0x39048c7000f718ed),
    starter(0x4020100a0000, 0, 0xfb306122278a0104),
    starter(0x4022140000, 0, 0x310b2c3c0681888c),
    starter(0x244280000, 0, 0x38b8dc504097f1b1),
    starter(0x20408500000, 0, 0xa24f60324e0a356),
    starter(0x2040810200000, 0, 0xbc40738c102e1088),
    starter(0x4081020400000, 0, 0x8ba46e9333006e5),
    starter(0x10080402000200, 0, 0xe041421970ac075f),
    starter(0x20100804000400, 0, 0x3c9bc8cf0fb216a7),
    starter(0x4020100a000a00, 0, 0x4ab0034803c3c27a),
    starter(0x402214001400, 0, 0x8b340558094030e2),
    starter(0x24428002800, 0, 0x2dcc008a81e0037c),
    starter(0x2040850005000, 0, 0xa8460073008db401),
    starter(0x4081020002000, 0, 0x86841bc082845050),
    starter(0x8102040004000, 0, 0x86dc347b0308d064),
    starter(0x8040200020400, 0, 0xe1059d0c0774d8f),
    starter(0x10080400040800, 0, 0x2091ab6260080a2d),
    starter(0x20100a000a1000, 0, 0x8ae11000fd0c07a0),
    starter(0x40221400142200, 0, 0x92c00401c600a038),
    starter(0x2442800284400, 0, 0xe806840152812005),
    starter(0x4085000500800, 0, 0x325982009362101b),
    starter(0x8102000201000, 0, 0x487e0bdbc6014d95),
    starter(0x10204000402000, 0, 0x9bd9060989068e91),
    starter(0x4020002040800, 0, 0xae3e1a409fe1783c),
    starter(0x8040004081000, 0, 0x60380334c1f03c1c),
    starter(0x100a000a102000, 0, 0x323abb300b680052),
    starter(0x22140014224000, 0, 0xc0a6a0180169010c),
    starter(0x44280028440200, 0, 0xbf321206001c00c8),
    starter(0x8500050080400, 0, 0x3a6b01c900e60120),
    starter(0x10200020100800, 0, 0xe1e70806837d1c11),
    starter(0x20400040201000, 0, 0x58081a0b89d742c4),
    starter(0x2000204081000, 0, 0xcc78180c35a050ea),
    starter(0x4000408102000, 0, 0x8cda51381f79c89d),
    starter(0xa000a10204000, 0, 0x4e520a1f00ca800),
    starter(0x14001422400000, 0, 0xbeee6ca01105080e),
    starter(0x28002844020000, 0, 0xd35b220e0a008401),
    starter(0x50005008040200, 0, 0xe0204e0f4ac16601),
    starter(0x20002010080400, 0, 0xfff00e0e185c440a),
    starter(0x40004020100800, 0, 0x13d00b571b00a657),
    starter(0x20408102000, 0, 0xd656c23030385fa4),
    starter(0x40810204000, 0, 0x53f194075a506220),
    starter(0xa1020400000, 0, 0x629edd444c500371),
    starter(0x142240000000, 0, 0x8b83298fc20218c7),
    starter(0x284402000000, 0, 0xfb26a5191024191f),
    starter(0x500804020000, 0, 0xe273e0ae222a01f8),
    starter(0x201008040200, 0, 0x320ac2f760c03ee),
    starter(0x402010080400, 0, 0x8c08b87803a2d30f),
    starter(0x2040810204000, 0, 0x6a630242d00c1018),
    starter(0x4081020400000, 0, 0x8ba46e9333006e5),
    starter(0xa102040000000, 0, 0xaed520c3424c3001),
    starter(0x14224000000000, 0, 0x53212edf7442120a),
    starter(0x28440200000000, 0, 0xc46c403b675cf402),
    starter(0x50080402000000, 0, 0xb59d3ca9b06d3603),
    starter(0x20100804020000, 0, 0xae5c4828a5041c02),
    starter(0x40201008040200, 0, 0x9fe0a01d1d0700b2),
};

constexpr MagicInfoStarter rook_starter[NUMSQ] = {
    starter(0x101010101017e, 0, 0xe880006cc0017918),
    starter(0x202020202027c, 1, 0xac0b0b02e681bc7c),
    starter(0x404040404047a, 1, 0xecf914a910699037),
    starter(0x8080808080876, 1, 0x3b2e5b2e44a608d2),
    starter(0x1010101010106e, 0, 0x960010788c202a00),
    starter(0x2020202020205e, 1, 0x9c1273ed8f9544ab),
    starter(0x4040404040403e, 0, 0xa400102836890204),
    starter(0x8080808080807e, 0, 0x150002039450d100),
    starter(0x1010101017e00, 0, 0x2f84800240009364),
    starter(0x2020202027c00, 0, 0x15e8802005c0018b),
    starter(0x4040404047a00, 0, 0x5f8200220151c280),
    starter(0x8080808087600, 0, 0xc1e1002010011902),
    starter(0x10101010106e00, 0, 0x688d003068003500),
    starter(0x20202020205e00, 0, 0xe3720010a85c6a00),
    starter(0x40404040403e00, 0, 0x20fc0004102a383d),
    starter(0x80808080807e00, 0, 0xf143003b00034082),
    starter(0x10101017e0100, 0, 0x359fe98007400086),
    starter(0x20202027c0200, 0, 0x3ee5020046028123),
    starter(0x40404047a0400, 0, 0x39e8b60022c20080),
    starter(0x8080808760800, 0, 0x12c7e1001901f002),
    starter(0x101010106e1000, 0, 0x3d160a00106a0022),
    starter(0x202020205e2000, 1, 0x7f5d2494ed4c950b),
    starter(0x404040403e4000, 0, 0xa5be9400501a2805),
    starter(0x808080807e8000, 0, 0xa24cee000d490984),
    starter(0x101017e010100, 0, 0x1e97c00180058471),
    starter(0x202027c020200, 0, 0xa64103850021c008),
    starter(0x404047a040400, 0, 0x2bc28202002340f2),
    starter(0x8080876080800, 0, 0xaf35b86100100101),
    starter(0x1010106e101000, 0, 0x167a001e00104b16),
    starter(0x2020205e202000, 0, 0x7286006200087084),
    starter(0x4040403e404000, 0, 0x73ac700400680146),
    starter(0x8080807e808000, 0, 0x322bf5020000c88c),
    starter(0x1017e01010100, 0, 0xd12c8a4003800461),
    starter(0x2027c02020200, 0, 0xd97b034a02002187),
    starter(0x4047a04040400, 0, 0xa6d7764082002201),
    starter(0x8087608080800, 0, 0xaf54ea4022001200),
    starter(0x10106e10101000, 0, 0xe52a032b6000600),
    starter(0x20205e20202000, 0, 0xdf6e0048c2001084),
    starter(0x40403e40404000, 0, 0x8ba4ad6e4c001810),
    starter(0x80807e80808000, 0, 0xd4252ae6660006ac),
    starter(0x17e0101010100, 0, 0x3b938128c002800a),
    starter(0x27c0202020200, 0, 0x5830022003d0c004),
    starter(0x47a0404040400, 0, 0x9e8de04600820010),
    starter(0x8760808080800, 0, 0xfa2b006170010019),
    starter(0x106e1010101000, 0, 0x3d160a00106a0022),
    starter(0x205e2020202000, 0, 0xe88a005048820024),
    starter(0x403e4040404000, 0, 0x4dbf90283d34002a),
    starter(0x807e8080808000, 0, 0x57033f8f034e0014),
    starter(0x7e010101010100, 0, 0xa09b02c28e5e0e00),
    starter(0x7c020202020200, 0, 0xa09b02c28e5e0e00),
    starter(0x7a040404040400, 0, 0x3c414420b0820600),
    starter(0x76080808080800, 0, 0xc3c9a0fa0012c200),
    starter(0x6e101010101000, 0, 0x31ddd60047603200),
    starter(0x5e202020202000, 0, 0x2c692040103c5801),
    starter(0x3e404040404000, 0, 0x461db03302282c00),
    starter(0x7e808080808000, 0, 0xff704507b3b40e00),
    starter(0x7e01010101010100, 0, 0xb67262c3d0820102),
    starter(0x7c02020202020200, 0, 0xd102f301834003a1),
    starter(0x7a04040404040400, 0, 0x1ea5e04472820036),
    starter(0x7608080808080800, 0, 0xd67700885d201001),
    starter(0x6e10101010101000, 0, 0xc4ce00d8204c5036),
    starter(0x5e20202020202000, 0, 0x6632003043341822),
    starter(0x3e40404040404000, 0, 0x5376df0824b00a14),
    starter(0x7e80808080808000, 0, 0xca450f7231015402),
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

constexpr uint64_t get_rook_moves(int32_t sq, uint64_t occupied)
{
    int32_t file = sq >> WIDTH_SHIFT;
    int32_t rank = sq & (WIDTH - 1);
    int32_t nsq = 0;
    uint64_t moves = 0;

    for (int32_t f = file + 1; f < HEIGHT; f++) {
        nsq = (f << WIDTH_SHIFT) + rank;
        moves |= (1ull << nsq);

        if ((1ull << nsq) & occupied) {
            break;
        }
    }

    for (int32_t f = file - 1; f >= 0; f--) {
        nsq = (f << WIDTH_SHIFT) + rank;
        moves |= (1ull << nsq);

        if ((1ull << nsq) & occupied) {
            break;
        }
    }

    for (int32_t r = rank + 1; r < WIDTH; r++) {
        nsq = (file << WIDTH_SHIFT) + r;
        moves |= (1ull << nsq);

        if ((1ull << nsq) & occupied) {
            break;
        }
    }

    for (int32_t r = rank - 1; r >= 0; r--) {
        nsq = (file << WIDTH_SHIFT) + r;
        moves |= (1ull << nsq);

        if ((1ull << nsq) & occupied) {
            break;
        }
    }

    return moves;
}

constexpr uint64_t get_bishop_moves(int32_t sq, uint64_t occupied)
{
    int32_t file = sq >> WIDTH_SHIFT;
    int32_t rank = sq & (WIDTH - 1);
    int32_t nsq = 0;
    uint64_t moves = 0;

    for (int32_t dir = 0; dir < 4; dir++) {
        for (int32_t off = 1; ; off++) {
            int32_t r = (dir & 1) ? rank + off : rank - off;
            if (r < 0 || r >= WIDTH) {
                break;
            }

            int32_t f = (dir & 2) ? file + off : file - off;
            if (f < 0 || f >= HEIGHT) {
                break;
            }

            nsq = r + (f << WIDTH_SHIFT);

            moves |= (1ull << nsq);

            if ((1ull << nsq) * occupied) {
                break;
            }
        }
    }


    return moves;
}

struct MoveArray {
    uint64_t arr[NUMSQ];

    constexpr MoveArray() : arr{} {};
};

constexpr auto make_knight_moves() {
    MoveArray moves = {};
    uint64_t sq_moves = 0;

    for (int32_t f = 0; f < HEIGHT; f++) {
        for (int32_t r = 0; r < WIDTH; r++) {
            sq_moves = 0;

            if ((r - 2) >= 0) {
                if ((f - 1) >= 0) {
                    sq_moves |= (1ull << ((r - 2) + ((f - 1) * WIDTH)));
                }
                if ((f + 1) < HEIGHT) {
                    sq_moves |= (1ull << ((r - 2) + ((f + 1) * WIDTH)));
                }
            }
            if ((r - 1) >= 0) {
                if ((f - 2) >= 0) {
                    sq_moves |= (1ull << ((r - 1) + ((f - 2) * WIDTH)));
                }
                if ((f + 2) < HEIGHT) {
                    sq_moves |= (1ull << ((r - 1) + ((f + 2) * WIDTH)));
                }
            }
            if ((r + 2) < WIDTH) {
                if ((f - 1) >= 0) {
                    sq_moves |= (1ull << ((r + 2) + ((f - 1) * WIDTH)));
                }
                if ((f + 1) < HEIGHT) {
                    sq_moves |= (1ull << ((r + 2) + ((f + 1) * WIDTH)));
                }
            }
            if ((r + 1) < WIDTH) {
                if ((f - 2) >= 0) {
                    sq_moves |= (1ull << ((r + 1) + ((f - 2) * WIDTH)));
                }
                if ((f + 2) < HEIGHT) {
                    sq_moves |= (1ull << ((r + 1) + ((f + 2) * WIDTH)));
                }
            }

            moves.arr[r + (f * WIDTH)] = sq_moves;
        }
    }

    return moves;
}

constexpr auto make_king_moves() {
    MoveArray moves = {};

    uint64_t sq_moves = 0;

    for (int32_t f = 0; f < HEIGHT; f++) {
        for (int32_t r = 0; r < WIDTH; r++) {
            sq_moves = 0;

            if ((r - 1) >= 0) {
                sq_moves |= (1ull << ((r - 1) + (f * WIDTH)));

                if ((f - 1) >= 0) {
                    sq_moves |= (1ull << ((r - 1) + ((f - 1) * WIDTH)));
                }
                if ((f + 1) < HEIGHT) {
                    sq_moves |= (1ull << ((r - 1) + ((f + 1) * WIDTH)));
                }
            }
            if ((f - 1) >= 0) {
                sq_moves |= (1ull << (r + ((f - 1) * WIDTH)));
            }
            if ((f + 1) < HEIGHT) {
                sq_moves |= (1ull << (r + ((f + 1) * WIDTH)));
            }
            if ((r + 1) < WIDTH) {
                sq_moves |= (1ull << ((r + 1) + (f * WIDTH)));

                if ((f - 1) >= 0) {
                    sq_moves |= (1ull << ((r + 1) + ((f - 1) * WIDTH)));
                }
                if ((f + 1) < HEIGHT) {
                    sq_moves |= (1ull << ((r + 1) + ((f + 1) * WIDTH)));
                }
            }

            moves.arr[r + (f * WIDTH)] = sq_moves;
        }
    }

    return moves;
}

constexpr size_t rook_mem_needed = max_magic_lut_mem(rook_starter);
constexpr size_t bishop_mem_needed = max_magic_lut_mem(bishop_starter);

struct LUTs {
    MagicInfoArray<bishop_mem_needed> lut_magic_bishop;
    MagicInfoArray<rook_mem_needed> lut_magic_rook;
    MoveArray lut_nk_moves[2];

    constexpr LUTs() :
        lut_nk_moves{ make_knight_moves(), make_king_moves() }
    {
        lut_magic_bishop = MagicInfoArray<bishop_mem_needed>::make_magic_bishop(bishop_starter);
        lut_magic_rook = MagicInfoArray<rook_mem_needed>::make_magic_rook(rook_starter);
    };
};

constexpr LUTs g_LUTs = {};

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

