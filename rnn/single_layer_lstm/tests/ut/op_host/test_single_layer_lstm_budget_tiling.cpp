/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_single_layer_lstm_budget_tiling.cpp
 * \brief Host-side on-chip budget tests for rnn/single_layer_lstm.
 *
 * Runs the SAME functions op_host calls -- SingleLayerLstmBudget::PickProjTiling / RecurrenceFits --
 * and checks two things the compiler cannot:
 *   1. every chunk pair the pickers return actually fits the dav_3510 capacities, and
 *   2. the shapes that MUST be refused are refused, rather than clamped to something that overflows.
 *
 * The budget headers are plain C++ by construction (no AscendC), which is what lets them be exercised
 * here at all. The file name ends in `_tiling` because that is the pattern the op_host UT CMake globs
 * for; it is a tiling test. An on-chip overflow has no error message at all, so "it looked right" is not an
 * acceptable level of assurance for this arithmetic.
 */
#include <cstdio>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "rnn/single_layer_lstm/op_host/arch35/single_layer_lstm_budget.h"

namespace {

void Expect(bool cond, const char* what) { EXPECT_TRUE(cond) << what; }

struct Shape {
    uint32_t t, b, i, h;
};

/* The shapes the reference implementation was measured on, plus the two edge cases that exposed real
 * bugs there: B=1 (the Mmad m==1 floor and the CeilDiv row split) and T=1. */
const Shape SHAPES[] = {
    {8, 16, 32, 64},
    {32, 64, 128, 128},
    {100, 32, 256, 128},
    {4, 8, 16, 32},
    {5, 1, 16, 32},
    {1, 8, 16, 64},
    {6, 12, 24, 64},
    {8, 8, 64, 128},
    /* Past each residency in turn, so the sweep exercises the streaming schedules and not only the
     * ones that fit whole: 512 streams the weights, 2048 at 64 rows per cluster streams h_{t-1} as
     * well, and both need phase A's N axis tiled to be accepted at all. */
    {8, 8, 32, 512},
    {4, 64, 32, 1024},
    {2, 1024, 32, 2048},
};

/* Element widths the operator is built for. Phase A carries the caller's width; phase B and the
 * epilogue are fp32 at all of them, which is why `inBytes` changes what PickProjTiling accepts and
 * leaves every phase B extent alone. fp16 and bf16 are the same width, so two values cover three
 * dtypes -- nothing below can tell them apart, and neither can the budget. */
const uint32_t IN_BYTES[] = {4U, 2U};

void CheckForward(uint32_t inBytes)
{
    /* ONE FRACTAL FOR BOTH EXTENTS AND ALL THREE DTYPES: the cube reads fp32 in both phases. */
    const uint32_t c0In = SingleLayerLstmFwd::C0F;
    for (const Shape& s : SHAPES) {
        char label[160];
        const uint32_t mBlk = SingleLayerLstmFwd::PickRowsPerBlock(s.b, SingleLayerLstmBudget::MAX_CLUSTERS);
        std::snprintf(label, sizeof(label), "fwd T=%u B=%u I=%u H=%u mBlk=%u inBytes=%u", s.t, s.b, s.i, s.h, mBlk,
                      inBytes);

        Expect(mBlk >= 1 && s.b % mBlk == 0, label);
        Expect(s.b / mBlk <= SingleLayerLstmBudget::MAX_CLUSTERS, label);
        Expect(SingleLayerLstmBudget::RecurrenceFits(s.b, s.h, s.t, mBlk, inBytes), label);

        uint32_t tChunk = 0;
        uint32_t kChunk = 0;
        uint32_t nc = 0; // columns of one gate per cube pass -- phase A's N chunk
        const bool ok = SingleLayerLstmBudget::PickProjTiling(mBlk, s.i, s.h, s.t, inBytes, &tChunk, &kChunk, &nc);
        /* The sweep still runs at both widths, but input_size's rule no longer separates them --
         * what `inBytes` still moves is phase B's UB staging and the workspace images. */
        if (s.i % c0In != 0) {
            Expect(!ok, label);
            continue;
        }
        Expect(ok, label);
        if (!ok) {
            continue;
        }
        /* tChunk must still divide T: the A tile lays row r's timesteps at NZ rows
         * [r*tChunk, r*tChunk + tc), so a short last chunk would leave GAPS between the batch rows.
         * kChunk carries no such rule -- its last block is simply narrower. */
        Expect(s.t % tChunk == 0, label);
        /* kChunk no longer has to divide input_size -- the last block is short. What it must be is
         * positive, no wider than input_size, and a whole number of fractals. */
        Expect(kChunk >= 1 && kChunk <= SingleLayerLstmCube::CeilAlign(s.i, c0In) && kChunk % c0In == 0, label);
        /* nChunk has no divisibility rule -- the last chunk is simply shorter, the way phase B's is
         * -- but it must be positive and must not exceed the gate it chunks. */
        Expect(nc >= 1 && nc <= s.h, label);

        /* Re-derive phase A's footprints independently of the picker and check them against the
         * capacities, so a picker that returns a too-large chunk is caught here rather than on chip. */
        /* Everything below is fp32: phase A's operands as well as its L0C tile and drained plane. */
        const uint32_t c0 = c0In;
        const uint32_t m = tChunk * mBlk;
        const uint32_t rows = SingleLayerLstmCube::CeilDiv(mBlk, 2);
        const uint32_t a1 = SingleLayerLstmCube::CeilDiv(s.i, c0) *
                            SingleLayerLstmCube::CeilAlign(m, SingleLayerLstmCube::CUBE_BLOCK) * c0;
        const uint32_t a2 = SingleLayerLstmCube::CeilDiv(kChunk, c0) *
                            SingleLayerLstmCube::CeilAlign(m, SingleLayerLstmCube::CUBE_BLOCK) * c0;
        /* nc, NOT s.h. PHASE A'S N AXIS IS TILED, so the B tile, the L0C accumulator, the bias tile
         * and the drained plane are all nChunk wide. Re-deriving them from hidden_size instead
         * would make this check pass a picker that returned a chunk too large to fit -- it would
         * simply be measuring a different kernel. */
        const uint32_t ncAl = SingleLayerLstmCube::CeilAlign(nc, SingleLayerLstmCube::CUBE_BLOCK);
        const uint32_t bt = SingleLayerLstmCube::CeilDiv(ncAl, c0) *
                            SingleLayerLstmCube::CeilAlign(kChunk, SingleLayerLstmCube::CUBE_BLOCK) * c0;
        const uint32_t cc = SingleLayerLstmCube::CeilAlign(m, SingleLayerLstmCube::CUBE_BLOCK) * ncAl;
        /* The bias tile is fp32 and sits in L1 beside the A and B tiles. It is in this sum because
         * SingleLayerLstmProj::Layout takes it; leaving it out was an under-count of BtElems(H)*4
         * bytes that only ever bites at the capacity boundary. */
        const uint32_t l1 = SingleLayerLstmCube::CeilAlign(a1 * 4U, SingleLayerLstmCube::C0_BYTES) +
                            SingleLayerLstmCube::CeilAlign(bt * 4U, SingleLayerLstmCube::C0_BYTES) +
                            SingleLayerLstmCube::CeilAlign(SingleLayerLstmCube::BtElems(nc) * 4U,
                                                           SingleLayerLstmCube::C0_BYTES);
        const uint32_t ub = SingleLayerLstmCube::CeilAlign(tChunk * rows * nc * 4U, SingleLayerLstmCube::C0_BYTES) +
                            SingleLayerLstmCube::CeilAlign(SingleLayerLstmBudget::GATES * nc * 4U,
                                                           SingleLayerLstmCube::C0_BYTES);
        Expect(a2 * 4U <= SingleLayerLstmCube::CAP_L0A, label);
        Expect(bt * 4U <= SingleLayerLstmCube::CAP_L0B, label);
        /* THE WHOLE L0C CAP AT EVERY DTYPE. It was halved at 2-byte operands against a measured,
         * unexplained fault; the operands are fp32 in this phase now, so the condition cannot be
         * reached and the halving is gone. If a fault ever appears with fp32 operands, this is the
         * line to tighten -- and the shape that showed the old one (T=448 B=8 I=16 H=64 fp16) is in
         * the device sweep. */
        Expect(cc * 4U <= SingleLayerLstmCube::CAP_L0C, label);
        Expect(l1 <= SingleLayerLstmCube::CAP_L1, label);
        Expect(ub <= SingleLayerLstmCube::CAP_UB, label);

        /* Phase B, read off the layout object the kernel itself builds. Fp32 at every dtype.
         * THE L0 EXTENTS ARE THE CHUNKED ONES: aTileElems is the whole A tile only while it is
         * resident, bChunkElems is one [kChunk, nChunk] tile rather than the L1 region bElems, and the
         * four L0C accumulators come out of one bump. Checking bElems here is what let a
         * hidden_size ceiling hide in a passing test. */
        const SingleLayerLstmFwd::Layout L(s.b, s.h, s.t, mBlk, inBytes);
        Expect(L.aTileElems * 4U <= SingleLayerLstmCube::CAP_L0A, label);
        /* L1 and L0A take the same A extent -- the split modes are unrepresentable. */
        Expect(L.aTileElems == (L.aResident ? L.aElems : L.aChunkElems), label);
        Expect(L.bChunkElems * 4U <= SingleLayerLstmCube::CAP_L0B, label);
        Expect(SingleLayerLstmBudget::GATES * L.cElems * 4U <= SingleLayerLstmCube::CAP_L0C, label);
        Expect(L.l1Bytes <= SingleLayerLstmCube::CAP_L1, label);
        Expect(L.ubBytes <= SingleLayerLstmCube::CAP_UB, label);
    }
}

/* The refusals. Each of these must come back false, because clamping instead would either overflow a
 * buffer on chip (no diagnostic at all) or make the kernel's addressing wrong. */
void CheckRefusals()
{
    uint32_t a = 0;
    uint32_t b = 0;
    uint32_t nc = 0;

    // I not a multiple of the fractal width: every strided UB copy's row pitch becomes
    // inexpressible in 32-byte blocks.
    Expect(!SingleLayerLstmBudget::PickProjTiling(8, 12, 64, 8, 4, &a, &b, &nc), "refuse I=12 at fp32");
    /* THE I RULE MOVES WITH THE DTYPE AND THE H RULE DOES NOT. I=24 is 3 fp32 fractals and one and a
     * half narrow ones, so it is legal at 4 bytes and illegal at 2. H=24 is legal at both, because
     * hidden_size is bounded by phase B and the epilogue, which are fp32 whatever the caller sent.
     * Checking only one width would leave either half of that free to be wrong. */
    Expect(SingleLayerLstmBudget::PickProjTiling(8, 24, 64, 8, 4, &a, &b, &nc), "accept I=24 at fp32");
    /* AND AT THE NARROW DTYPES TOO. This line read `!...` while phase A took the caller's width;
     * the cube reads fp32 in both phases now, so input_size follows the fp32 fractal at all three. */
    Expect(SingleLayerLstmBudget::PickProjTiling(8, 24, 64, 8, 2, &a, &b, &nc), "accept I=24 at fp16/bf16");
    Expect(SingleLayerLstmBudget::PickProjTiling(8, 32, 24, 8, 2, &a, &b, &nc), "accept H=24 at fp16/bf16");
    // H below the fp32 fractal width, at every dtype.
    Expect(!SingleLayerLstmBudget::PickProjTiling(8, 32, 12, 8, 4, &a, &b, &nc), "refuse H=12 at fp32");
    Expect(!SingleLayerLstmBudget::PickProjTiling(8, 32, 12, 8, 2, &a, &b, &nc), "refuse H=12 at fp16/bf16");
    Expect(!SingleLayerLstmBudget::RecurrenceFits(64, 12, 8, 8, 4), "refuse H=12 in the recurrence");
    Expect(!SingleLayerLstmBudget::RecurrenceFits(64, 12, 8, 8, 2), "refuse H=12 in the recurrence at fp16/bf16");
    /* mBlk NEED NOT DIVIDE THE BATCH. This line read `!...` while the split had to be exact; each
     * cluster now builds its Layout from its own row count, so the short last block is
     * self-consistent -- and requiring a divisor put the whole batch on one cluster whenever B had
     * no small factor. What is still refused is an mBlk larger than the batch. */
    Expect(SingleLayerLstmBudget::RecurrenceFits(65, 64, 8, 8, 4), "accept mBlk that does not divide B");
    Expect(!SingleLayerLstmBudget::RecurrenceFits(65, 64, 8, 66, 4), "refuse mBlk larger than B");
    /* THE BATCHES THAT USED TO HAVE NO USABLE SPLIT. A prime batch has no divisor at or below the
     * cluster cap other than 1, so the old rule handed every row to one cluster: B=4097 gave 4097
     * rows and B=2049 gave 683, and the per-AIV UB planes then had no chunk narrow enough. Ceiling
     * division caps them at ceil(B/16). */
    for (uint32_t b : {257U, 513U, 1025U, 2049U, 4097U}) {
        const uint32_t m = SingleLayerLstmFwd::PickRowsPerBlock(b, SingleLayerLstmBudget::MAX_CLUSTERS);
        Expect(m == SingleLayerLstmCube::CeilDiv(b, SingleLayerLstmBudget::MAX_CLUSTERS),
               "a batch with no small divisor still spreads over every cluster");
        Expect(SingleLayerLstmBudget::RecurrenceFits(b, 512, 2, m, 4), "... and the recurrence fits");
    }
    /* HIDDEN SIZE ON ITS OWN NO LONGER REFUSES ANYTHING, AND THAT IS THE PROPERTY UNDER TEST.
     * Both axes of the recurrence GEMM are tiled and each of the three residencies degrades into a
     * stream rather than into a refusal, so what H changes is the SCHEDULE. These cases pin the
     * transitions: past ~176 the four gates stop fitting L1 and stream from GM, and past the point
     * where [mBlk, H] fills L1 h_{t-1} streams out of hAll as well. Both widths, because every
     * buffer phase B counts is fp32 at every dtype.
     *
     * The old form of this block asserted the opposite -- "refuse H=1024" -- which was the correct
     * statement about the kernel that sent one whole gate to L0B. */
    Expect(SingleLayerLstmBudget::RecurrenceFits(16, 1024, 8, 16, 4), "accept H=1024 at fp32");
    Expect(SingleLayerLstmBudget::RecurrenceFits(16, 1024, 8, 16, 2), "accept H=1024 at fp16/bf16");
    Expect(SingleLayerLstmBudget::RecurrenceFits(16, 176, 8, 16, 4), "accept H=176 at fp32");
    Expect(SingleLayerLstmBudget::RecurrenceFits(16, 176, 8, 16, 2), "accept H=176 at fp16/bf16");
    Expect(SingleLayerLstmBudget::RecurrenceFits(16, 184, 8, 16, 4), "accept H=184: the weights stream");
    {
        const SingleLayerLstmFwd::Layout resident(16, 176, 8, 16, 4);
        const SingleLayerLstmFwd::Layout streamed(16, 184, 8, 16, 4);
        Expect(resident.wResident, "H=176: four gates still resident in L1");
        Expect(!streamed.wResident, "H=184: four gates no longer fit, so they stream");
        Expect(resident.aResident && streamed.aResident, "h_{t-1} still lives in L1 at both");
        /* The A tile is CeilAlign(mBlk, 16)*H*4 bytes, so at 16 rows per cluster it fills L1 just
         * past H=8128. Named as a formula rather than as the number, because the number moves with
         * mBlk and the formula does not. */
        const SingleLayerLstmFwd::Layout aStreamed(16, 8192, 8, 16, 4);
        Expect(!aStreamed.aResident, "H=8192 at 16 rows: h_{t-1} streams out of hAll instead");
        /* THE CASE THE SPLIT MODES GOT WRONG. At 64 rows per cluster and H=512 the whole A tile is
         * 128 KB: it fits L1 and does not fit L0A. While those were two flags it stayed in L1 and
         * SplitA re-read it once per column chunk, by which time the AIVs had overwritten the
         * leading columns with h_t -- device-measured as columns 96..511 wrong at B=1024 H=512
         * fp32. It must now come out STREAMED. */
        const SingleLayerLstmFwd::Layout aSplit(1024, 512, 8, 64, 4);
        Expect(aSplit.aElems * 4U > SingleLayerLstmCube::CAP_L0A, "B=1024 H=512: A does not fit L0A");
        Expect(!aSplit.aResident, "... so it does not stay in L1 either");
        Expect(aSplit.aTileElems == aSplit.aChunkElems, "... and both buffers hold one k-chunk");
        Expect(SingleLayerLstmBudget::RecurrenceFits(16, 8192, 8, 16, 4), "... and the shape is still accepted");
    }
    /* What the narrow dtypes add: fp32 images of BOTH slabs the cube reads, unconditionally -- the
     * cube never takes the caller's width, so residency no longer decides whether an image exists.
     * One copy for the whole grid, so neither scales with blockDim. */
    Expect(SingleLayerLstmBudget::ProjInputImageFloats(8, 16, 32, 4) == 0, "no x image at fp32");
    Expect(SingleLayerLstmBudget::WeightImageFloats(32, 176, 4) == 0, "no weight image at fp32");
    Expect(SingleLayerLstmBudget::ProjInputImageFloats(8, 16, 32, 2) == 8ULL * 16 * 32,
           "x image is [T, B, I] at fp16/bf16");
    Expect(SingleLayerLstmBudget::WeightImageFloats(32, 176, 2) == (32ULL + 176) * 4 * 176,
           "weight image is [I+H, 4H] at fp16/bf16, both phases' rows in one slab");
    Expect(SingleLayerLstmBudget::WeightImageFloats(32, 184, 2) == (32ULL + 184) * 4 * 184,
           "... and residency does not change it");
    // Zero extents.
    Expect(!SingleLayerLstmBudget::PickProjTiling(0, 32, 64, 8, 4, &a, &b, &nc), "refuse mBlk=0");
    Expect(!SingleLayerLstmBudget::PickProjTiling(8, 32, 64, 0, 4, &a, &b, &nc), "refuse T=0");
    Expect(!SingleLayerLstmBudget::PickProjTiling(8, 32, 64, 8, 0, &a, &b, &nc), "refuse inBytes=0");

    /* THE SHAPE THAT FIRST EXPOSED THE NARROW L0C BOUND, kept as a regression anchor. At
     * T=448 B=8 I=16 H=64 both widths used to pick the same chunk from the L0A rule -- tChunk 448,
     * m 896, a 224 KB L0C tile -- and on device only the narrow one raised AICORE error 171. The
     * answer then was to halve the cap for 2-byte operands. Phase A feeds the cube fp32 at every
     * dtype now, so the two widths pick the SAME tiling here and both are checked against the whole
     * cap. The device sweep still runs this shape at all three dtypes. */
    uint32_t tc32 = 0, kc32 = 0, nc32 = 0, tc16 = 0, kc16 = 0, nc16 = 0;
    Expect(SingleLayerLstmBudget::PickProjTiling(2, 16, 64, 448, 4, &tc32, &kc32, &nc32), "T=448 I=16 fp32 tiles");
    Expect(SingleLayerLstmBudget::PickProjTiling(2, 16, 64, 448, 2, &tc16, &kc16, &nc16), "T=448 I=16 fp16 tiles");
    Expect(tc32 == tc16 && kc32 == kc16 && nc32 == nc16,
           "the two widths pick the same phase A tiling: the cube sees fp32 either way");
    Expect(SingleLayerLstmCube::CeilAlign(tc32 * 2U, SingleLayerLstmCube::CUBE_BLOCK) *
                   SingleLayerLstmCube::CeilAlign(nc32, SingleLayerLstmCube::CUBE_BLOCK) * 4U <=
               SingleLayerLstmCube::CAP_L0C,
           "and it stays inside the whole L0C cap");
}

/* The row split has to be total and it has to COVER: for every batch in range the picker must return
 * a positive mBlk, the resulting cluster count must stay within bounds, and those clusters together
 * must account for every row with none left over and none of them empty. Exact division is no
 * longer required -- the last block may be short -- but a gap or an empty cluster would silently
 * drop or double-count rows, so both are checked here rather than left to the device. */
void CheckRowSplit()
{
    for (uint32_t batch = 1; batch <= 4100; ++batch) {
        const uint32_t mBlk = SingleLayerLstmFwd::PickRowsPerBlock(batch, SingleLayerLstmBudget::MAX_CLUSTERS);
        char label[80];
        std::snprintf(label, sizeof(label), "row split at B=%u", batch);
        Expect(mBlk >= 1, label);
        if (mBlk == 0) {
            continue;
        }
        const uint32_t blocks = SingleLayerLstmCube::CeilDiv(batch, mBlk);
        Expect(blocks <= SingleLayerLstmBudget::MAX_CLUSTERS, label);
        Expect((blocks - 1) * mBlk < batch, label); // no empty last cluster
        uint32_t covered = 0;
        for (uint32_t blk = 0; blk < blocks; ++blk) {
            const uint32_t rows = SingleLayerLstmFwd::RowsOfBlock(batch, mBlk, blk * mBlk);
            Expect(rows >= 1 && rows <= mBlk, label);
            covered += rows;
        }
        Expect(covered == batch, label);
    }
}

/* THE BLOCK WALK, WHICH IS WHAT MAKES A LARGE BATCH RUNNABLE AT ALL. `PickRowChunk` returns the
 * rows one cluster holds ON CHIP; the kernel then strides over the batch by blockDim * rowChunk.
 * Three things have to hold together or rows are silently dropped or computed twice:
 *   - both feasibility tests pass AT the returned chunk (not at some larger one),
 *   - blockDim stays within the cluster bound,
 *   - the strided walk covers every row of the batch EXACTLY once.
 * The last one is checked by replaying the kernel's own loop -- `for (base = cluster * mBlk;
 * base < B; base += blockDim * mBlk)` -- and counting how many times each row is visited.
 *
 * The shapes are the ones the forward case set refused before this existed: B=16384 at H=4096 put
 * 1024 rows on one cluster and a 2 MB h_{t-1} tile against L1's 512 KB. */
void CheckBlockWalk()
{
    struct Shape {
        uint32_t t, b, i, h;
    };
    static const Shape SHAPES[] = {
        {1, 16384, 32, 4096}, {1, 16384, 40, 4096}, {1, 16384, 4104, 512}, {1, 8192, 40, 8192},
        {1, 8192, 32, 512},   {3, 4097, 32, 128},   {8, 1024, 64, 1024},   {1, 2, 8192, 8192},
        {1, 2, 16384, 4096},  {2, 2, 16, 32},       {1, 8, 4096, 16384},   {32, 64, 128, 128},
    };
    for (uint32_t inBytes : IN_BYTES) {
        for (const Shape& s : SHAPES) {
            char label[128];
            std::snprintf(label, sizeof(label), "block walk T=%u B=%u I=%u H=%u %uB", s.t, s.b, s.i, s.h, inBytes);
            const uint32_t rows = SingleLayerLstmBudget::PickRowChunk(s.b, s.h, s.t, s.i, inBytes);
            Expect(rows >= 1 && rows <= s.b, label);
            if (rows == 0) {
                continue;
            }
            uint32_t tChunk = 0;
            uint32_t kChunk = 0;
            uint32_t nChunk = 0;
            Expect(SingleLayerLstmBudget::RecurrenceFits(s.b, s.h, s.t, rows, inBytes), label);
            Expect(SingleLayerLstmBudget::PickProjTiling(rows, s.i, s.h, s.t, inBytes, &tChunk, &kChunk, &nChunk),
                   label);

            uint32_t blockDim = SingleLayerLstmCube::CeilDiv(s.b, rows);
            if (blockDim > SingleLayerLstmBudget::MAX_CLUSTERS) {
                blockDim = SingleLayerLstmBudget::MAX_CLUSTERS;
            }
            Expect(blockDim >= 1 && blockDim <= SingleLayerLstmBudget::MAX_CLUSTERS, label);

            /* Replay the kernel's stride loop. A row visited twice is two clusters writing the same
             * output; a row visited never is a caller's buffer left at whatever it held. */
            std::vector<uint32_t> visits(s.b, 0U);
            for (uint32_t cluster = 0; cluster < blockDim; ++cluster) {
                for (uint32_t base = cluster * rows; base < s.b; base += blockDim * rows) {
                    const uint32_t n = SingleLayerLstmFwd::RowsOfBlock(s.b, rows, base);
                    Expect(n >= 1 && n <= rows, label);
                    for (uint32_t r = 0; r < n; ++r) {
                        visits[base + r] += 1U;
                    }
                }
            }
            bool covered = true;
            for (uint32_t r = 0; r < s.b; ++r) {
                if (visits[r] != 1U) {
                    covered = false;
                    break;
                }
            }
            Expect(covered, label);
        }
    }
}

} // namespace

class SingleLayerLstmBudgetTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SingleLayerLstmBudgetTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SingleLayerLstmBudgetTest TearDown" << std::endl; }
};

TEST_F(SingleLayerLstmBudgetTest, forward_pickers_return_tilings_that_fit)
{
    for (uint32_t inBytes : IN_BYTES) {
        CheckForward(inBytes);
    }
}

TEST_F(SingleLayerLstmBudgetTest, unsupported_shapes_are_refused_not_clamped) { CheckRefusals(); }

TEST_F(SingleLayerLstmBudgetTest, every_batch_in_range_has_a_row_split) { CheckRowSplit(); }

TEST_F(SingleLayerLstmBudgetTest, the_block_walk_covers_every_row_exactly_once) { CheckBlockWalk(); }

TEST_F(SingleLayerLstmBudgetTest, compensated_projection_budget_includes_all_scratch_planes)
{
    namespace C = SingleLayerLstmCube;
    namespace F = SingleLayerLstmFwd;
    namespace B = SingleLayerLstmBudget;
    EXPECT_FALSE(F::CompensateCubeK(F::CUBE_SUM_K_THRESHOLD));
    EXPECT_TRUE(F::CompensateCubeK(F::CUBE_SUM_K_THRESHOLD + F::C0F));
    EXPECT_EQ(F::PickKChunk(128U, 2U), 128U);
    EXPECT_EQ(F::PickKChunk(8192U, 2U), 32U);
    for (uint32_t input : {128U, 136U, 232U, 1024U, 2056U, 8192U}) {
        for (uint32_t hidden : {24U, 128U, 232U, 8192U}) {
            for (uint32_t batch : {1U, 2U, 15U, 257U}) {
                for (uint32_t width : IN_BYTES) {
                    const uint32_t steps = 5;
                    const uint32_t rows = B::PickRowChunk(batch, hidden, steps, input, width);
                    ASSERT_GT(rows, 0U);
                    uint32_t tc = 0, kc = 0, nc = 0;
                    ASSERT_TRUE(B::PickProjTiling(rows, input, hidden, steps, width, &tc, &kc, &nc));
                    const uint32_t work = C::CeilDiv(tc * rows, 2U) * nc;
                    C::Bump ub;
                    if (F::CompensateCubeK(input)) {
                        EXPECT_LE(kc, F::CUBE_SUM_K_CHUNK);
                        const uint32_t plane = C::CeilAlign(work, C::CMP_REPEAT_ELEMS);
                        // drain, sum, correction, y, tmp; then the compare mask.
                        for (uint32_t p = 0; p < 5; ++p) {
                            ub.TakeT<float>(plane);
                        }
                        ub.TakeT<uint8_t>(C::CeilAlign(plane, C::C0_BYTES));
                    } else {
                        ub.TakeT<float>(work);
                    }
                    EXPECT_LE(ub.cur, C::CAP_UB);
                    EXPECT_LE(C::BtElems(nc) * sizeof(float), C::CAP_BT);
                    const F::Layout recurrent(batch, hidden, steps, rows, width);
                    const uint32_t limit = F::CompensateCubeK(hidden) ? F::CUBE_SUM_K_CHUNK : F::CUBE_SUM_K_THRESHOLD;
                    EXPECT_LE(recurrent.kChunk, limit);
                    EXPECT_LE(recurrent.ubBytes, C::CAP_UB);
                    EXPECT_LE(recurrent.l1Bytes, C::CAP_L1);
                }
            }
        }
    }
}
