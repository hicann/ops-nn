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
 * \file single_layer_lstm_proj_impl.h
 * \brief Input projection into FP32 gates; tiling selects cube or compensated vector accumulation.
 */
#ifndef OPS_RNN_SINGLE_LAYER_LSTM_PROJ_IMPL_H
#define OPS_RNN_SINGLE_LAYER_LSTM_PROJ_IMPL_H

#include "kernel_operator.h"
#include "cube_helper.h"
#include "sync_helper.h"
#include "single_layer_lstm_layout.h"
#include "compensated_sum.h"

/* Operator-specific namespaces avoid collisions when headers are included together. */
namespace SingleLayerLstmProj {

/* Largest finite fp32. Spelled out rather than pulled from <cfloat>, which the kernel translation
 * unit does not include. */
constexpr float FLT_MAX_F = 3.40282347e+38F;

/* On-chip layout for one projection chunk. Both cores build it from the same scalars, which is how
 * they agree on every address without exchanging any. The host cannot instantiate it, so op_host
 * re-derives the same extents in SingleLayerLstmBudget::PickProjTiling from an element width -- the
 * two must be read together. Not templated on the caller's width: both operands reach the cube as
 * fp32 at every dtype, the narrow slabs having been widened into workspace once per launch, so the
 * fractal here is the fp32 one and `input_size` keeps a multiple-of-8 rule at all three dtypes. */
struct Layout {
    static constexpr uint32_t C0IN = SingleLayerLstmCube::C0<float>();

    uint32_t mBlk, tChunk, kChunk, nChunk, inSize, hid, batch, steps;
    uint32_t m;       // rows in one chunk = tChunk * mBlk
    uint32_t nChunks; // column chunks of one gate
    /* `a1Elems` is what the A tile costs IN L1. The whole [m, I] when it fits -- so one pass of x
     * serves every gate and every column chunk -- and one [m, kChunk] block when it does not, in
     * which case the load moves inside the k loop and x is re-read per (gate, column chunk). At
     * I=2056 with 80 aligned rows the whole tile is 658 KB against L1's 512, and that shape used to
     * be refused outright. */
    bool aResident;
    uint32_t a1Elems, bElems, cElems;
    uint32_t rowsMax, plane;
    uint32_t bOff, biasL1Off, btOff; // L1 / BT offsets
    uint32_t cubOff;                 // UB
    bool compensated;
    uint32_t sumOff, correctionOff, yOff, tmpOff, maskOff, maskElems;
    uint32_t l1Bytes, ubBytes;

    __aicore__ inline Layout(uint32_t mBlkIn, uint32_t tcIn, uint32_t kcIn, uint32_t ncIn, uint32_t iIn, uint32_t hIn,
                             uint32_t bIn, uint32_t tIn)
        : mBlk(mBlkIn),
          tChunk(tcIn),
          kChunk(kcIn),
          nChunk(ncIn),
          inSize(iIn),
          hid(hIn),
          batch(bIn),
          steps(tIn),
          m(tcIn * mBlkIn)
    {
        const uint32_t c0 = C0IN;
        nChunks = (nChunk == 0) ? 0 : SingleLayerLstmCube::CeilDiv(hid, nChunk);
        const uint32_t mAl = SingleLayerLstmCube::CeilAlign(m, SingleLayerLstmCube::CUBE_BLOCK);
        const uint32_t aWhole = SingleLayerLstmCube::CeilDiv(inSize, c0) * mAl * c0; // [m, I]
        const uint32_t aChunk = SingleLayerLstmCube::CeilDiv(kChunk, c0) * mAl * c0; // [m, kChunk]
        /* THE B TILE AND THE BIAS TABLE SHARE L1 WITH IT, so "the tile fits L1" is not the test --
         * SingleLayerLstmFwd::ProjAResident is, and op_host calls the same function while it is
         * choosing kChunk. A tile that fills L1 on its own leaves no column chunk runnable, which
         * is how I=8192 H=8192 B=2 came to be refused outright rather than run by streaming A. */
        aResident = SingleLayerLstmFwd::ProjAResident(m, inSize, kChunk, nChunk);
        a1Elems = aResident ? aWhole : aChunk;
        /* [kChunk, nChunk], not [kChunk, H]: the N axis is tiled here too. This tile, the L0C
         * accumulator and the drained UB plane were the three things in phase A that grew with
         * hidden_size and capped it at H=1024 in fp32 and, at batch 1024, H=512 in fp16.
         *
         * The column count is rounded up to a fractal before it is divided into C0 blocks. SplitB's
         * transpose granule is 16x16 elements, so at an N that is a multiple of 8 but not 16 it
         * reads one column block past the live data; allocating that block keeps the over-read
         * inside this tile. Mmad's n is likewise CeilAlign(n, 16) and the drain is given the real n,
         * so those columns are computed and never read. */
        const uint32_t nAl = SingleLayerLstmCube::CeilAlign(nChunk, SingleLayerLstmCube::CUBE_BLOCK);
        bElems = SingleLayerLstmCube::CeilDiv(nAl, c0) *
                 SingleLayerLstmCube::CeilAlign(kChunk, SingleLayerLstmCube::CUBE_BLOCK) * c0;
        /* L0C is fp32 whatever the operands are, so its extent is aligned to CUBE_BLOCK on both
         * axes and does NOT follow C0IN. */
        cElems = SingleLayerLstmCube::CeilAlign(m, SingleLayerLstmCube::CUBE_BLOCK) * nAl;
        rowsMax = SingleLayerLstmCube::DrainRowsMax(m, SingleLayerLstmFwd::SPLIT);
        compensated = SingleLayerLstmFwd::CompensateCubeK(inSize);
        plane = rowsMax * nChunk;
        if (compensated) {
            plane = SingleLayerLstmCube::CeilAlign(plane, SingleLayerLstmCube::CMP_REPEAT_ELEMS);
        }

        SingleLayerLstmCube::Bump l1;
        l1.TakeT<float>(a1Elems); // A tile at offset 0
        bOff = l1.TakeT<float>(bElems);
        /* The bias is fp32 in the operator's own signature, not the caller's width. The bias table
         * takes fp32 and nothing else (SingleLayerLstmCube::LoadBiasToBT), and `b` is [4H] -- small
         * enough that converting it once at the aclnn level costs less than a cross-core widening
         * round inside the phase A handshake would. See the note on `b` in single_layer_lstm_def.cpp. */
        biasL1Off = l1.TakeT<float>(SingleLayerLstmCube::BtElems(nChunk));
        l1Bytes = l1.cur;

        SingleLayerLstmCube::Bump bt;
        btOff = bt.TakeT<float>(SingleLayerLstmCube::BtElems(nChunk));

        SingleLayerLstmCube::Bump ub;
        cubOff = ub.TakeT<float>(plane);
        sumOff = correctionOff = yOff = tmpOff = maskOff = maskElems = 0;
        if (compensated) {
            sumOff = ub.TakeT<float>(plane);
            correctionOff = ub.TakeT<float>(plane);
            yOff = ub.TakeT<float>(plane);
            tmpOff = ub.TakeT<float>(plane);
            maskElems = SingleLayerLstmCube::CeilAlign(plane, SingleLayerLstmCube::C0_BYTES);
            maskOff = ub.TakeT<uint8_t>(maskElems);
        }
        ubBytes = ub.cur;
    }
};

/* `kCur` columns of x starting at column `k0`, for the `tc` timesteps of this chunk, laid into the
 * L1 A tile. One CopyInNd2Nz per batch row: that row's timesteps go contiguously at NZ row offset
 * r*tChunk, and consecutive timesteps of one row are B*I apart in x.
 *
 * THE OFFSET IS IN ELEMENTS, AND ONE NZ ROW IS c0 OF THEM. Inside a C0 block, row i sits at i*c0 --
 * which is why SingleLayerLstmCube::NzLayout::ColBlock multiplies by c0 as well. Writing
 * r*tChunk here (a ROW count used as an ELEMENT offset) overlaps every batch row onto the first
 * 1/c0 of its slot: it builds fine, runs fine, and the projection comes out wrong by O(1). */
__aicore__ inline void LoadATile(const AscendC::LocalTensor<float>& a1, const AscendC::GlobalTensor<float>& xGM,
                                 uint32_t t0, uint32_t tc, uint32_t k0, uint32_t kCur, uint32_t blockBase,
                                 const Layout& L)
{
    for (uint32_t r = 0; r < L.mBlk; ++r) {
        SingleLayerLstmCube::CopyInNd2Nz<float>(a1[r * L.tChunk * Layout::C0IN],
                                                xGM[(t0 * L.batch + blockBase + r) * L.inSize + k0], tc, kCur,
                                                L.batch * L.inSize, L.m);
    }
}

/* Cube half. `xGM` is [T, B, I] and `wGM` the fused [I+H, 4H], BOTH ALREADY FP32 -- at fp32 they
 * are the caller's own tensors, at fp16 and bf16 they are the workspace images the AIVs widened in
 * the prologue. Only the first I rows of w are ours. `bGM` is [4H] and is fp32 at every dtype. */
__aicore__ inline void CubeProject(AscendC::GlobalTensor<float> xGM, AscendC::GlobalTensor<float> wGM,
                                   AscendC::GlobalTensor<float> bGM, uint32_t blockBase, const Layout& L)
{
    AscendC::LocalTensor<float> a1(AscendC::TPosition::A1, 0, L.a1Elems);
    AscendC::LocalTensor<float> b1(AscendC::TPosition::B1, L.bOff, L.bElems);
    AscendC::LocalTensor<float> bl1(AscendC::TPosition::B1, L.biasL1Off, SingleLayerLstmCube::BtElems(L.nChunk));
    AscendC::LocalTensor<float> a2(AscendC::TPosition::A2, 0, L.a1Elems);
    AscendC::LocalTensor<float> b2(AscendC::TPosition::B2, 0, L.bElems);
    AscendC::LocalTensor<float> c(AscendC::TPosition::CO1, 0, L.cElems);
    AscendC::LocalTensor<float> bt(AscendC::TPosition::C2, L.btOff, SingleLayerLstmCube::BtElems(L.nChunk));
    AscendC::LocalTensor<float> cub(AscendC::TPosition::VECOUT, L.cubOff, L.plane);

    /* `tChunks` -- the TIME chunks. It was called nChunk until the N axis acquired a chunk of its
     * own; the two are different axes and the old name is now taken. */
    const uint32_t tChunks = SingleLayerLstmCube::CeilDiv(L.steps, L.tChunk);
    const uint32_t kBlocks = SingleLayerLstmCube::CeilDiv(L.inSize, L.kChunk); // the last one may be short
    const uint32_t gw = SingleLayerLstmFwd::GATES * L.hid;                     // row stride of w, in elements

    for (uint32_t ci = 0; ci < tChunks; ++ci) {
        const uint32_t t0 = ci * L.tChunk;
        const uint32_t tc = (t0 + L.tChunk <= L.steps) ? L.tChunk : (L.steps - t0);
        const uint32_t mNow = tc * L.mBlk;

        /* A tile: one CopyInNd2Nz per batch row, laying that row's `tc` timesteps contiguously at
         * NZ row offset r*tChunk. Consecutive timesteps of one row are B*I apart in x.
         *
         * ONCE PER TIME CHUNK WHEN THE WHOLE [m, I] FITS L1, which is the common case and the
         * schedule that shipped. When it does not, the same load runs inside the k loop for one
         * k-block at a time -- see the `!L.aResident` arm there. */
        if (L.aResident) {
            LoadATile(a1, xGM, t0, tc, 0, L.inSize, blockBase, L);
            SingleLayerLstmSync::WaitMte2ToMte1();
        }

        for (uint32_t g = 0; g < SingleLayerLstmFwd::GATES; ++g) {
            for (uint32_t nc = 0; nc < L.nChunks; ++nc) {
                const uint32_t nOff = nc * L.nChunk;
                const uint32_t nCur = (L.hid - nOff < L.nChunk) ? (L.hid - nOff) : L.nChunk;
                const uint32_t wCol = g * L.hid + nOff; // column origin of this tile in w and b

                /* Exactly nCur elements from GM, not BtElems(nCur). The bias-table burst is
                 * 64-byte granular so the L1 tile and BT slot are allocated at BtElems(nChunk), but
                 * the source `b[4H]` holds only nCur floats here: using the padded count walks off
                 * the end whenever nCur < 16 -- at H = 8 each gate would read 8 floats past its
                 * slice and gate 3 past the tensor. nCur is a multiple of 8, so the exact length is
                 * expressible. The BT pad beyond nCur may hold stale L1 bytes, which is harmless:
                 * Mmad's n is CeilAlign(nCur, 16) to match the L0C footprint, and the drain is given
                 * the real nCur. */
                SingleLayerLstmCube::CopyInFlat<float>(bl1, bGM[wCol], nCur);
                SingleLayerLstmSync::WaitMte2ToMte1();
                SingleLayerLstmCube::LoadBiasToBT(bt, bl1, nCur);

                for (uint32_t kb = 0; kb < kBlocks; ++kb) {
                    const uint32_t k0 = kb * L.kChunk;
                    /* THE LAST K BLOCK MAY BE SHORT, and every extent below takes `kCur` rather
                     * than kChunk for it. The L1 and L0B tiles are still allocated at the full
                     * kChunk, so the rows past kCur hold whatever was there -- Mmad is given the
                     * real k and never reads them, the same arrangement the N tail already uses. */
                    const uint32_t kCur = (L.inSize - k0 < L.kChunk) ? (L.inSize - k0) : L.kChunk;
                    if (!L.aResident) {
                        LoadATile(a1, xGM, t0, tc, k0, kCur, blockBase, L);
                        SingleLayerLstmSync::WaitMte2ToMte1();
                    }
                    /* w rows [kb*kChunk, +kChunk), columns [g*H + nOff, +nCur). A row range of an
                     * [I+H, 4H] matrix -- contiguous rows, stride 4H. */
                    SingleLayerLstmCube::CopyInNd2Nz<float>(b1, wGM[k0 * gw + wCol], kCur, nCur, gw, L.kChunk);
                    SingleLayerLstmSync::WaitMte2ToMte1();
                    /* L.m, not mNow: SplitA derives its srcStride from the `m` it is given, and that
                     * stride belongs to the PARENT L1 tile, which is always allocated for L.m rows.
                     * Handing it a smaller m makes it stride by the wrong amount and read the wrong
                     * rows -- silently. Mmad below is the one that gets the real extent. */
                    const uint32_t aOff = L.aResident ?
                                              (SingleLayerLstmCube::CeilAlign(L.m, SingleLayerLstmCube::CUBE_BLOCK) *
                                               k0) :
                                              0;
                    SingleLayerLstmCube::SplitA<float>(a2, a1[aOff], L.m, kCur);
                    /* parentK is the ALLOCATED tile height, not kCur: the source strides count
                     * fractals along K of the tile that was laid out, which is kChunk tall. */
                    SingleLayerLstmCube::SplitB<float>(b2, b1, kCur, nCur, L.kChunk);
                    SingleLayerLstmSync::WaitMte1ToM();
                    if (kb == 0) {
                        SingleLayerLstmCube::MmadBias(c, a2, b2, bt, mNow, nCur, kCur); // bias seeds L0C
                    } else if (L.compensated) {
                        SingleLayerLstmCube::MmadPlain(c, a2, b2, mNow, nCur, kCur);
                    } else {
                        SingleLayerLstmCube::MmadAccum(c, a2, b2, mNow, nCur, kCur);
                    }
                    /* The next k block reuses L0A/L0B, so its MTE1 loads must not overtake this
                     * Mmad's reads. Without this the last block silently wins parts of the earlier
                     * ones. */
                    AscendC::PipeBarrier<PIPE_ALL>();
                    if (L.compensated) {
                        SingleLayerLstmSync::WaitMToFix();
                        if (!(ci == 0 && g == 0 && nc == 0 && kb == 0)) {
                            SingleLayerLstmSync::CubeWaitVec();
                        }
                        SingleLayerLstmCube::DrainToUB(cub, c, mNow, nCur, SingleLayerLstmFwd::SPLIT);
                        SingleLayerLstmSync::CubeSignalVec();
                        SingleLayerLstmSync::WaitFixToM();
                    }
                }
                if (L.compensated) {
                    continue;
                }
                SingleLayerLstmSync::WaitMToFix();
                /* Not before the FIRST drain of all: nothing has been handed to the AIVs yet, so
                 * there is no round to wait for. Every later drain must wait, or it overwrites a UB
                 * tile the AIVs are still reading -- the V2C round is a write-after-read barrier as
                 * much as a data-ready signal. */
                if (!(ci == 0 && g == 0 && nc == 0)) {
                    SingleLayerLstmSync::CubeWaitVec();
                }
                SingleLayerLstmCube::DrainToUB(cub, c, mNow, nCur, SingleLayerLstmFwd::SPLIT);
                SingleLayerLstmSync::CubeSignalVec();
            }
        }
    }
}

/* Vector half: take each drained gate plane and scatter it into igates[t, row, g*H : (g+1)*H]. */
__aicore__ inline void VectorProject(AscendC::GlobalTensor<float> igGM, uint32_t blockBase, const Layout& L)
{
    AscendC::LocalTensor<float> cub(AscendC::TPosition::VECOUT, L.cubOff, L.plane);
    AscendC::LocalTensor<float> sum(AscendC::TPosition::VECCALC, L.sumOff, L.plane);
    AscendC::LocalTensor<float> correction(AscendC::TPosition::VECCALC, L.correctionOff, L.plane);
    AscendC::LocalTensor<float> yv(AscendC::TPosition::VECCALC, L.yOff, L.plane);
    AscendC::LocalTensor<float> tmp(AscendC::TPosition::VECCALC, L.tmpOff, L.plane);
    AscendC::LocalTensor<uint8_t> mask(AscendC::TPosition::VECCALC, L.maskOff,
                                       L.compensated ? L.maskElems : SingleLayerLstmCube::C0_BYTES);

    const uint32_t tChunks = SingleLayerLstmCube::CeilDiv(L.steps, L.tChunk);
    const uint32_t gw = SingleLayerLstmFwd::GATES * L.hid;

    for (uint32_t ci = 0; ci < tChunks; ++ci) {
        const uint32_t t0 = ci * L.tChunk;
        const uint32_t tc = (t0 + L.tChunk <= L.steps) ? L.tChunk : (L.steps - t0);
        const uint32_t mNow = tc * L.mBlk;
        const SingleLayerLstmCube::RowStripe stripe = SingleLayerLstmCube::DrainStripe(mNow, SingleLayerLstmFwd::SPLIT,
                                                                                       AscendC::GetSubBlockIdx());

        for (uint32_t g = 0; g < SingleLayerLstmFwd::GATES; ++g) {
            for (uint32_t nc = 0; nc < L.nChunks; ++nc) {
                const uint32_t nOff = nc * L.nChunk;
                const uint32_t nCur = (L.hid - nOff < L.nChunk) ? (L.hid - nOff) : L.nChunk;

                const uint32_t rounds = L.compensated ? SingleLayerLstmCube::CeilDiv(L.inSize, L.kChunk) : 1U;
                const uint32_t work = stripe.count * nCur;
                for (uint32_t kr = 0; kr < rounds; ++kr) {
                    SingleLayerLstmSync::VecWaitCube();
                    AscendC::PipeBarrier<PIPE_ALL>();
                    if (L.compensated && work != 0) {
                        if (kr == 0) {
                            AscendC::Adds(sum, cub, 0.0f, work);
                            AscendC::Duplicate(correction, 0.0f, work);
                        } else {
                            SingleLayerLstmVec::FoldCubePartial(sum, correction, cub, yv, tmp, mask, work);
                        }
                    }
                    if (kr + 1 < rounds) {
                        AscendC::PipeBarrier<PIPE_ALL>();
                        SingleLayerLstmSync::VecSignalCube();
                    }
                }
                if (L.compensated && work != 0) {
                    AscendC::Sub(sum, sum, correction, work);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                /* A subcore owning no rows still rides EVERY round: the V2C flag is a barrier and
                 * needs both subcores, or the cube waits forever. The round count is now
                 * tChunks * GATES * nChunks and must match the cube's loop exactly. */
                for (uint32_t i = 0; i < stripe.count; ++i) {
                    const uint32_t row = stripe.base + i; // (r, t) flattened, r-major
                    const uint32_t r = row / L.tChunk;
                    const uint32_t t = row % L.tChunk;
                    if (t >= tc || r >= L.mBlk) {
                        continue; // L0C padding at odd m
                    }
                    AscendC::DataCopy(igGM[((t0 + t) * L.batch + blockBase + r) * gw + g * L.hid + nOff],
                                      (L.compensated ? sum : cub)[i * nCur], nCur);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                if (!(ci == tChunks - 1 && g == SingleLayerLstmFwd::GATES - 1 && nc == L.nChunks - 1)) {
                    SingleLayerLstmSync::VecSignalCube();
                }
            }
        }
    }
    /* Flush on the PRODUCER, for the buffer this core wrote. Phase B's row split is not this
     * phase's M split, so an AIV can read igates rows a DIFFERENT core produced. Doing this on the
     * consumer instead has been measured, repeatedly, to be worse than doing nothing. */
    AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(igGM);
}

/* ============================ Phase A on the vector units ============================
 *
 * The cube accumulates K in one fp32 chain whose order is fixed in hardware, so the projection's
 * error grows as sqrt(K): measured against the correctly rounded result at batch 1, 4.55x the
 * rounding floor at K = 64, 7.86x at 256, 13.04x at 1024, 21.48x at 1687. The order is not tunable
 * from here -- capping the K block at 512, 256, 128 and 64 moved device time from 1.613 ms to
 * 0.980 ms while `y` came back bit-identical, because the L0C accumulation across k blocks is the
 * continuation of the cube's own order. Compensated (Neumaier) summation removes the growth and
 * needs the individual products, which only the vector units see.
 *
 * Offered only at small M, and only for phase A. The cube's M is padded to 16, so at m <= 8 more
 * than half its rows are padding: at T=4 B=1 I=705 H=982 the projection reached 10 M
 * multiply-adds/ms against 152 at B=16, so the time is the weight stream and moving the arithmetic
 * to the vector units costs little. Above that the cube is faster and no less accurate per output
 * element. Phase B keeps the cube: zeroing one weight matrix at a time on that shape gave 2.80e-07
 * with W_hh zeroed and 8.23e-08 with W_ih zeroed against 3.15e-07 for the pair, so phase A carries
 * the error -- `h` is bounded by tanh and sigmoid while `x` is not.
 *
 * The two AIVs split the 4H columns and share nothing: each owns a half, reads the whole of x for
 * this block's rows and writes its own columns. There is no cross-core round in this path, which is
 * why the cube half must skip phase A whenever it runs -- the handshake counts have to stay equal. */
struct VecLayout {
    uint32_t mBlk, inSize, hid, batch, steps, gw, m;
    uint32_t cBase, cw, kTile, cmpCw;
    uint32_t accOff, cmpOff, grpOff, yOff, tmpOff, wOff, xOff;
    uint32_t absOff, zeroOff, mskOff;
    uint32_t ubBytes;

    __aicore__ inline VecLayout(uint32_t mBlkIn, uint32_t iIn, uint32_t hIn, uint32_t bIn, uint32_t tIn,
                                uint32_t kTileIn, uint32_t sub)
        : mBlk(mBlkIn), inSize(iIn), hid(hIn), batch(bIn), steps(tIn)
    {
        gw = SingleLayerLstmFwd::GATES * hid;
        m = steps * mBlk;
        kTile = kTileIn;
        /* Halved on a 32-byte boundary so that both halves stay burst-aligned; the second AIV takes
         * whatever is left, which may be shorter and may be nothing at all. */
        const uint32_t half = SingleLayerLstmCube::CeilAlign(SingleLayerLstmCube::CeilDiv(gw, 2U), 8U);
        cBase = sub * half;
        cw = (cBase >= gw) ? 0U : ((gw - cBase < half) ? (gw - cBase) : half);

        SingleLayerLstmCube::Bump ub;
        accOff = ub.TakeT<float>(m * cw);
        cmpOff = ub.TakeT<float>(m * cw);
        grpOff = ub.TakeT<float>(SingleLayerLstmFwd::PROJ_VEC_NACC * cw);
        /* The non-finite guard below. `Compares` is the one level-2 op whose count must be a whole
         * number of vector repeats, so the plane it reads is rounded up and its tail is zeroed once
         * -- see SingleLayerLstmCube::CMP_REPEAT_ELEMS. */
        cmpCw = SingleLayerLstmCube::CeilAlign(cw, SingleLayerLstmCube::CMP_REPEAT_ELEMS);
        absOff = ub.TakeT<float>(cmpCw);
        zeroOff = ub.TakeT<float>(cw);
        mskOff = ub.TakeT<uint8_t>(SingleLayerLstmCube::CeilAlign(cmpCw, SingleLayerLstmCube::C0_BYTES));
        yOff = ub.TakeT<float>(cw);
        tmpOff = ub.TakeT<float>(cw);
        wOff = ub.TakeT<float>(kTile * cw);
        xOff = ub.TakeT<float>(m * inSize);
        ubBytes = ub.cur;
    }
};

/* igates[t, b, :] = bias + sum_k x[t, b, k] * w[k, :], for the rows of one batch block, with the sum
 * compensated at group granularity. Level 1, inside a group: PROJ_VEC_NACC interleaved partial sums,
 * one Axpy per term. Level 2, across groups: the group total goes into a Kahan accumulator, `comp`
 * carrying what the group addition could not represent.
 *
 * Two levels rather than per-term compensation because a plain fp32 chain rounds against its own
 * running total, so cutting the chain to GROUP terms and combining the pieces without loss leaves
 * only the error of the short chains. Measured over 30 draws of a 2008-term reduction: four
 * interleaved partial sums 1.64x better than one chain, sixteen 2.98x, two-level blocking 3.75x,
 * per-term compensation 12.78x. Per-term is the most accurate and costs six instructions where one
 * suffices; the two-level form keeps most of it at 1 + 12/PROJ_VEC_GROUP per term. */
__aicore__ inline void VectorProjectDirect(AscendC::GlobalTensor<float> xGM, AscendC::GlobalTensor<float> wGM,
                                           AscendC::GlobalTensor<float> bGM, AscendC::GlobalTensor<float> igGM,
                                           uint32_t blockBase, const VecLayout& V)
{
    if (V.cw == 0) {
        return; // this AIV owns no columns; 4H < 16 puts everything on the first
    }
    AscendC::LocalTensor<float> acc(AscendC::TPosition::VECCALC, V.accOff, V.m * V.cw);
    AscendC::LocalTensor<float> cmp(AscendC::TPosition::VECCALC, V.cmpOff, V.m * V.cw);
    AscendC::LocalTensor<float> grp(AscendC::TPosition::VECCALC, V.grpOff, SingleLayerLstmFwd::PROJ_VEC_NACC * V.cw);
    AscendC::LocalTensor<float> abv(AscendC::TPosition::VECCALC, V.absOff, V.cmpCw);
    AscendC::LocalTensor<float> zro(AscendC::TPosition::VECCALC, V.zeroOff, V.cw);
    AscendC::LocalTensor<uint8_t> msk(AscendC::TPosition::VECCALC, V.mskOff, V.cmpCw);
    AscendC::LocalTensor<float> yv(AscendC::TPosition::VECCALC, V.yOff, V.cw);
    AscendC::LocalTensor<float> tmp(AscendC::TPosition::VECCALC, V.tmpOff, V.cw);
    AscendC::LocalTensor<float> wst(AscendC::TPosition::VECCALC, V.wOff, V.kTile * V.cw);
    AscendC::LocalTensor<float> xst(AscendC::TPosition::VECCALC, V.xOff, V.m * V.inSize);

    /* x for every row this block owns, once. The rows are (t, r) flattened t-major, which is the
     * order the store at the end walks as well. */
    for (uint32_t t = 0; t < V.steps; ++t) {
        for (uint32_t r = 0; r < V.mBlk; ++r) {
            AscendC::DataCopy(xst[(t * V.mBlk + r) * V.inSize], xGM[((t * V.batch) + blockBase + r) * V.inSize],
                              V.inSize);
        }
    }
    for (uint32_t i = 0; i < V.m; ++i) {
        AscendC::DataCopy(acc[i * V.cw], bGM[V.cBase], V.cw); // the bias seeds the accumulator
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::Duplicate(cmp, static_cast<float>(0.0), V.m * V.cw);
    AscendC::Duplicate(zro, static_cast<float>(0.0), V.cw);
    /* Abs writes only the first cw lanes each time, so zeroing the rounded-up tail once is enough;
     * leaving it at whatever the plane last held would feed the comparator unrelated NaNs. */
    if (V.cmpCw > V.cw) {
        AscendC::Duplicate(abv[V.cw], static_cast<float>(0.0), V.cmpCw - V.cw);
    }

    for (uint32_t k0 = 0; k0 < V.inSize; k0 += V.kTile) {
        const uint32_t kc = (V.inSize - k0 < V.kTile) ? (V.inSize - k0) : V.kTile;
        /* w rows [k0, k0+kc) restricted to this AIV's columns. One burst of kc blocks, striding by
         * the full 4H row pitch. */
        SingleLayerLstmVec::LoadStripe(wst, wGM[k0 * V.gw + V.cBase], kc, V.cw, V.gw);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (uint32_t i = 0; i < V.m; ++i) {
            for (uint32_t g0 = 0; g0 < kc; g0 += SingleLayerLstmFwd::PROJ_VEC_GROUP) {
                const uint32_t gc = (kc - g0 < SingleLayerLstmFwd::PROJ_VEC_GROUP) ? (kc - g0) :
                                                                                     SingleLayerLstmFwd::PROJ_VEC_GROUP;
                AscendC::Duplicate(grp, static_cast<float>(0.0), SingleLayerLstmFwd::PROJ_VEC_NACC * V.cw);
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t kk = 0; kk < gc; ++kk) {
                    const float xv = xst.GetValue(i * V.inSize + k0 + g0 + kk);
                    /* dst += src * scalar, one instruction. The partial sum it lands in rotates so
                     * that consecutive terms do not queue behind one another. */
                    AscendC::Axpy(grp[(kk & (SingleLayerLstmFwd::PROJ_VEC_NACC - 1)) * V.cw], wst[(g0 + kk) * V.cw], xv,
                                  V.cw);
                }
                AscendC::PipeBarrier<PIPE_V>();
                /* Fold the interleaved partials pairwise, then hand ONE group total to the
                 * compensated accumulator. PROJ_VEC_NACC is a power of two, so this is exact-shaped
                 * and needs no odd-count branch. */
                for (uint32_t w = SingleLayerLstmFwd::PROJ_VEC_NACC >> 1; w >= 1; w >>= 1) {
                    for (uint32_t p = 0; p < w; ++p) {
                        AscendC::Add(grp[p * V.cw], grp[p * V.cw], grp[(p + w) * V.cw], V.cw);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Sub(yv, grp, cmp[i * V.cw], V.cw);
                AscendC::Add(tmp, acc[i * V.cw], yv, V.cw);
                AscendC::Sub(cmp[i * V.cw], tmp, acc[i * V.cw], V.cw);
                AscendC::Sub(cmp[i * V.cw], cmp[i * V.cw], yv, V.cw);
                /* The compensation must not outlive a non-finite accumulator. `cmp` is
                 * (acc + y - acc) - y, and when acc is +-inf that first difference is NaN, which
                 * then reaches every later group through `y = grp - cmp` and the closing
                 * `acc - cmp`, so the column comes back NaN where the cube returns +-inf. The suite
                 * reaches this through the bias alone: all twelve posinf/neginf b_ih and b_hh
                 * backward cases at T3_B5_I33_H17 failed this way the first time the path was
                 * widened past steps * batch <= 8. Once acc is non-finite there is no rounding left
                 * to carry and the correct carry is zero; NaN compares false, so one LT against
                 * FLT_MAX catches NaN and both infinities. */
                AscendC::Abs(abv, cmp[i * V.cw], V.cw);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Compares(msk, abv, FLT_MAX_F, AscendC::CMPMODE::LT, V.cmpCw);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Select(cmp[i * V.cw], msk, cmp[i * V.cw], zro, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                                V.cw);
                /* Adds by zero, not a DataCopy: the copy is a move pipe and would need a barrier
                 * against the vector writes on either side of it. */
                AscendC::Adds(acc[i * V.cw], tmp, static_cast<float>(0.0), V.cw);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    AscendC::Sub(acc, acc, cmp, V.m * V.cw);
    AscendC::PipeBarrier<PIPE_ALL>();
    for (uint32_t t = 0; t < V.steps; ++t) {
        for (uint32_t r = 0; r < V.mBlk; ++r) {
            AscendC::DataCopy(igGM[((t * V.batch) + blockBase + r) * V.gw + V.cBase], acc[(t * V.mBlk + r) * V.cw],
                              V.cw);
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(igGM);
}

/* The phase A -> phase B boundary. One full round, used purely as an AIV <-> AIV barrier: the two
 * subcores of a cluster have no direct handshake, so they synchronise THROUGH the cube. Needed
 * because phase A splits M by (r, t) while phase B splits by r, and at mBlk == 1 those differ. */
__aicore__ inline void CubeBoundary()
{
    SingleLayerLstmSync::CubeWaitVec();
    SingleLayerLstmSync::CubeSignalVec();
}

__aicore__ inline void VectorBoundary()
{
    SingleLayerLstmSync::VecSignalCube();
    SingleLayerLstmSync::VecWaitCube();
    AscendC::PipeBarrier<PIPE_ALL>();
}

} // namespace SingleLayerLstmProj

#endif // OPS_RNN_SINGLE_LAYER_LSTM_PROJ_IMPL_H
