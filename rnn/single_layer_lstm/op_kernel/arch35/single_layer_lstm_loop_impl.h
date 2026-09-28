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
 * \file single_layer_lstm_loop_impl.h
 * \brief Persistent FP32 recurrence. Batch rows are independent; h/c history uses [T+1,B,H].
 */
#ifndef OPS_RNN_SINGLE_LAYER_LSTM_LOOP_IMPL_H
#define OPS_RNN_SINGLE_LAYER_LSTM_LOOP_IMPL_H

#include "kernel_operator.h"
#include "cube_helper.h"
#include "vec_helper.h"
#include "sync_helper.h"
#include "single_layer_lstm_layout.h"
#include "compensated_sum.h"

/* Its own namespace. lstm_grad_impl.h defines its own Layout / cube / vector entry points with
 * different bodies; unqualified names would let the linker silently pick one.
 * SPLIT / GATES / C0F / PickRowsPerBlock / Layout are in single_layer_lstm_layout.h -- op_host needs them. */
namespace SingleLayerLstmFwd {

/* The L1 A-tile, [mBlk, H] NZ, compact -- so `rowsAligned` is its own aligned row count.
 * H must be a multiple of C0F or the feedback's column blocks do not line up. */
__aicore__ inline SingleLayerLstmCube::NzLayout ATileLayout(const Layout& L)
{
    return SingleLayerLstmCube::NzLayout(L.m, L.hid, C0F, L.m);
}

/* The L1 tile W_hh^T gate g occupies WHEN THE WEIGHTS ARE RESIDENT: [k=H, n=H] NZ, compact, fp32.
 * Shared by the cube's own fp32 load and by the AIV widening, so the two cannot disagree about
 * where a gate starts. Meaningless when they are streamed -- that mode's L1 tile is one
 * [kChunk, nChunk] staging pad, and the cube addresses it from offset 0. */
__aicore__ inline SingleLayerLstmCube::NzLayout BTileLayout(const Layout& L)
{
    return SingleLayerLstmCube::NzLayout(L.hid, L.n, C0F, L.hid);
}

/* The narrow-dtype prologue: every slab the cube will read, widened to fp32 in workspace, once.
 *
 * Mmad's two operands must share a width, and phase B is fp32 so that h_{t-1} is not rounded once
 * per timestep. Phase A used to keep the caller's width -- an fp16 x times an fp16 w is an exact
 * product and L0C accumulates in fp32, so on paper widening buys nothing. Measured, it buys about
 * 4x: at fp16, feeding the cube the same values as fp32 moves the operator from 92% of outputs
 * bit-identical to the correctly rounded result to 99.9%, and the worst element from 8 ulp to under
 * 1. The paper argument is right about the product and wrong about the reduction.
 *
 * Two slabs, one pass, one barrier: x [T, B, I] is phase A's A operand, and w [I+H, 4H] holds phase
 * A's B operand in rows [0, I) and W_hh^T in rows [I, I+H). Both are flat contiguous ranges, so one
 * routine widens either. The copies are written cooperatively by every AIV and published with a
 * single SyncAll -- one grid barrier per launch, not per timestep. One copy for the whole grid:
 * every cluster reads all of w. Round-robin over fixed-size chunks, so the bands are disjoint by
 * construction and the tail lands on whichever AIV draws it. */
template <typename TIn>
__aicore__ inline void WidenSlabToGm(AscendC::GlobalTensor<float> dstGM, AscendC::GlobalTensor<TIn> srcGM,
                                     uint32_t total, uint32_t aivIdx, uint32_t aivNum, const Layout& L)
{
    if (total == 0 || L.wStageElems == 0) {
        return;
    }
    AscendC::LocalTensor<TIn> raw(AscendC::TPosition::VECCALC, L.wraw, L.wStageElems);
    AscendC::LocalTensor<float> cvt(AscendC::TPosition::VECCALC, L.wcvt, L.wStageElems);
    const uint32_t chunks = SingleLayerLstmCube::CeilDiv(total, L.wStageElems);

    for (uint32_t ci = aivIdx; ci < chunks; ci += aivNum) {
        const uint32_t off = ci * L.wStageElems;
        const uint32_t elems = (total - off < L.wStageElems) ? (total - off) : L.wStageElems;
        SingleLayerLstmVec::WidenFromGm<TIn>(cvt, raw, srcGM[off], elems);
        AscendC::PipeBarrier<PIPE_ALL>();
        /* StoreToGm's fp32 arm is a plain DataCopy and wants a multiple of 8 elements. The chunk is
         * a multiple of 64, so only the LAST one can be short -- and it is `total` minus a multiple
         * of 64, so the caller has to hand over a total that is itself a multiple of 8. Both slabs
         * are: 4H*(I+H) with H a multiple of 8, and T*B*I with I a multiple of 8. */
        SingleLayerLstmVec::StoreToGm<float>(dstGM[off], cvt, elems);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

template <typename TIn>
__aicore__ inline void PrepareBias(AscendC::GlobalTensor<float> dstGM, AscendC::GlobalTensor<TIn> inputGM,
                                   AscendC::GlobalTensor<TIn> hiddenGM, bool hasHiddenBias, uint32_t total,
                                   uint32_t aivIdx, uint32_t aivNum, const Layout& L)
{
    // Reuse recurrence scratch before phase A; no additional UB allocation is needed.
    AscendC::LocalTensor<TIn> raw(AscendC::TPosition::VECCALC, L.nbuf, L.plane);
    AscendC::LocalTensor<float> input(AscendC::TPosition::VECCALC, L.t1, L.plane);
    AscendC::LocalTensor<float> hidden(AscendC::TPosition::VECCALC, L.t2, L.plane);
    const uint32_t chunks = SingleLayerLstmCube::CeilDiv(total, L.plane);
    for (uint32_t ci = aivIdx; ci < chunks; ci += aivNum) {
        const uint32_t off = ci * L.plane;
        const uint32_t elems = (total - off < L.plane) ? (total - off) : L.plane;
        SingleLayerLstmVec::WidenFromGm<TIn>(input, raw, inputGM[off], elems);
        AscendC::PipeBarrier<PIPE_ALL>();
        if (hasHiddenBias) {
            SingleLayerLstmVec::WidenFromGm<TIn>(hidden, raw, hiddenGM[off], elems);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Add(input, input, hidden, elems);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        SingleLayerLstmVec::StoreToGm<float>(dstGM[off], input, elems);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

/* The whole prologue, on the AIVs. Returns with the images published to the whole grid.
 * `cluster` and `blockDim` come from the caller rather than being derived from the Layout: an AIV's
 * GetBlockIdx() is already the flat subcore index on this part but its GetBlockNum() is the AIC
 * count, and the row split is ragged, so neither the hardware query nor L.b / L.m gives a band
 * assignment every cluster agrees on. A disagreement leaves chunks of the image unwritten. */
template <typename TIn>
__aicore__ inline void WidenInputs(AscendC::GlobalTensor<float> wF32, AscendC::GlobalTensor<TIn> wGM, uint32_t wElems,
                                   AscendC::GlobalTensor<float> xF32, AscendC::GlobalTensor<TIn> xGM, uint32_t xElems,
                                   AscendC::GlobalTensor<float> bF32, AscendC::GlobalTensor<TIn> bGM,
                                   AscendC::GlobalTensor<TIn> biasHhGM, bool hasBiasHh, uint32_t bElems,
                                   uint32_t cluster, uint32_t blockDim, const Layout& L)
{
    const uint32_t aivNum = blockDim * (SPLIT ? 2U : 1U);
    const uint32_t aivIdx = SPLIT ? (cluster * 2U + AscendC::GetSubBlockIdx()) : cluster;
    if constexpr (sizeof(TIn) != sizeof(float)) {
        WidenSlabToGm<TIn>(wF32, wGM, wElems, aivIdx, aivNum, L);
        WidenSlabToGm<TIn>(xF32, xGM, xElems, aivIdx, aivNum, L);
        AscendC::PipeBarrier<PIPE_ALL>();
        /* FLUSH ON THE PRODUCER, before the barrier that publishes the bands. Every cluster's cube
         * reads bands this core did not write, so the ordering the V2C flag gives inside a cluster
         * is not enough on its own -- the same argument phase A's igates rest on, and the same
         * placement (VectorProject flushes what it wrote, not what it is about to read). */
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(wF32);
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(xF32);
    }
    if (sizeof(TIn) != sizeof(float) || hasBiasHh) {
        PrepareBias<TIn>(bF32, bGM, biasHhGM, hasBiasHh, bElems, aivIdx, aivNum, L);
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(bF32);
        AscendC::SyncAll<true>();
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

/* ---------------------------------------------------------------------------------------------
 * Cube half
 * ------------------------------------------------------------------------------------------- */
template <typename TIn>
__aicore__ inline void CubeLoop(AscendC::GlobalTensor<float> wRecGM, AscendC::GlobalTensor<float> hAllGM,
                                uint32_t blockBase, const Layout& L)
{
    AscendC::LocalTensor<float> a1(AscendC::TPosition::A1, 0, L.aTileElems);
    AscendC::LocalTensor<float> a2(AscendC::TPosition::A2, 0, L.aTileElems);
    AscendC::LocalTensor<float> b2(AscendC::TPosition::B2, 0, L.bChunkElems);

    /* `wRecGM` IS ALWAYS THE FP32 IMAGE OF W_hh^T, [H, 4H] row-major, whatever the operator's
     * dtype. At fp32 it is the caller's own weight at a pointer offset; at fp16 and bf16 it is the
     * workspace copy the AIVs widened. The cube therefore has ONE load path instead of a
     * per-dtype one, and the dtype only decides WHO produced the bytes. */
    const uint32_t wPitch = GATES * L.hid; // GM row pitch of the fused [H, 4H] hidden weight
    const uint32_t mAligned = SingleLayerLstmCube::CeilAlign(L.m, SingleLayerLstmCube::CUBE_BLOCK);

    /* Resident: W_hh^T reaches L1 once for the whole sequence, which is the point of the kernel being
     * persistent. Gate g is the column range [g*H, (g+1)*H) of [H, 4H], so each tile is a compact
     * [k=H, n=H] load carrying the parent's row stride. One path for all three dtypes, `wRecGM`
     * being fp32 whoever produced it. Streaming: nothing is preloaded, the four [kChunk, nChunk]
     * tiles are pulled from GM inside the loop once per (n-chunk, k-chunk) per timestep. */
    if (L.wResident) {
        for (uint32_t g = 0; g < GATES; ++g) {
            AscendC::LocalTensor<float> b1(AscendC::TPosition::B1, L.bOff[g], L.bElems);
            SingleLayerLstmCube::CopyInNd2Nz<float>(b1, wRecGM[g * L.hid], L.hid, L.n, wPitch, L.hid);
        }
        SingleLayerLstmSync::WaitMte2ToMte1();
    }

    /* The K extent of whatever L1 tile SplitB reads -- a whole gate when resident, one staging pad
     * when streaming. It is the SOURCE STRIDE, not the load size, and the two stop being the same
     * number the moment K is tiled. `bRowsAligned` is the matching column-block stride in elements:
     * column block kb of an NZ tile starts at kb * rowsAligned * c0, and both chunk origins are
     * multiples of c0, so the tile at (kOff, nOff) begins at nOff*bRowsAligned + kOff*C0F. */
    const uint32_t bParentK = L.wResident ? L.hid : L.kChunk;
    const uint32_t bRowsAligned = SingleLayerLstmCube::CeilAlign(bParentK, SingleLayerLstmCube::CUBE_BLOCK);

    for (uint32_t s = 0; s < L.steps; ++s) {
        /* N outside, K inside. The four L0C tiles are sized for one n-chunk, so an n-chunk must be
         * finished before the next starts and the whole K axis runs inside it, accumulating into
         * those same tiles: k-chunk 0 seeds with MmadPlain and the rest add with MmadAccum. Long
         * reductions instead drain each independently seeded K block and compensate their fold in
         * VectorLoop -- a true change of summation order, which merely changing kChunk with
         * continued MmadAccum would not be. The price is that SplitA is re-issued per
         * (n-chunk, k-chunk); swapping the loops would need nChunks * GATES L0C tiles live at once,
         * the footprint that capped H in the first place. */
        for (uint32_t ch = 0; ch < L.nChunks; ++ch) {
            const uint32_t nOff = ch * L.nChunk;
            const uint32_t nCur = (L.n - nOff < L.nChunk) ? (L.n - nOff) : L.nChunk;

            /* One cross-core round per column chunk for short K, and one per K block within each
             * column chunk for compensated long K. The AIVs consume the drain one chunk at a time,
             * so their UB planes are [rowsMax, nChunk] rather than [rowsMax, H], which is what used
             * to bound hidden_size. The wait also carries the two orderings the loop needs: at
             * ch == 0 that h_{t-1} is complete in L1, and at every ch that the AIVs have finished
             * reading the previous drain. */
            SingleLayerLstmSync::CubeWaitVec();

            /* Once per timestep when the whole [m, H] A tile fits L0A, which is the schedule that
             * shipped and the common case -- every gate and every chunk then indexes into it. When
             * it does not fit, SplitA moves inside the k loop and reloads one [m, kChunk] chunk at
             * a time; nothing else about the loop changes. It sits under ch == 0 rather than above
             * the chunk loop because h_{t-1} is only known to be in L1 after the first wait. */
            if (ch == 0 && L.aResident) {
                SingleLayerLstmCube::SplitA<float>(a2, a1, L.m, L.hid);
            }

            for (uint32_t kc = 0; kc < L.kChunks; ++kc) {
                if (CompensateCubeK(L.k) && kc != 0) {
                    SingleLayerLstmSync::CubeWaitVec();
                }
                const uint32_t kOff = kc * L.kChunk;
                const uint32_t kCur = (L.k - kOff < L.kChunk) ? (L.k - kOff) : L.kChunk;

                /* Where h_{t-1} comes from when it is NOT resident: hAll slot s IS h_{s-1} -- the
                 * AIVs write it there every timestep for the backward, so the cube reads the same
                 * bytes rather than a second copy, and the UB->L1 feedback is not issued at all.
                 * The rows one cluster owns are contiguous in that [T+1, B, H] buffer, so the load
                 * is one Nd2Nz with the history's row pitch.
                 *
                 * IT IS SLOT s, WHILE THE AIVs ARE WRITING SLOT s+1 -- that separation is what
                 * makes re-reading A once per chunk safe here, and it is exactly what an L1 copy
                 * re-read per chunk would not have: see the note on `aResident`. */
                if (!L.aResident) {
                    SingleLayerLstmCube::CopyInNd2Nz<float>(a1, hAllGM[(s * L.mAll + blockBase) * L.hid + kOff], L.m,
                                                            kCur, L.hid, L.m);
                    SingleLayerLstmSync::WaitMte2ToMte1();
                    SingleLayerLstmCube::SplitA<float>(a2, a1, L.m, kCur);
                }

                /* STREAMING: refill the four staging pads. All four loads are issued before any of
                 * them is consumed, so MTE2 has four tiles in flight against one MTE1 wait. The
                 * pads were last read by the previous k-chunk's SplitB, which the PipeBarrier at
                 * the bottom of that gate loop already ordered against. */
                if (!L.wResident) {
                    for (uint32_t g = 0; g < GATES; ++g) {
                        AscendC::LocalTensor<float> b1(AscendC::TPosition::B1, L.bOff[g], L.bElems);
                        SingleLayerLstmCube::CopyInNd2Nz<float>(b1, wRecGM[kOff * wPitch + g * L.hid + nOff], kCur,
                                                                nCur, wPitch, bParentK);
                    }
                    SingleLayerLstmSync::WaitMte2ToMte1();
                }

                for (uint32_t g = 0; g < GATES; ++g) {
                    AscendC::LocalTensor<float> b1(AscendC::TPosition::B1, L.bOff[g], L.bElems);
                    AscendC::LocalTensor<float> c(AscendC::TPosition::CO1, L.cOff[g], L.cElems);
                    const uint32_t bIn = L.wResident ? (nOff * bRowsAligned + kOff * C0F) : 0;
                    SingleLayerLstmCube::SplitB<float>(b2, b1[bIn], kCur, nCur, bParentK);
                    SingleLayerLstmSync::WaitMte1ToM();
                    /* Resident: index into the whole tile already in L0A. Streaming: the pad holds
                     * this k-chunk and nothing else. */
                    const AscendC::LocalTensor<float> aIn = L.aResident ? a2[kOff * mAligned] : a2;
                    if (kc == 0 || CompensateCubeK(L.k)) {
                        SingleLayerLstmCube::MmadPlain(c, aIn, b2, L.m, nCur, kCur);
                    } else {
                        SingleLayerLstmCube::MmadAccum(c, aIn, b2, L.m, nCur, kCur);
                    }
                    /* The next gate reuses L0B, so its MTE1 load must not overtake this Mmad's
                     * reads. Without this the last gate silently wins parts of the earlier ones.
                     * It is PIPE_ALL, so it also orders this SplitB against the next k-chunk's
                     * MTE2 refill of the same staging pad. */
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                if (CompensateCubeK(L.k)) {
                    SingleLayerLstmSync::WaitMToFix();
                    for (uint32_t g = 0; g < GATES; ++g) {
                        AscendC::LocalTensor<float> c(AscendC::TPosition::CO1, L.cOff[g], L.cElems);
                        AscendC::LocalTensor<float> cUB(AscendC::TPosition::VECOUT, L.cub[g], L.plane);
                        SingleLayerLstmCube::DrainToUB(cUB, c, L.m, nCur, SPLIT);
                    }
                    SingleLayerLstmSync::CubeSignalVec();
                }
            }
            if (CompensateCubeK(L.k)) {
                continue;
            }
            SingleLayerLstmSync::WaitMToFix();
            for (uint32_t g = 0; g < GATES; ++g) {
                AscendC::LocalTensor<float> c(AscendC::TPosition::CO1, L.cOff[g], L.cElems);
                AscendC::LocalTensor<float> cUB(AscendC::TPosition::VECOUT, L.cub[g], L.plane);
                /* The destination is a COMPACT [rowsMax, nCur] plane -- the AIVs hold one chunk,
                 * not a whole gate -- so the default pitch is the right one. It was L.n back when
                 * the drain filled a column range of a gate-wide plane. */
                SingleLayerLstmCube::DrainToUB(cUB, c, L.m, nCur, SPLIT);
            }
            SingleLayerLstmSync::CubeSignalVec();
        }
    }
}

/* ---------------------------------------------------------------------------------------------
 * Vector half. Gate order is i, f, j, o (ops-nn `gate_order` "ifjo"); `store` holds them
 * POST-activation, which is exactly what lstm_grad reads back.
 * ------------------------------------------------------------------------------------------- */
template <typename TIn>
__aicore__ inline void VectorLoop(const SingleLayerLstmCube::RowStripe& stripe, uint32_t blockBase,
                                  AscendC::GlobalTensor<float> igGM, AscendC::GlobalTensor<TIn> h0GM,
                                  AscendC::GlobalTensor<TIn> c0GM, AscendC::GlobalTensor<float> hAllGM,
                                  AscendC::GlobalTensor<float> cAllGM, AscendC::GlobalTensor<float> storeGM,
                                  AscendC::GlobalTensor<TIn> coGM, uint32_t wantCo, const Layout& L)
{
    AscendC::LocalTensor<float> gi(AscendC::TPosition::VECCALC, L.gi, L.plane);
    AscendC::LocalTensor<float> gf(AscendC::TPosition::VECCALC, L.gf, L.plane);
    AscendC::LocalTensor<float> gg(AscendC::TPosition::VECCALC, L.gg, L.plane);
    AscendC::LocalTensor<float> go(AscendC::TPosition::VECCALC, L.go, L.plane);
    AscendC::LocalTensor<float> cc(AscendC::TPosition::VECCALC, L.cc, L.plane);
    AscendC::LocalTensor<float> hh(AscendC::TPosition::VECCALC, L.hh, L.plane);
    AscendC::LocalTensor<float> co(AscendC::TPosition::VECCALC, L.co, L.plane);
    AscendC::LocalTensor<float> t1(AscendC::TPosition::VECCALC, L.t1, L.plane);
    AscendC::LocalTensor<float> t2(AscendC::TPosition::VECCALC, L.t2, L.plane);
    AscendC::LocalTensor<float> t3(AscendC::TPosition::VECCALC, L.t3, L.plane);
    AscendC::LocalTensor<float> t4(AscendC::TPosition::VECCALC, L.t4, L.plane);
    AscendC::LocalTensor<uint8_t> msk(AscendC::TPosition::VECCALC, L.msk, L.mskElems);
    AscendC::LocalTensor<float> a1(AscendC::TPosition::A1, 0, L.aTileElems);
    /* One staging plane serves both directions of the dtype boundary here, because they never
     * overlap: init_h / init_c are widened once before the sweep, and tanhc is narrowed inside it.
     * Unread at fp32. */
    AscendC::LocalTensor<TIn> nstage(AscendC::TPosition::VECCALC, L.nbuf, L.plane);

    const uint32_t gRow = blockBase + stripe.base; // GLOBAL row this AIV starts at
    const uint32_t rowOff = gRow * L.hid;
    const uint32_t gw = GATES * L.hid; // row pitch of the igates / store blocks
    const SingleLayerLstmCube::NzLayout aLay = ATileLayout(L);

    /* A subcore that owns no rows still has to ride EVERY round: the V2C flag is a barrier and needs
     * both subcores, or the cube waits forever. There are now nChunks rounds per timestep, not one,
     * and this count has to follow the cube's loop exactly. */
    if (stripe.count == 0) {
        const uint32_t rounds = CompensateCubeK(L.k) ? L.kChunks : 1U;
        for (uint32_t r = 0; r < L.steps * L.nChunks * rounds; ++r) {
            SingleLayerLstmSync::VecSignalCube();
            SingleLayerLstmSync::VecWaitCube();
        }
        return;
    }

    /* The cell state lives in cAllGM, not in UB, and that is what unbounds hidden_size here. c_t is
     * [rowsMax, H] and is the only value the recurrence carries between timesteps, so holding it on
     * chip would put an H-sized plane back into the budget. It is already written to cAllGM every
     * step for the backward, so reading it back is one extra [rowsMax, nChunk] load per round. h_t
     * needs no such treatment: it is recomputed from o and tanh(c_t) every step. Slot 0 of the
     * [T+1, B, H] histories is h_0 / c_0, which is what lets the backward take h_prev / c_prev as
     * [:-1] and output_h / output_c as [1:] -- views, not copies. */
    for (uint32_t ch = 0; ch < L.nChunks; ++ch) {
        const uint32_t nOff = ch * L.nChunk;
        const uint32_t nCur = (L.hid - nOff < L.nChunk) ? (L.hid - nOff) : L.nChunk;
        SingleLayerLstmVec::LoadStripeWidened<TIn>(hh, nstage, h0GM[rowOff + nOff], stripe.count, nCur, L.hid);
        /* Between the two, not merely after them: both widen through the SAME staging plane, so the
         * second copy into it must not overtake the first Cast out of it. */
        AscendC::PipeBarrier<PIPE_ALL>();
        SingleLayerLstmVec::LoadStripeWidened<TIn>(cc, nstage, c0GM[rowOff + nOff], stripe.count, nCur, L.hid);
        AscendC::PipeBarrier<PIPE_ALL>();
        SingleLayerLstmVec::StoreStripe(hAllGM[rowOff + nOff], hh, stripe.count, nCur, L.hid);
        SingleLayerLstmVec::StoreStripe(cAllGM[rowOff + nOff], cc, stripe.count, nCur, L.hid);
        /* The SAME stripe the drain will hand back -- L1 has no write arbitration, so that identity
         * is the only thing keeping the two AIVs off each other's bytes (SingleLayerLstmCube::RowStripe).
         * `nOff / C0F` is the destination column block: the A tile is one [m, H] NZ image and this
         * chunk is a column range of it.
         *
         * Skipped entirely when the A tile no longer fits L1: the cube then reads h_{t-1} out of
         * hAll, which the StoreStripe above has just written. */
        if (L.aResident) {
            SingleLayerLstmVec::FeedbackToL1<float>(a1, hh, aLay, nCur, nOff / C0F, stripe);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    for (uint32_t t = 0; t < L.steps; ++t) {
        /* The history slot holding c_{t-1}: slot 0 is c_0, so step t reads slot t and writes t+1. */
        const uint32_t pOff = (t * L.mAll + gRow) * L.hid;
        const uint32_t oOff = ((t + 1) * L.mAll + gRow) * L.hid;
        const uint32_t sOff = (t * L.mAll + gRow) * gw;    // igates / store, step t
        const uint32_t cOff = (t * L.mAll + gRow) * L.hid; // tanhc is [T,B,H]: no slot 0

        for (uint32_t ch = 0; ch < L.nChunks; ++ch) {
            const uint32_t nOff = ch * L.nChunk;
            const uint32_t nCur = (L.hid - nOff < L.nChunk) ? (L.hid - nOff) : L.nChunk;
            const uint32_t work = stripe.count * nCur;

            AscendC::LocalTensor<float> u0(AscendC::TPosition::VECOUT, L.cub[0], L.plane);
            AscendC::LocalTensor<float> u1(AscendC::TPosition::VECOUT, L.cub[1], L.plane);
            AscendC::LocalTensor<float> u2(AscendC::TPosition::VECOUT, L.cub[2], L.plane);
            AscendC::LocalTensor<float> u3(AscendC::TPosition::VECOUT, L.cub[3], L.plane);
            const uint32_t rounds = CompensateCubeK(L.k) ? L.kChunks : 1U;
            for (uint32_t kr = 0; kr < rounds; ++kr) {
                /* Both subcores signal exactly once per independent partial.
                 * The barrier before release also orders vector reads/writes of
                 * the drain destinations against the cube's next Fixpipe. */
                AscendC::PipeBarrier<PIPE_ALL>();
                SingleLayerLstmSync::VecSignalCube();
                SingleLayerLstmSync::VecWaitCube();
                AscendC::PipeBarrier<PIPE_ALL>();
                if (kr == 0) {
                    AscendC::DataCopy(gi, u0, work);
                    AscendC::DataCopy(gf, u1, work);
                    AscendC::DataCopy(gg, u2, work);
                    AscendC::DataCopy(go, u3, work);
                    if (rounds > 1) {
                        AscendC::Duplicate(t1, 0.0f, work);
                        AscendC::Duplicate(t2, 0.0f, work);
                        AscendC::Duplicate(t3, 0.0f, work);
                        AscendC::Duplicate(t4, 0.0f, work);
                    }
                } else {
                    /* h/c archive is already in GM; hh/co are dead here, while
                     * cc remains untouched for the one-N-chunk recurrence. */
                    SingleLayerLstmVec::FoldCubePartial(gi, t1, u0, hh, co, msk, work);
                    SingleLayerLstmVec::FoldCubePartial(gf, t2, u1, hh, co, msk, work);
                    SingleLayerLstmVec::FoldCubePartial(gg, t3, u2, hh, co, msk, work);
                    SingleLayerLstmVec::FoldCubePartial(go, t4, u3, hh, co, msk, work);
                }
            }
            if (rounds > 1) {
                AscendC::Sub(gi, gi, t1, work);
                AscendC::Sub(gf, gf, t2, work);
                AscendC::Sub(gg, gg, t3, work);
                AscendC::Sub(go, go, t4, work);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            SingleLayerLstmVec::LoadStripe(t1, igGM[sOff + 0 * L.hid + nOff], stripe.count, nCur, gw);
            SingleLayerLstmVec::LoadStripe(t2, igGM[sOff + 1 * L.hid + nOff], stripe.count, nCur, gw);
            SingleLayerLstmVec::LoadStripe(t3, igGM[sOff + 2 * L.hid + nOff], stripe.count, nCur, gw);
            SingleLayerLstmVec::LoadStripe(t4, igGM[sOff + 3 * L.hid + nOff], stripe.count, nCur, gw);
            /* c_{t-1}. NOT READ BACK WHEN THE CHUNK IS THE WHOLE ROW: with one column chunk `cc`
             * still holds c_{t-1} from the previous timestep, exactly as it did before the H axis
             * was tiled, so the load would re-read what UB already has. It is only when the row is
             * split that a chunk's c cannot survive the other chunks and has to come from the
             * history buffer -- which the StoreStripe below wrote for the backward anyway. */
            if (L.nChunks != 1) {
                SingleLayerLstmVec::LoadStripe(cc, cAllGM[pOff + nOff], stripe.count, nCur, L.hid);
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            /* pre-activation gate = hgates + igates. The bias is already inside igates, seeded there
             * by phase A's bias table, so nothing here reads a bias. */
            AscendC::Add(gi, gi, t1, work);
            AscendC::Add(gf, gf, t2, work);
            AscendC::Add(gg, gg, t3, work);
            AscendC::Add(go, go, t4, work);
            AscendC::PipeBarrier<PIPE_ALL>();

            // c <- f*c + i*j, h <- o*tanhc, tanhc <- tanh(c). Every operation is elementwise over
            // columns, which is why chunking the H axis here changes nothing numerically.
            SingleLayerLstmVec::SingleLayerLstmGates(gi, gf, gg, go, cc, hh, co, t1, t2, t3, msk, work);
            AscendC::PipeBarrier<PIPE_ALL>();

            SingleLayerLstmVec::StoreStripe(hAllGM[oOff + nOff], hh, stripe.count, nCur, L.hid);
            SingleLayerLstmVec::StoreStripe(cAllGM[oOff + nOff], cc, stripe.count, nCur, L.hid);
            SingleLayerLstmVec::StoreStripe(storeGM[sOff + 0 * L.hid + nOff], gi, stripe.count, nCur, gw);
            SingleLayerLstmVec::StoreStripe(storeGM[sOff + 1 * L.hid + nOff], gf, stripe.count, nCur, gw);
            SingleLayerLstmVec::StoreStripe(storeGM[sOff + 2 * L.hid + nOff], gg, stripe.count, nCur, gw);
            SingleLayerLstmVec::StoreStripe(storeGM[sOff + 3 * L.hid + nOff], go, stripe.count, nCur, gw);
            /* Opt-in, and the branch is the point. `tanhc` is an archive output the backward can
             * also recompute; paying an unconditional [rows,H] store per step for a plane one
             * caller uses is a cost that should be asked for. The scalar branch is free next to the
             * store it guards. */
            if (wantCo != 0) {
                SingleLayerLstmVec::StoreStripeNarrowed<TIn>(coGM[cOff + nOff], nstage, co, stripe.count, nCur, L.hid);
            }

            /* h_t for the NEXT round's cube. On the last round nobody consumes it, which is
             * harmless. It rides the same MTE3 pipe as VecSignalCube, so program order is the
             * ordering -- and the cube's SplitA reads the whole A tile only at ch == 0 of the next
             * timestep, by which point every chunk has been written. The StoreStripe into hAll
             * above is the streaming mode's version of this same write. */
            if (L.aResident) {
                SingleLayerLstmVec::FeedbackToL1<float>(a1, hh, aLay, nCur, nOff / C0F, stripe);
            }
        }
    }
}

} // namespace SingleLayerLstmFwd

#endif // OPS_RNN_SINGLE_LAYER_LSTM_LOOP_IMPL_H
