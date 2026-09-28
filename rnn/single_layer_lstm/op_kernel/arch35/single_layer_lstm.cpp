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
 * \file single_layer_lstm.cpp
 * \brief Ascend950 forward: input projection, recurrent loop, then output scatter.
 */

/* Projection-only UT mode: omit phases B and C on both core types. It does not produce final outputs. */

#include "kernel_operator.h"
#include "single_layer_lstm_tiling_data.h"
#include "single_layer_lstm_loop_impl.h"
#include "single_layer_lstm_proj_impl.h"

/* DTYPE_W is the common floating-point IO dtype supplied by the build. */
using SingleLayerLstmIn = DTYPE_W;

namespace {

/* Scatter gate slots i,f,j,o into named outputs i,j,f,o; each AIV reads only its own rows. */
template <typename TIn>
__aicore__ inline void ScatterPlanes(const SingleLayerLstmCube::RowStripe& stripe, uint32_t blockBase,
                                     AscendC::GlobalTensor<float> hAllGM, AscendC::GlobalTensor<float> cAllGM,
                                     AscendC::GlobalTensor<float> storeGM, AscendC::GlobalTensor<TIn> outY,
                                     AscendC::GlobalTensor<TIn> outH, AscendC::GlobalTensor<TIn> outC,
                                     AscendC::GlobalTensor<TIn> outI, AscendC::GlobalTensor<TIn> outJ,
                                     AscendC::GlobalTensor<TIn> outF, AscendC::GlobalTensor<TIn> outO,
                                     const SingleLayerLstmFwd::Layout& L, uint32_t seqLenMax)
{
    if (stripe.count == 0) {
        return;
    }
    AscendC::LocalTensor<float> buf(AscendC::TPosition::VECCALC, L.t1, L.plane);
    AscendC::LocalTensor<float> zero(AscendC::TPosition::VECCALC, L.t2, L.plane);
    AscendC::LocalTensor<TIn> nbuf(AscendC::TPosition::VECCALC, L.nbuf, L.plane);
    AscendC::LocalTensor<TIn> nzbuf(AscendC::TPosition::VECCALC, L.nzero, L.plane);

    const uint32_t gRow = blockBase + stripe.base;
    const uint32_t gw = SingleLayerLstmFwd::GATES * L.hid;
    AscendC::Duplicate(zero, 0.0f, L.plane);
    AscendC::PipeBarrier<PIPE_ALL>();

    // store slot -> output tensor. Slot order is i, f, j, o; output order is i, j, f, o.
    AscendC::GlobalTensor<TIn> gate[SingleLayerLstmFwd::GATES] = {outI, outF, outJ, outO};

    /* All outputs are rounded to the declared dtype. */
    for (uint32_t t = 0; t < L.steps; ++t) {
        const uint32_t off = (t * L.mAll + gRow) * L.hid;
        const uint32_t sOff = (t * L.mAll + gRow) * gw;
        /* output_c = cAll[1:], output_h = hAll[1:] -- slot 0 holds init_c / init_h, so slot t+1 is
         * step t. `y` is the same quantity as `output_h`; both are declared outputs, so both are
         * written from the one buffer rather than one being derived downstream. */
        const uint32_t hOff = ((t + 1) * L.mAll + gRow) * L.hid;
        /* Every plane is zero past seq_length. An output that is merely "not meaningful" there
         * still has to be WRITTEN: the caller allocates the outputs and nothing zeroes them, so an
         * untouched tail returns whatever was in that memory. */
        const bool live = (t < seqLenMax);

        for (uint32_t ch = 0; ch < L.nChunks; ++ch) {
            const uint32_t nOff = ch * L.nChunk;
            const uint32_t nCur = (L.hid - nOff < L.nChunk) ? (L.hid - nOff) : L.nChunk;

            for (uint32_t g = 0; g < SingleLayerLstmFwd::GATES; ++g) {
                if (live) {
                    SingleLayerLstmVec::LoadStripe(buf, storeGM[sOff + g * L.hid + nOff], stripe.count, nCur, gw);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    SingleLayerLstmVec::StoreStripeNarrowed<TIn>(gate[g][off + nOff], nbuf, buf, stripe.count, nCur,
                                                                 L.hid);
                } else {
                    SingleLayerLstmVec::StoreStripeNarrowed<TIn>(gate[g][off + nOff], nzbuf, zero, stripe.count, nCur,
                                                                 L.hid);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
            }

            if (live) {
                SingleLayerLstmVec::LoadStripe(buf, cAllGM[hOff + nOff], stripe.count, nCur, L.hid);
                AscendC::PipeBarrier<PIPE_ALL>();
                SingleLayerLstmVec::StoreStripeNarrowed<TIn>(outC[off + nOff], nbuf, buf, stripe.count, nCur, L.hid);
                AscendC::PipeBarrier<PIPE_ALL>();
                SingleLayerLstmVec::LoadStripe(buf, hAllGM[hOff + nOff], stripe.count, nCur, L.hid);
                AscendC::PipeBarrier<PIPE_ALL>();
                /* output_h and y contain the same rounded hidden state. */
                SingleLayerLstmVec::StoreStripeNarrowed<TIn>(outH[off + nOff], nbuf, buf, stripe.count, nCur, L.hid);
                AscendC::PipeBarrier<PIPE_ALL>();
                SingleLayerLstmVec::StoreStripeNarrowed<TIn>(outY[off + nOff], nbuf, buf, stripe.count, nCur, L.hid);
            } else {
                SingleLayerLstmVec::StoreStripeNarrowed<TIn>(outC[off + nOff], nzbuf, zero, stripe.count, nCur, L.hid);
                AscendC::PipeBarrier<PIPE_ALL>();
                SingleLayerLstmVec::StoreStripeNarrowed<TIn>(outH[off + nOff], nzbuf, zero, stripe.count, nCur, L.hid);
                AscendC::PipeBarrier<PIPE_ALL>();
                SingleLayerLstmVec::StoreStripeNarrowed<TIn>(outY[off + nOff], nzbuf, zero, stripe.count, nCur, L.hid);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
}

/* `tanhc` is written straight into its output by phase B, so only its tail needs clearing. */
template <typename TIn>
__aicore__ inline void ClearTanhcTail(const SingleLayerLstmCube::RowStripe& stripe, uint32_t blockBase,
                                      AscendC::GlobalTensor<TIn> outTanhc, const SingleLayerLstmFwd::Layout& L,
                                      uint32_t seqLenMax)
{
    if (stripe.count == 0 || seqLenMax >= L.steps) {
        return;
    }
    AscendC::LocalTensor<float> zero(AscendC::TPosition::VECCALC, L.t2, L.plane);
    AscendC::LocalTensor<TIn> nzbuf(AscendC::TPosition::VECCALC, L.nzero, L.plane);
    const uint32_t gRow = blockBase + stripe.base;
    /* Re-made rather than inherited from ScatterPlanes: that function returns early on its own
     * conditions, so `zero` holding zeros here would be an assumption about a caller. */
    AscendC::Duplicate(zero, 0.0f, L.plane);
    AscendC::PipeBarrier<PIPE_ALL>();
    for (uint32_t t = seqLenMax; t < L.steps; ++t) {
        const uint32_t off = (t * L.mAll + gRow) * L.hid;
        for (uint32_t ch = 0; ch < L.nChunks; ++ch) {
            const uint32_t nOff = ch * L.nChunk;
            const uint32_t nCur = (L.hid - nOff < L.nChunk) ? (L.hid - nOff) : L.nChunk;
            SingleLayerLstmVec::StoreStripeNarrowed<TIn>(outTanhc[off + nOff], nzbuf, zero, stripe.count, nCur, L.hid);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
}

} // namespace

/* The entry uses MIX_AIC_1_2; build-time SSBUF and core-ratio options must match. */
extern "C" __global__ __aicore__ void single_layer_lstm(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR init_h, GM_ADDR init_c,
                                                        GM_ADDR seq_length, GM_ADDR bias_hh, GM_ADDR y,
                                                        GM_ADDR output_h, GM_ADDR output_c, GM_ADDR out_i,
                                                        GM_ADDR out_j, GM_ADDR out_f, GM_ADDR out_o, GM_ADDR out_tanhc,
                                                        GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(SingleLayerLstmTilingData);
    GET_TILING_DATA_WITH_STRUCT(SingleLayerLstmTilingData, tilingData, tiling);
    AscendC::InitSocState();

    /* seq_length reaches the kernel through TilingData: the host resolved it, or fell back to T.
     * Its VALUE is not read here -- see SingleLayerLstmTilingFunc for why the host read is guarded. */
    (void)seq_length;

    const uint32_t B = tilingData.batch;
    const uint32_t I = tilingData.inputSize;
    const uint32_t H = tilingData.hiddenSize;
    const uint32_t T = tilingData.timeStep;
    const uint32_t mBlk = tilingData.rowsPerBlock;

    /* ascend950 core-index convention, NOT the Atlas A2 one: an AIV's GetBlockIdx() is ALREADY the
     * flat subcore index, so the A2 idiom blockIdx*subBlockNum + subBlockIdx yields {0,3,4,7,...}
     * and half the clusters silently process nothing. */
    uint32_t cluster;
    if ASCEND_IS_AIV {
        cluster = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
    } else {
        cluster = AscendC::GetBlockIdx();
    }
    /* Walk disjoint mBlk-row blocks. Do not exit before the grid-wide weight-widening barrier. */
    const uint32_t gridStride = tilingData.blockDim * mBlk;

    /* workspace is already the user area for this entry; tiling offsets are relative to it. */
    __gm__ float* ws = reinterpret_cast<__gm__ float*>(workspace);

    /* THE WORKSPACE IS FP32 WHATEVER THE CALLER'S WIDTH IS. igates, hAll, cAll and store carry the
     * recurrence's own numbers, not the caller's -- narrowing them would put the rounding the
     * outputs take once per element onto every intermediate as well. */
    using TIn = SingleLayerLstmIn;
    AscendC::GlobalTensor<TIn> xGM, wGM, biasGM, biasHhGM, h0GM, c0GM;
    AscendC::GlobalTensor<float> bGM, igGM, hAllGM, cAllGM, storeGM;
    AscendC::GlobalTensor<TIn> oY, oH, oC, oI, oJ, oF, oO, oTanhc;
    xGM.SetGlobalBuffer(reinterpret_cast<__gm__ TIn*>(x));
    biasGM.SetGlobalBuffer(reinterpret_cast<__gm__ TIn*>(b));
    const bool hasBiasHh = tilingData.hasBiasHh != 0;
    if (hasBiasHh) {
        biasHhGM.SetGlobalBuffer(reinterpret_cast<__gm__ TIn*>(bias_hh));
    }
    wGM.SetGlobalBuffer(reinterpret_cast<__gm__ TIn*>(w));
    h0GM.SetGlobalBuffer(reinterpret_cast<__gm__ TIn*>(init_h));
    c0GM.SetGlobalBuffer(reinterpret_cast<__gm__ TIn*>(init_c));
    igGM.SetGlobalBuffer(ws + tilingData.offIgates);
    hAllGM.SetGlobalBuffer(ws + tilingData.offHAll);
    cAllGM.SetGlobalBuffer(ws + tilingData.offCAll);
    storeGM.SetGlobalBuffer(ws + tilingData.offStore);
    oY.SetGlobalBuffer(reinterpret_cast<__gm__ TIn*>(y));
    oH.SetGlobalBuffer(reinterpret_cast<__gm__ TIn*>(output_h));
    oC.SetGlobalBuffer(reinterpret_cast<__gm__ TIn*>(output_c));
    oI.SetGlobalBuffer(reinterpret_cast<__gm__ TIn*>(out_i));
    oJ.SetGlobalBuffer(reinterpret_cast<__gm__ TIn*>(out_j));
    oF.SetGlobalBuffer(reinterpret_cast<__gm__ TIn*>(out_f));
    oO.SetGlobalBuffer(reinterpret_cast<__gm__ TIn*>(out_o));
    oTanhc.SetGlobalBuffer(reinterpret_cast<__gm__ TIn*>(out_tanhc));

    /* Cube operands are FP32. w[I:] is the recurrent [H,4H] slab of the fused weight. */
    AscendC::GlobalTensor<float> xF32, wF32, wRec;
    if constexpr (sizeof(TIn) == sizeof(float)) {
        xF32.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        bGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
        wF32.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(w));
    } else {
        xF32.SetGlobalBuffer(ws + tilingData.offXF32);
        bGM.SetGlobalBuffer(ws + tilingData.offBF32);
        wF32.SetGlobalBuffer(ws + tilingData.offWF32);
    }
    wRec = wF32[static_cast<size_t>(I) * SingleLayerLstmFwd::GATES * H]; // W_hh^T, rows [I, I+H)
    if (hasBiasHh) {
        bGM.SetGlobalBuffer(ws + tilingData.offBF32);
    }

    /* THE WIDENING PROLOGUE, AND THE ONE HANDSHAKE IT COSTS. At fp16 and bf16 neither slab the cube
     * reads exists yet and only a vector core can convert, so the AIVs build both images, publish
     * them with a grid barrier, and then signal the cube once. A separate hidden bias also needs
     * this publication at FP32. Both cores use the same guard to keep the flag counts equal. */
    if ASCEND_IS_AIC {
        if (sizeof(TIn) != sizeof(float) || hasBiasHh) {
            SingleLayerLstmSync::CubeWaitVec();
        }
        /* ONE BLOCK OF ROWS PER ITERATION, PHASE A AND PHASE B TOGETHER. They are partitioned on
         * exactly the same rows, so lifting phase A out of this loop would make it project rows
         * whose igates the block's phase B has not reached yet -- and igates is sized for the whole
         * batch, so that would be correct but would hold T*B*4H floats live for no gain. Keeping
         * the two adjacent also keeps the cross-core round count per block identical on both
         * halves, which is what the handshake needs. */
        for (uint32_t base = cluster * mBlk; base < B; base += gridStride) {
            const uint32_t mThis = SingleLayerLstmFwd::RowsOfBlock(B, mBlk, base);
            const SingleLayerLstmProj::Layout PL(mThis, tilingData.tChunk, tilingData.kChunk, tilingData.projNChunk, I,
                                                 H, B, T);
            /* sizeof(TIn) is what sizes the narrow staging planes and the W_hh^T widening chunk.
             * op_host passes the same number to SingleLayerLstmBudget::RecurrenceFits, so the shape
             * this kernel runs was accepted against THIS layout and not against a re-derivation of
             * it. */
            const SingleLayerLstmFwd::Layout L(B, H, T, mThis, static_cast<uint32_t>(sizeof(TIn)));
            /* The vector half computes phase A alone on this path, with no cross-core round, so
             * the cube must not run its own -- the handshake counts would diverge and the launch
             * would hang. The boundary round below still happens on both halves. */
            if (tilingData.projVec == 0) {
                SingleLayerLstmProj::CubeProject(xF32, wF32, bGM, base, PL);
            }
            SingleLayerLstmProj::CubeBoundary();
#ifndef SINGLE_LAYER_LSTM_ONLY_PHASE_A
            SingleLayerLstmFwd::CubeLoop<TIn>(wRec, hAllGM, base, L);
#endif
            /* ONE MORE ROUND AT THE END OF EVERY BLOCK, AND IT IS NOT BOOKKEEPING. The next block's
             * phase A drains into the SAME VECOUT planes this block's last phase B round handed to
             * the AIVs, and the cube reaches that drain without waiting -- its first drain of a
             * chunk is the one that skips the wait, because in a single-block launch nothing had
             * been handed over yet. With blocks, something has: the AIVs are still copying the last
             * gates out of VECOUT. This is the same write-after-read barrier CubeBoundary provides
             * between phase A and phase B, at the other seam. Paid on the last block too, where it
             * is one idle round, rather than made conditional on a count both cores would have to
             * agree on. */
            SingleLayerLstmProj::CubeBoundary();
        }
    }
    if ASCEND_IS_AIV {
        /* ONCE PER LAUNCH, NOT ONCE PER ROW BLOCK. The image is whole-grid: every cluster's
         * cube reads all of w, so widening it again per block would redo the same bytes -- and the grid barrier inside
         * would then have to be crossed a different number of times by clusters holding different block counts, which
         * is a hang rather than an inefficiency.
         *
         * blockDim and the cluster index come from TilingData, NOT from a Layout's b / m: the band
         * assignment partitions a flat element range by AIV index, and a short last block would
         * otherwise give that cluster a different partition and leave chunks unwritten. `WL` is
         * built from the full `mBlk` for the same reason -- it is the value op_host checked, and
         * every cluster must get the same staging offsets out of it. */
        const SingleLayerLstmFwd::Layout WL(B, H, T, mBlk, static_cast<uint32_t>(sizeof(TIn)));
        SingleLayerLstmFwd::WidenInputs<TIn>(wF32, wGM, (I + H) * SingleLayerLstmFwd::GATES * H, xF32, xGM, T * B * I,
                                             bGM, biasGM, biasHhGM, hasBiasHh, SingleLayerLstmFwd::GATES * H, cluster,
                                             tilingData.blockDim, WL);
        if (sizeof(TIn) != sizeof(float) || hasBiasHh) {
            SingleLayerLstmSync::VecSignalCube();
        }
        for (uint32_t base = cluster * mBlk; base < B; base += gridStride) {
            const uint32_t mThis = SingleLayerLstmFwd::RowsOfBlock(B, mBlk, base);
            const SingleLayerLstmProj::Layout PL(mThis, tilingData.tChunk, tilingData.kChunk, tilingData.projNChunk, I,
                                                 H, B, T);
            const SingleLayerLstmFwd::Layout L(B, H, T, mThis, static_cast<uint32_t>(sizeof(TIn)));
            if (tilingData.projVec != 0) {
                const SingleLayerLstmProj::VecLayout VL2(mThis, I, H, B, T, tilingData.projVecKTile,
                                                         AscendC::GetSubBlockIdx());
                SingleLayerLstmProj::VectorProjectDirect(xF32, wF32, bGM, igGM, base, VL2);
            } else {
                SingleLayerLstmProj::VectorProject(igGM, base, PL);
            }
            SingleLayerLstmProj::VectorBoundary();
#ifndef SINGLE_LAYER_LSTM_ONLY_PHASE_A
            const SingleLayerLstmCube::RowStripe stripe = SingleLayerLstmCube::DrainStripe(
                L.m, SingleLayerLstmFwd::SPLIT, AscendC::GetSubBlockIdx());
            /* wantCo is 1 unconditionally: `tanhc` is a declared output, and an output nobody writes
             * returns whatever was in the caller's buffer. */
            SingleLayerLstmFwd::VectorLoop<TIn>(stripe, base, igGM, h0GM, c0GM, hAllGM, cAllGM, storeGM, oTanhc, 1, L);
            AscendC::PipeBarrier<PIPE_ALL>();
            ScatterPlanes<TIn>(stripe, base, hAllGM, cAllGM, storeGM, oY, oH, oC, oI, oJ, oF, oO, L,
                               tilingData.seqLenMax);
            ClearTanhcTail<TIn>(stripe, base, oTanhc, L, tilingData.seqLenMax);
#endif
            /* The other half of the block-boundary round. It is AFTER the two epilogue phases on
             * purpose: what the cube must not overtake is this core's last read of VECOUT, and
             * ScatterPlanes reads VECCALC only -- but ordering the signal behind both of them costs
             * nothing and keeps the rule "the cube may touch UB again only once the AIVs have
             * finished with this block" true as written. */
            SingleLayerLstmProj::VectorBoundary();
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}
