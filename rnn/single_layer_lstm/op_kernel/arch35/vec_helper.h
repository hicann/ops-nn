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
 * \file vec_helper.h
 * \brief AIV-side half of the dav_3510 LSTM kernels: gate arithmetic and the L1 feedback edge.
 *
 * The split from cube_helper.h is by WHO ISSUES the instruction, which is how the kernels are
 * structured (`if ASCEND_IS_AIC` / `if ASCEND_IS_AIV`) -- note that UB -> L1 lives HERE despite
 * targeting L1, because CopyUbufToCbuf is AIV-issued (PIPE_MTE3).
 */
#ifndef OPS_RNN_SINGLE_LAYER_LSTM_VEC_HELPER_H
#define OPS_RNN_SINGLE_LAYER_LSTM_VEC_HELPER_H

#include "kernel_operator.h"
#include "single_layer_lstm_gate_math.h"
#include "cube_helper.h" // NzLayout, RowStripe, CUBE_BLOCK, C0<T>()

/* 命名空间用算子全名。它们原先叫 ch / vh / sh (cube / vector / sync helper)、fwd、proj --
 * 按范式取的短名, 读起来顺, 但摆在顶层是抢地盘: 同名的 fwd::Layout 在另一份 RNN 实现里也
 * 存在过, 两者一旦被同一次编译收进来就是 struct 重定义, 而报错只会说 "redefinition of
 * struct fwd::Layout", 不会告诉你是哪两个算子撞了。仓内的同类都用全名 (LstmGradRegbase、
 * ThnnFusedLstmCellNS), 这里照办。 */
namespace SingleLayerLstmVec {

/* Recurrence feedback: UB -> L1, placing NZ column blocks by hand.
 *
 * Not DataCopy(l1, ub, Nd2NzParams): that overload builds a compact NZ tile via TransND2NZ into a
 * stack buffer and issues one flat contiguous burst, so its column-block stride is `nValue` and
 * dstNzC0Stride never reaches the addressing -- it cannot write into a column-block slice of a
 * larger NZ tile, which is exactly what a feedback edge needs. It also asserts GetTPipePtr() !=
 * nullptr, unusable in this no-TPipe style. The flat overload goes straight to CopyUbufToCbuf, so
 * the layout is ours to state; only the outer stride differs, which is why NzLayout keeps
 * `rowsAligned` separate from `rows`.
 *
 * This is a direct hardware path only because of the SSBUF macros. DataCopyUB2L1Impl has two
 * branches behind one API: with KFC_C310_SSBUF == 1 or __MIX_CORE_AIC_RATION__ != 1 it is
 * CopyUbufToCbuf, one on-chip MTE3 instruction; otherwise it allocates GM, copies UB -> GM and RPCs
 * the AIC to do GM -> L1. The kernel CMake passes -D__MIX_CORE_AIC_RATION__=2, so this takes the
 * direct path -- drop that macro and the identical source line becomes a GM round trip.
 *
 * Each parameter has exactly one job, because conflating two is a real bug: passing the whole tile's
 * layout makes `srcCols` come out as K instead of H, so the loop runs 6 column blocks instead of 4
 * with the wrong srcStride, row 0 right and every later row wrong.
 *
 *   dstTile       geometry of the whole destination L1 tile (its c0 and its parent row stride)
 *   srcCols       width of the source plane in UB -- not dstTile.cols
 *   dstColBlock0  first destination NZ column block to write
 *   stripe        the row range this core owns, and the same stripe the drain handed it. L1 has no
 *                 write arbitration, so that identity is the only thing keeping the AIVs apart.
 *   srcPitch      row pitch of the source plane in elements; 0 means the same as srcCols, the
 *                 compact case every h_t feedback uses. Separate because a caller may own a column
 *                 range of a wider plane -- the W_hh^T widening reads gate g out of a [rows, 4H]
 *                 slab, pitch 4H and width H.
 *
 * `src` is [stripe.count, srcPitch] row-major in UB, of which the first `srcCols` columns are read.
 * `srcCols` must be a multiple of c0: cp.srcStride counts 32-byte blocks, so a pitch that is not a
 * whole number of them is not expressible, and a CeilDiv does not patch it -- the stride comes out
 * wrong for every row, not just the last block. The M axis has no such constraint, which is why the
 * tail policy is asymmetric: batch is handled exactly in-kernel while I and H are required to be
 * multiples of c0 by op_host. */
template <typename T>
__aicore__ inline void FeedbackToL1(const AscendC::LocalTensor<T>& l1, const AscendC::LocalTensor<T>& src,
                                    const SingleLayerLstmCube::NzLayout& dstTile, uint32_t srcCols,
                                    uint32_t dstColBlock0, const SingleLayerLstmCube::RowStripe& stripe,
                                    uint32_t srcPitch = 0)
{
    const uint32_t c0 = dstTile.c0;
    const uint32_t pitch = (srcPitch == 0) ? srcCols : srcPitch;
    const uint32_t srcBlocks = srcCols / c0;
    const uint32_t pitchBlocks = pitch / c0;
    for (uint32_t j = 0; j < srcBlocks; ++j) {
        AscendC::DataCopyParams cp;
        cp.blockCount = static_cast<uint16_t>(stripe.count);   // one 32B block per row I own
        cp.blockLen = 1;                                       // c0 elements == 32B exactly
        cp.srcStride = static_cast<uint16_t>(pitchBlocks - 1); // skip this row's other SOURCE blocks
        cp.dstStride = 0;                                      // rows are contiguous in an NZ block
        AscendC::DataCopy(l1[dstTile.ColBlock(dstColBlock0 + j) + stripe.base * c0], src[j * c0], cp);
    }
}

/* The dtype boundary, vector side. Everything this kernel computes is fp32; the caller's tensors may
 * be fp16 or bf16, and these three helpers are the only places the vector half crosses that width.
 * The cube crosses it differently and does not come through here.
 *
 * DataCopyPad, not DataCopy, on the narrow side: its length is in bytes, so a run of `count` narrow
 * elements need not be a whole number of 32-byte blocks. Insisting that it were would raise
 * hidden_size's alignment rule from 8 to 16 for the narrow dtypes only.
 *
 * The two directions round differently. Widening is exact, so CAST_NONE names the absence of a
 * decision; narrowing has to choose, and CAST_NONE there truncates -- a half-ulp bias on every
 * output, in one direction, on every timestep. CAST_RINT is what the rest of rnn/ narrows with. At
 * fp32 all three collapse to the plain copy they wrap. */

/* GM (caller's width) -> UB, as fp32. `stage` is scratch and is unread at fp32. */
template <typename T>
__aicore__ inline void WidenFromGm(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<T>& stage,
                                   const AscendC::GlobalTensor<T>& gm, uint32_t count)
{
    if constexpr (sizeof(T) == sizeof(float)) {
        AscendC::DataCopy(dst, gm, count);
    } else {
        AscendC::DataCopyExtParams cp(1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0);
        AscendC::DataCopyPadExtParams<T> pp(true, 0, 0, 0);
        AscendC::DataCopyPad(stage, gm, cp, pp);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Cast(dst, stage, AscendC::RoundMode::CAST_NONE, count);
    }
}

/* The plane a store should read from: `stage`, holding the narrowed values, at fp16 and bf16; `src`
 * itself at fp32, where there is nothing to narrow.
 *
 * Returning the tensor rather than storing here is what lets a CONSTANT plane be narrowed once and
 * written on many timesteps -- the zero tail of every output does exactly that, and re-narrowing a
 * constant per write would be pure vector traffic. The caller must order the Cast against its own
 * store; every caller in this operator does it with the same PipeBarrier<PIPE_ALL> it uses
 * everywhere else. */
template <typename T>
__aicore__ inline AscendC::LocalTensor<T> Narrowed(const AscendC::LocalTensor<T>& stage,
                                                   const AscendC::LocalTensor<float>& src, uint32_t count)
{
    if constexpr (sizeof(T) == sizeof(float)) {
        (void)stage;
        (void)count;
        return src.template ReinterpretCast<T>();
    } else {
        AscendC::Cast(stage, src, AscendC::RoundMode::CAST_RINT, count);
        return stage;
    }
}

/* UB (already at the caller's width) -> GM. */
template <typename T>
__aicore__ inline void StoreToGm(const AscendC::GlobalTensor<T>& gm, const AscendC::LocalTensor<T>& src, uint32_t count)
{
    if constexpr (sizeof(T) == sizeof(float)) {
        AscendC::DataCopy(gm, src, count);
    } else {
        AscendC::DataCopyExtParams cp(1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0);
        AscendC::DataCopyPad(gm, src, cp);
    }
}

/* A column range of a [rows, pitch] block in GM, against a compact [rows, cols] plane in UB. These
 * exist because the vector half is tiled on the H axis: a plane it holds is `cols` wide while the GM
 * block it came from is `pitch` wide. cols must be a multiple of 8, which every chunk here is --
 * nChunk is a multiple of 16 and the tail is hidden_size minus a multiple of 16, hidden_size itself
 * being a multiple of 8 -- so at fp32 every row length and gap is a whole number of 32-byte blocks,
 * which is what DataCopyParams states them in. */
__aicore__ inline void LoadStripe(const AscendC::LocalTensor<float>& dst, const AscendC::GlobalTensor<float>& gm,
                                  uint32_t rows, uint32_t cols, uint32_t pitch)
{
    /* ONE BURST WHEN THE CHUNK IS THE WHOLE ROW, which is the common case: at hidden_size up to 128
     * there is a single column chunk and this whole family degenerates to the flat copies that were
     * here before the H axis was tiled. Expressing it as `rows` bursts of zero stride is not the
     * same instruction and was measured 3% slower end to end on T=32 B=64 I=128 H=128. */
    if (cols == pitch) {
        AscendC::DataCopy(dst, gm, rows * cols);
        return;
    }
    AscendC::DataCopyParams cp;
    cp.blockCount = static_cast<uint16_t>(rows);
    cp.blockLen = static_cast<uint16_t>(cols * sizeof(float) / AscendC::ONE_BLK_SIZE);
    cp.srcStride = static_cast<uint16_t>((pitch - cols) * sizeof(float) / AscendC::ONE_BLK_SIZE);
    cp.dstStride = 0;
    AscendC::DataCopy(dst, gm, cp);
}

__aicore__ inline void StoreStripe(const AscendC::GlobalTensor<float>& gm, const AscendC::LocalTensor<float>& src,
                                   uint32_t rows, uint32_t cols, uint32_t pitch)
{
    if (cols == pitch) {
        AscendC::DataCopy(gm, src, rows * cols); // see LoadStripe
        return;
    }
    AscendC::DataCopyParams cp;
    cp.blockCount = static_cast<uint16_t>(rows);
    cp.blockLen = static_cast<uint16_t>(cols * sizeof(float) / AscendC::ONE_BLK_SIZE);
    cp.srcStride = 0;
    cp.dstStride = static_cast<uint16_t>((pitch - cols) * sizeof(float) / AscendC::ONE_BLK_SIZE);
    AscendC::DataCopy(gm, src, cp);
}

/* The mirror of StoreStripeNarrowed: read a column range at the caller's width and widen it.
 * Same two arms, same reason -- see below. GM-side gaps are in BYTES here and UB-side gaps in
 * 32-byte blocks, which is the opposite assignment from the store; both are the hardware's. */
template <typename T>
__aicore__ inline void LoadStripeWidened(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<T>& stage,
                                         const AscendC::GlobalTensor<T>& gm, uint32_t rows, uint32_t cols,
                                         uint32_t pitch)
{
    constexpr uint32_t NARROW_ROW_ALIGN = AscendC::ONE_BLK_SIZE / 2;
    if constexpr (sizeof(T) == sizeof(float)) {
        (void)stage;
        LoadStripe(dst, gm.template ReinterpretCast<float>(), rows, cols, pitch);
    } else if (cols == pitch) {
        WidenFromGm<T>(dst, stage, gm, rows * cols); // one burst -- see LoadStripe
    } else if (cols % NARROW_ROW_ALIGN == 0) {
        AscendC::DataCopyExtParams cp(static_cast<uint16_t>(rows), cols * static_cast<uint32_t>(sizeof(T)),
                                      (pitch - cols) * static_cast<uint32_t>(sizeof(T)), 0, 0);
        AscendC::DataCopyPadExtParams<T> pp(true, 0, 0, 0);
        AscendC::DataCopyPad(stage, gm, cp, pp);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Cast(dst, stage, AscendC::RoundMode::CAST_NONE, rows * cols);
    } else {
        AscendC::DataCopyExtParams cp(1, cols * static_cast<uint32_t>(sizeof(T)), 0, 0, 0);
        AscendC::DataCopyPadExtParams<T> pp(true, 0, 0, 0);
        for (uint32_t r = 0; r < rows; ++r) {
            AscendC::DataCopyPad(stage, gm[r * pitch], cp, pp);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Cast(dst[r * cols], stage, AscendC::RoundMode::CAST_NONE, cols);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
}

/* The same store at the caller's width: narrow `src` through `stage`, then write the column range.
 *
 * WHY THE TWO ARMS, AND WHY THE SLOW ONE IS NOT AVOIDABLE. DataCopyPad states the UB-side gap
 * between rows in 32-BYTE BLOCKS, so it can only skip a row of `cols` 2-byte elements when cols is
 * a multiple of 16 -- and hidden_size is only guaranteed a multiple of 8, so the LAST column chunk
 * can be exactly 8 short of that. Its rows then go one at a time, each narrowed into the staging
 * plane at offset 0 so both the fp32 source (cols*4 is a multiple of 32) and the narrow source (the
 * plane's own base) are aligned. Only the instruction count differs between the arms; the bytes
 * written are the same. At fp32 `Narrowed` hands back `src` itself and this is one plain store. */
template <typename T>
__aicore__ inline void StoreStripeNarrowed(const AscendC::GlobalTensor<T>& gm, const AscendC::LocalTensor<T>& stage,
                                           const AscendC::LocalTensor<float>& src, uint32_t rows, uint32_t cols,
                                           uint32_t pitch)
{
    constexpr uint32_t NARROW_ROW_ALIGN = AscendC::ONE_BLK_SIZE / 2; // 16 two-byte elements per block
    if constexpr (sizeof(T) == sizeof(float)) {
        (void)stage;
        StoreStripe(gm.template ReinterpretCast<float>(), src, rows, cols, pitch);
    } else if (cols == pitch) {
        const AscendC::LocalTensor<T> n = Narrowed<T>(stage, src, rows * cols);
        AscendC::PipeBarrier<PIPE_ALL>();
        StoreToGm<T>(gm, n, rows * cols); // one burst -- see LoadStripe
    } else if (cols % NARROW_ROW_ALIGN == 0) {
        const AscendC::LocalTensor<T> n = Narrowed<T>(stage, src, rows * cols);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams cp(static_cast<uint16_t>(rows), cols * static_cast<uint32_t>(sizeof(T)), 0,
                                      (pitch - cols) * static_cast<uint32_t>(sizeof(T)), 0);
        AscendC::DataCopyPad(gm, n, cp);
    } else {
        for (uint32_t r = 0; r < rows; ++r) {
            const AscendC::LocalTensor<T> n = Narrowed<T>(stage, src[r * cols], cols);
            AscendC::PipeBarrier<PIPE_ALL>();
            StoreToGm<T>(gm[r * pitch], n, cols);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
}

} // namespace SingleLayerLstmVec

#endif // OPS_RNN_SINGLE_LAYER_LSTM_VEC_HELPER_H
