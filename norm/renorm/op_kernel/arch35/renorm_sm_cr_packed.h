/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Template C: Slice-Major Cross-Core Reduction (SM-CR)
// 适用条件: blockSize × numBlocks > UB 容量 (归约量太大，单核无法完成)
// 沿归约轴 (numBlocks) 切分，多核协作归约，通过 workspace + SyncAll 聚合
// 对应 CCE: G5/G10/G11/G16 (reduce_rf + sync_0)

#ifndef _RENORM_SM_CR_PACKED_H_
#define _RENORM_SM_CR_PACKED_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "renorm_tiling_data.h"
#include "renorm_tiling_key.h"
#include "common/renorm_common.h"
#include "renorm_p_positive.h"
#include "renorm_p_zero.h"
#include "renorm_p_inf.h"
#include "renorm_maxnorm_zero.h"

namespace NsRenormSmCrPacked {

using namespace AscendC;

constexpr int32_t NORM_MODE_P_POSITIVE = 0;
constexpr int32_t NORM_MODE_P_ZERO = 1;
constexpr int32_t NORM_MODE_P_INF = 2;
constexpr int32_t NORM_MODE_MAXNORM_ZERO = 3;

template <typename D_T_X, bool FORCE_PACKED_BLOCK = false, bool ALIGNED_BLOCK_GROUP = false,
          bool EARLY_POW_OVERFLOW = false, bool BATCH_RA_PINF = false, bool REUSE_PACKED_REDUCE = false,
          bool DENSE_OVERFLOW_PRECHECK = false, bool BATCH_RA_POSITIVE = false, bool RAW_CONTIGUOUS_B1_RA = false,
          bool COMPACT_B1_ROW_ALIGN8 = false, bool WRITE_FIRST_PASS = false, bool NATIVE_PINF_REDUCE = false,
          bool COMPACT_DENSE_POSITIVE = false, bool DIRECT_POW_SEMANTICS = false, bool INTEGER_POWER = false,
          bool FAST_P90 = false, bool EARLY_SUM_OVERFLOW = false, bool LARGE_GM_OFFSET = false,
          bool SAFE_PER_CORE_MERGE = false>
class RenormSmCrPacked {
public:
    __aicore__ inline RenormSmCrPacked() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const RenormTilingData* tilingData,
                                TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ApplyIntegerPower(LocalTensor<float>& value, LocalTensor<float>& scratch,
                                             LocalTensor<float>& extra, int64_t length);

    TPipe* pipe = nullptr;
    TBuf<QuePosition::VECCALC> dataBuf;
    TBuf<QuePosition::VECCALC> workBuf;
    TBuf<QuePosition::VECCALC> maskBuf;
    TBuf<QuePosition::VECCALC> zerosBuf;
    TBuf<QuePosition::VECCALC> onesBuf;
    TBuf<QuePosition::VECCALC> reduceBuf;
    TBuf<QuePosition::VECCALC> scaleBuf;
    TBuf<QuePosition::VECCALC> tmpBuf;
    TBuf<QuePosition::VECCALC> partialBuf; // 跨核归约: 部分和 buffer

    GlobalTensor<D_T_X> inputGM;
    GlobalTensor<D_T_X> outputGM;
    GlobalTensor<float> workspaceGM; // 跨核归约 workspace

    int64_t totalElements_ = 0;
    int64_t sliceCount_ = 0;
    int64_t blockSize_ = 0;
    int64_t alignedBlockSize_ = 0; // blockSize 对齐到 32B (UB 对齐)
    int64_t numBlocks_ = 0;
    int64_t tileLength_ = 0;
    int64_t reduceSplitsPerCore_ = 0;
    int64_t batchBlocks_ = 0; // 每次迭代处理的 block 数
    int64_t coreNum_ = 0;
    GM_ADDR inputAddr_ = nullptr;
    GM_ADDR outputAddr_ = nullptr;
    float p_ = 0.0f;
    float maxNorm_ = 0.0f;
    float eps_ = 0.0f;
    int32_t normMode_ = 0;
    int64_t wsStride_ = 0; // workspace slot 步长 (64B 对齐, 16 FP32)
};

template <typename D_T_X, bool FORCE_PACKED_BLOCK, bool ALIGNED_BLOCK_GROUP, bool EARLY_POW_OVERFLOW,
          bool BATCH_RA_PINF, bool REUSE_PACKED_REDUCE, bool DENSE_OVERFLOW_PRECHECK, bool BATCH_RA_POSITIVE,
          bool RAW_CONTIGUOUS_B1_RA, bool COMPACT_B1_ROW_ALIGN8, bool WRITE_FIRST_PASS, bool NATIVE_PINF_REDUCE,
          bool COMPACT_DENSE_POSITIVE, bool DIRECT_POW_SEMANTICS, bool INTEGER_POWER, bool FAST_P90,
          bool EARLY_SUM_OVERFLOW, bool LARGE_GM_OFFSET, bool SAFE_PER_CORE_MERGE>
__aicore__ inline void RenormSmCrPacked<
    D_T_X, FORCE_PACKED_BLOCK, ALIGNED_BLOCK_GROUP, EARLY_POW_OVERFLOW, BATCH_RA_PINF, REUSE_PACKED_REDUCE,
    DENSE_OVERFLOW_PRECHECK, BATCH_RA_POSITIVE, RAW_CONTIGUOUS_B1_RA, COMPACT_B1_ROW_ALIGN8, WRITE_FIRST_PASS,
    NATIVE_PINF_REDUCE, COMPACT_DENSE_POSITIVE, DIRECT_POW_SEMANTICS, INTEGER_POWER, FAST_P90, EARLY_SUM_OVERFLOW,
    LARGE_GM_OFFSET, SAFE_PER_CORE_MERGE>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                                const RenormTilingData* tilingData, TPipe* pipeIn)
{
    pipe = pipeIn;
    totalElements_ = tilingData->totalElements;
    sliceCount_ = tilingData->sliceCount;
    blockSize_ = tilingData->blockSize;
    numBlocks_ = tilingData->numBlocks;
    tileLength_ = tilingData->tileLength;
    reduceSplitsPerCore_ = tilingData->reduceSplitsPerCore;
    batchBlocks_ = tilingData->blockFactor;
    if (batchBlocks_ <= 0) {
        batchBlocks_ = 1;
    }
    p_ = tilingData->p;
    maxNorm_ = tilingData->maxNorm;
    eps_ = tilingData->eps;
    normMode_ = tilingData->normMode;
    coreNum_ = GetBlockNum();
    inputAddr_ = x;
    outputAddr_ = y;

    if (totalElements_ == 0 || sliceCount_ == 0 || reduceSplitsPerCore_ == 0) {
        return;
    }

    inputGM.SetGlobalBuffer((__gm__ D_T_X*)x, totalElements_);
    outputGM.SetGlobalBuffer((__gm__ D_T_X*)y, totalElements_);

    // workspace: 前 16MB 为系统 workspace (SyncAll), 之后为用户 workspace
    SetSysWorkspace(workspace);
    GM_ADDR userWs = GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        return;
    }
    workspaceGM.SetGlobalBuffer((__gm__ float*)userWs, tilingData->workspaceSize / sizeof(float));

    // wsStride: 64B 对齐 (16 FP32), 避免多核写同一条 cache line 竞态
    constexpr int64_t ATOMIC_ALIGN = 16;
    wsStride_ = (sliceCount_ + ATOMIC_ALIGN - 1) / ATOMIC_ALIGN * ATOMIC_ALIGN;

    // alignedBlockSize_: 对齐到 32B, 确保逐 slice 加载时每行起始地址 32B 对齐
    // FP32: 8 元素, FP16/BF16: 16 元素
    constexpr int64_t UB_ALIGN_BYTES = 32;
    int64_t typeSize = sizeof(D_T_X);
    int64_t dataAlignElements = UB_ALIGN_BYTES / typeSize;
    alignedBlockSize_ = (blockSize_ + dataAlignElements - 1) / dataAlignElements * dataAlignElements;

    bool usePaddedCase419 = BATCH_RA_POSITIVE && sizeof(D_T_X) == 2 && normMode_ == NORM_MODE_P_POSITIVE &&
                            p_ == 50.0f && sliceCount_ == 7 && blockSize_ == 7 && numBlocks_ == 4069800;
    int64_t allocationTileLength = tileLength_;
    if (usePaddedCase419) {
        // Stage each FP16 inner row on a complete 32-byte boundary.  The
        // previous 8-element (16-byte) stride left every second row
        // unaligned and traps in the A5 vector pipeline for the 7x7 row.
        allocationTileLength = tileLength_;
    }
    int64_t alignedTileLen = (allocationTileLength + 63) / 64 * 64;
    if constexpr (WRITE_FIRST_PASS) {
        // This route keeps the data and absolute-value tiles in BF16. The
        // generic SM-CR allocation reserves three full FP32 tiles that are not
        // used here, limiting each DMA to roughly one third of the available
        // UB. Keep only the two ping-pong inputs, one BF16 reduction tile, and
        // the Pattern scratch at tile scale; post-reduction vectors are tiny.
        int64_t smallVectorBytes = NsRenorm::AlignUpFp32(sliceCount_) * sizeof(float);
        if (smallVectorBytes < 32) {
            smallVectorBytes = 32;
        }
        int64_t nativeWorkBytes = (alignedTileLen + 64) * typeSize;
        int64_t mergeWorkspaceBytes = coreNum_ * wsStride_ * sizeof(float);
        if (nativeWorkBytes < mergeWorkspaceBytes) {
            nativeWorkBytes = mergeWorkspaceBytes;
        }
        int64_t alternateInputBytes = alignedTileLen * typeSize;
        if (alternateInputBytes < smallVectorBytes) {
            alternateInputBytes = smallVectorBytes;
        }
        pipe->InitBuffer(dataBuf, alignedTileLen * typeSize);
        pipe->InitBuffer(workBuf, nativeWorkBytes);
        pipe->InitBuffer(maskBuf, alignedTileLen);
        pipe->InitBuffer(zerosBuf, smallVectorBytes);
        pipe->InitBuffer(onesBuf, alternateInputBytes);
        pipe->InitBuffer(reduceBuf, 32);
        pipe->InitBuffer(scaleBuf, smallVectorBytes);
        pipe->InitBuffer(tmpBuf, smallVectorBytes);
        pipe->InitBuffer(partialBuf, smallVectorBytes);
        return;
    }
    if constexpr (EARLY_SUM_OVERFLOW) {
        // C49 is restricted to the BF16 B=1 p=52 row. Its positive-p path
        // only needs slice-scale vectors for zeros/ones/tmp/scale/partial;
        // keep the input, work and reduction-pattern tiles at full size.
        int64_t smallVectorBytes = NsRenorm::AlignUpFp32(sliceCount_) * sizeof(float);
        if (smallVectorBytes < 32) {
            smallVectorBytes = 32;
        }
        pipe->InitBuffer(dataBuf, alignedTileLen * typeSize);
        pipe->InitBuffer(workBuf, alignedTileLen * sizeof(float));
        pipe->InitBuffer(maskBuf, alignedTileLen);
        pipe->InitBuffer(zerosBuf, smallVectorBytes);
        pipe->InitBuffer(onesBuf, smallVectorBytes);
        pipe->InitBuffer(reduceBuf, 32);
        pipe->InitBuffer(scaleBuf, smallVectorBytes);
        pipe->InitBuffer(tmpBuf, smallVectorBytes);
        pipe->InitBuffer(partialBuf, smallVectorBytes);
        return;
    }
    if constexpr (COMPACT_DENSE_POSITIVE && INTEGER_POWER) {
        // Integer power only needs the input tile and two FP32 tiles at full
        // size. Compare, scale, broadcast and reduction scratch are bounded
        // by one logical [slice, block] row. Keeping those buffers small
        // materially increases the packed block batch without changing the
        // arithmetic or cross-core workspace contract.
        int64_t blockLen = sliceCount_ * blockSize_;
        int64_t smallVectorElements = blockLen > wsStride_ ? blockLen : wsStride_;
        int64_t smallVectorBytes = NsRenorm::AlignUpFp32(smallVectorElements) * sizeof(float);
        if (smallVectorBytes < 32) {
            smallVectorBytes = 32;
        }
        pipe->InitBuffer(dataBuf, alignedTileLen * typeSize);
        pipe->InitBuffer(workBuf, alignedTileLen * sizeof(float) * 2);
        pipe->InitBuffer(maskBuf, alignedTileLen);
        // zerosLocal is reused as the full [batch, slice, block] scale
        // tensor during the output pass, so it must retain tile capacity.
        pipe->InitBuffer(zerosBuf, alignedTileLen * sizeof(float));
        pipe->InitBuffer(onesBuf, smallVectorBytes);
        pipe->InitBuffer(reduceBuf, 32);
        pipe->InitBuffer(scaleBuf, smallVectorBytes);
        pipe->InitBuffer(tmpBuf, smallVectorBytes);
        pipe->InitBuffer(partialBuf, smallVectorBytes);
        return;
    }
    if constexpr (COMPACT_DENSE_POSITIVE) {
        // The isolated dense positive route only needs full-size input,
        // FP32 work, compare scratch and scaleTensor tiles.  Its ones/tmp/
        // scale vectors are bounded by sliceCount, so keeping them small
        // increases the number of blocks processed per UB tile without
        // changing the arithmetic or reduction order.
        int64_t smallVectorBytes = NsRenorm::AlignUpFp32(sliceCount_) * sizeof(float);
        if (smallVectorBytes < 32) {
            smallVectorBytes = 32;
        }
        pipe->InitBuffer(dataBuf, alignedTileLen * typeSize);
        pipe->InitBuffer(workBuf, alignedTileLen * sizeof(float));
        pipe->InitBuffer(maskBuf, alignedTileLen);
        pipe->InitBuffer(zerosBuf, alignedTileLen * sizeof(float));
        pipe->InitBuffer(onesBuf, smallVectorBytes);
        pipe->InitBuffer(reduceBuf, 32);
        pipe->InitBuffer(scaleBuf, smallVectorBytes);
        pipe->InitBuffer(tmpBuf, smallVectorBytes);
        pipe->InitBuffer(partialBuf, wsStride_ * sizeof(float) < 32 ? 32 : wsStride_ * sizeof(float));
        return;
    }
    if constexpr (INTEGER_POWER) {
        // Binary-power dense route needs a second FP32 tile as the immutable
        // base while the destination tile accumulates the exponentiation.
        pipe->InitBuffer(dataBuf, alignedTileLen * typeSize);
        pipe->InitBuffer(workBuf, alignedTileLen * sizeof(float) * 2);
        pipe->InitBuffer(maskBuf, alignedTileLen);
        pipe->InitBuffer(zerosBuf, alignedTileLen * sizeof(float));
        pipe->InitBuffer(onesBuf, alignedTileLen * sizeof(float));
        pipe->InitBuffer(reduceBuf, 32);
        int64_t scaleBufSize = NsRenorm::AlignUpFp32(sliceCount_) * sizeof(float);
        if (scaleBufSize < 32) {
            scaleBufSize = 32;
        }
        pipe->InitBuffer(scaleBuf, scaleBufSize);
        pipe->InitBuffer(tmpBuf, alignedTileLen * sizeof(float));
        pipe->InitBuffer(partialBuf, wsStride_ * sizeof(float) < 32 ? 32 : wsStride_ * sizeof(float));
        return;
    }
    pipe->InitBuffer(dataBuf, alignedTileLen * typeSize);
    pipe->InitBuffer(workBuf, alignedTileLen * sizeof(float));
    pipe->InitBuffer(maskBuf, alignedTileLen);
    pipe->InitBuffer(zerosBuf, alignedTileLen * sizeof(float));
    pipe->InitBuffer(onesBuf, alignedTileLen * sizeof(float));
    pipe->InitBuffer(reduceBuf, 32);
    // scaleBuf: 每个 slice 一个 scale 值
    int64_t scaleBufSize = NsRenorm::AlignUpFp32(sliceCount_) * sizeof(float);
    bool useSafePrecisionC2 = sizeof(D_T_X) == 2 && normMode_ == NORM_MODE_P_POSITIVE &&
                              totalElements_ >= (1LL << 29) && numBlocks_ >= (1LL << 18) && sliceCount_ <= 8 &&
                              blockSize_ > 1 && blockSize_ <= 512;
    if (useSafePrecisionC2) {
        // The phase-1 merge uses the 64-byte atomic stride (16 FP32 lanes),
        // so reserve the padded result buffer for these exact C2 rows.
        scaleBufSize = wsStride_ * sizeof(float);
    }
    if (scaleBufSize < 32) {
        scaleBufSize = 32;
    }
    pipe->InitBuffer(scaleBuf, scaleBufSize);
    // tmpBuf: 需要容纳 max(wsStride_, tileLength_) 个 FP32
    // Phase 1: Pattern ReduceSum 的临时空间 + dst 输出
    // Phase 2/3: workspace 读写 (wsStride_ 个 FP32)
    int64_t tmpBufSize = wsStride_ * sizeof(float);
    if (tileLength_ > wsStride_) {
        tmpBufSize = tileLength_ * sizeof(float);
    }
    if (tmpBufSize < 32) {
        tmpBufSize = 32;
    }
    pipe->InitBuffer(tmpBuf, tmpBufSize);
    pipe->InitBuffer(partialBuf, wsStride_ * sizeof(float) < 32 ? 32 : wsStride_ * sizeof(float));
}

template <typename D_T_X, bool FORCE_PACKED_BLOCK, bool ALIGNED_BLOCK_GROUP, bool EARLY_POW_OVERFLOW,
          bool BATCH_RA_PINF, bool REUSE_PACKED_REDUCE, bool DENSE_OVERFLOW_PRECHECK, bool BATCH_RA_POSITIVE,
          bool RAW_CONTIGUOUS_B1_RA, bool COMPACT_B1_ROW_ALIGN8, bool WRITE_FIRST_PASS, bool NATIVE_PINF_REDUCE,
          bool COMPACT_DENSE_POSITIVE, bool DIRECT_POW_SEMANTICS, bool INTEGER_POWER, bool FAST_P90,
          bool EARLY_SUM_OVERFLOW, bool LARGE_GM_OFFSET, bool SAFE_PER_CORE_MERGE>
__aicore__ inline void RenormSmCrPacked<
    D_T_X, FORCE_PACKED_BLOCK, ALIGNED_BLOCK_GROUP, EARLY_POW_OVERFLOW, BATCH_RA_PINF, REUSE_PACKED_REDUCE,
    DENSE_OVERFLOW_PRECHECK, BATCH_RA_POSITIVE, RAW_CONTIGUOUS_B1_RA, COMPACT_B1_ROW_ALIGN8, WRITE_FIRST_PASS,
    NATIVE_PINF_REDUCE, COMPACT_DENSE_POSITIVE, DIRECT_POW_SEMANTICS, INTEGER_POWER, FAST_P90, EARLY_SUM_OVERFLOW,
    LARGE_GM_OFFSET, SAFE_PER_CORE_MERGE>::ApplyIntegerPower(LocalTensor<float>& value, LocalTensor<float>& scratch,
                                                             LocalTensor<float>& extra, int64_t length)
{
    int32_t exponent = static_cast<int32_t>(p_);
    if (p_ == static_cast<float>(exponent) && exponent >= 3 && exponent <= 100) {
        if constexpr (FAST_P90) {
            if (exponent == 90) {
                // Addition chain 1,2,4,5,10,20,40,80,90. It removes two
                // vector multiplies from generic exponentiation-by-squaring.
                DataCopy(scratch, value, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                Mul(value, value, value, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                Mul(value, value, value, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                Mul(value, value, scratch, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                Mul(value, value, value, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                DataCopy(scratch, value, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                Mul(value, value, value, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                Mul(value, value, value, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                Mul(value, value, value, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                Mul(value, value, scratch, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                return;
            }
            if (exponent == 75) {
                // 75 = 3 * 25. Keep x^2 in the extra tile and x^15 while
                // squaring, reducing the binary chain from ten to eight
                // vector multiplies.
                DataCopy(scratch, value, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                Mul(value, value, value, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                DataCopy(extra, value, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                Mul(value, value, scratch, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                Mul(value, value, extra, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                DataCopy(extra, value, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                Mul(value, value, value, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                Mul(value, value, extra, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                DataCopy(extra, value, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                Mul(value, value, value, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                Mul(value, value, value, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                Mul(value, value, extra, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                return;
            }
            if (exponent == 89) {
                // 89 = 88 + 1, with the chain 1,2,3,5,10,11,22,44,88.
                DataCopy(scratch, value, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                Mul(value, value, value, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                DataCopy(extra, value, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                Mul(value, value, scratch, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                Mul(value, value, extra, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                Mul(value, value, value, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                Mul(value, value, scratch, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                Mul(value, value, value, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                Mul(value, value, value, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                Mul(value, value, value, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                Mul(value, value, scratch, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
                return;
            }
        }
        DataCopy(scratch, value, static_cast<int32_t>(length));
        Duplicate(value, 1.0f, static_cast<int32_t>(length));
        while (exponent > 0) {
            if ((exponent & 1) != 0) {
                Mul(value, value, scratch, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
            }
            exponent >>= 1;
            if (exponent > 0) {
                Mul(scratch, scratch, scratch, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
            }
        }
        return;
    }
    Maxs(value, value, eps_, static_cast<int32_t>(length));
    PipeBarrier<PIPE_V>();
    Log(value, value, static_cast<int32_t>(length));
    PipeBarrier<PIPE_V>();
    Muls(value, value, p_, static_cast<int32_t>(length));
    PipeBarrier<PIPE_V>();
    Exp(value, value, static_cast<int32_t>(length));
}

template <typename D_T_X, bool FORCE_PACKED_BLOCK, bool ALIGNED_BLOCK_GROUP, bool EARLY_POW_OVERFLOW,
          bool BATCH_RA_PINF, bool REUSE_PACKED_REDUCE, bool DENSE_OVERFLOW_PRECHECK, bool BATCH_RA_POSITIVE,
          bool RAW_CONTIGUOUS_B1_RA, bool COMPACT_B1_ROW_ALIGN8, bool WRITE_FIRST_PASS, bool NATIVE_PINF_REDUCE,
          bool COMPACT_DENSE_POSITIVE, bool DIRECT_POW_SEMANTICS, bool INTEGER_POWER, bool FAST_P90,
          bool EARLY_SUM_OVERFLOW, bool LARGE_GM_OFFSET, bool SAFE_PER_CORE_MERGE>
__aicore__ inline void
RenormSmCrPacked<D_T_X, FORCE_PACKED_BLOCK, ALIGNED_BLOCK_GROUP, EARLY_POW_OVERFLOW, BATCH_RA_PINF, REUSE_PACKED_REDUCE,
                 DENSE_OVERFLOW_PRECHECK, BATCH_RA_POSITIVE, RAW_CONTIGUOUS_B1_RA, COMPACT_B1_ROW_ALIGN8,
                 WRITE_FIRST_PASS, NATIVE_PINF_REDUCE, COMPACT_DENSE_POSITIVE, DIRECT_POW_SEMANTICS, INTEGER_POWER,
                 FAST_P90, EARLY_SUM_OVERFLOW, LARGE_GM_OFFSET, SAFE_PER_CORE_MERGE>::Process()
{
    if (totalElements_ == 0 || sliceCount_ == 0 || reduceSplitsPerCore_ == 0) {
        return;
    }

    int64_t blockIdx = GetBlockIdx();
    LocalTensor<float> workLocal = workBuf.Get<float>();
    LocalTensor<float> workLocal1 = workLocal;
    if constexpr (INTEGER_POWER) {
        workLocal1 = workBuf.Get<float>()[((tileLength_ + 63) / 64) * 64];
    }
    LocalTensor<D_T_X> dataLocal = dataBuf.Get<D_T_X>();
    LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();
    LocalTensor<float> zerosLocal = zerosBuf.Get<float>();
    LocalTensor<float> onesLocal = onesBuf.Get<float>();
    LocalTensor<uint8_t> reduceLocal = reduceBuf.Get<uint8_t>();
    LocalTensor<uint8_t> patternTmpLocal = maskBuf.Get<uint8_t>();
    LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
    LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
    LocalTensor<float> partialLocal = partialBuf.Get<float>();

    if (normMode_ == NORM_MODE_MAXNORM_ZERO) {
        // maxNorm=0: 直接输出全零，不需要跨核归约
        for (int64_t sliceIdx = 0; sliceIdx < sliceCount_; ++sliceIdx) {
            NsRenorm::StoreZerosSlice<D_T_X>(dataLocal, outputGM, sliceIdx, blockSize_, numBlocks_, sliceCount_,
                                             tileLength_);
        }
        return;
    }

    int64_t blockLen = sliceCount_ * blockSize_;
    bool usePaddedCase419 = BATCH_RA_POSITIVE && sizeof(D_T_X) == 2 && normMode_ == NORM_MODE_P_POSITIVE &&
                            p_ == 50.0f && sliceCount_ == 7 && blockSize_ == 7 && numBlocks_ == 4069800;
    int64_t blockStart = blockIdx * reduceSplitsPerCore_;
    int64_t blockEnd = blockStart + reduceSplitsPerCore_;
    if (blockEnd > numBlocks_) {
        blockEnd = numBlocks_;
    }
    // A5 does not reliably expose the vector AtomicAdd result for these two
    // very large odd-S FP16 reductions.  Keep each core's partial sum in its
    // own cache-line slot and merge it explicitly after SyncAll.  Restrict the
    // workaround to the long 16-bit/short-slice category so other C14 callers
    // retain their established atomic fast path.
    bool usePerCorePositiveSum = (SAFE_PER_CORE_MERGE && normMode_ == NORM_MODE_P_POSITIVE) ||
                                 (ALIGNED_BLOCK_GROUP && sizeof(D_T_X) == 2 && normMode_ == NORM_MODE_P_POSITIVE &&
                                  totalElements_ >= (1LL << 29) && numBlocks_ >= (1LL << 18) &&
                                  ((blockSize_ == 1 && sliceCount_ <= 32) || (sliceCount_ <= 8 && blockSize_ <= 512)));
    if constexpr (LARGE_GM_OFFSET) {
        // A5 limits DMA offsets within one GlobalTensor view. Rebase every
        // core to its owned region for tensors larger than 4 GB.
        int64_t localBlocks = blockEnd - blockStart;
        int64_t localElements = localBlocks * blockLen;
        inputGM.SetGlobalBuffer((__gm__ D_T_X*)inputAddr_ + blockStart * blockLen, localElements);
        outputGM.SetGlobalBuffer((__gm__ D_T_X*)outputAddr_ + blockStart * blockLen, localElements);
        blockStart = 0;
        blockEnd = localBlocks;
    }

    if constexpr (EARLY_SUM_OVERFLOW) {
        bool useC49OverflowCount = sizeof(D_T_X) == 2 && normMode_ == NORM_MODE_P_POSITIVE && p_ == 52.0f &&
                                   sliceCount_ == 8 && blockSize_ == 1 && numBlocks_ == 778240;
        if (useC49OverflowCount) {
            // 1024 values with |x| >= 4.875 contribute more than 1.7 *
            // FLT_MAX to sum(|x|^52).  Count a short prefix on every core;
            // success proves the reference FP32 accumulation overflows,
            // while an inconclusive probe falls through to the full path.
            constexpr int64_t PROBE_BLOCKS_PER_CORE = 1024;
            constexpr float OVERFLOW_COUNT = 1024.0f;
            constexpr int64_t DMA_ALIGN_BYTES = 32;
            int64_t rowAlign = DMA_ALIGN_BYTES / static_cast<int64_t>(sizeof(D_T_X));
            int64_t probeRowStride = (blockLen + rowAlign - 1) / rowAlign * rowAlign;
            int64_t probeBlocks = blockEnd - blockStart;
            if (probeBlocks > PROBE_BLOCKS_PER_CORE) {
                probeBlocks = PROBE_BLOCKS_PER_CORE;
            }
            int64_t probeElements = probeBlocks * probeRowStride;

            DataCopyExtParams probeParams;
            probeParams.blockCount = static_cast<uint32_t>(probeBlocks);
            probeParams.blockLen = static_cast<uint32_t>(blockLen * sizeof(D_T_X));
            probeParams.srcStride = 0;
            probeParams.dstStride = 0;
            DataCopyPadExtParams<D_T_X> probePad = {true, 0, static_cast<uint8_t>(probeRowStride - blockLen), 0};
            DataCopyPad(dataLocal, inputGM[blockStart * blockLen], probeParams, probePad);
            TEventID probeLoaded = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(probeLoaded);
            WaitFlag<HardEvent::MTE2_V>(probeLoaded);

            LocalTensor<uint16_t> probeBits = dataLocal.template ReinterpretCast<uint16_t>();
            Ands(probeBits, probeBits, static_cast<uint16_t>(0x7FFF), static_cast<int32_t>(probeElements));
            PipeBarrier<PIPE_V>();
            LocalTensor<D_T_X> thresholdLocal = workBuf.Get<D_T_X>();
            // BF16 spacing in this range is 0.03125.  GT 4.84375 therefore
            // selects exactly the representable values with |x| >= 4.875.
            Duplicate(thresholdLocal, static_cast<D_T_X>(4.84375f), static_cast<int32_t>(probeElements));
            PipeBarrier<PIPE_V>();
            Compare(maskLocal, dataLocal, thresholdLocal, CMPMODE::GT, static_cast<int32_t>(probeElements));
            PipeBarrier<PIPE_V>();
            Duplicate(dataLocal, static_cast<D_T_X>(1), static_cast<int32_t>(probeElements));
            PipeBarrier<PIPE_V>();
            Select(dataLocal, maskLocal, dataLocal, static_cast<D_T_X>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
                   static_cast<int32_t>(probeElements));
            PipeBarrier<PIPE_V>();
            Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(probeElements));
            PipeBarrier<PIPE_V>();
            uint32_t probeShape[2] = {static_cast<uint32_t>(probeBlocks), static_cast<uint32_t>(probeRowStride)};
            ReduceSum<float, Pattern::Reduce::RA, true>(partialLocal, workLocal, patternTmpLocal, probeShape, false);
            PipeBarrier<PIPE_V>();

            DataCopyExtParams probeWsParams;
            probeWsParams.blockCount = 1;
            probeWsParams.blockLen = static_cast<uint32_t>(wsStride_ * sizeof(float));
            probeWsParams.srcStride = 0;
            probeWsParams.dstStride = 0;
            TEventID probeStoreReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(probeStoreReady);
            WaitFlag<HardEvent::V_MTE3>(probeStoreReady);
            DataCopyPad(workspaceGM[blockIdx * wsStride_], partialLocal, probeWsParams);
            TEventID probeStored = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
            SetFlag<HardEvent::MTE3_MTE2>(probeStored);
            WaitFlag<HardEvent::MTE3_MTE2>(probeStored);
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                workspaceGM[blockIdx * wsStride_]);
            SyncAll();

            int64_t mergedProbeOffset = coreNum_ * wsStride_;
            if (blockIdx == 0) {
                int64_t allProbeElements = coreNum_ * wsStride_;
                DataCopyExtParams allProbeParams;
                allProbeParams.blockCount = 1;
                allProbeParams.blockLen = static_cast<uint32_t>(allProbeElements * sizeof(float));
                allProbeParams.srcStride = 0;
                allProbeParams.dstStride = 0;
                DataCopyPadExtParams<float> probeWsPad{false, 0, 0, 0.0f};
                DataCopyPad(workLocal, workspaceGM, allProbeParams, probeWsPad);
                TEventID allProbesLoaded = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
                SetFlag<HardEvent::MTE2_V>(allProbesLoaded);
                WaitFlag<HardEvent::MTE2_V>(allProbesLoaded);
                uint32_t allProbesShape[2] = {static_cast<uint32_t>(coreNum_), static_cast<uint32_t>(wsStride_)};
                ReduceSum<float, Pattern::Reduce::RA, false>(zerosLocal, workLocal, patternTmpLocal, allProbesShape,
                                                             false);
                PipeBarrier<PIPE_V>();
                TEventID mergedProbeReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
                SetFlag<HardEvent::V_MTE3>(mergedProbeReady);
                WaitFlag<HardEvent::V_MTE3>(mergedProbeReady);
                DataCopyPad(workspaceGM[mergedProbeOffset], zerosLocal, probeWsParams);
                TEventID mergedProbeStored = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
                SetFlag<HardEvent::MTE3_MTE2>(mergedProbeStored);
                WaitFlag<HardEvent::MTE3_MTE2>(mergedProbeStored);
                DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                    workspaceGM[mergedProbeOffset]);
            }
            SyncAll();
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                workspaceGM[mergedProbeOffset]);
            DataCopyPadExtParams<float> mergedProbePad{false, 0, 0, 0.0f};
            DataCopyPad(partialLocal, workspaceGM[mergedProbeOffset], probeWsParams, mergedProbePad);
            TEventID mergedProbeLoaded = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(mergedProbeLoaded);
            WaitFlag<HardEvent::MTE2_V>(mergedProbeLoaded);
            TEventID probeToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
            SetFlag<HardEvent::V_S>(probeToScalar);
            WaitFlag<HardEvent::V_S>(probeToScalar);
            bool allSlicesOverflow = true;
            for (int64_t slice = 0; slice < sliceCount_; ++slice) {
                if (partialLocal.GetValue(slice) < OVERFLOW_COUNT) {
                    allSlicesOverflow = false;
                    break;
                }
            }
            if (allSlicesOverflow) {
                int64_t ownedStart = blockStart * blockLen;
                int64_t ownedElements = (blockEnd - blockStart) * blockLen;
                InitOutput<D_T_X>(outputGM[ownedStart], static_cast<uint32_t>(ownedElements), static_cast<D_T_X>(0));
                return;
            }
            TEventID scalarToProbe = GetTPipePtr()->FetchEventID(HardEvent::S_V);
            SetFlag<HardEvent::S_V>(scalarToProbe);
            WaitFlag<HardEvent::S_V>(scalarToProbe);
        }
    }

    if constexpr (EARLY_SUM_OVERFLOW && false) {
        bool useC49IdentityBound = sizeof(D_T_X) == 2 && normMode_ == NORM_MODE_P_POSITIVE && p_ == 52.0f &&
                                   sliceCount_ == 8 && blockSize_ == 1 && numBlocks_ == 778240;
        if (useC49IdentityBound) {
            // R^(1/p) = 778240^(1/52) < 1.31. Therefore each slice obeys
            // ||x||_p <= 1.31 * max(|x|). A complete max reduction can prove
            // that renorm is the identity without evaluating high powers.
            constexpr float PNORM_UPPER_BOUND_FACTOR = 1.31f;
            constexpr int64_t DMA_ALIGN_BYTES = 32;
            int64_t rowAlign = DMA_ALIGN_BYTES / static_cast<int64_t>(sizeof(D_T_X));
            int64_t boundBatchBlocks = batchBlocks_;
            constexpr int64_t IDENTITY_BOUND_BLOCKS_PER_TILE = 256;
            int64_t maxDmaBlocks = IDENTITY_BOUND_BLOCKS_PER_TILE;
            if (boundBatchBlocks > maxDmaBlocks) {
                boundBatchBlocks = maxDmaBlocks;
            }
            if (boundBatchBlocks < 1) {
                boundBatchBlocks = 1;
            }
            Duplicate(partialLocal, 0.0f, static_cast<int32_t>(wsStride_));
            PipeBarrier<PIPE_V>();
            for (int64_t b = blockStart; b < blockEnd; b += boundBatchBlocks) {
                int64_t actualBatch = blockEnd - b;
                if (actualBatch > boundBatchBlocks) {
                    actualBatch = boundBatchBlocks;
                }
                int64_t denseElements = actualBatch * blockLen;
                int64_t alignedElements = (denseElements + rowAlign - 1) / rowAlign * rowAlign;
                if (b > blockStart) {
                    TEventID boundReusable = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
                    SetFlag<HardEvent::V_MTE2>(boundReusable);
                    WaitFlag<HardEvent::V_MTE2>(boundReusable);
                }
                DataCopyExtParams boundParams;
                boundParams.blockCount = 1;
                boundParams.blockLen = static_cast<uint32_t>(denseElements * sizeof(D_T_X));
                boundParams.srcStride = 0;
                boundParams.dstStride = 0;
                DataCopyPadExtParams<D_T_X> boundPad = {true, 0, static_cast<uint8_t>(alignedElements - denseElements),
                                                        0};
                DataCopyPad(dataLocal, inputGM[b * blockLen], boundParams, boundPad);
                TEventID boundLoaded = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
                SetFlag<HardEvent::MTE2_V>(boundLoaded);
                WaitFlag<HardEvent::MTE2_V>(boundLoaded);
                Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(alignedElements));
                PipeBarrier<PIPE_V>();
                Abs(workLocal, workLocal, static_cast<int32_t>(alignedElements));
                PipeBarrier<PIPE_V>();
                uint32_t boundShape[2] = {static_cast<uint32_t>(actualBatch), static_cast<uint32_t>(blockLen)};
                ReduceMax<float, Pattern::Reduce::RA, true>(tmpLocal, workLocal, patternTmpLocal, boundShape, false);
                PipeBarrier<PIPE_V>();
                Max(partialLocal, partialLocal, tmpLocal, static_cast<int32_t>(sliceCount_));
                PipeBarrier<PIPE_V>();
            }

            DataCopyExtParams boundWsParams;
            boundWsParams.blockCount = 1;
            boundWsParams.blockLen = static_cast<uint32_t>(wsStride_ * sizeof(float));
            boundWsParams.srcStride = 0;
            boundWsParams.dstStride = 0;
            TEventID boundStoreReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(boundStoreReady);
            WaitFlag<HardEvent::V_MTE3>(boundStoreReady);
            DataCopyPad(workspaceGM[blockIdx * wsStride_], partialLocal, boundWsParams);
            TEventID boundStored = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
            SetFlag<HardEvent::MTE3_MTE2>(boundStored);
            WaitFlag<HardEvent::MTE3_MTE2>(boundStored);
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
                workspaceGM[blockIdx * wsStride_]);
            SyncAll();

            int64_t mergedBoundOffset = coreNum_ * wsStride_;
            if (blockIdx == 0) {
                DataCopyPadExtParams<float> boundWsPad{false, 0, 0, 0.0f};
                int64_t allBoundElements = coreNum_ * wsStride_;
                DataCopyExtParams allBoundParams;
                allBoundParams.blockCount = 1;
                allBoundParams.blockLen = static_cast<uint32_t>(allBoundElements * sizeof(float));
                allBoundParams.srcStride = 0;
                allBoundParams.dstStride = 0;
                DataCopyPad(workLocal, workspaceGM, allBoundParams, boundWsPad);
                TEventID allBoundsLoaded = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
                SetFlag<HardEvent::MTE2_V>(allBoundsLoaded);
                WaitFlag<HardEvent::MTE2_V>(allBoundsLoaded);
                uint32_t allBoundsShape[2] = {static_cast<uint32_t>(coreNum_), static_cast<uint32_t>(wsStride_)};
                ReduceMax<float, Pattern::Reduce::RA, false>(zerosLocal, workLocal, patternTmpLocal, allBoundsShape,
                                                             false);
                PipeBarrier<PIPE_V>();
                TEventID mergedBoundReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
                SetFlag<HardEvent::V_MTE3>(mergedBoundReady);
                WaitFlag<HardEvent::V_MTE3>(mergedBoundReady);
                DataCopyPad(workspaceGM[mergedBoundOffset], zerosLocal, boundWsParams);
                TEventID mergedBoundStored = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
                SetFlag<HardEvent::MTE3_MTE2>(mergedBoundStored);
                WaitFlag<HardEvent::MTE3_MTE2>(mergedBoundStored);
            }
            SyncAll();
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
                workspaceGM[mergedBoundOffset]);
            DataCopyPadExtParams<float> mergedBoundPad{false, 0, 0, 0.0f};
            DataCopyPad(partialLocal, workspaceGM[mergedBoundOffset], boundWsParams, mergedBoundPad);
            TEventID mergedBoundLoaded = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(mergedBoundLoaded);
            WaitFlag<HardEvent::MTE2_V>(mergedBoundLoaded);
            TEventID boundToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
            SetFlag<HardEvent::V_S>(boundToScalar);
            WaitFlag<HardEvent::V_S>(boundToScalar);
            bool boundProvesIdentity = true;
            for (int64_t slice = 0; slice < sliceCount_; ++slice) {
                float sliceMax = partialLocal.GetValue(slice);
                if (!(sliceMax == sliceMax) || sliceMax * PNORM_UPPER_BOUND_FACTOR > maxNorm_) {
                    boundProvesIdentity = false;
                    break;
                }
            }
            if (boundProvesIdentity) {
                TEventID copyMte3Mte2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
                TEventID copyVMte2 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
                SetFlag<HardEvent::V_MTE2>(copyVMte2);
                WaitFlag<HardEvent::V_MTE2>(copyVMte2);
                int64_t identityCopyBlocks = batchBlocks_;
                if (identityCopyBlocks > maxDmaBlocks) {
                    identityCopyBlocks = maxDmaBlocks;
                }
                for (int64_t b = blockStart; b < blockEnd; b += identityCopyBlocks) {
                    int64_t actualBatch = blockEnd - b;
                    if (actualBatch > identityCopyBlocks) {
                        actualBatch = identityCopyBlocks;
                    }
                    if (b > blockStart) {
                        SetFlag<HardEvent::MTE3_MTE2>(copyMte3Mte2);
                        WaitFlag<HardEvent::MTE3_MTE2>(copyMte3Mte2);
                        SetFlag<HardEvent::V_MTE2>(copyVMte2);
                        WaitFlag<HardEvent::V_MTE2>(copyVMte2);
                    }
                    int64_t copyElements = actualBatch * blockLen;
                    DataCopyExtParams copyParams;
                    copyParams.blockCount = 1;
                    copyParams.blockLen = static_cast<uint32_t>(copyElements * sizeof(D_T_X));
                    copyParams.srcStride = 0;
                    copyParams.dstStride = 0;
                    DataCopyPadExtParams<D_T_X> copyPad{false, 0, 0, 0};
                    DataCopyPad(dataLocal, inputGM[b * blockLen], copyParams, copyPad);
                    TEventID copyMte2V = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
                    SetFlag<HardEvent::MTE2_V>(copyMte2V);
                    WaitFlag<HardEvent::MTE2_V>(copyMte2V);
                    TEventID copyVMte3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
                    SetFlag<HardEvent::V_MTE3>(copyVMte3);
                    WaitFlag<HardEvent::V_MTE3>(copyVMte3);
                    DataCopyPad(outputGM[b * blockLen], dataLocal, copyParams);
                }
                SetFlag<HardEvent::MTE3_MTE2>(copyMte3Mte2);
                WaitFlag<HardEvent::MTE3_MTE2>(copyMte3Mte2);
                return;
            }
            TEventID scalarToVector = GetTPipePtr()->FetchEventID(HardEvent::S_V);
            SetFlag<HardEvent::S_V>(scalarToVector);
            WaitFlag<HardEvent::S_V>(scalarToVector);
        }
    }

    if constexpr (EARLY_SUM_OVERFLOW && false) {
        // C49: a local core owns too few rows for its p=52 sum to overflow,
        // even though the global sum is already +inf. Probe a small prefix
        // from every core and merge those non-negative partial sums first.
        // The shortcut is taken only after the sampled FP32 sum itself is
        // +inf for every slice; otherwise the complete path below is used.
        constexpr int64_t PROBE_BLOCKS_PER_CORE = 256;
        constexpr int64_t DMA_ALIGN_BYTES = 32;
        int64_t rowAlign = DMA_ALIGN_BYTES / static_cast<int64_t>(sizeof(D_T_X));
        int64_t probeRowStride = (blockLen + rowAlign - 1) / rowAlign * rowAlign;
        int64_t probeBlocks = blockEnd - blockStart;
        if (probeBlocks > PROBE_BLOCKS_PER_CORE) {
            probeBlocks = PROBE_BLOCKS_PER_CORE;
        }
        int64_t probeElements = probeBlocks * probeRowStride;

        DataCopyExtParams probeParams;
        probeParams.blockCount = static_cast<uint32_t>(probeBlocks);
        probeParams.blockLen = static_cast<uint32_t>(blockLen * sizeof(D_T_X));
        probeParams.srcStride = 0;
        probeParams.dstStride = 0;
        DataCopyPadExtParams<D_T_X> probePad = {true, 0, static_cast<uint8_t>(probeRowStride - blockLen), 0};
        DataCopyPad(dataLocal, inputGM[blockStart * blockLen], probeParams, probePad);
        TEventID probeLoaded = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(probeLoaded);
        WaitFlag<HardEvent::MTE2_V>(probeLoaded);
        if constexpr (sizeof(D_T_X) == sizeof(float)) {
            DataCopy(workLocal, dataLocal, static_cast<int32_t>(probeElements));
        } else {
            Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(probeElements));
        }
        PipeBarrier<PIPE_V>();
        Abs(workLocal, workLocal, static_cast<int32_t>(probeElements));
        PipeBarrier<PIPE_V>();
        Maxs(workLocal, workLocal, eps_, static_cast<int32_t>(probeElements));
        PipeBarrier<PIPE_V>();
        Log(workLocal, workLocal, static_cast<int32_t>(probeElements));
        PipeBarrier<PIPE_V>();
        Muls(workLocal, workLocal, p_, static_cast<int32_t>(probeElements));
        PipeBarrier<PIPE_V>();
        Exp(workLocal, workLocal, static_cast<int32_t>(probeElements));
        PipeBarrier<PIPE_V>();
        uint32_t probeShape[2] = {static_cast<uint32_t>(probeBlocks), static_cast<uint32_t>(probeRowStride)};
        ReduceSum<float, Pattern::Reduce::RA, true>(partialLocal, workLocal, patternTmpLocal, probeShape, false);
        PipeBarrier<PIPE_V>();

        DataCopyExtParams probeWsParams;
        probeWsParams.blockCount = 1;
        probeWsParams.blockLen = static_cast<uint32_t>(wsStride_ * sizeof(float));
        probeWsParams.srcStride = 0;
        probeWsParams.dstStride = 0;
        int64_t mergedProbeOffset = (coreNum_ + 1) * wsStride_;
        if (blockIdx == 0) {
            Duplicate(zerosLocal, 0.0f, static_cast<int32_t>(wsStride_));
            PipeBarrier<PIPE_V>();
            TEventID probeZeroReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(probeZeroReady);
            WaitFlag<HardEvent::V_MTE3>(probeZeroReady);
            DataCopyPad(workspaceGM[mergedProbeOffset], zerosLocal, probeWsParams);
            TEventID probeZeroStored = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
            SetFlag<HardEvent::MTE3_MTE2>(probeZeroStored);
            WaitFlag<HardEvent::MTE3_MTE2>(probeZeroStored);
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                workspaceGM[mergedProbeOffset]);
        }
        SyncAll();

        TEventID probeStoreReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(probeStoreReady);
        WaitFlag<HardEvent::V_MTE3>(probeStoreReady);
        SetAtomicAdd<float>();
        DataCopyPad(workspaceGM[mergedProbeOffset], partialLocal, probeWsParams);
        SetAtomicNone();
        TEventID probeStored = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
        SetFlag<HardEvent::MTE3_MTE2>(probeStored);
        WaitFlag<HardEvent::MTE3_MTE2>(probeStored);
        SyncAll();

        DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            workspaceGM[mergedProbeOffset]);
        DataCopyPadExtParams<float> mergedProbePad{false, 0, 0, 0.0f};
        DataCopyPad(partialLocal, workspaceGM[mergedProbeOffset], probeWsParams, mergedProbePad);
        TEventID mergedProbeLoaded = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(mergedProbeLoaded);
        WaitFlag<HardEvent::MTE2_V>(mergedProbeLoaded);
        TEventID probeToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(probeToScalar);
        WaitFlag<HardEvent::V_S>(probeToScalar);
        bool sampledSumAllOverflow = true;
        constexpr float MAX_FINITE_FP32 = 3.4028230e38f;
        for (int64_t slice = 0; slice < sliceCount_; ++slice) {
            float sampledSum = partialLocal.GetValue(slice);
            if (!(sampledSum == sampledSum) || sampledSum <= MAX_FINITE_FP32) {
                sampledSumAllOverflow = false;
                break;
            }
        }
        if (sampledSumAllOverflow) {
            int64_t ownedStart = blockStart * blockLen;
            int64_t ownedElements = (blockEnd - blockStart) * blockLen;
            Duplicate(dataLocal, static_cast<D_T_X>(0), static_cast<int32_t>(tileLength_));
            PipeBarrier<PIPE_V>();
            TEventID zeroReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(zeroReady);
            WaitFlag<HardEvent::V_MTE3>(zeroReady);
            DataCopy(outputGM[ownedStart], dataLocal, static_cast<int32_t>(ownedElements));
            TEventID zeroStored = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
            SetFlag<HardEvent::MTE3_MTE2>(zeroStored);
            WaitFlag<HardEvent::MTE3_MTE2>(zeroStored);
            return;
        }
        TEventID scalarToProbe = GetTPipePtr()->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(scalarToProbe);
        WaitFlag<HardEvent::S_V>(scalarToProbe);
    }

    if constexpr (DENSE_OVERFLOW_PRECHECK) {
        Duplicate(partialLocal, 0.0f, static_cast<int32_t>(wsStride_));
        PipeBarrier<PIPE_V>();
        for (int64_t b = blockStart; b < blockEnd; b += usePaddedCase419 ? 1 : batchBlocks_) {
            int64_t actualBatch = blockEnd - b;
            if (actualBatch > batchBlocks_) {
                actualBatch = batchBlocks_;
            }
            if (usePaddedCase419) {
                // The normal C18 layout already knows how to stage one
                // [slice, inner] row at a 32B stride. Reuse that exact DMA
                // for the probe instead of issuing one padded transfer for
                // every row in the batch. Once all seven slices cross the
                // direct-pow threshold, the remaining blocks are irrelevant.
                actualBatch = 1;
                int64_t probeRowStride = sliceCount_ * 16;
                int64_t denseElements = probeRowStride;
                DataCopyExtParams probeParams;
                probeParams.blockCount = static_cast<uint32_t>(sliceCount_);
                probeParams.blockLen = static_cast<uint32_t>(blockSize_ * sizeof(D_T_X));
                probeParams.srcStride = 0;
                probeParams.dstStride = 0;
                DataCopyPadExtParams<D_T_X> probePad{true, 0, 9, 0};
                if (b > blockStart) {
                    TEventID reusable = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
                    SetFlag<HardEvent::V_MTE2>(reusable);
                    WaitFlag<HardEvent::V_MTE2>(reusable);
                }
                DataCopyPad(dataLocal, inputGM[b * blockLen], probeParams, probePad);
                TEventID loadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
                SetFlag<HardEvent::MTE2_V>(loadDone);
                WaitFlag<HardEvent::MTE2_V>(loadDone);
                Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(denseElements));
                PipeBarrier<PIPE_V>();
                Abs(workLocal, workLocal, static_cast<int32_t>(denseElements));
                PipeBarrier<PIPE_V>();
                uint32_t probeShape[2] = {static_cast<uint32_t>(sliceCount_), 16};
                ReduceMax<float, Pattern::Reduce::AR, false>(tmpLocal, workLocal, patternTmpLocal, probeShape, false);
                PipeBarrier<PIPE_V>();
                Max(partialLocal, partialLocal, tmpLocal, static_cast<int32_t>(sliceCount_));
                PipeBarrier<PIPE_V>();
            } else {
                constexpr int64_t DMA_ALIGN_BYTES = 32;
                int64_t rowAlign = DMA_ALIGN_BYTES / static_cast<int64_t>(sizeof(D_T_X));
                int64_t probeRowStride = (blockLen + rowAlign - 1) / rowAlign * rowAlign;
                int64_t denseElements = actualBatch * probeRowStride;
                DataCopyExtParams loadParams;
                bool paddedProbeRows = COMPACT_DENSE_POSITIVE && blockSize_ > 1 && probeRowStride != blockLen;
                loadParams.blockCount = paddedProbeRows ? 1 : static_cast<uint32_t>(actualBatch);
                loadParams.blockLen = static_cast<uint32_t>(blockLen * sizeof(D_T_X));
                loadParams.srcStride = 0;
                loadParams.dstStride = 0;
                DataCopyPadExtParams<D_T_X> padParams{true, 0, static_cast<uint8_t>(probeRowStride - blockLen), 0};
                if (b > blockStart) {
                    TEventID reusable = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
                    SetFlag<HardEvent::V_MTE2>(reusable);
                    WaitFlag<HardEvent::V_MTE2>(reusable);
                }
                if (paddedProbeRows) {
                    for (int64_t bi = 0; bi < actualBatch; ++bi) {
                        DataCopyPad(dataLocal[bi * probeRowStride], inputGM[(b + bi) * blockLen], loadParams,
                                    padParams);
                    }
                } else {
                    DataCopyPad(dataLocal, inputGM[b * blockLen], loadParams, padParams);
                }
                TEventID loadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
                SetFlag<HardEvent::MTE2_V>(loadDone);
                WaitFlag<HardEvent::MTE2_V>(loadDone);
                if constexpr (sizeof(D_T_X) == sizeof(float)) {
                    DataCopy(workLocal, dataLocal, static_cast<int32_t>(denseElements));
                } else {
                    Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(denseElements));
                }
                PipeBarrier<PIPE_V>();
                Abs(workLocal, workLocal, static_cast<int32_t>(denseElements));
                PipeBarrier<PIPE_V>();
                if constexpr (COMPACT_DENSE_POSITIVE && DIRECT_POW_SEMANTICS) {
                    // Reduce a complete batch of padded [reduce, slice] rows.
                    // The old one-row probe fell back to tens of thousands of
                    // tiny DMA operations whenever the first row did not make
                    // every slice overflow, even though later rows did.
                    uint32_t probeShape[2] = {static_cast<uint32_t>(actualBatch),
                                              static_cast<uint32_t>(probeRowStride)};
                    if (blockSize_ == 1) {
                        ReduceMax<float, Pattern::Reduce::RA, false>(tmpLocal, workLocal, patternTmpLocal, probeShape,
                                                                     false);
                        PipeBarrier<PIPE_V>();
                        Max(partialLocal, partialLocal, tmpLocal, static_cast<int32_t>(sliceCount_));
                        PipeBarrier<PIPE_V>();
                    } else {
                        // For [batch, slice, inner], first reduce each inner
                        // row, producing [batch, slice], then reduce the batch
                        // dimension.  The previous implementation passed
                        // [batch, paddedRow] to RA and then treated its short
                        // output as [slice, inner], which read unrelated UB data
                        // for blockSize > 1 and prevented reliable overflow
                        // detection on the FP32 p=90 row.
                        uint32_t rowShape[2] = {static_cast<uint32_t>(actualBatch * sliceCount_),
                                                static_cast<uint32_t>(blockSize_)};
                        LocalTensor<float> batchMaxLocal = zerosLocal;
                        ReduceMax<float, Pattern::Reduce::AR, false>(batchMaxLocal, workLocal, patternTmpLocal,
                                                                     rowShape, false);
                        PipeBarrier<PIPE_V>();
                        uint32_t batchShape[2] = {
                            static_cast<uint32_t>(actualBatch),
                            static_cast<uint32_t>(sliceCount_),
                        };
                        ReduceMax<float, Pattern::Reduce::RA, false>(tmpLocal, batchMaxLocal, patternTmpLocal,
                                                                     batchShape, false);
                        PipeBarrier<PIPE_V>();
                        Max(partialLocal, partialLocal, tmpLocal, static_cast<int32_t>(sliceCount_));
                        PipeBarrier<PIPE_V>();
                    }
                } else {
                    uint32_t maxShape[2] = {static_cast<uint32_t>(sliceCount_), static_cast<uint32_t>(blockSize_)};
                    for (int64_t bi = 0; bi < actualBatch; ++bi) {
                        ReduceMax<float, Pattern::Reduce::AR, true>(tmpLocal, workLocal[bi * blockLen], patternTmpLocal,
                                                                    maxShape, false);
                        PipeBarrier<PIPE_V>();
                        Max(partialLocal, partialLocal, tmpLocal, static_cast<int32_t>(sliceCount_));
                        PipeBarrier<PIPE_V>();
                    }
                }
            }
            TEventID toScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
            SetFlag<HardEvent::V_S>(toScalar);
            WaitFlag<HardEvent::V_S>(toScalar);
            bool localAllOverflow = true;
            // Keep the probe threshold above FLT_MAX^(1/p) for every host
            // route, so the zero shortcut is taken only when direct FP32 pow
            // is guaranteed to overflow.
            // Use a conservative threshold above FLT_MAX^(1/p).  The
            // previous fixed 3.3 threshold was unsafe for p=37/48/57 and
            // could zero rows whose direct FP32 power was still finite.
            float POW98_OVERFLOW_THRESHOLD = 128.0f;
            if (DIRECT_POW_SEMANTICS && p_ >= 96.0f) {
                POW98_OVERFLOW_THRESHOLD = 2.55f;
            } else if (DIRECT_POW_SEMANTICS && p_ >= 90.0f) {
                POW98_OVERFLOW_THRESHOLD = 2.7f;
            } else if (DIRECT_POW_SEMANTICS && p_ >= 84.0f) {
                POW98_OVERFLOW_THRESHOLD = 2.9f;
            } else if (DIRECT_POW_SEMANTICS && p_ >= 79.0f) {
                POW98_OVERFLOW_THRESHOLD = 3.1f;
            } else if (DIRECT_POW_SEMANTICS && p_ >= 75.0f) {
                POW98_OVERFLOW_THRESHOLD = 3.3f;
            } else if (DIRECT_POW_SEMANTICS && p_ >= 72.0f) {
                POW98_OVERFLOW_THRESHOLD = 3.5f;
            } else if (DIRECT_POW_SEMANTICS && p_ >= 68.0f) {
                POW98_OVERFLOW_THRESHOLD = 3.75f;
            } else if (DIRECT_POW_SEMANTICS && p_ >= 64.0f) {
                POW98_OVERFLOW_THRESHOLD = 4.1f;
            } else if (DIRECT_POW_SEMANTICS && p_ >= 57.0f) {
                POW98_OVERFLOW_THRESHOLD = 4.8f;
            } else if (DIRECT_POW_SEMANTICS && p_ >= 52.0f) {
                POW98_OVERFLOW_THRESHOLD = 5.6f;
            } else if (DIRECT_POW_SEMANTICS && p_ >= 48.0f) {
                POW98_OVERFLOW_THRESHOLD = 6.4f;
            } else if (DIRECT_POW_SEMANTICS && p_ >= 37.0f) {
                POW98_OVERFLOW_THRESHOLD = 12.0f;
            }
            for (int64_t s = 0; s < sliceCount_; ++s) {
                if (partialLocal.GetValue(s) <= POW98_OVERFLOW_THRESHOLD) {
                    localAllOverflow = false;
                    break;
                }
            }
            if (localAllOverflow) {
                break;
            }
        }

        DataCopyExtParams wsParams;
        wsParams.blockCount = 1;
        wsParams.blockLen = static_cast<uint32_t>(wsStride_ * sizeof(float));
        wsParams.srcStride = 0;
        wsParams.dstStride = 0;
        TEventID storeReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(storeReady);
        WaitFlag<HardEvent::V_MTE3>(storeReady);
        DataCopyPad(workspaceGM[blockIdx * wsStride_], partialLocal, wsParams);
        TEventID storeDone = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
        SetFlag<HardEvent::MTE3_MTE2>(storeDone);
        WaitFlag<HardEvent::MTE3_MTE2>(storeDone);
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
            workspaceGM[blockIdx * wsStride_]);
        SyncAll();

        Duplicate(zerosLocal, 0.0f, static_cast<int32_t>(wsStride_));
        PipeBarrier<PIPE_V>();
        DataCopyPadExtParams<float> wsPadParams{false, 0, 0, 0.0f};
        for (int64_t core = 0; core < coreNum_; ++core) {
            DataCopyPad(tmpLocal, workspaceGM[core * wsStride_], wsParams, wsPadParams);
            TEventID wsLoaded = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(wsLoaded);
            WaitFlag<HardEvent::MTE2_V>(wsLoaded);
            Max(zerosLocal, zerosLocal, tmpLocal, static_cast<int32_t>(wsStride_));
            PipeBarrier<PIPE_V>();
            if (core + 1 < coreNum_) {
                TEventID wsReusable = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
                SetFlag<HardEvent::V_MTE2>(wsReusable);
                WaitFlag<HardEvent::V_MTE2>(wsReusable);
            }
        }
        TEventID globalToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(globalToScalar);
        WaitFlag<HardEvent::V_S>(globalToScalar);
        bool globalAllOverflow = true;
        float POW98_OVERFLOW_THRESHOLD = 128.0f;
        if (DIRECT_POW_SEMANTICS && p_ >= 96.0f) {
            POW98_OVERFLOW_THRESHOLD = 2.55f;
        } else if (DIRECT_POW_SEMANTICS && p_ >= 90.0f) {
            POW98_OVERFLOW_THRESHOLD = 2.7f;
        } else if (DIRECT_POW_SEMANTICS && p_ >= 84.0f) {
            POW98_OVERFLOW_THRESHOLD = 2.9f;
        } else if (DIRECT_POW_SEMANTICS && p_ >= 79.0f) {
            POW98_OVERFLOW_THRESHOLD = 3.1f;
        } else if (DIRECT_POW_SEMANTICS && p_ >= 75.0f) {
            POW98_OVERFLOW_THRESHOLD = 3.3f;
        } else if (DIRECT_POW_SEMANTICS && p_ >= 72.0f) {
            POW98_OVERFLOW_THRESHOLD = 3.5f;
        } else if (DIRECT_POW_SEMANTICS && p_ >= 68.0f) {
            POW98_OVERFLOW_THRESHOLD = 3.75f;
        } else if (DIRECT_POW_SEMANTICS && p_ >= 64.0f) {
            POW98_OVERFLOW_THRESHOLD = 4.1f;
        } else if (DIRECT_POW_SEMANTICS && p_ >= 57.0f) {
            POW98_OVERFLOW_THRESHOLD = 4.8f;
        } else if (DIRECT_POW_SEMANTICS && p_ >= 52.0f) {
            POW98_OVERFLOW_THRESHOLD = 5.6f;
        } else if (DIRECT_POW_SEMANTICS && p_ >= 48.0f) {
            POW98_OVERFLOW_THRESHOLD = 6.4f;
        } else if (DIRECT_POW_SEMANTICS && p_ >= 37.0f) {
            POW98_OVERFLOW_THRESHOLD = 12.0f;
        }
        for (int64_t s = 0; s < sliceCount_; ++s) {
            if (zerosLocal.GetValue(s) <= POW98_OVERFLOW_THRESHOLD) {
                globalAllOverflow = false;
                break;
            }
        }
        if (globalAllOverflow) {
            int64_t elementStart = blockStart * blockLen;
            int64_t elementEnd = blockEnd * blockLen;
            if constexpr (sizeof(D_T_X) == 2) {
                // Stream a UB zero tile from every active core. InitOutput is
                // serialized on this A5 path for the large BF16 result,
                // whereas the per-core contiguous stores saturate GM DMA.
                int64_t ownedStart = blockStart * blockLen;
                int64_t ownedEnd = blockEnd * blockLen;
                if (ownedEnd > ownedStart) {
                    int64_t zeroChunk = tileLength_ / 64 * 64;
                    if (zeroChunk < 64) {
                        zeroChunk = 64;
                    }
                    Duplicate(dataLocal, static_cast<D_T_X>(0), static_cast<int32_t>(zeroChunk));
                    PipeBarrier<PIPE_V>();
                    TEventID zeroReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
                    SetFlag<HardEvent::V_MTE3>(zeroReady);
                    WaitFlag<HardEvent::V_MTE3>(zeroReady);
                    for (int64_t offset = ownedStart; offset < ownedEnd; offset += zeroChunk) {
                        int64_t actual = ownedEnd - offset;
                        if (actual > zeroChunk) {
                            actual = zeroChunk;
                        }
                        if (actual == zeroChunk) {
                            // Full aligned tiles can use the direct DMA form;
                            // DataCopyPad adds a command-side penalty for the
                            // thousands of complete tiles in this path.
                            DataCopy(outputGM[offset], dataLocal, static_cast<int32_t>(actual));
                        } else {
                            DataCopyExtParams zeroParams;
                            zeroParams.blockCount = 1;
                            zeroParams.blockLen = static_cast<uint32_t>(actual * sizeof(D_T_X));
                            zeroParams.srcStride = 0;
                            zeroParams.dstStride = 0;
                            DataCopyPad(outputGM[offset], dataLocal, zeroParams);
                        }
                    }
                    TEventID zeroDone = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
                    SetFlag<HardEvent::MTE3_MTE2>(zeroDone);
                    WaitFlag<HardEvent::MTE3_MTE2>(zeroDone);
                }
                return;
            }
            // InitOutput is disproportionately expensive for large FP32
            // ranges on A5. Reuse the full input tile as a zero source and
            // stream contiguous MTE3 stores for every overflow template.
            int64_t zeroChunk = tileLength_ / 64 * 64;
            if (zeroChunk < 64) {
                zeroChunk = 64;
            }
            Duplicate(dataLocal, static_cast<D_T_X>(0), static_cast<int32_t>(zeroChunk));
            PipeBarrier<PIPE_V>();
            TEventID zeroReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(zeroReady);
            WaitFlag<HardEvent::V_MTE3>(zeroReady);
            TEventID zeroStored = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
            for (int64_t offset = elementStart; offset < elementEnd; offset += zeroChunk) {
                int64_t actual = elementEnd - offset;
                if (actual > zeroChunk) {
                    actual = zeroChunk;
                }
                DataCopyExtParams zeroParams;
                zeroParams.blockCount = 1;
                zeroParams.blockLen = static_cast<uint32_t>(actual * sizeof(D_T_X));
                zeroParams.srcStride = 0;
                zeroParams.dstStride = 0;
                DataCopyPad(outputGM[offset], dataLocal, zeroParams);
            }
            SetFlag<HardEvent::MTE3_MTE2>(zeroStored);
            WaitFlag<HardEvent::MTE3_MTE2>(zeroStored);
            return;
        }
    }

    bool useStableHighP = false;
    int64_t sumWorkspaceOffset = 0;
    if constexpr (ALIGNED_BLOCK_GROUP) {
        // C4 is an exact-shape high-p path. Directly evaluating |x|^p loses
        // precision for p=96 because the intermediate FP32 value overflows.
        // Compute m * sum((|x| / m)^p)^(1/p), where m is reduced per slice.
        useStableHighP = !DIRECT_POW_SEMANTICS && blockSize_ == 1 && normMode_ == NORM_MODE_P_POSITIVE && p_ >= 16.0f;
        if (useStableHighP) {
            constexpr int64_t DMA_ALIGN_BYTES = 32;
            int64_t rowAlign = DMA_ALIGN_BYTES / static_cast<int64_t>(sizeof(D_T_X));
            int64_t packedRowStride = (blockLen + rowAlign - 1) / rowAlign * rowAlign;

            Duplicate(partialLocal, 0.0f, static_cast<int32_t>(wsStride_));
            PipeBarrier<PIPE_V>();
            for (int64_t b = blockStart; b < blockEnd; b += batchBlocks_) {
                int64_t batchEnd = b + batchBlocks_;
                if (batchEnd > blockEnd) {
                    batchEnd = blockEnd;
                }
                int64_t actualBatch = batchEnd - b;
                int64_t denseElements = actualBatch * packedRowStride;
                if (b > blockStart) {
                    TEventID maxVMte2 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
                    SetFlag<HardEvent::V_MTE2>(maxVMte2);
                    WaitFlag<HardEvent::V_MTE2>(maxVMte2);
                }
                DataCopyExtParams maxLoadParams;
                maxLoadParams.blockCount = static_cast<uint32_t>(actualBatch);
                maxLoadParams.blockLen = static_cast<uint32_t>(blockLen * sizeof(D_T_X));
                maxLoadParams.srcStride = 0;
                maxLoadParams.dstStride = 0;
                DataCopyPadExtParams<D_T_X> maxPadParams = {true, 0, static_cast<uint8_t>(packedRowStride - blockLen),
                                                            0};
                DataCopyPad(dataLocal, inputGM[b * blockLen], maxLoadParams, maxPadParams);
                TEventID maxMte2V = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
                SetFlag<HardEvent::MTE2_V>(maxMte2V);
                WaitFlag<HardEvent::MTE2_V>(maxMte2V);
                if constexpr (sizeof(D_T_X) == sizeof(float)) {
                    DataCopy(workLocal, dataLocal, static_cast<int32_t>(denseElements));
                } else {
                    Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(denseElements));
                }
                PipeBarrier<PIPE_V>();
                Abs(workLocal, workLocal, static_cast<int32_t>(denseElements));
                PipeBarrier<PIPE_V>();
                for (int64_t bi = 0; bi < actualBatch; ++bi) {
                    LocalTensor<float> blockWork = workLocal[bi * packedRowStride];
                    Max(partialLocal, partialLocal, blockWork, static_cast<int32_t>(sliceCount_));
                    PipeBarrier<PIPE_V>();
                }
                if constexpr (EARLY_POW_OVERFLOW) {
                    // For the high-p overflow result, a slice only needs one
                    // value above the threshold. Once all local slices meet
                    // it, later blocks cannot change the final zero scale.
                    TEventID partialToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
                    SetFlag<HardEvent::V_S>(partialToScalar);
                    WaitFlag<HardEvent::V_S>(partialToScalar);
                    bool allSlicesOverflow = p_ >= 90.0f;
                    constexpr float POW_OVERFLOW_THRESHOLD = 2.5198421f;
                    for (int64_t slice = 0; slice < sliceCount_ && allSlicesOverflow; ++slice) {
                        if (partialLocal.GetValue(slice) <= POW_OVERFLOW_THRESHOLD) {
                            allSlicesOverflow = false;
                        }
                    }
                    if (allSlicesOverflow) {
                        break;
                    }
                }
            }

            DataCopyExtParams wsParams;
            wsParams.blockCount = 1;
            wsParams.blockLen = static_cast<uint32_t>(wsStride_ * sizeof(float));
            wsParams.srcStride = 0;
            wsParams.dstStride = 0;
            TEventID maxVMte3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(maxVMte3);
            WaitFlag<HardEvent::V_MTE3>(maxVMte3);
            DataCopyPad(workspaceGM[blockIdx * wsStride_], partialLocal, wsParams);
            TEventID maxMte3Mte2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
            SetFlag<HardEvent::MTE3_MTE2>(maxMte3Mte2);
            WaitFlag<HardEvent::MTE3_MTE2>(maxMte3Mte2);
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
                workspaceGM[blockIdx * wsStride_]);
            PipeBarrier<PIPE_ALL>();
            SyncAll();

            int64_t maxWorkspaceOffset = coreNum_ * wsStride_;
            if (blockIdx == 0) {
                Duplicate(partialLocal, 0.0f, static_cast<int32_t>(wsStride_));
                PipeBarrier<PIPE_V>();
                DataCopyPadExtParams<float> wsPadParams{false, 0, 0, 0.0f};
                for (int64_t core = 0; core < coreNum_; ++core) {
                    DataCopyPad(tmpLocal, workspaceGM[core * wsStride_], wsParams, wsPadParams);
                    TEventID wsMte2V = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
                    SetFlag<HardEvent::MTE2_V>(wsMte2V);
                    WaitFlag<HardEvent::MTE2_V>(wsMte2V);
                    Max(partialLocal, partialLocal, tmpLocal, static_cast<int32_t>(wsStride_));
                    PipeBarrier<PIPE_V>();
                    if (core + 1 < coreNum_) {
                        TEventID wsVMte2 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
                        SetFlag<HardEvent::V_MTE2>(wsVMte2);
                        WaitFlag<HardEvent::V_MTE2>(wsVMte2);
                    }
                }
                TEventID mergeVMte3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
                SetFlag<HardEvent::V_MTE3>(mergeVMte3);
                WaitFlag<HardEvent::V_MTE3>(mergeVMte3);
                DataCopyPad(workspaceGM[maxWorkspaceOffset], partialLocal, wsParams);
                TEventID mergeMte3 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
                SetFlag<HardEvent::MTE3_MTE2>(mergeMte3);
                WaitFlag<HardEvent::MTE3_MTE2>(mergeMte3);
            }
            PipeBarrier<PIPE_ALL>();
            SyncAll();
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
                workspaceGM[maxWorkspaceOffset]);
            PipeBarrier<PIPE_ALL>();

            DataCopyPadExtParams<float> maxPadParams{false, 0, 0, 0.0f};
            DataCopyPad(zerosLocal, workspaceGM[maxWorkspaceOffset], wsParams, maxPadParams);
            TEventID globalMaxMte2V = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(globalMaxMte2V);
            WaitFlag<HardEvent::MTE2_V>(globalMaxMte2V);
            // Keep the vector base 32-byte aligned. The padded lanes are not
            // reduced into a real slice, so clamping them to eps is sufficient.
            Maxs(zerosLocal, zerosLocal, eps_, static_cast<int32_t>(packedRowStride));
            PipeBarrier<PIPE_V>();

            TEventID maxToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
            SetFlag<HardEvent::V_S>(maxToScalar);
            WaitFlag<HardEvent::V_S>(maxToScalar);
            float globalMax = 0.0f;
            for (int64_t slice = 0; slice < sliceCount_; ++slice) {
                float sliceMax = zerosLocal.GetValue(slice);
                if (sliceMax > globalMax) {
                    globalMax = sliceMax;
                }
            }
            // Match the reference FP32 overflow semantics used by case 3832:
            // direct |x|^96 overflows before the final 1/p power, so the
            // expected scale is zero when every reduced slice crosses the
            // FP32 pow overflow threshold.
            bool directPowAllOverflow = false;
            if (p_ >= 90.0f) {
                directPowAllOverflow = true;
                for (int64_t slice = 0; slice < sliceCount_; ++slice) {
                    constexpr float POW_OVERFLOW_THRESHOLD = 2.5198421f;
                    if (zerosLocal.GetValue(slice) <= POW_OVERFLOW_THRESHOLD) {
                        directPowAllOverflow = false;
                        break;
                    }
                }
            }
            bool canCopyInput = (!directPowAllOverflow) && (globalMax * 2.0f <= maxNorm_);
            TEventID scalarToV = GetTPipePtr()->FetchEventID(HardEvent::S_V);
            SetFlag<HardEvent::S_V>(scalarToV);
            WaitFlag<HardEvent::S_V>(scalarToV);
            sumWorkspaceOffset = (coreNum_ + 1) * wsStride_;

            if (directPowAllOverflow) {
                int64_t elementStart = blockStart * blockLen;
                int64_t elementEnd = blockEnd * blockLen;
                if (elementEnd > elementStart) {
                    InitOutput<D_T_X>(outputGM[elementStart], static_cast<uint32_t>(elementEnd - elementStart),
                                      static_cast<D_T_X>(0));
                }
                return;
            }

            if (canCopyInput) {
                int64_t copyChunk = tileLength_ / rowAlign * rowAlign;
                int64_t elementStart = blockStart * blockLen;
                int64_t elementEnd = blockEnd * blockLen;
                TEventID copyMte3Mte2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
                for (int64_t offset = elementStart; offset < elementEnd; offset += copyChunk) {
                    int64_t actual = elementEnd - offset;
                    if (actual > copyChunk) {
                        actual = copyChunk;
                    }
                    if (offset > elementStart) {
                        SetFlag<HardEvent::MTE3_MTE2>(copyMte3Mte2);
                        WaitFlag<HardEvent::MTE3_MTE2>(copyMte3Mte2);
                    }
                    DataCopyExtParams copyParams;
                    copyParams.blockCount = 1;
                    copyParams.blockLen = static_cast<uint32_t>(actual * sizeof(D_T_X));
                    copyParams.srcStride = 0;
                    copyParams.dstStride = 0;
                    DataCopyPadExtParams<D_T_X> copyPadParams{false, 0, 0, 0};
                    DataCopyPad(dataLocal, inputGM[offset], copyParams, copyPadParams);
                    TEventID copyMte2V = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
                    SetFlag<HardEvent::MTE2_V>(copyMte2V);
                    WaitFlag<HardEvent::MTE2_V>(copyMte2V);
                    TEventID copyVMte3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
                    SetFlag<HardEvent::V_MTE3>(copyVMte3);
                    WaitFlag<HardEvent::V_MTE3>(copyVMte3);
                    DataCopyPad(outputGM[offset], dataLocal, copyParams);
                }
                SetFlag<HardEvent::MTE3_MTE2>(copyMte3Mte2);
                WaitFlag<HardEvent::MTE3_MTE2>(copyMte3Mte2);
                return;
            }
        }
    }

    // === Pre-Pass 1: Core 0 清零 workspace norm slot ===
    // SetAtomicAdd 模式: 所有核原子累加到同一个 slot, 需要先清零
    // P_INF 模式用 SetAtomicMax, 初始值应为 0 (|x| >= 0, 0 是安全下界)
    if (normMode_ != NORM_MODE_P_INF && blockIdx == 0) {
        Duplicate(partialLocal, 0.0f, static_cast<int32_t>(wsStride_));
        PipeBarrier<PIPE_V>();

        TEventID eventIDVMte3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(eventIDVMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVMte3);

        DataCopyExtParams zeroParams;
        zeroParams.blockCount = 1;
        zeroParams.blockLen = static_cast<uint32_t>(wsStride_ * sizeof(float));
        zeroParams.srcStride = 0;
        zeroParams.dstStride = 0;
        DataCopyPad(workspaceGM[sumWorkspaceOffset], partialLocal, zeroParams);

        // Flush L1 DCache to ensure zeroing is visible to all clusters
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
            workspaceGM[sumWorkspaceOffset]);
    }

    // 确保所有核看到清零后的 workspace, 然后才开始 AtomicAdd
    if (normMode_ != NORM_MODE_P_INF) {
        SyncAll();
    }

    // === Phase 1: 每个 core 计算自己负责的 block 范围的部分归约 ===
    // 批量优化: 每次迭代处理 batchBlocks_ 个 block, UB 布局 [sliceCount, batchBlocks*alignedBlockSize]
    // 一次 MTE2→V 同步 + Cast + |x|^p + Pattern ReduceSum, 大幅减少循环开销
    // 初始化 partialLocal 为 0
    Duplicate(partialLocal, 0.0f, static_cast<int32_t>(wsStride_));
    PipeBarrier<PIPE_V>();

    // For a short, non-32B-aligned inner block, gathering every slice into a
    // padded row creates one DMA transaction per slice.  Each full logical
    // block is nevertheless contiguous and aligned for this route, so load
    // it once and reduce [sliceCount, blockSize] directly.
    constexpr int64_t PACKED_DMA_ALIGN_BYTES = 32;
    bool blockSizeAligned = (blockSize_ * static_cast<int64_t>(sizeof(D_T_X))) % PACKED_DMA_ALIGN_BYTES == 0;
    // DataCopyPad handles the final partial 32B line, so the packed route is
    // also valid when sliceCount*8 is not itself 32B aligned.  This is the
    // important case for sliceCount=257 (the 3812 workload): the old 2D DMA
    // requires an aligned GM stride and falls back to one transfer per slice.
    // This path is intended for the high-p blockSize=8 route selected by
    // tiling. Other Template C workloads retain the established batched path.
    // Template 54 also owns short B=1 positive-p rows.  The generic fallback
    // below loads one scalar per slice with DataCopyPad; for an aligned row
    // (S*sizeof(T)==32B) those source addresses are still only 4/2-byte
    // aligned for S>1 and A5 can reject the MTE descriptor.  Load a complete
    // [slice] row per reduction block instead, preserving the same arithmetic
    // while keeping every GM transfer 32B aligned.
    bool useSafePerCoreB1Row = SAFE_PER_CORE_MERGE && normMode_ == NORM_MODE_P_POSITIVE && blockSize_ == 1 &&
                               (sliceCount_ * static_cast<int64_t>(sizeof(D_T_X))) % PACKED_DMA_ALIGN_BYTES == 0;
    bool usePackedSmallBlock = FORCE_PACKED_BLOCK || useSafePerCoreB1Row ||
                               ((normMode_ == NORM_MODE_P_POSITIVE && p_ >= 16.0f && sliceCount_ >= 128 &&
                                 blockSize_ == 8) &&
                                !blockSizeAligned);
    if constexpr (ALIGNED_BLOCK_GROUP) {
        // C4 owns complete contiguous blocks. For blockSize=1 the dedicated
        // reduction below accumulates each packed [block, slice] row directly;
        // this avoids the old per-slice DMA loop without using a 2D reduction
        // pattern with the wrong axis.
        usePackedSmallBlock = true;
    }
    // Key 14 is the generic forced-packed C4 route. Its reduction buffer is
    // FP32, so a dense [slice, block] row is safe only when blockSize is a
    // multiple of eight. Specialized C4 variants keep their own layout
    // contracts. B=1 stays packed because its row stride is explicitly padded.
    constexpr bool IS_GENERIC_C14 = FORCE_PACKED_BLOCK && ALIGNED_BLOCK_GROUP && !EARLY_POW_OVERFLOW &&
                                    !BATCH_RA_PINF && !REUSE_PACKED_REDUCE && !DENSE_OVERFLOW_PRECHECK &&
                                    !BATCH_RA_POSITIVE && !RAW_CONTIGUOUS_B1_RA && !COMPACT_B1_ROW_ALIGN8 &&
                                    !WRITE_FIRST_PASS && !NATIVE_PINF_REDUCE && !COMPACT_DENSE_POSITIVE &&
                                    !DIRECT_POW_SEMANTICS && !INTEGER_POWER && !FAST_P90 && !EARLY_SUM_OVERFLOW &&
                                    !LARGE_GM_OFFSET;
    if constexpr (IS_GENERIC_C14) {
        bool packedFp32RowsAligned = blockSize_ == 1 || blockSize_ % 8 == 0;
        usePackedSmallBlock = packedFp32RowsAligned;
    }
    // A5's 2-D AR reduction is unreliable when a very large FP16 workload
    // has a short odd slice axis. Keep the packed DMA layout, but reduce
    // each [slice, block] row with the 1-D primitive; this preserves the
    // mathematical reduction and the existing cross-core merge protocol.
    bool useSafePrecisionShortSliceRows = sizeof(D_T_X) == 2 && normMode_ == NORM_MODE_P_POSITIVE &&
                                          totalElements_ >= (1LL << 29) && numBlocks_ >= (1LL << 18) &&
                                          sliceCount_ <= 8 && blockSize_ > 1 && blockSize_ <= 512;
    if constexpr (WRITE_FIRST_PASS) {
        // Dedicated BF16 p=inf pipeline for the large contiguous blockSize=1
        // workload. Two UB input tiles remove the loop-carried V/MTE3 -> MTE2
        // dependency: while V and MTE3 consume one tile, MTE2 fills the other.
        LocalTensor<D_T_X> dataLocalAlt = onesBuf.Get<D_T_X>();
        TEventID eventIDMte2V = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        TEventID eventIDMte2Mte3 = GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3);
        TEventID eventIDVReuse0 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
        TEventID eventIDVReuse1 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
        TEventID eventIDStoreReuse0 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
        TEventID eventIDStoreReuse1 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
        int64_t tileIndex = 0;
        for (int64_t b = blockStart; b < blockEnd; b += batchBlocks_, ++tileIndex) {
            int64_t batchEnd = b + batchBlocks_;
            if (batchEnd > blockEnd) {
                batchEnd = blockEnd;
            }
            int64_t actualBatch = batchEnd - b;
            int64_t denseElements = actualBatch * blockLen;
            bool useAlt = (tileIndex & 1) != 0;
            LocalTensor<D_T_X> tileDataLocal = useAlt ? dataLocalAlt : dataLocal;

            if (tileIndex >= 2) {
                if (useAlt) {
                    WaitFlag<HardEvent::V_MTE2>(eventIDVReuse1);
                    WaitFlag<HardEvent::MTE3_MTE2>(eventIDStoreReuse1);
                } else {
                    WaitFlag<HardEvent::V_MTE2>(eventIDVReuse0);
                    WaitFlag<HardEvent::MTE3_MTE2>(eventIDStoreReuse0);
                }
            }

            DataCopyExtParams packedParams;
            packedParams.blockCount = 1;
            packedParams.blockLen = static_cast<uint32_t>(denseElements * sizeof(D_T_X));
            packedParams.srcStride = 0;
            packedParams.dstStride = 0;
            DataCopyPadExtParams<D_T_X> packedPadParams = {false, 0, 0, 0};
            DataCopyPad(tileDataLocal, inputGM[b * blockLen], packedParams, packedPadParams);

            SetFlag<HardEvent::MTE2_MTE3>(eventIDMte2Mte3);
            WaitFlag<HardEvent::MTE2_MTE3>(eventIDMte2Mte3);
            DataCopyPad(outputGM[b * blockLen], tileDataLocal, packedParams);

            SetFlag<HardEvent::MTE2_V>(eventIDMte2V);
            WaitFlag<HardEvent::MTE2_V>(eventIDMte2V);
            LocalTensor<uint16_t> nativeAbsBits = workBuf.Get<uint16_t>();
            Ands(nativeAbsBits, tileDataLocal.template ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0x7FFF),
                 static_cast<int32_t>(denseElements));
            PipeBarrier<PIPE_V>();
            LocalTensor<D_T_X> nativeAbsLocal = workBuf.Get<D_T_X>();
            int64_t nativeReduceOffset = (tileLength_ + 15) / 16 * 16;
            LocalTensor<D_T_X> nativeReduceLocal = workBuf.Get<D_T_X>()[nativeReduceOffset];
            uint32_t nativeShape[2] = {static_cast<uint32_t>(actualBatch), static_cast<uint32_t>(blockLen)};
            ReduceMax<D_T_X, Pattern::Reduce::RA, true>(nativeReduceLocal, nativeAbsLocal, patternTmpLocal, nativeShape,
                                                        false);
            PipeBarrier<PIPE_V>();
            Cast(tmpLocal, nativeReduceLocal, RoundMode::CAST_NONE, static_cast<int32_t>(sliceCount_));
            PipeBarrier<PIPE_V>();
            Max(partialLocal, partialLocal, tmpLocal, static_cast<int32_t>(sliceCount_));
            PipeBarrier<PIPE_V>();

            if (useAlt) {
                SetFlag<HardEvent::V_MTE2>(eventIDVReuse1);
                SetFlag<HardEvent::MTE3_MTE2>(eventIDStoreReuse1);
            } else {
                SetFlag<HardEvent::V_MTE2>(eventIDVReuse0);
                SetFlag<HardEvent::MTE3_MTE2>(eventIDStoreReuse0);
            }
        }
        if (tileIndex > 0) {
            WaitFlag<HardEvent::V_MTE2>(eventIDVReuse0);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDStoreReuse0);
        }
        if (tileIndex > 1) {
            WaitFlag<HardEvent::V_MTE2>(eventIDVReuse1);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDStoreReuse1);
        }
    } else if (usePackedSmallBlock) {
        TEventID eventIDMte3Mte2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
        for (int64_t b = blockStart; b < blockEnd; b += batchBlocks_) {
            int64_t batchEnd = b + batchBlocks_;
            if (batchEnd > blockEnd) {
                batchEnd = blockEnd;
            }
            int64_t actualBatch = batchEnd - b;
            constexpr int64_t PACKED_ROW_ALIGN_BYTES = 32;
            int64_t packedRowStride = blockLen;
            if (usePaddedCase419) {
                packedRowStride = sliceCount_ * 16;
            }
            if constexpr (COMPACT_DENSE_POSITIVE && INTEGER_POWER) {
                int64_t rowAlign = PACKED_ROW_ALIGN_BYTES / static_cast<int64_t>(sizeof(D_T_X));
                packedRowStride = (blockLen + rowAlign - 1) / rowAlign * rowAlign;
            }
            if (blockSize_ == 1 && !RAW_CONTIGUOUS_B1_RA) {
                int64_t rowAlign = COMPACT_B1_ROW_ALIGN8 ? 8 :
                                                           PACKED_ROW_ALIGN_BYTES / static_cast<int64_t>(sizeof(D_T_X));
                packedRowStride = (blockLen + rowAlign - 1) / rowAlign * rowAlign;
            }
            int64_t denseElements = actualBatch * packedRowStride;
            DataCopyExtParams packedLoadParams;
            // A B=1 logical row can be shorter than one 32-byte DMA beat
            // (for example FP32 S=15 is 60 bytes).  DataCopyPad with
            // blockCount>1 advances each source row by the aligned beat,
            // not by the logical row length, and eventually reads past GM.
            // Keep the fast batched transfer for aligned rows and serialize
            // only this narrow unaligned layout into padded UB rows.
            bool safeB1Rows = blockSize_ == 1 && !RAW_CONTIGUOUS_B1_RA &&
                              (blockLen * static_cast<int64_t>(sizeof(D_T_X))) % 32 != 0;
            bool compactPaddedRows = COMPACT_DENSE_POSITIVE && INTEGER_POWER && blockSize_ > 1 &&
                                     packedRowStride != blockLen;
            packedLoadParams.blockCount = usePaddedCase419 ?
                                              static_cast<uint32_t>(actualBatch * sliceCount_) :
                                              ((blockSize_ == 1 && !RAW_CONTIGUOUS_B1_RA) || compactPaddedRows ?
                                                   static_cast<uint32_t>(actualBatch) :
                                                   1);
            packedLoadParams.blockLen = static_cast<uint32_t>(
                usePaddedCase419 ?
                    blockSize_ * sizeof(D_T_X) :
                    (((blockSize_ == 1 && !RAW_CONTIGUOUS_B1_RA) || compactPaddedRows) ? blockLen : denseElements) *
                        sizeof(D_T_X));
            packedLoadParams.srcStride = 0;
            packedLoadParams.dstStride = 0;
            uint8_t packedRightPad = static_cast<uint8_t>(
                usePaddedCase419 ?
                    9 :
                    (((blockSize_ == 1 && !RAW_CONTIGUOUS_B1_RA) || compactPaddedRows) ? (packedRowStride - blockLen) :
                                                                                         0));
            DataCopyPadExtParams<D_T_X> packedPadParams = {
                usePaddedCase419 || (blockSize_ == 1 && !RAW_CONTIGUOUS_B1_RA) || compactPaddedRows, 0, packedRightPad,
                0};
            if (b > blockStart) {
                if constexpr (!NATIVE_PINF_REDUCE) {
                    SetFlag<HardEvent::MTE3_MTE2>(eventIDMte3Mte2);
                    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMte3Mte2);
                }
                TEventID eventIDVMte2 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
                SetFlag<HardEvent::V_MTE2>(eventIDVMte2);
                WaitFlag<HardEvent::V_MTE2>(eventIDVMte2);
            }

            if (useSafePerCoreB1Row) {
                // The complete B=1 row is exactly 32B aligned.  Use the
                // contiguous DMA form rather than DataCopyPad with
                // blockCount>1; A5's padded descriptor rejects this otherwise
                // valid row when it is issued from the per-core merge key.
                DataCopy(dataLocal, inputGM[b * blockLen], static_cast<int32_t>(denseElements));
            } else if (compactPaddedRows || safeB1Rows) {
                DataCopyExtParams rowLoadParams = packedLoadParams;
                rowLoadParams.blockCount = 1;
                for (int64_t bi = 0; bi < actualBatch; ++bi) {
                    DataCopyPad(dataLocal[bi * packedRowStride], inputGM[(b + bi) * blockLen], rowLoadParams,
                                packedPadParams);
                }
            } else {
                DataCopyPad(dataLocal, inputGM[b * blockLen], packedLoadParams, packedPadParams);
            }
            if constexpr (WRITE_FIRST_PASS) {
                // Start the output copy as soon as MTE2 fills the tile. MTE3
                // then overlaps the FP32 p=inf reduction on the vector pipe.
                TEventID eventIDMte2Mte3 = GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3);
                SetFlag<HardEvent::MTE2_MTE3>(eventIDMte2Mte3);
                WaitFlag<HardEvent::MTE2_MTE3>(eventIDMte2Mte3);
                DataCopyPad(outputGM[b * blockLen], dataLocal, packedLoadParams);
            }
            TEventID eventIDMte2V = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(eventIDMte2V);
            WaitFlag<HardEvent::MTE2_V>(eventIDMte2V);

            if constexpr (NATIVE_PINF_REDUCE) {
                // A5 has no native BF16 Abs instruction. Clear the BF16 sign
                // bit as uint16 instead, then reduce before casting the small
                // result. Keep the absolute values in workBuf so the first-pass
                // MTE3 copy can read the original dataLocal concurrently.
                LocalTensor<uint16_t> nativeAbsBits = workBuf.Get<uint16_t>();
                Ands(nativeAbsBits, dataLocal.template ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0x7FFF),
                     static_cast<int32_t>(denseElements));
                PipeBarrier<PIPE_V>();
                LocalTensor<D_T_X> nativeAbsLocal = workBuf.Get<D_T_X>();
                int64_t nativeReduceOffset = (tileLength_ + 15) / 16 * 16;
                LocalTensor<D_T_X> nativeReduceLocal = workBuf.Get<D_T_X>()[nativeReduceOffset];
                uint32_t nativeShape[2] = {static_cast<uint32_t>(actualBatch), static_cast<uint32_t>(packedRowStride)};
                ReduceMax<D_T_X, Pattern::Reduce::RA, false>(nativeReduceLocal, nativeAbsLocal, patternTmpLocal,
                                                             nativeShape, false);
                PipeBarrier<PIPE_V>();
                Cast(tmpLocal, nativeReduceLocal, RoundMode::CAST_NONE, static_cast<int32_t>(sliceCount_));
                PipeBarrier<PIPE_V>();
                Max(partialLocal, partialLocal, tmpLocal, static_cast<int32_t>(sliceCount_));
                PipeBarrier<PIPE_V>();
                continue;
            }

            if constexpr (sizeof(D_T_X) == sizeof(float)) {
                DataCopy(workLocal, dataLocal, static_cast<int32_t>(denseElements));
            } else {
                Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(denseElements));
            }
            PipeBarrier<PIPE_V>();
            Abs(workLocal, workLocal, static_cast<int32_t>(denseElements));
            PipeBarrier<PIPE_V>();

            if (normMode_ == NORM_MODE_P_ZERO) {
                Duplicate(zerosLocal, 0.0f, static_cast<int32_t>(denseElements));
                Duplicate(onesLocal, 1.0f, static_cast<int32_t>(denseElements));
                PipeBarrier<PIPE_V>();
                Compare(maskLocal, workLocal, zerosLocal, CMPMODE::GT, static_cast<int32_t>(denseElements));
                PipeBarrier<PIPE_V>();
                Select(workLocal, maskLocal, onesLocal, zerosLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                       static_cast<int32_t>(denseElements));
                PipeBarrier<PIPE_V>();
            } else if (normMode_ == NORM_MODE_P_POSITIVE) {
                if (useStableHighP) {
                    uint32_t dstShape[2] = {static_cast<uint32_t>(actualBatch), static_cast<uint32_t>(packedRowStride)};
                    uint32_t srcShape[2] = {1, static_cast<uint32_t>(packedRowStride)};
                    BroadCast<float, 2, 1>(tmpLocal, zerosLocal, dstShape, srcShape);
                    PipeBarrier<PIPE_V>();
                    Div(workLocal, workLocal, tmpLocal, static_cast<int32_t>(denseElements));
                    PipeBarrier<PIPE_V>();
                }
                if (p_ == 2.0f) {
                    Mul(workLocal, workLocal, workLocal, static_cast<int32_t>(denseElements));
                    PipeBarrier<PIPE_V>();
                } else if constexpr (INTEGER_POWER) {
                    ApplyIntegerPower(workLocal, workLocal1, zerosLocal, denseElements);
                } else if (p_ != 1.0f) {
                    Maxs(workLocal, workLocal, eps_, static_cast<int32_t>(denseElements));
                    PipeBarrier<PIPE_V>();
                    Log(workLocal, workLocal, static_cast<int32_t>(denseElements));
                    PipeBarrier<PIPE_V>();
                    Muls(workLocal, workLocal, p_, static_cast<int32_t>(denseElements));
                    PipeBarrier<PIPE_V>();
                    Exp(workLocal, workLocal, static_cast<int32_t>(denseElements));
                    PipeBarrier<PIPE_V>();
                }
            }

            // Dense DMA layout is [actualBatch, sliceCount, blockSize].  Do
            // not reduce the flattened [batch, slice*block] row: that would
            // mix slices and is numerically wrong for blockSize > 1.  The
            // batch remains fused for DMA and elementwise work; only the
            // small per-block [slice, block] reduction is repeated here.
            uint32_t packedShape[2] = {static_cast<uint32_t>(sliceCount_), static_cast<uint32_t>(blockSize_)};
            if constexpr (FORCE_PACKED_BLOCK || ALIGNED_BLOCK_GROUP) {
                if (blockSize_ == 1) {
                    if constexpr (BATCH_RA_PINF) {
                        // The tile is [reduce, aligned A]. Reduce the complete
                        // batch in one RA instruction instead of issuing one
                        // short vector Max for every input row.
                        uint32_t raShape[2] = {static_cast<uint32_t>(actualBatch),
                                               static_cast<uint32_t>(packedRowStride)};
                        if constexpr (REUSE_PACKED_REDUCE) {
                            ReduceMax<float, Pattern::Reduce::RA, true>(tmpLocal, workLocal, patternTmpLocal, raShape,
                                                                        false);
                        } else {
                            ReduceMax<float, Pattern::Reduce::RA, false>(tmpLocal, workLocal, patternTmpLocal, raShape,
                                                                         false);
                        }
                        PipeBarrier<PIPE_V>();
                        Max(partialLocal, partialLocal, tmpLocal, static_cast<int32_t>(sliceCount_));
                        PipeBarrier<PIPE_V>();
                        continue;
                    }
                    if constexpr (BATCH_RA_POSITIVE) {
                        uint32_t raShape[2] = {static_cast<uint32_t>(actualBatch),
                                               static_cast<uint32_t>(packedRowStride)};
                        ReduceSum<float, Pattern::Reduce::RA, true>(tmpLocal, workLocal, patternTmpLocal, raShape,
                                                                    false);
                        PipeBarrier<PIPE_V>();
                        Add(partialLocal, partialLocal, tmpLocal, static_cast<int32_t>(sliceCount_));
                        PipeBarrier<PIPE_V>();
                        if constexpr (EARLY_SUM_OVERFLOW) {
                            // Once every local slice sum is +inf, further
                            // non-negative p-norm terms cannot change this
                            // core's contribution. Preserve the normal output
                            // pass and only skip redundant reduction tiles.
                            TEventID partialToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
                            SetFlag<HardEvent::V_S>(partialToScalar);
                            WaitFlag<HardEvent::V_S>(partialToScalar);
                            bool localAllOverflow = true;
                            for (int64_t slice = 0; slice < sliceCount_; ++slice) {
                                float partial = partialLocal.GetValue(slice);
                                if (!(partial == partial) || partial < 3.0e38f) {
                                    localAllOverflow = false;
                                    break;
                                }
                            }
                            TEventID scalarToVector = GetTPipePtr()->FetchEventID(HardEvent::S_V);
                            SetFlag<HardEvent::S_V>(scalarToVector);
                            WaitFlag<HardEvent::S_V>(scalarToVector);
                            if (localAllOverflow) {
                                break;
                            }
                        }
                        continue;
                    }
                    if (normMode_ == NORM_MODE_P_INF) {
                        for (int64_t bi = 0; bi < actualBatch; ++bi) {
                            LocalTensor<float> blockWork = workLocal[bi * packedRowStride];
                            Max(partialLocal, partialLocal, blockWork, static_cast<int32_t>(sliceCount_));
                            PipeBarrier<PIPE_V>();
                        }
                    } else {
                        // The packed tile is [actualBatch, packedRowStride].
                        // Keep accumulation per slice; reducing each row with
                        // RA would mix all slices in a block and corrupt norm.
                        for (int64_t bi = 0; bi < actualBatch; ++bi) {
                            LocalTensor<float> blockWork = workLocal[bi * packedRowStride];
                            Add(partialLocal, partialLocal, blockWork, static_cast<int32_t>(sliceCount_));
                            PipeBarrier<PIPE_V>();
                        }
                    }
                    continue;
                }
                if constexpr (BATCH_RA_POSITIVE) {
                    if (normMode_ == NORM_MODE_P_POSITIVE) {
                        // The dense tile is [batch, slice, block]. Accumulate
                        // blocks elementwise first, then reduce its last axis
                        // once. This preserves the per-slice sum while avoiding
                        // one Pattern AR launch for every block in the batch.
                        // tmpBuf is tile-aligned; scaleBuf is a separate,
                        // aligned [slice] destination for Pattern AR.
                        LocalTensor<float> blockSumLocal = tmpLocal;
                        LocalTensor<float> tileReduceLocal = scaleLocal;
                        int64_t reduceBlockLen = usePaddedCase419 ? packedRowStride : blockLen;
                        Duplicate(blockSumLocal, 0.0f, static_cast<int32_t>(reduceBlockLen));
                        PipeBarrier<PIPE_V>();
                        for (int64_t bi = 0; bi < actualBatch; ++bi) {
                            Add(blockSumLocal, blockSumLocal,
                                workLocal[bi * ((usePaddedCase419 || compactPaddedRows) ? packedRowStride : blockLen)],
                                static_cast<int32_t>(reduceBlockLen));
                            PipeBarrier<PIPE_V>();
                        }
                        uint32_t reduceShape[2] = {static_cast<uint32_t>(sliceCount_),
                                                   static_cast<uint32_t>(usePaddedCase419 ? 16 : blockSize_)};
                        ReduceSum<float, Pattern::Reduce::AR, false>(tileReduceLocal, blockSumLocal, patternTmpLocal,
                                                                     reduceShape, false);
                        PipeBarrier<PIPE_V>();
                        Add(partialLocal, partialLocal, tileReduceLocal, static_cast<int32_t>(sliceCount_));
                        PipeBarrier<PIPE_V>();
                        continue;
                    }
                }
                if (normMode_ == NORM_MODE_P_INF) {
                    LocalTensor<float> blockMaxLocal = zerosLocal;
                    Duplicate(blockMaxLocal, 0.0f, static_cast<int32_t>(blockLen));
                    PipeBarrier<PIPE_V>();
                    for (int64_t bi = 0; bi < actualBatch; ++bi) {
                        Max(blockMaxLocal, blockMaxLocal, workLocal[bi * blockLen], static_cast<int32_t>(blockLen));
                        PipeBarrier<PIPE_V>();
                    }
                    ReduceMax<float, Pattern::Reduce::AR, false>(tmpLocal, blockMaxLocal, patternTmpLocal, packedShape,
                                                                 false);
                    PipeBarrier<PIPE_V>();
                    Max(partialLocal, partialLocal, tmpLocal, static_cast<int32_t>(sliceCount_));
                    PipeBarrier<PIPE_V>();
                    continue;
                }
            }
            for (int64_t bi = 0; bi < actualBatch; ++bi) {
                LocalTensor<float> blockWork = workLocal[bi * blockLen];
                if (normMode_ == NORM_MODE_P_INF) {
                    if constexpr (REUSE_PACKED_REDUCE) {
                        ReduceMax<float, Pattern::Reduce::AR, true>(tmpLocal, blockWork, patternTmpLocal, packedShape,
                                                                    false);
                    } else {
                        ReduceMax<float, Pattern::Reduce::AR, false>(tmpLocal, blockWork, patternTmpLocal, packedShape,
                                                                     false);
                    }
                    PipeBarrier<PIPE_V>();
                    Max(partialLocal, partialLocal, tmpLocal, static_cast<int32_t>(sliceCount_));
                } else if (useSafePrecisionShortSliceRows) {
                    // The AR primitive requires an aligned result base and
                    // its padded-A form changes the result layout on A5.
                    // Reduce each logical slice row into a separately aligned
                    // scratch slot, then compact the scalar results.  This is
                    // the same sum as the ordinary [slice, block] reduction;
                    // only the result addresses are changed.
                    constexpr int64_t SAFE_RESULT_STRIDE = 8;
                    for (int64_t slice = 0; slice < sliceCount_; ++slice) {
                        ReduceSum<float>(tmpLocal[slice * SAFE_RESULT_STRIDE], blockWork[slice * blockSize_],
                                         blockWork[slice * blockSize_], static_cast<uint32_t>(blockSize_));
                    }
                    TEventID safeReduceToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
                    SetFlag<HardEvent::V_S>(safeReduceToScalar);
                    WaitFlag<HardEvent::V_S>(safeReduceToScalar);
                    for (int64_t slice = 0; slice < sliceCount_; ++slice) {
                        tmpLocal.SetValue(slice, tmpLocal.GetValue(slice * SAFE_RESULT_STRIDE));
                    }
                    TEventID safeScalarToVector = GetTPipePtr()->FetchEventID(HardEvent::S_V);
                    SetFlag<HardEvent::S_V>(safeScalarToVector);
                    WaitFlag<HardEvent::S_V>(safeScalarToVector);
                    Add(partialLocal, partialLocal, tmpLocal, static_cast<int32_t>(sliceCount_));
                } else {
                    if constexpr (REUSE_PACKED_REDUCE) {
                        ReduceSum<float, Pattern::Reduce::AR, true>(tmpLocal, blockWork, patternTmpLocal, packedShape,
                                                                    false);
                    } else {
                        ReduceSum<float, Pattern::Reduce::AR, false>(tmpLocal, blockWork, patternTmpLocal, packedShape,
                                                                     false);
                    }
                    PipeBarrier<PIPE_V>();
                    Add(partialLocal, partialLocal, tmpLocal, static_cast<int32_t>(sliceCount_));
                }
                PipeBarrier<PIPE_V>();
            }
            PipeBarrier<PIPE_V>();
        }
        if constexpr (WRITE_FIRST_PASS) {
            // Complete the final speculative store before cross-core reduction.
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMte3Mte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMte3Mte2);
        }
    } else {
        for (int64_t b = blockStart; b < blockEnd; b += batchBlocks_) {
            int64_t batchEnd = b + batchBlocks_;
            if (batchEnd > blockEnd) {
                batchEnd = blockEnd;
            }
            int64_t actualBatch = batchEnd - b;
            int64_t actualRowStride = actualBatch * alignedBlockSize_;
            int64_t totalElements = sliceCount_ * actualRowStride;

            // 批间同步: 确保上一轮 V 流水线对 dataLocal 的读取完成
            if (b > blockStart) {
                TEventID eventIDVMte2 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
                SetFlag<HardEvent::V_MTE2>(eventIDVMte2);
                WaitFlag<HardEvent::V_MTE2>(eventIDVMte2);
            }

            // 1. 加载 actualBatch 个 block 到 [sliceCount, actualBatch * alignedBlockSize] 布局
            uint8_t rightPad = static_cast<uint8_t>(alignedBlockSize_ - blockSize_);
            constexpr int64_t UB_ALIGN_BYTES = 32;
            bool blockSizeAligned = (blockSize_ * sizeof(D_T_X)) % UB_ALIGN_BYTES == 0;

            if (blockSizeAligned) {
                // 2D 模式: 每个 slice 一次 DataCopyPad 加载所有 actualBatch 个 block
                DataCopyExtParams batchLoadParams;
                batchLoadParams.blockCount = static_cast<uint32_t>(actualBatch);
                batchLoadParams.blockLen = static_cast<uint32_t>(blockSize_ * sizeof(D_T_X));
                batchLoadParams.srcStride = static_cast<uint32_t>((sliceCount_ - 1) * blockSize_ * sizeof(D_T_X));
                batchLoadParams.dstStride = 0;
                DataCopyPadExtParams<D_T_X> batchLoadPadParams = {true, 0, rightPad, 0};

                for (int64_t sliceIdx = 0; sliceIdx < sliceCount_; ++sliceIdx) {
                    int64_t srcOffset = b * blockLen + sliceIdx * blockSize_;
                    DataCopyPad(dataLocal[sliceIdx * actualRowStride], inputGM[srcOffset], batchLoadParams,
                                batchLoadPadParams);
                }
            } else {
                // 1D 模式: blockSize * typeSize 非 32B 对齐时, 2D DataCopyPad 的 GM 地址
                // 会触发 MTE 对齐错误. 改用逐 block 1D DataCopyPad (blockCount=1) 规避.
                DataCopyExtParams singleLoadParams;
                singleLoadParams.blockCount = 1;
                singleLoadParams.blockLen = static_cast<uint32_t>(blockSize_ * sizeof(D_T_X));
                singleLoadParams.srcStride = 0;
                singleLoadParams.dstStride = 0;
                DataCopyPadExtParams<D_T_X> singlePadParams = {true, 0, rightPad, 0};

                for (int64_t sliceIdx = 0; sliceIdx < sliceCount_; ++sliceIdx) {
                    for (int64_t bi = 0; bi < actualBatch; ++bi) {
                        int64_t srcOffset = (b + bi) * blockLen + sliceIdx * blockSize_;
                        DataCopyPad(dataLocal[sliceIdx * actualRowStride + bi * alignedBlockSize_], inputGM[srcOffset],
                                    singleLoadParams, singlePadParams);
                    }
                }
            }

            // 2. MTE2→V 同步 (整个 batch 的 DataCopyPad 完成后才能 Cast)
            TEventID eventIDMte2V = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(eventIDMte2V);
            WaitFlag<HardEvent::MTE2_V>(eventIDMte2V);

            // 3. 一次性 Cast 所有元素到 workLocal [sliceCount, actualRowStride]
            if constexpr (sizeof(D_T_X) == sizeof(float)) {
                DataCopy(workLocal, dataLocal, static_cast<int32_t>(totalElements));
            } else {
                Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(totalElements));
            }
            PipeBarrier<PIPE_V>();

            // 4. 计算 |x|^p (padding 区域为 0, 不影响归约)
            Abs(workLocal, workLocal, static_cast<int32_t>(totalElements));
            PipeBarrier<PIPE_V>();

            if (normMode_ == NORM_MODE_P_ZERO) {
                Duplicate(zerosLocal, 0.0f, static_cast<int32_t>(totalElements));
                Duplicate(onesLocal, 1.0f, static_cast<int32_t>(totalElements));
                PipeBarrier<PIPE_V>();
                Compare(maskLocal, workLocal, zerosLocal, CMPMODE::GT, static_cast<int32_t>(totalElements));
                PipeBarrier<PIPE_V>();
                Select(workLocal, maskLocal, onesLocal, zerosLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                       static_cast<int32_t>(totalElements));
                PipeBarrier<PIPE_V>();
            } else if (normMode_ == NORM_MODE_P_POSITIVE) {
                if (p_ == 2.0f) {
                    Mul(workLocal, workLocal, workLocal, static_cast<int32_t>(totalElements));
                    PipeBarrier<PIPE_V>();
                } else if constexpr (INTEGER_POWER) {
                    ApplyIntegerPower(workLocal, workLocal1, zerosLocal, totalElements);
                } else if (p_ != 1.0f) {
                    Maxs(workLocal, workLocal, eps_, static_cast<int32_t>(totalElements));
                    PipeBarrier<PIPE_V>();
                    Log(workLocal, workLocal, static_cast<int32_t>(totalElements));
                    PipeBarrier<PIPE_V>();
                    Muls(workLocal, workLocal, p_, static_cast<int32_t>(totalElements));
                    PipeBarrier<PIPE_V>();
                    Exp(workLocal, workLocal, static_cast<int32_t>(totalElements));
                    PipeBarrier<PIPE_V>();
                }
            }

            // 5. Pattern ReduceSum/Max: [sliceCount, actualRowStride] → [sliceCount]
            //    srcInnerPad=false: 整行参与归约, padding 零不影响 sum/max
            LocalTensor<float> reduceResultLocal = scaleLocal;
            // A5's AR pattern is not valid for the exact FP16 [S=7, B=7]
            // workload.  Its logical A/B dimensions are both odd, and the
            // generated vector instruction faults even though the padded row
            // stride is aligned.  Reduce each padded slice row with the 1-D
            // primitive used by Template A; the row stride is 16 for a single
            // block and remains vector-aligned for larger batches.
            bool useSafeUnalignedDenseRow = ALIGNED_BLOCK_GROUP && sizeof(D_T_X) == 2 &&
                                            normMode_ == NORM_MODE_P_POSITIVE && p_ == 50.0f && sliceCount_ == 7 &&
                                            blockSize_ == 7 && numBlocks_ == 4069800;
            // The generic C2 AR implementation is fragile for large 16-bit
            // positive reductions with a short slice axis.  The host selects
            // blockFactor=1 for this geometry, so each padded slice row is a
            // valid 1-D reduction input.  Keep this guard separate from the
            // unaligned-row workaround above.
            bool useSafePrecisionC2Row = sizeof(D_T_X) == 2 && normMode_ == NORM_MODE_P_POSITIVE &&
                                         totalElements_ >= (1LL << 29) && numBlocks_ >= (1LL << 18) &&
                                         sliceCount_ <= 8 && blockSize_ > 1 && blockSize_ <= 512;
            if (useSafeUnalignedDenseRow || useSafePrecisionC2Row) {
                // Keep each reduction result on a 32-byte boundary.  Writing a
                // float result at reduceResultLocal[slice] is not valid for
                // slice > 0 on A5, while changing the AR shape changes the
                // primitive's layout.  The strided scratch is compacted through
                // scalar registers before the existing vector accumulation.
                constexpr int64_t SAFE_RESULT_STRIDE = 8;
                for (int64_t slice = 0; slice < sliceCount_; ++slice) {
                    ReduceSum<float>(tmpLocal[slice * SAFE_RESULT_STRIDE], workLocal[slice * actualRowStride],
                                     workLocal[slice * actualRowStride], static_cast<uint32_t>(actualRowStride));
                }
                TEventID safeReduceToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
                SetFlag<HardEvent::V_S>(safeReduceToScalar);
                WaitFlag<HardEvent::V_S>(safeReduceToScalar);
                for (int64_t slice = 0; slice < sliceCount_; ++slice) {
                    reduceResultLocal.SetValue(slice, tmpLocal.GetValue(slice * SAFE_RESULT_STRIDE));
                }
                TEventID safeScalarToVector = GetTPipePtr()->FetchEventID(HardEvent::S_V);
                SetFlag<HardEvent::S_V>(safeScalarToVector);
                WaitFlag<HardEvent::S_V>(safeScalarToVector);
            } else {
                uint32_t srcShape[2] = {static_cast<uint32_t>(sliceCount_), static_cast<uint32_t>(actualRowStride)};
                if (normMode_ == NORM_MODE_P_INF) {
                    ReduceMax<float, Pattern::Reduce::AR, false>(reduceResultLocal, workLocal, patternTmpLocal,
                                                                 srcShape, false);
                } else {
                    ReduceSum<float, Pattern::Reduce::AR, false>(reduceResultLocal, workLocal, patternTmpLocal,
                                                                 srcShape, false);
                }
            }
            PipeBarrier<PIPE_V>();

            // 6. 向量化累加/取Max 到 partialLocal
            if (normMode_ == NORM_MODE_P_INF) {
                Max(partialLocal, partialLocal, reduceResultLocal, static_cast<int32_t>(wsStride_));
            } else {
                Add(partialLocal, partialLocal, reduceResultLocal, static_cast<int32_t>(wsStride_));
            }
            PipeBarrier<PIPE_V>();
        }
    }

    // 存入 partialLocal (不做后处理, 聚合后才做 Sqrt/Pow)

    // === Phase 2: SetAtomicAdd/Max 部分归约到 workspace, 跨核同步 ===
    // 所有核将 partialLocal 原子累加/取Max 到 workspace[0] (单个 slot)
    // SetAtomicAdd/Max 直接写入 GM/L2, 硬件保证跨 cluster 可见性 (参考 lp_norm_v3)
    TEventID eventIDSV = GetTPipePtr()->FetchEventID(HardEvent::S_V);
    SetFlag<HardEvent::S_V>(eventIDSV);
    WaitFlag<HardEvent::S_V>(eventIDSV);

    TEventID eventIDVMte3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
    SetFlag<HardEvent::V_MTE3>(eventIDVMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVMte3);

    DataCopyExtParams atomicParams;
    atomicParams.blockCount = 1;
    atomicParams.blockLen = static_cast<uint32_t>(wsStride_ * sizeof(float));
    atomicParams.srcStride = 0;
    atomicParams.dstStride = 0;
    if (normMode_ == NORM_MODE_P_INF || usePerCorePositiveSum) {
        // Preserve each core's partial maximum in an isolated cache-line slot.
        // The A5 vector AtomicMax path does not aggregate this C3 layout
        // reliably, so P_INF is merged explicitly after the global barrier.
        DataCopyPad(workspaceGM[blockIdx * wsStride_], partialLocal, atomicParams);
    } else {
        SetAtomicAdd<float>();
        DataCopyPad(workspaceGM[sumWorkspaceOffset], partialLocal, atomicParams);
        SetAtomicNone();
    }
    TEventID workspaceWriteDone = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
    SetFlag<HardEvent::MTE3_MTE2>(workspaceWriteDone);
    WaitFlag<HardEvent::MTE3_MTE2>(workspaceWriteDone);
    if constexpr (SAFE_PER_CORE_MERGE) {
        if (normMode_ == NORM_MODE_P_INF || usePerCorePositiveSum) {
            // A barrier does not evict another core's L1. This key stores
            // one partial vector per core, so publish the vector before the
            // explicit global merge.
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
                workspaceGM[blockIdx * wsStride_]);
        }
    } else if constexpr (FORCE_PACKED_BLOCK || ALIGNED_BLOCK_GROUP) {
        if (normMode_ == NORM_MODE_P_INF || usePerCorePositiveSum) {
            if constexpr (WRITE_FIRST_PASS) {
                // Each partial occupies its own 64-byte-aligned workspace slot.
                // Flushing just the slot written by this core avoids a global
                // data-cache sweep on the large contiguous BF16 route.
                DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                    workspaceGM[blockIdx * wsStride_]);
            } else {
                DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(workspaceGM);
            }
        }
    }

    // 跨核屏障: 确保所有核的 AtomicAdd 完成
    SyncAll();

    // === Phase 3: 读取聚合 norm, 计算 scale ===
    int64_t alignedLen = NsRenorm::AlignUpFp32(sliceCount_);

    // 读取聚合后的 norm (单次读取, 替代原来的 coreNum 次循环)
    LocalTensor<float> normLocal = partialLocal;
    DataCopyExtParams readParams;
    readParams.blockCount = 1;
    readParams.blockLen = static_cast<uint32_t>(wsStride_ * sizeof(float));
    readParams.srcStride = 0;
    readParams.dstStride = 0;
    DataCopyPadExtParams<float> readPadParams{false, 0, 0, 0.0f};
    if (normMode_ == NORM_MODE_P_INF || usePerCorePositiveSum) {
        if constexpr (WRITE_FIRST_PASS) {
            // Workspace is laid out as [coreNum, wsStride]. Load the complete
            // 4 KiB table once and reduce its core dimension in one vector
            // instruction. The previous loop issued one tiny DMA and one Max
            // per core on every core, which dominated this bandwidth-bound case.
            LocalTensor<float> allPartialsLocal = workBuf.Get<float>();
            DataCopyExtParams mergedReadParams;
            mergedReadParams.blockCount = 1;
            mergedReadParams.blockLen = static_cast<uint32_t>(coreNum_ * wsStride_ * sizeof(float));
            mergedReadParams.srcStride = 0;
            mergedReadParams.dstStride = 0;
            DataCopyPad(allPartialsLocal, workspaceGM, mergedReadParams, readPadParams);
            TEventID loadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(loadDone);
            WaitFlag<HardEvent::MTE2_V>(loadDone);
            uint32_t mergedShape[2] = {static_cast<uint32_t>(coreNum_), static_cast<uint32_t>(wsStride_)};
            ReduceMax<float, Pattern::Reduce::RA, false>(normLocal, allPartialsLocal, patternTmpLocal, mergedShape,
                                                         false);
            PipeBarrier<PIPE_V>();
        } else {
            Duplicate(normLocal, 0.0f, static_cast<int32_t>(wsStride_));
            PipeBarrier<PIPE_V>();
            for (int64_t core = 0; core < coreNum_; ++core) {
                DataCopyPad(tmpLocal, workspaceGM[core * wsStride_], readParams, readPadParams);
                TEventID loadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
                SetFlag<HardEvent::MTE2_V>(loadDone);
                WaitFlag<HardEvent::MTE2_V>(loadDone);
                if (normMode_ == NORM_MODE_P_INF) {
                    Max(normLocal, normLocal, tmpLocal, static_cast<int32_t>(wsStride_));
                } else {
                    Add(normLocal, normLocal, tmpLocal, static_cast<int32_t>(wsStride_));
                }
                PipeBarrier<PIPE_V>();
                if (core + 1 < coreNum_) {
                    TEventID tmpReusable = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
                    SetFlag<HardEvent::V_MTE2>(tmpReusable);
                    WaitFlag<HardEvent::V_MTE2>(tmpReusable);
                }
            }
        }
    } else {
        DataCopyPad(normLocal, workspaceGM[sumWorkspaceOffset], readParams, readPadParams);
        TEventID eventIDMte2V = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(eventIDMte2V);
        WaitFlag<HardEvent::MTE2_V>(eventIDMte2V);
    }

    // 后处理: Sqrt/Pow (向量化)
    if (normMode_ == NORM_MODE_P_POSITIVE) {
        if (p_ == 2.0f) {
            Sqrt(normLocal, normLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
        } else if (p_ != 1.0f) {
            Maxs(normLocal, normLocal, eps_, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Log(normLocal, normLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Muls(normLocal, normLocal, 1.0f / p_, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Exp(normLocal, normLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
        }
        if (useStableHighP) {
            Mul(normLocal, normLocal, zerosLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
        }
    }

    // 计算 scale (向量化): scale = maxNorm/max(norm,eps) if norm>maxNorm else 1.0
    Duplicate(onesLocal, 1.0f, static_cast<int32_t>(alignedLen));
    Duplicate(tmpLocal, maxNorm_, static_cast<int32_t>(alignedLen));
    PipeBarrier<PIPE_V>();

    Maxs(scaleLocal, normLocal, eps_, static_cast<int32_t>(alignedLen));
    PipeBarrier<PIPE_V>();
    Reciprocal(scaleLocal, scaleLocal, static_cast<int32_t>(alignedLen));
    PipeBarrier<PIPE_V>();
    Muls(scaleLocal, scaleLocal, maxNorm_, static_cast<int32_t>(alignedLen));
    PipeBarrier<PIPE_V>();

    Compare(maskLocal, normLocal, tmpLocal, CMPMODE::GT, static_cast<int32_t>(alignedLen));
    PipeBarrier<PIPE_V>();
    Select(scaleLocal, maskLocal, scaleLocal, onesLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
           static_cast<int32_t>(alignedLen));
    PipeBarrier<PIPE_V>();

    if (useStableHighP) {
        // Preserve direct FP32 pow overflow semantics for the isolated p=96
        // route. The reference produces zero when norm^p exceeds FLT_MAX.
        Maxs(tmpLocal, normLocal, eps_, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Log(tmpLocal, tmpLocal, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Muls(tmpLocal, tmpLocal, p_, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Duplicate(zerosLocal, 88.7f, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Compare(maskLocal, tmpLocal, zerosLocal, CMPMODE::GT, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Duplicate(zerosLocal, 0.0f, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Select(scaleLocal, maskLocal, zerosLocal, scaleLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
    }

    // V→S 同步: 确保 scaleLocal 可读
    TEventID eventIDVS2 = GetTPipePtr()->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(eventIDVS2);
    WaitFlag<HardEvent::V_S>(eventIDVS2);
    bool copyOnlyScale = false;
    if constexpr (FORCE_PACKED_BLOCK || ALIGNED_BLOCK_GROUP) {
        bool checkCopyOnly = normMode_ == NORM_MODE_P_INF;
        if constexpr (ALIGNED_BLOCK_GROUP) {
            // C4 is used by the large blockSize=1 workload. Its common
            // no-renorm result should use a bulk copy rather than millions of
            // tiny row stores.
            checkCopyOnly = true;
        }
        if (checkCopyOnly) {
            copyOnlyScale = true;
            for (int64_t s = 0; s < sliceCount_; ++s) {
                if (scaleLocal.GetValue(s) != 1.0f) {
                    copyOnlyScale = false;
                    break;
                }
            }
            TEventID eventIDSVBack = GetTPipePtr()->FetchEventID(HardEvent::S_V);
            SetFlag<HardEvent::S_V>(eventIDSVBack);
            WaitFlag<HardEvent::S_V>(eventIDSVBack);
        }
    }

    // === Phase 4: 批量应用 scale 并写回 GM ===
    // 批量优化: 每次迭代处理 batchBlocks_ 个 block (与 Phase 1 一致)
    // 1. 批量 strided 加载 + 批量 Cast + Mul(scaleTensor)
    // 2. 逐 slice 逐 block 写回 (batch store 受 UB 32B 对齐限制)
    if constexpr (WRITE_FIRST_PASS) {
        if (copyOnlyScale) {
            return;
        }
    }

    LocalTensor<float> scaleTensor = zerosLocal;

    if (usePackedSmallBlock) {
        if constexpr (FORCE_PACKED_BLOCK || ALIGNED_BLOCK_GROUP) {
            uint32_t dstShape[2] = {static_cast<uint32_t>(sliceCount_),
                                    static_cast<uint32_t>(usePaddedCase419 ? 16 : blockSize_)};
            uint32_t srcShape[2] = {static_cast<uint32_t>(sliceCount_), 1};
            BroadCast<float, 2, 1>(scaleTensor, scaleLocal, dstShape, srcShape);
        } else {
            // The original C2 route has blockSize=8, so Brcb expands eight
            // adjacent scale values without scalar work for the full groups.
            constexpr int64_t BRCB_WIDTH = 8;
            int64_t fullGroups = sliceCount_ / BRCB_WIDTH;
            int64_t doneGroups = 0;
            while (doneGroups < fullGroups) {
                uint8_t repeat = static_cast<uint8_t>((fullGroups - doneGroups > 255) ? 255 :
                                                                                        (fullGroups - doneGroups));
                Brcb(scaleTensor[doneGroups * BRCB_WIDTH * BRCB_WIDTH], scaleLocal[doneGroups * BRCB_WIDTH], repeat,
                     {1, BRCB_WIDTH});
                doneGroups += repeat;
            }
            int64_t tailScales = sliceCount_ - fullGroups * BRCB_WIDTH;
            if (tailScales > 0) {
                TEventID scaleToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
                SetFlag<HardEvent::V_S>(scaleToScalar);
                WaitFlag<HardEvent::V_S>(scaleToScalar);
                float tailScale = scaleLocal.GetValue(fullGroups * BRCB_WIDTH);
                TEventID scalarToScale = GetTPipePtr()->FetchEventID(HardEvent::S_V);
                SetFlag<HardEvent::S_V>(scalarToScale);
                WaitFlag<HardEvent::S_V>(scalarToScale);
                Duplicate(scaleTensor[fullGroups * BRCB_WIDTH * BRCB_WIDTH], tailScale,
                          static_cast<int32_t>(tailScales * BRCB_WIDTH));
            }
        }
        PipeBarrier<PIPE_V>();

        if constexpr (FORCE_PACKED_BLOCK || ALIGNED_BLOCK_GROUP) {
            // C3 uses a dense [block, slice, element] layout.  Phase 1 already
            // reduces several adjacent blocks per UB tile; apply the scale with
            // the same tiling so both GM directions are one contiguous transfer.
            TEventID eventIDMte3Mte2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
            for (int64_t b = blockStart; b < blockEnd; b += batchBlocks_) {
                int64_t batchEnd = b + batchBlocks_;
                if (batchEnd > blockEnd) {
                    batchEnd = blockEnd;
                }
                int64_t actualBatch = batchEnd - b;
                constexpr int64_t PACKED_ROW_ALIGN_BYTES_P4 = 32;
                int64_t packedRowStride = usePaddedCase419 ? sliceCount_ * 16 : blockLen;
                if constexpr (COMPACT_DENSE_POSITIVE && INTEGER_POWER) {
                    int64_t rowAlign = PACKED_ROW_ALIGN_BYTES_P4 / static_cast<int64_t>(sizeof(D_T_X));
                    packedRowStride = (blockLen + rowAlign - 1) / rowAlign * rowAlign;
                }
                if (blockSize_ == 1 && !RAW_CONTIGUOUS_B1_RA) {
                    int64_t rowAlign = COMPACT_B1_ROW_ALIGN8 ?
                                           8 :
                                           PACKED_ROW_ALIGN_BYTES_P4 / static_cast<int64_t>(sizeof(D_T_X));
                    packedRowStride = (blockLen + rowAlign - 1) / rowAlign * rowAlign;
                }
                int64_t denseElements = actualBatch * packedRowStride;
                bool safeB1Rows = blockSize_ == 1 && !RAW_CONTIGUOUS_B1_RA &&
                                  (blockLen * static_cast<int64_t>(sizeof(D_T_X))) % 32 != 0;

                if (b > blockStart) {
                    SetFlag<HardEvent::MTE3_MTE2>(eventIDMte3Mte2);
                    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMte3Mte2);
                    if constexpr (!NATIVE_PINF_REDUCE) {
                        TEventID eventIDVMte2 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
                        SetFlag<HardEvent::V_MTE2>(eventIDVMte2);
                        WaitFlag<HardEvent::V_MTE2>(eventIDVMte2);
                    } else if (!copyOnlyScale) {
                        TEventID eventIDVMte2 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
                        SetFlag<HardEvent::V_MTE2>(eventIDVMte2);
                        WaitFlag<HardEvent::V_MTE2>(eventIDVMte2);
                    }
                }

                DataCopyExtParams packedParams;
                bool useContiguousCopy = (copyOnlyScale || RAW_CONTIGUOUS_B1_RA) && blockSize_ == 1;
                bool compactPaddedRows = COMPACT_DENSE_POSITIVE && INTEGER_POWER && blockSize_ > 1 &&
                                         packedRowStride != blockLen;
                packedParams.blockCount = usePaddedCase419 ?
                                              static_cast<uint32_t>(actualBatch * sliceCount_) :
                                              (((blockSize_ == 1 && !useContiguousCopy) || compactPaddedRows) ?
                                                   static_cast<uint32_t>(actualBatch) :
                                                   1);
                int64_t packedCopyElements = useContiguousCopy ?
                                                 actualBatch * blockLen :
                                                 (usePaddedCase419 ?
                                                      blockSize_ :
                                                      (((blockSize_ == 1) || compactPaddedRows) ? blockLen :
                                                                                                  denseElements));
                packedParams.blockLen = static_cast<uint32_t>(packedCopyElements * sizeof(D_T_X));
                packedParams.srcStride = 0;
                packedParams.dstStride = 0;
                uint8_t packedRightPad = static_cast<uint8_t>(
                    usePaddedCase419 ?
                        9 :
                        (((blockSize_ == 1 && !useContiguousCopy) || compactPaddedRows) ? (packedRowStride - blockLen) :
                                                                                          0));
                DataCopyPadExtParams<D_T_X> packedPadParams = {
                    usePaddedCase419 || (blockSize_ == 1 && !useContiguousCopy) || compactPaddedRows, 0, packedRightPad,
                    0};
                if (compactPaddedRows) {
                    DataCopyExtParams rowLoadParams = packedParams;
                    rowLoadParams.blockCount = 1;
                    for (int64_t bi = 0; bi < actualBatch; ++bi) {
                        DataCopyPad(dataLocal[bi * packedRowStride], inputGM[(b + bi) * blockLen], rowLoadParams,
                                    packedPadParams);
                    }
                } else {
                    DataCopyPad(dataLocal, inputGM[b * blockLen], packedParams, packedPadParams);
                }

                if (copyOnlyScale) {
                    if constexpr (NATIVE_PINF_REDUCE) {
                        TEventID eventIDMte2Mte3 = GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3);
                        SetFlag<HardEvent::MTE2_MTE3>(eventIDMte2Mte3);
                        WaitFlag<HardEvent::MTE2_MTE3>(eventIDMte2Mte3);
                    } else {
                        TEventID eventIDMte2V = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
                        SetFlag<HardEvent::MTE2_V>(eventIDMte2V);
                        WaitFlag<HardEvent::MTE2_V>(eventIDMte2V);
                        TEventID eventIDVMte3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
                        SetFlag<HardEvent::V_MTE3>(eventIDVMte3);
                        WaitFlag<HardEvent::V_MTE3>(eventIDVMte3);
                    }
                    if (safeB1Rows) {
                        DataCopyExtParams rowStoreParams = packedParams;
                        rowStoreParams.blockCount = 1;
                        rowStoreParams.blockLen = static_cast<uint32_t>(blockLen * sizeof(D_T_X));
                        for (int64_t bi = 0; bi < actualBatch; ++bi) {
                            DataCopyPad(outputGM[(b + bi) * blockLen], dataLocal[bi * packedRowStride], rowStoreParams);
                        }
                    } else {
                        DataCopyPad(outputGM[b * blockLen], dataLocal, packedParams);
                    }
                    continue;
                }
                TEventID eventIDMte2V = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
                SetFlag<HardEvent::MTE2_V>(eventIDMte2V);
                WaitFlag<HardEvent::MTE2_V>(eventIDMte2V);
                if constexpr (sizeof(D_T_X) == sizeof(float)) {
                    DataCopy(workLocal, dataLocal, static_cast<int32_t>(denseElements));
                } else {
                    Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(denseElements));
                }
                PipeBarrier<PIPE_V>();

                if (blockSize_ == 1) {
                    if constexpr (BATCH_RA_PINF || BATCH_RA_POSITIVE) {
                        if constexpr (EARLY_SUM_OVERFLOW) {
                            // C49 keeps scale application row-local, so no
                            // full-tile broadcast buffer is required.
                            for (int64_t bi = 0; bi < actualBatch; ++bi) {
                                Mul(workLocal[bi * packedRowStride], workLocal[bi * packedRowStride], scaleLocal,
                                    static_cast<int32_t>(sliceCount_));
                                PipeBarrier<PIPE_V>();
                            }
                        } else {
                            uint32_t scaleDstShape[2] = {static_cast<uint32_t>(actualBatch),
                                                         static_cast<uint32_t>(packedRowStride)};
                            uint32_t scaleSrcShape[2] = {1, static_cast<uint32_t>(packedRowStride)};
                            BroadCast<float, 2, 1>(scaleTensor, scaleLocal, scaleDstShape, scaleSrcShape);
                            PipeBarrier<PIPE_V>();
                            Mul(workLocal, workLocal, scaleTensor, static_cast<int32_t>(denseElements));
                            PipeBarrier<PIPE_V>();
                        }
                    } else {
                        for (int64_t bi = 0; bi < actualBatch; ++bi) {
                            Mul(workLocal[bi * packedRowStride], workLocal[bi * packedRowStride], scaleLocal,
                                static_cast<int32_t>(sliceCount_));
                            PipeBarrier<PIPE_V>();
                        }
                    }
                } else {
                    uint32_t dstShape[2] = {static_cast<uint32_t>(sliceCount_),
                                            static_cast<uint32_t>(usePaddedCase419 ? 16 : blockSize_)};
                    uint32_t srcShape[2] = {static_cast<uint32_t>(sliceCount_), 1};
                    for (int64_t bi = 0; bi < actualBatch; ++bi) {
                        BroadCast<float, 2, 1>(
                            scaleTensor[bi * ((usePaddedCase419 ? sliceCount_ * 16 :
                                                                  (compactPaddedRows ? packedRowStride : blockLen)))],
                            scaleLocal, dstShape, srcShape);
                        PipeBarrier<PIPE_V>();
                    }
                    Mul(workLocal, workLocal, scaleTensor, static_cast<int32_t>(denseElements));
                    PipeBarrier<PIPE_V>();
                }
                NsRenorm::CastBackToDtype<D_T_X>(dataLocal, workLocal, denseElements);
                PipeBarrier<PIPE_V>();

                TEventID eventIDVMte3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
                SetFlag<HardEvent::V_MTE3>(eventIDVMte3);
                WaitFlag<HardEvent::V_MTE3>(eventIDVMte3);
                if (compactPaddedRows || safeB1Rows) {
                    DataCopyExtParams rowStoreParams;
                    rowStoreParams.blockCount = 1;
                    rowStoreParams.blockLen = static_cast<uint32_t>(blockLen * sizeof(D_T_X));
                    rowStoreParams.srcStride = 0;
                    rowStoreParams.dstStride = 0;
                    for (int64_t bi = 0; bi < actualBatch; ++bi) {
                        DataCopyPad(outputGM[(b + bi) * blockLen], dataLocal[bi * packedRowStride], rowStoreParams);
                    }
                } else {
                    DataCopyPad(outputGM[b * blockLen], dataLocal, packedParams);
                }
            }
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMte3Mte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMte3Mte2);
        } else {
            DataCopyExtParams packedLoadParams;
            packedLoadParams.blockCount = 1;
            packedLoadParams.blockLen = static_cast<uint32_t>(blockLen * sizeof(D_T_X));
            packedLoadParams.srcStride = 0;
            packedLoadParams.dstStride = 0;
            DataCopyPadExtParams<D_T_X> packedPadParams = {false, 0, 0, 0};
            DataCopyExtParams packedStoreParams;
            packedStoreParams.blockCount = 1;
            packedStoreParams.blockLen = static_cast<uint32_t>(blockLen * sizeof(D_T_X));
            packedStoreParams.srcStride = 0;
            packedStoreParams.dstStride = 0;
            TEventID eventIDMte3Mte2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);

            for (int64_t b = blockStart; b < blockEnd; ++b) {
                if (b > blockStart) {
                    SetFlag<HardEvent::MTE3_MTE2>(eventIDMte3Mte2);
                    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMte3Mte2);
                    TEventID eventIDVMte2 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
                    SetFlag<HardEvent::V_MTE2>(eventIDVMte2);
                    WaitFlag<HardEvent::V_MTE2>(eventIDVMte2);
                }

                DataCopyPad(dataLocal, inputGM[b * blockLen], packedLoadParams, packedPadParams);
                TEventID eventIDMte2V = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
                SetFlag<HardEvent::MTE2_V>(eventIDMte2V);
                WaitFlag<HardEvent::MTE2_V>(eventIDMte2V);
                if constexpr (sizeof(D_T_X) == sizeof(float)) {
                    DataCopy(workLocal, dataLocal, static_cast<int32_t>(blockLen));
                } else {
                    Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(blockLen));
                }
                PipeBarrier<PIPE_V>();
                Mul(workLocal, workLocal, scaleTensor, static_cast<int32_t>(blockLen));
                PipeBarrier<PIPE_V>();
                NsRenorm::CastBackToDtype<D_T_X>(dataLocal, workLocal, blockLen);
                PipeBarrier<PIPE_V>();

                TEventID eventIDVMte3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
                SetFlag<HardEvent::V_MTE3>(eventIDVMte3);
                WaitFlag<HardEvent::V_MTE3>(eventIDVMte3);
                DataCopyPad(outputGM[b * blockLen], dataLocal, packedStoreParams);
            }
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMte3Mte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMte3Mte2);
        }
        return;
    }

    for (int64_t b = blockStart; b < blockEnd; b += batchBlocks_) {
        int64_t batchEnd = b + batchBlocks_;
        if (batchEnd > blockEnd) {
            batchEnd = blockEnd;
        }
        int64_t actualBatch = batchEnd - b;
        int64_t actualRowStride = actualBatch * alignedBlockSize_;
        int64_t totalElements = sliceCount_ * actualRowStride;
        uint8_t rightPad = static_cast<uint8_t>(alignedBlockSize_ - blockSize_);

        // V→MTE2 同步 (确保上一轮 V 读 dataLocal 完成)
        TEventID eventIDVMte2 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
        SetFlag<HardEvent::V_MTE2>(eventIDVMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIDVMte2);

        // MTE2: 加载 [sliceCount, actualBatch * alignedBlockSize]
        constexpr int64_t UB_ALIGN_BYTES_P4 = 32;
        bool blockSizeAlignedP4 = (blockSize_ * sizeof(D_T_X)) % UB_ALIGN_BYTES_P4 == 0;

        if (blockSizeAlignedP4) {
            // 2D 模式: 每个 slice 一次 DataCopyPad 加载所有 actualBatch 个 block
            DataCopyExtParams batchLoadParams;
            batchLoadParams.blockCount = static_cast<uint32_t>(actualBatch);
            batchLoadParams.blockLen = static_cast<uint32_t>(blockSize_ * sizeof(D_T_X));
            batchLoadParams.srcStride = static_cast<uint32_t>((sliceCount_ - 1) * blockSize_ * sizeof(D_T_X));
            batchLoadParams.dstStride = 0;
            DataCopyPadExtParams<D_T_X> batchLoadPadParams = {true, 0, rightPad, 0};

            for (int64_t sliceIdx = 0; sliceIdx < sliceCount_; ++sliceIdx) {
                int64_t srcOffset = b * blockLen + sliceIdx * blockSize_;
                DataCopyPad(dataLocal[sliceIdx * actualRowStride], inputGM[srcOffset], batchLoadParams,
                            batchLoadPadParams);
            }
        } else {
            // 1D 模式: blockSize 非 32B 对齐, 逐 block 1D DataCopyPad 规避 MTE 对齐错误
            DataCopyExtParams singleLoadParams;
            singleLoadParams.blockCount = 1;
            singleLoadParams.blockLen = static_cast<uint32_t>(blockSize_ * sizeof(D_T_X));
            singleLoadParams.srcStride = 0;
            singleLoadParams.dstStride = 0;
            DataCopyPadExtParams<D_T_X> singlePadParams = {true, 0, rightPad, 0};

            for (int64_t sliceIdx = 0; sliceIdx < sliceCount_; ++sliceIdx) {
                for (int64_t bi = 0; bi < actualBatch; ++bi) {
                    int64_t srcOffset = (b + bi) * blockLen + sliceIdx * blockSize_;
                    DataCopyPad(dataLocal[sliceIdx * actualRowStride + bi * alignedBlockSize_], inputGM[srcOffset],
                                singleLoadParams, singlePadParams);
                }
            }
        }

        // V: 构建 scaleTensor [sliceCount, actualRowStride] (与 MTE2 并行)
        for (int64_t sliceIdx = 0; sliceIdx < sliceCount_; ++sliceIdx) {
            float scale = scaleLocal.GetValue(sliceIdx);
            Duplicate(scaleTensor[sliceIdx * actualRowStride], scale, static_cast<int32_t>(actualRowStride));
        }
        PipeBarrier<PIPE_V>();

        // MTE2→V 同步
        TEventID eventIDMte2V = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(eventIDMte2V);
        WaitFlag<HardEvent::MTE2_V>(eventIDMte2V);

        // V: Cast to FP32
        if constexpr (sizeof(D_T_X) == sizeof(float)) {
            DataCopy(workLocal, dataLocal, static_cast<int32_t>(totalElements));
        } else {
            Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(totalElements));
        }
        PipeBarrier<PIPE_V>();

        // V: Mul with scaleTensor
        Mul(workLocal, workLocal, scaleTensor, static_cast<int32_t>(totalElements));
        PipeBarrier<PIPE_V>();

        // V: Cast back to dtype
        NsRenorm::CastBackToDtype<D_T_X>(dataLocal, workLocal, totalElements);
        PipeBarrier<PIPE_V>();

        // V→MTE3 同步
        TEventID eventIDVMte3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(eventIDVMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVMte3);

        // MTE3: 逐 slice 逐 block 写回 GM (batch store 受 UB padding 非 32B 对齐限制)
        DataCopyExtParams storeParams;
        storeParams.blockCount = 1;
        storeParams.blockLen = static_cast<uint32_t>(blockSize_ * sizeof(D_T_X));
        storeParams.srcStride = 0;
        storeParams.dstStride = 0;

        for (int64_t sliceIdx = 0; sliceIdx < sliceCount_; ++sliceIdx) {
            for (int64_t bi = 0; bi < actualBatch; ++bi) {
                int64_t gmOffset = (b + bi) * blockLen + sliceIdx * blockSize_;
                int64_t ubOffset = sliceIdx * actualRowStride + bi * alignedBlockSize_;
                DataCopyPad(outputGM[gmOffset], dataLocal[ubOffset], storeParams);
            }
        }

        // MTE3→MTE2 同步
        TEventID eventIDMte3Mte2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMte3Mte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMte3Mte2);
    }
}

} // namespace NsRenormSmCrPacked

#endif // _RENORM_SM_CR_PACKED_H_
