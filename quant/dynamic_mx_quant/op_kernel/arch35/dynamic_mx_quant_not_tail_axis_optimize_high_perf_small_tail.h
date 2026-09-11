/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_mx_quant_not_tail_axis_optimize_high_perf_small_tail.h
 * \brief Small-tail (postAxisSize <= nAlignNum) high-perf template.
 *
 * Reimplemented using the large_tail.h framework (ProcessOneTask / CopyIn / ComputeAll /
 * CopyOut + ComputeScaleOcp + ComputeYVf). Because the post-axis is small, multiple rows
 * of one block are reduced via a binary-search max (referencing not_tail_axis_optimize.h).
 *
 * One block is processed per ComputeScaleOcp call; adjacent block scales are interleaved
 * into the packed mxscale output at the end of ComputeAll.
 *
 * NOTE: This first cut only implements the BF16 OCP (calcMode == MODE_ZERO) path. Other
 * dtype / calcMode combinations are guarded by `if constexpr` and early-return.
 */

#ifndef DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_OPTIMIZE_HIGH_PERF_SMALL_TAIL_H
#define DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_OPTIMIZE_HIGH_PERF_SMALL_TAIL_H

#include "dynamic_mx_quant_common.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "../inc/kernel_utils.h"

namespace DynamicMxQuant {
using namespace AscendC;

// Keeps the fp32->fp4 range-fit instruction sequence out of template-class member
// scope (bisheng backend "Unsupported Inst must be hoisted" on fp16->e1m2 otherwise).
template <RoundMode RM, bool IS_E1M2>
__attribute__((noinline)) __aicore__ void Fp32ToFp4RangeFit(Reg::RegTensor<float>& in)
{
    Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg zeroMask;
    Reg::MaskReg specialMask;
    Reg::MaskReg negInfMask;

    Reg::RegTensor<int32_t> negZero;
    Reg::RegTensor<int32_t> maxExpFP32;
    Reg::RegTensor<int32_t> exp0FP32;
    Reg::RegTensor<int32_t> exp1FP32;

    Reg::Duplicate(negZero, FP32_NEG_ZERO_BITS);
    Reg::Compare<int32_t, CMPMODE::EQ>(negInfMask, (Reg::RegTensor<int32_t>&)in, negZero, pregAll32);
    if constexpr (IS_E1M2) {
        Reg::Muls(in, in, FP4_SCALE_FACTOR, pregAll32);
        Reg::Compares<float, CMPMODE::LT>(specialMask, in, 0, pregAll32);
        Reg::Truncate<float, RM>(in, in, pregAll32);
        Reg::Muls(in, in, FP4_INV_SCALE_FACTOR, pregAll32);
    } else {
        Reg::Duplicate(maxExpFP32, FP32_MX_MAX_EXP);
        Reg::And(exp0FP32, (Reg::RegTensor<int32_t>&)in, maxExpFP32, pregAll32);
        Reg::ShiftRights(exp0FP32, exp0FP32, FP32_SHR_NUM, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, FP32_BIAS_NEG_VALUE, pregAll32);
        Reg::Maxs(exp0FP32, exp0FP32, 0, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, FP32_NEG_ONE, pregAll32);
        Reg::Muls(exp1FP32, exp0FP32, FP32_NEG_ONE, pregAll32);
        Reg::Adds(exp1FP32, exp1FP32, FP32_BIAS_VALUE, pregAll32);
        Reg::ShiftLefts(exp1FP32, exp1FP32, FP32_SHR_NUM, pregAll32);
        Reg::Mul(in, in, (Reg::RegTensor<float>&)exp1FP32, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, FP32_BIAS_VALUE, pregAll32);
        Reg::ShiftLefts(exp0FP32, exp0FP32, FP32_SHR_NUM, pregAll32);
        Reg::Compares<float, CMPMODE::LT>(specialMask, in, 0, pregAll32);
        Reg::Truncate<float, RM>(in, in, pregAll32);
        Reg::Mul(in, in, (Reg::RegTensor<float>&)exp0FP32, pregAll32);
    }
    Reg::Compares<float, CMPMODE::EQ>(zeroMask, in, 0, pregAll32);
    Reg::And(zeroMask, specialMask, zeroMask, pregAll32);
    Reg::Or(zeroMask, negInfMask, zeroMask, pregAll32);
    Reg::Select<int32_t>((Reg::RegTensor<int32_t>&)in, negZero, (Reg::RegTensor<int32_t>&)in, zeroMask);
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
class DynamicMxQuantNotTailAxisOptimizeSmallTail {
public:
    __aicore__ inline DynamicMxQuantNotTailAxisOptimizeSmallTail(){};
    __aicore__ inline void Init(TPipe* pipe, GM_ADDR x, GM_ADDR y, GM_ADDR mxScale,
                                const DynamicMxQuant4OptimizeTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const DynamicMxQuant4OptimizeTilingData* tilingData);
    __aicore__ inline void ProcessOneTask(int64_t blockIdx, int64_t blockCount);
    __aicore__ inline void CopyIn(int64_t offset, int64_t count);
    __aicore__ inline void ComputeAll(int64_t offset, int64_t count);
    __aicore__ inline void CopyOut(int64_t offset, int64_t count);

    // One-block scale computation (OCP / CeilAlg / CeilAlgOptimize, multi-row binary-search max).
    __aicore__ inline void ComputeScaleOcp(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                           __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint16_t* tmpAddr);
    __aicore__ inline void ComputeScaleOcpBf16(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                               __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint16_t* tmpAddr);
    __aicore__ inline void ComputeScaleOcpHalf(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                               __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint16_t* tmpAddr);
    __aicore__ inline void ComputeScaleOcpFp32(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                               __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint16_t* tmpAddr);
    __aicore__ inline void ComputeScaleCeilAlg(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                               __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint16_t* tmpAddr);
    __aicore__ inline void ComputeScaleCeilAlgBf16(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                                   __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint16_t* tmpAddr);
    __aicore__ inline void ComputeScaleCeilAlgHalf(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                                   __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint16_t* tmpAddr);
    __aicore__ inline void ComputeScaleCeilAlgFp32(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                                   __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint16_t* tmpAddr);
    // One-block Y computation.
    __aicore__ inline void ComputeYVf(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                      __ubuf__ uint16_t* tmpAddr, __ubuf__ uint8_t* yAddr);
    __aicore__ inline void ComputeYFromBf16(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                            __ubuf__ uint16_t* tmpAddr, __ubuf__ uint8_t* yAddr);
    __aicore__ inline void ComputeYFromHalf(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                            __ubuf__ uint16_t* tmpAddr, __ubuf__ uint8_t* yAddr);
    __aicore__ inline void ComputeYFromFp32(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                            __ubuf__ uint16_t* tmpAddr, __ubuf__ uint8_t* yAddr);
    // FP4-from-FP32 helper (shared by FP16 and FP32 Y paths).
    __aicore__ inline void ComputeFP4FromFp32(Reg::RegTensor<float>& in);
    // FP32 pre-processing for FP4 quantization (shared by FP16 and FP32 Y paths).
    __aicore__ inline void PreProcessFP32(Reg::RegTensor<float>& in);

    struct LoopParams {
        uint16_t N;
        uint16_t dataLenSingleLoop;
        uint16_t regLoop;
        uint16_t tailN;
        uint16_t dataLenTailLoop;
        uint16_t loopSize;
        uint16_t rowsPow2;
        uint16_t expOffsetInit;
        uint32_t loopNum0;
        uint32_t loopNum1;
        uint32_t tailLoopNum0;
        uint32_t tailLoopNum1;
    };
    __aicore__ inline LoopParams ComputeLoopParams(uint16_t dataLen, uint16_t blockCount);

private:
    // tiling data
    const DynamicMxQuant4OptimizeTilingData* tilingData_{nullptr};

    // pipe & queue & buf
    TPipe* pipe_{nullptr};
    TQue<QuePosition::VECIN, DB_BUFFER> inQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> mxScaleQueue_;
    TBuf<TPosition::VECCALC> tmpScaleBuf_; // per-block scales before interleave
    TBuf<TPosition::VECCALC> tmpBuf_;      // 1/scale (reciprocal) per block
    TBuf<TPosition::VECCALC> maxExpBuf_;   // scratch for binary-search reduce

    // gm
    GlobalTensor<xDtype> xGm_;
    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<uint8_t> mxScaleGm_;

    // tiling-derived params (small-tail field set)
    int64_t blkIdx_{0};
    int64_t usedCoreNum_{0};
    int64_t blockSize_{0};
    uint16_t tailBlockSize_{0}; // rows in the last (tail) block when needPadAxis_
    int64_t quantAxisSize_{0};
    uint32_t postAxisSize_{0};
    uint32_t alignedPostAxisSize_{0}; // nAlignSize
    uint32_t alignedOutputPostAxisSize_{0};
    uint32_t outputPostAxisSize_{0};
    int64_t blockNumInAxis_{0};    // mAlignBlockCount
    int64_t padBlockNumInAxis_{0}; // mAlignGroupCount * 2
    int64_t totalBlockNum_{0};
    int64_t blockNumPerTask_{0};
    int64_t totalTaskNum_{0};
    int64_t avgTaskNum_{0};
    int64_t tailTaskNum_{0};
    int64_t blockNumLastTask_{0};
    int64_t taskStartIdx_{0};
    int64_t taskEndIdx_{0};
    bool needPadAxis_{false};  // isPad: last block has tailBlockSize rows
    bool needPadBlock_{false}; // quantAxisIsOdd: a dummy block pads the last group
    bool needPadPostAxis_{false};

    // runtime per-task offset tracking (stateful across tasks, like multi_n.h)
    int64_t nextInRowOffset_{0};
    int64_t nextOutRowOffset_{0};
    uint32_t inStride_{0};
    uint32_t outStride_{0};
    uint32_t scaleStride_{0};
    uint32_t yUbStride_{0};
    DataCopyPadExtParams<xDtype> padParams_{false, 0, 0, 0};

    // dtype-dependent constants
    uint16_t dtypeYMaxExp_{0};
    uint16_t subNumForScale_{0};
    uint32_t invDtypeMax_{0};
    float invDstTypeMax_{0.0f};
    float maxLowBound_{0.0f};
};

// ---------------------------------------------------------------------------
// ParseTilingData
// ---------------------------------------------------------------------------
template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ParseTilingData(
    const DynamicMxQuant4OptimizeTilingData* tilingData)
{
    quantAxisSize_ = tilingData->quantAxisSize;
    postAxisSize_ = static_cast<uint32_t>(tilingData->postAxisSize);
    blockSize_ = tilingData->blockSize;
    usedCoreNum_ = tilingData->usedCoreNum;
    needPadAxis_ = tilingData->isPad == 1;
    tailBlockSize_ = static_cast<uint16_t>(tilingData->tailBlockSize);
    blockNumInAxis_ = tilingData->mAlignBlockCount;
    needPadBlock_ = tilingData->quantAxisIsOdd == 1;
    alignedPostAxisSize_ = static_cast<uint32_t>(tilingData->nAlignSize);
    padBlockNumInAxis_ = tilingData->mAlignGroupCount * DIGIT_TWO;
    totalBlockNum_ = tilingData->totalGroupNum * DIGIT_TWO;
    needPadPostAxis_ = tilingData->needPadPostAxis == 1;
    blockNumPerTask_ = tilingData->blockNumPerTask;
    totalTaskNum_ = tilingData->totalTaskNum;
    avgTaskNum_ = totalTaskNum_ / usedCoreNum_;
    tailTaskNum_ = totalTaskNum_ % usedCoreNum_;
    taskStartIdx_ = blkIdx_ * avgTaskNum_ + min(blkIdx_, tailTaskNum_);
    taskEndIdx_ = taskStartIdx_ + avgTaskNum_ + (blkIdx_ < tailTaskNum_ ? 1 : 0);
    blockNumLastTask_ = totalBlockNum_ - (totalTaskNum_ - 1) * blockNumPerTask_;

    if constexpr (IsSame<yDtype, fp4x2_e2m1_t>::value || IsSame<yDtype, fp4x2_e1m2_t>::value) {
        outputPostAxisSize_ = static_cast<uint32_t>(Ceil(postAxisSize_, DIGIT_TWO));
        alignedOutputPostAxisSize_ = (alignedPostAxisSize_ / DIGIT_TWO > Ops::Base::GetVRegSize() / 2) ?
                                         alignedPostAxisSize_ / DIGIT_TWO :
                                         Ops::Base::GetVRegSize() / 2;
    } else {
        outputPostAxisSize_ = postAxisSize_;
        alignedOutputPostAxisSize_ = (alignedPostAxisSize_ > Ops::Base::GetVRegSize() / 2) ?
                                         alignedPostAxisSize_ :
                                         Ops::Base::GetVRegSize() / 2;
    }

    padParams_.isPad = true;
    padParams_.leftPadding = 0;
    padParams_.rightPadding = static_cast<uint8_t>(AlignUp(postAxisSize_, ONE_BLK_SIZE / sizeof(xDtype)) -
                                                   postAxisSize_);
    padParams_.paddingValue = 0;
    inStride_ = alignedPostAxisSize_ / ONE_BLK_SIZE * sizeof(xDtype) -
                Ceil(postAxisSize_, ONE_BLK_SIZE / sizeof(xDtype));
    if constexpr (IsSame<yDtype, fp4x2_e2m1_t>::value || IsSame<yDtype, fp4x2_e1m2_t>::value) {
        yUbStride_ = alignedOutputPostAxisSize_;
        outStride_ = alignedOutputPostAxisSize_ / ONE_BLK_SIZE - Ceil(outputPostAxisSize_, ONE_BLK_SIZE);
    } else {
        yUbStride_ = alignedOutputPostAxisSize_;
        outStride_ = alignedOutputPostAxisSize_ / ONE_BLK_SIZE - Ceil(outputPostAxisSize_, ONE_BLK_SIZE);
    }
    // Interleave writes each group at stride alignedPostAxisSize_*2 bytes in UB;
    // srcGap must skip from end of one burst to start of the next group.
    scaleStride_ = static_cast<uint32_t>(alignedPostAxisSize_ * DIGIT_TWO / ONE_BLK_SIZE) -
                   static_cast<uint32_t>(ops::CeilAlign(postAxisSize_ * DIGIT_TWO, static_cast<int64_t>(ONE_BLK_SIZE)) /
                                         ONE_BLK_SIZE);

    int64_t taskBlockIdx = taskStartIdx_ * blockNumPerTask_;
    nextInRowOffset_ = taskBlockIdx / padBlockNumInAxis_ * quantAxisSize_ +
                       taskBlockIdx % padBlockNumInAxis_ * blockSize_;
    nextOutRowOffset_ = nextInRowOffset_;

    invDstTypeMax_ = tilingData->invDstTypeMax;
    maxLowBound_ = tilingData->maxLowBound;

    if constexpr (IsSame<yDtype, fp8_e4m3fn_t>::value) {
        dtypeYMaxExp_ = FP8_E4M3_MAX_EXP;
        invDtypeMax_ = FP8_E4M3_MAX_FLOAT_BITS;
    } else if constexpr (IsSame<yDtype, fp8_e5m2_t>::value) {
        dtypeYMaxExp_ = FP8_E5M2_MAX_EXP;
        invDtypeMax_ = FP8_E5M2_MAX_FLOAT_BITS;
    } else if constexpr (IsSame<yDtype, fp4x2_e2m1_t>::value) {
        dtypeYMaxExp_ = FP4_E2M1_BF16_MAX_EXP;
    }
    if (calcMode == MODE_TWO) {
        subNumForScale_ = static_cast<uint16_t>(tilingData->subNumForScale);
    } else {
        subNumForScale_ = dtypeYMaxExp_;
    }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::Init(
    TPipe* pipe, GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, const DynamicMxQuant4OptimizeTilingData* tilingData)
{
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    pipe_ = pipe;
    tilingData_ = tilingData;
    blkIdx_ = GetBlockIdx();
    ParseTilingData(tilingData);

    xGm_.SetGlobalBuffer((__gm__ xDtype*)(x));
    mxScaleGm_.SetGlobalBuffer((__gm__ uint8_t*)(mxScale));
    yGm_.SetGlobalBuffer((__gm__ uint8_t*)(y));

    // UB buffers (sized to the max task = blockNumPerTask_ blocks).
    // needPadBlock_ means the quant axis has an odd block count; only the very last
    // block in the axis is a dummy (padded to make groups even). All other blocks
    // in every task are real and must be copied in/out.
    int64_t inBufferSize = ops::CeilAlign(
        blockNumPerTask_ * blockSize_ * postAxisSize_ * static_cast<int64_t>(sizeof(xDtype)),
        static_cast<int64_t>(Ops::Base::GetUbBlockSize()));
    int64_t outBufferSize = ops::CeilAlign(blockNumPerTask_ * blockSize_ * yUbStride_,
                                           static_cast<int64_t>(Ops::Base::GetUbBlockSize()));
    // The mxScale interleave writes CeilDiv(blockNumPerTask,2) pairs of
    // alignedPostAxisSize_ bytes; the tmpScale stages one vRegStride per block. The
    // previous blockNumPerTask*VRegSize*2 over-allocation pushed the total UB past the
    // 256KB limit for dataLen=128 (VEC_ERROR 341 out-of-bounds).
    int64_t mxScaleBufferSize = ops::CeilAlign((blockNumPerTask_ + 1) / 2 * alignedPostAxisSize_ * 2,
                                               static_cast<int64_t>(Ops::Base::GetUbBlockSize()));
    int64_t tmpScaleBufferSize = ops::CeilAlign(blockNumPerTask_ * Ops::Base::GetVRegSize(),
                                                static_cast<int64_t>(Ops::Base::GetUbBlockSize()));
    int64_t tmpBufferSize = ops::CeilAlign(blockNumPerTask_ * Ops::Base::GetVRegSize(),
                                           static_cast<int64_t>(Ops::Base::GetUbBlockSize()));

    pipe_->InitBuffer(inQueue_, DB_BUFFER, inBufferSize);
    pipe_->InitBuffer(outQueue_, DB_BUFFER, outBufferSize);
    pipe_->InitBuffer(mxScaleQueue_, DB_BUFFER, mxScaleBufferSize);
    pipe_->InitBuffer(tmpScaleBuf_, tmpScaleBufferSize);
    pipe_->InitBuffer(tmpBuf_, tmpBufferSize);
    // The binary-search merge reads at expOffsetInit*2 bytes; for large dataLen this
    // exceeds VRegSize*2 (dataLen=32: offset 512B + 256B read = 768B) -> OOB read.
    pipe_->InitBuffer(maxExpBuf_, Ops::Base::GetVRegSize() * 4);
}

// ---------------------------------------------------------------------------
// Process: pipeline CopyIn / Compute / CopyOut across tasks.
// ---------------------------------------------------------------------------
template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::Process()
{
    if (blkIdx_ >= usedCoreNum_ || totalTaskNum_ == 0) {
        return;
    }

    int64_t blockIdx = taskStartIdx_ * blockNumPerTask_;
    int64_t blockCount = (taskStartIdx_ == totalTaskNum_ - 1) ? blockNumLastTask_ : blockNumPerTask_;
    CopyIn(blockIdx, blockCount);

    for (int64_t taskIdx = taskStartIdx_ + 1; taskIdx < taskEndIdx_; taskIdx++) {
        int64_t nextBlockIdx = taskIdx * blockNumPerTask_;
        int64_t nextBlockCount = (taskIdx == totalTaskNum_ - 1) ? blockNumLastTask_ : blockNumPerTask_;
        CopyIn(nextBlockIdx, nextBlockCount);
        ComputeAll(blockIdx, blockCount);
        CopyOut(blockIdx, blockCount);
        blockIdx = nextBlockIdx;
        blockCount = nextBlockCount;
    }
    ComputeAll(blockIdx, blockCount);
    CopyOut(blockIdx, blockCount);
}

// ---------------------------------------------------------------------------
// CopyIn: GM -> UB. Adapted from multi_n.h; rows land at alignedPostAxisSize_ stride.
// ---------------------------------------------------------------------------
template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::CopyIn(
    int64_t offset, int64_t count)
{
    int64_t rowOffset = (offset + count) / padBlockNumInAxis_ * quantAxisSize_ +
                        (offset + count) % padBlockNumInAxis_ * blockSize_;
    int64_t inOffset = nextInRowOffset_ * postAxisSize_;
    uint16_t nBurst = static_cast<uint16_t>(rowOffset - nextInRowOffset_);
    uint32_t blockLen = postAxisSize_ * sizeof(xDtype);
    nextInRowOffset_ = rowOffset;

    LocalTensor<xDtype> x = inQueue_.template AllocTensor<xDtype>();
    DataCopyExtParams copyParams = {nBurst, blockLen, 0, 0, 0};
    DataCopyPadExtParams<xDtype> compactPad{false, 0, 0, 0};
    DataCopyPad<xDtype, PaddingMode::Compact>(x, xGm_[inOffset], copyParams, compactPad);
    inQueue_.EnQue(x);
}

// ---------------------------------------------------------------------------
// CopyOut: UB -> GM for Y and mxScale. Adapted from multi_n.h.
// ---------------------------------------------------------------------------
template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::CopyOut(
    int64_t offset, int64_t count)
{
    int64_t rowOffset = (offset + count) / padBlockNumInAxis_ * quantAxisSize_ +
                        (offset + count) % padBlockNumInAxis_ * blockSize_;
    int64_t outOffset = nextOutRowOffset_ * outputPostAxisSize_;
    uint16_t totalRows = static_cast<uint16_t>(rowOffset - nextOutRowOffset_);
    uint32_t outBlockLen;
    // ComputeAll packs y rows contiguously for REAL blocks only (dummy blocks write
    // nothing; tail blocks write tailBlockSize_ rows). totalRows is derived from the
    // same rowOffset mapping, so it is the exact number of y rows produced here.
    // Using count*blockSize_ over-copies uninitialized UB and races with the next
    // task's GM writes on other cores.
    if constexpr (IsSame<yDtype, fp4x2_e2m1_t>::value || IsSame<yDtype, fp4x2_e1m2_t>::value) {
        outBlockLen = static_cast<uint32_t>(totalRows) * outputPostAxisSize_;
    } else {
        outBlockLen = static_cast<uint32_t>(totalRows) * postAxisSize_;
    }
    uint16_t scaleNBurst = static_cast<uint16_t>(count / DIGIT_TWO);
    uint32_t scaleBlockLen = postAxisSize_ * DIGIT_TWO;
    int64_t scaleOffset = offset * postAxisSize_;
    nextOutRowOffset_ = rowOffset;

    LocalTensor<uint8_t> y = outQueue_.template DeQue<uint8_t>();
    DataCopyExtParams copyOutParams = {1, outBlockLen, 0, 0, 0};
    DataCopyPad(yGm_[outOffset], y, copyOutParams);
    outQueue_.FreeTensor(y);

    LocalTensor<uint8_t> mxScale = mxScaleQueue_.template DeQue<uint8_t>();
    DataCopyExtParams copyScaleParams = {scaleNBurst, scaleBlockLen, scaleStride_, 0, 0};
    DataCopyPad(mxScaleGm_[scaleOffset], mxScale, copyScaleParams);
    mxScaleQueue_.FreeTensor(mxScale);
}

// ---------------------------------------------------------------------------
// ComputeAll: process `count` blocks; per block call ComputeScaleOcp + ComputeYVf;
// then interleave adjacent block-scale pairs into the packed mxscale output.
// ---------------------------------------------------------------------------
template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ComputeAll(
    int64_t offset, int64_t count)
{
    LocalTensor<xDtype> x = inQueue_.template DeQue<xDtype>();
    LocalTensor<uint8_t> mxScale = mxScaleQueue_.template AllocTensor<uint8_t>();
    LocalTensor<uint8_t> y = outQueue_.template AllocTensor<uint8_t>();
    LocalTensor<uint8_t> tmpScaleLocal = tmpScaleBuf_.Get<uint8_t>();
    LocalTensor<uint16_t> tmpLocal = tmpBuf_.Get<uint16_t>();

    auto xBase = (__ubuf__ xDtype*)x.GetPhyAddr();
    auto yBase = (__ubuf__ uint8_t*)y.GetPhyAddr();
    auto mxTmpScaleBase = (__ubuf__ uint8_t*)tmpScaleLocal.GetPhyAddr();
    auto tmpBase = (__ubuf__ uint16_t*)tmpLocal.GetPhyAddr();

    const uint16_t scaleDataLen = static_cast<uint16_t>(postAxisSize_);
    const uint16_t yDataLen = static_cast<uint16_t>(postAxisSize_);
    const int64_t vRegStride = static_cast<int64_t>(Ops::Base::GetVRegSize());

    int64_t ubRowOffset = 0;
    for (int64_t b = 0; b < count; ++b) {
        int64_t blockGlobalIdx = offset + b;
        int64_t blockInAxis = blockGlobalIdx % padBlockNumInAxis_;
        bool isDummy = needPadBlock_ && (blockInAxis == blockNumInAxis_);
        bool isTail = needPadAxis_ && (blockInAxis == blockNumInAxis_ - 1);
        uint16_t rows = isTail ? tailBlockSize_ : static_cast<uint16_t>(blockSize_);

        __ubuf__ xDtype* xAddr = xBase + ubRowOffset * postAxisSize_;
        __ubuf__ uint8_t* yAddr = yBase + ubRowOffset * outputPostAxisSize_;
        __ubuf__ uint8_t* scaleAddr = mxTmpScaleBase + b * vRegStride;
        __ubuf__ uint16_t* recipAddr = tmpBase + b * (vRegStride / static_cast<int64_t>(sizeof(uint16_t)));

        Duplicate<uint8_t>(tmpScaleLocal[b * vRegStride], static_cast<uint8_t>(0), static_cast<int32_t>(vRegStride));
        if (isDummy) {
            Duplicate<uint8_t>(tmpScaleLocal[b * vRegStride], static_cast<uint8_t>(0), alignedPostAxisSize_);
        } else {
            ComputeScaleOcp(scaleDataLen, rows, xAddr, scaleAddr, recipAddr);
            ComputeYVf(yDataLen, rows, xAddr, recipAddr, yAddr);
            ubRowOffset += rows;
        }
    }

    int64_t groupCount = (count + 1) / DIGIT_TWO;
    for (int64_t g = 0; g < groupCount; ++g) {
        int64_t b0 = g * 2;
        int64_t b1 = g * 2 + 1;
        if (b1 >= count) {
            Duplicate<uint8_t>(tmpScaleLocal[b1 * vRegStride], static_cast<uint8_t>(0), alignedPostAxisSize_);
        }
        Interleave(mxScale[g * alignedPostAxisSize_ * 2], mxScale[g * alignedPostAxisSize_ * 2 + alignedPostAxisSize_],
                   tmpScaleLocal[b0 * vRegStride], tmpScaleLocal[b1 * vRegStride], alignedPostAxisSize_);
    }

    inQueue_.FreeTensor(x);
    mxScaleQueue_.EnQue(mxScale);
    outQueue_.EnQue(y);
}

// ---------------------------------------------------------------------------
// ComputeScaleOcp: BF16 OCP (calcMode == MODE_ZERO) scale for one block.
// Multi-row binary-search max across `blockCount` rows (each `dataLen` wide).
// Produces `dataLen` scales (1 byte each) at mxScaleAddr and `dataLen` reciprocals
// (uint16 = BF16 1/scale) at tmpAddr.
// ---------------------------------------------------------------------------
template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ComputeScaleOcp(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScaleAddr,
    __ubuf__ uint16_t* tmpAddr)
{
    if constexpr (calcMode == MODE_ONE || calcMode == MODE_THREE) {
        ComputeScaleCeilAlg(dataLen, blockCount, xAddr, mxScaleAddr, tmpAddr);
    } else if constexpr (IsSame<xDtype, bfloat16_t>::value) {
        ComputeScaleOcpBf16(dataLen, blockCount, xAddr, mxScaleAddr, tmpAddr);
    } else if constexpr (IsSame<xDtype, half>::value) {
        ComputeScaleOcpHalf(dataLen, blockCount, xAddr, mxScaleAddr, tmpAddr);
    } else if constexpr (IsSame<xDtype, float>::value) {
        ComputeScaleOcpFp32(dataLen, blockCount, xAddr, mxScaleAddr, tmpAddr);
    } else {
        (void)dataLen;
        (void)blockCount;
        (void)xAddr;
        (void)mxScaleAddr;
        (void)tmpAddr;
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ComputeScaleOcpBf16(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScaleAddr,
    __ubuf__ uint16_t* tmpAddr)
{
    auto lp = ComputeLoopParams(dataLen, blockCount);
    const uint16_t N = lp.N;
    const uint16_t dataLenSingleLoop = lp.dataLenSingleLoop;
    const uint16_t regLoop = lp.regLoop;
    const uint16_t dataLenTailLoop = lp.dataLenTailLoop;
    const uint16_t loopSize = lp.loopSize;
    const uint16_t expOffsetInit = lp.expOffsetInit;

    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> x;
        Reg::RegTensor<uint16_t> expU16;
        Reg::RegTensor<uint16_t> expMaxU16;
        Reg::RegTensor<uint16_t> maxU16;
        Reg::RegTensor<uint16_t> mxScaleU16;
        Reg::RegTensor<uint8_t> mxScaleU8;
        Reg::RegTensor<uint16_t> recipU16;

        Reg::RegTensor<uint16_t> maxEleU16;
        Reg::RegTensor<uint16_t> biasU16;
        Reg::RegTensor<uint16_t> zeroU16;
        Reg::RegTensor<uint16_t> nanU16;
        Reg::RegTensor<uint16_t> specialExpU16;
        Reg::RegTensor<uint16_t> tgtMaxExpU16;
        Reg::RegTensor<uint16_t> subNumForScaleU16;
        Reg::RegTensor<uint16_t> fp8NanU16;
        Reg::RegTensor<uint16_t> absForX;

        Reg::MaskReg infMask;
        Reg::MaskReg zeroMask;
        Reg::MaskReg specialMask;
        Reg::MaskReg invalidDataMask;
        Reg::MaskReg pregAll8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();

        Reg::Duplicate(maxEleU16, BF16_MAX_EXP);
        Reg::Duplicate(biasU16, BF16_EXP_BIAS);
        Reg::Duplicate(zeroU16, 0);
        Reg::Duplicate(nanU16, BF16_NAN_CUSTOM);
        Reg::Duplicate(specialExpU16, BF16_SPECIAL_EXP_THRESHOLD);
        Reg::Duplicate(tgtMaxExpU16, dtypeYMaxExp_);
        Reg::Duplicate(subNumForScaleU16, subNumForScale_);
        Reg::Duplicate(fp8NanU16, FP8_DEFAULT_MAX_EXP);
        Reg::Duplicate(absForX, BF16_ABS_MASK);
        Reg::Duplicate(expMaxU16, 0);
        Reg::Duplicate(maxU16, 0);

        uint32_t pnumU32 = dataLenSingleLoop;
        uint32_t tailPnumU32 = dataLenTailLoop;
        Reg::MaskReg pnumMask16 = Reg::UpdateMask<uint16_t>(pnumU32);
        Reg::MaskReg tailPnumMask16 = Reg::UpdateMask<uint16_t>(tailPnumU32);
        uint32_t validU32 = postAxisSize_;
        Reg::MaskReg validMask16 = Reg::UpdateMask<uint16_t>(validU32);

        LocalTensor<uint16_t> maxExpTensor = maxExpBuf_.Get<uint16_t>();
        auto maxExpAddr16 = (__ubuf__ uint16_t*)maxExpTensor.GetPhyAddr();

        if constexpr (calcMode == MODE_ZERO) {
            for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); ++i) {
                Reg::UnalignRegForLoad uLd;
                Reg::LoadUnAlignPre(uLd, xAddr + static_cast<int64_t>(i) * dataLenSingleLoop);
                Reg::LoadUnAlign(x, uLd, xAddr + static_cast<int64_t>(i) * dataLenSingleLoop);
                Reg::And(expU16, (Reg::RegTensor<uint16_t>&)x, maxEleU16, pnumMask16);
                Reg::Max(expMaxU16, expMaxU16, expU16, pnumMask16);
            }
            Reg::UnalignRegForLoad uLd2;
            Reg::LoadUnAlignPre(uLd2, xAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop);
            Reg::LoadUnAlign(x, uLd2, xAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop);
            Reg::And(expU16, (Reg::RegTensor<uint16_t>&)x, maxEleU16, tailPnumMask16);
            Reg::Max(expU16, expMaxU16, expU16, tailPnumMask16);
            Reg::Copy<uint16_t, Reg::MaskMergeMode::MERGING>(expMaxU16, expU16, tailPnumMask16);
            Reg::RegTensor<uint16_t> zeroTmp16;
            Reg::Duplicate(zeroTmp16, 0);
            uint32_t staleNum16 = dataLenSingleLoop - tailPnumU32;
            Reg::MaskReg staleMask16 = Reg::UpdateMask<uint16_t>(staleNum16);
            Reg::Select<uint16_t>(maxU16, zeroTmp16, maxU16, staleMask16);
        } else {
            for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); ++i) {
                Reg::UnalignRegForLoad uLd;
                Reg::LoadUnAlignPre(uLd, xAddr + static_cast<int64_t>(i) * dataLenSingleLoop);
                Reg::LoadUnAlign(x, uLd, xAddr + static_cast<int64_t>(i) * dataLenSingleLoop);
                Reg::And(expU16, (Reg::RegTensor<uint16_t>&)x, absForX, pnumMask16);
                Reg::Max(maxU16, maxU16, expU16, pnumMask16);
            }
            Reg::UnalignRegForLoad uLd2;
            Reg::LoadUnAlignPre(uLd2, xAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop);
            Reg::LoadUnAlign(x, uLd2, xAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop);
            Reg::And(expU16, (Reg::RegTensor<uint16_t>&)x, absForX, tailPnumMask16);
            Reg::Max(expU16, maxU16, expU16, tailPnumMask16);
            Reg::Copy<uint16_t, Reg::MaskMergeMode::MERGING>(maxU16, expU16, tailPnumMask16);
        }

        if (loopSize > 0) {
            uint16_t expOffset = expOffsetInit;
            if constexpr (calcMode == MODE_ZERO) {
                Reg::StoreAlign(maxExpAddr16, expMaxU16, pnumMask16);
                Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
                uint32_t maskNum = static_cast<uint32_t>(dataLenSingleLoop) - static_cast<uint32_t>(expOffset);
                Reg::MaskReg mask = Reg::UpdateMask<uint16_t>(maskNum);
                Reg::UnalignRegForLoad uBs;
                Reg::LoadUnAlignPre(uBs, maxExpAddr16);
                Reg::LoadUnAlign(expMaxU16, uBs, maxExpAddr16);
                Reg::UnalignRegForLoad uBs2;
                Reg::LoadUnAlignPre(uBs2, maxExpAddr16 + expOffset);
                Reg::LoadUnAlign(expU16, uBs2, maxExpAddr16 + expOffset);
                Reg::Max(expU16, expMaxU16, expU16, mask);
                Reg::Copy<uint16_t, Reg::MaskMergeMode::MERGING>(expMaxU16, expU16, mask);
                for (uint16_t i = 0; i < loopSize; ++i) {
                    Reg::StoreAlign(maxExpAddr16, expMaxU16, pnumMask16);
                    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
                    expOffset = static_cast<uint16_t>(expOffset / DIGIT_TWO);
                    maskNum = static_cast<uint32_t>(expOffset);
                    mask = Reg::UpdateMask<uint16_t>(maskNum);
                    Reg::UnalignRegForLoad uBs;
                    Reg::LoadUnAlignPre(uBs, maxExpAddr16);
                    Reg::LoadUnAlign(expMaxU16, uBs, maxExpAddr16);
                    Reg::UnalignRegForLoad uBs2;
                    Reg::LoadUnAlignPre(uBs2, maxExpAddr16 + expOffset);
                    Reg::LoadUnAlign(expU16, uBs2, maxExpAddr16 + expOffset);
                    Reg::Max(expMaxU16, expMaxU16, expU16, mask);
                }
            } else {
                Reg::StoreAlign(maxExpAddr16, maxU16, pnumMask16);
                Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
                uint32_t maskNum = static_cast<uint32_t>(dataLenSingleLoop) - static_cast<uint32_t>(expOffset);
                Reg::MaskReg mask = Reg::UpdateMask<uint16_t>(maskNum);
                Reg::UnalignRegForLoad uBs3;
                Reg::LoadUnAlignPre(uBs3, maxExpAddr16);
                Reg::LoadUnAlign(maxU16, uBs3, maxExpAddr16);
                Reg::UnalignRegForLoad uBs2;
                Reg::LoadUnAlignPre(uBs2, maxExpAddr16 + expOffset);
                Reg::LoadUnAlign(expU16, uBs2, maxExpAddr16 + expOffset);
                Reg::Max(expU16, maxU16, expU16, mask);
                Reg::Copy<uint16_t, Reg::MaskMergeMode::MERGING>(maxU16, expU16, mask);
                for (uint16_t i = 0; i < loopSize; ++i) {
                    Reg::StoreAlign(maxExpAddr16, maxU16, pnumMask16);
                    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
                    expOffset = static_cast<uint16_t>(expOffset / DIGIT_TWO);
                    maskNum = static_cast<uint32_t>(expOffset);
                    mask = Reg::UpdateMask<uint16_t>(maskNum);
                    Reg::UnalignRegForLoad uBs3;
                    Reg::LoadUnAlignPre(uBs3, maxExpAddr16);
                    Reg::LoadUnAlign(maxU16, uBs3, maxExpAddr16);
                    Reg::UnalignRegForLoad uBs2;
                    Reg::LoadUnAlignPre(uBs2, maxExpAddr16 + expOffset);
                    Reg::LoadUnAlign(expU16, uBs2, maxExpAddr16 + expOffset);
                    Reg::Max(maxU16, maxU16, expU16, mask);
                }
            }
        }

        if constexpr (calcMode == MODE_ZERO) {
            Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxU16, maxEleU16, pregAll16);
            Reg::Compare<uint16_t, CMPMODE::LT>(invalidDataMask, expMaxU16, tgtMaxExpU16, pregAll16);
            Reg::Sub(expMaxU16, expMaxU16, subNumForScaleU16, pregAll16);
            Reg::Select<uint16_t>(expMaxU16, zeroU16, expMaxU16, invalidDataMask);
            Reg::ShiftRights(mxScaleU16, expMaxU16, BF16_SHR_NUM, pregAll16);
            Reg::Select<uint16_t>(mxScaleU16, mxScaleU16, fp8NanU16, infMask);
        } else if constexpr (calcMode == MODE_TWO) {
            Reg::And(expMaxU16, maxU16, maxEleU16, pregAll16);
            Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxU16, maxEleU16, pregAll16);
            Reg::Compare<uint16_t, CMPMODE::LT>(invalidDataMask, expMaxU16, tgtMaxExpU16, pregAll16);
            Reg::Sub(expMaxU16, maxU16, subNumForScaleU16, pregAll16);
            Reg::Select<uint16_t>(expMaxU16, zeroU16, expMaxU16, invalidDataMask);
            Reg::ShiftRights(mxScaleU16, expMaxU16, BF16_SHR_NUM, pregAll16);
            Reg::Select<uint16_t>(mxScaleU16, mxScaleU16, fp8NanU16, infMask);
        }
        Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScaleU8, mxScaleU16);
        Reg::MaskReg mxScaleMask = Reg::UpdateMask<uint8_t>(alignedPostAxisSize_);
        Reg::StoreAlign(mxScaleAddr, mxScaleU8, mxScaleMask);

        if constexpr (calcMode == MODE_ZERO) {
            Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxU16, zeroU16, pregAll16);
            Reg::Compare<uint16_t, CMPMODE::EQ>(specialMask, expMaxU16, biasU16, pregAll16);
            Reg::Sub(recipU16, biasU16, expMaxU16, pregAll16);
            Reg::Select<uint16_t>(recipU16, recipU16, nanU16, infMask);
            Reg::Select<uint16_t>(recipU16, recipU16, zeroU16, zeroMask);
            Reg::Select<uint16_t>(recipU16, specialExpU16, recipU16, specialMask);
        } else {
            Reg::And(expMaxU16, expMaxU16, maxEleU16, pregAll16);
            Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxU16, zeroU16, pregAll16);
            Reg::Compare<uint16_t, CMPMODE::EQ>(specialMask, expMaxU16, biasU16, pregAll16);
            Reg::Sub(recipU16, biasU16, expMaxU16, pregAll16);
            Reg::Select<uint16_t>(recipU16, recipU16, nanU16, infMask);
            Reg::Select<uint16_t>(recipU16, recipU16, zeroU16, zeroMask);
            Reg::Select<uint16_t>(recipU16, specialExpU16, recipU16, specialMask);
        }
        // Plain masked vector stores: ordered wrt the Y path's vector loads (the unalign
        // data-move path is not, which races the recip read for tail blocks).
        Reg::UnalignRegForStore uSt;
        auto scaleAddr = tmpAddr;
        for (uint16_t i = 0; i < N; ++i) {
            Reg::UnalignRegForStore uStRow;
            __ubuf__ uint8_t* rowAddr = (__ubuf__ uint8_t*)(scaleAddr + i * dataLen);
            Reg::StoreUnAlign(rowAddr, (Reg::RegTensor<uint8_t>&)recipU16, uStRow,
                              static_cast<uint32_t>(dataLen * sizeof(uint16_t)));
            Reg::StoreUnAlignPost(rowAddr, uStRow, 0);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        Reg::LoadAlign(recipU16, tmpAddr);
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ComputeScaleOcpHalf(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScaleAddr,
    __ubuf__ uint16_t* tmpAddr)
{
    auto lp = ComputeLoopParams(dataLen, blockCount);
    const uint16_t N = lp.N;
    const uint16_t dataLenSingleLoop = lp.dataLenSingleLoop;
    const uint16_t regLoop = lp.regLoop;
    const uint16_t dataLenTailLoop = lp.dataLenTailLoop;
    const uint16_t loopSize = lp.loopSize;
    const uint16_t expOffsetInit = lp.expOffsetInit;

    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> x;
        Reg::RegTensor<uint16_t> expU16;
        Reg::RegTensor<uint16_t> expMaxU16;
        Reg::RegTensor<uint16_t> mxScaleU16;
        Reg::RegTensor<uint8_t> mxScaleU8;
        Reg::RegTensor<uint16_t> recipU16;

        Reg::RegTensor<uint16_t> maxEleU16;
        Reg::RegTensor<uint16_t> fp16MaxEleU16;
        Reg::RegTensor<uint16_t> biasU16;
        Reg::RegTensor<uint16_t> zeroU16;
        Reg::RegTensor<uint16_t> nanU16;
        Reg::RegTensor<uint16_t> specialExpU16;
        Reg::RegTensor<uint16_t> tgtMaxExpU16;
        Reg::RegTensor<uint16_t> subNumForScaleU16;
        Reg::RegTensor<uint16_t> fp8NanU16;
        Reg::RegTensor<uint16_t> absForX;
        Reg::RegTensor<uint16_t> maxU16;

        Reg::MaskReg infMask;
        Reg::MaskReg zeroMask;
        Reg::MaskReg specialMask;
        Reg::MaskReg invalidDataMask;
        Reg::MaskReg pregAll8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();

        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                              Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
        static constexpr Reg::CastTrait castTraitCeilAlgHalf2Bf16 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                                     Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        Reg::Duplicate(maxEleU16, BF16_MAX_EXP);
        Reg::Duplicate(fp16MaxEleU16, FP16_INF);
        Reg::Duplicate(biasU16, BF16_EXP_BIAS);
        Reg::Duplicate(zeroU16, 0);
        Reg::Duplicate(nanU16, BF16_NAN_CUSTOM);
        Reg::Duplicate(specialExpU16, BF16_SPECIAL_EXP_THRESHOLD);
        Reg::Duplicate(tgtMaxExpU16, dtypeYMaxExp_);
        Reg::Duplicate(subNumForScaleU16, subNumForScale_);
        Reg::Duplicate(fp8NanU16, FP8_DEFAULT_MAX_EXP);
        Reg::Duplicate(absForX, BF16_ABS_MASK);
        Reg::Duplicate(expMaxU16, 0);
        Reg::Duplicate(maxU16, 0);

        uint32_t pnumU32 = dataLenSingleLoop;
        uint32_t tailPnumU32 = dataLenTailLoop;
        Reg::MaskReg pnumMask16 = Reg::UpdateMask<uint16_t>(pnumU32);
        Reg::MaskReg tailPnumMask16 = Reg::UpdateMask<uint16_t>(tailPnumU32);
        uint32_t validU32 = postAxisSize_;
        Reg::MaskReg validMask16 = Reg::UpdateMask<uint16_t>(validU32);

        LocalTensor<uint16_t> maxExpTensor = maxExpBuf_.Get<uint16_t>();
        auto maxExpAddr16 = (__ubuf__ uint16_t*)maxExpTensor.GetPhyAddr();

        if constexpr (calcMode == MODE_ZERO) {
            for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); ++i) {
                Reg::UnalignRegForLoad uLd;
                Reg::LoadUnAlignPre(uLd, xAddr + static_cast<int64_t>(i) * dataLenSingleLoop);
                Reg::LoadUnAlign(x, uLd, xAddr + static_cast<int64_t>(i) * dataLenSingleLoop);
                Reg::And(expU16, (Reg::RegTensor<uint16_t>&)x, fp16MaxEleU16, pnumMask16);
                Reg::CompareScalar<uint16_t, CMPMODE::EQ>(infMask, (Reg::RegTensor<uint16_t>&)expU16, FP16_INF,
                                                          pnumMask16);
                Reg::Cast<bfloat16_t, xDtype, castTraitHalf2Bf16>((Reg::RegTensor<bfloat16_t>&)expU16, x, pnumMask16);
                Reg::And(expU16, (Reg::RegTensor<uint16_t>&)expU16, maxEleU16, pnumMask16);
                Reg::Select<uint16_t>(expU16, maxEleU16, expU16, infMask);
                Reg::Max(expMaxU16, expMaxU16, expU16, pnumMask16);
            }
            Reg::UnalignRegForLoad uLd2;
            Reg::LoadUnAlignPre(uLd2, xAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop);
            Reg::LoadUnAlign(x, uLd2, xAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop);
            Reg::And(expU16, (Reg::RegTensor<uint16_t>&)x, fp16MaxEleU16, tailPnumMask16);
            Reg::CompareScalar<uint16_t, CMPMODE::EQ>(infMask, (Reg::RegTensor<uint16_t>&)expU16, FP16_INF,
                                                      tailPnumMask16);
            Reg::Cast<bfloat16_t, xDtype, castTraitHalf2Bf16>((Reg::RegTensor<bfloat16_t>&)expU16, x, tailPnumMask16);
            Reg::And(expU16, (Reg::RegTensor<uint16_t>&)expU16, maxEleU16, tailPnumMask16);
            Reg::Select<uint16_t>(expU16, maxEleU16, expU16, infMask);
            Reg::Max(expU16, expMaxU16, expU16, tailPnumMask16);
            Reg::Copy<uint16_t, Reg::MaskMergeMode::MERGING>(expMaxU16, expU16, tailPnumMask16);
            Reg::RegTensor<uint16_t> zeroTmp16;
            Reg::Duplicate(zeroTmp16, 0);
            uint32_t staleNum16 = dataLenSingleLoop - tailPnumU32;
            Reg::MaskReg staleMask16 = Reg::UpdateMask<uint16_t>(staleNum16);
            Reg::Select<uint16_t>(maxU16, zeroTmp16, maxU16, staleMask16);
        } else {
            for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); ++i) {
                Reg::UnalignRegForLoad uLd;
                Reg::LoadUnAlignPre(uLd, xAddr + static_cast<int64_t>(i) * dataLenSingleLoop);
                Reg::LoadUnAlign(x, uLd, xAddr + static_cast<int64_t>(i) * dataLenSingleLoop);
                Reg::And(expU16, (Reg::RegTensor<uint16_t>&)x, absForX, pnumMask16);
                Reg::Cast<bfloat16_t, xDtype, castTraitCeilAlgHalf2Bf16>((Reg::RegTensor<bfloat16_t>&)expU16,
                                                                         (Reg::RegTensor<xDtype>&)expU16, pnumMask16);
                Reg::Max(maxU16, maxU16, expU16, pnumMask16);
            }
            Reg::UnalignRegForLoad uLd2;
            Reg::LoadUnAlignPre(uLd2, xAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop);
            Reg::LoadUnAlign(x, uLd2, xAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop);
            Reg::And(expU16, (Reg::RegTensor<uint16_t>&)x, absForX, tailPnumMask16);
            Reg::Cast<bfloat16_t, xDtype, castTraitCeilAlgHalf2Bf16>((Reg::RegTensor<bfloat16_t>&)expU16,
                                                                     (Reg::RegTensor<xDtype>&)expU16, tailPnumMask16);
            Reg::Max(expU16, maxU16, expU16, tailPnumMask16);
            Reg::Copy<uint16_t, Reg::MaskMergeMode::MERGING>(maxU16, expU16, tailPnumMask16);
        }

        if (loopSize > 0) {
            uint16_t expOffset = expOffsetInit;
            if constexpr (calcMode == MODE_ZERO) {
                Reg::StoreAlign(maxExpAddr16, expMaxU16, pnumMask16);
                Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
                uint32_t maskNum = static_cast<uint32_t>(dataLenSingleLoop) - static_cast<uint32_t>(expOffset);
                Reg::MaskReg mask = Reg::UpdateMask<uint16_t>(maskNum);
                Reg::UnalignRegForLoad uBs;
                Reg::LoadUnAlignPre(uBs, maxExpAddr16);
                Reg::LoadUnAlign(expMaxU16, uBs, maxExpAddr16);
                Reg::UnalignRegForLoad uBs2;
                Reg::LoadUnAlignPre(uBs2, maxExpAddr16 + expOffset);
                Reg::LoadUnAlign(expU16, uBs2, maxExpAddr16 + expOffset);
                Reg::Max(expU16, expMaxU16, expU16, mask);
                Reg::Copy<uint16_t, Reg::MaskMergeMode::MERGING>(expMaxU16, expU16, mask);
                for (uint16_t i = 0; i < loopSize; ++i) {
                    Reg::StoreAlign(maxExpAddr16, expMaxU16, pnumMask16);
                    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
                    expOffset = static_cast<uint16_t>(expOffset / DIGIT_TWO);
                    maskNum = static_cast<uint32_t>(expOffset);
                    mask = Reg::UpdateMask<uint16_t>(maskNum);
                    Reg::UnalignRegForLoad uBs;
                    Reg::LoadUnAlignPre(uBs, maxExpAddr16);
                    Reg::LoadUnAlign(expMaxU16, uBs, maxExpAddr16);
                    Reg::UnalignRegForLoad uBs2;
                    Reg::LoadUnAlignPre(uBs2, maxExpAddr16 + expOffset);
                    Reg::LoadUnAlign(expU16, uBs2, maxExpAddr16 + expOffset);
                    Reg::Max(expMaxU16, expMaxU16, expU16, mask);
                }
            } else {
                Reg::StoreAlign(maxExpAddr16, maxU16, pnumMask16);
                Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
                uint32_t maskNum = static_cast<uint32_t>(dataLenSingleLoop) - static_cast<uint32_t>(expOffset);
                Reg::MaskReg mask = Reg::UpdateMask<uint16_t>(maskNum);
                Reg::UnalignRegForLoad uBs3;
                Reg::LoadUnAlignPre(uBs3, maxExpAddr16);
                Reg::LoadUnAlign(maxU16, uBs3, maxExpAddr16);
                Reg::UnalignRegForLoad uBs2;
                Reg::LoadUnAlignPre(uBs2, maxExpAddr16 + expOffset);
                Reg::LoadUnAlign(expU16, uBs2, maxExpAddr16 + expOffset);
                Reg::Max(expU16, maxU16, expU16, mask);
                Reg::Copy<uint16_t, Reg::MaskMergeMode::MERGING>(maxU16, expU16, mask);
                for (uint16_t i = 0; i < loopSize; ++i) {
                    Reg::StoreAlign(maxExpAddr16, maxU16, pnumMask16);
                    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
                    expOffset = static_cast<uint16_t>(expOffset / DIGIT_TWO);
                    maskNum = static_cast<uint32_t>(expOffset);
                    mask = Reg::UpdateMask<uint16_t>(maskNum);
                    Reg::UnalignRegForLoad uBs3;
                    Reg::LoadUnAlignPre(uBs3, maxExpAddr16);
                    Reg::LoadUnAlign(maxU16, uBs3, maxExpAddr16);
                    Reg::UnalignRegForLoad uBs2;
                    Reg::LoadUnAlignPre(uBs2, maxExpAddr16 + expOffset);
                    Reg::LoadUnAlign(expU16, uBs2, maxExpAddr16 + expOffset);
                    Reg::Max(maxU16, maxU16, expU16, mask);
                }
            }
        }

        if constexpr (calcMode == MODE_ZERO) {
            Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxU16, maxEleU16, pregAll16);
            Reg::Compare<uint16_t, CMPMODE::LT>(invalidDataMask, expMaxU16, tgtMaxExpU16, pregAll16);
            Reg::Sub(expMaxU16, expMaxU16, subNumForScaleU16, pregAll16);
            Reg::Select<uint16_t>(expMaxU16, zeroU16, expMaxU16, invalidDataMask);
            Reg::ShiftRights(mxScaleU16, expMaxU16, BF16_SHR_NUM, pregAll16);
            Reg::Select<uint16_t>(mxScaleU16, mxScaleU16, fp8NanU16, infMask);
        } else {
            Reg::And(expMaxU16, maxU16, maxEleU16, pregAll16);
            Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxU16, maxEleU16, pregAll16);
            Reg::Compare<uint16_t, CMPMODE::LT>(invalidDataMask, expMaxU16, tgtMaxExpU16, pregAll16);
            Reg::Sub(expMaxU16, maxU16, subNumForScaleU16, pregAll16);
            Reg::Select<uint16_t>(expMaxU16, zeroU16, expMaxU16, invalidDataMask);
            Reg::ShiftRights(mxScaleU16, expMaxU16, BF16_SHR_NUM, pregAll16);
            Reg::Select<uint16_t>(mxScaleU16, mxScaleU16, fp8NanU16, infMask);
        }
        Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScaleU8, mxScaleU16);
        Reg::MaskReg mxScaleMask = Reg::UpdateMask<uint8_t>(postAxisSize_);
        Reg::StoreAlign(mxScaleAddr, mxScaleU8, mxScaleMask);

        if constexpr (calcMode == MODE_ZERO) {
            Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxU16, zeroU16, pregAll16);
            Reg::Compare<uint16_t, CMPMODE::EQ>(specialMask, expMaxU16, biasU16, pregAll16);
            Reg::Sub(recipU16, biasU16, expMaxU16, pregAll16);
            Reg::Select<uint16_t>(recipU16, recipU16, nanU16, infMask);
            Reg::Select<uint16_t>(recipU16, recipU16, zeroU16, zeroMask);
            Reg::Select<uint16_t>(recipU16, specialExpU16, recipU16, specialMask);
        } else {
            Reg::And(expMaxU16, expMaxU16, maxEleU16, pregAll16);
            Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxU16, zeroU16, pregAll16);
            Reg::Compare<uint16_t, CMPMODE::EQ>(specialMask, expMaxU16, biasU16, pregAll16);
            Reg::Sub(recipU16, biasU16, expMaxU16, pregAll16);
            Reg::Select<uint16_t>(recipU16, recipU16, nanU16, infMask);
            Reg::Select<uint16_t>(recipU16, recipU16, zeroU16, zeroMask);
            Reg::Select<uint16_t>(recipU16, specialExpU16, recipU16, specialMask);
        }
        // Plain masked vector stores: ordered wrt the Y path's vector loads (the unalign
        // data-move path is not, which races the recip read for tail blocks).
        Reg::UnalignRegForStore uSt;
        auto scaleAddr = tmpAddr;
        for (uint16_t i = 0; i < N; ++i) {
            Reg::UnalignRegForStore uStRow;
            __ubuf__ uint8_t* rowAddr = (__ubuf__ uint8_t*)(scaleAddr + i * dataLen);
            Reg::StoreUnAlign(rowAddr, (Reg::RegTensor<uint8_t>&)recipU16, uStRow,
                              static_cast<uint32_t>(dataLen * sizeof(uint16_t)));
            Reg::StoreUnAlignPost(rowAddr, uStRow, 0);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        Reg::LoadAlign(recipU16, tmpAddr);
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ComputeScaleOcpFp32(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScaleAddr,
    __ubuf__ uint16_t* tmpAddr)
{
    auto lp = ComputeLoopParams(dataLen, blockCount);
    const uint16_t N = lp.N;
    const uint16_t dataLenSingleLoop = lp.dataLenSingleLoop;
    const uint16_t regLoop = lp.regLoop;
    const uint16_t dataLenTailLoop = lp.dataLenTailLoop;
    const uint16_t loopSize = lp.loopSize;
    const uint16_t expOffsetInit = lp.expOffsetInit;

    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> x;
        Reg::RegTensor<uint32_t> expU32;
        Reg::RegTensor<uint32_t> expMaxU32;
        Reg::RegTensor<uint16_t> expMaxU16;
        Reg::RegTensor<uint16_t> mxScaleU16;
        Reg::RegTensor<uint8_t> mxScaleU8;
        Reg::RegTensor<uint16_t> recipU16;

        Reg::RegTensor<uint32_t> maxEleU32;
        Reg::RegTensor<uint16_t> maxEleU16;
        Reg::RegTensor<uint16_t> biasU16;
        Reg::RegTensor<uint16_t> zeroU16;
        Reg::RegTensor<uint16_t> nanU16;
        Reg::RegTensor<uint16_t> specialExpU16;
        Reg::RegTensor<uint16_t> tgtMaxExpU16;
        Reg::RegTensor<uint16_t> subNumForScaleU16;
        Reg::RegTensor<uint16_t> fp8NanU16;
        Reg::RegTensor<uint32_t> absForX32;
        Reg::RegTensor<uint32_t> maxU32;
        Reg::RegTensor<uint16_t> maxU16;
        Reg::RegTensor<uint32_t> roundBiasU32;
        Reg::RegTensor<uint32_t> oneU32;

        Reg::MaskReg infMask;
        Reg::MaskReg zeroMask;
        Reg::MaskReg specialMask;
        Reg::MaskReg invalidDataMask;
        Reg::MaskReg pregAll8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        Reg::Duplicate(maxEleU32, FP32_MX_MAX_EXP);
        Reg::Duplicate(maxEleU16, BF16_MAX_EXP);
        Reg::Duplicate(biasU16, BF16_EXP_BIAS);
        Reg::Duplicate(zeroU16, 0);
        Reg::Duplicate(nanU16, BF16_NAN_CUSTOM);
        Reg::Duplicate(specialExpU16, BF16_SPECIAL_EXP_THRESHOLD);
        Reg::Duplicate(tgtMaxExpU16, dtypeYMaxExp_);
        Reg::Duplicate(subNumForScaleU16, subNumForScale_);
        Reg::Duplicate(fp8NanU16, FP8_DEFAULT_MAX_EXP);
        Reg::Duplicate(absForX32, FP32_ABS_MASK);
        Reg::Duplicate(oneU32, 1);
        Reg::Duplicate(expMaxU32, 0);
        Reg::Duplicate(maxU32, 0);

        uint32_t pnumU32 = dataLenSingleLoop;
        uint32_t tailPnumU32 = dataLenTailLoop;
        Reg::MaskReg pnumMask32 = Reg::UpdateMask<uint32_t>(pnumU32);
        Reg::MaskReg tailPnumMask32 = Reg::UpdateMask<uint32_t>(tailPnumU32);
        uint32_t validU32 = postAxisSize_;
        Reg::MaskReg validMask32 = Reg::UpdateMask<uint32_t>(validU32);
        Reg::MaskReg validMask16 = Reg::UpdateMask<uint16_t>(validU32);

        LocalTensor<uint16_t> maxExpTensor = maxExpBuf_.Get<uint16_t>();
        auto maxExpAddr32 = (__ubuf__ uint32_t*)maxExpTensor.GetPhyAddr();

        if constexpr (calcMode == MODE_ZERO) {
            for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); ++i) {
                Reg::UnalignRegForLoad uXLd;
                Reg::LoadUnAlignPre(uXLd, xAddr + static_cast<int64_t>(i) * dataLenSingleLoop);
                Reg::LoadUnAlign(x, uXLd, xAddr + static_cast<int64_t>(i) * dataLenSingleLoop);
                Reg::And(expU32, (Reg::RegTensor<uint32_t>&)x, maxEleU32, pnumMask32);
                Reg::Max(expMaxU32, expMaxU32, expU32, pnumMask32);
            }
            Reg::UnalignRegForLoad uXLd;
            Reg::LoadUnAlignPre(uXLd, xAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop);
            Reg::LoadUnAlign(x, uXLd, xAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop);
            Reg::And(expU32, (Reg::RegTensor<uint32_t>&)x, maxEleU32, tailPnumMask32);
            Reg::Max(expU32, expMaxU32, expU32, tailPnumMask32);
            Reg::Copy<uint32_t, Reg::MaskMergeMode::MERGING>(expMaxU32, expU32, tailPnumMask32);
            Reg::RegTensor<uint32_t> zeroTmp32;
            Reg::Duplicate(zeroTmp32, 0);
            uint32_t staleNum32 = dataLenSingleLoop - tailPnumU32;
            Reg::MaskReg staleMask32 = Reg::UpdateMask<uint32_t>(staleNum32);
            Reg::Select<uint32_t>(maxU32, zeroTmp32, maxU32, staleMask32);
        } else {
            for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); ++i) {
                Reg::UnalignRegForLoad uXLd;
                Reg::LoadUnAlignPre(uXLd, xAddr + static_cast<int64_t>(i) * dataLenSingleLoop);
                Reg::LoadUnAlign(x, uXLd, xAddr + static_cast<int64_t>(i) * dataLenSingleLoop);
                Reg::And(expU32, (Reg::RegTensor<uint32_t>&)x, absForX32, pnumMask32);
                Reg::Max(maxU32, maxU32, expU32, pnumMask32);
            }
            Reg::UnalignRegForLoad uXLd;
            Reg::LoadUnAlignPre(uXLd, xAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop);
            Reg::LoadUnAlign(x, uXLd, xAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop);
            Reg::And(expU32, (Reg::RegTensor<uint32_t>&)x, absForX32, tailPnumMask32);
            Reg::Max(expU32, maxU32, expU32, tailPnumMask32);
            Reg::Copy<uint32_t, Reg::MaskMergeMode::MERGING>(maxU32, expU32, tailPnumMask32);
        }

        if (loopSize > 0) {
            uint16_t expOffset = expOffsetInit;
            if constexpr (calcMode == MODE_ZERO) {
                Reg::StoreAlign(maxExpAddr32, expMaxU32, pnumMask32);
                Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
                uint32_t maskNum = static_cast<uint32_t>(dataLenSingleLoop) - static_cast<uint32_t>(expOffset);
                Reg::MaskReg mask = Reg::UpdateMask<uint32_t>(maskNum);
                Reg::UnalignRegForLoad uBs;
                Reg::LoadUnAlignPre(uBs, maxExpAddr32);
                Reg::LoadUnAlign(expMaxU32, uBs, maxExpAddr32);
                Reg::UnalignRegForLoad uBs2;
                Reg::LoadUnAlignPre(uBs2, maxExpAddr32 + expOffset);
                Reg::LoadUnAlign(expU32, uBs2, maxExpAddr32 + expOffset);
                Reg::Max(expU32, expMaxU32, expU32, mask);
                Reg::Copy<uint32_t, Reg::MaskMergeMode::MERGING>(expMaxU32, expU32, mask);
                for (uint16_t i = 0; i < loopSize; ++i) {
                    Reg::StoreAlign(maxExpAddr32, expMaxU32, pnumMask32);
                    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
                    expOffset = static_cast<uint16_t>(expOffset / DIGIT_TWO);
                    maskNum = static_cast<uint32_t>(expOffset);
                    mask = Reg::UpdateMask<uint32_t>(maskNum);
                    Reg::UnalignRegForLoad uBs;
                    Reg::LoadUnAlignPre(uBs, maxExpAddr32);
                    Reg::LoadUnAlign(expMaxU32, uBs, maxExpAddr32);
                    Reg::UnalignRegForLoad uBs2;
                    Reg::LoadUnAlignPre(uBs2, maxExpAddr32 + expOffset);
                    Reg::LoadUnAlign(expU32, uBs2, maxExpAddr32 + expOffset);
                    Reg::Max(expMaxU32, expMaxU32, expU32, mask);
                }
            } else {
                Reg::StoreAlign(maxExpAddr32, maxU32, pnumMask32);
                Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
                uint32_t maskNum = static_cast<uint32_t>(dataLenSingleLoop) - static_cast<uint32_t>(expOffset);
                Reg::MaskReg mask = Reg::UpdateMask<uint32_t>(maskNum);
                Reg::UnalignRegForLoad uBs3;
                Reg::LoadUnAlignPre(uBs3, maxExpAddr32);
                Reg::LoadUnAlign(maxU32, uBs3, maxExpAddr32);
                Reg::UnalignRegForLoad uBs2;
                Reg::LoadUnAlignPre(uBs2, maxExpAddr32 + expOffset);
                Reg::LoadUnAlign(expU32, uBs2, maxExpAddr32 + expOffset);
                Reg::Max(expU32, maxU32, expU32, mask);
                Reg::Copy<uint32_t, Reg::MaskMergeMode::MERGING>(maxU32, expU32, mask);
                for (uint16_t i = 0; i < loopSize; ++i) {
                    Reg::StoreAlign(maxExpAddr32, maxU32, pnumMask32);
                    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
                    expOffset = static_cast<uint16_t>(expOffset / DIGIT_TWO);
                    maskNum = static_cast<uint32_t>(expOffset);
                    mask = Reg::UpdateMask<uint32_t>(maskNum);
                    Reg::UnalignRegForLoad uBs3;
                    Reg::LoadUnAlignPre(uBs3, maxExpAddr32);
                    Reg::LoadUnAlign(maxU32, uBs3, maxExpAddr32);
                    Reg::UnalignRegForLoad uBs2;
                    Reg::LoadUnAlignPre(uBs2, maxExpAddr32 + expOffset);
                    Reg::LoadUnAlign(expU32, uBs2, maxExpAddr32 + expOffset);
                    Reg::Max(maxU32, maxU32, expU32, mask);
                }
            }
        }

        if constexpr (calcMode == MODE_ZERO) {
            Reg::ShiftRights(expMaxU32, expMaxU32, FP32_PACK_SHR_NUM, pregAll32);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(expMaxU16, expMaxU32);
            Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxU16, maxEleU16, pregAll16);
            Reg::Compare<uint16_t, CMPMODE::LT>(invalidDataMask, expMaxU16, tgtMaxExpU16, pregAll16);
            Reg::Sub(expMaxU16, expMaxU16, subNumForScaleU16, pregAll16);
            Reg::Select<uint16_t>(expMaxU16, zeroU16, expMaxU16, invalidDataMask);
            Reg::ShiftRights(mxScaleU16, expMaxU16, BF16_SHR_NUM, pregAll16);
            Reg::Select<uint16_t>(mxScaleU16, mxScaleU16, fp8NanU16, infMask);
        } else {
            Reg::ShiftRights(roundBiasU32, maxU32, FP32_PACK_SHR_NUM, pregAll32);
            Reg::And(roundBiasU32, roundBiasU32, oneU32, pregAll32);
            Reg::Adds(roundBiasU32, roundBiasU32, 0x7FFF, pregAll32);
            Reg::Add(maxU32, maxU32, roundBiasU32, pregAll32);
            Reg::ShiftRights(maxU32, maxU32, FP32_PACK_SHR_NUM, pregAll32);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(maxU16, maxU32);
            Reg::And(expMaxU16, maxU16, maxEleU16, pregAll16);
            Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxU16, maxEleU16, pregAll16);
            Reg::Compare<uint16_t, CMPMODE::LT>(invalidDataMask, expMaxU16, tgtMaxExpU16, pregAll16);
            Reg::Sub(expMaxU16, maxU16, subNumForScaleU16, pregAll16);
            Reg::Select<uint16_t>(expMaxU16, zeroU16, expMaxU16, invalidDataMask);
            Reg::ShiftRights(mxScaleU16, expMaxU16, BF16_SHR_NUM, pregAll16);
            Reg::Select<uint16_t>(mxScaleU16, mxScaleU16, fp8NanU16, infMask);
        }
        Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScaleU8, mxScaleU16);
        Reg::MaskReg mxScaleMask = Reg::UpdateMask<uint8_t>(postAxisSize_);
        Reg::StoreAlign(mxScaleAddr, mxScaleU8, mxScaleMask);

        if constexpr (calcMode == MODE_ZERO) {
            Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxU16, zeroU16, pregAll16);
            Reg::Compare<uint16_t, CMPMODE::EQ>(specialMask, expMaxU16, biasU16, pregAll16);
            Reg::Sub(recipU16, biasU16, expMaxU16, pregAll16);
            Reg::Select<uint16_t>(recipU16, recipU16, nanU16, infMask);
            Reg::Select<uint16_t>(recipU16, recipU16, zeroU16, zeroMask);
            Reg::Select<uint16_t>(recipU16, specialExpU16, recipU16, specialMask);
        } else {
            Reg::And(expMaxU16, expMaxU16, maxEleU16, pregAll16);
            Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxU16, zeroU16, pregAll16);
            Reg::Compare<uint16_t, CMPMODE::EQ>(specialMask, expMaxU16, biasU16, pregAll16);
            Reg::Sub(recipU16, biasU16, expMaxU16, pregAll16);
            Reg::Select<uint16_t>(recipU16, recipU16, nanU16, infMask);
            Reg::Select<uint16_t>(recipU16, recipU16, zeroU16, zeroMask);
            Reg::Select<uint16_t>(recipU16, specialExpU16, recipU16, specialMask);
        }
        // Broadcast the per-column recip to all N row-groups (StoreUnAlign form:
        // tolerates the sub-32B strides of small-postAxis shapes).
        Reg::UnalignRegForStore uSt;
        __ubuf__ uint16_t* scaleAddr = tmpAddr;
        // The fp32 Y path loads the recip raw and its cast extracts even u16 lanes, so the
        // recip values must sit at even u16 lanes: store [r0, 0, r1, 0, ...].
        Reg::RegTensor<uint16_t> recipInter0;
        Reg::RegTensor<uint16_t> recipInter1;
        Reg::Interleave(recipInter0, recipInter1, recipU16, zeroU16);
        for (uint16_t i = 0; i < N; ++i) {
            Reg::StoreUnAlign(scaleAddr, recipInter0, uSt, static_cast<uint32_t>(dataLen * 2));
        }
        Reg::StoreUnAlignPost(scaleAddr, uSt, 0);
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ComputeScaleCeilAlg(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScaleAddr,
    __ubuf__ uint16_t* tmpAddr)
{
    if constexpr (IsSame<xDtype, bfloat16_t>::value) {
        ComputeScaleCeilAlgBf16(dataLen, blockCount, xAddr, mxScaleAddr, tmpAddr);
    } else if constexpr (IsSame<xDtype, half>::value) {
        ComputeScaleCeilAlgHalf(dataLen, blockCount, xAddr, mxScaleAddr, tmpAddr);
    } else if constexpr (IsSame<xDtype, float>::value) {
        ComputeScaleCeilAlgFp32(dataLen, blockCount, xAddr, mxScaleAddr, tmpAddr);
    } else {
        (void)dataLen;
        (void)blockCount;
        (void)xAddr;
        (void)mxScaleAddr;
        (void)tmpAddr;
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ComputeScaleCeilAlgBf16(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScaleAddr,
    __ubuf__ uint16_t* tmpAddr)
{
    auto lp = ComputeLoopParams(dataLen, blockCount);
    const uint16_t N = lp.N;
    const uint16_t dataLenSingleLoop = lp.dataLenSingleLoop;
    const uint16_t regLoop = lp.regLoop;
    const uint16_t dataLenTailLoop = lp.dataLenTailLoop;
    const uint16_t loopSize = lp.loopSize;
    const uint16_t expOffsetInit = lp.expOffsetInit;

    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> x;
        Reg::RegTensor<uint16_t> expU16;
        Reg::RegTensor<uint16_t> maxU16;
        Reg::RegTensor<uint16_t> mxScaleU16;
        Reg::RegTensor<uint8_t> mxScaleU8;
        Reg::RegTensor<uint16_t> recipU16;
        Reg::RegTensor<uint16_t> maxEleU16;
        Reg::RegTensor<uint16_t> biasU16;
        Reg::RegTensor<uint16_t> zeroU16;
        Reg::RegTensor<uint16_t> nanU16;
        Reg::RegTensor<uint16_t> specialExpU16;
        Reg::RegTensor<uint16_t> absForX;
        Reg::RegTensor<uint32_t> manAbs0FP32;
        Reg::RegTensor<uint32_t> manAbs1FP32;
        Reg::RegTensor<uint32_t> mxScale0FP32;
        Reg::RegTensor<uint32_t> mxScale1FP32;
        Reg::RegTensor<uint32_t> manForFP32;
        Reg::RegTensor<uint32_t> invMax;
        Reg::RegTensor<float> dstTypeMaxReg;
        Reg::RegTensor<uint32_t> zeroU32;
        Reg::RegTensor<uint16_t> mxScale0BF16;
        Reg::RegTensor<uint16_t> mxScale1BF16;
        Reg::MaskReg infMask;
        Reg::MaskReg zeroMask;
        Reg::MaskReg specialMask;
        Reg::MaskReg p0;
        Reg::MaskReg p1;
        Reg::MaskReg p2;
        Reg::MaskReg p3;
        Reg::MaskReg pZeroBlock;
        Reg::MaskReg pregAll8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        static constexpr Reg::CastTrait castTraitZero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitOne = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        Reg::Duplicate(maxEleU16, BF16_MAX_EXP);
        Reg::Duplicate(biasU16, BF16_EXP_BIAS);
        Reg::Duplicate(zeroU16, 0);
        Reg::Duplicate(nanU16, BF16_NAN_CUSTOM);
        Reg::Duplicate(specialExpU16, BF16_SPECIAL_EXP_THRESHOLD);
        Reg::Duplicate(absForX, BF16_ABS_MASK);
        Reg::Duplicate(manForFP32, FP32_MX_MAN_MASK);
        Reg::Duplicate(zeroU32, FP32_NUMBER_ZERO);
        Reg::Duplicate(maxU16, 0);
        uint32_t pnumU32 = dataLenSingleLoop;
        uint32_t tailPnumU32 = dataLenTailLoop;
        Reg::MaskReg pnumMask16 = Reg::UpdateMask<uint16_t>(pnumU32);
        Reg::MaskReg tailPnumMask16 = Reg::UpdateMask<uint16_t>(tailPnumU32);
        uint32_t validU32 = postAxisSize_;
        Reg::MaskReg validMask16 = Reg::UpdateMask<uint16_t>(validU32);
        LocalTensor<uint16_t> maxExpTensor = maxExpBuf_.Get<uint16_t>();
        auto maxExpAddr16 = (__ubuf__ uint16_t*)maxExpTensor.GetPhyAddr();

        for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); ++i) {
            Reg::UnalignRegForLoad uXLd;
            Reg::LoadUnAlignPre(uXLd, xAddr + static_cast<int64_t>(i) * dataLenSingleLoop);
            Reg::LoadUnAlign(x, uXLd, xAddr + static_cast<int64_t>(i) * dataLenSingleLoop);
            Reg::And(expU16, (Reg::RegTensor<uint16_t>&)x, absForX, pnumMask16);
            Reg::Max(maxU16, maxU16, expU16, pnumMask16);
        }
        Reg::UnalignRegForLoad uXLd;
        Reg::LoadUnAlignPre(uXLd, xAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop);
        Reg::LoadUnAlign(x, uXLd, xAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop);
        Reg::And(expU16, (Reg::RegTensor<uint16_t>&)x, absForX, tailPnumMask16);
        Reg::Max(expU16, maxU16, expU16, tailPnumMask16);
        Reg::Copy<uint16_t, Reg::MaskMergeMode::MERGING>(maxU16, expU16, tailPnumMask16);

        if (loopSize > 0) {
            uint16_t expOffset = expOffsetInit;
            Reg::StoreAlign(maxExpAddr16, maxU16, pnumMask16);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            uint32_t maskNum = static_cast<uint32_t>(dataLenSingleLoop) - static_cast<uint32_t>(expOffset);
            Reg::MaskReg mask = Reg::UpdateMask<uint16_t>(maskNum);
            Reg::UnalignRegForLoad uBs3;
            Reg::LoadUnAlignPre(uBs3, maxExpAddr16);
            Reg::LoadUnAlign(maxU16, uBs3, maxExpAddr16);
            Reg::UnalignRegForLoad uBs2;
            Reg::LoadUnAlignPre(uBs2, maxExpAddr16 + expOffset);
            Reg::LoadUnAlign(expU16, uBs2, maxExpAddr16 + expOffset);
            Reg::Max(expU16, maxU16, expU16, mask);
            Reg::Copy<uint16_t, Reg::MaskMergeMode::MERGING>(maxU16, expU16, mask);
            for (uint16_t i = 0; i < loopSize; ++i) {
                Reg::StoreAlign(maxExpAddr16, maxU16, pnumMask16);
                Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
                expOffset = static_cast<uint16_t>(expOffset / DIGIT_TWO);
                maskNum = static_cast<uint32_t>(expOffset);
                mask = Reg::UpdateMask<uint16_t>(maskNum);
                Reg::UnalignRegForLoad uBs3;
                Reg::LoadUnAlignPre(uBs3, maxExpAddr16);
                Reg::LoadUnAlign(maxU16, uBs3, maxExpAddr16);
                Reg::UnalignRegForLoad uBs2;
                Reg::LoadUnAlignPre(uBs2, maxExpAddr16 + expOffset);
                Reg::LoadUnAlign(expU16, uBs2, maxExpAddr16 + expOffset);
                Reg::Max(maxU16, maxU16, expU16, mask);
            }
        }

        if constexpr (calcMode == MODE_ONE) {
            Reg::Duplicate(invMax, invDtypeMax_);
            Reg::Cast<float, xDtype, castTraitZero>((Reg::RegTensor<float>&)manAbs0FP32,
                                                    (Reg::RegTensor<xDtype>&)maxU16, pregAll16);
            Reg::Cast<float, xDtype, castTraitOne>((Reg::RegTensor<float>&)manAbs1FP32, (Reg::RegTensor<xDtype>&)maxU16,
                                                   pregAll16);
            Reg::CompareScalar<uint32_t, CMPMODE::NE>(pZeroBlock, manAbs0FP32, FP32_NUMBER_ZERO, pregAll32);
            Reg::Maxs((Reg::RegTensor<float>&)manAbs0FP32, (Reg::RegTensor<float>&)manAbs0FP32, maxLowBound_,
                      pregAll32);
            Reg::Mul((Reg::RegTensor<float>&)manAbs0FP32, (Reg::RegTensor<float>&)manAbs0FP32,
                     (Reg::RegTensor<float>&)invMax, pregAll32);
        } else {
            Reg::Duplicate(dstTypeMaxReg, invDstTypeMax_);
            Reg::Cast<float, xDtype, castTraitZero>((Reg::RegTensor<float>&)manAbs0FP32,
                                                    (Reg::RegTensor<xDtype>&)maxU16, pregAll16);
            Reg::Cast<float, xDtype, castTraitOne>((Reg::RegTensor<float>&)manAbs1FP32, (Reg::RegTensor<xDtype>&)maxU16,
                                                   pregAll16);
            Reg::Mul((Reg::RegTensor<float>&)manAbs0FP32, (Reg::RegTensor<float>&)manAbs0FP32,
                     (Reg::RegTensor<float>&)dstTypeMaxReg, pregAll32);
        }
        Reg::ShiftRights(mxScale0FP32, manAbs0FP32, FP32_SHR_NUM, pregAll32);
        Reg::And(manAbs0FP32, manAbs0FP32, manForFP32, pregAll32);
        Reg::CompareScalar<uint32_t, CMPMODE::GT>(p0, mxScale0FP32, FP32_NUMBER_ZERO, pregAll32);
        Reg::CompareScalar<uint32_t, CMPMODE::LT>(p0, mxScale0FP32, FP32_NUMBER_254, p0);
        Reg::CompareScalar<uint32_t, CMPMODE::GT>(p0, manAbs0FP32, FP32_NUMBER_ZERO, p0);
        if constexpr (calcMode == MODE_ONE) {
            Reg::CompareScalar<uint32_t, CMPMODE::EQ>(p1, mxScale0FP32, FP32_NUMBER_ZERO, pregAll32);
            Reg::CompareScalar<uint32_t, CMPMODE::GT>(p1, manAbs0FP32, FP32_NUMBER_HALF, p1);
            Reg::MaskXor(p0, p0, p1, pregAll32);
        }
        Reg::Adds(manAbs0FP32, mxScale0FP32, 1, p0);
        Reg::Select(mxScale0FP32, manAbs0FP32, mxScale0FP32, p0);
        if constexpr (calcMode == MODE_ONE) {
            Reg::Select<uint32_t>(mxScale0FP32, mxScale0FP32, zeroU32, pZeroBlock);
        }
        Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(mxScale0BF16, mxScale0FP32);
        if constexpr (calcMode == MODE_ONE) {
            Reg::CompareScalar<uint32_t, CMPMODE::NE>(pZeroBlock, manAbs1FP32, FP32_NUMBER_ZERO, pregAll32);
            Reg::Maxs((Reg::RegTensor<float>&)manAbs1FP32, (Reg::RegTensor<float>&)manAbs1FP32, maxLowBound_,
                      pregAll32);
            Reg::Mul((Reg::RegTensor<float>&)manAbs1FP32, (Reg::RegTensor<float>&)manAbs1FP32,
                     (Reg::RegTensor<float>&)invMax, pregAll32);
        } else {
            Reg::Mul((Reg::RegTensor<float>&)manAbs1FP32, (Reg::RegTensor<float>&)manAbs1FP32,
                     (Reg::RegTensor<float>&)dstTypeMaxReg, pregAll32);
        }
        Reg::ShiftRights(mxScale1FP32, manAbs1FP32, FP32_SHR_NUM, pregAll32);
        Reg::And(manAbs1FP32, manAbs1FP32, manForFP32, pregAll32);
        Reg::CompareScalar<uint32_t, CMPMODE::GT>(p2, mxScale1FP32, FP32_NUMBER_ZERO, pregAll32);
        Reg::CompareScalar<uint32_t, CMPMODE::LT>(p2, mxScale1FP32, FP32_NUMBER_254, p2);
        Reg::CompareScalar<uint32_t, CMPMODE::GT>(p2, manAbs1FP32, FP32_NUMBER_ZERO, p2);
        if constexpr (calcMode == MODE_ONE) {
            Reg::CompareScalar<uint32_t, CMPMODE::EQ>(p3, mxScale1FP32, FP32_NUMBER_ZERO, pregAll32);
            Reg::CompareScalar<uint32_t, CMPMODE::GT>(p3, manAbs1FP32, FP32_NUMBER_HALF, p3);
            Reg::MaskXor(p2, p3, p2, pregAll32);
        }
        Reg::Adds(manAbs1FP32, mxScale1FP32, 1, p2);
        Reg::Select(mxScale1FP32, manAbs1FP32, mxScale1FP32, p2);
        if constexpr (calcMode == MODE_ONE) {
            Reg::Select<uint32_t>(mxScale1FP32, mxScale1FP32, zeroU32, pZeroBlock);
        }
        Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(mxScale1BF16, mxScale1FP32);
        Reg::Interleave(mxScale0BF16, mxScale1BF16, mxScale0BF16, mxScale1BF16);
        Reg::ShiftLefts(mxScaleU16, mxScale0BF16, BF16_SHR_NUM, pregAll16);
        Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScaleU8, mxScale0BF16);
        Reg::MaskReg mxScaleMask = Reg::UpdateMask<uint8_t>(postAxisSize_);
        Reg::StoreAlign(mxScaleAddr, mxScaleU8, mxScaleMask);
        Reg::Compare<uint16_t, CMPMODE::NE>(infMask, mxScaleU16, maxEleU16, pregAll16);
        Reg::Compare<uint16_t, CMPMODE::EQ>(specialMask, mxScaleU16, biasU16, pregAll16);
        Reg::Sub(recipU16, biasU16, mxScaleU16, pregAll16);
        Reg::Select<uint16_t>(recipU16, recipU16, nanU16, infMask);
        Reg::Select<uint16_t>(recipU16, specialExpU16, recipU16, specialMask);
        // Plain masked vector stores: ordered wrt the Y path's vector loads (the unalign
        // data-move path is not, which races the recip read for tail blocks).
        Reg::UnalignRegForStore uSt;
        auto scaleAddr = tmpAddr;
        for (uint16_t i = 0; i < N; ++i) {
            Reg::UnalignRegForStore uStRow;
            __ubuf__ uint8_t* rowAddr = (__ubuf__ uint8_t*)(scaleAddr + i * dataLen);
            Reg::StoreUnAlign(rowAddr, (Reg::RegTensor<uint8_t>&)recipU16, uStRow,
                              static_cast<uint32_t>(dataLen * sizeof(uint16_t)));
            Reg::StoreUnAlignPost(rowAddr, uStRow, 0);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        Reg::LoadAlign(recipU16, tmpAddr);
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ComputeScaleCeilAlgHalf(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScaleAddr,
    __ubuf__ uint16_t* tmpAddr)
{
    ComputeScaleCeilAlgBf16(dataLen, blockCount, xAddr, mxScaleAddr, tmpAddr);
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ComputeScaleCeilAlgFp32(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScaleAddr,
    __ubuf__ uint16_t* tmpAddr)
{
    auto lp = ComputeLoopParams(dataLen, blockCount);
    const uint16_t N = lp.N;
    const uint16_t dataLenSingleLoop = lp.dataLenSingleLoop;
    const uint16_t regLoop = lp.regLoop;
    const uint16_t dataLenTailLoop = lp.dataLenTailLoop;
    const uint16_t loopSize = lp.loopSize;
    const uint16_t expOffsetInit = lp.expOffsetInit;

    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> x;
        Reg::RegTensor<uint32_t> expU32;
        Reg::RegTensor<uint32_t> maxU32;
        Reg::RegTensor<uint32_t> manAbs0FP32;
        Reg::RegTensor<uint32_t> mxScale0FP32;
        Reg::RegTensor<uint16_t> mxScale0BF16;
        Reg::RegTensor<uint16_t> mxScaleU16;
        Reg::RegTensor<uint8_t> mxScaleU8;
        Reg::RegTensor<uint16_t> recipU16;
        Reg::RegTensor<uint16_t> maxEleU16;
        Reg::RegTensor<uint16_t> biasU16;
        Reg::RegTensor<uint16_t> zeroU16;
        Reg::RegTensor<uint16_t> nanU16;
        Reg::RegTensor<uint16_t> specialExpU16;
        Reg::RegTensor<uint32_t> absForX32;
        Reg::RegTensor<uint32_t> manForFP32;
        Reg::RegTensor<uint32_t> invMax;
        Reg::RegTensor<float> dstTypeMaxReg;
        Reg::RegTensor<uint32_t> zeroU32;
        Reg::MaskReg infMask;
        Reg::MaskReg specialMask;
        Reg::MaskReg p0;
        Reg::MaskReg p1;
        Reg::MaskReg pZeroBlock;
        Reg::MaskReg pregAll8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::Duplicate(maxEleU16, BF16_MAX_EXP);
        Reg::Duplicate(biasU16, BF16_EXP_BIAS);
        Reg::Duplicate(zeroU16, 0);
        Reg::Duplicate(nanU16, BF16_NAN_CUSTOM);
        Reg::Duplicate(specialExpU16, BF16_SPECIAL_EXP_THRESHOLD);
        Reg::Duplicate(absForX32, FP32_ABS_MASK);
        Reg::Duplicate(manForFP32, FP32_MX_MAN_MASK);
        Reg::Duplicate(zeroU32, FP32_NUMBER_ZERO);
        Reg::Duplicate(maxU32, 0);
        uint32_t pnumU32 = dataLenSingleLoop;
        uint32_t tailPnumU32 = dataLenTailLoop;
        Reg::MaskReg pnumMask32 = Reg::UpdateMask<uint32_t>(pnumU32);
        Reg::MaskReg tailPnumMask32 = Reg::UpdateMask<uint32_t>(tailPnumU32);
        uint32_t validU32 = postAxisSize_;
        Reg::MaskReg validMask32 = Reg::UpdateMask<uint32_t>(validU32);
        LocalTensor<uint16_t> maxExpTensor = maxExpBuf_.Get<uint16_t>();
        auto maxExpAddr32 = (__ubuf__ uint32_t*)maxExpTensor.GetPhyAddr();

        for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); ++i) {
            Reg::UnalignRegForLoad uXLd;
            Reg::LoadUnAlignPre(uXLd, xAddr + static_cast<int64_t>(i) * dataLenSingleLoop);
            Reg::LoadUnAlign(x, uXLd, xAddr + static_cast<int64_t>(i) * dataLenSingleLoop);
            Reg::And(expU32, (Reg::RegTensor<uint32_t>&)x, absForX32, pnumMask32);
            Reg::Max(maxU32, maxU32, expU32, pnumMask32);
        }
        Reg::UnalignRegForLoad uXLd;
        Reg::LoadUnAlignPre(uXLd, xAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop);
        Reg::LoadUnAlign(x, uXLd, xAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop);
        Reg::And(expU32, (Reg::RegTensor<uint32_t>&)x, absForX32, tailPnumMask32);
        Reg::Max(expU32, maxU32, expU32, tailPnumMask32);
        Reg::Copy<uint32_t, Reg::MaskMergeMode::MERGING>(maxU32, expU32, tailPnumMask32);

        if (loopSize > 0) {
            uint16_t expOffset = expOffsetInit;
            Reg::StoreAlign(maxExpAddr32, maxU32, pnumMask32);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            uint32_t maskNum = static_cast<uint32_t>(dataLenSingleLoop) - static_cast<uint32_t>(expOffset);
            Reg::MaskReg mask = Reg::UpdateMask<uint32_t>(maskNum);
            Reg::UnalignRegForLoad uBs3;
            Reg::LoadUnAlignPre(uBs3, maxExpAddr32);
            Reg::LoadUnAlign(maxU32, uBs3, maxExpAddr32);
            Reg::UnalignRegForLoad uBs2;
            Reg::LoadUnAlignPre(uBs2, maxExpAddr32 + expOffset);
            Reg::LoadUnAlign(expU32, uBs2, maxExpAddr32 + expOffset);
            Reg::Max(expU32, maxU32, expU32, mask);
            Reg::Select<uint32_t>(maxU32, expU32, maxU32, mask);
            for (uint16_t i = 0; i < loopSize; ++i) {
                Reg::StoreAlign(maxExpAddr32, maxU32, pnumMask32);
                Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
                expOffset = static_cast<uint16_t>(expOffset / DIGIT_TWO);
                maskNum = static_cast<uint32_t>(expOffset);
                mask = Reg::UpdateMask<uint32_t>(maskNum);
                Reg::UnalignRegForLoad uBs3;
                Reg::LoadUnAlignPre(uBs3, maxExpAddr32);
                Reg::LoadUnAlign(maxU32, uBs3, maxExpAddr32);
                Reg::UnalignRegForLoad uBs2;
                Reg::LoadUnAlignPre(uBs2, maxExpAddr32 + expOffset);
                Reg::LoadUnAlign(expU32, uBs2, maxExpAddr32 + expOffset);
                Reg::Max(maxU32, maxU32, expU32, mask);
            }
        }

        Reg::Copy<uint32_t, Reg::MaskMergeMode::MERGING>(manAbs0FP32, maxU32, pregAll32);
        if constexpr (calcMode == MODE_ONE) {
            Reg::Duplicate(invMax, invDtypeMax_);
            Reg::CompareScalar<uint32_t, CMPMODE::NE>(pZeroBlock, manAbs0FP32, FP32_NUMBER_ZERO, pregAll32);
            Reg::Maxs((Reg::RegTensor<float>&)manAbs0FP32, (Reg::RegTensor<float>&)manAbs0FP32, maxLowBound_,
                      pregAll32);
            Reg::Mul((Reg::RegTensor<float>&)manAbs0FP32, (Reg::RegTensor<float>&)manAbs0FP32,
                     (Reg::RegTensor<float>&)invMax, pregAll32);
        } else {
            Reg::Duplicate(dstTypeMaxReg, invDstTypeMax_);
            Reg::Mul((Reg::RegTensor<float>&)manAbs0FP32, (Reg::RegTensor<float>&)manAbs0FP32,
                     (Reg::RegTensor<float>&)dstTypeMaxReg, pregAll32);
        }
        Reg::ShiftRights(mxScale0FP32, manAbs0FP32, FP32_SHR_NUM, pregAll32);
        Reg::And(manAbs0FP32, manAbs0FP32, manForFP32, pregAll32);
        Reg::CompareScalar<uint32_t, CMPMODE::GT>(p0, mxScale0FP32, FP32_NUMBER_ZERO, pregAll32);
        Reg::CompareScalar<uint32_t, CMPMODE::LT>(p0, mxScale0FP32, FP32_NUMBER_254, p0);
        Reg::CompareScalar<uint32_t, CMPMODE::GT>(p0, manAbs0FP32, FP32_NUMBER_ZERO, p0);
        if constexpr (calcMode == MODE_ONE) {
            Reg::CompareScalar<uint32_t, CMPMODE::EQ>(p1, mxScale0FP32, FP32_NUMBER_ZERO, pregAll32);
            Reg::CompareScalar<uint32_t, CMPMODE::GT>(p1, manAbs0FP32, FP32_NUMBER_HALF, p1);
            Reg::MaskXor(p0, p0, p1, pregAll32);
        }
        Reg::Adds(manAbs0FP32, mxScale0FP32, 1, p0);
        Reg::Select(mxScale0FP32, manAbs0FP32, mxScale0FP32, p0);
        if constexpr (calcMode == MODE_ONE) {
            Reg::Select<uint32_t>(mxScale0FP32, mxScale0FP32, zeroU32, pZeroBlock);
        }
        Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(mxScale0BF16, mxScale0FP32);
        Reg::ShiftLefts(mxScaleU16, mxScale0BF16, BF16_SHR_NUM, pregAll16);
        Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScaleU8, mxScale0BF16);
        Reg::MaskReg mxScaleMask = Reg::UpdateMask<uint8_t>(postAxisSize_);
        Reg::StoreAlign(mxScaleAddr, mxScaleU8, mxScaleMask);
        Reg::Compare<uint16_t, CMPMODE::NE>(infMask, mxScaleU16, maxEleU16, pregAll16);
        Reg::Compare<uint16_t, CMPMODE::EQ>(specialMask, mxScaleU16, biasU16, pregAll16);
        Reg::Sub(recipU16, biasU16, mxScaleU16, pregAll16);
        Reg::Select<uint16_t>(recipU16, recipU16, nanU16, infMask);
        Reg::Select<uint16_t>(recipU16, specialExpU16, recipU16, specialMask);
        Reg::UnalignRegForStore uSt;
        auto scaleAddr = tmpAddr;
        // The fp32 Y path loads the recip raw and its cast extracts even u16 lanes, so the
        // recip values must sit at even u16 lanes: store [r0, 0, r1, 0, ...].
        Reg::RegTensor<uint16_t> recipInter0;
        Reg::RegTensor<uint16_t> recipInter1;
        Reg::Interleave(recipInter0, recipInter1, recipU16, zeroU16);
        for (uint16_t i = 0; i < N; ++i) {
            Reg::StoreUnAlign(scaleAddr, recipInter0, uSt, static_cast<uint32_t>(dataLen * 2));
        }
        Reg::StoreUnAlignPost(scaleAddr, uSt, 0);
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        Reg::LoadAlign(recipU16, tmpAddr);
    }
}

// ---------------------------------------------------------------------------
// ComputeYVf: dispatch on xDtype.
// ---------------------------------------------------------------------------
template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ComputeYVf(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* tmpAddr, __ubuf__ uint8_t* yAddr)
{
    if constexpr (IsSame<xDtype, bfloat16_t>::value) {
        ComputeYFromBf16(dataLen, blockCount, xAddr, tmpAddr, yAddr);
    } else if constexpr (IsSame<xDtype, half>::value) {
        ComputeYFromHalf(dataLen, blockCount, xAddr, tmpAddr, yAddr);
    } else if constexpr (IsSame<xDtype, float>::value) {
        ComputeYFromFp32(dataLen, blockCount, xAddr, tmpAddr, yAddr);
    } else {
        (void)dataLen;
        (void)blockCount;
        (void)xAddr;
        (void)tmpAddr;
        (void)yAddr;
    }
}

// ---------------------------------------------------------------------------
// ComputeYFromBf16: BF16 -> FP8 / FP4 quantize for one block.
// `dataLen` = postAxisSize (valid elements per row).
// Rows in UB are at alignedPostAxisSize_ stride; Y rows at alignedOutputPostAxisSize_ stride.
// ---------------------------------------------------------------------------
template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ComputeYFromBf16(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* tmpAddr, __ubuf__ uint8_t* yAddr)
{
    auto lp = ComputeLoopParams(dataLen, blockCount);
    const uint16_t N = lp.N;
    const uint16_t dataLenSingleLoop = lp.dataLenSingleLoop;
    const uint16_t regLoop = lp.regLoop;
    const uint16_t dataLenTailLoop = lp.dataLenTailLoop;
    const uint32_t loopNum0 = lp.loopNum0;
    const uint32_t loopNum1 = lp.loopNum1;
    const uint32_t tailLoopNum0 = lp.tailLoopNum0;
    const uint32_t tailLoopNum1 = lp.tailLoopNum1;

    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> x;
        Reg::RegTensor<uint16_t> recipU16;
        Reg::RegTensor<bfloat16_t> valueBF16;
        Reg::RegTensor<float> x0FP32;
        Reg::RegTensor<float> x1FP32;
        Reg::RegTensor<float> recip0FP32;
        Reg::RegTensor<float> recip1FP32;
        Reg::RegTensor<yDtype> yZero;
        Reg::RegTensor<yDtype> yOne;
        Reg::RegTensor<uint16_t> yZeroU16;
        Reg::RegTensor<uint16_t> yOneU16;
        Reg::RegTensor<uint8_t> outZero;
        Reg::RegTensor<uint8_t> outOne;
        Reg::UnalignRegForLoad u1;

        uint32_t pnumU32 = dataLenSingleLoop;
        uint32_t tailPnumU32 = dataLenTailLoop;
        uint32_t storeLen0 = static_cast<uint32_t>(loopNum0) / DIGIT_TWO;
        uint32_t storeLen1 = static_cast<uint32_t>(loopNum1) / DIGIT_TWO;
        uint32_t tailStoreLen0 = static_cast<uint32_t>(tailLoopNum0) / DIGIT_TWO;
        uint32_t tailStoreLen1 = static_cast<uint32_t>(tailLoopNum1) / DIGIT_TWO;
        Reg::MaskReg storeMask0 = Reg::UpdateMask<uint16_t>(storeLen0);
        Reg::MaskReg storeMask1 = Reg::UpdateMask<uint16_t>(storeLen1);
        Reg::MaskReg tailStoreMask0 = Reg::UpdateMask<uint16_t>(tailStoreLen0);
        Reg::MaskReg tailStoreMask1 = Reg::UpdateMask<uint16_t>(tailStoreLen1);
        Reg::MaskReg pregAll8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pnumMask16 = Reg::UpdateMask<uint16_t>(pnumU32);
        Reg::MaskReg tailPnumMask16 = Reg::UpdateMask<uint16_t>(tailPnumU32);

        static constexpr Reg::CastTrait castTraitBf16toFp4 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                              Reg::MaskMergeMode::ZEROING, roundMode};
        static constexpr Reg::CastTrait castTraitBf16toFp32Zero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                   Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitBf16toFp32One = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                                  Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitFp32toYdtype = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                                 Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        // Full-width load is mandatory: the ZERO/ONE casts consume all 128 lanes.
        // Reg::DataCopy<u16, DIST_NORM> fills only the low half and corrupts rows >= 2.
        Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_NORM>(recipU16, tmpAddr);
        Reg::Cast<float, bfloat16_t, castTraitBf16toFp32Zero>(recip0FP32, (Reg::RegTensor<bfloat16_t>&)recipU16,
                                                              pregAll16);
        Reg::Cast<float, bfloat16_t, castTraitBf16toFp32One>(recip1FP32, (Reg::RegTensor<bfloat16_t>&)recipU16,
                                                             pregAll16);

        if constexpr (IsSame<yDtype, fp4x2_e2m1_t>::value || IsSame<yDtype, fp4x2_e1m2_t>::value) {
            for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); ++i) {
                __ubuf__ xDtype* rowX = xAddr + static_cast<int64_t>(i) * dataLenSingleLoop;
                __ubuf__ uint8_t* rowY = yAddr + static_cast<int64_t>(i) * dataLenSingleLoop / DIGIT_TWO;
                Reg::UnalignRegForLoad uXLd;
                Reg::LoadUnAlignPre(uXLd, rowX);
                Reg::LoadUnAlign(x, uXLd, rowX);
                Reg::Mul(valueBF16, x, (Reg::RegTensor<bfloat16_t>&)recipU16, pnumMask16);
                Reg::Cast<yDtype, bfloat16_t, castTraitBf16toFp4>(yZero, valueBF16, pnumMask16);
                // rowY is not guaranteed 32B-aligned (mid-task tail blocks shift yAddr), so aligned stores fault here.
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(yZeroU16, (Reg::RegTensor<uint32_t>&)yZero);
                Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(outZero, yZeroU16);
                Reg::UnalignRegForStore uYSt;
                __ubuf__ uint8_t* yPtr = rowY;
                Reg::StoreUnAlign(yPtr, outZero, uYSt, static_cast<uint32_t>(dataLenSingleLoop / DIGIT_TWO));
                Reg::StoreUnAlignPost(yPtr, uYSt, 0);
            }
            __ubuf__ xDtype* rowX = xAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop;
            // Skip (regLoop-1) FULL iterations, each dataLenSingleLoop/2 bytes. Using
            // dataLenTailLoop here displaces tail rows and leaves the block tail unwritten.
            __ubuf__ uint8_t* rowY = yAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop / DIGIT_TWO;
            Reg::UnalignRegForLoad uXLd;
            Reg::LoadUnAlignPre(uXLd, rowX);
            Reg::LoadUnAlign(x, uXLd, rowX);
            Reg::Mul(valueBF16, x, (Reg::RegTensor<bfloat16_t>&)recipU16, tailPnumMask16);
            Reg::Cast<yDtype, bfloat16_t, castTraitBf16toFp4>(yZero, valueBF16, tailPnumMask16);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(yZeroU16, (Reg::RegTensor<uint32_t>&)yZero);
            Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(outZero, yZeroU16);
            Reg::UnalignRegForStore uYSt;
            __ubuf__ uint8_t* yPtr = rowY;
            Reg::StoreUnAlign(yPtr, outZero, uYSt, static_cast<uint32_t>(dataLenTailLoop / DIGIT_TWO));
            Reg::StoreUnAlignPost(yPtr, uYSt, 0);
        } else {
            for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); ++i) {
                __ubuf__ xDtype* rowX = xAddr + static_cast<int64_t>(i) * dataLenSingleLoop;
                __ubuf__ uint8_t* rowY = yAddr + static_cast<int64_t>(i) * dataLenSingleLoop;
                Reg::UnalignRegForLoad uXLd;
                Reg::LoadUnAlignPre(uXLd, rowX);
                Reg::LoadUnAlign(x, uXLd, rowX);
                Reg::Cast<float, xDtype, castTraitBf16toFp32Zero>(x0FP32, x, pregAll16);
                Reg::Cast<float, xDtype, castTraitBf16toFp32One>(x1FP32, x, pregAll16);
                Reg::Mul(x0FP32, x0FP32, recip0FP32, pregAll32);
                Reg::Mul(x1FP32, x1FP32, recip1FP32, pregAll32);
                Reg::Interleave((Reg::RegTensor<float>&)x0FP32, (Reg::RegTensor<float>&)x1FP32,
                                (Reg::RegTensor<float>&)x0FP32, (Reg::RegTensor<float>&)x1FP32);
                Reg::Cast<yDtype, float, castTraitFp32toYdtype>(yZero, x0FP32, pregAll32);
                Reg::Cast<yDtype, float, castTraitFp32toYdtype>(yOne, x1FP32, pregAll32);
                Reg::Pack(yZeroU16, (Reg::RegTensor<uint32_t>&)yZero);
                Reg::Pack(outZero, yZeroU16);
                Reg::Pack(yOneU16, (Reg::RegTensor<uint32_t>&)yOne);
                Reg::Pack(outOne, yOneU16);
                // rowY / rowY+loopNum0 may be unaligned; byte-granular unalign stores keep the compact Y layout.
                Reg::UnalignRegForStore uYSt;
                __ubuf__ uint8_t* yPtr = rowY;
                Reg::StoreUnAlign(yPtr, outZero, uYSt, loopNum0);
                if (loopNum1 > 0) {
                    Reg::StoreUnAlign(yPtr, outOne, uYSt, loopNum1);
                }
                Reg::StoreUnAlignPost(yPtr, uYSt, 0);
            }
            __ubuf__ xDtype* rowX = xAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop;
            __ubuf__ uint8_t* rowY = yAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop;
            Reg::UnalignRegForLoad uXLd;
            Reg::LoadUnAlignPre(uXLd, rowX);
            Reg::LoadUnAlign(x, uXLd, rowX);
            Reg::Cast<float, xDtype, castTraitBf16toFp32Zero>(x0FP32, x, pregAll16);
            Reg::Cast<float, xDtype, castTraitBf16toFp32One>(x1FP32, x, pregAll16);
            Reg::Mul(x0FP32, x0FP32, recip0FP32, pregAll32);
            Reg::Mul(x1FP32, x1FP32, recip1FP32, pregAll32);
            Reg::Interleave((Reg::RegTensor<float>&)x0FP32, (Reg::RegTensor<float>&)x1FP32,
                            (Reg::RegTensor<float>&)x0FP32, (Reg::RegTensor<float>&)x1FP32);
            Reg::Cast<yDtype, float, castTraitFp32toYdtype>(yZero, x0FP32, pregAll32);
            Reg::Cast<yDtype, float, castTraitFp32toYdtype>(yOne, x1FP32, pregAll32);
            Reg::Pack(yZeroU16, (Reg::RegTensor<uint32_t>&)yZero);
            Reg::Pack(outZero, yZeroU16);
            Reg::Pack(yOneU16, (Reg::RegTensor<uint32_t>&)yOne);
            Reg::Pack(outOne, yOneU16);
            Reg::UnalignRegForStore uYSt;
            __ubuf__ uint8_t* yPtr = rowY;
            Reg::StoreUnAlign(yPtr, outZero, uYSt, tailLoopNum0);
            if (tailLoopNum1 > 0) {
                Reg::StoreUnAlign(yPtr, outOne, uYSt, tailLoopNum1);
            }
            Reg::StoreUnAlignPost(yPtr, uYSt, 0);
        }
    }
}

// ---------------------------------------------------------------------------
// ComputeYFromHalf: FP16 -> FP8 / FP4 quantize for one block.
// FP16 input is widened to FP32 (ZERO/ONE layout), multiplied by the BF16 1/scale
// (widened to FP32), then cast to the target type. FP4 needs ComputeFP4FromFp32.
// ---------------------------------------------------------------------------
template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ComputeYFromHalf(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* tmpAddr, __ubuf__ uint8_t* yAddr)
{
    auto lp = ComputeLoopParams(dataLen, blockCount);
    const uint16_t N = lp.N;
    const uint16_t dataLenSingleLoop = lp.dataLenSingleLoop;
    const uint16_t regLoop = lp.regLoop;
    const uint16_t dataLenTailLoop = lp.dataLenTailLoop;
    const uint32_t loopNum0 = lp.loopNum0;
    const uint32_t loopNum1 = lp.loopNum1;
    const uint32_t tailLoopNum0 = lp.tailLoopNum0;
    const uint32_t tailLoopNum1 = lp.tailLoopNum1;

    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> x;
        Reg::RegTensor<bfloat16_t> x0BF16;
        Reg::RegTensor<bfloat16_t> x1BF16;
        Reg::RegTensor<float> x0FP32;
        Reg::RegTensor<float> x1FP32;
        Reg::RegTensor<uint16_t> recipU16;
        Reg::RegTensor<float> recip0FP32;
        Reg::RegTensor<float> recip1FP32;
        Reg::RegTensor<yDtype> yZero;
        Reg::RegTensor<yDtype> yOne;
        Reg::RegTensor<uint16_t> yZeroU16;
        Reg::RegTensor<uint16_t> yOneU16;
        Reg::RegTensor<uint8_t> outZero;
        Reg::RegTensor<uint8_t> outOne;
        Reg::UnalignRegForLoad u1;

        uint32_t pnumU32 = dataLenSingleLoop;
        uint32_t tailPnumU32 = dataLenTailLoop;
        uint32_t storeLen0 = static_cast<uint32_t>(loopNum0) / DIGIT_TWO;
        uint32_t storeLen1 = static_cast<uint32_t>(loopNum1) / DIGIT_TWO;
        uint32_t tailStoreLen0 = static_cast<uint32_t>(tailLoopNum0) / DIGIT_TWO;
        uint32_t tailStoreLen1 = static_cast<uint32_t>(tailLoopNum1) / DIGIT_TWO;
        Reg::MaskReg storeMask0 = Reg::UpdateMask<uint16_t>(storeLen0);
        Reg::MaskReg storeMask1 = Reg::UpdateMask<uint16_t>(storeLen1);
        Reg::MaskReg tailStoreMask0 = Reg::UpdateMask<uint16_t>(tailStoreLen0);
        Reg::MaskReg tailStoreMask1 = Reg::UpdateMask<uint16_t>(tailStoreLen1);
        Reg::MaskReg pregAll8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pnumMask16 = Reg::UpdateMask<uint16_t>(pnumU32);
        Reg::MaskReg tailPnumMask16 = Reg::UpdateMask<uint16_t>(tailPnumU32);

        static constexpr Reg::CastTrait castTraitBf16toFp4 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                              Reg::MaskMergeMode::ZEROING, roundMode};
        static constexpr Reg::CastTrait castTraitXdtypeToFp32Zero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                     Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitXdtypeToFp32One = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                                    Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitFp32toBf16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                               Reg::MaskMergeMode::ZEROING, roundMode};
        static constexpr Reg::CastTrait castTraitFp32toYdtype = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                                 Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        // Full-width load is mandatory: the ZERO/ONE casts consume all 128 lanes.
        Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_NORM>(recipU16, tmpAddr);
        Reg::Cast<float, bfloat16_t, castTraitXdtypeToFp32Zero>(recip0FP32, (Reg::RegTensor<bfloat16_t>&)recipU16,
                                                                pregAll16);
        Reg::Cast<float, bfloat16_t, castTraitXdtypeToFp32One>(recip1FP32, (Reg::RegTensor<bfloat16_t>&)recipU16,
                                                               pregAll16);

        if constexpr (IsSame<yDtype, fp4x2_e2m1_t>::value || IsSame<yDtype, fp4x2_e1m2_t>::value) {
            for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); ++i) {
                __ubuf__ xDtype* rowX = xAddr + static_cast<int64_t>(i) * dataLenSingleLoop;
                __ubuf__ uint8_t* rowY = yAddr + static_cast<int64_t>(i) * dataLenSingleLoop / DIGIT_TWO;
                Reg::UnalignRegForLoad uXLd;
                Reg::LoadUnAlignPre(uXLd, rowX);
                Reg::LoadUnAlign(x, uXLd, rowX);
                Reg::Cast<float, xDtype, castTraitXdtypeToFp32Zero>(x0FP32, x, pregAll16);
                Reg::Cast<float, xDtype, castTraitXdtypeToFp32One>(x1FP32, x, pregAll16);
                Reg::Mul(x0FP32, x0FP32, recip0FP32, pregAll32);
                Reg::Mul(x1FP32, x1FP32, recip1FP32, pregAll32);
                ComputeFP4FromFp32(x0FP32);
                ComputeFP4FromFp32(x1FP32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBf16>(x0BF16, x0FP32, pregAll32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)x0BF16,
                                                                        (Reg::RegTensor<uint32_t>&)x0BF16);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBf16>(x1BF16, x1FP32, pregAll32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)x1BF16,
                                                                        (Reg::RegTensor<uint32_t>&)x1BF16);
                Reg::Interleave(x0BF16, x1BF16, x0BF16, x1BF16);
                Reg::Cast<yDtype, bfloat16_t, castTraitBf16toFp4>(yZero, (Reg::RegTensor<bfloat16_t>&)x0BF16,
                                                                  pnumMask16);
                // rowY is not guaranteed 32B-aligned; pack in-register and store byte-granular.
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(yZeroU16, (Reg::RegTensor<uint32_t>&)yZero);
                Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(outZero, yZeroU16);
                Reg::UnalignRegForStore uYSt;
                __ubuf__ uint8_t* yPtr = rowY;
                Reg::StoreUnAlign(yPtr, outZero, uYSt, static_cast<uint32_t>(dataLenSingleLoop / DIGIT_TWO));
                Reg::StoreUnAlignPost(yPtr, uYSt, 0);
            }
            __ubuf__ xDtype* rowX = xAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop;
            // Skip (regLoop-1) FULL iterations, each dataLenSingleLoop/2 bytes. Using
            // dataLenTailLoop here displaces tail rows and leaves the block tail unwritten.
            __ubuf__ uint8_t* rowY = yAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop / DIGIT_TWO;
            Reg::UnalignRegForLoad uXLd;
            Reg::LoadUnAlignPre(uXLd, rowX);
            Reg::LoadUnAlign(x, uXLd, rowX);
            Reg::Cast<float, xDtype, castTraitXdtypeToFp32Zero>(x0FP32, x, pregAll16);
            Reg::Cast<float, xDtype, castTraitXdtypeToFp32One>(x1FP32, x, pregAll16);
            Reg::Mul(x0FP32, x0FP32, recip0FP32, pregAll32);
            Reg::Mul(x1FP32, x1FP32, recip1FP32, pregAll32);
            ComputeFP4FromFp32(x0FP32);
            ComputeFP4FromFp32(x1FP32);
            Reg::Cast<bfloat16_t, float, castTraitFp32toBf16>(x0BF16, x0FP32, pregAll32);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)x0BF16,
                                                                    (Reg::RegTensor<uint32_t>&)x0BF16);
            Reg::Cast<bfloat16_t, float, castTraitFp32toBf16>(x1BF16, x1FP32, pregAll32);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)x1BF16,
                                                                    (Reg::RegTensor<uint32_t>&)x1BF16);
            Reg::Interleave(x0BF16, x1BF16, x0BF16, x1BF16);
            Reg::Cast<yDtype, bfloat16_t, castTraitBf16toFp4>(yZero, (Reg::RegTensor<bfloat16_t>&)x0BF16,
                                                              tailPnumMask16);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(yZeroU16, (Reg::RegTensor<uint32_t>&)yZero);
            Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(outZero, yZeroU16);
            Reg::UnalignRegForStore uYSt;
            __ubuf__ uint8_t* yPtr = rowY;
            Reg::StoreUnAlign(yPtr, outZero, uYSt, static_cast<uint32_t>(dataLenTailLoop / DIGIT_TWO));
            Reg::StoreUnAlignPost(yPtr, uYSt, 0);
        } else {
            for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); ++i) {
                __ubuf__ xDtype* rowX = xAddr + static_cast<int64_t>(i) * dataLenSingleLoop;
                __ubuf__ uint8_t* rowY = yAddr + static_cast<int64_t>(i) * dataLenSingleLoop;
                Reg::UnalignRegForLoad uXLd;
                Reg::LoadUnAlignPre(uXLd, rowX);
                Reg::LoadUnAlign(x, uXLd, rowX);
                Reg::Cast<float, xDtype, castTraitXdtypeToFp32Zero>(x0FP32, x, pregAll16);
                Reg::Cast<float, xDtype, castTraitXdtypeToFp32One>(x1FP32, x, pregAll16);
                Reg::Mul(x0FP32, x0FP32, recip0FP32, pregAll32);
                Reg::Mul(x1FP32, x1FP32, recip1FP32, pregAll32);
                Reg::Interleave((Reg::RegTensor<float>&)x0FP32, (Reg::RegTensor<float>&)x1FP32,
                                (Reg::RegTensor<float>&)x0FP32, (Reg::RegTensor<float>&)x1FP32);
                Reg::Cast<yDtype, float, castTraitFp32toYdtype>(yZero, x0FP32, pregAll32);
                Reg::Cast<yDtype, float, castTraitFp32toYdtype>(yOne, x1FP32, pregAll32);
                Reg::Pack(yZeroU16, (Reg::RegTensor<uint32_t>&)yZero);
                Reg::Pack(outZero, yZeroU16);
                Reg::Pack(yOneU16, (Reg::RegTensor<uint32_t>&)yOne);
                Reg::Pack(outOne, yOneU16);
                // rowY / rowY+loopNum0 may be unaligned; byte-granular unalign stores keep the compact Y layout.
                Reg::UnalignRegForStore uYSt;
                __ubuf__ uint8_t* yPtr = rowY;
                Reg::StoreUnAlign(yPtr, outZero, uYSt, loopNum0);
                if (loopNum1 > 0) {
                    Reg::StoreUnAlign(yPtr, outOne, uYSt, loopNum1);
                }
                Reg::StoreUnAlignPost(yPtr, uYSt, 0);
            }
            __ubuf__ xDtype* rowX = xAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop;
            __ubuf__ uint8_t* rowY = yAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop;
            Reg::UnalignRegForLoad uXLd;
            Reg::LoadUnAlignPre(uXLd, rowX);
            Reg::LoadUnAlign(x, uXLd, rowX);
            Reg::Cast<float, xDtype, castTraitXdtypeToFp32Zero>(x0FP32, x, pregAll16);
            Reg::Cast<float, xDtype, castTraitXdtypeToFp32One>(x1FP32, x, pregAll16);
            Reg::Mul(x0FP32, x0FP32, recip0FP32, pregAll32);
            Reg::Mul(x1FP32, x1FP32, recip1FP32, pregAll32);
            Reg::Interleave((Reg::RegTensor<float>&)x0FP32, (Reg::RegTensor<float>&)x1FP32,
                            (Reg::RegTensor<float>&)x0FP32, (Reg::RegTensor<float>&)x1FP32);
            Reg::Cast<yDtype, float, castTraitFp32toYdtype>(yZero, x0FP32, pregAll32);
            Reg::Cast<yDtype, float, castTraitFp32toYdtype>(yOne, x1FP32, pregAll32);
            Reg::Pack(yZeroU16, (Reg::RegTensor<uint32_t>&)yZero);
            Reg::Pack(outZero, yZeroU16);
            Reg::Pack(yOneU16, (Reg::RegTensor<uint32_t>&)yOne);
            Reg::Pack(outOne, yOneU16);
            Reg::UnalignRegForStore uYSt;
            __ubuf__ uint8_t* yPtr = rowY;
            Reg::StoreUnAlign(yPtr, outZero, uYSt, tailLoopNum0);
            if (tailLoopNum1 > 0) {
                Reg::StoreUnAlign(yPtr, outOne, uYSt, tailLoopNum1);
            }
            Reg::StoreUnAlignPost(yPtr, uYSt, 0);
        }
    }
}

// ---------------------------------------------------------------------------
// ComputeYFromFp32: FP32 -> FP8 / FP4 quantize for one block.
// 1/scale is loaded as BF16 bits then cast to FP32; x is already FP32.
// FP4 path uses PreProcessFP32 + FP32->BF16 cast.
// ---------------------------------------------------------------------------
template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ComputeYFromFp32(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* tmpAddr, __ubuf__ uint8_t* yAddr)
{
    auto lp = ComputeLoopParams(dataLen, blockCount);
    const uint16_t dataLenSingleLoop = lp.dataLenSingleLoop;
    const uint16_t regLoop = lp.regLoop;
    const uint16_t dataLenTailLoop = lp.dataLenTailLoop;
    const uint32_t loopNum0 = lp.loopNum0;
    const uint32_t tailLoopNum0 = lp.tailLoopNum0;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> x0FP32;
        Reg::RegTensor<bfloat16_t> x0BF16;
        Reg::RegTensor<uint16_t> recipU16;
        Reg::RegTensor<float> recip0FP32;
        Reg::RegTensor<yDtype> yZero;
        Reg::RegTensor<uint16_t> yZeroU16;
        Reg::RegTensor<uint8_t> outZero;
        Reg::UnalignRegForLoad u1;

        uint32_t storeLen0 = static_cast<uint32_t>(loopNum0) / DIGIT_TWO;
        uint32_t tailStoreLen0 = static_cast<uint32_t>(tailLoopNum0) / DIGIT_TWO;
        Reg::MaskReg storeMask0 = Reg::UpdateMask<uint16_t>(storeLen0);
        Reg::MaskReg tailStoreMask0 = Reg::UpdateMask<uint16_t>(tailStoreLen0);

        uint32_t fp4CastLen = static_cast<uint32_t>(dataLenSingleLoop);
        Reg::MaskReg fp4CastMask16 = Reg::UpdateMask<uint16_t>(fp4CastLen);
        uint32_t fp4StoreLen = static_cast<uint32_t>(dataLenSingleLoop / DIGIT_TWO);
        Reg::MaskReg fp4StoreMask8 = Reg::UpdateMask<uint8_t>(fp4StoreLen);
        uint32_t fp4TailCastLen = static_cast<uint32_t>(dataLenTailLoop);
        Reg::MaskReg fp4TailCastMask16 = Reg::UpdateMask<uint16_t>(fp4TailCastLen);
        uint32_t fp4TailStoreLen = static_cast<uint32_t>(dataLenTailLoop / DIGIT_TWO);
        Reg::MaskReg fp4TailStoreMask8 = Reg::UpdateMask<uint8_t>(fp4TailStoreLen);

        Reg::MaskReg pregAll8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        static constexpr Reg::CastTrait castTraitBf16toFp4 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                              Reg::MaskMergeMode::ZEROING, roundMode};
        static constexpr Reg::CastTrait castTraitBf16ToFp32Zero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                   Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        static constexpr Reg::CastTrait castTraitFp32toBf16Zero = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                                   Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
        static constexpr Reg::CastTrait castTraitFp32toYdtype = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                                 Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<uint16_t>&)recip0FP32, tmpAddr);
        Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32Zero>(recip0FP32, (Reg::RegTensor<bfloat16_t>&)recip0FP32,
                                                              pregAll16);

        if constexpr (IsSame<yDtype, fp4x2_e2m1_t>::value || IsSame<yDtype, fp4x2_e1m2_t>::value) {
            for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); ++i) {
                __ubuf__ xDtype* rowX = xAddr + static_cast<int64_t>(i) * dataLenSingleLoop;
                __ubuf__ uint8_t* rowY = yAddr + static_cast<int64_t>(i) * dataLenSingleLoop / DIGIT_TWO;
                Reg::UnalignRegForLoad uXLd;
                Reg::LoadUnAlignPre(uXLd, rowX);
                Reg::LoadUnAlign(x0FP32, uXLd, rowX);
                Reg::Mul(x0FP32, x0FP32, recip0FP32, pregAll32);
                ComputeFP4FromFp32(x0FP32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBf16Zero>(x0BF16, x0FP32, pregAll32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)x0BF16,
                                                                        (Reg::RegTensor<uint32_t>&)x0BF16);
                Reg::Cast<yDtype, bfloat16_t, castTraitBf16toFp4>(yZero, (Reg::RegTensor<bfloat16_t>&)x0BF16,
                                                                  fp4CastMask16);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(yZeroU16, (Reg::RegTensor<uint32_t>&)yZero);
                Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(outZero, yZeroU16);
                // rowY is not guaranteed 32B-aligned; store byte-granular to keep the compact Y layout.
                // +32 guarantees the valid bytes (<=32B) ride the first 32B vstus burst — stores
                // shorter than 32B never reach GM; the spill is overwritten by the next iteration.
                Reg::UnalignRegForStore uYSt;
                __ubuf__ uint8_t* yPtr = rowY;
                Reg::StoreUnAlign(yPtr, outZero, uYSt, static_cast<uint32_t>(fp4StoreLen + 32));
                Reg::StoreUnAlignPost(yPtr, uYSt, 0);
            }
            __ubuf__ xDtype* rowX = xAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop;
            // Skip (regLoop-1) FULL iterations, each dataLenSingleLoop/2 bytes. Using
            // dataLenTailLoop here displaces tail rows and leaves the block tail unwritten.
            __ubuf__ uint8_t* rowY = yAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop / DIGIT_TWO;
            Reg::UnalignRegForLoad uXLd;
            Reg::LoadUnAlignPre(uXLd, rowX);
            Reg::LoadUnAlign(x0FP32, uXLd, rowX);
            Reg::Mul(x0FP32, x0FP32, recip0FP32, pregAll32);
            ComputeFP4FromFp32(x0FP32);
            Reg::Cast<bfloat16_t, float, castTraitFp32toBf16Zero>(x0BF16, x0FP32, pregAll32);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)x0BF16,
                                                                    (Reg::RegTensor<uint32_t>&)x0BF16);
            Reg::Cast<yDtype, bfloat16_t, castTraitBf16toFp4>(yZero, (Reg::RegTensor<bfloat16_t>&)x0BF16,
                                                              fp4TailCastMask16);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(yZeroU16, (Reg::RegTensor<uint32_t>&)yZero);
            Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(outZero, yZeroU16);
            Reg::UnalignRegForStore uYSt;
            __ubuf__ uint8_t* yPtr = rowY;
            // Same +32 burst guarantee as the full-loop store above.
            Reg::StoreUnAlign(yPtr, outZero, uYSt, static_cast<uint32_t>(fp4TailStoreLen + 32));
            Reg::StoreUnAlignPost(yPtr, uYSt, 0);
        } else {
            for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); ++i) {
                __ubuf__ xDtype* rowX = xAddr + static_cast<int64_t>(i) * dataLenSingleLoop;
                __ubuf__ uint8_t* rowY = yAddr + static_cast<int64_t>(i) * dataLenSingleLoop;
                Reg::UnalignRegForLoad uXLd;
                Reg::LoadUnAlignPre(uXLd, rowX);
                Reg::LoadUnAlign(x0FP32, uXLd, rowX);
                Reg::Mul(x0FP32, x0FP32, recip0FP32, pregAll32);
                Reg::Cast<yDtype, float, castTraitFp32toYdtype>(yZero, x0FP32, pregAll32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(yZeroU16, (Reg::RegTensor<uint32_t>&)yZero);
                Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(outZero, yZeroU16);
                // rowY is not guaranteed 32B-aligned; store byte-granular to keep the compact Y layout.
                Reg::UnalignRegForStore uYSt;
                __ubuf__ uint8_t* yPtr = rowY;
                Reg::StoreUnAlign(yPtr, outZero, uYSt, loopNum0);
                Reg::StoreUnAlignPost(yPtr, uYSt, 0);
            }
            __ubuf__ xDtype* rowX = xAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop;
            __ubuf__ uint8_t* rowY = yAddr + static_cast<int64_t>(regLoop - 1) * dataLenSingleLoop;
            Reg::UnalignRegForLoad uLd;
            Reg::LoadUnAlignPre(uLd, rowX);
            Reg::LoadUnAlign(x0FP32, uLd, rowX);
            Reg::Mul(x0FP32, x0FP32, recip0FP32, pregAll32);
            Reg::Cast<yDtype, float, castTraitFp32toYdtype>(yZero, x0FP32, pregAll32);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(yZeroU16, (Reg::RegTensor<uint32_t>&)yZero);
            Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(outZero, yZeroU16);
            Reg::UnalignRegForStore uYSt;
            __ubuf__ uint8_t* yPtr = rowY;
            Reg::StoreUnAlign(yPtr, outZero, uYSt, tailLoopNum0);
            Reg::StoreUnAlignPost(yPtr, uYSt, 0);
        }
    }
}

// ---------------------------------------------------------------------------
// ComputeFP4FromFp32: scale + truncate an FP32 value into the FP4E2M1/E1M2 range.
// Adapted from large_tail.h ComputeFP4FromHalf (renamed: works for any FP32 input,
// whether sourced from FP16-widen or native FP32).
// ---------------------------------------------------------------------------
template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ComputeFP4FromFp32(
    Reg::RegTensor<float>& in)
{
    if constexpr (IsSame<yDtype, fp4x2_e1m2_t>::value) {
        Fp32ToFp4RangeFit<roundMode, true>(in);
    } else {
        Fp32ToFp4RangeFit<roundMode, false>(in);
    }
}

// ---------------------------------------------------------------------------
// PreProcessFP32: legacy alias for ComputeFP4FromFp32 (kept for parity with
// large_tail.h naming; delegates to ComputeFP4FromFp32).
// ---------------------------------------------------------------------------
template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::PreProcessFP32(
    Reg::RegTensor<float>& in)
{
    ComputeFP4FromFp32(in);
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline typename DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::LoopParams
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ComputeLoopParams(uint16_t dataLen,
                                                                                                   uint16_t blockCount)
{
    constexpr uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(xDtype);
    constexpr uint32_t vfNum = Ops::Base::GetVRegSize() / sizeof(float);
    int64_t vfDiv = static_cast<int64_t>(vfLen) / dataLen;
    int64_t rc = static_cast<int64_t>(blockCount);
    uint16_t N = static_cast<uint16_t>(rc < vfDiv ? rc : vfDiv);
    if (N == 0) {
        N = 1;
    }
    LoopParams p;
    p.N = N;
    p.dataLenSingleLoop = static_cast<uint16_t>(N * dataLen);
    p.regLoop = static_cast<uint16_t>(Ops::Base::CeilDiv(rc, static_cast<int64_t>(N)));
    p.tailN = static_cast<uint16_t>(blockCount % N);
    if (p.tailN == 0) {
        p.tailN = N;
    }
    p.dataLenTailLoop = static_cast<uint16_t>(p.tailN * dataLen);
    p.loopSize = static_cast<uint16_t>(DIGIT_SIXTY_THREE - AscendC::ScalarCountLeadingZero(static_cast<uint64_t>(N)));
    p.rowsPow2 = static_cast<uint16_t>(1u << p.loopSize);
    p.expOffsetInit = static_cast<uint16_t>(p.rowsPow2 * dataLen);
    p.loopNum0 = p.dataLenSingleLoop <= vfNum ? p.dataLenSingleLoop : vfNum;
    p.loopNum1 = p.dataLenSingleLoop <= vfNum ? 0 : (p.dataLenSingleLoop - vfNum);
    p.tailLoopNum0 = p.dataLenTailLoop <= vfNum ? p.dataLenTailLoop : vfNum;
    p.tailLoopNum1 = p.dataLenTailLoop <= vfNum ? 0 : (p.dataLenTailLoop - vfNum);
    return p;
}

} // namespace DynamicMxQuant
#endif // DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_OPTIMIZE_HIGH_PERF_SMALL_TAIL_H
