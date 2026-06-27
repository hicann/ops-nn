/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_mx_quant_not_tail_axis_optimize_high_perf.h
 * \brief
 */

#ifndef DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_OPTIMIZE_HIGH_PERF_H
#define DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_OPTIMIZE_HIGH_PERF_H

#include "dynamic_mx_quant_common.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"

namespace DynamicMxQuant {
using namespace AscendC;

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
class DynamicMxQuantNotTailAxisOptimizeSmallTail {
public:
    __aicore__ inline DynamicMxQuantNotTailAxisOptimizeSmallTail(){};
    __aicore__ inline void Init(
        TPipe* pipe, GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, const DynamicMxQuant4OptimizeTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const DynamicMxQuant4OptimizeTilingData* tilingData);
    template <const bool needPadBlock, const bool needPadAxis, const bool canInterleave>
    __aicore__ inline void ProcessWithPad();
    __aicore__ inline void CopyIn(int64_t offset, int64_t blockCount);
    template <const bool needPadBlock, const bool needPadAxis, const bool canInterleave>
    __aicore__ inline void SplitPreAxisCompute(int64_t offset, int64_t blockCount);
    template <const bool canInterleave, const bool canMaxLowBound>
    __aicore__ inline void Compute(
        __ubuf__ DTYPE_X* lhsXAddr, __ubuf__ DTYPE_X* rhsXAddr, __ubuf__ uint8_t* mxScaleAddr,
        __ubuf__ uint8_t* lhsYAddr, __ubuf__ uint8_t* rhsYAddr, uint16_t loop0, uint16_t loop1);
    __aicore__ inline void CopyOut(int64_t offset, int64_t blockCount);

private:
    struct ComputeRegisters {
        Reg::RegTensor<DTYPE_X> x;
        Reg::RegTensor<float> x0FP32;
        Reg::RegTensor<float> x1FP32;
        Reg::RegTensor<DTYPE_Y> y;
        Reg::RegTensor<uint16_t> yU16;
        Reg::RegTensor<uint16_t> expU16;
        Reg::RegTensor<uint32_t> expU32;
        Reg::RegTensor<uint32_t> absXU32;
        Reg::RegTensor<uint16_t> absMaxU16;
        Reg::RegTensor<uint32_t> absMaxU32;
        Reg::RegTensor<uint16_t> expMaxU16;
        Reg::RegTensor<uint32_t> expMaxU32;
        Reg::RegTensor<uint16_t> mxScale;
        Reg::RegTensor<uint16_t> mxScale1;
        Reg::RegTensor<uint32_t> mxScale0U32;
        Reg::RegTensor<uint32_t> mxScale1U32;
        Reg::RegTensor<uint32_t> mxScale0Add1U32;
        Reg::RegTensor<uint32_t> mxScale1Add1U32;
        Reg::RegTensor<uint32_t> roundBiasU32;
        Reg::RegTensor<float> exp0FP32;
        Reg::RegTensor<float> exp1FP32;
        Reg::RegTensor<float> max0FP32;
        Reg::RegTensor<float> max1FP32;
    };
    struct AuxRegisters {
        Reg::RegTensor<float> dstTypeMaxReg;
        Reg::RegTensor<uint16_t> maxEleU16; // also bf16 inf
        Reg::RegTensor<uint32_t> maxEleU32;
        Reg::RegTensor<uint16_t> fp16MaxEleU16;
        Reg::RegTensor<uint16_t> fp8NanU16;
        Reg::RegTensor<uint16_t> specialExpU16;
        Reg::RegTensor<uint16_t> zeroU16;
        Reg::RegTensor<uint16_t> nanU16;
        Reg::RegTensor<uint16_t> absForX;
        Reg::RegTensor<uint32_t> absForXFP32;
        Reg::RegTensor<uint32_t> manForFP32;
        Reg::RegTensor<uint32_t> oneU32;
        Reg::RegTensor<int32_t> negZeroI32;
        Reg::RegTensor<uint16_t> bf16NegInfU16;
        Reg::RegTensor<uint16_t> tgtMaxExpU16;
        Reg::RegTensor<uint16_t> subNumForScale;
        Reg::RegTensor<uint16_t> biasU16;
        Reg::RegTensor<int32_t> maxExpFP32; // for fp32 -> fp4x2_e2m1
        Reg::RegTensor<int32_t> exp0FP32;   // for fp32 -> fp4x2_e2m1
        Reg::RegTensor<int32_t> exp1FP32;   // for fp32 -> fp4x2_e2m1
        Reg::MaskReg infMask;
        Reg::MaskReg negInfMask; // reused as negZeroMask
        Reg::MaskReg zeroMask;
        Reg::MaskReg specialMask; // reused as negValueMask
        Reg::MaskReg p0;
        Reg::MaskReg p1;
        Reg::MaskReg p2;
        Reg::MaskReg p3;
        Reg::MaskReg p4;
        Reg::MaskReg p5;
        Reg::MaskReg p6;
        Reg::MaskReg maxLowBoundMask;

        Reg::RegTensor<uint32_t> invMax;
    };

    template <const bool canInterleave>
    __aicore__ inline void InitializeConstants(AuxRegisters& regs);

    template <const bool canInterleave>
    __aicore__ inline void ComputeMaxExponents(
        ComputeRegisters& lhsRegs, ComputeRegisters& rhsRegs, AuxRegisters& auxRegs, __ubuf__ DTYPE_X* lhsXAddr,
        __ubuf__ DTYPE_X* rhsXAddr, uint16_t loop0, uint16_t loop1);

     // 0: do nothing, 1: load from ub, 2: duplicate 0
    template <const uint8_t initRHS>
    __aicore__ inline void LoadData(
        ComputeRegisters& lhsRegs, ComputeRegisters& rhsRegs, AuxRegisters& auxRegs, __ubuf__ DTYPE_X* lhsXAddr,
        __ubuf__ DTYPE_X* rhsXAddr);

    __aicore__ inline void ComputeMax(ComputeRegisters& reg, AuxRegisters& auxRegs);
    __aicore__ inline void ComputeMaxFP32(ComputeRegisters& reg, AuxRegisters& auxRegs);
    __aicore__ inline void ComputeMaxNonFP32(ComputeRegisters& reg, AuxRegisters& auxRegs);

    template <const bool canMaxLowBound>
    __aicore__ inline void ComputeScalesAndSharedExp(ComputeRegisters& regs, AuxRegisters& auxRegs);
    template <const bool canMaxLowBound>
    __aicore__ inline void ComputeScalesAndSharedExpFP32(ComputeRegisters& regs, AuxRegisters& auxRegs);
    template <const bool canMaxLowBound>
    __aicore__ inline void ComputeScalesAndSharedExpNonFP32(ComputeRegisters& regs, AuxRegisters& auxRegs);

    template <const bool isEven, const bool canMaxLowBound>
    __aicore__ inline void ComputeScaleFromAbsMaxFP32(
        AuxRegisters& auxRegs, Reg::RegTensor<uint32_t>& absMaxU32, Reg::RegTensor<uint32_t>& mxScaleU32,
        Reg::RegTensor<uint32_t>& mxScaleAdd1U32, Reg::RegTensor<uint16_t>& mxScale);
    template <const bool isEven, const bool canMaxLowBound>
    __aicore__ inline void ComputeScaleFromAbsMaxNonFP32(
        AuxRegisters& auxRegs, Reg::RegTensor<uint16_t>& absMaxU16, Reg::RegTensor<uint32_t>& absMaxU32,
        Reg::RegTensor<uint32_t>& mxScaleU32, Reg::RegTensor<uint32_t>& mxScaleAdd1U32,
        Reg::RegTensor<uint16_t>& mxScale);

    template <const bool canInterleave>
    __aicore__ inline void QuantizeValues(
        ComputeRegisters& lhsRegs, ComputeRegisters& rhsRegs, AuxRegisters& auxRegs, __ubuf__ DTYPE_X* lhsXAddr,
        __ubuf__ DTYPE_X* rhsXAddr, __ubuf__ uint8_t* lhsYAddr, __ubuf__ uint8_t* rhsYAddr, uint16_t loop0,
        uint16_t loop1);

    template <const bool canInterleave>
    __aicore__ inline void QuantizeValuesFP32(
        ComputeRegisters& lhsRegs, ComputeRegisters& rhsRegs, AuxRegisters& auxRegs, __ubuf__ DTYPE_X* lhsXAddr,
        __ubuf__ DTYPE_X* rhsXAddr, __ubuf__ uint8_t* lhsYAddr, __ubuf__ uint8_t* rhsYAddr, uint16_t loop0,
        uint16_t loop1);
    template <const bool canInterleave>
    __aicore__ inline void QuantizeValuesNonFP32(
        ComputeRegisters& lhsRegs, ComputeRegisters& rhsRegs, AuxRegisters& auxRegs, __ubuf__ DTYPE_X* lhsXAddr,
        __ubuf__ DTYPE_X* rhsXAddr, __ubuf__ uint8_t* lhsYAddr, __ubuf__ uint8_t* rhsYAddr, uint16_t loop0,
        uint16_t loop1);

    // Additional helper function declarations
    template <const bool canInterleave, const bool withRHS>
    __aicore__ inline void ProcessQuantizationFP16(
        ComputeRegisters& lhsRegs, ComputeRegisters& rhsRegs, AuxRegisters& auxRegs, __ubuf__ uint8_t* lhsYAddr,
        __ubuf__ uint8_t* rhsYAddr);
    template <const bool canInterleave, const bool withRHS>
    __aicore__ inline void ProcessQuantizationFP32(
        ComputeRegisters& lhsRegs, ComputeRegisters& rhsRegs, AuxRegisters& auxRegs, __ubuf__ uint8_t* lhsYAddr,
        __ubuf__ uint8_t* rhsYAddr);

    template <const bool canInterleave, const bool withRHS>
    __aicore__ inline void ProcessQuantizationBF16ToFP4(
        ComputeRegisters& lhsRegs, ComputeRegisters& rhsRegs, AuxRegisters& auxRegs, __ubuf__ uint8_t* lhsYAddr,
        __ubuf__ uint8_t* rhsYAddr);

    template <const bool canInterleave>
    __aicore__ inline void ProcessFP4QuantizationNonFp32(
        ComputeRegisters& regs, AuxRegisters& auxRegs, __ubuf__ uint8_t* yAddr);

    __aicore__ inline void ComputeReciprocalScale(ComputeRegisters& regs, AuxRegisters& auxRegs, bool isModeOne);

    template <const bool canInterleave>
    __aicore__ inline void ProcessFP8QuantizationNonFP32(
        ComputeRegisters& regs, AuxRegisters& auxRegs, __ubuf__ uint8_t* yAddr);
    __aicore__ inline void PreProcessFP32(Reg::RegTensor<float>& in, AuxRegisters& auxRegs);

private:
    TPipe* pipe_;
    TQue<QuePosition::VECIN, DB_BUFFER> inQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue_, mxScaleQueue_;
    GlobalTensor<DTYPE_X> xGm_;
    GlobalTensor<uint8_t> mxScaleGm_;
    GlobalTensor<uint8_t> yGm_;

    int64_t blkIdx_{0}; // compute core index, not the calculated datablock index
    int64_t usedCoreNum_{0};
    // we dont need to pad the quant axis when the quant axis <= blockSize, or axisSize % blockSize == 0
    bool needPadAxis_{false};
    // we need to pad the block when the block num in axis is odd, for interleave output
    bool needPadBlock_{false};
    bool canInterleave_{false};
    bool needPadPostAxis_{false};
    int64_t quantAxisSize_{0};
    uint32_t alignedPostAxisSize_{0}, alignedOutputPostAxisSize_{0}, postAxisSize_{0}, outputPostAxisSize_{0};
    uint16_t blockSize_{0}, tailBlockSize_{0};
    int64_t blockNumInAxis_{0}, padBlockNumInAxis_{0}, totalBlockNum_{0}, blockNumPerTask_{0};
    int64_t totalTaskNum_{0}, avgTaskNum_{0}, tailTaskNum_{0}, blockNumLastTask_{0};
    int64_t taskStartIdx_{0}, taskEndIdx_{0};
    DataCopyPadExtParams<DTYPE_X> padParams_{false, 0, 0, 0};
    int64_t nextInRowOffset_, nextOutRowOffset_;
    uint32_t inStride_, outStride_, scaleStride_;
    uint16_t dtypeYMaxExp_{0};
    uint16_t subNumForScale_{0};
    uint32_t invDtypeMax_{0};
    float invDstTypeMax_{0};
    float maxLowBound_{0.0f};
};

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::Init(
    TPipe* pipe, GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, const DynamicMxQuant4OptimizeTilingData* tilingData)
{
#if (__NPU_ARCH__ == 3510)
    SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    pipe_ = pipe;
    blkIdx_ = GetBlockIdx();
    ParseTilingData(tilingData);
    xGm_.SetGlobalBuffer((__gm__ DTYPE_X*)(x));
    mxScaleGm_.SetGlobalBuffer((__gm__ uint8_t*)(mxScale));
    yGm_.SetGlobalBuffer((__gm__ uint8_t*)(y));
    pipe_->InitBuffer(inQueue_, DB_BUFFER, blockNumPerTask_ * alignedPostAxisSize_ * blockSize_ * sizeof(DTYPE_X));
    pipe_->InitBuffer(outQueue_, DB_BUFFER, blockNumPerTask_ * alignedOutputPostAxisSize_ * blockSize_);
    pipe_->InitBuffer(mxScaleQueue_, DB_BUFFER, blockNumPerTask_ * alignedPostAxisSize_);
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ParseTilingData(
    const DynamicMxQuant4OptimizeTilingData* tilingData)
{
    quantAxisSize_ = tilingData->quantAxisSize;
    postAxisSize_ = tilingData->postAxisSize;
    blockSize_ = tilingData->blockSize;
    usedCoreNum_ = tilingData->usedCoreNum;
    needPadAxis_ = tilingData->isPad;
    tailBlockSize_ = tilingData->tailBlockSize;
    blockNumInAxis_ = tilingData->mAlignBlockCount;
    needPadBlock_ = tilingData->quantAxisIsOdd;
    alignedPostAxisSize_ = tilingData->nAlignSize;
    canInterleave_ = alignedPostAxisSize_ * DIGIT_TWO * sizeof(DTYPE_X) <= Ops::Base::GetVRegSize();
    padBlockNumInAxis_ = tilingData->mAlignGroupCount * DIGIT_TWO;
    totalBlockNum_ = tilingData->totalGroupNum * DIGIT_TWO;
    needPadPostAxis_ = tilingData->needPadPostAxis;
    blockNumPerTask_ = tilingData->blockNumPerTask;
    totalTaskNum_ = tilingData->totalTaskNum;
    avgTaskNum_ = totalTaskNum_ / usedCoreNum_;
    tailTaskNum_ = totalTaskNum_ % usedCoreNum_;
    taskStartIdx_ = blkIdx_ * avgTaskNum_ + min(blkIdx_, tailTaskNum_);
    taskEndIdx_ = taskStartIdx_ + avgTaskNum_ + (blkIdx_ < tailTaskNum_ ? 1 : 0);
    blockNumLastTask_ = totalBlockNum_ - (totalTaskNum_ - 1) * blockNumPerTask_;
    if constexpr (IsSame<DTYPE_Y, fp4x2_e2m1_t>::value || IsSame<DTYPE_Y, fp4x2_e1m2_t>::value) {
        outputPostAxisSize_ = Ceil(postAxisSize_, DIGIT_TWO);
        alignedOutputPostAxisSize_ = alignedPostAxisSize_ / DIGIT_TWO;
    } else {
        outputPostAxisSize_ = postAxisSize_;
        alignedOutputPostAxisSize_ = alignedPostAxisSize_;
    }
    padParams_.rightPadding = AlignUp(postAxisSize_, ONE_BLK_SIZE / sizeof(DTYPE_X)) - postAxisSize_;
    uint32_t quantedVFBlock = DIGIT_FP4_SCALE_FACTOR; // for fp8, 256B/2/32B = 4
    if constexpr (IsSame<DTYPE_Y, fp4x2_e2m1_t>::value || IsSame<DTYPE_Y, fp4x2_e1m2_t>::value) {
        quantedVFBlock = DIGIT_TWO; // for fp4x2, 256B/4/32B = 2
    }
    inStride_ =
        alignedPostAxisSize_ / ONE_BLK_SIZE * sizeof(DTYPE_X) - Ceil(postAxisSize_, ONE_BLK_SIZE / sizeof(DTYPE_X));
    outStride_ = alignedOutputPostAxisSize_ / ONE_BLK_SIZE - Ceil(outputPostAxisSize_, ONE_BLK_SIZE);
    scaleStride_ = alignedPostAxisSize_ * DIGIT_TWO / ONE_BLK_SIZE - Ceil(postAxisSize_ * DIGIT_TWO, ONE_BLK_SIZE);

    nextInRowOffset_ = (taskStartIdx_ * blockNumPerTask_) / padBlockNumInAxis_ * quantAxisSize_ +
                       (taskStartIdx_ * blockNumPerTask_) % padBlockNumInAxis_ * blockSize_;
    nextOutRowOffset_ = nextInRowOffset_;
    // Determine target data type maximum exponent
    invDstTypeMax_ = tilingData->invDstTypeMax;
    maxLowBound_ = tilingData->maxLowBound;

    if constexpr (IsSame<DTYPE_Y, fp8_e4m3fn_t>::value) {
        dtypeYMaxExp_ = FP8_E4M3_MAX_EXP;
        invDtypeMax_ = FP8_E4M3_MAX_FLOAT_BITS;
    } else if constexpr (IsSame<DTYPE_Y, fp8_e5m2_t>::value) {
        dtypeYMaxExp_ = FP8_E5M2_MAX_EXP;
        invDtypeMax_ = FP8_E5M2_MAX_FLOAT_BITS;
    } else if constexpr (IsSame<DTYPE_Y, fp4x2_e2m1_t>::value) {
        dtypeYMaxExp_ = FP4_E2M1_BF16_MAX_EXP;
    }
    if (calcMode == MODE_TWO) {
        subNumForScale_ = static_cast<uint16_t>(tilingData->subNumForScale);
    } else {
        subNumForScale_ = dtypeYMaxExp_;
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::Process()
{
    if (blkIdx_ >= usedCoreNum_) {
        return;
    }

    if (needPadBlock_) {
        if (needPadAxis_) {
            if (canInterleave_) {
                ProcessWithPad<true, true, true>();
            } else {
                ProcessWithPad<true, true, false>();
            }
        } else {
            if (canInterleave_) {
                ProcessWithPad<true, false, true>();
            } else {
                ProcessWithPad<true, false, false>();
            }
        }
    } else {
        if (needPadAxis_) {
            if (canInterleave_) {
                ProcessWithPad<false, true, true>();
            } else {
                ProcessWithPad<false, true, false>();
            }
        } else {
            if (canInterleave_) {
                ProcessWithPad<false, false, true>();
            } else {
                ProcessWithPad<false, false, false>();
            }
        }
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
template <const bool needPadBlock, const bool needPadAxis, const bool canInterleave>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ProcessWithPad()
{
    int64_t blockIdx = taskStartIdx_ * blockNumPerTask_;
    int64_t blockCount = taskStartIdx_ == totalTaskNum_ - 1 ? blockNumLastTask_ : blockNumPerTask_;
    CopyIn(blockIdx, blockCount);

    for (int64_t taskIdx = taskStartIdx_ + 1; taskIdx < taskEndIdx_; taskIdx++) {
        int64_t nextBlockIdx = taskIdx * blockNumPerTask_;
        int64_t nextBlockCount = taskIdx == totalTaskNum_ - 1 ? blockNumLastTask_ : blockNumPerTask_;
        CopyIn(nextBlockIdx, nextBlockCount);
        SplitPreAxisCompute<needPadBlock, needPadAxis, canInterleave>(blockIdx, blockCount);
        CopyOut(blockIdx, blockCount);
        blockIdx = nextBlockIdx;
        blockCount = nextBlockCount;
    }
    SplitPreAxisCompute<needPadBlock, needPadAxis, canInterleave>(blockIdx, blockCount);
    CopyOut(blockIdx, blockCount);
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::CopyIn(
    int64_t offset, int64_t count)
{
    int64_t rowOffset =
        (offset + count) / padBlockNumInAxis_ * quantAxisSize_ + (offset + count) % padBlockNumInAxis_ * blockSize_;
    int64_t inOffset = nextInRowOffset_ * postAxisSize_;
    uint16_t nBurst = needPadPostAxis_ ? rowOffset - nextInRowOffset_ : 1;
    uint32_t blockLen = needPadPostAxis_ ? postAxisSize_ * sizeof(DTYPE_X) :
                                           (rowOffset - nextInRowOffset_) * postAxisSize_ * sizeof(DTYPE_X);
    nextInRowOffset_ = rowOffset;
    LocalTensor<DTYPE_X> x = inQueue_.AllocTensor<DTYPE_X>();
    DataCopyExtParams copyParams = {nBurst, blockLen, 0, inStride_, 0};
    DataCopyPad(x, xGm_[inOffset], copyParams, padParams_);
    inQueue_.EnQue(x);
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::CopyOut(
    int64_t offset, int64_t count)
{
    int64_t rowOffset =
        (offset + count) / padBlockNumInAxis_ * quantAxisSize_ + (offset + count) % padBlockNumInAxis_ * blockSize_;
    int64_t outOffset = nextOutRowOffset_ * outputPostAxisSize_;
    uint16_t outNBurst = needPadPostAxis_ ? rowOffset - nextOutRowOffset_ : 1;
    uint32_t outBlockLen =
        needPadPostAxis_ ? outputPostAxisSize_ : (rowOffset - nextOutRowOffset_) * outputPostAxisSize_;
    uint16_t scaleNBurst = needPadPostAxis_ ? count / 2 : 1;
    uint32_t scaleBlockLen = needPadPostAxis_ ? postAxisSize_ * DIGIT_TWO : count * postAxisSize_;
    int64_t scaleOffset = offset * postAxisSize_; // byte
    nextOutRowOffset_ = rowOffset;

    LocalTensor<uint8_t> y = outQueue_.DeQue<uint8_t>();
    DataCopyExtParams copyOutParams = {outNBurst, outBlockLen, outStride_, 0, 0};
    DataCopyPad(yGm_[outOffset], y, copyOutParams);
    outQueue_.FreeTensor(y);

    LocalTensor<uint8_t> mxScale = mxScaleQueue_.DeQue<uint8_t>();
    DataCopyExtParams copyScaleParams = {scaleNBurst, scaleBlockLen, scaleStride_, 0, 0};
    DataCopyPad(mxScaleGm_[scaleOffset], mxScale, copyScaleParams);
    mxScaleQueue_.FreeTensor(mxScale);
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
template <const bool needPadBlock, const bool needPadAxis, const bool canInterleave>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::SplitPreAxisCompute(
    int64_t offset, int64_t count)
{
    LocalTensor<DTYPE_X> x = inQueue_.DeQue<DTYPE_X>();
    LocalTensor<uint8_t> mxScale = mxScaleQueue_.AllocTensor<uint8_t>();
    LocalTensor<uint8_t> y = outQueue_.AllocTensor<uint8_t>();

    auto xAddr = (__ubuf__ DTYPE_X*)x.GetPhyAddr();
    auto mxScaleAddr = (__ubuf__ uint8_t*)mxScale.GetPhyAddr();
    auto yAddr = (__ubuf__ uint8_t*)y.GetPhyAddr();
    // NORMAL, NORMAL
    // NORMAL, TAIL (needPadAxis)
    // TAIL, PAD (needPadBlock)
    for (int64_t i = 0; i < count / DIGIT_TWO; ++i) {
        uint16_t loop0 = blockSize_;
        uint16_t loop1 = 0;
        if constexpr (needPadBlock || needPadAxis) { // all normal blocks
            int64_t lhsBlockIdx = offset + i * 2;
            bool isLhsNormal = lhsBlockIdx % padBlockNumInAxis_ < blockNumInAxis_ - 1;
            if (needPadBlock && !isLhsNormal) { // TAIL,PAD
                loop0 = 0;
                loop1 = tailBlockSize_;
            } else if (needPadAxis && isLhsNormal) {
                int64_t rhsBlockIdx = lhsBlockIdx + 1;
                bool isRhsNormal = rhsBlockIdx % padBlockNumInAxis_ < blockNumInAxis_ - 1;
                if (!isRhsNormal) { // NORMAL, TAIL
                    loop0 = tailBlockSize_;
                    loop1 = blockSize_ - tailBlockSize_;
                }
            }
        }
        if (loop0 == 0 && loop1 != 0) {
            Compute<canInterleave, false>(
                xAddr, xAddr + (loop0 + loop1) * alignedPostAxisSize_, mxScaleAddr, yAddr,
                yAddr + (loop0 + loop1) * alignedOutputPostAxisSize_, loop0, loop1);
        } else {
            Compute<canInterleave, true>(
                xAddr, xAddr + (loop0 + loop1) * alignedPostAxisSize_, mxScaleAddr, yAddr,
                yAddr + (loop0 + loop1) * alignedOutputPostAxisSize_, loop0, loop1);
        }

        xAddr += (2 * loop0 + loop1) * alignedPostAxisSize_;
        yAddr += (2 * loop0 + loop1) * alignedOutputPostAxisSize_;
        mxScaleAddr += 2 * alignedPostAxisSize_;
    }

    inQueue_.FreeTensor(x);
    mxScaleQueue_.EnQue(mxScale);
    outQueue_.EnQue(y);
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
template <const bool canInterleave, const bool canMaxLowBound>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::Compute(
    __ubuf__ DTYPE_X* lhsXAddr, __ubuf__ DTYPE_X* rhsXAddr, __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint8_t* lhsYAddr,
    __ubuf__ uint8_t* rhsYAddr, uint16_t loop0, uint16_t loop1)
{
    __VEC_SCOPE__
    {
        ComputeRegisters lhsRegs;
        ComputeRegisters rhsRegs;
        AuxRegisters auxRegs;

        // Initialize constants and register tensors
        InitializeConstants<canInterleave>(auxRegs);

        ComputeMaxExponents<canInterleave>(lhsRegs, rhsRegs, auxRegs, lhsXAddr, rhsXAddr, loop0, loop1);

        if constexpr (!IsSame<DTYPE_X, float>::value) {
            // Compute MX scales and shared exponents
            ComputeScalesAndSharedExp<canMaxLowBound>(lhsRegs, auxRegs);
            Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(
                (Reg::RegTensor<uint8_t>&)lhsRegs.mxScale, lhsRegs.mxScale);

            if constexpr (!canInterleave) {
                ComputeScalesAndSharedExp<canMaxLowBound>(rhsRegs, auxRegs);
                Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::HIGHEST>(
                    (Reg::RegTensor<uint8_t>&)rhsRegs.mxScale, rhsRegs.mxScale);
                Reg::Or(lhsRegs.mxScale, lhsRegs.mxScale, rhsRegs.mxScale, auxRegs.p0);
            }
            Reg::DataCopy(mxScaleAddr, (Reg::RegTensor<uint8_t>&)lhsRegs.mxScale, auxRegs.p2);
        } else {
            // Compute MX scales and shared exponents
            ComputeScalesAndSharedExp<canMaxLowBound>(lhsRegs, auxRegs);
            Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(
                (Reg::RegTensor<uint8_t>&)lhsRegs.mxScale, lhsRegs.mxScale);
            Reg::DataCopy(mxScaleAddr, (Reg::RegTensor<uint8_t>&)lhsRegs.mxScale, auxRegs.p2);
            if constexpr (!canInterleave) {
                ComputeScalesAndSharedExp<canMaxLowBound>(rhsRegs, auxRegs);
                Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(
                    (Reg::RegTensor<uint8_t>&)rhsRegs.mxScale, rhsRegs.mxScale);
                Reg::DataCopy(
                    mxScaleAddr + alignedPostAxisSize_, (Reg::RegTensor<uint8_t>&)rhsRegs.mxScale, auxRegs.p2);
            }
        }

        QuantizeValues<canInterleave>(lhsRegs, rhsRegs, auxRegs, lhsXAddr, rhsXAddr, lhsYAddr, rhsYAddr, loop0, loop1);
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
template <const bool canInterleave>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::InitializeConstants(AuxRegisters& regs)
{
    // Initialize mask registers
    regs.p0 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    regs.p1 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

    if constexpr (!IsSame<DTYPE_X, float>::value) {
        uint32_t scaleMask = alignedPostAxisSize_ * DIGIT_TWO;
        regs.p2 = Reg::UpdateMask<uint8_t>(scaleMask);
    } else {
        uint32_t scaleMask = alignedPostAxisSize_;
        regs.p2 = Reg::UpdateMask<uint8_t>(scaleMask);
    }
    // regs.p2 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

    uint32_t outputMask = alignedOutputPostAxisSize_ * DIGIT_TWO;
    if constexpr (IsSame<DTYPE_Y, fp4x2_e2m1_t>::value || IsSame<DTYPE_Y, fp4x2_e1m2_t>::value) {
        outputMask = alignedOutputPostAxisSize_ * DIGIT_FP4_SCALE_FACTOR;
    }
    regs.p3 = Reg::UpdateMask<uint8_t>(outputMask);

    // Initialize constant register tensors
    Reg::Duplicate(regs.maxEleU16, BF16_MAX_EXP);
    Reg::Duplicate(regs.maxEleU32, FP32_MX_MAX_EXP);
    Reg::Duplicate(regs.fp16MaxEleU16, FP16_INF);
    Reg::Duplicate(regs.fp8NanU16, FP8_DEFAULT_MAX_EXP);
    Reg::Duplicate(regs.biasU16, BF16_EXP_BIAS);
    Reg::Duplicate(regs.zeroU16, 0);
    Reg::Duplicate(regs.nanU16, BF16_NAN_CUSTOM);
    Reg::Duplicate(regs.specialExpU16, BF16_SPECIAL_EXP_THRESHOLD);
    Reg::Duplicate(regs.bf16NegInfU16, BF16_NEG_INF);
    Reg::Duplicate(regs.tgtMaxExpU16, dtypeYMaxExp_);
    Reg::Duplicate(regs.subNumForScale, subNumForScale_);

    Reg::Duplicate(regs.invMax, invDtypeMax_);
    Reg::Duplicate(regs.absForX, BF16_ABS_MASK);
    Reg::Duplicate(regs.absForXFP32, FP32_ABS_MASK);
    Reg::Duplicate(regs.manForFP32, FP32_MX_MAN_MASK);
    Reg::Duplicate(regs.dstTypeMaxReg, invDstTypeMax_);
    Reg::Duplicate(regs.oneU32, 1);

    if constexpr (IsSame<DTYPE_X, half>::value) {
        if constexpr (IsSame<DTYPE_Y, fp4x2_e2m1_t>::value) {
            Reg::Duplicate(regs.maxExpFP32, FP32_MX_MAX_EXP);
            Reg::Duplicate(regs.negZeroI32, FP32_NEG_ZERO_BITS);
        }
        if constexpr (IsSame<DTYPE_Y, fp4x2_e1m2_t>::value) {
            Reg::Duplicate(regs.negZeroI32, FP32_NEG_ZERO_BITS);
        }
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
template <const bool canInterleave>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ComputeMaxExponents(
    ComputeRegisters& lhsRegs, ComputeRegisters& rhsRegs, AuxRegisters& auxRegs, __ubuf__ DTYPE_X* lhsXAddr,
    __ubuf__ DTYPE_X* rhsXAddr, uint16_t loop0, uint16_t loop1)
{
    if constexpr (IsSame<DTYPE_X, float>::value) {
        Reg::Duplicate(lhsRegs.expMaxU32, 0);
        Reg::Duplicate(lhsRegs.absMaxU32, 0);
    } else {
        Reg::Duplicate(lhsRegs.expMaxU16, 0);
        Reg::Duplicate(lhsRegs.absMaxU16, 0);
    }

    if constexpr (!canInterleave) {
        if constexpr (IsSame<DTYPE_X, float>::value) {
            Reg::Duplicate(rhsRegs.expMaxU32, 0);
            Reg::Duplicate(rhsRegs.absMaxU32, 0);
        } else {
            Reg::Duplicate(rhsRegs.expMaxU16, 0);
            Reg::Duplicate(rhsRegs.absMaxU16, 0);
        }
    }

    uint16_t xOffset0 = 0;
    for (uint16_t i = 0; i < loop0; ++i) {
        LoadData<1>(
            lhsRegs, rhsRegs, auxRegs, lhsXAddr + xOffset0 * alignedPostAxisSize_,
            rhsXAddr + xOffset0 * alignedPostAxisSize_);
        ComputeMax(lhsRegs, auxRegs);
        if constexpr (!canInterleave) {
            ComputeMax(rhsRegs, auxRegs);
        }
        xOffset0++;
    }

    for (uint16_t i = 0; i < loop1; ++i) {
        LoadData<2>(lhsRegs, rhsRegs, auxRegs, lhsXAddr + xOffset0 * alignedPostAxisSize_, nullptr);
        ComputeMax(lhsRegs, auxRegs);
        if constexpr (!canInterleave) {
            ComputeMax(rhsRegs, auxRegs);
        }
        xOffset0++;
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
template <const uint8_t initRHS>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::LoadData(
    ComputeRegisters& lhsRegs, ComputeRegisters& rhsRegs, AuxRegisters& auxRegs, __ubuf__ DTYPE_X* lhsXAddr,
    __ubuf__ DTYPE_X* rhsXAddr)
{
    DataCopy(lhsRegs.x, lhsXAddr);
    if constexpr (initRHS == 1) {
        DataCopy(rhsRegs.x, rhsXAddr);
    } else if constexpr (initRHS == 2) {
        Reg::Duplicate(rhsRegs.x, 0);
    }
    Reg::Interleave(lhsRegs.x, rhsRegs.x, lhsRegs.x, rhsRegs.x);
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ComputeMax(
    ComputeRegisters& reg, AuxRegisters& auxRegs)
{
    if constexpr (IsSame<DTYPE_X, float>::value) {
        ComputeMaxFP32(reg, auxRegs);
    } else {
        ComputeMaxNonFP32(reg, auxRegs);
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ComputeMaxFP32(
    ComputeRegisters& reg, AuxRegisters& auxRegs)
{
    if constexpr (calcMode == MODE_ZERO) {
        Reg::And(reg.expU32, (Reg::RegTensor<uint32_t>&)reg.x, auxRegs.maxEleU32, auxRegs.p1);
        Reg::Max(reg.expMaxU32, reg.expMaxU32, reg.expU32, auxRegs.p1);
    } else {
        Reg::And(reg.absXU32, (Reg::RegTensor<uint32_t>&)reg.x, auxRegs.absForXFP32, auxRegs.p1);
        Reg::Max(reg.absMaxU32, reg.absMaxU32, reg.absXU32, auxRegs.p1);
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ComputeMaxNonFP32(
    ComputeRegisters& reg, AuxRegisters& auxRegs)
{
    if constexpr (calcMode == MODE_ZERO) {
        static constexpr Reg::CastTrait castTraitHalf2Bf16 = { Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
        if constexpr (IsSame<DTYPE_X, half>::value) {
            Reg::And(reg.expU16, (Reg::RegTensor<uint16_t>&)reg.x, auxRegs.fp16MaxEleU16, auxRegs.p0);
            Reg::CompareScalar<uint16_t, CMPMODE::EQ>(
                auxRegs.infMask, (Reg::RegTensor<uint16_t>&)reg.expU16, FP16_INF, auxRegs.p0);
            Reg::Cast<bfloat16_t, DTYPE_X, castTraitHalf2Bf16>(
                (Reg::RegTensor<bfloat16_t>&)reg.expU16, (Reg::RegTensor<DTYPE_X>&)reg.x, auxRegs.p0);
            Reg::Select<uint16_t>(
                (Reg::RegTensor<uint16_t>&)reg.expU16, auxRegs.maxEleU16, (Reg::RegTensor<uint16_t>&)reg.expU16,
                auxRegs.infMask);
            Reg::And(reg.expU16, (Reg::RegTensor<uint16_t>&)reg.expU16, auxRegs.maxEleU16, auxRegs.p0);
        } else {
            Reg::And(reg.expU16, (Reg::RegTensor<uint16_t>&)reg.x, auxRegs.maxEleU16, auxRegs.p0);
        }
        Reg::Max(reg.expMaxU16, reg.expMaxU16, reg.expU16, auxRegs.p0);
    } else {
        static constexpr Reg::CastTrait castTraitCublsHalf2Bf16 = { Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        Reg::And(reg.expU16, (Reg::RegTensor<uint16_t>&)reg.x, auxRegs.absForX, auxRegs.p0);
        if constexpr (calcMode == MODE_TWO && (IsSame<DTYPE_X, half>::value)) {
            Reg::Cast<bfloat16_t, DTYPE_X, castTraitCublsHalf2Bf16>(
                (Reg::RegTensor<bfloat16_t>&)reg.expU16, (Reg::RegTensor<DTYPE_X>&)reg.expU16, auxRegs.p0);
        }
        Reg::Max(reg.absMaxU16, reg.absMaxU16, reg.expU16, auxRegs.p0);
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
template <const bool canMaxLowBound>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ComputeScalesAndSharedExp(
    ComputeRegisters& regs, AuxRegisters& auxRegs)
{
    if constexpr (IsSame<DTYPE_X, float>::value) {
        ComputeScalesAndSharedExpFP32<canMaxLowBound>(regs, auxRegs);
    } else {
        ComputeScalesAndSharedExpNonFP32<canMaxLowBound>(regs, auxRegs);
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ComputeReciprocalScale(
    ComputeRegisters& regs, AuxRegisters& auxRegs, bool isModeOne)
{
    if (isModeOne) {
        Reg::Compare<uint16_t, CMPMODE::EQ>(auxRegs.infMask, regs.expMaxU16, auxRegs.maxEleU16, auxRegs.p0);
    } else {
        Reg::Compare<uint16_t, CMPMODE::EQ>(auxRegs.zeroMask, regs.expMaxU16, auxRegs.zeroU16, auxRegs.p0);
    }
    Reg::Compare<uint16_t, CMPMODE::EQ>(auxRegs.specialMask, regs.expMaxU16, auxRegs.biasU16, auxRegs.p0);
    Reg::Sub(regs.expMaxU16, auxRegs.biasU16, regs.expMaxU16, auxRegs.p0);
    Reg::Select<uint16_t>(regs.expMaxU16, auxRegs.nanU16, regs.expMaxU16, auxRegs.infMask);
    Reg::Select<uint16_t>(regs.expMaxU16, auxRegs.specialExpU16, regs.expMaxU16, auxRegs.specialMask);
    if (!isModeOne) {
        Reg::Select<uint16_t>(regs.expMaxU16, auxRegs.zeroU16, regs.expMaxU16, auxRegs.zeroMask);
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
template <const bool canMaxLowBound>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ComputeScalesAndSharedExpFP32(
    ComputeRegisters& regs, AuxRegisters& auxRegs)
{
    if constexpr (calcMode == MODE_ONE || calcMode == MODE_THREE) {
        ComputeScaleFromAbsMaxFP32<false, canMaxLowBound>(auxRegs, regs.absMaxU32, regs.mxScale0U32, regs.mxScale0Add1U32, regs.mxScale);
        Reg::ShiftLefts(regs.expMaxU16, regs.mxScale, BF16_SHR_NUM, auxRegs.p0);
        ComputeReciprocalScale(regs, auxRegs, true);
    } else {
        if constexpr (calcMode == MODE_TWO) {
            Reg::ShiftRights(regs.roundBiasU32, regs.absMaxU32, FP32_PACK_SHR_NUM, auxRegs.p1);
            // Round-to-nearest-even: add 0x7FFF + ((maxU32 >> 16) & 1)
            Reg::And(regs.roundBiasU32, regs.roundBiasU32, auxRegs.oneU32, auxRegs.p1);
            Reg::Adds(regs.roundBiasU32, regs.roundBiasU32, 0x7FFF, auxRegs.p1);
            Reg::Add(regs.absMaxU32, regs.absMaxU32, regs.roundBiasU32, auxRegs.p1);
            Reg::ShiftRights(regs.absMaxU32, regs.absMaxU32, FP32_PACK_SHR_NUM, auxRegs.p1);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(regs.absMaxU16, regs.absMaxU32);
            Reg::And(regs.expMaxU16, regs.absMaxU16, auxRegs.maxEleU16, auxRegs.p0);
            Reg::Compare<uint16_t, CMPMODE::EQ>(auxRegs.infMask, regs.expMaxU16, auxRegs.maxEleU16, auxRegs.p0);
            Reg::Compare<uint16_t, CMPMODE::LT>(auxRegs.specialMask, regs.expMaxU16, auxRegs.tgtMaxExpU16, auxRegs.p0);

            Reg::Sub(regs.expMaxU16, regs.absMaxU16, auxRegs.subNumForScale, auxRegs.p0);
            Reg::Select<uint16_t>(regs.expMaxU16, auxRegs.zeroU16, regs.expMaxU16, auxRegs.specialMask);
            Reg::ShiftRights(regs.mxScale, regs.expMaxU16, BF16_SHR_NUM, auxRegs.p0);

            Reg::Select<uint16_t>(regs.mxScale, auxRegs.fp8NanU16, regs.mxScale, auxRegs.infMask);
            Reg::And(regs.expMaxU16, regs.expMaxU16, auxRegs.maxEleU16, auxRegs.p0);
        } else {
            Reg::ShiftRights(regs.expMaxU32, regs.expMaxU32, FP32_PACK_SHR_NUM, auxRegs.p1);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(regs.expMaxU16, regs.expMaxU32);
            Reg::Compare<uint16_t, CMPMODE::EQ>(auxRegs.infMask, regs.expMaxU16, auxRegs.maxEleU16, auxRegs.p0);
            Reg::Compare<uint16_t, CMPMODE::LT>(auxRegs.specialMask, regs.expMaxU16, auxRegs.tgtMaxExpU16, auxRegs.p0);

            Reg::Sub(regs.expMaxU16, regs.expMaxU16, auxRegs.subNumForScale, auxRegs.p0);
            Reg::Select<uint16_t>(regs.expMaxU16, auxRegs.zeroU16, regs.expMaxU16, auxRegs.specialMask);
            Reg::ShiftRights(regs.mxScale, regs.expMaxU16, BF16_SHR_NUM, auxRegs.p0);

            Reg::Select<uint16_t>(regs.mxScale, auxRegs.fp8NanU16, regs.mxScale, auxRegs.infMask);
            Reg::And(regs.expMaxU16, regs.expMaxU16, auxRegs.maxEleU16, auxRegs.p0);
        }
        ComputeReciprocalScale(regs, auxRegs, false);
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
template <const bool canMaxLowBound>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ComputeScalesAndSharedExpNonFP32(
    ComputeRegisters& regs, AuxRegisters& auxRegs)
{
    if constexpr (calcMode == MODE_ONE || calcMode == MODE_THREE) {
        ComputeScaleFromAbsMaxNonFP32<false, canMaxLowBound>(
            auxRegs, regs.absMaxU16, (Reg::RegTensor<uint32_t>&)regs.max0FP32, regs.mxScale0U32, regs.mxScale0Add1U32,
            regs.mxScale);
        ComputeScaleFromAbsMaxNonFP32<true, canMaxLowBound>(
            auxRegs, regs.absMaxU16, (Reg::RegTensor<uint32_t>&)regs.max1FP32, regs.mxScale1U32, regs.mxScale1Add1U32,
            regs.mxScale1);

        Reg::Interleave(regs.mxScale, regs.mxScale1, regs.mxScale, regs.mxScale1);
        Reg::ShiftLefts(regs.expMaxU16, regs.mxScale, BF16_SHR_NUM, auxRegs.p0);
        ComputeReciprocalScale(regs, auxRegs, true);
    } else {
        if constexpr (calcMode == MODE_TWO) {
            Reg::And(regs.expMaxU16, regs.absMaxU16, auxRegs.maxEleU16, auxRegs.p0);
        }
        Reg::Compare<uint16_t, CMPMODE::EQ>(auxRegs.infMask, regs.expMaxU16, auxRegs.maxEleU16, auxRegs.p0);
        Reg::Compare<uint16_t, CMPMODE::LT>(auxRegs.specialMask, regs.expMaxU16, auxRegs.tgtMaxExpU16, auxRegs.p0);

        if constexpr (calcMode == MODE_TWO) {
            Reg::Sub(regs.expMaxU16, regs.absMaxU16, auxRegs.subNumForScale, auxRegs.p0);
        } else {
            Reg::Sub(regs.expMaxU16, regs.expMaxU16, auxRegs.subNumForScale, auxRegs.p0);
        }
        Reg::Select<uint16_t>(regs.expMaxU16, auxRegs.zeroU16, regs.expMaxU16, auxRegs.specialMask);
        Reg::ShiftRights(regs.mxScale, regs.expMaxU16, BF16_SHR_NUM, auxRegs.p0);

        Reg::Select<uint16_t>(regs.mxScale, auxRegs.fp8NanU16, regs.mxScale, auxRegs.infMask);
        if constexpr (calcMode == MODE_TWO) {
            Reg::And(regs.expMaxU16, regs.expMaxU16, auxRegs.maxEleU16, auxRegs.p0);
        }
        ComputeReciprocalScale(regs, auxRegs, false);
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
template <const bool isEven, const bool canMaxLowBound>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ComputeScaleFromAbsMaxFP32(
    AuxRegisters& auxRegs, Reg::RegTensor<uint32_t>& absMaxU32, Reg::RegTensor<uint32_t>& mxScaleU32,
    Reg::RegTensor<uint32_t>& mxScaleAdd1U32, Reg::RegTensor<uint16_t>& mxScale)
{
    if constexpr (calcMode == MODE_ONE) {
        if constexpr (IsSame<xDtype, float>::value) {
            if constexpr (canMaxLowBound) {
                Reg::MaskReg maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
                Reg::Maxs((Reg::RegTensor<float>&)absMaxU32, (Reg::RegTensor<float>&)absMaxU32, maxLowBound_, maskAll);
            } else {
                // padding scale do not handle maxs(absMax， maxLowBound_)
                Reg::MaskReg maskOdd = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
                Reg::MaskReg maskEven = Reg::CreateMask<float, Reg::MaskPattern::ALLF>();
                Reg::Interleave<float>(maskOdd, maskEven, maskOdd, maskEven);
                Reg::Maxs((Reg::RegTensor<float>&)absMaxU32, (Reg::RegTensor<float>&)absMaxU32, maxLowBound_, maskOdd);
            }
        } else {
            if constexpr (canMaxLowBound || !isEven) {
                Reg::MaskReg maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
                Reg::Maxs((Reg::RegTensor<float>&)absMaxU32, (Reg::RegTensor<float>&)absMaxU32, maxLowBound_, maskAll);
            }
        }
        Reg::Mul(
            (Reg::RegTensor<float>&)absMaxU32, (Reg::RegTensor<float>&)absMaxU32,
            (Reg::RegTensor<float>&)auxRegs.invMax, auxRegs.p1);
    } else {
        Reg::Mul(
            (Reg::RegTensor<float>&)absMaxU32, (Reg::RegTensor<float>&)absMaxU32, auxRegs.dstTypeMaxReg, auxRegs.p1);
    }

    Reg::ShiftRights(mxScaleU32, absMaxU32, FP32_SHR_NUM, auxRegs.p1);
    Reg::And(absMaxU32, absMaxU32, auxRegs.manForFP32, auxRegs.p1);

    Reg::CompareScalar<uint32_t, CMPMODE::GT>(auxRegs.p4, mxScaleU32, FP32_NUMBER_ZERO, auxRegs.p1);
    Reg::CompareScalar<uint32_t, CMPMODE::LT>(auxRegs.p5, mxScaleU32, FP32_NUMBER_254, auxRegs.p1);
    Reg::CompareScalar<uint32_t, CMPMODE::GT>(auxRegs.p6, absMaxU32, FP32_NUMBER_ZERO, auxRegs.p1);
    Reg::MaskAnd(auxRegs.p4, auxRegs.p4, auxRegs.p5, auxRegs.p1);
    Reg::MaskAnd(auxRegs.p4, auxRegs.p4, auxRegs.p6, auxRegs.p1);

    if constexpr (calcMode == MODE_ONE) {
        Reg::CompareScalar<uint32_t, CMPMODE::EQ>(auxRegs.p5, mxScaleU32, FP32_NUMBER_ZERO, auxRegs.p1);
        Reg::CompareScalar<uint32_t, CMPMODE::GT>(auxRegs.p6, absMaxU32, FP32_NUMBER_HALF, auxRegs.p1);
        Reg::MaskAnd(auxRegs.p5, auxRegs.p5, auxRegs.p6, auxRegs.p1);
        Reg::MaskXor(auxRegs.p4, auxRegs.p4, auxRegs.p5, auxRegs.p1);
    }

    Reg::Adds(mxScaleAdd1U32, mxScaleU32, 1, auxRegs.p4);
    Reg::Select(mxScaleU32, mxScaleAdd1U32, mxScaleU32, auxRegs.p4);

    Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(mxScale, mxScaleU32);
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
template <const bool isEven, const bool canMaxLowBound>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ComputeScaleFromAbsMaxNonFP32(
    AuxRegisters& auxRegs, Reg::RegTensor<uint16_t>& absMaxU16, Reg::RegTensor<uint32_t>& absMaxU32,
    Reg::RegTensor<uint32_t>& mxScaleU32, Reg::RegTensor<uint32_t>& mxScaleAdd1U32, Reg::RegTensor<uint16_t>& mxScale)
{
    if constexpr (isEven) {
        static constexpr Reg::CastTrait castTraitOne = { Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        Reg::Cast<float, DTYPE_X, castTraitOne>(
            (Reg::RegTensor<float>&)absMaxU32, (Reg::RegTensor<DTYPE_X>&)absMaxU16, auxRegs.p0);
    } else {
        static constexpr Reg::CastTrait castTraitZero = { Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        Reg::Cast<float, DTYPE_X, castTraitZero>(
            (Reg::RegTensor<float>&)absMaxU32, (Reg::RegTensor<DTYPE_X>&)absMaxU16, auxRegs.p0);
    }
    ComputeScaleFromAbsMaxFP32<isEven, canMaxLowBound>(auxRegs, absMaxU32, mxScaleU32, mxScaleAdd1U32, mxScale);
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
template <const bool canInterleave>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::QuantizeValues(
    ComputeRegisters& lhsRegs, ComputeRegisters& rhsRegs, AuxRegisters& auxRegs, __ubuf__ DTYPE_X* lhsXAddr,
    __ubuf__ DTYPE_X* rhsXAddr, __ubuf__ uint8_t* lhsYAddr, __ubuf__ uint8_t* rhsYAddr, uint16_t loop0, uint16_t loop1)
{
    if constexpr (IsSame<DTYPE_X, float>::value) {
        QuantizeValuesFP32<canInterleave>(
            lhsRegs, rhsRegs, auxRegs, lhsXAddr, rhsXAddr, lhsYAddr, rhsYAddr, loop0, loop1);
    } else {
        QuantizeValuesNonFP32<canInterleave>(
            lhsRegs, rhsRegs, auxRegs, lhsXAddr, rhsXAddr, lhsYAddr, rhsYAddr, loop0, loop1);
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
template <const bool canInterleave>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::QuantizeValuesFP32(
    ComputeRegisters& lhsRegs, ComputeRegisters& rhsRegs, AuxRegisters& auxRegs, __ubuf__ DTYPE_X* lhsXAddr,
    __ubuf__ DTYPE_X* rhsXAddr, __ubuf__ uint8_t* lhsYAddr, __ubuf__ uint8_t* rhsYAddr, uint16_t loop0, uint16_t loop1)
{
    uint16_t xOffset1 = 0;
    for (uint16_t i = 0; i < loop0; ++i) {
        LoadData<1>(
            lhsRegs, rhsRegs, auxRegs, lhsXAddr + xOffset1 * alignedPostAxisSize_,
            rhsXAddr + xOffset1 * alignedPostAxisSize_);
        ProcessQuantizationFP32<canInterleave, true>(
            lhsRegs, rhsRegs, auxRegs, lhsYAddr + xOffset1 * alignedOutputPostAxisSize_,
            rhsYAddr + xOffset1 * alignedOutputPostAxisSize_);
        xOffset1++;
    }
    for (uint16_t i = 0; i < loop1; ++i) {
        LoadData<0>(
            lhsRegs, rhsRegs, auxRegs, lhsXAddr + xOffset1 * alignedPostAxisSize_,
            rhsXAddr + xOffset1 * alignedPostAxisSize_);
        ProcessQuantizationFP32<canInterleave, false>(
            lhsRegs, rhsRegs, auxRegs, lhsYAddr + xOffset1 * alignedOutputPostAxisSize_,
            rhsYAddr + xOffset1 * alignedOutputPostAxisSize_);
        xOffset1++;
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
template <const bool canInterleave>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::QuantizeValuesNonFP32(
    ComputeRegisters& lhsRegs, ComputeRegisters& rhsRegs, AuxRegisters& auxRegs, __ubuf__ DTYPE_X* lhsXAddr,
    __ubuf__ DTYPE_X* rhsXAddr, __ubuf__ uint8_t* lhsYAddr, __ubuf__ uint8_t* rhsYAddr, uint16_t loop0, uint16_t loop1)
{
    uint16_t xOffset1 = 0;
    for (uint16_t i = 0; i < loop0; ++i) {
        LoadData<1>(
            lhsRegs, rhsRegs, auxRegs, lhsXAddr + xOffset1 * alignedPostAxisSize_,
            rhsXAddr + xOffset1 * alignedPostAxisSize_);
        if constexpr (
            IsSame<DTYPE_X, half>::value || IsSame<DTYPE_Y, fp8_e4m3fn_t>::value ||
            IsSame<DTYPE_Y, fp8_e5m2_t>::value) {
            ProcessQuantizationFP16<canInterleave, true>(
                lhsRegs, rhsRegs, auxRegs, lhsYAddr + xOffset1 * alignedOutputPostAxisSize_,
                rhsYAddr + xOffset1 * alignedOutputPostAxisSize_);
        } else {
            ProcessQuantizationBF16ToFP4<canInterleave, true>(
                lhsRegs, rhsRegs, auxRegs, lhsYAddr + xOffset1 * alignedOutputPostAxisSize_,
                rhsYAddr + xOffset1 * alignedOutputPostAxisSize_);
        }
        xOffset1++;
    }
    for (uint16_t i = 0; i < loop1; ++i) {
        LoadData<0>(
            lhsRegs, rhsRegs, auxRegs, lhsXAddr + xOffset1 * alignedPostAxisSize_,
            rhsXAddr + xOffset1 * alignedPostAxisSize_);
        if constexpr (
            IsSame<DTYPE_X, half>::value || IsSame<DTYPE_Y, fp8_e4m3fn_t>::value ||
            IsSame<DTYPE_Y, fp8_e5m2_t>::value) {
            ProcessQuantizationFP16<canInterleave, false>(
                lhsRegs, rhsRegs, auxRegs, lhsYAddr + xOffset1 * alignedOutputPostAxisSize_,
                rhsYAddr + xOffset1 * alignedOutputPostAxisSize_);
        } else {
            ProcessQuantizationBF16ToFP4<canInterleave, false>(
                lhsRegs, rhsRegs, auxRegs, lhsYAddr + xOffset1 * alignedOutputPostAxisSize_,
                rhsYAddr + xOffset1 * alignedOutputPostAxisSize_);
        }
        xOffset1++;
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
template <const bool canInterleave, const bool withRHS>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ProcessQuantizationFP32(
    ComputeRegisters& lhsRegs, ComputeRegisters& rhsRegs, AuxRegisters& auxRegs, __ubuf__ uint8_t* lhsYAddr,
    __ubuf__ uint8_t* rhsYAddr)
{
    static constexpr Reg::CastTrait castTraitZero = { Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr Reg::CastTrait castTraitOne = { Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    Reg::Cast<float, bfloat16_t, castTraitZero>(
        lhsRegs.exp0FP32, (Reg::RegTensor<bfloat16_t>&)lhsRegs.expMaxU16, auxRegs.p0);
    Reg::Cast<float, bfloat16_t, castTraitOne>(
        lhsRegs.exp1FP32, (Reg::RegTensor<bfloat16_t>&)lhsRegs.expMaxU16, auxRegs.p0);
    Reg::Interleave(lhsRegs.exp0FP32, lhsRegs.exp1FP32, lhsRegs.exp0FP32, lhsRegs.exp1FP32);
    Reg::Mul(lhsRegs.x0FP32, lhsRegs.x, lhsRegs.exp0FP32, auxRegs.p1);

    if (!canInterleave) {
        Reg::Cast<float, bfloat16_t, castTraitZero>(
            rhsRegs.exp0FP32, (Reg::RegTensor<bfloat16_t>&)rhsRegs.expMaxU16, auxRegs.p0);
        Reg::Cast<float, bfloat16_t, castTraitOne>(
            rhsRegs.exp1FP32, (Reg::RegTensor<bfloat16_t>&)rhsRegs.expMaxU16, auxRegs.p0);
        Reg::Interleave(rhsRegs.exp0FP32, rhsRegs.exp1FP32, rhsRegs.exp0FP32, rhsRegs.exp1FP32);
        Reg::Mul(rhsRegs.x0FP32, rhsRegs.x, rhsRegs.exp0FP32, auxRegs.p1);
    }

    Reg::DeInterleave(lhsRegs.x0FP32, rhsRegs.x0FP32, lhsRegs.x0FP32, rhsRegs.x0FP32);

    if constexpr (IsSame<DTYPE_Y, fp4x2_e2m1_t>::value || IsSame<DTYPE_Y, fp4x2_e1m2_t>::value) {
        static constexpr Reg::CastTrait castTraitDownZero = { Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, roundMode};
        static constexpr Reg::CastTrait castTraitDownZeroBF16Trunc = { Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
        // handle lhsregs
        PreProcessFP32(lhsRegs.x0FP32, auxRegs);
        Reg::Cast<bfloat16_t, float, castTraitDownZeroBF16Trunc>(
            (Reg::RegTensor<bfloat16_t>&)lhsRegs.yU16, lhsRegs.x0FP32, auxRegs.p1);
        Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(lhsRegs.yU16, (Reg::RegTensor<uint32_t>&)lhsRegs.yU16);
        Reg::Cast<DTYPE_Y, bfloat16_t, castTraitDownZero>(
            lhsRegs.y, (Reg::RegTensor<bfloat16_t>&)lhsRegs.yU16, auxRegs.p0);
        DataCopy<uint8_t, Reg::StoreDist::DIST_PACK4_B32>(lhsYAddr, (Reg::RegTensor<uint8_t>&)lhsRegs.y, auxRegs.p3);

        // handle rhsregs
        PreProcessFP32(rhsRegs.x0FP32, auxRegs);
        Reg::Cast<bfloat16_t, float, castTraitDownZeroBF16Trunc>(
            (Reg::RegTensor<bfloat16_t>&)rhsRegs.yU16, rhsRegs.x0FP32, auxRegs.p1);
        Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(rhsRegs.yU16, (Reg::RegTensor<uint32_t>&)rhsRegs.yU16);
        Reg::Cast<DTYPE_Y, bfloat16_t, castTraitDownZero>(
            rhsRegs.y, (Reg::RegTensor<bfloat16_t>&)rhsRegs.yU16, auxRegs.p0);
        DataCopy<uint8_t, Reg::StoreDist::DIST_PACK4_B32>(rhsYAddr, (Reg::RegTensor<uint8_t>&)rhsRegs.y, auxRegs.p3);
    } else {
        static constexpr Reg::CastTrait castTraitDownZero = { Reg::RegLayout::ZERO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        Reg::Cast<DTYPE_Y, float, castTraitDownZero>(lhsRegs.y, lhsRegs.x0FP32, auxRegs.p1);
        Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(
            (Reg::RegTensor<uint16_t>&)lhsRegs.y, (Reg::RegTensor<uint32_t>&)lhsRegs.y);
        DataCopy<uint8_t, Reg::StoreDist::DIST_PACK_B16>(lhsYAddr, (Reg::RegTensor<uint8_t>&)lhsRegs.y, auxRegs.p3);

        Reg::Cast<DTYPE_Y, float, castTraitDownZero>(rhsRegs.y, rhsRegs.x0FP32, auxRegs.p1);
        Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(
            (Reg::RegTensor<uint16_t>&)rhsRegs.y, (Reg::RegTensor<uint32_t>&)rhsRegs.y);
        DataCopy<uint8_t, Reg::StoreDist::DIST_PACK_B16>(rhsYAddr, (Reg::RegTensor<uint8_t>&)rhsRegs.y, auxRegs.p3);
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
template <const bool canInterleave, const bool withRHS>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ProcessQuantizationFP16(
    ComputeRegisters& lhsRegs, ComputeRegisters& rhsRegs, AuxRegisters& auxRegs, __ubuf__ uint8_t* lhsYAddr,
    __ubuf__ uint8_t* rhsYAddr)
{
    static constexpr Reg::CastTrait castTraitZero = { Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr Reg::CastTrait castTraitOne = { Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    Reg::Cast<float, bfloat16_t, castTraitZero>(
        lhsRegs.exp0FP32, (Reg::RegTensor<bfloat16_t>&)lhsRegs.expMaxU16, auxRegs.p0);
    Reg::Cast<float, bfloat16_t, castTraitOne>(
        rhsRegs.exp0FP32, (Reg::RegTensor<bfloat16_t>&)lhsRegs.expMaxU16, auxRegs.p0);
    if constexpr (!canInterleave) {
        Reg::Cast<float, bfloat16_t, castTraitZero>(
            lhsRegs.exp1FP32, (Reg::RegTensor<bfloat16_t>&)rhsRegs.expMaxU16, auxRegs.p0);
        Reg::Cast<float, bfloat16_t, castTraitOne>(
            rhsRegs.exp1FP32, (Reg::RegTensor<bfloat16_t>&)rhsRegs.expMaxU16, auxRegs.p0);
    }

    Reg::Cast<float, DTYPE_X, castTraitZero>(lhsRegs.x0FP32, lhsRegs.x, auxRegs.p0);
    Reg::Mul(lhsRegs.x0FP32, lhsRegs.x0FP32, lhsRegs.exp0FP32, auxRegs.p1);
    if constexpr (withRHS) {
        Reg::Cast<float, DTYPE_X, castTraitOne>(rhsRegs.x0FP32, lhsRegs.x, auxRegs.p0);
        Reg::Mul(rhsRegs.x0FP32, rhsRegs.x0FP32, rhsRegs.exp0FP32, auxRegs.p1);
    }
    if constexpr (!canInterleave) {
        Reg::Cast<float, DTYPE_X, castTraitZero>(lhsRegs.x1FP32, rhsRegs.x, auxRegs.p0);
        Reg::Mul(lhsRegs.x1FP32, lhsRegs.x1FP32, lhsRegs.exp1FP32, auxRegs.p1);
        if constexpr (withRHS) {
            Reg::Cast<float, DTYPE_X, castTraitOne>(rhsRegs.x1FP32, rhsRegs.x, auxRegs.p0);
            Reg::Mul(rhsRegs.x1FP32, rhsRegs.x1FP32, rhsRegs.exp1FP32, auxRegs.p1);
        }
    }

    if constexpr (IsSame<DTYPE_Y, fp4x2_e2m1_t>::value || IsSame<DTYPE_Y, fp4x2_e1m2_t>::value) {
        ProcessFP4QuantizationNonFp32<canInterleave>(lhsRegs, auxRegs, lhsYAddr);
        if constexpr (withRHS) {
            ProcessFP4QuantizationNonFp32<canInterleave>(rhsRegs, auxRegs, rhsYAddr);
        }
    } else {
        ProcessFP8QuantizationNonFP32<canInterleave>(lhsRegs, auxRegs, lhsYAddr);
        if constexpr (withRHS) {
            ProcessFP8QuantizationNonFP32<canInterleave>(rhsRegs, auxRegs, rhsYAddr);
        }
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
template <const bool canInterleave, const bool withRHS>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ProcessQuantizationBF16ToFP4(
    ComputeRegisters& lhsRegs, ComputeRegisters& rhsRegs, AuxRegisters& auxRegs, __ubuf__ uint8_t* lhsYAddr,
    __ubuf__ uint8_t* rhsYAddr)
{
    static constexpr Reg::CastTrait castTraitDownZero = { Reg::RegLayout::ZERO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING, roundMode};

    Reg::Mul(lhsRegs.x, lhsRegs.x, (Reg::RegTensor<DTYPE_X>&)lhsRegs.expMaxU16, auxRegs.p0);
    if constexpr (!canInterleave) {
        Reg::Mul(rhsRegs.x, rhsRegs.x, (Reg::RegTensor<DTYPE_X>&)rhsRegs.expMaxU16, auxRegs.p0);
    }
    Reg::DeInterleave(lhsRegs.x, rhsRegs.x, lhsRegs.x, rhsRegs.x);
    Reg::Cast<DTYPE_Y, bfloat16_t, castTraitDownZero>(lhsRegs.y, (Reg::RegTensor<bfloat16_t>&)lhsRegs.x, auxRegs.p0);
    DataCopy<uint8_t, Reg::StoreDist::DIST_PACK4_B32>(lhsYAddr, (Reg::RegTensor<uint8_t>&)lhsRegs.y, auxRegs.p3);
    if constexpr (withRHS) {
        Reg::Cast<DTYPE_Y, bfloat16_t, castTraitDownZero>(
            rhsRegs.y, (Reg::RegTensor<bfloat16_t>&)rhsRegs.x, auxRegs.p0);
        DataCopy<uint8_t, Reg::StoreDist::DIST_PACK4_B32>(rhsYAddr, (Reg::RegTensor<uint8_t>&)rhsRegs.y, auxRegs.p3);
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
template <const bool canInterleave>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ProcessFP4QuantizationNonFp32(
    ComputeRegisters& regs, AuxRegisters& auxRegs, __ubuf__ uint8_t* yAddr)
{
    static constexpr Reg::CastTrait castTraitDownZero = { Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, roundMode};
    // After Truncate in PreProcessFP32, FP32 value is at FP4E2M1 precision;
    // FP32→BF16 Cast is exact (FP4E2M1 values are exactly representable in BF16).
    static constexpr Reg::CastTrait castTraitDownZeroBF16 = { Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, roundMode};

    PreProcessFP32(regs.x0FP32, auxRegs);

    Reg::Cast<bfloat16_t, float, castTraitDownZeroBF16>(
        (Reg::RegTensor<bfloat16_t>&)regs.yU16, regs.x0FP32, auxRegs.p1);

    Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(regs.yU16, (Reg::RegTensor<uint32_t>&)regs.yU16);

    if constexpr (!canInterleave) {
        PreProcessFP32(regs.x1FP32, auxRegs);

        Reg::Cast<bfloat16_t, float, castTraitDownZeroBF16>(
            (Reg::RegTensor<bfloat16_t>&)regs.y, regs.x1FP32, auxRegs.p1);

        Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::HIGHEST>(
            (Reg::RegTensor<uint16_t>&)regs.y, (Reg::RegTensor<uint32_t>&)regs.y);

        Reg::Or<uint16_t>(
            (Reg::RegTensor<uint16_t>&)regs.yU16, (Reg::RegTensor<uint16_t>&)regs.yU16,
            (Reg::RegTensor<uint16_t>&)regs.y, auxRegs.p0);
    }

    Reg::Cast<DTYPE_Y, bfloat16_t, castTraitDownZero>(regs.y, (Reg::RegTensor<bfloat16_t>&)regs.yU16, auxRegs.p0);
    DataCopy<uint8_t, Reg::StoreDist::DIST_PACK4_B32>(yAddr, (Reg::RegTensor<uint8_t>&)regs.y, auxRegs.p3);
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
// Helper function for FP8 quantization from FP32
template <const bool canInterleave>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::ProcessFP8QuantizationNonFP32(
    ComputeRegisters& regs, AuxRegisters& auxRegs, __ubuf__ uint8_t* yAddr)
{
    static constexpr Reg::CastTrait castTraitDownZero = { Reg::RegLayout::ZERO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

    Reg::Cast<DTYPE_Y, float, castTraitDownZero>(regs.y, regs.x0FP32, auxRegs.p1);
    Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(
        (Reg::RegTensor<uint16_t>&)regs.y, (Reg::RegTensor<uint32_t>&)regs.y);

    if constexpr (!canInterleave) {
        Reg::Cast<DTYPE_Y, float, castTraitDownZero>((Reg::RegTensor<DTYPE_Y>&)regs.yU16, regs.x1FP32, auxRegs.p1);
        Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::HIGHEST>(regs.yU16, (Reg::RegTensor<uint32_t>&)regs.yU16);

        Reg::Or<uint16_t>((Reg::RegTensor<uint16_t>&)regs.y, regs.yU16, (Reg::RegTensor<uint16_t>&)regs.y, auxRegs.p0);
    }
    DataCopy<uint8_t, Reg::StoreDist::DIST_PACK_B16>(yAddr, (Reg::RegTensor<uint8_t>&)regs.y, auxRegs.p3);
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeSmallTail<xDtype, yDtype, roundMode, calcMode>::PreProcessFP32(
    Reg::RegTensor<float>& in, AuxRegisters& auxRegs)
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
    if constexpr (IsSame<DTYPE_Y, fp4x2_e1m2_t>::value) {
        Reg::Muls(in, in, FP4_SCALE_FACTOR, pregAll32);
        Reg::CompareScalar<float, CMPMODE::LT>(specialMask, in, 0, pregAll32);
        Reg::Truncate<float, roundMode>(in, in, pregAll32);
        Reg::Muls(in, in, FP4_INV_SCALE_FACTOR, pregAll32);
    } else { // fp4x2_e2m1
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
        Reg::CompareScalar<float, CMPMODE::LT>(specialMask, in, 0, pregAll32);
        Reg::Truncate<float, roundMode>(in, in, pregAll32);
        Reg::Mul(in, in, (Reg::RegTensor<float>&)exp0FP32, pregAll32);
    }
    Reg::CompareScalar<float, CMPMODE::EQ>(zeroMask, in, 0, pregAll32);
    Reg::MaskAnd(zeroMask, specialMask, zeroMask, pregAll32);
    Reg::MaskOr(zeroMask, negInfMask, zeroMask, pregAll32);
    Reg::Select<int32_t>((Reg::RegTensor<int32_t>&)in, negZero, (Reg::RegTensor<int32_t>&)in, zeroMask);
}

} // namespace DynamicMxQuant
#endif // DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_OPTIMIZE_HIGH_PERF_H
