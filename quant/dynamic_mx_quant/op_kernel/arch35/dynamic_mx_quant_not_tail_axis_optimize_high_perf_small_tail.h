/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

class DynamicMxQuantNotTailAxisOptimizeHighPerf {
public:
    __aicore__ inline DynamicMxQuantNotTailAxisOptimizeHighPerf(){};
    __aicore__ inline void Init(
        TPipe* pipe, GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, const DynamicMxQuant4OptimizeTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const DynamicMxQuant4OptimizeTilingData* tilingData);
    template <const bool needPadBlock, const bool needPadAxis, const bool canInterleave, const bool scaleAlg>
    __aicore__ inline void ProcessWithPad();
    __aicore__ inline void CopyIn(int64_t offset, int64_t blockCount);
    template <const bool needPadBlock, const bool needPadAxis, const bool canInterleave, const bool scaleAlg>
    __aicore__ inline void SplitPreAxisCompute(int64_t offset, int64_t blockCount);
    template <const bool canInterleave, const bool scaleAlg>
    __aicore__ inline void Compute(
        __ubuf__ DTYPE_X* lhsXAddr, __ubuf__ DTYPE_X* rhsXAddr, __ubuf__ uint8_t* mxScaleAddr,
        __ubuf__ uint8_t* lhsYAddr, __ubuf__ uint8_t* rhsYAddr, uint16_t loop0, uint16_t loop1);
    __aicore__ inline void CopyOut(int64_t offset, int64_t blockCount);

private:
    struct ComputeRegisters {
        AscendC::MicroAPI::RegTensor<DTYPE_X> xRegTensor;
        AscendC::MicroAPI::RegTensor<float> fp32RegTensor0;
        AscendC::MicroAPI::RegTensor<float> fp32RegTensor1;
        AscendC::MicroAPI::RegTensor<DTYPE_Y> yRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> u16RegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> expRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> expMaxRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> expMaxRegTensor1;
        AscendC::MicroAPI::RegTensor<uint16_t> mxScaleRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> mxScaleRegTensor1;
        AscendC::MicroAPI::RegTensor<uint32_t> mxScaleFP32RegTensor0;
        AscendC::MicroAPI::RegTensor<uint32_t> mxScaleFP32RegTensor1;
        AscendC::MicroAPI::RegTensor<uint32_t> mxScaleFP32Add1RegTensor0;
        AscendC::MicroAPI::RegTensor<uint32_t> mxScaleFP32Add1RegTensor1;
        AscendC::MicroAPI::RegTensor<float> expFP32RegTensor0;
        AscendC::MicroAPI::RegTensor<float> expFP32RegTensor1;
        AscendC::MicroAPI::RegTensor<float> maxFP32RegTensor0;
        AscendC::MicroAPI::RegTensor<float> maxFP32RegTensor1;
    };
    struct AuxRegisters {
        AscendC::MicroAPI::RegTensor<uint16_t> maxEleRegTensor; // also bf16 inf
        AscendC::MicroAPI::RegTensor<uint16_t> maxEleRegTensorHalf;
        AscendC::MicroAPI::RegTensor<uint16_t> fp8NanRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> specialExpRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> zeroRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> nanRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> absForX;
        AscendC::MicroAPI::RegTensor<uint32_t> manForFP32;
        AscendC::MicroAPI::RegTensor<int32_t> negZeroRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> bf16NegInfRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> tgtMaxExpRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> biasRegTensor;
        AscendC::MicroAPI::RegTensor<int32_t> maxExpFP32RegTensor; // for fp32 -> fp4x2_e2m1
        AscendC::MicroAPI::RegTensor<int32_t> expFP32RegTensor0;   // for fp32 -> fp4x2_e2m1
        AscendC::MicroAPI::RegTensor<int32_t> expFP32RegTensor1;   // for fp32 -> fp4x2_e2m1
        AscendC::MicroAPI::MaskReg infMask;
        AscendC::MicroAPI::MaskReg negInfMask; // reused as negZeroMask
        AscendC::MicroAPI::MaskReg zeroMask;
        AscendC::MicroAPI::MaskReg specialMask; // reused as negValueMask
        AscendC::MicroAPI::MaskReg p0;
        AscendC::MicroAPI::MaskReg p1;
        AscendC::MicroAPI::MaskReg p2;
        AscendC::MicroAPI::MaskReg p3;
        AscendC::MicroAPI::MaskReg p4;
        AscendC::MicroAPI::MaskReg p5;
        AscendC::MicroAPI::MaskReg p6;

        AscendC::MicroAPI::RegTensor<uint32_t> invMax;
    };

    template <RoundMode roundMode, const bool canInterleave, const bool scaleAlg>
    __aicore__ inline void ComputeWithRound(
        __ubuf__ DTYPE_X* lhsXAddr, __ubuf__ DTYPE_X* rhsXAddr, __ubuf__ uint8_t* mxScaleAddr,
        __ubuf__ uint8_t* lhsYAddr, __ubuf__ uint8_t* rhsYAddr, uint16_t loop0, uint16_t loop1);

    template <RoundMode roundMode, const bool canInterleave>
    __aicore__ inline void InitializeConstants(AuxRegisters& regs);

    template <const bool canInterleave, const bool scaleAlg>
    __aicore__ inline void ComputeMaxExponents(
        ComputeRegisters& lhsRegs, ComputeRegisters& rhsRegs, AuxRegisters& auxRegs, __ubuf__ DTYPE_X* lhsXAddr,
        __ubuf__ DTYPE_X* rhsXAddr, uint16_t loop0, uint16_t loop1);

    // 0: do nothing, 1: load from ub, 2: duplicate 0
    template <const uint8_t initRHS>
    __aicore__ inline void LoadData(
        ComputeRegisters& lhsRegs, ComputeRegisters& rhsRegs, AuxRegisters& auxRegs, __ubuf__ DTYPE_X* lhsXAddr,
        __ubuf__ DTYPE_X* rhsXAddr);

    template <RoundMode roundMode, const bool canInterleave, const bool scaleAlg>
    __aicore__ inline void ComputeScalesAndSharedExp(
        ComputeRegisters& Regs, AuxRegisters& auxRegs, __ubuf__ uint8_t* mxScaleAddr);

    template <RoundMode roundMode, const bool canInterleave>
    __aicore__ inline void QuantizeValues(
        ComputeRegisters& lhsRegs, ComputeRegisters& rhsRegs, AuxRegisters& auxRegs, __ubuf__ DTYPE_X* lhsXAddr,
        __ubuf__ DTYPE_X* rhsXAddr, __ubuf__ uint8_t* lhsYAddr, __ubuf__ uint8_t* rhsYAddr, uint16_t loop0,
        uint16_t loop1);

    // Additional helper function declarations
    template <RoundMode roundMode, const bool canInterleave, const bool withRHS>
    __aicore__ inline void ProcessFP32Quantization(
        ComputeRegisters& lhsRegs, ComputeRegisters& rhsRegs, AuxRegisters& auxRegs, __ubuf__ uint8_t* lhsYAddr,
        __ubuf__ uint8_t* rhsYAddr);

    template <RoundMode roundMode, const bool canInterleave, const bool withRHS>
    __aicore__ inline void ProcessBF16ToFP4Quantization(
        ComputeRegisters& lhsRegs, ComputeRegisters& rhsRegs, AuxRegisters& auxRegs, __ubuf__ uint8_t* lhsYAddr,
        __ubuf__ uint8_t* rhsYAddr);

    template <RoundMode roundMode, const bool canInterleave>
    __aicore__ inline void ProcessFP4QuantizationFromFP32(
        ComputeRegisters& regs, AuxRegisters& auxRegs, __ubuf__ uint8_t* yAddr);

    template <RoundMode roundMode, const bool canInterleave>
    __aicore__ inline void ProcessFP8QuantizationFromFP32(
        ComputeRegisters& regs, AuxRegisters& auxRegs, __ubuf__ uint8_t* yAddr);
    template <RoundMode roundMode>
    __aicore__ inline void PreProcessFP32(AscendC::MicroAPI::RegTensor<float>& in, AuxRegisters& auxRegs);

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
    bool scaleAlg_{false};
    int64_t preAxisSize_{0}, quantAxisSize_{0};
    uint32_t alignedPostAxisSize_{0}, alignedOutputPostAxisSize_{0}, postAxisSize_{0}, outputPostAxisSize_{0};
    uint16_t blockSize_{0}, tailBlockSize_{0};
    int64_t blockNumInAxis_{0}, padBlockNumInAxis_{0}, totalBlockNum_{0}, blockNumPerTask_{0};
    int64_t totalTaskNum_{0}, avgTaskNum_{0}, tailTaskNum_{0}, blockNumLastTask_{0};
    int64_t taskStartIdx_{0}, taskEndIdx_{0};
    int64_t roundMode_{0};
    DataCopyPadExtParams<DTYPE_X> padParams_{false, 0, 0, 0};
    int64_t nextInRowOffset_, nextOutRowOffset_;
    uint32_t inStride_, outStride_, scaleStride_;
    uint16_t DTYPE_Y_MAX_EXP{0};
    uint32_t INV_DTYPE_MAX{0};
};

__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeHighPerf::Init(
    TPipe* pipe, GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, const DynamicMxQuant4OptimizeTilingData* tilingData)
{
#if (__NPU_ARCH__ == 3101)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
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

__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeHighPerf::ParseTilingData(
    const DynamicMxQuant4OptimizeTilingData* tilingData)
{
    preAxisSize_ = tilingData->preAxisSize;
    quantAxisSize_ = tilingData->quantAxisSize;
    postAxisSize_ = tilingData->postAxisSize;
    blockSize_ = tilingData->blockSize;
    usedCoreNum_ = tilingData->usedCoreNum;
    needPadAxis_ = tilingData->isPad;
    tailBlockSize_ = tilingData->tailBlockSize;
    blockNumInAxis_ = tilingData->mAlignBlockSize;
    alignedPostAxisSize_ = tilingData->nAlignSize;
    roundMode_ = tilingData->roundMode;
    needPadBlock_ = tilingData->quantAxisIsOdd;
    scaleAlg_ =
        (IsSame<DTYPE_Y, fp4x2_e2m1_t>::value || IsSame<DTYPE_Y, fp4x2_e1m2_t>::value) ? false : tilingData->scaleAlg;
    canInterleave_ = alignedPostAxisSize_ * DIGIT_TWO * sizeof(DTYPE_X) <= Ops::Base::GetVRegSize();
    padBlockNumInAxis_ = tilingData->mAlignGroupSize * DIGIT_TWO;
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
    uint32_t quantedVFBlock = DIGIT_FOUR; // for fp8, 256B/2/32B = 4
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
    if constexpr (IsSame<DTYPE_Y, fp8_e4m3fn_t>::value) {
        DTYPE_Y_MAX_EXP = FP8_E4M3_MAX_EXP;
        INV_DTYPE_MAX = FP8_E4M3_MAX;
    } else if constexpr (IsSame<DTYPE_Y, fp8_e5m2_t>::value) {
        DTYPE_Y_MAX_EXP = FP8_E5M2_MAX_EXP;
        INV_DTYPE_MAX = FP8_E5M2_MAX;
    } else if constexpr (IsSame<DTYPE_Y, fp4x2_e2m1_t>::value) {
        DTYPE_Y_MAX_EXP = FP4_E2M1_BF16_MAX_EXP;
    }
}

__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeHighPerf::Process()
{
    if (blkIdx_ >= usedCoreNum_) {
        return;
    }
    if (scaleAlg_) {
        if (needPadBlock_) {
            if (needPadAxis_) {
                if (canInterleave_) {
                    ProcessWithPad<true, true, true, true>();
                } else {
                    ProcessWithPad<true, true, false, true>();
                }
            } else {
                if (canInterleave_) {
                    ProcessWithPad<true, false, true, true>();
                } else {
                    ProcessWithPad<true, false, false, true>();
                }
            }
        } else {
            if (needPadAxis_) {
                if (canInterleave_) {
                    ProcessWithPad<false, true, true, true>();
                } else {
                    ProcessWithPad<false, true, false, true>();
                }
            } else {
                if (canInterleave_) {
                    ProcessWithPad<false, false, true, true>();
                } else {
                    ProcessWithPad<false, false, false, true>();
                }
            }
        }
    } else {
        if (needPadBlock_) {
            if (needPadAxis_) {
                if (canInterleave_) {
                    ProcessWithPad<true, true, true, false>();
                } else {
                    ProcessWithPad<true, true, false, false>();
                }
            } else {
                if (canInterleave_) {
                    ProcessWithPad<true, false, true, false>();
                } else {
                    ProcessWithPad<true, false, false, false>();
                }
            }
        } else {
            if (needPadAxis_) {
                if (canInterleave_) {
                    ProcessWithPad<false, true, true, false>();
                } else {
                    ProcessWithPad<false, true, false, false>();
                }
            } else {
                if (canInterleave_) {
                    ProcessWithPad<false, false, true, false>();
                } else {
                    ProcessWithPad<false, false, false, false>();
                }
            }
        }
    }
}

template <const bool needPadBlock, const bool needPadAxis, const bool canInterleave, const bool scaleAlg>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeHighPerf::ProcessWithPad()
{
    int64_t blockIdx = taskStartIdx_ * blockNumPerTask_;
    int64_t blockCount = taskStartIdx_ == totalTaskNum_ - 1 ? blockNumLastTask_ : blockNumPerTask_;
    CopyIn(blockIdx, blockCount);

    for (int64_t taskIdx = taskStartIdx_ + 1; taskIdx < taskEndIdx_; taskIdx++) {
        int64_t nextBlockIdx = taskIdx * blockNumPerTask_;
        int64_t nextBlockCount = taskIdx == totalTaskNum_ - 1 ? blockNumLastTask_ : blockNumPerTask_;
        CopyIn(nextBlockIdx, nextBlockCount);
        SplitPreAxisCompute<needPadBlock, needPadAxis, canInterleave, scaleAlg>(blockIdx, blockCount);
        CopyOut(blockIdx, blockCount);
        blockIdx = nextBlockIdx;
        blockCount = nextBlockCount;
    }
    SplitPreAxisCompute<needPadBlock, needPadAxis, canInterleave, scaleAlg>(blockIdx, blockCount);
    CopyOut(blockIdx, blockCount);
}

__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeHighPerf::CopyIn(int64_t offset, int64_t count)
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

__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeHighPerf::CopyOut(int64_t offset, int64_t count)
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

template <const bool needPadBlock, const bool needPadAxis, const bool canInterleave, const bool scaleAlg>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeHighPerf::SplitPreAxisCompute(int64_t offset, int64_t count)
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
        Compute<canInterleave, scaleAlg>(
            xAddr, xAddr + (loop0 + loop1) * alignedPostAxisSize_, mxScaleAddr, yAddr,
            yAddr + (loop0 + loop1) * alignedOutputPostAxisSize_, loop0, loop1);
        xAddr += (2 * loop0 + loop1) * alignedPostAxisSize_;
        yAddr += (2 * loop0 + loop1) * alignedOutputPostAxisSize_;
        mxScaleAddr += 2 * alignedPostAxisSize_;
    }

    inQueue_.FreeTensor(x);
    mxScaleQueue_.EnQue(mxScale);
    outQueue_.EnQue(y);
}

template <const bool canInterleave, const bool scaleAlg>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeHighPerf::Compute(
    __ubuf__ DTYPE_X* lhsXAddr, __ubuf__ DTYPE_X* rhsXAddr, __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint8_t* lhsYAddr,
    __ubuf__ uint8_t* rhsYAddr, uint16_t loop0, uint16_t loop1)
{
    if constexpr (IsSame<DTYPE_Y, fp8_e4m3fn_t>::value || IsSame<DTYPE_Y, fp8_e5m2_t>::value) {
        ComputeWithRound<RoundMode::CAST_RINT, canInterleave, scaleAlg>(
            lhsXAddr, rhsXAddr, mxScaleAddr, lhsYAddr, rhsYAddr, loop0, loop1);
    } else if (roundMode_ == MODE_RINT) {
        ComputeWithRound<RoundMode::CAST_RINT, canInterleave, scaleAlg>(
            lhsXAddr, rhsXAddr, mxScaleAddr, lhsYAddr, rhsYAddr, loop0, loop1);
    } else if (roundMode_ == MODE_FLOOR) {
        ComputeWithRound<RoundMode::CAST_FLOOR, canInterleave, scaleAlg>(
            lhsXAddr, rhsXAddr, mxScaleAddr, lhsYAddr, rhsYAddr, loop0, loop1);
    } else if (roundMode_ == MODE_ROUND) {
        ComputeWithRound<RoundMode::CAST_ROUND, canInterleave, scaleAlg>(
            lhsXAddr, rhsXAddr, mxScaleAddr, lhsYAddr, rhsYAddr, loop0, loop1);
    }
}

template <RoundMode roundMode, const bool canInterleave, const bool scaleAlg>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeHighPerf::ComputeWithRound(
    __ubuf__ DTYPE_X* lhsXAddr, __ubuf__ DTYPE_X* rhsXAddr, __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint8_t* lhsYAddr,
    __ubuf__ uint8_t* rhsYAddr, uint16_t loop0, uint16_t loop1)
{
    __VEC_SCOPE__
    {
        ComputeRegisters lhsRegs;
        ComputeRegisters rhsRegs;
        AuxRegisters auxRegs;

        // Initialize constants and register tensors
        InitializeConstants<roundMode, canInterleave>(auxRegs);

        // Compute maximum exponents from input data
        ComputeMaxExponents<canInterleave, scaleAlg>(lhsRegs, rhsRegs, auxRegs, lhsXAddr, rhsXAddr, loop0, loop1);

        // Compute MX scales and shared exponents
        ComputeScalesAndSharedExp<roundMode, canInterleave, scaleAlg>(lhsRegs, auxRegs, mxScaleAddr);
        AscendC::MicroAPI::Pack<uint8_t, uint16_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
            (AscendC::MicroAPI::RegTensor<uint8_t>&)lhsRegs.mxScaleRegTensor, lhsRegs.mxScaleRegTensor);

        if constexpr (!canInterleave) {
            ComputeScalesAndSharedExp<roundMode, canInterleave, scaleAlg>(rhsRegs, auxRegs, mxScaleAddr);
            AscendC::MicroAPI::Pack<uint8_t, uint16_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(
                (AscendC::MicroAPI::RegTensor<uint8_t>&)rhsRegs.mxScaleRegTensor, rhsRegs.mxScaleRegTensor);
            AscendC::MicroAPI::Or(
                lhsRegs.mxScaleRegTensor, lhsRegs.mxScaleRegTensor, rhsRegs.mxScaleRegTensor, auxRegs.p0);
        }

        AscendC::MicroAPI::DataCopy(
            mxScaleAddr, (AscendC::MicroAPI::RegTensor<uint8_t>&)lhsRegs.mxScaleRegTensor, auxRegs.p2);

        if constexpr (
            IsSame<DTYPE_X, half>::value || IsSame<DTYPE_Y, fp8_e4m3fn_t>::value ||
            IsSame<DTYPE_Y, fp8_e5m2_t>::value) {
            static constexpr AscendC::MicroAPI::CastTrait castTraitZero = {
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
            static constexpr AscendC::MicroAPI::CastTrait castTraitOne = {
                AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

            AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitZero>(
                lhsRegs.expFP32RegTensor0, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)lhsRegs.expMaxRegTensor,
                auxRegs.p0);
            AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitOne>(
                rhsRegs.expFP32RegTensor0, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)lhsRegs.expMaxRegTensor,
                auxRegs.p0);
            if constexpr (!canInterleave) {
                AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitZero>(
                    lhsRegs.expFP32RegTensor1, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)rhsRegs.expMaxRegTensor,
                    auxRegs.p0);
                AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitOne>(
                    rhsRegs.expFP32RegTensor1, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)rhsRegs.expMaxRegTensor,
                    auxRegs.p0);
            }
        }

        // Quantize values to output format
        QuantizeValues<roundMode, canInterleave>(
            lhsRegs, rhsRegs, auxRegs, lhsXAddr, rhsXAddr, lhsYAddr, rhsYAddr, loop0, loop1);
    }
}

template <RoundMode roundMode, const bool canInterleave>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeHighPerf::InitializeConstants(AuxRegisters& regs)
{
    // Initialize mask registers
    regs.p0 = AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
    regs.p1 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

    uint32_t scaleMask = alignedPostAxisSize_ * DIGIT_TWO;
    regs.p2 = AscendC::MicroAPI::UpdateMask<uint8_t>(scaleMask);

    uint32_t outputMask = alignedOutputPostAxisSize_ * DIGIT_TWO;
    if constexpr (IsSame<DTYPE_Y, fp4x2_e2m1_t>::value || IsSame<DTYPE_Y, fp4x2_e1m2_t>::value) {
        outputMask = alignedOutputPostAxisSize_ * DIGIT_FOUR;
    }
    regs.p3 = AscendC::MicroAPI::UpdateMask<uint8_t>(outputMask);

    // Initialize constant register tensors
    AscendC::MicroAPI::Duplicate(regs.maxEleRegTensor, MAX_EXP_FOR_BF16);
    AscendC::MicroAPI::Duplicate(regs.maxEleRegTensorHalf, HALF_INF);
    AscendC::MicroAPI::Duplicate(regs.fp8NanRegTensor, MAX_EXP_FOR_FP8);
    AscendC::MicroAPI::Duplicate(regs.biasRegTensor, BF16_EXP_BIAS);
    AscendC::MicroAPI::Duplicate(regs.zeroRegTensor, 0);
    AscendC::MicroAPI::Duplicate(regs.nanRegTensor, NAN_CUSTOMIZATION);
    AscendC::MicroAPI::Duplicate(regs.specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
    AscendC::MicroAPI::Duplicate(regs.bf16NegInfRegTensor, BF16_NEG_INF);
    AscendC::MicroAPI::Duplicate(regs.tgtMaxExpRegTensor, DTYPE_Y_MAX_EXP);

    AscendC::MicroAPI::Duplicate(regs.invMax, INV_DTYPE_MAX);
    AscendC::MicroAPI::Duplicate(regs.absForX, ABS_FOR_UINT16);
    AscendC::MicroAPI::Duplicate(regs.manForFP32, MAN_FOR_FP32);

    if constexpr (IsSame<DTYPE_X, half>::value) {
        if constexpr (IsSame<DTYPE_Y, fp4x2_e2m1_t>::value) {
            AscendC::MicroAPI::Duplicate(regs.maxExpFP32RegTensor, MAX_EXP_FOR_FP32);
            AscendC::MicroAPI::Duplicate(regs.negZeroRegTensor, NEG_ZERO);
        }
        if constexpr (IsSame<DTYPE_Y, fp4x2_e1m2_t>::value) {
            AscendC::MicroAPI::Duplicate(regs.negZeroRegTensor, NEG_ZERO);
        }
    }
}

template <const uint8_t initRHS>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeHighPerf::LoadData(
    ComputeRegisters& lhsRegs, ComputeRegisters& rhsRegs, AuxRegisters& auxRegs, __ubuf__ DTYPE_X* lhsXAddr,
    __ubuf__ DTYPE_X* rhsXAddr)
{
    DataCopy(lhsRegs.xRegTensor, lhsXAddr);
    if constexpr (initRHS == 1) {
        DataCopy(rhsRegs.xRegTensor, rhsXAddr);
    } else if constexpr (initRHS == 2) {
        AscendC::MicroAPI::Duplicate(rhsRegs.xRegTensor, 0);
    }
    AscendC::MicroAPI::Interleave(lhsRegs.xRegTensor, rhsRegs.xRegTensor, lhsRegs.xRegTensor, rhsRegs.xRegTensor);
}

template <const bool canInterleave, const bool scaleAlg>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeHighPerf::ComputeMaxExponents(
    ComputeRegisters& lhsRegs, ComputeRegisters& rhsRegs, AuxRegisters& auxRegs, __ubuf__ DTYPE_X* lhsXAddr,
    __ubuf__ DTYPE_X* rhsXAddr, uint16_t loop0, uint16_t loop1)
{
    static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
        AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
    AscendC::MicroAPI::Duplicate(lhsRegs.expMaxRegTensor, 0);
    if constexpr (!canInterleave) {
        AscendC::MicroAPI::Duplicate(rhsRegs.expMaxRegTensor, 0);
    }

    uint16_t xOffset0 = 0;

    // Process main loops
    for (uint16_t i = 0; i < loop0; ++i) {
        LoadData<1>(
            lhsRegs, rhsRegs, auxRegs, lhsXAddr + xOffset0 * alignedPostAxisSize_,
            rhsXAddr + xOffset0 * alignedPostAxisSize_);

        if constexpr (scaleAlg) {
            AscendC::MicroAPI::And(
                lhsRegs.expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)lhsRegs.xRegTensor, auxRegs.absForX,
                auxRegs.p0); // |x|
            AscendC::MicroAPI::Max(
                lhsRegs.expMaxRegTensor, lhsRegs.expMaxRegTensor, lhsRegs.expRegTensor, auxRegs.p0); // max |x|
        } else {
            if constexpr (IsSame<DTYPE_X, half>::value) {
                AscendC::MicroAPI::And(
                    lhsRegs.expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)lhsRegs.xRegTensor,
                    auxRegs.maxEleRegTensorHalf, auxRegs.p0);
                AscendC::MicroAPI::CompareScalar<uint16_t, CMPMODE::EQ>(
                    auxRegs.infMask, (AscendC::MicroAPI::RegTensor<uint16_t>&)lhsRegs.expRegTensor, HALF_INF,
                    auxRegs.p0);
                AscendC::MicroAPI::Cast<bfloat16_t, DTYPE_X, castTraitHalf2Bf16>(
                    (AscendC::MicroAPI::RegTensor<bfloat16_t>&)lhsRegs.expRegTensor,
                    (AscendC::MicroAPI::RegTensor<DTYPE_X>&)lhsRegs.xRegTensor, auxRegs.p0);
                AscendC::MicroAPI::Select<uint16_t>(
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)lhsRegs.expRegTensor, auxRegs.maxEleRegTensor,
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)lhsRegs.expRegTensor, auxRegs.infMask);
                AscendC::MicroAPI::And(
                    lhsRegs.expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)lhsRegs.expRegTensor,
                    auxRegs.maxEleRegTensor, auxRegs.p0);
            } else {
                AscendC::MicroAPI::And(
                    lhsRegs.expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)lhsRegs.xRegTensor,
                    auxRegs.maxEleRegTensor, auxRegs.p0);
            }
            AscendC::MicroAPI::Max(lhsRegs.expMaxRegTensor, lhsRegs.expMaxRegTensor, lhsRegs.expRegTensor, auxRegs.p0);
        }
        if constexpr (!canInterleave) {
            if constexpr (scaleAlg) {
                AscendC::MicroAPI::And(
                    rhsRegs.expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)rhsRegs.xRegTensor, auxRegs.absForX,
                    auxRegs.p0); // |x|
                AscendC::MicroAPI::Max(
                    rhsRegs.expMaxRegTensor, rhsRegs.expMaxRegTensor, rhsRegs.expRegTensor, auxRegs.p0); // max |x|
            } else {
                if constexpr (IsSame<DTYPE_X, half>::value) {
                    AscendC::MicroAPI::And(
                        rhsRegs.expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)rhsRegs.xRegTensor,
                        auxRegs.maxEleRegTensorHalf, auxRegs.p0);
                    AscendC::MicroAPI::CompareScalar<uint16_t, CMPMODE::EQ>(
                        auxRegs.infMask, (AscendC::MicroAPI::RegTensor<uint16_t>&)rhsRegs.expRegTensor, HALF_INF,
                        auxRegs.p0);
                    AscendC::MicroAPI::Cast<bfloat16_t, DTYPE_X, castTraitHalf2Bf16>(
                        (AscendC::MicroAPI::RegTensor<bfloat16_t>&)rhsRegs.expRegTensor,
                        (AscendC::MicroAPI::RegTensor<DTYPE_X>&)rhsRegs.xRegTensor, auxRegs.p0);
                    AscendC::MicroAPI::Select<uint16_t>(
                        (AscendC::MicroAPI::RegTensor<uint16_t>&)rhsRegs.expRegTensor, auxRegs.maxEleRegTensor,
                        (AscendC::MicroAPI::RegTensor<uint16_t>&)rhsRegs.expRegTensor, auxRegs.infMask);
                    AscendC::MicroAPI::And(
                        rhsRegs.expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)rhsRegs.expRegTensor,
                        auxRegs.maxEleRegTensor, auxRegs.p0);
                } else {
                    AscendC::MicroAPI::And(
                        rhsRegs.expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)rhsRegs.xRegTensor,
                        auxRegs.maxEleRegTensor, auxRegs.p0);
                }
                AscendC::MicroAPI::Max(
                    rhsRegs.expMaxRegTensor, rhsRegs.expMaxRegTensor, rhsRegs.expRegTensor, auxRegs.p0);
            }
        }
        xOffset0++;
    }

    // Process tail loops
    for (uint16_t i = 0; i < loop1; ++i) {
        LoadData<2>(lhsRegs, rhsRegs, auxRegs, lhsXAddr + xOffset0 * alignedPostAxisSize_, nullptr);

        if constexpr (scaleAlg) {
            AscendC::MicroAPI::And(
                lhsRegs.expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)lhsRegs.xRegTensor, auxRegs.absForX,
                auxRegs.p0); // |x|
            AscendC::MicroAPI::Max(
                lhsRegs.expMaxRegTensor, lhsRegs.expMaxRegTensor, lhsRegs.expRegTensor, auxRegs.p0); // max |x|
        } else {
            if constexpr (IsSame<DTYPE_X, half>::value) {
                AscendC::MicroAPI::And(
                    lhsRegs.expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)lhsRegs.xRegTensor,
                    auxRegs.maxEleRegTensorHalf, auxRegs.p0);
                AscendC::MicroAPI::CompareScalar<uint16_t, CMPMODE::EQ>(
                    auxRegs.infMask, (AscendC::MicroAPI::RegTensor<uint16_t>&)lhsRegs.expRegTensor, HALF_INF,
                    auxRegs.p0);
                AscendC::MicroAPI::Cast<bfloat16_t, DTYPE_X, castTraitHalf2Bf16>(
                    (AscendC::MicroAPI::RegTensor<bfloat16_t>&)lhsRegs.expRegTensor,
                    (AscendC::MicroAPI::RegTensor<DTYPE_X>&)lhsRegs.xRegTensor, auxRegs.p0);
                AscendC::MicroAPI::Select<uint16_t>(
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)lhsRegs.expRegTensor, auxRegs.maxEleRegTensor,
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)lhsRegs.expRegTensor, auxRegs.infMask);
                AscendC::MicroAPI::And(
                    lhsRegs.expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)lhsRegs.expRegTensor,
                    auxRegs.maxEleRegTensor, auxRegs.p0);
            } else {
                AscendC::MicroAPI::And(
                    lhsRegs.expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)lhsRegs.xRegTensor,
                    auxRegs.maxEleRegTensor, auxRegs.p0);
            }
            AscendC::MicroAPI::Max(lhsRegs.expMaxRegTensor, lhsRegs.expMaxRegTensor, lhsRegs.expRegTensor, auxRegs.p0);
        }
        if constexpr (!canInterleave) {
            if constexpr (scaleAlg) {
                AscendC::MicroAPI::And(
                    rhsRegs.expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)rhsRegs.xRegTensor, auxRegs.absForX,
                    auxRegs.p0); // |x|
                AscendC::MicroAPI::Max(
                    rhsRegs.expMaxRegTensor, rhsRegs.expMaxRegTensor, rhsRegs.expRegTensor, auxRegs.p0); // max |x|
            } else {
                if constexpr (IsSame<DTYPE_X, half>::value) {
                    AscendC::MicroAPI::And(
                        rhsRegs.expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)rhsRegs.xRegTensor,
                        auxRegs.maxEleRegTensorHalf, auxRegs.p0);
                    AscendC::MicroAPI::CompareScalar<uint16_t, CMPMODE::EQ>(
                        auxRegs.infMask, (AscendC::MicroAPI::RegTensor<uint16_t>&)rhsRegs.expRegTensor, HALF_INF,
                        auxRegs.p0);
                    AscendC::MicroAPI::Cast<bfloat16_t, DTYPE_X, castTraitHalf2Bf16>(
                        (AscendC::MicroAPI::RegTensor<bfloat16_t>&)rhsRegs.expRegTensor,
                        (AscendC::MicroAPI::RegTensor<DTYPE_X>&)rhsRegs.xRegTensor, auxRegs.p0);
                    AscendC::MicroAPI::Select<uint16_t>(
                        (AscendC::MicroAPI::RegTensor<uint16_t>&)rhsRegs.expRegTensor, auxRegs.maxEleRegTensor,
                        (AscendC::MicroAPI::RegTensor<uint16_t>&)rhsRegs.expRegTensor, auxRegs.infMask);
                    AscendC::MicroAPI::And(
                        rhsRegs.expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)rhsRegs.expRegTensor,
                        auxRegs.maxEleRegTensor, auxRegs.p0);
                } else {
                    AscendC::MicroAPI::And(
                        rhsRegs.expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)rhsRegs.xRegTensor,
                        auxRegs.maxEleRegTensor, auxRegs.p0);
                }
                AscendC::MicroAPI::Max(
                    rhsRegs.expMaxRegTensor, rhsRegs.expMaxRegTensor, rhsRegs.expRegTensor, auxRegs.p0);
            }
        }
        xOffset0++;
    }
}

template <RoundMode roundMode, const bool canInterleave, const bool scaleAlg>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeHighPerf::ComputeScalesAndSharedExp(
    ComputeRegisters& Regs, AuxRegisters& auxRegs, __ubuf__ uint8_t* mxScaleAddr)
{
    if constexpr (scaleAlg) {
        static constexpr AscendC::MicroAPI::CastTrait castTraitZero = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTraitOne = {
            AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        // Calculate elements at odd positions
        AscendC::MicroAPI::Compare<uint16_t, CMPMODE::EQ>(
            auxRegs.zeroMask, Regs.expMaxRegTensor, auxRegs.zeroRegTensor, auxRegs.p0);
        AscendC::MicroAPI::Cast<float, DTYPE_X, castTraitZero>(
            Regs.maxFP32RegTensor0, (AscendC::MicroAPI::RegTensor<DTYPE_X>&)Regs.expMaxRegTensor, auxRegs.p0);
        AscendC::MicroAPI::Mul(
            Regs.maxFP32RegTensor0, Regs.maxFP32RegTensor0, (AscendC::MicroAPI::RegTensor<float>&)auxRegs.invMax,
            auxRegs.p1);
        AscendC::MicroAPI::ShiftRights(
            Regs.mxScaleFP32RegTensor0, (AscendC::MicroAPI::RegTensor<uint32_t>&)Regs.maxFP32RegTensor0,
            SHR_NUM_FOR_FP32, auxRegs.p1);
        AscendC::MicroAPI::And(
            (AscendC::MicroAPI::RegTensor<uint32_t>&)Regs.maxFP32RegTensor0,
            (AscendC::MicroAPI::RegTensor<uint32_t>&)Regs.maxFP32RegTensor0,
            (AscendC::MicroAPI::RegTensor<uint32_t>&)auxRegs.manForFP32, auxRegs.p1); // 提取尾数

        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(
            auxRegs.p4, (AscendC::MicroAPI::RegTensor<uint32_t>&)Regs.mxScaleFP32RegTensor0, NUMBER_ZERO, auxRegs.p1);
        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(
            auxRegs.p5, (AscendC::MicroAPI::RegTensor<uint32_t>&)Regs.mxScaleFP32RegTensor0, NUMBER_TWO_FIVE_FOUR,
            auxRegs.p1);
        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(
            auxRegs.p6, (AscendC::MicroAPI::RegTensor<uint32_t>&)Regs.maxFP32RegTensor0, NUMBER_ZERO, auxRegs.p1);
        AscendC::MicroAPI::MaskAnd(auxRegs.p4, auxRegs.p4, auxRegs.p5, auxRegs.p1);
        AscendC::MicroAPI::MaskAnd(auxRegs.p4, auxRegs.p4, auxRegs.p6, auxRegs.p1);

        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(
            auxRegs.p5, (AscendC::MicroAPI::RegTensor<uint32_t>&)Regs.mxScaleFP32RegTensor0, NUMBER_ZERO, auxRegs.p1);
        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(
            auxRegs.p6, (AscendC::MicroAPI::RegTensor<uint32_t>&)Regs.maxFP32RegTensor0, NUMBER_HALF, auxRegs.p1);
        AscendC::MicroAPI::MaskAnd(auxRegs.p5, auxRegs.p5, auxRegs.p6, auxRegs.p1);
        AscendC::MicroAPI::MaskXor(auxRegs.p4, auxRegs.p4, auxRegs.p5, auxRegs.p1);

        AscendC::MicroAPI::Adds(Regs.mxScaleFP32Add1RegTensor0, Regs.mxScaleFP32RegTensor0, 1, auxRegs.p4);
        AscendC::MicroAPI::Select(
            Regs.mxScaleFP32RegTensor0, Regs.mxScaleFP32Add1RegTensor0, Regs.mxScaleFP32RegTensor0, auxRegs.p4);

        AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
            (AscendC::MicroAPI::RegTensor<uint16_t>&)Regs.mxScaleRegTensor, Regs.mxScaleFP32RegTensor0);

        // Calculate elements at even positions
        AscendC::MicroAPI::Cast<float, DTYPE_X, castTraitOne>(
            Regs.maxFP32RegTensor1, (AscendC::MicroAPI::RegTensor<DTYPE_X>&)Regs.expMaxRegTensor, auxRegs.p0);
        AscendC::MicroAPI::Mul(
            Regs.maxFP32RegTensor1, Regs.maxFP32RegTensor1, (AscendC::MicroAPI::RegTensor<float>&)auxRegs.invMax,
            auxRegs.p1);
        AscendC::MicroAPI::ShiftRights(
            Regs.mxScaleFP32RegTensor1, (AscendC::MicroAPI::RegTensor<uint32_t>&)Regs.maxFP32RegTensor1,
            SHR_NUM_FOR_FP32, auxRegs.p1);
        AscendC::MicroAPI::And(
            (AscendC::MicroAPI::RegTensor<uint32_t>&)Regs.maxFP32RegTensor1,
            (AscendC::MicroAPI::RegTensor<uint32_t>&)Regs.maxFP32RegTensor1,
            (AscendC::MicroAPI::RegTensor<uint32_t>&)auxRegs.manForFP32, auxRegs.p1); // 提取尾数

        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(
            auxRegs.p4, (AscendC::MicroAPI::RegTensor<uint32_t>&)Regs.mxScaleFP32RegTensor1, NUMBER_ZERO, auxRegs.p1);
        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(
            auxRegs.p5, (AscendC::MicroAPI::RegTensor<uint32_t>&)Regs.mxScaleFP32RegTensor1, NUMBER_TWO_FIVE_FOUR,
            auxRegs.p1);
        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(
            auxRegs.p6, (AscendC::MicroAPI::RegTensor<uint32_t>&)Regs.maxFP32RegTensor1, NUMBER_ZERO, auxRegs.p1);
        AscendC::MicroAPI::MaskAnd(auxRegs.p4, auxRegs.p4, auxRegs.p5, auxRegs.p1);
        AscendC::MicroAPI::MaskAnd(auxRegs.p4, auxRegs.p4, auxRegs.p6, auxRegs.p1);

        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(
            auxRegs.p5, (AscendC::MicroAPI::RegTensor<uint32_t>&)Regs.mxScaleFP32RegTensor1, NUMBER_ZERO, auxRegs.p1);
        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(
            auxRegs.p6, (AscendC::MicroAPI::RegTensor<uint32_t>&)Regs.maxFP32RegTensor1, NUMBER_HALF, auxRegs.p1);
        AscendC::MicroAPI::MaskAnd(auxRegs.p5, auxRegs.p5, auxRegs.p6, auxRegs.p1);
        AscendC::MicroAPI::MaskXor(auxRegs.p4, auxRegs.p4, auxRegs.p5, auxRegs.p1);

        AscendC::MicroAPI::Adds(Regs.mxScaleFP32Add1RegTensor1, Regs.mxScaleFP32RegTensor1, 1, auxRegs.p4);
        AscendC::MicroAPI::Select(
            Regs.mxScaleFP32RegTensor1, Regs.mxScaleFP32Add1RegTensor1, Regs.mxScaleFP32RegTensor1, auxRegs.p4);

        AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
            (AscendC::MicroAPI::RegTensor<uint16_t>&)Regs.mxScaleRegTensor1, Regs.mxScaleFP32RegTensor1);

        AscendC::MicroAPI::Interleave(
            Regs.mxScaleRegTensor, Regs.mxScaleRegTensor1, Regs.mxScaleRegTensor, Regs.mxScaleRegTensor1);

        AscendC::MicroAPI::ShiftLefts(Regs.expMaxRegTensor, Regs.mxScaleRegTensor, SHR_NUM_FOR_BF16, auxRegs.p0);

        AscendC::MicroAPI::Compare<uint16_t, CMPMODE::EQ>(
            auxRegs.infMask, Regs.expMaxRegTensor, auxRegs.maxEleRegTensor, auxRegs.p0);
        
    } else {
        AscendC::MicroAPI::Compare<uint16_t, CMPMODE::EQ>(
            auxRegs.infMask, Regs.expMaxRegTensor, auxRegs.maxEleRegTensor, auxRegs.p0);
        AscendC::MicroAPI::Compare<uint16_t, CMPMODE::EQ>(
            auxRegs.zeroMask, Regs.expMaxRegTensor, auxRegs.zeroRegTensor, auxRegs.p0);

        AscendC::MicroAPI::Max(Regs.expMaxRegTensor, Regs.expMaxRegTensor, auxRegs.tgtMaxExpRegTensor, auxRegs.p0);
        AscendC::MicroAPI::Sub(Regs.expMaxRegTensor, Regs.expMaxRegTensor, auxRegs.tgtMaxExpRegTensor, auxRegs.p0);
        AscendC::MicroAPI::ShiftRights(Regs.mxScaleRegTensor, Regs.expMaxRegTensor, SHR_NUM_FOR_BF16, auxRegs.p0);

        AscendC::MicroAPI::Select<uint16_t>(
            Regs.mxScaleRegTensor, auxRegs.fp8NanRegTensor, Regs.mxScaleRegTensor, auxRegs.infMask);
        AscendC::MicroAPI::Select<uint16_t>(
            Regs.mxScaleRegTensor, auxRegs.zeroRegTensor, Regs.mxScaleRegTensor, auxRegs.zeroMask);
    }

    // Calculate 1/scale
    AscendC::MicroAPI::Compare<uint16_t, CMPMODE::EQ>(
        auxRegs.specialMask, Regs.expMaxRegTensor, auxRegs.biasRegTensor, auxRegs.p0);
    AscendC::MicroAPI::Sub(Regs.expMaxRegTensor, auxRegs.biasRegTensor, Regs.expMaxRegTensor, auxRegs.p0);
    AscendC::MicroAPI::Select<uint16_t>(
        Regs.expMaxRegTensor, auxRegs.nanRegTensor, Regs.expMaxRegTensor, auxRegs.infMask);
    AscendC::MicroAPI::Select<uint16_t>(
        Regs.expMaxRegTensor, auxRegs.zeroRegTensor, Regs.expMaxRegTensor, auxRegs.zeroMask);
    AscendC::MicroAPI::Select<uint16_t>(
        Regs.expMaxRegTensor, auxRegs.specialExpRegTensor, Regs.expMaxRegTensor, auxRegs.specialMask);
}

template <RoundMode roundMode, const bool canInterleave>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeHighPerf::QuantizeValues(
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
            ProcessFP32Quantization<roundMode, canInterleave, true>(
                lhsRegs, rhsRegs, auxRegs, lhsYAddr + xOffset1 * alignedOutputPostAxisSize_,
                rhsYAddr + xOffset1 * alignedOutputPostAxisSize_);
        } else {
            ProcessBF16ToFP4Quantization<roundMode, canInterleave, true>(
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
            ProcessFP32Quantization<roundMode, canInterleave, false>(
                lhsRegs, rhsRegs, auxRegs, lhsYAddr + xOffset1 * alignedOutputPostAxisSize_,
                rhsYAddr + xOffset1 * alignedOutputPostAxisSize_);
        } else {
            ProcessBF16ToFP4Quantization<roundMode, canInterleave, false>(
                lhsRegs, rhsRegs, auxRegs, lhsYAddr + xOffset1 * alignedOutputPostAxisSize_,
                rhsYAddr + xOffset1 * alignedOutputPostAxisSize_);
        }
        xOffset1++;
    }
}

template <RoundMode roundMode, const bool canInterleave, const bool withRHS>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeHighPerf::ProcessFP32Quantization(
    ComputeRegisters& lhsRegs, ComputeRegisters& rhsRegs, AuxRegisters& auxRegs, __ubuf__ uint8_t* lhsYAddr,
    __ubuf__ uint8_t* rhsYAddr)
{
    static constexpr AscendC::MicroAPI::CastTrait castTraitZero = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr AscendC::MicroAPI::CastTrait castTraitOne = {
        AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    AscendC::MicroAPI::Cast<float, DTYPE_X, castTraitZero>(lhsRegs.fp32RegTensor0, lhsRegs.xRegTensor, auxRegs.p0);
    AscendC::MicroAPI::Mul(lhsRegs.fp32RegTensor0, lhsRegs.fp32RegTensor0, lhsRegs.expFP32RegTensor0, auxRegs.p1);
    if constexpr (withRHS) {
        AscendC::MicroAPI::Cast<float, DTYPE_X, castTraitOne>(rhsRegs.fp32RegTensor0, lhsRegs.xRegTensor, auxRegs.p0);
        AscendC::MicroAPI::Mul(rhsRegs.fp32RegTensor0, rhsRegs.fp32RegTensor0, rhsRegs.expFP32RegTensor0, auxRegs.p1);
    }
    if constexpr (!canInterleave) {
        AscendC::MicroAPI::Cast<float, DTYPE_X, castTraitZero>(lhsRegs.fp32RegTensor1, rhsRegs.xRegTensor, auxRegs.p0);
        AscendC::MicroAPI::Mul(lhsRegs.fp32RegTensor1, lhsRegs.fp32RegTensor1, lhsRegs.expFP32RegTensor1, auxRegs.p1);
        if constexpr (withRHS) {
            AscendC::MicroAPI::Cast<float, DTYPE_X, castTraitOne>(
                rhsRegs.fp32RegTensor1, rhsRegs.xRegTensor, auxRegs.p0);
            AscendC::MicroAPI::Mul(
                rhsRegs.fp32RegTensor1, rhsRegs.fp32RegTensor1, rhsRegs.expFP32RegTensor1, auxRegs.p1);
        }
    }

    if constexpr (IsSame<DTYPE_Y, fp4x2_e2m1_t>::value || IsSame<DTYPE_Y, fp4x2_e1m2_t>::value) {
        // FP32 cannot directly cast to fp4x2_e2m1_t or fp4x2_e1m2_t, so cast to bfloat16 first
        ProcessFP4QuantizationFromFP32<roundMode, canInterleave>(lhsRegs, auxRegs, lhsYAddr);
        if constexpr (withRHS) {
            ProcessFP4QuantizationFromFP32<roundMode, canInterleave>(rhsRegs, auxRegs, rhsYAddr);
        }
    } else {
        // FP32 -> FP8
        ProcessFP8QuantizationFromFP32<roundMode, canInterleave>(lhsRegs, auxRegs, lhsYAddr);
        if constexpr (withRHS) {
            ProcessFP8QuantizationFromFP32<roundMode, canInterleave>(rhsRegs, auxRegs, rhsYAddr);
        }
    }
}

template <RoundMode roundMode, const bool canInterleave, const bool withRHS>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeHighPerf::ProcessBF16ToFP4Quantization(
    ComputeRegisters& lhsRegs, ComputeRegisters& rhsRegs, AuxRegisters& auxRegs, __ubuf__ uint8_t* lhsYAddr,
    __ubuf__ uint8_t* rhsYAddr)
{
    static constexpr AscendC::MicroAPI::CastTrait castTraitDownZero = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
        roundMode};

    // bfloat16 -> fp4x2_e2m1_t or fp4x2_e1m2_t
    AscendC::MicroAPI::Mul(
        lhsRegs.xRegTensor, lhsRegs.xRegTensor, (AscendC::MicroAPI::RegTensor<DTYPE_X>&)lhsRegs.expMaxRegTensor,
        auxRegs.p0);
    if constexpr (!canInterleave) {
        AscendC::MicroAPI::Mul(
            rhsRegs.xRegTensor, rhsRegs.xRegTensor, (AscendC::MicroAPI::RegTensor<DTYPE_X>&)rhsRegs.expMaxRegTensor,
            auxRegs.p0);
    }
    AscendC::MicroAPI::DeInterleave(lhsRegs.xRegTensor, rhsRegs.xRegTensor, lhsRegs.xRegTensor, rhsRegs.xRegTensor);
    AscendC::MicroAPI::Cast<DTYPE_Y, bfloat16_t, castTraitDownZero>(
        lhsRegs.yRegTensor, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)lhsRegs.xRegTensor, auxRegs.p0);
    DataCopy<uint8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
        lhsYAddr, (AscendC::MicroAPI::RegTensor<uint8_t>&)lhsRegs.yRegTensor, auxRegs.p3);
    if constexpr (withRHS) {
        AscendC::MicroAPI::Cast<DTYPE_Y, bfloat16_t, castTraitDownZero>(
            rhsRegs.yRegTensor, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)rhsRegs.xRegTensor, auxRegs.p0);
        DataCopy<uint8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
            rhsYAddr, (AscendC::MicroAPI::RegTensor<uint8_t>&)rhsRegs.yRegTensor, auxRegs.p3);
    }
}

// Helper function for FP4 quantization from FP32
template <RoundMode roundMode, const bool canInterleave>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeHighPerf::ProcessFP4QuantizationFromFP32(
    ComputeRegisters& regs, AuxRegisters& auxRegs, __ubuf__ uint8_t* yAddr)
{
    static constexpr AscendC::MicroAPI::CastTrait castTraitDownZero = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, roundMode};
    static constexpr AscendC::MicroAPI::CastTrait castTraitDownZeroBF16 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, roundMode};

    PreProcessFP32<roundMode>(regs.fp32RegTensor0, auxRegs);

    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitDownZeroBF16>(
        (AscendC::MicroAPI::RegTensor<bfloat16_t>&)regs.u16RegTensor, regs.fp32RegTensor0, auxRegs.p1);
    AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
        regs.u16RegTensor, (AscendC::MicroAPI::RegTensor<uint32_t>&)regs.u16RegTensor);

    if (!canInterleave) {
        PreProcessFP32<roundMode>(regs.fp32RegTensor1, auxRegs);

        AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitDownZeroBF16>(
            (AscendC::MicroAPI::RegTensor<bfloat16_t>&)regs.yRegTensor, regs.fp32RegTensor1, auxRegs.p1);
        AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(
            (AscendC::MicroAPI::RegTensor<uint16_t>&)regs.yRegTensor,
            (AscendC::MicroAPI::RegTensor<uint32_t>&)regs.yRegTensor);

        AscendC::MicroAPI::Or<uint16_t>(
            (AscendC::MicroAPI::RegTensor<uint16_t>&)regs.u16RegTensor,
            (AscendC::MicroAPI::RegTensor<uint16_t>&)regs.u16RegTensor,
            (AscendC::MicroAPI::RegTensor<uint16_t>&)regs.yRegTensor, auxRegs.p0);
    }

    AscendC::MicroAPI::Cast<DTYPE_Y, bfloat16_t, castTraitDownZero>(
        regs.yRegTensor, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)regs.u16RegTensor, auxRegs.p0);
    DataCopy<uint8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
        yAddr, (AscendC::MicroAPI::RegTensor<uint8_t>&)regs.yRegTensor, auxRegs.p3);
}

// Helper function for FP8 quantization from FP32
template <RoundMode roundMode, const bool canInterleave>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeHighPerf::ProcessFP8QuantizationFromFP32(
    ComputeRegisters& regs, AuxRegisters& auxRegs, __ubuf__ uint8_t* yAddr)
{
    static constexpr AscendC::MicroAPI::CastTrait castTraitDownZero = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
        roundMode};

    AscendC::MicroAPI::Cast<DTYPE_Y, float, castTraitDownZero>(regs.yRegTensor, regs.fp32RegTensor0, auxRegs.p1);
    AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
        (AscendC::MicroAPI::RegTensor<uint16_t>&)regs.yRegTensor,
        (AscendC::MicroAPI::RegTensor<uint32_t>&)regs.yRegTensor);

    if constexpr (!canInterleave) {
        AscendC::MicroAPI::Cast<DTYPE_Y, float, castTraitDownZero>(
            (AscendC::MicroAPI::RegTensor<DTYPE_Y>&)regs.u16RegTensor, regs.fp32RegTensor1, auxRegs.p1);
        AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(
            regs.u16RegTensor, (AscendC::MicroAPI::RegTensor<uint32_t>&)regs.u16RegTensor);

        AscendC::MicroAPI::Or<uint16_t>(
            (AscendC::MicroAPI::RegTensor<uint16_t>&)regs.yRegTensor, regs.u16RegTensor,
            (AscendC::MicroAPI::RegTensor<uint16_t>&)regs.yRegTensor, auxRegs.p0);
    }
    DataCopy<uint8_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(
        yAddr, (AscendC::MicroAPI::RegTensor<uint8_t>&)regs.yRegTensor, auxRegs.p3);
}

template <RoundMode roundMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeHighPerf::PreProcessFP32(
    AscendC::MicroAPI::RegTensor<float>& in, AuxRegisters& auxRegs)
{
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::EQ>(
        /*negzero*/ auxRegs.negInfMask, (AscendC::MicroAPI::RegTensor<int32_t>&)in, auxRegs.negZeroRegTensor,
        auxRegs.p1);
    if constexpr (IsSame<DTYPE_Y, fp4x2_e1m2_t>::value) {
        AscendC::MicroAPI::Muls(in, in, FOUR, auxRegs.p1);
        AscendC::MicroAPI::CompareScalar<float, CMPMODE::LT>(/*negvalue*/ auxRegs.specialMask, in, 0, auxRegs.p1);
        AscendC::MicroAPI::Truncate<float, roundMode>(in, in, auxRegs.p1);
        AscendC::MicroAPI::Muls(in, in, ONE_FOURTH, auxRegs.p1);
    } else { // fp4x2_e2m1
        AscendC::MicroAPI::And(
            auxRegs.expFP32RegTensor0, (AscendC::MicroAPI::RegTensor<int32_t>&)in, auxRegs.maxExpFP32RegTensor,
            auxRegs.p1);
        AscendC::MicroAPI::ShiftRights(
            auxRegs.expFP32RegTensor0, auxRegs.expFP32RegTensor0, SHR_NUM_FOR_FP32, auxRegs.p1);
        AscendC::MicroAPI::Adds(auxRegs.expFP32RegTensor0, auxRegs.expFP32RegTensor0, FP32_BIAS_NEG, auxRegs.p1);
        AscendC::MicroAPI::Maxs(auxRegs.expFP32RegTensor0, auxRegs.expFP32RegTensor0, 0, auxRegs.p1);
        AscendC::MicroAPI::Adds(auxRegs.expFP32RegTensor0, auxRegs.expFP32RegTensor0, NEG_ONE, auxRegs.p1);
        AscendC::MicroAPI::Muls(auxRegs.expFP32RegTensor1, auxRegs.expFP32RegTensor0, NEG_ONE, auxRegs.p1);
        AscendC::MicroAPI::Adds(auxRegs.expFP32RegTensor1, auxRegs.expFP32RegTensor1, FP32_BIAS, auxRegs.p1);
        AscendC::MicroAPI::ShiftLefts(
            auxRegs.expFP32RegTensor1, auxRegs.expFP32RegTensor1, SHR_NUM_FOR_FP32, auxRegs.p1);

        AscendC::MicroAPI::Mul(in, in, (AscendC::MicroAPI::RegTensor<float>&)auxRegs.expFP32RegTensor1, auxRegs.p1);
        AscendC::MicroAPI::Adds(auxRegs.expFP32RegTensor0, auxRegs.expFP32RegTensor0, FP32_BIAS, auxRegs.p1);
        AscendC::MicroAPI::ShiftLefts(
            auxRegs.expFP32RegTensor0, auxRegs.expFP32RegTensor0, SHR_NUM_FOR_FP32, auxRegs.p1);
        AscendC::MicroAPI::CompareScalar<float, CMPMODE::LT>(/*negvalue*/ auxRegs.specialMask, in, 0, auxRegs.p1);
        AscendC::MicroAPI::Truncate<float, roundMode>(in, in, auxRegs.p1);
        AscendC::MicroAPI::Mul(in, in, (AscendC::MicroAPI::RegTensor<float>&)auxRegs.expFP32RegTensor0, auxRegs.p1);
    }
    AscendC::MicroAPI::CompareScalar<float, CMPMODE::EQ>(auxRegs.zeroMask, in, 0, auxRegs.p1);
    AscendC::MicroAPI::MaskAnd(auxRegs.zeroMask, auxRegs.specialMask, auxRegs.zeroMask, auxRegs.p1);
    AscendC::MicroAPI::MaskOr(auxRegs.zeroMask, auxRegs.negInfMask, auxRegs.zeroMask, auxRegs.p1);
    AscendC::MicroAPI::Select<int32_t>(
        (AscendC::MicroAPI::RegTensor<int32_t>&)in, auxRegs.negZeroRegTensor,
        (AscendC::MicroAPI::RegTensor<int32_t>&)in, auxRegs.zeroMask);
}

} // namespace DynamicMxQuant
#endif // DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_OPTIMIZE_HIGH_PERF_H
