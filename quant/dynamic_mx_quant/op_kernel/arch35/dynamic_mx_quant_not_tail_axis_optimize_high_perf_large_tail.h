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
 * \file dynamic_mx_quant_hp.h
 * \brief
 */

#ifndef DYNAMIC_MX_QUANT_HP_2000_H
#define DYNAMIC_MX_QUANT_HP_2000_H

#include "dynamic_mx_quant_common.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "../inc/kernel_utils.h"

namespace DynamicMxQuant {
using namespace AscendC;

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
class DynamicMxQuantNotTailAxisOptimizeLargeTail {
public:
    __aicore__ inline DynamicMxQuantNotTailAxisOptimizeLargeTail(const DynamicMxQuant4OptimizeTilingData* tilingData,
                                                                 TPipe* pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR mxScale);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitParams();
    __aicore__ inline void InitCalcParams(int64_t index);
    __aicore__ inline void ProcessOneLoop(int64_t index);
    __aicore__ inline void CopyOut(int64_t yOffset, int64_t scaleOutOffset, int64_t blockCount, int64_t dataLen);
    __aicore__ inline void CopyIn(int64_t offset, int64_t blockCount, int64_t dataLen);
    __aicore__ inline void ComputeAll(int64_t blockCount, int64_t dataLen);
    __aicore__ inline void ComputeScaleCuBlas(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                              __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint16_t* tmpAddr);
    __aicore__ inline void ComputeScaleOcp(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                           __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint16_t* tmpAddr);
    __aicore__ inline void ComputeScaleCuBlasOptimize(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                                      __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint16_t* tmpAddr);
    __aicore__ inline void ComputeYVf(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                      __ubuf__ uint16_t* tmpAddr, __ubuf__ uint8_t* yAddr);
    __aicore__ inline void ComputeYFromHalf(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                            __ubuf__ uint16_t* tmpAddr, __ubuf__ uint8_t* yAddr);
    __aicore__ inline void ComputeYFromBf16(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                            __ubuf__ uint16_t* tmpAddr, __ubuf__ uint8_t* yAddr);
    __aicore__ inline void ComputeYFromFp32(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                            __ubuf__ uint16_t* tmpAddr, __ubuf__ uint8_t* yAddr);
    __aicore__ inline void ComputeFP4FromHalf(Reg::RegTensor<float>& Reg);

private:
    // tiling data
    const DynamicMxQuant4OptimizeTilingData* tilingData_;

    // pipe & queue & buf
    TPipe* pipe_;
    TQue<QuePosition::VECIN, DB_BUFFER> inQueue;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue;
    TQue<QuePosition::VECOUT, DB_BUFFER> mxScaleQueue;
    TBuf<TPosition::VECCALC> tmpScaleBuf;
    TBuf<TPosition::VECCALC> tmpBuf;

    // gm
    GlobalTensor<xDtype> xGm_;
    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<uint8_t> mxScaleGm_;

    // base varible
    int64_t blockIdx_ = 0;
    int64_t scaleRowCountPerBatch_ = 0;
    int64_t blockCountPerBatch_ = 0;
    int64_t nAlignBlockCount_ = 0;
    int64_t mAlignBlockCount_ = 0;
    int64_t postAxisSize_ = 0;
    int64_t quantAxisSize_ = 0;
    uint32_t invDtypeMax_ = 0;
    uint16_t dtypeYMaxExp_ = 0;
    uint16_t subNumForScale_ = 0;
    int64_t blockSize_ = 0;
    float invDstTypeMax_ = 0;
    float maxLowBound_ = 0.0f;

    // runtime varible
    int64_t calcRow_ = 0;
    int64_t calcCol_ = 0;
    int64_t ubOffset_ = 0;
    int64_t blockOffset_ = 0;
    int64_t loopPerCore_ = 0;
    int64_t ubRowLen_ = 0;
    int64_t ubRowLenTail_ = 0;
    int64_t ubRowCount_ = 0;
    int64_t ubRowCountTail_ = 0;
    int64_t scaleOffset_ = 0;
    int64_t dataLen8Align_ = 0;
    int64_t dataLen16Align_ = 0;
    int64_t dataLen32Align_ = 0;
    int64_t dataLen64Align_ = 0;
    bool scaleNeedsPad_ = false;

    int64_t nAlignBlockCountMinus1_ = 0;
    int64_t mAlignBlockCountMinus1_ = 0;
    int64_t quantAxisSizeMulPostAxis_ = 0;
    int64_t groupPerUbMulPostAxisMul2_ = 0;
    int64_t ubRowLenMul2_ = 0;
    int64_t ubRowCountMulPostAxis_ = 0;
};

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeLargeTail<xDtype, yDtype, roundMode, calcMode>::InitParams()
{
    blockIdx_ = GetBlockIdx();
    int64_t headCoreNum = tilingData_->totalTaskNum <= tilingData_->totalCoreNum ?
                              tilingData_->totalTaskNum :
                              tilingData_->totalTaskNum % tilingData_->usedCoreNum;
    if (blockIdx_ < headCoreNum) {
        loopPerCore_ = tilingData_->loopNumPerHeadCore;
        blockOffset_ = blockIdx_ * loopPerCore_;
    } else {
        loopPerCore_ = tilingData_->loopNumPerTailCore;
        blockOffset_ = headCoreNum * tilingData_->loopNumPerHeadCore + (blockIdx_ - headCoreNum) * loopPerCore_;
    }
    blockSize_ = tilingData_->blockSize;
    // 一次vf计算的行长度，如果是tail后续处理
    ubRowLen_ = tilingData_->ubRowLen;
    ubRowLenTail_ = tilingData_->ubRowLenTail;
    // 一次UB计算的行数，如果是tail后续处理
    ubRowCount_ = tilingData_->ubRowCount;
    ubRowCountTail_ = tilingData_->ubRowCountTail;
    // 一个batch轴的block个数
    blockCountPerBatch_ = tilingData_->blockCountPerBatch;
    // 一个batch轴的scale行数
    scaleRowCountPerBatch_ = tilingData_->scaleRowCountPerBatch;
    // 行方向block个数
    nAlignBlockCount_ = tilingData_->nAlignBlockCount;
    // 列方向block个数
    mAlignBlockCount_ = tilingData_->mAlignBlockCount;
    postAxisSize_ = tilingData_->postAxisSize;
    quantAxisSize_ = tilingData_->quantAxisSize;
    invDstTypeMax_ = tilingData_->invDstTypeMax;
    maxLowBound_ = tilingData_->maxLowBound;

    nAlignBlockCountMinus1_ = nAlignBlockCount_ - 1;
    mAlignBlockCountMinus1_ = mAlignBlockCount_ - 1;
    quantAxisSizeMulPostAxis_ = quantAxisSize_ * postAxisSize_;
    groupPerUbMulPostAxisMul2_ = tilingData_->groupPerUb * DIGIT_TWO * postAxisSize_;
    ubRowLenMul2_ = ubRowLen_ * DIGIT_TWO;
    ubRowCountMulPostAxis_ = ubRowCount_ * postAxisSize_;

    if constexpr (IsSame<DTYPE_Y, fp8_e4m3fn_t>::value) {
        dtypeYMaxExp_ = FP8_E4M3_MAX_EXP;
        invDtypeMax_ = FP8_E4M3_MAX_FLOAT_BITS;
    } else if constexpr (IsSame<DTYPE_Y, fp8_e5m2_t>::value) {
        dtypeYMaxExp_ = FP8_E5M2_MAX_EXP;
        invDtypeMax_ = FP8_E5M2_MAX_FLOAT_BITS;
    } else if constexpr (IsSame<DTYPE_Y, fp4x2_e2m1_t>::value) {
        dtypeYMaxExp_ = FP4_E2M1_BF16_MAX_EXP;
    }

    if constexpr (calcMode == MODE_TWO) {
        subNumForScale_ = static_cast<uint16_t>(tilingData_->subNumForScale);
    } else {
        subNumForScale_ = dtypeYMaxExp_;
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeLargeTail<xDtype, yDtype, roundMode, calcMode>::InitCalcParams(
    int64_t index)
{
    int64_t curBlockOffset = blockOffset_ + index;
    int64_t curBlockBatchOffset = curBlockOffset / blockCountPerBatch_;
    int64_t curBlockRowOffset = curBlockOffset % blockCountPerBatch_ / nAlignBlockCount_;
    int64_t curBlockColOffset = curBlockOffset % blockCountPerBatch_ - curBlockRowOffset * nAlignBlockCount_;

    bool isLastCol = (curBlockColOffset == nAlignBlockCountMinus1_);
    bool isLastRow = (curBlockRowOffset == mAlignBlockCountMinus1_);

    calcRow_ = isLastCol ? ubRowLenTail_ : ubRowLen_;
    calcCol_ = isLastRow ? ubRowCountTail_ : ubRowCount_;

    ubOffset_ = curBlockBatchOffset * quantAxisSizeMulPostAxis_ + curBlockRowOffset * ubRowCountMulPostAxis_ +
                curBlockColOffset * ubRowLen_;
    scaleOffset_ = curBlockBatchOffset * scaleRowCountPerBatch_ * postAxisSize_ +
                   curBlockRowOffset * groupPerUbMulPostAxisMul2_ + curBlockColOffset * ubRowLenMul2_;

    scaleNeedsPad_ = tilingData_->quantAxisIsOdd && isLastRow;

    dataLen8Align_ = (calcRow_ + 7) & (~7);
    dataLen16Align_ = (calcRow_ + 15) & (~15);
    dataLen32Align_ = (calcRow_ + 31) & (~31);
    dataLen64Align_ = (calcRow_ + 63) & (~63);
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeLargeTail<xDtype, yDtype, roundMode, calcMode>::ProcessOneLoop(
    int64_t index)
{
    CopyIn(ubOffset_, calcCol_, calcRow_);
    ComputeAll(calcCol_, calcRow_);
    CopyOut(ubOffset_, scaleOffset_, calcCol_, calcRow_);
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeLargeTail<xDtype, yDtype, roundMode, calcMode>::CopyIn(
    int64_t offset, int64_t blockCount, int64_t dataLen)
{
    int64_t rightPadding = Ops::Base::CeilAlign(dataLen * static_cast<int64_t>(sizeof(xDtype)), UB_BLOCK_SIZE_) /
                               sizeof(xDtype) -
                           dataLen;
    DataCopyExtParams copyInParams = {static_cast<uint16_t>(blockCount),
                                      static_cast<uint32_t>(dataLen * sizeof(xDtype)),
                                      static_cast<uint32_t>((postAxisSize_ - dataLen) * sizeof(xDtype)),
                                      static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<xDtype> padParams{true, 0, static_cast<uint8_t>(rightPadding), 0};

    LocalTensor<xDtype> xLocal = inQueue.template AllocTensor<xDtype>();
    DataCopyPad(xLocal, xGm_[offset], copyInParams, padParams);
    inQueue.template EnQue(xLocal);
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeLargeTail<xDtype, yDtype, roundMode, calcMode>::ComputeAll(
    int64_t blockCount, int64_t dataLen)
{
    LocalTensor<xDtype> x = inQueue.template DeQue<xDtype>();
    LocalTensor<uint8_t> mxScale = mxScaleQueue.template AllocTensor<uint8_t>();
    LocalTensor<uint8_t> y = outQueue.template AllocTensor<uint8_t>();
    LocalTensor<uint8_t> tmpScaleLocal = tmpScaleBuf.Get<uint8_t>();
    LocalTensor<uint16_t> tmpLocal = tmpBuf.Get<uint16_t>();

    auto xAddr = (__ubuf__ xDtype*)x.GetPhyAddr();
    auto yAddr = (__ubuf__ uint8_t*)y.GetPhyAddr();
    auto mxScaleAddr = (__ubuf__ uint8_t*)mxScale.GetPhyAddr();
    auto mxTmpScaleAddr = (__ubuf__ uint8_t*)tmpScaleLocal.GetPhyAddr();
    auto tmpAddr = (__ubuf__ uint16_t*)tmpLocal.GetPhyAddr();

    int64_t calcBlockLoop = Ops::Base::CeilDiv(calcCol_, blockSize_) - 1;
    int64_t calcBlockTail = calcCol_ % blockSize_ == 0 ? blockSize_ : calcCol_ % blockSize_;
    int64_t blockNum = ops::CeilAlign(calcBlockLoop + 1, DIGIT_TWO);

    int64_t xOffset = 0;
    if constexpr (IsSame<xDtype, float>::value) {
        xOffset = blockSize_ * dataLen8Align_;
    } else {
        xOffset = blockSize_ * dataLen16Align_;
    }
    int64_t yOffset = blockSize_ * dataLen64Align_ / DIGIT_TWO;
    if constexpr ((IsSame<yDtype, float8_e4m3_t>::value) || (IsSame<yDtype, float8_e5m2_t>::value)) {
        yOffset = blockSize_ * dataLen32Align_;
    }
    int64_t scaleUbOffset = dataLen32Align_;
    int64_t tmpOffset = dataLen16Align_;
    for (int64_t i = 0; i < calcBlockLoop; i++) {
        if constexpr (calcMode == MODE_THREE || calcMode == MODE_ONE) {
            ComputeScaleCuBlas(dataLen, static_cast<uint16_t>(blockSize_), xAddr, mxTmpScaleAddr, tmpAddr);
        } else if constexpr (calcMode == MODE_TWO) {
            ComputeScaleCuBlasOptimize(dataLen, static_cast<uint16_t>(blockSize_), xAddr, mxTmpScaleAddr, tmpAddr);
        } else {
            ComputeScaleOcp(dataLen, static_cast<uint16_t>(blockSize_), xAddr, mxTmpScaleAddr, tmpAddr);
        }
        ComputeYVf(dataLen, static_cast<uint16_t>(blockSize_), xAddr, tmpAddr, yAddr);
        xAddr = xAddr + xOffset;
        yAddr = yAddr + yOffset;
        mxTmpScaleAddr = mxTmpScaleAddr + scaleUbOffset;
        tmpAddr = tmpAddr + tmpOffset;
    }
    if constexpr (calcMode == MODE_THREE || calcMode == MODE_ONE) {
        ComputeScaleCuBlas(dataLen, static_cast<uint16_t>(calcBlockTail), xAddr, mxTmpScaleAddr, tmpAddr);
    } else if constexpr (calcMode == MODE_TWO) {
        ComputeScaleCuBlasOptimize(dataLen, static_cast<uint16_t>(calcBlockTail), xAddr, mxTmpScaleAddr, tmpAddr);
    } else {
        ComputeScaleOcp(dataLen, static_cast<uint16_t>(calcBlockTail), xAddr, mxTmpScaleAddr, tmpAddr);
    }
    ComputeYVf(dataLen, static_cast<uint16_t>(calcBlockTail), xAddr, tmpAddr, yAddr);
    if (scaleNeedsPad_) {
        int64_t padScaleOffset = (calcBlockLoop + 1) * dataLen32Align_;
        Duplicate<uint8_t>(tmpScaleLocal[padScaleOffset], (uint8_t)0, dataLen32Align_);
    }

    for (int64_t i = 1; i < blockNum; i += 2) {
        Interleave(mxScale[(i - 1) * dataLen32Align_], mxScale[i * dataLen32Align_],
                   tmpScaleLocal[(i - 1) * dataLen32Align_], tmpScaleLocal[i * dataLen32Align_], dataLen32Align_);
    }

    mxScaleQueue.template EnQue(mxScale);
    outQueue.template EnQue(y);
    inQueue.template FreeTensor(x);
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeLargeTail<xDtype, yDtype, roundMode, calcMode>::CopyOut(
    int64_t yOffset, int64_t scaleOutOffset, int64_t blockCount, int64_t dataLen)
{
    uint16_t outBurst = 0;
    uint32_t outBlockLen = 0;
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;
    int64_t YOffset = yOffset;
    uint32_t scaleSrcStride = DIGIT_TWO * ops::CeilDiv(dataLen, static_cast<int64_t>(32)) -
                              ops::CeilDiv(DIGIT_TWO * dataLen, static_cast<int64_t>(32));

    if constexpr (IsSame<yDtype, fp4x2_e2m1_t>::value || IsSame<yDtype, fp4x2_e1m2_t>::value) {
        outBurst = blockCount;
        outBlockLen = dataLen / DIGIT_TWO * sizeof(uint8_t);
        srcStride = 0;
        dstStride = (postAxisSize_ - dataLen) / DIGIT_TWO * sizeof(uint8_t);
        YOffset = yOffset / DIGIT_TWO;
    } else {
        outBurst = blockCount;
        outBlockLen = dataLen * sizeof(uint8_t);
        srcStride = 0;
        dstStride = (postAxisSize_ - dataLen) * sizeof(uint8_t);
        YOffset = yOffset;
    }
    DataCopyExtParams yCopyOutParams = {static_cast<uint16_t>(outBurst), static_cast<uint32_t>(outBlockLen),
                                        static_cast<uint32_t>(srcStride), static_cast<uint32_t>(dstStride),
                                        static_cast<uint32_t>(0)};

    DataCopyExtParams scaleCopyOutParams = {
        static_cast<uint16_t>(ops::CeilDiv(blockCount, DIGIT_TWO * blockSize_)),
        static_cast<uint32_t>(dataLen * DIGIT_TWO * sizeof(uint8_t)), static_cast<uint32_t>(scaleSrcStride),
        static_cast<uint32_t>(DIGIT_TWO * (postAxisSize_ - dataLen) * sizeof(uint8_t)), static_cast<uint32_t>(0)};

    LocalTensor<uint8_t> yLocal = outQueue.template DeQue<uint8_t>();
    DataCopyPad(yGm_[YOffset], yLocal, yCopyOutParams);
    outQueue.FreeTensor(yLocal);

    LocalTensor<uint8_t> mxScaleLocal = mxScaleQueue.template DeQue<uint8_t>();
    DataCopyPad(mxScaleGm_[scaleOutOffset], mxScaleLocal, scaleCopyOutParams);
    mxScaleQueue.FreeTensor(mxScaleLocal);
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeLargeTail<xDtype, yDtype, roundMode, calcMode>::ComputeScaleCuBlas(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScaleAddr,
    __ubuf__ uint16_t* tmpAddr)
{
    uint16_t loopNum = blockCount / 2;
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint32_t> x0U32;
        Reg::RegTensor<uint32_t> x1U32;
        Reg::RegTensor<uint32_t> x0AbsU32;
        Reg::RegTensor<uint32_t> x1AbsU32;
        Reg::RegTensor<uint16_t> x0U16;
        Reg::RegTensor<uint16_t> x1U16;
        Reg::RegTensor<bfloat16_t> xBF16;
        Reg::RegTensor<uint16_t> x0AbsU16;     // x绝对值
        Reg::RegTensor<uint16_t> x1AbsU16;     // x绝对值
        Reg::RegTensor<uint16_t> maxU16;       // 绝对值最大值
        Reg::RegTensor<uint16_t> mxScaleBF16;  // 16位指数
        Reg::RegTensor<uint16_t> mxScale0BF16; // 奇数位16位指数
        Reg::RegTensor<uint16_t> mxScale1BF16; // 偶数位16位指数
        Reg::RegTensor<uint16_t> maxEleFP16;
        Reg::RegTensor<uint32_t> manAbs0FP32; // 尾数与指数加1
        Reg::RegTensor<uint32_t> manAbs1FP32;
        Reg::RegTensor<uint32_t> mxScale0FP32; // 指数
        Reg::RegTensor<uint32_t> mxScale1FP32;
        Reg::RegTensor<uint16_t> reversedShareExpBF16; // 1/scale
        Reg::RegTensor<uint8_t> mxScale;

        Reg::RegTensor<uint32_t> manForFP32;
        Reg::RegTensor<uint16_t> biasU16;
        Reg::RegTensor<uint16_t> maxEleU16;
        Reg::RegTensor<uint16_t> zeroU16;
        Reg::RegTensor<uint32_t> invMax;
        Reg::RegTensor<uint16_t> absForXU16;
        Reg::RegTensor<uint32_t> absForXU32;
        Reg::RegTensor<uint16_t> nanU16;
        Reg::RegTensor<uint16_t> specialExpU16;
        Reg::RegTensor<float> dstTypeMaxReg;

        Reg::MaskReg p0;
        Reg::MaskReg p1;
        Reg::MaskReg p2;
        Reg::MaskReg p3;
        Reg::MaskReg pregAll8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        static constexpr Reg::CastTrait castTraitFp32toBF16Zero = {
            Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_TRUNC};
        static constexpr Reg::CastTrait castTraitFp32toBF16One = {
            Reg::RegLayout::ONE, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_TRUNC};
        static constexpr Reg::CastTrait castTraitZero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitOne = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        Reg::Duplicate(absForXU16, BF16_ABS_MASK);
        Reg::Duplicate(absForXU32, FP32_ABS_MASK);
        Reg::Duplicate(maxEleFP16, FP16_INF);
        Reg::Duplicate(manForFP32, FP32_MX_MAN_MASK);
        Reg::Duplicate(invMax, invDtypeMax_);
        Reg::Duplicate(dstTypeMaxReg, invDstTypeMax_);
        Reg::Duplicate(maxU16, 0);
        Reg::Duplicate(maxEleU16, BF16_MAX_EXP);
        Reg::Duplicate(biasU16, BF16_EXP_BIAS);
        Reg::Duplicate(zeroU16, 0);
        Reg::Duplicate(nanU16, BF16_NAN_CUSTOM);
        Reg::Duplicate(specialExpU16, BF16_SPECIAL_EXP_THRESHOLD);
        if constexpr (IsSame<xDtype, float>::value) {
            Reg::Duplicate(manAbs0FP32, 0);
            for (uint16_t j = 0; j <= loopNum; j++) {
                DataCopy((Reg::RegTensor<xDtype>&)x0U32, xAddr + j * dataLen8Align_);
                DataCopy((Reg::RegTensor<xDtype>&)x1U32, xAddr + (blockCount - j - 1) * dataLen8Align_);
                Reg::And(x0AbsU32, x0U32, absForXU32, pregAll32);
                Reg::And(x1AbsU32, x1U32, absForXU32, pregAll32);
                Reg::Max(x0AbsU32, x1AbsU32, x0AbsU32, pregAll32);
                Reg::Max(manAbs0FP32, manAbs0FP32, x0AbsU32, pregAll32);
            }
        } else {
            for (uint16_t j = 0; j <= loopNum; j++) {
                DataCopy((Reg::RegTensor<xDtype>&)x0U16, xAddr + j * dataLen16Align_);
                DataCopy((Reg::RegTensor<xDtype>&)x1U16, xAddr + (blockCount - j - 1) * dataLen16Align_);
                Reg::And(x0AbsU16, x0U16, absForXU16, pregAll16);
                Reg::And(x1AbsU16, x1U16, absForXU16, pregAll16);
                Reg::Max(x0AbsU16, x1AbsU16, x0AbsU16, pregAll16);
                Reg::Max(maxU16, maxU16, x0AbsU16, pregAll16);
            }

            Reg::Cast<float, xDtype, castTraitZero>((Reg::RegTensor<float>&)manAbs0FP32,
                                                    (Reg::RegTensor<xDtype>&)maxU16, pregAll16);
            Reg::Cast<float, xDtype, castTraitOne>((Reg::RegTensor<float>&)manAbs1FP32, (Reg::RegTensor<xDtype>&)maxU16,
                                                   pregAll16);
        }

        if constexpr (calcMode == MODE_ONE) {
            Reg::Maxs((Reg::RegTensor<float>&)manAbs0FP32, (Reg::RegTensor<float>&)manAbs0FP32, maxLowBound_,
                      pregAll32);
            Reg::Mul((Reg::RegTensor<float>&)manAbs0FP32, (Reg::RegTensor<float>&)manAbs0FP32,
                     (Reg::RegTensor<float>&)invMax, pregAll32);
        } else {
            Reg::Mul((Reg::RegTensor<float>&)manAbs0FP32, (Reg::RegTensor<float>&)manAbs0FP32, dstTypeMaxReg,
                     pregAll32);
        }

        Reg::ShiftRights(mxScale0FP32, manAbs0FP32, FP32_SHR_NUM, pregAll32);
        // 提取尾数
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

        Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)mxScale0BF16, mxScale0FP32);

        if constexpr (!IsSame<xDtype, float>::value) {
            if constexpr (calcMode == MODE_ONE) {
                Reg::Maxs((Reg::RegTensor<float>&)manAbs1FP32, (Reg::RegTensor<float>&)manAbs1FP32, maxLowBound_,
                          pregAll32);
                Reg::Mul((Reg::RegTensor<float>&)manAbs1FP32, (Reg::RegTensor<float>&)manAbs1FP32,
                         (Reg::RegTensor<float>&)invMax, pregAll32);
            } else {
                Reg::Mul((Reg::RegTensor<float>&)manAbs1FP32, (Reg::RegTensor<float>&)manAbs1FP32, dstTypeMaxReg,
                         pregAll32);
            }

            Reg::ShiftRights(mxScale1FP32, manAbs1FP32, FP32_SHR_NUM, pregAll32);
            // 提取尾数
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

            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)mxScale1BF16,
                                                                    mxScale1FP32);

            Reg::Interleave(mxScale0BF16, mxScale1BF16, mxScale0BF16, mxScale1BF16);
        }

        Reg::ShiftLefts(mxScaleBF16, mxScale0BF16, BF16_SHR_NUM, pregAll16);
        Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint8_t>&)mxScale, mxScale0BF16);

        DataCopy(mxScaleAddr, mxScale, pregAll8);

        // 求1/scale
        Reg::Compare<uint16_t, CMPMODE::NE>(p0, mxScaleBF16, maxEleU16, pregAll16);
        Reg::Compare<uint16_t, CMPMODE::EQ>(p1, mxScaleBF16, biasU16, pregAll16);
        Reg::Sub(reversedShareExpBF16, biasU16, mxScaleBF16, pregAll16);
        Reg::Select<uint16_t>(reversedShareExpBF16, reversedShareExpBF16, nanU16, p0);
        Reg::Select<uint16_t>(reversedShareExpBF16, specialExpU16, reversedShareExpBF16, p1);
        DataCopy(tmpAddr, reversedShareExpBF16, pregAll16);
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeLargeTail<xDtype, yDtype, roundMode, calcMode>::ComputeScaleCuBlasOptimize(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScaleAddr,
    __ubuf__ uint16_t* tmpAddr)
{
    uint16_t loopNum = blockCount / 2;
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint32_t> x0U32;
        Reg::RegTensor<uint32_t> x1U32;
        Reg::RegTensor<uint32_t> exp0U32;
        Reg::RegTensor<uint32_t> exp1U32;
        Reg::RegTensor<uint16_t> x0U16;
        Reg::RegTensor<uint16_t> x1U16;
        Reg::RegTensor<uint16_t> expMaxU16;
        Reg::RegTensor<uint32_t> expMaxU32;
        Reg::RegTensor<uint16_t> mxScaleBF16;
        Reg::RegTensor<uint16_t> reversedShareExpBF16;
        Reg::RegTensor<uint8_t> mxScale;
        Reg::RegTensor<uint16_t> x0AbsU16; // x绝对值
        Reg::RegTensor<uint16_t> x1AbsU16; // x绝对值
        Reg::RegTensor<uint32_t> x0AbsU32; // x绝对值
        Reg::RegTensor<uint32_t> x1AbsU32; // x绝对值
        Reg::RegTensor<uint16_t> maxU16;   // 绝对值最大值
        Reg::RegTensor<uint32_t> maxU32;   // 绝对值最大值

        Reg::RegTensor<uint16_t> specialExpU16;
        Reg::RegTensor<uint16_t> biasU16;
        Reg::RegTensor<uint16_t> zeroU16;
        Reg::RegTensor<uint16_t> nanU16;
        Reg::RegTensor<uint16_t> maxEleU16;
        Reg::RegTensor<uint32_t> maxEleU32;
        Reg::RegTensor<uint16_t> dtypeYMaxExp;
        Reg::RegTensor<uint16_t> subNumForScale;
        Reg::RegTensor<uint16_t> fp8NanU16;
        Reg::RegTensor<uint16_t> absForXU16;
        Reg::RegTensor<uint32_t> absForXU32;

        Reg::RegTensor<bfloat16_t> xBF16;
        Reg::RegTensor<uint32_t> roundBiasU32;
        Reg::RegTensor<uint32_t> oneU32;

        Reg::MaskReg infMask;
        Reg::MaskReg zeroMask;
        Reg::MaskReg invalidDataMask;
        Reg::MaskReg pregAll8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        static constexpr Reg::CastTrait castTraitFp32toBF16Zero = {
            Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_TRUNC};
        static constexpr Reg::CastTrait castTraitFp32toBF16One = {
            Reg::RegLayout::ONE, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_TRUNC};
        static constexpr Reg::CastTrait castTraitCublsHalf2Bf16 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                                   Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        Reg::Duplicate(maxEleU16, BF16_MAX_EXP);
        Reg::Duplicate(maxEleU32, FP32_MX_MAX_EXP);
        Reg::Duplicate(subNumForScale, subNumForScale_);
        Reg::Duplicate(absForXU16, BF16_ABS_MASK);
        Reg::Duplicate(absForXU32, FP32_ABS_MASK);
        Reg::Duplicate(dtypeYMaxExp, dtypeYMaxExp_);
        Reg::Duplicate(fp8NanU16, FP8_DEFAULT_MAX_EXP);
        Reg::Duplicate(biasU16, BF16_EXP_BIAS);
        Reg::Duplicate(zeroU16, 0);
        Reg::Duplicate(nanU16, BF16_NAN_CUSTOM);
        Reg::Duplicate(specialExpU16, BF16_SPECIAL_EXP_THRESHOLD);
        Reg::Duplicate(expMaxU16, 0);
        Reg::Duplicate(expMaxU32, 0);
        Reg::Duplicate(maxU16, 0);
        Reg::Duplicate(maxU32, 0);
        Reg::Duplicate(oneU32, 1);

        for (uint16_t j = 0; j <= loopNum; j++) {
            if constexpr (IsSame<xDtype, float>::value) {
                DataCopy((Reg::RegTensor<xDtype>&)x0U32, xAddr + j * dataLen8Align_);
                DataCopy((Reg::RegTensor<xDtype>&)x1U32, xAddr + (blockCount - j - 1) * dataLen8Align_);
                Reg::And(x0AbsU32, x0U32, absForXU32, pregAll32);
                Reg::And(x1AbsU32, x1U32, absForXU32, pregAll32);
                Reg::Max(x0AbsU32, x0AbsU32, x1AbsU32, pregAll32);
                Reg::Max(maxU32, maxU32, x0AbsU32, pregAll32);
            } else {
                DataCopy((Reg::RegTensor<xDtype>&)x0U16, xAddr + j * dataLen16Align_);
                DataCopy((Reg::RegTensor<xDtype>&)x1U16, xAddr + (blockCount - j - 1) * dataLen16Align_);
                Reg::And(x0AbsU16, x0U16, absForXU16, pregAll16);
                Reg::And(x1AbsU16, x1U16, absForXU16, pregAll16);
                if constexpr (IsSame<xDtype, half>::value) {
                    Reg::Cast<bfloat16_t, xDtype, castTraitCublsHalf2Bf16>(
                        (Reg::RegTensor<bfloat16_t>&)x0AbsU16, (Reg::RegTensor<float16_t>&)x0AbsU16, pregAll16);
                    Reg::Cast<bfloat16_t, xDtype, castTraitCublsHalf2Bf16>(
                        (Reg::RegTensor<bfloat16_t>&)x1AbsU16, (Reg::RegTensor<float16_t>&)x1AbsU16, pregAll16);
                }
                Reg::Max(x1AbsU16, x1AbsU16, x0AbsU16, pregAll16);
                Reg::Max(maxU16, maxU16, x1AbsU16, pregAll16);
            }
        }

        if constexpr (IsSame<xDtype, float>::value) {
            Reg::ShiftRights(roundBiasU32, maxU32, FP32_PACK_SHR_NUM, pregAll32);
            Reg::And(roundBiasU32, roundBiasU32, oneU32, pregAll32);
            Reg::Adds(roundBiasU32, roundBiasU32, 0x7FFF, pregAll32);
            Reg::Add(maxU32, maxU32, roundBiasU32, pregAll32);
            Reg::ShiftRights(maxU32, maxU32, FP32_PACK_SHR_NUM, pregAll32);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(maxU16, maxU32);
            Reg::And(expMaxU16, maxU16, maxEleU16, pregAll16);
        } else {
            Reg::And(expMaxU16, maxU16, maxEleU16, pregAll16);
        }

        Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxU16, maxEleU16, pregAll16);
        Reg::Compare<uint16_t, CMPMODE::LT>(invalidDataMask, expMaxU16, dtypeYMaxExp, pregAll16);

        Reg::Sub(expMaxU16, maxU16, subNumForScale, pregAll16);

        Reg::Select<uint16_t>(expMaxU16, zeroU16, expMaxU16, invalidDataMask);
        Reg::ShiftRights(mxScaleBF16, expMaxU16, BF16_SHR_NUM, pregAll16);
        Reg::Select<uint16_t>(mxScaleBF16, mxScaleBF16, fp8NanU16, infMask);

        Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint8_t>&)mxScale, mxScaleBF16);
        DataCopy(mxScaleAddr, mxScale, pregAll8);
        // 求1/scale
        Reg::And(expMaxU16, expMaxU16, maxEleU16, pregAll16);
        Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxU16, zeroU16, pregAll16);
        Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMaxU16, biasU16, pregAll16);
        Reg::Sub(reversedShareExpBF16, biasU16, expMaxU16, pregAll16);
        Reg::Select<uint16_t>(reversedShareExpBF16, reversedShareExpBF16, nanU16, infMask);
        Reg::Select<uint16_t>(reversedShareExpBF16, reversedShareExpBF16, zeroU16, zeroMask);
        Reg::Select<uint16_t>(reversedShareExpBF16, specialExpU16, reversedShareExpBF16, invalidDataMask);
        DataCopy(tmpAddr, reversedShareExpBF16, pregAll16);
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeLargeTail<xDtype, yDtype, roundMode, calcMode>::ComputeScaleOcp(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScaleAddr,
    __ubuf__ uint16_t* tmpAddr)
{
    uint16_t loopNum = blockCount / 2;
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint32_t> x0U32;
        Reg::RegTensor<uint32_t> x1U32;
        Reg::RegTensor<uint32_t> exp0U32;
        Reg::RegTensor<uint32_t> exp1U32;
        Reg::RegTensor<uint32_t> expMaxU32;
        Reg::RegTensor<uint16_t> x0U16;
        Reg::RegTensor<uint16_t> x1U16;
        Reg::RegTensor<uint16_t> exp0U16;
        Reg::RegTensor<uint16_t> exp1U16;
        Reg::RegTensor<uint16_t> expMaxU16;
        Reg::RegTensor<uint16_t> mxScaleBF16;
        Reg::RegTensor<uint16_t> reversedShareExpBF16;
        Reg::RegTensor<uint8_t> mxScale;
        Reg::RegTensor<uint16_t> absU16; // x绝对值
        Reg::RegTensor<uint16_t> maxU16; // 绝对值最大值

        Reg::RegTensor<uint16_t> specialExpU16;
        Reg::RegTensor<uint16_t> biasU16;
        Reg::RegTensor<uint16_t> zeroU16;
        Reg::RegTensor<uint16_t> nanU16;
        Reg::RegTensor<uint16_t> maxEleU16;
        Reg::RegTensor<uint32_t> maxEleU32;
        Reg::RegTensor<uint16_t> maxEleFP16;
        Reg::RegTensor<uint16_t> dtypeYMaxExp;
        Reg::RegTensor<uint16_t> subNumForScale;
        Reg::RegTensor<uint16_t> fp8NanU16;
        Reg::RegTensor<uint16_t> absForXU16;

        Reg::MaskReg infMask;
        Reg::MaskReg zeroMask;
        Reg::MaskReg invalidDataMask;
        Reg::MaskReg pregAll8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        static constexpr Reg::CastTrait castTraitOcpHalf2Bf16 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                                 Reg::MaskMergeMode::ZEROING,
                                                                 AscendC::RoundMode::CAST_TRUNC};
        static constexpr Reg::CastTrait castTraitFp32toBF16Zero = {
            Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_TRUNC};
        static constexpr Reg::CastTrait castTraitFp32toBF16One = {
            Reg::RegLayout::ONE, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_TRUNC};

        Reg::Duplicate(maxEleU16, BF16_MAX_EXP);
        Reg::Duplicate(maxEleU32, FP32_MX_MAX_EXP);
        Reg::Duplicate(subNumForScale, subNumForScale_);
        Reg::Duplicate(absForXU16, BF16_ABS_MASK);
        Reg::Duplicate(maxEleFP16, FP16_INF);
        Reg::Duplicate(dtypeYMaxExp, dtypeYMaxExp_);
        Reg::Duplicate(fp8NanU16, FP8_DEFAULT_MAX_EXP);
        Reg::Duplicate(biasU16, BF16_EXP_BIAS);
        Reg::Duplicate(zeroU16, 0);
        Reg::Duplicate(nanU16, BF16_NAN_CUSTOM);
        Reg::Duplicate(specialExpU16, BF16_SPECIAL_EXP_THRESHOLD);
        Reg::Duplicate(expMaxU16, 0);
        Reg::Duplicate(expMaxU32, 0);
        Reg::Duplicate(maxU16, 0);

        for (uint16_t j = 0; j <= loopNum; j++) {
            if constexpr (IsSame<xDtype, float>::value) {
                DataCopy((Reg::RegTensor<xDtype>&)x0U32, xAddr + j * dataLen8Align_);
                DataCopy((Reg::RegTensor<xDtype>&)x1U32, xAddr + (blockCount - j - 1) * dataLen8Align_);
                Reg::And(exp0U32, x0U32, maxEleU32, pregAll32);
                Reg::And(exp1U32, x1U32, maxEleU32, pregAll32);
                Reg::Max(exp0U32, exp0U32, exp1U32, pregAll32);
                Reg::Max(expMaxU32, expMaxU32, exp0U32, pregAll32);
            } else {
                DataCopy((Reg::RegTensor<xDtype>&)x0U16, xAddr + j * dataLen16Align_);
                DataCopy((Reg::RegTensor<xDtype>&)x1U16, xAddr + (blockCount - j - 1) * dataLen16Align_);
                if constexpr (IsSame<xDtype, half>::value) {
                    Reg::Cast<bfloat16_t, xDtype, castTraitOcpHalf2Bf16>((Reg::RegTensor<bfloat16_t>&)exp0U16,
                                                                         (Reg::RegTensor<float16_t>&)x0U16, pregAll16);
                    Reg::And(exp0U16, (Reg::RegTensor<uint16_t>&)exp0U16, maxEleU16, pregAll16);
                    Reg::Cast<bfloat16_t, xDtype, castTraitOcpHalf2Bf16>((Reg::RegTensor<bfloat16_t>&)exp1U16,
                                                                         (Reg::RegTensor<float16_t>&)x1U16, pregAll16);
                    Reg::And(exp1U16, (Reg::RegTensor<uint16_t>&)exp1U16, maxEleU16, pregAll16);
                } else {
                    Reg::And(exp0U16, (Reg::RegTensor<uint16_t>&)x0U16, maxEleU16, pregAll16);
                    Reg::And(exp1U16, (Reg::RegTensor<uint16_t>&)x1U16, maxEleU16, pregAll16);
                }
                Reg::Max(exp0U16, exp0U16, exp1U16, pregAll16);
                Reg::Max(expMaxU16, expMaxU16, exp0U16, pregAll16);
            }
        }
        if constexpr (IsSame<xDtype, float>::value) {
            Reg::ShiftRights(expMaxU32, expMaxU32, FP32_PACK_SHR_NUM, pregAll32);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(expMaxU16, expMaxU32);
        }

        Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxU16, maxEleU16, pregAll16);
        Reg::Compare<uint16_t, CMPMODE::LT>(invalidDataMask, expMaxU16, dtypeYMaxExp, pregAll16);

        Reg::Sub(expMaxU16, expMaxU16, subNumForScale, pregAll16);

        Reg::Select<uint16_t>(expMaxU16, zeroU16, expMaxU16, invalidDataMask);
        Reg::ShiftRights(mxScaleBF16, expMaxU16, BF16_SHR_NUM, pregAll16);
        Reg::Select<uint16_t>(mxScaleBF16, mxScaleBF16, fp8NanU16, infMask);

        DataCopy<uint8_t, Reg::StoreDist::DIST_PACK_B16>(mxScaleAddr, (Reg::RegTensor<uint8_t>&)mxScaleBF16, pregAll8);

        // 求1/scale
        Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxU16, zeroU16, pregAll16);
        Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMaxU16, biasU16, pregAll16);
        Reg::Sub(reversedShareExpBF16, biasU16, expMaxU16, pregAll16);
        Reg::Select<uint16_t>(reversedShareExpBF16, reversedShareExpBF16, nanU16, infMask);
        Reg::Select<uint16_t>(reversedShareExpBF16, reversedShareExpBF16, zeroU16, zeroMask);
        Reg::Select<uint16_t>(reversedShareExpBF16, specialExpU16, reversedShareExpBF16, invalidDataMask);
        DataCopy(tmpAddr, reversedShareExpBF16, pregAll16);
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeLargeTail<xDtype, yDtype, roundMode, calcMode>::ComputeYVf(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* tmpAddr, __ubuf__ uint8_t* yAddr)
{
    if constexpr (IsSame<xDtype, half>::value) {
        // half: fp16 → fp32 widen via ZERO/ONE cast；float: layout-only register split
        ComputeYFromHalf(dataLen, blockCount, xAddr, tmpAddr, yAddr);
    } else if constexpr (IsSame<xDtype, bfloat16_t>::value) {
        ComputeYFromBf16(dataLen, blockCount, xAddr, tmpAddr, yAddr);
    } else if constexpr (IsSame<xDtype, float>::value) {
        ComputeYFromFp32(dataLen, blockCount, xAddr, tmpAddr, yAddr);
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeLargeTail<xDtype, yDtype, roundMode, calcMode>::ComputeYFromHalf(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* tmpAddr, __ubuf__ uint8_t* yAddr)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> x;
        Reg::RegTensor<bfloat16_t> x0BF16;
        Reg::RegTensor<bfloat16_t> x1BF16;
        Reg::RegTensor<float> x0FP32;
        Reg::RegTensor<float> x1FP32;
        Reg::RegTensor<uint16_t> reversedShareExpBF16;
        Reg::RegTensor<float> reversedShareExp0FP32;
        Reg::RegTensor<float> reversedShareExp1FP32;
        Reg::RegTensor<yDtype> yZeroFP8;
        Reg::RegTensor<yDtype> yOneFP8;
        Reg::RegTensor<yDtype> yZeroFP4;

        Reg::MaskReg pregAll8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        static constexpr Reg::CastTrait castTraitBf16toFp4 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                              Reg::MaskMergeMode::ZEROING, roundMode};
        static constexpr Reg::CastTrait castTraitXdtypetoFp32Zero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                     Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitXdtypetoFp32One = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                                    Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitFp32toBf16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                               Reg::MaskMergeMode::ZEROING, roundMode};
        static constexpr Reg::CastTrait castTraitFp32toYdtype = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                                 Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        Reg::DataCopy<uint16_t, Reg::LoadDist::DIST_NORM>(reversedShareExpBF16, tmpAddr);

        Reg::Cast<float, bfloat16_t, castTraitXdtypetoFp32Zero>(
            reversedShareExp0FP32, (Reg::RegTensor<bfloat16_t>&)reversedShareExpBF16, pregAll16);
        Reg::Cast<float, bfloat16_t, castTraitXdtypetoFp32One>(
            reversedShareExp1FP32, (Reg::RegTensor<bfloat16_t>&)reversedShareExpBF16, pregAll16);

        for (uint16_t j = 0; j < blockCount; j++) {
            Reg::DataCopy<xDtype, Reg::LoadDist::DIST_NORM>(x, xAddr + j * dataLen16Align_);

            Reg::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(x0FP32, x, pregAll16);
            Reg::Cast<float, xDtype, castTraitXdtypetoFp32One>(x1FP32, x, pregAll16);

            Reg::Mul(x0FP32, x0FP32, reversedShareExp0FP32, pregAll32);
            Reg::Mul(x1FP32, x1FP32, reversedShareExp1FP32, pregAll32);

            if constexpr (IsSame<yDtype, fp4x2_e2m1_t>::value || IsSame<yDtype, fp4x2_e1m2_t>::value) {
                ComputeFP4FromHalf(x0FP32);
                ComputeFP4FromHalf(x1FP32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBf16>((Reg::RegTensor<bfloat16_t>&)x0BF16, x0FP32,
                                                                  pregAll32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)x0BF16,
                                                                        (Reg::RegTensor<uint32_t>&)x0BF16);

                Reg::Cast<bfloat16_t, float, castTraitFp32toBf16>((Reg::RegTensor<bfloat16_t>&)x1BF16, x1FP32,
                                                                  pregAll32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)x1BF16,
                                                                        (Reg::RegTensor<uint32_t>&)x1BF16);

                Reg::Interleave(x0BF16, x1BF16, x0BF16, x1BF16);

                Reg::Cast<yDtype, bfloat16_t, castTraitBf16toFp4>(yZeroFP4, (Reg::RegTensor<bfloat16_t>&)x0BF16,
                                                                  pregAll16);

                DataCopy<uint8_t, Reg::StoreDist::DIST_PACK4_B32>(yAddr + (j * dataLen64Align_ / DIGIT_TWO),
                                                                  (Reg::RegTensor<uint8_t>&)yZeroFP4, pregAll8);
            } else {
                Reg::Cast<yDtype, float, castTraitFp32toYdtype>(yZeroFP8, (Reg::RegTensor<float>&)x0FP32, pregAll32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)yZeroFP8,
                                                                        (Reg::RegTensor<uint32_t>&)yZeroFP8);

                Reg::Cast<yDtype, float, castTraitFp32toYdtype>(yOneFP8, (Reg::RegTensor<float>&)x1FP32, pregAll32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)yOneFP8,
                                                                        (Reg::RegTensor<uint32_t>&)yOneFP8);

                Reg::Interleave((Reg::RegTensor<uint16_t>&)yZeroFP8, (Reg::RegTensor<uint16_t>&)yOneFP8,
                                (Reg::RegTensor<uint16_t>&)yZeroFP8, (Reg::RegTensor<uint16_t>&)yOneFP8);

                DataCopy<uint8_t, Reg::StoreDist::DIST_PACK_B16>(yAddr + (j * dataLen32Align_),
                                                                 (Reg::RegTensor<uint8_t>&)yZeroFP8, pregAll8);
            }
        }
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeLargeTail<xDtype, yDtype, roundMode, calcMode>::ComputeFP4FromHalf(
    Reg::RegTensor<float>& Reg)
{
    Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg zeroMask;
    Reg::MaskReg specialMask;
    Reg::MaskReg negInfMask;

    Reg::RegTensor<int32_t> negZeroI32;
    Reg::RegTensor<int32_t> maxExpFP32;
    Reg::RegTensor<int32_t> exp0FP32;
    Reg::RegTensor<int32_t> exp1FP32;

    Reg::Duplicate(negZeroI32, FP32_NEG_ZERO_BITS);

    Reg::Compare<int32_t, CMPMODE::EQ>(
        /*negzero*/ negInfMask, (Reg::RegTensor<int32_t>&)Reg, negZeroI32, pregAll32);
    if constexpr (IsSame<yDtype, fp4x2_e1m2_t>::value) {
        Reg::Muls(Reg, Reg, FP4_SCALE_FACTOR, pregAll32);
        Reg::CompareScalar<float, CMPMODE::LT>(/*negvalue*/ specialMask, Reg, 0, pregAll32);
        Reg::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        Reg::Muls(Reg, Reg, FP4_INV_SCALE_FACTOR, pregAll32);
    } else { // fp4x2_e2m1
        Reg::Duplicate(maxExpFP32, FP32_MX_MAX_EXP);
        Reg::And(exp0FP32, (Reg::RegTensor<int32_t>&)Reg, maxExpFP32, pregAll32);
        Reg::ShiftRights(exp0FP32, exp0FP32, FP32_SHR_NUM, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, FP32_BIAS_NEG_VALUE, pregAll32);
        Reg::Maxs(exp0FP32, exp0FP32, 0, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, FP32_NEG_ONE, pregAll32);
        Reg::Muls(exp1FP32, exp0FP32, FP32_NEG_ONE, pregAll32);
        Reg::Adds(exp1FP32, exp1FP32, FP32_BIAS_VALUE, pregAll32);
        Reg::ShiftLefts(exp1FP32, exp1FP32, FP32_SHR_NUM, pregAll32);

        Reg::Mul(Reg, Reg, (Reg::RegTensor<float>&)exp1FP32, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, FP32_BIAS_VALUE, pregAll32);
        Reg::ShiftLefts(exp0FP32, exp0FP32, FP32_SHR_NUM, pregAll32);
        Reg::CompareScalar<float, CMPMODE::LT>(/*negvalue*/ specialMask, Reg, 0, pregAll32);
        Reg::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        Reg::Mul(Reg, Reg, (Reg::RegTensor<float>&)exp0FP32, pregAll32);
    }
    Reg::CompareScalar<float, CMPMODE::EQ>(zeroMask, Reg, 0, pregAll32);
    Reg::MaskAnd(zeroMask, specialMask, zeroMask, pregAll32);
    Reg::MaskOr(zeroMask, negInfMask, zeroMask, pregAll32);
    Reg::Select<int32_t>((Reg::RegTensor<int32_t>&)Reg, negZeroI32, (Reg::RegTensor<int32_t>&)Reg, zeroMask);
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeLargeTail<xDtype, yDtype, roundMode, calcMode>::ComputeYFromBf16(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* tmpAddr, __ubuf__ uint8_t* yAddr)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> x;
        Reg::RegTensor<bfloat16_t> valueBF16;
        Reg::RegTensor<float> x0FP32;
        Reg::RegTensor<float> x1FP32;
        Reg::RegTensor<uint16_t> reversedShareExpBF16;
        Reg::RegTensor<float> reversedShareExp0FP32;
        Reg::RegTensor<float> reversedShareExp1FP32;
        Reg::RegTensor<yDtype> yZeroFP8;
        Reg::RegTensor<yDtype> yOneFP8;
        Reg::RegTensor<yDtype> yZeroFP4;

        Reg::MaskReg pregAll8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        static constexpr Reg::CastTrait castTraitBf16toFp4 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                              Reg::MaskMergeMode::ZEROING, roundMode};
        static constexpr Reg::CastTrait castTraitXdtypetoFp32Zero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                     Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitXdtypetoFp32One = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                                    Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitFp32toYdtype = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                                 Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        Reg::DataCopy<uint16_t, Reg::LoadDist::DIST_NORM>(reversedShareExpBF16, tmpAddr);

        Reg::Cast<float, bfloat16_t, castTraitXdtypetoFp32Zero>(
            reversedShareExp0FP32, (Reg::RegTensor<bfloat16_t>&)reversedShareExpBF16, pregAll16);
        Reg::Cast<float, bfloat16_t, castTraitXdtypetoFp32One>(
            reversedShareExp1FP32, (Reg::RegTensor<bfloat16_t>&)reversedShareExpBF16, pregAll16);

        for (uint16_t j = 0; j < blockCount; j++) {
            Reg::DataCopy<xDtype, Reg::LoadDist::DIST_NORM>(x, xAddr + j * dataLen16Align_);

            if constexpr (IsSame<yDtype, fp4x2_e2m1_t>::value || IsSame<yDtype, fp4x2_e1m2_t>::value) {
                Reg::Mul(valueBF16, x, (Reg::RegTensor<bfloat16_t>&)reversedShareExpBF16, pregAll16);
                Reg::Cast<yDtype, bfloat16_t, castTraitBf16toFp4>(yZeroFP4, valueBF16, pregAll16);

                DataCopy<uint8_t, Reg::StoreDist::DIST_PACK4_B32>(yAddr + (j * dataLen64Align_ / DIGIT_TWO),
                                                                  (Reg::RegTensor<uint8_t>&)yZeroFP4, pregAll8);
            } else {
                Reg::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(x0FP32, x, pregAll16);
                Reg::Cast<float, xDtype, castTraitXdtypetoFp32One>(x1FP32, x, pregAll16);

                Reg::Mul(x0FP32, x0FP32, reversedShareExp0FP32, pregAll32);
                Reg::Mul(x1FP32, x1FP32, reversedShareExp1FP32, pregAll32);
                Reg::Cast<yDtype, float, castTraitFp32toYdtype>(yZeroFP8, (Reg::RegTensor<float>&)x0FP32, pregAll32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)yZeroFP8,
                                                                        (Reg::RegTensor<uint32_t>&)yZeroFP8);

                Reg::Cast<yDtype, float, castTraitFp32toYdtype>(yOneFP8, (Reg::RegTensor<float>&)x1FP32, pregAll32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)yOneFP8,
                                                                        (Reg::RegTensor<uint32_t>&)yOneFP8);

                Reg::Interleave((Reg::RegTensor<uint16_t>&)yZeroFP8, (Reg::RegTensor<uint16_t>&)yOneFP8,
                                (Reg::RegTensor<uint16_t>&)yZeroFP8, (Reg::RegTensor<uint16_t>&)yOneFP8);

                DataCopy<uint8_t, Reg::StoreDist::DIST_PACK_B16>(yAddr + (j * dataLen32Align_),
                                                                 (Reg::RegTensor<uint8_t>&)yZeroFP8, pregAll8);
            }
        }
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void
DynamicMxQuantNotTailAxisOptimizeLargeTail<xDtype, yDtype, roundMode, calcMode>::ComputeYFromFp32(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* tmpAddr, __ubuf__ uint8_t* yAddr)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> x;
        Reg::RegTensor<bfloat16_t> valueBF16;
        Reg::RegTensor<bfloat16_t> x0BF16;
        Reg::RegTensor<bfloat16_t> x1BF16;
        Reg::RegTensor<float> x0FP32;
        Reg::RegTensor<float> x1FP32;
        Reg::RegTensor<uint16_t> reversedShareExpBF16;
        Reg::RegTensor<float> reversedShareExp0FP32;
        Reg::RegTensor<float> reversedShareExp1FP32;
        Reg::RegTensor<uint32_t> yZeroU32;
        Reg::RegTensor<uint16_t> yZeroU16;
        Reg::RegTensor<uint8_t> yZeroU8;
        Reg::RegTensor<yDtype> yZeroFP8;
        Reg::RegTensor<yDtype> yOneFP8;
        Reg::RegTensor<yDtype> yZeroFP4;
        Reg::MaskReg pregAll8;
        Reg::MaskReg pregAll16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        uint32_t maskLen = dataLen32Align_;
        static constexpr Reg::CastTrait castTraitBf16toFp4 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                              Reg::MaskMergeMode::ZEROING, roundMode};
        static constexpr Reg::CastTrait castTraitXdtypetoFp32Zero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                     Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitXdtypetoFp32One = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                                    Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitFp32toYdtype = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                                 Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTraitFp32toBF16Zero = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                                   Reg::MaskMergeMode::ZEROING, roundMode};
        static constexpr Reg::CastTrait castTraitFp32toBF16One = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                                  Reg::MaskMergeMode::ZEROING, roundMode};

        DataCopy<uint16_t, Reg::LoadDist::DIST_UNPACK_B16>((Reg::RegTensor<uint16_t>&)reversedShareExp0FP32, tmpAddr);
        Reg::Cast<float, bfloat16_t, castTraitXdtypetoFp32Zero>(
            reversedShareExp0FP32, (Reg::RegTensor<bfloat16_t>&)reversedShareExp0FP32, pregAll16);

        for (uint16_t j = 0; j < blockCount; j++) {
            DataCopy<xDtype, Reg::LoadDist::DIST_NORM>(x0FP32, xAddr + j * dataLen8Align_);
            if constexpr (IsSame<yDtype, fp4x2_e2m1_t>::value || IsSame<yDtype, fp4x2_e1m2_t>::value) {
                maskLen = dataLen64Align_ * DIGIT_TWO;
                pregAll8 = Reg::UpdateMask<uint8_t>(maskLen);
                Reg::Mul(x0FP32, x0FP32, reversedShareExp0FP32, pregAll32);
                ComputeFP4FromHalf(x0FP32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16Zero>(x0BF16, x0FP32, pregAll32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)x0BF16,
                                                                        (Reg::RegTensor<uint32_t>&)x0BF16);

                Reg::Cast<yDtype, bfloat16_t, castTraitBf16toFp4>(yZeroFP4, (Reg::RegTensor<bfloat16_t>&)x0BF16,
                                                                  pregAll16);
                DataCopy<uint8_t, Reg::StoreDist::DIST_PACK4_B32>(yAddr + (j * dataLen64Align_ / DIGIT_TWO),
                                                                  (Reg::RegTensor<uint8_t>&)yZeroFP4, pregAll8);
            } else {
                maskLen = dataLen32Align_;
                pregAll8 = Reg::UpdateMask<uint8_t>(maskLen);
                Reg::Mul(x0FP32, x0FP32, reversedShareExp0FP32, pregAll32);
                Reg::Cast<yDtype, float, castTraitFp32toYdtype>((Reg::RegTensor<yDtype>&)yZeroU32, x0FP32, pregAll32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(yZeroU16, yZeroU32);
                Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(yZeroU8, yZeroU16);
                DataCopy<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM>(
                    yAddr, yZeroU8, dataLen32Align_, pregAll8);
            }
        }
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeLargeTail<xDtype, yDtype, roundMode, calcMode>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR mxScale)
{
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    // init block params
    InitParams();

    xGm_.SetGlobalBuffer((__gm__ xDtype*)(x));
    yGm_.SetGlobalBuffer((__gm__ uint8_t*)(y));
    mxScaleGm_.SetGlobalBuffer((__gm__ uint8_t*)(mxScale));

    int64_t scaleUbRowCount = ops::CeilDiv(ubRowCount_, DIGIT_TWO * blockSize_) * DIGIT_TWO;

    int64_t inBufferSize_ = ops::CeilAlign(ubRowLen_ * ubRowCount_ * static_cast<int64_t>(sizeof(xDtype)),
                                           static_cast<int64_t>(Ops::Base::GetUbBlockSize()));

    int64_t mxScaleBufferSize_ = ops::CeilAlign(ubRowLen_ * scaleUbRowCount,
                                                static_cast<int64_t>(Ops::Base::GetUbBlockSize()));

    int64_t outBufferSize_ = ops::CeilAlign(static_cast<int64_t>(ubRowLen_ * ubRowCount_),
                                            static_cast<int64_t>(Ops::Base::GetUbBlockSize()));

    int64_t tmpBufferSize_ = ops::CeilAlign(ubRowLen_ * scaleUbRowCount * static_cast<int64_t>(sizeof(xDtype)),
                                            static_cast<int64_t>(Ops::Base::GetUbBlockSize()));

    int64_t tmpScaleBufferSize_ = mxScaleBufferSize_;

    pipe_->InitBuffer(inQueue, DB_BUFFER, inBufferSize_);
    pipe_->InitBuffer(mxScaleQueue, DB_BUFFER, mxScaleBufferSize_);
    pipe_->InitBuffer(outQueue, DB_BUFFER, outBufferSize_);
    pipe_->InitBuffer(tmpBuf, tmpBufferSize_);
    pipe_->InitBuffer(tmpScaleBuf, tmpScaleBufferSize_);
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeLargeTail<xDtype, yDtype, roundMode, calcMode>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    for (int64_t i = 0; i < loopPerCore_; i++) {
        InitCalcParams(i);
        ProcessOneLoop(i);
    }
}

} // namespace DynamicMxQuant
#endif // DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_FP8_H
