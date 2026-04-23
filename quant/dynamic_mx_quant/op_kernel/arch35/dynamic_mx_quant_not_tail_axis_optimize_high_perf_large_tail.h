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
class DynamicMxQuantHP2000 {
public:
    __aicore__ inline DynamicMxQuantHP2000(const DynamicMxQuant4OptimizeTilingData* tilingData, TPipe* pipe)
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
    __aicore__ inline void ComputeScaleCuBlas(
        uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScaleAddr,
        __ubuf__ uint16_t* tmpAddr);
    __aicore__ inline void ComputeScaleOcp(
        uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScaleAddr,
        __ubuf__ uint16_t* tmpAddr);
    __aicore__ inline void ComputeScaleCuBlasOptimize(
        uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScaleAddr,
        __ubuf__ uint16_t* tmpAddr);
    __aicore__ inline void ComputeYVf(
        uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* tmpAddr,
        __ubuf__ uint8_t* yAddr);
    __aicore__ inline void ComputeYFromHalf(
        uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* tmpAddr,
        __ubuf__ uint8_t* yAddr);
    __aicore__ inline void ComputeYFromBf16(
        uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* tmpAddr,
        __ubuf__ uint8_t* yAddr);
    __aicore__ inline void ComputeFP4FromHalf(AscendC::MicroAPI::RegTensor<float>& Reg);

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
    uint32_t INV_DTYPE_MAX = 0;
    uint16_t DTYPE_Y_MAX_EXP = 0;
    uint16_t subNumForScale_ = 0;
    int64_t blockSize_ = 0;
    float invDstTypeMax_ = 0;

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
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype, roundMode, calcMode>::InitParams()
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

    nAlignBlockCountMinus1_ = nAlignBlockCount_ - 1;
    mAlignBlockCountMinus1_ = mAlignBlockCount_ - 1;
    quantAxisSizeMulPostAxis_ = quantAxisSize_ * postAxisSize_;
    groupPerUbMulPostAxisMul2_ = tilingData_->groupPerUb * DIGIT_TWO * postAxisSize_;
    ubRowLenMul2_ = ubRowLen_ * DIGIT_TWO;
    ubRowCountMulPostAxis_ = ubRowCount_ * postAxisSize_;

    if constexpr (IsSame<DTYPE_Y, fp8_e4m3fn_t>::value) {
        DTYPE_Y_MAX_EXP = FP8_E4M3_MAX_EXP;
        INV_DTYPE_MAX = FP8_E4M3_MAX;
    } else if constexpr (IsSame<DTYPE_Y, fp8_e5m2_t>::value) {
        DTYPE_Y_MAX_EXP = FP8_E5M2_MAX_EXP;
        INV_DTYPE_MAX = FP8_E5M2_MAX;
    } else if constexpr (IsSame<DTYPE_Y, fp4x2_e2m1_t>::value) {
        DTYPE_Y_MAX_EXP = FP4_E2M1_BF16_MAX_EXP;
    }

    if constexpr (calcMode == ModeTwo) {
        subNumForScale_ = static_cast<uint16_t>(tilingData_->subNumForScale);
    } else {
        subNumForScale_ = DTYPE_Y_MAX_EXP;
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype, roundMode, calcMode>::InitCalcParams(int64_t index)
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

    dataLen16Align_ = (calcRow_ + 15) & (~15);
    dataLen32Align_ = (calcRow_ + 31) & (~31);
    dataLen64Align_ = (calcRow_ + 63) & (~63);
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype, roundMode, calcMode>::ProcessOneLoop(int64_t index)
{
    CopyIn(ubOffset_, calcCol_, calcRow_);
    ComputeAll(calcCol_, calcRow_);
    CopyOut(ubOffset_, scaleOffset_, calcCol_, calcRow_);
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype, roundMode, calcMode>::CopyIn(
    int64_t offset, int64_t blockCount, int64_t dataLen)
{
    int64_t rightPadding =
        Ops::Base::CeilAlign(dataLen * static_cast<int64_t>(sizeof(xDtype)), static_cast<int64_t>(32)) /
            sizeof(xDtype) -
        dataLen;
    DataCopyExtParams copyInParams = {
        static_cast<uint16_t>(blockCount), static_cast<uint32_t>(dataLen * sizeof(xDtype)),
        static_cast<uint32_t>((postAxisSize_ - dataLen) * sizeof(xDtype)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)};
    DataCopyPadExtParams<xDtype> padParams{true, 0, static_cast<uint8_t>(rightPadding), 0};

    LocalTensor<xDtype> xLocal = inQueue.template AllocTensor<xDtype>();
    DataCopyPad(xLocal, xGm_[offset], copyInParams, padParams);
    inQueue.template EnQue(xLocal);
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype, roundMode, calcMode>::ComputeAll(
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

    int64_t xOffset = blockSize_ * dataLen16Align_;
    int64_t yOffset = blockSize_ * dataLen64Align_ / DIGIT_TWO;
    if constexpr ((IsSame<yDtype, float8_e4m3_t>::value) || (IsSame<yDtype, float8_e5m2_t>::value)) {
        yOffset = blockSize_ * dataLen32Align_;
    }
    int64_t scaleUbOffset = dataLen32Align_;
    int64_t tmpOffset = dataLen16Align_;

    for (int64_t i = 0; i < calcBlockLoop; i++) {
        if constexpr (calcMode == ModeThree || calcMode == ModeOne) {
            ComputeScaleCuBlas(dataLen, static_cast<uint16_t>(blockSize_), xAddr, mxTmpScaleAddr, tmpAddr);
        } else if constexpr (calcMode == ModeTwo) {
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
    if constexpr (calcMode == ModeThree || calcMode == ModeOne) {
        ComputeScaleCuBlas(dataLen, static_cast<uint16_t>(calcBlockTail), xAddr, mxTmpScaleAddr, tmpAddr);
    } else if constexpr (calcMode == ModeTwo) {
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
        Interleave(
            mxScale[(i - 1) * dataLen32Align_], mxScale[i * dataLen32Align_], tmpScaleLocal[(i - 1) * dataLen32Align_],
            tmpScaleLocal[i * dataLen32Align_], dataLen32Align_);
    }

    mxScaleQueue.template EnQue(mxScale);
    outQueue.template EnQue(y);
    inQueue.template FreeTensor(x);
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype, roundMode, calcMode>::CopyOut(
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
    DataCopyExtParams yCopyOutParams = {
        static_cast<uint16_t>(outBurst), static_cast<uint32_t>(outBlockLen), static_cast<uint32_t>(srcStride),
        static_cast<uint32_t>(dstStride), static_cast<uint32_t>(0)};

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
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype, roundMode, calcMode>::ComputeScaleCuBlas(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScaleAddr,
    __ubuf__ uint16_t* tmpAddr)
{
    uint16_t loopNum = blockCount / 2;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<xDtype> xRegTensor0;
        AscendC::MicroAPI::RegTensor<xDtype> xRegTensor1;
        AscendC::MicroAPI::RegTensor<bfloat16_t> xBF16RegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> absRegTensor0;       // x绝对值
        AscendC::MicroAPI::RegTensor<uint16_t> absRegTensor1;       // x绝对值
        AscendC::MicroAPI::RegTensor<uint16_t> maxRegTensor;        // 绝对值最大值
        AscendC::MicroAPI::RegTensor<uint16_t> mxScale16RegTensor;  // 16位指数
        AscendC::MicroAPI::RegTensor<uint16_t> mxScale16RegTensor0; // 奇数位16位指数
        AscendC::MicroAPI::RegTensor<uint16_t> mxScale16RegTensor1; // 偶数位16位指数
        AscendC::MicroAPI::RegTensor<uint16_t> maxEleRegTensorFP16;
        AscendC::MicroAPI::RegTensor<uint32_t> manAbsFP32RegTensor0; // 尾数与指数加1
        AscendC::MicroAPI::RegTensor<uint32_t> manAbsFP32RegTensor1;
        AscendC::MicroAPI::RegTensor<uint32_t> mxScaleFP32RegTensor0; // 指数
        AscendC::MicroAPI::RegTensor<uint32_t> mxScaleFP32RegTensor1;
        AscendC::MicroAPI::RegTensor<uint16_t> reversedShareExpRegTensor; // 1/scale
        AscendC::MicroAPI::RegTensor<uint8_t> mxScale;

        AscendC::MicroAPI::RegTensor<uint32_t> manForFP32;
        AscendC::MicroAPI::RegTensor<uint16_t> biasRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> maxEleRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> zeroRegTensor;
        AscendC::MicroAPI::RegTensor<uint32_t> invMax;
        AscendC::MicroAPI::RegTensor<uint16_t> absForX;
        AscendC::MicroAPI::RegTensor<uint16_t> nanRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> specialExpRegTensor;
        AscendC::MicroAPI::RegTensor<float> dstTypeMaxReg;

        AscendC::MicroAPI::MaskReg p0;
        AscendC::MicroAPI::MaskReg p1;
        AscendC::MicroAPI::MaskReg p2;
        AscendC::MicroAPI::MaskReg p3;
        AscendC::MicroAPI::MaskReg pregAll8 =
            AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg pregAll16 =
            AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg pregAll32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

        static constexpr AscendC::MicroAPI::CastTrait castTraitZero = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTraitOne = {
            AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        AscendC::MicroAPI::Duplicate(absForX, ABS_FOR_UINT16);
        AscendC::MicroAPI::Duplicate(maxEleRegTensorFP16, HALF_INF);
        AscendC::MicroAPI::Duplicate(manForFP32, MAN_FOR_FP32);
        AscendC::MicroAPI::Duplicate(invMax, INV_DTYPE_MAX);
        AscendC::MicroAPI::Duplicate(dstTypeMaxReg, invDstTypeMax_);
        AscendC::MicroAPI::Duplicate(maxRegTensor, 0);
        AscendC::MicroAPI::Duplicate(maxEleRegTensor, MAX_EXP_FOR_BF16);
        AscendC::MicroAPI::Duplicate(biasRegTensor, BF16_EXP_BIAS);
        AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
        AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);

        for (uint16_t j = 0; j <= loopNum; j++) {
            DataCopy(xRegTensor0, xAddr + j * dataLen16Align_);
            DataCopy(xRegTensor1, xAddr + (blockCount - j - 1) * dataLen16Align_);
            AscendC::MicroAPI::And(
                absRegTensor0, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor0, absForX, pregAll16);
            AscendC::MicroAPI::And(
                absRegTensor1, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor1, absForX, pregAll16);
            AscendC::MicroAPI::Max(absRegTensor0, absRegTensor1, absRegTensor0, pregAll16);
            AscendC::MicroAPI::Max(maxRegTensor, maxRegTensor, absRegTensor0, pregAll16);
        }

        AscendC::MicroAPI::Cast<float, xDtype, castTraitZero>(
            (AscendC::MicroAPI::RegTensor<float>&)manAbsFP32RegTensor0,
            (AscendC::MicroAPI::RegTensor<xDtype>&)maxRegTensor, pregAll16);
        if constexpr (calcMode == ModeOne) {
            AscendC::MicroAPI::Mul(
                (AscendC::MicroAPI::RegTensor<float>&)manAbsFP32RegTensor0,
                (AscendC::MicroAPI::RegTensor<float>&)manAbsFP32RegTensor0,
                (AscendC::MicroAPI::RegTensor<float>&)invMax, pregAll32);
        } else {
            AscendC::MicroAPI::Mul(
                (AscendC::MicroAPI::RegTensor<float>&)manAbsFP32RegTensor0,
                (AscendC::MicroAPI::RegTensor<float>&)manAbsFP32RegTensor0, dstTypeMaxReg, pregAll32);
        }

        AscendC::MicroAPI::ShiftRights(mxScaleFP32RegTensor0, manAbsFP32RegTensor0, SHR_NUM_FOR_FP32, pregAll32);
        AscendC::MicroAPI::And(manAbsFP32RegTensor0, manAbsFP32RegTensor0, manForFP32,
                               pregAll32); // 提取尾数

        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, mxScaleFP32RegTensor0, NUMBER_ZERO, pregAll32);
        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(p0, mxScaleFP32RegTensor0, NUMBER_TWO_FIVE_FOUR, p0);
        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, manAbsFP32RegTensor0, NUMBER_ZERO, p0);

        if constexpr (calcMode == ModeOne) {
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(p1, mxScaleFP32RegTensor0, NUMBER_ZERO, pregAll32);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p1, manAbsFP32RegTensor0, NUMBER_HALF, p1);
            AscendC::MicroAPI::MaskXor(p0, p0, p1, pregAll32);
        }

        AscendC::MicroAPI::Adds(manAbsFP32RegTensor0, mxScaleFP32RegTensor0, 1, p0);
        AscendC::MicroAPI::Select(mxScaleFP32RegTensor0, manAbsFP32RegTensor0, mxScaleFP32RegTensor0, p0);

        AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
            (AscendC::MicroAPI::RegTensor<uint16_t>&)mxScale16RegTensor0, mxScaleFP32RegTensor0);

        AscendC::MicroAPI::Cast<float, xDtype, castTraitOne>(
            (AscendC::MicroAPI::RegTensor<float>&)manAbsFP32RegTensor1,
            (AscendC::MicroAPI::RegTensor<xDtype>&)maxRegTensor, pregAll16);

        if constexpr (calcMode == ModeOne) {
            AscendC::MicroAPI::Mul(
                (AscendC::MicroAPI::RegTensor<float>&)manAbsFP32RegTensor1,
                (AscendC::MicroAPI::RegTensor<float>&)manAbsFP32RegTensor1,
                (AscendC::MicroAPI::RegTensor<float>&)invMax, pregAll32);
        } else {
            AscendC::MicroAPI::Mul(
                (AscendC::MicroAPI::RegTensor<float>&)manAbsFP32RegTensor1,
                (AscendC::MicroAPI::RegTensor<float>&)manAbsFP32RegTensor1, dstTypeMaxReg, pregAll32);
        }

        AscendC::MicroAPI::ShiftRights(mxScaleFP32RegTensor1, manAbsFP32RegTensor1, SHR_NUM_FOR_FP32, pregAll32);
        AscendC::MicroAPI::And(manAbsFP32RegTensor1, manAbsFP32RegTensor1, manForFP32,
                               pregAll32); // 提取尾数

        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p2, mxScaleFP32RegTensor1, NUMBER_ZERO, pregAll32);
        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(p2, mxScaleFP32RegTensor1, NUMBER_TWO_FIVE_FOUR, p2);
        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p2, manAbsFP32RegTensor1, NUMBER_ZERO, p2);

        if constexpr (calcMode == ModeOne) {
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(p3, mxScaleFP32RegTensor1, NUMBER_ZERO, pregAll32);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p3, manAbsFP32RegTensor1, NUMBER_HALF, p3);
            AscendC::MicroAPI::MaskXor(p2, p3, p2, pregAll32);
        }

        AscendC::MicroAPI::Adds(manAbsFP32RegTensor1, mxScaleFP32RegTensor1, 1, p2);
        AscendC::MicroAPI::Select(mxScaleFP32RegTensor1, manAbsFP32RegTensor1, mxScaleFP32RegTensor1, p2);

        AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
            (AscendC::MicroAPI::RegTensor<uint16_t>&)mxScale16RegTensor1, mxScaleFP32RegTensor1);

        AscendC::MicroAPI::Interleave(
            mxScale16RegTensor0, mxScale16RegTensor1, mxScale16RegTensor0, mxScale16RegTensor1);
        AscendC::MicroAPI::ShiftLefts(mxScale16RegTensor, mxScale16RegTensor0, SHR_NUM_FOR_BF16, pregAll16);
        AscendC::MicroAPI::Pack<uint8_t, uint16_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
            (AscendC::MicroAPI::RegTensor<uint8_t>&)mxScale, mxScale16RegTensor0);

        DataCopy(mxScaleAddr, mxScale, pregAll8);

        // 求1/scale
        AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(p0, mxScale16RegTensor, maxEleRegTensor, pregAll16);
        AscendC::MicroAPI::Compare<uint16_t, CMPMODE::EQ>(p1, mxScale16RegTensor, biasRegTensor, pregAll16);
        AscendC::MicroAPI::Sub(reversedShareExpRegTensor, biasRegTensor, mxScale16RegTensor, pregAll16);
        AscendC::MicroAPI::Select<uint16_t>(reversedShareExpRegTensor, reversedShareExpRegTensor, nanRegTensor, p0);
        AscendC::MicroAPI::Select<uint16_t>(
            reversedShareExpRegTensor, specialExpRegTensor, reversedShareExpRegTensor, p1);
        DataCopy(tmpAddr, reversedShareExpRegTensor, pregAll16);
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype, roundMode, calcMode>::ComputeScaleCuBlasOptimize(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScaleAddr,
    __ubuf__ uint16_t* tmpAddr)
{
    uint16_t loopNum = blockCount / 2;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<xDtype> xRegTensor0;
        AscendC::MicroAPI::RegTensor<xDtype> xRegTensor1;
        AscendC::MicroAPI::RegTensor<uint16_t> expMaxRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> mxScaleRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> reversedShareExpRegTensor;
        AscendC::MicroAPI::RegTensor<uint8_t> mxScale;
        AscendC::MicroAPI::RegTensor<uint16_t> absRegTensor0; // x绝对值
        AscendC::MicroAPI::RegTensor<uint16_t> absRegTensor1; // x绝对值
        AscendC::MicroAPI::RegTensor<uint16_t> maxRegTensor;  // 绝对值最大值

        AscendC::MicroAPI::RegTensor<uint16_t> specialExpRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> biasRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> zeroRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> nanRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> maxEleRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> dtypeYMaxExp;
        AscendC::MicroAPI::RegTensor<uint16_t> subNumForScale;
        AscendC::MicroAPI::RegTensor<uint16_t> fp8NanRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> absForX;

        AscendC::MicroAPI::MaskReg infMask;
        AscendC::MicroAPI::MaskReg zeroMask;
        AscendC::MicroAPI::MaskReg invalidDataMask;
        AscendC::MicroAPI::MaskReg pregAll8 =
            AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg pregAll16 =
            AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();

        static constexpr AscendC::MicroAPI::CastTrait castTraitCublsHalf2Bf16 = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        AscendC::MicroAPI::Duplicate(maxEleRegTensor, MAX_EXP_FOR_BF16);
        AscendC::MicroAPI::Duplicate(subNumForScale, subNumForScale_);
        AscendC::MicroAPI::Duplicate(absForX, ABS_FOR_UINT16);
        AscendC::MicroAPI::Duplicate(dtypeYMaxExp, DTYPE_Y_MAX_EXP);
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        AscendC::MicroAPI::Duplicate(biasRegTensor, BF16_EXP_BIAS);
        AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
        AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
        AscendC::MicroAPI::Duplicate(expMaxRegTensor, 0);
        AscendC::MicroAPI::Duplicate(maxRegTensor, 0);

        for (uint16_t j = 0; j <= loopNum; j++) {
            DataCopy(xRegTensor0, xAddr + j * dataLen16Align_);
            DataCopy(xRegTensor1, xAddr + (blockCount - j - 1) * dataLen16Align_);
            AscendC::MicroAPI::And(
                absRegTensor0, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor0, absForX, pregAll16);
            AscendC::MicroAPI::And(
                absRegTensor1, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor1, absForX, pregAll16);
            if constexpr (IsSame<xDtype, half>::value) {
                AscendC::MicroAPI::Cast<bfloat16_t, xDtype, castTraitCublsHalf2Bf16>(
                    (AscendC::MicroAPI::RegTensor<bfloat16_t>&)absRegTensor0,
                    (AscendC::MicroAPI::RegTensor<float16_t>&)absRegTensor0, pregAll16);
                AscendC::MicroAPI::Cast<bfloat16_t, xDtype, castTraitCublsHalf2Bf16>(
                    (AscendC::MicroAPI::RegTensor<bfloat16_t>&)absRegTensor1,
                    (AscendC::MicroAPI::RegTensor<float16_t>&)absRegTensor1, pregAll16);
            }
            AscendC::MicroAPI::Max(absRegTensor1, absRegTensor1, absRegTensor0, pregAll16);
            AscendC::MicroAPI::Max(maxRegTensor, maxRegTensor, absRegTensor1, pregAll16);
        }
        AscendC::MicroAPI::And(expMaxRegTensor, maxRegTensor, maxEleRegTensor, pregAll16);

        AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxRegTensor, maxEleRegTensor, pregAll16);
        AscendC::MicroAPI::Compare<uint16_t, CMPMODE::LT>(invalidDataMask, expMaxRegTensor, dtypeYMaxExp, pregAll16);

        AscendC::MicroAPI::Sub(expMaxRegTensor, maxRegTensor, subNumForScale, pregAll16);

        AscendC::MicroAPI::Select<uint16_t>(expMaxRegTensor, zeroRegTensor, expMaxRegTensor, invalidDataMask);
        AscendC::MicroAPI::ShiftRights(mxScaleRegTensor, expMaxRegTensor, SHR_NUM_FOR_BF16, pregAll16);
        AscendC::MicroAPI::Select<uint16_t>(mxScaleRegTensor, mxScaleRegTensor, fp8NanRegTensor, infMask);

        AscendC::MicroAPI::Pack<uint8_t, uint16_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
            (AscendC::MicroAPI::RegTensor<uint8_t>&)mxScale, mxScaleRegTensor);
        DataCopy(mxScaleAddr, mxScale, pregAll8);
        // 求1/scale
        AscendC::MicroAPI::And(expMaxRegTensor, expMaxRegTensor, maxEleRegTensor, pregAll16);
        AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxRegTensor, zeroRegTensor, pregAll16);
        AscendC::MicroAPI::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMaxRegTensor, biasRegTensor, pregAll16);
        AscendC::MicroAPI::Sub(reversedShareExpRegTensor, biasRegTensor, expMaxRegTensor, pregAll16);
        AscendC::MicroAPI::Select<uint16_t>(
            reversedShareExpRegTensor, reversedShareExpRegTensor, nanRegTensor, infMask);
        AscendC::MicroAPI::Select<uint16_t>(
            reversedShareExpRegTensor, reversedShareExpRegTensor, zeroRegTensor, zeroMask);
        AscendC::MicroAPI::Select<uint16_t>(
            reversedShareExpRegTensor, specialExpRegTensor, reversedShareExpRegTensor, invalidDataMask);
        DataCopy(tmpAddr, reversedShareExpRegTensor, pregAll16);
    }
}
template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype, roundMode, calcMode>::ComputeScaleOcp(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScaleAddr,
    __ubuf__ uint16_t* tmpAddr)
{
    uint16_t loopNum = blockCount / 2;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<xDtype> xRegTensor;
        AscendC::MicroAPI::RegTensor<xDtype> xRegTensor1;
        AscendC::MicroAPI::RegTensor<bfloat16_t> xBF16RegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> expRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> expRegTensor1;
        AscendC::MicroAPI::RegTensor<uint16_t> expMaxRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> mxScaleRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> reversedShareExpRegTensor;
        AscendC::MicroAPI::RegTensor<uint8_t> mxScale;
        AscendC::MicroAPI::RegTensor<uint16_t> absRegTensor; // x绝对值
        AscendC::MicroAPI::RegTensor<uint16_t> maxRegTensor; // 绝对值最大值

        AscendC::MicroAPI::RegTensor<uint16_t> specialExpRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> biasRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> zeroRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> nanRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> maxEleRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> maxEleRegTensorFP16;
        AscendC::MicroAPI::RegTensor<uint16_t> dtypeYMaxExp;
        AscendC::MicroAPI::RegTensor<uint16_t> subNumForScale;
        AscendC::MicroAPI::RegTensor<uint16_t> fp8NanRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> absForX;

        AscendC::MicroAPI::MaskReg infMask;
        AscendC::MicroAPI::MaskReg zeroMask;
        AscendC::MicroAPI::MaskReg invalidDataMask;
        AscendC::MicroAPI::MaskReg pregAll8 =
            AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg pregAll16 =
            AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();

        static constexpr AscendC::MicroAPI::CastTrait castTraitOcpHalf2Bf16 = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

        AscendC::MicroAPI::Duplicate(maxEleRegTensor, MAX_EXP_FOR_BF16);
        AscendC::MicroAPI::Duplicate(subNumForScale, subNumForScale_);
        AscendC::MicroAPI::Duplicate(absForX, ABS_FOR_UINT16);
        AscendC::MicroAPI::Duplicate(maxEleRegTensorFP16, HALF_INF);
        AscendC::MicroAPI::Duplicate(dtypeYMaxExp, DTYPE_Y_MAX_EXP);
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        AscendC::MicroAPI::Duplicate(biasRegTensor, BF16_EXP_BIAS);
        AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
        AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
        AscendC::MicroAPI::Duplicate(expMaxRegTensor, 0);
        AscendC::MicroAPI::Duplicate(maxRegTensor, 0);

        for (uint16_t j = 0; j <= loopNum; j++) {
            DataCopy(xRegTensor, xAddr + j * dataLen16Align_);
            DataCopy(xRegTensor1, xAddr + (blockCount - j - 1) * dataLen16Align_);

            if constexpr (IsSame<xDtype, half>::value) {
                AscendC::MicroAPI::Cast<bfloat16_t, xDtype, castTraitOcpHalf2Bf16>(
                    (AscendC::MicroAPI::RegTensor<bfloat16_t>&)expRegTensor,
                    (AscendC::MicroAPI::RegTensor<float16_t>&)xRegTensor, pregAll16);
                AscendC::MicroAPI::And(
                    expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)expRegTensor, maxEleRegTensor, pregAll16);
                AscendC::MicroAPI::Cast<bfloat16_t, xDtype, castTraitOcpHalf2Bf16>(
                    (AscendC::MicroAPI::RegTensor<bfloat16_t>&)expRegTensor1,
                    (AscendC::MicroAPI::RegTensor<float16_t>&)xRegTensor1, pregAll16);
                AscendC::MicroAPI::And(
                    expRegTensor1, (AscendC::MicroAPI::RegTensor<uint16_t>&)expRegTensor1, maxEleRegTensor, pregAll16);
            } else {
                AscendC::MicroAPI::And(
                    expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor, maxEleRegTensor, pregAll16);
                AscendC::MicroAPI::And(
                    expRegTensor1, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor1, maxEleRegTensor, pregAll16);
            }
            AscendC::MicroAPI::Max(expRegTensor, expRegTensor, expRegTensor1, pregAll16);
            AscendC::MicroAPI::Max(expMaxRegTensor, expMaxRegTensor, expRegTensor, pregAll16);
        }
        AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxRegTensor, maxEleRegTensor, pregAll16);
        AscendC::MicroAPI::Compare<uint16_t, CMPMODE::LT>(invalidDataMask, expMaxRegTensor, dtypeYMaxExp, pregAll16);

        AscendC::MicroAPI::Sub(expMaxRegTensor, expMaxRegTensor, subNumForScale, pregAll16);

        AscendC::MicroAPI::Select<uint16_t>(expMaxRegTensor, zeroRegTensor, expMaxRegTensor, invalidDataMask);
        AscendC::MicroAPI::ShiftRights(mxScaleRegTensor, expMaxRegTensor, SHR_NUM_FOR_BF16, pregAll16);
        AscendC::MicroAPI::Select<uint16_t>(mxScaleRegTensor, mxScaleRegTensor, fp8NanRegTensor, infMask);

        DataCopy<uint8_t, Reg::StoreDist::DIST_PACK_B16>(
            mxScaleAddr, (AscendC::MicroAPI::RegTensor<uint8_t>&)mxScaleRegTensor, pregAll8);

        // 求1/scale
        AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxRegTensor, zeroRegTensor, pregAll16);

        AscendC::MicroAPI::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMaxRegTensor, biasRegTensor, pregAll16);
        AscendC::MicroAPI::Sub(reversedShareExpRegTensor, biasRegTensor, expMaxRegTensor, pregAll16);
        AscendC::MicroAPI::Select<uint16_t>(
            reversedShareExpRegTensor, reversedShareExpRegTensor, nanRegTensor, infMask);
        AscendC::MicroAPI::Select<uint16_t>(
            reversedShareExpRegTensor, reversedShareExpRegTensor, zeroRegTensor, zeroMask);
        AscendC::MicroAPI::Select<uint16_t>(
            reversedShareExpRegTensor, specialExpRegTensor, reversedShareExpRegTensor, invalidDataMask);
        DataCopy(tmpAddr, reversedShareExpRegTensor, pregAll16);
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype, roundMode, calcMode>::ComputeYVf(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* tmpAddr, __ubuf__ uint8_t* yAddr)
{
    if constexpr (IsSame<xDtype, half>::value) {
        ComputeYFromHalf(dataLen, blockCount, xAddr, tmpAddr, yAddr);
    } else {
        ComputeYFromBf16(dataLen, blockCount, xAddr, tmpAddr, yAddr);
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype, roundMode, calcMode>::ComputeYFromHalf(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* tmpAddr, __ubuf__ uint8_t* yAddr)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<xDtype> xRegTensor;
        AscendC::MicroAPI::RegTensor<bfloat16_t> xBF16RegTensor0;
        AscendC::MicroAPI::RegTensor<bfloat16_t> xBF16RegTensor1;
        AscendC::MicroAPI::RegTensor<float> xFP32RegTensor0;
        AscendC::MicroAPI::RegTensor<float> xFP32RegTensor1;
        AscendC::MicroAPI::RegTensor<uint16_t> reversedShareExpRegTensor;
        AscendC::MicroAPI::RegTensor<float> reversedShareExpFP32RegTensor0;
        AscendC::MicroAPI::RegTensor<float> reversedShareExpFP32RegTensor1;
        AscendC::MicroAPI::RegTensor<yDtype> yZeroFP8;
        AscendC::MicroAPI::RegTensor<yDtype> yOneFP8;
        AscendC::MicroAPI::RegTensor<yDtype> yZeroFP4;

        AscendC::MicroAPI::MaskReg pregAll8 =
            AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg pregAll16 =
            AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg pregAll32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

        static constexpr AscendC::MicroAPI::CastTrait castTraitBf16toFp4 = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, roundMode};
        static constexpr AscendC::MicroAPI::CastTrait castTraitXdtypetoFp32Zero = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTraitXdtypetoFp32One = {
            AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTraitFp32toBf16 = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, roundMode};
        static constexpr AscendC::MicroAPI::CastTrait castTraitFp32toYdtype = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        AscendC::MicroAPI::DataCopy<uint16_t, MicroAPI::LoadDist::DIST_NORM>(reversedShareExpRegTensor, tmpAddr);

        AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitXdtypetoFp32Zero>(
            reversedShareExpFP32RegTensor0, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)reversedShareExpRegTensor,
            pregAll16);
        AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitXdtypetoFp32One>(
            reversedShareExpFP32RegTensor1, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)reversedShareExpRegTensor,
            pregAll16);

        for (uint16_t j = 0; j < blockCount; j++) {
            AscendC::MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_NORM>(xRegTensor, xAddr + j * dataLen16Align_);

            AscendC::MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(xFP32RegTensor0, xRegTensor, pregAll16);
            AscendC::MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32One>(xFP32RegTensor1, xRegTensor, pregAll16);

            AscendC::MicroAPI::Mul(xFP32RegTensor0, xFP32RegTensor0, reversedShareExpFP32RegTensor0, pregAll32);
            AscendC::MicroAPI::Mul(xFP32RegTensor1, xFP32RegTensor1, reversedShareExpFP32RegTensor1, pregAll32);

            if constexpr (IsSame<yDtype, fp4x2_e2m1_t>::value || IsSame<yDtype, fp4x2_e1m2_t>::value) {
                ComputeFP4FromHalf(xFP32RegTensor0);
                ComputeFP4FromHalf(xFP32RegTensor1);
                AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBf16>(
                    (AscendC::MicroAPI::RegTensor<bfloat16_t>&)xBF16RegTensor0, xFP32RegTensor0, pregAll32);
                AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)xBF16RegTensor0,
                    (AscendC::MicroAPI::RegTensor<uint32_t>&)xBF16RegTensor0);

                AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBf16>(
                    (AscendC::MicroAPI::RegTensor<bfloat16_t>&)xBF16RegTensor1, xFP32RegTensor1, pregAll32);
                AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)xBF16RegTensor1,
                    (AscendC::MicroAPI::RegTensor<uint32_t>&)xBF16RegTensor1);

                AscendC::MicroAPI::Interleave(xBF16RegTensor0, xBF16RegTensor1, xBF16RegTensor0, xBF16RegTensor1);

                AscendC::MicroAPI::Cast<yDtype, bfloat16_t, castTraitBf16toFp4>(
                    yZeroFP4, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)xBF16RegTensor0, pregAll16);

                DataCopy<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                    yAddr + (j * dataLen64Align_ / DIGIT_TWO), (AscendC::MicroAPI::RegTensor<uint8_t>&)yZeroFP4,
                    pregAll8);
            } else {
                AscendC::MicroAPI::Cast<yDtype, float, castTraitFp32toYdtype>(
                    yZeroFP8, (AscendC::MicroAPI::RegTensor<float>&)xFP32RegTensor0, pregAll32);
                AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)yZeroFP8,
                    (AscendC::MicroAPI::RegTensor<uint32_t>&)yZeroFP8);

                AscendC::MicroAPI::Cast<yDtype, float, castTraitFp32toYdtype>(
                    yOneFP8, (AscendC::MicroAPI::RegTensor<float>&)xFP32RegTensor1, pregAll32);
                AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)yOneFP8, (AscendC::MicroAPI::RegTensor<uint32_t>&)yOneFP8);

                AscendC::MicroAPI::Interleave(
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)yZeroFP8, (AscendC::MicroAPI::RegTensor<uint16_t>&)yOneFP8,
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)yZeroFP8,
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)yOneFP8);

                DataCopy<uint8_t, Reg::StoreDist::DIST_PACK_B16>(
                    yAddr + (j * dataLen32Align_), (AscendC::MicroAPI::RegTensor<uint8_t>&)yZeroFP8, pregAll8);
            }
        }
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype, roundMode, calcMode>::ComputeFP4FromHalf(
    AscendC::MicroAPI::RegTensor<float>& Reg)
{
    AscendC::MicroAPI::MaskReg pregAll32 =
        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg zeroMask;
    AscendC::MicroAPI::MaskReg specialMask;
    AscendC::MicroAPI::MaskReg negInfMask;

    AscendC::MicroAPI::RegTensor<int32_t> negZeroRegTensor;
    AscendC::MicroAPI::RegTensor<int32_t> maxExpFP32RegTensor;
    AscendC::MicroAPI::RegTensor<int32_t> expFP32RegTensor0;
    AscendC::MicroAPI::RegTensor<int32_t> expFP32RegTensor1;

    AscendC::MicroAPI::Duplicate(negZeroRegTensor, NEG_ZERO);

    AscendC::MicroAPI::Compare<int32_t, CMPMODE::EQ>(
        /*negzero*/ negInfMask, (AscendC::MicroAPI::RegTensor<int32_t>&)Reg, negZeroRegTensor, pregAll32);
    if constexpr (IsSame<yDtype, fp4x2_e1m2_t>::value) {
        AscendC::MicroAPI::Muls(Reg, Reg, FOUR, pregAll32);
        AscendC::MicroAPI::CompareScalar<float, CMPMODE::LT>(/*negvalue*/ specialMask, Reg, 0, pregAll32);
        AscendC::MicroAPI::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        AscendC::MicroAPI::Muls(Reg, Reg, ONE_FOURTH, pregAll32);
    } else { // fp4x2_e2m1
        AscendC::MicroAPI::Duplicate(maxExpFP32RegTensor, MAX_EXP_FOR_FP32);
        AscendC::MicroAPI::And(
            expFP32RegTensor0, (AscendC::MicroAPI::RegTensor<int32_t>&)Reg, maxExpFP32RegTensor, pregAll32);
        AscendC::MicroAPI::ShiftRights(expFP32RegTensor0, expFP32RegTensor0, SHR_NUM_FOR_FP32, pregAll32);
        AscendC::MicroAPI::Adds(expFP32RegTensor0, expFP32RegTensor0, FP32_BIAS_NEG, pregAll32);
        AscendC::MicroAPI::Maxs(expFP32RegTensor0, expFP32RegTensor0, 0, pregAll32);
        AscendC::MicroAPI::Adds(expFP32RegTensor0, expFP32RegTensor0, NEG_ONE, pregAll32);
        AscendC::MicroAPI::Muls(expFP32RegTensor1, expFP32RegTensor0, NEG_ONE, pregAll32);
        AscendC::MicroAPI::Adds(expFP32RegTensor1, expFP32RegTensor1, FP32_BIAS, pregAll32);
        AscendC::MicroAPI::ShiftLefts(expFP32RegTensor1, expFP32RegTensor1, SHR_NUM_FOR_FP32, pregAll32);

        AscendC::MicroAPI::Mul(Reg, Reg, (AscendC::MicroAPI::RegTensor<float>&)expFP32RegTensor1, pregAll32);
        AscendC::MicroAPI::Adds(expFP32RegTensor0, expFP32RegTensor0, FP32_BIAS, pregAll32);
        AscendC::MicroAPI::ShiftLefts(expFP32RegTensor0, expFP32RegTensor0, SHR_NUM_FOR_FP32, pregAll32);
        AscendC::MicroAPI::CompareScalar<float, CMPMODE::LT>(/*negvalue*/ specialMask, Reg, 0, pregAll32);
        AscendC::MicroAPI::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        AscendC::MicroAPI::Mul(Reg, Reg, (AscendC::MicroAPI::RegTensor<float>&)expFP32RegTensor0, pregAll32);
    }
    AscendC::MicroAPI::CompareScalar<float, CMPMODE::EQ>(zeroMask, Reg, 0, pregAll32);
    AscendC::MicroAPI::MaskAnd(zeroMask, specialMask, zeroMask, pregAll32);
    AscendC::MicroAPI::MaskOr(zeroMask, negInfMask, zeroMask, pregAll32);
    AscendC::MicroAPI::Select<int32_t>(
        (AscendC::MicroAPI::RegTensor<int32_t>&)Reg, negZeroRegTensor, (AscendC::MicroAPI::RegTensor<int32_t>&)Reg,
        zeroMask);
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype, roundMode, calcMode>::ComputeYFromBf16(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* tmpAddr, __ubuf__ uint8_t* yAddr)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<xDtype> xRegTensor;
        AscendC::MicroAPI::RegTensor<bfloat16_t> valueRegTensor;
        AscendC::MicroAPI::RegTensor<float> xFP32RegTensor0;
        AscendC::MicroAPI::RegTensor<float> xFP32RegTensor1;
        AscendC::MicroAPI::RegTensor<uint16_t> reversedShareExpRegTensor;
        AscendC::MicroAPI::RegTensor<float> reversedShareExpFP32RegTensor0;
        AscendC::MicroAPI::RegTensor<float> reversedShareExpFP32RegTensor1;
        AscendC::MicroAPI::RegTensor<yDtype> yZeroFP8;
        AscendC::MicroAPI::RegTensor<yDtype> yOneFP8;
        AscendC::MicroAPI::RegTensor<yDtype> yZeroFP4;

        AscendC::MicroAPI::MaskReg pregAll8 =
            AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg pregAll16 =
            AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg pregAll32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

        static constexpr AscendC::MicroAPI::CastTrait castTraitBf16toFp4 = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, roundMode};
        static constexpr AscendC::MicroAPI::CastTrait castTraitXdtypetoFp32Zero = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTraitXdtypetoFp32One = {
            AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTraitFp32toYdtype = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        AscendC::MicroAPI::DataCopy<uint16_t, MicroAPI::LoadDist::DIST_NORM>(reversedShareExpRegTensor, tmpAddr);

        AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitXdtypetoFp32Zero>(
            reversedShareExpFP32RegTensor0, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)reversedShareExpRegTensor,
            pregAll16);
        AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitXdtypetoFp32One>(
            reversedShareExpFP32RegTensor1, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)reversedShareExpRegTensor,
            pregAll16);

        for (uint16_t j = 0; j < blockCount; j++) {
            AscendC::MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_NORM>(xRegTensor, xAddr + j * dataLen16Align_);

            if constexpr (IsSame<yDtype, fp4x2_e2m1_t>::value || IsSame<yDtype, fp4x2_e1m2_t>::value) {
                AscendC::MicroAPI::Mul(
                    valueRegTensor, xRegTensor, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)reversedShareExpRegTensor,
                    pregAll16);
                AscendC::MicroAPI::Cast<yDtype, bfloat16_t, castTraitBf16toFp4>(yZeroFP4, valueRegTensor, pregAll16);

                DataCopy<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                    yAddr + (j * dataLen64Align_ / DIGIT_TWO), (AscendC::MicroAPI::RegTensor<uint8_t>&)yZeroFP4,
                    pregAll8);
            } else {
                AscendC::MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(
                    xFP32RegTensor0, xRegTensor, pregAll16);
                AscendC::MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32One>(
                    xFP32RegTensor1, xRegTensor, pregAll16);

                AscendC::MicroAPI::Mul(xFP32RegTensor0, xFP32RegTensor0, reversedShareExpFP32RegTensor0, pregAll32);
                AscendC::MicroAPI::Mul(xFP32RegTensor1, xFP32RegTensor1, reversedShareExpFP32RegTensor1, pregAll32);
                AscendC::MicroAPI::Cast<yDtype, float, castTraitFp32toYdtype>(
                    yZeroFP8, (AscendC::MicroAPI::RegTensor<float>&)xFP32RegTensor0, pregAll32);
                AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)yZeroFP8,
                    (AscendC::MicroAPI::RegTensor<uint32_t>&)yZeroFP8);

                AscendC::MicroAPI::Cast<yDtype, float, castTraitFp32toYdtype>(
                    yOneFP8, (AscendC::MicroAPI::RegTensor<float>&)xFP32RegTensor1, pregAll32);
                AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)yOneFP8, (AscendC::MicroAPI::RegTensor<uint32_t>&)yOneFP8);

                AscendC::MicroAPI::Interleave(
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)yZeroFP8, (AscendC::MicroAPI::RegTensor<uint16_t>&)yOneFP8,
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)yZeroFP8,
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)yOneFP8);

                DataCopy<uint8_t, Reg::StoreDist::DIST_PACK_B16>(
                    yAddr + (j * dataLen32Align_), (AscendC::MicroAPI::RegTensor<uint8_t>&)yZeroFP8, pregAll8);
            }
        }
    }
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype, roundMode, calcMode>::Init(
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

    int64_t inBufferSize_ = ops::CeilAlign(
        ubRowLen_ * ubRowCount_ * static_cast<int64_t>(sizeof(xDtype)),
        static_cast<int64_t>(Ops::Base::GetUbBlockSize()));

    int64_t mxScaleBufferSize_ =
        ops::CeilAlign(ubRowLen_ * scaleUbRowCount, static_cast<int64_t>(Ops::Base::GetUbBlockSize()));

    int64_t outBufferSize_ = ops::CeilAlign(
        static_cast<int64_t>(ubRowLen_ * ubRowCount_), static_cast<int64_t>(Ops::Base::GetUbBlockSize()));

    int64_t tmpBufferSize_ = ops::CeilAlign(
        ubRowLen_ * scaleUbRowCount * static_cast<int64_t>(sizeof(xDtype)),
        static_cast<int64_t>(Ops::Base::GetUbBlockSize()));

    int64_t tmpScaleBufferSize_ = mxScaleBufferSize_;

    pipe_->InitBuffer(inQueue, DB_BUFFER, inBufferSize_);
    pipe_->InitBuffer(mxScaleQueue, DB_BUFFER, mxScaleBufferSize_);
    pipe_->InitBuffer(outQueue, DB_BUFFER, outBufferSize_);
    pipe_->InitBuffer(tmpBuf, tmpBufferSize_);
    pipe_->InitBuffer(tmpScaleBuf, tmpScaleBufferSize_);
}

template <typename xDtype, typename yDtype, RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype, roundMode, calcMode>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    for (int64_t i = 0; i < loopPerCore_; i++) {
        // init runtime ub params
        InitCalcParams(i);
        ProcessOneLoop(i);
    }
}

} // namespace DynamicMxQuant
#endif // DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_FP8_H
