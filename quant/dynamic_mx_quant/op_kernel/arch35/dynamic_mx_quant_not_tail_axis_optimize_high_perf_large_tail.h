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

namespace DynamicMxQuant {
using namespace AscendC;

template <typename xDtype, typename yDtype>
class DynamicMxQuantHP2000 {
public:
    __aicore__ inline DynamicMxQuantHP2000(const DynamicMxQuant4OptimizeTilingData* tilingData, TPipe* pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR mxScale);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitParams();
    __aicore__ inline void InitCalcParams(int64_t index);
    template <const bool scaleAlg, RoundMode roundMode>
    __aicore__ inline void ProcessOneLoop(int64_t index);
    __aicore__ inline void CopyOut(int64_t yOffset, int64_t scaleOutOffset, int64_t blockCount, int64_t dataLen);
    __aicore__ inline void CopyIn(int64_t offset, int64_t blockCount, int64_t dataLen);
    template <const bool scaleAlg, RoundMode roundMode>
    __aicore__ inline void ComputeAll(int64_t blockCount, int64_t dataLen);
    __aicore__ inline void ComputeScaleCuBlas(
        uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScaleAddr,
        __ubuf__ uint16_t* tmpAddr);
    __aicore__ inline void ComputeScaleOcp(
        uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScaleAddr,
        __ubuf__ uint16_t* tmpAddr);
    template <RoundMode roundMode>
    __aicore__ inline void ComputeYVf(
        uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* tmpAddr,
        __ubuf__ uint8_t* yAddr);
    template <RoundMode roundMode>
    __aicore__ inline void ComputeYFromHalf(
        uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* tmpAddr,
        __ubuf__ uint8_t* yAddr);
    template <RoundMode roundMode>
    __aicore__ inline void ComputeYFromBf16(
        uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* tmpAddr,
        __ubuf__ uint8_t* yAddr);
    template <RoundMode roundMode>
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
    int64_t blockOffset_ = 0;
    int64_t loopPerCore_ = 0;
    int64_t ubRowLen_ = 0;
    int64_t ubRowLenTail_ = 0;
    int64_t ubRowCount_ = 0;
    int64_t ubRowCountTail_ = 0;
    int64_t mGroupSizeInOneN_ = 0;
    int64_t roundMode_ = 0;
    int64_t blockCountPerPage_ = 0;
    uint32_t INV_DTYPE_MAX = 0;
    uint16_t DTYPE_Y_MAX_EXP = 0;
    uint16_t blockSize_ = 0;

    // runtime varible
    int64_t calcRow_ = 0;
    int64_t calcCol_ = 0;
    int64_t ubOffset_ = 0;
    int64_t scaleOffset_ = 0;
    int64_t calcBlockLoop_ = 0;
    int64_t calcBlockTail_ = 0;
    int64_t dataLen16Align_ = 0;
    int64_t dataLen32Align_ = 0;
    int64_t dataLen64Align_ = 0;
    bool scaleNeedsPad_ = false;
    bool scaleAlg_ = false;
};

template <typename xDtype, typename yDtype>
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype>::InitParams()
{
    blockIdx_ = GetBlockIdx();
    int64_t headCoreNum = tilingData_->totalTaskNum <= tilingData_->totalCoreNum ?
                              tilingData_->totalTaskNum :
                              tilingData_->totalTaskNum % tilingData_->usedCoreNum;
    if (blockIdx_ < headCoreNum) {
        loopPerCore_ = tilingData_->rowPerHeadCore;
        blockOffset_ = blockIdx_ * loopPerCore_;
    } else {
        loopPerCore_ = tilingData_->rowPerTailCore;
        blockOffset_ = headCoreNum * tilingData_->rowPerHeadCore + (blockIdx_ - headCoreNum) * loopPerCore_;
    }

    blockSize_ = static_cast<uint16_t>(tilingData_->blockSize);

    // 一次vf计算的行长度，如果是tail后续处理
    ubRowLen_ = 128;
    ubRowLenTail_ = tilingData_->postAxisSize % ubRowLen_ == 0 ? ubRowLen_ : tilingData_->postAxisSize % ubRowLen_;

    // 一次UB计算的行数，如果是tail后续处理
    ubRowCount_ = tilingData_->groupPerUb * DIGIT_TWO * tilingData_->blockSize;
    ubRowCountTail_ =
        tilingData_->quantAxisSize % ubRowCount_ == 0 ? ubRowCount_ : tilingData_->quantAxisSize % ubRowCount_;

    blockCountPerPage_ = tilingData_->nAlignBlockSize * ((tilingData_->quantAxisSize + ubRowCount_ - 1) / ubRowCount_);

    mGroupSizeInOneN_ = (tilingData_->quantAxisSize + DIGIT_TWO * tilingData_->blockSize - 1) /
                        (DIGIT_TWO * tilingData_->blockSize) * DIGIT_TWO; // 每页的scale行数

    if constexpr (IsSame<DTYPE_Y, fp8_e4m3fn_t>::value) {
        DTYPE_Y_MAX_EXP = FP8_E4M3_MAX_EXP;
        INV_DTYPE_MAX = FP8_E4M3_MAX;
    } else if constexpr (IsSame<DTYPE_Y, fp8_e5m2_t>::value) {
        DTYPE_Y_MAX_EXP = FP8_E5M2_MAX_EXP;
        INV_DTYPE_MAX = FP8_E5M2_MAX;
    } else if constexpr (IsSame<DTYPE_Y, fp4x2_e2m1_t>::value) {
        DTYPE_Y_MAX_EXP = FP4_E2M1_BF16_MAX_EXP;
    }

    roundMode_ = tilingData_->roundMode;
    scaleAlg_ =
        (IsSame<DTYPE_Y, fp4x2_e2m1_t>::value || IsSame<DTYPE_Y, fp4x2_e1m2_t>::value) ? false : tilingData_->scaleAlg;
}

template <typename xDtype, typename yDtype>
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype>::InitCalcParams(int64_t index)
{
    calcRow_ = ((blockOffset_ + index) % tilingData_->nAlignBlockSize == tilingData_->nAlignBlockSize - 1) ?
                   ubRowLenTail_ :
                   ubRowLen_; // 本次ub计算的列数
    calcCol_ = ((blockOffset_ + index) / tilingData_->nAlignBlockSize % tilingData_->mAlignBlockSize) ==
                       (tilingData_->mAlignBlockSize - 1) ?
                   ubRowCountTail_ :
                   ubRowCount_; // 本次ub计算的行数

    ubOffset_ = (blockOffset_ + index) / blockCountPerPage_ * tilingData_->quantAxisSize * tilingData_->postAxisSize +
                (blockOffset_ + index) % blockCountPerPage_ / tilingData_->nAlignBlockSize * ubRowCount_ *
                    tilingData_->postAxisSize +
                (blockOffset_ + index) % blockCountPerPage_ % tilingData_->nAlignBlockSize * ubRowLen_;
    scaleOffset_ = (blockOffset_ + index) / blockCountPerPage_ * mGroupSizeInOneN_ * tilingData_->postAxisSize +
                   (blockOffset_ + index) % blockCountPerPage_ / tilingData_->nAlignBlockSize *
                       tilingData_->groupPerUb * DIGIT_TWO * tilingData_->postAxisSize +
                   (blockOffset_ + index) % blockCountPerPage_ % tilingData_->nAlignBlockSize * ubRowLen_ * DIGIT_TWO;

    calcBlockLoop_ = (calcCol_ + tilingData_->blockSize - 1) / tilingData_->blockSize;
    calcBlockTail_ = calcCol_ % tilingData_->blockSize;

    scaleNeedsPad_ = tilingData_->quantAxisIsOdd &&
                     ((blockOffset_ + index) / tilingData_->nAlignBlockSize % tilingData_->mAlignBlockSize) ==
                         (tilingData_->mAlignBlockSize - 1);

    // Align for fp16, fp8, fp4
    dataLen16Align_ = (calcRow_ + 15) / 16 * 16;
    dataLen32Align_ = (calcRow_ + 31) / 32 * 32;
    dataLen64Align_ = (calcRow_ + 63) / 64 * 64;
}

template <typename xDtype, typename yDtype>
template <const bool scaleAlg, RoundMode roundMode>
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype>::ProcessOneLoop(int64_t index)
{
    CopyIn(ubOffset_, calcCol_, calcRow_);
    ComputeAll<scaleAlg, roundMode>(calcCol_, calcRow_);
    CopyOut(ubOffset_, scaleOffset_, calcCol_, calcRow_);
}

template <typename xDtype, typename yDtype>
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype>::CopyIn(int64_t offset, int64_t blockCount, int64_t dataLen)
{
    int64_t rightPadding = (dataLen * sizeof(xDtype) + 31) / 32 * 32 / sizeof(xDtype) - dataLen;
    DataCopyExtParams copyInParams = {
        static_cast<uint16_t>(blockCount), static_cast<uint32_t>(dataLen * sizeof(xDtype)),
        static_cast<uint32_t>((tilingData_->postAxisSize - dataLen) * sizeof(xDtype)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)};
    DataCopyPadExtParams<xDtype> padParams{true, 0, static_cast<uint8_t>(rightPadding), 0};

    LocalTensor<xDtype> xLocal = inQueue.template AllocTensor<xDtype>();
    DataCopyPad(xLocal, xGm_[offset], copyInParams, padParams);
    inQueue.template EnQue(xLocal);
}

template <typename xDtype, typename yDtype>
template <const bool scaleAlg, RoundMode roundMode>
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype>::ComputeAll(int64_t blockCount, int64_t dataLen)
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

    int64_t xOffset = 0;
    int64_t yOffset = 0;
    int64_t scaleUbOffset = 0;
    int64_t tmpOffset = 0;

    int64_t calcLoop = calcBlockTail_ == 0 ? calcBlockLoop_ : (calcBlockLoop_ - 1);
    for (int64_t i = 0; i < calcLoop; i++) {
        xOffset = i * tilingData_->blockSize * dataLen16Align_;
        if constexpr ((IsSame<yDtype, float8_e4m3_t>::value) || (IsSame<yDtype, float8_e5m2_t>::value)) {
            yOffset = i * tilingData_->blockSize * dataLen32Align_;
        } else {
            yOffset = i * tilingData_->blockSize * dataLen64Align_ / DIGIT_TWO;
        }
        scaleUbOffset = i * dataLen32Align_;
        tmpOffset = i * dataLen16Align_;
        if constexpr (scaleAlg) {
            ComputeScaleCuBlas(
                dataLen, blockSize_, xAddr + xOffset, mxTmpScaleAddr + scaleUbOffset, tmpAddr + tmpOffset);
        } else {
            ComputeScaleOcp(dataLen, blockSize_, xAddr + xOffset, mxTmpScaleAddr + scaleUbOffset, tmpAddr + tmpOffset);
        }
        ComputeYVf<roundMode>(dataLen, blockSize_, xAddr + xOffset, tmpAddr + tmpOffset, yAddr + yOffset);
    }
    if (calcBlockTail_ != 0) {
        xOffset = calcLoop * tilingData_->blockSize * dataLen16Align_;
        if constexpr ((IsSame<yDtype, float8_e4m3_t>::value) || (IsSame<yDtype, float8_e5m2_t>::value)) {
            yOffset = calcLoop * tilingData_->blockSize * dataLen32Align_;
        } else {
            yOffset = calcLoop * tilingData_->blockSize * dataLen64Align_ / DIGIT_TWO;
        }
        scaleUbOffset = calcLoop * dataLen32Align_;
        tmpOffset = calcLoop * dataLen16Align_;
        if constexpr (scaleAlg) {
            ComputeScaleCuBlas(
                dataLen, static_cast<uint16_t>(calcBlockTail_), xAddr + xOffset, mxTmpScaleAddr + scaleUbOffset,
                tmpAddr + tmpOffset);
        } else {
            ComputeScaleOcp(
                dataLen, static_cast<uint16_t>(calcBlockTail_), xAddr + xOffset, mxTmpScaleAddr + scaleUbOffset,
                tmpAddr + tmpOffset);
        }
        ComputeYVf<roundMode>(
            dataLen, static_cast<uint16_t>(calcBlockTail_), xAddr + xOffset, tmpAddr + tmpOffset, yAddr + yOffset);
    }

    if (scaleNeedsPad_) {
        int64_t padScaleOffset = (calcCol_ + tilingData_->blockSize - 1) / tilingData_->blockSize * dataLen32Align_;
        Duplicate<uint8_t>(tmpScaleLocal[padScaleOffset], (uint8_t)0, dataLen32Align_);
    }

    for (int64_t i = 0; i < ((calcBlockLoop_ + 1) / DIGIT_TWO * DIGIT_TWO); i++) {
        if (i % DIGIT_TWO == 1) {
            Interleave(
                mxScale[(i - 1) * dataLen32Align_], mxScale[i * dataLen32Align_],
                tmpScaleLocal[(i - 1) * dataLen32Align_], tmpScaleLocal[i * dataLen32Align_], dataLen32Align_);
        }
    }

    mxScaleQueue.template EnQue(mxScale);
    outQueue.template EnQue(y);
    inQueue.template FreeTensor(x);
}

template <typename xDtype, typename yDtype>
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype>::CopyOut(
    int64_t yOffset, int64_t scaleOutOffset, int64_t blockCount, int64_t dataLen)
{
    uint16_t outBurst = 0;
    uint32_t outBlockLen = 0;
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;
    int64_t YOffset = yOffset;
    uint32_t scaleSrcStride = DIGIT_TWO * ((dataLen + 31) / 32) - (DIGIT_TWO * dataLen + 31) / 32;

    if constexpr (IsSame<yDtype, fp4x2_e2m1_t>::value || IsSame<yDtype, fp4x2_e1m2_t>::value) {
        outBurst = blockCount;
        outBlockLen = dataLen / DIGIT_TWO * sizeof(uint8_t);
        srcStride = 0;
        dstStride = (tilingData_->postAxisSize - dataLen) / DIGIT_TWO * sizeof(uint8_t);
        YOffset = yOffset / DIGIT_TWO;
    } else {
        outBurst = blockCount;
        outBlockLen = dataLen * sizeof(uint8_t);
        srcStride = 0;
        dstStride = (tilingData_->postAxisSize - dataLen) * sizeof(uint8_t);
        YOffset = yOffset;
    }
    DataCopyExtParams yCopyOutParams = {
        static_cast<uint16_t>(outBurst), static_cast<uint32_t>(outBlockLen), static_cast<uint32_t>(srcStride),
        static_cast<uint32_t>(dstStride), static_cast<uint32_t>(0)};

    DataCopyExtParams scaleCopyOutParams = {
        static_cast<uint16_t>(
            (blockCount + DIGIT_TWO * tilingData_->blockSize - 1) / (DIGIT_TWO * tilingData_->blockSize)),
        static_cast<uint32_t>(dataLen * DIGIT_TWO * sizeof(uint8_t)), static_cast<uint32_t>(scaleSrcStride),
        static_cast<uint32_t>(DIGIT_TWO * (tilingData_->postAxisSize - dataLen) * sizeof(uint8_t)),
        static_cast<uint32_t>(0)};

    LocalTensor<uint8_t> yLocal = outQueue.template DeQue<uint8_t>();
    DataCopyPad(yGm_[YOffset], yLocal, yCopyOutParams);
    outQueue.FreeTensor(yLocal);

    LocalTensor<uint8_t> mxScaleLocal = mxScaleQueue.template DeQue<uint8_t>();
    DataCopyPad(mxScaleGm_[scaleOutOffset], mxScaleLocal, scaleCopyOutParams);
    mxScaleQueue.FreeTensor(mxScaleLocal);
}

template <typename xDtype, typename yDtype>
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype>::ComputeScaleCuBlas(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScaleAddr,
    __ubuf__ uint16_t* tmpAddr)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<xDtype> xRegTensor;
        AscendC::MicroAPI::RegTensor<bfloat16_t> xBF16RegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> absRegTensor;        // x绝对值
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
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

        AscendC::MicroAPI::Duplicate(absForX, ABS_FOR_UINT16);
        AscendC::MicroAPI::Duplicate(maxEleRegTensorFP16, HALF_INF);
        AscendC::MicroAPI::Duplicate(manForFP32, MAN_FOR_FP32);
        AscendC::MicroAPI::Duplicate(invMax, INV_DTYPE_MAX);
        AscendC::MicroAPI::Duplicate(maxRegTensor, 0);
        AscendC::MicroAPI::Duplicate(maxEleRegTensor, MAX_EXP_FOR_BF16);
        AscendC::MicroAPI::Duplicate(biasRegTensor, BF16_EXP_BIAS);
        AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
        AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);

        for (uint16_t j = 0; j < blockCount; j++) {
            DataCopy(xRegTensor, xAddr + j * dataLen16Align_);
            AscendC::MicroAPI::And(
                absRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor, absForX, pregAll16);
            AscendC::MicroAPI::Max(maxRegTensor, maxRegTensor, absRegTensor, pregAll16);
        }

        AscendC::MicroAPI::Cast<float, xDtype, castTraitZero>(
            (AscendC::MicroAPI::RegTensor<float>&)manAbsFP32RegTensor0,
            (AscendC::MicroAPI::RegTensor<xDtype>&)maxRegTensor, pregAll16);
        AscendC::MicroAPI::Mul(
            (AscendC::MicroAPI::RegTensor<float>&)manAbsFP32RegTensor0,
            (AscendC::MicroAPI::RegTensor<float>&)manAbsFP32RegTensor0, (AscendC::MicroAPI::RegTensor<float>&)invMax,
            pregAll32);
        AscendC::MicroAPI::ShiftRights(
            mxScaleFP32RegTensor0, (AscendC::MicroAPI::RegTensor<uint32_t>&)manAbsFP32RegTensor0, SHR_NUM_FOR_FP32,
            pregAll32);
        AscendC::MicroAPI::And(
            (AscendC::MicroAPI::RegTensor<uint32_t>&)manAbsFP32RegTensor0,
            (AscendC::MicroAPI::RegTensor<uint32_t>&)manAbsFP32RegTensor0,
            (AscendC::MicroAPI::RegTensor<uint32_t>&)manForFP32, pregAll32); // 提取尾数

        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(
            p0, (AscendC::MicroAPI::RegTensor<uint32_t>&)mxScaleFP32RegTensor0, NUMBER_ZERO, pregAll32);
        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(
            p0, (AscendC::MicroAPI::RegTensor<uint32_t>&)mxScaleFP32RegTensor0, NUMBER_TWO_FIVE_FOUR, p0);
        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(
            p0, (AscendC::MicroAPI::RegTensor<uint32_t>&)manAbsFP32RegTensor0, NUMBER_ZERO, p0);

        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(
            p1, (AscendC::MicroAPI::RegTensor<uint32_t>&)mxScaleFP32RegTensor0, NUMBER_ZERO, pregAll32);
        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(
            p1, (AscendC::MicroAPI::RegTensor<uint32_t>&)manAbsFP32RegTensor0, NUMBER_HALF, p1);
        AscendC::MicroAPI::MaskXor(p0, p0, p1, pregAll32);

        AscendC::MicroAPI::Adds(manAbsFP32RegTensor0, mxScaleFP32RegTensor0, 1, p0);
        AscendC::MicroAPI::Select(mxScaleFP32RegTensor0, manAbsFP32RegTensor0, mxScaleFP32RegTensor0, p0);

        AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
            (AscendC::MicroAPI::RegTensor<uint16_t>&)mxScale16RegTensor0, mxScaleFP32RegTensor0);

        AscendC::MicroAPI::Cast<float, xDtype, castTraitOne>(
            (AscendC::MicroAPI::RegTensor<float>&)manAbsFP32RegTensor1,
            (AscendC::MicroAPI::RegTensor<xDtype>&)maxRegTensor, pregAll16);
        AscendC::MicroAPI::Mul(
            (AscendC::MicroAPI::RegTensor<float>&)manAbsFP32RegTensor1,
            (AscendC::MicroAPI::RegTensor<float>&)manAbsFP32RegTensor1, (AscendC::MicroAPI::RegTensor<float>&)invMax,
            pregAll32);
        AscendC::MicroAPI::ShiftRights(
            mxScaleFP32RegTensor1, (AscendC::MicroAPI::RegTensor<uint32_t>&)manAbsFP32RegTensor1, SHR_NUM_FOR_FP32,
            pregAll32);
        AscendC::MicroAPI::And(
            (AscendC::MicroAPI::RegTensor<uint32_t>&)manAbsFP32RegTensor1,
            (AscendC::MicroAPI::RegTensor<uint32_t>&)manAbsFP32RegTensor1,
            (AscendC::MicroAPI::RegTensor<uint32_t>&)manForFP32, pregAll32); // 提取尾数

        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(
            p2, (AscendC::MicroAPI::RegTensor<uint32_t>&)mxScaleFP32RegTensor1, NUMBER_ZERO, pregAll32);
        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(
            p2, (AscendC::MicroAPI::RegTensor<uint32_t>&)mxScaleFP32RegTensor1, NUMBER_TWO_FIVE_FOUR, p2);
        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(
            p2, (AscendC::MicroAPI::RegTensor<uint32_t>&)manAbsFP32RegTensor1, NUMBER_ZERO, p2);

        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(
            p3, (AscendC::MicroAPI::RegTensor<uint32_t>&)mxScaleFP32RegTensor1, NUMBER_ZERO, pregAll32);
        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(
            p3, (AscendC::MicroAPI::RegTensor<uint32_t>&)manAbsFP32RegTensor1, NUMBER_HALF, p3);
        AscendC::MicroAPI::MaskXor(p2, p3, p2, pregAll32);

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

template <typename xDtype, typename yDtype>
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype>::ComputeScaleOcp(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScaleAddr,
    __ubuf__ uint16_t* tmpAddr)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<xDtype> xRegTensor;
        AscendC::MicroAPI::RegTensor<bfloat16_t> xBF16RegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> expRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> expMaxRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> mxScaleRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> reversedShareExpRegTensor;
        AscendC::MicroAPI::RegTensor<uint8_t> mxScale;

        AscendC::MicroAPI::RegTensor<uint16_t> specialExpRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> biasRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> zeroRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> nanRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> maxEleRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> maxEleRegTensorFP16;
        AscendC::MicroAPI::RegTensor<uint16_t> fp8MaxExpRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> fp8NanRegTensor;

        AscendC::MicroAPI::MaskReg infMask;
        AscendC::MicroAPI::MaskReg zeroMask;
        AscendC::MicroAPI::MaskReg invalidDataMask;
        AscendC::MicroAPI::MaskReg pregAll8 =
            AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg pregAll16 =
            AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();

        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

        AscendC::MicroAPI::Duplicate(maxEleRegTensor, MAX_EXP_FOR_BF16);
        AscendC::MicroAPI::Duplicate(maxEleRegTensorFP16, HALF_INF);
        AscendC::MicroAPI::Duplicate(fp8MaxExpRegTensor, DTYPE_Y_MAX_EXP);
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        AscendC::MicroAPI::Duplicate(biasRegTensor, BF16_EXP_BIAS);
        AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
        AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
        AscendC::MicroAPI::Duplicate(expMaxRegTensor, 0);

        for (uint16_t j = 0; j < blockCount; j++) {
            DataCopy(xRegTensor, xAddr + j * dataLen16Align_);
            if constexpr (IsSame<xDtype, half>::value) {
                AscendC::MicroAPI::And(
                    expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor, maxEleRegTensorFP16, pregAll16);
                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::EQ>(
                    infMask, expRegTensor, maxEleRegTensorFP16, pregAll16);
                AscendC::MicroAPI::Cast<bfloat16_t, xDtype, castTraitHalf2Bf16>(
                    (AscendC::MicroAPI::RegTensor<bfloat16_t>&)expRegTensor,
                    (AscendC::MicroAPI::RegTensor<float16_t>&)xRegTensor, pregAll16);
                AscendC::MicroAPI::Select<uint16_t>(expRegTensor, maxEleRegTensor, expRegTensor, infMask);
                AscendC::MicroAPI::And(
                    expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)expRegTensor, maxEleRegTensor, pregAll16);
            } else {
                AscendC::MicroAPI::And(
                    expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor, maxEleRegTensor, pregAll16);
            }
            AscendC::MicroAPI::Max(expMaxRegTensor, expMaxRegTensor, expRegTensor, pregAll16);
        }

        AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxRegTensor, maxEleRegTensor, pregAll16);
        AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxRegTensor, zeroRegTensor, pregAll16);
        AscendC::MicroAPI::Compare<uint16_t, CMPMODE::LE>(
            invalidDataMask, expMaxRegTensor, fp8MaxExpRegTensor, pregAll16);
        AscendC::MicroAPI::Select<uint16_t>(expMaxRegTensor, fp8MaxExpRegTensor, expMaxRegTensor, invalidDataMask);
        AscendC::MicroAPI::Sub(expMaxRegTensor, expMaxRegTensor, fp8MaxExpRegTensor, pregAll16);

        AscendC::MicroAPI::ShiftRights(mxScaleRegTensor, expMaxRegTensor, SHR_NUM_FOR_BF16, pregAll16);
        AscendC::MicroAPI::Select<uint16_t>(mxScaleRegTensor, mxScaleRegTensor, fp8NanRegTensor, infMask);
        AscendC::MicroAPI::Select<uint16_t>(mxScaleRegTensor, mxScaleRegTensor, zeroRegTensor, zeroMask);

        AscendC::MicroAPI::Pack<uint8_t, uint16_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
            (AscendC::MicroAPI::RegTensor<uint8_t>&)mxScale, mxScaleRegTensor);
        DataCopy(mxScaleAddr, mxScale, pregAll8);
        // 求1/scale
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

template <typename xDtype, typename yDtype>
template <RoundMode roundMode>
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype>::ComputeYVf(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* tmpAddr, __ubuf__ uint8_t* yAddr)
{
    if constexpr (IsSame<xDtype, half>::value) {
        ComputeYFromHalf<roundMode>(dataLen, blockCount, xAddr, tmpAddr, yAddr);
    } else {
        ComputeYFromBf16<roundMode>(dataLen, blockCount, xAddr, tmpAddr, yAddr);
    }
}

template <typename xDtype, typename yDtype>
template <RoundMode roundMode>
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype>::ComputeYFromHalf(
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
            AscendC::MicroAPI::MaskMergeMode::ZEROING, roundMode};

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
                ComputeFP4FromHalf<roundMode>(xFP32RegTensor0);
                ComputeFP4FromHalf<roundMode>(xFP32RegTensor1);
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

                AscendC::MicroAPI::Pack<uint8_t, uint16_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                    (AscendC::MicroAPI::RegTensor<uint8_t>&)yZeroFP8,
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)yZeroFP8);

                DataCopy(yAddr + (j * dataLen32Align_), (AscendC::MicroAPI::RegTensor<uint8_t>&)yZeroFP8, pregAll8);
            }
        }
    }
}

template <typename xDtype, typename yDtype>
template <RoundMode roundMode>
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype>::ComputeFP4FromHalf(
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

template <typename xDtype, typename yDtype>
template <RoundMode roundMode>
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype>::ComputeYFromBf16(
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
            AscendC::MicroAPI::MaskMergeMode::ZEROING, roundMode};

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
                AscendC::MicroAPI::Mul(
                    valueRegTensor, xRegTensor, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)reversedShareExpRegTensor,
                    pregAll16);
                AscendC::MicroAPI::Cast<yDtype, bfloat16_t, castTraitBf16toFp4>(yZeroFP4, valueRegTensor, pregAll16);

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

                AscendC::MicroAPI::Pack<uint8_t, uint16_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                    (AscendC::MicroAPI::RegTensor<uint8_t>&)yZeroFP8,
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)yZeroFP8);

                DataCopy(yAddr + (j * dataLen32Align_), (AscendC::MicroAPI::RegTensor<uint8_t>&)yZeroFP8, pregAll8);
            }
        }
    }
}

template <typename xDtype, typename yDtype>
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR mxScale)
{
#if (__NPU_ARCH__ == 3101)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    // init block params
    InitParams();

    xGm_.SetGlobalBuffer((__gm__ xDtype*)(x));
    yGm_.SetGlobalBuffer((__gm__ uint8_t*)(y));
    mxScaleGm_.SetGlobalBuffer((__gm__ uint8_t*)(mxScale));

    int64_t inBufferSize_ = Ops::Base::CeilAlign(
        static_cast<int64_t>(ubRowLen_ * ubRowCount_ * sizeof(xDtype)),
        static_cast<int64_t>(Ops::Base::GetUbBlockSize()));
    int64_t mxScaleBufferSize_ = Ops::Base::CeilAlign(
        static_cast<int64_t>(
            ubRowLen_ * ((ubRowCount_ + DIGIT_TWO * tilingData_->blockSize - 1) / (DIGIT_TWO * tilingData_->blockSize) *
                         DIGIT_TWO)),
        static_cast<int64_t>(Ops::Base::GetUbBlockSize()));
    int64_t outBufferSize_ = Ops::Base::CeilAlign(
        static_cast<int64_t>(ubRowLen_ * ubRowCount_), static_cast<int64_t>(Ops::Base::GetUbBlockSize()));
    int64_t tmpBufferSize_ = Ops::Base::CeilAlign(
        static_cast<int64_t>(
            ubRowLen_ *
            ((ubRowCount_ + DIGIT_TWO * tilingData_->blockSize - 1) / (DIGIT_TWO * tilingData_->blockSize) *
             DIGIT_TWO) *
            sizeof(xDtype)),
        static_cast<int64_t>(Ops::Base::GetUbBlockSize()));
    int64_t tmpScaleBufferSize_ = Ops::Base::CeilAlign(
        static_cast<int64_t>(
            ubRowLen_ * ((ubRowCount_ + DIGIT_TWO * tilingData_->blockSize - 1) / (DIGIT_TWO * tilingData_->blockSize) *
                         DIGIT_TWO)),
        static_cast<int64_t>(Ops::Base::GetUbBlockSize()));

    pipe_->InitBuffer(inQueue, DB_BUFFER, inBufferSize_);
    pipe_->InitBuffer(mxScaleQueue, DB_BUFFER, mxScaleBufferSize_);
    pipe_->InitBuffer(outQueue, DB_BUFFER, outBufferSize_);
    pipe_->InitBuffer(tmpBuf, tmpBufferSize_);
    pipe_->InitBuffer(tmpScaleBuf, tmpScaleBufferSize_);
}

template <typename xDtype, typename yDtype>
__aicore__ inline void DynamicMxQuantHP2000<xDtype, yDtype>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    for (int64_t i = 0; i < loopPerCore_; i++) {
        // init runtime ub params
        InitCalcParams(i);
        if constexpr (IsSame<DTYPE_Y, fp8_e4m3fn_t>::value || IsSame<DTYPE_Y, fp8_e5m2_t>::value) {
            if (scaleAlg_) {
                ProcessOneLoop<true, RoundMode::CAST_RINT>(i);
            } else {
                ProcessOneLoop<false, RoundMode::CAST_RINT>(i);
            }
        } else if (roundMode_ == MODE_RINT) {
            if (scaleAlg_) {
                ProcessOneLoop<true, RoundMode::CAST_RINT>(i);
            } else {
                ProcessOneLoop<false, RoundMode::CAST_RINT>(i);
            }
        } else if (roundMode_ == MODE_ROUND) {
            if (scaleAlg_) {
                ProcessOneLoop<true, RoundMode::CAST_ROUND>(i);
            } else {
                ProcessOneLoop<false, RoundMode::CAST_ROUND>(i);
            }
        } else if (roundMode_ == MODE_FLOOR) {
            if (scaleAlg_) {
                ProcessOneLoop<true, RoundMode::CAST_FLOOR>(i);
            } else {
                ProcessOneLoop<false, RoundMode::CAST_FLOOR>(i);
            }
        }
    }
}

} // namespace DynamicMxQuant
#endif // DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_FP8_H
