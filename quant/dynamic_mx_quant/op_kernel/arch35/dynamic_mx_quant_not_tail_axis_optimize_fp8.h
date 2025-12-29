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
 * \file dynamic_mx_quant_not_tail_axis_optimize_fp8.h
 * \brief
 */

#ifndef DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_OPTIMIZE_FP8_H
#define DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_OPTIMIZE_FP8_H

#include "dynamic_mx_quant_not_tail_axis_base_fp8.h"
#include "op_kernel/math_util.h"

namespace DynamicMxQuant {
using namespace AscendC;

template <typename T, typename U, const bool isTail>
class DynamicMxQuantNotTailAxisOptimizeFP8 : public DynamicMxQuantBaseFP8<T, U, isTail> {
public:
    __aicore__ inline DynamicMxQuantNotTailAxisOptimizeFP8(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR workspace, const DynamicMxQuantTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SplitPreAxisCompute(int64_t ubFactor, int64_t blockSizeIdx);
    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void ComputeOCP(
        int64_t dataLen, int64_t blockCount, __ubuf__ T* xAddr, __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint8_t* yAddr);
    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void ComputeCuBLAS(
        int64_t dataLen, int64_t blockCount, __ubuf__ T* xAddr, __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint8_t* yAddr);

private:
    TBuf<QuePosition::VECCALC> maxExpBuf_;
    int64_t blockLoopOffset_ = 0;
    int64_t blockOffset_ = 0;
    int64_t scaleBlockOffset_ = 0;
    int64_t bufferSize_ = 0;
};

template <typename T, typename U, const bool isTail>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeFP8<T, U, isTail>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR workspace, const DynamicMxQuantTilingData* tilingData)
{
    this->BaseInit(x, y, mxScale, workspace, tilingData);
    if (this->blockIdx_ >= this->usedCoreNum_) {
        return;
    }
    blockLoopOffset_ = this->blockIdx_ * this->blockFactor_;

    scaleBlockOffset_ = blockLoopOffset_ * this->ubFactor_ * this->postAxisSize_;
    if (this->isPad_) {
        int64_t blockLoopMod = blockLoopOffset_ * this->ubFactor_ % this->blockSizeNumInAxis_;
        int64_t fullAxisNum = blockLoopOffset_ * this->ubFactor_ / this->blockSizeNumInAxis_;
        blockOffset_ = fullAxisNum * (this->fullBlockSizeNumInAxis_ * this->blockSize_ + this->tailBlockSize_) *
                           this->postAxisSize_ +
                       blockLoopMod * this->blockSize_ * this->postAxisSize_;
    } else {
        blockOffset_ = this->blockSize_ * scaleBlockOffset_;
    }
    bufferSize_ = this->ubFactor_ * this->blockSize_ * this->postAxisSize_ * sizeof(T);

    this->xGm_.SetGlobalBuffer((__gm__ T*)(x) + blockOffset_);
    this->mxScaleGm_.SetGlobalBuffer((__gm__ uint8_t*)(mxScale) + scaleBlockOffset_);
    this->workspaceGm_.SetGlobalBuffer((__gm__ uint8_t*)(workspace) + scaleBlockOffset_);
    this->yGm_.SetGlobalBuffer((__gm__ uint8_t*)(y) + blockOffset_);

    bufferSize_ = Ops::Base::CeilAlign(bufferSize_, static_cast<int64_t>(Ops::Base::GetUbBlockSize()));
    this->pipe_.InitBuffer(this->inQueue_, DB_BUFFER, bufferSize_);
    this->pipe_.InitBuffer(this->mxScaleQueue_, DB_BUFFER, bufferSize_);
    this->pipe_.InitBuffer(this->outQueue_, DB_BUFFER, bufferSize_);
    this->pipe_.InitBuffer(maxExpBuf_, Ops::Base::GetVRegSize());
}

template <typename T, typename U, const bool isTail>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeFP8<T, U, isTail>::Process()
{
    if (this->blockIdx_ >= this->usedCoreNum_) {
        return;
    }
    int64_t loopSize = this->isTailBlock_ ? this->tailBlockFactor_ : this->blockFactor_;
    int64_t blockSizeNumInPreCore = blockLoopOffset_ * this->ubFactor_;
    int64_t scaleDataLen = this->ubFactor_ * this->postAxisSize_;
    int64_t offset = 0;
    for (int64_t loopIter = 0; loopIter < loopSize - 1; loopIter++) {
        int64_t blockSizeIdx = blockSizeNumInPreCore + loopIter * this->ubFactor_;
        int64_t dataLen = this->CalcDataLen(this->ubFactor_, blockSizeIdx, scaleDataLen);
        this->InitCopyParams(1, dataLen);
        this->CopyIn(offset);
        SplitPreAxisCompute(this->ubFactor_, blockSizeIdx);
        this->CopyOut(offset, loopIter * scaleDataLen, scaleDataLen);
        offset += dataLen;
    }
    int64_t ubFactor = this->isTailBlock_ ? this->tailUbFactor_ : this->ubFactor_;
    scaleDataLen = ubFactor * this->postAxisSize_;
    int64_t blockSizeIdx = blockSizeNumInPreCore + (loopSize - 1) * this->ubFactor_;
    int64_t dataLen = this->CalcDataLen(ubFactor, blockSizeIdx, scaleDataLen);
    this->InitCopyParams(1, dataLen);
    this->CopyIn(offset);
    SplitPreAxisCompute(ubFactor, blockSizeIdx);
    this->CopyOut(offset, (loopSize - 1) * this->ubFactor_ * this->postAxisSize_, scaleDataLen);
}

template <typename T, typename U, const bool isTail>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeFP8<T, U, isTail>::SplitPreAxisCompute(
    int64_t ubFactor, int64_t blockSizeIdx)
{
    LocalTensor<T> x = this->inQueue_.template DeQue<T>();
    LocalTensor<uint8_t> mxScale = this->mxScaleQueue_.template AllocTensor<uint8_t>();
    LocalTensor<uint8_t> y = this->outQueue_.template AllocTensor<uint8_t>();

    int64_t offset = 0;
    for (int64_t i = 0; i < ubFactor; i++) {
        auto xAddr = (__ubuf__ T*)x.GetPhyAddr() + offset;
        auto mxScaleAddr = (__ubuf__ uint8_t*)mxScale.GetPhyAddr() + i * this->postAxisSize_;
        auto yAddr = (__ubuf__ uint8_t*)y.GetPhyAddr() + offset;
        int64_t blockCount = this->BlockCountInCurCompute(blockSizeIdx + i + 1);
        offset += blockCount * this->postAxisSize_;
        if (this->scaleAlg_) {
            ComputeCuBLAS<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(
                this->postAxisSize_, blockCount, xAddr, mxScaleAddr, yAddr);
        } else {
            ComputeOCP<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(
                this->postAxisSize_, blockCount, xAddr, mxScaleAddr, yAddr);
        }
    }
    this->inQueue_.template FreeTensor(x);
    this->mxScaleQueue_.template EnQue(mxScale);
    this->outQueue_.template EnQue(y);
}

template <typename T, typename U, const bool isTail>
template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeFP8<T, U, isTail>::ComputeOCP(
    int64_t dataLen, int64_t blockCount, __ubuf__ T* xAddr, __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint8_t* yAddr)
{
    constexpr uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(T);     // 寄存器单次处理能处理的长度
    constexpr uint32_t vfNum = Ops::Base::GetVRegSize() / sizeof(float); // cast到FP32后单个寄存器中的元素个数
    uint16_t rowsSingleLoop =
        static_cast<uint16_t>(min(blockCount, static_cast<int64_t>(vfLen) / dataLen)); // 单次处理能处理的行数
    uint16_t dataLenSingleLoop = rowsSingleLoop * static_cast<uint16_t>(dataLen);      // 单次处理长度
    uint16_t regLoop = Ceil(static_cast<uint16_t>(blockCount), rowsSingleLoop);        // 循环数
    uint16_t rowsTailLoop = static_cast<uint16_t>(blockCount) % rowsSingleLoop;        // 尾循环处理的行数
    uint32_t loopNum0 = dataLenSingleLoop <= vfNum ? dataLenSingleLoop : vfNum;
    uint32_t loopNum1 = dataLenSingleLoop <= vfNum ? 0 : (dataLenSingleLoop - vfNum);
    if (rowsTailLoop == 0) {
        rowsTailLoop = rowsSingleLoop;
    }
    uint16_t dataLenTailLoop = rowsTailLoop * static_cast<uint16_t>(dataLen); // 尾循环处理的长度
    uint32_t tailLoopNum0 = dataLenTailLoop <= vfNum ? dataLenTailLoop : vfNum;
    uint32_t tailLoopNum1 = dataLenTailLoop <= vfNum ? 0 : (dataLenTailLoop - vfNum);
    uint16_t loopSize =
        static_cast<uint16_t>(DIGIT_SIXTY_THREE - AscendC::ScalarCountLeadingZero(static_cast<uint64_t>(rowsSingleLoop))); // 求最大指数行的二分次数
    uint16_t rows = 1 << loopSize; // 最接近rowsSingleLoop的2次方数
    uint16_t expOffset = rows * static_cast<uint16_t>(dataLen);
    uint16_t FP8_BF16_MAX_EXP = 0;
    if constexpr (IsSame<U, fp8_e4m3fn_t>::value) {
        FP8_BF16_MAX_EXP = FP8_E4M3_MAX_EXP;
    } else if constexpr (IsSame<U, fp8_e5m2_t>::value) {
        FP8_BF16_MAX_EXP = FP8_E5M2_MAX_EXP;
    }

    LocalTensor<uint16_t> maxExpTensor = maxExpBuf_.Get<uint16_t>();
    auto maxExpAddr = (__ubuf__ uint16_t*)maxExpTensor.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> xRegTensor;
        AscendC::MicroAPI::RegTensor<bfloat16_t> xBF16RegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> expRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> expMaxRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> maxEleRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> fp8NanRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> fp8MaxExpRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> shareExpRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> mxScaleRegTensor;
        AscendC::MicroAPI::RegTensor<uint8_t> mxScale;
        AscendC::MicroAPI::RegTensor<uint16_t> specialExpRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> reversedShareExpRegTensor;
        AscendC::MicroAPI::RegTensor<float> reversedShareExpRegTensorFP32Zero;
        AscendC::MicroAPI::RegTensor<float> reversedShareExpRegTensorFP32One;
        AscendC::MicroAPI::RegTensor<uint16_t> biasRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> zeroRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> nanRegTensor;
        AscendC::MicroAPI::RegTensor<uint8_t> outZero;
        AscendC::MicroAPI::RegTensor<uint8_t> outOne;
        AscendC::MicroAPI::RegTensor<uint16_t> yRegTensorZero;
        AscendC::MicroAPI::RegTensor<uint16_t> yRegTensorOne;
        AscendC::MicroAPI::RegTensor<float> yZero;
        AscendC::MicroAPI::RegTensor<float> yOne;
        AscendC::MicroAPI::RegTensor<U> yZeroFP8;
        AscendC::MicroAPI::RegTensor<U> yOneFP8;
        AscendC::MicroAPI::RegTensor<bfloat16_t> valueRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> invalidMaskFp16;
        AscendC::MicroAPI::RegTensor<uint16_t> xSelectRegTensor;
        AscendC::MicroAPI::UnalignReg u0;
        AscendC::MicroAPI::UnalignReg u1;
        AscendC::MicroAPI::MaskReg zeroMask;
        AscendC::MicroAPI::MaskReg infMask;
        AscendC::MicroAPI::MaskReg invalidDataMask;
        AscendC::MicroAPI::MaskReg specialDataMask;
        AscendC::MicroAPI::MaskReg maskAll = AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        static constexpr AscendC::MicroAPI::CastTrait castTraitZero = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTraitOne = {
            AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTrait32to8 = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC}; 
        AscendC::MicroAPI::Duplicate(fp8MaxExpRegTensor, FP8_BF16_MAX_EXP);
        AscendC::MicroAPI::Duplicate(maxEleRegTensor, MAX_EXP_FOR_BF16);
        AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        AscendC::MicroAPI::Duplicate(biasRegTensor, BF16_EXP_BIAS);
        AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
        AscendC::MicroAPI::Duplicate(invalidMaskFp16, INVALID_FLOAT16);

        uint32_t pnum = dataLenSingleLoop;
        uint32_t tailPnum = dataLenTailLoop;
        AscendC::MicroAPI::MaskReg pnumMask = AscendC::MicroAPI::UpdateMask<uint16_t>(pnum);
        AscendC::MicroAPI::MaskReg tailPnumMask = AscendC::MicroAPI::UpdateMask<uint16_t>(tailPnum);
        AscendC::MicroAPI::Duplicate(expMaxRegTensor, 0);
        for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); i++) {
            this->template LoadData<RoundMode::CAST_TRUNC, roundMode>(xAddr, i * dataLenSingleLoop, xRegTensor,
                pnumMask);
            if constexpr (IsSame<T, half>::value) {
                AscendC::MicroAPI::And(
                    xSelectRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor, invalidMaskFp16, pnumMask);
                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(
                    invalidDataMask, xSelectRegTensor, invalidMaskFp16, pnumMask);
                AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(xBF16RegTensor, xRegTensor, pnumMask);
                AscendC::MicroAPI::And(expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xBF16RegTensor,
                maxEleRegTensor, pnumMask);
                AscendC::MicroAPI::Select<uint16_t>(expRegTensor, expRegTensor, maxEleRegTensor, invalidDataMask);
            } else {
                AscendC::MicroAPI::And(expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor,
                maxEleRegTensor, pnumMask);
            }
            AscendC::MicroAPI::Max(expMaxRegTensor, expMaxRegTensor, expRegTensor, pnumMask);
        }
        this->template LoadData<RoundMode::CAST_TRUNC, roundMode>(xAddr, (regLoop - 1) * dataLenSingleLoop, xRegTensor,
            tailPnumMask);
        if constexpr (IsSame<T, half>::value) {
            AscendC::MicroAPI::And(
                    xSelectRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor, invalidMaskFp16, tailPnumMask);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(
                    invalidDataMask, xSelectRegTensor, invalidMaskFp16, tailPnumMask);
            AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(xBF16RegTensor, xRegTensor, tailPnumMask);
            AscendC::MicroAPI::And(expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xBF16RegTensor,
            maxEleRegTensor, tailPnumMask);
            AscendC::MicroAPI::Select<uint16_t>(expRegTensor, expRegTensor, maxEleRegTensor, invalidDataMask);
        } else {
            AscendC::MicroAPI::And(expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor,
            maxEleRegTensor, tailPnumMask);
        }
        AscendC::MicroAPI::Max(expRegTensor, expMaxRegTensor, expRegTensor, tailPnumMask);
        AscendC::MicroAPI::Copy<uint16_t, AscendC::MicroAPI::MaskMergeMode::MERGING>(expMaxRegTensor, expRegTensor,
            tailPnumMask);
        // 二分法求rowsSingleLoop行中的最大行
        AscendC::MicroAPI::DataCopy(maxExpAddr, expMaxRegTensor, pnumMask);
        AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        uint32_t maskNum = dataLenSingleLoop - expOffset;
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::UpdateMask<uint16_t>(maskNum);
        AscendC::MicroAPI::DataCopyUnAlignPre(u0, maxExpAddr);
        AscendC::MicroAPI::DataCopyUnAlign(expMaxRegTensor, u0, maxExpAddr);
        AscendC::MicroAPI::DataCopyUnAlignPre(u0, maxExpAddr + expOffset);
        AscendC::MicroAPI::DataCopyUnAlign(expRegTensor, u0, maxExpAddr + expOffset);
        AscendC::MicroAPI::Max(expRegTensor, expMaxRegTensor, expRegTensor, mask);
        AscendC::MicroAPI::Copy<uint16_t, AscendC::MicroAPI::MaskMergeMode::MERGING>(expMaxRegTensor, expRegTensor,
            mask);
        for (uint16_t i = 0; i < loopSize; i++) {
            AscendC::MicroAPI::DataCopy(maxExpAddr, expMaxRegTensor, pnumMask);
            AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
            expOffset /= DIGIT_TWO;
            maskNum = expOffset;
            mask = AscendC::MicroAPI::UpdateMask<uint16_t>(maskNum);
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, maxExpAddr);
            AscendC::MicroAPI::DataCopyUnAlign(expMaxRegTensor, u0, maxExpAddr);
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, maxExpAddr + expOffset);
            AscendC::MicroAPI::DataCopyUnAlign(expRegTensor, u0, maxExpAddr + expOffset);
            AscendC::MicroAPI::Max(expMaxRegTensor, expMaxRegTensor, expRegTensor, mask);
        }
        maskNum = static_cast<uint32_t>(dataLen);
        mask = AscendC::MicroAPI::UpdateMask<uint16_t>(maskNum);
        AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxRegTensor, maxEleRegTensor, mask);
        AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxRegTensor, zeroRegTensor, mask);
        AscendC::MicroAPI::Compare<uint16_t, CMPMODE::LT>(invalidDataMask, expMaxRegTensor, fp8MaxExpRegTensor,
            mask);
        AscendC::MicroAPI::Select<uint16_t>(expMaxRegTensor, fp8MaxExpRegTensor, expMaxRegTensor, invalidDataMask);
        AscendC::MicroAPI::Sub(expMaxRegTensor, expMaxRegTensor, fp8MaxExpRegTensor, mask);
        AscendC::MicroAPI::ShiftRights(mxScaleRegTensor, expMaxRegTensor, SHR_NUM_FOR_BF16, mask);
        AscendC::MicroAPI::Select<uint16_t>(mxScaleRegTensor, mxScaleRegTensor, fp8NanRegTensor, infMask);
        AscendC::MicroAPI::Select<uint16_t>(mxScaleRegTensor, mxScaleRegTensor, zeroRegTensor, zeroMask);
        AscendC::MicroAPI::Pack(mxScale, mxScaleRegTensor);
        AscendC::MicroAPI::DataCopyUnAlign(mxScaleAddr, mxScale, u1, dataLen);
        AscendC::MicroAPI::DataCopyUnAlignPost(mxScaleAddr, u1, 0);

        // 求1/scale
        AscendC::MicroAPI::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, expMaxRegTensor, biasRegTensor, mask);
        AscendC::MicroAPI::Sub(reversedShareExpRegTensor, biasRegTensor, expMaxRegTensor, mask);
        AscendC::MicroAPI::Select<uint16_t>(reversedShareExpRegTensor, reversedShareExpRegTensor, nanRegTensor,
            infMask);
        AscendC::MicroAPI::Select<uint16_t>(reversedShareExpRegTensor, reversedShareExpRegTensor, zeroRegTensor,
            zeroMask);
        AscendC::MicroAPI::Select<uint16_t>(reversedShareExpRegTensor, specialExpRegTensor, reversedShareExpRegTensor,
            specialDataMask);

        auto scaleAddr = maxExpAddr;
        for (uint16_t i = 0; i < rowsSingleLoop; i++) {
            AscendC::MicroAPI::DataCopyUnAlign(scaleAddr, reversedShareExpRegTensor, u1, dataLen);
            AscendC::MicroAPI::DataCopyUnAlignPost(scaleAddr, u1, 0);
        }
        AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        AscendC::MicroAPI::DataCopy(reversedShareExpRegTensor, maxExpAddr);
        // 求data value
        for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); i++) {
            this->template LoadData<toBf16RoundMode, roundMode, true>(xAddr, i * dataLenSingleLoop, xRegTensor,
                pnumMask);
            if constexpr (IsSame<T, half>::value) {
                AscendC::MicroAPI::Cast<float, T, castTraitZero>(yZero, xRegTensor, pnumMask);
                AscendC::MicroAPI::Cast<float, T, castTraitOne>(yOne, xRegTensor, pnumMask);
                AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitZero>(reversedShareExpRegTensorFP32Zero, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, pnumMask);
                AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitOne>(reversedShareExpRegTensorFP32One, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, pnumMask);
                AscendC::MicroAPI::Mul(yZero, yZero, reversedShareExpRegTensorFP32Zero, maskAll);
                AscendC::MicroAPI::Mul(yOne, yOne, reversedShareExpRegTensorFP32One, maskAll);
            } else {
                AscendC::MicroAPI::Mul(valueRegTensor, xRegTensor,
                    (AscendC::MicroAPI::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, pnumMask);
                AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitZero>(yZero, valueRegTensor, maskAll);
                AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitOne>(yOne, valueRegTensor, maskAll);
            }
            AscendC::MicroAPI::Interleave(yZero, yOne, yZero, yOne);
            AscendC::MicroAPI::Cast<U, float, castTrait32to8>(yZeroFP8, yZero, maskAll);
            AscendC::MicroAPI::Cast<U, float, castTrait32to8>(yOneFP8, yOne, maskAll);
            AscendC::MicroAPI::Pack(yRegTensorZero, (AscendC::MicroAPI::RegTensor<uint32_t>&)yZeroFP8);
            AscendC::MicroAPI::Pack(outZero, yRegTensorZero);
            AscendC::MicroAPI::Pack(yRegTensorOne, (AscendC::MicroAPI::RegTensor<uint32_t>&)yOneFP8);
            AscendC::MicroAPI::Pack(outOne, yRegTensorOne);
            auto addr0 = yAddr + i * dataLenSingleLoop;
            AscendC::MicroAPI::DataCopyUnAlign(addr0, outZero, u1, loopNum0);
            AscendC::MicroAPI::DataCopyUnAlignPost(addr0, u1, 0);
            auto addr1 = yAddr + i * dataLenSingleLoop + loopNum0;
            AscendC::MicroAPI::DataCopyUnAlign(addr1, outOne, u1, loopNum1);
            AscendC::MicroAPI::DataCopyUnAlignPost(addr1, u1, 0);
        }
        this->template LoadData<toBf16RoundMode, roundMode, true>(xAddr, (regLoop - 1) * dataLenSingleLoop, xRegTensor,
            tailPnumMask);
        if constexpr (IsSame<T, half>::value) {
            AscendC::MicroAPI::Cast<float, T, castTraitZero>(yZero, xRegTensor, tailPnumMask);
            AscendC::MicroAPI::Cast<float, T, castTraitOne>(yOne, xRegTensor, tailPnumMask);
            AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitZero>(reversedShareExpRegTensorFP32Zero, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, tailPnumMask);
            AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitOne>(reversedShareExpRegTensorFP32One, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, tailPnumMask);
            AscendC::MicroAPI::Mul(yZero, yZero, reversedShareExpRegTensorFP32Zero, maskAll);
            AscendC::MicroAPI::Mul(yOne, yOne, reversedShareExpRegTensorFP32One, maskAll);
        } else {
            AscendC::MicroAPI::Mul(valueRegTensor, xRegTensor,
                (AscendC::MicroAPI::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, tailPnumMask);
            AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitZero>(yZero, valueRegTensor, maskAll);
            AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitOne>(yOne, valueRegTensor, maskAll);
        }
        AscendC::MicroAPI::Interleave(yZero, yOne, yZero, yOne);
        AscendC::MicroAPI::Cast<U, float, castTrait32to8>(yZeroFP8, yZero, maskAll);
        AscendC::MicroAPI::Cast<U, float, castTrait32to8>(yOneFP8, yOne, maskAll);
        AscendC::MicroAPI::Pack(yRegTensorZero, (AscendC::MicroAPI::RegTensor<uint32_t>&)yZeroFP8);
        AscendC::MicroAPI::Pack(outZero, yRegTensorZero);
        AscendC::MicroAPI::Pack(yRegTensorOne, (AscendC::MicroAPI::RegTensor<uint32_t>&)yOneFP8);
        AscendC::MicroAPI::Pack(outOne, yRegTensorOne);
        auto addr0 = yAddr + (regLoop - 1) * dataLenSingleLoop;
        AscendC::MicroAPI::DataCopyUnAlign(addr0, outZero, u1, tailLoopNum0);
        AscendC::MicroAPI::DataCopyUnAlignPost(addr0, u1, 0);
        auto addr1 = yAddr + (regLoop - 1) * dataLenSingleLoop + tailLoopNum0;
        AscendC::MicroAPI::DataCopyUnAlign(addr1, outOne, u1, tailLoopNum1);
        AscendC::MicroAPI::DataCopyUnAlignPost(addr1, u1, 0);
    }
}

template <typename T, typename U, const bool isTail>
template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimizeFP8<T, U, isTail>::ComputeCuBLAS(
    int64_t dataLen, int64_t blockCount, __ubuf__ T* xAddr, __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint8_t* yAddr)
{
    constexpr uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(T); // 寄存器单次处理能处理的长度
    uint16_t rowsSingleLoop =
        static_cast<uint16_t>(min(blockCount, static_cast<int64_t>(vfLen) / dataLen)); // 单次处理能处理的行数
    uint16_t dataLenSingleLoop = rowsSingleLoop * static_cast<uint16_t>(dataLen);      // 单次处理长度
    uint16_t regLoop = Ceil(static_cast<uint16_t>(blockCount), rowsSingleLoop);        // 循环数
    uint16_t rowsTailLoop = static_cast<uint16_t>(blockCount) % rowsSingleLoop;        // 尾循环处理的行数
    if (rowsTailLoop == 0) {
        rowsTailLoop = rowsSingleLoop;
    }
    uint16_t dataLenTailLoop = rowsTailLoop * static_cast<uint16_t>(dataLen); // 尾循环处理的长度
    uint16_t loopSize =
        static_cast<uint16_t>(DIGIT_SIXTY_THREE - clz(static_cast<uint64_t>(rowsSingleLoop))); // 求最大指数行的二分次数
    uint16_t rows = 1 << loopSize; // 最接近rowsSingleLoop的2次方数
    uint16_t expOffset = rows * static_cast<uint16_t>(dataLen);
    uint32_t INV_DTYPE_MAX = 0;
    if constexpr (IsSame<U, fp8_e4m3fn_t>::value) {
        INV_DTYPE_MAX = 0x3b124925;
    } else if constexpr (IsSame<U, fp8_e5m2_t>::value) {
        INV_DTYPE_MAX = 0x37924925;
    }

    LocalTensor<uint16_t> maxExpTensor = maxExpBuf_.Get<uint16_t>();
    auto maxExpAddr = (__ubuf__ uint16_t*)maxExpTensor.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> xRegTensor;
        AscendC::MicroAPI::RegTensor<bfloat16_t> xBF16RegTensor;

        AscendC::MicroAPI::RegTensor<uint32_t> absMaskFP32RegTensor;
        AscendC::MicroAPI::RegTensor<uint32_t> maxFP32RegTensor;
        AscendC::MicroAPI::RegTensor<uint32_t> invMaxRegTensor;
        AscendC::MicroAPI::RegTensor<uint32_t> fp32ZeroRegTensor;
        // 指数位
        AscendC::MicroAPI::RegTensor<uint32_t> expFP32RegTensor;
        // 尾数位
        AscendC::MicroAPI::RegTensor<uint32_t> manFP32RegTensor;
        AscendC::MicroAPI::RegTensor<uint32_t> scaleBias;
        AscendC::MicroAPI::RegTensor<uint32_t> extractExpRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> tmp16RegTensor;
        AscendC::MicroAPI::MaskReg p0;
        AscendC::MicroAPI::MaskReg p1;
        AscendC::MicroAPI::MaskReg zeroMask;
        AscendC::MicroAPI::MaskReg infMask;

        AscendC::MicroAPI::RegTensor<uint16_t> expRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> expMaxRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> reversedShareExpRegTensor;
        AscendC::MicroAPI::RegTensor<float> reversedShareExpRegTensorFP32Zero;
        AscendC::MicroAPI::RegTensor<float> reversedShareExpRegTensorFP32One;
        AscendC::MicroAPI::RegTensor<uint32_t> nanRegTensor;
        AscendC::MicroAPI::RegTensor<uint8_t> outZero;
        AscendC::MicroAPI::RegTensor<float> yZero;
        AscendC::MicroAPI::RegTensor<float> yOne;
        AscendC::MicroAPI::RegTensor<U> yZeroFP8;
        AscendC::MicroAPI::RegTensor<U> yOneFP8;
        AscendC::MicroAPI::UnalignReg u0;
        AscendC::MicroAPI::UnalignReg u1;

        static constexpr AscendC::MicroAPI::CastTrait castTraitZero = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTraitOne = {
            AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTrait32to8 = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
        static constexpr AscendC::MicroAPI::CastTrait castTraitFp32ToBf16 = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, toBf16RoundMode};
        AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION_FP32);
        AscendC::MicroAPI::Duplicate(tmp16RegTensor, BF16_MAX);
        AscendC::MicroAPI::Duplicate(absMaskFP32RegTensor, MAX_EXP_FOR_FP32);
        AscendC::MicroAPI::Duplicate(invMaxRegTensor, INV_DTYPE_MAX);
        AscendC::MicroAPI::Duplicate(fp32ZeroRegTensor, 0);
        AscendC::MicroAPI::Duplicate(manFP32RegTensor, MAN_MASK_FP32);
        AscendC::MicroAPI::Duplicate(scaleBias, FP32_EXP_BIAS);

        uint32_t pnum = dataLenSingleLoop;
        uint32_t tailPnum = dataLenTailLoop;
        AscendC::MicroAPI::MaskReg pnumMask = AscendC::MicroAPI::UpdateMask<uint16_t>(pnum);
        AscendC::MicroAPI::MaskReg tailPnumMask = AscendC::MicroAPI::UpdateMask<uint16_t>(tailPnum);
        AscendC::MicroAPI::Duplicate(expMaxRegTensor, 0);
        for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); i++) {
            this->template LoadData<RoundMode::UNKNOWN, roundMode>(xAddr, i * dataLenSingleLoop, xRegTensor, pnumMask);
            AscendC::MicroAPI::And(
                expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor, tmp16RegTensor, pnumMask);
            AscendC::MicroAPI::Max(expMaxRegTensor, expMaxRegTensor, expRegTensor, pnumMask);
        }
        this->template LoadData<RoundMode::UNKNOWN, roundMode>(
            xAddr, (regLoop - 1) * dataLenSingleLoop, xRegTensor, tailPnumMask);
        AscendC::MicroAPI::And(
            expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor, tmp16RegTensor, tailPnumMask);
        AscendC::MicroAPI::Max(expRegTensor, expMaxRegTensor, expRegTensor, tailPnumMask);
        AscendC::MicroAPI::Copy<uint16_t, AscendC::MicroAPI::MaskMergeMode::MERGING>(
            expMaxRegTensor, expRegTensor, tailPnumMask);

        // 二分法求rowsSingleLoop行中的最大行
        AscendC::MicroAPI::DataCopy(maxExpAddr, expMaxRegTensor, pnumMask);
        AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        uint32_t maskNum = dataLenSingleLoop - expOffset;
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::UpdateMask<uint16_t>(maskNum);
        AscendC::MicroAPI::DataCopyUnAlignPre(u0, maxExpAddr + expOffset);
        AscendC::MicroAPI::DataCopyUnAlign(expRegTensor, u0, maxExpAddr + expOffset);
        AscendC::MicroAPI::Max(expRegTensor, expMaxRegTensor, expRegTensor, mask);
        AscendC::MicroAPI::Copy<uint16_t, AscendC::MicroAPI::MaskMergeMode::MERGING>(
            expMaxRegTensor, expRegTensor, mask);
        for (uint16_t i = 0; i < loopSize; i++) {
            AscendC::MicroAPI::DataCopy(maxExpAddr, expMaxRegTensor, pnumMask);
            AscendC::MicroAPI::LocalMemBar<
                AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
            expOffset /= DIGIT_TWO;
            maskNum = expOffset;
            mask = AscendC::MicroAPI::UpdateMask<uint16_t>(maskNum);
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, maxExpAddr + expOffset);
            AscendC::MicroAPI::DataCopyUnAlign(expRegTensor, u0, maxExpAddr + expOffset);
            AscendC::MicroAPI::Max(expMaxRegTensor, expMaxRegTensor, expRegTensor, mask);
        }
        maskNum = static_cast<uint32_t>(dataLen);
        mask = AscendC::MicroAPI::CreateMask<uint32_t>();
        maskNum = static_cast<uint32_t>(dataLen);
        // 求scale
        AscendC::MicroAPI::Interleave(expMaxRegTensor, tmp16RegTensor, expMaxRegTensor, tmp16RegTensor);
        AscendC::MicroAPI::Cast<float, T, castTraitZero>(
            (AscendC::MicroAPI::RegTensor<float>&)maxFP32RegTensor, (AscendC::MicroAPI::RegTensor<T>&)expMaxRegTensor,
            mask);

        AscendC::MicroAPI::Compare<uint32_t, CMPMODE::LT>(infMask, maxFP32RegTensor, absMaskFP32RegTensor, mask);
        AscendC::MicroAPI::Compare<uint32_t, CMPMODE::NE>(zeroMask, maxFP32RegTensor, fp32ZeroRegTensor, mask);

        AscendC::MicroAPI::Mul(
            (AscendC::MicroAPI::RegTensor<float>&)maxFP32RegTensor,
            (AscendC::MicroAPI::RegTensor<float>&)maxFP32RegTensor,
            (AscendC::MicroAPI::RegTensor<float>&)invMaxRegTensor, mask);
        AscendC::MicroAPI::Duplicate(invMaxRegTensor, MAX_EXP_FOR_FP8);
        // 右移获取指数位
        AscendC::MicroAPI::ShiftRights(expFP32RegTensor, maxFP32RegTensor, SHR_NUM_FOR_FP32, mask);
        // And获取尾数位
        AscendC::MicroAPI::And(manFP32RegTensor, maxFP32RegTensor, manFP32RegTensor, mask);

        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, expFP32RegTensor, NUMBER_ZERO, mask);
        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(p0, expFP32RegTensor, NUMBER_TWO_FIVE_FOUR, p0);
        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, manFP32RegTensor, NUMBER_ZERO, p0);

        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(p1, expFP32RegTensor, NUMBER_ZERO, mask);
        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p1, manFP32RegTensor, NUMBER_HALF, p1);

        AscendC::MicroAPI::MaskOr(p0, p0, p1, mask);
        AscendC::MicroAPI::Adds(extractExpRegTensor, expFP32RegTensor, 1, mask);
        // 根据情况选择指数位是否加一
        AscendC::MicroAPI::Select<uint32_t>(extractExpRegTensor, extractExpRegTensor, expFP32RegTensor, p0);

        AscendC::MicroAPI::Select<uint32_t>(extractExpRegTensor, extractExpRegTensor, invMaxRegTensor, infMask);
        AscendC::MicroAPI::Select<uint32_t>(extractExpRegTensor, extractExpRegTensor, fp32ZeroRegTensor, zeroMask);
        AscendC::MicroAPI::Pack(tmp16RegTensor, extractExpRegTensor);
        AscendC::MicroAPI::Pack(outZero, tmp16RegTensor);
        // 搬出mxScale
        AscendC::MicroAPI::DataCopyUnAlign(mxScaleAddr, outZero, u1, dataLen);
        AscendC::MicroAPI::DataCopyUnAlignPost(mxScaleAddr, u1, 0);

        // 求1/scale
        AscendC::MicroAPI::ShiftLefts(extractExpRegTensor, extractExpRegTensor, SHR_NUM_FOR_FP32, mask);
        AscendC::MicroAPI::Sub(extractExpRegTensor, scaleBias, extractExpRegTensor, mask);
        AscendC::MicroAPI::Select<uint32_t>(extractExpRegTensor, extractExpRegTensor, nanRegTensor, infMask);
        AscendC::MicroAPI::Select<uint32_t>(extractExpRegTensor, extractExpRegTensor, fp32ZeroRegTensor, zeroMask);

        AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitFp32ToBf16>(
            (AscendC::MicroAPI::RegTensor<bfloat16_t>&)reversedShareExpRegTensor,
            (AscendC::MicroAPI::RegTensor<float>&)extractExpRegTensor, mask);
        AscendC::MicroAPI::DeInterleave(
            reversedShareExpRegTensor, tmp16RegTensor, reversedShareExpRegTensor, tmp16RegTensor);
        auto scaleAddr = maxExpAddr;
        for (uint16_t i = 0; i < rowsSingleLoop; i++) {
            AscendC::MicroAPI::DataCopyUnAlign(scaleAddr, reversedShareExpRegTensor, u1, dataLen);
            AscendC::MicroAPI::DataCopyUnAlignPost(scaleAddr, u1, 0);
        }
        AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        AscendC::MicroAPI::DataCopy(reversedShareExpRegTensor, maxExpAddr);
        mask = AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        if constexpr (IsSame<T, half>::value) {
            AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitZero>(
                reversedShareExpRegTensorFP32Zero, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)reversedShareExpRegTensor,
                pnumMask);
            AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitOne>(
                reversedShareExpRegTensorFP32One, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)reversedShareExpRegTensor,
                pnumMask);
        }
        // 求data value
        for (uint16_t i = 0; i <= static_cast<uint16_t>(regLoop - 1); i++) {
            this->template LoadData<RoundMode::UNKNOWN, roundMode, true>(
                xAddr, i * dataLenSingleLoop, xRegTensor, pnumMask);
            if constexpr (IsSame<T, half>::value) {
                AscendC::MicroAPI::Cast<float, T, castTraitZero>(yZero, xRegTensor, pnumMask);
                AscendC::MicroAPI::Cast<float, T, castTraitOne>(yOne, xRegTensor, pnumMask);
                AscendC::MicroAPI::Mul(yZero, yZero, reversedShareExpRegTensorFP32Zero, mask);
                AscendC::MicroAPI::Mul(yOne, yOne, reversedShareExpRegTensorFP32One, mask);
            } else {
                AscendC::MicroAPI::Mul(
                    xBF16RegTensor, xRegTensor, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)reversedShareExpRegTensor,
                    pnumMask);
                AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitZero>(yZero, xBF16RegTensor, mask);
                AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitOne>(yOne, xBF16RegTensor, mask);
            }
            AscendC::MicroAPI::Cast<U, float, castTrait32to8>(yZeroFP8, yZero, mask);
            AscendC::MicroAPI::Cast<U, float, castTrait32to8>(yOneFP8, yOne, mask);
            // 寄存器复用
            AscendC::MicroAPI::Pack(tmp16RegTensor, (AscendC::MicroAPI::RegTensor<uint32_t>&)yZeroFP8);
            AscendC::MicroAPI::Pack(expRegTensor, (AscendC::MicroAPI::RegTensor<uint32_t>&)yOneFP8);
            AscendC::MicroAPI::Interleave(tmp16RegTensor, expRegTensor, tmp16RegTensor, expRegTensor);
            AscendC::MicroAPI::Pack(outZero, tmp16RegTensor);
            auto addr0 = yAddr + i * dataLenSingleLoop;
            AscendC::MicroAPI::DataCopyUnAlign(addr0, outZero, u1, dataLenSingleLoop);
            AscendC::MicroAPI::DataCopyUnAlignPost(addr0, u1, 0);
        }
    }
}

} // namespace DynamicMxQuant
#endif // DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_OPTIMIZE_FP8_H
