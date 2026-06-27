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
    uint16_t loopSize = static_cast<uint16_t>(
        DIGIT_SIXTY_THREE -
        AscendC::ScalarCountLeadingZero(static_cast<uint64_t>(rowsSingleLoop))); // 求最大指数行的二分次数
    uint16_t rows = 1 << loopSize;                                               // 最接近rowsSingleLoop的2次方数
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
        if constexpr (IsSame<T, float>::value) {
            return;
        }
        Reg::RegTensor<T> x;
        Reg::RegTensor<bfloat16_t> xBF16RegTensor;
        Reg::RegTensor<uint16_t> exp;
        Reg::RegTensor<uint16_t> expMax;
        Reg::RegTensor<uint16_t> maxEle;
        Reg::RegTensor<uint16_t> fp8Nan;
        Reg::RegTensor<uint16_t> fp8MaxExpRegTensor;
        Reg::RegTensor<uint16_t> shareExpRegTensor;
        Reg::RegTensor<uint16_t> mxScaleInt;
        Reg::RegTensor<uint8_t> mxScale;
        Reg::RegTensor<uint16_t> specialExp;
        Reg::RegTensor<uint16_t> reversedShareExpRegTensor;
        Reg::RegTensor<float> reversedShareExpRegTensorFP32Zero;
        Reg::RegTensor<float> reversedShareExpRegTensorFP32One;
        Reg::RegTensor<uint16_t> bias;
        Reg::RegTensor<uint16_t> zero;
        Reg::RegTensor<uint16_t> nan;
        Reg::RegTensor<uint8_t> outZero;
        Reg::RegTensor<uint8_t> outOne;
        Reg::RegTensor<uint16_t> yRegTensorZero;
        Reg::RegTensor<uint16_t> yRegTensorOne;
        Reg::RegTensor<float> yZero;
        Reg::RegTensor<float> yOne;
        Reg::RegTensor<U> yZeroFP8;
        Reg::RegTensor<U> yOneFP8;
        Reg::RegTensor<bfloat16_t> valueRegTensor;
        Reg::RegTensor<uint16_t> invalidMaskFp16;
        Reg::RegTensor<uint16_t> xSelectRegTensor;
        Reg::UnalignReg u0;
        Reg::UnalignReg u1;
        Reg::MaskReg zeroMask;
        Reg::MaskReg infMask;
        Reg::MaskReg invalidDataMask;
        Reg::MaskReg specialDataMask;
        Reg::MaskReg maskAll = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        static constexpr Reg::CastTrait castTraitZero = {
            Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitOne = {
            Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTrait32to8 = {
            Reg::RegLayout::ZERO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {
            Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
        Reg::Duplicate(fp8MaxExpRegTensor, FP8_BF16_MAX_EXP);
        Reg::Duplicate(maxEle, BF16_MAX_EXP);
        Reg::Duplicate(nan, BF16_NAN_CUSTOM);
        Reg::Duplicate(fp8Nan, FP8_DEFAULT_MAX_EXP);
        Reg::Duplicate(bias, BF16_EXP_BIAS);
        Reg::Duplicate(zero, 0);
        Reg::Duplicate(specialExp, BF16_SPECIAL_EXP_THRESHOLD);
        Reg::Duplicate(invalidMaskFp16, FP16_INVALID);

        uint32_t pnum = dataLenSingleLoop;
        uint32_t tailPnum = dataLenTailLoop;
        Reg::MaskReg pnumMask = Reg::UpdateMask<uint16_t>(pnum);
        Reg::MaskReg tailPnumMask = Reg::UpdateMask<uint16_t>(tailPnum);
        Reg::Duplicate(expMax, 0);
        for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); i++) {
            this->template LoadData<RoundMode::CAST_TRUNC, roundMode>(xAddr, i * dataLenSingleLoop, x, pnumMask);
            if constexpr (IsSame<T, half>::value) {
                Reg::And(xSelectRegTensor, (Reg::RegTensor<uint16_t>&)x, invalidMaskFp16, pnumMask);
                Reg::Compare<uint16_t, CMPMODE::NE>(invalidDataMask, xSelectRegTensor, invalidMaskFp16, pnumMask);
                Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(xBF16RegTensor, x, pnumMask);
                Reg::And(exp, (Reg::RegTensor<uint16_t>&)xBF16RegTensor, maxEle, pnumMask);
                Reg::Select<uint16_t>(exp, exp, maxEle, invalidDataMask);
            } else if constexpr (IsSame<T, bfloat16_t>::value) {
                Reg::And(exp, (Reg::RegTensor<uint16_t>&)x, maxEle, pnumMask);
            }
            Reg::Max(expMax, expMax, exp, pnumMask);
        }
        this->template LoadData<RoundMode::CAST_TRUNC, roundMode>(
            xAddr, (regLoop - 1) * dataLenSingleLoop, x, tailPnumMask);
        if constexpr (IsSame<T, half>::value) {
            Reg::And(xSelectRegTensor, (Reg::RegTensor<uint16_t>&)x, invalidMaskFp16, tailPnumMask);
            Reg::Compare<uint16_t, CMPMODE::NE>(invalidDataMask, xSelectRegTensor, invalidMaskFp16, tailPnumMask);
            Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(xBF16RegTensor, x, tailPnumMask);
            Reg::And(exp, (Reg::RegTensor<uint16_t>&)xBF16RegTensor, maxEle, tailPnumMask);
            Reg::Select<uint16_t>(exp, exp, maxEle, invalidDataMask);
        } else if constexpr (IsSame<T, bfloat16_t>::value) {
            Reg::And(exp, (Reg::RegTensor<uint16_t>&)x, maxEle, tailPnumMask);
        }
        Reg::Max(exp, expMax, exp, tailPnumMask);
        Reg::Copy<uint16_t, Reg::MaskMergeMode::MERGING>(expMax, exp, tailPnumMask);
        // 二分法求rowsSingleLoop行中的最大行
        Reg::DataCopy(maxExpAddr, expMax, pnumMask);
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        uint32_t maskNum = dataLenSingleLoop - expOffset;
        Reg::MaskReg mask = Reg::UpdateMask<uint16_t>(maskNum);
        Reg::DataCopyUnAlignPre(u0, maxExpAddr);
        Reg::DataCopyUnAlign(expMax, u0, maxExpAddr);
        Reg::DataCopyUnAlignPre(u0, maxExpAddr + expOffset);
        Reg::DataCopyUnAlign(exp, u0, maxExpAddr + expOffset);
        Reg::Max(exp, expMax, exp, mask);
        Reg::Copy<uint16_t, Reg::MaskMergeMode::MERGING>(expMax, exp, mask);
        for (uint16_t i = 0; i < loopSize; i++) {
            Reg::DataCopy(maxExpAddr, expMax, pnumMask);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            expOffset /= DIGIT_TWO;
            maskNum = expOffset;
            mask = Reg::UpdateMask<uint16_t>(maskNum);
            Reg::DataCopyUnAlignPre(u0, maxExpAddr);
            Reg::DataCopyUnAlign(expMax, u0, maxExpAddr);
            Reg::DataCopyUnAlignPre(u0, maxExpAddr + expOffset);
            Reg::DataCopyUnAlign(exp, u0, maxExpAddr + expOffset);
            Reg::Max(expMax, expMax, exp, mask);
        }
        maskNum = static_cast<uint32_t>(dataLen);
        mask = Reg::UpdateMask<uint16_t>(maskNum);
        Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expMax, maxEle, mask);
        Reg::Compare<uint16_t, CMPMODE::LT>(invalidDataMask, expMax, fp8MaxExpRegTensor, mask);
        Reg::Select<uint16_t>(expMax, fp8MaxExpRegTensor, expMax, invalidDataMask);
        Reg::Sub(expMax, expMax, fp8MaxExpRegTensor, mask);
        Reg::ShiftRights(mxScaleInt, expMax, BF16_SHR_NUM, mask);
        Reg::Select<uint16_t>(mxScaleInt, mxScaleInt, fp8Nan, infMask);
        Reg::Pack(mxScale, mxScaleInt);
        Reg::DataCopyUnAlign(mxScaleAddr, mxScale, u1, dataLen);
        Reg::DataCopyUnAlignPost(mxScaleAddr, u1, 0);

        // 求1/scale
        Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMax, zero, mask);
        Reg::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, expMax, bias, mask);
        Reg::Sub(reversedShareExpRegTensor, bias, expMax, mask);
        Reg::Select<uint16_t>(reversedShareExpRegTensor, reversedShareExpRegTensor, nan, infMask);
        Reg::Select<uint16_t>(reversedShareExpRegTensor, reversedShareExpRegTensor, zero, zeroMask);
        Reg::Select<uint16_t>(reversedShareExpRegTensor, specialExp, reversedShareExpRegTensor, specialDataMask);

        auto scaleAddr = maxExpAddr;
        for (uint16_t i = 0; i < rowsSingleLoop; i++) {
            Reg::DataCopyUnAlign(scaleAddr, reversedShareExpRegTensor, u1, dataLen);
            Reg::DataCopyUnAlignPost(scaleAddr, u1, 0);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        Reg::DataCopy(reversedShareExpRegTensor, maxExpAddr);
        // 求data value
        for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); i++) {
            this->template LoadData<toBf16RoundMode, roundMode, true>(xAddr, i * dataLenSingleLoop, x, pnumMask);
            if constexpr (IsSame<T, half>::value) {
                Reg::Cast<float, T, castTraitZero>(yZero, x, pnumMask);
                Reg::Cast<float, T, castTraitOne>(yOne, x, pnumMask);
                Reg::Cast<float, bfloat16_t, castTraitZero>(
                    reversedShareExpRegTensorFP32Zero, (Reg::RegTensor<bfloat16_t>&)reversedShareExpRegTensor,
                    pnumMask);
                Reg::Cast<float, bfloat16_t, castTraitOne>(
                    reversedShareExpRegTensorFP32One, (Reg::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, pnumMask);
                Reg::Mul(yZero, yZero, reversedShareExpRegTensorFP32Zero, maskAll);
                Reg::Mul(yOne, yOne, reversedShareExpRegTensorFP32One, maskAll);
            } else if constexpr (IsSame<T, bfloat16_t>::value) {
                Reg::Mul(valueRegTensor, x, (Reg::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, pnumMask);
                Reg::Cast<float, bfloat16_t, castTraitZero>(yZero, valueRegTensor, maskAll);
                Reg::Cast<float, bfloat16_t, castTraitOne>(yOne, valueRegTensor, maskAll);
            }
            Reg::Interleave(yZero, yOne, yZero, yOne);
            Reg::Cast<U, float, castTrait32to8>(yZeroFP8, yZero, maskAll);
            Reg::Cast<U, float, castTrait32to8>(yOneFP8, yOne, maskAll);
            Reg::Pack(yRegTensorZero, (Reg::RegTensor<uint32_t>&)yZeroFP8);
            Reg::Pack(outZero, yRegTensorZero);
            Reg::Pack(yRegTensorOne, (Reg::RegTensor<uint32_t>&)yOneFP8);
            Reg::Pack(outOne, yRegTensorOne);
            auto addr0 = yAddr + i * dataLenSingleLoop;
            Reg::DataCopyUnAlign(addr0, outZero, u1, loopNum0);
            Reg::DataCopyUnAlignPost(addr0, u1, 0);
            auto addr1 = yAddr + i * dataLenSingleLoop + loopNum0;
            Reg::DataCopyUnAlign(addr1, outOne, u1, loopNum1);
            Reg::DataCopyUnAlignPost(addr1, u1, 0);
        }
        this->template LoadData<toBf16RoundMode, roundMode, true>(
            xAddr, (regLoop - 1) * dataLenSingleLoop, x, tailPnumMask);
        if constexpr (IsSame<T, half>::value) {
            Reg::Cast<float, T, castTraitZero>(yZero, x, tailPnumMask);
            Reg::Cast<float, T, castTraitOne>(yOne, x, tailPnumMask);
            Reg::Cast<float, bfloat16_t, castTraitZero>(
                reversedShareExpRegTensorFP32Zero, (Reg::RegTensor<bfloat16_t>&)reversedShareExpRegTensor,
                tailPnumMask);
            Reg::Cast<float, bfloat16_t, castTraitOne>(
                reversedShareExpRegTensorFP32One, (Reg::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, tailPnumMask);
            Reg::Mul(yZero, yZero, reversedShareExpRegTensorFP32Zero, maskAll);
            Reg::Mul(yOne, yOne, reversedShareExpRegTensorFP32One, maskAll);
        } else if constexpr (IsSame<T, bfloat16_t>::value) {
            Reg::Mul(valueRegTensor, x, (Reg::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, tailPnumMask);
            Reg::Cast<float, bfloat16_t, castTraitZero>(yZero, valueRegTensor, maskAll);
            Reg::Cast<float, bfloat16_t, castTraitOne>(yOne, valueRegTensor, maskAll);
        }
        Reg::Interleave(yZero, yOne, yZero, yOne);
        Reg::Cast<U, float, castTrait32to8>(yZeroFP8, yZero, maskAll);
        Reg::Cast<U, float, castTrait32to8>(yOneFP8, yOne, maskAll);
        Reg::Pack(yRegTensorZero, (Reg::RegTensor<uint32_t>&)yZeroFP8);
        Reg::Pack(outZero, yRegTensorZero);
        Reg::Pack(yRegTensorOne, (Reg::RegTensor<uint32_t>&)yOneFP8);
        Reg::Pack(outOne, yRegTensorOne);
        auto addr0 = yAddr + (regLoop - 1) * dataLenSingleLoop;
        Reg::DataCopyUnAlign(addr0, outZero, u1, tailLoopNum0);
        Reg::DataCopyUnAlignPost(addr0, u1, 0);
        auto addr1 = yAddr + (regLoop - 1) * dataLenSingleLoop + tailLoopNum0;
        Reg::DataCopyUnAlign(addr1, outOne, u1, tailLoopNum1);
        Reg::DataCopyUnAlignPost(addr1, u1, 0);
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
    uint16_t loopSize = static_cast<uint16_t>(
        DIGIT_SIXTY_THREE -
        AscendC::ScalarCountLeadingZero(static_cast<uint64_t>(rowsSingleLoop))); // 求最大指数行的二分次数
    uint16_t rows = 1 << loopSize;                                               // 最接近rowsSingleLoop的2次方数
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
        if constexpr (IsSame<T, float>::value) {
            return;
        }
        Reg::RegTensor<T> x;
        Reg::RegTensor<bfloat16_t> xBF16RegTensor;

        Reg::RegTensor<uint32_t> absMaskFP32RegTensor;
        Reg::RegTensor<uint32_t> maxFP32RegTensor;
        Reg::RegTensor<uint32_t> invMaxRegTensor;
        Reg::RegTensor<uint32_t> fp32ZeroRegTensor;
        // 指数位
        Reg::RegTensor<uint32_t> expFP32RegTensor;
        // 尾数位
        Reg::RegTensor<uint32_t> manFP32RegTensor;
        Reg::RegTensor<uint32_t> scaleBias;
        Reg::RegTensor<uint32_t> extractExpRegTensor;
        Reg::RegTensor<uint16_t> tmp16RegTensor;
        Reg::MaskReg p0;
        Reg::MaskReg p1;
        Reg::MaskReg zeroMask;
        Reg::MaskReg infMask;

        Reg::RegTensor<uint16_t> exp;
        Reg::RegTensor<uint16_t> expMax;
        Reg::RegTensor<uint16_t> reversedShareExpRegTensor;
        Reg::RegTensor<float> reversedShareExpRegTensorFP32Zero;
        Reg::RegTensor<float> reversedShareExpRegTensorFP32One;
        Reg::RegTensor<uint32_t> nan;
        Reg::RegTensor<uint8_t> outZero;
        Reg::RegTensor<float> yZero;
        Reg::RegTensor<float> yOne;
        Reg::RegTensor<U> yZeroFP8;
        Reg::RegTensor<U> yOneFP8;
        Reg::UnalignReg u0;
        Reg::UnalignReg u1;

        static constexpr Reg::CastTrait castTraitZero = {
            Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitOne = {
            Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTrait32to8 = {
            Reg::RegLayout::ZERO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {
            Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
        static constexpr Reg::CastTrait castTraitFp32ToBf16 = {
            Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, toBf16RoundMode};
        Reg::Duplicate(nan, FP32_NAN_CUSTOM);
        Reg::Duplicate(tmp16RegTensor, BF16_MAX);
        Reg::Duplicate(absMaskFP32RegTensor, FP32_MX_MAX_EXP);
        Reg::Duplicate(invMaxRegTensor, INV_DTYPE_MAX);
        Reg::Duplicate(fp32ZeroRegTensor, 0);
        Reg::Duplicate(manFP32RegTensor, FP32_MX_MAN_MASK);
        Reg::Duplicate(scaleBias, FP32_MX_EXP_BIAS);

        uint32_t pnum = dataLenSingleLoop;
        uint32_t tailPnum = dataLenTailLoop;
        Reg::MaskReg pnumMask = Reg::UpdateMask<uint16_t>(pnum);
        Reg::MaskReg tailPnumMask = Reg::UpdateMask<uint16_t>(tailPnum);
        Reg::Duplicate(expMax, 0);
        for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); i++) {
            this->template LoadData<RoundMode::UNKNOWN, roundMode>(xAddr, i * dataLenSingleLoop, x, pnumMask);
            Reg::And(exp, (Reg::RegTensor<uint16_t>&)x, tmp16RegTensor, pnumMask);
            Reg::Max(expMax, expMax, exp, pnumMask);
        }
        this->template LoadData<RoundMode::UNKNOWN, roundMode>(
            xAddr, (regLoop - 1) * dataLenSingleLoop, x, tailPnumMask);
        Reg::And(exp, (Reg::RegTensor<uint16_t>&)x, tmp16RegTensor, tailPnumMask);
        Reg::Max(exp, expMax, exp, tailPnumMask);
        Reg::Copy<uint16_t, Reg::MaskMergeMode::MERGING>(expMax, exp, tailPnumMask);

        // 二分法求rowsSingleLoop行中的最大行
        Reg::DataCopy(maxExpAddr, expMax, pnumMask);
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        uint32_t maskNum = dataLenSingleLoop - expOffset;
        Reg::MaskReg mask = Reg::UpdateMask<uint16_t>(maskNum);
        Reg::DataCopyUnAlignPre(u0, maxExpAddr + expOffset);
        Reg::DataCopyUnAlign(exp, u0, maxExpAddr + expOffset);
        Reg::Max(exp, expMax, exp, mask);
        Reg::Copy<uint16_t, Reg::MaskMergeMode::MERGING>(expMax, exp, mask);
        for (uint16_t i = 0; i < loopSize; i++) {
            Reg::DataCopy(maxExpAddr, expMax, pnumMask);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            expOffset /= DIGIT_TWO;
            maskNum = expOffset;
            mask = Reg::UpdateMask<uint16_t>(maskNum);
            Reg::DataCopyUnAlignPre(u0, maxExpAddr + expOffset);
            Reg::DataCopyUnAlign(exp, u0, maxExpAddr + expOffset);
            Reg::Max(expMax, expMax, exp, mask);
        }
        maskNum = static_cast<uint32_t>(dataLen);
        mask = Reg::CreateMask<uint32_t>();
        maskNum = static_cast<uint32_t>(dataLen);
        // 求scale
        Reg::Interleave(expMax, tmp16RegTensor, expMax, tmp16RegTensor);
        Reg::Cast<float, T, castTraitZero>((Reg::RegTensor<float>&)maxFP32RegTensor, (Reg::RegTensor<T>&)expMax, mask);

        Reg::Compare<uint32_t, CMPMODE::LT>(infMask, maxFP32RegTensor, absMaskFP32RegTensor, mask);
        Reg::Compare<uint32_t, CMPMODE::NE>(zeroMask, maxFP32RegTensor, fp32ZeroRegTensor, mask);

        Reg::Maxs((Reg::RegTensor<float>&)maxFP32RegTensor, (Reg::RegTensor<float>&)maxFP32RegTensor, this->maxLowBound_, mask);

        Reg::Mul(
            (Reg::RegTensor<float>&)maxFP32RegTensor, (Reg::RegTensor<float>&)maxFP32RegTensor,
            (Reg::RegTensor<float>&)invMaxRegTensor, mask);
        Reg::Duplicate(invMaxRegTensor, FP8_DEFAULT_MAX_EXP);
        // 右移获取指数位
        Reg::ShiftRights(expFP32RegTensor, maxFP32RegTensor, FP32_SHR_NUM, mask);
        // And获取尾数位
        Reg::And(manFP32RegTensor, maxFP32RegTensor, manFP32RegTensor, mask);

        Reg::CompareScalar<uint32_t, CMPMODE::GT>(p0, expFP32RegTensor, FP32_NUMBER_ZERO, mask);
        Reg::CompareScalar<uint32_t, CMPMODE::LT>(p0, expFP32RegTensor, FP32_NUMBER_254, p0);
        Reg::CompareScalar<uint32_t, CMPMODE::GT>(p0, manFP32RegTensor, FP32_NUMBER_ZERO, p0);

        Reg::CompareScalar<uint32_t, CMPMODE::EQ>(p1, expFP32RegTensor, FP32_NUMBER_ZERO, mask);
        Reg::CompareScalar<uint32_t, CMPMODE::GT>(p1, manFP32RegTensor, FP32_NUMBER_HALF, p1);

        Reg::MaskOr(p0, p0, p1, mask);
        Reg::Adds(extractExpRegTensor, expFP32RegTensor, 1, mask);
        // 根据情况选择指数位是否加一
        Reg::Select<uint32_t>(extractExpRegTensor, extractExpRegTensor, expFP32RegTensor, p0);

        Reg::Select<uint32_t>(extractExpRegTensor, extractExpRegTensor, invMaxRegTensor, infMask);
        Reg::Select<uint32_t>(extractExpRegTensor, extractExpRegTensor, fp32ZeroRegTensor, zeroMask);
        Reg::Pack(tmp16RegTensor, extractExpRegTensor);
        Reg::Pack(outZero, tmp16RegTensor);
        // 搬出mxScale
        Reg::DataCopyUnAlign(mxScaleAddr, outZero, u1, dataLen);
        Reg::DataCopyUnAlignPost(mxScaleAddr, u1, 0);

        // 求1/scale
        Reg::ShiftLefts(extractExpRegTensor, extractExpRegTensor, FP32_SHR_NUM, mask);
        Reg::Sub(extractExpRegTensor, scaleBias, extractExpRegTensor, mask);
        Reg::Select<uint32_t>(extractExpRegTensor, extractExpRegTensor, nan, infMask);
        Reg::Select<uint32_t>(extractExpRegTensor, extractExpRegTensor, fp32ZeroRegTensor, zeroMask);

        Reg::Cast<bfloat16_t, float, castTraitFp32ToBf16>(
            (Reg::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, (Reg::RegTensor<float>&)extractExpRegTensor, mask);
        Reg::DeInterleave(reversedShareExpRegTensor, tmp16RegTensor, reversedShareExpRegTensor, tmp16RegTensor);
        auto scaleAddr = maxExpAddr;
        for (uint16_t i = 0; i < rowsSingleLoop; i++) {
            Reg::DataCopyUnAlign(scaleAddr, reversedShareExpRegTensor, u1, dataLen);
            Reg::DataCopyUnAlignPost(scaleAddr, u1, 0);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        Reg::DataCopy(reversedShareExpRegTensor, maxExpAddr);
        mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        if constexpr (IsSame<T, half>::value) {
            Reg::Cast<float, bfloat16_t, castTraitZero>(
                reversedShareExpRegTensorFP32Zero, (Reg::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, pnumMask);
            Reg::Cast<float, bfloat16_t, castTraitOne>(
                reversedShareExpRegTensorFP32One, (Reg::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, pnumMask);
        }
        // 求data value
        for (uint16_t i = 0; i <= static_cast<uint16_t>(regLoop - 1); i++) {
            this->template LoadData<RoundMode::UNKNOWN, roundMode, true>(xAddr, i * dataLenSingleLoop, x, pnumMask);
            if constexpr (IsSame<T, half>::value) {
                Reg::Cast<float, T, castTraitZero>(yZero, x, pnumMask);
                Reg::Cast<float, T, castTraitOne>(yOne, x, pnumMask);
                Reg::Mul(yZero, yZero, reversedShareExpRegTensorFP32Zero, mask);
                Reg::Mul(yOne, yOne, reversedShareExpRegTensorFP32One, mask);
            } else if constexpr (IsSame<T, bfloat16_t>::value) {
                Reg::Mul(xBF16RegTensor, x, (Reg::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, pnumMask);
                Reg::Cast<float, bfloat16_t, castTraitZero>(yZero, xBF16RegTensor, mask);
                Reg::Cast<float, bfloat16_t, castTraitOne>(yOne, xBF16RegTensor, mask);
            }
            Reg::Cast<U, float, castTrait32to8>(yZeroFP8, yZero, mask);
            Reg::Cast<U, float, castTrait32to8>(yOneFP8, yOne, mask);
            // 寄存器复用
            Reg::Pack(tmp16RegTensor, (Reg::RegTensor<uint32_t>&)yZeroFP8);
            Reg::Pack(exp, (Reg::RegTensor<uint32_t>&)yOneFP8);
            Reg::Interleave(tmp16RegTensor, exp, tmp16RegTensor, exp);
            Reg::Pack(outZero, tmp16RegTensor);
            auto addr0 = yAddr + i * dataLenSingleLoop;
            Reg::DataCopyUnAlign(addr0, outZero, u1, dataLenSingleLoop);
            Reg::DataCopyUnAlignPost(addr0, u1, 0);
        }
    }
}

} // namespace DynamicMxQuant
#endif // DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_OPTIMIZE_FP8_H
