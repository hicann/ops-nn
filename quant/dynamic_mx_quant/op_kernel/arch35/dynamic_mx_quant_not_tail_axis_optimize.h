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
 * \file dynamic_mx_quant_not_tail_axis_optimize.h
 * \brief
 */

#ifndef DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_OPTIMIZE_H
#define DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_OPTIMIZE_H

#include "dynamic_mx_quant_not_tail_axis_base.h"
#include "op_kernel/math_util.h"

namespace DynamicMxQuant {
using namespace AscendC;

template <typename T, typename U, const bool ISTAIL>
class DynamicMxQuantNotTailAxisOptimize : public DynamicMxQuantBase<T, U, ISTAIL> {
public:
    __aicore__ inline DynamicMxQuantNotTailAxisOptimize(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR workspace, const DynamicMxQuantTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SplitPreAxisCompute(int64_t ubFactor, int64_t blockSizeIdx);
    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode, const int64_t calcMode>
    __aicore__ inline void ComputeOcp(
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
    int64_t calcMode_ = 0;
    uint32_t subNumForFP32Scale_ = 0;
    uint16_t subNumForBF16Scale_ = 0;
    using calcType = typename std::conditional<IsSame<T, half>::value, float, T>::type;
    using calcTypeInt = typename std::conditional<IsSame<T, half>::value, uint32_t, uint16_t>::type;
};

template <typename T, typename U, const bool ISTAIL>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimize<T, U, ISTAIL>::Init(
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

    calcMode_ = tilingData->calcMode;
    subNumForBF16Scale_ = static_cast<uint16_t>(tilingData->subNumForScale);
    subNumForFP32Scale_ = tilingData->subNumForFP16Scale;

    this->xGm_.SetGlobalBuffer((__gm__ T*)(x) + blockOffset_);
    this->mxScaleGm_.SetGlobalBuffer((__gm__ uint8_t*)(mxScale) + scaleBlockOffset_);
    this->workspaceGm_.SetGlobalBuffer((__gm__ uint8_t*)(workspace) + scaleBlockOffset_);
    this->yGm_.SetGlobalBuffer((__gm__ uint8_t*)(y) + blockOffset_ / DIGIT_TWO);

    bufferSize_ = Ops::Base::CeilAlign(bufferSize_, static_cast<int64_t>(Ops::Base::GetUbBlockSize()));
    this->pipe_.InitBuffer(this->inQueue_, DB_BUFFER, bufferSize_);
    this->pipe_.InitBuffer(this->mxScaleQueue_, DB_BUFFER, bufferSize_);
    this->pipe_.InitBuffer(this->outQueue_, DB_BUFFER, bufferSize_);
    this->pipe_.InitBuffer(maxExpBuf_, Ops::Base::GetVRegSize());
}

template <typename T, typename U, const bool ISTAIL>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimize<T, U, ISTAIL>::Process()
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

template <typename T, typename U, const bool ISTAIL>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimize<T, U, ISTAIL>::SplitPreAxisCompute(
    int64_t ubFactor, int64_t blockSizeIdx)
{
    LocalTensor<T> x = this->inQueue_.template DeQue<T>();
    LocalTensor<uint8_t> mxScale = this->mxScaleQueue_.template AllocTensor<uint8_t>();
    LocalTensor<uint8_t> y = this->outQueue_.template AllocTensor<uint8_t>();

    int64_t offset = 0;
    for (int64_t i = 0; i < ubFactor; i++) {
        auto xAddr = (__ubuf__ T*)x.GetPhyAddr() + offset;
        auto mxScaleAddr = (__ubuf__ uint8_t*)mxScale.GetPhyAddr() + i * this->postAxisSize_;
        auto yAddr = (__ubuf__ uint8_t*)y.GetPhyAddr() + offset / DIGIT_TWO;
        int64_t blockCount = this->BlockCountInCurCompute(blockSizeIdx + i + 1);
        offset += blockCount * this->postAxisSize_;
        if (calcMode_ == ModeZero) {
            if (this->roundMode_ == MODE_RINT) {
                ComputeOcp<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT, ModeZero>(
                    this->postAxisSize_, blockCount, xAddr, mxScaleAddr, yAddr);
            } else if (this->roundMode_ == MODE_FLOOR) {
                ComputeOcp<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR, ModeZero>(
                    this->postAxisSize_, blockCount, xAddr, mxScaleAddr, yAddr);
            } else if (this->roundMode_ == MODE_ROUND) {
                ComputeOcp<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND, ModeZero>(
                    this->postAxisSize_, blockCount, xAddr, mxScaleAddr, yAddr);
            }
        } else if (calcMode_ == ModeTwo) {
            if (this->roundMode_ == MODE_RINT) {
                ComputeOcp<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT, ModeTwo>(
                    this->postAxisSize_, blockCount, xAddr, mxScaleAddr, yAddr);
            } else if (this->roundMode_ == MODE_FLOOR) {
                ComputeOcp<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR, ModeTwo>(
                    this->postAxisSize_, blockCount, xAddr, mxScaleAddr, yAddr);
            } else if (this->roundMode_ == MODE_ROUND) {
                ComputeOcp<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND, ModeTwo>(
                    this->postAxisSize_, blockCount, xAddr, mxScaleAddr, yAddr);
            }
        } else {
            if (this->roundMode_ == MODE_RINT) {
                ComputeCuBLAS<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(
                    this->postAxisSize_, blockCount, xAddr, mxScaleAddr, yAddr);
            } else if (this->roundMode_ == MODE_FLOOR) {
                ComputeCuBLAS<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(
                    this->postAxisSize_, blockCount, xAddr, mxScaleAddr, yAddr);
            } else if (this->roundMode_ == MODE_ROUND) {
                ComputeCuBLAS<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(
                    this->postAxisSize_, blockCount, xAddr, mxScaleAddr, yAddr);
            }
        }
    }
    this->inQueue_.template FreeTensor(x);
    this->mxScaleQueue_.template EnQue(mxScale);
    this->outQueue_.template EnQue(y);
}

template <typename T, typename U, const bool ISTAIL>
template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimize<T, U, ISTAIL>::ComputeCuBLAS(
    int64_t dataLen, int64_t blockCount, __ubuf__ T* xAddr, __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint8_t* yAddr)
{
    static constexpr AscendC::MicroAPI::CastTrait castTraitZero = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    constexpr uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(calcType); // 寄存器单次处理能处理的长度
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

    LocalTensor<calcTypeInt> maxExpTensor = maxExpBuf_.Get<calcTypeInt>();
    auto maxExpAddr = (__ubuf__ calcTypeInt*)maxExpTensor.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<calcType> xReg;
        AscendC::MicroAPI::RegTensor<calcTypeInt> absReg;
        AscendC::MicroAPI::RegTensor<calcTypeInt> xMaxReg;
        AscendC::MicroAPI::RegTensor<uint32_t> xFP32MaxReg;
        AscendC::MicroAPI::RegTensor<uint32_t> expFP32Reg;    // 指数
        AscendC::MicroAPI::RegTensor<uint32_t> manFP32Reg;    // 尾数
        AscendC::MicroAPI::RegTensor<uint32_t> extractExpReg; // 指数 + 1
        AscendC::MicroAPI::RegTensor<uint16_t> expBF16Reg;
        AscendC::MicroAPI::RegTensor<uint8_t> mxScale;
        AscendC::MicroAPI::RegTensor<uint8_t> out;
        AscendC::MicroAPI::RegTensor<calcTypeInt> scaleReprocal; // 1/scale

        AscendC::MicroAPI::RegTensor<calcTypeInt> absForXReg;
        AscendC::MicroAPI::RegTensor<calcTypeInt> zeroReg;
        AscendC::MicroAPI::RegTensor<calcTypeInt> biasReg;
        AscendC::MicroAPI::RegTensor<calcTypeInt> infForXReg;
        AscendC::MicroAPI::RegTensor<uint32_t> manForFP32Reg;
        AscendC::MicroAPI::RegTensor<float> dstTypeMaxReg;
        AscendC::MicroAPI::RegTensor<calcTypeInt> nanRegTensor;
        AscendC::MicroAPI::RegTensor<calcTypeInt> specialExpRegTensor;

        AscendC::MicroAPI::UnalignReg u0;
        AscendC::MicroAPI::UnalignReg u1;
        AscendC::MicroAPI::MaskReg zeroMask;
        AscendC::MicroAPI::MaskReg infMask;
        AscendC::MicroAPI::MaskReg specialDataMask;
        AscendC::MicroAPI::MaskReg p0;
        AscendC::MicroAPI::MaskReg mask32;
        AscendC::MicroAPI::MaskReg mask16 =
            AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();

        AscendC::MicroAPI::Duplicate(dstTypeMaxReg, this->invDstTypeMax_);
        AscendC::MicroAPI::Duplicate(biasReg, this->maxBias_);
        AscendC::MicroAPI::Duplicate(absForXReg, this->absForX_);
        AscendC::MicroAPI::Duplicate(infForXReg, this->maxExp_);
        AscendC::MicroAPI::Duplicate(manForFP32Reg, MAN_MASK_FLOAT);
        AscendC::MicroAPI::Duplicate(nanRegTensor, this->nanValue_);
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, this->specialExp_);

        uint32_t pnum = dataLenSingleLoop;
        uint32_t tailPnum = dataLenTailLoop;
        AscendC::MicroAPI::MaskReg pnumMask = AscendC::MicroAPI::UpdateMask<calcTypeInt>(pnum);
        AscendC::MicroAPI::MaskReg tailPnumMask = AscendC::MicroAPI::UpdateMask<calcTypeInt>(tailPnum);
        AscendC::MicroAPI::Duplicate(xMaxReg, 0);
        for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); i++) {
            this->template LoadData<calcType>(xAddr, i * dataLenSingleLoop, xReg, pnumMask);
            AscendC::MicroAPI::And(absReg, (AscendC::MicroAPI::RegTensor<calcTypeInt>&)xReg, absForXReg, pnumMask);
            AscendC::MicroAPI::Max(xMaxReg, xMaxReg, absReg, pnumMask);
        }
        this->template LoadData<calcType>(xAddr, (regLoop - 1) * dataLenSingleLoop, xReg, tailPnumMask);
        AscendC::MicroAPI::And(absReg, (AscendC::MicroAPI::RegTensor<calcTypeInt>&)xReg, absForXReg, tailPnumMask);
        AscendC::MicroAPI::Max(absReg, xMaxReg, absReg, tailPnumMask);
        AscendC::MicroAPI::Copy<calcTypeInt, AscendC::MicroAPI::MaskMergeMode::MERGING>(xMaxReg, absReg, tailPnumMask);
        // 二分法求rowsSingleLoop行中的最大行
        AscendC::MicroAPI::DataCopy(maxExpAddr, xMaxReg, pnumMask);
        AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        uint32_t maskNum = dataLenSingleLoop - expOffset;
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::UpdateMask<calcTypeInt>(maskNum);
        AscendC::MicroAPI::DataCopyUnAlignPre(u0, maxExpAddr);
        AscendC::MicroAPI::DataCopyUnAlign(xMaxReg, u0, maxExpAddr);
        AscendC::MicroAPI::DataCopyUnAlignPre(u0, maxExpAddr + expOffset);
        AscendC::MicroAPI::DataCopyUnAlign(absReg, u0, maxExpAddr + expOffset);
        AscendC::MicroAPI::Max(absReg, xMaxReg, absReg, mask);
        AscendC::MicroAPI::Copy<calcTypeInt, AscendC::MicroAPI::MaskMergeMode::MERGING>(xMaxReg, absReg, mask);
        for (uint16_t i = 0; i < loopSize; i++) {
            AscendC::MicroAPI::DataCopy(maxExpAddr, xMaxReg, pnumMask);
            AscendC::MicroAPI::LocalMemBar<
                AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
            expOffset /= DIGIT_TWO;
            maskNum = expOffset;
            mask = AscendC::MicroAPI::UpdateMask<calcTypeInt>(maskNum);
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, maxExpAddr);
            AscendC::MicroAPI::DataCopyUnAlign(xMaxReg, u0, maxExpAddr);
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, maxExpAddr + expOffset);
            AscendC::MicroAPI::DataCopyUnAlign(absReg, u0, maxExpAddr + expOffset);
            AscendC::MicroAPI::Max(xMaxReg, xMaxReg, absReg, mask);
        }
        // 求scale
        maskNum = static_cast<uint32_t>(dataLen);
        mask32 = AscendC::MicroAPI::UpdateMask<uint32_t>(maskNum);
        if constexpr (IsSame<T, bfloat16_t>::value) {
            AscendC::MicroAPI::Interleave(xMaxReg, zeroReg, xMaxReg, zeroReg);
            AscendC::MicroAPI::Cast<float, T, castTraitZero>(
                (AscendC::MicroAPI::RegTensor<float>&)xFP32MaxReg, (AscendC::MicroAPI::RegTensor<T>&)xMaxReg, mask16);
            AscendC::MicroAPI::Mul(
                (AscendC::MicroAPI::RegTensor<float>&)xFP32MaxReg, (AscendC::MicroAPI::RegTensor<float>&)xFP32MaxReg,
                dstTypeMaxReg, mask32);
        } else {
            AscendC::MicroAPI::Mul(
                (AscendC::MicroAPI::RegTensor<float>&)xFP32MaxReg, (AscendC::MicroAPI::RegTensor<float>&)xMaxReg,
                dstTypeMaxReg, mask32);
        }
        // 右移获取指数位
        AscendC::MicroAPI::ShiftRights(expFP32Reg, xFP32MaxReg, SHR_NUM_FOR_FP32, mask32);
        // And获取尾数位
        AscendC::MicroAPI::And(manFP32Reg, xFP32MaxReg, manForFP32Reg, mask32);
        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, expFP32Reg, NUMBER_ZERO, mask32);
        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(p0, expFP32Reg, NUMBER_TWO_FIVE_FOUR, p0);
        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, manFP32Reg, NUMBER_ZERO, p0);
        AscendC::MicroAPI::Adds(extractExpReg, expFP32Reg, 1, mask32);
        // 根据情况选择指数位是否加一
        AscendC::MicroAPI::Select<uint32_t>(expFP32Reg, extractExpReg, expFP32Reg, p0);

        AscendC::MicroAPI::Pack(expBF16Reg, expFP32Reg);
        AscendC::MicroAPI::Pack(mxScale, expBF16Reg);
        AscendC::MicroAPI::DataCopyUnAlign(mxScaleAddr, mxScale, u1, dataLen);
        AscendC::MicroAPI::DataCopyUnAlignPost(mxScaleAddr, u1, 0);

        AscendC::MicroAPI::ShiftLefts(expBF16Reg, expBF16Reg, SHR_NUM_FOR_BF16, mask16);
        AscendC::MicroAPI::ShiftLefts(expFP32Reg, expFP32Reg, SHR_NUM_FOR_FP32, mask16);

        // 求1/scale
        if constexpr (IsSame<T, half>::value) {
            AscendC::MicroAPI::Compare<calcTypeInt, CMPMODE::NE>(infMask, expFP32Reg, infForXReg, mask32);
            AscendC::MicroAPI::Compare<calcTypeInt, CMPMODE::EQ>(specialDataMask, expFP32Reg, biasReg, mask32);
            AscendC::MicroAPI::Sub(scaleReprocal, biasReg, expFP32Reg, mask32);
        } else {
            AscendC::MicroAPI::Compare<calcTypeInt, CMPMODE::NE>(infMask, expBF16Reg, infForXReg, mask16);
            AscendC::MicroAPI::Compare<calcTypeInt, CMPMODE::EQ>(specialDataMask, expBF16Reg, biasReg, mask16);
            AscendC::MicroAPI::Sub(scaleReprocal, biasReg, expBF16Reg, mask16);
        }
        AscendC::MicroAPI::Select<calcTypeInt>(scaleReprocal, scaleReprocal, nanRegTensor, infMask);
        AscendC::MicroAPI::Select<calcTypeInt>(scaleReprocal, specialExpRegTensor, scaleReprocal, specialDataMask);

        auto scaleAddr = maxExpAddr;
        for (uint16_t i = 0; i < rowsSingleLoop; i++) {
            AscendC::MicroAPI::DataCopyUnAlign(scaleAddr, scaleReprocal, u1, dataLen);
            AscendC::MicroAPI::DataCopyUnAlignPost(scaleAddr, u1, 0);
        }
        AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        AscendC::MicroAPI::DataCopy(scaleReprocal, maxExpAddr);

        // 求data value
        for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); i++) {
            this->template LoadData<calcType>(xAddr, i * dataLenSingleLoop, xReg, pnumMask);
            CalcElement<roundMode, U, calcType, calcTypeInt>(xReg, scaleReprocal, infForXReg, out, pnumMask);
            auto addr = yAddr + (i * dataLenSingleLoop) / DIGIT_TWO;
            AscendC::MicroAPI::DataCopyUnAlign(addr, out, u1, dataLenSingleLoop / DIGIT_TWO);
            AscendC::MicroAPI::DataCopyUnAlignPost(addr, u1, 0);
        }
        this->template LoadData<calcType>(xAddr, (regLoop - 1) * dataLenSingleLoop, xReg, tailPnumMask);
        CalcElement<roundMode, U, calcType, calcTypeInt>(xReg, scaleReprocal, infForXReg, out, tailPnumMask);
        auto addr = yAddr + ((regLoop - 1) * dataLenSingleLoop) / DIGIT_TWO;
        AscendC::MicroAPI::DataCopyUnAlign(addr, out, u1, dataLenTailLoop / DIGIT_TWO);
        AscendC::MicroAPI::DataCopyUnAlignPost(addr, u1, 0);
    }
}
template <typename T, typename U, const bool ISTAIL>
template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimize<T, U, ISTAIL>::ComputeOcp(
    int64_t dataLen, int64_t blockCount, __ubuf__ T* xAddr, __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint8_t* yAddr)
{
    constexpr uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(calcType); // 寄存器单次处理能处理的长度
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

    LocalTensor<calcTypeInt> maxExpTensor = maxExpBuf_.Get<calcTypeInt>();
    auto maxExpAddr = (__ubuf__ calcTypeInt*)maxExpTensor.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<calcType> xReg;
        AscendC::MicroAPI::RegTensor<calcTypeInt> absReg;
        AscendC::MicroAPI::RegTensor<calcTypeInt> xMaxReg;
        AscendC::MicroAPI::RegTensor<calcTypeInt> expMaxReg;
        AscendC::MicroAPI::RegTensor<calcTypeInt> expReg;
        AscendC::MicroAPI::RegTensor<calcTypeInt> mxScaleReg;
        AscendC::MicroAPI::RegTensor<calcTypeInt> fp8NanRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> fp16MxScale;
        AscendC::MicroAPI::RegTensor<uint8_t> mxScale;
        AscendC::MicroAPI::RegTensor<calcTypeInt> specialExpRegTensor;
        AscendC::MicroAPI::RegTensor<calcTypeInt> scaleReprocal;
        AscendC::MicroAPI::RegTensor<calcTypeInt> biasRegTensor;
        AscendC::MicroAPI::RegTensor<calcTypeInt> nanRegTensor;
        AscendC::MicroAPI::RegTensor<calcTypeInt> subNumForScale;
        AscendC::MicroAPI::RegTensor<uint8_t> out;

        AscendC::MicroAPI::RegTensor<calcTypeInt> absForXReg;
        AscendC::MicroAPI::RegTensor<calcTypeInt> infForXReg;
        AscendC::MicroAPI::RegTensor<calcTypeInt> fp4MaxExpReg;
        AscendC::MicroAPI::RegTensor<calcTypeInt> zeroReg;

        AscendC::MicroAPI::UnalignReg u0;
        AscendC::MicroAPI::UnalignReg u1;
        AscendC::MicroAPI::MaskReg zeroMask;
        AscendC::MicroAPI::MaskReg infMask;
        AscendC::MicroAPI::MaskReg invalidDataMask;
        AscendC::MicroAPI::MaskReg specialDataMask;

        AscendC::MicroAPI::Duplicate(infForXReg, this->maxExp_);
        AscendC::MicroAPI::Duplicate(zeroReg, 0);
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, this->f8Emax_);
        AscendC::MicroAPI::Duplicate(fp4MaxExpReg, this->f4Emax_);
        AscendC::MicroAPI::Duplicate(nanRegTensor, this->nanValue_);
        AscendC::MicroAPI::Duplicate(biasRegTensor, this->maxBias_);
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, this->specialExp_);

        uint32_t pnum = dataLenSingleLoop;
        uint32_t tailPnum = dataLenTailLoop;
        uint32_t maskNum = dataLenSingleLoop - expOffset;
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::UpdateMask<calcTypeInt>(maskNum);
        AscendC::MicroAPI::MaskReg pnumMask = AscendC::MicroAPI::UpdateMask<calcTypeInt>(pnum);
        AscendC::MicroAPI::MaskReg tailPnumMask = AscendC::MicroAPI::UpdateMask<calcTypeInt>(tailPnum);
        if constexpr (calcMode == ModeTwo) {
            AscendC::MicroAPI::Duplicate(absForXReg, this->absForX_);
            AscendC::MicroAPI::Duplicate(xMaxReg, 0);
            if constexpr (IsSame<T, half>::value) {
                AscendC::MicroAPI::Duplicate(subNumForScale, subNumForFP32Scale_);
            } else {
                AscendC::MicroAPI::Duplicate(subNumForScale, subNumForBF16Scale_);
            }

            for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); i++) {
                this->template LoadData<calcType>(xAddr, i * dataLenSingleLoop, xReg, pnumMask);
                AscendC::MicroAPI::And(absReg, (AscendC::MicroAPI::RegTensor<calcTypeInt>&)xReg, absForXReg, pnumMask);
                AscendC::MicroAPI::Max(xMaxReg, xMaxReg, absReg, pnumMask);
            }
            this->template LoadData<calcType>(xAddr, (regLoop - 1) * dataLenSingleLoop, xReg, tailPnumMask);
            AscendC::MicroAPI::And(absReg, (AscendC::MicroAPI::RegTensor<calcTypeInt>&)xReg, absForXReg, tailPnumMask);
            AscendC::MicroAPI::Max(absReg, xMaxReg, absReg, tailPnumMask);
            AscendC::MicroAPI::Copy<calcTypeInt, AscendC::MicroAPI::MaskMergeMode::MERGING>(
                xMaxReg, absReg, tailPnumMask);
            // 二分法求rowsSingleLoop行中的最大行
            AscendC::MicroAPI::DataCopy(maxExpAddr, xMaxReg, pnumMask);
            AscendC::MicroAPI::LocalMemBar<
                AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, maxExpAddr);
            AscendC::MicroAPI::DataCopyUnAlign(xMaxReg, u0, maxExpAddr);
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, maxExpAddr + expOffset);
            AscendC::MicroAPI::DataCopyUnAlign(absReg, u0, maxExpAddr + expOffset);
            AscendC::MicroAPI::Max(absReg, xMaxReg, absReg, mask);
            AscendC::MicroAPI::Copy<calcTypeInt, AscendC::MicroAPI::MaskMergeMode::MERGING>(xMaxReg, absReg, mask);
            for (uint16_t i = 0; i < loopSize; i++) {
                AscendC::MicroAPI::DataCopy(maxExpAddr, xMaxReg, pnumMask);
                AscendC::MicroAPI::LocalMemBar<
                    AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
                expOffset /= DIGIT_TWO;
                maskNum = expOffset;
                mask = AscendC::MicroAPI::UpdateMask<calcTypeInt>(maskNum);
                AscendC::MicroAPI::DataCopyUnAlignPre(u0, maxExpAddr);
                AscendC::MicroAPI::DataCopyUnAlign(xMaxReg, u0, maxExpAddr);
                AscendC::MicroAPI::DataCopyUnAlignPre(u0, maxExpAddr + expOffset);
                AscendC::MicroAPI::DataCopyUnAlign(absReg, u0, maxExpAddr + expOffset);
                AscendC::MicroAPI::Max(xMaxReg, xMaxReg, absReg, mask);
            }
            maskNum = static_cast<uint32_t>(dataLen);
            mask = AscendC::MicroAPI::UpdateMask<calcTypeInt>(maskNum);
            // calcMode=1时，前面求 数的最大值，现提取指数
            AscendC::MicroAPI::And(expMaxReg, xMaxReg, infForXReg, mask);

            AscendC::MicroAPI::Compare<calcTypeInt, CMPMODE::NE>(infMask, expMaxReg, infForXReg, mask);
            AscendC::MicroAPI::Compare<calcTypeInt, CMPMODE::NE>(zeroMask, expMaxReg, zeroReg, mask);
            if constexpr (IsSame<U, fp4x2_e2m1_t>::value) {
                AscendC::MicroAPI::Compare<calcTypeInt, CMPMODE::LT>(invalidDataMask, expMaxReg, fp4MaxExpReg, mask);
                AscendC::MicroAPI::Sub(expMaxReg, xMaxReg, subNumForScale, mask);
                AscendC::MicroAPI::Select<calcTypeInt>(expMaxReg, zeroReg, expMaxReg, invalidDataMask);
                AscendC::MicroAPI::And(expMaxReg, expMaxReg, infForXReg, mask);
            }
        } else {
            AscendC::MicroAPI::Duplicate(expMaxReg, 0);
            for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); i++) {
                this->template LoadData<calcType>(xAddr, i * dataLenSingleLoop, xReg, pnumMask);
                AscendC::MicroAPI::And(expReg, (AscendC::MicroAPI::RegTensor<calcTypeInt>&)xReg, infForXReg, pnumMask);
                AscendC::MicroAPI::Max(expMaxReg, expMaxReg, expReg, pnumMask);
            }
            this->template LoadData<calcType>(xAddr, (regLoop - 1) * dataLenSingleLoop, xReg, tailPnumMask);
            AscendC::MicroAPI::And(expReg, (AscendC::MicroAPI::RegTensor<calcTypeInt>&)xReg, infForXReg, tailPnumMask);
            AscendC::MicroAPI::Max(expReg, expMaxReg, expReg, tailPnumMask);
            AscendC::MicroAPI::Copy<calcTypeInt, AscendC::MicroAPI::MaskMergeMode::MERGING>(
                expMaxReg, expReg, tailPnumMask);
            // 二分法求rowsSingleLoop行中的最大行
            AscendC::MicroAPI::DataCopy(maxExpAddr, expMaxReg, pnumMask);
            AscendC::MicroAPI::LocalMemBar<
                AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, maxExpAddr);
            AscendC::MicroAPI::DataCopyUnAlign(expMaxReg, u0, maxExpAddr);
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, maxExpAddr + expOffset);
            AscendC::MicroAPI::DataCopyUnAlign(expReg, u0, maxExpAddr + expOffset);
            AscendC::MicroAPI::Max(expReg, expMaxReg, expReg, mask);
            AscendC::MicroAPI::Copy<calcTypeInt, AscendC::MicroAPI::MaskMergeMode::MERGING>(expMaxReg, expReg, mask);
            for (uint16_t i = 0; i < loopSize; i++) {
                AscendC::MicroAPI::DataCopy(maxExpAddr, expMaxReg, pnumMask);
                AscendC::MicroAPI::LocalMemBar<
                    AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
                expOffset /= DIGIT_TWO;
                maskNum = expOffset;
                mask = AscendC::MicroAPI::UpdateMask<calcTypeInt>(maskNum);
                AscendC::MicroAPI::DataCopyUnAlignPre(u0, maxExpAddr);
                AscendC::MicroAPI::DataCopyUnAlign(expMaxReg, u0, maxExpAddr);
                AscendC::MicroAPI::DataCopyUnAlignPre(u0, maxExpAddr + expOffset);
                AscendC::MicroAPI::DataCopyUnAlign(expReg, u0, maxExpAddr + expOffset);
                AscendC::MicroAPI::Max(expMaxReg, expMaxReg, expReg, mask);
            }
            maskNum = static_cast<uint32_t>(dataLen);
            mask = AscendC::MicroAPI::UpdateMask<calcTypeInt>(maskNum);
            AscendC::MicroAPI::Compare<calcTypeInt, CMPMODE::NE>(infMask, expMaxReg, infForXReg, mask);
            AscendC::MicroAPI::Compare<calcTypeInt, CMPMODE::NE>(zeroMask, expMaxReg, zeroReg, mask);
            if constexpr (IsSame<U, fp4x2_e2m1_t>::value) {
                AscendC::MicroAPI::Compare<calcTypeInt, CMPMODE::LT>(invalidDataMask, expMaxReg, fp4MaxExpReg, mask);
                AscendC::MicroAPI::Select<calcTypeInt>(expMaxReg, fp4MaxExpReg, expMaxReg, invalidDataMask);
                AscendC::MicroAPI::Sub(expMaxReg, expMaxReg, fp4MaxExpReg, mask);
            }
        }
        AscendC::MicroAPI::ShiftRights(mxScaleReg, expMaxReg, this->shrNum_, mask);
        AscendC::MicroAPI::Select<calcTypeInt>(mxScaleReg, mxScaleReg, fp8NanRegTensor, infMask);
        AscendC::MicroAPI::Select<calcTypeInt>(mxScaleReg, mxScaleReg, zeroReg, zeroMask);
        if constexpr (IsSame<T, half>::value) {
            AscendC::MicroAPI::Pack(fp16MxScale, mxScaleReg);
            AscendC::MicroAPI::Pack(mxScale, fp16MxScale);
        } else {
            AscendC::MicroAPI::Pack(mxScale, mxScaleReg);
        }
        AscendC::MicroAPI::DataCopyUnAlign(mxScaleAddr, mxScale, u1, dataLen);
        AscendC::MicroAPI::DataCopyUnAlignPost(mxScaleAddr, u1, 0);

        // 求1/scale
        AscendC::MicroAPI::Compare<calcTypeInt, CMPMODE::EQ>(specialDataMask, expMaxReg, biasRegTensor, mask);
        AscendC::MicroAPI::Sub(scaleReprocal, biasRegTensor, expMaxReg, mask);
        AscendC::MicroAPI::Select<calcTypeInt>(scaleReprocal, scaleReprocal, nanRegTensor, infMask);
        AscendC::MicroAPI::Select<calcTypeInt>(scaleReprocal, scaleReprocal, zeroReg, zeroMask);
        AscendC::MicroAPI::Select<calcTypeInt>(scaleReprocal, specialExpRegTensor, scaleReprocal, specialDataMask);

        auto scaleAddr = maxExpAddr;
        for (uint16_t i = 0; i < rowsSingleLoop; i++) {
            AscendC::MicroAPI::DataCopyUnAlign(scaleAddr, scaleReprocal, u1, dataLen);
            AscendC::MicroAPI::DataCopyUnAlignPost(scaleAddr, u1, 0);
        }
        AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        AscendC::MicroAPI::DataCopy(scaleReprocal, maxExpAddr);

        // 求data value
        for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); i++) {
            this->template LoadData<calcType>(xAddr, i * dataLenSingleLoop, xReg, pnumMask);
            CalcElement<roundMode, U, calcType, calcTypeInt>(xReg, scaleReprocal, infForXReg, out, pnumMask);
            auto addr = yAddr + (i * dataLenSingleLoop) / DIGIT_TWO;
            AscendC::MicroAPI::DataCopyUnAlign(addr, out, u1, dataLenSingleLoop / DIGIT_TWO);
            AscendC::MicroAPI::DataCopyUnAlignPost(addr, u1, 0);
        }
        this->template LoadData<calcType>(xAddr, (regLoop - 1) * dataLenSingleLoop, xReg, tailPnumMask);
        CalcElement<roundMode, U, calcType, calcTypeInt>(xReg, scaleReprocal, infForXReg, out, tailPnumMask);
        auto addr = yAddr + ((regLoop - 1) * dataLenSingleLoop) / DIGIT_TWO;
        AscendC::MicroAPI::DataCopyUnAlign(addr, out, u1, dataLenTailLoop / DIGIT_TWO);
        AscendC::MicroAPI::DataCopyUnAlignPost(addr, u1, 0);
    }
}

} // namespace DynamicMxQuant
#endif // DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_OPTIMIZE_H
