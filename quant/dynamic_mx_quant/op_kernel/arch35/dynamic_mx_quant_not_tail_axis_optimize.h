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
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR workspace,
                                const DynamicMxQuantTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SplitPreAxisCompute(int64_t ubFactor, int64_t blockSizeIdx);
    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode, const int64_t calcMode>
    __aicore__ inline void ComputeOcp(int64_t dataLen, int64_t blockCount, __ubuf__ T* xAddr,
                                      __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint8_t* yAddr);
    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void ComputeCuBLAS(int64_t dataLen, int64_t blockCount, __ubuf__ T* xAddr,
                                         __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint8_t* yAddr);

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
__aicore__ inline void DynamicMxQuantNotTailAxisOptimize<T, U, ISTAIL>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR mxScale,
                                                                             GM_ADDR workspace,
                                                                             const DynamicMxQuantTilingData* tilingData)
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
__aicore__ inline void DynamicMxQuantNotTailAxisOptimize<T, U, ISTAIL>::SplitPreAxisCompute(int64_t ubFactor,
                                                                                            int64_t blockSizeIdx)
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
        if (calcMode_ == MODE_ZERO) {
            if (this->roundMode_ == MODE_RINT) {
                ComputeOcp<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT, MODE_ZERO>(this->postAxisSize_, blockCount,
                                                                                   xAddr, mxScaleAddr, yAddr);
            } else if (this->roundMode_ == MODE_FLOOR) {
                ComputeOcp<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR, MODE_ZERO>(this->postAxisSize_, blockCount,
                                                                                    xAddr, mxScaleAddr, yAddr);
            } else if (this->roundMode_ == MODE_ROUND) {
                ComputeOcp<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND, MODE_ZERO>(this->postAxisSize_, blockCount,
                                                                                    xAddr, mxScaleAddr, yAddr);
            }
        } else if (calcMode_ == MODE_TWO) {
            if (this->roundMode_ == MODE_RINT) {
                ComputeOcp<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT, MODE_TWO>(this->postAxisSize_, blockCount,
                                                                                  xAddr, mxScaleAddr, yAddr);
            } else if (this->roundMode_ == MODE_FLOOR) {
                ComputeOcp<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR, MODE_TWO>(this->postAxisSize_, blockCount,
                                                                                   xAddr, mxScaleAddr, yAddr);
            } else if (this->roundMode_ == MODE_ROUND) {
                ComputeOcp<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND, MODE_TWO>(this->postAxisSize_, blockCount,
                                                                                   xAddr, mxScaleAddr, yAddr);
            }
        } else {
            if (this->roundMode_ == MODE_RINT) {
                ComputeCuBLAS<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(this->postAxisSize_, blockCount, xAddr,
                                                                           mxScaleAddr, yAddr);
            } else if (this->roundMode_ == MODE_FLOOR) {
                ComputeCuBLAS<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(this->postAxisSize_, blockCount, xAddr,
                                                                            mxScaleAddr, yAddr);
            } else if (this->roundMode_ == MODE_ROUND) {
                ComputeCuBLAS<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(this->postAxisSize_, blockCount, xAddr,
                                                                            mxScaleAddr, yAddr);
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
    static constexpr Reg::CastTrait castTraitZero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                     Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    constexpr uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(calcType); // 寄存器单次处理能处理的长度
    uint16_t rowsSingleLoop = 0;
    if (blockCount < static_cast<int64_t>(vfLen) / dataLen) {
        rowsSingleLoop = static_cast<uint16_t>(blockCount);
    } else {
        rowsSingleLoop = static_cast<uint16_t>(static_cast<int64_t>(vfLen) / dataLen);
    }                                                                             // 单次处理能处理的行数
    uint16_t dataLenSingleLoop = rowsSingleLoop * static_cast<uint16_t>(dataLen); // 单次处理长度
    uint16_t regLoop = Ceil(static_cast<uint16_t>(blockCount), rowsSingleLoop);   // 循环数
    uint16_t rowsTailLoop = static_cast<uint16_t>(blockCount) % rowsSingleLoop;   // 尾循环处理的行数
    if (rowsTailLoop == 0) {
        rowsTailLoop = rowsSingleLoop;
    }
    uint16_t dataLenTailLoop = rowsTailLoop * static_cast<uint16_t>(dataLen); // 尾循环处理的长度
    uint16_t loopSize = static_cast<uint16_t>(DIGIT_SIXTY_THREE - AscendC::ScalarCountLeadingZero(static_cast<uint64_t>(
                                                                      rowsSingleLoop))); // 求最大指数行的二分次数
    uint16_t rows = 1 << loopSize; // 最接近rowsSingleLoop的2次方数
    uint16_t expOffset = rows * static_cast<uint16_t>(dataLen);

    LocalTensor<calcTypeInt> maxExpTensor = maxExpBuf_.Get<calcTypeInt>();
    auto maxExpAddr = (__ubuf__ calcTypeInt*)maxExpTensor.GetPhyAddr();
    __VEC_SCOPE__
    {
        if constexpr (IsSame<T, float>::value) {
            return;
        }
        Reg::RegTensor<calcType> xReg;
        Reg::RegTensor<calcTypeInt> absReg;
        Reg::RegTensor<calcTypeInt> xMaxReg;
        Reg::RegTensor<uint32_t> xFP32MaxReg;
        Reg::RegTensor<uint32_t> expFP32Reg;    // 指数
        Reg::RegTensor<uint32_t> manFP32Reg;    // 尾数
        Reg::RegTensor<uint32_t> extractExpReg; // 指数 + 1
        Reg::RegTensor<uint16_t> expBF16Reg;
        Reg::RegTensor<uint8_t> mxScale;
        Reg::RegTensor<uint8_t> out;
        Reg::RegTensor<calcTypeInt> scaleReprocal; // 1/scale

        Reg::RegTensor<calcTypeInt> absForXReg;
        Reg::RegTensor<calcTypeInt> zeroReg;
        Reg::RegTensor<calcTypeInt> biasReg;
        Reg::RegTensor<calcTypeInt> infForXReg;
        Reg::RegTensor<uint32_t> manForFP32Reg;
        Reg::RegTensor<float> dstTypeMaxReg;
        Reg::RegTensor<calcTypeInt> nan;
        Reg::RegTensor<calcTypeInt> specialExp;

        Reg::UnalignReg u0;
        Reg::UnalignReg u1;
        Reg::MaskReg zeroMask;
        Reg::MaskReg infMask;
        Reg::MaskReg specialDataMask;
        Reg::MaskReg p0;
        Reg::MaskReg mask32;
        Reg::MaskReg mask16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();

        Reg::Duplicate(dstTypeMaxReg, this->invDstTypeMax_);
        Reg::Duplicate(biasReg, this->maxBias_);
        Reg::Duplicate(absForXReg, this->absForX_);
        Reg::Duplicate(infForXReg, this->maxExp_);
        Reg::Duplicate(manForFP32Reg, FP32_MX_MAN_MASK);
        Reg::Duplicate(nan, this->nanValue_);
        Reg::Duplicate(specialExp, this->specialExp_);

        uint32_t pnum = dataLenSingleLoop;
        uint32_t tailPnum = dataLenTailLoop;
        Reg::MaskReg pnumMask = Reg::UpdateMask<calcTypeInt>(pnum);
        Reg::MaskReg tailPnumMask = Reg::UpdateMask<calcTypeInt>(tailPnum);
        Reg::Duplicate(xMaxReg, 0);
        for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); i++) {
            this->template LoadData<calcType>(xAddr, i * dataLenSingleLoop, xReg, pnumMask);
            Reg::And(absReg, (Reg::RegTensor<calcTypeInt>&)xReg, absForXReg, pnumMask);
            Reg::Max(xMaxReg, xMaxReg, absReg, pnumMask);
        }
        this->template LoadData<calcType>(xAddr, (regLoop - 1) * dataLenSingleLoop, xReg, tailPnumMask);
        Reg::And(absReg, (Reg::RegTensor<calcTypeInt>&)xReg, absForXReg, tailPnumMask);
        Reg::Max(absReg, xMaxReg, absReg, tailPnumMask);
        Reg::Copy<calcTypeInt, Reg::MaskMergeMode::MERGING>(xMaxReg, absReg, tailPnumMask);
        // 二分法求rowsSingleLoop行中的最大行
        Reg::DataCopy(maxExpAddr, xMaxReg, pnumMask);
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        uint32_t maskNum = dataLenSingleLoop - expOffset;
        Reg::MaskReg mask = Reg::UpdateMask<calcTypeInt>(maskNum);
        Reg::DataCopyUnAlignPre(u0, maxExpAddr);
        Reg::DataCopyUnAlign(xMaxReg, u0, maxExpAddr);
        Reg::DataCopyUnAlignPre(u0, maxExpAddr + expOffset);
        Reg::DataCopyUnAlign(absReg, u0, maxExpAddr + expOffset);
        Reg::Max(absReg, xMaxReg, absReg, mask);
        Reg::Copy<calcTypeInt, Reg::MaskMergeMode::MERGING>(xMaxReg, absReg, mask);
        for (uint16_t i = 0; i < loopSize; i++) {
            Reg::DataCopy(maxExpAddr, xMaxReg, pnumMask);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            expOffset /= DIGIT_TWO;
            maskNum = expOffset;
            mask = Reg::UpdateMask<calcTypeInt>(maskNum);
            Reg::DataCopyUnAlignPre(u0, maxExpAddr);
            Reg::DataCopyUnAlign(xMaxReg, u0, maxExpAddr);
            Reg::DataCopyUnAlignPre(u0, maxExpAddr + expOffset);
            Reg::DataCopyUnAlign(absReg, u0, maxExpAddr + expOffset);
            Reg::Max(xMaxReg, xMaxReg, absReg, mask);
        }
        // 求scale
        maskNum = static_cast<uint32_t>(dataLen);
        mask32 = Reg::UpdateMask<uint32_t>(maskNum);
        if constexpr (IsSame<T, bfloat16_t>::value) {
            Reg::Interleave(xMaxReg, zeroReg, xMaxReg, zeroReg);
            Reg::Cast<float, T, castTraitZero>((Reg::RegTensor<float>&)xFP32MaxReg, (Reg::RegTensor<T>&)xMaxReg,
                                               mask16);
            Reg::Mul((Reg::RegTensor<float>&)xFP32MaxReg, (Reg::RegTensor<float>&)xFP32MaxReg, dstTypeMaxReg, mask32);
        } else {
            Reg::Mul((Reg::RegTensor<float>&)xFP32MaxReg, (Reg::RegTensor<float>&)xMaxReg, dstTypeMaxReg, mask32);
        }
        // 右移获取指数位
        Reg::ShiftRights(expFP32Reg, xFP32MaxReg, FP32_SHR_NUM, mask32);
        // And获取尾数位
        Reg::And(manFP32Reg, xFP32MaxReg, manForFP32Reg, mask32);
        Reg::CompareScalar<uint32_t, CMPMODE::GT>(p0, expFP32Reg, FP32_NUMBER_ZERO, mask32);
        Reg::CompareScalar<uint32_t, CMPMODE::LT>(p0, expFP32Reg, FP32_NUMBER_254, p0);
        Reg::CompareScalar<uint32_t, CMPMODE::GT>(p0, manFP32Reg, FP32_NUMBER_ZERO, p0);
        Reg::Adds(extractExpReg, expFP32Reg, 1, mask32);
        // 根据情况选择指数位是否加一
        Reg::Select<uint32_t>(expFP32Reg, extractExpReg, expFP32Reg, p0);

        Reg::Pack(expBF16Reg, expFP32Reg);
        Reg::Pack(mxScale, expBF16Reg);
        Reg::DataCopyUnAlign(mxScaleAddr, mxScale, u1, dataLen);
        Reg::DataCopyUnAlignPost(mxScaleAddr, u1, 0);

        Reg::ShiftLefts(expBF16Reg, expBF16Reg, BF16_SHR_NUM, mask16);
        Reg::ShiftLefts(expFP32Reg, expFP32Reg, FP32_SHR_NUM, mask16);

        // 求1/scale
        if constexpr (IsSame<T, half>::value) {
            Reg::Compare<calcTypeInt, CMPMODE::NE>(infMask, expFP32Reg, infForXReg, mask32);
            Reg::Compare<calcTypeInt, CMPMODE::EQ>(specialDataMask, expFP32Reg, biasReg, mask32);
            Reg::Sub(scaleReprocal, biasReg, expFP32Reg, mask32);
        } else {
            Reg::Compare<calcTypeInt, CMPMODE::NE>(infMask, expBF16Reg, infForXReg, mask16);
            Reg::Compare<calcTypeInt, CMPMODE::EQ>(specialDataMask, expBF16Reg, biasReg, mask16);
            Reg::Sub(scaleReprocal, biasReg, expBF16Reg, mask16);
        }
        Reg::Select<calcTypeInt>(scaleReprocal, scaleReprocal, nan, infMask);
        Reg::Select<calcTypeInt>(scaleReprocal, specialExp, scaleReprocal, specialDataMask);

        auto scaleAddr = maxExpAddr;
        for (uint16_t i = 0; i < rowsSingleLoop; i++) {
            Reg::DataCopyUnAlign(scaleAddr, scaleReprocal, u1, dataLen);
            Reg::DataCopyUnAlignPost(scaleAddr, u1, 0);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        Reg::DataCopy(scaleReprocal, maxExpAddr);

        // 求data value
        for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); i++) {
            this->template LoadData<calcType>(xAddr, i * dataLenSingleLoop, xReg, pnumMask);
            CalcElement<roundMode, U, calcType, calcTypeInt>(xReg, scaleReprocal, infForXReg, out, pnumMask);
            auto addr = yAddr + (i * dataLenSingleLoop) / DIGIT_TWO;
            Reg::DataCopyUnAlign(addr, out, u1, dataLenSingleLoop / DIGIT_TWO);
            Reg::DataCopyUnAlignPost(addr, u1, 0);
        }
        this->template LoadData<calcType>(xAddr, (regLoop - 1) * dataLenSingleLoop, xReg, tailPnumMask);
        CalcElement<roundMode, U, calcType, calcTypeInt>(xReg, scaleReprocal, infForXReg, out, tailPnumMask);
        auto addr = yAddr + ((regLoop - 1) * dataLenSingleLoop) / DIGIT_TWO;
        Reg::DataCopyUnAlign(addr, out, u1, dataLenTailLoop / DIGIT_TWO);
        Reg::DataCopyUnAlignPost(addr, u1, 0);
    }
}
template <typename T, typename U, const bool ISTAIL>
template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimize<T, U, ISTAIL>::ComputeOcp(int64_t dataLen, int64_t blockCount,
                                                                                   __ubuf__ T* xAddr,
                                                                                   __ubuf__ uint8_t* mxScaleAddr,
                                                                                   __ubuf__ uint8_t* yAddr)
{
    constexpr uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(calcType); // 寄存器单次处理能处理的长度
    uint16_t rowsSingleLoop = 0;
    if (blockCount < static_cast<int64_t>(vfLen) / dataLen) {
        rowsSingleLoop = static_cast<uint16_t>(blockCount);
    } else {
        rowsSingleLoop = static_cast<uint16_t>(static_cast<int64_t>(vfLen) / dataLen);
    }                                                                             // 单次处理能处理的行数
    uint16_t dataLenSingleLoop = rowsSingleLoop * static_cast<uint16_t>(dataLen); // 单次处理长度
    uint16_t regLoop = Ceil(static_cast<uint16_t>(blockCount), rowsSingleLoop);   // 循环数
    uint16_t rowsTailLoop = static_cast<uint16_t>(blockCount) % rowsSingleLoop;   // 尾循环处理的行数
    if (rowsTailLoop == 0) {
        rowsTailLoop = rowsSingleLoop;
    }
    uint16_t dataLenTailLoop = rowsTailLoop * static_cast<uint16_t>(dataLen); // 尾循环处理的长度
    uint16_t loopSize = static_cast<uint16_t>(DIGIT_SIXTY_THREE - AscendC::ScalarCountLeadingZero(static_cast<uint64_t>(
                                                                      rowsSingleLoop))); // 求最大指数行的二分次数
    uint16_t rows = 1 << loopSize; // 最接近rowsSingleLoop的2次方数
    uint16_t expOffset = rows * static_cast<uint16_t>(dataLen);

    LocalTensor<calcTypeInt> maxExpTensor = maxExpBuf_.Get<calcTypeInt>();
    auto maxExpAddr = (__ubuf__ calcTypeInt*)maxExpTensor.GetPhyAddr();
    __VEC_SCOPE__
    {
        if constexpr (IsSame<T, float>::value) {
            return;
        }
        Reg::RegTensor<calcType> xReg;
        Reg::RegTensor<calcTypeInt> absReg;
        Reg::RegTensor<calcTypeInt> xMaxReg;
        Reg::RegTensor<calcTypeInt> expMaxReg;
        Reg::RegTensor<calcTypeInt> expReg;
        Reg::RegTensor<calcTypeInt> mxScaleReg;
        Reg::RegTensor<calcTypeInt> fp8Nan;
        Reg::RegTensor<uint16_t> fp16MxScale;
        Reg::RegTensor<uint8_t> mxScale;
        Reg::RegTensor<calcTypeInt> specialExp;
        Reg::RegTensor<calcTypeInt> scaleReprocal;
        Reg::RegTensor<calcTypeInt> bias;
        Reg::RegTensor<calcTypeInt> nan;
        Reg::RegTensor<calcTypeInt> subNumForScale;
        Reg::RegTensor<uint8_t> out;

        Reg::RegTensor<calcTypeInt> absForXReg;
        Reg::RegTensor<calcTypeInt> infForXReg;
        Reg::RegTensor<calcTypeInt> fp4MaxExpReg;
        Reg::RegTensor<calcTypeInt> zeroReg;

        Reg::UnalignReg u0;
        Reg::UnalignReg u1;
        Reg::MaskReg zeroMask;
        Reg::MaskReg infMask;
        Reg::MaskReg invalidDataMask;
        Reg::MaskReg specialDataMask;

        Reg::Duplicate(infForXReg, this->maxExp_);
        Reg::Duplicate(zeroReg, 0);
        Reg::Duplicate(fp8Nan, this->f8Emax_);
        Reg::Duplicate(fp4MaxExpReg, this->f4Emax_);
        Reg::Duplicate(nan, this->nanValue_);
        Reg::Duplicate(bias, this->maxBias_);
        Reg::Duplicate(specialExp, this->specialExp_);

        uint32_t pnum = dataLenSingleLoop;
        uint32_t tailPnum = dataLenTailLoop;
        uint32_t maskNum = dataLenSingleLoop - expOffset;
        Reg::MaskReg mask = Reg::UpdateMask<calcTypeInt>(maskNum);
        Reg::MaskReg pnumMask = Reg::UpdateMask<calcTypeInt>(pnum);
        Reg::MaskReg tailPnumMask = Reg::UpdateMask<calcTypeInt>(tailPnum);
        if constexpr (calcMode == MODE_TWO) {
            Reg::Duplicate(absForXReg, this->absForX_);
            Reg::Duplicate(xMaxReg, 0);
            if constexpr (IsSame<T, half>::value) {
                Reg::Duplicate(subNumForScale, subNumForFP32Scale_);
            } else {
                Reg::Duplicate(subNumForScale, subNumForBF16Scale_);
            }

            for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); i++) {
                this->template LoadData<calcType>(xAddr, i * dataLenSingleLoop, xReg, pnumMask);
                Reg::And(absReg, (Reg::RegTensor<calcTypeInt>&)xReg, absForXReg, pnumMask);
                Reg::Max(xMaxReg, xMaxReg, absReg, pnumMask);
            }
            this->template LoadData<calcType>(xAddr, (regLoop - 1) * dataLenSingleLoop, xReg, tailPnumMask);
            Reg::And(absReg, (Reg::RegTensor<calcTypeInt>&)xReg, absForXReg, tailPnumMask);
            Reg::Max(absReg, xMaxReg, absReg, tailPnumMask);
            Reg::Copy<calcTypeInt, Reg::MaskMergeMode::MERGING>(xMaxReg, absReg, tailPnumMask);
            // 二分法求rowsSingleLoop行中的最大行
            Reg::DataCopy(maxExpAddr, xMaxReg, pnumMask);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            Reg::DataCopyUnAlignPre(u0, maxExpAddr);
            Reg::DataCopyUnAlign(xMaxReg, u0, maxExpAddr);
            Reg::DataCopyUnAlignPre(u0, maxExpAddr + expOffset);
            Reg::DataCopyUnAlign(absReg, u0, maxExpAddr + expOffset);
            Reg::Max(absReg, xMaxReg, absReg, mask);
            Reg::Copy<calcTypeInt, Reg::MaskMergeMode::MERGING>(xMaxReg, absReg, mask);
            for (uint16_t i = 0; i < loopSize; i++) {
                Reg::DataCopy(maxExpAddr, xMaxReg, pnumMask);
                Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
                expOffset /= DIGIT_TWO;
                maskNum = expOffset;
                mask = Reg::UpdateMask<calcTypeInt>(maskNum);
                Reg::DataCopyUnAlignPre(u0, maxExpAddr);
                Reg::DataCopyUnAlign(xMaxReg, u0, maxExpAddr);
                Reg::DataCopyUnAlignPre(u0, maxExpAddr + expOffset);
                Reg::DataCopyUnAlign(absReg, u0, maxExpAddr + expOffset);
                Reg::Max(xMaxReg, xMaxReg, absReg, mask);
            }
            maskNum = static_cast<uint32_t>(dataLen);
            mask = Reg::UpdateMask<calcTypeInt>(maskNum);
            // calcMode=1时，前面求 数的最大值，现提取指数
            Reg::And(expMaxReg, xMaxReg, infForXReg, mask);
            Reg::Compare<calcTypeInt, CMPMODE::NE>(infMask, expMaxReg, infForXReg, mask);

            if constexpr (IsSame<U, fp4x2_e2m1_t>::value) {
                Reg::Compare<calcTypeInt, CMPMODE::LT>(invalidDataMask, expMaxReg, fp4MaxExpReg, mask);
                Reg::Sub(expMaxReg, xMaxReg, subNumForScale, mask);
                Reg::Select<calcTypeInt>(expMaxReg, zeroReg, expMaxReg, invalidDataMask);
                Reg::And(expMaxReg, expMaxReg, infForXReg, mask);
            }
        } else {
            Reg::Duplicate(expMaxReg, 0);
            for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); i++) {
                this->template LoadData<calcType>(xAddr, i * dataLenSingleLoop, xReg, pnumMask);
                Reg::And(expReg, (Reg::RegTensor<calcTypeInt>&)xReg, infForXReg, pnumMask);
                Reg::Max(expMaxReg, expMaxReg, expReg, pnumMask);
            }
            this->template LoadData<calcType>(xAddr, (regLoop - 1) * dataLenSingleLoop, xReg, tailPnumMask);
            Reg::And(expReg, (Reg::RegTensor<calcTypeInt>&)xReg, infForXReg, tailPnumMask);
            Reg::Max(expReg, expMaxReg, expReg, tailPnumMask);
            Reg::Copy<calcTypeInt, Reg::MaskMergeMode::MERGING>(expMaxReg, expReg, tailPnumMask);
            // 二分法求rowsSingleLoop行中的最大行
            Reg::DataCopy(maxExpAddr, expMaxReg, pnumMask);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            Reg::DataCopyUnAlignPre(u0, maxExpAddr);
            Reg::DataCopyUnAlign(expMaxReg, u0, maxExpAddr);
            Reg::DataCopyUnAlignPre(u0, maxExpAddr + expOffset);
            Reg::DataCopyUnAlign(expReg, u0, maxExpAddr + expOffset);
            Reg::Max(expReg, expMaxReg, expReg, mask);
            Reg::Copy<calcTypeInt, Reg::MaskMergeMode::MERGING>(expMaxReg, expReg, mask);
            for (uint16_t i = 0; i < loopSize; i++) {
                Reg::DataCopy(maxExpAddr, expMaxReg, pnumMask);
                Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
                expOffset /= DIGIT_TWO;
                maskNum = expOffset;
                mask = Reg::UpdateMask<calcTypeInt>(maskNum);
                Reg::DataCopyUnAlignPre(u0, maxExpAddr);
                Reg::DataCopyUnAlign(expMaxReg, u0, maxExpAddr);
                Reg::DataCopyUnAlignPre(u0, maxExpAddr + expOffset);
                Reg::DataCopyUnAlign(expReg, u0, maxExpAddr + expOffset);
                Reg::Max(expMaxReg, expMaxReg, expReg, mask);
            }
            maskNum = static_cast<uint32_t>(dataLen);
            mask = Reg::UpdateMask<calcTypeInt>(maskNum);
            Reg::Compare<calcTypeInt, CMPMODE::NE>(infMask, expMaxReg, infForXReg, mask);
            if constexpr (IsSame<U, fp4x2_e2m1_t>::value) {
                Reg::Compare<calcTypeInt, CMPMODE::LT>(invalidDataMask, expMaxReg, fp4MaxExpReg, mask);
                Reg::Select<calcTypeInt>(expMaxReg, fp4MaxExpReg, expMaxReg, invalidDataMask);
                Reg::Sub(expMaxReg, expMaxReg, fp4MaxExpReg, mask);
            }
        }
        Reg::ShiftRights(mxScaleReg, expMaxReg, this->shrNum_, mask);
        Reg::Select<calcTypeInt>(mxScaleReg, mxScaleReg, fp8Nan, infMask);
        if constexpr (IsSame<T, half>::value) {
            Reg::Pack(fp16MxScale, mxScaleReg);
            Reg::Pack(mxScale, fp16MxScale);
        } else {
            Reg::Pack(mxScale, mxScaleReg);
        }
        Reg::DataCopyUnAlign(mxScaleAddr, mxScale, u1, dataLen);
        Reg::DataCopyUnAlignPost(mxScaleAddr, u1, 0);

        // 求1/scale
        Reg::Compare<calcTypeInt, CMPMODE::NE>(zeroMask, expMaxReg, zeroReg, mask);
        Reg::Compare<calcTypeInt, CMPMODE::EQ>(specialDataMask, expMaxReg, bias, mask);
        Reg::Sub(scaleReprocal, bias, expMaxReg, mask);
        Reg::Select<calcTypeInt>(scaleReprocal, scaleReprocal, nan, infMask);
        Reg::Select<calcTypeInt>(scaleReprocal, scaleReprocal, zeroReg, zeroMask);
        Reg::Select<calcTypeInt>(scaleReprocal, specialExp, scaleReprocal, specialDataMask);

        auto scaleAddr = maxExpAddr;
        for (uint16_t i = 0; i < rowsSingleLoop; i++) {
            Reg::DataCopyUnAlign(scaleAddr, scaleReprocal, u1, dataLen);
            Reg::DataCopyUnAlignPost(scaleAddr, u1, 0);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        Reg::DataCopy(scaleReprocal, maxExpAddr);

        // 求data value
        for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); i++) {
            this->template LoadData<calcType>(xAddr, i * dataLenSingleLoop, xReg, pnumMask);
            CalcElement<roundMode, U, calcType, calcTypeInt>(xReg, scaleReprocal, infForXReg, out, pnumMask);
            auto addr = yAddr + (i * dataLenSingleLoop) / DIGIT_TWO;
            Reg::DataCopyUnAlign(addr, out, u1, dataLenSingleLoop / DIGIT_TWO);
            Reg::DataCopyUnAlignPost(addr, u1, 0);
        }
        this->template LoadData<calcType>(xAddr, (regLoop - 1) * dataLenSingleLoop, xReg, tailPnumMask);
        CalcElement<roundMode, U, calcType, calcTypeInt>(xReg, scaleReprocal, infForXReg, out, tailPnumMask);
        auto addr = yAddr + ((regLoop - 1) * dataLenSingleLoop) / DIGIT_TWO;
        Reg::DataCopyUnAlign(addr, out, u1, dataLenTailLoop / DIGIT_TWO);
        Reg::DataCopyUnAlignPost(addr, u1, 0);
    }
}

} // namespace DynamicMxQuant
#endif // DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_OPTIMIZE_H
