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
    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void Compute(
        int64_t dataLen, int64_t blockCount, __ubuf__ T* xAddr, __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint8_t* yAddr);

private:
    TBuf<QuePosition::VECCALC> maxExpBuf_;
    int64_t blockLoopOffset_ = 0;
    int64_t blockOffset_ = 0;
    int64_t scaleBlockOffset_ = 0;
    int64_t bufferSize_ = 0;
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
        if (this->roundMode_ == MODE_RINT) {
            Compute<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(
                this->postAxisSize_, blockCount, xAddr, mxScaleAddr, yAddr);
        } else if (this->roundMode_ == MODE_FLOOR) {
            Compute<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(
                this->postAxisSize_, blockCount, xAddr, mxScaleAddr, yAddr);
        } else if (this->roundMode_ == MODE_ROUND) {
            Compute<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(
                this->postAxisSize_, blockCount, xAddr, mxScaleAddr, yAddr);
        }
    }
    this->inQueue_.template FreeTensor(x);
    this->mxScaleQueue_.template EnQue(mxScale);
    this->outQueue_.template EnQue(y);
}

template <typename T, typename U, const bool ISTAIL>
template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicMxQuantNotTailAxisOptimize<T, U, ISTAIL>::Compute(
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
    uint16_t loopSize =
        static_cast<uint16_t>(DIGIT_SIXTY_THREE - AscendC::ScalarCountLeadingZero(static_cast<uint64_t>(rowsSingleLoop))); // 求最大指数行的二分次数
    uint16_t rows = 1 << loopSize; // 最接近rowsSingleLoop的2次方数
    uint16_t expOffset = rows * static_cast<uint16_t>(dataLen);

    LocalTensor<calcTypeInt> maxExpTensor = maxExpBuf_.Get<calcTypeInt>();
    auto maxExpAddr = (__ubuf__ calcTypeInt*)maxExpTensor.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<calcType> xRegTensor;
        AscendC::MicroAPI::RegTensor<calcTypeInt> expRegTensor;
        AscendC::MicroAPI::RegTensor<calcTypeInt> expMaxRegTensor;
        AscendC::MicroAPI::RegTensor<calcTypeInt> maxEleRegTensor;
        AscendC::MicroAPI::RegTensor<calcTypeInt> fp8NanRegTensor;
        AscendC::MicroAPI::RegTensor<calcTypeInt> fp4MaxExpRegTensor;
        AscendC::MicroAPI::RegTensor<calcTypeInt> mxScaleRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> fp16MxScale;
        AscendC::MicroAPI::RegTensor<uint8_t> mxScale;
        AscendC::MicroAPI::RegTensor<calcTypeInt> specialExpRegTensor;
        AscendC::MicroAPI::RegTensor<calcTypeInt> scaleReprocal;
        AscendC::MicroAPI::RegTensor<calcTypeInt> biasRegTensor;
        AscendC::MicroAPI::RegTensor<calcTypeInt> zeroRegTensor;
        AscendC::MicroAPI::RegTensor<calcTypeInt> nanRegTensor;
        AscendC::MicroAPI::RegTensor<uint8_t> out;
        AscendC::MicroAPI::UnalignReg u0;
        AscendC::MicroAPI::UnalignReg u1;
        AscendC::MicroAPI::MaskReg zeroMask;
        AscendC::MicroAPI::MaskReg infMask;
        AscendC::MicroAPI::MaskReg invalidDataMask;
        AscendC::MicroAPI::MaskReg specialDataMask;

        AscendC::MicroAPI::Duplicate(fp4MaxExpRegTensor, this->f4Emax_);
        AscendC::MicroAPI::Duplicate(maxEleRegTensor, this->maxExp_);
        AscendC::MicroAPI::Duplicate(nanRegTensor, this->nanValue_);
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, this->f8Emax_);
        AscendC::MicroAPI::Duplicate(biasRegTensor, this->maxBias_);
        AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, this->specialExp_);

        uint32_t pnum = dataLenSingleLoop;
        uint32_t tailPnum = dataLenTailLoop;
        AscendC::MicroAPI::MaskReg pnumMask = AscendC::MicroAPI::UpdateMask<calcTypeInt>(pnum);
        AscendC::MicroAPI::MaskReg tailPnumMask = AscendC::MicroAPI::UpdateMask<calcTypeInt>(tailPnum);
        AscendC::MicroAPI::Duplicate(expMaxRegTensor, 0);
        for (uint16_t i = 0; i < static_cast<uint16_t>(regLoop - 1); i++) {
            this->template LoadData<calcType>(xAddr, i * dataLenSingleLoop, xRegTensor, pnumMask);
            AscendC::MicroAPI::And(
                expRegTensor, (AscendC::MicroAPI::RegTensor<calcTypeInt>&)xRegTensor, maxEleRegTensor, pnumMask);
            AscendC::MicroAPI::Max(expMaxRegTensor, expMaxRegTensor, expRegTensor, pnumMask);
        }
        this->template LoadData<calcType>(xAddr, (regLoop - 1) * dataLenSingleLoop, xRegTensor, tailPnumMask);
        AscendC::MicroAPI::And(
            expRegTensor, (AscendC::MicroAPI::RegTensor<calcTypeInt>&)xRegTensor, maxEleRegTensor, tailPnumMask);
        AscendC::MicroAPI::Max(expRegTensor, expMaxRegTensor, expRegTensor, tailPnumMask);
        AscendC::MicroAPI::Copy<calcTypeInt, AscendC::MicroAPI::MaskMergeMode::MERGING>(
            expMaxRegTensor, expRegTensor, tailPnumMask);
        // 二分法求rowsSingleLoop行中的最大行
        AscendC::MicroAPI::DataCopy(maxExpAddr, expMaxRegTensor, pnumMask);
        AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        uint32_t maskNum = dataLenSingleLoop - expOffset;
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::UpdateMask<calcTypeInt>(maskNum);
        AscendC::MicroAPI::DataCopyUnAlignPre(u0, maxExpAddr);
        AscendC::MicroAPI::DataCopyUnAlign(expMaxRegTensor, u0, maxExpAddr);
        AscendC::MicroAPI::DataCopyUnAlignPre(u0, maxExpAddr + expOffset);
        AscendC::MicroAPI::DataCopyUnAlign(expRegTensor, u0, maxExpAddr + expOffset);
        AscendC::MicroAPI::Max(expRegTensor, expMaxRegTensor, expRegTensor, mask);
        AscendC::MicroAPI::Copy<calcTypeInt, AscendC::MicroAPI::MaskMergeMode::MERGING>(
            expMaxRegTensor, expRegTensor, mask);
        for (uint16_t i = 0; i < loopSize; i++) {
            AscendC::MicroAPI::DataCopy(maxExpAddr, expMaxRegTensor, pnumMask);
            AscendC::MicroAPI::LocalMemBar<
                AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
            expOffset /= DIGIT_TWO;
            maskNum = expOffset;
            mask = AscendC::MicroAPI::UpdateMask<calcTypeInt>(maskNum);
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, maxExpAddr);
            AscendC::MicroAPI::DataCopyUnAlign(expMaxRegTensor, u0, maxExpAddr);
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, maxExpAddr + expOffset);
            AscendC::MicroAPI::DataCopyUnAlign(expRegTensor, u0, maxExpAddr + expOffset);
            AscendC::MicroAPI::Max(expMaxRegTensor, expMaxRegTensor, expRegTensor, mask);
        }
        maskNum = static_cast<uint32_t>(dataLen);
        mask = AscendC::MicroAPI::UpdateMask<calcTypeInt>(maskNum);
        AscendC::MicroAPI::Compare<calcTypeInt, CMPMODE::NE>(infMask, expMaxRegTensor, maxEleRegTensor, mask);
        AscendC::MicroAPI::Compare<calcTypeInt, CMPMODE::NE>(zeroMask, expMaxRegTensor, zeroRegTensor, mask);
        if constexpr (IsSame<U, fp4x2_e2m1_t>::value) {
            AscendC::MicroAPI::Compare<calcTypeInt, CMPMODE::LT>(
                invalidDataMask, expMaxRegTensor, fp4MaxExpRegTensor, mask);
            AscendC::MicroAPI::Select<calcTypeInt>(
                expMaxRegTensor, fp4MaxExpRegTensor, expMaxRegTensor, invalidDataMask);
            AscendC::MicroAPI::Sub(expMaxRegTensor, expMaxRegTensor, fp4MaxExpRegTensor, mask);
        }
        AscendC::MicroAPI::ShiftRights(mxScaleRegTensor, expMaxRegTensor, this->shrNum_, mask);
        AscendC::MicroAPI::Select<calcTypeInt>(mxScaleRegTensor, mxScaleRegTensor, fp8NanRegTensor, infMask);
        AscendC::MicroAPI::Select<calcTypeInt>(mxScaleRegTensor, mxScaleRegTensor, zeroRegTensor, zeroMask);
        if constexpr (IsSame<T, half>::value) {
            AscendC::MicroAPI::Pack(fp16MxScale, mxScaleRegTensor);
            AscendC::MicroAPI::Pack(mxScale, fp16MxScale);
        } else {
            AscendC::MicroAPI::Pack(mxScale, mxScaleRegTensor);
        }
        AscendC::MicroAPI::DataCopyUnAlign(mxScaleAddr, mxScale, u1, dataLen);
        AscendC::MicroAPI::DataCopyUnAlignPost(mxScaleAddr, u1, 0);

        // 求1/scale
        AscendC::MicroAPI::Compare<calcTypeInt, CMPMODE::EQ>(specialDataMask, expMaxRegTensor, biasRegTensor, mask);
        AscendC::MicroAPI::Sub(scaleReprocal, biasRegTensor, expMaxRegTensor, mask);
        AscendC::MicroAPI::Select<calcTypeInt>(scaleReprocal, scaleReprocal, nanRegTensor, infMask);
        AscendC::MicroAPI::Select<calcTypeInt>(scaleReprocal, scaleReprocal, zeroRegTensor, zeroMask);
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
            this->template LoadData<calcType>(xAddr, i * dataLenSingleLoop, xRegTensor, pnumMask);
            CalcElement<roundMode, U, calcType, calcTypeInt>(xRegTensor, scaleReprocal, maxEleRegTensor, out, pnumMask);
            auto addr = yAddr + (i * dataLenSingleLoop) / DIGIT_TWO;
            AscendC::MicroAPI::DataCopyUnAlign(addr, out, u1, dataLenSingleLoop / DIGIT_TWO);
            AscendC::MicroAPI::DataCopyUnAlignPost(addr, u1, 0);
        }
        this->template LoadData<calcType>(xAddr, (regLoop - 1) * dataLenSingleLoop, xRegTensor, tailPnumMask);
        CalcElement<roundMode, U, calcType, calcTypeInt>(xRegTensor, scaleReprocal, maxEleRegTensor, out, tailPnumMask);
        auto addr = yAddr + ((regLoop - 1) * dataLenSingleLoop) / DIGIT_TWO;
        AscendC::MicroAPI::DataCopyUnAlign(addr, out, u1, dataLenTailLoop / DIGIT_TWO);
        AscendC::MicroAPI::DataCopyUnAlignPost(addr, u1, 0);
    }
}

} // namespace DynamicMxQuant
#endif // DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_OPTIMIZE_H
