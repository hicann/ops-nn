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
 * \file dynamic_mx_quant_not_tail_axis.h
 * \brief
 */

#ifndef DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_H
#define DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_H

#include "dynamic_mx_quant_not_tail_axis_base.h"
#include "op_kernel/math_util.h"

namespace DynamicMxQuant {
using namespace AscendC;

template <typename T, typename U, const bool ISTAIL>
class DynamicMxQuantNotTailAxis : public DynamicMxQuantBase<T, U, ISTAIL> {
public:
    __aicore__ inline DynamicMxQuantNotTailAxis(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR workspace, const DynamicMxQuantTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SplitPostAxisCompute(int64_t dataLen, int64_t blockCount);
    __aicore__ inline void SplitPreAxisCompute(int64_t ubFactor, int64_t blockSizeIdx);
    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void Compute(
        int64_t dataLen, int64_t blockCount, __ubuf__ T* xAddr, __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint8_t* yAddr);
    __aicore__ inline bool IsTailLoopInUbDim(int64_t loopIdx);
    __aicore__ inline bool IsNeedPadAndTailInAxis(int64_t curLoopIdxInAllCore);

private:
    int64_t blockLoopOffset_ = 0;
    int64_t blockOffset_ = 0;
    int64_t scaleBlockOffset_ = 0;
    int64_t bufferSize_ = 0;
    using calcType = typename std::conditional<IsSame<T, half>::value, float, T>::type;
    using calcTypeInt = typename std::conditional<IsSame<T, half>::value, uint32_t, uint16_t>::type;
};

template <typename T, typename U, const bool ISTAIL>
__aicore__ inline void DynamicMxQuantNotTailAxis<T, U, ISTAIL>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR workspace, const DynamicMxQuantTilingData* tilingData)
{
    this->BaseInit(x, y, mxScale, workspace, tilingData);
    if (this->blockIdx_ >= this->usedCoreNum_) {
        return;
    }
    blockLoopOffset_ = this->blockIdx_ * this->blockFactor_;
    if (this->ubDim_ == DIM2) {
        scaleBlockOffset_ =
            blockLoopOffset_ / this->uo_ * this->postAxisSize_ + blockLoopOffset_ % this->uo_ * this->ubFactor_;
        if (this->isPad_) {
            int64_t fullAxisNum = blockLoopOffset_ / (this->uo_ * this->blockSizeNumInAxis_);
            int64_t blockLoopMod = blockLoopOffset_ % (this->uo_ * this->blockSizeNumInAxis_);
            blockOffset_ = fullAxisNum * (this->fullBlockSizeNumInAxis_ * this->blockSize_ + this->tailBlockSize_) *
                           this->postAxisSize_;
            if (blockLoopMod <= this->uo_ * this->fullBlockSizeNumInAxis_) {
                blockOffset_ += blockLoopMod / this->uo_ * this->blockSize_ * this->postAxisSize_ +
                                blockLoopMod % this->uo_ * this->ubFactor_;
            } else {
                blockOffset_ += this->fullBlockSizeNumInAxis_ * this->blockSize_ * this->postAxisSize_ +
                                (blockLoopMod - this->uo_ * this->fullBlockSizeNumInAxis_) * this->ubFactor_;
            }
        } else {
            blockOffset_ = blockLoopOffset_ / this->uo_ * this->postAxisSize_ * this->blockSize_ +
                           blockLoopOffset_ % this->uo_ * this->ubFactor_;
        }
        bufferSize_ = this->ubFactor_ * this->blockSize_ * sizeof(T);
    } else {
        scaleBlockOffset_ = blockLoopOffset_ * this->ubFactor_ * this->postAxisSize_;
        if (this->isPad_) {
            int64_t fullAxisNum = blockLoopOffset_ * this->ubFactor_ / this->blockSizeNumInAxis_;
            int64_t blockLoopMod = blockLoopOffset_ * this->ubFactor_ % this->blockSizeNumInAxis_;
            blockOffset_ = fullAxisNum * (this->fullBlockSizeNumInAxis_ * this->blockSize_ + this->tailBlockSize_) *
                               this->postAxisSize_ +
                           blockLoopMod * this->blockSize_ * this->postAxisSize_;
        } else {
            blockOffset_ = scaleBlockOffset_ * this->blockSize_;
        }
        bufferSize_ = this->ubFactor_ * this->blockSize_ * this->postAxisSize_ * sizeof(T);
    }

    this->xGm_.SetGlobalBuffer((__gm__ T*)(x) + blockOffset_);
    this->yGm_.SetGlobalBuffer((__gm__ uint8_t*)(y) + blockOffset_ / DIGIT_TWO);
    this->mxScaleGm_.SetGlobalBuffer((__gm__ uint8_t*)(mxScale) + scaleBlockOffset_);
    this->workspaceGm_.SetGlobalBuffer((__gm__ uint8_t*)(workspace) + scaleBlockOffset_);

    bufferSize_ = Ops::Base::CeilAlign(bufferSize_, static_cast<int64_t>(Ops::Base::GetUbBlockSize()));
    this->pipe_.InitBuffer(this->inQueue_, DB_BUFFER, bufferSize_);
    this->pipe_.InitBuffer(this->mxScaleQueue_, DB_BUFFER, bufferSize_);
    this->pipe_.InitBuffer(this->outQueue_, DB_BUFFER, bufferSize_);
}

template <typename T, typename U, const bool ISTAIL>
__aicore__ inline void DynamicMxQuantNotTailAxis<T, U, ISTAIL>::Process()
{
    if (this->blockIdx_ >= this->usedCoreNum_) {
        return;
    }
    int64_t loopNum = this->isTailBlock_ ? this->tailBlockFactor_ : this->blockFactor_;
    if (this->ubDim_ == DIM2) {
        int64_t xGmOffset = 0;
        int64_t scaleGmOffset = 0;
        for (int64_t loopIdx = 1; loopIdx <= loopNum; loopIdx++) {
            int64_t curLoopIdxInAllCore = loopIdx + blockLoopOffset_;
            bool isTailLoopInUbDim = IsTailLoopInUbDim(curLoopIdxInAllCore);
            int64_t dataLen = isTailLoopInUbDim ? this->tailUbFactor_ : this->ubFactor_;
            int64_t blockCount = IsNeedPadAndTailInAxis(curLoopIdxInAllCore) ? this->tailBlockSize_ : this->blockSize_;
            this->InitCopyParams(blockCount, dataLen);
            this->CopyIn(xGmOffset);
            SplitPostAxisCompute(dataLen, blockCount);
            this->CopyOut(xGmOffset, scaleGmOffset, dataLen);
            xGmOffset += dataLen;
            scaleGmOffset += dataLen;
            if (isTailLoopInUbDim) {
                xGmOffset += this->postAxisSize_ * (blockCount - 1);
            }
        }
    } else {
        int64_t blockSizeNumInPreCore = blockLoopOffset_ * this->ubFactor_;
        int64_t scaleDataLen = this->ubFactor_ * this->postAxisSize_;
        int64_t offset = 0;
        for (int64_t loopIdx = 0; loopIdx < loopNum - 1; loopIdx++) {
            int64_t blockSizeIdx = blockSizeNumInPreCore + loopIdx * this->ubFactor_;
            int64_t dataLen = this->CalcDataLen(this->ubFactor_, blockSizeIdx, scaleDataLen);
            this->InitCopyParams(1, dataLen);
            this->CopyIn(offset);
            SplitPreAxisCompute(this->ubFactor_, blockSizeIdx);
            this->CopyOut(offset, loopIdx * scaleDataLen, scaleDataLen);
            offset += dataLen;
        }
        int64_t ubFactor = this->isTailBlock_ ? this->tailUbFactor_ : this->ubFactor_;
        scaleDataLen = ubFactor * this->postAxisSize_;
        int64_t blockSizeIdx = blockSizeNumInPreCore + (loopNum - 1) * this->ubFactor_;
        int64_t dataLen = this->CalcDataLen(ubFactor, blockSizeIdx, scaleDataLen);
        this->InitCopyParams(1, dataLen);
        this->CopyIn(offset);
        SplitPreAxisCompute(ubFactor, blockSizeIdx);
        this->CopyOut(offset, (loopNum - 1) * this->ubFactor_ * this->postAxisSize_, scaleDataLen);
    }
}

template <typename T, typename U, const bool ISTAIL>
__aicore__ inline bool DynamicMxQuantNotTailAxis<T, U, ISTAIL>::IsTailLoopInUbDim(int64_t curLoopIdxInAllCore)
{
    return curLoopIdxInAllCore >= this->uo_ && curLoopIdxInAllCore % this->uo_ == 0;
}

template <typename T, typename U, const bool ISTAIL>
__aicore__ inline bool DynamicMxQuantNotTailAxis<T, U, ISTAIL>::IsNeedPadAndTailInAxis(int64_t curLoopIdxInAllCore)
{
    return this->isPad_ &&
           ((curLoopIdxInAllCore != 0 && curLoopIdxInAllCore % (this->blockSizeNumInAxis_ * this->uo_) == 0) ||
            (curLoopIdxInAllCore % (this->blockSizeNumInAxis_ * this->uo_)) >
                this->fullBlockSizeNumInAxis_ * this->uo_);
}

template <typename T, typename U, const bool ISTAIL>
__aicore__ inline void DynamicMxQuantNotTailAxis<T, U, ISTAIL>::SplitPostAxisCompute(
    int64_t dataLen, int64_t blockCount)
{
    LocalTensor<T> x = this->inQueue_.template DeQue<T>();
    LocalTensor<uint8_t> mxScale = this->mxScaleQueue_.template AllocTensor<uint8_t>();
    LocalTensor<uint8_t> y = this->outQueue_.template AllocTensor<uint8_t>();
    auto xAddr = (__ubuf__ T*)x.GetPhyAddr();
    auto mxScaleAddr = (__ubuf__ uint8_t*)mxScale.GetPhyAddr();
    auto yAddr = (__ubuf__ uint8_t*)y.GetPhyAddr();

    if (this->roundMode_ == MODE_RINT) {
        Compute<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(dataLen, blockCount, xAddr, mxScaleAddr, yAddr);
    } else if (this->roundMode_ == MODE_ROUND) {
        Compute<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(dataLen, blockCount, xAddr, mxScaleAddr, yAddr);
    } else if (this->roundMode_ == MODE_FLOOR) {
        Compute<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(dataLen, blockCount, xAddr, mxScaleAddr, yAddr);
    }
    this->mxScaleQueue_.template EnQue(mxScale);
    this->outQueue_.template EnQue(y);
    this->inQueue_.template FreeTensor(x);
}

template <typename T, typename U, const bool ISTAIL>
__aicore__ inline void DynamicMxQuantNotTailAxis<T, U, ISTAIL>::SplitPreAxisCompute(
    int64_t ubFactor, int64_t blockSizeIdx)
{
    LocalTensor<T> x = this->inQueue_.template DeQue<T>();
    LocalTensor<uint8_t> mxScale = this->mxScaleQueue_.template AllocTensor<uint8_t>();
    LocalTensor<uint8_t> y = this->outQueue_.template AllocTensor<uint8_t>();

    int64_t offset = 0;
    for (int64_t i = 0; i < ubFactor; i++) {
        int64_t blockCount = this->BlockCountInCurCompute(blockSizeIdx + i + 1);
        auto xAddr = (__ubuf__ T*)x.GetPhyAddr() + offset;
        auto mxScaleAddr = (__ubuf__ uint8_t*)mxScale.GetPhyAddr() + i * this->postAxisSize_;
        auto yAddr = (__ubuf__ uint8_t*)y.GetPhyAddr() + offset / DIGIT_TWO;
        offset += blockCount * this->postAxisSize_;
        if (this->roundMode_ == MODE_RINT) {
            Compute<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(
                this->postAxisSize_, blockCount, xAddr, mxScaleAddr, yAddr);
        } else if (this->roundMode_ == MODE_ROUND) {
            Compute<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(
                this->postAxisSize_, blockCount, xAddr, mxScaleAddr, yAddr);
        } else if (this->roundMode_ == MODE_FLOOR) {
            Compute<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(
                this->postAxisSize_, blockCount, xAddr, mxScaleAddr, yAddr);
        }
    }
    this->mxScaleQueue_.template EnQue(mxScale);
    this->outQueue_.template EnQue(y);
    this->inQueue_.template FreeTensor(x);
}

template <typename T, typename U, const bool ISTAIL>
template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicMxQuantNotTailAxis<T, U, ISTAIL>::Compute(
    int64_t dataLen, int64_t blockCount, __ubuf__ T* xAddr, __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint8_t* yAddr)
{
    constexpr uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(calcType);           // 寄存器单次处理能处理的长度
    uint16_t regLoop = static_cast<uint16_t>(dataLen) / static_cast<uint16_t>(vfLen); // 当前loop需要处理的长度
    uint16_t tailVfLen = static_cast<uint16_t>(dataLen) % static_cast<uint16_t>(vfLen);
    int64_t outDataLenAlign = this->ubDim_ == DIM2 ?
                                  (dataLen + OUT_ELE_NUM_ONE_BLK - 1) / OUT_ELE_NUM_ONE_BLK * OUT_ELE_NUM_ONE_BLK :
                                  dataLen;
    constexpr uint16_t step = ISTAIL ? DIGIT_TWO : 1;
    if constexpr (ISTAIL) {
        tailVfLen = DIGIT_TWO;
        outDataLenAlign = 1;
    }
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<calcType> xRegTensor;
        AscendC::MicroAPI::RegTensor<calcTypeInt> expRegTensor;
        AscendC::MicroAPI::RegTensor<calcTypeInt> expMaxRegTensor;
        AscendC::MicroAPI::RegTensor<calcTypeInt> maxEleRegTensor;
        AscendC::MicroAPI::RegTensor<calcTypeInt> fp4MaxExpRegTensor;
        AscendC::MicroAPI::RegTensor<calcTypeInt> fp8NanRegTensor;
        AscendC::MicroAPI::RegTensor<calcTypeInt> mxScaleRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> fp16MxScale;
        AscendC::MicroAPI::RegTensor<uint8_t> mxScale;
        AscendC::MicroAPI::RegTensor<calcTypeInt> scaleReprocal;
        AscendC::MicroAPI::RegTensor<calcTypeInt> specialExpRegTensor;
        AscendC::MicroAPI::RegTensor<calcTypeInt> biasRegTensor;
        AscendC::MicroAPI::RegTensor<calcTypeInt> zeroRegTensor;
        AscendC::MicroAPI::RegTensor<calcTypeInt> nanRegTensor;
        AscendC::MicroAPI::RegTensor<uint8_t> out;
        AscendC::MicroAPI::UnalignReg u1;
        AscendC::MicroAPI::MaskReg infMask;
        AscendC::MicroAPI::MaskReg zeroMask;
        AscendC::MicroAPI::MaskReg invalidDataMask;
        AscendC::MicroAPI::MaskReg specialDataMask;

        AscendC::MicroAPI::Duplicate(maxEleRegTensor, this->maxExp_);
        AscendC::MicroAPI::Duplicate(fp4MaxExpRegTensor, this->f4Emax_);
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, this->f8Emax_);
        AscendC::MicroAPI::Duplicate(biasRegTensor, this->maxBias_);
        AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
        AscendC::MicroAPI::Duplicate(nanRegTensor, this->nanValue_);
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, this->specialExp_);
        for (uint16_t i = 0; i < regLoop; i++) {
            uint32_t pnum = vfLen;
            AscendC::MicroAPI::MaskReg p0 = AscendC::MicroAPI::UpdateMask<calcTypeInt>(pnum);
            this->template LoadData<calcType>(xAddr, i * vfLen, xRegTensor, p0);
            AscendC::MicroAPI::And(
                expMaxRegTensor, (AscendC::MicroAPI::RegTensor<calcTypeInt>&)xRegTensor, maxEleRegTensor, p0);
            for (uint16_t j = 1; j < static_cast<uint16_t>(blockCount); j++) {
                this->template LoadData<calcType>(xAddr, j * dataLen + i * vfLen, xRegTensor, p0);
                AscendC::MicroAPI::And(
                    expRegTensor, (AscendC::MicroAPI::RegTensor<calcTypeInt>&)xRegTensor, maxEleRegTensor, p0);
                AscendC::MicroAPI::Max(expMaxRegTensor, expMaxRegTensor, expRegTensor, p0);
            }
            AscendC::MicroAPI::Compare<calcTypeInt, CMPMODE::NE>(infMask, expMaxRegTensor, maxEleRegTensor, p0);
            AscendC::MicroAPI::Compare<calcTypeInt, CMPMODE::NE>(zeroMask, expMaxRegTensor, zeroRegTensor, p0);
            if constexpr (IsSame<U, fp4x2_e2m1_t>::value) {
                AscendC::MicroAPI::Compare<calcTypeInt, CMPMODE::LE>(
                    invalidDataMask, expMaxRegTensor, fp4MaxExpRegTensor, p0);
                AscendC::MicroAPI::Select<calcTypeInt>(
                    expMaxRegTensor, fp4MaxExpRegTensor, expMaxRegTensor, invalidDataMask);
                AscendC::MicroAPI::Sub(expMaxRegTensor, expMaxRegTensor, fp4MaxExpRegTensor, p0);
            }
            AscendC::MicroAPI::ShiftRights(mxScaleRegTensor, expMaxRegTensor, this->shrNum_, p0);
            AscendC::MicroAPI::Select<calcTypeInt>(mxScaleRegTensor, mxScaleRegTensor, fp8NanRegTensor, infMask);
            AscendC::MicroAPI::Select<calcTypeInt>(mxScaleRegTensor, mxScaleRegTensor, zeroRegTensor, zeroMask);
            if constexpr (IsSame<T, half>::value) {
                AscendC::MicroAPI::Pack(fp16MxScale, mxScaleRegTensor);
                AscendC::MicroAPI::Pack(mxScale, fp16MxScale);
            } else {
                AscendC::MicroAPI::Pack(mxScale, mxScaleRegTensor);
            }
            AscendC::MicroAPI::DataCopyUnAlign(mxScaleAddr, mxScale, u1, vfLen);
            AscendC::MicroAPI::DataCopyUnAlignPost(mxScaleAddr, u1, 0);
            // 求1/scale
            AscendC::MicroAPI::Compare<calcTypeInt, CMPMODE::EQ>(specialDataMask, expMaxRegTensor, biasRegTensor, p0);
            AscendC::MicroAPI::Sub(scaleReprocal, biasRegTensor, expMaxRegTensor, p0);
            AscendC::MicroAPI::Select<calcTypeInt>(scaleReprocal, scaleReprocal, nanRegTensor, infMask);
            AscendC::MicroAPI::Select<calcTypeInt>(scaleReprocal, scaleReprocal, zeroRegTensor, zeroMask);
            AscendC::MicroAPI::Select<calcTypeInt>(scaleReprocal, specialExpRegTensor, scaleReprocal, specialDataMask);

            // 求data value
            for (uint16_t j = 0; j < static_cast<uint16_t>(blockCount); j++) {
                this->template LoadData<calcType>(xAddr, j * dataLen + i * vfLen, xRegTensor, p0);
                CalcElement<roundMode, U, calcType, calcTypeInt>(xRegTensor, scaleReprocal, maxEleRegTensor, out, p0);
                auto addr = yAddr + (j * outDataLenAlign + i * vfLen) / DIGIT_TWO;
                AscendC::MicroAPI::DataCopyUnAlign(addr, out, u1, vfLen / DIGIT_TWO);
                AscendC::MicroAPI::DataCopyUnAlignPost(addr, u1, 0);
            }
        }

        if (tailVfLen != 0) {
            uint32_t tailPnum = tailVfLen;
            AscendC::MicroAPI::MaskReg p1 = AscendC::MicroAPI::UpdateMask<calcTypeInt>(tailPnum);
            this->template LoadData<calcType>(xAddr, regLoop * vfLen, xRegTensor, p1);
            AscendC::MicroAPI::And(
                expMaxRegTensor, (AscendC::MicroAPI::RegTensor<calcTypeInt>&)xRegTensor, maxEleRegTensor, p1);
            for (uint16_t j = 1; j < static_cast<uint16_t>(blockCount); j++) {
                this->template LoadData<calcType>(xAddr, regLoop * vfLen + j * dataLen, xRegTensor, p1);
                AscendC::MicroAPI::And(
                    expRegTensor, (AscendC::MicroAPI::RegTensor<calcTypeInt>&)xRegTensor, maxEleRegTensor, p1);
                AscendC::MicroAPI::Max(expMaxRegTensor, expMaxRegTensor, expRegTensor, p1);
            }
            AscendC::MicroAPI::Compare<calcTypeInt, CMPMODE::NE>(infMask, expMaxRegTensor, maxEleRegTensor, p1);
            AscendC::MicroAPI::Compare<calcTypeInt, CMPMODE::NE>(zeroMask, expMaxRegTensor, zeroRegTensor, p1);
            if constexpr (IsSame<U, fp4x2_e2m1_t>::value) {
                AscendC::MicroAPI::Compare<calcTypeInt, CMPMODE::LE>(
                    invalidDataMask, expMaxRegTensor, fp4MaxExpRegTensor, p1);
                AscendC::MicroAPI::Select<calcTypeInt>(
                    expMaxRegTensor, fp4MaxExpRegTensor, expMaxRegTensor, invalidDataMask);
                AscendC::MicroAPI::Sub(expMaxRegTensor, expMaxRegTensor, fp4MaxExpRegTensor, p1);
            }
            AscendC::MicroAPI::ShiftRights(mxScaleRegTensor, expMaxRegTensor, this->shrNum_, p1);
            AscendC::MicroAPI::Select<calcTypeInt>(mxScaleRegTensor, mxScaleRegTensor, fp8NanRegTensor, infMask);
            AscendC::MicroAPI::Select<calcTypeInt>(mxScaleRegTensor, mxScaleRegTensor, zeroRegTensor, zeroMask);
            if constexpr (IsSame<T, half>::value) {
                AscendC::MicroAPI::Pack(fp16MxScale, mxScaleRegTensor);
                AscendC::MicroAPI::Pack(mxScale, fp16MxScale);
            } else {
                AscendC::MicroAPI::Pack(mxScale, mxScaleRegTensor);
            }
            AscendC::MicroAPI::DataCopyUnAlign(mxScaleAddr, mxScale, u1, tailVfLen);
            AscendC::MicroAPI::DataCopyUnAlignPost(mxScaleAddr, u1, 0);
            // 求1/scale
            AscendC::MicroAPI::Compare<calcTypeInt, CMPMODE::EQ>(specialDataMask, expMaxRegTensor, biasRegTensor, p1);
            AscendC::MicroAPI::Sub(scaleReprocal, biasRegTensor, expMaxRegTensor, p1);
            AscendC::MicroAPI::Select<calcTypeInt>(scaleReprocal, specialExpRegTensor, scaleReprocal, specialDataMask);
            AscendC::MicroAPI::Select<calcTypeInt>(scaleReprocal, scaleReprocal, nanRegTensor, infMask);
            AscendC::MicroAPI::Select<calcTypeInt>(scaleReprocal, scaleReprocal, zeroRegTensor, zeroMask);
            if constexpr (ISTAIL) {
                AscendC::MicroAPI::Duplicate(scaleReprocal, scaleReprocal, p1);
            }

            // 求data value
            for (uint16_t j = 0; j < static_cast<uint16_t>(blockCount); j += step) {
                this->template LoadData<calcType>(xAddr, regLoop * vfLen + j * dataLen, xRegTensor, p1);
                CalcElement<roundMode, U, calcType, calcTypeInt>(xRegTensor, scaleReprocal, maxEleRegTensor, out, p1);
                auto addr = yAddr + (regLoop * vfLen + j * outDataLenAlign) / DIGIT_TWO;
                AscendC::MicroAPI::DataCopyUnAlign(addr, out, u1, tailVfLen / DIGIT_TWO);
                AscendC::MicroAPI::DataCopyUnAlignPost(addr, u1, 0);
            }
        }
    }
}

} // namespace DynamicMxQuant
#endif // DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_H
