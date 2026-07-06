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
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR workspace,
                                const DynamicMxQuantTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SplitPostAxisCompute(int64_t dataLen, int64_t blockCount);
    __aicore__ inline void SplitPreAxisCompute(int64_t ubFactor, int64_t blockSizeIdx);
    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode, const int64_t calcMode>
    __aicore__ inline void ComputeOcp(int64_t dataLen, int64_t blockCount, __ubuf__ T* xAddr,
                                      __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint8_t* yAddr);
    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void ComputeCuBLAS(int64_t dataLen, int64_t blockCount, __ubuf__ T* xAddr,
                                         __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint8_t* yAddr);
    __aicore__ inline bool IsTailLoopInUbDim(int64_t loopIdx);
    __aicore__ inline bool IsNeedPadAndTailInAxis(int64_t curLoopIdxInAllCore);

private:
    TBuf<QuePosition::VECCALC> maxExpBuf_;
    int64_t blockLoopOffset_ = 0;
    int64_t blockOffset_ = 0;
    int64_t scaleBlockOffset_ = 0;
    int64_t bufferSize_ = 0;
    int64_t calcMode_ = 0;
    uint32_t subNumForFP32Scale_ = 0;
    uint16_t subNumForBF16Scale_ = 0;
    float invDstTypeMax_ = 0.0;
    using calcType = typename std::conditional<IsSame<T, half>::value, float, T>::type;
    using calcTypeInt = typename std::conditional<IsSame<T, half>::value, uint32_t, uint16_t>::type;
};

template <typename T, typename U, const bool ISTAIL>
__aicore__ inline void DynamicMxQuantNotTailAxis<T, U, ISTAIL>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR mxScale,
                                                                     GM_ADDR workspace,
                                                                     const DynamicMxQuantTilingData* tilingData)
{
    invDstTypeMax_ = tilingData->invDstTypeMax;
    this->BaseInit(x, y, mxScale, workspace, tilingData);
    if (this->blockIdx_ >= this->usedCoreNum_) {
        return;
    }
    blockLoopOffset_ = this->blockIdx_ * this->blockFactor_;
    if (this->ubDim_ == DIM2) {
        scaleBlockOffset_ = blockLoopOffset_ / this->uo_ * this->postAxisSize_ +
                            blockLoopOffset_ % this->uo_ * this->ubFactor_;
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

    calcMode_ = tilingData->calcMode;
    subNumForBF16Scale_ = static_cast<uint16_t>(tilingData->subNumForScale);
    subNumForFP32Scale_ = tilingData->subNumForFP16Scale;

    this->xGm_.SetGlobalBuffer((__gm__ T*)(x) + blockOffset_);
    this->yGm_.SetGlobalBuffer((__gm__ uint8_t*)(y) + blockOffset_ / DIGIT_TWO);
    this->mxScaleGm_.SetGlobalBuffer((__gm__ uint8_t*)(mxScale) + scaleBlockOffset_);
    this->workspaceGm_.SetGlobalBuffer((__gm__ uint8_t*)(workspace) + scaleBlockOffset_);

    bufferSize_ = Ops::Base::CeilAlign(bufferSize_, static_cast<int64_t>(Ops::Base::GetUbBlockSize()));
    this->pipe_.InitBuffer(this->inQueue_, DB_BUFFER, bufferSize_);
    this->pipe_.InitBuffer(this->mxScaleQueue_, DB_BUFFER, bufferSize_);
    this->pipe_.InitBuffer(this->outQueue_, DB_BUFFER, bufferSize_);
    this->pipe_.InitBuffer(maxExpBuf_, Ops::Base::GetVRegSize());
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
__aicore__ inline void DynamicMxQuantNotTailAxis<T, U, ISTAIL>::SplitPostAxisCompute(int64_t dataLen,
                                                                                     int64_t blockCount)
{
    LocalTensor<T> x = this->inQueue_.template DeQue<T>();
    LocalTensor<uint8_t> mxScale = this->mxScaleQueue_.template AllocTensor<uint8_t>();
    LocalTensor<uint8_t> y = this->outQueue_.template AllocTensor<uint8_t>();
    auto xAddr = (__ubuf__ T*)x.GetPhyAddr();
    auto mxScaleAddr = (__ubuf__ uint8_t*)mxScale.GetPhyAddr();
    auto yAddr = (__ubuf__ uint8_t*)y.GetPhyAddr();

    if (calcMode_ == MODE_ZERO) {
        if (this->roundMode_ == MODE_RINT) {
            ComputeOcp<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT, 0>(dataLen, blockCount, xAddr, mxScaleAddr, yAddr);
        } else if (this->roundMode_ == MODE_FLOOR) {
            ComputeOcp<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR, 0>(dataLen, blockCount, xAddr, mxScaleAddr, yAddr);
        } else if (this->roundMode_ == MODE_ROUND) {
            ComputeOcp<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND, 0>(dataLen, blockCount, xAddr, mxScaleAddr, yAddr);
        }
    } else if (calcMode_ == MODE_TWO) {
        if (this->roundMode_ == MODE_RINT) {
            ComputeOcp<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT, 2>(dataLen, blockCount, xAddr, mxScaleAddr, yAddr);
        } else if (this->roundMode_ == MODE_FLOOR) {
            ComputeOcp<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR, 2>(dataLen, blockCount, xAddr, mxScaleAddr, yAddr);
        } else if (this->roundMode_ == MODE_ROUND) {
            ComputeOcp<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND, 2>(dataLen, blockCount, xAddr, mxScaleAddr, yAddr);
        }
    } else {
        if (this->roundMode_ == MODE_RINT) {
            ComputeCuBLAS<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(dataLen, blockCount, xAddr, mxScaleAddr, yAddr);
        } else if (this->roundMode_ == MODE_FLOOR) {
            ComputeCuBLAS<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(dataLen, blockCount, xAddr, mxScaleAddr, yAddr);
        } else if (this->roundMode_ == MODE_ROUND) {
            ComputeCuBLAS<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(dataLen, blockCount, xAddr, mxScaleAddr, yAddr);
        }
    }
    this->mxScaleQueue_.template EnQue(mxScale);
    this->outQueue_.template EnQue(y);
    this->inQueue_.template FreeTensor(x);
}

template <typename T, typename U, const bool ISTAIL>
__aicore__ inline void DynamicMxQuantNotTailAxis<T, U, ISTAIL>::SplitPreAxisCompute(int64_t ubFactor,
                                                                                    int64_t blockSizeIdx)
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
        if (calcMode_ == MODE_ZERO) {
            if (this->roundMode_ == MODE_RINT) {
                ComputeOcp<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT, 0>(this->postAxisSize_, blockCount, xAddr,
                                                                           mxScaleAddr, yAddr);
            } else if (this->roundMode_ == MODE_FLOOR) {
                ComputeOcp<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR, 0>(this->postAxisSize_, blockCount, xAddr,
                                                                            mxScaleAddr, yAddr);
            } else if (this->roundMode_ == MODE_ROUND) {
                ComputeOcp<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND, 0>(this->postAxisSize_, blockCount, xAddr,
                                                                            mxScaleAddr, yAddr);
            }
        } else if (calcMode_ == MODE_TWO) {
            if (this->roundMode_ == MODE_RINT) {
                ComputeOcp<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT, 2>(this->postAxisSize_, blockCount, xAddr,
                                                                           mxScaleAddr, yAddr);
            } else if (this->roundMode_ == MODE_FLOOR) {
                ComputeOcp<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR, 2>(this->postAxisSize_, blockCount, xAddr,
                                                                            mxScaleAddr, yAddr);
            } else if (this->roundMode_ == MODE_ROUND) {
                ComputeOcp<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND, 2>(this->postAxisSize_, blockCount, xAddr,
                                                                            mxScaleAddr, yAddr);
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
    this->mxScaleQueue_.template EnQue(mxScale);
    this->outQueue_.template EnQue(y);
    this->inQueue_.template FreeTensor(x);
}

template <typename T, typename U, const bool ISTAIL>
template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode, const int64_t calcMode>
__aicore__ inline void DynamicMxQuantNotTailAxis<T, U, ISTAIL>::ComputeOcp(int64_t dataLen, int64_t blockCount,
                                                                           __ubuf__ T* xAddr,
                                                                           __ubuf__ uint8_t* mxScaleAddr,
                                                                           __ubuf__ uint8_t* yAddr)
{
    constexpr uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(calcType); // 寄存器单次处理能处理的长度
    uint16_t regLoop = static_cast<uint16_t>(dataLen) / static_cast<uint16_t>(vfLen); // 当前loop需要处理的长度
    uint16_t tailVfLen = static_cast<uint16_t>(dataLen) % static_cast<uint16_t>(vfLen);
    int64_t outDataLenAlign = this->ubDim_ == DIM2 ?
                                  (dataLen + FP4_OUT_ELE_PER_BLK - 1) / FP4_OUT_ELE_PER_BLK * FP4_OUT_ELE_PER_BLK :
                                  dataLen;
    constexpr uint16_t step = ISTAIL ? DIGIT_TWO : 1;
    if constexpr (ISTAIL) {
        tailVfLen = DIGIT_TWO;
        outDataLenAlign = 1;
    }
    __VEC_SCOPE__
    {
        if constexpr (IsSame<T, float>::value) {
            return;
        }
        Reg::RegTensor<calcType> x;
        Reg::RegTensor<calcTypeInt> absReg;
        Reg::RegTensor<calcTypeInt> exp;
        Reg::RegTensor<calcTypeInt> expMax;
        Reg::RegTensor<calcTypeInt> maxEle;
        Reg::RegTensor<calcTypeInt> absForX;
        Reg::RegTensor<calcTypeInt> fp4MaxExp;
        Reg::RegTensor<calcTypeInt> fp8Nan;
        Reg::RegTensor<calcTypeInt> mxScaleInt;
        Reg::RegTensor<calcTypeInt> xMaxReg;
        Reg::RegTensor<uint16_t> fp16MxScale;
        Reg::RegTensor<uint8_t> mxScale;
        Reg::RegTensor<calcTypeInt> scaleReprocal;
        Reg::RegTensor<calcTypeInt> specialExp;
        Reg::RegTensor<calcTypeInt> bias;
        Reg::RegTensor<calcTypeInt> zero;
        Reg::RegTensor<calcTypeInt> nan;
        Reg::RegTensor<calcTypeInt> subNumForScale;
        Reg::RegTensor<uint8_t> out;
        Reg::UnalignReg u1;
        Reg::MaskReg infMask;
        Reg::MaskReg zeroMask;
        Reg::MaskReg invalidDataMask;
        Reg::MaskReg specialDataMask;

        Reg::Duplicate(maxEle, this->maxExp_);
        Reg::Duplicate(fp4MaxExp, this->f4Emax_);
        Reg::Duplicate(fp8Nan, this->f8Emax_);
        Reg::Duplicate(bias, this->maxBias_);
        Reg::Duplicate(zero, 0);
        Reg::Duplicate(nan, this->nanValue_);
        Reg::Duplicate(specialExp, this->specialExp_);

        if constexpr (calcMode == MODE_TWO) {
            Reg::Duplicate(absForX, this->absForX_);
            if constexpr (IsSame<T, half>::value) {
                Reg::Duplicate(subNumForScale, subNumForFP32Scale_);
            } else if constexpr (IsSame<T, bfloat16_t>::value) {
                Reg::Duplicate(subNumForScale, subNumForBF16Scale_);
            }

            for (uint16_t i = 0; i < regLoop; i++) {
                uint32_t pnum = vfLen;
                Reg::MaskReg p0 = Reg::UpdateMask<calcTypeInt>(pnum);
                this->template LoadData<calcType>(xAddr, i * vfLen, x, p0);
                Reg::And(xMaxReg, (Reg::RegTensor<calcTypeInt>&)x, absForX,
                         p0);                                                      // 提取指数
                for (uint16_t j = 1; j < static_cast<uint16_t>(blockCount); j++) { // 遍历block求最大值
                    this->template LoadData<calcType>(xAddr, j * dataLen + i * vfLen, x, p0);
                    Reg::And(absReg, (Reg::RegTensor<calcTypeInt>&)x, absForX, p0);
                    Reg::Max(xMaxReg, xMaxReg, absReg, p0);
                }
                // calcMode=2时，前面求数的最大值，处理最大值提取有效指数位
                Reg::And(expMax, xMaxReg, maxEle, p0);

                Reg::Compare<calcTypeInt, CMPMODE::NE>(infMask, expMax, maxEle, p0);
                if constexpr (IsSame<U, fp4x2_e2m1_t>::value) {
                    Reg::Compare<calcTypeInt, CMPMODE::LT>(invalidDataMask, expMax, fp4MaxExp, p0);
                    Reg::Sub(expMax, xMaxReg, subNumForScale, p0);
                    Reg::Select<calcTypeInt>(expMax, zero, expMax, invalidDataMask);
                    Reg::And(expMax, expMax, maxEle, p0);
                }
                Reg::ShiftRights(mxScaleInt, expMax, this->shrNum_, p0); // 计算scale
                Reg::Select<calcTypeInt>(mxScaleInt, mxScaleInt, fp8Nan, infMask);
                if constexpr (IsSame<T, half>::value) {
                    Reg::Pack(fp16MxScale, mxScaleInt);
                    Reg::Pack(mxScale, fp16MxScale);
                } else if constexpr (IsSame<T, bfloat16_t>::value) {
                    Reg::Pack(mxScale, mxScaleInt);
                }
                Reg::DataCopyUnAlign(mxScaleAddr, mxScale, u1, vfLen);
                Reg::DataCopyUnAlignPost(mxScaleAddr, u1, 0);
                // 求1/scale
                Reg::Compare<calcTypeInt, CMPMODE::EQ>(specialDataMask, expMax, bias, p0);
                Reg::Compare<calcTypeInt, CMPMODE::NE>(zeroMask, expMax, zero, p0);
                Reg::Sub(scaleReprocal, bias, expMax, p0);
                Reg::Select<calcTypeInt>(scaleReprocal, scaleReprocal, nan, infMask);
                Reg::Select<calcTypeInt>(scaleReprocal, scaleReprocal, zero, zeroMask);
                Reg::Select<calcTypeInt>(scaleReprocal, specialExp, scaleReprocal, specialDataMask);

                // 求data value
                for (uint16_t j = 0; j < static_cast<uint16_t>(blockCount); j++) {
                    this->template LoadData<calcType>(xAddr, j * dataLen + i * vfLen, x, p0);
                    CalcElement<roundMode, U, calcType, calcTypeInt>(x, scaleReprocal, maxEle, out, p0);
                    auto addr = yAddr + (j * outDataLenAlign + i * vfLen) / DIGIT_TWO;
                    Reg::DataCopyUnAlign(addr, out, u1, vfLen / DIGIT_TWO);
                    Reg::DataCopyUnAlignPost(addr, u1, 0);
                }
            }
            if (tailVfLen != 0) {
                uint32_t tailPnum = tailVfLen;
                Reg::MaskReg p1 = Reg::UpdateMask<calcTypeInt>(tailPnum);
                this->template LoadData<calcType>(xAddr, regLoop * vfLen, x, p1);
                Reg::And(xMaxReg, (Reg::RegTensor<calcTypeInt>&)x, absForX, p1);
                for (uint16_t j = 1; j < static_cast<uint16_t>(blockCount); j++) {
                    this->template LoadData<calcType>(xAddr, regLoop * vfLen + j * dataLen, x, p1);
                    Reg::And(absReg, (Reg::RegTensor<calcTypeInt>&)x, absForX, p1);
                    Reg::Max(xMaxReg, xMaxReg, absReg, p1);
                }
                Reg::And(expMax, xMaxReg, maxEle, p1);

                Reg::Compare<calcTypeInt, CMPMODE::NE>(infMask, expMax, maxEle, p1);
                if constexpr (IsSame<U, fp4x2_e2m1_t>::value) {
                    Reg::Compare<calcTypeInt, CMPMODE::LE>(invalidDataMask, expMax, fp4MaxExp, p1);
                    Reg::Sub(expMax, xMaxReg, subNumForScale, p1);
                    Reg::Select<calcTypeInt>(expMax, zero, expMax, invalidDataMask);
                    Reg::And(expMax, expMax, maxEle, p1);
                }
                Reg::ShiftRights(mxScaleInt, expMax, this->shrNum_, p1);
                Reg::Select<calcTypeInt>(mxScaleInt, mxScaleInt, fp8Nan, infMask);
                if constexpr (IsSame<T, half>::value) {
                    Reg::Pack(fp16MxScale, mxScaleInt);
                    Reg::Pack(mxScale, fp16MxScale);
                } else if constexpr (IsSame<T, bfloat16_t>::value) {
                    Reg::Pack(mxScale, mxScaleInt);
                }
                Reg::DataCopyUnAlign(mxScaleAddr, mxScale, u1, tailVfLen);
                Reg::DataCopyUnAlignPost(mxScaleAddr, u1, 0);
                // 求1/scale
                Reg::Compare<calcTypeInt, CMPMODE::NE>(zeroMask, expMax, zero, p1);
                Reg::Compare<calcTypeInt, CMPMODE::EQ>(specialDataMask, expMax, bias, p1);
                Reg::Sub(scaleReprocal, bias, expMax, p1);
                Reg::Select<calcTypeInt>(scaleReprocal, specialExp, scaleReprocal, specialDataMask);
                Reg::Select<calcTypeInt>(scaleReprocal, scaleReprocal, nan, infMask);
                Reg::Select<calcTypeInt>(scaleReprocal, scaleReprocal, zero, zeroMask);
                if constexpr (ISTAIL) {
                    Reg::Duplicate(scaleReprocal, scaleReprocal, p1);
                }

                // 求data value
                for (uint16_t j = 0; j < static_cast<uint16_t>(blockCount); j += step) {
                    this->template LoadData<calcType>(xAddr, regLoop * vfLen + j * dataLen, x, p1);
                    CalcElement<roundMode, U, calcType, calcTypeInt>(x, scaleReprocal, maxEle, out, p1);
                    auto addr = yAddr + (regLoop * vfLen + j * outDataLenAlign) / DIGIT_TWO;
                    Reg::DataCopyUnAlign(addr, out, u1, tailVfLen / DIGIT_TWO);
                    Reg::DataCopyUnAlignPost(addr, u1, 0);
                }
            }
        } else {
            for (uint16_t i = 0; i < regLoop; i++) {
                uint32_t pnum = vfLen;
                Reg::MaskReg p0 = Reg::UpdateMask<calcTypeInt>(pnum);
                this->template LoadData<calcType>(xAddr, i * vfLen, x, p0);
                Reg::And(expMax, (Reg::RegTensor<calcTypeInt>&)x, maxEle, p0);
                for (uint16_t j = 1; j < static_cast<uint16_t>(blockCount); j++) {
                    this->template LoadData<calcType>(xAddr, j * dataLen + i * vfLen, x, p0);
                    Reg::And(exp, (Reg::RegTensor<calcTypeInt>&)x, maxEle, p0);
                    Reg::Max(expMax, expMax, exp, p0);
                }
                Reg::Compare<calcTypeInt, CMPMODE::NE>(infMask, expMax, maxEle, p0);
                if constexpr (IsSame<U, fp4x2_e2m1_t>::value) {
                    Reg::Compare<calcTypeInt, CMPMODE::LE>(invalidDataMask, expMax, fp4MaxExp, p0);
                    Reg::Select<calcTypeInt>(expMax, fp4MaxExp, expMax, invalidDataMask);
                    Reg::Sub(expMax, expMax, fp4MaxExp, p0);
                }
                Reg::ShiftRights(mxScaleInt, expMax, this->shrNum_, p0);
                Reg::Select<calcTypeInt>(mxScaleInt, mxScaleInt, fp8Nan, infMask);
                if constexpr (IsSame<T, half>::value) {
                    Reg::Pack(fp16MxScale, mxScaleInt);
                    Reg::Pack(mxScale, fp16MxScale);
                } else if constexpr (IsSame<T, bfloat16_t>::value) {
                    Reg::Pack(mxScale, mxScaleInt);
                }
                Reg::DataCopyUnAlign(mxScaleAddr, mxScale, u1, vfLen);
                Reg::DataCopyUnAlignPost(mxScaleAddr, u1, 0);
                // 求1/scale
                Reg::Compare<calcTypeInt, CMPMODE::NE>(zeroMask, expMax, zero, p0);
                Reg::Compare<calcTypeInt, CMPMODE::EQ>(specialDataMask, expMax, bias, p0);
                Reg::Sub(scaleReprocal, bias, expMax, p0);
                Reg::Select<calcTypeInt>(scaleReprocal, scaleReprocal, nan, infMask);
                Reg::Select<calcTypeInt>(scaleReprocal, scaleReprocal, zero, zeroMask);
                Reg::Select<calcTypeInt>(scaleReprocal, specialExp, scaleReprocal, specialDataMask);

                // 求data value
                for (uint16_t j = 0; j < static_cast<uint16_t>(blockCount); j++) {
                    this->template LoadData<calcType>(xAddr, j * dataLen + i * vfLen, x, p0);
                    CalcElement<roundMode, U, calcType, calcTypeInt>(x, scaleReprocal, maxEle, out, p0);
                    auto addr = yAddr + (j * outDataLenAlign + i * vfLen) / DIGIT_TWO;
                    Reg::DataCopyUnAlign(addr, out, u1, vfLen / DIGIT_TWO);
                    Reg::DataCopyUnAlignPost(addr, u1, 0);
                }
            }
            if (tailVfLen != 0) {
                uint32_t tailPnum = tailVfLen;
                Reg::MaskReg p1 = Reg::UpdateMask<calcTypeInt>(tailPnum);
                this->template LoadData<calcType>(xAddr, regLoop * vfLen, x, p1);
                Reg::And(expMax, (Reg::RegTensor<calcTypeInt>&)x, maxEle, p1);
                for (uint16_t j = 1; j < static_cast<uint16_t>(blockCount); j++) {
                    this->template LoadData<calcType>(xAddr, regLoop * vfLen + j * dataLen, x, p1);
                    Reg::And(exp, (Reg::RegTensor<calcTypeInt>&)x, maxEle, p1);
                    Reg::Max(expMax, expMax, exp, p1);
                }
                Reg::Compare<calcTypeInt, CMPMODE::NE>(infMask, expMax, maxEle, p1);
                if constexpr (IsSame<U, fp4x2_e2m1_t>::value) {
                    Reg::Compare<calcTypeInt, CMPMODE::LE>(invalidDataMask, expMax, fp4MaxExp, p1);
                    Reg::Select<calcTypeInt>(expMax, fp4MaxExp, expMax, invalidDataMask);
                    Reg::Sub(expMax, expMax, fp4MaxExp, p1);
                }
                Reg::ShiftRights(mxScaleInt, expMax, this->shrNum_, p1);
                Reg::Select<calcTypeInt>(mxScaleInt, mxScaleInt, fp8Nan, infMask);
                if constexpr (IsSame<T, half>::value) {
                    Reg::Pack(fp16MxScale, mxScaleInt);
                    Reg::Pack(mxScale, fp16MxScale);
                } else if constexpr (IsSame<T, bfloat16_t>::value) {
                    Reg::Pack(mxScale, mxScaleInt);
                }
                Reg::DataCopyUnAlign(mxScaleAddr, mxScale, u1, tailVfLen);
                Reg::DataCopyUnAlignPost(mxScaleAddr, u1, 0);
                // 求1/scale
                Reg::Compare<calcTypeInt, CMPMODE::NE>(zeroMask, expMax, zero, p1);
                Reg::Compare<calcTypeInt, CMPMODE::EQ>(specialDataMask, expMax, bias, p1);
                Reg::Sub(scaleReprocal, bias, expMax, p1);
                Reg::Select<calcTypeInt>(scaleReprocal, specialExp, scaleReprocal, specialDataMask);
                Reg::Select<calcTypeInt>(scaleReprocal, scaleReprocal, nan, infMask);
                Reg::Select<calcTypeInt>(scaleReprocal, scaleReprocal, zero, zeroMask);
                if constexpr (ISTAIL) {
                    Reg::Duplicate(scaleReprocal, scaleReprocal, p1);
                }

                // 求data value
                for (uint16_t j = 0; j < static_cast<uint16_t>(blockCount); j += step) {
                    this->template LoadData<calcType>(xAddr, regLoop * vfLen + j * dataLen, x, p1);
                    CalcElement<roundMode, U, calcType, calcTypeInt>(x, scaleReprocal, maxEle, out, p1);
                    auto addr = yAddr + (regLoop * vfLen + j * outDataLenAlign) / DIGIT_TWO;
                    Reg::DataCopyUnAlign(addr, out, u1, tailVfLen / DIGIT_TWO);
                    Reg::DataCopyUnAlignPost(addr, u1, 0);
                }
            }
        }
    }
}

template <typename T, typename U, const bool ISTAIL>
template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicMxQuantNotTailAxis<T, U, ISTAIL>::ComputeCuBLAS(int64_t dataLen, int64_t blockCount,
                                                                              __ubuf__ T* xAddr,
                                                                              __ubuf__ uint8_t* mxScaleAddr,
                                                                              __ubuf__ uint8_t* yAddr)
{
    constexpr uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(calcType); // 寄存器单次处理能处理的长度
    uint16_t regLoop = static_cast<uint16_t>(dataLen) / static_cast<uint16_t>(vfLen); // 当前loop需要处理的长度
    uint16_t tailVfLen = static_cast<uint16_t>(dataLen) % static_cast<uint16_t>(vfLen);
    int64_t outDataLenAlign = this->ubDim_ == DIM2 ?
                                  (dataLen + FP4_OUT_ELE_PER_BLK - 1) / FP4_OUT_ELE_PER_BLK * FP4_OUT_ELE_PER_BLK :
                                  dataLen;
    constexpr uint16_t step = ISTAIL ? DIGIT_TWO : 1;
    if constexpr (ISTAIL) {
        tailVfLen = DIGIT_TWO;
        outDataLenAlign = 1;
    }
    __VEC_SCOPE__
    {
        if constexpr (IsSame<T, float>::value) {
            return;
        }
        Reg::RegTensor<calcType> xReg;
        Reg::RegTensor<calcTypeInt> absReg;
        Reg::RegTensor<calcTypeInt> xMaxReg;
        Reg::RegTensor<calcTypeInt> x2MaxReg;
        Reg::RegTensor<uint32_t> xFP32MaxReg;
        Reg::RegTensor<uint32_t> xFP32MaxReg2;
        Reg::RegTensor<uint32_t> expFP32Reg; // 指数
        Reg::RegTensor<uint32_t> expFP32Reg2;
        Reg::RegTensor<uint32_t> manFP32Reg; // 尾数
        Reg::RegTensor<uint32_t> manFP32Reg2;
        Reg::RegTensor<uint32_t> extractExpReg; // 指数 + 1
        Reg::RegTensor<uint32_t> extractExpReg2;
        Reg::RegTensor<uint16_t> expBF16Reg;
        Reg::RegTensor<uint16_t> expBF16Reg2;
        Reg::RegTensor<uint8_t> mxScale;
        Reg::RegTensor<uint8_t> out;
        Reg::RegTensor<calcTypeInt> scaleReprocal; // 1/scale
        Reg::RegTensor<uint32_t> zeroRegTensor32;
        Reg::RegTensor<uint16_t> zeroRegTensor16;
        Reg::RegTensor<float> invDstTypeMax;

        Reg::RegTensor<calcTypeInt> absForXReg;
        Reg::RegTensor<calcTypeInt> biasReg;
        Reg::RegTensor<calcTypeInt> infForXReg;
        Reg::RegTensor<uint32_t> infForFP32Reg;
        Reg::RegTensor<float> dstTypeMaxReg;
        Reg::RegTensor<calcTypeInt> nan;
        Reg::RegTensor<calcTypeInt> specialExp;
        Reg::RegTensor<uint32_t> expAndreShareExpFP32RegTensor;
        Reg::RegTensor<uint32_t> manAndmxScaleFP32RegTensor;

        Reg::UnalignReg u0;
        Reg::UnalignReg u1;
        Reg::MaskReg p2;
        Reg::MaskReg infMask;
        Reg::MaskReg specialDataMask;
        Reg::MaskReg preMaskScale = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg preMaskScale2 = Reg::CreateMask<bfloat16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg p0;
        Reg::MaskReg p1;
        Reg::MaskReg pregAll16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();

        Reg::Duplicate(dstTypeMaxReg, this->dstTypeMax_);
        Reg::Duplicate(biasReg, this->maxBias_);
        Reg::Duplicate(absForXReg, this->absForX_);
        Reg::Duplicate(infForXReg, this->maxExp_);
        Reg::Duplicate(infForFP32Reg, FP32_MX_MAN_MASK);
        Reg::Duplicate(nan, this->nanValue_);
        Reg::Duplicate(specialExp, this->specialExp_);
        Reg::Duplicate(zeroRegTensor32, 0);
        Reg::Duplicate(zeroRegTensor16, 0);
        Reg::Duplicate(xMaxReg, 0);
        Reg::Duplicate(x2MaxReg, 0);
        Reg::Duplicate(invDstTypeMax, invDstTypeMax_);

        static constexpr Reg::CastTrait castTraitZero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitOne = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        uint32_t pnum = vfLen;
        p0 = AscendC::Reg::UpdateMask<calcTypeInt>(pnum);
        for (uint16_t i = 0; i < regLoop; i++) {
            this->template LoadData<calcType>(xAddr, i * vfLen, xReg, p0);
            AscendC::Reg::And(xMaxReg, (Reg::RegTensor<calcTypeInt>&)xReg, absForXReg, p0);
            for (uint16_t j = 1; j < static_cast<uint16_t>(blockCount); j++) {
                this->template LoadData<calcType>(xAddr, j * dataLen + i * vfLen, xReg, p0);
                AscendC::Reg::And(absReg, (Reg::RegTensor<calcTypeInt>&)xReg, absForXReg, p0);
                AscendC::Reg::Max(xMaxReg, xMaxReg, absReg, p0);
            }
            // 求scale
            if constexpr (IsSame<T, bfloat16_t>::value) {
                AscendC::Reg::Interleave(xMaxReg, zeroRegTensor16, xMaxReg, zeroRegTensor16);
                AscendC::Reg::Cast<float, T, castTraitZero>((AscendC::Reg::RegTensor<float>&)xFP32MaxReg,
                                                            (AscendC::Reg::RegTensor<T>&)xMaxReg, pregAll16);
                AscendC::Reg::Cast<float, T, castTraitZero>((AscendC::Reg::RegTensor<float>&)xFP32MaxReg2,
                                                            (AscendC::Reg::RegTensor<T>&)zeroRegTensor16, pregAll16);
                AscendC::Reg::Mul((AscendC::Reg::RegTensor<float>&)xFP32MaxReg,
                                  (AscendC::Reg::RegTensor<float>&)xFP32MaxReg, invDstTypeMax, preMaskScale);
                AscendC::Reg::Mul((AscendC::Reg::RegTensor<float>&)xFP32MaxReg2,
                                  (AscendC::Reg::RegTensor<float>&)xFP32MaxReg2, invDstTypeMax, preMaskScale);
                // 右移获取指数位
                AscendC::Reg::ShiftRights(expFP32Reg, xFP32MaxReg, FP32_SHR_NUM, preMaskScale);
                AscendC::Reg::ShiftRights(expFP32Reg2, xFP32MaxReg2, FP32_SHR_NUM, preMaskScale);
                // And获取尾数位
                AscendC::Reg::And(manFP32Reg, xFP32MaxReg, infForFP32Reg, preMaskScale);
                AscendC::Reg::And(manFP32Reg2, xFP32MaxReg2, infForFP32Reg, preMaskScale);
                AscendC::Reg::CompareScalar<uint32_t, CMPMODE::GT>(p1, expFP32Reg, FP32_NUMBER_ZERO, preMaskScale);
                AscendC::Reg::CompareScalar<uint32_t, CMPMODE::LT>(p1, expFP32Reg, FP32_NUMBER_254, p1);
                AscendC::Reg::CompareScalar<uint32_t, CMPMODE::GT>(p1, manFP32Reg, FP32_NUMBER_ZERO, p1);
                AscendC::Reg::CompareScalar<uint32_t, CMPMODE::GT>(p2, expFP32Reg2, FP32_NUMBER_ZERO, preMaskScale);
                AscendC::Reg::CompareScalar<uint32_t, CMPMODE::LT>(p2, expFP32Reg2, FP32_NUMBER_254, p2);
                AscendC::Reg::CompareScalar<uint32_t, CMPMODE::GT>(p2, manFP32Reg2, FP32_NUMBER_ZERO, p2);
                AscendC::Reg::Adds(extractExpReg, expFP32Reg, 1, preMaskScale);
                AscendC::Reg::Adds(extractExpReg2, expFP32Reg2, 1, preMaskScale);
                // 根据情况选择指数位是否加一
                AscendC::Reg::Select<uint32_t>(expFP32Reg, extractExpReg, expFP32Reg, p1);
                AscendC::Reg::Select<uint32_t>(expFP32Reg2, extractExpReg2, expFP32Reg2, p2);
                AscendC::Reg::DeInterleave(
                    (Reg::RegTensor<uint16_t>&)expFP32Reg, (Reg::RegTensor<uint16_t>&)expFP32Reg2,
                    (Reg::RegTensor<uint16_t>&)expFP32Reg, (Reg::RegTensor<uint16_t>&)expFP32Reg2);
                AscendC::Reg::Pack(mxScale, (Reg::RegTensor<uint16_t>&)expFP32Reg);
                AscendC::Reg::DataCopyUnAlign(mxScaleAddr, mxScale, u1, vfLen);
                AscendC::Reg::DataCopyUnAlignPost(mxScaleAddr, u1, 0);
                AscendC::Reg::ShiftLefts((Reg::RegTensor<uint16_t>&)expFP32Reg, (Reg::RegTensor<uint16_t>&)expFP32Reg,
                                         BF16_SHR_NUM, pregAll16);
                AscendC::Reg::Compare<calcTypeInt, CMPMODE::NE>(infMask, (Reg::RegTensor<uint16_t>&)expFP32Reg,
                                                                infForXReg, pregAll16);
                AscendC::Reg::Compare<calcTypeInt, CMPMODE::EQ>(specialDataMask, (Reg::RegTensor<uint16_t>&)expFP32Reg,
                                                                biasReg, pregAll16);
                AscendC::Reg::Sub(scaleReprocal, biasReg, (Reg::RegTensor<uint16_t>&)expFP32Reg, pregAll16);
                AscendC::Reg::Select<calcTypeInt>(scaleReprocal, scaleReprocal, nan, infMask);
                AscendC::Reg::Select<calcTypeInt>(scaleReprocal, specialExp, scaleReprocal, specialDataMask);
                for (uint16_t j = 0; j < static_cast<uint16_t>(blockCount); j++) {
                    this->template LoadData<calcType>(xAddr, j * dataLen + i * vfLen, xReg, p0);
                    CalcElement<roundMode, U, calcType, calcTypeInt>(xReg, scaleReprocal, infForXReg, out,
                                                                     preMaskScale2);
                    auto addr = yAddr + (j * outDataLenAlign + i * vfLen) / DIGIT_TWO;
                    AscendC::Reg::DataCopyUnAlign(addr, out, u1, vfLen / DIGIT_TWO);
                    AscendC::Reg::DataCopyUnAlignPost(addr, u1, 0);
                }
            } else if constexpr (IsSame<T, half>::value) {
                AscendC::Reg::Mul((AscendC::Reg::RegTensor<float>&)xFP32MaxReg,
                                  (AscendC::Reg::RegTensor<float>&)xMaxReg, invDstTypeMax, preMaskScale);
                // 右移获取指数位
                AscendC::Reg::ShiftRights(expFP32Reg, xFP32MaxReg, FP32_SHR_NUM, preMaskScale);
                // And获取尾数位
                AscendC::Reg::And(manFP32Reg, xFP32MaxReg, infForFP32Reg, preMaskScale);
                AscendC::Reg::CompareScalar<uint32_t, CMPMODE::GT>(p1, expFP32Reg, FP32_NUMBER_ZERO, preMaskScale);
                AscendC::Reg::CompareScalar<uint32_t, CMPMODE::LT>(p1, expFP32Reg, FP32_NUMBER_254, p1);
                AscendC::Reg::CompareScalar<uint32_t, CMPMODE::GT>(p1, manFP32Reg, FP32_NUMBER_ZERO, p1);
                AscendC::Reg::Adds(extractExpReg, expFP32Reg, 1, preMaskScale);
                // 根据情况选择指数位是否加一
                AscendC::Reg::Select<uint32_t>(expFP32Reg, extractExpReg, expFP32Reg, p1);

                AscendC::Reg::Pack(expBF16Reg, expFP32Reg);
                AscendC::Reg::Pack(mxScale, expBF16Reg);
                AscendC::Reg::DataCopyUnAlign(mxScaleAddr, mxScale, u1, vfLen);
                AscendC::Reg::DataCopyUnAlignPost(mxScaleAddr, u1, 0);

                AscendC::Reg::ShiftLefts(expFP32Reg, expFP32Reg, FP32_SHR_NUM, preMaskScale);

                // 求1/scale
                AscendC::Reg::Compare<calcTypeInt, CMPMODE::NE>(infMask, expFP32Reg, infForXReg, preMaskScale);
                AscendC::Reg::Compare<calcTypeInt, CMPMODE::EQ>(specialDataMask, expFP32Reg, biasReg, preMaskScale);
                AscendC::Reg::Sub(scaleReprocal, biasReg, expFP32Reg, preMaskScale);
                AscendC::Reg::Select<calcTypeInt>(scaleReprocal, scaleReprocal, nan, infMask);
                AscendC::Reg::Select<calcTypeInt>(scaleReprocal, specialExp, scaleReprocal, specialDataMask);
                // 求data value
                for (uint16_t j = 0; j < static_cast<uint16_t>(blockCount); j++) {
                    this->template LoadData<calcType>(xAddr, j * dataLen + i * vfLen, xReg, p0);
                    CalcElement<roundMode, U, calcType, calcTypeInt>(xReg, scaleReprocal, infForXReg, out, p0);
                    auto addr = yAddr + (j * outDataLenAlign + i * vfLen) / DIGIT_TWO;
                    AscendC::Reg::DataCopyUnAlign(addr, out, u1, vfLen / DIGIT_TWO);
                    AscendC::Reg::DataCopyUnAlignPost(addr, u1, 0);
                }
            }
        }
        uint32_t tailPnum = tailVfLen;
        p2 = Reg::UpdateMask<calcTypeInt>(tailPnum);
        if (tailVfLen != 0) {
            this->template LoadData<calcType>(xAddr, regLoop * vfLen, xReg, p2);
            AscendC::Reg::And(x2MaxReg, (Reg::RegTensor<calcTypeInt>&)xReg, absForXReg, p2);
            for (uint16_t j = 1; j < static_cast<uint16_t>(blockCount); j++) {
                this->template LoadData<calcType>(xAddr, regLoop * vfLen + j * dataLen, xReg, p2);
                AscendC::Reg::And(absReg, (Reg::RegTensor<calcTypeInt>&)xReg, absForXReg, p2);
                AscendC::Reg::Max(x2MaxReg, x2MaxReg, absReg, p2);
            }
            if constexpr (IsSame<T, bfloat16_t>::value) {
                AscendC::Reg::Interleave(x2MaxReg, zeroRegTensor16, x2MaxReg, zeroRegTensor16);
                AscendC::Reg::Cast<float, T, castTraitZero>((AscendC::Reg::RegTensor<float>&)xFP32MaxReg,
                                                            (AscendC::Reg::RegTensor<T>&)x2MaxReg, pregAll16);
                AscendC::Reg::Cast<float, T, castTraitZero>((AscendC::Reg::RegTensor<float>&)xFP32MaxReg2,
                                                            (AscendC::Reg::RegTensor<T>&)zeroRegTensor16, pregAll16);
                AscendC::Reg::Mul((AscendC::Reg::RegTensor<float>&)xFP32MaxReg,
                                  (AscendC::Reg::RegTensor<float>&)xFP32MaxReg, invDstTypeMax, preMaskScale);
                AscendC::Reg::Mul((AscendC::Reg::RegTensor<float>&)xFP32MaxReg2,
                                  (AscendC::Reg::RegTensor<float>&)xFP32MaxReg2, invDstTypeMax, preMaskScale);

                AscendC::Reg::ShiftRights(expFP32Reg, xFP32MaxReg, FP32_SHR_NUM, preMaskScale);
                AscendC::Reg::ShiftRights(expFP32Reg2, xFP32MaxReg2, FP32_SHR_NUM, preMaskScale);
                // And获取尾数位
                AscendC::Reg::And(manFP32Reg, xFP32MaxReg, infForFP32Reg, preMaskScale);
                AscendC::Reg::And(manFP32Reg2, xFP32MaxReg2, infForFP32Reg, preMaskScale);
                AscendC::Reg::CompareScalar<uint32_t, CMPMODE::GT>(p1, expFP32Reg, FP32_NUMBER_ZERO, preMaskScale);
                AscendC::Reg::CompareScalar<uint32_t, CMPMODE::LT>(p1, expFP32Reg, FP32_NUMBER_254, p1);
                AscendC::Reg::CompareScalar<uint32_t, CMPMODE::GT>(p1, manFP32Reg, FP32_NUMBER_ZERO, p1);
                AscendC::Reg::CompareScalar<uint32_t, CMPMODE::GT>(p2, expFP32Reg2, FP32_NUMBER_ZERO, preMaskScale);
                AscendC::Reg::CompareScalar<uint32_t, CMPMODE::LT>(p2, expFP32Reg2, FP32_NUMBER_254, p2);
                AscendC::Reg::CompareScalar<uint32_t, CMPMODE::GT>(p2, manFP32Reg2, FP32_NUMBER_ZERO, p2);
                AscendC::Reg::Adds(extractExpReg, expFP32Reg, 1, preMaskScale);
                AscendC::Reg::Adds(extractExpReg2, expFP32Reg2, 1, preMaskScale);
                // 根据情况选择指数位是否加一
                AscendC::Reg::Select<uint32_t>(expFP32Reg, extractExpReg, expFP32Reg, p1);
                AscendC::Reg::Select<uint32_t>(expFP32Reg2, extractExpReg2, expFP32Reg2, p2);
                AscendC::Reg::DeInterleave(
                    (Reg::RegTensor<uint16_t>&)expFP32Reg, (Reg::RegTensor<uint16_t>&)expFP32Reg2,
                    (Reg::RegTensor<uint16_t>&)expFP32Reg, (Reg::RegTensor<uint16_t>&)expFP32Reg2);
                AscendC::Reg::Pack(mxScale, (Reg::RegTensor<uint16_t>&)expFP32Reg);
                AscendC::Reg::DataCopyUnAlign(mxScaleAddr, mxScale, u1, tailVfLen);
                AscendC::Reg::DataCopyUnAlignPost(mxScaleAddr, u1, 0);

                AscendC::Reg::ShiftLefts((Reg::RegTensor<uint16_t>&)expFP32Reg, (Reg::RegTensor<uint16_t>&)expFP32Reg,
                                         BF16_SHR_NUM, pregAll16);
                AscendC::Reg::Compare<calcTypeInt, CMPMODE::NE>(infMask, (Reg::RegTensor<uint16_t>&)expFP32Reg,
                                                                infForXReg, pregAll16);
                AscendC::Reg::Compare<calcTypeInt, CMPMODE::EQ>(specialDataMask, (Reg::RegTensor<uint16_t>&)expFP32Reg,
                                                                biasReg, pregAll16);
                AscendC::Reg::Sub(scaleReprocal, biasReg, (Reg::RegTensor<uint16_t>&)expFP32Reg, pregAll16);
                AscendC::Reg::Select<calcTypeInt>(scaleReprocal, scaleReprocal, nan, infMask);
                AscendC::Reg::Select<calcTypeInt>(scaleReprocal, specialExp, scaleReprocal, specialDataMask);
                if constexpr (ISTAIL) {
                    Reg::Duplicate(scaleReprocal, scaleReprocal, preMaskScale2);
                }
                for (uint16_t j = 0; j < static_cast<uint16_t>(blockCount); j += step) {
                    this->template LoadData<calcType>(xAddr, regLoop * vfLen + j * dataLen, xReg, p2);
                    CalcElement<roundMode, U, calcType, calcTypeInt>(xReg, scaleReprocal, infForXReg, out,
                                                                     preMaskScale2);
                    auto addr = yAddr + (regLoop * vfLen + j * outDataLenAlign) / DIGIT_TWO;
                    Reg::DataCopyUnAlign(addr, out, u1, tailVfLen / DIGIT_TWO);
                    Reg::DataCopyUnAlignPost(addr, u1, 0);
                }

            } else if constexpr (IsSame<T, half>::value) {
                AscendC::Reg::Mul((Reg::RegTensor<float>&)xFP32MaxReg, (Reg::RegTensor<float>&)x2MaxReg, invDstTypeMax,
                                  preMaskScale);
                Reg::ShiftRights(expFP32Reg, xFP32MaxReg, FP32_SHR_NUM, preMaskScale);

                // And获取尾数位
                AscendC::Reg::And(manFP32Reg, xFP32MaxReg, infForFP32Reg, preMaskScale);
                AscendC::Reg::CompareScalar<uint32_t, CMPMODE::GT>(p1, expFP32Reg, FP32_NUMBER_ZERO, preMaskScale);
                AscendC::Reg::CompareScalar<uint32_t, CMPMODE::LT>(p1, expFP32Reg, FP32_NUMBER_254, p1);
                AscendC::Reg::CompareScalar<uint32_t, CMPMODE::GT>(p1, manFP32Reg, FP32_NUMBER_ZERO, p1);
                AscendC::Reg::Adds(extractExpReg, expFP32Reg, 1, preMaskScale);
                // 根据情况选择指数位是否加一
                AscendC::Reg::Select<uint32_t>(expFP32Reg, extractExpReg, expFP32Reg, p1);

                AscendC::Reg::Pack(expBF16Reg, expFP32Reg);
                AscendC::Reg::Pack(mxScale, expBF16Reg);
                AscendC::Reg::DataCopyUnAlign(mxScaleAddr, mxScale, u1, tailVfLen);
                AscendC::Reg::DataCopyUnAlignPost(mxScaleAddr, u1, 0);

                AscendC::Reg::ShiftLefts(expFP32Reg, expFP32Reg, FP32_SHR_NUM, preMaskScale);
                // 求1/scale
                AscendC::Reg::Compare<calcTypeInt, CMPMODE::NE>(infMask, expFP32Reg, infForXReg, preMaskScale);
                AscendC::Reg::Compare<calcTypeInt, CMPMODE::EQ>(specialDataMask, expFP32Reg, biasReg, preMaskScale);
                AscendC::Reg::Sub(scaleReprocal, biasReg, expFP32Reg, preMaskScale);

                AscendC::Reg::Select<calcTypeInt>(scaleReprocal, scaleReprocal, nan, infMask);
                AscendC::Reg::Select<calcTypeInt>(scaleReprocal, specialExp, scaleReprocal, specialDataMask);
                if constexpr (ISTAIL) {
                    Reg::Duplicate(scaleReprocal, scaleReprocal, preMaskScale);
                }

                // 求data value
                for (uint16_t j = 0; j < static_cast<uint16_t>(blockCount); j += step) {
                    this->template LoadData<calcType>(xAddr, regLoop * vfLen + j * dataLen, xReg, p2);
                    CalcElement<roundMode, U, calcType, calcTypeInt>(xReg, scaleReprocal, infForXReg, out, p2);
                    auto addr = yAddr + (regLoop * vfLen + j * outDataLenAlign) / DIGIT_TWO;
                    Reg::DataCopyUnAlign(addr, out, u1, tailVfLen / DIGIT_TWO);
                    Reg::DataCopyUnAlignPost(addr, u1, 0);
                }
            }
        }
    }
}
} // namespace DynamicMxQuant
#endif // DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_H
