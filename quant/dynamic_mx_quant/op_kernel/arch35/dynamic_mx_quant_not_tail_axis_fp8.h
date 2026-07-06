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
 * \file dynamic_mx_quant_not_tail_axis_fp8.h
 * \brief
 */

#ifndef DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_FP8_H
#define DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_FP8_H

#include "dynamic_mx_quant_not_tail_axis_base_fp8.h"
#include "op_kernel/math_util.h"

namespace DynamicMxQuant {
using namespace AscendC;

template <typename T, typename U, const bool isTail>
class DynamicMxQuantNotTailAxisFP8 : public DynamicMxQuantBaseFP8<T, U, isTail> {
public:
    __aicore__ inline DynamicMxQuantNotTailAxisFP8(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR workspace,
                                const DynamicMxQuantTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SplitPostAxisCompute(int64_t dataLen, int64_t blockCount);
    __aicore__ inline void SplitPreAxisCompute(int64_t ubFactor, int64_t blockSizeIdx);
    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void ComputeOCP(int64_t dataLen, int64_t blockCount, __ubuf__ T* xAddr,
                                      __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint8_t* yAddr);
    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void ComputecuBLAS(int64_t dataLen, int64_t blockCount, __ubuf__ T* xAddr,
                                         __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint8_t* yAddr);
    __aicore__ inline bool IsTailLoopInUbDim(int64_t loopIdx);
    __aicore__ inline bool IsNeedPadAndTailInAxis(int64_t curLoopIdxInAllCore);

private:
    int64_t blockLoopOffset_ = 0;
    int64_t blockOffset_ = 0;
    int64_t scaleBlockOffset_ = 0;
    int64_t bufferSize_ = 0;
};

template <typename T, typename U, const bool isTail>
__aicore__ inline void DynamicMxQuantNotTailAxisFP8<T, U, isTail>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR mxScale,
                                                                        GM_ADDR workspace,
                                                                        const DynamicMxQuantTilingData* tilingData)
{
    this->BaseInit(x, y, mxScale, workspace, tilingData);
    if (this->blockIdx_ >= this->usedCoreNum_) {
        return;
    }
    blockLoopOffset_ = this->blockIdx_ * this->blockFactor_;
    if (this->ubDim_ == DIM2) { // 非尾轴场景，切axis之后的轴合轴成的post
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
    } else { // 尾轴场景，切axis之前的轴合轴成的pre
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
    this->yGm_.SetGlobalBuffer((__gm__ uint8_t*)(y) + blockOffset_);
    this->mxScaleGm_.SetGlobalBuffer((__gm__ uint8_t*)(mxScale) + scaleBlockOffset_);
    this->workspaceGm_.SetGlobalBuffer((__gm__ uint8_t*)(workspace) + scaleBlockOffset_);

    bufferSize_ = Ops::Base::CeilAlign(bufferSize_, static_cast<int64_t>(Ops::Base::GetUbBlockSize()));
    this->pipe_.InitBuffer(this->inQueue_, DB_BUFFER, bufferSize_);
    this->pipe_.InitBuffer(this->mxScaleQueue_, DB_BUFFER, bufferSize_);
    this->pipe_.InitBuffer(this->outQueue_, DB_BUFFER, bufferSize_);
}

template <typename T, typename U, const bool isTail>
__aicore__ inline void DynamicMxQuantNotTailAxisFP8<T, U, isTail>::Process()
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

template <typename T, typename U, const bool isTail>
__aicore__ inline bool DynamicMxQuantNotTailAxisFP8<T, U, isTail>::IsTailLoopInUbDim(int64_t curLoopIdxInAllCore)
{
    return curLoopIdxInAllCore >= this->uo_ && curLoopIdxInAllCore % this->uo_ == 0;
}

template <typename T, typename U, const bool isTail>
__aicore__ inline bool DynamicMxQuantNotTailAxisFP8<T, U, isTail>::IsNeedPadAndTailInAxis(int64_t curLoopIdxInAllCore)
{
    return this->isPad_ &&
           ((curLoopIdxInAllCore != 0 && curLoopIdxInAllCore % (this->blockSizeNumInAxis_ * this->uo_) == 0) ||
            (curLoopIdxInAllCore % (this->blockSizeNumInAxis_ * this->uo_)) >
                this->fullBlockSizeNumInAxis_ * this->uo_);
}

template <typename T, typename U, const bool isTail>
__aicore__ inline void DynamicMxQuantNotTailAxisFP8<T, U, isTail>::SplitPostAxisCompute(int64_t dataLen,
                                                                                        int64_t blockCount)
{
    LocalTensor<T> x = this->inQueue_.template DeQue<T>();
    LocalTensor<uint8_t> mxScale = this->mxScaleQueue_.template AllocTensor<uint8_t>();
    LocalTensor<uint8_t> y = this->outQueue_.template AllocTensor<uint8_t>();
    auto xAddr = (__ubuf__ T*)x.GetPhyAddr();
    auto yAddr = (__ubuf__ uint8_t*)y.GetPhyAddr();
    auto mxScaleAddr = (__ubuf__ uint8_t*)mxScale.GetPhyAddr();
    if (this->scaleAlg_ == 0) {
        ComputeOCP<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(dataLen, blockCount, xAddr, mxScaleAddr, yAddr);
    } else {
        ComputecuBLAS<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(dataLen, blockCount, xAddr, mxScaleAddr, yAddr);
    }
    this->mxScaleQueue_.template EnQue(mxScale);
    this->outQueue_.template EnQue(y);
    this->inQueue_.template FreeTensor(x);
}

template <typename T, typename U, const bool isTail>
__aicore__ inline void DynamicMxQuantNotTailAxisFP8<T, U, isTail>::SplitPreAxisCompute(int64_t ubFactor,
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
        auto yAddr = (__ubuf__ uint8_t*)y.GetPhyAddr() + offset;
        offset += blockCount * this->postAxisSize_;
        if (this->scaleAlg_ == 0) {
            ComputeOCP<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(this->postAxisSize_, blockCount, xAddr, mxScaleAddr,
                                                                    yAddr);
        } else {
            ComputecuBLAS<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(this->postAxisSize_, blockCount, xAddr,
                                                                       mxScaleAddr, yAddr);
        }
    }
    this->mxScaleQueue_.template EnQue(mxScale);
    this->outQueue_.template EnQue(y);
    this->inQueue_.template FreeTensor(x);
}

template <typename T, typename U, const bool isTail>
template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicMxQuantNotTailAxisFP8<T, U, isTail>::ComputeOCP(int64_t dataLen, int64_t blockCount,
                                                                              __ubuf__ T* xAddr,
                                                                              __ubuf__ uint8_t* mxScaleAddr,
                                                                              __ubuf__ uint8_t* yAddr)
{
    constexpr uint32_t vfNum16 = Ops::Base::GetVRegSize() / sizeof(T); // 寄存器单次处理能处理的长度
    constexpr uint32_t vfNum32 = Ops::Base::GetVRegSize() / sizeof(float); // cast到FP32后单个寄存器中的元素个数
    uint16_t regLoop = static_cast<uint16_t>(dataLen) / static_cast<uint16_t>(vfNum16); // 当前loop需要处理的长度
    uint16_t tailVfLen = static_cast<uint16_t>(dataLen) % static_cast<uint16_t>(vfNum16);
    uint32_t loopNum0 = dataLen <= vfNum32 ? vfNum16 : vfNum32;
    uint32_t loopNum1 = dataLen <= vfNum32 ? 0 : (vfNum16 - vfNum32);
    uint32_t tailLoopNum0 = tailVfLen <= vfNum32 ? tailVfLen : vfNum32;
    uint32_t tailLoopNum1 = tailVfLen <= vfNum32 ? 0 : (tailVfLen - vfNum32);
    int64_t outDataLenAlign = this->ubDim_ == DIM2 ?
                                  (dataLen + FP8_OUT_ELE_PER_BLK - 1) / FP8_OUT_ELE_PER_BLK * FP8_OUT_ELE_PER_BLK :
                                  dataLen;
    uint16_t FP8_BF16_MAX_EXP = 0;
    if constexpr (IsSame<U, fp8_e4m3fn_t>::value) {
        FP8_BF16_MAX_EXP = FP8_E4M3_MAX_EXP;
    } else if constexpr (IsSame<U, fp8_e5m2_t>::value) {
        FP8_BF16_MAX_EXP = FP8_E5M2_MAX_EXP;
    }
    if constexpr (isTail) {
        outDataLenAlign = 1;
    }
    __VEC_SCOPE__
    {
        if constexpr (IsSame<T, float>::value) {
            return;
        }
        Reg::RegTensor<T> x;
        Reg::RegTensor<bfloat16_t> xBF16;
        Reg::RegTensor<uint16_t> exp;
        Reg::RegTensor<uint16_t> expMax;
        Reg::RegTensor<uint16_t> maxEle;
        Reg::RegTensor<uint16_t> fp8MaxExp;
        Reg::RegTensor<uint16_t> fp8Nan;
        Reg::RegTensor<uint16_t> shareExp;
        Reg::RegTensor<uint16_t> mxScaleInt;

        Reg::RegTensor<uint16_t> reversedShareExp;
        Reg::RegTensor<uint16_t> specialExp;
        Reg::RegTensor<uint16_t> bias;
        Reg::RegTensor<uint16_t> zero;
        Reg::RegTensor<uint16_t> nan;
        Reg::RegTensor<uint16_t> y0;
        Reg::RegTensor<uint16_t> y1;
        Reg::RegTensor<uint16_t> invalidMaskU16;
        Reg::RegTensor<uint16_t> xSelect;

        Reg::RegTensor<uint8_t> mxScale;
        Reg::RegTensor<uint8_t> outZero;
        Reg::RegTensor<uint8_t> outOne;
        Reg::RegTensor<bfloat16_t> valueRegTensor;
        Reg::RegTensor<float> reversedShareExpRegTensorFP32Zero;
        Reg::RegTensor<float> reversedShareExpRegTensorFP32One;
        Reg::RegTensor<float> yZero;
        Reg::RegTensor<float> yOne;
        Reg::RegTensor<U> yZeroFP8;
        Reg::RegTensor<U> yOneFP8;

        Reg::UnalignReg u0;
        Reg::UnalignReg u1;
        Reg::MaskReg p0;
        Reg::MaskReg p1;
        Reg::MaskReg infMask;
        Reg::MaskReg zeroMask;
        Reg::MaskReg invalidDataMask;
        Reg::MaskReg specialDataMask;

        Reg::MaskReg maskAll = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        static constexpr Reg::CastTrait castTraitZero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitOne = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTrait32to8 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                              Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

        Reg::Duplicate(maxEle, BF16_MAX_EXP);
        Reg::Duplicate(fp8MaxExp, FP8_BF16_MAX_EXP);
        Reg::Duplicate(fp8Nan, FP8_DEFAULT_MAX_EXP);
        Reg::Duplicate(bias, BF16_EXP_BIAS);
        Reg::Duplicate(zero, 0);
        Reg::Duplicate(nan, BF16_NAN_CUSTOM);
        Reg::Duplicate(specialExp, BF16_SPECIAL_EXP_THRESHOLD);
        Reg::Duplicate(invalidMaskU16, FP16_INVALID);

        uint32_t pnum = vfNum16;
        p0 = Reg::UpdateMask<uint16_t>(pnum);
        for (uint16_t i = 0; i < regLoop; i++) {
            this->template LoadData<RoundMode::CAST_TRUNC, roundMode>(xAddr, i * vfNum16, x, p0);
            if constexpr (IsSame<T, half>::value) {
                Reg::And(xSelect, (Reg::RegTensor<uint16_t>&)x, invalidMaskU16, p0);
                Reg::Compare<uint16_t, CMPMODE::NE>(invalidDataMask, xSelect, invalidMaskU16, p0);
                Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(xBF16, x, p0);
                Reg::And(expMax, (Reg::RegTensor<uint16_t>&)xBF16, maxEle, p0);
                Reg::Select<uint16_t>(expMax, expMax, maxEle, invalidDataMask);
            } else if constexpr (IsSame<T, bfloat16_t>::value) {
                Reg::And(expMax, (Reg::RegTensor<uint16_t>&)x, maxEle, p0);
            }
            for (uint16_t j = 1; j < static_cast<uint16_t>(blockCount); j++) {
                this->template LoadData<RoundMode::CAST_TRUNC, roundMode>(xAddr, j * dataLen + i * vfNum16, x, p0);
                if constexpr (IsSame<T, half>::value) {
                    Reg::And(xSelect, (Reg::RegTensor<uint16_t>&)x, invalidMaskU16, p0);
                    Reg::Compare<uint16_t, CMPMODE::NE>(invalidDataMask, xSelect, invalidMaskU16, p0);
                    Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(xBF16, x, p0);
                    Reg::And(exp, (Reg::RegTensor<uint16_t>&)xBF16, maxEle, p0);
                    Reg::Select<uint16_t>(exp, exp, maxEle, invalidDataMask);
                } else if constexpr (IsSame<T, bfloat16_t>::value) {
                    Reg::And(exp, (Reg::RegTensor<uint16_t>&)x, maxEle, p0);
                }
                Reg::Max(expMax, expMax, exp, p0);
            }
            Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expMax, maxEle, p0);
            Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMax, fp8MaxExp, p0);
            Reg::Select<uint16_t>(expMax, fp8MaxExp, expMax, invalidDataMask);
            Reg::Sub(expMax, expMax, fp8MaxExp, p0);
            Reg::ShiftRights(mxScaleInt, expMax, BF16_SHR_NUM, p0);
            Reg::Select<uint16_t>(mxScaleInt, mxScaleInt, fp8Nan, infMask);
            Reg::Pack(mxScale, mxScaleInt);
            Reg::DataCopyUnAlign(mxScaleAddr, mxScale, u1, vfNum16);
            Reg::DataCopyUnAlignPost(mxScaleAddr, u1, 0);
            // 求1/scale
            Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMax, zero, p0);
            Reg::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, expMax, bias, p0);
            Reg::Sub(reversedShareExp, bias, expMax, p0);
            Reg::Select<uint16_t>(reversedShareExp, reversedShareExp, nan, infMask);
            Reg::Select<uint16_t>(reversedShareExp, reversedShareExp, zero, zeroMask);
            Reg::Select<uint16_t>(reversedShareExp, specialExp, reversedShareExp, specialDataMask);

            // 求data value
            for (uint16_t j = 0; j < static_cast<uint16_t>(blockCount); j++) {
                this->template LoadData<toBf16RoundMode, roundMode, true>(xAddr, j * dataLen + i * vfNum16, x, p0);
                if constexpr (IsSame<T, half>::value) {
                    Reg::Cast<float, T, castTraitZero>(yZero, x, p0);
                    Reg::Cast<float, T, castTraitOne>(yOne, x, p0);
                    Reg::Cast<float, bfloat16_t, castTraitZero>(reversedShareExpRegTensorFP32Zero,
                                                                (Reg::RegTensor<bfloat16_t>&)reversedShareExp, p0);
                    Reg::Cast<float, bfloat16_t, castTraitOne>(reversedShareExpRegTensorFP32One,
                                                               (Reg::RegTensor<bfloat16_t>&)reversedShareExp, p0);
                    Reg::Mul(yZero, yZero, reversedShareExpRegTensorFP32Zero, maskAll);
                    Reg::Mul(yOne, yOne, reversedShareExpRegTensorFP32One, maskAll);
                } else if constexpr (IsSame<T, bfloat16_t>::value) {
                    Reg::Mul(valueRegTensor, x, (Reg::RegTensor<bfloat16_t>&)reversedShareExp, p0);
                    Reg::Cast<float, bfloat16_t, castTraitZero>(yZero, valueRegTensor, maskAll);
                    Reg::Cast<float, bfloat16_t, castTraitOne>(yOne, valueRegTensor, maskAll);
                }
                Reg::Interleave(yZero, yOne, yZero, yOne);
                Reg::Cast<U, float, castTrait32to8>(yZeroFP8, yZero, maskAll);
                Reg::Cast<U, float, castTrait32to8>(yOneFP8, yOne, maskAll);
                Reg::Pack(y0, (Reg::RegTensor<uint32_t>&)yZeroFP8);
                Reg::Pack(outZero, y0);
                Reg::Pack(y1, (Reg::RegTensor<uint32_t>&)yOneFP8);
                Reg::Pack(outOne, y1);
                auto addr0 = yAddr + (j * outDataLenAlign + i * vfNum16);
                Reg::DataCopyUnAlign(addr0, outZero, u1, loopNum0);
                Reg::DataCopyUnAlignPost(addr0, u1, 0);
                auto addr1 = yAddr + (j * outDataLenAlign + i * vfNum16) + loopNum0;
                Reg::DataCopyUnAlign(addr1, outOne, u1, loopNum1);
                Reg::DataCopyUnAlignPost(addr1, u1, 0);
            }
        }

        uint32_t tailPnum = tailVfLen;
        p1 = Reg::UpdateMask<T>(tailPnum);
        if (tailVfLen != 0) {
            this->template LoadData<RoundMode::CAST_TRUNC, roundMode>(xAddr, regLoop * vfNum16, x, p1);
            if constexpr (IsSame<T, half>::value) {
                Reg::And(xSelect, (Reg::RegTensor<uint16_t>&)x, invalidMaskU16, p1);
                Reg::Compare<uint16_t, CMPMODE::NE>(invalidDataMask, xSelect, invalidMaskU16, p1);
                Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(xBF16, x, p1);
                Reg::And(expMax, (Reg::RegTensor<uint16_t>&)xBF16, maxEle, p1);
                Reg::Select<uint16_t>(expMax, expMax, maxEle, invalidDataMask);
            } else if constexpr (IsSame<T, bfloat16_t>::value) {
                Reg::And(expMax, (Reg::RegTensor<uint16_t>&)x, maxEle, p1);
            }
            for (uint16_t j = 1; j < static_cast<uint16_t>(blockCount); j++) {
                this->template LoadData<RoundMode::CAST_TRUNC, roundMode>(xAddr, regLoop * vfNum16 + j * dataLen, x,
                                                                          p1);
                if constexpr (IsSame<T, half>::value) {
                    Reg::And(xSelect, (Reg::RegTensor<uint16_t>&)x, invalidMaskU16, p1);
                    Reg::Compare<uint16_t, CMPMODE::NE>(invalidDataMask, xSelect, invalidMaskU16, p1);
                    Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(xBF16, x, p1);
                    Reg::And(exp, (Reg::RegTensor<uint16_t>&)xBF16, maxEle, p1);
                    Reg::Select<uint16_t>(exp, exp, maxEle, invalidDataMask);
                } else if constexpr (IsSame<T, bfloat16_t>::value) {
                    Reg::And(exp, (Reg::RegTensor<uint16_t>&)x, maxEle, p1);
                }
                Reg::Max(expMax, expMax, exp, p1);
            }
            Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expMax, maxEle, p1);
            Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMax, fp8MaxExp, p1);
            Reg::Select<uint16_t>(expMax, fp8MaxExp, expMax, invalidDataMask);
            Reg::Sub(expMax, expMax, fp8MaxExp, p1);
            Reg::ShiftRights(mxScaleInt, expMax, BF16_SHR_NUM, p1);
            Reg::Select<uint16_t>(mxScaleInt, mxScaleInt, fp8Nan, infMask);
            Reg::Pack(mxScale, mxScaleInt);
            Reg::DataCopyUnAlign(mxScaleAddr, mxScale, u1, tailVfLen);
            Reg::DataCopyUnAlignPost(mxScaleAddr, u1, 0);
            // 求1/scale
            Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMax, zero, p1);
            Reg::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, expMax, bias, p1);
            Reg::Sub(reversedShareExp, bias, expMax, p1);
            Reg::Select<uint16_t>(reversedShareExp, specialExp, reversedShareExp, specialDataMask);
            Reg::Select<uint16_t>(reversedShareExp, reversedShareExp, nan, infMask);
            Reg::Select<uint16_t>(reversedShareExp, reversedShareExp, zero, zeroMask);
            if constexpr (isTail) {
                Reg::Duplicate(reversedShareExp, reversedShareExp, p1);
            }
            // 求data value
            for (uint16_t j = 0; j < static_cast<uint16_t>(blockCount); j++) {
                this->template LoadData<toBf16RoundMode, roundMode, true>(xAddr, regLoop * vfNum16 + j * dataLen, x,
                                                                          p1);
                if constexpr (IsSame<T, half>::value) {
                    Reg::Cast<float, T, castTraitZero>(yZero, x, p1);
                    Reg::Cast<float, T, castTraitOne>(yOne, x, p1);
                    Reg::Cast<float, bfloat16_t, castTraitZero>(reversedShareExpRegTensorFP32Zero,
                                                                (Reg::RegTensor<bfloat16_t>&)reversedShareExp, p1);
                    Reg::Cast<float, bfloat16_t, castTraitOne>(reversedShareExpRegTensorFP32One,
                                                               (Reg::RegTensor<bfloat16_t>&)reversedShareExp, p1);
                    Reg::Mul(yZero, yZero, reversedShareExpRegTensorFP32Zero, maskAll);
                    Reg::Mul(yOne, yOne, reversedShareExpRegTensorFP32One, maskAll);
                } else if constexpr (IsSame<T, bfloat16_t>::value) {
                    Reg::Mul(valueRegTensor, x, (Reg::RegTensor<bfloat16_t>&)reversedShareExp, p1);
                    Reg::Cast<float, bfloat16_t, castTraitZero>(yZero, valueRegTensor, maskAll);
                    Reg::Cast<float, bfloat16_t, castTraitOne>(yOne, valueRegTensor, maskAll);
                }
                Reg::Interleave(yZero, yOne, yZero, yOne);
                Reg::Cast<U, float, castTrait32to8>(yZeroFP8, yZero, maskAll);
                Reg::Cast<U, float, castTrait32to8>(yOneFP8, yOne, maskAll);
                Reg::Pack(y0, (Reg::RegTensor<uint32_t>&)yZeroFP8);
                Reg::Pack(outZero, y0);
                Reg::Pack(y1, (Reg::RegTensor<uint32_t>&)yOneFP8);
                Reg::Pack(outOne, y1);
                auto addr0 = yAddr + (regLoop * vfNum16 + j * outDataLenAlign);
                Reg::DataCopyUnAlign(addr0, outZero, u1, tailLoopNum0);
                Reg::DataCopyUnAlignPost(addr0, u1, 0);
                auto addr1 = yAddr + (regLoop * vfNum16 + j * outDataLenAlign) + tailLoopNum0;
                Reg::DataCopyUnAlign(addr1, outOne, u1, tailLoopNum1);
                Reg::DataCopyUnAlignPost(addr1, u1, 0);
            }
        }
    }
}

template <typename T, typename U, const bool isTail>
template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicMxQuantNotTailAxisFP8<T, U, isTail>::ComputecuBLAS(int64_t dataLen, int64_t blockCount,
                                                                                 __ubuf__ T* xAddr,
                                                                                 __ubuf__ uint8_t* mxScaleAddr,
                                                                                 __ubuf__ uint8_t* yAddr)
{
    constexpr uint32_t vfNum16 = Ops::Base::GetVRegSize() / sizeof(T); // 寄存器单次能处理的长度
    constexpr uint32_t vfNum32 = Ops::Base::GetVRegSize() / sizeof(float); // cast到FP32后单个寄存器中的元素个数
    uint16_t regLoop = static_cast<uint16_t>(dataLen) / static_cast<uint16_t>(vfNum32);
    uint16_t tailVfLen = static_cast<uint16_t>(dataLen) % static_cast<uint16_t>(vfNum32);
    uint32_t singleLoopNum = dataLen <= vfNum32 ? vfNum16 : vfNum32;
    uint32_t singleTailLoopNum = tailVfLen;
    int64_t outDataLenAlign = this->ubDim_ == DIM2 ?
                                  (dataLen + FP8_OUT_ELE_PER_BLK - 1) / FP8_OUT_ELE_PER_BLK * FP8_OUT_ELE_PER_BLK :
                                  dataLen;
    float inv_dtype_max = 0;
    if constexpr (IsSame<U, fp8_e4m3fn_t>::value) {
        inv_dtype_max = FP8_E4M3_INV_MAX; // 1.0f / 448.0f
    } else if constexpr (IsSame<U, fp8_e5m2_t>::value) {
        inv_dtype_max = FP8_E5M2_INV_MAX; // 1.0f / 57344.0f
    }
    if constexpr (isTail) {
        outDataLenAlign = 1;
    }
    __VEC_SCOPE__
    {
        if constexpr (IsSame<T, float>::value) {
            return;
        }
        Reg::RegTensor<T> x;
        Reg::RegTensor<uint16_t> absMaskBF16;
        Reg::RegTensor<uint16_t> fullZero;
        Reg::RegTensor<uint32_t> zeroRegTensor32;
        Reg::RegTensor<uint32_t> manMaskFP32;
        Reg::RegTensor<uint32_t> fp8Nan;
        Reg::RegTensor<uint32_t> bias;
        Reg::RegTensor<uint32_t> nan;

        Reg::RegTensor<uint16_t> expScaleMxYZeroRegTensor;
        Reg::RegTensor<uint16_t> expMaxreShareExpFP16YOneRegTensor;
        Reg::RegTensor<uint32_t> expMaxAndAddOneFP32RegTensor;
        Reg::RegTensor<uint32_t> expAndreShareExpFP32RegTensor;
        Reg::RegTensor<uint32_t> manAndmxScaleFP32RegTensor;

        Reg::RegTensor<float> yZero;
        Reg::RegTensor<float> yOne;
        Reg::RegTensor<float> reversedShareExpRegTensorFP32Zero;
        Reg::RegTensor<float> reversedShareExpRegTensorFP32One;
        Reg::RegTensor<bfloat16_t> valueRegTensor;
        Reg::RegTensor<U> yZeroFP8;
        Reg::RegTensor<uint8_t> outZeromxScaleFp8;

        Reg::UnalignReg u1;
        Reg::MaskReg infMask;
        Reg::MaskReg zeroMask;
        Reg::MaskReg p0;
        Reg::MaskReg p1;
        Reg::MaskReg p2;
        Reg::MaskReg p3;
        Reg::MaskReg preMaskScale = Reg::CreateMask<uint32_t>();
        Reg::MaskReg maskAll = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();

        static constexpr Reg::CastTrait castTraitZero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitOne = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTrait32to8 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTraitHalf2Float = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                               Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        Reg::Duplicate(absMaskBF16, BF16_ABS_MASK);
        Reg::Duplicate(zeroRegTensor32, 0);
        Reg::Duplicate(manMaskFP32, FP32_MX_MAN_MASK);
        Reg::Duplicate(fp8Nan, FP8_MAX_EXP_IN_FP32);
        Reg::Duplicate(bias, FP32_MX_EXP_BIAS_CUBLAS);
        Reg::Duplicate(nan, BF16_NAN_CUSTOM);
        Reg::Duplicate(fullZero, 0);

        uint32_t pnum = vfNum32;
        p0 = Reg::UpdateMask<T>(pnum); // 前vfNum位有效加载
        for (uint16_t i = 0; i < regLoop; i++) {
            this->template LoadData<RoundMode::CAST_TRUNC, roundMode>(xAddr, i * vfNum32, x, p0);
            Reg::And((Reg::RegTensor<uint16_t>&)expMaxreShareExpFP16YOneRegTensor, (Reg::RegTensor<uint16_t>&)x,
                     absMaskBF16, p0);
            for (uint16_t j = 1; j < static_cast<uint16_t>(blockCount); j++) {
                this->template LoadData<RoundMode::CAST_TRUNC, roundMode>(xAddr, j * dataLen + i * vfNum32, x, p0);
                Reg::And((Reg::RegTensor<uint16_t>&)expScaleMxYZeroRegTensor, (Reg::RegTensor<uint16_t>&)x, absMaskBF16,
                         p0);
                Reg::Max(expMaxreShareExpFP16YOneRegTensor,
                         (Reg::RegTensor<uint16_t>&)expMaxreShareExpFP16YOneRegTensor,
                         (Reg::RegTensor<uint16_t>&)expScaleMxYZeroRegTensor, p0);
            }
            Reg::Interleave(expMaxreShareExpFP16YOneRegTensor, fullZero, expMaxreShareExpFP16YOneRegTensor, fullZero);
            Reg::Cast<float, T, castTraitHalf2Float>((Reg::RegTensor<float>&)expMaxAndAddOneFP32RegTensor,
                                                     (Reg::RegTensor<T>&)expMaxreShareExpFP16YOneRegTensor,
                                                     preMaskScale);
            // 校验
            Reg::CompareScalar<uint32_t, CMPMODE::LT>(infMask, expMaxAndAddOneFP32RegTensor, FP32_MX_MAX_EXP,
                                                      preMaskScale);
            Reg::Compare<uint32_t, CMPMODE::NE>(zeroMask, expMaxAndAddOneFP32RegTensor, zeroRegTensor32, preMaskScale);
            Reg::Maxs((Reg::RegTensor<float>&)expMaxAndAddOneFP32RegTensor,
                      (Reg::RegTensor<float>&)expMaxAndAddOneFP32RegTensor, this->maxLowBound_, preMaskScale);
            Reg::Muls((Reg::RegTensor<float>&)expMaxAndAddOneFP32RegTensor,
                      (Reg::RegTensor<float>&)expMaxAndAddOneFP32RegTensor, inv_dtype_max, preMaskScale);
            Reg::ShiftRights(expAndreShareExpFP32RegTensor, expMaxAndAddOneFP32RegTensor, FP32_SHR_NUM,
                             preMaskScale);                                                                // Exp
            Reg::And(manAndmxScaleFP32RegTensor, expMaxAndAddOneFP32RegTensor, manMaskFP32, preMaskScale); // Man
            // 条件
            Reg::CompareScalar<uint32_t, CMPMODE::GT>(p1, expAndreShareExpFP32RegTensor, FP32_NUMBER_ZERO,
                                                      preMaskScale);
            Reg::CompareScalar<uint32_t, CMPMODE::LT>(p2, expAndreShareExpFP32RegTensor, FP32_NUMBER_254, preMaskScale);
            Reg::CompareScalar<uint32_t, CMPMODE::GT>(p3, manAndmxScaleFP32RegTensor, FP32_NUMBER_ZERO, preMaskScale);
            Reg::MaskAnd(p1, p1, p2, preMaskScale);
            Reg::MaskAnd(p1, p1, p3, preMaskScale);
            Reg::CompareScalar<uint32_t, CMPMODE::EQ>(p2, expAndreShareExpFP32RegTensor, FP32_NUMBER_ZERO,
                                                      preMaskScale);
            Reg::CompareScalar<uint32_t, CMPMODE::GT>(p3, manAndmxScaleFP32RegTensor, FP32_NUMBER_HALF, preMaskScale);
            Reg::MaskAnd(p2, p2, p3, preMaskScale);
            Reg::MaskXor(p1, p1, p2, preMaskScale);
            // 向上取整
            Reg::Adds(expMaxAndAddOneFP32RegTensor, expAndreShareExpFP32RegTensor, 1, preMaskScale);
            Reg::Select(manAndmxScaleFP32RegTensor, expMaxAndAddOneFP32RegTensor, expAndreShareExpFP32RegTensor, p1);
            // 校验
            Reg::Select<uint32_t>(manAndmxScaleFP32RegTensor, manAndmxScaleFP32RegTensor, fp8Nan, infMask);
            Reg::Select<uint32_t>(manAndmxScaleFP32RegTensor, manAndmxScaleFP32RegTensor, zeroRegTensor32, zeroMask);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(expScaleMxYZeroRegTensor,
                                                                    manAndmxScaleFP32RegTensor);
            Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(outZeromxScaleFp8, expScaleMxYZeroRegTensor);
            Reg::DataCopyUnAlign(mxScaleAddr, outZeromxScaleFp8, u1, vfNum32); // 从寄存器搬到UB
            Reg::DataCopyUnAlignPost(mxScaleAddr, u1, 0);
            // 求1/scale
            Reg::ShiftLefts(manAndmxScaleFP32RegTensor, manAndmxScaleFP32RegTensor, BF16_SHR_NUM, preMaskScale);
            Reg::Sub(expAndreShareExpFP32RegTensor, bias, manAndmxScaleFP32RegTensor, preMaskScale);
            // 校验
            Reg::Select<uint32_t>(expAndreShareExpFP32RegTensor, expAndreShareExpFP32RegTensor, nan, infMask);
            Reg::Select<uint32_t>(expAndreShareExpFP32RegTensor, expAndreShareExpFP32RegTensor, zeroRegTensor32,
                                  zeroMask);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(expMaxreShareExpFP16YOneRegTensor,
                                                                    expAndreShareExpFP32RegTensor);
            // 求data value
            for (uint16_t j = 0; j < static_cast<uint16_t>(blockCount); j++) {
                this->template LoadData<toBf16RoundMode, roundMode, true>(xAddr, j * dataLen + i * vfNum32, x, p0);
                if constexpr (IsSame<T, half>::value) {
                    Reg::Cast<float, T, castTraitZero>(yZero, x, p0);
                    Reg::Cast<float, T, castTraitOne>(yOne, x, p0);
                    Reg::Cast<float, bfloat16_t, castTraitZero>(
                        reversedShareExpRegTensorFP32Zero,
                        (Reg::RegTensor<bfloat16_t>&)expMaxreShareExpFP16YOneRegTensor, p0);
                    Reg::Cast<float, bfloat16_t, castTraitOne>(
                        reversedShareExpRegTensorFP32One,
                        (Reg::RegTensor<bfloat16_t>&)expMaxreShareExpFP16YOneRegTensor, p0);
                    Reg::Mul(yZero, yZero, reversedShareExpRegTensorFP32Zero, maskAll);
                    Reg::Mul(yOne, yOne, reversedShareExpRegTensorFP32One, maskAll);
                } else if constexpr (IsSame<T, bfloat16_t>::value) {
                    Reg::Mul(valueRegTensor, x, (Reg::RegTensor<bfloat16_t>&)expMaxreShareExpFP16YOneRegTensor, p0);
                    Reg::Cast<float, bfloat16_t, castTraitZero>(yZero, valueRegTensor, maskAll);
                    Reg::Cast<float, bfloat16_t, castTraitOne>(yOne, valueRegTensor, maskAll);
                }
                Reg::Interleave(yZero, yOne, yZero, yOne);
                Reg::Cast<U, float, castTrait32to8>(yZeroFP8, yZero, maskAll);
                Reg::Pack(expScaleMxYZeroRegTensor, (Reg::RegTensor<uint32_t>&)yZeroFP8);
                Reg::Pack(outZeromxScaleFp8, expScaleMxYZeroRegTensor);
                auto addr = yAddr + (j * outDataLenAlign + i * vfNum32);
                Reg::DataCopyUnAlign(addr, outZeromxScaleFp8, u1, singleLoopNum);
                Reg::DataCopyUnAlignPost(addr, u1, 0);
            }
        }

        uint32_t tailPnum = tailVfLen;
        p0 = Reg::UpdateMask<T>(tailPnum); // 前tailvfLen有效加载
        if (tailVfLen != 0) {
            this->template LoadData<RoundMode::CAST_TRUNC, roundMode>(xAddr, regLoop * vfNum32, x, p0);
            Reg::And((Reg::RegTensor<uint16_t>&)expMaxreShareExpFP16YOneRegTensor, (Reg::RegTensor<uint16_t>&)x,
                     absMaskBF16, p0);
            for (uint16_t j = 1; j < static_cast<uint16_t>(blockCount); j++) {
                this->template LoadData<RoundMode::CAST_TRUNC, roundMode>(xAddr, regLoop * vfNum32 + j * dataLen, x,
                                                                          p0);
                Reg::And((Reg::RegTensor<uint16_t>&)expScaleMxYZeroRegTensor, (Reg::RegTensor<uint16_t>&)x, absMaskBF16,
                         p0);
                Reg::Max(expMaxreShareExpFP16YOneRegTensor,
                         (Reg::RegTensor<uint16_t>&)expMaxreShareExpFP16YOneRegTensor,
                         (Reg::RegTensor<uint16_t>&)expScaleMxYZeroRegTensor, p0);
            }
            Reg::Interleave(expMaxreShareExpFP16YOneRegTensor, fullZero, expMaxreShareExpFP16YOneRegTensor, fullZero);
            Reg::Cast<float, T, castTraitHalf2Float>((Reg::RegTensor<float>&)expMaxAndAddOneFP32RegTensor,
                                                     (Reg::RegTensor<T>&)expMaxreShareExpFP16YOneRegTensor,
                                                     preMaskScale);
            // 校验
            Reg::CompareScalar<uint32_t, CMPMODE::LT>(infMask, expMaxAndAddOneFP32RegTensor, FP32_MX_MAX_EXP,
                                                      preMaskScale);
            Reg::Compare<uint32_t, CMPMODE::NE>(zeroMask, expMaxAndAddOneFP32RegTensor, zeroRegTensor32, preMaskScale);
            Reg::Maxs((Reg::RegTensor<float>&)expMaxAndAddOneFP32RegTensor,
                      (Reg::RegTensor<float>&)expMaxAndAddOneFP32RegTensor, this->maxLowBound_, preMaskScale);
            Reg::Muls((Reg::RegTensor<float>&)expMaxAndAddOneFP32RegTensor,
                      (Reg::RegTensor<float>&)expMaxAndAddOneFP32RegTensor, inv_dtype_max, preMaskScale);
            Reg::ShiftRights(expAndreShareExpFP32RegTensor, expMaxAndAddOneFP32RegTensor, FP32_SHR_NUM,
                             preMaskScale);                                                                // Exp
            Reg::And(manAndmxScaleFP32RegTensor, expMaxAndAddOneFP32RegTensor, manMaskFP32, preMaskScale); // Man
            // 条件
            Reg::CompareScalar<uint32_t, CMPMODE::GT>(p1, expAndreShareExpFP32RegTensor, FP32_NUMBER_ZERO,
                                                      preMaskScale);
            Reg::CompareScalar<uint32_t, CMPMODE::LT>(p2, expAndreShareExpFP32RegTensor, FP32_NUMBER_254, preMaskScale);
            Reg::CompareScalar<uint32_t, CMPMODE::GT>(p3, manAndmxScaleFP32RegTensor, FP32_NUMBER_ZERO, preMaskScale);
            Reg::MaskAnd(p1, p1, p2, preMaskScale);
            Reg::MaskAnd(p1, p1, p3, preMaskScale);
            Reg::CompareScalar<uint32_t, CMPMODE::EQ>(p2, expAndreShareExpFP32RegTensor, FP32_NUMBER_ZERO,
                                                      preMaskScale);
            Reg::CompareScalar<uint32_t, CMPMODE::GT>(p3, manAndmxScaleFP32RegTensor, FP32_NUMBER_HALF, preMaskScale);
            Reg::MaskAnd(p2, p2, p3, preMaskScale);
            Reg::MaskXor(p1, p1, p2, preMaskScale);
            // 向上取整
            Reg::Adds(expMaxAndAddOneFP32RegTensor, expAndreShareExpFP32RegTensor, 1, preMaskScale);
            Reg::Select(manAndmxScaleFP32RegTensor, expMaxAndAddOneFP32RegTensor, expAndreShareExpFP32RegTensor, p1);
            // 校验
            Reg::Select<uint32_t>(manAndmxScaleFP32RegTensor, manAndmxScaleFP32RegTensor, fp8Nan, infMask);
            Reg::Select<uint32_t>(manAndmxScaleFP32RegTensor, manAndmxScaleFP32RegTensor, zeroRegTensor32, zeroMask);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(expScaleMxYZeroRegTensor,
                                                                    manAndmxScaleFP32RegTensor);
            Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(outZeromxScaleFp8, expScaleMxYZeroRegTensor);
            Reg::DataCopyUnAlign(mxScaleAddr, outZeromxScaleFp8, u1, tailVfLen); // 从寄存器搬到UB
            Reg::DataCopyUnAlignPost(mxScaleAddr, u1, 0);
            // 求1/scale
            Reg::ShiftLefts(manAndmxScaleFP32RegTensor, manAndmxScaleFP32RegTensor, BF16_SHR_NUM, preMaskScale);
            Reg::Sub(expAndreShareExpFP32RegTensor, bias, manAndmxScaleFP32RegTensor, preMaskScale);
            // 校验
            Reg::Select<uint32_t>(expAndreShareExpFP32RegTensor, expAndreShareExpFP32RegTensor, nan, infMask);
            Reg::Select<uint32_t>(expAndreShareExpFP32RegTensor, expAndreShareExpFP32RegTensor, zeroRegTensor32,
                                  zeroMask);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(expMaxreShareExpFP16YOneRegTensor,
                                                                    expAndreShareExpFP32RegTensor);
            if constexpr (isTail) {
                Reg::Duplicate(expMaxreShareExpFP16YOneRegTensor, expMaxreShareExpFP16YOneRegTensor, p0);
            }
            // 求data value
            for (uint16_t j = 0; j < static_cast<uint16_t>(blockCount); j++) {
                this->template LoadData<toBf16RoundMode, roundMode, true>(xAddr, regLoop * vfNum32 + j * dataLen, x,
                                                                          p0);
                if constexpr (IsSame<T, half>::value) {
                    Reg::Cast<float, T, castTraitZero>(yZero, x, p0);
                    Reg::Cast<float, T, castTraitOne>(yOne, x, p0);
                    Reg::Cast<float, bfloat16_t, castTraitZero>(
                        reversedShareExpRegTensorFP32Zero,
                        (Reg::RegTensor<bfloat16_t>&)expMaxreShareExpFP16YOneRegTensor, p0);
                    Reg::Cast<float, bfloat16_t, castTraitOne>(
                        reversedShareExpRegTensorFP32One,
                        (Reg::RegTensor<bfloat16_t>&)expMaxreShareExpFP16YOneRegTensor, p0);
                    Reg::Mul(yZero, yZero, reversedShareExpRegTensorFP32Zero, maskAll);
                    Reg::Mul(yOne, yOne, reversedShareExpRegTensorFP32One, maskAll);
                } else if constexpr (IsSame<T, bfloat16_t>::value) {
                    Reg::Mul(valueRegTensor, x, (Reg::RegTensor<bfloat16_t>&)expMaxreShareExpFP16YOneRegTensor, p0);
                    Reg::Cast<float, bfloat16_t, castTraitZero>(yZero, valueRegTensor, maskAll);
                    Reg::Cast<float, bfloat16_t, castTraitOne>(yOne, valueRegTensor, maskAll);
                }
                Reg::Interleave(yZero, yOne, yZero, yOne);
                Reg::Cast<U, float, castTrait32to8>(yZeroFP8, yZero, maskAll);
                Reg::Pack(expScaleMxYZeroRegTensor, (Reg::RegTensor<uint32_t>&)yZeroFP8);
                Reg::Pack(outZeromxScaleFp8, expScaleMxYZeroRegTensor);
                auto addr = yAddr + (regLoop * vfNum32 + j * outDataLenAlign);
                Reg::DataCopyUnAlign(addr, outZeromxScaleFp8, u1, singleTailLoopNum);
                Reg::DataCopyUnAlignPost(addr, u1, 0);
            }
        }
    }
}
} // namespace DynamicMxQuant
#endif // DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_FP8_H
