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
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR workspace, const DynamicMxQuantTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SplitPostAxisCompute(int64_t dataLen, int64_t blockCount);
    __aicore__ inline void SplitPreAxisCompute(int64_t ubFactor, int64_t blockSizeIdx);
    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void ComputeOCP(
        int64_t dataLen, int64_t blockCount, __ubuf__ T* xAddr, __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint8_t* yAddr);
    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void ComputecuBLAS(
        int64_t dataLen, int64_t blockCount, __ubuf__ T* xAddr, __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint8_t* yAddr);
    __aicore__ inline bool IsTailLoopInUbDim(int64_t loopIdx);
    __aicore__ inline bool IsNeedPadAndTailInAxis(int64_t curLoopIdxInAllCore);

private:
    int64_t blockLoopOffset_ = 0;
    int64_t blockOffset_ = 0;
    int64_t scaleBlockOffset_ = 0;
    int64_t bufferSize_ = 0;
};

template <typename T, typename U, const bool isTail>
__aicore__ inline void DynamicMxQuantNotTailAxisFP8<T, U, isTail>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR workspace, const DynamicMxQuantTilingData* tilingData)
{
    this->BaseInit(x, y, mxScale, workspace, tilingData);
    if (this->blockIdx_ >= this->usedCoreNum_) {
        return;
    }
    blockLoopOffset_ = this->blockIdx_ * this->blockFactor_;
    if (this->ubDim_ == DIM2) { // 非尾轴场景，切axis之后的轴合轴成的post
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
__aicore__ inline void DynamicMxQuantNotTailAxisFP8<T, U, isTail>::SplitPostAxisCompute(
    int64_t dataLen, int64_t blockCount)
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
__aicore__ inline void DynamicMxQuantNotTailAxisFP8<T, U, isTail>::SplitPreAxisCompute(
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
        auto yAddr = (__ubuf__ uint8_t*)y.GetPhyAddr() + offset;
        offset += blockCount * this->postAxisSize_;
        if (this->scaleAlg_ == 0) {
            ComputeOCP<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(
                this->postAxisSize_, blockCount, xAddr, mxScaleAddr, yAddr);
        } else {
            ComputecuBLAS<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(
                this->postAxisSize_, blockCount, xAddr, mxScaleAddr, yAddr);
        }
    }
    this->mxScaleQueue_.template EnQue(mxScale);
    this->outQueue_.template EnQue(y);
    this->inQueue_.template FreeTensor(x);
}

template <typename T, typename U, const bool isTail>
template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicMxQuantNotTailAxisFP8<T, U, isTail>::ComputeOCP(
    int64_t dataLen, int64_t blockCount, __ubuf__ T* xAddr, __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint8_t* yAddr)
{
    constexpr uint32_t vfNum16 = Ops::Base::GetVRegSize() / sizeof(T);     // 寄存器单次处理能处理的长度
    constexpr uint32_t vfNum32 = Ops::Base::GetVRegSize() / sizeof(float); // cast到FP32后单个寄存器中的元素个数
    uint16_t regLoop = static_cast<uint16_t>(dataLen) / static_cast<uint16_t>(vfNum16); // 当前loop需要处理的长度
    uint16_t tailVfLen = static_cast<uint16_t>(dataLen) % static_cast<uint16_t>(vfNum16);
    uint32_t loopNum0 = dataLen <= vfNum32 ? vfNum16 : vfNum32;
    uint32_t loopNum1 = dataLen <= vfNum32 ? 0 : (vfNum16 - vfNum32);
    uint32_t tailLoopNum0 = tailVfLen <= vfNum32 ? tailVfLen : vfNum32;
    uint32_t tailLoopNum1 = tailVfLen <= vfNum32 ? 0 : (tailVfLen - vfNum32);
    int64_t outDataLenAlign = this->ubDim_ == DIM2 ? (dataLen + OUT_ELE_NUM_ONE_BLK_FP8 - 1) / OUT_ELE_NUM_ONE_BLK_FP8 *
                                                         OUT_ELE_NUM_ONE_BLK_FP8 :
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
        AscendC::MicroAPI::RegTensor<T> xRegTensor;
        AscendC::MicroAPI::RegTensor<bfloat16_t> xBF16RegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> expRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> expMaxRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> maxEleRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> fp8MaxExpRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> fp8NanRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> shareExpRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> mxScaleRegTensor;

        AscendC::MicroAPI::RegTensor<uint16_t> reversedShareExpRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> specialExpRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> biasRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> zeroRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> nanRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> yRegTensorZero;
        AscendC::MicroAPI::RegTensor<uint16_t> yRegTensorOne;
        AscendC::MicroAPI::RegTensor<uint16_t> invalidMaskFp16;
        AscendC::MicroAPI::RegTensor<uint16_t> xSelectRegTensor;

        AscendC::MicroAPI::RegTensor<uint8_t> mxScale;
        AscendC::MicroAPI::RegTensor<uint8_t> outZero;
        AscendC::MicroAPI::RegTensor<uint8_t> outOne;
        AscendC::MicroAPI::RegTensor<bfloat16_t> valueRegTensor;
        AscendC::MicroAPI::RegTensor<float> reversedShareExpRegTensorFP32Zero;
        AscendC::MicroAPI::RegTensor<float> reversedShareExpRegTensorFP32One;
        AscendC::MicroAPI::RegTensor<float> yZero;
        AscendC::MicroAPI::RegTensor<float> yOne;
        AscendC::MicroAPI::RegTensor<U> yZeroFP8;
        AscendC::MicroAPI::RegTensor<U> yOneFP8;

        AscendC::MicroAPI::UnalignReg u0;
        AscendC::MicroAPI::UnalignReg u1;
        AscendC::MicroAPI::MaskReg p0;
        AscendC::MicroAPI::MaskReg p1;
        AscendC::MicroAPI::MaskReg infMask;
        AscendC::MicroAPI::MaskReg zeroMask;
        AscendC::MicroAPI::MaskReg invalidDataMask;
        AscendC::MicroAPI::MaskReg specialDataMask;

        AscendC::MicroAPI::MaskReg maskAll =
            AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
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

        AscendC::MicroAPI::Duplicate(maxEleRegTensor, MAX_EXP_FOR_BF16);
        AscendC::MicroAPI::Duplicate(fp8MaxExpRegTensor, FP8_BF16_MAX_EXP);
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        AscendC::MicroAPI::Duplicate(biasRegTensor, BF16_EXP_BIAS);
        AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
        AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
        AscendC::MicroAPI::Duplicate(invalidMaskFp16, INVALID_FLOAT16);

        uint32_t pnum = vfNum16;
        p0 = AscendC::MicroAPI::UpdateMask<uint16_t>(pnum);
        for (uint16_t i = 0; i < regLoop; i++) {
            this->template LoadData<RoundMode::CAST_TRUNC, roundMode>(xAddr, i * vfNum16, xRegTensor, p0);
            if constexpr (IsSame<T, half>::value) {
                AscendC::MicroAPI::And(
                    xSelectRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor, invalidMaskFp16, p0);
                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(
                    invalidDataMask, xSelectRegTensor, invalidMaskFp16, p0);
                AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(xBF16RegTensor, xRegTensor, p0);
                AscendC::MicroAPI::And(
                    expMaxRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xBF16RegTensor, maxEleRegTensor, p0);
                AscendC::MicroAPI::Select<uint16_t>(expMaxRegTensor, expMaxRegTensor, maxEleRegTensor, invalidDataMask);
            } else {
                AscendC::MicroAPI::And(
                    expMaxRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor, maxEleRegTensor, p0);
            }
            for (uint16_t j = 1; j < static_cast<uint16_t>(blockCount); j++) {
                this->template LoadData<RoundMode::CAST_TRUNC, roundMode>(
                    xAddr, j * dataLen + i * vfNum16, xRegTensor, p0);
                if constexpr (IsSame<T, half>::value) {
                    AscendC::MicroAPI::And(
                        xSelectRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor, invalidMaskFp16, p0);
                    AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(
                        invalidDataMask, xSelectRegTensor, invalidMaskFp16, p0);
                    AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(xBF16RegTensor, xRegTensor, p0);
                    AscendC::MicroAPI::And(
                        expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xBF16RegTensor, maxEleRegTensor, p0);
                    AscendC::MicroAPI::Select<uint16_t>(expRegTensor, expRegTensor, maxEleRegTensor, invalidDataMask);
                } else {
                    AscendC::MicroAPI::And(
                        expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor, maxEleRegTensor, p0);
                }
                AscendC::MicroAPI::Max(expMaxRegTensor, expMaxRegTensor, expRegTensor, p0);
            }
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxRegTensor, maxEleRegTensor, p0);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxRegTensor, zeroRegTensor, p0);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMaxRegTensor, fp8MaxExpRegTensor, p0);
            AscendC::MicroAPI::Select<uint16_t>(expMaxRegTensor, fp8MaxExpRegTensor, expMaxRegTensor, invalidDataMask);
            AscendC::MicroAPI::Sub(expMaxRegTensor, expMaxRegTensor, fp8MaxExpRegTensor, p0);
            AscendC::MicroAPI::ShiftRights(mxScaleRegTensor, expMaxRegTensor, SHR_NUM_FOR_BF16, p0);
            AscendC::MicroAPI::Select<uint16_t>(mxScaleRegTensor, mxScaleRegTensor, fp8NanRegTensor, infMask);
            AscendC::MicroAPI::Select<uint16_t>(mxScaleRegTensor, mxScaleRegTensor, zeroRegTensor, zeroMask);
            AscendC::MicroAPI::Pack(mxScale, mxScaleRegTensor);
            AscendC::MicroAPI::DataCopyUnAlign(mxScaleAddr, mxScale, u1, vfNum16);
            AscendC::MicroAPI::DataCopyUnAlignPost(mxScaleAddr, u1, 0);
            // 求1/scale
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, expMaxRegTensor, biasRegTensor, p0);
            AscendC::MicroAPI::Sub(reversedShareExpRegTensor, biasRegTensor, expMaxRegTensor, p0);
            AscendC::MicroAPI::Select<uint16_t>(
                reversedShareExpRegTensor, reversedShareExpRegTensor, nanRegTensor, infMask);
            AscendC::MicroAPI::Select<uint16_t>(
                reversedShareExpRegTensor, reversedShareExpRegTensor, zeroRegTensor, zeroMask);
            AscendC::MicroAPI::Select<uint16_t>(
                reversedShareExpRegTensor, specialExpRegTensor, reversedShareExpRegTensor, specialDataMask);

            // 求data value
            for (uint16_t j = 0; j < static_cast<uint16_t>(blockCount); j++) {
                this->template LoadData<toBf16RoundMode, roundMode, true>(
                    xAddr, j * dataLen + i * vfNum16, xRegTensor, p0);
                if constexpr (IsSame<T, half>::value) {
                    AscendC::MicroAPI::Cast<float, T, castTraitZero>(yZero, xRegTensor, p0);
                    AscendC::MicroAPI::Cast<float, T, castTraitOne>(yOne, xRegTensor, p0);
                    AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitZero>(
                        reversedShareExpRegTensorFP32Zero,
                        (AscendC::MicroAPI::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, p0);
                    AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitOne>(
                        reversedShareExpRegTensorFP32One,
                        (AscendC::MicroAPI::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, p0);
                    AscendC::MicroAPI::Mul(yZero, yZero, reversedShareExpRegTensorFP32Zero, maskAll);
                    AscendC::MicroAPI::Mul(yOne, yOne, reversedShareExpRegTensorFP32One, maskAll);
                } else {
                    AscendC::MicroAPI::Mul(
                        valueRegTensor, xRegTensor,
                        (AscendC::MicroAPI::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, p0);
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
                auto addr0 = yAddr + (j * outDataLenAlign + i * vfNum16);
                AscendC::MicroAPI::DataCopyUnAlign(addr0, outZero, u1, loopNum0);
                AscendC::MicroAPI::DataCopyUnAlignPost(addr0, u1, 0);
                auto addr1 = yAddr + (j * outDataLenAlign + i * vfNum16) + loopNum0;
                AscendC::MicroAPI::DataCopyUnAlign(addr1, outOne, u1, loopNum1);
                AscendC::MicroAPI::DataCopyUnAlignPost(addr1, u1, 0);
            }
        }

        uint32_t tailPnum = tailVfLen;
        p1 = AscendC::MicroAPI::UpdateMask<T>(tailPnum);
        if (tailVfLen != 0) {
            this->template LoadData<RoundMode::CAST_TRUNC, roundMode>(xAddr, regLoop * vfNum16, xRegTensor, p1);
            if constexpr (IsSame<T, half>::value) {
                AscendC::MicroAPI::And(
                    xSelectRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor, invalidMaskFp16, p1);
                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(
                    invalidDataMask, xSelectRegTensor, invalidMaskFp16, p1);
                AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(xBF16RegTensor, xRegTensor, p1);
                AscendC::MicroAPI::And(
                    expMaxRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xBF16RegTensor, maxEleRegTensor, p1);
                AscendC::MicroAPI::Select<uint16_t>(expMaxRegTensor, expMaxRegTensor, maxEleRegTensor, invalidDataMask);
            } else {
                AscendC::MicroAPI::And(
                    expMaxRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor, maxEleRegTensor, p1);
            }
            for (uint16_t j = 1; j < static_cast<uint16_t>(blockCount); j++) {
                this->template LoadData<RoundMode::CAST_TRUNC, roundMode>(
                    xAddr, regLoop * vfNum16 + j * dataLen, xRegTensor, p1);
                if constexpr (IsSame<T, half>::value) {
                    AscendC::MicroAPI::And(
                        xSelectRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor, invalidMaskFp16, p1);
                    AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(
                        invalidDataMask, xSelectRegTensor, invalidMaskFp16, p1);
                    AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(xBF16RegTensor, xRegTensor, p1);
                    AscendC::MicroAPI::And(
                        expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xBF16RegTensor, maxEleRegTensor, p1);
                    AscendC::MicroAPI::Select<uint16_t>(expRegTensor, expRegTensor, maxEleRegTensor, invalidDataMask);
                } else {
                    AscendC::MicroAPI::And(
                        expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor, maxEleRegTensor, p1);
                }
                AscendC::MicroAPI::Max(expMaxRegTensor, expMaxRegTensor, expRegTensor, p1);
            }
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxRegTensor, maxEleRegTensor, p1);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxRegTensor, zeroRegTensor, p1);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMaxRegTensor, fp8MaxExpRegTensor, p1);
            AscendC::MicroAPI::Select<uint16_t>(expMaxRegTensor, fp8MaxExpRegTensor, expMaxRegTensor, invalidDataMask);
            AscendC::MicroAPI::Sub(expMaxRegTensor, expMaxRegTensor, fp8MaxExpRegTensor, p1);
            AscendC::MicroAPI::ShiftRights(mxScaleRegTensor, expMaxRegTensor, SHR_NUM_FOR_BF16, p1);
            AscendC::MicroAPI::Select<uint16_t>(mxScaleRegTensor, mxScaleRegTensor, fp8NanRegTensor, infMask);
            AscendC::MicroAPI::Select<uint16_t>(mxScaleRegTensor, mxScaleRegTensor, zeroRegTensor, zeroMask);
            AscendC::MicroAPI::Pack(mxScale, mxScaleRegTensor);
            AscendC::MicroAPI::DataCopyUnAlign(mxScaleAddr, mxScale, u1, tailVfLen);
            AscendC::MicroAPI::DataCopyUnAlignPost(mxScaleAddr, u1, 0);
            // 求1/scale
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, expMaxRegTensor, biasRegTensor, p1);
            AscendC::MicroAPI::Sub(reversedShareExpRegTensor, biasRegTensor, expMaxRegTensor, p1);
            AscendC::MicroAPI::Select<uint16_t>(
                reversedShareExpRegTensor, specialExpRegTensor, reversedShareExpRegTensor, specialDataMask);
            AscendC::MicroAPI::Select<uint16_t>(
                reversedShareExpRegTensor, reversedShareExpRegTensor, nanRegTensor, infMask);
            AscendC::MicroAPI::Select<uint16_t>(
                reversedShareExpRegTensor, reversedShareExpRegTensor, zeroRegTensor, zeroMask);
            if constexpr (isTail) {
                AscendC::MicroAPI::Duplicate(reversedShareExpRegTensor, reversedShareExpRegTensor, p1);
            }
            // 求data value
            for (uint16_t j = 0; j < static_cast<uint16_t>(blockCount); j++) {
                this->template LoadData<toBf16RoundMode, roundMode, true>(
                    xAddr, regLoop * vfNum16 + j * dataLen, xRegTensor, p1);
                if constexpr (IsSame<T, half>::value) {
                    AscendC::MicroAPI::Cast<float, T, castTraitZero>(yZero, xRegTensor, p1);
                    AscendC::MicroAPI::Cast<float, T, castTraitOne>(yOne, xRegTensor, p1);
                    AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitZero>(
                        reversedShareExpRegTensorFP32Zero,
                        (AscendC::MicroAPI::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, p1);
                    AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitOne>(
                        reversedShareExpRegTensorFP32One,
                        (AscendC::MicroAPI::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, p1);
                    AscendC::MicroAPI::Mul(yZero, yZero, reversedShareExpRegTensorFP32Zero, maskAll);
                    AscendC::MicroAPI::Mul(yOne, yOne, reversedShareExpRegTensorFP32One, maskAll);
                } else {
                    AscendC::MicroAPI::Mul(
                        valueRegTensor, xRegTensor,
                        (AscendC::MicroAPI::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, p1);
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
                auto addr0 = yAddr + (regLoop * vfNum16 + j * outDataLenAlign);
                AscendC::MicroAPI::DataCopyUnAlign(addr0, outZero, u1, tailLoopNum0);
                AscendC::MicroAPI::DataCopyUnAlignPost(addr0, u1, 0);
                auto addr1 = yAddr + (regLoop * vfNum16 + j * outDataLenAlign) + tailLoopNum0;
                AscendC::MicroAPI::DataCopyUnAlign(addr1, outOne, u1, tailLoopNum1);
                AscendC::MicroAPI::DataCopyUnAlignPost(addr1, u1, 0);
            }
        }
    }
}

template <typename T, typename U, const bool isTail>
template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicMxQuantNotTailAxisFP8<T, U, isTail>::ComputecuBLAS(
    int64_t dataLen, int64_t blockCount, __ubuf__ T* xAddr, __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint8_t* yAddr)
{
    constexpr uint32_t vfNum16 = Ops::Base::GetVRegSize() / sizeof(T);     // 寄存器单次能处理的长度
    constexpr uint32_t vfNum32 = Ops::Base::GetVRegSize() / sizeof(float); // cast到FP32后单个寄存器中的元素个数
    uint16_t regLoop = static_cast<uint16_t>(dataLen) / static_cast<uint16_t>(vfNum32);
    uint16_t tailVfLen = static_cast<uint16_t>(dataLen) % static_cast<uint16_t>(vfNum32);
    uint32_t singleLoopNum = dataLen <= vfNum32 ? vfNum16 : vfNum32;
    uint32_t singleTailLoopNum = tailVfLen;
    int64_t outDataLenAlign = this->ubDim_ == DIM2 ? (dataLen + OUT_ELE_NUM_ONE_BLK_FP8 - 1) / OUT_ELE_NUM_ONE_BLK_FP8 *
                                                         OUT_ELE_NUM_ONE_BLK_FP8 :
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
        AscendC::MicroAPI::RegTensor<T> xRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> absMaskBF16;
        AscendC::MicroAPI::RegTensor<uint16_t> fullZero;
        AscendC::MicroAPI::RegTensor<uint32_t> zeroRegTensor32;
        AscendC::MicroAPI::RegTensor<uint32_t> manMaskFP32;
        AscendC::MicroAPI::RegTensor<uint32_t> fp8NanRegTensor;
        AscendC::MicroAPI::RegTensor<uint32_t> biasRegTensor;
        AscendC::MicroAPI::RegTensor<uint32_t> nanRegTensor;

        AscendC::MicroAPI::RegTensor<uint16_t> expScaleMxYZeroRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> expMaxreShareExpFP16YOneRegTensor;
        AscendC::MicroAPI::RegTensor<uint32_t> expMaxAndAddOneFP32RegTensor;
        AscendC::MicroAPI::RegTensor<uint32_t> expAndreShareExpFP32RegTensor;
        AscendC::MicroAPI::RegTensor<uint32_t> manAndmxScaleFP32RegTensor;

        AscendC::MicroAPI::RegTensor<float> yZero;
        AscendC::MicroAPI::RegTensor<float> yOne;
        AscendC::MicroAPI::RegTensor<float> reversedShareExpRegTensorFP32Zero;
        AscendC::MicroAPI::RegTensor<float> reversedShareExpRegTensorFP32One;
        AscendC::MicroAPI::RegTensor<bfloat16_t> valueRegTensor;
        AscendC::MicroAPI::RegTensor<U> yZeroFP8;
        AscendC::MicroAPI::RegTensor<uint8_t> outZeromxScaleFp8;

        AscendC::MicroAPI::UnalignReg u1;
        AscendC::MicroAPI::MaskReg infMask;
        AscendC::MicroAPI::MaskReg zeroMask;
        AscendC::MicroAPI::MaskReg p0;
        AscendC::MicroAPI::MaskReg p1;
        AscendC::MicroAPI::MaskReg p2;
        AscendC::MicroAPI::MaskReg p3;
        AscendC::MicroAPI::MaskReg preMaskScale = AscendC::MicroAPI::CreateMask<uint32_t>();
        AscendC::MicroAPI::MaskReg maskAll =
            AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();

        static constexpr AscendC::MicroAPI::CastTrait castTraitZero = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTraitOne = {
            AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTrait32to8 = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Float = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        AscendC::MicroAPI::Duplicate(absMaskBF16, ABS_FOR_UINT16);
        AscendC::MicroAPI::Duplicate(zeroRegTensor32, 0);
        AscendC::MicroAPI::Duplicate(manMaskFP32, MAN_FOR_FP32);
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8_IN_FP32);
        AscendC::MicroAPI::Duplicate(biasRegTensor, FP32_EXP_BIAS_CUBLAS);
        AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION_PACK);
        AscendC::MicroAPI::Duplicate(fullZero, 0);

        uint32_t pnum = vfNum32;
        p0 = AscendC::MicroAPI::UpdateMask<T>(pnum); // 前vfNum位有效加载
        for (uint16_t i = 0; i < regLoop; i++) {
            this->template LoadData<RoundMode::CAST_TRUNC, roundMode>(xAddr, i * vfNum32, xRegTensor, p0);
            AscendC::MicroAPI::And(
                (AscendC::MicroAPI::RegTensor<uint16_t>&)expMaxreShareExpFP16YOneRegTensor,
                (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor, absMaskBF16, p0);
            for (uint16_t j = 1; j < static_cast<uint16_t>(blockCount); j++) {
                this->template LoadData<RoundMode::CAST_TRUNC, roundMode>(
                    xAddr, j * dataLen + i * vfNum32, xRegTensor, p0);
                AscendC::MicroAPI::And(
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)expScaleMxYZeroRegTensor,
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor, absMaskBF16, p0);
                AscendC::MicroAPI::Max(
                    expMaxreShareExpFP16YOneRegTensor,
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)expMaxreShareExpFP16YOneRegTensor,
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)expScaleMxYZeroRegTensor, p0);
            }
            AscendC::MicroAPI::Interleave(
                expMaxreShareExpFP16YOneRegTensor, fullZero, expMaxreShareExpFP16YOneRegTensor, fullZero);
            AscendC::MicroAPI::Cast<float, T, castTraitHalf2Float>(
                (AscendC::MicroAPI::RegTensor<float>&)expMaxAndAddOneFP32RegTensor,
                (AscendC::MicroAPI::RegTensor<T>&)expMaxreShareExpFP16YOneRegTensor, preMaskScale);
            // 校验
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(
                infMask, expMaxAndAddOneFP32RegTensor, MAX_EXP_FOR_FP32, preMaskScale);
            AscendC::MicroAPI::Compare<uint32_t, CMPMODE::NE>(
                zeroMask, expMaxAndAddOneFP32RegTensor, zeroRegTensor32, preMaskScale);
            AscendC::MicroAPI::Muls(
                (AscendC::MicroAPI::RegTensor<float>&)expMaxAndAddOneFP32RegTensor,
                (AscendC::MicroAPI::RegTensor<float>&)expMaxAndAddOneFP32RegTensor, inv_dtype_max, preMaskScale);
            AscendC::MicroAPI::ShiftRights(
                expAndreShareExpFP32RegTensor, expMaxAndAddOneFP32RegTensor, SHR_NUM_FOR_FP32, preMaskScale); // Exp
            AscendC::MicroAPI::And(
                manAndmxScaleFP32RegTensor, expMaxAndAddOneFP32RegTensor, manMaskFP32, preMaskScale); // Man
            // 条件
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(
                p1, expAndreShareExpFP32RegTensor, NUMBER_ZERO, preMaskScale);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(
                p2, expAndreShareExpFP32RegTensor, NUMBER_TWO_FIVE_FOUR, preMaskScale);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(
                p3, manAndmxScaleFP32RegTensor, NUMBER_ZERO, preMaskScale);
            AscendC::MicroAPI::MaskAnd(p1, p1, p2, preMaskScale);
            AscendC::MicroAPI::MaskAnd(p1, p1, p3, preMaskScale);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(
                p2, expAndreShareExpFP32RegTensor, NUMBER_ZERO, preMaskScale);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(
                p3, manAndmxScaleFP32RegTensor, NUMBER_HALF, preMaskScale);
            AscendC::MicroAPI::MaskAnd(p2, p2, p3, preMaskScale);
            AscendC::MicroAPI::MaskXor(p1, p1, p2, preMaskScale);
            // 向上取整
            AscendC::MicroAPI::Adds(expMaxAndAddOneFP32RegTensor, expAndreShareExpFP32RegTensor, 1, preMaskScale);
            AscendC::MicroAPI::Select(
                manAndmxScaleFP32RegTensor, expMaxAndAddOneFP32RegTensor, expAndreShareExpFP32RegTensor, p1);
            // 校验
            AscendC::MicroAPI::Select<uint32_t>(
                manAndmxScaleFP32RegTensor, manAndmxScaleFP32RegTensor, fp8NanRegTensor, infMask);
            AscendC::MicroAPI::Select<uint32_t>(
                manAndmxScaleFP32RegTensor, manAndmxScaleFP32RegTensor, zeroRegTensor32, zeroMask);
            AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                expScaleMxYZeroRegTensor, manAndmxScaleFP32RegTensor);
            AscendC::MicroAPI::Pack<uint8_t, uint16_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                outZeromxScaleFp8, expScaleMxYZeroRegTensor);
            AscendC::MicroAPI::DataCopyUnAlign(mxScaleAddr, outZeromxScaleFp8, u1, vfNum32); // 从寄存器搬到UB
            AscendC::MicroAPI::DataCopyUnAlignPost(mxScaleAddr, u1, 0);
            // 求1/scale
            AscendC::MicroAPI::ShiftLefts(
                manAndmxScaleFP32RegTensor, manAndmxScaleFP32RegTensor, SHR_NUM_FOR_BF16, preMaskScale);
            AscendC::MicroAPI::Sub(
                expAndreShareExpFP32RegTensor, biasRegTensor, manAndmxScaleFP32RegTensor, preMaskScale);
            // 校验
            AscendC::MicroAPI::Select<uint32_t>(
                expAndreShareExpFP32RegTensor, expAndreShareExpFP32RegTensor, nanRegTensor, infMask);
            AscendC::MicroAPI::Select<uint32_t>(
                expAndreShareExpFP32RegTensor, expAndreShareExpFP32RegTensor, zeroRegTensor32, zeroMask);
            AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                expMaxreShareExpFP16YOneRegTensor, expAndreShareExpFP32RegTensor);
            // 求data value
            for (uint16_t j = 0; j < static_cast<uint16_t>(blockCount); j++) {
                this->template LoadData<toBf16RoundMode, roundMode, true>(
                    xAddr, j * dataLen + i * vfNum32, xRegTensor, p0);
                if constexpr (IsSame<T, half>::value) {
                    AscendC::MicroAPI::Cast<float, T, castTraitZero>(yZero, xRegTensor, p0);
                    AscendC::MicroAPI::Cast<float, T, castTraitOne>(yOne, xRegTensor, p0);
                    AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitZero>(
                        reversedShareExpRegTensorFP32Zero,
                        (AscendC::MicroAPI::RegTensor<bfloat16_t>&)expMaxreShareExpFP16YOneRegTensor, p0);
                    AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitOne>(
                        reversedShareExpRegTensorFP32One,
                        (AscendC::MicroAPI::RegTensor<bfloat16_t>&)expMaxreShareExpFP16YOneRegTensor, p0);
                    AscendC::MicroAPI::Mul(yZero, yZero, reversedShareExpRegTensorFP32Zero, maskAll);
                    AscendC::MicroAPI::Mul(yOne, yOne, reversedShareExpRegTensorFP32One, maskAll);
                } else {
                    AscendC::MicroAPI::Mul(
                        valueRegTensor, xRegTensor,
                        (AscendC::MicroAPI::RegTensor<bfloat16_t>&)expMaxreShareExpFP16YOneRegTensor, p0);
                    AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitZero>(yZero, valueRegTensor, maskAll);
                    AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitOne>(yOne, valueRegTensor, maskAll);
                }
                AscendC::MicroAPI::Interleave(yZero, yOne, yZero, yOne);
                AscendC::MicroAPI::Cast<U, float, castTrait32to8>(yZeroFP8, yZero, maskAll);
                AscendC::MicroAPI::Pack(expScaleMxYZeroRegTensor, (AscendC::MicroAPI::RegTensor<uint32_t>&)yZeroFP8);
                AscendC::MicroAPI::Pack(outZeromxScaleFp8, expScaleMxYZeroRegTensor);
                auto addr = yAddr + (j * outDataLenAlign + i * vfNum32);
                AscendC::MicroAPI::DataCopyUnAlign(addr, outZeromxScaleFp8, u1, singleLoopNum);
                AscendC::MicroAPI::DataCopyUnAlignPost(addr, u1, 0);
            }
        }

        uint32_t tailPnum = tailVfLen;
        p0 = AscendC::MicroAPI::UpdateMask<T>(tailPnum); // 前tailvfLen有效加载
        if (tailVfLen != 0) {
            this->template LoadData<RoundMode::CAST_TRUNC, roundMode>(xAddr, regLoop * vfNum32, xRegTensor, p0);
            AscendC::MicroAPI::And(
                (AscendC::MicroAPI::RegTensor<uint16_t>&)expMaxreShareExpFP16YOneRegTensor,
                (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor, absMaskBF16, p0);
            for (uint16_t j = 1; j < static_cast<uint16_t>(blockCount); j++) {
                this->template LoadData<RoundMode::CAST_TRUNC, roundMode>(
                    xAddr, regLoop * vfNum32 + j * dataLen, xRegTensor, p0);
                AscendC::MicroAPI::And(
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)expScaleMxYZeroRegTensor,
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor, absMaskBF16, p0);
                AscendC::MicroAPI::Max(
                    expMaxreShareExpFP16YOneRegTensor,
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)expMaxreShareExpFP16YOneRegTensor,
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)expScaleMxYZeroRegTensor, p0);
            }
            AscendC::MicroAPI::Interleave(
                expMaxreShareExpFP16YOneRegTensor, fullZero, expMaxreShareExpFP16YOneRegTensor, fullZero);
            AscendC::MicroAPI::Cast<float, T, castTraitHalf2Float>(
                (AscendC::MicroAPI::RegTensor<float>&)expMaxAndAddOneFP32RegTensor,
                (AscendC::MicroAPI::RegTensor<T>&)expMaxreShareExpFP16YOneRegTensor, preMaskScale);
            // 校验
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(
                infMask, expMaxAndAddOneFP32RegTensor, MAX_EXP_FOR_FP32, preMaskScale);
            AscendC::MicroAPI::Compare<uint32_t, CMPMODE::NE>(
                zeroMask, expMaxAndAddOneFP32RegTensor, zeroRegTensor32, preMaskScale);
            AscendC::MicroAPI::Muls(
                (AscendC::MicroAPI::RegTensor<float>&)expMaxAndAddOneFP32RegTensor,
                (AscendC::MicroAPI::RegTensor<float>&)expMaxAndAddOneFP32RegTensor, inv_dtype_max, preMaskScale);
            AscendC::MicroAPI::ShiftRights(
                expAndreShareExpFP32RegTensor, expMaxAndAddOneFP32RegTensor, SHR_NUM_FOR_FP32, preMaskScale); // Exp
            AscendC::MicroAPI::And(
                manAndmxScaleFP32RegTensor, expMaxAndAddOneFP32RegTensor, manMaskFP32, preMaskScale); // Man
            // 条件
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(
                p1, expAndreShareExpFP32RegTensor, NUMBER_ZERO, preMaskScale);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(
                p2, expAndreShareExpFP32RegTensor, NUMBER_TWO_FIVE_FOUR, preMaskScale);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(
                p3, manAndmxScaleFP32RegTensor, NUMBER_ZERO, preMaskScale);
            AscendC::MicroAPI::MaskAnd(p1, p1, p2, preMaskScale);
            AscendC::MicroAPI::MaskAnd(p1, p1, p3, preMaskScale);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(
                p2, expAndreShareExpFP32RegTensor, NUMBER_ZERO, preMaskScale);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(
                p3, manAndmxScaleFP32RegTensor, NUMBER_HALF, preMaskScale);
            AscendC::MicroAPI::MaskAnd(p2, p2, p3, preMaskScale);
            AscendC::MicroAPI::MaskXor(p1, p1, p2, preMaskScale);
            // 向上取整
            AscendC::MicroAPI::Adds(expMaxAndAddOneFP32RegTensor, expAndreShareExpFP32RegTensor, 1, preMaskScale);
            AscendC::MicroAPI::Select(
                manAndmxScaleFP32RegTensor, expMaxAndAddOneFP32RegTensor, expAndreShareExpFP32RegTensor, p1);
            // 校验
            AscendC::MicroAPI::Select<uint32_t>(
                manAndmxScaleFP32RegTensor, manAndmxScaleFP32RegTensor, fp8NanRegTensor, infMask);
            AscendC::MicroAPI::Select<uint32_t>(
                manAndmxScaleFP32RegTensor, manAndmxScaleFP32RegTensor, zeroRegTensor32, zeroMask);
            AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                expScaleMxYZeroRegTensor, manAndmxScaleFP32RegTensor);
            AscendC::MicroAPI::Pack<uint8_t, uint16_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                outZeromxScaleFp8, expScaleMxYZeroRegTensor);
            AscendC::MicroAPI::DataCopyUnAlign(mxScaleAddr, outZeromxScaleFp8, u1, tailVfLen); // 从寄存器搬到UB
            AscendC::MicroAPI::DataCopyUnAlignPost(mxScaleAddr, u1, 0);
            // 求1/scale
            AscendC::MicroAPI::ShiftLefts(
                manAndmxScaleFP32RegTensor, manAndmxScaleFP32RegTensor, SHR_NUM_FOR_BF16, preMaskScale);
            AscendC::MicroAPI::Sub(
                expAndreShareExpFP32RegTensor, biasRegTensor, manAndmxScaleFP32RegTensor, preMaskScale);
            // 校验
            AscendC::MicroAPI::Select<uint32_t>(
                expAndreShareExpFP32RegTensor, expAndreShareExpFP32RegTensor, nanRegTensor, infMask);
            AscendC::MicroAPI::Select<uint32_t>(
                expAndreShareExpFP32RegTensor, expAndreShareExpFP32RegTensor, zeroRegTensor32, zeroMask);
            AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                expMaxreShareExpFP16YOneRegTensor, expAndreShareExpFP32RegTensor);
            if constexpr (isTail) {
                AscendC::MicroAPI::Duplicate(expMaxreShareExpFP16YOneRegTensor, expMaxreShareExpFP16YOneRegTensor, p0);
            }
            // 求data value
            for (uint16_t j = 0; j < static_cast<uint16_t>(blockCount); j++) {
                this->template LoadData<toBf16RoundMode, roundMode, true>(
                    xAddr, regLoop * vfNum32 + j * dataLen, xRegTensor, p0);
                if constexpr (IsSame<T, half>::value) {
                    AscendC::MicroAPI::Cast<float, T, castTraitZero>(yZero, xRegTensor, p0);
                    AscendC::MicroAPI::Cast<float, T, castTraitOne>(yOne, xRegTensor, p0);
                    AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitZero>(
                        reversedShareExpRegTensorFP32Zero,
                        (AscendC::MicroAPI::RegTensor<bfloat16_t>&)expMaxreShareExpFP16YOneRegTensor, p0);
                    AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitOne>(
                        reversedShareExpRegTensorFP32One,
                        (AscendC::MicroAPI::RegTensor<bfloat16_t>&)expMaxreShareExpFP16YOneRegTensor, p0);
                    AscendC::MicroAPI::Mul(yZero, yZero, reversedShareExpRegTensorFP32Zero, maskAll);
                    AscendC::MicroAPI::Mul(yOne, yOne, reversedShareExpRegTensorFP32One, maskAll);
                } else {
                    AscendC::MicroAPI::Mul(
                        valueRegTensor, xRegTensor,
                        (AscendC::MicroAPI::RegTensor<bfloat16_t>&)expMaxreShareExpFP16YOneRegTensor, p0);
                    AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitZero>(yZero, valueRegTensor, maskAll);
                    AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitOne>(yOne, valueRegTensor, maskAll);
                }
                AscendC::MicroAPI::Interleave(yZero, yOne, yZero, yOne);
                AscendC::MicroAPI::Cast<U, float, castTrait32to8>(yZeroFP8, yZero, maskAll);
                AscendC::MicroAPI::Pack(expScaleMxYZeroRegTensor, (AscendC::MicroAPI::RegTensor<uint32_t>&)yZeroFP8);
                AscendC::MicroAPI::Pack(outZeromxScaleFp8, expScaleMxYZeroRegTensor);
                auto addr = yAddr + (regLoop * vfNum32 + j * outDataLenAlign);
                AscendC::MicroAPI::DataCopyUnAlign(addr, outZeromxScaleFp8, u1, singleTailLoopNum);
                AscendC::MicroAPI::DataCopyUnAlignPost(addr, u1, 0);
            }
        }
    }
}
} // namespace DynamicMxQuant
#endif // DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_FP8_H
