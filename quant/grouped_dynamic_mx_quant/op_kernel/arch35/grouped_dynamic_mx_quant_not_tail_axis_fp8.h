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
 * \file grouped_dynamic_mx_quant_not_tail_axis_fp8.h
 * \brief
 */

#ifndef GROUPED_DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_FP8_H
#define GROUPED_DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_FP8_H
#include "grouped_dynamic_mx_quant_common.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace GroupedDynamicMxQuant {
using namespace AscendC;
#define FLOAT_OVERFLOW_MODE_CTRL 60
template <typename T, typename U>
class GroupedDynamicMxQuantBaseFP8 {
public:
    __aicore__ inline GroupedDynamicMxQuantBaseFP8() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR mxScale, const GroupedDynamicMxQuantTilingData &tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const GroupedDynamicMxQuantTilingData &tilingData);
    __aicore__ inline void SplitPostAxisCompute(int64_t dataLen, int64_t blockCount);
    __aicore__ inline void CopyIn(int64_t offset, int64_t blockCount, int64_t dataLen);
    __aicore__ inline void CopyOut(int64_t xOffset, int64_t scaleOffset, int64_t blockCount, int64_t dataLen);
    __aicore__ inline void Compute(int64_t dataLen, int64_t blockCount,
        __ubuf__ T *xAddr, __ubuf__ uint8_t *mxScaleAddr, __ubuf__ uint8_t *yAddr);
private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, DB_BUFFER> inQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> mxScaleQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue_;
    TBuf<TPosition::VECCALC> calcBuf;
    GlobalTensor<T> xGm_;
    GlobalTensor<int32_t> groupIndexGm_;
    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<uint8_t> mxScaleGm_;

    int64_t blockIdx_ = 0;     // 核id
    int64_t usedCoreNum_ = 0;     // 使用的核数
    int64_t blockFactor_ = 0;     // 整核计算[g, cacheline]次数
    int64_t tailBlockFactor_ = 0; // 尾核计算[g, cacheline]次数
    int64_t blockLoopOffset_ = 0; // 当前核起始偏移
    int64_t uo_ = 0;             // n轴计算cacheline次数
    int64_t maxUbCol_ = 0; 
    int64_t ubFactor_ = 0;       // cacheline大小
    int64_t tailUbFactor_ = 0;   // n轴cacheline尾块
    int64_t preAxisSize_ = 0; // m轴大小
    int64_t postAxisSize_ = 0; // n轴大小
    int64_t blockSize_ = 0;     // 使用核数，仅支持32
};

template <typename T, typename U>
__aicore__ inline void GroupedDynamicMxQuantBaseFP8<T, U>::ParseTilingData(const GroupedDynamicMxQuantTilingData &tilingData)
{
    usedCoreNum_ = tilingData.usedCoreNum;
    uo_ = tilingData.uo;
    maxUbCol_ = tilingData.maxUbCol;
    ubFactor_ = tilingData.ubFactor;
    tailUbFactor_ = tilingData.tailUbFactor;
    blockFactor_ = tilingData.blockFactor;
    tailBlockFactor_ = tilingData.tailBlockFactor;
    blockSize_ = tilingData.blockSize;
    preAxisSize_ = tilingData.preAxisSize;
    postAxisSize_ = tilingData.postAxisSize;
}

template <typename T, typename U>
__aicore__ inline void GroupedDynamicMxQuantBaseFP8<T, U>::CopyIn(int64_t offset, int64_t blockCount, int64_t dataLen)
{
    DataCopyExtParams inCopyParams_ = {
        static_cast<uint16_t>(blockCount),
        static_cast<uint32_t>(dataLen * sizeof(T)),
        static_cast<uint32_t>((postAxisSize_ - dataLen) * sizeof(T)),
        static_cast<uint32_t>((dataLen +31)/32*2-(dataLen+15)/16),
        static_cast<uint32_t>(0)
    };
    
    DataCopyPadExtParams<T> padParams_ = { false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0) };
    LocalTensor<T> x = inQueue_.AllocTensor<T>();
    DataCopyPad(x, xGm_[offset], inCopyParams_, padParams_);
    inQueue_.EnQue(x);
}

template <typename T, typename U>
__aicore__ inline void GroupedDynamicMxQuantBaseFP8<T, U>::CopyOut(int64_t xOffset, int64_t scaleOffset,
    int64_t blockCount, int64_t dataLen)
{
    DataCopyExtParams outCopyParams_ = {
        static_cast<uint16_t>(blockCount),
        static_cast<uint32_t>(dataLen * sizeof(uint8_t)),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>((postAxisSize_ - dataLen) * sizeof(uint8_t)),
        static_cast<uint32_t>(0)
    };

    DataCopyExtParams scaleCopyOutParams = {
        static_cast<uint16_t>((blockCount + 63) / 64), static_cast<uint32_t>(dataLen * 2 * sizeof(uint8_t)),
        static_cast<uint32_t>(2 * ((dataLen + 31) / 32 * 32 - dataLen) / 32),  // 搬入pad做完交织后变成2倍pad
        static_cast<uint32_t>(2 * (postAxisSize_ - dataLen) * sizeof(uint8_t)), static_cast<uint32_t>(0)
    };

    LocalTensor<uint8_t> y = outQueue_.DeQue<uint8_t>();
    DataCopyPad(yGm_[xOffset], y, outCopyParams_);
    outQueue_.FreeTensor(y);

    LocalTensor<uint8_t> mxScale = mxScaleQueue_.DeQue<uint8_t>();
    DataCopyPad(mxScaleGm_[scaleOffset], mxScale, scaleCopyOutParams);
    mxScaleQueue_.FreeTensor(mxScale);
}

template <typename T, typename U>
__aicore__ inline void GroupedDynamicMxQuantBaseFP8<T, U>::Init(GM_ADDR x, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR mxScale,
    const GroupedDynamicMxQuantTilingData &tilingData)
{
    #if (__NPU_ARCH__ == 3101)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL,FLOAT_OVERFLOW_MODE_CTRL>(0);
    #endif
    blockIdx_ = GetBlockIdx();
    ParseTilingData(tilingData);
    if (this->blockIdx_ >= this->usedCoreNum_) {
        return;
    }
    
    int64_t bufferSize_ = this->ubFactor_ * this->maxUbCol_ * sizeof(T);

    this->xGm_.SetGlobalBuffer((__gm__ T *)(x));
    this->groupIndexGm_.SetGlobalBuffer((__gm__ int32_t *)(groupIndex)); // groupIndex getvalue方式加载
    this->yGm_.SetGlobalBuffer((__gm__ uint8_t *)(y));
    this->mxScaleGm_.SetGlobalBuffer((__gm__ uint8_t *)(mxScale));

    bufferSize_ = Ops::Base::CeilAlign(bufferSize_, static_cast<int64_t>(Ops::Base::GetUbBlockSize()));
    this->pipe_.InitBuffer(this->inQueue_, DB_BUFFER, bufferSize_);
    this->pipe_.InitBuffer(this->mxScaleQueue_, DB_BUFFER, bufferSize_);
    this->pipe_.InitBuffer(this->outQueue_, DB_BUFFER, bufferSize_);
    this->pipe_.InitBuffer(this->calcBuf, bufferSize_);
}

template <typename T, typename U>
__aicore__ inline void GroupedDynamicMxQuantBaseFP8<T, U>::Process()
{
    if (this->blockIdx_ >= this->usedCoreNum_) {
        return;
    }
    
    bool isTailBlock = blockIdx_ == (usedCoreNum_ - 1);
    int64_t loopNum = isTailBlock ? this->tailBlockFactor_ : this->blockFactor_;
    blockLoopOffset_ = this->blockIdx_ * this->blockFactor_;

    for (int64_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
        int64_t curLoopIdxInAllCore = loopIdx + blockLoopOffset_;
        int64_t gIdx = curLoopIdxInAllCore / uo_;
        int64_t nIdx = curLoopIdxInAllCore % uo_;
        int64_t gIdxValueStart = 0;
        if (gIdx > 0) {
            gIdxValueStart = groupIndexGm_.GetValue(gIdx -1);
        }
        int64_t gIdxValueEnd = groupIndexGm_.GetValue(gIdx);
        assert((0 <= gIdxValueEnd && gIdxValueEnd <= preAxisSize_), "groupIndex %lld = %lld out of range[0 %lld]!\n", gIdx, gIdxValueEnd, preAxisSize_);
        assert((gIdxValueStart <= gIdxValueEnd), "groupIndex %lld = %lld less than previous groupIdx = %lld!\n", gIdx, gIdxValueEnd, gIdxValueStart);
        int64_t groupSizeIdx = gIdxValueEnd - gIdxValueStart;
        int64_t rowOffset = gIdxValueStart * postAxisSize_ + nIdx * ubFactor_;
        int64_t rowGroupOffset = 2*(gIdxValueStart /64 + gIdx) * postAxisSize_ + 2 * nIdx * ubFactor_; // e8m0_2
        bool isTailLoopInUbDim = nIdx == uo_ - 1;
        int64_t dataLen = isTailLoopInUbDim ? tailUbFactor_ : ubFactor_;
        int64_t inLoopNum = Ops::Base::CeilDiv(groupSizeIdx, maxUbCol_);
        for (int64_t j = 0; j < inLoopNum; j++){
            int64_t blockCount = (j == inLoopNum - 1)? groupSizeIdx - j * maxUbCol_: maxUbCol_;
            int64_t xGmOffset = rowOffset + j * maxUbCol_ * postAxisSize_;
            int64_t scaleGmOffset = rowGroupOffset + j * (maxUbCol_/32) * postAxisSize_;
            CopyIn(xGmOffset, blockCount, dataLen);
            SplitPostAxisCompute(blockCount, (dataLen+31)/32*32);
            CopyOut(xGmOffset, scaleGmOffset, blockCount, dataLen);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void GroupedDynamicMxQuantBaseFP8<T, U>::SplitPostAxisCompute(int64_t blockCount, int64_t dataLen)
{
    LocalTensor<T> x = this->inQueue_.template DeQue<T>();
    LocalTensor<uint8_t> mxScale = this->mxScaleQueue_.template AllocTensor<uint8_t>();
    LocalTensor<uint8_t> y = this->outQueue_.template AllocTensor<uint8_t>();
    LocalTensor<uint8_t> tmpBuff = calcBuf.Get<uint8_t>();

    int64_t rowNumFullCount = blockCount / 32;
    int64_t rowNumFullAlign32 = (blockCount+31) / 32;
    int64_t rowNumFullAlign64 = (blockCount+63) / 64;
    int64_t rowNumResSize = blockCount % 32;

    int64_t offset = 0;
    for(int64_t i=0; i < rowNumFullCount; i++){
        offset = i * 32 * dataLen;
        auto xAddr = (__ubuf__ T *)x.GetPhyAddr() + offset;
        auto mxScaleAddr = (__ubuf__ uint8_t *)tmpBuff.GetPhyAddr() + i * dataLen;
        auto yAddr = (__ubuf__ uint8_t *)y.GetPhyAddr() + offset;
        Compute(dataLen, 32, xAddr, mxScaleAddr, yAddr);
    }
    if (rowNumResSize!=0) {
        int64_t i = rowNumFullCount;
        offset = i * 32 * dataLen;
        auto xAddr = (__ubuf__ T *)x.GetPhyAddr() + offset;
        auto mxScaleAddr = (__ubuf__ uint8_t *)tmpBuff.GetPhyAddr() + i * dataLen;
        auto yAddr = (__ubuf__ uint8_t *)y.GetPhyAddr() + offset;
        Compute(dataLen, rowNumResSize, xAddr, mxScaleAddr, yAddr);
    }
    if (rowNumFullAlign32 % 2 != 0){
        Duplicate<uint8_t>(tmpBuff[rowNumFullAlign32 * dataLen], (uint8_t)0, 128); // e8m0_2偶数pad 0
    }
    for(int64_t i=0; i < rowNumFullAlign64; i++){
        Interleave(mxScale[2*i*dataLen], mxScale[(2*i+1)*dataLen], tmpBuff[2*i*dataLen], tmpBuff[(2*i+1)*dataLen], dataLen);
    }
    this->mxScaleQueue_.template EnQue(mxScale);
    this->outQueue_.template EnQue(y);
    this->inQueue_.template FreeTensor(x);
}

template <typename T, typename U>
__aicore__ inline void GroupedDynamicMxQuantBaseFP8<T, U>::Compute(int64_t dataLen, int64_t blockCount,
    __ubuf__ T *xAddr, __ubuf__ uint8_t *mxScaleAddr, __ubuf__ uint8_t *yAddr)
{
    constexpr uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(T); // 寄存器单次处理能处理的长度
    constexpr uint32_t vfNum = Ops::Base::GetVRegSize() / sizeof(float); // cast到FP32后单个寄存器中的元素个数
    uint16_t regLoop = static_cast<uint16_t>(dataLen) / static_cast<uint16_t>(vfLen); // 当前loop需要处理的长度
    uint16_t tailVfLen = static_cast<uint16_t>(dataLen) % static_cast<uint16_t>(vfLen);
    uint32_t loopNum0 = dataLen <= vfNum ? vfLen : vfNum;
    uint32_t loopNum1 = dataLen <= vfNum ? 0 : (vfLen - vfNum);
    uint32_t tailLoopNum0 = tailVfLen <= vfNum ? tailVfLen : vfNum;
    uint32_t tailLoopNum1 = tailVfLen <= vfNum ? 0 : (tailVfLen - vfNum);
    int64_t outDataLenAlign = dataLen;
    uint16_t FP8_BF16_MAX_EXP = 0;
    if constexpr (IsSame<U, fp8_e4m3fn_t>::value){
        FP8_BF16_MAX_EXP = FP8_E4M3_MAX_EXP;
    } else if constexpr (IsSame<U, fp8_e5m2_t>::value){
        FP8_BF16_MAX_EXP = FP8_E5M2_MAX_EXP;
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
        AscendC::MicroAPI::RegTensor<uint8_t> mxScale;
        AscendC::MicroAPI::RegTensor<uint16_t> reversedShareExpRegTensor;
        AscendC::MicroAPI::RegTensor<float> reversedShareExpRegTensorFP32Zero;
        AscendC::MicroAPI::RegTensor<float> reversedShareExpRegTensorFP32One;
        AscendC::MicroAPI::RegTensor<uint16_t> specialExpRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> biasRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> zeroRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> nanRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> yRegTensorZero;
        AscendC::MicroAPI::RegTensor<uint16_t> yRegTensorOne;
        AscendC::MicroAPI::RegTensor<uint8_t> outZero;
        AscendC::MicroAPI::RegTensor<uint8_t> outOne;
        AscendC::MicroAPI::RegTensor<float> yZero;
        AscendC::MicroAPI::RegTensor<float> yOne;
        AscendC::MicroAPI::RegTensor<U> yZeroFP8;
        AscendC::MicroAPI::RegTensor<U> yOneFP8;
        AscendC::MicroAPI::RegTensor<bfloat16_t> valueRegTensor;
        AscendC::MicroAPI::UnalignReg u0;
        AscendC::MicroAPI::UnalignReg u1;
        AscendC::MicroAPI::MaskReg infMask;
        AscendC::MicroAPI::MaskReg zeroMask;
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
        AscendC::MicroAPI::Duplicate(maxEleRegTensor, MAX_EXP_FOR_BF16);
        AscendC::MicroAPI::Duplicate(fp8MaxExpRegTensor, FP8_BF16_MAX_EXP);
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        AscendC::MicroAPI::Duplicate(biasRegTensor, BF16_EXP_BIAS);
        AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
        AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
        for (uint16_t i = 0; i < regLoop; i++) {
            uint32_t pnum = vfLen;
            AscendC::MicroAPI::MaskReg p0 = AscendC::MicroAPI::UpdateMask<uint16_t>(pnum);
            DataCopy(xRegTensor, xAddr + i * vfLen);
            if constexpr (IsSame<T, half>::value) {
                AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(xBF16RegTensor, xRegTensor, p0);
                AscendC::MicroAPI::And(expMaxRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xBF16RegTensor,
                maxEleRegTensor, p0);
            } else {
                AscendC::MicroAPI::And(expMaxRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor,
                maxEleRegTensor, p0);
            }
            for (uint16_t j = 1; j < static_cast<uint16_t>(blockCount); j++) {
                DataCopy(xRegTensor, xAddr + j * dataLen + i * vfLen);
                if constexpr (IsSame<T, half>::value) {
                    AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(xBF16RegTensor, xRegTensor, p0);
                    AscendC::MicroAPI::And(expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xBF16RegTensor,
                    maxEleRegTensor, p0);
                } else {
                    AscendC::MicroAPI::And(expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor,
                    maxEleRegTensor, p0);
                }   
                AscendC::MicroAPI::Max(expMaxRegTensor, expMaxRegTensor, expRegTensor, p0);
            }
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxRegTensor, maxEleRegTensor, p0);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxRegTensor, zeroRegTensor, p0);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMaxRegTensor,
                fp8MaxExpRegTensor, p0);
            AscendC::MicroAPI::Select<uint16_t>(expMaxRegTensor, fp8MaxExpRegTensor, expMaxRegTensor,
                invalidDataMask);
            AscendC::MicroAPI::Sub(expMaxRegTensor, expMaxRegTensor, fp8MaxExpRegTensor, p0);
            AscendC::MicroAPI::ShiftRights(mxScaleRegTensor, expMaxRegTensor, SHR_NUM_FOR_BF16, p0);
            AscendC::MicroAPI::Select<uint16_t>(mxScaleRegTensor, mxScaleRegTensor, fp8NanRegTensor, infMask);
            AscendC::MicroAPI::Select<uint16_t>(mxScaleRegTensor, mxScaleRegTensor, zeroRegTensor, zeroMask);
            AscendC::MicroAPI::Pack(mxScale, mxScaleRegTensor);
            AscendC::MicroAPI::DataCopyUnAlign(mxScaleAddr, mxScale, u1, vfLen);
            AscendC::MicroAPI::DataCopyUnAlignPost(mxScaleAddr, u1, 0);
            // 求1/scale
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, expMaxRegTensor, biasRegTensor, p0);
            AscendC::MicroAPI::Sub(reversedShareExpRegTensor, biasRegTensor, expMaxRegTensor, p0);
            AscendC::MicroAPI::Select<uint16_t>(reversedShareExpRegTensor, reversedShareExpRegTensor, nanRegTensor,
                infMask);
            AscendC::MicroAPI::Select<uint16_t>(reversedShareExpRegTensor, reversedShareExpRegTensor, zeroRegTensor,
                zeroMask);
            AscendC::MicroAPI::Select<uint16_t>(reversedShareExpRegTensor, specialExpRegTensor,
                reversedShareExpRegTensor, specialDataMask);

            // 求data value
            for (uint16_t j = 0; j < static_cast<uint16_t>(blockCount); j++) {
                DataCopy(xRegTensor, xAddr + j * dataLen + i * vfLen);
                if constexpr (IsSame<T, half>::value) {
                    AscendC::MicroAPI::Cast<float, T, castTraitZero>(yZero, xRegTensor, p0);
                    AscendC::MicroAPI::Cast<float, T, castTraitOne>(yOne, xRegTensor, p0);
                    AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitZero>(reversedShareExpRegTensorFP32Zero, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, p0);
                    AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitOne>(reversedShareExpRegTensorFP32One, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, p0);
                    AscendC::MicroAPI::Mul(yZero, yZero, reversedShareExpRegTensorFP32Zero, maskAll);
                    AscendC::MicroAPI::Mul(yOne, yOne, reversedShareExpRegTensorFP32One, maskAll);
                } else {
                    AscendC::MicroAPI::Mul(valueRegTensor, xRegTensor,
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
                auto addr0 = yAddr + (j * outDataLenAlign + i * vfLen);
                AscendC::MicroAPI::DataCopyUnAlign(addr0, outZero, u1, loopNum0);
                AscendC::MicroAPI::DataCopyUnAlignPost(addr0, u1, 0);
                auto addr1 = yAddr + (j * outDataLenAlign + i * vfLen) + loopNum0;
                AscendC::MicroAPI::DataCopyUnAlign(addr1, outOne, u1, loopNum1);
                AscendC::MicroAPI::DataCopyUnAlignPost(addr1, u1, 0);
            }
        }

        if (tailVfLen != 0) {
            uint32_t tailPnum = tailVfLen;
            AscendC::MicroAPI::MaskReg p1 = AscendC::MicroAPI::UpdateMask<T>(tailPnum);
            DataCopy(xRegTensor, xAddr + regLoop * vfLen);
             if constexpr (IsSame<T, half>::value) {
                AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(xBF16RegTensor, xRegTensor, p1);
                AscendC::MicroAPI::And(expMaxRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xBF16RegTensor,
                maxEleRegTensor, p1);
            } else {
                AscendC::MicroAPI::And(expMaxRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor,
                maxEleRegTensor, p1);
            }
            for (uint16_t j = 1; j < static_cast<uint16_t>(blockCount); j++) {
                DataCopy(xRegTensor, xAddr + regLoop * vfLen + j * dataLen);
                if constexpr (IsSame<T, half>::value) {
                    AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(xBF16RegTensor, xRegTensor, p1);
                    AscendC::MicroAPI::And(expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xBF16RegTensor,
                    maxEleRegTensor, p1);
                } else {
                    AscendC::MicroAPI::And(expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor,
                    maxEleRegTensor, p1);
                }
                AscendC::MicroAPI::Max(expMaxRegTensor, expMaxRegTensor, expRegTensor, p1);
            }
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxRegTensor, maxEleRegTensor, p1);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxRegTensor, zeroRegTensor, p1);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMaxRegTensor,
                fp8MaxExpRegTensor, p1);
            AscendC::MicroAPI::Select<uint16_t>(expMaxRegTensor, fp8MaxExpRegTensor, expMaxRegTensor,
                invalidDataMask);
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
            AscendC::MicroAPI::Select<uint16_t>(reversedShareExpRegTensor, specialExpRegTensor,
                reversedShareExpRegTensor, specialDataMask);
            AscendC::MicroAPI::Select<uint16_t>(reversedShareExpRegTensor, reversedShareExpRegTensor, nanRegTensor,
                infMask);
            AscendC::MicroAPI::Select<uint16_t>(reversedShareExpRegTensor, reversedShareExpRegTensor, zeroRegTensor,
                zeroMask);
            // 求data value
            for (uint16_t j = 0; j < static_cast<uint16_t>(blockCount); j++) {
                DataCopy(xRegTensor, xAddr + regLoop * vfLen + j * dataLen);
                if constexpr (IsSame<T, half>::value) {
                    AscendC::MicroAPI::Cast<float, T, castTraitZero>(yZero, xRegTensor, p1);
                    AscendC::MicroAPI::Cast<float, T, castTraitOne>(yOne, xRegTensor, p1);
                    AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitZero>(reversedShareExpRegTensorFP32Zero, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, p1);
                    AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitOne>(reversedShareExpRegTensorFP32One, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, p1);
                    AscendC::MicroAPI::Mul(yZero, yZero, reversedShareExpRegTensorFP32Zero, maskAll);
                    AscendC::MicroAPI::Mul(yOne, yOne, reversedShareExpRegTensorFP32One, maskAll);
                } else {
                    AscendC::MicroAPI::Mul(valueRegTensor, xRegTensor,
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
                auto addr0 = yAddr + (regLoop * vfLen + j * outDataLenAlign);
                AscendC::MicroAPI::DataCopyUnAlign(addr0, outZero, u1, tailLoopNum0);
                AscendC::MicroAPI::DataCopyUnAlignPost(addr0, u1, 0);
                auto addr1 = yAddr + (regLoop * vfLen + j * outDataLenAlign) + tailLoopNum0;
                AscendC::MicroAPI::DataCopyUnAlign(addr1, outOne, u1, tailLoopNum1);
                AscendC::MicroAPI::DataCopyUnAlignPost(addr1, u1, 0);
            }
        }
    }
}

} // namespace GroupedDynamicMxQuant
#endif // GROUPED_DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_FP8_H
