/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "grouped_dynamic_mx_quant_tilingdata.h"
#include "grouped_dynamic_mx_quant_struct.h"
#include "kernel_operator.h"
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
    __aicore__ inline void ComputeOCP(int64_t dataLen, int64_t blockCount,
        __ubuf__ T *xAddr, __ubuf__ uint8_t *mxScaleAddr, __ubuf__ uint8_t *yAddr);
    __aicore__ inline void ComputecuBLAS(int64_t dataLen, int64_t blockCount,
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
    int64_t blockSize_ = 0;     // 量化数据块大小，仅支持32
    int64_t scaleAlg_ = 0;
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
    scaleAlg_ = tilingData.scaleAlg;
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
        static_cast<uint32_t>((dataLen + 31) / 32 * 2 - (dataLen + 15) / 16),
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
#if (__NPU_ARCH__ == 3510)
    SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
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
        assert((0 <= gIdxValueEnd && gIdxValueEnd <= preAxisSize_), "groupIndex %lld = %lld out of range[0 %lld]!\n",
            gIdx, gIdxValueEnd, preAxisSize_);
        assert((gIdxValueStart <= gIdxValueEnd), "groupIndex %lld = %lld less than previous groupIdx = %lld!\n",
            gIdx, gIdxValueEnd, gIdxValueStart);
        int64_t groupSizeIdx = gIdxValueEnd - gIdxValueStart;
        int64_t rowOffset = gIdxValueStart * postAxisSize_ + nIdx * ubFactor_;
        int64_t rowGroupOffset = 2 * (gIdxValueStart / 64 + gIdx) * postAxisSize_ + 2 * nIdx * ubFactor_; // e8m0_2
        bool isTailLoopInUbDim = nIdx == uo_ - 1;
        int64_t dataLen = isTailLoopInUbDim ? tailUbFactor_ : ubFactor_;
        int64_t inLoopNum = Ops::Base::CeilDiv(groupSizeIdx, maxUbCol_);
        for (int64_t j = 0; j < inLoopNum; j++){
            int64_t blockCount = (j == inLoopNum - 1)? groupSizeIdx - j * maxUbCol_: maxUbCol_;
            int64_t xGmOffset = rowOffset + j * maxUbCol_ * postAxisSize_;
            int64_t scaleGmOffset = rowGroupOffset + j * (maxUbCol_ / 32) * postAxisSize_;
            CopyIn(xGmOffset, blockCount, dataLen);
            SplitPostAxisCompute(blockCount, (dataLen + 31) / 32 * 32);
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
    int64_t rowNumFullAlign32 = (blockCount + 31) / 32;
    int64_t rowNumFullAlign64 = (blockCount + 63) / 64;
    int64_t rowNumResSize = blockCount % 32;

    int64_t offset = 0;
    if (scaleAlg_ == 0) {
        for(int64_t i = 0; i < rowNumFullCount; i++) {
            offset = i * 32 * dataLen;
            auto xAddr = (__ubuf__ T *)x.GetPhyAddr() + offset;
            auto mxScaleAddr = (__ubuf__ uint8_t *)tmpBuff.GetPhyAddr() + i * dataLen;
            auto yAddr = (__ubuf__ uint8_t *)y.GetPhyAddr() + offset;
            ComputeOCP(dataLen, 32, xAddr, mxScaleAddr, yAddr);
        }
    } else if (scaleAlg_ == 1) {
        for(int64_t i = 0; i < rowNumFullCount; i++) {
            offset = i * 32 * dataLen;
            auto xAddr = (__ubuf__ T *)x.GetPhyAddr() + offset;
            auto mxScaleAddr = (__ubuf__ uint8_t *)tmpBuff.GetPhyAddr() + i * dataLen;
            auto yAddr = (__ubuf__ uint8_t *)y.GetPhyAddr() + offset;
            ComputecuBLAS(dataLen, 32, xAddr, mxScaleAddr, yAddr);
        }
    }

    if (rowNumResSize != 0) {
        int64_t i = rowNumFullCount;
        offset = i * 32 * dataLen;
        auto xAddr = (__ubuf__ T *)x.GetPhyAddr() + offset;
        auto mxScaleAddr = (__ubuf__ uint8_t *)tmpBuff.GetPhyAddr() + i * dataLen;
        auto yAddr = (__ubuf__ uint8_t *)y.GetPhyAddr() + offset;
        if (scaleAlg_ == 0) {
            ComputeOCP(dataLen, rowNumResSize, xAddr, mxScaleAddr, yAddr);
        } else if (scaleAlg_ == 1) {
            ComputecuBLAS(dataLen, rowNumResSize, xAddr, mxScaleAddr, yAddr);
        }
    }
    if (rowNumFullAlign32 % 2 != 0) {
        Duplicate<uint8_t>(tmpBuff[rowNumFullAlign32 * dataLen], (uint8_t)0, 128); // e8m0_2偶数pad 0
    }
    for(int64_t i = 0; i < rowNumFullAlign64; i++){
        Interleave(mxScale[2 * i * dataLen], mxScale[(2 * i + 1) * dataLen], tmpBuff[2 * i * dataLen],
            tmpBuff[(2 * i + 1) * dataLen], dataLen);
    }
    this->mxScaleQueue_.template EnQue(mxScale);
    this->outQueue_.template EnQue(y);
    this->inQueue_.template FreeTensor(x);
}

template <typename T, typename U>
__aicore__ inline void GroupedDynamicMxQuantBaseFP8<T, U>::ComputeOCP(int64_t dataLen, int64_t blockCount,
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
    int64_t blockCountLoop = blockCount / 2;
    if constexpr (IsSame<U, fp8_e4m3fn_t>::value){
        FP8_BF16_MAX_EXP = FP8_E4M3_MAX_EXP;
    } else if constexpr (IsSame<U, fp8_e5m2_t>::value){
        FP8_BF16_MAX_EXP = FP8_E5M2_MAX_EXP;
    }

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> xRegTensor;
        Reg::RegTensor<uint16_t> expRegTensor;
        Reg::RegTensor<uint16_t> expMaxRegTensor;
        Reg::RegTensor<T> xRegTensor0;
        Reg::RegTensor<T> xRegTensor1;
        Reg::RegTensor<bfloat16_t> xBF16RegTensor0;
        Reg::RegTensor<bfloat16_t> xBF16RegTensor1;
        Reg::RegTensor<uint16_t> expMaxRegTensor0;
        Reg::RegTensor<uint16_t> expMaxRegTensor1;
        Reg::RegTensor<uint16_t> maxEleRegTensor;
        Reg::RegTensor<uint16_t> fp8MaxExpRegTensor;
        Reg::RegTensor<uint16_t> fp8NanRegTensor;
        Reg::RegTensor<uint16_t> mxScaleRegTensor;
        Reg::RegTensor<uint8_t> mxScale;
        Reg::RegTensor<uint16_t> reversedShareExpRegTensor;
        Reg::RegTensor<float> reversedShareExpRegTensorFP32Zero;
        Reg::RegTensor<float> reversedShareExpRegTensorFP32One;
        Reg::RegTensor<uint16_t> specialExpRegTensor;
        Reg::RegTensor<uint16_t> biasRegTensor;
        Reg::RegTensor<uint16_t> zeroRegTensor;
        Reg::RegTensor<uint16_t> nanRegTensor;
        Reg::RegTensor<uint16_t> yRegTensorZero;
        Reg::RegTensor<uint16_t> yRegTensorOne;
        Reg::RegTensor<uint8_t> outZero;
        Reg::RegTensor<uint8_t> outOne;
        Reg::RegTensor<float> yZero;
        Reg::RegTensor<float> yOne;
        Reg::RegTensor<U> yZeroFP8;
        Reg::RegTensor<U> yOneFP8;
        Reg::RegTensor<bfloat16_t> valueRegTensor;

        Reg::UnalignReg u0;
        Reg::MaskReg infMask;
        Reg::MaskReg zeroMask;
        Reg::MaskReg invalidDataMask;
        Reg::MaskReg specialDataMask;
        Reg::MaskReg maskAll = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();

        static constexpr Reg::CastTrait castTraitZero = {
            Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
            Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitOne = {
            Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
            Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTrait32to8 = {
            Reg::RegLayout::ZERO, Reg::SatMode::SAT,
            Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {
            Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
            Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC}; 

        Reg::Duplicate(maxEleRegTensor, MAX_EXP_FOR_BF16);
        Reg::Duplicate(fp8MaxExpRegTensor, FP8_BF16_MAX_EXP);
        Reg::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        Reg::Duplicate(biasRegTensor, BF16_EXP_BIAS);
        Reg::Duplicate(zeroRegTensor, 0);
        Reg::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        Reg::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
        Reg::Duplicate(expMaxRegTensor, 0);

        for (uint16_t i = 0; i < regLoop; i++) {
            uint32_t pnum = vfLen;
            Reg::MaskReg p0 = Reg::UpdateMask<uint16_t>(pnum);
            for (uint16_t j = 0; j <= static_cast<uint16_t>(blockCountLoop); j++) {
                Reg::LoadAlign(xRegTensor0, xAddr + j * dataLen + i * vfLen);
                Reg::LoadAlign(xRegTensor1, xAddr + (static_cast<uint16_t>(blockCount) - j - 1) * dataLen + i * vfLen);
                if constexpr (IsSame<T, half>::value) {
                    Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(xBF16RegTensor0, xRegTensor0, p0);
                    Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(xBF16RegTensor1, xRegTensor1, p0);
                    Reg::And(expMaxRegTensor0, (Reg::RegTensor<uint16_t>&)xBF16RegTensor0, maxEleRegTensor, p0);
                    Reg::And(expMaxRegTensor1, (Reg::RegTensor<uint16_t>&)xBF16RegTensor1, maxEleRegTensor, p0);
                } else {
                    Reg::And(expMaxRegTensor0, (Reg::RegTensor<uint16_t>&)xRegTensor0, maxEleRegTensor, p0);
                    Reg::And(expMaxRegTensor1, (Reg::RegTensor<uint16_t>&)xRegTensor1, maxEleRegTensor, p0);
                }   
                Reg::Max(expRegTensor, expMaxRegTensor0, expMaxRegTensor1, p0);
                Reg::Max(expMaxRegTensor, expRegTensor, expMaxRegTensor, p0);
            }
            Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxRegTensor, maxEleRegTensor, p0);
            Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxRegTensor, zeroRegTensor, p0);
            Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMaxRegTensor, fp8MaxExpRegTensor, p0);
            Reg::Select<uint16_t>(expMaxRegTensor, fp8MaxExpRegTensor, expMaxRegTensor, invalidDataMask);
            Reg::Sub(expMaxRegTensor, expMaxRegTensor, fp8MaxExpRegTensor, p0);
            Reg::ShiftRights(mxScaleRegTensor, expMaxRegTensor, SHR_NUM_FOR_BF16, p0);
            Reg::Select<uint16_t>(mxScaleRegTensor, mxScaleRegTensor, fp8NanRegTensor, infMask);
            Reg::Select<uint16_t>(mxScaleRegTensor, mxScaleRegTensor, zeroRegTensor, zeroMask);
            Reg::Pack(mxScale, mxScaleRegTensor);
            Reg::StoreUnAlign(mxScaleAddr, mxScale, u0, vfLen);
            Reg::StoreUnAlignPost(mxScaleAddr, u0, 0);

            // 求1/scale
            Reg::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, expMaxRegTensor, biasRegTensor, p0);
            Reg::Sub(reversedShareExpRegTensor, biasRegTensor, expMaxRegTensor, p0);
            Reg::Select<uint16_t>(reversedShareExpRegTensor, reversedShareExpRegTensor, nanRegTensor, infMask);
            Reg::Select<uint16_t>(reversedShareExpRegTensor, reversedShareExpRegTensor, zeroRegTensor, zeroMask);
            Reg::Select<uint16_t>(reversedShareExpRegTensor, specialExpRegTensor, reversedShareExpRegTensor, specialDataMask);

            // 求data value
            if constexpr (IsSame<T, half>::value) {
                Reg::Cast<float, bfloat16_t, castTraitZero>(reversedShareExpRegTensorFP32Zero,
                    (Reg::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, p0);
                Reg::Cast<float, bfloat16_t, castTraitOne>(reversedShareExpRegTensorFP32One,
                    (Reg::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, p0);
            }
            for (uint16_t j = 0; j < static_cast<uint16_t>(blockCount); j++) {
                Reg::LoadAlign(xRegTensor, xAddr + j * dataLen + i * vfLen);
                if constexpr (IsSame<T, half>::value) {
                    Reg::Cast<float, T, castTraitZero>(yZero, xRegTensor, p0);
                    Reg::Cast<float, T, castTraitOne>(yOne, xRegTensor, p0);
                    Reg::Mul(yZero, yZero, reversedShareExpRegTensorFP32Zero, maskAll);
                    Reg::Mul(yOne, yOne, reversedShareExpRegTensorFP32One, maskAll);
                } else {
                    Reg::Mul(valueRegTensor, xRegTensor,
                        (Reg::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, p0);
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
                auto addr0 = yAddr + (j * outDataLenAlign + i * vfLen);
                Reg::StoreUnAlign(addr0, outZero, u0, loopNum0);
                Reg::StoreUnAlignPost(addr0, u0, 0);
                auto addr1 = yAddr + (j * outDataLenAlign + i * vfLen) + loopNum0;
                Reg::StoreUnAlign(addr1, outOne, u0, loopNum1);
                Reg::StoreUnAlignPost(addr1, u0, 0);
            }
        }

        if (tailVfLen != 0) {
            uint32_t tailPnum = tailVfLen;
            Reg::MaskReg p1 = Reg::UpdateMask<T>(tailPnum);
            for (uint16_t k = 0; k <= static_cast<uint16_t>(blockCountLoop); k++) {
                Reg::LoadAlign(xRegTensor0, xAddr + k * dataLen + regLoop * vfLen);
                Reg::LoadAlign(xRegTensor1, xAddr + (static_cast<uint16_t>(blockCount) - k - 1) * dataLen + regLoop * vfLen);
                if constexpr (IsSame<T, half>::value) {
                    Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(xBF16RegTensor0, xRegTensor0, p1);
                    Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(xBF16RegTensor1, xRegTensor1, p1);
                    Reg::And(expMaxRegTensor0, (Reg::RegTensor<uint16_t>&)xBF16RegTensor0, maxEleRegTensor, p1);
                    Reg::And(expMaxRegTensor1, (Reg::RegTensor<uint16_t>&)xBF16RegTensor1, maxEleRegTensor, p1);
                } else {
                    Reg::And(expMaxRegTensor0, (Reg::RegTensor<uint16_t>&)xRegTensor0, maxEleRegTensor, p1);
                    Reg::And(expMaxRegTensor1, (Reg::RegTensor<uint16_t>&)xRegTensor1, maxEleRegTensor, p1);
                }   
                Reg::Max(expRegTensor, expMaxRegTensor0, expMaxRegTensor1, p1);
                Reg::Max(expMaxRegTensor, expRegTensor, expMaxRegTensor, p1);
            }
            Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxRegTensor, maxEleRegTensor, p1);
            Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxRegTensor, zeroRegTensor, p1);
            Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMaxRegTensor, fp8MaxExpRegTensor, p1);
            Reg::Select<uint16_t>(expMaxRegTensor, fp8MaxExpRegTensor, expMaxRegTensor, invalidDataMask);
            Reg::Sub(expMaxRegTensor, expMaxRegTensor, fp8MaxExpRegTensor, p1);
            Reg::ShiftRights(mxScaleRegTensor, expMaxRegTensor, SHR_NUM_FOR_BF16, p1);
            Reg::Select<uint16_t>(mxScaleRegTensor, mxScaleRegTensor, fp8NanRegTensor, infMask);
            Reg::Select<uint16_t>(mxScaleRegTensor, mxScaleRegTensor, zeroRegTensor, zeroMask);
            Reg::Pack(mxScale, mxScaleRegTensor);
            Reg::StoreUnAlign(mxScaleAddr, mxScale, u0, tailVfLen);
            Reg::StoreUnAlignPost(mxScaleAddr, u0, 0);

            // 求1/scale
            Reg::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, expMaxRegTensor, biasRegTensor, p1);
            Reg::Sub(reversedShareExpRegTensor, biasRegTensor, expMaxRegTensor, p1);
            Reg::Select<uint16_t>(reversedShareExpRegTensor, specialExpRegTensor,
                reversedShareExpRegTensor, specialDataMask);
            Reg::Select<uint16_t>(reversedShareExpRegTensor, reversedShareExpRegTensor, nanRegTensor,
                infMask);
            Reg::Select<uint16_t>(reversedShareExpRegTensor, reversedShareExpRegTensor, zeroRegTensor,
                zeroMask);

            // 求data value
            if constexpr (IsSame<T, half>::value) {
                Reg::Cast<float, bfloat16_t, castTraitZero>(reversedShareExpRegTensorFP32Zero,
                    (Reg::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, p1);
                Reg::Cast<float, bfloat16_t, castTraitOne>(reversedShareExpRegTensorFP32One,
                    (Reg::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, p1);
            }
            for (uint16_t j = 0; j < static_cast<uint16_t>(blockCount); j++) {
                Reg::LoadAlign(xRegTensor, xAddr + regLoop * vfLen + j * dataLen);
                if constexpr (IsSame<T, half>::value) {
                    Reg::Cast<float, T, castTraitZero>(yZero, xRegTensor, p1);
                    Reg::Cast<float, T, castTraitOne>(yOne, xRegTensor, p1);
                    Reg::Mul(yZero, yZero, reversedShareExpRegTensorFP32Zero, maskAll);
                    Reg::Mul(yOne, yOne, reversedShareExpRegTensorFP32One, maskAll);
                } else {
                    Reg::Mul(valueRegTensor, xRegTensor,
                        (Reg::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, p1);
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
                auto addr0 = yAddr + (regLoop * vfLen + j * outDataLenAlign);
                Reg::StoreUnAlign(addr0, outZero, u0, tailLoopNum0);
                Reg::StoreUnAlignPost(addr0, u0, 0);
                auto addr1 = yAddr + (regLoop * vfLen + j * outDataLenAlign) + tailLoopNum0;
                Reg::StoreUnAlign(addr1, outOne, u0, tailLoopNum1);
                Reg::StoreUnAlignPost(addr1, u0, 0);
            }
        }
    }
}

template <typename T, typename U>
__aicore__ inline void GroupedDynamicMxQuantBaseFP8<T, U>::ComputecuBLAS(int64_t dataLen, int64_t blockCount,
    __ubuf__ T *xAddr, __ubuf__ uint8_t *mxScaleAddr, __ubuf__ uint8_t *yAddr)
{
    constexpr uint32_t vfNum16 = Ops::Base::GetVRegSize() / sizeof(T);
    constexpr uint32_t vfNum32 = Ops::Base::GetVRegSize() / sizeof(float);
    uint16_t regLoop = static_cast<uint16_t>(dataLen) / static_cast<uint16_t>(vfNum32);
    uint16_t tailVfLen = static_cast<uint16_t>(dataLen) % static_cast<uint16_t>(vfNum32);
    uint32_t singleLoopNum = dataLen <= vfNum32 ? vfNum16 : vfNum32;
    uint32_t singleTailLoopNum = tailVfLen;
    int64_t outDataLenAlign = (dataLen + 32 - 1) / 32 * 32;
    float inv_dtype_max = 0;
    int64_t blockCountLoop = blockCount / 2;
    if constexpr (IsSame<U, fp8_e4m3fn_t>::value) {
        inv_dtype_max = FP8_E4M3_INV_MAX;   // 1.0f / 448.0f
    } else if constexpr (IsSame<U, fp8_e5m2_t>::value) {
        inv_dtype_max = FP8_E5M2_INV_MAX;   // 1.0f / 57344.0f
    }

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> xRegTensor;
        Reg::RegTensor<T> xRegTensor0;
        Reg::RegTensor<T> xRegTensor1;
        Reg::RegTensor<uint16_t> absMaskBF16Reg;
        Reg::RegTensor<uint16_t> fullZeroReg;
        Reg::RegTensor<uint32_t> zeroRegTensor32;
        Reg::RegTensor<uint32_t> manMaskFP32Reg;
        Reg::RegTensor<uint32_t> fp8NanRegTensor32;
        Reg::RegTensor<uint32_t> biasRegTensor32;
        Reg::RegTensor<uint32_t> nanRegTensor32;
        Reg::RegTensor<uint16_t> expScaleMxYZeroRegTensor;
        Reg::RegTensor<uint16_t> expMaxRegTensor;
        Reg::RegTensor<uint16_t> expMaxRegTensor0;
        Reg::RegTensor<uint16_t> expMaxRegTensor1;
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

        static constexpr Reg::CastTrait castTraitZero = {
            Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitOne = {
            Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTrait32to8 = {
            Reg::RegLayout::ZERO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {
            Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
        static constexpr Reg::CastTrait castTraitHalf2Float = {
            Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        Reg::Duplicate(absMaskBF16Reg, ABS_FOR_UINT16);
        Reg::Duplicate(zeroRegTensor32, 0);
        Reg::Duplicate(manMaskFP32Reg, MAN_FOR_FP32);
        Reg::Duplicate(fp8NanRegTensor32, MAX_EXP_FOR_FP8_IN_FP32);
        Reg::Duplicate(biasRegTensor32, FP32_EXP_BIAS_CUBLAS);
        Reg::Duplicate(nanRegTensor32, NAN_CUSTOMIZATION_PACK);
        Reg::Duplicate(fullZeroReg, 0);

        uint32_t pnum = vfNum32;
        p0 = Reg::UpdateMask<T>(pnum);
        for (uint16_t i = 0; i < regLoop; i++) {
            Reg::Duplicate(expMaxreShareExpFP16YOneRegTensor, 0);
            for (uint16_t j = 0; j <= static_cast<uint16_t>(blockCountLoop); j++) {
                Reg::LoadAlign(xRegTensor0, xAddr + j * dataLen + i * vfNum32);
                Reg::LoadAlign(xRegTensor1, xAddr + (static_cast<uint16_t>(blockCount) - j - 1) * dataLen + i * vfNum32);
                Reg::And(expMaxRegTensor0, (Reg::RegTensor<uint16_t>&)xRegTensor0, absMaskBF16Reg, p0);
                Reg::And(expMaxRegTensor1, (Reg::RegTensor<uint16_t>&)xRegTensor1, absMaskBF16Reg, p0);
                Reg::Max(expMaxRegTensor, expMaxRegTensor0, expMaxRegTensor1, p0);
                Reg::Max(expMaxreShareExpFP16YOneRegTensor, expMaxRegTensor, expMaxreShareExpFP16YOneRegTensor, p0);
            }
            // 2. BF16→FP32
            Reg::Interleave(expMaxreShareExpFP16YOneRegTensor, fullZeroReg,
                expMaxreShareExpFP16YOneRegTensor, fullZeroReg);
            Reg::Cast<float, T, castTraitHalf2Float>(
                (Reg::RegTensor<float>&)expMaxAndAddOneFP32RegTensor,
                (Reg::RegTensor<T>&)expMaxreShareExpFP16YOneRegTensor, preMaskScale);
            // 3. Inf/Zero校验
            Reg::CompareScalar<uint32_t, CMPMODE::LT>(
                infMask, expMaxAndAddOneFP32RegTensor, MAX_EXP_FOR_FP32, preMaskScale);
            Reg::Compare<uint32_t, CMPMODE::NE>(
                zeroMask, expMaxAndAddOneFP32RegTensor, zeroRegTensor32, preMaskScale);
            // 4. S_fp32 = Amax / Amax(DType)
            Reg::Muls((Reg::RegTensor<float>&)expMaxAndAddOneFP32RegTensor,
                (Reg::RegTensor<float>&)expMaxAndAddOneFP32RegTensor, inv_dtype_max, preMaskScale);
            // 5. 提取 Exp 和 Man
            Reg::ShiftRights(expAndreShareExpFP32RegTensor,
                expMaxAndAddOneFP32RegTensor, SHR_NUM_FOR_FP32, preMaskScale);
            Reg::And(manAndmxScaleFP32RegTensor,
                expMaxAndAddOneFP32RegTensor, manMaskFP32Reg, preMaskScale);
            // 6. 条件向上取整
            Reg::CompareScalar<uint32_t, CMPMODE::GT>(
                p1, expAndreShareExpFP32RegTensor, NUMBER_ZERO, preMaskScale);
            Reg::CompareScalar<uint32_t, CMPMODE::LT>(
                p2, expAndreShareExpFP32RegTensor, NUMBER_TWO_FIVE_FOUR, preMaskScale);
            Reg::CompareScalar<uint32_t, CMPMODE::GT>(
                p3, manAndmxScaleFP32RegTensor, NUMBER_ZERO, preMaskScale);
            Reg::MaskAnd(p1, p1, p2, preMaskScale);
            Reg::MaskAnd(p1, p1, p3, preMaskScale);
            Reg::CompareScalar<uint32_t, CMPMODE::EQ>(
                p2, expAndreShareExpFP32RegTensor, NUMBER_ZERO, preMaskScale);
            Reg::CompareScalar<uint32_t, CMPMODE::GT>(
                p3, manAndmxScaleFP32RegTensor, NUMBER_HALF, preMaskScale);
            Reg::MaskAnd(p2, p2, p3, preMaskScale);
            Reg::MaskXor(p1, p1, p2, preMaskScale);
            Reg::Adds(expMaxAndAddOneFP32RegTensor, expAndreShareExpFP32RegTensor, 1, preMaskScale);
            Reg::Select(manAndmxScaleFP32RegTensor,
                expMaxAndAddOneFP32RegTensor, expAndreShareExpFP32RegTensor, p1);
            // 7. 异常值处理
            Reg::Select<uint32_t>(manAndmxScaleFP32RegTensor,
                manAndmxScaleFP32RegTensor, fp8NanRegTensor32, infMask);
            Reg::Select<uint32_t>(manAndmxScaleFP32RegTensor,
                manAndmxScaleFP32RegTensor, zeroRegTensor32, zeroMask);
            // 8. 输出 mxscale（E8M0）
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(
                expScaleMxYZeroRegTensor, manAndmxScaleFP32RegTensor);
            Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(
                outZeromxScaleFp8, expScaleMxYZeroRegTensor);
            Reg::StoreUnAlign(mxScaleAddr, outZeromxScaleFp8, u1, vfNum32);
            Reg::StoreUnAlignPost(mxScaleAddr, u1, 0);
            // 9. 求 1/scale
            Reg::ShiftLefts(manAndmxScaleFP32RegTensor,
                manAndmxScaleFP32RegTensor, SHR_NUM_FOR_BF16, preMaskScale);
            Reg::Sub(expAndreShareExpFP32RegTensor,
                biasRegTensor32, manAndmxScaleFP32RegTensor, preMaskScale);
            Reg::Select<uint32_t>(expAndreShareExpFP32RegTensor,
                expAndreShareExpFP32RegTensor, nanRegTensor32, infMask);
            Reg::Select<uint32_t>(expAndreShareExpFP32RegTensor,
                expAndreShareExpFP32RegTensor, zeroRegTensor32, zeroMask);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(
                expMaxreShareExpFP16YOneRegTensor, expAndreShareExpFP32RegTensor);
            // 10. 量化
            if constexpr (IsSame<T, half>::value) {
                Reg::Cast<float, bfloat16_t, castTraitZero>(reversedShareExpRegTensorFP32Zero,
                    (Reg::RegTensor<bfloat16_t>&)expMaxreShareExpFP16YOneRegTensor, p0);
                Reg::Cast<float, bfloat16_t, castTraitOne>(reversedShareExpRegTensorFP32One,
                    (Reg::RegTensor<bfloat16_t>&)expMaxreShareExpFP16YOneRegTensor, p0);
            }
            for (uint16_t j = 0; j < static_cast<uint16_t>(blockCount); j++) {
                Reg::LoadAlign(xRegTensor, xAddr + j * dataLen + i * vfNum32);
                if constexpr (IsSame<T, half>::value) {
                    Reg::Cast<float, T, castTraitZero>(yZero, xRegTensor, p0);
                    Reg::Cast<float, T, castTraitOne>(yOne, xRegTensor, p0);
                    Reg::Mul(yZero, yZero, reversedShareExpRegTensorFP32Zero, maskAll);
                    Reg::Mul(yOne, yOne, reversedShareExpRegTensorFP32One, maskAll);
                } else {
                    Reg::Mul(valueRegTensor, xRegTensor,
                        (Reg::RegTensor<bfloat16_t>&)expMaxreShareExpFP16YOneRegTensor, p0);
                    Reg::Cast<float, bfloat16_t, castTraitZero>(yZero, valueRegTensor, maskAll);
                    Reg::Cast<float, bfloat16_t, castTraitOne>(yOne, valueRegTensor, maskAll);
                }
                Reg::Interleave(yZero, yOne, yZero, yOne);
                Reg::Cast<U, float, castTrait32to8>(yZeroFP8, yZero, maskAll);
                Reg::Pack(expScaleMxYZeroRegTensor, (Reg::RegTensor<uint32_t>&)yZeroFP8);
                Reg::Pack(outZeromxScaleFp8, expScaleMxYZeroRegTensor);
                auto addr = yAddr + (j * outDataLenAlign + i * vfNum32);
                Reg::StoreUnAlign(addr, outZeromxScaleFp8, u1, singleLoopNum);
                Reg::StoreUnAlignPost(addr, u1, 0);
            }
        }

        if (tailVfLen != 0) {
            uint32_t tailPnum = tailVfLen;
            p0 = Reg::UpdateMask<T>(tailPnum);
            Reg::Duplicate(expMaxreShareExpFP16YOneRegTensor, 0);
            for (uint16_t k = 0; k <= static_cast<uint16_t>(blockCountLoop); k++) {
                Reg::LoadAlign(xRegTensor0, xAddr + k * dataLen + regLoop * vfNum32);
                Reg::LoadAlign(xRegTensor1, xAddr + (static_cast<uint16_t>(blockCount) - k - 1) * dataLen + regLoop * vfNum32);
                Reg::And(expMaxRegTensor0, (Reg::RegTensor<uint16_t>&)xRegTensor0, absMaskBF16Reg, p0);
                Reg::And(expMaxRegTensor1, (Reg::RegTensor<uint16_t>&)xRegTensor1, absMaskBF16Reg, p0);
                Reg::Max(expMaxRegTensor, expMaxRegTensor0, expMaxRegTensor1, p0);
                Reg::Max(expMaxreShareExpFP16YOneRegTensor, expMaxRegTensor, expMaxreShareExpFP16YOneRegTensor, p0);
            }
            // 2. BF16→FP32
            Reg::Interleave(expMaxreShareExpFP16YOneRegTensor, fullZeroReg,
                expMaxreShareExpFP16YOneRegTensor, fullZeroReg);
            Reg::Cast<float, T, castTraitHalf2Float>(
                (Reg::RegTensor<float>&)expMaxAndAddOneFP32RegTensor,
                (Reg::RegTensor<T>&)expMaxreShareExpFP16YOneRegTensor, preMaskScale);
            // 3. Inf/Zero校验
            Reg::CompareScalar<uint32_t, CMPMODE::LT>(
                infMask, expMaxAndAddOneFP32RegTensor, MAX_EXP_FOR_FP32, preMaskScale);
            Reg::Compare<uint32_t, CMPMODE::NE>(
                zeroMask, expMaxAndAddOneFP32RegTensor, zeroRegTensor32, preMaskScale);
            // 4. S_fp32 = Amax / Amax(DType)
            Reg::Muls((Reg::RegTensor<float>&)expMaxAndAddOneFP32RegTensor,
                (Reg::RegTensor<float>&)expMaxAndAddOneFP32RegTensor, inv_dtype_max, preMaskScale);
            // 5. 提取 Exp 和 Man
            Reg::ShiftRights(expAndreShareExpFP32RegTensor,
                expMaxAndAddOneFP32RegTensor, SHR_NUM_FOR_FP32, preMaskScale);
            Reg::And(manAndmxScaleFP32RegTensor,
                expMaxAndAddOneFP32RegTensor, manMaskFP32Reg, preMaskScale);
            // 6. 条件向上取整
            Reg::CompareScalar<uint32_t, CMPMODE::GT>(
                p1, expAndreShareExpFP32RegTensor, NUMBER_ZERO, preMaskScale);
            Reg::CompareScalar<uint32_t, CMPMODE::LT>(
                p2, expAndreShareExpFP32RegTensor, NUMBER_TWO_FIVE_FOUR, preMaskScale);
            Reg::CompareScalar<uint32_t, CMPMODE::GT>(
                p3, manAndmxScaleFP32RegTensor, NUMBER_ZERO, preMaskScale);
            Reg::MaskAnd(p1, p1, p2, preMaskScale);
            Reg::MaskAnd(p1, p1, p3, preMaskScale);
            Reg::CompareScalar<uint32_t, CMPMODE::EQ>(
                p2, expAndreShareExpFP32RegTensor, NUMBER_ZERO, preMaskScale);
            Reg::CompareScalar<uint32_t, CMPMODE::GT>(
                p3, manAndmxScaleFP32RegTensor, NUMBER_HALF, preMaskScale);
            Reg::MaskAnd(p2, p2, p3, preMaskScale);
            Reg::MaskXor(p1, p1, p2, preMaskScale);
            Reg::Adds(expMaxAndAddOneFP32RegTensor, expAndreShareExpFP32RegTensor, 1, preMaskScale);
            Reg::Select(manAndmxScaleFP32RegTensor,
                expMaxAndAddOneFP32RegTensor, expAndreShareExpFP32RegTensor, p1);
            // 7. 异常值处理
            Reg::Select<uint32_t>(manAndmxScaleFP32RegTensor,
                manAndmxScaleFP32RegTensor, fp8NanRegTensor32, infMask);
            Reg::Select<uint32_t>(manAndmxScaleFP32RegTensor,
                manAndmxScaleFP32RegTensor, zeroRegTensor32, zeroMask);
            // 8. 输出 mxscale（E8M0）
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(
                expScaleMxYZeroRegTensor, manAndmxScaleFP32RegTensor);
            Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(
                outZeromxScaleFp8, expScaleMxYZeroRegTensor);
            Reg::StoreUnAlign(mxScaleAddr, outZeromxScaleFp8, u1, tailVfLen);
            Reg::StoreUnAlignPost(mxScaleAddr, u1, 0);
            // 9. 求 1/scale
            Reg::ShiftLefts(manAndmxScaleFP32RegTensor,
                manAndmxScaleFP32RegTensor, SHR_NUM_FOR_BF16, preMaskScale);
            Reg::Sub(expAndreShareExpFP32RegTensor,
                biasRegTensor32, manAndmxScaleFP32RegTensor, preMaskScale);
            Reg::Select<uint32_t>(expAndreShareExpFP32RegTensor,
                expAndreShareExpFP32RegTensor, nanRegTensor32, infMask);
            Reg::Select<uint32_t>(expAndreShareExpFP32RegTensor,
                expAndreShareExpFP32RegTensor, zeroRegTensor32, zeroMask);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(
                expMaxreShareExpFP16YOneRegTensor, expAndreShareExpFP32RegTensor);
            // 10. 量化
            if constexpr (IsSame<T, half>::value) {
                Reg::Cast<float, bfloat16_t, castTraitZero>(reversedShareExpRegTensorFP32Zero,
                    (Reg::RegTensor<bfloat16_t>&)expMaxreShareExpFP16YOneRegTensor, p0);
                Reg::Cast<float, bfloat16_t, castTraitOne>(reversedShareExpRegTensorFP32One,
                    (Reg::RegTensor<bfloat16_t>&)expMaxreShareExpFP16YOneRegTensor, p0);
            }
            for (uint16_t j = 0; j < static_cast<uint16_t>(blockCount); j++) {
                Reg::LoadAlign(xRegTensor, xAddr + regLoop * vfNum32 + j * dataLen);
                if constexpr (IsSame<T, half>::value) {
                    Reg::Cast<float, T, castTraitZero>(yZero, xRegTensor, p0);
                    Reg::Cast<float, T, castTraitOne>(yOne, xRegTensor, p0);
                    Reg::Mul(yZero, yZero, reversedShareExpRegTensorFP32Zero, maskAll);
                    Reg::Mul(yOne, yOne, reversedShareExpRegTensorFP32One, maskAll);
                } else {
                    Reg::Mul(valueRegTensor, xRegTensor,
                        (Reg::RegTensor<bfloat16_t>&)expMaxreShareExpFP16YOneRegTensor, p0);
                    Reg::Cast<float, bfloat16_t, castTraitZero>(yZero, valueRegTensor, maskAll);
                    Reg::Cast<float, bfloat16_t, castTraitOne>(yOne, valueRegTensor, maskAll);
                }
                Reg::Interleave(yZero, yOne, yZero, yOne);
                Reg::Cast<U, float, castTrait32to8>(yZeroFP8, yZero, maskAll);
                Reg::Pack(expScaleMxYZeroRegTensor, (Reg::RegTensor<uint32_t>&)yZeroFP8);
                Reg::Pack(outZeromxScaleFp8, expScaleMxYZeroRegTensor);
                auto addr = yAddr + (regLoop * vfNum32 + j * outDataLenAlign);
                Reg::StoreUnAlign(addr, outZeromxScaleFp8, u1, singleTailLoopNum);
                Reg::StoreUnAlignPost(addr, u1, 0);
            }
        }
    }
}

} // namespace GroupedDynamicMxQuant
#endif // GROUPED_DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_FP8_H
