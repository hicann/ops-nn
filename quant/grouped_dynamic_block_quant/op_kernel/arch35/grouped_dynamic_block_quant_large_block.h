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
 * \file grouped_dynamic_block_quant_large_block.h
 * \brief
 */

#ifndef GROUPED_DYNAMIC_BLOCK_QUANT_LARGE_BLOCK_H
#define GROUPED_DYNAMIC_BLOCK_QUANT_LARGE_BLOCK_H
#define FLOAT_OVERFLOW_MODE_CTRL 60
#include "kernel_operator.h"
#include "../inc/platform.h"

namespace GroupedDynamicBlockQuant {
using namespace AscendC;

template <typename T, typename U, int64_t RMode>
class GroupedDynamicBlockQuantLargeBlock {
public:
    __aicore__ inline GroupedDynamicBlockQuantLargeBlock(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR groupList, GM_ADDR yOut, GM_ADDR scaleOut,
        const GroupedDynamicBlockQuantTilingData& tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const GroupedDynamicBlockQuantTilingData& tilingData);
    __aicore__ inline void CopyIn(int64_t offset, int64_t rowNum, int64_t dataLen);
    __aicore__ inline void ComputeXTmpMax(
        int64_t rowNum, int64_t colNum, __local_mem__ T* xLocalAddr, __local_mem__ T* xLocalMaxTmp);
    __aicore__ inline void ComputeScaleVF(__local_mem__ float* scaleLocalTmp, __local_mem__ T* xLocalMaxTmp);
    __aicore__ inline void ComputeOutVF(
        int64_t xTotalNum, __local_mem__ T* xLocalAddr, __local_mem__ float* scaleLocal, __local_mem__ U* outLocal);
    __aicore__ inline void CopyOutScale(int64_t baseScaleOffset);
    __aicore__ inline void CopyOutY(int64_t rowNum, int64_t colNum, int64_t baseYGmOffset);

private:
    int64_t totalCoreNum_ = 0;
    int64_t headCoreNum_ = 0;
    int64_t tailCoreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t vfLen_ = 0;
    int64_t roundMode_ = 0;
    int64_t dstType_ = 0;
    int64_t batchNum_ = 0;
    int64_t blockIdx_ = 0;    // 核ID
    int64_t usedCoreNum_ = 0; // 实际使用的核数
    int64_t blockFactor_ = 0;
    int64_t tailBlockFactor_ = 0;
    int64_t blockLoopOffset_ = 0; // 当前核起始偏移
    int64_t uo_ = 0;              // N轴计算cacheline次数
    int64_t maxUbRow_ = 0;
    int64_t rowNum_ = 0;
    int64_t colNum_ = 0;
    int64_t scaleRowNum_ = 0;
    int64_t scaleColNum_ = 0;
    int64_t groupNum_ = 0;
    int64_t rowBlockSize_ = 0;
    int64_t colBlockSize_ = 0;
    float minScale_ = 0;
    int64_t fullBlockNum = 0;
    int64_t resBlockRowNum = 0;
    int64_t groupBlockNumHeadCore_ = 0;
    int64_t groupBlockNumTailCore_ = 0;
    float fp8MaxExpValue_ = 0.0f;

    uint16_t infValue_ = 0;
    constexpr static int64_t DB_BUFFER = 2;
    constexpr static int64_t DIGIT_ONE = 1;

    constexpr static float FP8_E5M2_MAX_VALUE = 57344.0f;
    constexpr static float FP8_E4M3_MAX_VALUE = 448.0f;
    constexpr static float HIFP8_MAX_VALUE = 32768.0f;

    constexpr static uint16_t MAX_EXP_FOR_BF16 = 0x7fff;
    constexpr static uint16_t FP16_INF_VALUE = 0x7c00;
    constexpr static uint16_t BF16_INF_VALUE = 0x7f80;

    TPipe pipe_;
    TQue<QuePosition::VECIN, DB_BUFFER> inQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> scaleQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue_;
    TBuf<QuePosition::VECCALC> xLocalMaxBuffer_;
    GlobalTensor<T> xGm_;
    GlobalTensor<int32_t> groupListGm_;
    GlobalTensor<U> yOutGm_;
    GlobalTensor<float> scaleOutGm_;

    static constexpr AscendC::MicroAPI::CastTrait castTrait32to8 = []() {
        if constexpr (RMode == 1) {
            return AscendC::MicroAPI::CastTrait{
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        } else if constexpr (RMode == 4) {
            return AscendC::MicroAPI::CastTrait{
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
        } else if constexpr (RMode == 7) {
            return AscendC::MicroAPI::CastTrait{
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_HYBRID};
        }
    }();
    static constexpr AscendC::MicroAPI::CastTrait castTraitT2Float = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr AscendC::MicroAPI::CastTrait castTraitZero = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr AscendC::MicroAPI::CastTrait castTraitOne = {
        AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
};

template <typename T, typename U, int64_t RMode>
__aicore__ inline void GroupedDynamicBlockQuantLargeBlock<T, U, RMode>::ParseTilingData(
    const GroupedDynamicBlockQuantTilingData& tilingData)
{
    totalCoreNum_ = tilingData.totalCoreNum;
    usedCoreNum_ = tilingData.usedCoreNum;
    headCoreNum_ = tilingData.headCoreNum;
    tailCoreNum_ = tilingData.tailCoreNum;
    ubSize_ = tilingData.ubSize;
    vfLen_ = tilingData.vfLen;
    minScale_ = tilingData.minScale;
    roundMode_ = tilingData.roundMode;
    dstType_ = tilingData.dstType;
    rowBlockSize_ = tilingData.rowBlockSize;
    colBlockSize_ = tilingData.colBlockSize;
    batchNum_ = tilingData.batchNum;
    rowNum_ = tilingData.rowNum;
    colNum_ = tilingData.colNum;
    scaleRowNum_ = tilingData.scaleRowNum;
    scaleColNum_ = tilingData.scaleColNum;
    uo_ = tilingData.uo;
    blockFactor_ = tilingData.blockFactor;
    tailBlockFactor_ = tilingData.tailBlockFactor;
    groupNum_ = tilingData.groupNum;
    maxUbRow_ = tilingData.maxUbRow;
    groupBlockNumHeadCore_ = tilingData.groupBlockNumHeadCore;
    groupBlockNumTailCore_ = tilingData.groupBlockNumTailCore;
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void GroupedDynamicBlockQuantLargeBlock<T, U, RMode>::Init(
    GM_ADDR x, GM_ADDR groupList, GM_ADDR yOut, GM_ADDR scaleOut, const GroupedDynamicBlockQuantTilingData& tilingData)
{
#if (__NPU_ARCH__ == 3101)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    ParseTilingData(tilingData);
    blockIdx_ = GetBlockIdx();
    if (this->blockIdx_ >= this->usedCoreNum_) {
        return;
    }

    if constexpr (IsSameType<U, fp8_e5m2_t>::value) {
        fp8MaxExpValue_ = FP8_E5M2_MAX_VALUE;
    } else if constexpr (IsSameType<U, fp8_e4m3fn_t>::value) {
        fp8MaxExpValue_ = FP8_E4M3_MAX_VALUE;
    } else if constexpr (IsSameType<U, hifloat8_t>::value) {
        fp8MaxExpValue_ = HIFP8_MAX_VALUE;
    }

    if constexpr (IsSameType<T, half>::value) {
        infValue_ = FP16_INF_VALUE;
    } else {
        infValue_ = BF16_INF_VALUE;
    }

    this->xGm_.SetGlobalBuffer((__gm__ T*)(x));
    this->groupListGm_.SetGlobalBuffer((__gm__ int32_t*)(groupList));
    this->yOutGm_.SetGlobalBuffer((__gm__ U*)(yOut));
    this->scaleOutGm_.SetGlobalBuffer((__gm__ float*)(scaleOut));

    this->pipe_.InitBuffer(this->inQueue_, DB_BUFFER, colBlockSize_ * maxUbRow_ * sizeof(T));
    this->pipe_.InitBuffer(this->outQueue_, DB_BUFFER, colBlockSize_ * maxUbRow_ * sizeof(uint8_t));
    this->pipe_.InitBuffer(this->scaleQueue_, DB_BUFFER, 32);
    this->pipe_.InitBuffer(xLocalMaxBuffer_, AscendC::VECTOR_REG_WIDTH);
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void GroupedDynamicBlockQuantLargeBlock<T, U, RMode>::CopyIn(
    int64_t offset, int64_t rowNum, int64_t dataLen)
{
    DataCopyExtParams inCopyParams_ = {
        static_cast<uint16_t>(rowNum), static_cast<uint32_t>(dataLen * sizeof(T)),
        static_cast<uint32_t>((colNum_ - dataLen) * sizeof(T)), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<T> padParams_ = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0)};
    LocalTensor<T> x = inQueue_.AllocTensor<T>();

    DataCopyPad<T, PaddingMode::Compact>(x, xGm_[offset], inCopyParams_, padParams_);
    inQueue_.EnQue(x);
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void GroupedDynamicBlockQuantLargeBlock<T, U, RMode>::Process()
{
    if (this->blockIdx_ >= this->usedCoreNum_) {
        return;
    }
    bool isTailBlock = blockIdx_ >= headCoreNum_;
    int64_t batchGroupNum = groupNum_ * uo_;
    int64_t preTotalGroupNum = 0;
    int64_t groupLoopNum = 0;
    if (isTailBlock) {
        preTotalGroupNum = groupBlockNumHeadCore_ * headCoreNum_ + (blockIdx_ - headCoreNum_) * groupBlockNumTailCore_;
        groupLoopNum = this->groupBlockNumTailCore_;
    } else {
        preTotalGroupNum = groupBlockNumHeadCore_ * blockIdx_;
        groupLoopNum = this->groupBlockNumHeadCore_;
    }

    for (int64_t groupLoopIdx = 0; groupLoopIdx < groupLoopNum; groupLoopIdx++) {
        int64_t nowGroupNum = preTotalGroupNum + groupLoopIdx;
        int64_t bIdx = nowGroupNum / batchGroupNum;
        int64_t nowGroupNumInBatch = nowGroupNum % batchGroupNum;

        int64_t nIdx = nowGroupNumInBatch / groupNum_;
        int64_t gIdx = nowGroupNumInBatch % groupNum_;
        int64_t gIdxValueStart = gIdx == 0 ? 0 : groupListGm_.GetValue(gIdx - 1);
        int64_t gIdxValueEnd = groupListGm_.GetValue(gIdx);

        int64_t xBaseGmOffset = bIdx * rowNum_ * colNum_ + nIdx * blockFactor_ + gIdxValueStart * colNum_;
        int64_t scaleBaseGmOffset =
            bIdx * scaleRowNum_ * scaleColNum_ + nIdx + (gIdxValueStart / rowBlockSize_ + gIdx) * scaleColNum_;
        int64_t groupSize = gIdxValueEnd - gIdxValueStart;
        int64_t blockLoopNum = ops::Ceil(groupSize, rowBlockSize_);
        int64_t dataLen = (nIdx == uo_ - 1) ? tailBlockFactor_ : blockFactor_;

        for (int64_t blockLoopIdx = 0; blockLoopIdx < blockLoopNum; blockLoopIdx++) {
            int64_t xBlockGmOffset = xBaseGmOffset + blockLoopIdx * rowBlockSize_ * colNum_;
            int64_t scaleGmOffset = scaleBaseGmOffset + blockLoopIdx * scaleColNum_;
            int64_t blockRowNum =
                (blockLoopIdx == blockLoopNum - 1) ? groupSize - blockLoopIdx * rowBlockSize_ : rowBlockSize_;

            int64_t ubLoopNum = ops::Ceil(blockRowNum, maxUbRow_);
            int64_t xGmOffset = 0;
            int64_t rowNum = 0;

            LocalTensor<T> xLocalMaxTmp = xLocalMaxBuffer_.Get<T>();
            AscendC::Duplicate(xLocalMaxTmp, static_cast<T>(0), xLocalMaxTmp.GetSize());
            __local_mem__ T* xLocalMaxTmpAddr = (__local_mem__ T*)xLocalMaxTmp.GetPhyAddr();

            LocalTensor<T> inLocal;
            __local_mem__ T* xLocalAddr;

            // compute Max(abs(x))
            for (int64_t ubLoopNumIdx = ubLoopNum - 1; ubLoopNumIdx >= 0; ubLoopNumIdx--) {
                xGmOffset = xBlockGmOffset + ubLoopNumIdx * maxUbRow_ * colNum_;
                rowNum = (ubLoopNumIdx == ubLoopNum - 1) ? blockRowNum - ubLoopNumIdx * maxUbRow_ : maxUbRow_;
                CopyIn(xGmOffset, rowNum, dataLen);
                inLocal = inQueue_.DeQue<T>();
                xLocalAddr = (__local_mem__ T*)inLocal.GetPhyAddr();
                ComputeXTmpMax(rowNum, dataLen, xLocalAddr, xLocalMaxTmpAddr);
                if (ubLoopNumIdx != 0) {
                    inQueue_.FreeTensor(inLocal);
                }
            }

            AscendC::ReduceMax<uint16_t>(
                (AscendC::LocalTensor<uint16_t>&)xLocalMaxTmp, (AscendC::LocalTensor<uint16_t>&)xLocalMaxTmp,
                (AscendC::LocalTensor<uint16_t>&)xLocalMaxTmp, xLocalMaxTmp.GetSize());
            LocalTensor<float> scaleLocal = scaleQueue_.AllocTensor<float>();
            __local_mem__ float* scaleLocalAddr = (__local_mem__ float*)scaleLocal.GetPhyAddr();
            ComputeScaleVF(scaleLocalAddr, xLocalMaxTmpAddr);

            LocalTensor<U> outLocal = outQueue_.AllocTensor<U>();
            __local_mem__ U* outLocalAddr = (__local_mem__ U*)outLocal.GetPhyAddr();

            // 分段计算Y
            ComputeOutVF(rowNum * dataLen, xLocalAddr, scaleLocalAddr, outLocalAddr);

            inQueue_.FreeTensor(inLocal);
            outQueue_.EnQue(outLocal);
            CopyOutY(rowNum, dataLen, xGmOffset);

            for (int64_t ubLoopNumIdx = ubLoopNum - 1; ubLoopNumIdx > 0; ubLoopNumIdx--) {
                xGmOffset = xBlockGmOffset + ubLoopNumIdx * maxUbRow_ * colNum_;
                rowNum = (ubLoopNumIdx == ubLoopNum - 1) ? blockRowNum - ubLoopNumIdx * maxUbRow_ : maxUbRow_;
                CopyIn(xGmOffset, rowNum, dataLen);

                inLocal = inQueue_.DeQue<T>();
                xLocalAddr = (__local_mem__ T*)inLocal.GetPhyAddr();

                outLocal = outQueue_.AllocTensor<U>();
                outLocalAddr = (__local_mem__ U*)outLocal.GetPhyAddr();
                ComputeOutVF(rowNum * dataLen, xLocalAddr, scaleLocalAddr, outLocalAddr);

                inQueue_.FreeTensor(inLocal);
                outQueue_.EnQue(outLocal);
                CopyOutY(rowNum, dataLen, xGmOffset);
            }

            scaleQueue_.EnQue(scaleLocal);
            CopyOutScale(scaleGmOffset);
        }
    }
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void GroupedDynamicBlockQuantLargeBlock<T, U, RMode>::ComputeXTmpMax(
    int64_t rowNum, int64_t colNum, __local_mem__ T* xLocalAddr, __local_mem__ T* xLocalMaxTmp)
{
    uint32_t xTotalNum = rowNum * colNum;
    uint32_t dtypeSize = sizeof(T);
    uint16_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint16_t vfLoop = Ceil(xTotalNum, VL);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> vreg1;
        AscendC::MicroAPI::RegTensor<T> vreg2;
        AscendC::MicroAPI::RegTensor<T> vreg3;
        AscendC::MicroAPI::RegTensor<T> vLocalTmpMaxReg;

        AscendC::MicroAPI::Duplicate((AscendC::MicroAPI::RegTensor<uint16_t>&)vreg2, MAX_EXP_FOR_BF16);
        AscendC::MicroAPI::MaskReg preg0;
        AscendC::MicroAPI::MaskReg preg2;
        AscendC::MicroAPI::MaskReg maskAll =
            AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();

        AscendC::MicroAPI::DataCopy(vLocalTmpMaxReg, xLocalMaxTmp);

        for (uint16_t i = 0; i < vfLoop; i++) {
            preg0 = AscendC::MicroAPI::UpdateMask<T>(xTotalNum);
            AscendC::MicroAPI::DataCopy(vreg1, xLocalAddr + i * VL);
            AscendC::MicroAPI::And(
                (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg3, (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg1,
                (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg2, preg0);
            AscendC::MicroAPI::CompareScalar<uint16_t, CMPMODE::LT>(
                preg2, (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg3, infValue_, preg0);
            AscendC::MicroAPI::Max<T, AscendC::MicroAPI::MaskMergeMode::MERGING>(
                vLocalTmpMaxReg, vLocalTmpMaxReg, vreg3, preg2);
        }
        AscendC::MicroAPI::DataCopy(xLocalMaxTmp, vLocalTmpMaxReg, maskAll);
    }
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void GroupedDynamicBlockQuantLargeBlock<T, U, RMode>::ComputeScaleVF(
    __local_mem__ float* scaleLocalTmp, __local_mem__ T* xLocalMaxTmp)
{
    static constexpr AscendC::MicroAPI::DivSpecificMode mode = {AscendC::MicroAPI::MaskMergeMode::ZEROING, false};
    uint32_t scaleNum = 1;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> vreg1;
        AscendC::MicroAPI::RegTensor<float> vreg2;
        AscendC::MicroAPI::RegTensor<float> vreg3;
        AscendC::MicroAPI::RegTensor<float> reciprocalScale;
        AscendC::MicroAPI::RegTensor<float> minScaleReg;
        AscendC::MicroAPI::RegTensor<float> vreg5;
        AscendC::MicroAPI::RegTensor<float> vreg6;

        AscendC::MicroAPI::MaskReg preg0;

        preg0 = AscendC::MicroAPI::UpdateMask<T>(scaleNum);

        AscendC::MicroAPI::Duplicate(vreg3, fp8MaxExpValue_);
        AscendC::MicroAPI::Duplicate(reciprocalScale, 1.0f);
        AscendC::MicroAPI::Duplicate(minScaleReg, minScale_);
        AscendC::MicroAPI::Div<float, &mode>(reciprocalScale, reciprocalScale, minScaleReg, preg0);

        AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg1, xLocalMaxTmp);
        AscendC::MicroAPI::Cast<float, T, castTraitT2Float>(vreg2, vreg1, preg0);

        AscendC::MicroAPI::Div<float, &mode>(vreg5, vreg2, vreg3, preg0);
        AscendC::MicroAPI::Min(vreg6, vreg5, reciprocalScale, preg0);

        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
            scaleLocalTmp, vreg6, preg0);
    }
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void GroupedDynamicBlockQuantLargeBlock<T, U, RMode>::ComputeOutVF(
    int64_t xTotalNum, __local_mem__ T* xLocalAddr, __local_mem__ float* scaleLocal, __local_mem__ U* outLocal)
{
    uint32_t dtypeSize = sizeof(float);
    uint16_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint16_t vfLoop = (xTotalNum + VL - 1) / VL;
    uint32_t dataNum = xTotalNum;
    static constexpr AscendC::MicroAPI::DivSpecificMode mode = {AscendC::MicroAPI::MaskMergeMode::ZEROING, false};
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> vreg1;
        AscendC::MicroAPI::RegTensor<float> vreg2;
        AscendC::MicroAPI::RegTensor<float> vreg3;
        AscendC::MicroAPI::RegTensor<float> vreg4;
        AscendC::MicroAPI::RegTensor<float> vreg5;
        AscendC::MicroAPI::RegTensor<float> vreg7;
        AscendC::MicroAPI::RegTensor<U> outReg;

        AscendC::MicroAPI::MaskReg preg0;
        AscendC::MicroAPI::MaskReg preg1;

        preg0 = AscendC::MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vreg2, scaleLocal);
        for (uint16_t i = 0; i < static_cast<uint16_t>(vfLoop); i++) {
            preg0 = AscendC::MicroAPI::UpdateMask<float>(dataNum);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg1, xLocalAddr + i * VL);
            AscendC::MicroAPI::Cast<float, T, castTraitT2Float>(vreg4, vreg1, preg0);
            AscendC::MicroAPI::Div<float, &mode>(vreg5, vreg4, vreg2, preg0);

            AscendC::MicroAPI::Muls(vreg3, vreg4, 0.0f, preg0);
            AscendC::MicroAPI::Compare<float, CMPMODE::NE>(preg1, vreg3, vreg3, preg0);
            AscendC::MicroAPI::Select(vreg7, vreg4, vreg5, preg1);

            AscendC::MicroAPI::Cast<U, float, castTrait32to8>(outReg, vreg7, preg0);

            MicroAPI::DataCopy<U, MicroAPI::StoreDist::DIST_PACK4_B32>(outLocal + i * VL, outReg, preg0);
        }
    }
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void GroupedDynamicBlockQuantLargeBlock<T, U, RMode>::CopyOutScale(int64_t baseScaleOffset)
{
    LocalTensor<float> scaleLocal = scaleQueue_.DeQue<float>();

    DataCopyExtParams scaleCopyParams = {1, static_cast<uint32_t>(1 * sizeof(float)), 0, 0, 0};
    DataCopyPad(scaleOutGm_[baseScaleOffset], scaleLocal, scaleCopyParams);
    scaleQueue_.FreeTensor(scaleLocal);
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void GroupedDynamicBlockQuantLargeBlock<T, U, RMode>::CopyOutY(
    int64_t rowNum, int64_t colNum, int64_t baseYGmOffset)
{
    LocalTensor<U> outLocal = outQueue_.DeQue<U>();
    DataCopyExtParams outCopyParams = {
        static_cast<uint16_t>(rowNum), static_cast<uint32_t>(colNum * sizeof(U)), 0,
        static_cast<uint32_t>((colNum_ - colNum) * sizeof(U)), 0};
    DataCopyPad<U, PaddingMode::Compact>(yOutGm_[baseYGmOffset], outLocal, outCopyParams);
    outQueue_.FreeTensor(outLocal);
}
} // namespace GroupedDynamicBlockQuant
#endif // GROUPED_DYNAMIC_BLOCK_QUANT_LARGE_BLOCK_H