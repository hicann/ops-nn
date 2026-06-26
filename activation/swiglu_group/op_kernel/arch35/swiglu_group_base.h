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
 * \file swiglu_group_base.h
 * \brief
 */

#ifndef SWIGLU_GROUP_BASE_H
#define SWIGLU_GROUP_BASE_H

#include "kernel_operator.h"

namespace SwigluGroup {
using namespace AscendC;
using namespace AscendC::Reg;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t DOUBLE_BUFFER_NUM = 2;
constexpr int32_t VL_FP32 = 64;
constexpr uint32_t REPEAT_SIZE = 256;
constexpr uint16_t FOUR_UNFOLD = 4;

__aicore__ inline int32_t CeilDiv(int32_t a, int b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline int32_t CeilAlign(int32_t a, int b)
{
    return CeilDiv(a, b) * b;
}

template <typename T>
__aicore__ inline int32_t RoundUp(int32_t num)
{
    int32_t elemNum = BLOCK_SIZE / sizeof(T);
    return CeilAlign(num, elemNum);
}

constexpr AscendC::Reg::CastTrait castTraitB162B32Even = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::Reg::CastTrait castTraitB322B16Even = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

template <typename T>
__simd_callee__ inline void LoadInputData(RegTensor<float>& dst, __ubuf__ T* src, MaskReg pregLoop, uint32_t srcOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadAlign(dst, src + srcOffset);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(tmp, src + srcOffset);
        Cast<float, T, castTraitB162B32Even>(dst, tmp, pregLoop);
    }
}

template <typename T>
__simd_callee__ inline void StoreOutputData(
    __ubuf__ T* dst, RegTensor<float>& src, MaskReg pregLoop, uint32_t dstOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        StoreAlign(dst + dstOffset, src, pregLoop);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        Cast<T, float, castTraitB322B16Even>(tmp, src, pregLoop);
        StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(dst + dstOffset, tmp, pregLoop);
    }
}

__simd_callee__ inline void VFSwiGlu(RegTensor<float>& y, RegTensor<float>& x0, RegTensor<float>& x1,
    RegTensor<float>& vreg, MaskReg pregLoop)
{
    Muls(vreg, x0, static_cast<float>(-1.0f), pregLoop);
    Exp(vreg, vreg, pregLoop);
    Adds(vreg, vreg, static_cast<float>(1.0f), pregLoop);
    Div(vreg, x0, vreg, pregLoop);
    Mul(y, vreg, x1, pregLoop);
}

template <typename T, bool hasTopkWeight = false, bool hasClampValue = false, bool singleLoop = false>
__simd_vf__ inline void VFProcessSwigluVf(__ubuf__ T* yLocalAddr, __ubuf__ T* x0LocalAddr,
    __ubuf__ T* x1LocalAddr, __ubuf__ float* topkWeightLocalAddr, uint16_t loopCount,
    uint32_t sregNum, uint32_t curColNumAlign, const uint16_t curRowNum, float clampValue)
{
    RegTensor<float> weight;
    RegTensor<float> x0;
    RegTensor<float> x1;
    RegTensor<float> y;
    RegTensor<float> tmp;
    MaskReg pregLoop = CreateMask<float>();
    if constexpr (singleLoop) {
        uint32_t sreg = sregNum;
        MaskReg pregFixed = UpdateMask<float>(sreg);
        for (uint16_t i = 0; i < curRowNum; i++) {
            if constexpr (hasTopkWeight) {
                LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(weight, topkWeightLocalAddr + i);
            }
            LoadInputData<T>(x0, x0LocalAddr, pregFixed, i * curColNumAlign);
            LoadInputData<T>(x1, x1LocalAddr, pregFixed, i * curColNumAlign);
            if constexpr (hasClampValue) {
                Mins(x0, x0, clampValue, pregFixed);
                Maxs(x1, x1, -clampValue, pregFixed);
                Mins(x1, x1, clampValue, pregFixed);
            }
            VFSwiGlu(y, x0, x1, tmp, pregFixed);
            if constexpr (hasTopkWeight) {
                Mul(y, y, weight, pregFixed);
            }
            StoreOutputData<T>(yLocalAddr, y, pregFixed, i * curColNumAlign);
        }
    } else {
        for (uint16_t i = 0; i < curRowNum; i++) {
            if constexpr (hasTopkWeight) {
                LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(weight, topkWeightLocalAddr + i);
            }
            uint32_t sreg = sregNum;
            for (uint16_t j = 0; j < loopCount; j++) {
                pregLoop = UpdateMask<float>(sreg);
                LoadInputData<T>(x0, x0LocalAddr, pregLoop, j * VL_FP32 + i * curColNumAlign);
                LoadInputData<T>(x1, x1LocalAddr, pregLoop, j * VL_FP32 + i * curColNumAlign);
                if constexpr (hasClampValue) {
                    Mins(x0, x0, clampValue, pregLoop);
                    Maxs(x1, x1, -clampValue, pregLoop);
                    Mins(x1, x1, clampValue, pregLoop);
                }
                VFSwiGlu(y, x0, x1, tmp, pregLoop);
                if constexpr (hasTopkWeight) {
                    Mul(y, y, weight, pregLoop);
                }
                StoreOutputData<T>(yLocalAddr, y, pregLoop, j * VL_FP32 + i * curColNumAlign);
            }
        }
    }
}

template <typename T, bool hasTopkWeight = false, bool hasClampValue = false>
__aicore__ inline void VFProcessSwiglu(
    const LocalTensor<T>& yLocal, const LocalTensor<T>& x0Local, const LocalTensor<T>& x1Local,
    const LocalTensor<float>& topkWeightLocal,
    const uint16_t curRowNum, const uint32_t curColNum, float clampValue)
{
    __ubuf__ T* yLocalAddr = (__ubuf__ T*)yLocal.GetPhyAddr();
    __ubuf__ T* x0LocalAddr = (__ubuf__ T*)x0Local.GetPhyAddr();
    __ubuf__ T* x1LocalAddr = (__ubuf__ T*)x1Local.GetPhyAddr();
    __ubuf__ float* topkWeightLocalAddr =
        hasTopkWeight ? (__ubuf__ float*)topkWeightLocal.GetPhyAddr() : nullptr;
    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint32_t sregNum = curColNum;
    uint32_t curColNumAlign = RoundUp<T>(curColNum);
    if (loopCount == 1) {
        AscendC::VF_CALL<VFProcessSwigluVf<T, hasTopkWeight, hasClampValue, true>>(
            yLocalAddr, x0LocalAddr, x1LocalAddr, topkWeightLocalAddr, loopCount, sregNum, curColNumAlign, curRowNum,
            clampValue);
    } else {
        AscendC::VF_CALL<VFProcessSwigluVf<T, hasTopkWeight, hasClampValue, false>>(
            yLocalAddr, x0LocalAddr, x1LocalAddr, topkWeightLocalAddr, loopCount, sregNum, curColNumAlign, curRowNum,
            clampValue);
    }
}

template <typename T>
__aicore__ inline void SwigluGroupDispatcher(const LocalTensor<T>& yLocal, const LocalTensor<T>& x0Local,
    const LocalTensor<T>& x1Local, const LocalTensor<float>& topkWeightLocal, float clampValue,
    const uint16_t curRowNum, const uint32_t curColNum, int32_t maskBit)
{
    if (maskBit == 0b00) {
        VFProcessSwiglu<T, false, false>(yLocal, x0Local, x1Local, topkWeightLocal, curRowNum, curColNum, clampValue);
    } else if (maskBit == 0b01) {
        VFProcessSwiglu<T, true, false>(yLocal, x0Local, x1Local, topkWeightLocal, curRowNum, curColNum, clampValue);
    } else if (maskBit == 0b10) {
        VFProcessSwiglu<T, false, true>(yLocal, x0Local, x1Local, topkWeightLocal, curRowNum, curColNum, clampValue);
    } else if (maskBit == 0b11) {
        VFProcessSwiglu<T, true, true>(yLocal, x0Local, x1Local, topkWeightLocal, curRowNum, curColNum, clampValue);
    }
}

template <typename T, bool withUbReduce = false>
__simd_vf__ inline void VFProcessGroupIndexSmallVf(
    __ubuf__ T* yLocalAddr, __ubuf__ T* xLocalAddr, uint16_t curColNum, uint16_t vlLen, uint16_t loopCount)
{
    RegTensor<T> x;
    RegTensor<T> sum;
    MaskReg pregMain = CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    MaskReg pregMerge = CreateMask<T, AscendC::Reg::MaskPattern::VL1>();
    Duplicate(sum, static_cast<T>(0), pregMain);
    uint32_t sreg = curColNum;
    MaskReg pregLoop;
    for (uint16_t i = 0; i < loopCount; i++) {
        pregLoop = UpdateMask<T>(sreg);
        LoadAlign(x, xLocalAddr + i * vlLen);
        Adds(x, x, static_cast<T>(0), pregLoop);
        Add(sum, sum, x, pregMain);
    }
    ReduceSum(sum, sum, pregMain);
    if (withUbReduce) {
        RegTensor<T> origin;
        LoadAlign(origin, yLocalAddr);
        Add(sum, sum, origin, pregMerge);
    }
    StoreAlign(yLocalAddr, sum, pregMerge);
}

template <typename T, bool withUbReduce = false>
__simd_vf__ inline void VFProcessGroupIndexLargeVf(__ubuf__ T* yLocalAddr, __ubuf__ T* xLocalAddr, uint16_t vlLen,
    uint16_t fourLoopCount, uint16_t tailLoopNum, uint32_t tailReminder)
{
    RegTensor<T> x0;
    RegTensor<T> x1;
    RegTensor<T> x2;
    RegTensor<T> x3;
    RegTensor<T> sum0;
    RegTensor<T> sum1;
    RegTensor<T> sum2;
    RegTensor<T> sum3;
    MaskReg pregMain = CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    MaskReg pregMerge = CreateMask<T, AscendC::Reg::MaskPattern::VL1>();
    Duplicate(sum0, static_cast<T>(0), pregMain);
    Duplicate(sum1, static_cast<T>(0), pregMain);
    Duplicate(sum2, static_cast<T>(0), pregMain);
    Duplicate(sum3, static_cast<T>(0), pregMain);
    MaskReg pregLoop;
    for (uint16_t i = 0; i < fourLoopCount; i++) {
        LoadAlign(x0, xLocalAddr + i * FOUR_UNFOLD * vlLen);
        Add(sum0, sum0, x0, pregMain);
        LoadAlign(x1, xLocalAddr + (i * FOUR_UNFOLD + 1) * vlLen);
        Add(sum1, sum1, x1, pregMain);
        LoadAlign(x2, xLocalAddr + (i * FOUR_UNFOLD + 2) * vlLen);
        Add(sum2, sum2, x2, pregMain);
        LoadAlign(x3, xLocalAddr + (i * FOUR_UNFOLD + 3) * vlLen);
        Add(sum3, sum3, x3, pregMain);
    }
    uint32_t sreg = tailReminder;
    for (uint16_t i = 0; i < tailLoopNum; i++) {
        pregLoop = UpdateMask<T>(sreg);
        LoadAlign(x0, xLocalAddr + (fourLoopCount * FOUR_UNFOLD + i) * vlLen);
        Adds(x0, x0, static_cast<T>(0), pregLoop);
        Add(sum0, sum0, x0, pregMain);
    }
    Add(sum0, sum0, sum1, pregMain);
    Add(sum2, sum2, sum3, pregMain);
    Add(sum0, sum0, sum2, pregMain);
    ReduceSum(sum0, sum0, pregMain);
    if (withUbReduce) {
        RegTensor<T> origin;
        LoadAlign(origin, yLocalAddr);
        Add(sum0, sum0, origin, pregMerge);
    }
    StoreAlign(yLocalAddr, sum0, pregMerge);
}

template <typename T, bool withUbReduce = false>
__aicore__ inline void VFProcessGroupIndex(const LocalTensor<T>& yLocal, const LocalTensor<T>& xLocal,
    uint16_t curColNum)
{
    __ubuf__ T* yLocalAddr = (__ubuf__ T*)yLocal.GetPhyAddr();
    __ubuf__ T* xLocalAddr = (__ubuf__ T*)xLocal.GetPhyAddr();
    uint16_t vlLen = REPEAT_SIZE / sizeof(T);
    uint16_t loopCount = CeilDiv(curColNum, vlLen);
    uint16_t fullBlocks = curColNum / vlLen;
    uint16_t fourLoopCount = fullBlocks / FOUR_UNFOLD;
    uint16_t tailLoopNum = loopCount - fourLoopCount * FOUR_UNFOLD;
    uint32_t tailReminder = curColNum - fourLoopCount * vlLen * FOUR_UNFOLD;
    if (loopCount < FOUR_UNFOLD) {
        AscendC::VF_CALL<VFProcessGroupIndexSmallVf<T, withUbReduce>>(
            yLocalAddr, xLocalAddr, curColNum, vlLen, loopCount);
    } else {
        AscendC::VF_CALL<VFProcessGroupIndexLargeVf<T, withUbReduce>>(
            yLocalAddr, xLocalAddr, vlLen, fourLoopCount, tailLoopNum, tailReminder);
    }
}

template <typename T>
__aicore__ inline void CopyIn(
    const GlobalTensor<T>& inputGm, const LocalTensor<T>& inputTensor, const uint16_t nBurst, const uint32_t copyLen,
    uint32_t srcStride = 0)
{
    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen * sizeof(T);
    dataCoptExtParams.srcStride = srcStride * sizeof(T);
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(inputTensor, inputGm, dataCoptExtParams, dataCopyPadExtParams);
}

__aicore__ inline void SetDefaultBlockTiling(
    const SwigluGroupTilingData* tilingData, int64_t& usedCoreNums, int64_t& rowOfFormerBlock,
    int64_t& rowOfTailBlock, int64_t& rowLoopOfFormerBlock, int64_t& rowLoopOfTailBlock,
    int64_t& tailRowFactorOfFormerBlock, int64_t& tailRowFactorOfTailBlock)
{
    rowOfFormerBlock = tilingData->rowOfFormerBlock;
    rowOfTailBlock = tilingData->rowOfTailBlock;
    rowLoopOfFormerBlock = tilingData->rowLoopOfFormerBlock;
    rowLoopOfTailBlock = tilingData->rowLoopOfTailBlock;
    tailRowFactorOfFormerBlock = tilingData->tailRowFactorOfFormerBlock;
    tailRowFactorOfTailBlock = tilingData->tailRowFactorOfTailBlock;
    usedCoreNums = GetBlockNum();
}

__aicore__ inline void SetGroupIndexBlockTiling(
    const SwigluGroupTilingData* tilingData, int64_t realBs, int64_t& usedCoreNums,
    int64_t& rowOfFormerBlock, int64_t& rowOfTailBlock, int64_t& rowLoopOfFormerBlock,
    int64_t& rowLoopOfTailBlock, int64_t& tailRowFactorOfFormerBlock, int64_t& tailRowFactorOfTailBlock)
{
    rowOfFormerBlock = CeilDiv(realBs, static_cast<int64_t>(tilingData->coreNum));
    usedCoreNums = CeilDiv(realBs, rowOfFormerBlock) < tilingData->coreNum
        ? CeilDiv(realBs, rowOfFormerBlock)
        : tilingData->coreNum;
    rowOfTailBlock = realBs - (usedCoreNums - 1) * rowOfFormerBlock;

    rowLoopOfFormerBlock = CeilDiv(rowOfFormerBlock, tilingData->rowFactor);
    rowLoopOfTailBlock = CeilDiv(rowOfTailBlock, tilingData->rowFactor);
    tailRowFactorOfFormerBlock = rowOfFormerBlock % tilingData->rowFactor == 0
        ? tilingData->rowFactor
        : rowOfFormerBlock % tilingData->rowFactor;
    tailRowFactorOfTailBlock = rowOfTailBlock % tilingData->rowFactor == 0
        ? tilingData->rowFactor
        : rowOfTailBlock % tilingData->rowFactor;
}

template <typename TBufPoolType>
__aicore__ inline void ProcessGroupIndexTiling(
    GM_ADDR groupIndex, const SwigluGroupTilingData* tilingData, TBufPoolType& tBufPool,
    TQue<QuePosition::VECIN, 1>& groupIndexQue, TBuf<QuePosition::VECCALC>& groupIndexSumBuf,
    GlobalTensor<int64_t>& groupIndexGm, LocalTensor<int64_t>& groupSumLocal, bool& hasGroupIndex,
    int64_t& usedCoreNums, int64_t& rowOfFormerBlock, int64_t& rowOfTailBlock, int64_t& rowLoopOfFormerBlock,
    int64_t& rowLoopOfTailBlock, int64_t& tailRowFactorOfFormerBlock, int64_t& tailRowFactorOfTailBlock)
{
    if (groupIndex == nullptr) {
        SetDefaultBlockTiling(tilingData, usedCoreNums, rowOfFormerBlock, rowOfTailBlock, rowLoopOfFormerBlock,
            rowLoopOfTailBlock, tailRowFactorOfFormerBlock, tailRowFactorOfTailBlock);
        return;
    }

    hasGroupIndex = true;
    groupIndexGm.SetGlobalBuffer((__gm__ int64_t*)groupIndex);
    tBufPool.InitBuffer(groupIndexQue, DOUBLE_BUFFER_NUM, RoundUp<int64_t>(tilingData->gFactor) * sizeof(int64_t));
    tBufPool.InitBuffer(groupIndexSumBuf, BLOCK_SIZE);
    groupSumLocal = groupIndexSumBuf.Get<int64_t>();
    for (int64_t idx = 0; idx < tilingData->gLoop; idx++) {
        int64_t curGFactor = (idx == tilingData->gLoop - 1) ? tilingData->tailGFactor : tilingData->gFactor;
        LocalTensor<int64_t> groupIndexLocal = groupIndexQue.template AllocTensor<int64_t>();
        CopyIn(groupIndexGm[idx * tilingData->gFactor], groupIndexLocal, 1, curGFactor);
        groupIndexQue.template EnQue(groupIndexLocal);
        groupIndexLocal = groupIndexQue.template DeQue<int64_t>();
        if (idx == 0) {
            VFProcessGroupIndex<int64_t, false>(groupSumLocal, groupIndexLocal, curGFactor);
        } else {
            VFProcessGroupIndex<int64_t, true>(groupSumLocal, groupIndexLocal, curGFactor);
        }
        groupIndexQue.template FreeTensor(groupIndexLocal);
    }
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventId);
    WaitFlag<HardEvent::V_S>(eventId);
    int64_t groupSum = groupSumLocal.GetValue(0);
    int64_t realBs = groupSum > tilingData->bs ? tilingData->bs : groupSum;
    SetGroupIndexBlockTiling(tilingData, realBs, usedCoreNums, rowOfFormerBlock, rowOfTailBlock,
        rowLoopOfFormerBlock, rowLoopOfTailBlock, tailRowFactorOfFormerBlock, tailRowFactorOfTailBlock);
    tBufPool.Reset();
}

template <typename T, AscendC::PaddingMode mode = AscendC::PaddingMode::Normal>
__aicore__ inline void CopyOut(
    const LocalTensor<T>& outputTensor, const GlobalTensor<T>& outputGm, const uint16_t nBurst, const uint32_t copyLen,
    uint32_t dstStride = 0)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = nBurst;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = dstStride * sizeof(T);
    DataCopyPad<T, mode>(outputGm, outputTensor, dataCopyParams);
}
} // namespace SwigluGroup

#endif
