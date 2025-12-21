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
 * \file group_norm_silu_regbase_two_pass.h
 * \brief
 */

#ifndef GROUP_NORM_SILU_REGBASE_TWO_PASS_H_
#define GROUP_NORM_SILU_REGBASE_TWO_PASS_H_

#include "group_norm_silu_regbase_base.h"

namespace GroupNormSilu {
using namespace AscendC;
template <typename T1, typename T2, int32_t BUFFER_NUM = 2>
class GroupNormSiluTwoPass
{
public:
    __aicore__ inline GroupNormSiluTwoPass(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR silu, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace,
        const GroupNormSiluRegbaseTilingData* tilingData)
    {
        tiling = tilingData;
        blockIdx = GetBlockIdx();
        blockNum = GetBlockNum();
        xGm.SetGlobalBuffer((__gm__ T1*)x);
        if (gamma != nullptr) {
            hasGamma = true;
            gammaGm.SetGlobalBuffer((__gm__ T2*)gamma);
        }
        if (beta != nullptr) {
            hasBeta = true;
            betaGm.SetGlobalBuffer((__gm__ T2*)beta);
        }
        siluGm.SetGlobalBuffer((__gm__ T1*)silu);
        if constexpr (sizeof(T2) == sizeof(float)) {
            meanT2Gm.SetGlobalBuffer((__gm__ T2*)mean);
            rstdT2Gm.SetGlobalBuffer((__gm__ T2*)rstd);
        } else {
            meanGm.SetGlobalBuffer((__gm__ T1*)mean);
            rstdGm.SetGlobalBuffer((__gm__ T1*)rstd);
        }
        ParseTilingData();
        InitInnerBuffer();
    }

    __aicore__ inline void Process()
    {
        uint32_t numPerCoreLoop = CeilDiv(numPerCore, innerNumPerCore);
        uint32_t numPerCoreTail = numPerCore % innerNumPerCore == 0 ? innerNumPerCore : numPerCore % innerNumPerCore;
        uint32_t numPerCoreOneLoop = innerNumPerCore;
        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        for (uint64_t i = 0; i < numPerCoreLoop; i++) {
            if (i > 0) {
                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            }
            if (i == numPerCoreLoop - 1) {
                numPerCoreOneLoop = numPerCoreTail;
            }
            CalNormalize(i * innerNumPerCore, numPerCoreOneLoop);
            uint64_t gmOffset = blockIdx * tiling->numPerCore + i * innerNumPerCore;
            if constexpr (sizeof(T2) == sizeof(float)) {
                LocalTensor<T2> meanOutTensorNew = meanOutTensor.template ReinterpretCast<T2>();
                LocalTensor<T2> rstdOutTensorNew = rstdOutTensor.template ReinterpretCast<T2>();
                ProcessMeanAndRstd<T2>(
                    meanTensor, meanOutTensorNew, meanT2Gm, rstdTensor, rstdOutTensorNew, rstdT2Gm, gmOffset,
                    numPerCoreOneLoop);
            } else {
                ProcessMeanAndRstd<T1>(
                    meanTensor, meanOutTensor, meanGm, rstdTensor, rstdOutTensor, rstdGm, gmOffset, numPerCoreOneLoop);
            }
            if (i < numPerCoreLoop - 1) {
                SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            }
        }
    }

private:
    __aicore__ inline void CalNormalize(uint64_t offset, uint32_t numPerCoreTmp)
    {
        int64_t numPerCoreExtent = CeilDiv(numPerCoreTmp, onceNumPerCore);
        uint32_t numPerCoreTail = numPerCoreTmp % onceNumPerCore == 0 ? onceNumPerCore : numPerCoreTmp % onceNumPerCore;
        uint32_t numPerCoreProcess = onceNumPerCore;
        uint64_t xGmBaseOffset = blockIdx * tiling->numPerCore * elemNum + offset * elemNum;
        ProcessGammaAndBeta(offset);
        auto eventIDMte2ToVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        auto eventIDMte2ToVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        auto eventIDVToMte3Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        auto eventIDVToMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        auto eventIDVToMte2Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        auto eventIDVToMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        auto eventIDMte3ToVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        auto eventIDMte3ToVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        __local_mem__ float* dichotomyAddLocal = (__local_mem__ float*)dichotomyAddTensor.GetPhyAddr();
        for (int64_t i = 0; i < numPerCoreExtent; i++) {
            if (i == numPerCoreExtent - 1) {
                numPerCoreProcess = numPerCoreTail;
            }
            bool isPing = (i % BUFFER_NUM) == 0;
            if (i > 1) {
                WaitFlag<HardEvent::V_MTE2>(isPing ? eventIDVToMte2Ping : eventIDVToMte2Pong);
            }
            uint32_t xGmOffset = xGmBaseOffset + i * onceNumPerCore * elemNum;
            uint32_t xUbOffset = isPing * processSize;
            CopyX2UB<T1>(xGm[xGmOffset], xTensor[xUbOffset], numPerCoreProcess, elemNum);
            SetFlag<HardEvent::MTE2_V>(isPing ? eventIDMte2ToVPing : eventIDMte2ToVPong);
            WaitFlag<HardEvent::MTE2_V>(isPing ? eventIDMte2ToVPing : eventIDMte2ToVPong);
            __local_mem__ T1* xLocal = (__local_mem__ T1*)xTensor[xUbOffset].GetPhyAddr();
            __local_mem__ float* meanLocal = (__local_mem__ float*)meanTensor[onceNumPerCore * i].GetPhyAddr();
            __local_mem__ float* rstdLocal = (__local_mem__ float*)rstdTensor[onceNumPerCore * i].GetPhyAddr();
            if (i > 1) {
                WaitFlag<HardEvent::MTE3_V>(isPing ? eventIDMte3ToVPing : eventIDMte3ToVPong);
            }
            CalMeanAndRstd<T1>(
                xLocal, meanLocal, rstdLocal, dichotomyAddLocal, numPerCoreProcess, dichotomyAddPower, dichotomyAddK,
                dichotomyAddLastNum, powerOfTwoForReduce, elemNum, eps);
            NormalizeAndSwish(xUbOffset, offset + i * onceNumPerCore, numPerCoreProcess, i);
            if (i < numPerCoreExtent - BUFFER_NUM) {
                SetFlag<HardEvent::V_MTE2>(isPing ? eventIDVToMte2Ping : eventIDVToMte2Pong);
            }
            SetFlag<HardEvent::V_MTE3>(isPing ? eventIDVToMte3Ping : eventIDVToMte3Pong);
            WaitFlag<HardEvent::V_MTE3>(isPing ? eventIDVToMte3Ping : eventIDVToMte3Pong);
            CopySilu2Gm<T1>(siluGm[xGmOffset], yTensor[xUbOffset], numPerCoreProcess, elemNum);
            if (i < numPerCoreExtent - BUFFER_NUM) {
                SetFlag<HardEvent::MTE3_V>(isPing ? eventIDMte3ToVPing : eventIDMte3ToVPong);
            }
        }
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIDMte2ToVPing);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIDMte2ToVPong);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIDVToMte3Ping);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIDVToMte3Pong);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIDVToMte2Ping);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIDVToMte2Pong);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIDMte3ToVPing);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIDMte3ToVPong);
    }

    __aicore__ inline void NormalizeAndSwish(
        uint32_t xUbOffset, uint32_t numPerCoreoffset, int64_t numPerCoreProcess, uint32_t numPerCoreLoop)
    {
        if (isFold) {
            return NormalizeAndSwishFold(xUbOffset, numPerCoreoffset, numPerCoreProcess, numPerCoreLoop);
        }
        return NormalizeAndSwishCommon(xUbOffset, numPerCoreoffset, numPerCoreProcess, numPerCoreLoop);
    }

    __aicore__ inline void NormalizeAndSwishCommon(
        uint32_t xUbOffset, uint32_t numPerCoreoffset, int64_t numPerCoreProcess, uint32_t numPerCoreLoop)
    {
        __local_mem__ T1* xLocal = (__local_mem__ T1*)xTensor[xUbOffset].GetPhyAddr();
        __local_mem__ T1* yOutLocal = (__local_mem__ T1*)yTensor[xUbOffset].GetPhyAddr();
        for (int64_t i = 0; i < numPerCoreProcess; i++) {
            uint64_t gammaOffset = ((blockIdx * tiling->numPerCore + numPerCoreoffset + i) % numGroups) * shapeD;
            uint64_t betaOffset = gammaOffset;
            __local_mem__ T1* xLocal = (__local_mem__ T1*)xTensor[xUbOffset + i * elemNumAlign].GetPhyAddr();
            __local_mem__ T1* yOutLocal = (__local_mem__ T1*)yTensor[xUbOffset + i * elemNumAlign].GetPhyAddr();
            __local_mem__ T2* gammaLocal =
                hasGamma ? (__local_mem__ T2*)gammaTensor[gammaOffset].GetPhyAddr() : nullptr;
            __local_mem__ T2* betaLocal = hasBeta ? (__local_mem__ T2*)betaTensor[betaOffset].GetPhyAddr() : nullptr;
            __local_mem__ float* meanLocal =
                (__local_mem__ float*)meanTensor[numPerCoreLoop * onceNumPerCore + i].GetPhyAddr();
            __local_mem__ float* rstdLocal =
                (__local_mem__ float*)rstdTensor[numPerCoreLoop * onceNumPerCore + i].GetPhyAddr();
            VFNormalizeAndSwishUnAlign<T1, T2>(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, yOutLocal, shapeD, hwNum, activateSilu, hasGamma,
                hasBeta);
        }
    }

    // HW <= 32场景或者是gamma/beta均为空的场景，对VF做融合，融合后最少一个VF,最多3个VF
    // 假设当前需要处理的A轴范围为:[curNumPerCoreHead, curNumPerCoreTail),那么区间划分存在如下几种情况
    // case1: A轴范围不包含numGroups
    // [curNumPerCoreHead, curNumPerCoreTail)
    // case2: A轴范围包含numGroups
    // [curNumPerCoreHead, n * numGroups) [n * numGroups, m * numGroups) [m * numGroups, curNumPerCoreTail)
    // 其中：n * numGroups = curNumPerCoreHeadUpAlign, m * numGroups = curNumPerCoreTailDownAlign
    // 每个区间对应一个VF
    __aicore__ inline void NormalizeAndSwishFold(
        uint32_t xUbOffset, uint32_t numPerCoreoffset, int64_t numPerCoreProcess, uint32_t numPerCoreLoop)
    {
        uint64_t curNumPerCoreHead = blockIdx * tiling->numPerCore + numPerCoreoffset;
        uint64_t curNumPerCoreTail = curNumPerCoreHead + numPerCoreProcess;
        uint64_t curNumPerCoreHeadUpAlign = CeilDiv(curNumPerCoreHead, numGroups) * numGroups;
        uint64_t curNumPerCoreTailDownAlign = (curNumPerCoreTail / numGroups) * numGroups;
        uint64_t gammaOffset = ((blockIdx * tiling->numPerCore + numPerCoreoffset) % numGroups) * elemNumAlign;
        uint64_t betaOffset = gammaOffset;
        __local_mem__ T1* xLocal = (__local_mem__ T1*)xTensor[xUbOffset].GetPhyAddr();
        __local_mem__ T1* yOutLocal = (__local_mem__ T1*)yTensor[xUbOffset].GetPhyAddr();
        __local_mem__ T2* gammaLocal = hasGamma ? (__local_mem__ T2*)gammaTensor[gammaOffset].GetPhyAddr() : nullptr;
        __local_mem__ T2* betaLocal = hasBeta ? (__local_mem__ T2*)betaTensor[betaOffset].GetPhyAddr() : nullptr;
        __local_mem__ float* meanLocal = (__local_mem__ float*)meanTensor[numPerCoreLoop * onceNumPerCore].GetPhyAddr();
        __local_mem__ float* rstdLocal = (__local_mem__ float*)rstdTensor[numPerCoreLoop * onceNumPerCore].GetPhyAddr();
        // case1
        if (curNumPerCoreTailDownAlign < curNumPerCoreHeadUpAlign) {
            VFNormalizeAndSwishFold(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, yOutLocal, 1, numPerCoreProcess, shapeD * hwNum,
                activateSilu, hasGamma, hasBeta);
            return;
        }

        // case2
        // 区间1：[curNumPerCoreHead, curNumPerCoreHeadUpAlign)
        if (curNumPerCoreHeadUpAlign - curNumPerCoreHead > 0) {
            VFNormalizeAndSwishFold(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, yOutLocal, 1,
                curNumPerCoreHeadUpAlign - curNumPerCoreHead, shapeD * hwNum, activateSilu, hasGamma, hasBeta);
        }
        // 区间2：[curNumPerCoreHeadUpAlign, curNumPerCoreTailDownAlign)
        uint32_t groupsCount = (curNumPerCoreTailDownAlign - curNumPerCoreHeadUpAlign) / numGroups;
        xLocal = xLocal + (curNumPerCoreHeadUpAlign - curNumPerCoreHead) * elemNumAlign;
        yOutLocal = yOutLocal + (curNumPerCoreHeadUpAlign - curNumPerCoreHead) * elemNumAlign;
        meanLocal = meanLocal + (curNumPerCoreHeadUpAlign - curNumPerCoreHead);
        rstdLocal = rstdLocal + (curNumPerCoreHeadUpAlign - curNumPerCoreHead);
        gammaLocal = hasGamma ? (__local_mem__ T2*)gammaTensor.GetPhyAddr() : nullptr;
        betaLocal = hasBeta ? (__local_mem__ T2*)betaTensor.GetPhyAddr() : nullptr;
        if (groupsCount > 0) {
            VFNormalizeAndSwishFold(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, yOutLocal, groupsCount, numGroups, shapeD * hwNum,
                activateSilu, hasGamma, hasBeta);
        }
        // 区间3: [curNumPerCoreTailDownAlign, curNumPerCoreTail)
        if (curNumPerCoreTail - curNumPerCoreTailDownAlign > 0) {
            xLocal = xLocal + groupsCount * numGroups * elemNumAlign;
            yOutLocal = yOutLocal + groupsCount * numGroups * elemNumAlign;
            meanLocal = meanLocal + groupsCount * numGroups;
            rstdLocal = rstdLocal + groupsCount * numGroups;
            VFNormalizeAndSwishFold(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, yOutLocal, 1,
                curNumPerCoreTail - curNumPerCoreTailDownAlign, shapeD * hwNum, activateSilu, hasGamma, hasBeta);
        }
    }

    __aicore__ inline void ProcessGammaAndBeta(uint64_t curNumPerCore)
    {
        if (curNumPerCore != 0) {
            return;
        }
        if (isFold) {
            CopyGammaAndBeta2UBByNDDMA<T2>(
                gammaGm, betaGm, gammaTensor, betaTensor, numGroups, shapeD, hwNum, elemNumAlign, hasGamma, hasBeta);
        } else {
            CopyGammaAndBeta2UB<T2>(gammaGm, betaGm, gammaTensor, betaTensor, 1, shapeC, hasGamma, hasBeta);
        }
        if (hasGamma || hasBeta) {
            auto eventIDMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
            SetFlag<HardEvent::MTE2_V>(eventIDMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMte2ToV);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIDMte2ToV);
        }
    }

    __aicore__ inline void ParseTilingData()
    {
        if (blockIdx == blockNum - 1) {
            numPerCore = tiling->numLastCore;
        } else {
            numPerCore = tiling->numPerCore;
        }
        numGroups = tiling->numGroups;
        ubSize = tiling->ubSize;
        elemNum = tiling->elemNum;
        shapeC = tiling->shapeC;
        shapeD = tiling->shapeD;
        eps = tiling->epsilon;
        activateSilu = tiling->activateSilu;
        hwNum = tiling->hwNum;
        processSize = tiling->processSize;
        elemNumAlign = RoundUp<T1>(elemNum);
        hwNumAlign = RoundUp<T1>(hwNum);
        onceNumPerCore = processSize / elemNumAlign;
        dichotomyAddPower = tiling->dichotomyAddPower;
        dichotomyAddK = tiling->dichotomyAddK;
        dichotomyAddLastNum = tiling->dichotomyAddLastNum;
        powerOfTwoForReduce = tiling->powerOfTwoForReduce;
        isFold = hwNum <= NDDMA_SIZE || (!hasGamma && !hasBeta);
    }

    __aicore__ inline void InitInnerBuffer()
    {
        pipe.InitBuffer(innerBuf, ubSize);
        LocalTensor<T1> ubTensor = innerBuf.template Get<T1>();
        int32_t xSize = processSize * BUFFER_NUM;
        int32_t ySize = processSize * BUFFER_NUM;
        int32_t realNumPerCore = numPerCore > innerNumPerCore ? innerNumPerCore : numPerCore;
        int32_t meanSize = RoundUp<T1>(realNumPerCore * (FLOAT_BYTE_SIZE / sizeof(T1)));
        int32_t rstdSize = RoundUp<T1>(realNumPerCore * (FLOAT_BYTE_SIZE / sizeof(T1)));
        int32_t dichotomySize = RoundUp<T1>((dichotomyAddPower / FP32_ONE_REPEAT) * (FLOAT_BYTE_SIZE / sizeof(T1)));
        int32_t meanOutSize = 0;
        int32_t rstdOutSize = 0;
        if constexpr (IsSameType<T2, half>::value || IsSameType<T2, bfloat16_t>::value) {
            meanOutSize = RoundUp<T1>(realNumPerCore);
            rstdOutSize = RoundUp<T1>(realNumPerCore);
        }

        int32_t yOffset = xSize;
        int32_t meanOffset = yOffset + ySize;
        int32_t rstdOffset = meanOffset + meanSize;
        int32_t dichotomyAddOffset = rstdOffset + rstdSize;

        xTensor = ubTensor;
        yTensor = ubTensor[yOffset];
        meanTensor = ubTensor[meanOffset].template ReinterpretCast<float>();
        rstdTensor = ubTensor[rstdOffset].template ReinterpretCast<float>();
        dichotomyAddTensor = ubTensor[dichotomyAddOffset].template ReinterpretCast<float>();
        int32_t curOffset = dichotomyAddOffset + dichotomySize;

        if constexpr (IsSameType<T2, half>::value || IsSameType<T2, bfloat16_t>::value) {
            meanOutTensor = ubTensor[curOffset];
            curOffset += meanOutSize;
            rstdOutTensor = ubTensor[curOffset];
            curOffset += rstdOutSize;
        }

        if (hasGamma) {
            int32_t gammaSize = isFold ? RoundUp<T1>(static_cast<unsigned long>(numGroups * elemNumAlign) * (sizeof(T2) / sizeof(T1))) :
                                         RoundUp<T1>(static_cast<unsigned long>(shapeC) * (sizeof(T2) / sizeof(T1)));
            gammaTensor = ubTensor[curOffset].template ReinterpretCast<T2>();
            curOffset += gammaSize;
        }
        if (hasBeta) {
            betaTensor = ubTensor[curOffset].template ReinterpretCast<T2>();
        }
    }

private:
    const GroupNormSiluRegbaseTilingData* tiling;
    TPipe pipe;
    // input GM tensors
    GlobalTensor<T1> xGm;
    GlobalTensor<T2> gammaGm;
    GlobalTensor<T2> betaGm;

    // output GM tensors
    GlobalTensor<T1> siluGm;
    GlobalTensor<T1> meanGm;
    GlobalTensor<T1> rstdGm;
    GlobalTensor<T2> meanT2Gm;
    GlobalTensor<T2> rstdT2Gm;

    TBuf<> innerBuf;

    // tiling parameters
    int64_t blockIdx;
    int64_t blockNum;
    int64_t blockElemNum;
    bool hasGamma{false};
    bool hasBeta{false};
    int64_t numPerCore;
    int64_t elemNum;
    int64_t elemNumAlign;
    int64_t ubSize;
    int64_t shapeC;
    int64_t shapeD;
    int64_t hwNum;
    int64_t hwNumAlign;
    int64_t processSize;
    int64_t numGroups;
    int64_t innerNumPerCore{MAX_ONCE_NUM_PER_CORE};
    int64_t onceNumPerCore;
    int64_t dichotomyAddPower;
    int64_t dichotomyAddK;
    int64_t dichotomyAddLastNum;
    int64_t powerOfTwoForReduce;
    float eps;
    bool activateSilu{true};
    bool isFold{false};

    LocalTensor<T1> xTensor;
    LocalTensor<T2> gammaTensor;
    LocalTensor<T2> betaTensor;
    LocalTensor<float> meanTensor;
    LocalTensor<float> rstdTensor;
    LocalTensor<float> dichotomyAddTensor;
    LocalTensor<T1> meanOutTensor;
    LocalTensor<T1> rstdOutTensor;
    // output ub tensor
    LocalTensor<T1> yTensor;
};

} // namespace GroupNormSilu

#endif
