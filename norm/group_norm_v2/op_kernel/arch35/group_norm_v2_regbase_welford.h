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
 * \file group_norm_v2_regbase_welford.h
 * \brief
 */
#ifndef GROUP_NORM_V2_REGBASE_WELFORD_H_
#define GROUP_NORM_V2_REGBASE_WELFORD_H_

#include "group_norm_v2_regbase_base.h"
namespace GroupNormV2 {
using namespace AscendC;
template <typename T1, typename T2, int32_t BUFFER_NUM = 2>
class GroupNormV2Welford {
public:
    __aicore__ inline GroupNormV2Welford(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace,
        const GroupNormV2TilingData* tilingData)
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
        yGm.SetGlobalBuffer((__gm__ T1*)y);
        meanGm.SetGlobalBuffer((__gm__ T1*)mean);
        rstdGm.SetGlobalBuffer((__gm__ T1*)rstd);

        ParseTilingData();
        InitInnerBuffer();
    }

    __aicore__ inline void Process()
    {
        uint32_t numPerCoreLoop = CeilDiv(numPerCore, innerNumPerCore);
        uint32_t numPerCoreTail = numPerCore % innerNumPerCore == 0 ? innerNumPerCore : numPerCore % innerNumPerCore;
        uint32_t numPerCoreOneLoop = innerNumPerCore;
        event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        for (uint64_t i = 0; i < numPerCoreLoop; i++) {
            if (i > 0) {
                WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            }
            if (i == numPerCoreLoop - 1) {
                numPerCoreOneLoop = numPerCoreTail;
            }
            CalNormalize(i * innerNumPerCore, numPerCoreOneLoop);
            ProcessMeanAndRstd<T1>(
                meanTensor, meanOutTensor, meanGm, rstdTensor, rstdOutTensor, rstdGm,
                blockIdx * tiling->numPerCore + i * innerNumPerCore, numPerCoreOneLoop);
            if (i < numPerCoreLoop - 1) {
                SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            }
        }
    }

private:
    __aicore__ inline void CalNormalize(uint64_t offset, uint32_t curNumPerCore)
    {
        auto eventIDMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        for (uint64_t i = 0; i < curNumPerCore; i++) {
            if (i > 0) {
                WaitFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
            }
            CalMeanAndRstdByWelford(offset + i, i);
            NormalizeAndSwish(offset + i, i);
            if (i < curNumPerCore - 1) {
                SetFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
            }
        }
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
    }

    __aicore__ inline void CalMeanAndRstdByWelford(uint64_t curNumPerCore, uint64_t curInnerNumPerCore)
    {
        __local_mem__ float* tmpMeanLocal = (__local_mem__ float*)tMeanTensor.GetPhyAddr();
        __local_mem__ float* tmpVarLocal = (__local_mem__ float*)tVarTensor.GetPhyAddr();
        __local_mem__ float* meanLocal = (__local_mem__ float*)meanTensor.GetPhyAddr();
        __local_mem__ float* rstdLocal = (__local_mem__ float*)rstdTensor.GetPhyAddr();
        __local_mem__ float* dichotomyAddLocal = (__local_mem__ float*)dichotomyAddTensor.GetPhyAddr();
        uint64_t xGmOffset = blockIdx * tiling->numPerCore * elemNum;
        uint32_t welfordLen = parallelN;
        count = 0;
        auto eventIDMte2ToVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        auto eventIDMte2ToVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        auto eventIDVToMte2Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        auto eventIDVToMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        for (int64_t i = 0; i < welfordLoopCount; i++) {
            if (i == welfordLoopCount - 1) {
                welfordLen = welfordLoopTail;
            }
            int64_t xPhase1Offset = (i % BUFFER_NUM) * parallelNAlign;
            bool isPing = (i % BUFFER_NUM) == 0;
            if (i > BUFFER_NUM - 1) {
                WaitFlag<HardEvent::V_MTE2>(isPing ? eventIDVToMte2Ping : eventIDVToMte2Pong);
            }
            CopyX2UB<T1>(
                xGm[xGmOffset + curNumPerCore * elemNum + i * parallelN], xPhase1Tensor[xPhase1Offset], 1, welfordLen);
            SetFlag<HardEvent::MTE2_V>(isPing ? eventIDMte2ToVPing : eventIDMte2ToVPong);
            WaitFlag<HardEvent::MTE2_V>(isPing ? eventIDMte2ToVPing : eventIDMte2ToVPong);
            __local_mem__ T1* x1Local = (__local_mem__ T1*)xPhase1Tensor[xPhase1Offset].GetPhyAddr();
            count = count + 1;
            float scale = (float)1.0 / static_cast<float>(count);
            VFWelfordParallelUpdate<T1>(x1Local, tmpMeanLocal, tmpVarLocal, i, welfordLen, scale);
            if (i < welfordLoopCount - BUFFER_NUM) {
                SetFlag<HardEvent::V_MTE2>(isPing ? eventIDVToMte2Ping : eventIDVToMte2Pong);
            }
        }
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIDMte2ToVPing);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIDMte2ToVPong);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIDVToMte2Ping);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIDVToMte2Pong);
        ProcessGammaAndBeta(curNumPerCore);
        float reduceScale = float(1.0) / static_cast<float>(elemNum);
        float scale = float(1.0) / static_cast<float>(parallelN);
        VFWelfordParallelFinalize(
            meanLocal, rstdLocal, tmpMeanLocal, tmpVarLocal, dichotomyAddLocal, parallelN, dichotomyAddPower,
            dichotomyAddK, dichotomyAddLastNum, curInnerNumPerCore, welfordLoopTail, reduceScale, scale, count, eps,
            welfordAlign);
        auto eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
    }
    __aicore__ inline void NormalizeAndSwish(uint64_t curNumPerCore, uint64_t curInnerNumPerCore)
    {
        auto eventIDMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        SetFlag<HardEvent::MTE2_V>(eventIDMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMte2ToV);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIDMte2ToV);
        if (innerLoopNum == 1) {
            NormalizeAndSwishByChannel(curNumPerCore, curInnerNumPerCore);
        } else {
            NormalizeAndSwishByHW(curNumPerCore, curInnerNumPerCore);
        }
    }

    __aicore__ inline void ProcessGammaAndBeta(uint64_t curNumPerCore)
    {
        // 保证normalize阶段MTE2流水在welfordMTE2流水完成之后启动，避免不必要的带宽竞争
        PipeBarrier<PIPE_MTE2>();
        if (curNumPerCore != 0) {
            return;
        }
        CopyGammaAndBeta2UB<T2>(gammaGm, betaGm, gammaTensor, betaTensor, 1, shapeC);
    }

    __aicore__ inline void NormalizeAndSwishByChannel(int64_t curNumPerCore, uint64_t curInnerNumPerCore)
    {
        int64_t inputBaseOffset = blockIdx * tiling->numPerCore * elemNum;
        int64_t outputBaseOffset = blockIdx * tiling->numPerCore * elemNum;
        int64_t hwNumAlign = RoundUp<T1>(hwNum);
        int64_t rowsCount = processSize / hwNumAlign;
        int32_t reduceCount = hwNum;
        uint64_t gammaBaseOffset = ((blockIdx * tiling->numPerCore + curNumPerCore) % numGroups) * shapeD;
        uint64_t betaBaseOffset = gammaBaseOffset;
        // normalize阶段，采用double-buffer流水排布
        // VECTOR流水启动依赖MTE2_V的正向同步和MTE3_V的反向同步
        // MTE2流水启动依赖V_MTE2的反向同步,
        // MTE3流水启动依赖MTE3_V的正向同步
        // 正向同步
        auto eventIDMte2ToVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        auto eventIDMte2ToVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        auto eventIDVToMte3Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        auto eventIDVToMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        // 反向同步
        auto eventIDVToMte2Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        auto eventIDVToMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        auto eventIDMte3ToVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        auto eventIDMte3ToVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        for (int16_t i = 0; i < loopNum; i++) { // for D
            bool isPing = (i % BUFFER_NUM) == 0;
            int64_t inputGmOffset = inputBaseOffset + hwNum * rowsCount * i + elemNum * curNumPerCore;
            int64_t inputUbOffset = isPing ? 0 : processSize;
            if (i == loopNum - 1) {
                rowsCount = loopTail / hwNumAlign;
            }
            if (i > 1) {
                WaitFlag<HardEvent::V_MTE2>(isPing ? eventIDVToMte2Ping : eventIDVToMte2Pong);
            }
            CopyX2UB(xGm[inputGmOffset], xPhase2Tensor[inputUbOffset], rowsCount, hwNum);
            SetFlag<HardEvent::MTE2_V>(isPing ? eventIDMte2ToVPing : eventIDMte2ToVPong);
            WaitFlag<HardEvent::MTE2_V>(isPing ? eventIDMte2ToVPing : eventIDMte2ToVPong);
            uint64_t gammaOffset = gammaBaseOffset + i * (processSize / hwNumAlign);
            uint64_t betaOffset = gammaOffset;
            __local_mem__ T1* xLocal = (__local_mem__ T1*)xPhase2Tensor[inputUbOffset].GetPhyAddr();
            __local_mem__ T2* gammaLocal =
                hasGamma ? (__local_mem__ T2*)gammaTensor[gammaOffset].GetPhyAddr() : nullptr;
            __local_mem__ T2* betaLocal = hasBeta ? (__local_mem__ T2*)betaTensor[betaOffset].GetPhyAddr() : nullptr;
            __local_mem__ float* meanLocal = (__local_mem__ float*)meanTensor[curInnerNumPerCore].GetPhyAddr();
            __local_mem__ float* rstdLocal = (__local_mem__ float*)rstdTensor[curInnerNumPerCore].GetPhyAddr();
            __local_mem__ T1* yOutLocal = (__local_mem__ T1*)yTensor[inputUbOffset].GetPhyAddr();
            if (i > 1) {
                WaitFlag<HardEvent::MTE3_V>(isPing ? eventIDMte3ToVPing : eventIDMte3ToVPong);
            }
            VFNormalizeAlign<T1, T2>(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, yOutLocal, rowsCount, reduceCount);
            SetFlag<HardEvent::V_MTE3>(isPing ? eventIDVToMte3Ping : eventIDVToMte3Pong);
            WaitFlag<HardEvent::V_MTE3>(isPing ? eventIDVToMte3Ping : eventIDVToMte3Pong);
            if (i < loopNum - BUFFER_NUM) {
                SetFlag<HardEvent::V_MTE2>(isPing ? eventIDVToMte2Ping : eventIDVToMte2Pong);
            }
            int32_t outputGmOffset =
                outputBaseOffset + hwNum * (processSize / hwNumAlign) * i + elemNum * curNumPerCore;
            CopyY2Gm<T1>(yGm[outputGmOffset], yTensor[inputUbOffset], rowsCount, hwNum);
            if (i < loopNum - BUFFER_NUM) {
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

    __aicore__ inline void NormalizeAndSwishByHW(int64_t curNumPerCore, uint64_t curInnerNumPerCore)
    {
        uint64_t inputBaseOffset = blockIdx * tiling->numPerCore * elemNum;
        uint64_t outputBaseOffset = blockIdx * tiling->numPerCore * elemNum;
        int64_t totalSize = processSize;
        int64_t tailSize = innerLoopTail;
        uint64_t gammaBaseOffset = ((blockIdx * tiling->numPerCore + curNumPerCore) % numGroups) * shapeD;
        uint64_t betaBaseOffset = gammaBaseOffset;
        // 正向同步
        auto eventIDMte2ToVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        auto eventIDMte2ToVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        auto eventIDVToMte3Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        auto eventIDVToMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        // 反向同步
        auto eventIDVToMte2Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        auto eventIDVToMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        auto eventIDMte3ToVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        auto eventIDMte3ToVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        for (int64_t i = 0; i < loopNum; i++) {
            int64_t copyLen = totalSize;
            uint64_t gammaOffset = gammaBaseOffset + i;
            uint64_t betaOffset = gammaOffset;
            for (int64_t j = 0; j < innerLoopNum; j++) {
                int64_t inputOffset = inputBaseOffset + totalSize * j + hwNum * i + elemNum * curNumPerCore;
                auto extent = i * innerLoopNum + j;
                bool isPing = (extent % BUFFER_NUM) == 0;
                int64_t inputUbOffset = isPing ? 0 : processSize;
                if (j == innerLoopNum - 1) {
                    copyLen = tailSize;
                }
                if (extent > 1) {
                    WaitFlag<HardEvent::V_MTE2>(isPing ? eventIDVToMte2Ping : eventIDVToMte2Pong);
                }
                CopyX2UB(xGm[inputOffset], xPhase2Tensor[inputUbOffset], 1, copyLen);
                SetFlag<HardEvent::MTE2_V>(isPing ? eventIDMte2ToVPing : eventIDMte2ToVPong);
                WaitFlag<HardEvent::MTE2_V>(isPing ? eventIDMte2ToVPing : eventIDMte2ToVPong);
                __local_mem__ T1* xLocal = (__local_mem__ T1*)xPhase2Tensor[inputUbOffset].GetPhyAddr();
                __local_mem__ T2* gammaLocal =
                    hasGamma ? (__local_mem__ T2*)gammaTensor[gammaOffset].GetPhyAddr() : nullptr;
                __local_mem__ T2* betaLocal =
                    hasBeta ? (__local_mem__ T2*)betaTensor[betaOffset].GetPhyAddr() : nullptr;
                __local_mem__ float* meanLocal = (__local_mem__ float*)meanTensor[curInnerNumPerCore].GetPhyAddr();
                __local_mem__ float* rstdLocal = (__local_mem__ float*)rstdTensor[curInnerNumPerCore].GetPhyAddr();
                __local_mem__ T1* yOutLocal = (__local_mem__ T1*)yTensor[inputUbOffset].GetPhyAddr();
                if (extent > 1) {
                    WaitFlag<HardEvent::MTE3_V>(isPing ? eventIDMte3ToVPing : eventIDMte3ToVPong);
                }
                VFNormalizeAlign<T1, T2>(xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, yOutLocal, 1, copyLen);
                SetFlag<HardEvent::V_MTE3>(isPing ? eventIDVToMte3Ping : eventIDVToMte3Pong);
                WaitFlag<HardEvent::V_MTE3>(isPing ? eventIDVToMte3Ping : eventIDVToMte3Pong);
                if (extent < loopNum * innerLoopNum - BUFFER_NUM) {
                    SetFlag<HardEvent::V_MTE2>(isPing ? eventIDVToMte2Ping : eventIDVToMte2Pong);
                }
                uint64_t outputGmOffset = outputBaseOffset + totalSize * j + hwNum * i + elemNum * curNumPerCore;
                int32_t outputLen = copyLen;
                CopyY2Gm<T1>(yGm[outputGmOffset], yTensor[inputUbOffset], 1, copyLen);
                if (extent < loopNum * innerLoopNum - BUFFER_NUM) {
                    SetFlag<HardEvent::MTE3_V>(isPing ? eventIDMte3ToVPing : eventIDMte3ToVPong);
                }
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

    __aicore__ inline void ParseTilingData()
    {
        if (blockIdx == blockNum - 1) {
            numPerCore = tiling->numLastCore;
        } else {
            numPerCore = tiling->numPerCore;
        }
        numGroups = tiling->numGroups;
        ubSize = tiling->ubSize;
        ubElemNum = ubSize / sizeof(T1);
        elemNum = tiling->elemNum;
        parallelN = tiling->parallelN;
        parallelNAlign = RoundUp<T1>(parallelN);
        shapeC = tiling->shapeC;
        shapeD = tiling->shapeD;
        eps = tiling->epsilon;
        hwNum = tiling->hwNum;
        loopNum = tiling->loopNum;
        loopTail = tiling->loopTail;
        processSize = tiling->processSize;
        innerLoopNum = tiling->innerLoopNum;
        innerLoopTail = tiling->innerLoopTail;
        welfordLoopCount = CeilDiv(elemNum, parallelN);
        welfordAlign = elemNum % parallelN == 0;
        welfordLoopTail = welfordAlign ? parallelN : elemNum - parallelN * (welfordLoopCount - 1);
        dichotomyAddPower = tiling->dichotomyAddPower;
        dichotomyAddK = tiling->dichotomyAddK;
        dichotomyAddLastNum = tiling->dichotomyAddLastNum;
    }

    __aicore__ inline void InitInnerBuffer()
    {
        pipe.InitBuffer(innerBuf, ubSize);
        LocalTensor<T1> ubTensor = innerBuf.template Get<T1>();

        int32_t xPhase1Size = parallelNAlign * BUFFER_NUM;
        int32_t tMeanSize = RoundUp<T1>(parallelNAlign * (FLOAT_BYTE_SIZE / sizeof(T1)));
        int32_t tVarSize = RoundUp<T1>(parallelNAlign * (FLOAT_BYTE_SIZE / sizeof(T1)));
        int32_t tMeanOffset = xPhase1Size;
        int32_t tVarOffset = tMeanOffset + tMeanSize;
        int32_t dichotomyAddOffset = tVarOffset + tVarSize;
        int32_t realNumPerCore = numPerCore > innerNumPerCore ? innerNumPerCore : numPerCore;
        int32_t meanSize = RoundUp<T1>(realNumPerCore * (FLOAT_BYTE_SIZE / sizeof(T1)));
        int32_t rstdSize = RoundUp<T1>(realNumPerCore * (FLOAT_BYTE_SIZE / sizeof(T1)));
        int32_t gammaSize = hasGamma ? RoundUp<T1>(shapeC * (sizeof(T2) / sizeof(T1))) : 0;
        int32_t betaSize = hasBeta ? RoundUp<T1>(shapeC * (sizeof(T2) / sizeof(T1))) : 0;
        int32_t meanOutSize = 0;
        int32_t rstdOutSize = 0;
        if constexpr (IsSameType<T1, half>::value || IsSameType<T1, bfloat16_t>::value) {
            meanOutSize = RoundUp<T1>(realNumPerCore);
            rstdOutSize = RoundUp<T1>(realNumPerCore);
        }
        int32_t meanOffset = ubElemNum - meanSize - rstdSize - gammaSize - betaSize - meanOutSize - rstdOutSize;
        int32_t rstdOffset = meanOffset + meanSize;

        xPhase1Tensor = ubTensor;
        tMeanTensor = ubTensor[tMeanOffset].template ReinterpretCast<float>();
        tVarTensor = ubTensor[tVarOffset].template ReinterpretCast<float>();
        dichotomyAddTensor = ubTensor[dichotomyAddOffset].template ReinterpretCast<float>();
        meanTensor = ubTensor[meanOffset].template ReinterpretCast<float>();
        rstdTensor = ubTensor[rstdOffset].template ReinterpretCast<float>();

        int32_t xPhase2Size = processSize * BUFFER_NUM;
        int32_t yOutSize = processSize * BUFFER_NUM;
        xPhase2Tensor = ubTensor;
        yTensor = ubTensor[xPhase2Size];
        int32_t curOffset = rstdOffset + rstdSize;

        if constexpr (IsSameType<T1, half>::value || IsSameType<T1, bfloat16_t>::value) {
            meanOutTensor = ubTensor[curOffset];
            curOffset += meanOutSize;
            rstdOutTensor = ubTensor[curOffset];
            curOffset += rstdOutSize;
        }

        if (hasGamma) {
            gammaTensor = ubTensor[curOffset].template ReinterpretCast<T2>();
            curOffset += gammaSize;
        }
        if (hasBeta) {
            betaTensor = ubTensor[curOffset].template ReinterpretCast<T2>();
        }
    }

private:
    const GroupNormV2TilingData* tiling;
    TPipe pipe;

    GlobalTensor<T1> xGm;
    GlobalTensor<T2> gammaGm;
    GlobalTensor<T2> betaGm;

    GlobalTensor<T1> yGm;
    GlobalTensor<T1> meanGm;
    GlobalTensor<T1> rstdGm;

    TBuf<> innerBuf;

    int64_t blockIdx;
    int64_t blockNum;
    bool hasGamma{false};
    bool hasBeta{false};
    int64_t numPerCore;
    int64_t numGroups;
    int64_t elemNum;
    int64_t ubSize;
    int64_t parallelN;
    int64_t parallelNAlign;
    int64_t welfordLoopCount;
    int64_t welfordLoopTail;
    bool welfordAlign{false};
    int64_t shapeC;
    int64_t shapeD;
    int64_t ubElemNum;
    int64_t hwNum;
    int64_t loopNum;
    int64_t loopTail;
    int64_t processSize;
    int64_t innerLoopNum;
    int64_t innerLoopTail;
    int64_t innerNumPerCore{MAX_ONCE_NUM_PER_CORE};
    int64_t dichotomyAddPower;
    int64_t dichotomyAddK;
    int64_t dichotomyAddLastNum;
    float eps;

    int64_t count{0};

    LocalTensor<T1> xPhase1Tensor;
    LocalTensor<T1> xPhase2Tensor;
    LocalTensor<T2> gammaTensor;
    LocalTensor<T2> betaTensor;

    LocalTensor<float> tMeanTensor;
    LocalTensor<float> tVarTensor;
    LocalTensor<float> dichotomyAddTensor;
    LocalTensor<float> meanTensor;
    LocalTensor<float> rstdTensor;
    LocalTensor<T1> meanOutTensor;
    LocalTensor<T1> rstdOutTensor;

    LocalTensor<T1> yTensor;
};
} // namespace GroupNormV2

#endif
