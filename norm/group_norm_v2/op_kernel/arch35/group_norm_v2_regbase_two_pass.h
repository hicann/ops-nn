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

#ifndef GROUP_NORM_V2_REGBASE_TWO_PASS_H_
#define GROUP_NORM_V2_REGBASE_TWO_PASS_H_

#include "group_norm_v2_regbase_base.h"

namespace GroupNormV2 {
using namespace AscendC;
template <typename T1, typename T2, int32_t BUFFER_NUM = 2>
class GroupNormV2TwoPass {
public:
  __aicore__ inline GroupNormV2TwoPass(){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd,
                              GM_ADDR workspace, const GroupNormV2TilingData* tilingData) {
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

  __aicore__ inline void Process() {
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
      uint64_t gmOffset = blockIdx * tiling->numPerCore + i * innerNumPerCore;
      ProcessMeanAndRstd<T1>(meanTensor, meanOutTensor, meanGm, rstdTensor, rstdOutTensor, rstdGm, gmOffset,
                             numPerCoreOneLoop);
      if (i < numPerCoreLoop - 1) {
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
      }
    }
  }

private:
  __aicore__ inline void CalNormalize(uint64_t offset, uint32_t numPerCoreTmp) {
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
      CalMeanAndRstd<T1>(xLocal, meanLocal, rstdLocal, dichotomyAddLocal, numPerCoreProcess, dichotomyAddPower,
                         dichotomyAddK, dichotomyAddLastNum, elemNum, reduceScale, eps);
      NormalizeAndSwish(xUbOffset, offset + i * onceNumPerCore, numPerCoreProcess, i);
      if (i < numPerCoreExtent - BUFFER_NUM) {
        SetFlag<HardEvent::V_MTE2>(isPing ? eventIDVToMte2Ping : eventIDVToMte2Pong);
      }
      SetFlag<HardEvent::V_MTE3>(isPing ? eventIDVToMte3Ping : eventIDVToMte3Pong);
      WaitFlag<HardEvent::V_MTE3>(isPing ? eventIDVToMte3Ping : eventIDVToMte3Pong);
      CopyY2Gm<T1>(yGm[xGmOffset], yTensor[xUbOffset], numPerCoreProcess, elemNum);
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

  __aicore__ inline void NormalizeAndSwish(uint32_t xUbOffset, uint32_t numPerCoreoffset, int64_t numPerCoreProcess,
                                           uint32_t numPerCoreLoop) {
    __local_mem__ T1* xLocal = (__local_mem__ T1*)xTensor[xUbOffset].GetPhyAddr();
    __local_mem__ T1* yOutLocal = (__local_mem__ T1*)yTensor[xUbOffset].GetPhyAddr();
    for (int64_t i = 0; i < numPerCoreProcess; i++) {
      uint64_t gammaOffset = ((blockIdx * tiling->numPerCore + numPerCoreoffset + i) % numGroups) * shapeD;
      uint64_t betaOffset = gammaOffset;
      __local_mem__ T1* xLocal = (__local_mem__ T1*)xTensor[xUbOffset + i * elemNumAlign].GetPhyAddr();
      __local_mem__ T1* yOutLocal = (__local_mem__ T1*)yTensor[xUbOffset + i * elemNumAlign].GetPhyAddr();
      __local_mem__ T2* gammaLocal = hasGamma ? (__local_mem__ T2*)gammaTensor[gammaOffset].GetPhyAddr() : nullptr;
      __local_mem__ T2* betaLocal = hasBeta ? (__local_mem__ T2*)betaTensor[betaOffset].GetPhyAddr() : nullptr;
      __local_mem__ float* meanLocal =
          (__local_mem__ float*)meanTensor[numPerCoreLoop * onceNumPerCore + i].GetPhyAddr();
      __local_mem__ float* rstdLocal =
          (__local_mem__ float*)rstdTensor[numPerCoreLoop * onceNumPerCore + i].GetPhyAddr();
      VFNormalizeUnAlign<T1, T2>(xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, yOutLocal, shapeD, hwNum);
    }
  }

  __aicore__ inline void ProcessGammaAndBeta(uint64_t curNumPerCore) {
    if (curNumPerCore != 0) {
      return;
    }
    CopyGammaAndBeta2UB<T2>(gammaGm, betaGm, gammaTensor, betaTensor, 1, shapeC);
    auto eventIDMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    SetFlag<HardEvent::MTE2_V>(eventIDMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMte2ToV);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIDMte2ToV);
  }

  __aicore__ inline void ParseTilingData() {
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
    hwNum = tiling->hwNum;
    processSize = tiling->processSize;
    elemNumAlign = RoundUp<T1>(elemNum);
    hwNumAlign = RoundUp<T1>(hwNum);
    onceNumPerCore = processSize / elemNumAlign;
    reduceScale = (float)1.0 / static_cast<float>(elemNum);
    dichotomyAddPower = tiling->dichotomyAddPower;
    dichotomyAddK = tiling->dichotomyAddK;
    dichotomyAddLastNum = tiling->dichotomyAddLastNum;
  }

  __aicore__ inline void InitInnerBuffer() {
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
    if constexpr (IsSameType<T1, half>::value || IsSameType<T1, bfloat16_t>::value) {
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

    if constexpr (IsSameType<T1, half>::value || IsSameType<T1, bfloat16_t>::value) {
      meanOutTensor = ubTensor[curOffset];
      curOffset += meanOutSize;
      rstdOutTensor = ubTensor[curOffset];
      curOffset += rstdOutSize;
    }

    if (hasGamma) {
      int32_t gammaSize = RoundUp<T1>(shapeC * (sizeof(T2) / sizeof(T1)));
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
  // input GM tensors
  GlobalTensor<T1> xGm;
  GlobalTensor<T2> gammaGm;
  GlobalTensor<T2> betaGm;

  // output GM tensors
  GlobalTensor<T1> yGm;
  GlobalTensor<T1> meanGm;
  GlobalTensor<T1> rstdGm;

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
  float eps;
  float reduceScale;
  bool activateSilu{true};

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

}  // namespace GroupNormV2

#endif
