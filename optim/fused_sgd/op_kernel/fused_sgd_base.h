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
 * \file fused_sgd_base.h
 * \brief
 */

#ifndef FUSED_SGD_BASE_H
#define FUSED_SGD_BASE_H

#include "kernel_operator.h"
#include "fused_sgd_tiling_data.h"

namespace FusedSgd {
using namespace AscendC;
constexpr int32_t BYTE_ONE_BLOCK = 32;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t INDEX_PARAMS = 0;
constexpr int32_t INDEX_GRADS = 1;
constexpr int32_t INDEX_MOMENTUM_BUFFER = 2;

template <typename T>
class FusedSgdBase
{
public:
    __aicore__ inline FusedSgdBase(){};
    __aicore__ inline void InitData(const FusedSgdTilingData& tiling);
    __aicore__ inline void PipeSync();

protected:
    float weightDecay;
    float momentum;
    float lr;
    float dampening;
    uint64_t nesterov;
    uint64_t maximize;
    uint64_t isFirstStep;
    uint64_t useGradScale;
    uint64_t useMomentum;
    uint64_t tensorNum;
    uint64_t tensorsPerCore;
    uint64_t usedCoreNum;
    uint64_t coreCalcMax;
};

template <typename T>
__aicore__ inline void FusedSgdBase<T>::InitData(const FusedSgdTilingData& tiling)
{
    weightDecay = tiling.weightDecay;
    momentum = tiling.momentum;
    lr = tiling.lr;
    dampening = tiling.dampening;
    nesterov = tiling.nesterov;
    maximize = tiling.maximize;
    isFirstStep = tiling.isFirstStep;
    useGradScale = tiling.useGradScale;
    useMomentum = tiling.useMomentum;
    tensorNum = tiling.tensorNum;
    tensorsPerCore = tiling.tensorsPerCore;
    usedCoreNum = tiling.usedCoreNum;
    coreCalcMax = tiling.coreCalcMax;
}

template <AscendC::HardEvent hardEvent>
__aicore__ inline void PipeSync()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(hardEvent));
    AscendC::SetFlag<hardEvent>(eventID);
    AscendC::WaitFlag<hardEvent>(eventID);
}

} // namespace FusedSgd

#endif // FUSED_SGD_BASE_H