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
 * \file fused_adamw_base.h
 * \brief
 */

#ifndef FUSED_ADAMW_BASE_H
#define FUSED_ADAMW_BASE_H

#include "kernel_operator.h"
#include "fused_adamw_tiling_data.h"

namespace FusedAdamW {
using namespace AscendC;
constexpr int32_t BYTE_ONE_BLOCK = 32;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t INDEX_PARAMS = 0;
constexpr int32_t INDEX_GRADS = 1;
constexpr int32_t INDEX_EXP_AVG = 2;
constexpr int32_t INDEX_EXP_AVG_SQ = 3;
constexpr int32_t INDEX_MAX_EXP_AVG_SQ = 4;
constexpr int32_t INDEX_STEP = 5;
constexpr int32_t TENSOR_COUNT_NO_AMSGRAD = 5;
constexpr int32_t TENSOR_COUNT_AMSGRAD = 6;
constexpr int32_t TENSOR_COUNT_OUT_NO_AMSGRAD = 3;
constexpr int32_t TENSOR_COUNT_OUT_AMSGRAD = 4;
constexpr int32_t BLOCK_SIZE_FOR_FLOAT32 = 8;

template <typename T>
class FusedAdamWBase {
public:
    __aicore__ inline FusedAdamWBase(){};
    __aicore__ inline void InitData(const FusedAdamWTilingData& tiling);

protected:
    float lr{0.0f};
    float beta1{0.0f};
    float beta2{0.0f};
    float weightDecay{0.0f};
    float eps{0.0f};
    uint64_t amsgrad{0};
    uint64_t maximize{0};
    uint64_t useGradScale{0};
    uint64_t useFoundInf{0};
    uint64_t tensorNum{0};
    uint64_t tensorsPerCore{0};
    uint64_t usedCoreNum{0};
    uint64_t coreCalcMax{0};
    uint64_t usedRealCoreNum{0};
    uint64_t lastCoreTensor{0};
    float stepCount{0.0f};
};

template <typename T>
__aicore__ inline void FusedAdamWBase<T>::InitData(const FusedAdamWTilingData& tiling)
{
    lr = tiling.lr;
    beta1 = tiling.beta1;
    beta2 = tiling.beta2;
    weightDecay = tiling.weightDecay;
    eps = tiling.eps;
    amsgrad = tiling.amsgrad;
    maximize = tiling.maximize;
    useGradScale = tiling.useGradScale;
    useFoundInf = tiling.useFoundInf;
    tensorNum = tiling.tensorNum;
    tensorsPerCore = tiling.tensorsPerCore;
    usedCoreNum = tiling.usedCoreNum;
    coreCalcMax = tiling.coreCalcMax;
    usedRealCoreNum = tiling.usedRealCoreNum;
    lastCoreTensor = tiling.lastCoreTensor;
    stepCount = tiling.stepCount;
}

template <AscendC::HardEvent hardEvent>
__aicore__ inline void PipeSync()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(hardEvent));
    AscendC::SetFlag<hardEvent>(eventID);
    AscendC::WaitFlag<hardEvent>(eventID);
}

} // namespace FusedAdamW

#endif // FUSED_ADAMW_BASE_H_
