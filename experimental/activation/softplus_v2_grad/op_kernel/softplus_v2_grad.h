/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SOFTPLUS_BACKWARD_KERNEL_H
#define SOFTPLUS_BACKWARD_KERNEL_H

#include <cmath>

#include "kernel_operator.h"
#include "softplus_v2_grad_tiling_data.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t BLOCK_SIZE = 512; // 改为512字节对齐

template <typename T>
class KernelSoftplusV2Grad {
public:
    __aicore__ inline KernelSoftplusV2Grad() {}
    __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR gradOutput, GM_ADDR self, GM_ADDR gradInput, GM_ADDR workspace,
                                const SoftplusV2GradTilingData& tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> gradOutputQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> selfQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> gradInputQueue;

    TBuf<QuePosition::VECCALC> mainComputeBuf;
    TBuf<QuePosition::VECCALC> intermediateBuf;
    TBuf<QuePosition::VECCALC> maskBuf;
    TBuf<QuePosition::VECCALC> tempBuf1;
    TBuf<QuePosition::VECCALC> tempBuf2;

    GlobalTensor<T> gradOutputGm;
    GlobalTensor<T> selfGm;
    GlobalTensor<T> gradInputGm;

    uint64_t typeLength;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
    uint64_t processDataNum_computes;
    uint64_t tailprocessDataNum_computes;
    uint64_t currentProcessDataNumComputes;
    float betaVal;
    float thresholdVal;
};

extern "C" __global__ __aicore__ void softplus_v2_grad(GM_ADDR gradOutput, GM_ADDR self, GM_ADDR gradInput,
                                                       GM_ADDR workspace, GM_ADDR tiling);

#endif // SOFTPLUS_BACKWARD_KERNEL_H
