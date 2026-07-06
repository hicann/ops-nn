/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "softplus_v2_grad.h"

// 移除kernel侧的对齐函数，因为host侧已处理

template <typename T>
__aicore__ inline void KernelSoftplusV2Grad<T>::Init(TPipe* pipeIn, GM_ADDR gradOutput, GM_ADDR self, GM_ADDR gradInput,
                                                     GM_ADDR workspace, const SoftplusV2GradTilingData& tilingData)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

    this->pipe = pipeIn;
    uint32_t coreIdx = GetBlockIdx();
    this->betaVal = tilingData.beta;
    this->thresholdVal = tilingData.threshold;
    this->typeLength = tilingData.typeLength;

    // 核心数据分配
    if (coreIdx < tilingData.tailBlockNum) {
        this->coreDataNum = tilingData.bigCoreDataNum;
        this->tileNum = tilingData.finalBigTileNum;
        this->tailDataNum = tilingData.bigTailDataNum;
        this->processDataNum_computes = tilingData.bigprocessDataNum_computes;
        this->tailprocessDataNum_computes = tilingData.tailbigprocessDataNum_computes;
    } else {
        this->coreDataNum = tilingData.smallCoreDataNum;
        this->tileNum = tilingData.finalSmallTileNum;
        this->tailDataNum = tilingData.smallTailDataNum;
        this->processDataNum_computes = tilingData.smallprocessDataNum_computes;
        this->tailprocessDataNum_computes = tilingData.tailsmallprocessDataNum_computes;
    }
    this->tileDataNum = tilingData.tileDataNum;

    // 全局偏移量
    uint32_t globalOffset = 0;
    if (coreIdx < tilingData.tailBlockNum) {
        globalOffset = coreIdx * tilingData.bigCoreDataNum;
    } else {
        globalOffset = tilingData.tailBlockNum * tilingData.bigCoreDataNum +
                       (coreIdx - tilingData.tailBlockNum) * tilingData.smallCoreDataNum;
    }

    // 绑定全局缓冲区
    auto gradOutputT = reinterpret_cast<__gm__ T*>(gradOutput);
    auto selfT = reinterpret_cast<__gm__ T*>(self);
    auto gradInputT = reinterpret_cast<__gm__ T*>(gradInput);

    gradOutputGm.SetGlobalBuffer(gradOutputT + globalOffset, this->coreDataNum);
    selfGm.SetGlobalBuffer(selfT + globalOffset, this->coreDataNum);
    gradInputGm.SetGlobalBuffer(gradInputT + globalOffset, this->coreDataNum);

    // 使用512字节对齐（DataCopy）和256字节对齐（计算）
    uint32_t dataCopyAlignment = 512; // DataCopy改为512字节对齐
    uint32_t computeAlignment = 256;

    // 由于host侧已处理对齐，kernel侧直接使用对齐后的长度
    uint32_t bufferSizeCompute = this->processDataNum_computes * sizeof(float);
    uint32_t bufferSizeCopy = this->tileDataNum * typeLength;

    // 确保缓冲区大小满足对齐要求
    bufferSizeCompute = (bufferSizeCompute + computeAlignment - 1) / computeAlignment * computeAlignment;
    bufferSizeCopy = (bufferSizeCopy + dataCopyAlignment - 1) / dataCopyAlignment * dataCopyAlignment;

    // 初始化缓冲区
    pipe->InitBuffer(gradOutputQueue, BUFFER_NUM, bufferSizeCopy);
    pipe->InitBuffer(selfQueue, BUFFER_NUM, bufferSizeCopy);
    pipe->InitBuffer(gradInputQueue, BUFFER_NUM, bufferSizeCopy);

    // 3 --> 申请3个连续长度为bufferSizeCompute 缓冲区
    pipe->InitBuffer(mainComputeBuf, bufferSizeCompute * 3);
    pipe->InitBuffer(intermediateBuf, bufferSizeCompute);
    // 4 --> 除4是因为mask使用int占空间为float的1/4
    pipe->InitBuffer(maskBuf, bufferSizeCompute / 4);
    pipe->InitBuffer(tempBuf1, bufferSizeCompute);
    pipe->InitBuffer(tempBuf2, bufferSizeCompute);
}

template <typename T>
__aicore__ inline void KernelSoftplusV2Grad<T>::Process()
{
    int32_t loopCount = this->tileNum;
    for (int32_t i = 0; i < loopCount - 1; i++) {
        this->processDataNum = this->tileDataNum;
        this->currentProcessDataNumComputes = this->processDataNum_computes;
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
    this->processDataNum = this->tailDataNum;
    this->currentProcessDataNumComputes = this->tailprocessDataNum_computes;
    CopyIn(loopCount - 1);
    Compute(loopCount - 1);
    CopyOut(loopCount - 1);
}

template <typename T>
__aicore__ inline void KernelSoftplusV2Grad<T>::CopyIn(int32_t progress)
{
    // host侧已处理对齐，kernel侧直接使用实际数据长度
    uint32_t copyLength = this->processDataNum;
    uint32_t offset = progress * this->tileDataNum;

    LocalTensor<T> gradOutputLocal = gradOutputQueue.AllocTensor<T>();
    LocalTensor<T> selfLocal = selfQueue.AllocTensor<T>();
    // 直接使用实际长度，host侧已确保对齐
    DataCopy(gradOutputLocal, gradOutputGm[offset], copyLength);
    DataCopy(selfLocal, selfGm[offset], copyLength);
    gradOutputQueue.EnQue(gradOutputLocal);
    selfQueue.EnQue(selfLocal);
}

template <typename T>
__aicore__ inline void KernelSoftplusV2Grad<T>::CopyOut(int32_t progress)
{
    // host侧已处理对齐，kernel侧直接使用实际数据长度
    uint32_t copyLength = this->processDataNum;
    uint32_t offset = progress * this->tileDataNum;

    LocalTensor<T> yLocal = gradInputQueue.DeQue<T>();
    DataCopy(gradInputGm[offset], yLocal, copyLength);
    gradInputQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void KernelSoftplusV2Grad<T>::Compute(int32_t progress)
{
    const float one_val = 1.0f;
    const float neg_one = -1.0f;

    // 使用实际计算长度，host侧已确保对齐
    uint32_t computeLength = this->currentProcessDataNumComputes;

    LocalTensor<float> mainCompute = mainComputeBuf.Get<float>();
    LocalTensor<float> intermediate = intermediateBuf.Get<float>();
    LocalTensor<uint8_t> linearMask = maskBuf.Get<uint8_t>();
    LocalTensor<float> temp1 = tempBuf1.Get<float>();
    LocalTensor<float> temp2 = tempBuf2.Get<float>();

    uint32_t tensorSize = this->currentProcessDataNumComputes;
    LocalTensor<float> gradOutputFloat = mainCompute;
    LocalTensor<float> selfFloat = mainCompute[tensorSize];
    LocalTensor<float> gradInputFloat = mainCompute[tensorSize * 2];

    // 数据类型转换
    LocalTensor<T> gradOutputLocal = gradOutputQueue.DeQue<T>();
    LocalTensor<T> selfLocal = selfQueue.DeQue<T>();

    // 根据模板类型进行不同的转换处理
    if constexpr (std::is_same_v<T, float>) {
        // float类型直接拷贝
        DataCopy(gradOutputFloat, gradOutputLocal, this->processDataNum);
        DataCopy(selfFloat, selfLocal, this->processDataNum);
    } else {
        // half或bfloat16需要类型转换
        Cast(gradOutputFloat, gradOutputLocal, RoundMode::CAST_NONE, this->processDataNum);
        Cast(selfFloat, selfLocal, RoundMode::CAST_NONE, this->processDataNum);
    }

    gradOutputQueue.FreeTensor(gradOutputLocal);
    selfQueue.FreeTensor(selfLocal);

    // 优化后的计算流程
    Muls(intermediate, selfFloat, this->betaVal, computeLength);
    CompareScalar(linearMask, intermediate, this->thresholdVal, CMPMODE::GT, computeLength);

    Muls(intermediate, intermediate, neg_one, computeLength);
    Exp(temp1, intermediate, computeLength);
    Adds(temp2, temp1, one_val, computeLength);

    Duplicate(intermediate, one_val, tensorSize);
    Div(temp1, intermediate, temp2, computeLength);

    Select(intermediate, linearMask, intermediate, temp1, SELMODE::VSEL_TENSOR_TENSOR_MODE, computeLength);

    Mul(gradInputFloat, gradOutputFloat, intermediate, computeLength);

    // 结果类型转换
    LocalTensor<T> outputLocal = gradInputQueue.AllocTensor<T>();

    // 根据模板类型进行不同的转换处理
    if constexpr (std::is_same_v<T, float>) {
        // float类型直接拷贝
        DataCopy(outputLocal, gradInputFloat, this->processDataNum);
    } else {
        // half或bfloat16需要类型转换
        Cast(outputLocal, gradInputFloat, RoundMode::CAST_RINT, this->processDataNum);
    }

    gradInputQueue.EnQue(outputLocal);
}

extern "C" __global__ __aicore__ void softplus_v2_grad(GM_ADDR gradOutput, GM_ADDR self, GM_ADDR gradInput,
                                                       GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SoftplusV2GradTilingData);
    GET_TILING_DATA_WITH_STRUCT(SoftplusV2GradTilingData, tilingData, tiling);

    TPipe pipe;

    KernelSoftplusV2Grad<DTYPE_GRADOUTPUT> op;
    op.Init(&pipe, gradOutput, self, gradInput, workspace, tilingData);
    op.Process();
}
