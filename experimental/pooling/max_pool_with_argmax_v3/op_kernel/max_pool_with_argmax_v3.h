/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#ifndef MAX_POOL_WITH_ARGMAX_V3_H
#define MAX_POOL_WITH_ARGMAX_V3_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "max_pool_with_argmax_v3_tiling_data.h"
#include "max_pool_with_argmax_v3_tiling_key.h"

namespace NsMaxPoolWithArgmaxV3 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; 

// ---- type trait helper ----
// Used to distinguish bfloat16_t from half (both sizeof == 2) in if constexpr branches.
template <typename A, typename B>
struct IsSameType { static constexpr bool value = false; };
template <typename A>
struct IsSameType<A, A> { static constexpr bool value = true; };

template <typename T>
__aicore__ inline T FloatToT(float val)
{
    if constexpr (sizeof(T) == 4) {
        return val;
    } else {
        // half: static_cast is supported
        return static_cast<T>(val);
    }
}

template <>
__aicore__ inline bfloat16_t FloatToT<bfloat16_t>(float val)
{
    // bfloat16 = upper 16 bits of float32 (truncation, no rounding for simplicity)
    uint32_t bits;
    *(reinterpret_cast<float*>(&bits)) = val;
    uint16_t bf16bits = static_cast<uint16_t>(bits >> 16);
    bfloat16_t result;
    *(reinterpret_cast<uint16_t*>(&result)) = bf16bits;
    return result;
}

// Convert a value of type T to float
template <typename T>
__aicore__ inline float TToFloat(T val)
{
    if constexpr (sizeof(T) == 4) {
        return val;
    } else {
        // half: static_cast is supported
        return static_cast<float>(val);
    }
}

template <>
__aicore__ inline float TToFloat<bfloat16_t>(bfloat16_t val)
{
    uint16_t bf16bits;
    *(reinterpret_cast<bfloat16_t*>(&bf16bits)) = val;
    uint32_t bits = static_cast<uint32_t>(bf16bits) << 16;
    float result;
    *(reinterpret_cast<uint32_t*>(&result)) = bits;
    return result;
}

template <typename T>
class MaxPoolWithArgmaxV3 {
public:
    __aicore__ inline MaxPoolWithArgmaxV3() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR argmax,
                                const MaxPoolWithArgmaxV3TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessOneSlice(int64_t sliceIdx);
    __aicore__ inline void CopyInRows(int64_t sliceIdx, int64_t hOutStart, int64_t numRows);
    __aicore__ inline void ComputeTile(int64_t sliceIdx, int64_t hOutStart, int64_t numRows);
    __aicore__ inline void CopyOut(int64_t sliceIdx, int64_t hOutStart, int64_t numRows);

private:
    // Pipeline
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yOutputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> argmaxOutputQueue;

    // GM tensors
    GlobalTensor<T> inputGM;
    GlobalTensor<T> yGM;
    GlobalTensor<int64_t> argmaxGM;

    // Tiling parameters
    const MaxPoolWithArgmaxV3TilingData* tiling_;

    // Core-local state
    int64_t startSlice_;
    int64_t endSlice_;
};

template <typename T>
__aicore__ inline void MaxPoolWithArgmaxV3<T>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR argmax,
    const MaxPoolWithArgmaxV3TilingData* tilingData)
{
    tiling_ = tilingData;

    // 计算本核任务范围
    int64_t blockIdx = GetBlockIdx();
    startSlice_ = blockIdx * tiling_->slicesPerCore;
    endSlice_ = startSlice_ + tiling_->slicesPerCore;
    if (endSlice_ > tiling_->totalSlices) {
        endSlice_ = tiling_->totalSlices;
    }

    // 设置 GM 指针
    // For empty tensors (totalSlices=0), use size >= 1 to avoid zero-length GM buffers
    int64_t inputTotalElems = tiling_->totalSlices * tiling_->inputHeight * tiling_->inputWidth;
    int64_t outputTotalElems = tiling_->totalSlices * tiling_->outputHeight * tiling_->outputWidth;
    if (inputTotalElems == 0) inputTotalElems = 1;
    if (outputTotalElems == 0) outputTotalElems = 1;
    inputGM.SetGlobalBuffer((__gm__ T*)x, inputTotalElems);
    yGM.SetGlobalBuffer((__gm__ T*)y, outputTotalElems);
    argmaxGM.SetGlobalBuffer((__gm__ int64_t*)argmax, outputTotalElems);

    // 初始化 pipe buffer
    // inputQueue: 需容纳 totalInputRows 行, 其中
    // totalInputRows = dilationH * (kH - 1) + 1 + (outputRowsPerTile - 1) * strideH
    int64_t totalInputRows = static_cast<int64_t>(tiling_->dilationH) * (tiling_->kernelH - 1) + 1
                             + (tiling_->outputRowsPerTile - 1) * tiling_->strideH;
    int64_t inputBufSize = totalInputRows * tiling_->inputWidthAligned * static_cast<int64_t>(sizeof(T));
    pipe.InitBuffer(inputQueue, BUFFER_NUM, inputBufSize);

    // yOutputQueue: outputRowsPerTile 行输出
    int64_t yBufSize = tiling_->outputRowsPerTile * tiling_->outputWidthAligned * static_cast<int64_t>(sizeof(T));
    pipe.InitBuffer(yOutputQueue, BUFFER_NUM, yBufSize);

    // argmaxOutputQueue: outputRowsPerTile 行输出
    int64_t argmaxWidthAligned = (tiling_->outputWidth * 8 + 31) / 32 * 32 / 8;
    if (argmaxWidthAligned == 0) argmaxWidthAligned = 4; // minimum 32-byte alignment for int64_t
    int64_t argmaxBufSize = tiling_->outputRowsPerTile * argmaxWidthAligned * static_cast<int64_t>(sizeof(int64_t));
    pipe.InitBuffer(argmaxOutputQueue, BUFFER_NUM, argmaxBufSize);
}

template <typename T>
__aicore__ inline void MaxPoolWithArgmaxV3<T>::Process()
{
    for (int64_t sliceIdx = startSlice_; sliceIdx < endSlice_; sliceIdx++) {
        ProcessOneSlice(sliceIdx);
    }
}

template <typename T>
__aicore__ inline void MaxPoolWithArgmaxV3<T>::ProcessOneSlice(int64_t sliceIdx)
{
    // 迭代二：按 outputRowsPerTile 分组处理
    for (int64_t hOut = 0; hOut < tiling_->outputHeight; hOut += tiling_->outputRowsPerTile) {
        int64_t numRows = tiling_->outputRowsPerTile;
        if (hOut + numRows > tiling_->outputHeight) {
            numRows = tiling_->outputHeight - hOut;
        }
        CopyInRows(sliceIdx, hOut, numRows);
        ComputeTile(sliceIdx, hOut, numRows);
        CopyOut(sliceIdx, hOut, numRows);
    }
}

template <typename T>
__aicore__ inline void MaxPoolWithArgmaxV3<T>::CopyInRows(
    int64_t sliceIdx, int64_t hOutStart, int64_t numRows)
{
    LocalTensor<T> inputLocal = inputQueue.AllocTensor<T>();

    int64_t sliceOffset = sliceIdx * tiling_->inputHeight * tiling_->inputWidth;

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(tiling_->inputWidth * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPadParams padParams{false, 0, 0, 0};

    // 计算需要加载的输入行范围
    // inputRowStart = hOutStart * strideH - padH
    // inputRowEnd = (hOutStart + numRows - 1) * strideH - padH + dilationH * (kH - 1)
    int64_t inputRowStart = hOutStart * tiling_->strideH - tiling_->padH;
    int64_t inputRowEnd = (hOutStart + numRows - 1) * tiling_->strideH - tiling_->padH
                          + static_cast<int64_t>(tiling_->dilationH) * (tiling_->kernelH - 1);
    int64_t totalInputRows = inputRowEnd - inputRowStart + 1;

    // 加载所有需要的输入行
    for (int64_t i = 0; i < totalInputRows; i++) {
        int64_t ih = inputRowStart + i;
        int64_t bufRowOffset = i * tiling_->inputWidthAligned;

        if (ih >= 0 && ih < tiling_->inputHeight) {
            int64_t gmOffset = sliceOffset + ih * tiling_->inputWidth;
            DataCopyPad(inputLocal[bufRowOffset], inputGM[gmOffset], copyParams, padParams);
        } else {
            // padding 行：填充 dtype 最小值
            T minVal;
            if constexpr (sizeof(T) == 4) {
                // float32: -FLT_MAX
                minVal = static_cast<T>(-3.402823466e+38f);
            } else if constexpr (IsSameType<T, bfloat16_t>::value) {
                // bfloat16: minimum finite value ~= -3.389e+38
                // bfloat16 has same exponent range as float32, so -65504 is NOT sufficient
                minVal = FloatToT<T>(-3.3895314e+38f);
            } else {
                // float16 (half): -65504
                minVal = static_cast<T>(-65504.0f);
            }
            Duplicate(inputLocal[bufRowOffset], minVal, tiling_->inputWidthAligned);
        }
    }

    inputQueue.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void MaxPoolWithArgmaxV3<T>::ComputeTile(
    int64_t sliceIdx, int64_t hOutStart, int64_t numRows)
{
    LocalTensor<T> inputLocal = inputQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutputQueue.AllocTensor<T>();
    LocalTensor<int64_t> argmaxLocal = argmaxOutputQueue.AllocTensor<int64_t>();

    // inputRowStart 是 UB 中第一行对应的原始输入行号
    int64_t inputRowStart = hOutStart * tiling_->strideH - tiling_->padH;

    // 标量循环：对每个输出位置 (hOut, wOut)
    for (int64_t r = 0; r < numRows; r++) {
        int64_t hOut = hOutStart + r;
        for (int64_t wOut = 0; wOut < tiling_->outputWidth; wOut++) {
            // 使用 float 做标量比较
            float maxValF;
            if constexpr (sizeof(T) == 4) {
                maxValF = -3.402823466e+38f;
            } else if constexpr (IsSameType<T, bfloat16_t>::value) {
                // bfloat16 has same exponent range as float32
                maxValF = -3.3895314e+38f;
            } else {
                maxValF = -65504.0f;
            }
            int64_t maxIdx = 0;

            // 遍历 kernel 窗口
            for (int32_t kh = 0; kh < tiling_->kernelH; kh++) {
                for (int32_t kw = 0; kw < tiling_->kernelW; kw++) {
                    int64_t ih = hOut * tiling_->strideH - tiling_->padH + kh * tiling_->dilationH;
                    int64_t iw = wOut * tiling_->strideW - tiling_->padW + kw * tiling_->dilationW;

                    if (ih >= 0 && ih < tiling_->inputHeight && iw >= 0 && iw < tiling_->inputWidth) {
                        // 从 UB 中读取：bufRowIdx = ih - inputRowStart
                        int64_t bufRowIdx = ih - inputRowStart;
                        int64_t bufIdx = bufRowIdx * tiling_->inputWidthAligned + iw;
                        T val = inputLocal.GetValue(bufIdx);

                        // 转换为 float 进行比较
                        float valF = TToFloat<T>(val);
                        if (valF > maxValF) {
                            maxValF = valF;
                            maxIdx = ih * tiling_->inputWidth + iw;
                        }
                    }
                    // padding 位置跳过（值为 -inf，不会被选为最大值）
                }
            }

            // 回写结果
            T maxVal = FloatToT<T>(maxValF);
            int64_t outIdx = r * tiling_->outputWidth + wOut;
            yLocal.SetValue(outIdx, maxVal);
            argmaxLocal.SetValue(outIdx, maxIdx);
        }
    }

    yOutputQueue.EnQue(yLocal);
    argmaxOutputQueue.EnQue(argmaxLocal);
    inputQueue.FreeTensor(inputLocal);
}

template <typename T>
__aicore__ inline void MaxPoolWithArgmaxV3<T>::CopyOut(
    int64_t sliceIdx, int64_t hOutStart, int64_t numRows)
{
    int64_t sliceOutOffset = sliceIdx * tiling_->outputHeight * tiling_->outputWidth;
    int64_t tileOffset = sliceOutOffset + hOutStart * tiling_->outputWidth;

    LocalTensor<T> yLocal = yOutputQueue.DeQue<T>();
    LocalTensor<int64_t> argmaxLocal = argmaxOutputQueue.DeQue<int64_t>();

    // 搬运 y 输出 (numRows 行)
    DataCopyParams yCopyParams;
    yCopyParams.blockCount = 1;
    yCopyParams.blockLen = static_cast<uint32_t>(numRows * tiling_->outputWidth * sizeof(T));
    yCopyParams.srcStride = 0;
    yCopyParams.dstStride = 0;
    DataCopyPad(yGM[tileOffset], yLocal, yCopyParams);

    // 搬运 argmax 输出 (numRows 行)
    DataCopyParams argmaxCopyParams;
    argmaxCopyParams.blockCount = 1;
    argmaxCopyParams.blockLen = static_cast<uint32_t>(numRows * tiling_->outputWidth * sizeof(int64_t));
    argmaxCopyParams.srcStride = 0;
    argmaxCopyParams.dstStride = 0;
    DataCopyPad(argmaxGM[tileOffset], argmaxLocal, argmaxCopyParams);

    yOutputQueue.FreeTensor(yLocal);
    argmaxOutputQueue.FreeTensor(argmaxLocal);
}

} // namespace NsMaxPoolWithArgmaxV3
#endif // MAX_POOL_WITH_ARGMAX_V3_H
