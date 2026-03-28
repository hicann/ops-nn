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

/**
 * \file foreach_div_scalar.h
 * \brief ForeachDivScalar kernel class definition (arch32 / Ascend910B)
 *
 * Processes a TensorList by dividing each tensor's elements by a scalar value.
 * Uses Muls(x, 1/scalar) for performance.
 *
 * Uses the official ListTensorDesc API to decode DYNAMIC TensorList descriptors.
 */
#ifndef FOREACH_DIV_SCALAR_H
#define FOREACH_DIV_SCALAR_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"
#include "foreach_div_scalar_tiling_data.h"
#include "foreach_div_scalar_tiling_key.h"

namespace NsForeachDivScalar {

using namespace AscendC;

template <typename T>
class ForeachDivScalar {
public:
    __aicore__ inline ForeachDivScalar() {};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scalar, GM_ADDR y,
                                 const ForeachDivScalarTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(GlobalTensor<T>& inGM, int64_t gmOffset, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(GlobalTensor<T>& outGM, int64_t gmOffset, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inputQueue;
    TQue<QuePosition::VECOUT, 1> outputQueue;
    TBuf<QuePosition::VECCALC> bf16TmpBuf_;  // Pre-allocated buffer for bf16 intermediate fp32

    const ForeachDivScalarTilingData* tilingData_ = nullptr;
    float scalarReciprocalF32_ = 0.0f;

    // ListTensorDesc for accessing DYNAMIC TensorList
    ListTensorDesc xListDesc_;
    ListTensorDesc yListDesc_;

    int64_t ubFactor_ = 0;
};

template <typename T>
__aicore__ inline void ForeachDivScalar<T>::Init(
    GM_ADDR x, GM_ADDR scalar, GM_ADDR y,
    const ForeachDivScalarTilingData* tilingData)
{
    tilingData_ = tilingData;
    ubFactor_ = tilingData->ubFactor;

    // Decode DYNAMIC TensorList descriptors using official API
    xListDesc_.Init(x);
    yListDesc_.Init(y);

    // Initialize buffers
    pipe.InitBuffer(inputQueue, 1, ubFactor_ * sizeof(T));
    pipe.InitBuffer(outputQueue, 1, ubFactor_ * sizeof(T));

    // Pre-allocate bf16 intermediate buffer (only used for bf16 path)
    if constexpr (!std::is_same_v<T, half> && !std::is_same_v<T, float>) {
        pipe.InitBuffer(bf16TmpBuf_, ubFactor_ * sizeof(float));
    }

    // Read scalar from GM (scalar is REQUIRED, so GM_ADDR is direct device address)
    // Read 32 bytes (minimum DMA block) and interpret based on scalarDtype
    {
        GlobalTensor<uint8_t> scalarGM;
        scalarGM.SetGlobalBuffer((__gm__ uint8_t*)scalar, 32);
        TBuf<QuePosition::VECCALC> scalarBuf;
        pipe.InitBuffer(scalarBuf, 32);
        LocalTensor<uint8_t> scalarLocal = scalarBuf.Get<uint8_t>();
        DataCopy(scalarLocal, scalarGM, 32);
        PipeBarrier<PIPE_ALL>();

        float scalarVal = 0.0f;
        if (tilingData->scalarDtype == 2) {
            // double: read as uint64, extract mantissa and exponent to float
            LocalTensor<uint64_t> u64Local = scalarLocal.ReinterpretCast<uint64_t>();
            uint64_t dbits = u64Local.GetValue(0);
            // IEEE 754 double -> float conversion via bit manipulation
            uint32_t sign = static_cast<uint32_t>((dbits >> 63) & 1);
            int32_t dexp = static_cast<int32_t>((dbits >> 52) & 0x7FF);
            uint64_t dmant = dbits & 0x000FFFFFFFFFFFFFULL;
            uint32_t fbits;
            if (dexp == 0) { fbits = sign << 31; }  // zero/subnormal -> zero
            else if (dexp == 0x7FF) { fbits = (sign << 31) | 0x7F800000 | static_cast<uint32_t>(dmant >> 29); }  // inf/nan
            else {
                int32_t fexp = dexp - 1023 + 127;
                if (fexp >= 255) { fbits = (sign << 31) | 0x7F800000; }  // overflow -> inf
                else if (fexp <= 0) { fbits = sign << 31; }  // underflow -> zero
                else { fbits = (sign << 31) | (static_cast<uint32_t>(fexp) << 23) | static_cast<uint32_t>(dmant >> 29); }
            }
            scalarVal = *reinterpret_cast<float*>(&fbits);
        } else if (tilingData->scalarDtype == 1) {
            // fp16: read 2 bytes as uint16, convert to float
            LocalTensor<half> hLocal = scalarLocal.ReinterpretCast<half>();
            half hVal = hLocal.GetValue(0);
            scalarVal = static_cast<float>(hVal);
        } else {
            // float: read 4 bytes as float
            LocalTensor<float> fLocal = scalarLocal.ReinterpretCast<float>();
            scalarVal = fLocal.GetValue(0);
        }
        scalarReciprocalF32_ = 1.0f / scalarVal;
    }
}

template <typename T>
__aicore__ inline void ForeachDivScalar<T>::Process()
{
    uint32_t tensorNum = tilingData_->tensorNum;
    int64_t totalElements = tilingData_->totalElements;
    int64_t blockIdx = GetBlockIdx();
    int64_t blockFactor = tilingData_->blockFactor;  // elements per core

    // Element-granularity multi-core splitting
    int64_t coreStartElem = blockIdx * blockFactor;
    int64_t coreEndElem = coreStartElem + blockFactor;
    if (coreEndElem > totalElements) coreEndElem = totalElements;
    if (coreStartElem >= totalElements) return;

    // Compute cumulative offsets dynamically (tensorOffsets removed from TilingData)
    int64_t tStart = 0;
    for (uint32_t t = 0; t < tensorNum && t < MAX_TENSOR_NUM; t++) {
        int64_t tLen = tilingData_->tensorLengths[t];
        int64_t tEnd = tStart + tLen;

        // Compute overlap between this core's element range and this tensor's range
        int64_t overlapStart = (coreStartElem > tStart) ? coreStartElem : tStart;
        int64_t overlapEnd = (coreEndElem < tEnd) ? coreEndElem : tEnd;
        if (overlapStart >= overlapEnd) {
            tStart = tEnd;
            continue;
        }

        // Get actual data addresses from descriptor using official ListTensorDesc API
        __gm__ T* xDataPtr = xListDesc_.GetDataPtr<T>(t);
        __gm__ T* yDataPtr = yListDesc_.GetDataPtr<T>(t);

        GlobalTensor<T> inGM;
        GlobalTensor<T> outGM;
        inGM.SetGlobalBuffer(xDataPtr, tLen);
        outGM.SetGlobalBuffer(yDataPtr, tLen);

        int64_t localOffset = overlapStart - tStart;  // offset within this tensor
        int64_t processLen = overlapEnd - overlapStart;

        // UB tile loop to process the overlap portion
        int64_t loopCount = (processLen + ubFactor_ - 1) / ubFactor_;
        for (int64_t i = 0; i < loopCount; i++) {
            int64_t tileOffset = i * ubFactor_;
            int64_t currentNum = (i == loopCount - 1)
                ? (processLen - ubFactor_ * i) : ubFactor_;
            CopyIn(inGM, localOffset + tileOffset, currentNum);
            Compute(currentNum);
            CopyOut(outGM, localOffset + tileOffset, currentNum);
        }

        tStart = tEnd;
    }
}

template <typename T>
__aicore__ inline void ForeachDivScalar<T>::CopyIn(GlobalTensor<T>& inGM, int64_t gmOffset, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueue.template AllocTensor<T>();
    // Use DataCopyPad: auto-handles non-aligned tail blocks, avoids GM over-read
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, inGM[gmOffset], copyParams, padParams);
    inputQueue.template EnQue<T>(xLocal);
}

template <typename T>
__aicore__ inline void ForeachDivScalar<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueue.template DeQue<T>();
    LocalTensor<T> yLocal = outputQueue.template AllocTensor<T>();

    constexpr int64_t elemPerBlock = 32 / sizeof(T);
    int64_t alignedNum = ((currentNum + elemPerBlock - 1) / elemPerBlock) * elemPerBlock;

    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float>) {
        T scalarReciprocal = static_cast<T>(scalarReciprocalF32_);
        Muls(yLocal, xLocal, scalarReciprocal, alignedNum);
    } else {
        // bf16 path: use pre-allocated TBuf for intermediate fp32 buffers
        LocalTensor<float> tmpFloat = bf16TmpBuf_.Get<float>();
        Cast(tmpFloat, xLocal, RoundMode::CAST_NONE, alignedNum);
        Muls(tmpFloat, tmpFloat, scalarReciprocalF32_, alignedNum);
        Cast(yLocal, tmpFloat, RoundMode::CAST_ROUND, alignedNum);
    }

    outputQueue.template EnQue<T>(yLocal);
    inputQueue.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void ForeachDivScalar<T>::CopyOut(GlobalTensor<T>& outGM, int64_t gmOffset, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueue.template DeQue<T>();
    // Use DataCopyPad: auto-discards padding on CopyOut, only writes valid data to GM
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outGM[gmOffset], yLocal, copyParams);
    outputQueue.FreeTensor(yLocal);
}

} // namespace NsForeachDivScalar
#endif // FOREACH_DIV_SCALAR_H
