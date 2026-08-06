/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file embedding_dense_grad_scale.h
 * \brief scale-by-freq post-processing kernel of embedding_dense_grad
 */
#ifndef EMBEDDING_DENSE_GRAD_SCALE_H
#define EMBEDDING_DENSE_GRAD_SCALE_H

#include <type_traits>
#include "kernel_operator.h"
#include "embedding_dense_grad_tiling_key.h"

// UB scratch buffer size for SyncAll API (bytes), 256B enough for up to 64 cores
constexpr uint32_t SYNC_UB_BUF_SIZE_SCALE = 256;

namespace AscendC {
template <typename T>
class EmbeddingDenseGradScaleKernel {
    static constexpr bool kFp16 = std::is_same<T, half>::value;

public:
    __aicore__ inline EmbeddingDenseGradScaleKernel() = delete;

    __aicore__ inline EmbeddingDenseGradScaleKernel(GM_ADDR output, GM_ADDR workspace,
                                                    const EmbeddingDenseGradTilingData& tiling, TPipe& pipe)
    {
        InitParams(tiling);
        SetGmAddr(output, workspace);
        InitBuffers(pipe);
    }

    __aicore__ inline void Process()
    {
        SyncAll(workGm, workspace, GetBlockNum());
        auto iterations = dimSize / ubProcessNum;
        auto tail = dimSize % ubProcessNum;
        if constexpr (kFp16) {
            for (int i = 0; i < batchSize; i++) {
                float idxCount = indicesCountGm.GetValue(i);
                float scale = (idxCount >= 2) ? (1.0f / idxCount) : 1.0f;
                for (int j = 0; j < iterations; j++) {
                    ProcessTileHP(ubProcessNum, i * dimSize + j * ubProcessNum, scale);
                }
                if (tail > 0) {
                    ProcessTileHP(tail, i * dimSize + iterations * ubProcessNum, scale);
                }
            }
        } else {
            for (int i = 0; i < batchSize; i++) {
                float idxCount = indicesCountGm.GetValue(i);
                if (idxCount < 2)
                    continue;
                for (int j = 0; j < iterations; j++) {
                    CopyIn(ubProcessNum, i * dimSize + j * ubProcessNum);
                    Compute(ubProcessNum, 1 / idxCount);
                    CopyOut(ubProcessNum, i * dimSize + j * ubProcessNum);
                }
                if (tail > 0) {
                    CopyIn(tail, i * dimSize + iterations * ubProcessNum);
                    Compute(tail, 1 / idxCount);
                    CopyOut(tail, i * dimSize + iterations * ubProcessNum);
                }
            }
        }
    }

    __aicore__ inline void ProcessTileHP(int len, uint64_t offset, float scale)
    {
        queBind.AllocTensor<float>(fp32Tensor);
        DataCopy(fp32Tensor, fp32AccumGm[offset], CeilAlign(len, BLOCK_SIZE / sizeof(float)));
        event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(evt);
        WaitFlag<HardEvent::MTE2_V>(evt);
        if (scale != 1.0f) {
            Muls(fp32Tensor, fp32Tensor, scale, len);
        }
        LocalTensor<T> outTensor = fp32Tensor.template ReinterpretCast<T>();
        Cast(outTensor, fp32Tensor, RoundMode::CAST_NONE, len);
        queBind.EnQue<float>(fp32Tensor);
        queBind.DeQue<float>(fp32Tensor);
        LocalTensor<T> dT = fp32Tensor.template ReinterpretCast<T>();
        DataCopy(outputGm[offset], dT, CeilAlign(len, gradBlockElemNum));
        queBind.FreeTensor<float>(fp32Tensor);
    }

private:
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 0> queBind;
    TBuf<TPosition::VECCALC> workBuf;
    GlobalTensor<T> outputGm;
    GlobalTensor<float> indicesCountGm;
    GlobalTensor<int32_t> workGm;
    GlobalTensor<float> fp32AccumGm;

    LocalTensor<T> gradTensor;
    LocalTensor<float> fp32Tensor;
    LocalTensor<int32_t> workspace;

    uint64_t blockIdx;
    uint64_t outGlobalOffset;
    uint64_t idxGlobalOffset;

    uint64_t gradBlockElemNum;
    uint64_t dimSize;
    uint64_t batchSize;
    uint64_t ubProcessNum;
    int64_t numWeights;

    __aicore__ inline void InitParams(const EmbeddingDenseGradTilingData& tiling)
    {
        blockIdx = GetBlockIdx();
        dimSize = tiling.dimSize;
        ubProcessNum = tiling.scaleUbProcessNum;
        numWeights = tiling.numWeights;
        gradBlockElemNum = BLOCK_SIZE / sizeof(T);

        outGlobalOffset = blockIdx * tiling.scaleFormerBatchSize * dimSize;
        idxGlobalOffset = blockIdx * tiling.scaleFormerBatchSize;

        if (blockIdx < tiling.scaleFormerCoreNum) {
            batchSize = tiling.scaleFormerBatchSize;
        } else {
            batchSize = tiling.scaleTailBatchSize;
            outGlobalOffset -= (blockIdx - tiling.scaleFormerCoreNum) * dimSize;
            idxGlobalOffset -= (blockIdx - tiling.scaleFormerCoreNum);
        }
    }

    __aicore__ inline void InitBuffers(TPipe& pipe)
    {
        if constexpr (kFp16) {
            pipe.InitBuffer(queBind, BUFFER_NUM, ubProcessNum * sizeof(float));
        } else {
            pipe.InitBuffer(queBind, BUFFER_NUM, ubProcessNum * sizeof(T));
        }
        pipe.InitBuffer(workBuf, SYNC_UB_BUF_SIZE_SCALE);
        workspace = workBuf.Get<int32_t>();
    }

    __aicore__ inline void SetGmAddr(GM_ADDR output, GM_ADDR workspace)
    {
        outputGm.SetGlobalBuffer((__gm__ T*)output + outGlobalOffset);

        workGm.SetGlobalBuffer((__gm__ int32_t*)workspace + CeilAlign(numWeights, 8));
        indicesCountGm.SetGlobalBuffer((__gm__ float*)workspace + idxGlobalOffset);
        if constexpr (kFp16) {
            fp32AccumGm.SetGlobalBuffer((__gm__ float*)workspace + WORKSPACE_HEADER_FLOATS + CeilAlign(numWeights, 8) +
                                        outGlobalOffset);
        }
        InitGlobalMemory(workGm, 64, 0);
    }
    __aicore__ inline void CopyIn(int copyLen, uint64_t offset)
    {
        queBind.AllocTensor<T>(gradTensor);
        auto alignTCopyLen = CeilAlign(copyLen, gradBlockElemNum);
        DataCopy(gradTensor, outputGm[offset], alignTCopyLen);
        PipeBarrier<PIPE_ALL>();
        queBind.EnQue<T>(gradTensor);
    }
    __aicore__ inline void Compute(int cols, float scale)
    {
        queBind.DeQue<T>(gradTensor);
        auto alignTCols = CeilAlign(cols, gradBlockElemNum);
        Muls(gradTensor, gradTensor, (T)(scale - 1), cols);
        if (cols != alignTCols) {
            uint64_t mask[2] = {(1ul << gradBlockElemNum) - (1ul << (cols % gradBlockElemNum)), 0};
            Duplicate(gradTensor[alignTCols - gradBlockElemNum], (T)0.0f, mask, 1, 1, 8);
        }
        queBind.EnQue<T>(gradTensor);
    }

    __aicore__ inline void CopyOut(int copyLen, uint64_t offset)
    {
        queBind.DeQue<T>(gradTensor);
        auto alignTCopyLen = CeilAlign(copyLen, gradBlockElemNum);
        SetAtomicAdd<T>();
        DataCopy(outputGm[offset], gradTensor, alignTCopyLen);
        SetAtomicNone();

        queBind.FreeTensor<T>(gradTensor);
    }

    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b)
    {
        return (a + b - 1) / b;
    }
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilAlign(T1 a, T2 b)
    {
        return (a + b - 1) / b * b;
    }
};
} // namespace AscendC
#endif
