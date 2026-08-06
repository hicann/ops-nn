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
 * \file embedding_dense_grad.h
 * \brief kernel implementation of embedding_dense_grad
 */
#ifndef EMBEDDING_DENSE_GRAD_H
#define EMBEDDING_DENSE_GRAD_H
#include <type_traits>
#include "kernel_operator.h"
#include "embedding_dense_grad_tiling_key.h"

// UB scratch buffer size for SyncAll API (bytes)
constexpr uint32_t SYNC_UB_BUF_SIZE = 512;
namespace AscendC {
template <typename T, typename U, int tilingKey>
class EmbeddingDenseGradKernel {
    static constexpr bool kFp16 = std::is_same<T, half>::value;
    using ComputeT = typename std::conditional<kFp16, float, T>::type;

public:
    __aicore__ inline EmbeddingDenseGradKernel() = delete;

    __aicore__ inline EmbeddingDenseGradKernel(GM_ADDR grad, GM_ADDR indices, GM_ADDR y, GM_ADDR workspace,
                                               const EmbeddingDenseGradTilingData& tiling, TPipe& pipe)
    {
        InitParams(tiling);
        SetGmAddr(grad, indices, y, workspace);
        InitBuffers(pipe);
    }

    __aicore__ inline void Process()
    {
        if constexpr (tilingKey == EMBEDDING_DENSE_GRAD_SCH_MODE_SINGLE_ROW) {
            for (int i = 0; i < batchSize; i++) {
                if constexpr (kFp16) {
                    CopyInHP(dimSize, i * dimSize);
                    CopyOutHP(dimSize, 0, i, true);
                } else {
                    CopyIn(dimSize, i * dimSize);
                    CopyOut(dimSize, 0, i, true);
                }
            }

        } else if constexpr (tilingKey == EMBEDDING_DENSE_GRAD_SCH_MODE_SEGMENTED) {
            auto iterations = dimSize / ubProcessNum;
            auto tail = iterations == 0 ? dimSize : dimSize % ubProcessNum;
            for (int i = 0; i < batchSize; i++) {
                for (int j = 0; j < iterations; j++) {
                    if constexpr (kFp16) {
                        CopyInHP(ubProcessNum, i * dimSize + j * ubProcessNum);
                        CopyOutHP(ubProcessNum, j * ubProcessNum, i, j == 0);
                    } else {
                        CopyIn(ubProcessNum, i * dimSize + j * ubProcessNum);
                        CopyOut(ubProcessNum, j * ubProcessNum, i, j == 0);
                    }
                }
                if (tail > 0) {
                    if constexpr (kFp16) {
                        CopyInHP(tail, i * dimSize + iterations * ubProcessNum);
                        CopyOutHP(tail, iterations * ubProcessNum, i, iterations == 0);
                    } else {
                        CopyIn(tail, i * dimSize + iterations * ubProcessNum);
                        CopyOut(tail, iterations * ubProcessNum, i, iterations == 0);
                    }
                }
            }

        } else if constexpr (tilingKey == EMBEDDING_DENSE_GRAD_SCH_MODE_PACKED) {
            ProcessPacked();
        }
    }

    __aicore__ inline void CastOutput()
    {
        if constexpr (kFp16) {
            SyncAll(workGm, workspace.ReinterpretCast<int32_t>(), GetBlockNum());
            uint64_t total = (uint64_t)numWeights * dimSize;
            uint64_t blockNum = GetBlockNum();
            uint64_t perCore = CeilAlign(CeilDiv(total, blockNum), gradBlockElemNum);
            uint64_t start = blockIdx * perCore;
            if (start >= total) {
                return;
            }
            uint64_t end = (start + perCore > total) ? total : start + perCore;
            for (uint64_t off = start; off < end; off += ubProcessNum) {
                uint32_t len = (off + ubProcessNum <= end) ? (uint32_t)ubProcessNum : (uint32_t)(end - off);
                queBind.AllocTensor<float>(fp32Tensor);
                DataCopy(fp32Tensor, fp32AccumGm[off], CeilAlign(len, BLOCK_SIZE / sizeof(float)));
                event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(evt);
                WaitFlag<HardEvent::MTE2_V>(evt);
                LocalTensor<T> castTensor = fp32Tensor.template ReinterpretCast<T>();
                Cast(castTensor, fp32Tensor, RoundMode::CAST_NONE, len);
                queBind.EnQue<float>(fp32Tensor);
                queBind.DeQue<float>(fp32Tensor);
                LocalTensor<T> outTensor = fp32Tensor.template ReinterpretCast<T>();
                DataCopy(outputGm[off], outTensor, CeilAlign(len, gradBlockElemNum));
                queBind.FreeTensor<float>(fp32Tensor);
            }
        }
    }

private:
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 0> queBind;
    TQue<TPosition::VECIN, 0> idxQue;
    TBuf<TPosition::VECCALC> workBuf;
    TBuf<TPosition::VECCALC> idxBuf;
    GlobalTensor<T> gradGm;
    GlobalTensor<T> outputGm;
    GlobalTensor<U> indicesGm;
    GlobalTensor<int32_t> workGm;
    GlobalTensor<float> indicesCountGM;
    GlobalTensor<float> fp32AccumGm;

    LocalTensor<T> gradTensor;
    LocalTensor<float> fp32Tensor;
    LocalTensor<U> idxTensor;
    LocalTensor<float> idxCountTensor;
    LocalTensor<uint8_t> workspace;

    uint64_t blockIdx;
    uint64_t gradGlobalOffset;
    uint64_t idxGlobalOffset;

    uint64_t gradBlockElemNum;
    uint64_t dimSize;
    uint64_t batchSize;
    uint64_t ubProcessNum;

    int64_t numWeights;

    int64_t paddingIdx;
    bool scaleGradByFreq;

    __aicore__ inline void InitParams(const EmbeddingDenseGradTilingData& tiling)
    {
        blockIdx = GetBlockIdx();
        numWeights = tiling.numWeights;
        dimSize = tiling.dimSize;
        paddingIdx = tiling.paddingIdx;
        scaleGradByFreq = tiling.scaleGradByFreq;
        ubProcessNum = tiling.ubProcessNum;
        gradBlockElemNum = BLOCK_SIZE / sizeof(T);

        gradGlobalOffset = blockIdx * tiling.formerBatchSize * dimSize;
        idxGlobalOffset = blockIdx * tiling.formerBatchSize;

        if (blockIdx < tiling.formerCoreNum) {
            batchSize = tiling.formerBatchSize;
        } else {
            batchSize = tiling.tailBatchSize;
            gradGlobalOffset -= (blockIdx - tiling.formerCoreNum) * dimSize;
            idxGlobalOffset -= (blockIdx - tiling.formerCoreNum);
        }
    }

    __aicore__ inline void InitBuffers(TPipe& pipe)
    {
        pipe.InitBuffer(workBuf, SYNC_UB_BUF_SIZE);
        workspace = workBuf.Get<uint8_t>();
        SyncAll(workGm, workspace.ReinterpretCast<int32_t>(), GetBlockNum());

        pipe.InitBuffer(queBind, BUFFER_NUM, ubProcessNum * sizeof(ComputeT));

        pipe.InitBuffer(idxBuf, 64);
        idxCountTensor = idxBuf.Get<float>();
        Duplicate(idxCountTensor, 0.0f, 8);
        idxCountTensor.SetValue(0, 1);
    }

    __aicore__ inline void SetGmAddr(GM_ADDR grad, GM_ADDR indices, GM_ADDR output, GM_ADDR workspace)
    {
        gradGm.SetGlobalBuffer((__gm__ T*)grad + gradGlobalOffset, batchSize * dimSize);
        indicesGm.SetGlobalBuffer((__gm__ U*)indices + idxGlobalOffset);
        outputGm.SetGlobalBuffer((__gm__ T*)output);
        auto align8NumWeights = CeilAlign(numWeights, 8);
        workGm.SetGlobalBuffer((__gm__ int32_t*)workspace + align8NumWeights, (uint64_t)64);

        indicesCountGM.SetGlobalBuffer((__gm__ float*)workspace);
        if constexpr (kFp16) {
            fp32AccumGm.SetGlobalBuffer((__gm__ float*)workspace + WORKSPACE_HEADER_FLOATS + align8NumWeights,
                                        (uint64_t)numWeights * dimSize);
        }
        if (blockIdx == 0) {
            if (scaleGradByFreq) {
                InitGlobalMemory(indicesCountGM, align8NumWeights, 0.0f);
            }
            if constexpr (kFp16) {
                InitGlobalMemory(fp32AccumGm, CeilAlign((uint64_t)numWeights * dimSize, gradBlockElemNum), 0.0f);
            }
        }
#ifndef __CCE_KT_TEST__
        InitGlobalMemory(workGm, 64, 0);
#endif
    }

    __aicore__ inline void ProcessPacked()
    {
        uint64_t K = ubProcessNum / dimSize;
        if (K == 0) {
            K = 1;
        }
        SetAtomicAdd<T>();
        for (uint64_t base = 0; base < batchSize; base += K) {
            uint64_t rows = (base + K <= batchSize) ? K : (batchSize - base);
            queBind.AllocTensor<T>(gradTensor);
            DataCopy(gradTensor, gradGm[base * dimSize], rows * dimSize);
            queBind.EnQue<T>(gradTensor);
            queBind.DeQue<T>(gradTensor);
            for (uint64_t j = 0; j < rows; j++) {
                int idx = indicesGm.GetValue(base + j);
                if (paddingIdx != idx) {
                    DataCopy(outputGm[idx * dimSize], gradTensor[j * dimSize], dimSize);
                }
            }
            queBind.FreeTensor<T>(gradTensor);
        }
        SetAtomicNone();
    }

    __aicore__ inline void CopyInHP(int cols, uint64_t gradOffset)
    {
        queBind.AllocTensor<float>(fp32Tensor);
        LocalTensor<T> halfTensor = fp32Tensor.template ReinterpretCast<T>();
        DataCopy(halfTensor, gradGm[gradOffset], CeilAlign(cols, gradBlockElemNum));
        event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(evt);
        WaitFlag<HardEvent::MTE2_V>(evt);
        Cast(fp32Tensor, halfTensor, RoundMode::CAST_NONE, cols);
        auto alignFp32Cols = CeilAlign(cols, BLOCK_SIZE / sizeof(float));
        if (cols != alignFp32Cols) {
            Duplicate(fp32Tensor[cols], 0.0f, alignFp32Cols - cols);
        }
        queBind.EnQue<float>(fp32Tensor);
    }
    __aicore__ inline void CopyOutHP(uint64_t cols, uint64_t iterationOffset, uint64_t idxOffset, bool isFirst)
    {
        queBind.DeQue<float>(fp32Tensor);
        auto alignFp32Cols = CeilAlign(cols, BLOCK_SIZE / sizeof(float));
        int idx = indicesGm.GetValue(idxOffset);
        if (paddingIdx != idx) {
            SetAtomicAdd<float>();
            DataCopy(fp32AccumGm[idx * dimSize + iterationOffset], fp32Tensor, alignFp32Cols);
            SetAtomicNone();
            if (scaleGradByFreq && isFirst) {
                SetAtomicAdd<float>();
                DataCopy(indicesCountGM[idx], idxCountTensor, 8);
                SetAtomicNone();
            }
        }
        queBind.FreeTensor<float>(fp32Tensor);
    }

    __aicore__ inline void CopyIn(int cols, uint64_t gradOffset)
    {
        queBind.AllocTensor<T>(gradTensor);
        auto alignTCols = CeilAlign(cols, gradBlockElemNum);
        DataCopy(gradTensor, gradGm[gradOffset], alignTCols);
        PipeBarrier<PIPE_ALL>();
        queBind.EnQue<T>(gradTensor);
        queBind.DeQue<T>(gradTensor);
        if (cols != alignTCols) {
            uint64_t mask[2] = {(1ul << gradBlockElemNum) - (1ul << (cols % gradBlockElemNum)), 0};
            Duplicate(gradTensor[alignTCols - gradBlockElemNum], (T)0.0f, mask, 1, 1, 8);
        }
    }
    __aicore__ inline void CopyOut(uint64_t cols, uint64_t iterationOffset, uint64_t idxOffset, bool isFirst)
    {
        auto alignTCols = CeilAlign(cols, gradBlockElemNum);
        int idx = indicesGm.GetValue(idxOffset);
        if (paddingIdx != idx) {
            SetAtomicAdd<T>();
            DataCopy(outputGm[idx * dimSize + iterationOffset], gradTensor, alignTCols);
            SetAtomicNone();
            if (scaleGradByFreq && isFirst) {
                SetAtomicAdd<float>();
                DataCopy(indicesCountGM[idx], idxCountTensor, 8);
                SetAtomicNone();
            }
        }

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
