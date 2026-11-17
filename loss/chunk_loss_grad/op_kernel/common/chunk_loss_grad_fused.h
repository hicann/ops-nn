/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#pragma once
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
namespace ChunkLossGradNative {
using namespace AscendC;
constexpr uint16_t LOGITS_READY = 8;
constexpr uint16_t G_READY = 9;
constexpr uint16_t PARTIAL_READY = 10;
constexpr uint16_t SLOT_RELEASED = 11;
// A2 ND2NZ srcDValue is uint16_t. Enable the Matmul stride guard so
// H >= UINT16_MAX uses row-wise copies instead of truncating the GM row stride.
// TPipe initializes the pure-AIC MDL backend synchronization state.
constexpr auto MM_CONFIG = GetMDLConfig(true, false, 0, false, false, false, false, true, true, false, false, false);
template <class InputType>
using LogitsMatmul = matmul::MatmulImpl<matmul::MatmulType<TPosition::GM, CubeFormat::ND, InputType>,
                                        matmul::MatmulType<TPosition::GM, CubeFormat::ND, InputType, true>,
                                        matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>,
                                        matmul::MatmulType<TPosition::GM, CubeFormat::ND, InputType>, MM_CONFIG>;
template <class InputType>
using WeightMatmul = matmul::MatmulImpl<matmul::MatmulType<TPosition::GM, CubeFormat::ND, InputType, true>,
                                        matmul::MatmulType<TPosition::GM, CubeFormat::ND, InputType>,
                                        matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>,
                                        matmul::MatmulType<TPosition::GM, CubeFormat::ND, InputType>, MM_CONFIG>;

template <class InputType, class Tiling>
class Kernel {
public:
    __aicore__ inline void Init(GM_ADDR hidden, GM_ADDR weight, GM_ADDR labels, GM_ADDR maximum, GM_ADDR sum,
                                GM_ADDR grad, GM_ADDR output, GM_ADDR workspace, const Tiling& tiling)
    {
        tilingData = tiling;
        group = GetBlockIdx();
        lane = 0;
        if ASCEND_IS_AIV {
            group = GetBlockIdx() / 2;
            lane = GetBlockIdx() % 2;
        }
        hiddenGm.SetGlobalBuffer((__gm__ InputType*)hidden);
        weightGm.SetGlobalBuffer((__gm__ InputType*)weight);
        labelsGm.SetGlobalBuffer((__gm__ int64_t*)labels);
        softmaxMaxGm.SetGlobalBuffer((__gm__ float*)maximum);
        softmaxSumGm.SetGlobalBuffer((__gm__ float*)sum);
        upstream.SetGlobalBuffer((__gm__ float*)grad);
        gradWeightGm.SetGlobalBuffer((__gm__ InputType*)output);
        auto base = GetUserWorkspace(workspace) + uint64_t(group) * tilingData.coreWorkspaceBytes;
        logits.SetGlobalBuffer((__gm__ float*)base);
        logitsGradGm.SetGlobalBuffer((__gm__ InputType*)(base + tilingData.logitsBytes));
        partial.SetGlobalBuffer((__gm__ float*)(base + tilingData.logitsBytes + tilingData.logitsGradBytes));
        accum.SetGlobalBuffer(
            (__gm__ float*)(base + tilingData.logitsBytes + tilingData.logitsGradBytes + tilingData.partialBytes));
        if ASCEND_IS_AIC {
            logitsMatmul.Init(&tilingData.logitsTiling, &pipe);
            gradWeightMatmul.Init(&tilingData.weightTiling, &pipe);
        }
        if ASCEND_IS_AIV {
            pipe.InitBuffer(inputQ, 1, tilingData.vectorChunk * sizeof(float));
            pipe.InitBuffer(accumQ, 1, tilingData.vectorChunk * sizeof(float));
            pipe.InitBuffer(outputQ, 1, tilingData.vectorChunk * sizeof(InputType));
        }
    }
    __aicore__ inline void Process()
    {
        for (uint64_t vocabOffset = uint64_t(group) * tilingData.vocabTile; vocabOffset < tilingData.vocabSize;
             vocabOffset += uint64_t(tilingData.coreNum) * tilingData.vocabTile) {
            uint32_t vocabCount = tilingData.vocabSize - vocabOffset < tilingData.vocabTile ?
                                      tilingData.vocabSize - vocabOffset :
                                      tilingData.vocabTile;
            uint32_t pitch = (vocabCount + 15) / 16 * 16;
            for (uint64_t tokenOffset = 0; tokenOffset < tilingData.numTokens; tokenOffset += tilingData.tokenTile) {
                uint32_t tokenCount = tilingData.numTokens - tokenOffset < tilingData.tokenTile ?
                                          tilingData.numTokens - tokenOffset :
                                          tilingData.tokenTile;
                if ASCEND_IS_AIC {
                    ComputeLogits(tokenOffset, vocabOffset, tokenCount, vocabCount);
                    CrossCoreSetFlag<2, PIPE_FIX>(LOGITS_READY);
                    CrossCoreWaitFlag(G_READY);
                    ComputeWeight(tokenOffset, tokenCount, vocabCount, pitch);
                    CrossCoreSetFlag<2, PIPE_FIX>(PARTIAL_READY);
                    CrossCoreWaitFlag(SLOT_RELEASED);
                }
                if ASCEND_IS_AIV {
                    CrossCoreWaitFlag(LOGITS_READY);
                    MakeG(tokenOffset, vocabOffset, tokenCount, vocabCount, pitch);
                    CrossCoreSetFlag<2, PIPE_MTE3>(G_READY);
                    CrossCoreWaitFlag(PARTIAL_READY);
                    // One output writer per group also avoids unaligned row boundary races.
                    if (lane == 0) {
                        ReduceWeight(tokenOffset, vocabOffset, tokenCount, vocabCount);
                    }
                    CrossCoreSetFlag<2, PIPE_MTE3>(SLOT_RELEASED);
                }
            }
        }
        if ASCEND_IS_AIC {
            // End releases queue events; each object stays live across all token/vocab windows.
            logitsMatmul.End();
            gradWeightMatmul.End();
        }
    }

private:
    __aicore__ inline void ComputeLogits(uint64_t tokenOffset, uint64_t vocabOffset, uint32_t tokenCount,
                                         uint32_t vocabCount)
    {
        logitsMatmul.SetOrgShape(tokenCount, vocabCount, tilingData.hiddenDim);
        logitsMatmul.SetSingleShape(tokenCount, vocabCount, tilingData.hiddenDim);
        logitsMatmul.SetTensorA(hiddenGm[tokenOffset * tilingData.hiddenDim], false);
        logitsMatmul.SetTensorB(weightGm[vocabOffset * tilingData.hiddenDim], true);
        logitsMatmul.IterateAll(logits);
    }
    __aicore__ inline void ComputeWeight(uint64_t tokenOffset, uint32_t tokenCount, uint32_t vocabCount, uint32_t pitch)
    {
        // Transposed A is physically [nc,pitch]; C is physically [vc,H].
        gradWeightMatmul.SetOrgShape(pitch, tilingData.hiddenDim, tokenCount);
        gradWeightMatmul.SetSingleShape(vocabCount, tilingData.hiddenDim, tokenCount);
        gradWeightMatmul.SetTensorA(logitsGradGm, true);
        gradWeightMatmul.SetTensorB(hiddenGm[tokenOffset * tilingData.hiddenDim], false);
        gradWeightMatmul.IterateAll(partial);
    }
    __aicore__ inline void MakeG(uint64_t tokenOffset, uint64_t vocabOffset, uint32_t tokenCount, uint32_t vocabCount,
                                 uint32_t pitch)
    {
        const float normalizationScale = tilingData.alpha;
        for (uint32_t row = lane; row < tokenCount; row += 2) {
            const float maximum = softmaxMaxGm.GetValue(tokenOffset + row);
            const float inverseSum = 1.0f / softmaxSumGm.GetValue(tokenOffset + row);
            const int64_t label = labelsGm.GetValue(tokenOffset + row);
            const bool ignored = label == tilingData.ignoreIndex;
            const int64_t target = ignored ? -1 : label - static_cast<int64_t>(vocabOffset);
            auto values = inputQ.template AllocTensor<float>();
            DataCopyExtParams read{1, uint32_t(vocabCount * sizeof(float)), 0, 0, 0};
            DataCopyPad(values, logits[uint64_t(row) * vocabCount], read, DataCopyPadExtParams<float>{false, 0, 0, 0});
            inputQ.EnQue(values);
            values = inputQ.template DeQue<float>();
            Adds(values, values, -maximum, vocabCount);
            PipeBarrier<PIPE_V>();
            Exp(values, values, vocabCount);
            PipeBarrier<PIPE_V>();
            Muls(values, values, inverseSum, vocabCount);
            if (target >= 0 && target < vocabCount) {
                auto vs = pipe.FetchEventID<HardEvent::V_S>();
                SetFlag<HardEvent::V_S>(vs);
                WaitFlag<HardEvent::V_S>(vs);
                values.SetValue(target, values.GetValue(target) - 1.0f);
                auto sv = pipe.FetchEventID<HardEvent::S_V>();
                SetFlag<HardEvent::S_V>(sv);
                WaitFlag<HardEvent::S_V>(sv);
            }
            PipeBarrier<PIPE_V>();
            Muls(values, values, normalizationScale, vocabCount);
            if (ignored) {
                PipeBarrier<PIPE_V>();
                Duplicate(values, 0.0f, vocabCount);
                PipeBarrier<PIPE_V>();
            }
            auto low = outputQ.template AllocTensor<InputType>();
            Duplicate(low, InputType(0), pitch);
            PipeBarrier<PIPE_V>();
            Cast(low, values, RoundMode::CAST_RINT, vocabCount);
            outputQ.EnQue(low);
            low = outputQ.template DeQue<InputType>();
            DataCopy(logitsGradGm[uint64_t(row) * pitch], low, pitch);
            outputQ.FreeTensor(low);
            inputQ.FreeTensor(values);
        }
    }
    __aicore__ inline void ReduceWeight(uint64_t tokenOffset, uint64_t vocabOffset, uint32_t tokenCount,
                                        uint32_t vocabCount)
    {
        const bool final = tokenOffset + tokenCount == tilingData.numTokens;
        const float upstreamScale = final ? upstream.GetValue(0) : 1.0f;
        for (uint32_t row = 0; row < vocabCount; ++row) {
            for (uint64_t col = 0; col < tilingData.hiddenDim; col += tilingData.vectorChunk) {
                uint32_t count = tilingData.hiddenDim - col < tilingData.vectorChunk ? tilingData.hiddenDim - col :
                                                                                       tilingData.vectorChunk;
                uint64_t offset = uint64_t(row) * tilingData.hiddenDim + col;
                DataCopyExtParams copy{1, uint32_t(count * sizeof(float)), 0, 0, 0};
                auto partialTile = inputQ.template AllocTensor<float>();
                DataCopyPad(partialTile, partial[offset], copy, DataCopyPadExtParams<float>{false, 0, 0, 0});
                inputQ.EnQue(partialTile);
                partialTile = inputQ.template DeQue<float>();
                if (tokenOffset != 0) {
                    auto previousPartial = accumQ.template AllocTensor<float>();
                    DataCopyPad(previousPartial, accum[offset], copy, DataCopyPadExtParams<float>{false, 0, 0, 0});
                    accumQ.EnQue(previousPartial);
                    previousPartial = accumQ.template DeQue<float>();
                    Add(partialTile, partialTile, previousPartial, count);
                    accumQ.FreeTensor(previousPartial);
                }
                if (final) {
                    PipeBarrier<PIPE_V>();
                    Muls(partialTile, partialTile, upstreamScale, count);
                    auto low = outputQ.template AllocTensor<InputType>();
                    PipeBarrier<PIPE_V>();
                    Cast(low, partialTile, RoundMode::CAST_RINT, count);
                    outputQ.EnQue(low);
                    low = outputQ.template DeQue<InputType>();
                    DataCopyExtParams write{1, uint32_t(count * sizeof(InputType)), 0, 0, 0};
                    DataCopyPad(gradWeightGm[(vocabOffset + row) * tilingData.hiddenDim + col], low, write);
                    outputQ.FreeTensor(low);
                } else {
                    auto vm = pipe.FetchEventID<HardEvent::V_MTE3>();
                    SetFlag<HardEvent::V_MTE3>(vm);
                    WaitFlag<HardEvent::V_MTE3>(vm);
                    DataCopyPad(accum[offset], partialTile, copy);
                    auto mv = pipe.FetchEventID<HardEvent::MTE3_V>();
                    SetFlag<HardEvent::MTE3_V>(mv);
                    WaitFlag<HardEvent::MTE3_V>(mv);
                }
                inputQ.FreeTensor(partialTile);
            }
        }
    }
    Tiling tilingData;
    uint32_t group, lane;
    TPipe pipe;
    LogitsMatmul<InputType> logitsMatmul;
    WeightMatmul<InputType> gradWeightMatmul;
    GlobalTensor<InputType> hiddenGm, weightGm, logitsGradGm, gradWeightGm;
    GlobalTensor<int64_t> labelsGm;
    GlobalTensor<float> softmaxMaxGm, softmaxSumGm, upstream, logits, partial, accum;
    TQue<QuePosition::VECIN, 1> inputQ, accumQ;
    TQue<QuePosition::VECOUT, 1> outputQ;
};
} // namespace ChunkLossGradNative
