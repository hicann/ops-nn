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
#include "chunk_loss_grad_vector.h"
#include "lib/matmul_intf.h"
namespace ChunkLossGradLegacy {
using namespace ChunkLossGradArch35;
using namespace AscendC;
constexpr uint16_t LOGITS_READY = 4;
constexpr uint16_t G_READY = 5;
constexpr uint16_t PARTIAL_READY = 6;
constexpr uint16_t SLOT_RELEASED = 7;
// Pure-AIC MDL backend; TPipe initializes its synchronization state in the constructor.
constexpr auto MM_CONFIG = GetMDLConfig(false, false, 0, false, false, false, false, true, true, false, false, false);
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

// Each handshake covers the AIC and both AIVs in one MIX group.
// Wait blocks Scalar issue, so all later dependent DMA/compute instructions follow it.
class GroupSync {
public:
    __aicore__ inline void PublishLogits() { CrossCoreSetFlag<2, PIPE_FIX>(LOGITS_READY); }
    __aicore__ inline void WaitLogits() { CrossCoreWaitFlag<2, PIPE_S>(LOGITS_READY); }
    __aicore__ inline void PublishGradient() { CrossCoreSetFlag<2, PIPE_MTE3>(G_READY); }
    __aicore__ inline void WaitGradient() { CrossCoreWaitFlag<2, PIPE_S>(G_READY); }
    __aicore__ inline void PublishPartial() { CrossCoreSetFlag<2, PIPE_FIX>(PARTIAL_READY); }
    __aicore__ inline void WaitPartial() { CrossCoreWaitFlag<2, PIPE_S>(PARTIAL_READY); }
    __aicore__ inline void ReleaseSlot() { CrossCoreSetFlag<2, PIPE_MTE3>(SLOT_RELEASED); }
    __aicore__ inline void WaitSlot() { CrossCoreWaitFlag<2, PIPE_S>(SLOT_RELEASED); }
};

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
            const float alphaValue = tilingData.alpha;
            // Keep g bounded before low-precision Cube input conversion.
            finalAlpha = (alphaValue > 1.0f || alphaValue < (1.0f / 65504.0f)) ? alphaValue : 1.0f;
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
                    sync.PublishLogits();
                    sync.WaitGradient();
                    ComputeWeight(tokenOffset, tokenCount, vocabCount, pitch);
                    sync.PublishPartial();
                    sync.WaitSlot();
                }
                if ASCEND_IS_AIV {
                    sync.WaitLogits();
                    MakeG(tokenOffset, vocabOffset, tokenCount, vocabCount, pitch);
                    sync.PublishGradient();
                    sync.WaitPartial();
                    // One output writer per group also avoids unaligned row boundary races.
                    if (lane == 0) {
                        ReduceWeight(tokenOffset, vocabOffset, tokenCount, vocabCount);
                    }
                    sync.ReleaseSlot();
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
        const float alphaValue = finalAlpha == 1.0f ? tilingData.alpha : 1.0f;
        for (uint32_t row = lane; row < tokenCount; row += 2) {
            const float maximum = softmaxMaxGm.GetValue(tokenOffset + row);
            const float sumValue = softmaxSumGm.GetValue(tokenOffset + row);
            const int64_t label = labelsGm.GetValue(tokenOffset + row);
            const bool ignored = label == tilingData.ignoreIndex;
            const int32_t target = ignored ? -1 : label - static_cast<int64_t>(vocabOffset);
            auto values = inputQ.template AllocTensor<float>();
            DataCopyExtParams read{1, uint32_t(vocabCount * sizeof(float)), 0, 0, 0};
            DataCopyPad(values, logits[uint64_t(row) * vocabCount], read, DataCopyPadExtParams<float>{false, 0, 0, 0});
            inputQ.EnQue(values);
            values = inputQ.template DeQue<float>();
            auto low = outputQ.template AllocTensor<InputType>();
            if (IsSameType<InputType, bfloat16_t>::value || alphaValue > 1.0f) {
                MakeGradientVF<InputType, true>((__ubuf__ InputType*)low.GetPhyAddr(),
                                                (__ubuf__ float*)values.GetPhyAddr(), vocabCount, pitch, maximum,
                                                sumValue, alphaValue, target, ignored);
            } else {
                MakeGradientVF<InputType, false>((__ubuf__ InputType*)low.GetPhyAddr(),
                                                 (__ubuf__ float*)values.GetPhyAddr(), vocabCount, pitch, maximum,
                                                 sumValue, alphaValue, target, ignored);
            }
            outputQ.EnQue(low);
            low = outputQ.template DeQue<InputType>();
            DataCopy(logitsGradGm[uint64_t(row) * pitch], low, pitch);
            outputQ.FreeTensor(low);
            inputQ.FreeTensor(values);
        }
    }
    template <bool addOld, bool final>
    __aicore__ inline void Accumulate(LocalTensor<float> partialTile, LocalTensor<float> previousPartial,
                                      LocalTensor<InputType> low, uint32_t count, float upstreamScale)
    {
        if (final && finalAlpha != 1.0f) {
            AccumulateVF<InputType, addOld, final, true>(
                (__ubuf__ float*)partialTile.GetPhyAddr(), (__ubuf__ float*)previousPartial.GetPhyAddr(),
                (__ubuf__ InputType*)low.GetPhyAddr(), count, upstreamScale, finalAlpha);
        } else {
            AccumulateVF<InputType, addOld, final, false>(
                (__ubuf__ float*)partialTile.GetPhyAddr(), (__ubuf__ float*)previousPartial.GetPhyAddr(),
                (__ubuf__ InputType*)low.GetPhyAddr(), count, upstreamScale, finalAlpha);
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
                auto previousPartial = accumQ.template AllocTensor<float>();
                if (tokenOffset != 0) {
                    DataCopyPad(previousPartial, accum[offset], copy, DataCopyPadExtParams<float>{false, 0, 0, 0});
                    accumQ.EnQue(previousPartial);
                    previousPartial = accumQ.template DeQue<float>();
                }
                auto low = outputQ.template AllocTensor<InputType>();
                if (final) {
                    if (tokenOffset != 0) {
                        Accumulate<true, true>(partialTile, previousPartial, low, count, upstreamScale);
                    } else {
                        Accumulate<false, true>(partialTile, previousPartial, low, count, upstreamScale);
                    }
                    outputQ.EnQue(low);
                    low = outputQ.template DeQue<InputType>();
                    DataCopyExtParams write{1, uint32_t(count * sizeof(InputType)), 0, 0, 0};
                    DataCopyPad(gradWeightGm[(vocabOffset + row) * tilingData.hiddenDim + col], low, write);
                } else {
                    if (tokenOffset != 0) {
                        Accumulate<true, false>(partialTile, previousPartial, low, count, upstreamScale);
                    } else {
                        Accumulate<false, false>(partialTile, previousPartial, low, count, upstreamScale);
                    }
                    auto vm = pipe.FetchEventID<HardEvent::V_MTE3>();
                    SetFlag<HardEvent::V_MTE3>(vm);
                    WaitFlag<HardEvent::V_MTE3>(vm);
                    DataCopyPad(accum[offset], partialTile, copy);
                    auto mv = pipe.FetchEventID<HardEvent::MTE3_V>();
                    SetFlag<HardEvent::MTE3_V>(mv);
                    WaitFlag<HardEvent::MTE3_V>(mv);
                }
                outputQ.FreeTensor(low);
                accumQ.FreeTensor(previousPartial);
                inputQ.FreeTensor(partialTile);
            }
        }
    }
    GroupSync sync;
    Tiling tilingData;
    uint32_t group, lane;
    float finalAlpha = 1.0f;
    TPipe pipe;
    LogitsMatmul<InputType> logitsMatmul;
    WeightMatmul<InputType> gradWeightMatmul;
    GlobalTensor<InputType> hiddenGm, weightGm, logitsGradGm, gradWeightGm;
    GlobalTensor<int64_t> labelsGm;
    GlobalTensor<float> softmaxMaxGm, softmaxSumGm, upstream, logits, partial, accum;
    TQue<QuePosition::VECIN, 1> inputQ, accumQ;
    TQue<QuePosition::VECOUT, 1> outputQ;
};
} // namespace ChunkLossGradLegacy
