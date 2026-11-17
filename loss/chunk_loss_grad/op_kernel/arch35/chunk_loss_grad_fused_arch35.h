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
#include "chunk_loss_grad_cube.h"

namespace ChunkLossGradOpt {
using namespace AscendC;
struct Chunk {
    uint64_t vocabOffset, tokenOffset;
    uint32_t vocabCount, tokenCount, tokenPitch;
    bool first, final;
};

// The public kernel owns mode2 IDs 2..9. The selected Blaze Basic block uses
// only local mutexes, not cross-core counters. Each event has one ID per slot.
class GroupSync {
public:
    __aicore__ inline void LogitsReady(uint32_t slot) { CrossCoreSetFlag<2, PIPE_FIX>(2 + slot); }
    __aicore__ inline void WaitLogits(uint32_t slot) { CrossCoreWaitFlag<2, PIPE_S>(2 + slot); }
    __aicore__ inline void GradientReady(uint32_t slot) { CrossCoreSetFlag<2, PIPE_MTE3>(4 + slot); }
    __aicore__ inline void WaitGradient(uint32_t slot) { CrossCoreWaitFlag<2, PIPE_S>(4 + slot); }
    __aicore__ inline void PartialReady(uint32_t slot) { CrossCoreSetFlag<2, PIPE_FIX>(6 + slot); }
    __aicore__ inline void WaitPartial(uint32_t slot) { CrossCoreWaitFlag<2, PIPE_S>(6 + slot); }
    // P is reusable once both AIVs finish reading/accumulating it. Output UB is
    // independent of P; final GM output has its own MTE3 drain before exit.
    __aicore__ inline void PartialFree(uint32_t slot) { CrossCoreSetFlag<2, PIPE_V>(8 + slot); }
    __aicore__ inline void WaitFree(uint32_t slot) { CrossCoreWaitFlag<2, PIPE_S>(8 + slot); }
};

template <class InputType, class Tiling>
class Kernel {
public:
    __aicore__ inline void Init(GM_ADDR hidden, GM_ADDR weight, GM_ADDR labels, GM_ADDR maximum, GM_ADDR sum,
                                GM_ADDR grad, GM_ADDR output, GM_ADDR workspace, const Tiling& tiling)
    {
        tilingData = tiling;
        hiddenAddr = hidden;
        weightAddr = weight;
        group = GetBlockIdx();
        lane = 0;
        if ASCEND_IS_AIV {
            group = GetBlockIdx() / 2;
            lane = GetBlockIdx() % 2;
        }
        base = GetUserWorkspace(workspace) + uint64_t(group) * tilingData.coreWorkspaceBytes;
        labelsGm.SetGlobalBuffer((__gm__ int64_t*)labels);
        softmaxMaxGm.SetGlobalBuffer((__gm__ float*)maximum);
        softmaxSumGm.SetGlobalBuffer((__gm__ float*)sum);
        upstream.SetGlobalBuffer((__gm__ float*)grad);
        gradWeightGm.SetGlobalBuffer((__gm__ InputType*)output);
        for (uint32_t slot = 0; slot < tilingData.slotCount; ++slot) {
            logits[slot].SetGlobalBuffer((__gm__ float*)(base + slot * tilingData.logitsBytes),
                                         tilingData.logitsBytes / sizeof(float));
            logitsGradGm[slot].SetGlobalBuffer(
                (__gm__ InputType*)(base + tilingData.slotCount * tilingData.logitsBytes +
                                    slot * tilingData.logitsGradBytes),
                tilingData.logitsGradBytes / sizeof(InputType));
            partial[slot].SetGlobalBuffer(
                (__gm__ float*)(base + tilingData.slotCount * (tilingData.logitsBytes + tilingData.logitsGradBytes) +
                                slot * tilingData.partialBytes),
                tilingData.partialBytes / sizeof(float));
        }
        accumGm.SetGlobalBuffer(
            (__gm__ float*)(base + tilingData.slotCount * (tilingData.logitsBytes + tilingData.logitsGradBytes +
                                                           tilingData.partialBytes)));
        nChunks = (tilingData.numTokens + tilingData.tokenTile - 1) / tilingData.tokenTile;
        const uint64_t tasks = (tilingData.vocabSize + tilingData.vocabTile - 1) / tilingData.vocabTile;
        chunks = group < tasks ? ((tasks - 1 - group) / tilingData.coreNum + 1) * nChunks : 0;
        if ASCEND_IS_AIV {
            InitVector();
        }
    }

    __aicore__ inline void Process()
    {
        if (chunks == 0) {
            return;
        }
        if (tilingData.scheduleMode == 2 && chunks > 1) {
            ProcessPipeline();
        } else {
            ProcessSerial();
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    __aicore__ inline void InitVector()
    {
        const uint32_t rows = (tilingData.vocabTile + 1) / 2;
        pipe.InitBuffer(logitBuf, rows * tilingData.cacheElements * sizeof(float));
        pipe.InitBuffer(gBuf, rows * tilingData.tokenPitch * sizeof(InputType));
        pipe.InitBuffer(maxBuf, tilingData.cacheElements * sizeof(float));
        pipe.InitBuffer(sumBuf, tilingData.cacheElements * sizeof(float));
        pipe.InitBuffer(labelBuf, tilingData.cacheElements * sizeof(int64_t));
        pipe.InitBuffer(label32Buf, tilingData.cacheElements * sizeof(int32_t));
        pipe.InitBuffer(partialBuf, (tilingData.vectorChunk + 64) * sizeof(float));
        pipe.InitBuffer(outputBuf, tilingData.vectorChunk * sizeof(InputType));
        if (tilingData.residentAccum) {
            pipe.InitBuffer(accumBuf, (tilingData.accumElements + 64) * sizeof(float));
        } else if (nChunks > 1) {
            pipe.InitBuffer(accumBuf, (tilingData.vectorChunk + 64) * sizeof(float));
        }
        pipe.InitBuffer(paramBuf, 64);
        Duplicate(logitBuf.template Get<float>(), 0.0f, rows * tilingData.cacheElements);
        Duplicate(maxBuf.template Get<float>(), 0.0f, tilingData.cacheElements);
        Duplicate(sumBuf.template Get<float>(), 1.0f, tilingData.cacheElements);
        Duplicate(labelBuf.template Get<float>(), 0.0f, tilingData.cacheElements * 2);
        Duplicate(label32Buf.template Get<int32_t>(), 0, tilingData.cacheElements);
        Duplicate(partialBuf.template Get<float>(), 0.0f, tilingData.vectorChunk + 64);
        if (tilingData.residentAccum || nChunks > 1) {
            Duplicate(accumBuf.template Get<float>(), 0.0f,
                      (tilingData.residentAccum ? tilingData.accumElements : tilingData.vectorChunk) + 64);
        }
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        auto params = paramBuf.template Get<float>();
        DataCopyExtParams copy{1, sizeof(float), 0, 0, 0};
        DataCopyPad(params[8], upstream, copy, DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(0);
        WaitFlag<HardEvent::MTE2_S>(0);
        // One-time scalar setup from UB, not per-row GM scalar access.
        alphaValue = tilingData.alpha;
        upstreamScale = params.GetValue(8);
        deferred = alphaValue > 1.0f || alphaValue < (1.0f / 65504.0f);
    }
    __aicore__ inline Chunk GetChunk(uint64_t chunkIndex)
    {
        const uint64_t task = group + (chunkIndex / nChunks) * tilingData.coreNum;
        const uint64_t vocabOffset = task * tilingData.vocabTile;
        const uint64_t tokenOffset = (chunkIndex % nChunks) * tilingData.tokenTile;
        const uint32_t vocabCount = tilingData.vocabSize - vocabOffset < tilingData.vocabTile ?
                                        tilingData.vocabSize - vocabOffset :
                                        tilingData.vocabTile;
        const uint32_t tokenCount = tilingData.numTokens - tokenOffset < tilingData.tokenTile ?
                                        tilingData.numTokens - tokenOffset :
                                        tilingData.tokenTile;
        // Keep physical G rows and writer ownership stable across N tails.
        return {vocabOffset,
                tokenOffset,
                vocabCount,
                tokenCount,
                tilingData.tokenPitch,
                tokenOffset == 0,
                tokenOffset + tokenCount == tilingData.numTokens};
    }
    __aicore__ inline void ComputeLogits(uint64_t chunkIndex, uint32_t slot)
    {
        const auto chunk = GetChunk(chunkIndex);
        logitsCube.Compute(weightAddr + chunk.vocabOffset * tilingData.hiddenDim * sizeof(InputType),
                           hiddenAddr + chunk.tokenOffset * tilingData.hiddenDim * sizeof(InputType),
                           (GM_ADDR)logits[slot].GetPhyAddr(), chunk.vocabCount, chunk.tokenCount, tilingData.hiddenDim,
                           tilingData.hiddenDim, CubeBaseM());
    }
    __aicore__ inline void ComputeWeight(uint64_t chunkIndex, uint32_t slot)
    {
        const auto chunk = GetChunk(chunkIndex);
        gradWeightCube.Compute((GM_ADDR)logitsGradGm[slot].GetPhyAddr(),
                               hiddenAddr + chunk.tokenOffset * tilingData.hiddenDim * sizeof(InputType),
                               (GM_ADDR)partial[slot].GetPhyAddr(), chunk.vocabCount, tilingData.hiddenDim,
                               chunk.tokenCount, chunk.tokenPitch, CubeBaseM());
    }
    __aicore__ inline uint32_t CubeBaseM() const
    {
        // The tiny-H batching branch keeps its existing M16 backend. Only
        // the host's large dense partition selects a larger Cube base shape.
        return tilingData.numTokens >= 128 && tilingData.hiddenDim >= 128 && tilingData.vocabSize >= 1024 ?
                   tilingData.vocabTile :
                   16;
    }
    __aicore__ inline void ProcessSerial()
    {
        for (int64_t chunkIndex = 0; chunkIndex < int64_t(chunks); ++chunkIndex) {
            const uint32_t slot = chunkIndex % tilingData.slotCount;
            if ASCEND_IS_AIC {
                ComputeLogits(chunkIndex, slot);
                sync.LogitsReady(slot);
                sync.WaitGradient(slot);
                ComputeWeight(chunkIndex, slot);
                sync.PartialReady(slot);
                sync.WaitFree(slot);
            }
            if ASCEND_IS_AIV {
                sync.WaitLogits(slot);
                MakeG(chunkIndex, slot);
                sync.GradientReady(slot);
                sync.WaitPartial(slot);
                Reduce(chunkIndex, slot);
                sync.PartialFree(slot);
            }
        }
    }
    __aicore__ inline void ProcessPipeline()
    {
        if ASCEND_IS_AIC {
            ComputeLogits(0, 0);
            sync.LogitsReady(0);
            for (int64_t chunkIndex = 0; chunkIndex < int64_t(chunks); ++chunkIndex) {
                if (chunkIndex + 1 < chunks) {
                    ComputeLogits(chunkIndex + 1, (chunkIndex + 1) % 2);
                    sync.LogitsReady((chunkIndex + 1) % 2);
                }
                sync.WaitGradient(chunkIndex % 2);
                if (chunkIndex >= 2) {
                    sync.WaitFree((chunkIndex - 2) % 2);
                }
                ComputeWeight(chunkIndex, chunkIndex % 2);
                sync.PartialReady(chunkIndex % 2);
            }
            for (int64_t chunkIndex = chunks > 2 ? int64_t(chunks) - 2 : 0; chunkIndex < int64_t(chunks);
                 ++chunkIndex) {
                sync.WaitFree(chunkIndex % 2);
            }
        }
        if ASCEND_IS_AIV {
            for (int64_t chunkIndex = 0; chunkIndex < int64_t(chunks); ++chunkIndex) {
                sync.WaitLogits(chunkIndex % 2);
                MakeG(chunkIndex, chunkIndex % 2);
                sync.GradientReady(chunkIndex % 2);
                if (chunkIndex != 0) {
                    sync.WaitPartial((chunkIndex - 1) % 2);
                    Reduce(chunkIndex - 1, (chunkIndex - 1) % 2);
                    sync.PartialFree((chunkIndex - 1) % 2);
                }
            }
            // Process() has already rejected the empty group.
            const int64_t last = int64_t(chunks) - 1;
            sync.WaitPartial(last % 2);
            Reduce(last, last % 2);
            sync.PartialFree(last % 2);
        }
    }
    __aicore__ inline void MakeG(uint64_t chunkIndex, uint32_t slot)
    {
        const auto chunk = GetChunk(chunkIndex);
        const uint32_t half = (tilingData.vocabTile + 1) / 2;
        const uint32_t first = lane * half;
        const uint32_t rows = first < chunk.vocabCount ?
                                  (chunk.vocabCount - first < half ? chunk.vocabCount - first : half) :
                                  0;
        if (rows == 0) {
            return;
        }
        // Reuse of input UB waits for all preceding VF readers. Reuse of g UB
        // separately waits for its previous DMA reader (G slots in GM differ).
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        auto values = logitBuf.template Get<float>();
        auto maxima = maxBuf.template Get<float>();
        auto sums = sumBuf.template Get<float>();
        auto labels = labelBuf.template Get<int64_t>();
        auto labels32 = label32Buf.template Get<int32_t>();
        DataCopyExtParams matrix{
            uint16_t(rows), uint32_t(chunk.tokenCount * sizeof(float)), 0,
            uint32_t((tilingData.cacheElements * sizeof(float) - (chunk.tokenCount * sizeof(float) + 31) / 32 * 32) /
                     32),
            0};
        DataCopyPad(values, logits[slot][uint64_t(first) * chunk.tokenCount], matrix,
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        DataCopyExtParams cache{1, uint32_t(chunk.tokenCount * sizeof(float)), 0, 0, 0};
        DataCopyPad(maxima, softmaxMaxGm[chunk.tokenOffset], cache, DataCopyPadExtParams<float>{false, 0, 0, 0});
        DataCopyPad(sums, softmaxSumGm[chunk.tokenOffset], cache, DataCopyPadExtParams<float>{false, 0, 0, 0});
        cache.blockLen = chunk.tokenCount * sizeof(int64_t);
        DataCopyPad(labels, labelsGm[chunk.tokenOffset], cache, DataCopyPadExtParams<int64_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        NarrowLabels((__ubuf__ int32_t*)labels32.GetPhyAddr(), (__ubuf__ int64_t*)labels.GetPhyAddr(), chunk.tokenCount,
                     tilingData.ignoreIndex);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
        auto low = gBuf.template Get<InputType>();
        const float gradientScale = deferred ? 1.0f : alphaValue;
        if (IsSameType<InputType, bfloat16_t>::value) {
            MakeGradientBlock<InputType, true>(
                (__ubuf__ InputType*)low.GetPhyAddr(), (__ubuf__ float*)values.GetPhyAddr(),
                (__ubuf__ float*)maxima.GetPhyAddr(), (__ubuf__ float*)sums.GetPhyAddr(),
                (__ubuf__ int32_t*)labels32.GetPhyAddr(), rows, chunk.tokenCount, tilingData.cacheElements,
                chunk.tokenPitch, gradientScale, chunk.vocabOffset + first);
        } else {
            MakeGradientBlock<InputType, false>(
                (__ubuf__ InputType*)low.GetPhyAddr(), (__ubuf__ float*)values.GetPhyAddr(),
                (__ubuf__ float*)maxima.GetPhyAddr(), (__ubuf__ float*)sums.GetPhyAddr(),
                (__ubuf__ int32_t*)labels32.GetPhyAddr(), rows, chunk.tokenCount, tilingData.cacheElements,
                chunk.tokenPitch, gradientScale, chunk.vocabOffset + first);
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyExtParams write{1, uint32_t(rows * chunk.tokenPitch * sizeof(InputType)), 0, 0, 0};
        DataCopyPad(logitsGradGm[slot][uint64_t(first) * chunk.tokenPitch], low, write);
    }
    template <bool add, bool final>
    __aicore__ inline void Accumulate(LocalTensor<float> partialTile, LocalTensor<float> acc,
                                      LocalTensor<InputType> low, uint32_t count)
    {
        if (deferred) {
            AccumulateBlock<InputType, add, final, true>(
                (__ubuf__ float*)partialTile.GetPhyAddr(), (__ubuf__ float*)acc.GetPhyAddr(),
                (__ubuf__ InputType*)low.GetPhyAddr(), count, upstreamScale, alphaValue);
        } else {
            AccumulateBlock<InputType, add, final, false>(
                (__ubuf__ float*)partialTile.GetPhyAddr(), (__ubuf__ float*)acc.GetPhyAddr(),
                (__ubuf__ InputType*)low.GetPhyAddr(), count, upstreamScale, alphaValue);
        }
    }
    __aicore__ inline void Reduce(uint64_t chunkIndex, uint32_t slot)
    {
        const auto chunk = GetChunk(chunkIndex);
        const uint64_t total = uint64_t(chunk.vocabCount) * tilingData.hiddenDim;
        // GM accumulation reuses the same allocation for successive V tasks.
        // A short V tail must not transfer its lower part to the other AIV.
        const uint64_t partition = tilingData.residentAccum ? total :
                                                              uint64_t(tilingData.vocabTile) * tilingData.hiddenDim;
        const uint64_t split = ((partition + 1) / 2 + 15) / 16 * 16;
        const uint64_t cut = split < total ? split : total;
        const uint64_t begin = lane == 0 ? 0 : cut;
        const uint64_t end = lane == 0 ? cut : total;
        auto partialTile = partialBuf.template Get<float>();
        // Single-window nonresident route instantiates only <addOld=false,
        // final=true>; the VF never dereferences its accumulator parameter.
        // Alias a valid P buffer instead of allocating unused UB storage.
        auto accumulator = tilingData.residentAccum || nChunks > 1 ? accumBuf.template Get<float>() : partialTile;
        auto low = outputBuf.template Get<InputType>();
        for (uint64_t offset = begin; offset < end; offset += tilingData.vectorChunk) {
            const uint32_t count = end - offset < tilingData.vectorChunk ? end - offset : tilingData.vectorChunk;
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
            if (!tilingData.residentAccum) {
                SetFlag<HardEvent::MTE3_MTE2>(0);
                WaitFlag<HardEvent::MTE3_MTE2>(0);
            }
            DataCopyExtParams read{1, uint32_t(count * sizeof(float)), 0, 0, 0};
            DataCopyPad(partialTile, partial[slot][offset], read, DataCopyPadExtParams<float>{false, 0, 0, 0});
            auto acc = tilingData.residentAccum ? accumulator[offset - begin] : accumulator;
            if (!tilingData.residentAccum && !chunk.first) {
                DataCopyPad(acc, accumGm[offset], read, DataCopyPadExtParams<float>{false, 0, 0, 0});
            }
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
            if (chunk.final) {
                if (chunk.first) {
                    Accumulate<false, true>(partialTile, acc, low, count);
                } else {
                    Accumulate<true, true>(partialTile, acc, low, count);
                }
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                DataCopyExtParams write{1, uint32_t(count * sizeof(InputType)), 0, 0, 0};
                DataCopyPad(gradWeightGm[chunk.vocabOffset * tilingData.hiddenDim + offset], low, write);
            } else {
                if (chunk.first) {
                    Accumulate<false, false>(partialTile, acc, low, count);
                } else {
                    Accumulate<true, false>(partialTile, acc, low, count);
                }
                if (!tilingData.residentAccum) {
                    SetFlag<HardEvent::V_MTE3>(0);
                    WaitFlag<HardEvent::V_MTE3>(0);
                    DataCopyPad(accumGm[offset], acc, read);
                }
            }
        }
    }
    Tiling tilingData;
    TPipe pipe;
    GroupSync sync;
    Cube<InputType, true> logitsCube;
    Cube<InputType, false> gradWeightCube;
    GM_ADDR hiddenAddr;
    GM_ADDR weightAddr;
    GM_ADDR base;
    uint32_t group, lane;
    uint64_t chunks, nChunks;
    float alphaValue = 1.0f, upstreamScale = 1.0f;
    bool deferred = false;
    GlobalTensor<float> logits[2], partial[2], accumGm, softmaxMaxGm, softmaxSumGm, upstream;
    GlobalTensor<int64_t> labelsGm;
    GlobalTensor<InputType> logitsGradGm[2], gradWeightGm;
    TBuf<QuePosition::VECCALC> logitBuf, gBuf, maxBuf, sumBuf, labelBuf, label32Buf;
    TBuf<QuePosition::VECCALC> partialBuf, outputBuf, accumBuf, paramBuf;
};
} // namespace ChunkLossGradOpt
