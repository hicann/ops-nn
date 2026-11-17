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
#include "chunk_loss_grad_direct.h"

namespace ChunkLossGradPGm {
using namespace AscendC;
using ChunkLossGradDirect::DirectBlock;
using ChunkLossGradDirect::MakeGradientNZ;
using ChunkLossGradDirect::MakeGradientNZReuseMetadata;
constexpr uint32_t CV_N = 128;
constexpr uint32_t G_L1_BASE = 256 * 1024;

class DirectSync {
public:
    __aicore__ inline void LReady(uint32_t softmaxSumGm) { CrossCoreSetFlag<2, PIPE_FIX>(2 + softmaxSumGm); }
    __aicore__ inline void WaitL(uint32_t softmaxSumGm) { CrossCoreWaitFlag<2, PIPE_V>(2 + softmaxSumGm); }
    __aicore__ inline void LFree(uint32_t softmaxSumGm) { CrossCoreSetFlag<2, PIPE_V>(10 + softmaxSumGm); }
    __aicore__ inline void WaitLFree(uint32_t softmaxSumGm) { CrossCoreWaitFlag<2, PIPE_FIX>(10 + softmaxSumGm); }
    __aicore__ inline void GReady(uint32_t softmaxSumGm) { CrossCoreSetFlag<2, PIPE_MTE3>(4 + softmaxSumGm); }
    __aicore__ inline void WaitG(uint32_t softmaxSumGm) { CrossCoreWaitFlag<2, PIPE_MTE1>(4 + softmaxSumGm); }
    __aicore__ inline void PReady(uint32_t softmaxSumGm) { CrossCoreSetFlag<2, PIPE_FIX>(6 + softmaxSumGm); }
    __aicore__ inline void WaitP(uint32_t softmaxSumGm) { CrossCoreWaitFlag<2, PIPE_MTE2>(6 + softmaxSumGm); }
    __aicore__ inline void PFree(uint32_t softmaxSumGm) { CrossCoreSetFlag<2, PIPE_V>(8 + softmaxSumGm); }
    __aicore__ inline void WaitPFree(uint32_t softmaxSumGm) { CrossCoreWaitFlag<2, PIPE_FIX>(8 + softmaxSumGm); }
};

template <class Tiling>
class BlockScheduler {
public:
    __aicore__ inline void Init(const Tiling& data)
    {
        tilingData = data;
        group = GetBlockIdx();
        lane = 0;
        if ASCEND_IS_AIV {
            group = GetBlockIdx() / 2;
            lane = GetBlockIdx() % 2;
        }
        const uint64_t total = (tilingData.vocabSize + tilingData.vocabTile - 1) / tilingData.vocabTile;
        tasks = group < total ? (total - 1 - group) / tilingData.coreNum + 1 : 0;
    }
    __aicore__ inline uint64_t V0(uint64_t i) const { return (group + i * tilingData.coreNum) * tilingData.vocabTile; }
    __aicore__ inline uint32_t Rows(uint64_t i) const
    {
        const uint64_t rest = tilingData.vocabSize - V0(i);
        return rest < tilingData.vocabTile ? rest : tilingData.vocabTile;
    }
    uint32_t group = 0, lane = 0;
    uint64_t tasks = 0;

private:
    Tiling tilingData;
};

template <class InputType, class Tiling>
class VectorStage {
    static constexpr uint32_t options = IsSameType<InputType, bfloat16_t>::value ?
                                            ChunkLossGradOptions::BUILD_MASK :
                                            (ChunkLossGradOptions::BUILD_MASK & ChunkLossGradOptions::REUSE_METADATA);

public:
    __aicore__ inline void Init(GM_ADDR labels, GM_ADDR maximum, GM_ADDR sum, GM_ADDR grad, GM_ADDR output,
                                GM_ADDR partialBase, const Tiling& data, uint32_t laneId)
    {
        tilingData = data;
        lane = laneId;
        // This first allocation is identical on AIC and both AIVs. AIC's
        // Fixpipe destination uses this logical UB offset, split along M.
        pipe.InitBuffer(logitBuf, 2 * (tilingData.vocabTile / 2) * CV_N * sizeof(float));
        if ASCEND_IS_AIV {
            labelsGm.SetGlobalBuffer((__gm__ int64_t*)labels);
            softmaxMaxGm.SetGlobalBuffer((__gm__ float*)maximum);
            softmaxSumGm.SetGlobalBuffer((__gm__ float*)sum);
            upstream.SetGlobalBuffer((__gm__ float*)grad);
            gradWeightGm.SetGlobalBuffer((__gm__ InputType*)output);
            pBase = partialBase;
            pipe.InitBuffer(gBuf, (tilingData.vocabTile + 1) * CV_N * sizeof(InputType));
            pipe.InitBuffer(maxBuf, tilingData.cacheElements * sizeof(float));
            pipe.InitBuffer(sumBuf, tilingData.cacheElements * sizeof(float));
            pipe.InitBuffer(labelBuf, tilingData.cacheElements * sizeof(int64_t));
            pipe.InitBuffer(label32Buf, tilingData.cacheElements * sizeof(int32_t));
            pipe.InitBuffer(partialBuf, (tilingData.vectorChunk + 64) * sizeof(float));
            pipe.InitBuffer(outputBuf, tilingData.vectorChunk * sizeof(InputType));
            pipe.InitBuffer(paramBuf, 64);
            Duplicate(logitBuf.template Get<float>(), 0.0f, tilingData.vocabTile * CV_N);
            Duplicate(maxBuf.template Get<float>(), 0.0f, tilingData.cacheElements);
            Duplicate(sumBuf.template Get<float>(), 1.0f, tilingData.cacheElements);
            Duplicate(labelBuf.template Get<float>(), 0.0f, 2 * tilingData.cacheElements);
            Duplicate(label32Buf.template Get<int32_t>(), 0, tilingData.cacheElements);
            Duplicate(partialBuf.template Get<float>(), 0.0f, tilingData.vectorChunk + 64);
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
            auto params = paramBuf.template Get<float>();
            DataCopyExtParams copyParams{1, 4, 0, 0, 0};
            DataCopyPad(params[8], upstream, copyParams, DataCopyPadExtParams<float>{false, 0, 0, 0});
            // The complete call's immutable metadata fits in UB. Load once
            // per AIV, then all V tasks reuse slices of this cache.
            copyParams.blockLen = tilingData.numTokens * sizeof(float);
            DataCopyPad(maxBuf.template Get<float>(), softmaxMaxGm, copyParams,
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
            DataCopyPad(sumBuf.template Get<float>(), softmaxSumGm, copyParams,
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
            copyParams.blockLen = tilingData.numTokens * sizeof(int64_t);
            DataCopyPad(labelBuf.template Get<int64_t>(), labelsGm, copyParams,
                        DataCopyPadExtParams<int64_t>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            ChunkLossGradOpt::NarrowLabels((__ubuf__ int32_t*)label32Buf.template Get<int32_t>().GetPhyAddr(),
                                           (__ubuf__ int64_t*)labelBuf.template Get<int64_t>().GetPhyAddr(),
                                           tilingData.numTokens, tilingData.ignoreIndex);
            SetFlag<HardEvent::MTE2_S>(0);
            WaitFlag<HardEvent::MTE2_S>(0);
            alphaValue = tilingData.alpha;
            upstreamScale = params.GetValue(8);
            deferred = alphaValue > 1.0f || alphaValue < (1.0f / 65504.0f);
            if constexpr (options & ChunkLossGradOptions::HOIST_SUM_RECIPROCAL) {
                // A valid normalized forward cache has sum in [1,V]. Limit
                // the reciprocal candidate to this finite, normal range;
                // arbitrary tiny/large/nonfinite sums retain exact Div.
                sumReciprocalSafe = true;
                auto cachedSum = sumBuf.template Get<float>();
                for (uint32_t i = 0; i < tilingData.numTokens; ++i) {
                    const float denominator = cachedSum.GetValue(i);
                    sumReciprocalSafe = sumReciprocalSafe && denominator >= 1.0f &&
                                        denominator <= float(tilingData.vocabSize);
                }
            }
            sync.LFree(0);
            sync.LFree(1);
        }
    }
    __aicore__ inline uint64_t LogitAddress(uint32_t slot)
    {
        return uint64_t(logitBuf.template Get<float>()[slot * (tilingData.vocabTile / 2) * CV_N].GetPhyAddr());
    }
    __aicore__ inline void MakeFullG(uint64_t task, uint64_t vocabOffset, uint32_t vocabCount)
    {
        const uint32_t half = vocabCount / 2;
        const uint32_t first = lane * half;
        const uint32_t storageRows = lane == 0 ? half : tilingData.vocabTile - half;
        const uint64_t logitsGradBytes = (uint64_t(tilingData.vocabTile) * tilingData.tokenPitch * sizeof(InputType) +
                                          511) /
                                         512 * 512;
        for (uint32_t tokenOffset = 0; tokenOffset < tilingData.numTokens; tokenOffset += CV_N) {
            const uint32_t tokenCount = tilingData.numTokens - tokenOffset < CV_N ? tilingData.numTokens - tokenOffset :
                                                                                    CV_N;
            const uint32_t tokenPitch = (tokenCount + 15) / 16 * 16;
            const uint32_t slot = fragment++ % 2;
            sync.WaitL(slot);
            auto maxima = maxBuf.template Get<float>()[tokenOffset];
            auto sums = sumBuf.template Get<float>()[tokenOffset];
            auto labels32 = label32Buf.template Get<int32_t>()[tokenOffset];
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
            auto values = logitBuf.template Get<float>()[slot * (tilingData.vocabTile / 2) * CV_N];
            auto nz = gBuf.template Get<InputType>();
            const float gradientScale = deferred ? 1.0f : alphaValue;
            if constexpr (options & ChunkLossGradOptions::REUSE_METADATA) {
                if (IsSameType<InputType, bfloat16_t>::value) {
                    if constexpr (options & ChunkLossGradOptions::HOIST_SUM_RECIPROCAL) {
                        if (sumReciprocalSafe) {
                            MakeGradientNZReuseMetadata<InputType, true, true>(
                                (__ubuf__ InputType*)nz.GetPhyAddr(), (__ubuf__ float*)values.GetPhyAddr(),
                                (__ubuf__ float*)maxima.GetPhyAddr(), (__ubuf__ float*)sums.GetPhyAddr(),
                                (__ubuf__ int32_t*)labels32.GetPhyAddr(), half, storageRows, tokenCount, tokenPitch,
                                gradientScale, vocabOffset + first);
                        } else {
                            MakeGradientNZReuseMetadata<InputType, true, false>(
                                (__ubuf__ InputType*)nz.GetPhyAddr(), (__ubuf__ float*)values.GetPhyAddr(),
                                (__ubuf__ float*)maxima.GetPhyAddr(), (__ubuf__ float*)sums.GetPhyAddr(),
                                (__ubuf__ int32_t*)labels32.GetPhyAddr(), half, storageRows, tokenCount, tokenPitch,
                                gradientScale, vocabOffset + first);
                        }
                    } else {
                        MakeGradientNZReuseMetadata<InputType, true, false>(
                            (__ubuf__ InputType*)nz.GetPhyAddr(), (__ubuf__ float*)values.GetPhyAddr(),
                            (__ubuf__ float*)maxima.GetPhyAddr(), (__ubuf__ float*)sums.GetPhyAddr(),
                            (__ubuf__ int32_t*)labels32.GetPhyAddr(), half, storageRows, tokenCount, tokenPitch,
                            gradientScale, vocabOffset + first);
                    }
                } else {
                    MakeGradientNZReuseMetadata<InputType, false>(
                        (__ubuf__ InputType*)nz.GetPhyAddr(), (__ubuf__ float*)values.GetPhyAddr(),
                        (__ubuf__ float*)maxima.GetPhyAddr(), (__ubuf__ float*)sums.GetPhyAddr(),
                        (__ubuf__ int32_t*)labels32.GetPhyAddr(), half, storageRows, tokenCount, tokenPitch,
                        gradientScale, vocabOffset + first);
                }
            } else if (IsSameType<InputType, bfloat16_t>::value) {
                MakeGradientNZ<InputType, true>(
                    (__ubuf__ InputType*)nz.GetPhyAddr(), (__ubuf__ float*)values.GetPhyAddr(),
                    (__ubuf__ float*)maxima.GetPhyAddr(), (__ubuf__ float*)sums.GetPhyAddr(),
                    (__ubuf__ int32_t*)labels32.GetPhyAddr(), half, storageRows, tokenCount, tokenPitch, gradientScale,
                    vocabOffset + first);
            } else {
                MakeGradientNZ<InputType, false>(
                    (__ubuf__ InputType*)nz.GetPhyAddr(), (__ubuf__ float*)values.GetPhyAddr(),
                    (__ubuf__ float*)maxima.GetPhyAddr(), (__ubuf__ float*)sums.GetPhyAddr(),
                    (__ubuf__ int32_t*)labels32.GetPhyAddr(), half, storageRows, tokenCount, tokenPitch, gradientScale,
                    vocabOffset + first);
            }
            sync.LFree(slot);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            const uint64_t address = G_L1_BASE + (task % 2) * logitsGradBytes +
                                     uint64_t(tokenOffset) * tilingData.vocabTile * sizeof(InputType) +
                                     first * 16 * sizeof(InputType);
            // Same grouped NZ UB->L1 copy as FAG: lengths/gaps are32B blocks.
            asc_copy_ub2l1((__cbuf__ void*)address, (__ubuf__ void*)nz.GetPhyAddr(), uint16_t(tokenPitch / 16),
                           uint16_t(storageRows), uint16_t(1), uint16_t(tilingData.vocabTile - storageRows));
        }
        sync.GReady(task % 2);
    }
    __aicore__ inline void Reduce(uint64_t task, uint64_t vocabOffset, uint32_t vocabCount)
    {
        sync.WaitP(task % 2);
        GlobalTensor<float> pgm;
        pgm.SetGlobalBuffer((__gm__ float*)(pBase + (task % 2) * tilingData.partialBytes), tilingData.partialBytes / 4);
        const uint64_t total = uint64_t(vocabCount) * tilingData.hiddenDim;
        const uint64_t split = (((uint64_t(tilingData.vocabTile) * tilingData.hiddenDim + 1) / 2 + 15) / 16) * 16;
        const uint64_t cut = split < total ? split : total;
        const uint64_t begin = lane == 0 ? 0 : cut, end = lane == 0 ? cut : total;
        auto partialTile = partialBuf.template Get<float>();
        auto low = outputBuf.template Get<InputType>();
        for (uint64_t off = begin; off < end; off += tilingData.vectorChunk) {
            const uint32_t count = end - off < tilingData.vectorChunk ? end - off : tilingData.vectorChunk;
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
            SetFlag<HardEvent::MTE3_MTE2>(0);
            WaitFlag<HardEvent::MTE3_MTE2>(0);
            DataCopyExtParams copyParams{1, count * 4, 0, 0, 0};
            DataCopyPad(partialTile, pgm[off], copyParams, DataCopyPadExtParams<float>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
            if (deferred) {
                ChunkLossGradOpt::AccumulateBlock<InputType, false, true, true>(
                    (__ubuf__ float*)partialTile.GetPhyAddr(), (__ubuf__ float*)partialTile.GetPhyAddr(),
                    (__ubuf__ InputType*)low.GetPhyAddr(), count, upstreamScale, alphaValue);
            } else {
                ChunkLossGradOpt::AccumulateBlock<InputType, false, true, false>(
                    (__ubuf__ float*)partialTile.GetPhyAddr(), (__ubuf__ float*)partialTile.GetPhyAddr(),
                    (__ubuf__ InputType*)low.GetPhyAddr(), count, upstreamScale, alphaValue);
            }
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            copyParams.blockLen = count * sizeof(InputType);
            DataCopyPad(gradWeightGm[vocabOffset * tilingData.hiddenDim + off], low, copyParams);
        }
        sync.PFree(task % 2);
    }

private:
    Tiling tilingData;
    TPipe pipe;
    DirectSync sync;
    uint32_t lane = 0, fragment = 0;
    float alphaValue = 1.0f, upstreamScale = 1.0f;
    bool deferred = false;
    bool sumReciprocalSafe = false;
    GM_ADDR pBase;
    GlobalTensor<float> softmaxMaxGm, softmaxSumGm, upstream;
    GlobalTensor<int64_t> labelsGm;
    GlobalTensor<InputType> gradWeightGm;
    TBuf<QuePosition::VECCALC> logitBuf, gBuf, maxBuf, sumBuf, labelBuf, label32Buf, partialBuf, outputBuf, paramBuf;
};

// The universal kernel owns the scheduler, Block and Vector epilogue at the
// same lifetime level. All math remains in this single MIX kernel.
template <class InputType, class Tiling>
class GemmUniversal {
public:
    __aicore__ inline void Init(GM_ADDR hidden, GM_ADDR weight, GM_ADDR labels, GM_ADDR maximum, GM_ADDR sum,
                                GM_ADDR grad, GM_ADDR output, GM_ADDR workspace, const Tiling& data)
    {
        tilingData = data;
        hiddenAddr = hidden;
        weightAddr = weight;
        scheduler.Init(tilingData);
        base = GetUserWorkspace(workspace) + uint64_t(scheduler.group) * tilingData.coreWorkspaceBytes;
        vec.Init(labels, maximum, sum, grad, output, base, tilingData, scheduler.lane);
    }
    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC {
            ProduceG(0);
            for (uint64_t i = 0; i < scheduler.tasks; ++i) {
                if (i + 1 < scheduler.tasks) {
                    ProduceG(i + 1);
                }
                sync.WaitG(i % 2);
                if (i >= 2) {
                    sync.WaitPFree(i % 2);
                }
                Weight(i);
                sync.PReady(i % 2);
            }
            for (uint64_t i = scheduler.tasks > 2 ? scheduler.tasks - 2 : 0; i < scheduler.tasks; ++i) {
                sync.WaitPFree(i % 2);
            }
            sync.WaitLFree(0);
            sync.WaitLFree(1);
            block.Drain();
        }
        if ASCEND_IS_AIV {
            for (uint64_t i = 0; i < scheduler.tasks; ++i) {
                vec.MakeFullG(i, scheduler.V0(i), scheduler.Rows(i));
                if (i) {
                    vec.Reduce(i - 1, scheduler.V0(i - 1), scheduler.Rows(i - 1));
                }
            }
            const uint64_t last = scheduler.tasks - 1;
            vec.Reduce(last, scheduler.V0(last), scheduler.Rows(last));
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    __aicore__ inline void ProduceG(uint64_t task)
    {
        const uint64_t vocabOffset = scheduler.V0(task);
        const uint32_t vocabCount = scheduler.Rows(task);
        auto la = Blaze::Gemm::MakeNDExtLayout<InputType>(vocabCount, tilingData.hiddenDim, tilingData.hiddenDim);
        auto ga = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(
                (__gm__ InputType*)(weightAddr + vocabOffset * tilingData.hiddenDim * sizeof(InputType))),
            la);
        for (uint32_t tokenOffset = 0; tokenOffset < tilingData.numTokens; tokenOffset += CV_N) {
            const uint32_t tokenCount = tilingData.numTokens - tokenOffset < CV_N ? tilingData.numTokens - tokenOffset :
                                                                                    CV_N;
            const uint32_t slot = fragment++ % 2;
            auto lb = asc::te::frame_layout_format<asc::te::dn_ext_layout_ptn,
                                                   asc::te::layout_trait_default<InputType>>{}(tilingData.hiddenDim,
                                                                                               tokenCount);
            auto gb = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::gm>(
                    (__gm__ InputType*)(hiddenAddr + uint64_t(tokenOffset) * tilingData.hiddenDim * sizeof(InputType))),
                lb);
            auto l0cLayout = Blaze::Gemm::MakeNDExtLayout<float>(vocabCount, tokenCount, CV_N);
            auto ub = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub>((__ubuf__ float*)vec.LogitAddress(slot)), l0cLayout);
            sync.WaitLFree(slot);
            block.template Run<false, true>(ga, gb, ub, vocabCount, tokenCount, tilingData.hiddenDim);
            sync.LReady(slot);
        }
        block.Drain();
    }
    __aicore__ inline void Weight(uint64_t task)
    {
        const uint32_t vocabCount = scheduler.Rows(task);
        const uint64_t bytes = (uint64_t(tilingData.vocabTile) * tilingData.tokenPitch * sizeof(InputType) + 511) /
                               512 * 512;
        auto la = asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::layout_trait_default<InputType>>{}(
            tilingData.vocabTile, tilingData.tokenPitch);
        auto ga = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, InputType>(G_L1_BASE + (task % 2) * bytes), la);
        auto lb = Blaze::Gemm::MakeNDExtLayout<InputType>(tilingData.numTokens, tilingData.hiddenDim,
                                                          tilingData.hiddenDim);
        auto gb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>((__gm__ InputType*)hiddenAddr), lb);
        auto l0cLayout = Blaze::Gemm::MakeNDExtLayout<float>(vocabCount, tilingData.hiddenDim, tilingData.hiddenDim);
        auto gc = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>((__gm__ float*)(base + (task % 2) * tilingData.partialBytes)),
            l0cLayout);
        for (uint32_t h0 = 0; h0 < tilingData.hiddenDim; h0 += 256) {
            const uint32_t hc = tilingData.hiddenDim - h0 < 256 ? tilingData.hiddenDim - h0 : 256;
            auto b = gb.slice(asc::te::make_coord(0, int64_t(h0)),
                              asc::te::make_shape(int64_t(tilingData.numTokens), int64_t(hc)));
            auto chunk = gc.slice(asc::te::make_coord(0, int64_t(h0)),
                                  asc::te::make_shape(int64_t(vocabCount), int64_t(hc)));
            block.template Run<true, false>(ga, b, chunk, vocabCount, hc, tilingData.numTokens);
        }
        block.Drain();
    }
    Tiling tilingData;
    BlockScheduler<Tiling> scheduler;
    VectorStage<InputType, Tiling> vec;
    DirectBlock<InputType> block;
    DirectSync sync;
    GM_ADDR hiddenAddr;
    GM_ADDR weightAddr;
    GM_ADDR base;
    uint32_t fragment = 0;
};
} // namespace ChunkLossGradPGm
