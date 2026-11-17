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
#include "chunk_loss_grad_direct_block.h"
#include "chunk_loss_grad_direct_vector.h"
#include "chunk_loss_grad_direct_options.h"

namespace ChunkLossGradDirect {
using namespace AscendC;
constexpr uint32_t CV_N = 128;
constexpr uint32_t P_H = 256;
constexpr uint32_t G_L1_BASE = 256 * 1024;

template <bool separate>
class TransferCounter {
public:
    __aicore__ inline uint32_t Next(bool forP)
    {
        if constexpr (separate) {
            return forP ? partialTile++ % 2 : l++ % 2;
        }
        return shared++ % 2;
    }

private:
    uint32_t shared = 0, l = 0, partialTile = 0;
};

class DirectSync {
public:
    __aicore__ inline void TileReady(uint32_t softmaxSumGm) { CrossCoreSetFlag<2, PIPE_FIX>(2 + softmaxSumGm); }
    __aicore__ inline void WaitTile(uint32_t softmaxSumGm) { CrossCoreWaitFlag<2, PIPE_V>(2 + softmaxSumGm); }
    __aicore__ inline void TileFree(uint32_t softmaxSumGm) { CrossCoreSetFlag<2, PIPE_V>(10 + softmaxSumGm); }
    __aicore__ inline void WaitTileFree(uint32_t softmaxSumGm) { CrossCoreWaitFlag<2, PIPE_FIX>(10 + softmaxSumGm); }
    __aicore__ inline void GReady(uint32_t softmaxSumGm) { CrossCoreSetFlag<2, PIPE_MTE3>(4 + softmaxSumGm); }
    __aicore__ inline void WaitG(uint32_t softmaxSumGm) { CrossCoreWaitFlag<2, PIPE_MTE1>(4 + softmaxSumGm); }
    __aicore__ inline void PReady(uint32_t softmaxSumGm) { CrossCoreSetFlag<2, PIPE_FIX>(6 + softmaxSumGm); }
    __aicore__ inline void WaitP(uint32_t softmaxSumGm) { CrossCoreWaitFlag<2, PIPE_V>(6 + softmaxSumGm); }
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
    static constexpr bool separate = (options & ChunkLossGradOptions::SEPARATE_L_P) != 0;
    static constexpr bool doubleOutput = (options & ChunkLossGradOptions::OUTPUT_DOUBLE_BUFFER) != 0;

public:
    __aicore__ inline void Init(GM_ADDR labels, GM_ADDR maximum, GM_ADDR sum, GM_ADDR grad, GM_ADDR output,
                                const Tiling& data, uint32_t laneId)
    {
        tilingData = data;
        lane = laneId;
        // This first allocation is identical on AIC and both AIVs. AIC's
        // Fixpipe destination uses this logical UB offset, split along M.
        // L and P use the same two slots, with a common transfer sequence.
        // P needs the larger H256 pitch; M0 keeps its N128 row pitch.
        pipe.InitBuffer(sharedBuf, 2 * (tilingData.vocabTile / 2) * P_H * sizeof(float));
        if constexpr (separate) {
            pipe.InitBuffer(logitBuf, 2 * (tilingData.vocabTile / 2) * CV_N * sizeof(float));
        }
        if ASCEND_IS_AIV {
            labelsGm.SetGlobalBuffer((__gm__ int64_t*)labels);
            softmaxMaxGm.SetGlobalBuffer((__gm__ float*)maximum);
            softmaxSumGm.SetGlobalBuffer((__gm__ float*)sum);
            upstream.SetGlobalBuffer((__gm__ float*)grad);
            gradWeightGm.SetGlobalBuffer((__gm__ InputType*)output);
            pipe.InitBuffer(gBuf, (tilingData.vocabTile + 1) * CV_N * sizeof(InputType));
            pipe.InitBuffer(maxBuf, tilingData.cacheElements * sizeof(float));
            pipe.InitBuffer(sumBuf, tilingData.cacheElements * sizeof(float));
            pipe.InitBuffer(labelBuf, tilingData.cacheElements * sizeof(int64_t));
            pipe.InitBuffer(label32Buf, tilingData.cacheElements * sizeof(int32_t));
            pipe.InitBuffer(outputBuf, (doubleOutput ? 2 : 1) * tilingData.vectorChunk * sizeof(InputType));
            pipe.InitBuffer(paramBuf, 64);
            Duplicate(sharedBuf.template Get<float>(), 0.0f, tilingData.vocabTile * P_H);
            if constexpr (separate) {
                Duplicate(logitBuf.template Get<float>(), 0.0f, tilingData.vocabTile * CV_N);
            }
            Duplicate(maxBuf.template Get<float>(), 0.0f, tilingData.cacheElements);
            Duplicate(sumBuf.template Get<float>(), 1.0f, tilingData.cacheElements);
            Duplicate(labelBuf.template Get<float>(), 0.0f, 2 * tilingData.cacheElements);
            Duplicate(label32Buf.template Get<int32_t>(), 0, tilingData.cacheElements);
            Duplicate(outputBuf.template Get<InputType>(), InputType(0),
                      (doubleOutput ? 2 : 1) * tilingData.vectorChunk);
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
            sync.TileFree(0);
            sync.TileFree(1);
            if constexpr (separate) {
                sync.PFree(0);
                sync.PFree(1);
            }
            if constexpr (doubleOutput) {
                ReleaseOutput(0);
                ReleaseOutput(1);
                ReleaseOutput(2);
            }
        }
    }
    __aicore__ inline uint64_t TileAddress(uint32_t slot, bool forP = false)
    {
        if (separate && !forP) {
            return uint64_t(logitBuf.template Get<float>()[slot * (tilingData.vocabTile / 2) * CV_N].GetPhyAddr());
        }
        return uint64_t(sharedBuf.template Get<float>()[slot * (tilingData.vocabTile / 2) * P_H].GetPhyAddr());
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
            const uint32_t slot = transfers.Next(false);
            sync.WaitTile(slot);
            auto maxima = maxBuf.template Get<float>()[tokenOffset];
            auto sums = sumBuf.template Get<float>()[tokenOffset];
            auto labels32 = label32Buf.template Get<int32_t>()[tokenOffset];
            if constexpr (doubleOutput) {
                WaitOutput(2);
            } else {
                SetFlag<HardEvent::MTE3_V>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
            }
            auto values = separate ? logitBuf.template Get<float>()[slot * (tilingData.vocabTile / 2) * CV_N] :
                                     sharedBuf.template Get<float>()[slot * (tilingData.vocabTile / 2) * P_H];
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
            sync.TileFree(slot);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            const uint64_t address = G_L1_BASE + (task % 2) * logitsGradBytes +
                                     uint64_t(tokenOffset) * tilingData.vocabTile * sizeof(InputType) +
                                     first * 16 * sizeof(InputType);
            // Same grouped NZ UB->L1 copy as FAG: lengths/gaps are32B blocks.
            asc_copy_ub2l1((__cbuf__ void*)address, (__ubuf__ void*)nz.GetPhyAddr(), uint16_t(tokenPitch / 16),
                           uint16_t(storageRows), uint16_t(1), uint16_t(tilingData.vocabTile - storageRows));
            if constexpr (doubleOutput) {
                ReleaseOutput(2);
            }
        }
        sync.GReady(task % 2);
    }
    __aicore__ inline void Finalize(uint64_t vocabOffset, uint32_t vocabCount)
    {
        const uint32_t rows = vocabCount / 2;
        const uint64_t first = vocabOffset + lane * rows;
        for (uint32_t h0 = 0; h0 < tilingData.hiddenDim; h0 += P_H) {
            const uint32_t hc = tilingData.hiddenDim - h0 < P_H ? tilingData.hiddenDim - h0 : P_H;
            const uint32_t pitch = (hc + 15) / 16 * 16;
            const uint32_t slot = transfers.Next(true);
            if constexpr (separate) {
                sync.WaitP(slot);
            } else {
                sync.WaitTile(slot);
            }
            const uint32_t outputSlot = doubleOutput ? outputSequence++ % 2 : 0;
            if constexpr (doubleOutput) {
                WaitOutput(outputSlot);
            } else {
                SetFlag<HardEvent::MTE3_V>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
            }
            auto low = outputBuf.template Get<InputType>()[outputSlot * tilingData.vectorChunk];
            auto partialTile = sharedBuf.template Get<float>()[slot * (tilingData.vocabTile / 2) * P_H];
            if (hc == P_H) {
                // Full H256 rows are contiguous in both UB tensors. Use
                // the existing flat VF instead of one inner loop per row.
                const uint32_t count = rows * P_H;
                if (deferred) {
                    ChunkLossGradOpt::AccumulateBlock<InputType, false, true, true>(
                        (__ubuf__ float*)partialTile.GetPhyAddr(), (__ubuf__ float*)partialTile.GetPhyAddr(),
                        (__ubuf__ InputType*)low.GetPhyAddr(), count, upstreamScale, alphaValue);
                } else {
                    ChunkLossGradOpt::AccumulateBlock<InputType, false, true, false>(
                        (__ubuf__ float*)partialTile.GetPhyAddr(), (__ubuf__ float*)partialTile.GetPhyAddr(),
                        (__ubuf__ InputType*)low.GetPhyAddr(), count, upstreamScale, alphaValue);
                }
            } else if (deferred) {
                FinalizeRows<InputType, true>((__ubuf__ InputType*)low.GetPhyAddr(),
                                              (__ubuf__ float*)partialTile.GetPhyAddr(), rows, hc, pitch, upstreamScale,
                                              alphaValue);
            } else {
                FinalizeRows<InputType, false>((__ubuf__ InputType*)low.GetPhyAddr(),
                                               (__ubuf__ float*)partialTile.GetPhyAddr(), rows, hc, pitch,
                                               upstreamScale, alphaValue);
            }
            if constexpr (separate) {
                sync.PFree(slot);
            } else {
                sync.TileFree(slot);
            }
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopyExtParams copyParams{uint16_t(rows), uint32_t(hc * sizeof(InputType)), 0,
                                         uint32_t((tilingData.hiddenDim - hc) * sizeof(InputType)), 0};
            DataCopyPad(gradWeightGm[first * tilingData.hiddenDim + h0], low, copyParams);
            if constexpr (doubleOutput) {
                ReleaseOutput(outputSlot);
            }
        }
    }
    __aicore__ inline void Drain()
    {
        if constexpr (doubleOutput) {
            WaitOutput(0);
            WaitOutput(1);
            WaitOutput(2);
        }
    }

private:
    __aicore__ inline void WaitOutput(uint32_t id) { WaitFlag<HardEvent::MTE3_V>(id); }
    __aicore__ inline void ReleaseOutput(uint32_t id) { SetFlag<HardEvent::MTE3_V>(id); }
    Tiling tilingData;
    TPipe pipe;
    DirectSync sync;
    TransferCounter<separate> transfers;
    uint32_t lane = 0, outputSequence = 0;
    bool sumReciprocalSafe = false;
    float alphaValue = 1.0f, upstreamScale = 1.0f;
    bool deferred = false;
    GlobalTensor<float> softmaxMaxGm, softmaxSumGm, upstream;
    GlobalTensor<int64_t> labelsGm;
    GlobalTensor<InputType> gradWeightGm;
    TBuf<QuePosition::VECCALC> sharedBuf, logitBuf, gBuf, maxBuf, sumBuf, labelBuf, label32Buf, outputBuf, paramBuf;
};

// The universal kernel owns the scheduler, Block and Vector epilogue at the
// same lifetime level. All math remains in this single MIX kernel.
template <class InputType, class Tiling>
class GemmUniversal {
    static constexpr bool separate = IsSameType<InputType, bfloat16_t>::value &&
                                     ((ChunkLossGradOptions::BUILD_MASK & ChunkLossGradOptions::SEPARATE_L_P) != 0);

public:
    __aicore__ inline void Init(GM_ADDR hidden, GM_ADDR weight, GM_ADDR labels, GM_ADDR maximum, GM_ADDR sum,
                                GM_ADDR grad, GM_ADDR output, GM_ADDR workspace, const Tiling& data)
    {
        tilingData = data;
        hiddenAddr = hidden;
        weightAddr = weight;
        scheduler.Init(tilingData);
        vec.Init(labels, maximum, sum, grad, output, tilingData, scheduler.lane);
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
                Weight(i);
            }
            sync.WaitTileFree(0);
            sync.WaitTileFree(1);
            if constexpr (separate) {
                sync.WaitPFree(0);
                sync.WaitPFree(1);
            }
            block.Drain();
        }
        if ASCEND_IS_AIV {
            vec.MakeFullG(0, scheduler.V0(0), scheduler.Rows(0));
            for (uint64_t i = 0; i < scheduler.tasks; ++i) {
                if (i + 1 < scheduler.tasks) {
                    vec.MakeFullG(i + 1, scheduler.V0(i + 1), scheduler.Rows(i + 1));
                }
                vec.Finalize(scheduler.V0(i), scheduler.Rows(i));
            }
            vec.Drain();
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    struct LogitsEpilogue {
        GemmUniversal* owner;
        template <class C>
        __aicore__ inline void operator()(C& l0cTensor, uint32_t tokenCount)
        {
            const uint32_t slot = owner->transfers.Next(false);
            auto l0cLayout = Blaze::Gemm::MakeNDExtLayout<float>(rows, tokenCount, CV_N);
            auto ub = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub>((__ubuf__ float*)owner->vec.TileAddress(slot)), l0cLayout);
            owner->sync.WaitTileFree(slot);
            auto copyParams = asc::te::make_copy(asc::te::copy_l0c_to_ub{}, Blaze::Gemm::Tile::CopyL0C2UBTraitSplitM{});
            asc::te::l0c_to_ub_params params{asc::te::unit_flag_mode::enable_update};
            asc::te::copy(copyParams.with(params), ub, l0cTensor);
            owner->sync.TileReady(slot);
        }
        uint32_t rows;
    };
    __aicore__ inline void ProduceG(uint64_t task)
    {
        const uint64_t vocabOffset = scheduler.V0(task);
        const uint32_t vocabCount = scheduler.Rows(task);
        auto la = Blaze::Gemm::MakeNDExtLayout<InputType>(vocabCount, tilingData.hiddenDim, tilingData.hiddenDim);
        auto ga = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(
                (__gm__ InputType*)(weightAddr + vocabOffset * tilingData.hiddenDim * sizeof(InputType))),
            la);
        if constexpr (ChunkLossGradOptions::BUILD_MASK & ChunkLossGradOptions::REUSE_LOGITS_WEIGHT) {
            if ((tilingData.scheduleMode >> ChunkLossGradOptions::MODE_SHIFT) &
                ChunkLossGradOptions::REUSE_LOGITS_WEIGHT) {
                auto lb = asc::te::frame_layout_format<asc::te::dn_ext_layout_ptn,
                                                       asc::te::layout_trait_default<InputType>>{}(
                    tilingData.hiddenDim, tilingData.numTokens);
                auto gb = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::gm>((__gm__ InputType*)hiddenAddr), lb);
                constexpr uint32_t groupN = (ChunkLossGradOptions::BUILD_MASK &
                                             ChunkLossGradOptions::LOGITS_GROUP_TWO) ?
                                                256 :
                                                512;
                LogitsEpilogue epilogue{this, vocabCount};
                for (uint32_t tokenOffset = 0; tokenOffset < tilingData.numTokens; tokenOffset += groupN) {
                    const uint32_t tokenCount = tilingData.numTokens - tokenOffset < groupN ?
                                                    tilingData.numTokens - tokenOffset :
                                                    groupN;
                    auto b = gb.slice(asc::te::make_coord(0, int64_t(tokenOffset)),
                                      asc::te::make_shape(int64_t(tilingData.hiddenDim), int64_t(tokenCount)));
                    block.template RunLogitsReuse<(ChunkLossGradOptions::BUILD_MASK &
                                                   ChunkLossGradOptions::REUSE_LOGITS_L0A) != 0>(
                        ga, b, vocabCount, tokenCount, tilingData.hiddenDim, epilogue);
                }
                return;
            }
        }
        for (uint32_t tokenOffset = 0; tokenOffset < tilingData.numTokens; tokenOffset += CV_N) {
            const uint32_t tokenCount = tilingData.numTokens - tokenOffset < CV_N ? tilingData.numTokens - tokenOffset :
                                                                                    CV_N;
            const uint32_t slot = transfers.Next(false);
            auto lb = asc::te::frame_layout_format<asc::te::dn_ext_layout_ptn,
                                                   asc::te::layout_trait_default<InputType>>{}(tilingData.hiddenDim,
                                                                                               tokenCount);
            auto gb = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::gm>(
                    (__gm__ InputType*)(hiddenAddr + uint64_t(tokenOffset) * tilingData.hiddenDim * sizeof(InputType))),
                lb);
            auto l0cLayout = Blaze::Gemm::MakeNDExtLayout<float>(vocabCount, tokenCount, CV_N);
            auto ub = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub>((__ubuf__ float*)vec.TileAddress(slot)), l0cLayout);
            sync.WaitTileFree(slot);
            block.template Run<false, true>(ga, gb, ub, vocabCount, tokenCount, tilingData.hiddenDim);
            sync.TileReady(slot);
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
        for (uint32_t h0 = 0; h0 < tilingData.hiddenDim; h0 += P_H) {
            const uint32_t hc = tilingData.hiddenDim - h0 < P_H ? tilingData.hiddenDim - h0 : P_H;
            const uint32_t slot = transfers.Next(true);
            auto b = gb.slice(asc::te::make_coord(0, int64_t(h0)),
                              asc::te::make_shape(int64_t(tilingData.numTokens), int64_t(hc)));
            auto l0cLayout = Blaze::Gemm::MakeNDExtLayout<float>(vocabCount, hc, P_H);
            auto chunk = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub>((__ubuf__ float*)vec.TileAddress(slot, true)), l0cLayout);
            if constexpr (separate) {
                sync.WaitPFree(slot);
            } else {
                sync.WaitTileFree(slot);
            }
            block.template Run<true, true>(ga, b, chunk, vocabCount, hc, tilingData.numTokens);
            if constexpr (separate) {
                sync.PReady(slot);
            } else {
                sync.TileReady(slot);
            }
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
    TransferCounter<separate> transfers;
};
} // namespace ChunkLossGradDirect
