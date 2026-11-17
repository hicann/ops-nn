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
namespace ChunkLossGradArch22Opt {
using namespace AscendC;
constexpr uint16_t LOGITS_READY = 8;
constexpr uint16_t G_READY = 9;
constexpr uint16_t PARTIAL_READY = 10;
constexpr uint16_t SLOT_RELEASED = 11;
constexpr uint32_t VOCAB_TILE = 128;
constexpr uint32_t MAX_LANE_ROWS = 256;
constexpr uint32_t BATCH_ROWS = 16;
constexpr uint32_t BATCH_COUNT = MAX_LANE_ROWS / BATCH_ROWS;
constexpr uint32_t STATE_BYTES = MAX_LANE_ROWS * (sizeof(float) * 2 + sizeof(int64_t));
constexpr uint32_t SCALAR_BYTES = 64;
constexpr uint32_t BROADCAST_FLOATS = MAX_LANE_ROWS * 8;
// The one-window path needs no accumulator queue. Its budget can hold both
// fully broadcast coefficient arrays for the complete invocation.
constexpr uint32_t VECTOR_ELEMENTS = BATCH_ROWS * VOCAB_TILE;
constexpr uint32_t OPT_UB_BYTES = 2 * VECTOR_ELEMENTS * (sizeof(float) + sizeof(half)) + STATE_BYTES + SCALAR_BYTES +
                                  BROADCAST_FLOATS * sizeof(float) * 2;
static_assert(OPT_UB_BYTES <= 4096 * (sizeof(float) * 2 + sizeof(half)) + 8192,
              "Vector buffers must fit the UB bound checked by host tiling");
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
        if ASCEND_IS_AIC {
            logitsMatmul.Init(&tilingData.logitsTiling, &pipe);
            gradWeightMatmul.Init(&tilingData.weightTiling, &pipe);
        }
        if ASCEND_IS_AIV {
            pipe.InitBuffer(inputBuf, 2 * VECTOR_ELEMENTS * sizeof(float));
            pipe.InitBuffer(outputBuf, 2 * VECTOR_ELEMENTS * sizeof(InputType));
            pipe.InitBuffer(stateBuf, STATE_BYTES + SCALAR_BYTES);
            pipe.InitBuffer(broadcastBuf, BROADCAST_FLOATS * sizeof(float) * 2);
            InitEvents();
            CacheVectorState();
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
            const uint32_t tokenCount = static_cast<uint32_t>(tilingData.numTokens);
            if ASCEND_IS_AIC {
                ComputeLogits(0, vocabOffset, tokenCount, vocabCount);
                CrossCoreSetFlag<2, PIPE_FIX>(LOGITS_READY);
                CrossCoreWaitFlag(G_READY);
                ComputeWeight(0, tokenCount, vocabCount, pitch);
                CrossCoreSetFlag<2, PIPE_FIX>(PARTIAL_READY);
                CrossCoreWaitFlag(SLOT_RELEASED);
            }
            if ASCEND_IS_AIV {
                CrossCoreWaitFlag(LOGITS_READY);
                MakeGBatched(vocabOffset);
                CrossCoreSetFlag<2, PIPE_MTE3>(G_READY);
                CrossCoreWaitFlag(PARTIAL_READY);
                // H%128==0: both lanes own complete 32-byte output blocks.
                ReduceWeight(vocabOffset, vocabCount);
                CrossCoreSetFlag<2, PIPE_MTE3>(SLOT_RELEASED);
            }
        }
        if ASCEND_IS_AIV {
            DrainEvents();
        }
        if ASCEND_IS_AIC {
            // End releases queue events; each object stays live across all token/vocab windows.
            logitsMatmul.End();
            gradWeightMatmul.End();
        }
    }

private:
    __aicore__ inline void InitEvents()
    {
        for (uint32_t slot = 0; slot < 2; ++slot) {
            inputReady[slot] = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_V>());
            inputFree[slot] = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE2>());
            outputReady[slot] = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE3>());
            outputFree[slot] = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_V>());
            SetFlag<HardEvent::V_MTE2>(inputFree[slot]);
            SetFlag<HardEvent::MTE3_V>(outputFree[slot]);
        }
    }

    __aicore__ inline void DrainEvents()
    {
        for (uint32_t slot = 0; slot < 2; ++slot) {
            WaitFlag<HardEvent::V_MTE2>(inputFree[slot]);
            WaitFlag<HardEvent::MTE3_V>(outputFree[slot]);
            pipe.ReleaseEventID<HardEvent::MTE2_V>(inputReady[slot]);
            pipe.ReleaseEventID<HardEvent::V_MTE2>(inputFree[slot]);
            pipe.ReleaseEventID<HardEvent::V_MTE3>(outputReady[slot]);
            pipe.ReleaseEventID<HardEvent::MTE3_V>(outputFree[slot]);
        }
    }

    __aicore__ inline LocalTensor<float> Input(uint32_t slot)
    {
        return inputBuf.GetWithOffset<float>(VECTOR_ELEMENTS, slot * VECTOR_ELEMENTS * sizeof(float));
    }

    __aicore__ inline void Prefetch(uint32_t slot, GlobalTensor<float> source, uint32_t count)
    {
        WaitFlag<HardEvent::V_MTE2>(inputFree[slot]);
        auto value = Input(slot);
        DataCopyExtParams copy{1, uint32_t(count * sizeof(float)), 0, 0, 0};
        DataCopyPad(value, source, copy, DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(inputReady[slot]);
    }

    __aicore__ inline void Store(uint32_t slot, LocalTensor<float>& value, GlobalTensor<InputType> destination,
                                 uint32_t count)
    {
        WaitFlag<HardEvent::MTE3_V>(outputFree[slot]);
        auto low = outputBuf.GetWithOffset<InputType>(VECTOR_ELEMENTS, slot * VECTOR_ELEMENTS * sizeof(InputType));
        Cast(low, value, RoundMode::CAST_RINT, count);
        SetFlag<HardEvent::V_MTE2>(inputFree[slot]);
        SetFlag<HardEvent::V_MTE3>(outputReady[slot]);
        WaitFlag<HardEvent::V_MTE3>(outputReady[slot]);
        DataCopyExtParams copy{1, uint32_t(count * sizeof(InputType)), 0, 0, 0};
        DataCopyPad(destination, low, copy);
        SetFlag<HardEvent::MTE3_V>(outputFree[slot]);
    }

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
    __aicore__ inline void CacheVectorState()
    {
        // Key1 is a single token window, with N/H/V all multiples of 128.
        laneRows = static_cast<uint32_t>(tilingData.numTokens) / 2;
        laneBegin = lane * laneRows;
        auto maximum = stateBuf.GetWithOffset<float>(MAX_LANE_ROWS, 0);
        auto inverseSum = stateBuf.GetWithOffset<float>(MAX_LANE_ROWS, MAX_LANE_ROWS * sizeof(float));
        auto labels = stateBuf.GetWithOffset<int64_t>(MAX_LANE_ROWS, MAX_LANE_ROWS * sizeof(float) * 2);
        auto grad = stateBuf.GetWithOffset<float>(8, STATE_BYTES + 32);
        DataCopyExtParams floats{1, uint32_t(laneRows * sizeof(float)), 0, 0, 0};
        DataCopyExtParams integers{1, uint32_t(laneRows * sizeof(int64_t)), 0, 0, 0};
        DataCopyExtParams scalar{1, sizeof(float), 0, 0, 0};
        DataCopyPad(maximum, softmaxMaxGm[laneBegin], floats, DataCopyPadExtParams<float>{false, 0, 0, 0});
        DataCopyPad(inverseSum, softmaxSumGm[laneBegin], floats, DataCopyPadExtParams<float>{false, 0, 0, 0});
        DataCopyPad(labels, labelsGm[laneBegin], integers, DataCopyPadExtParams<int64_t>{false, 0, 0, 0});
        DataCopyPad(grad, upstream, scalar, DataCopyPadExtParams<float>{false, 0, 0, 0});
        auto ms = pipe.FetchEventID<HardEvent::MTE2_S>();
        SetFlag<HardEvent::MTE2_S>(ms);
        WaitFlag<HardEvent::MTE2_S>(ms);
        normalizationScale = tilingData.alpha;
        outputScale = grad.GetValue(0);
        for (uint32_t batch = 0; batch < BATCH_COUNT; ++batch) {
            targetHeads[batch] = -1;
            ignoredRows[batch] = 0;
        }
        uint32_t targets = 0;
        for (uint32_t row = 0; row < laneRows; ++row) {
            // Setup only: preserve the original scalar FP32 reciprocal.
            inverseSum.SetValue(row, 1.0f / inverseSum.GetValue(row));
            const int64_t label = labels.GetValue(row);
            if (label == tilingData.ignoreIndex) {
                ignoredRows[row / BATCH_ROWS] |= 1U << (row % BATCH_ROWS);
            } else if (label >= 0 && label < static_cast<int64_t>(tilingData.vocabSize)) {
                const uint32_t tile = static_cast<uint32_t>(label) / VOCAB_TILE;
                if (tile % tilingData.coreNum == group) {
                    const uint32_t batch = row / BATCH_ROWS;
                    targetTiles[targets] = tile;
                    targetOffsets[targets] = (row % BATCH_ROWS) * VOCAB_TILE +
                                             static_cast<uint32_t>(label) % VOCAB_TILE;
                    targetNext[targets] = targetHeads[batch];
                    targetHeads[batch] = static_cast<int32_t>(targets++);
                }
            }
        }
        auto sv = pipe.FetchEventID<HardEvent::S_V>();
        SetFlag<HardEvent::S_V>(sv);
        WaitFlag<HardEvent::S_V>(sv);
        auto brcbMax = broadcastBuf.GetWithOffset<float>(BROADCAST_FLOATS, 0);
        auto brcbInverse = broadcastBuf.GetWithOffset<float>(BROADCAST_FLOATS, BROADCAST_FLOATS * sizeof(float));
        Brcb(brcbMax, maximum, uint8_t(laneRows / 8), {1, 8});
        Brcb(brcbInverse, inverseSum, uint8_t(laneRows / 8), {1, 8});
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CorrectTargets(LocalTensor<float>& values, uint32_t batch, uint32_t tile)
    {
        int32_t target = targetHeads[batch];
        while (target >= 0 && targetTiles[target] != tile) {
            target = targetNext[target];
        }
        if (target < 0) {
            return;
        }
        auto vs = pipe.FetchEventID<HardEvent::V_S>();
        SetFlag<HardEvent::V_S>(vs);
        WaitFlag<HardEvent::V_S>(vs);
        while (target >= 0) {
            if (targetTiles[target] == tile) {
                const uint32_t offset = targetOffsets[target];
                values.SetValue(offset, values.GetValue(offset) - 1.0f);
            }
            target = targetNext[target];
        }
        auto sv = pipe.FetchEventID<HardEvent::S_V>();
        SetFlag<HardEvent::S_V>(sv);
        WaitFlag<HardEvent::S_V>(sv);
    }

    __aicore__ inline void MakeGBatched(uint64_t vocabOffset)
    {
        auto brcbMax = broadcastBuf.GetWithOffset<float>(BROADCAST_FLOATS, 0);
        auto brcbInverse = broadcastBuf.GetWithOffset<float>(BROADCAST_FLOATS, BROADCAST_FLOATS * sizeof(float));
        // FP32 block=8 elements; each repeat advances one 128-column row,
        // and src1 advances one block containing eight copies of its scalar.
        const BinaryRepeatParams broadcast{1, 1, 0, VOCAB_TILE / 8, VOCAB_TILE / 8, 1};
        Prefetch(0, logits[uint64_t(laneBegin) * VOCAB_TILE], VECTOR_ELEMENTS);
        for (uint32_t first = 0; first < laneRows; first += BATCH_ROWS) {
            const uint32_t slot = (first / BATCH_ROWS) & 1;
            const uint32_t count = VECTOR_ELEMENTS;
            WaitFlag<HardEvent::MTE2_V>(inputReady[slot]);
            if (first + BATCH_ROWS < laneRows) {
                Prefetch(slot ^ 1, logits[uint64_t(laneBegin + first + BATCH_ROWS) * VOCAB_TILE], count);
            }
            auto values = Input(slot);
            Sub(values, values, brcbMax[first * 8], uint64_t(64), uint8_t(BATCH_ROWS), broadcast);
            Sub(values[64], values[64], brcbMax[first * 8], uint64_t(64), uint8_t(BATCH_ROWS), broadcast);
            PipeBarrier<PIPE_V>();
            Exp(values, values, count);
            PipeBarrier<PIPE_V>();
            Mul(values, values, brcbInverse[first * 8], uint64_t(64), uint8_t(BATCH_ROWS), broadcast);
            Mul(values[64], values[64], brcbInverse[first * 8], uint64_t(64), uint8_t(BATCH_ROWS), broadcast);
            CorrectTargets(values, first / BATCH_ROWS, static_cast<uint32_t>(vocabOffset / VOCAB_TILE));
            PipeBarrier<PIPE_V>();
            Muls(values, values, normalizationScale, count);
            PipeBarrier<PIPE_V>();
            const uint32_t ignored = ignoredRows[first / BATCH_ROWS];
            if (ignored != 0) {
                for (uint32_t row = 0; row < BATCH_ROWS; ++row) {
                    if (ignored & (1U << row)) {
                        Duplicate(values[row * VOCAB_TILE], 0.0f, VOCAB_TILE);
                    }
                }
                PipeBarrier<PIPE_V>();
            }
            Store(slot, values, logitsGradGm[uint64_t(laneBegin + first) * VOCAB_TILE], count);
        }
    }

    __aicore__ inline void ReduceWeight(uint64_t vocabOffset, uint32_t vocabCount)
    {
        const uint32_t rowsPerLane = (vocabCount + 1) / 2;
        const uint32_t firstRow = lane * rowsPerLane;
        const uint32_t endRow = vocabCount < firstRow + rowsPerLane ? vocabCount : firstRow + rowsPerLane;
        const uint64_t begin = uint64_t(firstRow) * tilingData.hiddenDim;
        const uint64_t end = uint64_t(endRow) * tilingData.hiddenDim;
        const uint32_t firstCount = end - begin < VECTOR_ELEMENTS ? end - begin : VECTOR_ELEMENTS;
        Prefetch(0, partial[begin], firstCount);
        uint32_t iteration = 0;
        for (uint64_t offset = begin; offset < end; offset += VECTOR_ELEMENTS, ++iteration) {
            const uint32_t slot = iteration & 1;
            const uint32_t count = end - offset < VECTOR_ELEMENTS ? end - offset : VECTOR_ELEMENTS;
            WaitFlag<HardEvent::MTE2_V>(inputReady[slot]);
            if (offset + VECTOR_ELEMENTS < end) {
                const uint64_t next = offset + VECTOR_ELEMENTS;
                const uint32_t nextCount = end - next < VECTOR_ELEMENTS ? end - next : VECTOR_ELEMENTS;
                Prefetch(slot ^ 1, partial[next], nextCount);
            }
            auto value = Input(slot);
            Muls(value, value, outputScale, count);
            PipeBarrier<PIPE_V>();
            Store(slot, value, gradWeightGm[vocabOffset * tilingData.hiddenDim + offset], count);
        }
    }

    Tiling tilingData;
    uint32_t group, lane;
    uint32_t laneBegin, laneRows;
    float normalizationScale, outputScale;
    int32_t targetHeads[BATCH_COUNT];
    uint32_t ignoredRows[BATCH_COUNT];
    int32_t targetNext[MAX_LANE_ROWS];
    uint32_t targetTiles[MAX_LANE_ROWS], targetOffsets[MAX_LANE_ROWS];
    TPipe pipe;
    LogitsMatmul<InputType> logitsMatmul;
    WeightMatmul<InputType> gradWeightMatmul;
    GlobalTensor<InputType> hiddenGm, weightGm, logitsGradGm, gradWeightGm;
    GlobalTensor<int64_t> labelsGm;
    GlobalTensor<float> softmaxMaxGm, softmaxSumGm, upstream, logits, partial;
    event_t inputReady[2], inputFree[2], outputReady[2], outputFree[2];
    TBuf<TPosition::VECCALC> inputBuf, outputBuf;
    TBuf<TPosition::VECCALC> stateBuf, broadcastBuf;
};
} // namespace ChunkLossGradArch22Opt
