/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_elements_v2.h
 * \brief
 */
#ifndef SCATTER_ELEMENTS_V2_H
#define SCATTER_ELEMENTS_V2_H
#include "kernel_operator.h"

#define IS_CAST_INT (is_same<U, int64_t>::value)
using namespace AscendC;

namespace ScatterElementsV2NS {
constexpr uint64_t SCATTER_MODE_NONE = 1;
constexpr uint64_t SCATTER_MODE_ADD = 2;
constexpr uint64_t SCATTER_MODE_MUL = 3;
constexpr uint64_t SCATTER_MODE_MIN = 4;
constexpr uint64_t SCATTER_MODE_MAX = 5;
constexpr uint64_t SCATTER_MODE_MEAN = 6;
constexpr uint64_t SCATTER_MODE_REDUCTION_BEGIN = SCATTER_MODE_ADD;
constexpr uint64_t SCATTER_MODE_REDUCTION_END = SCATTER_MODE_MEAN;

__aicore__ inline int64_t FloorDivInt64(int64_t value, int32_t divisor)
{
    if (divisor == 0) {
        return 0;
    }
    int64_t quotient = value / static_cast<int64_t>(divisor);
    int64_t remainder = value % static_cast<int64_t>(divisor);
    if (remainder != 0 && value < 0) {
        --quotient;
    }
    return quotient;
}

template <typename T>
__aicore__ inline T MeanDivideValue(T value, int32_t divisor)
{
    if (divisor == 0) {
        return static_cast<T>(0);
    }
    return value / static_cast<T>(divisor);
}

template <>
__aicore__ inline int8_t MeanDivideValue<int8_t>(int8_t value, int32_t divisor)
{
    return static_cast<int8_t>(FloorDivInt64(static_cast<int64_t>(value), divisor));
}

template <>
__aicore__ inline int16_t MeanDivideValue<int16_t>(int16_t value, int32_t divisor)
{
    return static_cast<int16_t>(FloorDivInt64(static_cast<int64_t>(value), divisor));
}

template <>
__aicore__ inline int32_t MeanDivideValue<int32_t>(int32_t value, int32_t divisor)
{
    return static_cast<int32_t>(FloorDivInt64(static_cast<int64_t>(value), divisor));
}

template <>
__aicore__ inline int64_t MeanDivideValue<int64_t>(int64_t value, int32_t divisor)
{
    return FloorDivInt64(value, divisor);
}

// CPU waits for the vector unit to finish computing.
__aicore__ inline void PIPE_V_S()
{
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
}

// CPU waits for the MTE2 unit to finish moving data.
__aicore__ inline void PIPE_MTE2_S()
{
    event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
}

// MTE3 waits for the CPU to finish computing.
__aicore__ inline void PIPE_S_MTE3()
{
    event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
}
} // namespace ScatterElementsV2NS

template <typename Tp, Tp v>
struct integral_constant {
    static constexpr Tp value = v;
};
using false_type = integral_constant<bool, false>;
using true_type = integral_constant<bool, true>;
template <typename, typename>
struct is_same : public false_type {};
template <typename Tp>
struct is_same<Tp, Tp> : public true_type {};

constexpr int INT32_OFFSET = 31;
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t SMALL_MODE = 1;

template <typename T, typename U>
class KernelScatterElementsV2 {
public:
    __aicore__ inline KernelScatterElementsV2() {}
    __aicore__ inline void Init(const ScatterElementsV2TilingData* __restrict tiling_data, TPipe* tmpPipe,
                                GM_ADDR input, GM_ADDR indices, GM_ADDR updates)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        pipe = tmpPipe;
        coreId = GetBlockIdx();
        LoadTilingData(tiling_data);
        InitGlobalBuffers(input, indices, updates);

        if (modeFlag == SMALL_MODE) {
            InitSmallModeBuffers();
        } else {
            InitScatterModeBuffers(tiling_data);
        }

        InitLocalTensors();
    }

    __aicore__ inline void CopyInIndex(int indicesIndex)
    {
        if constexpr (IS_CAST_INT) {
            DataCopyPadGm2UBImpl((__ubuf__ uint32_t*)indicesLocal.GetPhyAddr(),
                                 (__gm__ uint32_t*)indicesGm[indicesIndex].GetPhyAddr(), indicesExtParams,
                                 padParams); // datacopypad int64
        } else {
            DataCopyPad(indices32Local, indicesGm[indicesIndex], indicesExtParams, uPadParams);
        }
    }

    __aicore__ inline void ScatterSetValue(int k, int kIndex)
    {
        if constexpr (is_same<T, double>::value) {
            inputLocal.SetValue(kIndex, updatesLocal.GetValue(k));
        } else {
            if (mode == 1) {
                inputLocal.SetValue(kIndex, updatesLocal.GetValue(k));
                return;
            }
            bool useUpdateOnly = false;
            if (NeedHitCount()) {
                int hitCount = countLocal.GetValue(kIndex);
                countLocal.SetValue(kIndex, hitCount + 1);
                useUpdateOnly = includeSelf == 0 && hitCount == 0;
            }
            if constexpr (IsCastFloatType()) {
                if (IsCastFloat()) {
                    float inputValue = inputTemp.GetValue(kIndex);
                    float updateValue = updatesTemp.GetValue(k);
                    inputTemp.SetValue(kIndex, ReduceValue<float>(inputValue, updateValue, useUpdateOnly));
                    return;
                }
                return;
            }
            if constexpr (!IsCastFloatType()) {
                T inputValue = inputLocal.GetValue(kIndex);
                T updateValue = updatesLocal.GetValue(k);
                inputLocal.SetValue(kIndex, ReduceValue<T>(inputValue, updateValue, useUpdateOnly));
            }
        }
    }

    __aicore__ inline void ZeroFill(LocalTensor<T>& tensor, uint64_t count)
    {
        if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value || is_same<T, int16_t>::value ||
                      is_same<T, uint16_t>::value || is_same<T, int32_t>::value || is_same<T, uint32_t>::value ||
                      is_same<T, float>::value) {
            Duplicate(tensor, static_cast<T>(0), count);
        } else if constexpr (is_same<T, int8_t>::value || is_same<T, uint8_t>::value) {
            // DAV_C220 does not expose Duplicate for byte tensors. The local
            // allocation is 32-byte aligned, so zeroing its uint16 view is
            // equivalent and keeps initialization on the vector pipeline.
            auto tensorU16 = tensor.template ReinterpretCast<uint16_t>();
            Duplicate(tensorU16, static_cast<uint16_t>(0), count / 2);
        } else if constexpr (is_same<T, int64_t>::value || is_same<T, double>::value) {
            // The same bitwise-zero rule applies to 64-bit integer and FP64.
            auto tensorU32 = tensor.template ReinterpretCast<uint32_t>();
            Duplicate(tensorU32, static_cast<uint32_t>(0), count * 2);
        } else {
            for (uint64_t i = 0; i < count; ++i) {
                tensor.SetValue(i, static_cast<T>(0));
            }
        }
    }

    __aicore__ inline void InitHitCount(uint64_t count)
    {
        if (NeedHitCount()) {
            for (uint64_t i = 0; i < count; ++i) {
                countLocal.SetValue(i, 0);
            }
        }
    }

    __aicore__ inline void CalcMeanValue(uint64_t count)
    {
        if constexpr (!is_same<T, double>::value) {
            if (mode == ScatterElementsV2NS::SCATTER_MODE_MEAN) {
                for (uint64_t i = 0; i < count; ++i) {
                    int hitCount = countLocal.GetValue(i);
                    if (hitCount == 0) {
                        continue;
                    }
                    int divisor = includeSelf != 0 ? hitCount + 1 : hitCount;
                    if constexpr (IsCastFloatType()) {
                        if (IsCastFloat()) {
                            inputTemp.SetValue(
                                i, ScatterElementsV2NS::MeanDivideValue<float>(inputTemp.GetValue(i), divisor));
                            continue;
                        }
                    } else {
                        inputLocal.SetValue(i,
                                            ScatterElementsV2NS::MeanDivideValue<T>(inputLocal.GetValue(i), divisor));
                    }
                }
            }
        }
    }

    // The none reduction is ordered overwrite semantics. Keep it on a direct path so the
    // hot scalar update loop does not carry reduction-mode and hit-count branches.
    __aicore__ inline void ProcessNoneSmall()
    {
        for (uint64_t index = 0; index < indicesLoop; ++index) {
            uint64_t baseIndex = coreId * oneTime + index * indicesEach;
            uint64_t indicesIndex = baseIndex * indicesOneTime;
            uint64_t inputIndex = baseIndex * inputOneTime;
            uint64_t updatesIndex = baseIndex * updatesOneTime;
            uint64_t currentIndices = indicesEach;
            if (index == indicesLoop - 1) {
                currentIndices = indicesLast;
            }
            inputExtParams = {(uint16_t)1, static_cast<uint32_t>(currentIndices * inputOneTime * sizeof(T)), 0, 0, 0};
            indicesExtParams = {(uint16_t)1, static_cast<uint32_t>(currentIndices * indicesOneTime * sizeof(U)), 0, 0,
                                0};
            updatesExtParams = {(uint16_t)1, static_cast<uint32_t>(currentIndices * updatesOneTime * sizeof(T)), 0, 0,
                                0};
            // There is no previous output DMA to wait for on the first tile.
            if (index != 0) {
                // inputLocal is reused by both vector zero-fill and MTE2.
                PipeBarrier<PIPE_ALL>();
            }
            CopyInIndex(indicesIndex);
            DataCopyPad(updatesLocal, updatesGm[updatesIndex], updatesExtParams, tPadParams);
            if (useZeroBase != 0) {
                ZeroFill(inputLocal, inputAlign);
            } else {
                DataCopyPad(inputLocal, inputGm[inputIndex], inputExtParams, tPadParams);
            }
            ScatterElementsV2NS::PIPE_MTE2_S();
            if constexpr (IS_CAST_INT) {
                PIPE_MTE2_V();
                Cast<int, U>(indices32Local, indicesLocal, RoundMode::CAST_NONE, indicesAlign);
                ScatterElementsV2NS::PIPE_V_S();
            }
            if (useZeroBase != 0) {
                ScatterElementsV2NS::PIPE_V_S();
            }
            if constexpr (is_same<T, int32_t>::value) {
                ScatterNoneInt32Rows(static_cast<uint32_t>(currentIndices));
            } else {
                for (uint64_t j = 0; j < currentIndices; ++j) {
                    for (uint64_t k = 0; k < indicesOneTime; ++k) {
                        auto upIndex = j * updatesOneTime + k;
                        auto inIndex = j * inputOneTime + indices32Local.GetValue(j * indicesOneTime + k);
                        inputLocal.SetValue(inIndex, updatesLocal.GetValue(upIndex));
                    }
                }
            }
            ScatterElementsV2NS::PIPE_S_MTE3();
            DataCopyPad(inputGm[inputIndex], inputLocal, inputExtParams);
        }
        FreeLocalTensors();
    }

    __aicore__ inline void ProcessNoneScatter()
    {
        for (uint64_t index = start; index < start + currentNum; ++index) {
            uint64_t inputIndex = index * inputOneTime + currentPiece * inputOnePiece;
            uint64_t indicesIndex = index * indicesOneTime, updatesIndex = index * updatesOneTime;

            for (uint64_t i = 0; i < inputLoop; ++i) {
                PipeBarrier<PIPE_ALL>();
                uint64_t currentInput = pieceEach;
                if (i == inputLoop - 1) {
                    currentInput = pieceLast;
                }
                inputExtParams = {(uint16_t)1, static_cast<uint32_t>(currentInput * sizeof(T)), 0, 0, 0};
                if (useZeroBase != 0) {
                    ZeroFill(inputLocal, inputAlign);
                } else {
                    DataCopyPad(inputLocal, inputGm[inputIndex + i * pieceEach], inputExtParams, tPadParams);
                }

                for (uint64_t j = 0; j < indicesLoop; ++j) {
                    uint64_t currentIndices = indicesEach;
                    if (j == indicesLoop - 1) {
                        currentIndices = indicesLast;
                    }
                    indicesExtParams = {(uint16_t)1, static_cast<uint32_t>(currentIndices * sizeof(U)), 0, 0, 0};
                    updatesExtParams = {(uint16_t)1, static_cast<uint32_t>(currentIndices * sizeof(T)), 0, 0, 0};
                    CopyInIndex(indicesIndex + j * indicesEach);
                    DataCopyPad(updatesLocal, updatesGm[updatesIndex + j * indicesEach], updatesExtParams, tPadParams);
                    ScatterElementsV2NS::PIPE_MTE2_S();
                    PIPE_MTE2_V();
                    if constexpr (IS_CAST_INT) {
                        Cast<int, U>(indices32Local, indicesLocal, RoundMode::CAST_NONE, indicesAlign);
                        PipeBarrier<PIPE_V>();
                    }
                    Adds(indices32Local, indices32Local,
                         static_cast<int>(-i * pieceEach - currentPiece * inputOnePiece),
                         static_cast<int>(indicesAlign));
                    PIPE_V_S();
                    for (uint64_t k = 0; k < currentIndices; ++k) {
                        auto kIndex = indices32Local.GetValue(k);
                        if (kIndex < 0 || kIndex >= currentInput) {
                            continue;
                        }
                        inputLocal.SetValue(kIndex, updatesLocal.GetValue(k));
                    }
                    // Do not overwrite indices/updates while the scalar loop
                    // or the index-adjustment vector operation still uses them.
                    PipeBarrier<PIPE_ALL>();
                }
                ScatterElementsV2NS::PIPE_S_MTE3();
                DataCopyPad(inputGm[inputIndex + i * pieceEach], inputLocal, inputExtParams);
            }
        }
        FreeLocalTensors();
    }

    __aicore__ inline void ProcessNoneStableBucket()
    {
        for (uint64_t index = start; index < start + currentNum; ++index) {
            uint64_t inputIndex = index * inputOneTime + currentPiece * inputOnePiece;
            uint64_t indicesIndex = index * indicesOneTime;
            uint64_t updatesIndex = index * updatesOneTime;
            const int64_t pieceBase = static_cast<int64_t>(currentPiece * inputOnePiece);
            for (uint64_t j = 0; j < indicesLoop; ++j) {
                uint64_t currentIndices = j == indicesLoop - 1 ? indicesLast : indicesEach;
                indicesExtParams = {(uint16_t)1, static_cast<uint32_t>(currentIndices * sizeof(U)), 0, 0, 0};
                updatesExtParams = {(uint16_t)1, static_cast<uint32_t>(currentIndices * sizeof(T)), 0, 0, 0};
                PipeBarrier<PIPE_ALL>();
                CopyInIndex(indicesIndex + j * indicesEach);
                DataCopyPad(updatesLocal, updatesGm[updatesIndex + j * indicesEach], updatesExtParams, tPadParams);
                ScatterElementsV2NS::PIPE_MTE2_S();
                if constexpr (IS_CAST_INT) {
                    PIPE_MTE2_V();
                    Cast<int, U>(indices32Local, indicesLocal, RoundMode::CAST_NONE, indicesAlign);
                    ScatterElementsV2NS::PIPE_V_S();
                }
                Duplicate(bucketOffsetsLocal, 0, inputLoop * 2);
                ScatterElementsV2NS::PIPE_V_S();
                for (uint64_t k = 0; k < currentIndices; ++k) {
                    int64_t localIndex = static_cast<int64_t>(indices32Local.GetValue(k)) - pieceBase;
                    if (localIndex >= 0 && localIndex < static_cast<int64_t>(inputOnePiece)) {
                        uint64_t tile = static_cast<uint64_t>(localIndex) / pieceEach;
                        bucketOffsetsLocal.SetValue(tile, bucketOffsetsLocal.GetValue(tile) + 1);
                    }
                }
                int offset = 0;
                for (uint64_t tile = 0; tile < inputLoop; ++tile) {
                    int count = bucketOffsetsLocal.GetValue(tile);
                    bucketOffsetsLocal.SetValue(tile, offset);
                    bucketOffsetsLocal.SetValue(inputLoop + tile, offset);
                    offset += count;
                }
                for (uint64_t k = 0; k < currentIndices; ++k) {
                    int64_t localIndex = static_cast<int64_t>(indices32Local.GetValue(k)) - pieceBase;
                    if (localIndex >= 0 && localIndex < static_cast<int64_t>(inputOnePiece)) {
                        uint64_t tile = static_cast<uint64_t>(localIndex) / pieceEach;
                        int position = bucketOffsetsLocal.GetValue(inputLoop + tile);
                        // Keep the update ordinal, not a byte offset. This avoids an
                        // unnecessary UB-to-UB gather and is independent of T's width.
                        bucketPositionsLocal.SetValue(position, static_cast<int>(k));
                        bucketOffsetsLocal.SetValue(inputLoop + tile, position + 1);
                    }
                }
                for (uint64_t tile = 0; tile < inputLoop; ++tile) {
                    // Every bucket reuses the same inputLocal allocation.
                    // Complete its previous writeback before loading/zeroing it.
                    PipeBarrier<PIPE_ALL>();
                    int begin = bucketOffsetsLocal.GetValue(tile);
                    int end = bucketOffsetsLocal.GetValue(inputLoop + tile);
                    uint64_t currentInput = tile == inputLoop - 1 ? pieceLast : pieceEach;
                    inputExtParams = {(uint16_t)1, static_cast<uint32_t>(currentInput * sizeof(T)), 0, 0, 0};
                    // Only the first source chunk starts from zero. Later
                    // chunks must preserve updates already written to GM.
                    if (useZeroBase != 0 && j == 0) {
                        ZeroFill(inputLocal, inputAlign);
                        ScatterElementsV2NS::PIPE_V_S();
                    } else {
                        DataCopyPad(inputLocal, inputGm[inputIndex + tile * pieceEach], inputExtParams, tPadParams);
                    }
                    ScatterElementsV2NS::PIPE_MTE2_S();
                    for (int position = begin; position < end; ++position) {
                        int updatePosition = bucketPositionsLocal.GetValue(position);
                        int localIndex = indices32Local.GetValue(updatePosition) - static_cast<int>(pieceBase) -
                                         static_cast<int>(tile * pieceEach);
                        inputLocal.SetValue(localIndex, updatesLocal.GetValue(updatePosition));
                    }
                    ScatterElementsV2NS::PIPE_S_MTE3();
                    DataCopyPad(inputGm[inputIndex + tile * pieceEach], inputLocal, inputExtParams);
                }
            }
        }
        FreeLocalTensors();
    }

    __aicore__ inline void ProcessSmall()
    {
        for (uint64_t index = 0; index < indicesLoop; ++index) {
            uint64_t baseIndex = coreId * oneTime + index * indicesEach;
            uint64_t indicesIndex = baseIndex * indicesOneTime;
            uint64_t inputIndex = baseIndex * inputOneTime;
            uint64_t updatesIndex = baseIndex * updatesOneTime;
            uint64_t currentIndices = indicesEach;
            if (index == indicesLoop - 1) {
                currentIndices = indicesLast;
            }
            inputExtParams = {(uint16_t)1, static_cast<uint32_t>(currentIndices * inputOneTime * sizeof(T)), 0, 0, 0};
            indicesExtParams = {(uint16_t)1, static_cast<uint32_t>(currentIndices * indicesOneTime * sizeof(U)), 0, 0,
                                0};
            updatesExtParams = {(uint16_t)1, static_cast<uint32_t>(currentIndices * updatesOneTime * sizeof(T)), 0, 0,
                                0};
            PIPE_MTE3_MTE2();
            CopyInIndex(indicesIndex);
            DataCopyPad(updatesLocal, updatesGm[updatesIndex], updatesExtParams, tPadParams);
            DataCopyPad(inputLocal, inputGm[inputIndex], inputExtParams, tPadParams);
            CastInputToFloat(inputAlign);
            CastUpdatesToFloat(updatesAlign);
            if constexpr (IS_CAST_INT) {
                PIPE_MTE2_V();
                Cast<int, U>(indices32Local, indicesLocal, RoundMode::CAST_NONE, indicesAlign);
            }
            InitHitCount(inputAlign);
            PipeBarrier<PIPE_ALL>();
            for (uint64_t j = 0; j < currentIndices; ++j) {
                for (uint64_t k = 0; k < indicesOneTime; ++k) {
                    auto upIndex = j * updatesOneTime + k;
                    auto inIndex = j * inputOneTime + indices32Local.GetValue(j * indicesOneTime + k);
                    ScatterSetValue(upIndex, inIndex);
                }
            }
            PipeBarrier<PIPE_ALL>();
            CalcMeanValue(inputAlign);
            CastFloatToInput(inputAlign);
            DataCopyPad(inputGm[inputIndex], inputLocal, inputExtParams);
        }
        FreeLocalTensors();
    }

    __aicore__ inline void ProcessScatter()
    {
        for (uint64_t index = start; index < start + currentNum; ++index) {
            uint64_t inputIndex = index * inputOneTime + currentPiece * inputOnePiece;
            uint64_t indicesIndex = index * indicesOneTime, updatesIndex = index * updatesOneTime;

            for (uint64_t i = 0; i < inputLoop; ++i) {
                PIPE_MTE3_MTE2();
                uint64_t currentInput = pieceEach;
                if (i == inputLoop - 1) {
                    currentInput = pieceLast;
                }
                inputExtParams = {(uint16_t)1, static_cast<uint32_t>(currentInput * sizeof(T)), 0, 0, 0};
                DataCopyPad(inputLocal, inputGm[inputIndex + i * pieceEach], inputExtParams, tPadParams);

                CastInputToFloat(inputAlign);
                InitHitCount(inputAlign);

                for (uint64_t j = 0; j < indicesLoop; ++j) {
                    uint64_t currentIndices = indicesEach;
                    if (j == indicesLoop - 1) {
                        currentIndices = indicesLast;
                    }
                    indicesExtParams = {(uint16_t)1, static_cast<uint32_t>(currentIndices * sizeof(U)), 0, 0, 0};
                    updatesExtParams = {(uint16_t)1, static_cast<uint32_t>(currentIndices * sizeof(T)), 0, 0, 0};
                    CopyInIndex(indicesIndex + j * indicesEach);
                    DataCopyPad(updatesLocal, updatesGm[updatesIndex + j * indicesEach], updatesExtParams, tPadParams);
                    PIPE_MTE2_V();
                    if constexpr (IS_CAST_INT) {
                        Cast<int, U>(indices32Local, indicesLocal, RoundMode::CAST_NONE, indicesAlign);
                        PipeBarrier<PIPE_V>();
                    }
                    CastUpdatesToFloat(updatesAlign);
                    Adds(indices32Local, indices32Local,
                         static_cast<int>(-i * pieceEach - currentPiece * inputOnePiece),
                         static_cast<int>(indicesAlign));
                    PIPE_V_S();
                    for (uint64_t k = 0; k < currentIndices; ++k) {
                        auto kIndex = indices32Local.GetValue(k);
                        if (kIndex < 0 || kIndex >= currentInput) {
                            continue;
                        }
                        ScatterSetValue(k, kIndex);
                    }
                }
                PipeBarrier<PIPE_ALL>();
                CalcMeanValue(inputAlign);
                CastFloatToInput(inputAlign);
                DataCopyPad(inputGm[inputIndex + i * pieceEach], inputLocal, inputExtParams);
            }
        }
        FreeLocalTensors();
    }

    __aicore__ inline void PIPE_MTE3_MTE2()
    {
        int32_t eventIDMTE3ToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    }

    __aicore__ inline void PIPE_MTE2_V()
    {
        int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    }

    __aicore__ inline void PIPE_V_MTE3()
    {
        int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    }

    __aicore__ inline void PIPE_V_S()
    {
        int32_t eventIDVToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIDVToS);
        WaitFlag<HardEvent::V_S>(eventIDVToS);
    }

private:
    __aicore__ inline void ScatterNoneInt32Rows(uint32_t rowCount)
    {
        // The three UB allocations do not alias. Hoist their addresses out of
        // the hot loop and preload independent indices/values in groups of eight.
        // Ordered volatile stores retain last-update-wins for repeated indices;
        // this path does not assume that MaxUnpool indices are unique.
        auto* __restrict dst = reinterpret_cast<__ubuf__ volatile int32_t*>(inputLocal.GetPhyAddr());
        auto* __restrict offsets = reinterpret_cast<__ubuf__ int32_t*>(indices32Local.GetPhyAddr());
        auto* __restrict src = reinterpret_cast<__ubuf__ int32_t*>(updatesLocal.GetPhyAddr());
        const uint32_t width = static_cast<uint32_t>(indicesOneTime);
        constexpr uint32_t UNROLL = 8;
        const uint32_t alignedWidth = width / UNROLL * UNROLL;
        for (uint32_t row = 0; row < rowCount; ++row) {
            uint32_t k = 0;
            for (; k < alignedWidth; k += UNROLL) {
                const int32_t i0 = offsets[k];
                const int32_t i1 = offsets[k + 1];
                const int32_t i2 = offsets[k + 2];
                const int32_t i3 = offsets[k + 3];
                const int32_t i4 = offsets[k + 4];
                const int32_t i5 = offsets[k + 5];
                const int32_t i6 = offsets[k + 6];
                const int32_t i7 = offsets[k + 7];
                const int32_t v0 = src[k];
                const int32_t v1 = src[k + 1];
                const int32_t v2 = src[k + 2];
                const int32_t v3 = src[k + 3];
                const int32_t v4 = src[k + 4];
                const int32_t v5 = src[k + 5];
                const int32_t v6 = src[k + 6];
                const int32_t v7 = src[k + 7];
                dst[i0] = v0;
                dst[i1] = v1;
                dst[i2] = v2;
                dst[i3] = v3;
                dst[i4] = v4;
                dst[i5] = v5;
                dst[i6] = v6;
                dst[i7] = v7;
            }
            for (; k < width; ++k) {
                dst[offsets[k]] = src[k];
            }
            dst += inputOneTime;
            offsets += indicesOneTime;
            src += updatesOneTime;
        }
    }

    __aicore__ inline void LoadTilingData(const ScatterElementsV2TilingData* __restrict tiling_data)
    {
        usedCoreNum = tiling_data->usedCoreNum;
        eachNum = tiling_data->eachNum;
        inputCount = tiling_data->inputCount;
        indicesCount = tiling_data->indicesCount;
        updatesCount = tiling_data->updatesCount;
        inputOneTime = tiling_data->inputOneTime;
        indicesOneTime = tiling_data->indicesOneTime;
        updatesOneTime = tiling_data->updatesOneTime;
        inputLoop = tiling_data->inputLoop;
        indicesLoop = tiling_data->indicesLoop;
        inputEach = tiling_data->inputEach;
        indicesEach = tiling_data->indicesEach;
        inputLast = tiling_data->inputLast;
        indicesLast = tiling_data->indicesLast;
        inputAlign = tiling_data->inputAlign;
        indicesAlign = tiling_data->indicesAlign;
        updatesAlign = tiling_data->updatesAlign;
        inputOnePiece = tiling_data->inputOnePiece;
        modeFlag = tiling_data->modeFlag;
        mode = tiling_data->mode;
        includeSelf = tiling_data->includeSelf;
        lastIndicesLoop = tiling_data->lastIndicesLoop;
        lastIndicesEach = tiling_data->lastIndicesEach;
        lastIndicesLast = tiling_data->lastIndicesLast;
        oneTime = tiling_data->oneTime;
        // M carries the execution plan selected by host tiling.
        useStableBucket = tiling_data->M;
        // include_self=false is the zero-initialized MaxUnpool/none contract.
        useZeroBase = (tiling_data->includeSelf == 0) ? 1 : 0;
    }

    __aicore__ inline void InitGlobalBuffers(GM_ADDR input, GM_ADDR indices, GM_ADDR updates)
    {
        inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input), inputCount);
        indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(indices), indicesCount);
        updatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(updates), updatesCount);
    }

    __aicore__ inline void InitIoBuffers()
    {
        pipe->InitBuffer(inQueueSelf, BUFFER_NUM, inputAlign * sizeof(T));
        if constexpr (IS_CAST_INT) {
            pipe->InitBuffer(inQueueIndics, BUFFER_NUM, indicesAlign * sizeof(U));
        }
        pipe->InitBuffer(inQueueUpdates, BUFFER_NUM, updatesAlign * sizeof(T));
    }

    __aicore__ inline void InitCalcBuffers()
    {
        if constexpr (IsCastFloatType()) {
            if (IsCastFloat()) {
                pipe->InitBuffer(calcSelfBuf, inputAlign * sizeof(float));
                pipe->InitBuffer(calcUpdatesBuf, updatesAlign * sizeof(float));
                inputTemp = calcSelfBuf.Get<float>();
                updatesTemp = calcUpdatesBuf.Get<float>();
            }
        }
        if (NeedHitCount()) {
            pipe->InitBuffer(calcCountBuf, inputAlign * sizeof(int));
            countLocal = calcCountBuf.Get<int>();
        }
    }

    __aicore__ inline void InitSmallModeBuffers()
    {
        indicesLoop = coreId == (usedCoreNum - 1) ? lastIndicesLoop : indicesLoop;
        indicesEach = coreId == (usedCoreNum - 1) ? lastIndicesEach : indicesEach;
        indicesLast = coreId == (usedCoreNum - 1) ? lastIndicesLast : indicesLast;
        inputAlign = (indicesEach * inputOneTime + dataAlign - 1) / dataAlign * dataAlign;
        indicesAlign = (indicesEach * indicesOneTime + dataAlign - 1) / dataAlign * dataAlign;
        updatesAlign = (indicesEach * updatesOneTime + dataAlign - 1) / dataAlign * dataAlign;
        InitIoBuffers();
        if (useStableBucket != 0) {
            pipe->InitBuffer(calcBucketPositionsBuf, indicesAlign * sizeof(int));
            pipe->InitBuffer(calcBucketOffsetsBuf, inputLoop * 2 * sizeof(int));
            bucketPositionsLocal = calcBucketPositionsBuf.Get<int>();
            bucketOffsetsLocal = calcBucketOffsetsBuf.Get<int>();
        }
        InitCalcBuffers();
    }

    __aicore__ inline void InitScatterModeBuffers(const ScatterElementsV2TilingData* __restrict tiling_data)
    {
        pieceEach = inputEach;
        pieceLast = inputLast;
        if (eachNum == 0) {
            uint32_t eachPiece = tiling_data->eachPiece;
            start = coreId / eachPiece;
            currentPiece = coreId % eachPiece;
            currentNum = 1;
            if (currentPiece == eachPiece - 1) {
                auto tmpOnePiece = inputOneTime - inputOnePiece * (eachPiece - 1);
                pieceEach = (tmpOnePiece + inputLoop - 1) / inputLoop;
                pieceLast = tmpOnePiece - pieceEach * (inputLoop - 1);
            }
        } else {
            uint32_t extraTaskCore = tiling_data->extraTaskCore;
            currentPiece = 0;
            currentNum = coreId < extraTaskCore ? (eachNum + 1) : eachNum;
            start = coreId * eachNum + (coreId < extraTaskCore ? coreId : extraTaskCore);
        }
        InitCalcBuffers();
        InitIoBuffers();
        if (useStableBucket != 0) {
            pipe->InitBuffer(calcBucketPositionsBuf, indicesAlign * sizeof(int));
            pipe->InitBuffer(calcBucketOffsetsBuf, inputLoop * 2 * sizeof(int));
            bucketPositionsLocal = calcBucketPositionsBuf.Get<int>();
            bucketOffsetsLocal = calcBucketOffsetsBuf.Get<int>();
        }
    }

    __aicore__ inline void InitLocalTensors()
    {
        pipe->InitBuffer(calcIndices32Buf, indicesAlign * sizeof(int));
        indices32Local = calcIndices32Buf.Get<int>();
        inputLocal = inQueueSelf.AllocTensor<T>();
        if constexpr (IS_CAST_INT) {
            indicesLocal = inQueueIndics.AllocTensor<U>();
        }
        updatesLocal = inQueueUpdates.AllocTensor<T>();
        padParams = {false, 0, 0, 0};
        tPadParams = {false, 0, 0, static_cast<T>(0)};
        uPadParams = {false, 0, 0, static_cast<U>(0)};
    }

    __aicore__ inline void CastInputToFloat(uint64_t count)
    {
        if constexpr (IsCastFloatType()) {
            if (IsCastFloat()) {
                PIPE_MTE2_V();
                Cast(inputTemp, inputLocal, RoundMode::CAST_NONE, count);
            }
        }
    }

    __aicore__ inline void CastUpdatesToFloat(uint64_t count)
    {
        if constexpr (IsCastFloatType()) {
            if (IsCastFloat()) {
                Cast(updatesTemp, updatesLocal, RoundMode::CAST_NONE, count);
            }
        }
    }

    __aicore__ inline void CastFloatToInput(uint64_t count)
    {
        if constexpr (IsCastFloatType()) {
            if (IsCastFloat()) {
                Cast(inputLocal, inputTemp, RoundMode::CAST_RINT, count);
                PIPE_V_MTE3();
            }
        }
    }

    __aicore__ inline void FreeLocalTensors()
    {
        inQueueSelf.FreeTensor(inputLocal);
        if constexpr (IS_CAST_INT) {
            inQueueIndics.FreeTensor(indicesLocal);
        }
        inQueueUpdates.FreeTensor(updatesLocal);
    }

    __aicore__ inline bool IsReduceMode() const
    {
        return mode >= ScatterElementsV2NS::SCATTER_MODE_REDUCTION_BEGIN &&
               mode <= ScatterElementsV2NS::SCATTER_MODE_REDUCTION_END;
    }

    __aicore__ inline bool NeedHitCount() const
    {
        if (!IsReduceMode()) {
            return false;
        }
        // Hit count is required only where the result depends on the number of hits:
        // include_self=false (the first hit must replace the loaded self, later hits
        // accumulate) and mean (division by hit count). include_self=true add/mul/min/max
        // always combine over the already-loaded self, so skip the per-chunk count
        // bookkeeping that dominated wide-row performance.
        return includeSelf == 0 || mode == ScatterElementsV2NS::SCATTER_MODE_MEAN;
    }

    // fp16/bf16 reductions accumulate in fp32 and cast back once (CAST_RINT) for precision.
    // This is independent of hit-count bookkeeping: include_self=true add/mul/min/max need
    // no counts but still require the fp32 intermediate.
    __aicore__ inline bool IsCastFloat() const { return IsCastFloatType() && IsReduceMode(); }

    __aicore__ static constexpr bool IsCastFloatType()
    {
        return is_same<T, half>::value || is_same<T, bfloat16_t>::value;
    }

    template <typename DataType>
    __aicore__ inline DataType ReduceValue(DataType inputValue, DataType updateValue, bool useUpdateOnly) const
    {
        if (useUpdateOnly) {
            return updateValue;
        }
        if (mode == ScatterElementsV2NS::SCATTER_MODE_ADD || mode == ScatterElementsV2NS::SCATTER_MODE_MEAN) {
            return static_cast<DataType>(inputValue + updateValue);
        }
        if (mode == ScatterElementsV2NS::SCATTER_MODE_MUL) {
            return static_cast<DataType>(inputValue * updateValue);
        }
        if (mode == ScatterElementsV2NS::SCATTER_MODE_MIN) {
            return inputValue < updateValue ? inputValue : updateValue;
        }
        if (mode == ScatterElementsV2NS::SCATTER_MODE_MAX) {
            return inputValue > updateValue ? inputValue : updateValue;
        }
        return updateValue;
    }

    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueSelf, inQueueIndics, inQueueUpdates;
    TBuf<QuePosition::VECCALC> calcSelfBuf, calcUpdatesBuf, calcIndices32Buf, calcCountBuf;
    TBuf<QuePosition::VECCALC> calcBucketPositionsBuf, calcBucketOffsetsBuf;
    GlobalTensor<T> inputGm, updatesGm;
    GlobalTensor<U> indicesGm;
    LocalTensor<float> inputTemp, updatesTemp;
    LocalTensor<int> indicesTemp, indices32Local, countLocal, bucketPositionsLocal, bucketOffsetsLocal;
    LocalTensor<U> indicesLocal;
    LocalTensor<T> inputLocal, updatesLocal;
    DataCopyPadExtParams<uint32_t> padParams;
    DataCopyPadExtParams<T> tPadParams;
    DataCopyPadExtParams<U> uPadParams;
    DataCopyExtParams inputExtParams, indicesExtParams, updatesExtParams;
    uint32_t coreId;
    uint64_t usedCoreNum;
    uint64_t modeFlag;
    uint64_t mode;
    uint64_t includeSelf;
    uint64_t currentNum;
    uint64_t eachNum;
    uint64_t start;
    uint64_t inputAlign;
    uint64_t indicesAlign;
    uint64_t updatesAlign;
    uint64_t inputCount;
    uint64_t indicesCount;
    uint64_t updatesCount;
    uint64_t inputOneTime;
    uint64_t indicesOneTime;
    uint64_t updatesOneTime;
    uint64_t inputLoop;
    uint64_t indicesLoop;
    uint64_t inputEach;
    uint64_t indicesEach;
    uint64_t inputLast;
    uint64_t indicesLast;
    uint64_t currentPiece;
    uint64_t inputOnePiece;
    uint64_t pieceEach;
    uint64_t pieceLast;
    uint64_t lastIndicesLoop;
    uint64_t lastIndicesEach;
    uint64_t lastIndicesLast;
    uint64_t oneTime;
    uint64_t useStableBucket;
    uint64_t useZeroBase;
    uint32_t dataAlign = 32;
};
#endif // SCATTER_ELEMENTS_V2_H
