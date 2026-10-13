/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MAX_POOL2D_WITH_MASK_WINDOW_H
#define MAX_POOL2D_WITH_MASK_WINDOW_H

#include "kernel_operator.h"
#include "max_pool2d_with_mask_tiling_data.h"

namespace MaxPool2dWithMask {
using namespace AscendC;

// A bounded-memory fallback for windows that cannot fit the transposed vector tile.
// Compare ordered IEEE bit patterns so FP16/BF16 subnormals and signed zeros survive.
template <typename T, bool LARGE = false, bool CACHE_PLANE = false, bool DIRECT_SINGLE = false, bool WIDE = false,
          bool EQUIVALENT = false>
class WindowKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR out, GM_ADDR indices, TPipe* pipe,
                                const MaxPool2dWithMaskWindowTilingData* tiling)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        const int64_t outputs = tiling_->validIndexBytes / sizeof(int32_t);
        outputTile_ = tiling_->windowOutputTile;
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(x));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(out));
        indices_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));
        indexBytes_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(indices));
        if constexpr (DIRECT_SINGLE) {
            pipe_->InitBuffer(directInput_, INPUT_BYTES);
            pipe_->InitBuffer(directOutput_, 32);
            pipe_->InitBuffer(directIndex_, 32);
        } else {
            pipe_->InitBuffer(inputQueue_, tiling_->windowPrefetch != 0 ? 2 : 1, INPUT_BYTES);
            pipe_->InitBuffer(outputQueue_, 1, MAX_POOL2D_WINDOW_OUTPUT_ELEMENTS * sizeof(uint32_t));
            pipe_->InitBuffer(indexQueue_, 1, MAX_POOL2D_WINDOW_OUTPUT_ELEMENTS * sizeof(int32_t));
        }
        if constexpr (sizeof(T) == 2) {
            if (!EQUIVALENT && GetBlockIdx() * outputTile_ >= outputs) {
                return;
            }
            pipe_->InitBuffer(keysBuffer_, CAPACITY * sizeof(float));
            if constexpr (!LARGE) {
                pipe_->InitBuffer(tempBuffer_, CAPACITY * sizeof(float));
            }
            pipe_->InitBuffer(reduceBuffer_, CAPACITY / 64 * 2 * sizeof(float));
            if constexpr (!LARGE) {
                pipe_->InitBuffer(maskBuffer_, CAPACITY / 8);
            }
            pipe_->InitBuffer(resultBuffer_, WIDE ? 128 : 64);
            if constexpr (CACHE_PLANE) {
                // Keep offsets separate from both gathered and persistent FP32 keys.
                pipe_->InitBuffer(windowBuffer_, CAPACITY * sizeof(int32_t));
            }
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (DIRECT_SINGLE) {
            PrepareDirectSingle();
        }
        if constexpr (LARGE && sizeof(T) == 2) {
            // Prefetch the first window before its reduction.
            const int64_t first = GetBlockIdx() * outputTile_;
            if (first < tiling_->validIndexBytes / sizeof(int32_t)) {
                earlyPlan_ = PlanWindow(first);
                LoadWindow(earlyPlan_);
                earlyInputReady_ = true;
            }
        }
        if constexpr (EQUIVALENT) {
            ProcessEquivalentWindows();
        } else if constexpr (DIRECT_SINGLE) {
            ProcessDirectSingle();
        } else if constexpr (CACHE_PLANE) {
            ProcessCachedPlanes();
        } else if (outputTile_ == 1) {
            ProcessOutputs<true>();
        } else {
            ProcessOutputs<false>();
        }
    }

private:
    struct AxisGroups {
        int64_t first;
        int64_t repeats;
        int64_t count;
    };

    __aicore__ inline AxisGroups GroupAxis(int64_t size, int64_t kernel, int64_t stride, int64_t pad, int64_t output)
    {
        const int64_t lower = size + pad > kernel ? size + pad - kernel : 0;
        const int64_t first = lower / stride + (lower % stride != 0);
        const int64_t last = pad / stride < output - 1 ? pad / stride : output - 1;
        if (last <= first) {
            return {output, 1, output};
        }
        const int64_t repeats = last - first + 1;
        return {first, repeats, output - repeats + 1};
    }

    __aicore__ inline void ProcessEquivalentWindows()
    {
        const auto height = GroupAxis(tiling_->inputHeight, tiling_->kh, tiling_->sh, tiling_->pt,
                                      tiling_->outputHeight);
        const auto width = GroupAxis(tiling_->inputWidth, tiling_->kw, tiling_->sw, tiling_->pl, tiling_->outputWidth);
        const int64_t groupsPerSlice = height.count * width.count;
        const int64_t total = tiling_->nc * groupsPerSlice;
        for (int64_t task = GetBlockIdx(); task < total; task += GetBlockNum()) {
            const int64_t slice = task / groupsPerSlice;
            const int64_t rowGroup = task % groupsPerSlice / width.count;
            const int64_t colGroup = task % width.count;
            const int64_t row = rowGroup <= height.first ? rowGroup : rowGroup + height.repeats - 1;
            const int64_t col = colGroup <= width.first ? colGroup : colGroup + width.repeats - 1;
            const int64_t rows = rowGroup == height.first ? height.repeats : 1;
            const int64_t columns = colGroup == width.first ? width.repeats : 1;
            const int64_t first = (slice * tiling_->outputHeight + row) * tiling_->outputWidth + col;
            auto values = outputQueue_.AllocTensor<uint16_t>();
            auto indices = indexQueue_.AllocTensor<int32_t>();
            // Equal clipped rectangles have the same stable argmax, including
            // its original signed-zero bits and absolute per-plane index.
            ReduceWindow(first, values, indices, 0);
            const uint16_t bits = values.GetValue(0);
            const int32_t index = indices.GetValue(0);
            Sync<HardEvent::S_V>();
            Duplicate(values.ReinterpretCast<int16_t>(), static_cast<int16_t>(bits), MAX_POOL2D_WINDOW_OUTPUT_ELEMENTS);
            Duplicate(indices, index, MAX_POOL2D_WINDOW_OUTPUT_ELEMENTS);
            outputQueue_.EnQue(values);
            indexQueue_.EnQue(indices);
            values = outputQueue_.DeQue<uint16_t>();
            indices = indexQueue_.DeQue<int32_t>();
            for (int64_t r = 0; r < rows; ++r) {
                for (int64_t c = 0; c < columns; c += MAX_POOL2D_WINDOW_OUTPUT_ELEMENTS) {
                    const int64_t remaining = columns - c;
                    const uint32_t count = remaining < MAX_POOL2D_WINDOW_OUTPUT_ELEMENTS ?
                                               remaining :
                                               MAX_POOL2D_WINDOW_OUTPUT_ELEMENTS;
                    const int64_t offset = first + r * tiling_->outputWidth + c;
                    DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(uint16_t)), 0, 0, 0};
                    DataCopyPad(output_[offset], values, copy);
                    copy.blockLen = count * sizeof(int32_t);
                    DataCopyPad(indices_[offset], indices, copy);
                }
            }
            Sync<HardEvent::MTE3_S>();
            outputQueue_.FreeTensor(values);
            indexQueue_.FreeTensor(indices);
        }
    }

    __aicore__ inline void ProcessDirectSingle()
    {
        if (GetBlockIdx() >= tiling_->validIndexBytes / sizeof(int32_t)) {
            return;
        }
        const WindowPlan& plan = directPlan_;
        auto values = directOutput_.Get<uint16_t>();
        auto indices = directIndex_.Get<int32_t>();
        if (plan.rows == 0 || plan.width == 0) {
            values.SetValue(0, 0);
            indices.SetValue(0, 0);
        } else {
            auto input = directInput_.Get<uint16_t>();
            Sync<HardEvent::MTE2_V>();
            const uint32_t elements = plan.rows * plan.pitch;
            ReduceInput16(input, elements);
            const uint32_t position = static_cast<uint32_t>(ReductionPosition(elements));
            values.SetValue(0, input.GetValue(position));
            indices.SetValue(0, static_cast<int32_t>((plan.row + position / plan.pitch) * tiling_->inputWidth +
                                                     plan.column + position % plan.pitch));
        }
        Sync<HardEvent::S_MTE3>();
        DataCopyExtParams copy{1, sizeof(uint16_t), 0, 0, 0};
        DataCopyPad(output_[GetBlockIdx()], values, copy);
        copy.blockLen = sizeof(int32_t);
        DataCopyPad(indices_[GetBlockIdx()], indices, copy);
        // All buffers stay live until kernel completion; there is no next tile.
    }

    template <bool SINGLE>
    __aicore__ inline void ProcessOutputs()
    {
        const int64_t outputs = tiling_->validIndexBytes / sizeof(int32_t);
        const int64_t tile = SINGLE ? 1 : outputTile_;
        // Each output owns exact-length DMA stores. No scalar GM write can
        // evict a cache line containing an adjacent core's output.
        for (int64_t first = GetBlockIdx() * tile; first < outputs; first += GetBlockNum() * tile) {
            const int64_t count = SINGLE ? 1 : (outputs - first < tile ? outputs - first : tile);
            auto values = outputQueue_.AllocTensor<uint16_t>();
            auto indices = indexQueue_.AllocTensor<int32_t>();
            if constexpr (sizeof(T) == 2) {
                if constexpr (LARGE) {
                    ReducePrefetchedWindows(first, count, values, indices);
                } else if (tiling_->windowPrefetch != 0) {
                    ReducePrefetchedWindows(first, count, values, indices);
                } else {
                    for (int64_t i = 0; i < count; ++i) {
                        ReduceWindow(first + i, values, indices, i);
                    }
                }
            } else {
                for (int64_t i = 0; i < count; ++i) {
                    ReduceWindow(first + i, values, indices, i);
                }
            }
            const event_t event = static_cast<event_t>(pipe_->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(event);
            WaitFlag<HardEvent::S_MTE3>(event);
            outputQueue_.EnQue(values);
            indexQueue_.EnQue(indices);
            values = outputQueue_.DeQue<uint16_t>();
            indices = indexQueue_.DeQue<int32_t>();
            DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            DataCopyPad(output_[first * WORDS], values, copy);
            copy.blockLen = static_cast<uint32_t>(count * sizeof(int32_t));
            DataCopyPad(indices_[first], indices, copy);
            // Queue events order vector work; scalar stores on the next tile
            // also need the previous output DMA to finish before buffer reuse.
            const event_t copied = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(copied);
            WaitFlag<HardEvent::MTE3_S>(copied);
            outputQueue_.FreeTensor(values);
            indexQueue_.FreeTensor(indices);
        }
    }

    __aicore__ inline void ReduceWindow(int64_t output, LocalTensor<uint16_t>& values, LocalTensor<int32_t>& indices,
                                        int64_t localOutput)
    {
        const int64_t plane = tiling_->outputHeight * tiling_->outputWidth;
        const int64_t slice = plane == 1 ? output : output / plane;
        const int64_t row = plane == 1 ? 0 : output % plane / tiling_->outputWidth;
        const int64_t column = plane == 1 ? 0 : output % tiling_->outputWidth;
        const int64_t top = row * static_cast<int64_t>(tiling_->sh) - static_cast<int64_t>(tiling_->pt);
        const int64_t left = column * static_cast<int64_t>(tiling_->sw) - static_cast<int64_t>(tiling_->pl);
        // The A2 attributes retain their full width; the shared legacy fields are int32.
        const int64_t bottom = top + static_cast<int64_t>(tiling_->kh) < tiling_->inputHeight ?
                                   top + static_cast<int64_t>(tiling_->kh) :
                                   tiling_->inputHeight;
        const int64_t right = left + static_cast<int64_t>(tiling_->kw) < tiling_->inputWidth ?
                                  left + static_cast<int64_t>(tiling_->kw) :
                                  tiling_->inputWidth;
        const int64_t inputBase = slice * tiling_->inputHeight * tiling_->inputWidth;
        if constexpr (sizeof(T) == 2) {
            ReduceWindow16(inputBase, top, left, bottom, right, values, indices, localOutput);
            return;
        }
        bool initialized = false;
        uint32_t maximumKey = 0;
        uint32_t maximumBits = 0;
        int32_t maximumIndex = 0;
        for (int64_t ih = top > 0 ? top : 0; ih < bottom; ++ih) {
            for (int64_t iw = left > 0 ? left : 0; iw < right;) {
                constexpr int64_t capacity = INPUT_BYTES / sizeof(T);
                const int64_t count = right - iw < capacity ? right - iw : capacity;
                auto input = inputQueue_.AllocTensor<uint16_t>();
                DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<uint16_t> padding{false, 0, 0, 0};
                DataCopyPad(input, input_[(inputBase + ih * tiling_->inputWidth + iw) * WORDS], copy, padding);
                inputQueue_.EnQue(input);
                input = inputQueue_.DeQue<uint16_t>();
                const event_t ready = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_S));
                SetFlag<HardEvent::MTE2_S>(ready);
                WaitFlag<HardEvent::MTE2_S>(ready);
                for (int64_t i = 0; i < count; ++i) {
                    uint32_t bits = input.GetValue(i * WORDS);
                    if constexpr (sizeof(T) == 4) {
                        bits |= static_cast<uint32_t>(input.GetValue(i * WORDS + 1)) << 16;
                    }
                    uint32_t key = (bits & SIGN) != 0 ? (~bits & MASK) : (bits ^ SIGN);
                    if ((bits & (SIGN - 1)) == 0) {
                        key = SIGN;
                    }
                    if (!initialized || key > maximumKey) {
                        initialized = true;
                        maximumKey = key;
                        maximumBits = bits;
                        maximumIndex = static_cast<int32_t>(ih * tiling_->inputWidth + iw + i);
                    }
                }
                const event_t consumed = static_cast<event_t>(pipe_->FetchEventID(HardEvent::S_MTE2));
                SetFlag<HardEvent::S_MTE2>(consumed);
                WaitFlag<HardEvent::S_MTE2>(consumed);
                inputQueue_.FreeTensor(input);
                iw += count;
            }
        }
        values.SetValue(localOutput * WORDS, static_cast<uint16_t>(maximumBits));
        if constexpr (sizeof(T) == 4) {
            values.SetValue(localOutput * WORDS + 1, static_cast<uint16_t>(maximumBits >> 16));
        }
        indices.SetValue(localOutput, maximumIndex);
    }

    template <HardEvent EVENT>
    __aicore__ inline void Sync()
    {
        const event_t event = static_cast<event_t>(pipe_->FetchEventID(EVENT));
        SetFlag<EVENT>(event);
        WaitFlag<EVENT>(event);
    }

    __aicore__ inline void ReduceValues(LocalTensor<float>& result, LocalTensor<float>& source,
                                        LocalTensor<float>& work, uint32_t count)
    {
        // Each pass retains the first occurrence among equal maxima. Value
        // lanes occupy the even positions of each (value, position) pair.
        const uint32_t repeats = count / 64;
        if (repeats > 32 && repeats < 64) {
            Duplicate(work, -65536.0f, 128);
            PipeBarrier<PIPE_V>();
        }
        WholeReduceMax(work, source, 64, repeats, 1, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
        PipeBarrier<PIPE_V>();
        const uint64_t pairMask = 0x5555555555555555ULL;
        uint64_t mask[2] = {repeats >= 32 ? pairMask : pairMask & ((1ULL << (2 * repeats)) - 1), 0};
        WholeReduceMax(result, work, mask, repeats > 32 ? 2 : 1, 1, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
        if (repeats > 32) {
            PipeBarrier<PIPE_V>();
            uint64_t finalMask[2] = {5, 0};
            WholeReduceMax(result[8], result, finalMask, 1, 1, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
        }
    }

    __aicore__ inline void ReduceInput16(LocalTensor<uint16_t>& input, uint32_t elements)
    {
        auto keys = keysBuffer_.Get<float>();
        auto temp = tempBuffer_.Get<float>();
        auto work = reduceBuffer_.Get<float>();
        auto mask = maskBuffer_.Get<uint8_t>();
        auto result = resultBuffer_.Get<float>();
        const uint32_t reduceElements = (elements + 63) / 64 * 64;
        Cast(keys, input.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, elements);
        PipeBarrier<PIPE_V>();
        if (reduceElements > elements) {
            Duplicate(keys[elements], 0.0f, reduceElements - elements);
            PipeBarrier<PIPE_V>();
        }
        Muls(temp, keys, -1.0f, reduceElements);
        PipeBarrier<PIPE_V>();
        Adds(temp, temp, -32768.0f, reduceElements);
        CompareScalar(mask, keys, 0.0f, CMPMODE::LT, reduceElements);
        PipeBarrier<PIPE_V>();
        Select(keys, mask, temp, keys, SELMODE::VSEL_TENSOR_TENSOR_MODE, reduceElements);
        PipeBarrier<PIPE_V>();
        if (reduceElements > elements) {
            Duplicate(keys[elements], -65536.0f, reduceElements - elements);
            PipeBarrier<PIPE_V>();
        }
        ReduceValues(result, keys, work, reduceElements);
        Sync<HardEvent::V_S>();
    }

    template <bool MINIMUM = false>
    __aicore__ inline void ReduceLargeValues(LocalTensor<float>& result, LocalTensor<float>& source,
                                             LocalTensor<float>& work, uint32_t count)
    {
        // Each pass retains the first occurrence among equal extrema. Value
        // lanes occupy the even positions of each (value, position) pair.
        const uint32_t repeats = count / 64;
        if (repeats > 32 && repeats % 32 != 0) {
            Duplicate(work, MINIMUM ? 65536.0f : -65536.0f, (repeats + 31) / 32 * 64);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (MINIMUM) {
            WholeReduceMin(work, source, 64, repeats, 1, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
        } else {
            WholeReduceMax(work, source, 64, repeats, 1, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
        }
        PipeBarrier<PIPE_V>();
        const uint64_t pairMask = 0x5555555555555555ULL;
        uint64_t mask[2] = {repeats >= 32 ? pairMask : pairMask & ((1ULL << (2 * repeats)) - 1), 0};
        if constexpr (MINIMUM) {
            WholeReduceMin(result, work, mask, (repeats + 31) / 32, 1, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
        } else {
            WholeReduceMax(result, work, mask, (repeats + 31) / 32, 1, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
        }
        if (repeats > 32) {
            PipeBarrier<PIPE_V>();
            uint64_t finalMask[2] = {pairMask & ((1ULL << (2 * ((repeats + 31) / 32))) - 1), 0};
            if constexpr (MINIMUM) {
                WholeReduceMin(result[WIDE ? 16 : 8], result, finalMask, 1, 1, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
            } else {
                WholeReduceMax(result[WIDE ? 16 : 8], result, finalMask, 1, 1, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
            }
        }
    }

    struct ReducedWindow {
        float key;
        int32_t position;
    };

    __aicore__ inline int32_t ReductionPosition(uint32_t count)
    {
        auto result = resultBuffer_.Get<uint32_t>();
        uint32_t pair = result.GetValue(1);
        if (count > 2048) {
            const uint32_t group = result.GetValue(WIDE ? 17 : 9);
            pair = group / 2 * 64 + result.GetValue(group + 1);
        }
        return static_cast<int32_t>(pair / 2 * 64 + reduceBuffer_.Get<uint32_t>().GetValue(pair + 1));
    }

    __aicore__ inline ReducedWindow ReduceLargeInput16(LocalTensor<uint16_t>& input, uint32_t elements)
    {
        auto keys = keysBuffer_.Get<float>();
        auto work = reduceBuffer_.Get<float>();
        auto result = resultBuffer_.Get<float>();
        const uint32_t reduceElements = (elements + 63) / 64 * 64;
        // Initialize the last complete vector before copying valid ranks into
        // it, so the fill address stays aligned even for a packed GM tail.
        if (reduceElements > elements) {
            Duplicate(keys[reduceElements - 64], -65536.0f, 64);
            PipeBarrier<PIPE_V>();
        }
        // Signed raw bits have the same order as positive IEEE values. Their
        // minimum is the largest nonpositive value. Casting the integers is exact
        // and preserves subnormals without transforming every element's sign.
        Cast(keys, input.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, elements);
        PipeBarrier<PIPE_V>();
        if (elements <= MAX_POOL2D_WINDOW_LARGE_INPUT_BYTES / 4) {
            // Two arrays fit for small windows. Transform signs in one vector
            // pass sequence so all-negative data does not need another reduction.
            auto temp = keys[CAPACITY / 2];
            auto mask = reduceBuffer_.Get<uint8_t>();
            if (reduceElements > elements) {
                Duplicate(keys[elements], 0.0f, reduceElements - elements);
                PipeBarrier<PIPE_V>();
            }
            Muls(temp, keys, -1.0f, reduceElements);
            PipeBarrier<PIPE_V>();
            Adds(temp, temp, -32768.0f, reduceElements);
            CompareScalar(mask, keys, 0.0f, CMPMODE::LT, reduceElements);
            PipeBarrier<PIPE_V>();
            Select(keys, mask, temp, keys, SELMODE::VSEL_TENSOR_TENSOR_MODE, reduceElements);
            PipeBarrier<PIPE_V>();
            if (reduceElements > elements) {
                Duplicate(keys[elements], -65536.0f, reduceElements - elements);
                PipeBarrier<PIPE_V>();
            }
            ReduceLargeValues(result, keys, work, reduceElements);
            Sync<HardEvent::V_S>();
            return {result.GetValue(reduceElements > 2048 ? (WIDE ? 16 : 8) : 0), ReductionPosition(reduceElements)};
        }
        ReduceLargeValues(result, keys, work, reduceElements);
        Sync<HardEvent::V_S>();
        const float largest = result.GetValue(reduceElements > 2048 ? (WIDE ? 16 : 8) : 0);
        const int32_t largestPosition = ReductionPosition(reduceElements);
        if (largest > 0) {
            return {largest, largestPosition};
        }
        // Scalar reads of the first result must finish before vector work
        // overwrites it. No positive input exists, so reduce the negative bits.
        Sync<HardEvent::S_V>();
        if (reduceElements > elements) {
            const uint32_t tailStart = reduceElements - 64;
            Duplicate(keys[tailStart], 65536.0f, 64);
            PipeBarrier<PIPE_V>();
            Cast(keys[tailStart], input.ReinterpretCast<int16_t>()[tailStart], RoundMode::CAST_NONE,
                 elements - tailStart);
            PipeBarrier<PIPE_V>();
        }
        ReduceLargeValues<true>(result, keys, work, reduceElements);
        Sync<HardEvent::V_S>();
        const float smallest = result.GetValue(reduceElements > 2048 ? (WIDE ? 16 : 8) : 0);
        const int32_t smallestPosition = ReductionPosition(reduceElements);
        if (largest == 0) {
            // +0 and -0 tie numerically; retain the first input's original bits.
            return {0,
                    smallest == -32768.0f && smallestPosition < largestPosition ? smallestPosition : largestPosition};
        }
        return {-32768.0f - smallest, smallestPosition};
    }

    struct WindowPlan {
        int64_t base;
        int64_t row;
        int64_t column;
        int32_t rows;
        int32_t width;
        int32_t pitch;
    };

    __aicore__ inline WindowPlan PlanWindow(int64_t output)
    {
        const int64_t plane = tiling_->outputHeight * tiling_->outputWidth;
        const int64_t row = DIRECT_SINGLE && plane == 1 ? 0 : output % plane / tiling_->outputWidth;
        const int64_t column = output % tiling_->outputWidth;
        const int64_t top = row * static_cast<int64_t>(tiling_->sh) - static_cast<int64_t>(tiling_->pt);
        const int64_t left = column * static_cast<int64_t>(tiling_->sw) - static_cast<int64_t>(tiling_->pl);
        const int64_t bottom = top + static_cast<int64_t>(tiling_->kh) < tiling_->inputHeight ?
                                   top + static_cast<int64_t>(tiling_->kh) :
                                   tiling_->inputHeight;
        const int64_t right = left + static_cast<int64_t>(tiling_->kw) < tiling_->inputWidth ?
                                  left + static_cast<int64_t>(tiling_->kw) :
                                  tiling_->inputWidth;
        const int64_t firstRow = top > 0 ? top : 0;
        const int64_t firstColumn = left > 0 ? left : 0;
        const int32_t rows = bottom > firstRow ? static_cast<int32_t>(bottom - firstRow) : 0;
        const int32_t width = right > firstColumn ? static_cast<int32_t>(right - firstColumn) : 0;
        int32_t pitch = (width + 15) / 16 * 16;
        if constexpr (LARGE) {
            // Complete rows are contiguous in GM. The large reduction initializes
            // its final aligned vector before casting the exact packed tail.
            const int64_t elements = static_cast<int64_t>(rows) * width;
            if (width == tiling_->inputWidth &&
                (elements > MAX_POOL2D_WINDOW_LARGE_INPUT_BYTES / 4 || elements % 16 == 0)) {
                pitch = width;
            }
        }
        return {(DIRECT_SINGLE && plane == 1 ? output : output / plane) * tiling_->inputHeight * tiling_->inputWidth,
                firstRow,
                firstColumn,
                rows,
                width,
                pitch};
    }

    __aicore__ inline void PrepareDirectSingle()
    {
        if (GetBlockIdx() >= tiling_->validIndexBytes / sizeof(int32_t)) {
            return;
        }
        directPlan_ = PlanWindow(GetBlockIdx());
        const WindowPlan& plan = directPlan_;
        if (plan.rows != 0 && plan.width != 0) {
            auto input = directInput_.Get<uint16_t>();
            DataCopyExtParams copy{static_cast<uint16_t>(plan.rows), static_cast<uint32_t>(plan.width * 2),
                                   plan.rows > 1 ? static_cast<uint32_t>((tiling_->inputWidth - plan.width) * 2) : 0, 0,
                                   0};
            DataCopyPadExtParams<uint16_t> padding{true, 0, static_cast<uint8_t>(plan.pitch - plan.width), 0xFF80};
            DataCopyPad(input, input_[plan.base + plan.row * tiling_->inputWidth + plan.column], copy, padding);
        }
    }

    __aicore__ inline void LoadWindow(const WindowPlan& plan)
    {
        auto input = inputQueue_.AllocTensor<uint16_t>();
        if (plan.rows != 0 && plan.width != 0) {
#if ORIG_DTYPE_X == DT_BF16
            constexpr uint16_t negativeInfinity = 0xFF80;
#else
            constexpr uint16_t negativeInfinity = 0xFC00;
#endif
            DataCopyExtParams copy{static_cast<uint16_t>(plan.rows), static_cast<uint32_t>(plan.width * 2),
                                   plan.rows > 1 ? static_cast<uint32_t>((tiling_->inputWidth - plan.width) * 2) : 0, 0,
                                   0};
            DataCopyPadExtParams<uint16_t> padding{true, 0, static_cast<uint8_t>(plan.pitch - plan.width),
                                                   negativeInfinity};
            if constexpr (LARGE) {
                if (plan.width == tiling_->inputWidth && plan.pitch == plan.width) {
                    copy.blockCount = 1;
                    copy.blockLen = static_cast<uint32_t>(plan.rows * plan.width * sizeof(uint16_t));
                    padding.isPad = false;
                }
            }
            DataCopyPad(input, input_[plan.base + plan.row * tiling_->inputWidth + plan.column], copy, padding);
        }
        inputQueue_.EnQue(input);
    }

    __aicore__ inline void ReducePrefetchedWindows(int64_t first, int64_t count, LocalTensor<uint16_t>& values,
                                                   LocalTensor<int32_t>& indices)
    {
        WindowPlan current;
        if constexpr (LARGE) {
            if (earlyInputReady_) {
                current = earlyPlan_;
                earlyInputReady_ = false;
            } else {
                current = PlanWindow(first);
                LoadWindow(current);
            }
        } else {
            current = PlanWindow(first);
            LoadWindow(current);
        }
        for (int64_t i = 0; i < count; ++i) {
            auto input = inputQueue_.DeQue<uint16_t>();
            WindowPlan next{};
            if (i + 1 < count) {
                next = PlanWindow(first + i + 1);
                LoadWindow(next);
            }
            if (current.rows == 0 || current.width == 0) {
                values.SetValue(i, 0);
                indices.SetValue(i, 0);
            } else {
                const uint32_t elements = current.rows * current.pitch;
                int32_t position;
                if constexpr (LARGE) {
                    position = ReduceLargeInput16(input, elements).position;
                } else {
                    ReduceInput16(input, elements);
                    auto result = resultBuffer_.Get<uint32_t>();
                    uint32_t pair = result.GetValue(1);
                    if (elements > 2048) {
                        const uint32_t group = result.GetValue(9);
                        pair = group / 2 * 64 + result.GetValue(group + 1);
                    }
                    position = static_cast<int32_t>(pair / 2 * 64 + reduceBuffer_.Get<uint32_t>().GetValue(pair + 1));
                }
                values.SetValue(i, input.GetValue(position));
                indices.SetValue(i,
                                 static_cast<int32_t>((current.row + position / current.pitch) * tiling_->inputWidth +
                                                      current.column + position % current.pitch));
            }
            Sync<HardEvent::S_MTE2>();
            inputQueue_.FreeTensor(input);
            current = next;
        }
    }

    __aicore__ inline void PrepareCachedKeys(LocalTensor<uint16_t>& cached, uint32_t elements)
    {
        auto keys = tempBuffer_.Get<float>();
        auto scratch = keysBuffer_.Get<float>();
        auto mask = maskBuffer_.Get<uint8_t>();
        const uint32_t aligned = (elements + 63) / 64 * 64;
        Duplicate(keys, 0.0f, aligned);
        PipeBarrier<PIPE_V>();
        Cast(keys, cached.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, elements);
        PipeBarrier<PIPE_V>();
        Muls(scratch, keys, -1.0f, aligned);
        PipeBarrier<PIPE_V>();
        Adds(scratch, scratch, -32768.0f, aligned);
        CompareScalar(mask, keys, 0.0f, CMPMODE::LT, aligned);
        PipeBarrier<PIPE_V>();
        Select(keys, mask, scratch, keys, SELMODE::VSEL_TENSOR_TENSOR_MODE, aligned);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void GatherCachedWindow(LocalTensor<uint16_t>& cached, const WindowPlan& plan,
                                              LocalTensor<uint16_t>& values, LocalTensor<int32_t>& indices,
                                              int64_t localOutput)
    {
        if (plan.rows == 0 || plan.width == 0) {
            values.SetValue(localOutput, 0);
            indices.SetValue(localOutput, 0);
            return;
        }
        auto offsets = windowBuffer_.Get<int32_t>();
        auto window = keysBuffer_.Get<float>();
        auto cachedKeys = tempBuffer_.Get<float>();
        if (cachedWindowRows_ != plan.rows || cachedWindowWidth_ != plan.width) {
            cachedWindowRows_ = plan.rows;
            cachedWindowWidth_ = plan.width;
            cachedWindowPitch_ = 1;
            uint32_t shift = 0;
            while (cachedWindowPitch_ < static_cast<uint32_t>(plan.width)) {
                cachedWindowPitch_ *= 2;
                ++shift;
            }
            cachedWindowElements_ = (plan.rows * cachedWindowPitch_ + 63) / 64 * 64;
            const uint32_t elements = cachedWindowElements_;
            auto rows = keysBuffer_.Get<int32_t>();
            CreateVecIndex(offsets, static_cast<int32_t>(0), elements);
            PipeBarrier<PIPE_V>();
            ShiftRight(rows, offsets, static_cast<int32_t>(shift), elements);
            PipeBarrier<PIPE_V>();
            // Complete the last vector using the last real row. All duplicates
            // follow their original elements and preserve the first-argmax result.
            Mins(rows, rows, plan.rows - 1, elements);
            PipeBarrier<PIPE_V>();
            if (shift == 0) {
                Duplicate(offsets, static_cast<int32_t>(0), elements);
            } else {
                ShiftLeft(offsets.ReinterpretCast<uint32_t>(), offsets.ReinterpretCast<uint32_t>(), 32U - shift,
                          elements);
                PipeBarrier<PIPE_V>();
                ShiftRight(offsets.ReinterpretCast<uint32_t>(), offsets.ReinterpretCast<uint32_t>(), 32U - shift,
                           elements);
            }
            Muls(rows, rows, static_cast<int32_t>(tiling_->inputWidth), elements);
            PipeBarrier<PIPE_V>();
            // Duplicate the last valid column into row padding. Its real occurrence
            // always comes first, so the first-argmax reduction cannot select padding.
            Mins(offsets, offsets, plan.width - 1, elements);
            PipeBarrier<PIPE_V>();
            Add(offsets, offsets, rows, elements);
            PipeBarrier<PIPE_V>();
            Muls(offsets, offsets, static_cast<int32_t>(sizeof(float)), elements);
            PipeBarrier<PIPE_V>();
        }
        const uint32_t elements = cachedWindowElements_;
        const uint32_t pitch = cachedWindowPitch_;
        const uint32_t base = static_cast<uint32_t>((plan.row * tiling_->inputWidth + plan.column) * sizeof(float));
        Gather(window, cachedKeys, offsets.ReinterpretCast<uint32_t>(), base, elements);
        PipeBarrier<PIPE_V>();
        auto result = resultBuffer_.Get<float>();
        auto work = reduceBuffer_.Get<float>();
        ReduceValues(result, window, work, elements);
        Sync<HardEvent::V_S>();
        const uint32_t position = static_cast<uint32_t>(ReductionPosition(elements));
        const int32_t index = static_cast<int32_t>((plan.row + position / pitch) * tiling_->inputWidth + plan.column +
                                                   position % pitch);
        values.SetValue(localOutput, cached.GetValue(index));
        indices.SetValue(localOutput, index);
        Sync<HardEvent::S_V>();
    }

    __aicore__ inline void ProcessCachedPlanes()
    {
        const int64_t inputPlane = tiling_->inputHeight * tiling_->inputWidth;
        const int64_t outputPlane = tiling_->outputHeight * tiling_->outputWidth;
        for (int64_t slice = GetBlockIdx(); slice < tiling_->nc; slice += GetBlockNum()) {
            auto cached = inputQueue_.AllocTensor<uint16_t>();
            DataCopyExtParams copy{1, static_cast<uint32_t>(inputPlane * 2), 0, 0, 0};
            DataCopyPadExtParams<uint16_t> padding{false, 0, 0, 0};
            DataCopyPad(cached, input_[slice * inputPlane], copy, padding);
            inputQueue_.EnQue(cached);
            cached = inputQueue_.DeQue<uint16_t>();
            PrepareCachedKeys(cached, static_cast<uint32_t>(inputPlane));
            for (int64_t first = 0; first < outputPlane; first += MAX_POOL2D_WINDOW_OUTPUT_ELEMENTS) {
                const int64_t count = outputPlane - first < MAX_POOL2D_WINDOW_OUTPUT_ELEMENTS ?
                                          outputPlane - first :
                                          MAX_POOL2D_WINDOW_OUTPUT_ELEMENTS;
                auto values = outputQueue_.AllocTensor<uint16_t>();
                auto indices = indexQueue_.AllocTensor<int32_t>();
                for (int64_t i = 0; i < count; ++i) {
                    const WindowPlan plan = PlanWindow(slice * outputPlane + first + i);
                    GatherCachedWindow(cached, plan, values, indices, i);
                }
                Sync<HardEvent::S_MTE3>();
                outputQueue_.EnQue(values);
                indexQueue_.EnQue(indices);
                values = outputQueue_.DeQue<uint16_t>();
                indices = indexQueue_.DeQue<int32_t>();
                copy.blockLen = static_cast<uint32_t>(count * 2);
                DataCopyPad(output_[slice * outputPlane + first], values, copy);
                copy.blockLen = static_cast<uint32_t>(count * sizeof(int32_t));
                DataCopyPad(indices_[slice * outputPlane + first], indices, copy);
                Sync<HardEvent::MTE3_S>();
                outputQueue_.FreeTensor(values);
                indexQueue_.FreeTensor(indices);
            }
            Sync<HardEvent::S_MTE2>();
            inputQueue_.FreeTensor(cached);
        }
    }

    __aicore__ inline void ReduceWindow16(int64_t inputBase, int64_t top, int64_t left, int64_t bottom, int64_t right,
                                          LocalTensor<uint16_t>& values, LocalTensor<int32_t>& indices,
                                          int64_t localOutput)
    {
        const int64_t firstRow = top > 0 ? top : 0;
        const int64_t firstCol = left > 0 ? left : 0;
        const int64_t width = right - firstCol;
        if (firstRow >= bottom || width <= 0) {
            values.SetValue(localOutput, 0);
            indices.SetValue(localOutput, 0);
            return;
        }
        auto work = reduceBuffer_.Get<float>();
        auto result = resultBuffer_.Get<float>();
        bool initialized = false;
        float maximum = 0;
        uint16_t maximumBits = 0;
        int32_t maximumIndex = 0;
        // Treat raw IEEE 16-bit values as signed integers. Positive bit patterns
        // are already ordered; negative patterns become -32768 - signedBits.
        // All operations are exact normal FP32 integers, including subnormals'
        // ranks. Both zero encodings map to zero without changing output bits.
#if ORIG_DTYPE_X == DT_BF16
        constexpr uint16_t negativeInfinity = 0xFF80;
#else
        constexpr uint16_t negativeInfinity = 0xFC00;
#endif
        for (int64_t row = firstRow; row < bottom;) {
            const int64_t rowWidth = width < CAPACITY ? width : CAPACITY;
            const int64_t alignedWidth = (rowWidth + 15) / 16 * 16;
            const int64_t capacityRows = width <= CAPACITY ? CAPACITY / alignedWidth : 1;
            const int64_t rows = bottom - row < capacityRows ? bottom - row : capacityRows;
            for (int64_t col = firstCol; col < right;) {
                const int64_t count = right - col < CAPACITY ? right - col : CAPACITY;
                const int64_t pitch = (count + 15) / 16 * 16;
                const uint32_t elements = static_cast<uint32_t>(rows * pitch);
                // CompareScalar requires complete 256-byte repeats. Initialize
                // the extra lanes before either comparison, then exclude them
                // from both reductions with a rank below every legal input.
                const uint32_t reduceElements = (elements + 63) / 64 * 64;
                auto input = inputQueue_.AllocTensor<uint16_t>();
                DataCopyExtParams copy{static_cast<uint16_t>(rows), static_cast<uint32_t>(count * 2),
                                       rows > 1 ? static_cast<uint32_t>((tiling_->inputWidth - count) * 2) : 0, 0, 0};
                DataCopyPadExtParams<uint16_t> padding{true, 0, static_cast<uint8_t>(pitch - count), negativeInfinity};
                DataCopyPad(input, input_[inputBase + row * tiling_->inputWidth + col], copy, padding);
                inputQueue_.EnQue(input);
                input = inputQueue_.DeQue<uint16_t>();
                ReduceInput16(input, elements);
                const float candidate = result.GetValue(reduceElements > 2048 ? 8 : 0);
                if (!initialized || candidate > maximum) {
                    uint32_t pair = result.ReinterpretCast<uint32_t>().GetValue(1);
                    if (reduceElements > 2048) {
                        const uint32_t group = result.ReinterpretCast<uint32_t>().GetValue(9);
                        pair = group / 2 * 64 + result.ReinterpretCast<uint32_t>().GetValue(group + 1);
                    }
                    const int32_t position = static_cast<int32_t>(pair / 2 * 64 +
                                                                  work.ReinterpretCast<uint32_t>().GetValue(pair + 1));
                    maximum = candidate;
                    maximumBits = input.GetValue(position);
                    maximumIndex = static_cast<int32_t>((row + position / pitch) * tiling_->inputWidth + col +
                                                        position % pitch);
                    initialized = true;
                }
                Sync<HardEvent::S_MTE2>();
                inputQueue_.FreeTensor(input);
                col += count;
            }
            row += rows;
        }
        values.SetValue(localOutput, maximumBits);
        indices.SetValue(localOutput, maximumIndex);
    }

    WindowPlan directPlan_{};
    WindowPlan earlyPlan_{};
    bool earlyInputReady_ = false;
    uint32_t cachedWindowRows_ = 0;
    uint32_t cachedWindowWidth_ = 0;
    uint32_t cachedWindowPitch_ = 0;
    uint32_t cachedWindowElements_ = 0;
    int64_t outputTile_ = 1;
    static constexpr int64_t INPUT_BYTES = WIDE ? MAX_POOL2D_WINDOW_WIDE_INPUT_BYTES :
                                                  (LARGE ? MAX_POOL2D_WINDOW_LARGE_INPUT_BYTES :
                                                           MAX_POOL2D_WINDOW_INPUT_BYTES);
    static constexpr int64_t CAPACITY = INPUT_BYTES / sizeof(uint16_t);
    TBuf<> directInput_;
    TBuf<> directOutput_;
    TBuf<> directIndex_;
    TBuf<> keysBuffer_;
    TBuf<> tempBuffer_;
    TBuf<> reduceBuffer_;
    TBuf<> maskBuffer_;
    TBuf<> resultBuffer_;
    TBuf<> windowBuffer_;
    static constexpr int64_t WORDS = sizeof(T) / sizeof(uint16_t);
    static constexpr uint32_t SIGN = sizeof(T) == 4 ? 0x80000000U : 0x8000U;
    static constexpr uint32_t MASK = sizeof(T) == 4 ? 0xFFFFFFFFU : 0xFFFFU;
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 2> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TQue<QuePosition::VECOUT, 1> indexQueue_;
    GlobalTensor<uint16_t> input_;
    GlobalTensor<uint16_t> output_;
    GlobalTensor<int32_t> indices_;
    GlobalTensor<uint8_t> indexBytes_;
    const MaxPool2dWithMaskWindowTilingData* tiling_ = nullptr;
};
// One lane per spatial output; keep raw integer bits for exact BF16 results.
template <bool SCAN = false, bool COMPACT = false>
class SpatialSmallKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR out, GM_ADDR indices, TPipe* pipe,
                                const MaxPool2dWithMaskWindowTilingData* tiling)
    {
        t_ = tiling;
        pipe_ = pipe;
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(x));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(out));
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));
        bytes_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(indices));
        const uint32_t usedWidth = (t_->outputWidth - 1) * t_->sw + t_->kw;
        const uint32_t width = SCAN ? t_->inputWidth + t_->pl : usedWidth;
        // Each scan segment must remain within its row, including ceilMode
        // inputs narrower than the window's largest power-of-two segment.
        if constexpr (SCAN) {
            while (2U * inputPitch_ <= t_->kw) {
                inputPitch_ *= 2;
                ++inputShift_;
            }
        }
        // Scan queries clip neutral right padding. Direct Gather still needs
        // the full window span to be physically present.
        while (inputPitch_ < width || (!SCAN && inputPitch_ < usedWidth)) {
            inputPitch_ *= 2;
            ++inputShift_;
        }
        if constexpr (COMPACT) {
            inputPitch_ = (usedWidth + 15) / 16 * 16;
        }
        while (outputPitch_ < t_->outputWidth) {
            outputPitch_ *= 2;
            ++outputShift_;
        }
        const uint32_t usedRows = (t_->outputHeight - 1) * t_->sh + t_->kh;
        const uint32_t rows = t_->inputHeight + t_->pt;
        inputCount_ = (((rows > usedRows ? rows : usedRows) * inputPitch_ + 63) / 64) * 64;
        outputCount_ = (t_->outputHeight * outputPitch_ + 63) / 64 * 64;
        pipe_->InitBuffer(raw16_, 8192 * 2);
        pipe_->InitBuffer(raw32_, 8192 * 4);
        pipe_->InitBuffer(keys_, 8192 * 4);
        pipe_->InitBuffer(temp_, 8192 * 4);
        pipe_->InitBuffer(offsets_, 2048 * 4);
        pipe_->InitBuffer(bestPosition_, 2048 * 4);
        pipe_->InitBuffer(maximum_, 2048 * 4);
        pipe_->InitBuffer(candidate_, 2048 * 4);
        pipe_->InitBuffer(scratch_, 2048 * 4);
        pipe_->InitBuffer(resultIndex_, 2048 * 4);
        pipe_->InitBuffer(resultValue_, 2048 * 2);
        pipe_->InitBuffer(paddingMask_, 2048);
        pipe_->InitBuffer(compareMask_, 256);
    }

    __aicore__ inline void Process()
    {
        const uint32_t rows = t_->windowOutputTile;
        const uint32_t tiles = (t_->outputHeight + rows - 1) / rows;
        for (uint64_t task = GetBlockIdx(); task < t_->nc * tiles; task += GetBlockNum()) {
            const uint64_t slice = task / tiles;
            outputRowStart_ = (task % tiles) * rows;
            outputRows_ = rows < t_->outputHeight - outputRowStart_ ? rows : t_->outputHeight - outputRowStart_;
            inputRowStart_ = static_cast<int64_t>(outputRowStart_) * static_cast<int64_t>(t_->sh) -
                             static_cast<int64_t>(t_->pt);
            const uint32_t neededRows = (outputRows_ - 1) * t_->sh + t_->kh;
            inputCount_ = (neededRows * inputPitch_ + 63) / 64 * 64;
            outputCount_ = (outputRows_ * outputPitch_ + 63) / 64 * 64;
            PrepareGeometry();
            LoadPlane(slice);
            PoolPlane();
            StorePlane(slice);
        }
    }

private:
    template <HardEvent E>
    __aicore__ inline void Sync()
    {
        const event_t event = static_cast<event_t>(pipe_->FetchEventID(E));
        SetFlag<E>(event);
        WaitFlag<E>(event);
    }

    __aicore__ inline void PrepareGeometry()
    {
        if constexpr (!COMPACT) {
            auto column = temp_.Get<int32_t>();
            auto columnFloat = keys_.Get<float>();
            auto mask = paddingMask_.Get<uint8_t>();
            CreateVecIndex(column, int32_t{0}, inputCount_);
            PipeBarrier<PIPE_V>();
            ShiftLeft(column.ReinterpretCast<uint32_t>(), column.ReinterpretCast<uint32_t>(), 32U - inputShift_,
                      inputCount_);
            PipeBarrier<PIPE_V>();
            ShiftRight(column.ReinterpretCast<uint32_t>(), column.ReinterpretCast<uint32_t>(), 32U - inputShift_,
                       inputCount_);
            PipeBarrier<PIPE_V>();
            Cast(columnFloat, column, RoundMode::CAST_NONE, inputCount_);
            PipeBarrier<PIPE_V>();
            CompareScalar(mask, columnFloat, static_cast<float>(static_cast<int32_t>(t_->pl)), CMPMODE::LT,
                          inputCount_);
            CompareScalar(mask[1024], columnFloat, static_cast<float>(static_cast<int32_t>(t_->pl + t_->inputWidth)),
                          CMPMODE::GE, inputCount_);
            PipeBarrier<PIPE_V>();
            Or(mask.ReinterpretCast<uint16_t>(), mask.ReinterpretCast<uint16_t>(),
               mask[1024].ReinterpretCast<uint16_t>(), inputCount_ / 16);
        }
        auto offsets = offsets_.Get<int32_t>();
        auto rows = scratch_.Get<int32_t>();
        CreateVecIndex(offsets, int32_t{0}, outputCount_);
        PipeBarrier<PIPE_V>();
        ShiftRight(rows, offsets, static_cast<int32_t>(outputShift_), outputCount_);
        ShiftLeft(offsets.ReinterpretCast<uint32_t>(), offsets.ReinterpretCast<uint32_t>(), 32U - outputShift_,
                  outputCount_);
        PipeBarrier<PIPE_V>();
        ShiftRight(offsets.ReinterpretCast<uint32_t>(), offsets.ReinterpretCast<uint32_t>(), 32U - outputShift_,
                   outputCount_);
        Mins(rows, rows, static_cast<int32_t>(outputRows_ - 1), outputCount_);
        PipeBarrier<PIPE_V>();
        Mins(offsets, offsets, static_cast<int32_t>(t_->outputWidth - 1), outputCount_);
        if constexpr (COMPACT) {
            PipeBarrier<PIPE_V>();
            auto base = resultIndex_.Get<int32_t>();
            auto columns = candidate_.Get<int32_t>();
            Muls(base, rows, static_cast<int32_t>(t_->sh * t_->inputWidth), outputCount_);
            Muls(columns, offsets, static_cast<int32_t>(t_->sw), outputCount_);
            PipeBarrier<PIPE_V>();
            Add(base, base, columns, outputCount_);
            PipeBarrier<PIPE_V>();
            Adds(base, base, static_cast<int32_t>(inputRowStart_ * t_->inputWidth), outputCount_);
        }
        Muls(rows, rows, static_cast<int32_t>(t_->sh * inputPitch_ * 4), outputCount_);
        PipeBarrier<PIPE_V>();
        Muls(offsets, offsets, static_cast<int32_t>(t_->sw * 4), outputCount_);
        PipeBarrier<PIPE_V>();
        Add(offsets, offsets, rows, outputCount_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadPlane(int64_t slice)
    {
        auto raw = raw16_.Get<uint16_t>();
        Duplicate(raw.ReinterpretCast<int16_t>(), COMPACT ? int16_t{-128} : int16_t{0}, inputCount_);
        Sync<HardEvent::V_MTE2>();
        const int64_t firstRow = inputRowStart_ > 0 ? inputRowStart_ : 0;
        const int64_t proposedEnd = inputRowStart_ + (outputRows_ - 1) * t_->sh + t_->kh;
        const int64_t endRow = proposedEnd < t_->inputHeight ? proposedEnd : t_->inputHeight;
        const uint32_t topPadding = firstRow - inputRowStart_;
        const uint32_t loadRows = endRow - firstRow;
        // Hardware padding is limited to 32 bytes on each side. Complete
        // left-padding blocks already contain zeros; start DMA after them.
        const uint32_t leftBlocks = SCAN ? (t_->pl / 16) * 16 : 0;
        const uint32_t leftRemainder = t_->pl - leftBlocks;
        // Only columns observed by an output window need to be materialized.
        // The GM row stride still uses the full physical input width.
        const uint32_t requiredWidth = (t_->outputWidth - 1) * t_->sw + t_->kw - t_->pl;
        const uint32_t loadWidth = !SCAN && requiredWidth < t_->inputWidth ? requiredWidth : t_->inputWidth;
        DataCopyExtParams copy{static_cast<uint16_t>(loadRows), loadWidth * 2,
                               static_cast<uint32_t>((t_->inputWidth - loadWidth) * 2),
                               static_cast<uint32_t>(inputPitch_ / 16 - (loadWidth + leftRemainder + 15) / 16), 0};
        DataCopyPadExtParams<uint16_t> pad{true, static_cast<uint8_t>(leftRemainder),
                                           COMPACT ? static_cast<uint8_t>((16 - loadWidth % 16) % 16) : uint8_t{0},
                                           COMPACT ? uint16_t{0xFF80} : uint16_t{0}};
        DataCopyPad(raw[topPadding * inputPitch_ + leftBlocks],
                    x_[slice * t_->inputHeight * t_->inputWidth + firstRow * t_->inputWidth], copy, pad);
        Sync<HardEvent::MTE2_V>();
        auto rawFloat = raw32_.Get<float>();
        auto keys = keys_.Get<float>();
        auto temp = temp_.Get<float>();
        auto mask = paddingMask_.Get<uint8_t>();
        Cast(rawFloat, raw.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, inputCount_);
        PipeBarrier<PIPE_V>();
        Muls(keys, rawFloat, -1.0f, inputCount_);
        CompareScalar(mask[1024], rawFloat, 0.0f, CMPMODE::LT, inputCount_);
        PipeBarrier<PIPE_V>();
        Adds(keys, keys, -32768.0f, inputCount_);
        PipeBarrier<PIPE_V>();
        Select(keys, mask[1024], keys, rawFloat, SELMODE::VSEL_TENSOR_TENSOR_MODE, inputCount_);
        Duplicate(temp, -65536.0f, inputCount_);
        PipeBarrier<PIPE_V>();
        if constexpr (SCAN) {
            EncodeKeys(topPadding, loadRows);
            return;
        }
        if constexpr (!COMPACT) {
            Select(keys, mask, temp, keys, SELMODE::VSEL_TENSOR_TENSOR_MODE, inputCount_);
            PipeBarrier<PIPE_V>();
        }
        if (topPadding != 0) {
            Duplicate(keys, -65536.0f, topPadding * inputPitch_);
        }
        const uint32_t last = (topPadding + loadRows) * inputPitch_;
        if (last < inputCount_) {
            Duplicate(keys[last], -65536.0f, inputCount_ - last);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void EncodeKeys(uint32_t topPadding, uint32_t loadRows)
    {
        auto keys = keys_.Get<float>();
        auto temp = temp_.Get<float>();
        auto mask = paddingMask_.Get<uint8_t>();
        // Encode (ordered BF16 rank, first position) into a positive normal FP32
        // bit pattern. No integer-to-float rounding is used for the combined key.
        auto ranks = temp.ReinterpretCast<int32_t>();
        Cast(ranks, keys, RoundMode::CAST_RINT, inputCount_);
        PipeBarrier<PIPE_V>();
        Adds(ranks, ranks, int32_t{32768}, inputCount_);
        PipeBarrier<PIPE_V>();
        ShiftLeft(ranks.ReinterpretCast<uint32_t>(), ranks.ReinterpretCast<uint32_t>(), 13U, inputCount_);
        auto words = keys.ReinterpretCast<int32_t>();
        CreateVecIndex(words, int32_t{0}, inputCount_);
        PipeBarrier<PIPE_V>();
        Muls(words, words, int32_t{-1}, inputCount_);
        PipeBarrier<PIPE_V>();
        Adds(words, words, int32_t{0x20000000 + 8191}, inputCount_);
        PipeBarrier<PIPE_V>();
        Add(words, words, ranks, inputCount_);
        PipeBarrier<PIPE_V>();
        Duplicate(temp, 0.0f, inputCount_);
        PipeBarrier<PIPE_V>();
        Select(keys, mask, temp, keys, SELMODE::VSEL_TENSOR_TENSOR_MODE, inputCount_);
        PipeBarrier<PIPE_V>();
        if (topPadding != 0) {
            Duplicate(keys, 0.0f, topPadding * inputPitch_);
        }
        const uint32_t last = (topPadding + loadRows) * inputPitch_;
        if (last < inputCount_) {
            Duplicate(keys[last], 0.0f, inputCount_ - last);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ScanHorizontal(LocalTensor<float>& state, uint32_t blockShift, bool reverse)
    {
        auto offsets = offsets_.Get<int32_t>();
        auto blocks = bestPosition_.Get<int32_t>();
        auto positions = scratch_.Get<int32_t>();
        auto candidate = candidate_.Get<float>();
        const uint32_t block = 1U << blockShift;
        for (uint32_t distance = 1; distance < block; distance *= 2) {
            for (uint32_t first = 0; first < inputCount_; first += 2048) {
                const uint32_t count = inputCount_ - first < 2048 ? inputCount_ - first : 2048;
                CreateVecIndex(offsets, static_cast<int32_t>(first), count);
                PipeBarrier<PIPE_V>();
                ShiftRight(blocks, offsets, static_cast<int32_t>(blockShift), count);
                PipeBarrier<PIPE_V>();
                ShiftLeft(blocks.ReinterpretCast<uint32_t>(), blocks.ReinterpretCast<uint32_t>(), blockShift, count);
                PipeBarrier<PIPE_V>();
                Sub(positions, offsets, blocks, count);
                PipeBarrier<PIPE_V>();
                Adds(positions, positions, reverse ? static_cast<int32_t>(distance) : -static_cast<int32_t>(distance),
                     count);
                PipeBarrier<PIPE_V>();
                if (reverse) {
                    Mins(positions, positions, static_cast<int32_t>(block - 1), count);
                } else {
                    Maxs(positions, positions, int32_t{0}, count);
                }
                PipeBarrier<PIPE_V>();
                Add(offsets, blocks, positions, count);
                PipeBarrier<PIPE_V>();
                Muls(offsets, offsets, int32_t{4}, count);
                PipeBarrier<PIPE_V>();
                Gather(candidate, state, offsets.ReinterpretCast<uint32_t>(), 0, count);
                PipeBarrier<PIPE_V>();
                Max(state[first], state[first], candidate, count);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void PoolScanned()
    {
        auto prefix = temp_.Get<float>();
        auto suffix = keys_.Get<float>();
        Adds(prefix, suffix, 0.0f, inputCount_);
        PipeBarrier<PIPE_V>();
        uint32_t shift = 0;
        while ((2U << shift) <= t_->kw) {
            ++shift;
        }
        ScanHorizontal(prefix, shift, false);
        ScanHorizontal(suffix, shift, true);
        auto offsets = offsets_.Get<int32_t>();
        auto ends = bestPosition_.Get<int32_t>();
        auto positions = scratch_.Get<int32_t>();
        auto maximum = maximum_.Get<float>();
        auto candidate = candidate_.Get<float>();
        auto horizontal = raw32_.Get<float>();
        const uint32_t rows = (outputRows_ - 1) * t_->sh + t_->kh;
        const uint32_t horizontalCount = (rows * outputPitch_ + 63) / 64 * 64;
        const bool clipRight = (t_->outputWidth - 1) * t_->sw + t_->kw > inputPitch_;
        auto crossBlock = compareMask_.Get<uint8_t>();
        for (uint32_t first = 0; first < horizontalCount; first += 2048) {
            const uint32_t count = horizontalCount - first < 2048 ? horizontalCount - first : 2048;
            CreateVecIndex(offsets, static_cast<int32_t>(first), count);
            PipeBarrier<PIPE_V>();
            ShiftRight(ends, offsets, static_cast<int32_t>(outputShift_), count);
            ShiftLeft(offsets.ReinterpretCast<uint32_t>(), offsets.ReinterpretCast<uint32_t>(), 32U - outputShift_,
                      count);
            PipeBarrier<PIPE_V>();
            ShiftRight(offsets.ReinterpretCast<uint32_t>(), offsets.ReinterpretCast<uint32_t>(), 32U - outputShift_,
                       count);
            Mins(ends, ends, static_cast<int32_t>(rows - 1), count);
            PipeBarrier<PIPE_V>();
            Mins(offsets, offsets, static_cast<int32_t>(t_->outputWidth - 1), count);
            Muls(ends, ends, static_cast<int32_t>(inputPitch_), count);
            PipeBarrier<PIPE_V>();
            Muls(offsets, offsets, static_cast<int32_t>(t_->sw), count);
            PipeBarrier<PIPE_V>();
            if (clipRight) {
                Cast(positions.ReinterpretCast<float>(), offsets, RoundMode::CAST_NONE, count);
                PipeBarrier<PIPE_V>();
                CompareScalar(crossBlock, positions.ReinterpretCast<float>(),
                              static_cast<float>(static_cast<int32_t>(inputPitch_ - (1U << shift))), CMPMODE::LT,
                              count);
                PipeBarrier<PIPE_V>();
                Adds(positions, ends, static_cast<int32_t>(inputPitch_ - 1), count);
                PipeBarrier<PIPE_V>();
            }
            Add(offsets, offsets, ends, count);
            PipeBarrier<PIPE_V>();
            Adds(ends, offsets, static_cast<int32_t>(t_->kw - 1), count);
            PipeBarrier<PIPE_V>();
            if (clipRight) {
                Min(ends, ends, positions, count);
                PipeBarrier<PIPE_V>();
            }
            ShiftRight(positions, offsets, static_cast<int32_t>(shift), count);
            PipeBarrier<PIPE_V>();
            Adds(positions, positions, int32_t{2}, count);
            PipeBarrier<PIPE_V>();
            ShiftLeft(positions.ReinterpretCast<uint32_t>(), positions.ReinterpretCast<uint32_t>(), shift, count);
            PipeBarrier<PIPE_V>();
            Adds(positions, positions, int32_t{-1}, count);
            PipeBarrier<PIPE_V>();
            Min(positions, positions, ends, count);
            Muls(offsets, offsets, int32_t{4}, count);
            Muls(ends, ends, int32_t{4}, count);
            PipeBarrier<PIPE_V>();
            Muls(positions, positions, int32_t{4}, count);
            Gather(maximum, suffix, offsets.ReinterpretCast<uint32_t>(), 0, count);
            Gather(candidate, prefix, ends.ReinterpretCast<uint32_t>(), 0, count);
            PipeBarrier<PIPE_V>();
            if (clipRight) {
                Adds(horizontal[first], maximum, 0.0f, count);
                PipeBarrier<PIPE_V>();
            }
            Max(maximum, maximum, candidate, count);
            PipeBarrier<PIPE_V>();
            Gather(candidate, prefix, positions.ReinterpretCast<uint32_t>(), 0, count);
            PipeBarrier<PIPE_V>();
            if (clipRight) {
                Max(maximum, maximum, candidate, count);
                PipeBarrier<PIPE_V>();
                // Within the final segment, prefix[b] would include values
                // before a. Its suffix alone exactly covers the clipped window.
                Select(horizontal[first], crossBlock, maximum, horizontal[first], SELMODE::VSEL_TENSOR_TENSOR_MODE,
                       count);
            } else {
                Max(horizontal[first], maximum, candidate, count);
            }
            PipeBarrier<PIPE_V>();
        }
        // Vertical consumers need only the horizontal winner states.
        CreateVecIndex(offsets, int32_t{0}, outputCount_);
        PipeBarrier<PIPE_V>();
        ShiftRight(positions, offsets, static_cast<int32_t>(outputShift_), outputCount_);
        ShiftLeft(offsets.ReinterpretCast<uint32_t>(), offsets.ReinterpretCast<uint32_t>(), 32U - outputShift_,
                  outputCount_);
        PipeBarrier<PIPE_V>();
        ShiftRight(offsets.ReinterpretCast<uint32_t>(), offsets.ReinterpretCast<uint32_t>(), 32U - outputShift_,
                   outputCount_);
        Mins(positions, positions, static_cast<int32_t>(outputRows_ - 1), outputCount_);
        PipeBarrier<PIPE_V>();
        Mins(offsets, offsets, static_cast<int32_t>(t_->outputWidth - 1), outputCount_);
        Muls(positions, positions, static_cast<int32_t>(t_->sh * outputPitch_ * 4), outputCount_);
        PipeBarrier<PIPE_V>();
        Muls(offsets, offsets, int32_t{4}, outputCount_);
        PipeBarrier<PIPE_V>();
        Add(offsets, offsets, positions, outputCount_);
        Duplicate(maximum, 0.0f, outputCount_);
        PipeBarrier<PIPE_V>();
        for (uint32_t kh = 0; kh < t_->kh; ++kh) {
            Gather(candidate, horizontal, offsets.ReinterpretCast<uint32_t>(), kh * outputPitch_ * 4, outputCount_);
            PipeBarrier<PIPE_V>();
            Max(maximum, maximum, candidate, outputCount_);
            PipeBarrier<PIPE_V>();
        }
        auto best = bestPosition_.Get<int32_t>();
        auto position = scratch_.Get<int32_t>();
        Duplicate(position, int32_t{8191}, outputCount_);
        PipeBarrier<PIPE_V>();
        And(best.ReinterpretCast<uint16_t>(), maximum.ReinterpretCast<uint16_t>(), position.ReinterpretCast<uint16_t>(),
            outputCount_ * 2);
        PipeBarrier<PIPE_V>();
        Muls(best, best, int32_t{-1}, outputCount_);
        PipeBarrier<PIPE_V>();
        Adds(best, best, int32_t{8191}, outputCount_);
        PipeBarrier<PIPE_V>();
        ShiftLeft(best.ReinterpretCast<uint32_t>(), best.ReinterpretCast<uint32_t>(), 2U, outputCount_);
        // Restore original values only after horizontal states are dead.
        Cast(raw32_.Get<float>(), raw16_.Get<int16_t>(), RoundMode::CAST_NONE, inputCount_);
        PipeBarrier<PIPE_V>();
        // Recover original raw bits, including the sign of a first-occurrence zero.
        Gather(candidate, raw32_.Get<float>(), best.ReinterpretCast<uint32_t>(), 0, outputCount_);
        PipeBarrier<PIPE_V>();
        Cast(resultValue_.Get<int16_t>(), candidate, RoundMode::CAST_RINT, outputCount_);
        auto indices = resultIndex_.Get<int32_t>();
        ShiftRight(position, best, int32_t{2}, outputCount_);
        PipeBarrier<PIPE_V>();
        ShiftRight(indices, position, static_cast<int32_t>(inputShift_), outputCount_);
        ShiftLeft(position.ReinterpretCast<uint32_t>(), position.ReinterpretCast<uint32_t>(), 32U - inputShift_,
                  outputCount_);
        PipeBarrier<PIPE_V>();
        ShiftRight(position.ReinterpretCast<uint32_t>(), position.ReinterpretCast<uint32_t>(), 32U - inputShift_,
                   outputCount_);
        Adds(indices, indices, static_cast<int32_t>(inputRowStart_), outputCount_);
        PipeBarrier<PIPE_V>();
        Muls(indices, indices, static_cast<int32_t>(t_->inputWidth), outputCount_);
        Adds(position, position, -static_cast<int32_t>(t_->pl), outputCount_);
        PipeBarrier<PIPE_V>();
        Add(indices, indices, position, outputCount_);
    }

    __aicore__ inline void PoolCompact()
    {
        auto maximum = maximum_.Get<float>();
        auto candidate = candidate_.Get<float>();
        auto best = bestPosition_.Get<int32_t>();
        auto position = scratch_.Get<int32_t>();
        auto offsets = offsets_.Get<int32_t>();
        auto keys = keys_.Get<float>();
        auto mask = compareMask_.Get<uint8_t>();
        // With zero padding and a valid output start, its first position is real.
        Gather(maximum, keys, offsets.ReinterpretCast<uint32_t>(), 0, outputCount_);
        Duplicate(best, int32_t{0}, outputCount_);
        PipeBarrier<PIPE_V>();
        for (uint32_t winner = 1; winner < 4; ++winner) {
            const uint32_t delta = ((winner / 2) * inputPitch_ + winner % 2) * 4;
            Gather(candidate, keys, offsets.ReinterpretCast<uint32_t>(), delta, outputCount_);
            Duplicate(position, static_cast<int32_t>(winner), outputCount_);
            PipeBarrier<PIPE_V>();
            Compare(mask, candidate, maximum, CMPMODE::GT, outputCount_);
            PipeBarrier<PIPE_V>();
            Select(maximum, mask, candidate, maximum, SELMODE::VSEL_TENSOR_TENSOR_MODE, outputCount_);
            Select(best.ReinterpretCast<float>(), mask, position.ReinterpretCast<float>(),
                   best.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, outputCount_);
            PipeBarrier<PIPE_V>();
        }
        // A two-bit row-major winner restores either address independently.
        auto indices = resultIndex_.Get<int32_t>();
        auto rowDelta = candidate_.Get<int32_t>();
        ShiftRight(position, best, int32_t{1}, outputCount_);
        ShiftLeft(best.ReinterpretCast<uint32_t>(), best.ReinterpretCast<uint32_t>(), 31U, outputCount_);
        PipeBarrier<PIPE_V>();
        ShiftRight(best.ReinterpretCast<uint32_t>(), best.ReinterpretCast<uint32_t>(), 31U, outputCount_);
        Muls(rowDelta, position, static_cast<int32_t>(t_->inputWidth), outputCount_);
        PipeBarrier<PIPE_V>();
        Add(indices, indices, rowDelta, outputCount_);
        PipeBarrier<PIPE_V>();
        Add(indices, indices, best, outputCount_);
        Muls(position, position, static_cast<int32_t>(inputPitch_ * 4), outputCount_);
        PipeBarrier<PIPE_V>();
        Muls(best, best, int32_t{4}, outputCount_);
        PipeBarrier<PIPE_V>();
        Add(best, best, position, outputCount_);
        PipeBarrier<PIPE_V>();
        Add(best, best, offsets, outputCount_);
        PipeBarrier<PIPE_V>();
        Gather(candidate, raw32_.Get<float>(), best.ReinterpretCast<uint32_t>(), 0, outputCount_);
        PipeBarrier<PIPE_V>();
        Cast(resultValue_.Get<int16_t>(), candidate, RoundMode::CAST_RINT, outputCount_);
    }

    __aicore__ inline void PoolPlane()
    {
        if constexpr (SCAN) {
            PoolScanned();
            return;
        }
        if constexpr (COMPACT) {
            PoolCompact();
            return;
        }
        auto maximum = maximum_.Get<float>();
        auto candidate = candidate_.Get<float>();
        auto best = bestPosition_.Get<int32_t>();
        auto position = scratch_.Get<int32_t>();
        auto offsets = offsets_.Get<int32_t>();
        auto keys = keys_.Get<float>();
        auto mask = compareMask_.Get<uint8_t>();
        Duplicate(maximum, -65536.0f, outputCount_);
        Duplicate(best, int32_t{0}, outputCount_);
        PipeBarrier<PIPE_V>();
        for (uint32_t kh = 0; kh < t_->kh; ++kh) {
            for (uint32_t kw = 0; kw < t_->kw; ++kw) {
                const uint32_t delta = (kh * inputPitch_ + kw) * 4;
                Gather(candidate, keys, offsets.ReinterpretCast<uint32_t>(), delta, outputCount_);
                Adds(position, offsets, static_cast<int32_t>(delta), outputCount_);
                PipeBarrier<PIPE_V>();
                Compare(mask, candidate, maximum, CMPMODE::GT, outputCount_);
                PipeBarrier<PIPE_V>();
                Select(maximum, mask, candidate, maximum, SELMODE::VSEL_TENSOR_TENSOR_MODE, outputCount_);
                Select(best.ReinterpretCast<float>(), mask, position.ReinterpretCast<float>(),
                       best.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, outputCount_);
                PipeBarrier<PIPE_V>();
            }
        }
        // Recover original raw bits, including the sign of a first-occurrence zero.
        Gather(candidate, raw32_.Get<float>(), best.ReinterpretCast<uint32_t>(), 0, outputCount_);
        PipeBarrier<PIPE_V>();
        Cast(resultValue_.Get<int16_t>(), candidate, RoundMode::CAST_RINT, outputCount_);
        auto indices = resultIndex_.Get<int32_t>();
        ShiftRight(position, best, int32_t{2}, outputCount_);
        PipeBarrier<PIPE_V>();
        ShiftRight(indices, position, static_cast<int32_t>(inputShift_), outputCount_);
        ShiftLeft(position.ReinterpretCast<uint32_t>(), position.ReinterpretCast<uint32_t>(), 32U - inputShift_,
                  outputCount_);
        PipeBarrier<PIPE_V>();
        ShiftRight(position.ReinterpretCast<uint32_t>(), position.ReinterpretCast<uint32_t>(), 32U - inputShift_,
                   outputCount_);
        Adds(indices, indices, static_cast<int32_t>(inputRowStart_), outputCount_);
        PipeBarrier<PIPE_V>();
        Muls(indices, indices, static_cast<int32_t>(t_->inputWidth), outputCount_);
        Adds(position, position, -static_cast<int32_t>(t_->pl), outputCount_);
        PipeBarrier<PIPE_V>();
        Add(indices, indices, position, outputCount_);
    }

    __aicore__ inline void StorePlane(int64_t slice)
    {
        Sync<HardEvent::V_MTE3>();
        const int64_t plane = t_->outputHeight * t_->outputWidth;
        DataCopyExtParams copy{static_cast<uint16_t>(outputRows_), static_cast<uint32_t>(t_->outputWidth * 2),
                               static_cast<uint32_t>(outputPitch_ / 16 - (t_->outputWidth + 15) / 16), 0, 0};
        DataCopyPad(y_[slice * plane + outputRowStart_ * t_->outputWidth], resultValue_.Get<uint16_t>(), copy);
        copy.blockLen = t_->outputWidth * 4;
        copy.srcStride = outputPitch_ / 8 - (t_->outputWidth + 7) / 8;
        DataCopyPad(index_[slice * plane + outputRowStart_ * t_->outputWidth], resultIndex_.Get<int32_t>(), copy);
        Sync<HardEvent::MTE3_V>();
    }

    const MaxPool2dWithMaskWindowTilingData* t_ = nullptr;
    TPipe* pipe_ = nullptr;
    uint32_t inputPitch_ = 16;
    uint32_t outputPitch_ = 16;
    uint32_t inputShift_ = 4;
    uint32_t outputShift_ = 4;
    uint32_t inputCount_ = 0;
    uint32_t outputCount_ = 0;
    uint32_t outputRowStart_ = 0;
    uint32_t outputRows_ = 0;
    int64_t inputRowStart_ = 0;
    TBuf<> raw16_;
    TBuf<> raw32_;
    TBuf<> keys_;
    TBuf<> temp_;
    TBuf<> offsets_;
    TBuf<> bestPosition_;
    TBuf<> maximum_;
    TBuf<> candidate_;
    TBuf<> scratch_;
    TBuf<> resultIndex_;
    TBuf<> resultValue_;
    TBuf<> paddingMask_;
    TBuf<> compareMask_;
    GlobalTensor<uint16_t> x_;
    GlobalTensor<uint16_t> y_;
    GlobalTensor<int32_t> index_;
    GlobalTensor<uint8_t> bytes_;
};

} // namespace MaxPool2dWithMask
#endif
