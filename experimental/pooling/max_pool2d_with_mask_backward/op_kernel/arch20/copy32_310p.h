// clang-format off
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// clang-format on

#ifndef MAXPOOL310P_COPY32_H
#define MAXPOOL310P_COPY32_H
#include "kernel_operator.h"
#define MAXPOOL310P_PLAN_FUNCTION __aicore__
#include "copy32_plan.h"
#undef MAXPOOL310P_PLAN_FUNCTION
#include <type_traits>

namespace MaxPool310P {
using namespace AscendC;

// Synchronous correctness baseline for FP16, FP32 and INT32 index transport.
// The caller owns the entire output interval. Different cores must not overlap
// their actual writes (including short-tensor padding). Local tensors start at
// 32B-aligned addresses and have at least "capacity" elements.
template <typename T, bool CachedMask = false>
class Copy32 {
public:
    using U = typename std::conditional<sizeof(T) == 2, uint16_t, uint32_t>::type;
    static constexpr uint32_t BLOCK = 32 / sizeof(T);

    __aicore__ inline void Init(TPipe* pipe)
    {
        static_assert(sizeof(T) == 2 || sizeof(T) == 4, "Only 16/32-bit elements");
        pipe_ = pipe;
        pipe_->InitBuffer(readTail_, 32);
        pipe_->InitBuffer(writeTail_, 32);
        pipe_->InitBuffer(pattern_, 32);
    }

    // Valid data occupies dst[0,count); padding up to ceil32(count) is zero.
    // Returns false without any DMA for an invalid range or absent short backing.
    __aicore__ inline bool In(const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint64_t offset, uint32_t count,
                              uint64_t total, uint64_t physical, uint32_t capacity)
    {
        const auto p = PlanIn(total, physical, offset, count, capacity, BLOCK);
        if (!p.valid || count == 0)
            return p.valid;
        if (p.mainCount)
            DataCopy(dst, src[offset], p.mainCount);
        if (p.tailCount)
            DataCopy(readTail_.Get<T>(), src[p.tailGm], BLOCK);
        Sync<HardEvent::MTE2_V>();
        if (p.tailCount) {
            auto tailDst = dst[p.tailUb].template ReinterpretCast<U>();
            Duplicate(tailDst, static_cast<U>(0), BLOCK);
            PipeBarrier<PIPE_V>();
            Select(tailDst, readTail_.Get<U>(), p.skip, p.tailCount, p.span);
        }
        Sync<HardEvent::V_S>();
        return true;
    }

    // Ordinary stores only: never use this overlapping-tail algorithm under an
    // atomic-add mode. For count>=BLOCK every store is within the logical range.
    __aicore__ inline bool Out(const GlobalTensor<T>& dst, uint64_t offset, const LocalTensor<T>& src, uint32_t count,
                               uint64_t total, uint64_t physical, uint32_t capacity)
    {
        const auto p = PlanOut(total, physical, offset, count, capacity, BLOCK);
        if (!p.valid || count == 0)
            return p.valid;
        Sync<HardEvent::MTE2_V>();
        Sync<HardEvent::S_V>();
        if (p.tailCount) {
            auto tail = writeTail_.Get<U>();
            Duplicate(tail, static_cast<U>(0), BLOCK);
            PipeBarrier<PIPE_V>();
            Select(tail, src[p.tailUb].template ReinterpretCast<U>(), p.skip, p.tailCount, p.span);
        }
        Sync<HardEvent::V_MTE3>();
        if (p.mainCount)
            DataCopy(dst[offset], src, p.mainCount);
        if (p.tailCount) {
            // Documented requirement for two DMA writes to overlapping GM.
            PipeBarrier<PIPE_MTE3>();
            DataCopy(dst[p.tailGm], writeTail_.Get<T>(), BLOCK);
        }
        Sync<HardEvent::MTE3_V>();
        Sync<HardEvent::MTE3_S>();
        return true;
    }

private:
    template <HardEvent E>
    __aicore__ inline void Sync()
    {
        const auto event = pipe_->FetchEventID(E);
        SetFlag<E>(event);
        WaitFlag<E>(event);
    }

    __aicore__ inline void Select(const LocalTensor<U>& dst, const LocalTensor<U>& src, uint32_t skip, uint32_t count,
                                  uint32_t span)
    {
        // At most two source data blocks: all source and destination bases are
        // aligned. Scalar work builds a small UB mask; it never accesses GM.
        auto pattern = pattern_.Get<U>();
        if constexpr (CachedMask) {
            // PlanIn/Out gather at most two blocks: <=32 elements for FP16,
            // <=16 for FP32. The complete selection fits in one uint32 word.
            if (!patternReady_) {
                Duplicate(pattern_.Get<uint32_t>(), static_cast<uint32_t>(0), 8);
                PipeBarrier<PIPE_V>();
            }
            if (!patternReady_ || skip != patternSkip_ || count != patternCount_) {
                Sync<HardEvent::V_S>();
                pattern_.Get<uint32_t>().SetValue(0, static_cast<uint32_t>(((1ULL << count) - 1) << skip));
                Sync<HardEvent::S_V>();
                patternReady_ = true;
                patternSkip_ = skip;
                patternCount_ = count;
            }
        } else {
            Sync<HardEvent::V_S>();
            constexpr uint32_t bits = 8 * sizeof(U);
            for (uint32_t word = 0; word < 32 / sizeof(U); ++word) {
                U value = 0;
                for (uint32_t bit = 0; bit < bits; ++bit) {
                    const uint32_t pos = word * bits + bit;
                    if (pos >= skip && pos - skip < count)
                        value |= static_cast<U>(1ULL << bit);
                }
                pattern.SetValue(word, value);
            }
            Sync<HardEvent::S_V>();
        }
        uint64_t selected = 0;
        GatherMask<U, U, GatherMaskMode::VERSION_V1>(dst, src, pattern, true, span, GatherMaskParams{1, 1, 8, 0},
                                                     selected);
        PipeBarrier<PIPE_V>();
    }

    bool patternReady_ = false;
    uint32_t patternSkip_ = 0, patternCount_ = 0;
    TPipe* pipe_ = nullptr;
    TBuf<TPosition::VECCALC> readTail_, writeTail_, pattern_;
};
} // namespace MaxPool310P
#endif
