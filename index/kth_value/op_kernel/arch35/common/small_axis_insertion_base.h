/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file small_axis_insertion_base.h
 * \brief Common SIMT kernels and constants shared by sort and kth_value small_axis_insertion kernels.
 */

#ifndef SMALL_AXIS_INSERTION_BASE_H
#define SMALL_AXIS_INSERTION_BASE_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"
#include "util_type_simd.h"

namespace SmallAxisCommon {
using namespace AscendC;

constexpr uint32_t INSERTION_THREAD_NUM = 1024;
// DataCopy hardware limit for blockCount (batch mode max segments per transfer).
constexpr uint32_t MAX_DATACOPY_BLOCK_COUNT = 4095;

/**
 * @brief SIMT gather kernel for non-last-axis insertion-sort batches.
 * @tparam T Input storage data type
 */
template <typename T>
__simt_vf__ LAUNCH_BOUND(INSERTION_THREAD_NUM) __aicore__
    void SimtLoadNonLastInsertionBatch(uint32_t totalElems, uint32_t validSegs, uint32_t segmentLen,
                                       uint32_t valueRowElems, uint64_t outerBaseOffset, uint64_t innerStart,
                                       uint64_t innerSize, __gm__ volatile T* input, __ubuf__ T* output)
{
    // Load a non-last-axis tile as sort-major segments:
    // GM [axis, inner] -> UB [inner segment, axis].
    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x); idx < totalElems; idx += INSERTION_THREAD_NUM) {
        uint32_t axis = idx / validSegs;
        uint32_t seg = idx - axis * validSegs;
        output[seg * valueRowElems + axis] = input[outerBaseOffset + static_cast<uint64_t>(axis) * innerSize +
                                                   innerStart + seg];
    }
}

/**
 * @brief SIMT per-segment insertion sort used by small-axis sch 5.
 * @tparam T Sort value data type
 * @tparam IDX_T Output index data type stored in UB
 * @tparam IsDescend Sort order: true for descending, false for ascending
 * @tparam OrderNan Whether floating-point NaNs follow the sort order contract
 */
template <typename T, typename IDX_T, bool IsDescend, bool OrderNan = false>
__simt_vf__ LAUNCH_BOUND(INSERTION_THREAD_NUM) __aicore__
    void SimtInsertionSortSegments(uint32_t validSegs, uint32_t segmentLen, uint32_t valueRowElems,
                                   uint32_t idxRowElems, __ubuf__ T* valueBase, __ubuf__ IDX_T* idxBase)
{
    for (int32_t seg = threadIdx.x; seg < static_cast<int32_t>(validSegs);
         seg += static_cast<int32_t>(INSERTION_THREAD_NUM)) {
        uint32_t segId = static_cast<uint32_t>(seg);
        uint32_t valueBaseOffset = segId * valueRowElems;
        uint32_t idxBaseOffset = segId * idxRowElems;
        for (uint32_t elem = 0; elem < segmentLen; ++elem) {
            idxBase[idxBaseOffset + elem] = static_cast<IDX_T>(elem);
        }

        for (uint32_t elem = 1; elem < segmentLen; ++elem) {
            T keyValue = valueBase[valueBaseOffset + elem];
            IDX_T keyIdx = idxBase[idxBaseOffset + elem];
            uint32_t insertPos = elem;
            while (insertPos > 0) {
                T prevValue = valueBase[valueBaseOffset + insertPos - 1];
                bool needMove;
                if constexpr (OrderNan && (IsSameType<T, float>::value || IsSameType<T, half>::value ||
                                           IsSameType<T, bfloat16_t>::value)) {
                    bool prevNan = prevValue != prevValue;
                    bool keyNan = keyValue != keyValue;
                    if (prevNan != keyNan) {
                        needMove = IsDescend ? keyNan : prevNan;
                    } else if (prevNan) {
                        needMove = false;
                    } else if constexpr (IsDescend) {
                        needMove = prevValue < keyValue;
                    } else {
                        needMove = prevValue > keyValue;
                    }
                } else if constexpr (IsDescend) {
                    needMove = prevValue < keyValue;
                } else {
                    needMove = prevValue > keyValue;
                }
                if (!needMove) {
                    break;
                }
                valueBase[valueBaseOffset + insertPos] = prevValue;
                idxBase[idxBaseOffset + insertPos] = idxBase[idxBaseOffset + insertPos - 1];
                --insertPos;
            }
            valueBase[valueBaseOffset + insertPos] = keyValue;
            idxBase[idxBaseOffset + insertPos] = keyIdx;
        }
    }
}

/**
 * @brief CRTP base class for small-axis insertion-sort kernels.
 *        Contains shared Process loop, UB buffer setup, data loading, dtype conversion, and insertion sort.
 *        Derived classes provide batch mapping details and output store behavior.
 * @tparam Derived CRTP derived type
 * @tparam T Input/output storage data type
 * @tparam CONVERT_TYPE Data type used for insertion-sort comparisons in UB
 * @tparam IDX_T Index data type stored in UB during insertion sort
 * @tparam IsDescend Sort order: true for descending, false for ascending
 * @tparam OrderNan Whether floating-point NaNs follow the sort order contract
 */
template <typename Derived, typename T, typename CONVERT_TYPE, typename IDX_T, bool IsDescend, bool OrderNan = false>
class SmallAxisInsertionBase {
public:
    __aicore__ inline void Process()
    {
        Derived* op = static_cast<Derived*>(this);
        if (op->IsProcessInvalid()) {
            return;
        }
        uint32_t batchesPerCore = (batchNum_ + blockDim_ - 1U) / blockDim_;
        uint32_t startBatch = blockIdx_ * batchesPerCore;
        uint32_t endBatch = batchNum_ < startBatch + batchesPerCore ? batchNum_ : startBatch + batchesPerCore;
        for (uint32_t batchId = startBatch; batchId < endBatch; ++batchId) {
            uint32_t validSegs = op->ComputeValidSegs(batchId);
            if (validSegs == 0U) {
                continue;
            }
            op->ProcessBatch(batchId, validSegs);
        }
    }

    __aicore__ inline void InitInsertionBuffers(TPipe* pipe)
    {
        pipe_ = pipe;
        if constexpr (!IsSameType<T, CONVERT_TYPE>::value) {
            castRowStride_ = ROUND_UP_AGLIN(segmentLen_ * sizeof(T)) / sizeof(T);
            valueRowStride_ = castRowStride_;
        } else {
            valueRowStride_ = ROUND_UP_AGLIN(segmentLen_ * sizeof(CONVERT_TYPE)) / sizeof(CONVERT_TYPE);
        }
        indexRowStride_ = ROUND_UP_AGLIN(segmentLen_ * sizeof(IDX_T)) / sizeof(IDX_T);

        pipe_->InitBuffer(valueBuf_, segmentsPerBatch_ * valueRowStride_ * sizeof(CONVERT_TYPE));
        pipe_->InitBuffer(idxBuf_, segmentsPerBatch_ * indexRowStride_ * sizeof(IDX_T));
        if constexpr (!IsSameType<T, CONVERT_TYPE>::value) {
            pipe_->InitBuffer(castBuf_, segmentsPerBatch_ * castRowStride_ * sizeof(T));
        }

        values_ = valueBuf_.template Get<CONVERT_TYPE>();
        indices_ = idxBuf_.template Get<IDX_T>();
    }

    __aicore__ inline void LoadContiguousBatch(GlobalTensor<T>& inputGm, int64_t inputStart, uint32_t validSegs,
                                               bool padTail)
    {
        if (validSegs == 0U) {
            return;
        }
        uint32_t dstRowStride = IsSameType<T, CONVERT_TYPE>::value ? valueRowStride_ : castRowStride_;
        uint32_t copiedRowElems = padTail ? ROUND_UP_AGLIN(segmentLen_ * sizeof(T)) / sizeof(T) : segmentLen_;
        uint32_t dstStrideBlocks = ((dstRowStride - copiedRowElems) * sizeof(T)) / UB_BLOCK_SIZE;
        DataCopyExtParams copyParam{static_cast<uint16_t>(validSegs), static_cast<uint32_t>(segmentLen_ * sizeof(T)), 0,
                                    dstStrideBlocks, 0};
        DataCopyPadExtParams<T> padParams{
            padTail, 0, static_cast<uint8_t>(padTail ? (copiedRowElems - segmentLen_) : 0U), static_cast<T>(0)};
        if constexpr (!IsSameType<T, CONVERT_TYPE>::value) {
            LocalTensor<T> loadBuf = castBuf_.template Get<T>();
            DataCopyPad(loadBuf, inputGm[inputStart], copyParam, padParams);
            event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);
            Cast(values_, loadBuf, RoundMode::CAST_NONE, validSegs * valueRowStride_);
        } else {
            DataCopyPad(values_, inputGm[inputStart], copyParam, padParams);
            event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);
        }
    }

    __aicore__ inline void LoadNonLastBatch(GlobalTensor<T>& inputGm, uint64_t outerBaseOffset, uint64_t innerStart,
                                            uint64_t innerSize, uint32_t validSegs)
    {
        uint32_t totalElems = validSegs * segmentLen_;
        if constexpr (!IsSameType<T, CONVERT_TYPE>::value) {
            LocalTensor<T> loadBuf = castBuf_.template Get<T>();
            asc_vf_call<SimtLoadNonLastInsertionBatch<T>>(
                dim3(INSERTION_THREAD_NUM), totalElems, validSegs, segmentLen_, castRowStride_, outerBaseOffset,
                innerStart, innerSize, (__gm__ volatile T*)inputGm.GetPhyAddr(), (__ubuf__ T*)loadBuf.GetPhyAddr());
            Cast(values_, loadBuf, RoundMode::CAST_NONE, validSegs * valueRowStride_);
        } else {
            asc_vf_call<SimtLoadNonLastInsertionBatch<T>>(
                dim3(INSERTION_THREAD_NUM), totalElems, validSegs, segmentLen_, valueRowStride_, outerBaseOffset,
                innerStart, innerSize, (__gm__ volatile T*)inputGm.GetPhyAddr(), (__ubuf__ T*)values_.GetPhyAddr());
        }
    }

    __aicore__ inline void SortBatch(uint32_t validSegs)
    {
        asc_vf_call<SimtInsertionSortSegments<CONVERT_TYPE, IDX_T, IsDescend, OrderNan>>(
            dim3(INSERTION_THREAD_NUM), validSegs, segmentLen_, valueRowStride_, indexRowStride_,
            (__ubuf__ CONVERT_TYPE*)values_.GetPhyAddr(), (__ubuf__ IDX_T*)indices_.GetPhyAddr());
    }

protected:
    TPipe* pipe_ = nullptr;

    TBuf<TPosition::VECCALC> valueBuf_; // CONVERT_TYPE buffer for sort operations
    TBuf<TPosition::VECCALC> castBuf_;  // T buffer for GM ↔ UB cast (bf16↔float transfer)
    TBuf<TPosition::VECCALC> idxBuf_;

    LocalTensor<CONVERT_TYPE> values_;
    LocalTensor<IDX_T> indices_;

    uint32_t blockIdx_ = 0;
    uint32_t blockDim_ = 0;
    uint32_t segmentLen_ = 0;
    uint32_t segmentsPerBatch_ = 0;
    uint32_t batchNum_ = 0;
    uint32_t valueRowStride_ = 0;
    uint32_t castRowStride_ = 0;
    uint32_t indexRowStride_ = 0;
};

} // namespace SmallAxisCommon

#endif
