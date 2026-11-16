/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SOFTMAX_FOCAL_LOSS_GRAD_PARALLEL_H
#define SOFTMAX_FOCAL_LOSS_GRAD_PARALLEL_H

#include "softmax_focal_loss_grad_nd_vec.h"

namespace SoftmaxFocalLossGrad {

template <typename T, typename TW, uint64_t hasWeight, uint64_t gammaMode>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight, gammaMode>::StorePartial(int64_t chunk)
{
    // Publish the raw chunk expansion before its merge into the temporary accumulator.
    const bool exact = ExactPath();
    LocalTensor<float> source = exact ? dfBuf_.template Get<float>() : accBuf_.template Get<float>();
    const int64_t sourceStride = exact ? VL_FP32 * ROW_SLOT_ELEM : accStride_;
    LocalTensor<float> stage = gradQue_.template AllocTensor<float>();
    auto src = (__ubuf__ float*)source.GetPhyAddr();
    auto dst = (__ubuf__ float*)stage.GetPhyAddr();
    uint32_t slotLanes = ROW_SLOT_ELEM;
    __VEC_SCOPE__
    {
        RegTensor<float> value;
        MaskReg mask = UpdateMask<float>(slotLanes);
        // Four high components followed by four low components, one 32B slot each.
        for (uint16_t slot = 0; slot < 8; ++slot) {
            AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(value, src + slot * sourceStride);
            AscendC::Reg::StoreAlign(dst + slot * ROW_SLOT_ELEM, value, mask);
        }
    }
    gradQue_.template EnQue<float>(stage);
    stage = gradQue_.template DeQue<float>();
    DataCopy(partialGm_[chunk * 8 * ROW_SLOT_ELEM], stage, 8 * ROW_SLOT_ELEM);
    gradQue_.FreeTensor(stage);
}

template <typename T, typename TW, uint64_t hasWeight, uint64_t gammaMode>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight, gammaMode>::MergePartials(int64_t row)
{
    LocalTensor<float> acc = accBuf_.template Get<float>();
    Duplicate(acc, 0.0f, static_cast<int32_t>(8 * accStride_));
    PipeBarrier<PIPE_V>();
    // The first pass has released the input queue, which can hold a batch of partials.
    int64_t queueBytes = (rowsPerTile_ * colsPerChunk_ + VL_FP32) * sizeof(T);
    int64_t batchChunks = queueBytes / (8 * BLOCK_SIZE);
    constexpr int64_t MAX_BATCH_CHUNKS = 64;
    batchChunks = batchChunks > MAX_BATCH_CHUNKS ? MAX_BATCH_CHUNKS : batchChunks;
    for (int64_t first = 0; first < chunkNum_; first += batchChunks) {
        int64_t count = chunkNum_ - first;
        count = count > batchChunks ? batchChunks : count;
        LocalTensor<float> parts = predQue_.template AllocTensor<float>();
        DataCopy(parts, partialGm_[(row * chunkNum_ + first) * 8 * ROW_SLOT_ELEM],
                 static_cast<uint32_t>(count * 8 * ROW_SLOT_ELEM));
        predQue_.template EnQue<float>(parts);
        parts = predQue_.template DeQue<float>();
        auto data = (__ubuf__ float*)parts.GetPhyAddr();
        for (int64_t offset = 0; offset < count; ++offset) {
            auto rowHi = data + offset * 8 * ROW_SLOT_ELEM;
            auto rowLo = rowHi + 4 * ROW_SLOT_ELEM;
            // Consume chunks in their original order using the existing pair merge.
            MergeRowSlots((__ubuf__ float*)acc.GetPhyAddr(), rowHi, rowLo, ROW_SLOT_ELEM, 0, 1);
        }
        predQue_.FreeTensor(parts);
    }
    if (sizeof(T) != sizeof(half) || r_ <= static_cast<int64_t>(VL_FP32)) {
        for (uint16_t slot = 0; slot < 4; ++slot) {
            Add(acc[slot * accStride_], acc[slot * accStride_], acc[(slot + 4) * accStride_], 1);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename T, typename TW, uint64_t hasWeight, uint64_t gammaMode>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight, gammaMode>::ProcessSplitRow()
{
    int64_t block = static_cast<int64_t>(GetBlockIdx());
    int64_t coresPerRow = realCoreNum_ / a_;
    int64_t row = block / coresPerRow;
    int64_t rowCore = block % coresPerRow;
    int64_t baseChunks = chunkNum_ / coresPerRow;
    int64_t extraCores = chunkNum_ % coresPerRow;
    int64_t first = rowCore * baseChunks + (rowCore < extraCores ? rowCore : extraCores);
    int64_t count = baseChunks + (rowCore < extraCores ? 1 : 0);
    LocalTensor<float> acc = accBuf_.template Get<float>();
    LocalTensor<float> wf = wfBuf_.template Get<float>();
    LocalTensor<float> wd = wdBuf_.template Get<float>();
    LocalTensor<float> ce = ceBuf_.template Get<float>();
    LocalTensor<float> wt = wtBuf_.template Get<float>();
    for (int64_t chunk = first; chunk < first + count; ++chunk) {
        int64_t len = r_ - chunk * colsPerChunk_;
        len = len > colsPerChunk_ ? colsPerChunk_ : len;
        Duplicate(acc, 0.0f, static_cast<int32_t>(8 * accStride_));
        CopyIn(row, chunk * colsPerChunk_, 1, len);
        ComputeSums(1, len);
        ExactRowSum4((__ubuf__ float*)wf.GetPhyAddr(), (__ubuf__ float*)wd.GetPhyAddr(),
                     (__ubuf__ float*)ce.GetPhyAddr(), (__ubuf__ float*)wt.GetPhyAddr(),
                     (__ubuf__ float*)acc.GetPhyAddr(), 1, len);
        StorePartial(row * chunkNum_ + chunk);
    }
    // All launched cores publish disjoint chunks before any core reads the row sums.
    PipeBarrier<PIPE_MTE3>();
    SyncAll();
    MergePartials(row);
    for (int64_t chunk = first; chunk < first + count; ++chunk) {
        int64_t len = r_ - chunk * colsPerChunk_;
        len = len > colsPerChunk_ ? colsPerChunk_ : len;
        CopyIn(row, chunk * colsPerChunk_, 1, len);
        ComputeGrad(1, len);
        CopyOut(row, chunk * colsPerChunk_, 1, len);
    }
}

} // namespace SoftmaxFocalLossGrad
#endif
