/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_NORM_CENTRALIZATION_GENERIC_CONTIGUOUS_IMPL_H_
#define OPS_NORM_CENTRALIZATION_GENERIC_CONTIGUOUS_IMPL_H_

__aicore__ inline void ProcessSmallInnerOneTask(int64_t outer)
{
    const int64_t blockElements = tiling_->smallContiguousBlockElements;
    const int64_t base = ContiguousBase(outer, 0);
    CopyInContiguous(base, blockElements);
    LocalTensor<T> input = inQueue_.DeQue<T>();
    LocalTensor<T> output = outQueue_.AllocTensor<T>();
    const float sum = ReduceSegmentToScalar(input, blockElements);
    const float mean = sum / static_cast<float>(tiling_->reduceCount);
    CentralizeSegmentScalar(input, output, blockElements, mean);
    PipeBarrier<PIPE_V>();
    outQueue_.EnQue(output);
    inQueue_.FreeTensor(input);
    CopyOutContiguous(base, blockElements);
}

__aicore__ inline void ProcessSmallWholeBlockTask(int64_t outer)
{
    const int64_t meanElements = tiling_->smallContiguousMeanElements;
    const int64_t rowElements = tiling_->contiguousInnerCount;
    const int64_t rowStride = AlignToVector(rowElements);
    const int64_t base = ContiguousBase(outer, 0);
    CopyInSmallPadded(base, rowElements, rowStride);
    LocalTensor<T> input = inQueue_.DeQue<T>();
    LocalTensor<T> output = outQueue_.AllocTensor<T>();
    LocalTensor<float> sum = sumBuf_.Get<float>();
    const uint16_t numSlots = static_cast<uint16_t>(tiling_->numAccSlots > 0 ? tiling_->numAccSlots : 1);
    const int64_t sumStride = AlignToVector(tiling_->smallContiguousMeanElements);
    for (uint16_t slot = 0; slot < numSlots; ++slot) {
        ZeroSumTileAt(sum, static_cast<int64_t>(slot) * sumStride, meanElements);
    }
    PipeBarrier<PIPE_V>();
    uint16_t slot = 0;
    for (int64_t r = 0; r < tiling_->reduceCount; ++r) {
        const int64_t rowOffset = r * rowStride;
        AccumulateContiguousAt(input, sum, static_cast<int64_t>(slot) * sumStride, rowElements, rowOffset);
        slot = static_cast<uint16_t>((slot + 1 == numSlots) ? 0 : (slot + 1));
    }
    FoldAccumulatorSlots(sum, numSlots, sumStride, meanElements);
    PipeBarrier<PIPE_V>();
    ScaleSumToMeanAt(sum, 0, meanElements);
    PipeBarrier<PIPE_V>();
    for (int64_t r = 0; r < tiling_->reduceCount; ++r) {
        const int64_t rowOffset = r * rowStride;
        CentralizeContiguousAt(input, output, sum, 0, rowElements, rowOffset, rowOffset);
    }
    PipeBarrier<PIPE_V>();
    outQueue_.EnQue(output);
    inQueue_.FreeTensor(input);
    CopyOutSmallPadded(base, rowElements, rowStride);
}

__aicore__ inline void ProcessSmallContiguous(uint32_t core, uint32_t cores)
{
    const int64_t taskCount = tiling_->contiguousOuterCount;
    for (int64_t task = core; task < taskCount; task += cores) {
        if (tiling_->smallContiguousMode == 1) {
            ProcessSmallInnerOneTask(task);
        } else {
            ProcessSmallWholeBlockTask(task);
        }
    }
}

__aicore__ inline void AccumulateRowsSmallInner(LocalTensor<T>& input, LocalTensor<float>& sum, int64_t validRows,
                                                int64_t innerCount, int64_t rowStride)
{
    __local_mem__ T* inputAddr = (__local_mem__ T*)input.GetPhyAddr();
    __local_mem__ float* sumAddr = (__local_mem__ float*)sum.GetPhyAddr();
    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> sumReg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        uint32_t activeCount = static_cast<uint32_t>(innerCount);
        const uint16_t rowLoops = static_cast<uint16_t>(validRows);
        MaskReg preg = activeCount == V_LENGTH ? pregFull : UpdateMask<float>(activeCount);
        LoadRegForDtype<float>(sumAddr, sumReg, preg, 0);
        for (uint16_t r = 0; r < rowLoops; ++r) {
            LoadRegForDtype<T>(inputAddr, xReg, preg, static_cast<uint32_t>(r * rowStride));
            Add(sumReg, sumReg, xReg, preg);
        }
        StoreRegForDtype<float>(sumAddr, sumReg, preg, 0);
    }
}

__aicore__ inline void CentralizeRowsSmallInner(LocalTensor<T>& input, LocalTensor<T>& output, LocalTensor<float>& mean,
                                                int64_t validRows, int64_t innerCount, int64_t rowStride)
{
    __local_mem__ T* inputAddr = (__local_mem__ T*)input.GetPhyAddr();
    __local_mem__ T* outputAddr = (__local_mem__ T*)output.GetPhyAddr();
    __local_mem__ float* meanAddr = (__local_mem__ float*)mean.GetPhyAddr();
    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> meanReg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        uint32_t activeCount = static_cast<uint32_t>(innerCount);
        const uint16_t rowLoops = static_cast<uint16_t>(validRows);
        MaskReg preg = activeCount == V_LENGTH ? pregFull : UpdateMask<float>(activeCount);
        LoadRegForDtype<float>(meanAddr, meanReg, preg, 0);
        for (uint16_t r = 0; r < rowLoops; ++r) {
            const uint32_t rowOffset = static_cast<uint32_t>(r * rowStride);
            LoadRegForDtype<T>(inputAddr, xReg, preg, rowOffset);
            Sub(xReg, xReg, meanReg, preg);
            StoreRegForDtype<T>(outputAddr, xReg, preg, rowOffset);
        }
    }
}

__aicore__ inline void AccumulateRowsSmallInnerKahan(LocalTensor<T>& input, LocalTensor<float>& sum, int64_t sumOffset,
                                                     int64_t corrOffset, int64_t validRows, int64_t innerCount,
                                                     int64_t rowStride)
{
    __local_mem__ T* inputAddr = (__local_mem__ T*)input.GetPhyAddr();
    __local_mem__ float* sumAddr = (__local_mem__ float*)sum.GetPhyAddr();
    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> sumReg;
        RegTensor<float> corrReg;
        RegTensor<float> tReg;
        RegTensor<float> c1Reg;
        RegTensor<float> c2Reg;
        RegTensor<float> addReg;
        RegTensor<float> absSum;
        RegTensor<float> absX;
        MaskReg dom;
        MaskReg fin;
        RegTensor<float> zeroReg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        uint32_t activeCount = static_cast<uint32_t>(innerCount);
        const uint16_t rowLoops = static_cast<uint16_t>(validRows);
        MaskReg preg = activeCount == V_LENGTH ? pregFull : UpdateMask<float>(activeCount);
        Duplicate(zeroReg, 0.0f, pregFull);
        LoadRegForDtype<float>(sumAddr, sumReg, preg, static_cast<uint32_t>(sumOffset));
        LoadRegForDtype<float>(sumAddr, corrReg, preg, static_cast<uint32_t>(corrOffset));
        for (uint16_t r = 0; r < rowLoops; ++r) {
            LoadRegForDtype<T>(inputAddr, xReg, preg, static_cast<uint32_t>(r * rowStride));
            Add(tReg, sumReg, xReg, preg);
            Abs(absSum, sumReg, preg);
            Abs(absX, xReg, preg);
            Compare<float, CMPMODE::GE>(dom, absSum, absX, preg);
            Sub(c1Reg, sumReg, tReg, preg);
            Add(c1Reg, c1Reg, xReg, preg);
            Sub(c2Reg, xReg, tReg, preg);
            Add(c2Reg, c2Reg, sumReg, preg);
            Select(addReg, c1Reg, c2Reg, dom);
            Abs(absSum, tReg, preg);
            Compares<float, CMPMODE::LE>(fin, absSum, 3.4028235e38f, preg);
            Select(addReg, addReg, zeroReg, fin);
            Add(corrReg, corrReg, addReg, preg);
            sumReg = tReg;
        }
        StoreRegForDtype<float>(sumAddr, sumReg, preg, static_cast<uint32_t>(sumOffset));
        StoreRegForDtype<float>(sumAddr, corrReg, preg, static_cast<uint32_t>(corrOffset));
    }
}

__aicore__ inline void CentralizeRowsSmallInnerHiLo(LocalTensor<T>& input, LocalTensor<T>& output,
                                                    LocalTensor<float>& mean, int64_t meanOffset, int64_t meanLowOffset,
                                                    int64_t validRows, int64_t innerCount, int64_t rowStride)
{
    __local_mem__ T* inputAddr = (__local_mem__ T*)input.GetPhyAddr();
    __local_mem__ T* outputAddr = (__local_mem__ T*)output.GetPhyAddr();
    __local_mem__ float* meanAddr = (__local_mem__ float*)mean.GetPhyAddr();
    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> hiReg;
        RegTensor<float> loReg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        uint32_t activeCount = static_cast<uint32_t>(innerCount);
        const uint16_t rowLoops = static_cast<uint16_t>(validRows);
        MaskReg preg = activeCount == V_LENGTH ? pregFull : UpdateMask<float>(activeCount);
        LoadRegForDtype<float>(meanAddr, hiReg, preg, static_cast<uint32_t>(meanOffset));
        LoadRegForDtype<float>(meanAddr, loReg, preg, static_cast<uint32_t>(meanLowOffset));
        for (uint16_t r = 0; r < rowLoops; ++r) {
            const uint32_t rowOffset = static_cast<uint32_t>(r * rowStride);
            LoadRegForDtype<T>(inputAddr, xReg, preg, rowOffset);
            Sub(xReg, xReg, hiReg, preg);
            Sub(xReg, xReg, loReg, preg);
            StoreRegForDtype<T>(outputAddr, xReg, preg, rowOffset);
        }
    }
}

__aicore__ inline void ProcessLargeContiguousSmallInnerTask(int64_t outer)
{
    const int64_t innerCount = tiling_->contiguousInnerCount;
    const int64_t reduceTile = tiling_->largeContiguousReduceTile;
    const int64_t rowStride = AlignToVector(innerCount);
    const int64_t base = ContiguousBase(outer, 0);
    LocalTensor<float> sum = sumBuf_.Get<float>();
    const int64_t corrOffset = AlignToVector(innerCount);
    ZeroSumTile(sum, innerCount);
    ZeroSumTileAt(sum, corrOffset, innerCount);
    PipeBarrier<PIPE_V>();

    for (int64_t reduceOffset = 0; reduceOffset < tiling_->reduceCount; reduceOffset += reduceTile) {
        const int64_t validRows = Min(reduceTile, tiling_->reduceCount - reduceOffset);
        CopyInRowsPadded(base + reduceOffset * innerCount, validRows, innerCount, rowStride);
        LocalTensor<T> input = inQueue_.DeQue<T>();
        AccumulateRowsSmallInnerKahan(input, sum, 0, corrOffset, validRows, innerCount, rowStride);
        PipeBarrier<PIPE_V>();
        inQueue_.FreeTensor(input);
    }

    FinalizeMeanHiLoAt(sum, 0, corrOffset, innerCount);
    PipeBarrier<PIPE_V>();

    for (int64_t reduceOffset = 0; reduceOffset < tiling_->reduceCount; reduceOffset += reduceTile) {
        const int64_t validRows = Min(reduceTile, tiling_->reduceCount - reduceOffset);
        const int64_t gmOffset = base + reduceOffset * innerCount;
        CopyInRowsPadded(gmOffset, validRows, innerCount, rowStride);
        LocalTensor<T> input = inQueue_.DeQue<T>();
        LocalTensor<T> output = outQueue_.AllocTensor<T>();
        CentralizeRowsSmallInnerHiLo(input, output, sum, 0, corrOffset, validRows, innerCount, rowStride);
        PipeBarrier<PIPE_V>();
        outQueue_.EnQue(output);
        inQueue_.FreeTensor(input);
        CopyOutRowsPadded(gmOffset, validRows, innerCount, rowStride);
    }
}

__aicore__ inline void ProcessLargeContiguousSmallInner(uint32_t core, uint32_t cores)
{
    const int64_t taskCount = tiling_->contiguousOuterCount;
    for (int64_t task = core; task < taskCount; task += cores) {
        ProcessLargeContiguousSmallInnerTask(task);
    }
}

__aicore__ inline void ProcessContiguousTask(int64_t outer, int64_t innerOffset, int64_t validElements)
{
    LocalTensor<float> sum = sumBuf_.Get<float>();
    const uint16_t numSlots = static_cast<uint16_t>(tiling_->numAccSlots > 0 ? tiling_->numAccSlots : 1);
    const int64_t sumStride = kLargeChunkElements;
    const int64_t corrBase = static_cast<int64_t>(numSlots) * sumStride;
    for (uint16_t slot = 0; slot < numSlots; ++slot) {
        ZeroSumTileAt(sum, static_cast<int64_t>(slot) * sumStride, validElements);
        ZeroSumTileAt(sum, corrBase + static_cast<int64_t>(slot) * sumStride, validElements);
    }
    PipeBarrier<PIPE_V>();

    const int64_t base = ContiguousBase(outer, innerOffset);
    uint16_t slot = 0;
    for (int64_t r = 0; r < tiling_->reduceCount; ++r) {
        CopyInContiguous(base + r * tiling_->contiguousInnerCount, validElements);
        LocalTensor<T> input = inQueue_.DeQue<T>();
        AccumulateContiguousNeumaierAt(input, sum, static_cast<int64_t>(slot) * sumStride,
                                       corrBase + static_cast<int64_t>(slot) * sumStride, validElements);
        PipeBarrier<PIPE_V>();
        inQueue_.FreeTensor(input);
        slot = static_cast<uint16_t>((slot + 1 == numSlots) ? 0 : (slot + 1));
    }

    FoldAccumulatorSlotsNeumaier(sum, numSlots, sumStride, corrBase, validElements);
    PipeBarrier<PIPE_V>();
    FinalizeMeanHiLoAt(sum, 0, corrBase, validElements);
    PipeBarrier<PIPE_V>();

    for (int64_t r = 0; r < tiling_->reduceCount; ++r) {
        const int64_t offset = base + r * tiling_->contiguousInnerCount;
        CopyInContiguous(offset, validElements);
        LocalTensor<T> input = inQueue_.DeQue<T>();
        LocalTensor<T> output = outQueue_.AllocTensor<T>();
        CentralizeContiguousHiLoAt(input, output, sum, 0, corrBase, validElements);
        PipeBarrier<PIPE_V>();
        outQueue_.EnQue(output);
        inQueue_.FreeTensor(input);
        CopyOutContiguous(offset, validElements);
    }
}

__aicore__ inline void ProcessContiguous(uint32_t core, uint32_t cores)
{
    const int64_t tileElements = tiling_->contiguousInnerTileElements > 0 ?
                                     Min(tiling_->contiguousInnerTileElements, kLargeChunkElements) :
                                     kLargeChunkElements;
    const int64_t innerTileCount = tiling_->contiguousInnerTileCount;
    const int64_t taskCount = tiling_->contiguousOuterCount * innerTileCount;
    for (int64_t task = core; task < taskCount; task += cores) {
        const int64_t outer = task / innerTileCount;
        const int64_t innerTile = task - outer * innerTileCount;
        const int64_t innerOffset = innerTile * tileElements;
        const int64_t validElements = Min(tileElements, tiling_->contiguousInnerCount - innerOffset);
        ProcessContiguousTask(outer, innerOffset, validElements);
    }
}

#endif
