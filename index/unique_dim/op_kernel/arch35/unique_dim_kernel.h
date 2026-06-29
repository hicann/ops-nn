/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UNIQUE_DIM_KERNEL_H
#define UNIQUE_DIM_KERNEL_H

#include "kernel_tiling/kernel_tiling.h"
#include "unique_dim_constant.h"
#include "unique_dim_simt.h"
#include "unique_dim_simt_sort.h"

// ============================================================
// Shape output helpers (shared between kernel class and apt entry)
// ============================================================
__aicore__ inline void FillEmptyShapeValues(LocalTensor<uint64_t> &shapeTensor)
{
    shapeTensor.SetValue(SHAPE0_SIZE_IDX, UINT64_SHAPE_DIM_ONE);
    shapeTensor.SetValue(SHAPE0_DIM0_IDX, 0);
    shapeTensor.SetValue(SHAPE1_SIZE_IDX, UINT64_SHAPE_DIM_ONE);
    shapeTensor.SetValue(SHAPE1_DIM0_IDX, 0);
    shapeTensor.SetValue(SHAPE2_SIZE_IDX, UINT64_SHAPE_DIM_ONE);
    shapeTensor.SetValue(SHAPE2_DIM0_IDX, 0);
}

__aicore__ inline void SetShapeCopyParams(DataCopyExtParams &params)
{
    params.blockCount = 1;
    params.blockLen = SHAPE_LEN * sizeof(uint64_t);
    params.srcStride = 0;
    params.dstStride = 0;
}

// ============================================================
// Main Kernel Class
// ============================================================
template <typename T, bool RETURN_INVERSE, bool RETURN_COUNTS>
class UniqueDimKernel {
public:
    __aicore__ inline UniqueDimKernel(TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR valueOut, GM_ADDR inverseOut, GM_ADDR countsOut,
                                GM_ADDR shapeOut, GM_ADDR workspace, const UniqueDimTilingData *tiling)
    {
        coreId_ = GetBlockIdx();
        coreNum_ = GetBlockNum();
        isFinalCore_ = (coreId_ == coreNum_ - 1);

        numInp_ = tiling->numInp;
        rowLen_ = tiling->rowLen;
        elementsPerCore_ = tiling->elementsPerCore;
        tailCoreElements_ = tiling->tailCoreElements;
        blockDimX_ = static_cast<int32_t>(tiling->blockDimX);
        tileSize_ = tiling->tileSize;

        dim_ = tiling->dim;
        outerSize_ = tiling->outerSize;
        innerSize_ = tiling->innerSize;
        transposeDstOffset_ = tiling->transposeDstOffset;

        inputDimNum_ = tiling->inputDimNum;
        inputDims_[0] = tiling->inputDim0;
        inputDims_[1] = tiling->inputDim1;
        inputDims_[2] = tiling->inputDim2;
        inputDims_[3] = tiling->inputDim3;
        inputDims_[4] = tiling->inputDim4;
        inputDims_[5] = tiling->inputDim5;
        inputDims_[6] = tiling->inputDim6;
        inputDims_[7] = tiling->inputDim7;

        coreStart_ = coreId_ * elementsPerCore_;
        coreLen_ = isFinalCore_ ? tailCoreElements_ : elementsPerCore_;
        // Clamp: cores beyond numInp_ have no sort work
        if (coreStart_ >= numInp_) {
            coreLen_ = 0;
        } else if (coreStart_ + coreLen_ > numInp_) {
            coreLen_ = numInp_ - coreStart_;
        }

        inputFlatGm_.SetGlobalBuffer((__gm__ T *)(x));
        valueOutGm_.SetGlobalBuffer((__gm__ T *)(valueOut));
        inverseOutGm_.SetGlobalBuffer((__gm__ int64_t *)(inverseOut));
        countsOutGm_.SetGlobalBuffer((__gm__ int64_t *)(countsOut));
        shapeOutGm_.SetGlobalBuffer((__gm__ uint64_t *)(shapeOut));

        ws_ = workspace;
        indicesGm_.SetGlobalBuffer((__gm__ uint32_t *)(ws_ + tiling->indicesOffset));
        sortBufGm_.SetGlobalBuffer((__gm__ uint32_t *)(ws_ + tiling->sortBufOffset));
        flagsGm_.SetGlobalBuffer((__gm__ uint32_t *)(ws_ + tiling->flagsOffset));
        positionsGm_.SetGlobalBuffer((__gm__ uint32_t *)(ws_ + tiling->positionsOffset));
        partialSumGm_.SetGlobalBuffer((__gm__ uint32_t *)(ws_ + tiling->partialSumOffset));
        globalPrefixGm_.SetGlobalBuffer((__gm__ uint32_t *)(ws_ + tiling->globalPrefixOffset));

        // UB single buffer: 2 * tileSize * 8B = 8KB (fits __local_mem__ limit)
        // First half [0..tileSize-1] = bufA, second half [tileSize..2*tileSize-1] = bufB
        uint32_t blockSize = GetDataBlockSizeInBytes();
        int64_t sortBufUbSize = CeilAlign(tileSize_ * sizeof(int64_t), blockSize);
        pipe_->InitBuffer(ubBuf_, 2 * sortBufUbSize);

        // Cache UB pointer for Simt (single pointer, halves via index offset)
        ubBufPtr64_ = (__local_mem__ int64_t *)ubBuf_.Get<int64_t>().GetPhyAddr();

        int64_t shapeBufSize = CeilAlign(SHAPE_LEN * sizeof(uint64_t), blockSize);
        pipe_->InitBuffer(shapeBuf_, shapeBufSize);

        effectiveInputFlat_ = (__gm__ T *)inputFlatGm_.GetPhyAddr();
    }

    __aicore__ inline void Process()
    {
        if (numInp_ == 0) {
            OutputEmptyShape();
            return;
        }

        Phase0_InitIndices();
        SyncAll();

        Phase_transpose();

        Phase1_BlockSort();
        SyncAll();

        Phase2_CrossCoreMerge();

        Phase3_AdjacentDiff();
        SyncAll();

        Phase4_PrefixSum();

        if constexpr (RETURN_INVERSE) {
            Phase5_ScatterInverse();
            SyncAll();
        }

        Phase6_UniqueDetectAndGather();

        Phase_transpose_back();

        if constexpr (RETURN_COUNTS) {
            Phase7_ComputeCounts();
        }

        Phase8_OutputShape();
    }

private:
    // ============================================================
    // Helpers
    // ============================================================
    __aicore__ inline int64_t CeilAlign(int64_t size, uint32_t alignment)
    {
        return (alignment == 0) ? size : (size + alignment - 1) / alignment * alignment;
    }

    __aicore__ inline int64_t Min64(int64_t a, int64_t b) { return a < b ? a : b; }

    __aicore__ inline void SplitWorkByCore(int64_t total, int64_t &myStart, int64_t &myLen)
    {
        int64_t perCore = CeilDiv64(total, coreNum_);
        myStart = coreId_ * perCore;
        myLen = Min64(perCore, total - myStart);
        if (myStart >= total) {
            myLen = 0;
        }
    }

    __aicore__ inline int64_t CeilDiv64(int64_t a, int64_t b) { return (b == 0) ? 0 : (a + b - 1) / b; }

    template <HardEvent ent>
    __aicore__ inline void SyncEvent()
    {
        event_t event = static_cast<event_t>(pipe_->FetchEventID(ent));
        SetFlag<ent>(event);
        WaitFlag<ent>(event);
    }

    // ============================================================
    // Phase 0: Init indices
    // ============================================================
    __aicore__ inline void Phase0_InitIndices()
    {
        asc_vf_call<SimtInitIndices>(
            dim3(blockDimX_),
            coreStart_, coreLen_,
            (__gm__ uint32_t *)indicesGm_.GetPhyAddr());
    }

    // ============================================================
    // Phase transpose: transpose input data if dim!=0
    // Rearranges [outerSize, numInp, innerSize] → [numInp, outerSize, innerSize]
    // so that slices along the target dim become contiguous rows.
    // Splits work across cores: each core handles its own range in flat index space.
    // ============================================================
    __aicore__ inline void Phase_transpose()
    {
        if (dim_ == 0) {
            return;
        }
        int64_t total = outerSize_ * numInp_ * innerSize_;
        int64_t myStart, myLen;
        SplitWorkByCore(total, myStart, myLen);

        if (myLen > 0) {
            asc_vf_call<SimtTransposeDim<T>>(
                dim3(GM_BLOCK_DIM),
                myStart, myLen,
                outerSize_, numInp_, innerSize_,
                effectiveInputFlat_,
                (__gm__ T *)(ws_ + transposeDstOffset_));
        }
        effectiveInputFlat_ = (__gm__ T *)(ws_ + transposeDstOffset_);
        SyncAll();
    }

    // ============================================================
    // Phase transpose back: transpose valueOut back to original layout
    // valueOut is [numOut, outerSize, innerSize] (transposed) after gather.
    // Transpose to [outerSize, numOut, innerSize] (original dim order), then copy back.
    // Reuses the same workspace buffer (no longer needed for input transpose).
    // ============================================================
    __aicore__ inline void Phase_transpose_back()
    {
        if (dim_ == 0) {
            return;
        }
        int64_t numOut = numOut_;
        int64_t total = numOut * outerSize_ * innerSize_;
        int64_t myStart, myLen;
        SplitWorkByCore(total, myStart, myLen);

        // valueOut [numOut, outerSize, innerSize] → workspace [outerSize, numOut, innerSize]
        if (myLen > 0) {
            asc_vf_call<SimtTransposeDim<T>>(
                dim3(GM_BLOCK_DIM),
                myStart, myLen,
                numOut, outerSize_, innerSize_,
                (__gm__ T *)valueOutGm_.GetPhyAddr(),
                (__gm__ T *)(ws_ + transposeDstOffset_));
        }
        SyncAll();

        // Copy workspace → valueOut (also split across cores)
        int64_t totalCopy = numOut * rowLen_;
        int64_t copyStart, copyLen;
        SplitWorkByCore(totalCopy, copyStart, copyLen);
        if (copyLen > 0) {
            asc_vf_call<SimtCopyFlat<T>>(
                dim3(GM_BLOCK_DIM),
                copyStart, copyLen,
                (__gm__ T *)(ws_ + transposeDstOffset_),
                (__gm__ T *)valueOutGm_.GetPhyAddr());
        }
        SyncAll();
    }

    // ============================================================
    // Phase 1: Per-core sort — tile sort + intra-core merge in single VF_CALL
    // ============================================================
    __aicore__ inline void Phase1_BlockSort()
    {
        if (coreLen_ <= 1) {
            return;
        }

        int64_t numLocalTiles = CeilDiv64(coreLen_, tileSize_);

        // Single VF_CALL: sort tiles in UB + multi-round merge on GM
        asc_vf_call<SimtPhase1AllInOne<T>>(
            dim3(blockDimX_),
            coreStart_, coreLen_, tileSize_, numLocalTiles,
            (__gm__ uint32_t *)indicesGm_.GetPhyAddr(),
            (__gm__ uint32_t *)sortBufGm_.GetPhyAddr(),
            effectiveInputFlat_,
            rowLen_,
            ubBufPtr64_);
    }

    // ============================================================
    // Phase 2 helper: Copy unmerged tail from src to dst
    // ============================================================
    __aicore__ inline void Phase2_CopyUnmergedTail(int32_t round, int64_t width)
    {
        int64_t numValidPairs = (2 * width == 0) ? 0 : (numInp_ - width - 1) / (2 * width) + 1;
        int64_t tailOff = numValidPairs * 2 * width;
        int64_t tailLen = numInp_ - tailOff;
        if (tailLen > 0) {
            if (coreId_ == 0) {
                __gm__ uint32_t *tailSrc = (round % 2 == 0) ?
                    (__gm__ uint32_t *)indicesGm_.GetPhyAddr() :
                    (__gm__ uint32_t *)sortBufGm_.GetPhyAddr();
                __gm__ uint32_t *tailDst = (round % 2 == 0) ?
                    (__gm__ uint32_t *)sortBufGm_.GetPhyAddr() :
                    (__gm__ uint32_t *)indicesGm_.GetPhyAddr();
                asc_vf_call<SimtCopyRange>(
                    dim3(blockDimX_),
                    tailOff, tailOff + tailLen, tailSrc, tailDst);
            }
            PipeBarrier<PIPE_ALL>();
            SyncAll();
        }
    }

    // ============================================================
    // Phase 2 helper: Copy sortBuf → indices if odd rounds
    // ============================================================
    __aicore__ inline void Phase2_CopyBackIfOdd(int32_t round)
    {
        if (round % 2 == 1) {
            int64_t copyPerCore = CeilDiv64(numInp_, coreNum_);
            int64_t copyStart = coreId_ * copyPerCore;
            int64_t copyEnd = Min64(copyStart + copyPerCore, numInp_);
            if (copyStart < copyEnd) {
                asc_vf_call<SimtCopyRange>(
                    dim3(blockDimX_),
                    copyStart, copyEnd,
                    (__gm__ uint32_t *)sortBufGm_.GetPhyAddr(),
                    (__gm__ uint32_t *)indicesGm_.GetPhyAddr());
            }
            PipeBarrier<PIPE_ALL>();
            SyncAll();
        }
    }

    // ============================================================
    // Phase 2: Cross-core merge — all-core cooperative via merge path
    // All cores cooperate on each pair. Pairs are sequential within a round.
    // ============================================================
    __aicore__ inline void Phase2_CrossCoreMerge()
    {
        int64_t width = elementsPerCore_;
        int32_t round = 0;

        while (width < numInp_) {
            int64_t numPairs = CeilDiv64(numInp_, 2 * width);

            __gm__ uint32_t *src = (round % 2 == 0) ?
                (__gm__ uint32_t *)indicesGm_.GetPhyAddr() :
                (__gm__ uint32_t *)sortBufGm_.GetPhyAddr();
            __gm__ uint32_t *dst = (round % 2 == 0) ?
                (__gm__ uint32_t *)sortBufGm_.GetPhyAddr() :
                (__gm__ uint32_t *)indicesGm_.GetPhyAddr();

            for (int64_t p = 0; p < numPairs; p++) {
                int64_t mergeStart = p * 2 * width;
                int64_t mergeMid = Min64(mergeStart + width, numInp_);
                int64_t mergeEnd = Min64(mergeStart + 2 * width, numInp_);

                if (mergeMid < mergeEnd) {
                    asc_vf_call<SimtMergePathPair<T>>(
                        dim3(blockDimX_),
                        coreId_, coreNum_,
                        mergeStart, mergeMid, mergeEnd,
                        src, dst,
                        effectiveInputFlat_,
                        rowLen_);
                }
                PipeBarrier<PIPE_ALL>();
                SyncAll();
            }

            Phase2_CopyUnmergedTail(round, width);

            width *= 2;
            round++;
        }

        Phase2_CopyBackIfOdd(round);
    }

    // ============================================================
    // Phase 3: Adjacent diff
    // ============================================================
    __aicore__ inline void Phase3_AdjacentDiff()
    {
        asc_vf_call<SimtAdjacentDiff<T>>(
            dim3(blockDimX_),
            coreStart_, coreLen_,
            (__gm__ uint32_t *)indicesGm_.GetPhyAddr(),
            (__gm__ uint32_t *)flagsGm_.GetPhyAddr(),
            effectiveInputFlat_,
            rowLen_);
    }

    // ============================================================
    // Phase 4: Prefix sum
    // ============================================================
    __aicore__ inline void Phase4_PrefixSum()
    {
        int64_t psStride = MAGIC_GM_PAGE_SIZE / sizeof(uint32_t);
        asc_vf_call<SimtLocalScan>(
            dim3(blockDimX_),
            coreStart_, coreLen_,
            (__gm__ uint32_t *)flagsGm_.GetPhyAddr(),
            coreId_,
            (__gm__ uint32_t *)partialSumGm_.GetPhyAddr(),
            psStride);
        SyncAll();

        if (coreId_ == 0) {
            asc_vf_call<SimtGlobalPrefixScan>(
                dim3(blockDimX_),
                coreNum_, psStride,
                (__gm__ uint32_t *)partialSumGm_.GetPhyAddr(),
                (__gm__ uint32_t *)globalPrefixGm_.GetPhyAddr());
        }

        SyncAll();

        // Read totalUnique from globalPrefix[coreNum] (written by SimtGlobalPrefixScan, flushed by MTE3_S)
        __gm__ uint32_t *gpPtr = (__gm__ uint32_t *)globalPrefixGm_.GetPhyAddr();
        totalUnique_ = static_cast<int64_t>(gpPtr[coreNum_]);

        asc_vf_call<SimtAddGlobalPrefix>(
            dim3(blockDimX_),
            coreStart_, coreLen_,
            (__gm__ uint32_t *)flagsGm_.GetPhyAddr(),
            coreId_,
            (__gm__ uint32_t *)globalPrefixGm_.GetPhyAddr());
        SyncAll();
    }

    // ============================================================
    // Phase 5: Scatter inverse
    // ============================================================
    __aicore__ inline void Phase5_ScatterInverse()
    {
        asc_vf_call<SimtScatterInverse>(
            dim3(blockDimX_),
            coreStart_, coreLen_,
            (__gm__ uint32_t *)indicesGm_.GetPhyAddr(),
            (__gm__ uint32_t *)flagsGm_.GetPhyAddr(),
            (__gm__ int64_t *)inverseOutGm_.GetPhyAddr());
    }

    // ============================================================
    // Phase 6: Unique detect + compact + gather
    // ============================================================
    __aicore__ inline void Phase6_UniqueDetectAndGather()
    {
        int64_t returnCountsFlag = RETURN_COUNTS ? 1 : 0;
        asc_vf_call<SimtUniqueDetect>(
            dim3(blockDimX_),
            coreStart_, coreLen_,
            (__gm__ uint32_t *)indicesGm_.GetPhyAddr(),
            (__gm__ uint32_t *)flagsGm_.GetPhyAddr(),
            (__gm__ uint32_t *)sortBufGm_.GetPhyAddr(),
            (__gm__ uint32_t *)positionsGm_.GetPhyAddr(),
            returnCountsFlag);

        if (coreId_ == 0) {
            asc_vf_call<SimtWriteValue>(
                dim3(blockDimX_),
                (__gm__ uint32_t *)globalPrefixGm_.GetPhyAddr(),
                coreNum_, static_cast<uint32_t>(totalUnique_));
        }
        SyncAll();

        __gm__ uint32_t *gpPtr2 = (__gm__ uint32_t *)globalPrefixGm_.GetPhyAddr();
        int64_t numOut = static_cast<int64_t>(gpPtr2[coreNum_]);

        // Split the gather over total output elements (numOut * rowLen) so all
        // cores participate even when numOut is small but rows are long, and
        // launch with GM_BLOCK_DIM (2048) threads like the transpose/copy path.
        int64_t gatherTotal = numOut * rowLen_;
        int64_t eStart, eLen;
        SplitWorkByCore(gatherTotal, eStart, eLen);

        if (eLen > 0) {
            asc_vf_call<SimtGatherRows<T>>(
                dim3(GM_BLOCK_DIM),
                eStart, eLen,
                (__gm__ uint32_t *)sortBufGm_.GetPhyAddr(),
                effectiveInputFlat_,
                (__gm__ T *)valueOutGm_.GetPhyAddr(),
                rowLen_);
        }

        numOut_ = numOut;

        if constexpr (RETURN_COUNTS) {
            if (coreId_ == 0) {
                asc_vf_call<SimtWriteValue>(
                    dim3(blockDimX_),
                    (__gm__ uint32_t *)positionsGm_.GetPhyAddr(),
                    numOut, static_cast<uint32_t>(numInp_));
            }
            SyncAll();
        } else {
            SyncAll();
        }
    }

    // ============================================================
    // Phase 7: Compute counts
    // ============================================================
    __aicore__ inline void Phase7_ComputeCounts()
    {
        int64_t numOut = numOut_;
        int64_t cntPerCore = CeilDiv64(numOut, coreNum_);
        int64_t cntStart = coreId_ * cntPerCore;
        int64_t cntEnd = Min64(cntStart + cntPerCore, numOut);

        if (cntStart < cntEnd) {
            asc_vf_call<SimtComputeCounts>(
                dim3(blockDimX_),
                cntStart, cntEnd,
                (__gm__ uint32_t *)positionsGm_.GetPhyAddr(),
                (__gm__ int64_t *)countsOutGm_.GetPhyAddr());
        }
        SyncAll();
    }

    // ============================================================
    // Phase 8: Output shape
    // ============================================================
    __aicore__ inline void Phase8_OutputShape()
    {
        if (!isFinalCore_) {
            return;
        }

        int64_t numOut = numOut_;

        LocalTensor<uint64_t> shapeTensor = shapeBuf_.Get<uint64_t>();
        Duplicate(shapeTensor, (uint64_t)1, SHAPE_LEN);
        SyncEvent<HardEvent::V_S>();

        // Shape0: valueOut — always preserve original ndim, replace dim_k with numOut
        shapeTensor.SetValue(SHAPE0_SIZE_IDX, 0x80000000 | inputDimNum_);
        for (int64_t i = 0; i < inputDimNum_; i++) {
            int64_t dimVal = (i == dim_) ? numOut : inputDims_[i];
            shapeTensor.SetValue(SHAPE0_DIM0_IDX + i, dimVal);
        }

        shapeTensor.SetValue(SHAPE1_SIZE_IDX, UINT64_SHAPE_DIM_ONE);
        if constexpr (RETURN_INVERSE) {
            shapeTensor.SetValue(SHAPE1_DIM0_IDX, numInp_);
        } else {
            shapeTensor.SetValue(SHAPE1_DIM0_IDX, 1);
        }

        shapeTensor.SetValue(SHAPE2_SIZE_IDX, UINT64_SHAPE_DIM_ONE);
        if constexpr (RETURN_COUNTS) {
            shapeTensor.SetValue(SHAPE2_DIM0_IDX, numOut);
        } else {
            shapeTensor.SetValue(SHAPE2_DIM0_IDX, 1);
        }

        DataCopyExtParams dataCopyParams;
        SetShapeCopyParams(dataCopyParams);

        SyncEvent<HardEvent::S_MTE3>();
        DataCopyPad(shapeOutGm_, shapeTensor, dataCopyParams);
    }

    __aicore__ inline void OutputEmptyShape()
    {
        if (!isFinalCore_) {
            return;
        }

        LocalTensor<uint64_t> shapeTensor = shapeBuf_.Get<uint64_t>();
        Duplicate(shapeTensor, (uint64_t)1, SHAPE_LEN);
        SyncEvent<HardEvent::V_S>();

        FillEmptyShapeValues(shapeTensor);

        DataCopyExtParams dataCopyParams;
        SetShapeCopyParams(dataCopyParams);

        SyncEvent<HardEvent::S_MTE3>();
        DataCopyPad(shapeOutGm_, shapeTensor, dataCopyParams);
    }

private:
    TPipe *pipe_;

    // UB buffers: combined sort/tile merge buffer + shape output
    TBuf<TPosition::VECCALC> ubBuf_;
    TBuf<TPosition::VECCALC> shapeBuf_;

    // Cached UB pointer for Simt (single pointer, halves via index offset)
    __local_mem__ int64_t *ubBufPtr64_ = nullptr;

    // GM workspace sub-buffers (uint32_t to save ~50% workspace memory)
    GM_ADDR ws_;
    GlobalTensor<T> inputFlatGm_;
    GlobalTensor<uint32_t> indicesGm_;
    GlobalTensor<uint32_t> sortBufGm_;
    GlobalTensor<uint32_t> flagsGm_;
    GlobalTensor<uint32_t> positionsGm_;
    GlobalTensor<uint32_t> partialSumGm_;
    GlobalTensor<uint32_t> globalPrefixGm_;

    // Output GM (actual operator outputs, not workspace)
    GlobalTensor<T> valueOutGm_;
    GlobalTensor<int64_t> inverseOutGm_;
    GlobalTensor<int64_t> countsOutGm_;
    GlobalTensor<uint64_t> shapeOutGm_;

    // Parameters
    int64_t numInp_ = 0;
    int64_t rowLen_ = 0;
    int32_t coreId_ = 0;
    int64_t coreNum_ = 1;
    int64_t elementsPerCore_ = 0;
    int64_t tailCoreElements_ = 0;
    int32_t blockDimX_ = 256;
    int64_t tileSize_ = 1024;
    int64_t coreStart_ = 0;
    int64_t coreLen_ = 0;
    bool isFinalCore_ = false;

    // Transpose support for dim!=0
    int64_t dim_ = 0;
    int64_t outerSize_ = 0;
    int64_t innerSize_ = 0;
    int64_t transposeDstOffset_ = 0;
    __gm__ T *effectiveInputFlat_ = nullptr;

    // Original input shape for output shape construction
    int64_t inputDimNum_ = 0;
    int64_t inputDims_[8] = {0};

    // Runtime result
    int64_t totalUnique_ = 0;
    int64_t numOut_ = 0;
};

#endif // UNIQUE_DIM_KERNEL_H
