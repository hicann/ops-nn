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
 * \file merge_more_core_base.h
 * \brief Common constants, free functions, and CRTP base class shared by sort and kth_value
 *        merge_sort_more_core kernels.
 */

#ifndef MERGE_MORE_CORE_BASE_H
#define MERGE_MORE_CORE_BASE_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "merge_sort_constants.h"
#include "ping_pong_merge_sort.h"
#include "util_type_simd.h"

namespace MergeMoreCoreCommon {

using namespace AscendC;

using MergeSortConstants::DEALING_EXTRACT_NUM_ONCE;
using MergeSortConstants::DEALING_SORT_NUM_ONCE;
using MergeSortConstants::FP32_DTYPE_BYTES;
using MergeSortConstants::MERGE_LIST_MAX_NUM;
using MergeSortConstants::MERGE_MORE_BUFFER_NUM;
using MergeSortConstants::MERGE_WORKSPACE_BUFFER_NUM;
using MergeSortConstants::THREE_WAY_MERGE_LIST_NUM;
using MergeSortConstants::TWO_WAY_MERGE_LIST_NUM;
using MergeSortConstants::UB_BLOCK_BYTES;
using MergeSortConstants::XOR_OP_VALUE_FP;
using MergeSortConstants::XOR_OP_VALUE_HALF;

// ======================== CRTP Base Class ========================

/*!
 * \brief CRTP base class containing member variables and functions shared between
 *        MergeSortBigSize (sort) and KthValueMergeSortMoreCore (kth_value).
 *
 * Derived classes must provide: Init(), InitMergeBuffers(), ExtractAndCopyOut().
 * All access to member variables uses `this->` to satisfy C++ two-phase name lookup
 * for dependent base classes.
 * @tparam Derived CRTP derived type
 * @tparam T Input/output storage data type
 * @tparam CONVERT_TYPE Data type used for sort and merge workspace
 * @tparam IS_DESCEND Sort order: true for descending, false for ascending
 * @tparam INDEX_TYPE Output index data type
 */
template <typename Derived, typename T, typename CONVERT_TYPE, bool IS_DESCEND, typename INDEX_TYPE>
class MergeMoreCoreBase {
public:
    // ===== Queues =====
    TQue<QuePosition::VECIN, MERGE_MORE_BUFFER_NUM> inputQueue_;
    TQue<QuePosition::VECOUT, MERGE_MORE_BUFFER_NUM> outValueQueue_;
    TQue<QuePosition::VECOUT, MERGE_MORE_BUFFER_NUM> outIndexQueue_;
    TQue<QuePosition::VECOUT, MERGE_MORE_BUFFER_NUM> sortedQueue_;
    TQue<QuePosition::VECIN, MERGE_MORE_BUFFER_NUM> copyInQueue_;
    TQue<QuePosition::VECOUT, MERGE_MORE_BUFFER_NUM> castValueQueue_;
    TQue<QuePosition::VECOUT, MERGE_MORE_BUFFER_NUM> castIndexQueue_;

    // ===== Global Tensors =====
    GlobalTensor<T> inputValueGm_;
    GlobalTensor<T> outValueGm_;
    GlobalTensor<INDEX_TYPE> outIndexGm_;
    GlobalTensor<CONVERT_TYPE> workspaceGm_[2]; // ping-pong buffer
    GlobalTensor<CONVERT_TYPE> workspaceInput_;
    GlobalTensor<CONVERT_TYPE> workspaceOutput_;

    // ===== Local Buffers =====
    TBuf<QuePosition::VECCALC> sortedValueUb_;
    TBuf<QuePosition::VECCALC> sortedValueIndexUb_;
    TBuf<QuePosition::VECCALC> sortTempBuf_;

    // ===== Scalar Members =====
    TPipe* pipe_ = nullptr;
    uint32_t blockIdx_ = 0;
    uint32_t numTileData_ = 0;
    uint32_t outputLastDimValue_ = 0;
    uint32_t frontCoreNum_ = 0;
    uint32_t rowGroupIdx_ = 0;
    uint32_t rowIdx_ = 0;
    uint32_t rowCoreIdx_ = 0;
    uint32_t rowsPerRound_ = 0;
    uint32_t sortLoopTimes_ = 1;
    uint32_t logicalBlockSize_ = 0;
    bool syncMergeSortMode_ = false;
    int64_t batchNum_ = 0;
    uint32_t vfLenFp32_ = Ops::Base::GetVRegSize() / FP32_DTYPE_BYTES;
    int64_t rowDataOffset_ = 0;
    int64_t rowWorkspaceOffset_ = 0;

    // ===== Merge-sort state =====
    int64_t listNum_{0};
    int64_t remainListNum_{0};
    int64_t outOffset_{0};
    int64_t offsets_[4] = {0};
    int64_t listRemainElements_[4] = {0};
    int64_t currentElements_{0};
    int64_t currentTailElements_{0};
    int64_t dealLengths_[4] = {0};
    int64_t allRemainElements_{0};
    int64_t curLoopSortedNum_{0};
    int64_t onceMaxElements_{0};
    uint16_t elementCountList_[4] = {0};
    uint16_t validBitTail_;
    uint32_t listSortedNums_[4] = {0};
    uint32_t workSpaceFlag_ = 0;

    // ======================== Shared Member Functions ========================

    __aicore__ inline void Process()
    {
        if (this->frontCoreNum_ == 0U || this->rowsPerRound_ == 0U || this->sortLoopTimes_ == 0U) {
            return;
        }
        for (uint32_t loopIdx = 0; loopIdx < this->sortLoopTimes_; loopIdx++) {
            this->rowIdx_ = loopIdx * this->rowsPerRound_ + this->rowGroupIdx_;
            // In multi-round mode, the last round may have fewer rows than rowsPerRound_.
            // Cores assigned to out-of-range rows skip real work but still join SyncAll to avoid deadlock.
            bool rowActive = static_cast<int64_t>(this->rowIdx_) < this->batchNum_;
            if (rowActive) {
                static_cast<Derived*>(this)->PrepareRowWorkspace();
                static_cast<Derived*>(this)->InitSortBuffers();
            }

            int64_t offsetPerCore = 0;
            uint32_t tileNum = 0;
            if (rowActive) {
                uint32_t phase1BlockSize = this->syncMergeSortMode_ ? this->logicalBlockSize_ : this->numTileData_;
                offsetPerCore = static_cast<int64_t>(phase1BlockSize) * static_cast<int64_t>(this->rowCoreIdx_);
                if (this->rowCoreIdx_ < this->frontCoreNum_ - 1) {
                    tileNum = phase1BlockSize;
                } else if (this->rowCoreIdx_ == this->frontCoreNum_ - 1) {
                    tileNum = this->outputLastDimValue_ - phase1BlockSize * (this->frontCoreNum_ - 1);
                }
            }
            if (rowActive && tileNum > 0U) {
                if (this->syncMergeSortMode_) {
                    this->SortLogicalBlock(tileNum, offsetPerCore);
                } else {
                    this->SortInSingleCore(tileNum, offsetPerCore);
                }
            }
            SyncAll();
            this->pipe_->Reset();

            if (rowActive) {
                static_cast<Derived*>(this)->InitMergeBuffers();
                if (this->syncMergeSortMode_ && tileNum > 0U) {
                    this->workspaceInput_ = this->workspaceGm_[0];
                    this->workspaceOutput_ = this->workspaceGm_[1];
                    this->MergeLogicalBlock(tileNum, offsetPerCore);
                    this->ClearCache();
                }
            }
            if (this->syncMergeSortMode_) {
                SyncAll();
            }

            InitMergeState(this->syncMergeSortMode_ ? this->logicalBlockSize_ : this->numTileData_,
                           this->syncMergeSortMode_ ? 1U : 0U);
            MergeAcrossCores(rowActive);
            this->pipe_->Reset();
        }
    }

    // The one-round, non-sync schedule has one active row per core group. Keeping it in a dedicated
    // kernel instance removes the generic row-activity and sync-mode branches from this hot orchestration path.
    __aicore__ inline void ProcessDirect()
    {
        if (this->frontCoreNum_ == 0U) {
            return;
        }
        int64_t offsetPerCore = static_cast<int64_t>(this->numTileData_) * this->rowCoreIdx_;
        uint32_t tileNum = this->rowCoreIdx_ < this->frontCoreNum_ - 1 ?
                               this->numTileData_ :
                               this->outputLastDimValue_ - this->numTileData_ * (this->frontCoreNum_ - 1);
        SortInSingleCore(tileNum, offsetPerCore);
        SyncAll();
        this->pipe_->Reset();

        static_cast<Derived*>(this)->InitMergeBuffers();
        InitMergeState(this->numTileData_, 0U);
        MergeAcrossCores(true);
    }

    __aicore__ inline void InitMergeState(int64_t blockElements, uint32_t workspaceFlag)
    {
        this->workSpaceFlag_ = workspaceFlag;
        this->listNum_ = this->frontCoreNum_;
        this->currentElements_ = blockElements;
        this->currentTailElements_ = this->outputLastDimValue_ - blockElements * (this->frontCoreNum_ - 1);
    }

    // Inactive row groups advance the same merge levels and participate in the
    // barriers inside SortInMultiCore/FinalSortAndCopy, without accessing row workspace.
    __aicore__ inline void MergeAcrossCores(bool rowActive)
    {
        while (this->listNum_ > MERGE_LIST_MAX_NUM) {
            if (rowActive) {
                this->workspaceInput_ = this->workspaceGm_[this->workSpaceFlag_];
                this->workspaceOutput_ = this->workspaceGm_[1U - this->workSpaceFlag_];
            }
            SortInMultiCore(rowActive);
            uint32_t currentCoreNum = Ops::Base::CeilDiv(this->listNum_, static_cast<int64_t>(MERGE_LIST_MAX_NUM));
            uint32_t remainListNum = this->listNum_ - (currentCoreNum - 1U) * MERGE_LIST_MAX_NUM;
            this->currentTailElements_ = this->currentElements_ * (remainListNum - 1U) + this->currentTailElements_;
            this->listNum_ = currentCoreNum;
            this->currentElements_ *= MERGE_LIST_MAX_NUM;
            this->workSpaceFlag_ = 1U - this->workSpaceFlag_;
        }
        if (rowActive) {
            this->workspaceInput_ = this->workspaceGm_[this->workSpaceFlag_];
            this->workspaceOutput_ = this->workspaceGm_[1U - this->workSpaceFlag_];
        }
        FinalSortAndCopy(rowActive);
    }

    // Sync-merge mode: sort a logical block (larger than numTileData_) in multiple passes,
    // each pass handling one numTileData_-sized chunk and flushing to workspaceGm_[0].
    __aicore__ inline void SortLogicalBlock(uint32_t tileNum, int64_t offsetPerCore)
    {
        uint32_t sortedNum = 0;
        while (sortedNum < tileNum) {
            uint32_t curTileNum = this->numTileData_;
            if (curTileNum > tileNum - sortedNum) {
                curTileNum = tileNum - sortedNum;
            }
            this->SortInSingleCore(curTileNum, offsetPerCore + static_cast<int64_t>(sortedNum));
            event_t eventIdMTE3ToS = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(eventIdMTE3ToS);
            WaitFlag<HardEvent::MTE3_S>(eventIdMTE3ToS);
            sortedNum += curTileNum;
        }
    }

    // Sync-merge mode: merge the locally sorted chunks (produced by SortLogicalBlock) into one
    // contiguous sorted logical block, using the streaming merge loop (CopyIn → MrgSort → CopyOut).
    __aicore__ inline void MergeLogicalBlock(uint32_t tileNum, int64_t offsetPerCore)
    {
        this->listNum_ = Ops::Base::CeilDiv(tileNum, this->numTileData_);
        this->currentElements_ = this->numTileData_;
        this->currentTailElements_ = tileNum - this->numTileData_ * (this->listNum_ - 1);
        this->InitLogicalBlockMerge(tileNum, offsetPerCore);
        for (; this->allRemainElements_ > 0;) {
            CopyInMultiCore();
            UpdateMrgParam();
            DealingMergeSort();
            UpdateSortInfo();
            CopyOutMultiCore();
        }
    }

    // Set up per-run offsets and remaining element counts for the logical block merge.
    // Each run corresponds to one numTileData_-sized chunk sorted in SortLogicalBlock.
    __aicore__ inline void InitLogicalBlockMerge(uint32_t tileNum, int64_t offsetPerCore)
    {
        this->outOffset_ = GetSortLen<CONVERT_TYPE>(offsetPerCore);
        for (int64_t i = 0; i < MERGE_LIST_MAX_NUM; i++) {
            if (i < this->listNum_ - 1) {
                this->listRemainElements_[i] = this->numTileData_;
                this->offsets_[i] = GetSortOffset<CONVERT_TYPE>(offsetPerCore + i * this->numTileData_);
                this->allRemainElements_ += this->listRemainElements_[i];
            } else if (i == this->listNum_ - 1) {
                this->listRemainElements_[i] = this->currentTailElements_;
                this->offsets_[i] = GetSortOffset<CONVERT_TYPE>(offsetPerCore + i * this->numTileData_);
                this->allRemainElements_ += this->currentTailElements_;
            } else {
                this->listRemainElements_[i] = 0;
            }
        }
        if (this->listNum_ == 1) {
            this->currentTailElements_ = tileNum;
        }
    }

    __aicore__ inline void SortInSingleCore(uint32_t tileNum, int64_t offsetPerCore)
    {
        // Phase 1: each front core sorts one contiguous chunk of a row and writes Sort-API packed records into
        // workspaceGm_[0]. Later phases merge these sorted runs.
        CopyInData(tileNum, offsetPerCore);
        LocalTensor<T> inputLocal = this->inputQueue_.template DeQue<T>();
        LocalTensor<CONVERT_TYPE> sortedValueLocal = this->sortedValueUb_.template Get<CONVERT_TYPE>();
        LocalTensor<uint32_t> sortedValueIndexLocal = this->sortedValueIndexUb_.template Get<uint32_t>();
        static_cast<Derived*>(this)->OnInputLoaded(inputLocal, tileNum);
        InitIndexLocal(tileNum, sortedValueIndexLocal, offsetPerCore);
        DoSort(tileNum, inputLocal, sortedValueLocal, sortedValueIndexLocal);
        CopyOutWorkSpace(tileNum, offsetPerCore, sortedValueLocal);
        this->inputQueue_.FreeTensor(inputLocal);
    }

    // Copy tileNum elements from GM to UB, padding the tail to alignment with defaultValue
    // (−INF for descending, NAN for ascending) so Sort API processes a full aligned block.
    __aicore__ inline void CopyInData(uint32_t tileNum, int64_t offsetPerCore)
    {
        LocalTensor<T> inputLocal = this->inputQueue_.template AllocTensor<T>();
        T defaultValue = IS_DESCEND ? static_cast<T>(-INFINITY) : static_cast<T>(NAN);
        uint32_t alignTile = ROUND_UP_AGLIN(tileNum);
        Duplicate(inputLocal, defaultValue, alignTile);

        event_t eventId = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);

        uint32_t currTileSizeAlign = ROUND_UP_AGLIN(tileNum * sizeof(T)) / sizeof(T);
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = tileNum * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<T> padParams;
        padParams.isPad = true;
        padParams.rightPadding = currTileSizeAlign - tileNum;
        padParams.paddingValue = static_cast<T>(defaultValue);

        AscendC::DataCopyPad(inputLocal, this->inputValueGm_[this->rowDataOffset_ + offsetPerCore], copyParams,
                             padParams);
        this->inputQueue_.EnQue(inputLocal);
    }

    __aicore__ inline void InitIndexLocal(uint32_t tileNum, LocalTensor<uint32_t> sortedValueIndexLocal,
                                          int64_t offsetPerCore)
    {
        PipeBarrier<PIPE_ALL>();
        LocalTensor<int32_t> tempIndexLocal = sortedValueIndexLocal.ReinterpretCast<int32_t>();
        ArithProgression<int32_t>(tempIndexLocal, offsetPerCore, 1, tileNum);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void DoSort(uint32_t tileNum, LocalTensor<T> inputLocal,
                                  LocalTensor<CONVERT_TYPE> sortedValueLocal,
                                  LocalTensor<uint32_t> sortedValueIndexLocal)
    {
        AscendC::LocalTensor<CONVERT_TYPE> sortTempLocal = this->sortTempBuf_.template Get<CONVERT_TYPE>();
        uint32_t aglinTileNum = ROUND_UP_AGLIN(tileNum);
        uint32_t sortRepeatTimes = Ops::Base::CeilDiv(aglinTileNum, DEALING_SORT_NUM_ONCE);

        static_cast<Derived*>(this)->PrepareInputForSort(inputLocal, aglinTileNum);
        if constexpr (!IS_DESCEND) {
            FlipSignBit(inputLocal, aglinTileNum);
        }
        bool resultInSorted = PingPongMergeSortCommon::SortToProposal(sortedValueLocal, sortTempLocal, inputLocal,
                                                                      sortedValueIndexLocal, sortRepeatTimes);
        if (!resultInSorted) {
            DataCopy(sortedValueLocal, sortTempLocal, GetSortOffset<CONVERT_TYPE>(aglinTileNum));
        }
    }

    __aicore__ inline void FlipSignBit(LocalTensor<CONVERT_TYPE> xLocal, uint32_t aglinTileNum)
    {
        if constexpr (std::is_same<float, CONVERT_TYPE>::value) {
            AscendC::LocalTensor<int32_t> castTensor = xLocal.template ReinterpretCast<int32_t>();
            AscendC::Adds(castTensor, castTensor, XOR_OP_VALUE_FP, aglinTileNum);
        } else if constexpr (std::is_same<half, CONVERT_TYPE>::value) {
            AscendC::LocalTensor<int16_t> castTensor = xLocal.template ReinterpretCast<int16_t>();
            AscendC::Adds(castTensor, castTensor, XOR_OP_VALUE_HALF, aglinTileNum);
        }
    }

    __aicore__ inline void CopyOutWorkSpace(uint32_t tileNum, int64_t offsetPerCore,
                                            LocalTensor<CONVERT_TYPE> sortedValueLocal)
    {
        event_t eventIdVToMTE3 = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = GetSortLen<CONVERT_TYPE>(tileNum) * sizeof(CONVERT_TYPE);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(this->workspaceGm_[0][GetSortLen<CONVERT_TYPE>(offsetPerCore)], sortedValueLocal, copyParams);
    }

    __aicore__ inline void CopyOutMultiCore()
    {
        LocalTensor<CONVERT_TYPE> sortTempBuffer = this->sortedQueue_.template DeQue<CONVERT_TYPE>();

        uint32_t len = this->curLoopSortedNum_;
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = GetSortLen<CONVERT_TYPE>(len) * sizeof(CONVERT_TYPE);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(this->workspaceOutput_[this->outOffset_], sortTempBuffer, copyParams);
        this->outOffset_ += GetSortLen<CONVERT_TYPE>(len);
        this->sortedQueue_.template FreeTensor<CONVERT_TYPE>(sortTempBuffer);
    }

    __aicore__ inline void SortInMultiCore(bool rowActive)
    {
        // Phase 2: every active row core merges up to 4 sorted runs. The merged output is written to the alternate
        // workspace buffer, and Process/derived code swaps the input/output workspace for the next merge level.
        uint32_t needCoreNum = Ops::Base::CeilDiv(this->listNum_, static_cast<int64_t>(MERGE_LIST_MAX_NUM));
        if (rowActive && this->rowCoreIdx_ < needCoreNum) {
            MultiCoreInit();
            for (; this->allRemainElements_ > 0;) {
                CopyInMultiCore();
                UpdateMrgParam();
                DealingMergeSort();
                UpdateSortInfo();
                CopyOutMultiCore();
            }
            ClearCache();
        }
        SyncAll();
    }

    __aicore__ inline void FinalSortAndCopy(bool rowActive)
    {
        // Phase 3: once at most 4 runs remain for the row, rowCoreIdx_ 0 performs the final merge and lets the
        // derived kernel extract values/indices directly to the operator outputs.
        if (rowActive && this->rowCoreIdx_ == 0) {
            FinalSortInit();
            for (; this->allRemainElements_ > 0;) {
                CopyInMultiCore();
                UpdateMrgParam();
                DealingMergeSort();
                UpdateSortInfo();
                static_cast<Derived*>(this)->ExtractAndCopyOut();
            }
            ClearCache();
        }
        SyncAll();
    }

    __aicore__ inline void MultiCoreInit()
    {
        // Each row core owns one 4-run merge group. offsets_ point into workspaceInput_, while
        // listRemainElements_ tracks how many packed records are still unread for each run.
        this->outOffset_ = GetSortLen<CONVERT_TYPE>(this->rowCoreIdx_ * MERGE_LIST_MAX_NUM * this->currentElements_);
        for (int64_t i = 0; i < MERGE_LIST_MAX_NUM; i++) {
            uint32_t blockNum = this->rowCoreIdx_ * MERGE_LIST_MAX_NUM + i;
            if (blockNum < this->listNum_ - 1) {
                this->listRemainElements_[i] = this->currentElements_;
                this->offsets_[i] = GetSortOffset<CONVERT_TYPE>(blockNum * this->currentElements_);
                this->allRemainElements_ += this->listRemainElements_[i];
            } else if (blockNum == this->listNum_ - 1) {
                this->listRemainElements_[i] = this->currentTailElements_;
                this->offsets_[i] = GetSortOffset<CONVERT_TYPE>(blockNum * this->currentElements_);
                this->allRemainElements_ += this->currentTailElements_;
            } else {
                this->listRemainElements_[i] = 0;
            }
        }
    }

    __aicore__ inline void FinalSortInit()
    {
        // Final merge is always performed by rowCoreIdx_ 0, so it starts from the first remaining run.
        this->outOffset_ = 0;
        for (int64_t i = 0; i < MERGE_LIST_MAX_NUM; i++) {
            if (i < this->listNum_ - 1) {
                this->listRemainElements_[i] = this->currentElements_;
                this->offsets_[i] = GetSortOffset<CONVERT_TYPE>(i * this->currentElements_);
                this->allRemainElements_ += this->listRemainElements_[i];
            } else if (i == this->listNum_ - 1) {
                this->listRemainElements_[i] = this->currentTailElements_;
                this->offsets_[i] = GetSortOffset<CONVERT_TYPE>(i * this->currentElements_);
                this->allRemainElements_ += this->currentTailElements_;
            } else {
                this->listRemainElements_[i] = 0;
            }
        }
    }

    __aicore__ inline void CopyInMultiCore()
    {
        LocalTensor<CONVERT_TYPE> ubMainInput = this->copyInQueue_.template AllocTensor<CONVERT_TYPE>();
        this->remainListNum_ = 0;
        for (int64_t i = 0, j = 0; i < MERGE_LIST_MAX_NUM; i++) {
            // Copy at most onceMaxElements_ from each non-empty run so one MrgSort call can merge the current window.
            this->dealLengths_[i] = (this->onceMaxElements_ > this->listRemainElements_[i] ?
                                         this->listRemainElements_[i] :
                                         this->onceMaxElements_);
            if (this->dealLengths_[i] > 0) {
                DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = GetSortLen<CONVERT_TYPE>(this->dealLengths_[i]) * sizeof(CONVERT_TYPE);
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                DataCopyPadExtParams<CONVERT_TYPE> padParams{false, 0, 0, 0};
                DataCopyPad(ubMainInput[GetSortLen<CONVERT_TYPE>(this->onceMaxElements_) * i],
                            this->workspaceInput_[this->offsets_[i]], copyParams, padParams);
                this->elementCountList_[j] = this->dealLengths_[i];
                this->remainListNum_ += 1;
                j++;
            }
        }
        this->copyInQueue_.EnQue(ubMainInput);
    }

    __aicore__ inline void UpdateMrgParam()
    {
        // MrgSort accepts four source lists. validBitTail marks which of the four are real for this window.
        if (this->remainListNum_ == TWO_WAY_MERGE_LIST_NUM) {
            this->elementCountList_[TWO_WAY_MERGE_LIST_NUM] = 0;
            this->elementCountList_[THREE_WAY_MERGE_LIST_NUM] = 0;
            this->validBitTail_ = 0b0011;
        } else if (this->remainListNum_ == THREE_WAY_MERGE_LIST_NUM) {
            this->elementCountList_[THREE_WAY_MERGE_LIST_NUM] = 0;
            this->validBitTail_ = 0b0111;
        } else if (this->remainListNum_ == MERGE_LIST_MAX_NUM) {
            this->validBitTail_ = 0b1111;
        } else {
            this->elementCountList_[1] = 0;
            this->elementCountList_[TWO_WAY_MERGE_LIST_NUM] = 0;
            this->elementCountList_[THREE_WAY_MERGE_LIST_NUM] = 0;
            this->validBitTail_ = 0b0001;
        }
    }

    // Perform MrgSort on 2/3/4 source runs. Unused source-list slots are filled with a duplicate
    // of tmpUbInputs[0] because MrgSort requires valid tensor addresses for all four parameters;
    // validBitTail_ controls which lists actually participate.
    __aicore__ inline void DealingMergeSort()
    {
        LocalTensor<CONVERT_TYPE> sortTempBuffer = this->sortedQueue_.template AllocTensor<CONVERT_TYPE>();
        LocalTensor<CONVERT_TYPE> ubMainInput = this->copyInQueue_.template DeQue<CONVERT_TYPE>();
        LocalTensor<CONVERT_TYPE> tmpUbInputs[4];
        for (int64_t i = 0, j = 0; i < MERGE_LIST_MAX_NUM; i++) {
            if (this->dealLengths_[i] > 0) {
                tmpUbInputs[j] = ubMainInput[GetSortLen<CONVERT_TYPE>(this->onceMaxElements_) * i];
                j++;
            }
        }
        if (this->remainListNum_ == TWO_WAY_MERGE_LIST_NUM) {
            // Unused source-list arguments are duplicated because validBitTail controls which lists participate.
            MrgSortSrcList sortListTail = MrgSortSrcList(tmpUbInputs[0], tmpUbInputs[1], tmpUbInputs[0],
                                                         tmpUbInputs[0]);
            MrgSort<CONVERT_TYPE, true>(sortTempBuffer, sortListTail, this->elementCountList_, this->listSortedNums_,
                                        this->validBitTail_, 1);
        } else if (this->remainListNum_ == THREE_WAY_MERGE_LIST_NUM) {
            MrgSortSrcList sortListTail = MrgSortSrcList(tmpUbInputs[0], tmpUbInputs[1], tmpUbInputs[2],
                                                         tmpUbInputs[0]);
            MrgSort<CONVERT_TYPE, true>(sortTempBuffer, sortListTail, this->elementCountList_, this->listSortedNums_,
                                        this->validBitTail_, 1);
        } else if (this->remainListNum_ == MERGE_LIST_MAX_NUM) {
            MrgSortSrcList sortListTail = MrgSortSrcList(tmpUbInputs[0], tmpUbInputs[1], tmpUbInputs[2],
                                                         tmpUbInputs[3]);
            MrgSort<CONVERT_TYPE, true>(sortTempBuffer, sortListTail, this->elementCountList_, this->listSortedNums_,
                                        this->validBitTail_, 1);
        } else {
            AscendC::Copy(sortTempBuffer, tmpUbInputs[0],
                          ROUND_UP_AGLIN(GetSortLen<CONVERT_TYPE>(this->elementCountList_[0]) * sizeof(CONVERT_TYPE)) /
                              sizeof(CONVERT_TYPE));
            this->listSortedNums_[0] = this->elementCountList_[0];
        }
        this->sortedQueue_.EnQue(sortTempBuffer);
        this->copyInQueue_.FreeTensor(ubMainInput);
    }

    __aicore__ inline void UpdateSortInfo()
    {
        this->curLoopSortedNum_ = 0;
        for (int64_t i = 0, j = 0; i < MERGE_LIST_MAX_NUM; i++) {
            if (this->dealLengths_[i] > 0) {
                // Consume the number of records that MrgSort actually produced from this source run.
                this->listRemainElements_[i] -= this->listSortedNums_[j];
                this->allRemainElements_ -= this->listSortedNums_[j];
                // Advance by packed Sort-API record length, not raw element count.
                this->offsets_[i] += GetSortOffset<CONVERT_TYPE>(this->listSortedNums_[j]);
                // curLoopSortedNum_ is the copy-out length for the merged window.
                this->curLoopSortedNum_ += this->listSortedNums_[j];
                j++;
            }
        }
    }

    __aicore__ inline void ClearCache()
    {
        this->allRemainElements_ = 0;
        this->outOffset_ = 0;
        this->remainListNum_ = 0;
        for (int64_t i = 0; i < MERGE_LIST_MAX_NUM; i++) {
            this->offsets_[i] = 0;
            this->listRemainElements_[i] = 0;
            this->elementCountList_[i] = 0;
        }
    }
};

} // namespace MergeMoreCoreCommon

#endif // MERGE_MORE_CORE_BASE_H
