/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file index_put_with_sort_gather_data.h
 * \brief
 */

#ifndef INDEX_PUT_WITH_SORT_GATHER_DATA
#define INDEX_PUT_WITH_SORT_GATHER_DATA

#include "index_put_with_sort_scatter_data.h"

namespace IndexPutWithSort {
using namespace AscendC;

template<typename IT, typename CT>
class GatherDataOp : public ScatterDataInKernelOp<IT, CT> {
public:
    __aicore__ inline GatherDataOp() {}
    __aicore__ inline void Init(GM_ADDR out, GM_ADDR linear_index, GM_ADDR pos_idx, 
                                GM_ADDR values, GM_ADDR userWorkspace,
                                const IndexPutWithSortTilingData* tilingData, TPipe* tpipe) {
        this->InitTilingData(tilingData, tpipe);
        this->InitLocalTensor(INDEX_UB_NUMS, VALUES_UB_NUMS, VALUES_UB_NUMS);
        this->InitGmTensor(out, linear_index, pos_idx, values, userWorkspace);
    }

    __aicore__ inline void Process() {
        this->GetHeadTailIndexValue();
        // 索引切块
        uint64_t coreIndicesNums = this->coreEndIndex - this->coreStartIndex;
        uint64_t indicesBlocks = coreIndicesNums / INDEX_UB_NUMS;
        uint64_t indicesLeft = coreIndicesNums - indicesBlocks * INDEX_UB_NUMS;
        // 按块处理
        for (uint64_t i = 0; i <= indicesBlocks; i++) {
            if (i == indicesBlocks && indicesLeft == 0) {
                break;
            }
            uint64_t indicesNums = (i == indicesBlocks ? indicesLeft : INDEX_UB_NUMS);
            uint64_t indicesStart = this->coreStartIndex + i * INDEX_UB_NUMS;
            ProcessIndicesBlock(indicesStart, indicesNums);
        }
        this->CopyOutSyncData();
        SyncAll();
        this->PIPE_CORE_S();
        this->indexBlockLastValue = -1;
        this->ProcessCacheData(this->sliceSize, 0);
    }

    /**
        @brief 处理一整块索引对应的计算任务，一整块索引中索引个数是INDEX_UB_NUMS
        @param indicesStart 该块索引在indexGm上的起始位置，单位为元素个数
        @param indicesNums 该块索引的索引个数，整块时是INDEX_UB_NUMS，尾块时个数不定。
        因为ub容纳的vlaues和out有限，索引处理一整块索引时需要继续划分为batch，每个batch里的索引对应的values和out可以完全搬入ub
        处理每个batch任务时使用double buffer。
    */
    __aicore__ inline void ProcessIndicesBlock(uint64_t indicesStart, uint64_t indicesNums) {
        // 搬入index, pos
        this->CopyInIndices(indicesStart, indicesNums);
        this->CopyInPosIdx(indicesStart, indicesNums);
        // 由尾轴大小，计算出一次性能搬入多少行数据，对out的搬运要求同索引只搬一次，pos没有同索引，所以values全搬
        uint64_t batchSize = VALUES_UB_NUMS / this->sliceSizeAligned;
        uint64_t batches = indicesNums / batchSize;
        uint64_t batchLeft = indicesNums - batches * batchSize;
        for (uint64_t i = 0; i <= batches; i++) {
            if (i == batches && batchLeft == 0) {
                break;
            }
            uint64_t nums = (i == batches ? batchLeft : batchSize);
            uint64_t indexStartOfBatch = i * batchSize;
            CopyInDataOfBatch(indexStartOfBatch, nums);
            PIPE_MTE2_S();
            ComputeOfBatch(indexStartOfBatch, nums);
            PIPE_MTE3_S();
        }
    }

    /**
        @brief 当Ub上的某批索引对应的values和out都被搬入之后，完成这批索引的计算
        先识别首值索引段，尾值索引段，及中间的索引段。然后调用相应的函数分别处理三段。
        @param indexStart 该批次索引在ub上的起始下标
        @param indicesNums 该批次索引个数
    */
    __aicore__ inline void ComputeOfBatch(uint64_t indexStart, uint64_t indicesNums) {
        if (this->castEnable) {
            BatchCastIT2CT(indexStart, indicesNums);
        }
        uint64_t midSegmentStart = indexStart;
        uint64_t midSegmentEnd = indexStart + indicesNums;
        ComputeSegmentEdges(midSegmentStart, midSegmentEnd);
        uint64_t frontIndices = 0;
        if (midSegmentStart > indexStart) {
            // 有首值索引
            ComputeHeadOfBatch(indexStart, midSegmentStart, frontIndices);
            frontIndices += midSegmentStart - indexStart;
        }

        if (midSegmentEnd > midSegmentStart) {
            // 有非首值索引且非尾值索引的中值段
            uint64_t batchIndexEnd = indexStart + indicesNums;
            ComputeMidOfBatch(midSegmentStart, midSegmentEnd, frontIndices, batchIndexEnd);
            CopyOutMidOfBatch(midSegmentStart, midSegmentEnd, frontIndices);
            frontIndices += midSegmentEnd - midSegmentStart;
        }

        if (midSegmentEnd < indexStart + indicesNums && midSegmentEnd >= midSegmentStart) {
            // 有尾值索引
            ComputeTailOfBatch(midSegmentEnd, indexStart + indicesNums, frontIndices);
        }
    }

    /**
        @brief 将一批索引对应的values和out做类型转化
        @param indicesNums 该批次索引个数
    */
    __aicore__ inline void BatchCastIT2CT(uint64_t batchIndexStart, uint64_t indicesNums) {
        // fp16/bf16->fp32 eg [000111].fp16 => [111].fp32
        uint64_t bufferStartIT = VALUES_UB_NUMS;
        LocalTensor<IT> valuesLocalIT = this->valuesLocal[bufferStartIT];
        LocalTensor<IT> outLocalIT = this->outLocal[bufferStartIT];
        uint64_t bufferStartCT = 0;
        LocalTensor<CT> valuesLocalCT = this->valuesLocal[bufferStartCT].template ReinterpretCast<CT>();
        LocalTensor<CT> outLocalCT = this->outLocal[bufferStartCT].template ReinterpretCast<CT>();
        Cast(valuesLocalCT, valuesLocalIT, RoundMode::CAST_NONE, indicesNums * this->sliceSizeAligned);
        int32_t firstIndexValue = this->indexLocal.GetValue(batchIndexStart);
        if (firstIndexValue == this->indexBlockLastValue) {
            if (indicesNums > 1) {
                uint64_t aligned = this->sliceSizeAligned;
                Cast(outLocalCT[aligned], outLocalIT[aligned], RoundMode::CAST_NONE, (indicesNums - 1) * aligned);
            }
        } else {
            Cast(outLocalCT, outLocalIT, RoundMode::CAST_NONE, indicesNums * this->sliceSizeAligned);
        }
        PIPE_V_S();
    }

    /**
        @brief 处理首值索引段
        @param segmentStart 段起始位置
        @param segmentEnd 段截止位置
    */
    __aicore__ inline void ComputeHeadOfBatch(uint64_t segmentStart, uint64_t segmentEnd, uint64_t frontIndices) {
        GroupAccumulation(segmentStart, segmentEnd, frontIndices, false);
        PIPE_V_S();
        uint64_t src = frontIndices * this->sliceSizeAligned;
        uint64_t dst = GetBlockIdx() * BUFFER_NUM * this->sliceSize;
        auto srcTensor = this->valuesLocal.template ReinterpretCast<CT>();
        this->accumulate == 1 ? this->AddToCache(src, dst, srcTensor, this->sliceSize) :
                                this->CopyToCache(src, dst, srcTensor, this->sliceSize);
    }

    /**
        @brief 处理尾值索引段
        @param segmentStart 段起始位置
        @param segmentEnd 段截止位置
    */
    __aicore__ inline void ComputeTailOfBatch(uint64_t segmentStart, uint64_t segmentEnd, uint64_t frontIndices) {
        GroupAccumulation(segmentStart, segmentEnd, frontIndices, false);
        PIPE_V_S();
        uint64_t src = frontIndices * this->sliceSizeAligned;
        uint64_t dst = GetBlockIdx() * BUFFER_NUM * this->sliceSize + this->sliceSize;
        auto srcTensor = this->valuesLocal.template ReinterpretCast<CT>();
        this->accumulate == 1 ? this->AddToCache(src, dst, srcTensor, this->sliceSize) :
                                this->CopyToCache(src, dst, srcTensor, this->sliceSize);
    }

    /**
        @brief 对非首值索引且非尾值索引的中值段索引完成计算
        主要是将同值索引累加到第一次出现的位置，对于多个同值索引使用分组累加提升精度
        @param segmentStart 中值段索引在ub上的起始下标
        @param segmentEnd 中值段索引在ub上的截止位置
    */
    __aicore__ inline void ComputeMidOfBatch(uint64_t segmentStart, uint64_t segmentEnd, uint64_t frontIndices,
                                             uint64_t batchIndexEnd) {
        uint64_t segmentNums = segmentEnd - segmentStart;
        if (segmentNums == 1) { //必须注重分支覆盖率
            GroupAccumulation(segmentStart, segmentEnd, frontIndices, this->castEnable, batchIndexEnd);
        } else {
            uint64_t pre = 0;
            int32_t preIndexValue = this->indexLocal.GetValue(segmentStart + pre);
            for (uint64_t i = 1; i < segmentNums; i++) {
                int32_t indexValue = this->indexLocal.GetValue(segmentStart + i);
                if (indexValue != preIndexValue) {
                    // 遇到不相等的索引值，前面相等的索引值使用分组累加
                    GroupAccumulation(segmentStart + pre, segmentStart + i, frontIndices + pre, this->castEnable, batchIndexEnd);
                    pre = i;
                    preIndexValue = indexValue;
                }
                if (i == segmentNums - 1) {
                    // 最后一个索引值，无论与前面是相等还是不等，都是同索引范围都是[pre, i + 1)
                    GroupAccumulation(segmentStart + pre, segmentStart + i + 1, frontIndices + pre, this->castEnable, batchIndexEnd);
                }
            }
        }
        PIPE_V_S();
    }

    __aicore__ inline void CopyOutMidOfBatch(uint64_t segmentStart, uint64_t segmentEnd, uint64_t frontIndices) {
        uint64_t segmentNums = segmentEnd - segmentStart;
        if (segmentNums == 1) { //必须注重分支覆盖率
            int32_t indexValue = this->indexLocal.GetValue(segmentStart);
            OneIndexCopyToOut(frontIndices, 0, indexValue);
        } else {
            uint64_t pre = 0;
            int32_t preIndexValue = this->indexLocal.GetValue(segmentStart + pre);
            for (uint64_t i = 1; i < segmentNums; i++) {
                int32_t indexValue = this->indexLocal.GetValue(segmentStart + i);
                if (indexValue != preIndexValue) {
                    OneIndexCopyToOut(frontIndices, pre, preIndexValue);
                    pre = i;
                    preIndexValue = indexValue;
                }
                if (i == segmentNums - 1) {
                    OneIndexCopyToOut(frontIndices, pre, preIndexValue);
                }
            }
        }
        PIPE_MTE3_S();
    }

    __aicore__ inline void OneIndexCopyToOut(uint64_t frontIndices, uint64_t indexOffset, int32_t indexValue) {
        uint64_t src = (frontIndices + indexOffset) * this->sliceSizeAligned;
        src += src * this->castEnable;
        uint64_t dst = indexValue * this->sliceSize;
        this->CopyToOut(src, dst, this->valuesLocal, this->sliceSize);
    }

    /**
        @brief 同索引分组累加
        @param indexStart 同索引段在ub上的起始下标
        @param indexEnd 同索引段在ub上的截止下标
        @param dataStart 记录indicesStart对应的“valuesLocal”上的位置，因为同索引相加时起始索引对应的values起始并不是0
    */
    __aicore__ inline void GroupAccumulation(uint64_t indexStart, uint64_t indexEnd, uint64_t dataStart, bool needCast=false,
                                             uint64_t batchIndexEnd=0) {
        if (this->accumulate == 0) {
            // indexEnd - 1处的values转移至indexStart处，此时已经是升精度的。
            // 如果是头尾进来，不用再执行cast。如果是mid进来需要cast到IT
            uint64_t src = (dataStart + (indexEnd - indexStart - 1)) * this->sliceSizeAligned;
            uint64_t dst = dataStart * this->sliceSizeAligned;
            if (needCast) {
                auto srcTensor = this->valuesLocal.template ReinterpretCast<CT>()[src];
                auto dstTensor = this->valuesLocal.template ReinterpretCast<CT>()[dst].template ReinterpretCast<IT>();
                Cast(dstTensor, srcTensor, RoundMode::CAST_RINT, this->sliceSizeAligned);
            } else {
                auto srcTensor = this->valuesLocal.template ReinterpretCast<CT>()[src];
                auto dstTensor = this->valuesLocal.template ReinterpretCast<CT>()[dst];
                DataCopy(dstTensor, srcTensor, this->sliceSizeAligned);
            }
            PIPE_V_S();
            return;
        }
        // 同索引分组累加
        uint64_t indicesNums = indexEnd - indexStart;
        uint64_t indicesGroups = 0;
        uint64_t indicesLeft = 0;
        TaskDivision<uint64_t>(indicesNums, GROUP_NUM, indicesGroups, indicesLeft);
        auto valuesLocalCT = this->valuesLocal.template ReinterpretCast<CT>();
        auto outLocalCT = this->outLocal.template ReinterpretCast<CT>();
        // 组内累加
        for (uint64_t i = 0; i <= indicesGroups; i++) {
            if (i == indicesGroups && indicesLeft == 0) {
                break;
            }
            uint64_t indicesNumsOfGroup = (i == indicesGroups ? indicesLeft : GROUP_NUM);
            uint64_t dataStartOfGroup = dataStart + i * GROUP_NUM;
            for (uint64_t j = 1; j < indicesNumsOfGroup; j++) {
                uint64_t dst = dataStartOfGroup * this->sliceSizeAligned;
                uint64_t src = (dataStartOfGroup + j) * this->sliceSizeAligned;
                Add(valuesLocalCT[dst], valuesLocalCT[dst], valuesLocalCT[src], this->sliceSizeAligned);
                PipeBarrier<PIPE_V>();
            }
        }

        // 组间累加
        for (uint64_t i = 1; i <= indicesGroups; i++) {
            if (i == indicesGroups && indicesLeft == 0) {
                break;
            }
            uint64_t dst = dataStart * this->sliceSizeAligned;
            uint64_t src = (dataStart + i * GROUP_NUM) * this->sliceSizeAligned;
            Add(valuesLocalCT[dst], valuesLocalCT[dst], valuesLocalCT[src], this->sliceSizeAligned);
            PipeBarrier<PIPE_V>();
        }

        // 非头尾索引，第一组的累加核与out累加，避免总累加和与out累加导致吃掉out
        int32_t indexValue = this->indexLocal.GetValue(indexStart);
        if (indexValue != this->headIndexValue && indexValue != this->tailIndexValue) {
            uint64_t dst = dataStart * this->sliceSizeAligned;
            Add(valuesLocalCT[dst], valuesLocalCT[dst], outLocalCT[dst], this->sliceSizeAligned);
            PipeBarrier<PIPE_V>();
            if (indexEnd == batchIndexEnd && this->castEnable) {
                this->indexBlockLastValue = indexValue;
                DataCopy(outLocalCT, valuesLocalCT[dst], this->sliceSizeAligned);
                PipeBarrier<PIPE_V>();
            }
        }

        // 类型转换
        if (needCast) {
            uint64_t dst = dataStart * this->sliceSizeAligned;
            auto valuesLocalIT = valuesLocalCT[dst].template ReinterpretCast<IT>();
            Cast(valuesLocalIT, valuesLocalCT[dst], RoundMode::CAST_RINT, this->sliceSizeAligned);
        }
    }

    /**
        @brief 对某batch索引，查找其非首值索引非尾值索引段的起始范围
        @param segmentStart 传入的时候是batch索引在ub上的起始下标，计算完成后会刷新为中值段的起始位置
        @param segmentEnd 传入的时候是batch索引在ub上的截止下标，计算完成后会刷新为中值段的截止位置
    */
    __aicore__ inline void ComputeSegmentEdges(uint64_t& segmentStart, uint64_t& segmentEnd) {
        int64_t start = segmentStart;
        int64_t end = segmentEnd;
        segmentStart = end;
        segmentEnd = start;
        if (this->indexLocal.GetValue(end - 1) == this->headIndexValue) {
            // 该段"尾索引值"和该核"首索引值"相同，认为没有中间段，无需搜索
            return;
        }
        for(int64_t i = start; i < end; i++) {
            int32_t indexValue = this->indexLocal.GetValue(i);
            if (indexValue != this->headIndexValue) {
                segmentStart = i;
                break;
            }
        }
        for (int64_t i = end - 1; i >= start; i--) {
            int32_t indexValue = this->indexLocal.GetValue(i);
            if (indexValue != this->tailIndexValue) {
                segmentEnd = i + 1;
                break;
            }
        }
    }

    /**
        @brief 为Ub上一个batch的索引，搬入values和out
        注意：一个batch的索引在ub上的起始下标不是0. 但一个batch的索引对应的values和out在ub上的起始下标是0.
        @param indicesStart 该batch索引在ub上的起始下标
        @param indicesNums 该batch索引的个数
    */
    __aicore__ inline void CopyInDataOfBatch(uint64_t indicesStart, uint64_t indicesNums) {
        if (this->accumulate == 0) {
            this->CopyInSingleOfBatch(indicesStart, indicesNums);
            return;
        }
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(IT) * this->sliceSize), 0, 0, 0};
        DataCopyPadExtParams<IT> padParams{true, 0, 0, 0};
        uint64_t bufferStart = 0;
        if (this->castEnable) {
            bufferStart = VALUES_UB_NUMS;
        }
        LocalTensor<IT> valuesLocalNow = this->valuesLocal[bufferStart];
        LocalTensor<IT> outLocalNow = this->outLocal[bufferStart];
        int32_t preIndexValue = this->indexLocal.GetValue(indicesStart);
        int32_t prePosIdxValue = this->posIdxLocal.GetValue(indicesStart);

        DataCopyPad(valuesLocalNow, this->valuesGm[prePosIdxValue * this->sliceSize], copyParams, padParams);
        DataCopyPad(outLocalNow, this->outGm[preIndexValue * this->sliceSize], copyParams, padParams);
        for (uint64_t i = 1; i < indicesNums; i++) {
            int32_t indexValue = this->indexLocal.GetValue(indicesStart + i);
            int32_t posIdxValue = this->posIdxLocal.GetValue(indicesStart + i);
            DataCopyPad(valuesLocalNow[i * this->sliceSizeAligned], this->valuesGm[posIdxValue * this->sliceSize], copyParams, padParams);
            // 同索引直接调过，不同索引立即搬入。牺牲scalar 提升mte2
            if (indexValue == preIndexValue) {
                continue;
            } else {
                DataCopyPad(outLocalNow[i * this->sliceSizeAligned], this->outGm[indexValue * this->sliceSize], copyParams, padParams);
                preIndexValue = indexValue;
            }
        }
    }

    __aicore__ inline void CopyInSingleOfBatch(uint64_t indicesStart, uint64_t indicesNums) {
        uint64_t bufferStart = this->castEnable * VALUES_UB_NUMS;
        LocalTensor<IT> valuesLocalNow = this->valuesLocal[bufferStart];

        if (indicesNums == 1) { //必须注重分支覆盖率
            int32_t posIdxValue = this->posIdxLocal.GetValue(indicesStart);
            uint64_t src = posIdxValue * this->sliceSize;
            CopyGm2Ub<IT>(valuesLocalNow, this->valuesGm[src], this->sliceSize);
            PIPE_MTE2_S();
        }
        // 遍历索引Tensor，同索引只搬入最后一个对应的values
        uint64_t pre = 0;
        int32_t preIndexValue = this->indexLocal.GetValue(indicesStart + pre);
        for (uint64_t i = 1; i < indicesNums; i++) {
            int32_t indexValue = this->indexLocal.GetValue(indicesStart + i);
            if (indexValue != preIndexValue) {
                // 搬入i - 1
                int32_t posIdxValue = this->posIdxLocal.GetValue(indicesStart + i - 1);
                uint64_t src = posIdxValue * this->sliceSize;
                uint64_t dst = (i - 1) * this->sliceSizeAligned;
                CopyGm2Ub<IT>(valuesLocalNow[dst], this->valuesGm[src], this->sliceSize);
                pre = i;
                preIndexValue = indexValue;
            }
            if (i == indicesNums - 1) {
                // 搬入 i
                int32_t posIdxValue = this->posIdxLocal.GetValue(indicesStart + i);
                uint64_t src = posIdxValue * this->sliceSize;
                uint64_t dst = i * this->sliceSizeAligned;
                CopyGm2Ub<IT>(valuesLocalNow[dst], this->valuesGm[src], this->sliceSize);
            }
        }
        PIPE_MTE2_S();
    }
};

}
#endif // INDEX_PUT_WITH_SORT_SCATTER_DATA