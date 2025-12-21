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
 * \file index_put_with_sort_scatter_data.h
 * \brief
 */

#ifndef INDEX_PUT_WITH_SORT_SCATTER_DATA
#define INDEX_PUT_WITH_SORT_SCATTER_DATA

#include "index_put_with_sort_base.h"

namespace IndexPutWithSort {
using namespace AscendC;

template<typename IT, typename CT>
class ScatterDataBetweenKernelOp {
public:
    __aicore__ inline ScatterDataBetweenKernelOp() {}

    __aicore__ inline void Init(GM_ADDR out, GM_ADDR linear_index, GM_ADDR pos_idx, GM_ADDR values,
                                GM_ADDR userWorkspace, const IndexPutWithSortTilingData* tilingData, TPipe* tpipe) {
        this->InitTilingData(tilingData, tpipe);
        this->CoreSplitData();
        this->InitGmTensor(out, linear_index, pos_idx, values);
        this->InitLocalTensor(INDEX_UB_NUMS, TBUF_BLOCK_SIZE * BUFFER_NUM, TBUF_BLOCK_SIZE);
    }

    /**
        @brief 核对自己分到的部分尾轴进行切块，以保证后续可以整块搬入
    */
    __aicore__ inline void CoreSplitData() {
        uint64_t coreDataLength = this->coreEndId - this->coreStartId;
        if (coreDataLength <= TBUF_BLOCK_SIZE) {
            this->frontBlocks = 1;
            this->frontBlockSize = coreDataLength;
        } else {
            uint64_t halfTbufSize = TBUF_BLOCK_SIZE / BUFFER_NUM;
            this->frontBlocks = coreDataLength / (halfTbufSize);
            uint64_t left = coreDataLength - this->frontBlocks * (halfTbufSize);
            if (left == 0) {
                this->frontBlockSize = halfTbufSize;
            } else {
                this->frontBlocks -= 1;
                this->frontBlockSize = halfTbufSize;
                this->lastBlockSize = this->frontBlockSize + left;
            }
        }
    }

    /**
        @brief 读取tilingData中的参数，并计算出核分到的尾轴数据范围
        @param tilingData tiling数据
        @param tpipe 系统资源
    */
    __aicore__ inline void InitTilingData(const IndexPutWithSortTilingData* tilingData, TPipe* tpipe) {
        this->pipe = tpipe;
        this->sliceSize = tilingData->sliceSize;
        this->allIndicesNums = tilingData->indicesNums;
        this->accumulate = tilingData->accumulate;

        if (GetBlockIdx() < tilingData->frontCoreNums) {
            this->coreStartId = GetBlockIdx() * tilingData->frontCoreDataLength;
            this->coreEndId = this->coreStartId + tilingData->frontCoreDataLength;
        } else {
            this->coreStartId = tilingData->frontCoreNums * tilingData->frontCoreDataLength + 
                  (GetBlockIdx() - tilingData->frontCoreNums) * tilingData->tailCoreDataLength;
            this->coreEndId = this->coreStartId + tilingData->tailCoreDataLength;
        }
    }

    /**
        @brief 将GM上的地址初始化为对应的GmTensor
    */
    __aicore__ inline void InitGmTensor(GM_ADDR out, GM_ADDR linear_index, GM_ADDR pos_idx, 
                                        GM_ADDR values) {
        this->outGm.SetGlobalBuffer(reinterpret_cast<__gm__ IT*>(out));
        this->indexGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(linear_index));
        this->posIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(pos_idx));
        this->valuesGm.SetGlobalBuffer(reinterpret_cast<__gm__ IT*>(values));
    }

    /**
        @brief 分配Ub上空间，valuesUbBuf有两块数据，后一块用于搬入values数据，前一块用于存储分组的累加
    */
    __aicore__ inline void InitLocalTensor(uint64_t indexUbSize, uint64_t valueUbSize, uint64_t outUbSize) {
        this->pipe->InitBuffer(this->indexUbBuf, indexUbSize * sizeof(int32_t));
        this->pipe->InitBuffer(this->posIdxUbBuf, indexUbSize * sizeof(int32_t));
        this->pipe->InitBuffer(this->valuesUbBuf, valueUbSize * sizeof(int32_t));
        this->pipe->InitBuffer(this->outUbBuf, outUbSize * sizeof(int32_t));

        this->indexLocal = this->indexUbBuf.template Get<int32_t>();
        this->posIdxLocal = this->posIdxUbBuf.template Get<int32_t>();
        this->valuesLocal = this->valuesUbBuf.template Get<IT>();
        this->outLocal = this->outUbBuf.template Get<IT>();
    }

    /**
        @brief 处理流程起始函数
    */
    __aicore__ inline void Process() {
        // 循环处理每个尾轴块
        for (uint64_t i = 0; i <= this->frontBlocks; i++) {
            if (i == this->frontBlocks && this->lastBlockSize == 0) {
                break;
            }
            uint64_t blockSize = (i == this->frontBlocks ? this->lastBlockSize : this->frontBlockSize);
            uint64_t blockStart = this->coreStartId + i * this->frontBlockSize;
            this->indexBlockLastValue = -1;
            this->ProcessDataBlockWithIndices(blockStart, blockSize, 0, this->allIndicesNums);
        }
    }

    /**
        @brief 处理指定范围内的尾轴数据，并指定GM上索引起始范围
        @param dataBlockStart "尾轴数据"的起始位置
        @param dataBlockSize 处理的数据长度
        @param indicesStart 在indexGm上的起始下标
        @param indicesNums 索引个数
    */
    // 这个函数要打造成和big_tail分支适配
    __aicore__ inline void ProcessDataBlockWithIndices(uint64_t dataBlockStart, uint64_t dataBlockSize,
                                                       uint64_t indicesStart, uint64_t indicesNums) {
        // 索引切块
        uint64_t indicesBlockSize = INDEX_UB_NUMS;
        uint64_t indicesBlocks = 0;
        uint64_t indicesLeft = 0;
        TaskDivision(indicesNums, indicesBlockSize, indicesBlocks, indicesLeft);
        // 按块处理
        for (uint64_t i = 0; i <= indicesBlocks; i++) {
            if (i == indicesBlocks && indicesLeft == 0) {
                break;
            }
            uint64_t indicesBlockNums = (i == indicesBlocks ? indicesLeft : indicesBlockSize);
            uint64_t indicesBlockStart = indicesStart + i * indicesBlockSize;
            this->ProcessDataBlockWithIndicesBlock(dataBlockStart, dataBlockSize, indicesBlockStart, indicesBlockNums);
        }
    }

    /**
        @brief 处理一块尾轴数据和对应的一块索引，“一块尾轴数据”和“一块索引”可完全搬入UB，不用再切块了
        @param dataBlockStart "尾轴数据"的起始位置
        @param dataBlockSize 处理的数据长度
        @param indicesBlockStart 在indexGm上的起始下标
        @param indicesBlockNums 索引个数
    */
    __aicore__ inline void ProcessDataBlockWithIndicesBlock(uint64_t dataBlockStart, uint64_t dataBlockSize,
                                                            uint64_t indicesBlockStart, uint64_t indicesBlockNums) {
        // 搬入index, pos
        this->CopyInIndices(indicesBlockStart, indicesBlockNums);
        this->CopyInPosIdx(indicesBlockStart, indicesBlockNums);

        if (indicesBlockNums == 1) { //必须注重分支覆盖率
            this->InitOutLocal(0, dataBlockSize, dataBlockStart);
            this->ProcessGivenIndex<IT>(0, 1, dataBlockSize, dataBlockStart, this->valuesGm, this->castEnable);
            this->ThrowOutResult(0, dataBlockSize, dataBlockStart);
            int32_t indexValue = this->indexLocal.GetValue(0);
            if (indexValue != this->headIndexValue && indexValue != this->tailIndexValue && this->castEnable) {
                this->indexBlockLastValue = indexValue;
            }
        }
        // 遍历索引Tensor，同索引处理一次
        uint64_t pre = 0;
        int32_t preIndexValue = this->indexLocal.GetValue(pre);
        for (uint64_t i = 1; i < indicesBlockNums; i++) {
            int32_t indexValue = this->indexLocal.GetValue(i);
            if (indexValue != preIndexValue) {
                this->InitOutLocal(pre, dataBlockSize, dataBlockStart);
                this->ProcessGivenIndex<IT>(pre, i, dataBlockSize, dataBlockStart, this->valuesGm, this->castEnable);
                this->ThrowOutResult(pre, dataBlockSize, dataBlockStart);
                pre = i;
                preIndexValue = indexValue;
            }
            if (i == indicesBlockNums - 1) {
                this->InitOutLocal(pre, dataBlockSize, dataBlockStart);
                this->ProcessGivenIndex<IT>(pre, i + 1, dataBlockSize, dataBlockStart, this->valuesGm, this->castEnable);
                this->ThrowOutResult(pre, dataBlockSize, dataBlockStart);
                int32_t indexValue = this->indexLocal.GetValue(pre);
                if (indexValue != this->headIndexValue && indexValue != this->tailIndexValue && this->castEnable) {
                    this->indexBlockLastValue = indexValue;
                }
            }
        }
    }

    /**
        @brief 处理一段同值索引段
        @param indexStartId 索引段在ub上的起始下标
        @param indexEndId 索引段在ub上的截止下标
    */
    template <typename IC>
    __aicore__ inline void ProcessGivenIndex(uint64_t indexStartId, uint64_t indexEndId, 
                                             uint64_t blockSize, uint64_t blockStart,
                                             GlobalTensor<IC> srcGm, bool needCast=false) {
        if (this->accumulate == 0) {
            this->ProcessLastIndex<IC>(indexEndId - 1, blockSize, blockStart, srcGm, needCast);
            return;
        }
        // 同索引切块
        uint64_t indicesNums = indexEndId - indexStartId;
        uint64_t batches = 0;
        uint64_t batchLeft = 0;
        TaskDivision<uint64_t>(indicesNums, GROUP_NUM, batches, batchLeft);

        // 循环处理索引块
        for (uint64_t i = 0; i <= batches; i++) {
            if (i == batches && batchLeft == 0) {
                break;
            }
            uint64_t batchNums = (i == batches ? batchLeft : GROUP_NUM);
            uint64_t batchStart = indexStartId + i * GROUP_NUM;
            // 置零valuesLocal前半块，置零后作为该索引块的累加和
            auto valuesLocalCT = this->valuesLocal.template ReinterpretCast<CT>();
            Duplicate(valuesLocalCT, (CT)0, TBUF_BLOCK_SIZE);
            PIPE_V_S();
            // batchNums个索引搬入并累加到valuesLocal前半段
            this->ProcessBatchNums<IC>(batchStart, batchNums, blockSize, blockStart, srcGm, needCast);
            // 索引块累加和加到outLocal上
            LocalTensor<CT> dstTensor = this->outLocal.template ReinterpretCast<CT>();
            LocalTensor<CT> srcTensor = this->valuesLocal.template ReinterpretCast<CT>();
            Add(dstTensor, dstTensor, srcTensor, blockSize);
            PIPE_V_S();
        }
    }

    // 第一阶段进来，IC为IT, needcast为True时需要cast
    // 第二阶段进来，IC为CT, neadcast一定为false
    template <typename IC>
    __aicore__ inline void ProcessLastIndex(uint64_t indexId, uint64_t blockSize, uint64_t blockStart,
                                             GlobalTensor<IC> srcGm, bool needCast=false) {
        if (blockSize == 0) {
            return;
        }
        uint64_t src = this->posIdxLocal.GetValue(indexId) * this->sliceSize + blockStart;
        this->CopyInValues<IC>(src, 0, blockSize, srcGm);
        // 2. 升精度
        uint64_t blockSizeAligned = ((blockSize + ALIGNED_NUM - 1) / ALIGNED_NUM) * ALIGNED_NUM;
        if (needCast) {
            if (!this->accumulate && this->castEnable) {
                DataCopy(this->outLocal, this->valuesLocal, blockSizeAligned);
            } else {
                LocalTensor<CT> dstTensor = this->outLocal.template ReinterpretCast<CT>();
                Cast(dstTensor, this->valuesLocal, RoundMode::CAST_NONE, blockSize);
            }
        } else {
            auto srcTensor = this->valuesLocal.template ReinterpretCast<CT>();
            auto dstTensor = this->outLocal.template ReinterpretCast<CT>();
            uint64_t blockSizeAligned = ((blockSize + ALIGNED_NUM - 1) / ALIGNED_NUM) * ALIGNED_NUM;
            DataCopy(dstTensor, srcTensor, blockSizeAligned);
        }
        PIPE_V_S();
    }

    // outLocal搬出或者累加出去
    __aicore__ inline void ThrowOutResult(uint64_t indexStartId, uint64_t blockSize, uint64_t blockStart, bool fromCache=false) {
        int32_t indexValue = this->indexLocal.GetValue(indexStartId);
        if (indexValue == this->headIndexValue) {
            uint64_t src = 0;
            uint64_t dst = GetBlockIdx() * BUFFER_NUM * this->sliceSize + blockStart;
            LocalTensor<CT> srcTensor = this->outLocal.template ReinterpretCast<CT>();
            this->accumulate == 1 ? this->AddToCache(src, dst, srcTensor, blockSize) :
                                    this->CopyToCache(src, dst, srcTensor, blockSize);
        } else if (indexValue == this->tailIndexValue) {
            uint64_t src = 0;
            uint64_t dst = GetBlockIdx() * BUFFER_NUM * this->sliceSize + this->sliceSize + blockStart;
            LocalTensor<CT> srcTensor = this->outLocal.template ReinterpretCast<CT>();
            this->accumulate == 1 ? this->AddToCache(src, dst, srcTensor, blockSize) :
                                    this->CopyToCache(src, dst, srcTensor, blockSize);
        } else {
            // fp32前面未搬入out, 需要累加到out上
            if (this->castEnable || (fromCache && this->castEnable)) {
                // 转换为IT
                uint64_t blockSizeAligned = ((blockSize + ALIGNED_NUM - 1) / ALIGNED_NUM) * ALIGNED_NUM;
                if (this->castEnable) {
                    LocalTensor<CT> dstTensor = this->valuesLocal.template ReinterpretCast<CT>();
                    LocalTensor<CT> srcTensor = this->outLocal.template ReinterpretCast<CT>();
                    DataCopy(dstTensor, srcTensor, blockSizeAligned); // 保留outLocal值供下段索引使用防止掉精度
                    if (!(!this->accumulate && this->castEnable)) {
                        Cast(this->outLocal, srcTensor, RoundMode::CAST_RINT, blockSize);
                    }
                    PIPE_V_S();
                }
                // 搬运至outGm
                uint64_t src = 0;
                uint64_t dst = indexValue * this->sliceSize + blockStart;
                CopyToOut(src, dst, this->outLocal, blockSize);
                PIPE_MTE3_S();
                // 保留outLocal值供下段索引使用防止掉精度
                LocalTensor<CT> srcTensor = this->valuesLocal.template ReinterpretCast<CT>();
                LocalTensor<CT> dstTensor = this->outLocal.template ReinterpretCast<CT>();
                DataCopy(dstTensor, srcTensor, blockSizeAligned);
                PIPE_V_S();
            } else {
                uint64_t src = 0;
                uint64_t dst = indexValue * this->sliceSize + blockStart;
                this->accumulate == 1 ? this->AddToOut(src, dst, this->outLocal, blockSize) : // out值最后加
                                        this->CopyToOut(src, dst, this->outLocal, blockSize);
                PIPE_MTE3_S();
            }
        }
    }

    /**
        @brief 处理outLocal，当需要升精度计算时搬入outGm，不需要升精度时置零outLocal
        @param indexStartId 索引段在ub上的起始下标
    */
    __aicore__ inline void InitOutLocal(uint64_t indexStartId, uint64_t blockSize, uint64_t blockStart) {
        if (this->accumulate == 0) {
            return;
        }
        int32_t indexValue = this->indexLocal.GetValue(indexStartId);
        if (indexValue == this->indexBlockLastValue) {
            return; //索引段首值索引与上个索引段尾值索引相同，使用它的累加和，不用去out搬入
        }
        if (this->castEnable && indexValue != this->headIndexValue && indexValue != this->tailIndexValue) {
            uint64_t src = indexValue * this->sliceSize + blockStart;
            uint64_t dst = TBUF_BLOCK_SIZE;
            CopyInOut(src, dst, blockSize);
            // 转为CT
            LocalTensor<IT> srcTensor = this->outLocal[TBUF_BLOCK_SIZE];
            LocalTensor<CT> dstTensor = this->outLocal.template ReinterpretCast<CT>();
            Cast(dstTensor, srcTensor, RoundMode::CAST_NONE, blockSize);
            PIPE_V_S();
        } else {
            auto dstTensor = this->outLocal.template ReinterpretCast<CT>();
            Duplicate(dstTensor, (CT)0, TBUF_BLOCK_SIZE);
            PIPE_V_S();
        }
    }

    template <typename IC> // IC为IT或CT，第一阶段为IT从valuesGm搬入数据，第二阶段为CT从cacheSpaceGm搬入数据
    __aicore__ inline void ProcessBatchNums(uint64_t batchIndexStart, uint64_t nums, uint64_t blockSize, uint64_t blockStart,
                                            GlobalTensor<IC> srcGm, bool needCast) {
        for (uint64_t i = 0; i < nums; i++) {
            // 1. 往valuesLocal后半段搬入一行
            uint64_t src = this->posIdxLocal.GetValue(batchIndexStart + i) * this->sliceSize + blockStart;
            uint64_t dst = TBUF_BLOCK_SIZE + TBUF_BLOCK_SIZE * BUFFER_NUM * needCast;
            this->CopyInValues<IC>(src, dst, blockSize, srcGm);
            // 2. 升精度
            if (needCast) {
                uint64_t src = TBUF_BLOCK_SIZE * 3; // 当IT != CT时，将valuesLocal分成4块，最后一块是刚搬入的数据
                LocalTensor<IT> srcTensor = this->valuesLocal[src];
                LocalTensor<CT> dstTensor = this->valuesLocal.template ReinterpretCast<CT>()[TBUF_BLOCK_SIZE];
                Cast(dstTensor, srcTensor, RoundMode::CAST_NONE, blockSize);
                PIPE_V_S();
            }
            // 3. 加到valuesLocal前半段
            LocalTensor<CT> srcTensor = this->valuesLocal.template ReinterpretCast<CT>()[TBUF_BLOCK_SIZE];
            LocalTensor<CT> dstTensor = this->valuesLocal.template ReinterpretCast<CT>();
            Add(dstTensor, dstTensor, srcTensor, blockSize);
            PIPE_V_S();
        }
    }

    /**
        @brief 从outGm向outLocal搬入一段数据，数据类型为IT
        @param src 搬运起始位置，单位为元素
        @param dst 搬运目的位置，单位为元素
        @param count 搬运数目，单位为元素
    */
    __aicore__ inline void CopyInOut(uint64_t src, uint64_t dst, uint64_t count) {
        CopyGm2Ub<IT>(this->outLocal[dst], this->outGm[src], count);
        PIPE_MTE2_S();
    }

    template <typename IC>
    __aicore__ inline void CopyInValues(const uint64_t src, const uint64_t dst, const int64_t count, GlobalTensor<IC> srcGm) {
        auto dstTensor = this->valuesLocal.template ReinterpretCast<IC>();
        CopyGm2Ub<IC>(dstTensor[dst], srcGm[src], count);
        PIPE_MTE2_S();
    }

    /**
        @brief 数据累加至out，只有fp32/int32使用到
        @param src 由索引在indexLocal上的下标计算得来的valuesLocal搬运起始点
        @param dst 由索引值计算得来的目的地址起始点，单位是元素个数
    */
    __aicore__ inline void AddToOut(uint64_t src, uint64_t dst, const LocalTensor<IT>& srcTensor, uint64_t blockSize) {
        AddUb2Gm<IT>(this->outGm[dst], srcTensor[src], blockSize);
        PIPE_MTE3_S();
    }

    /**
        @brief 数据搬运至out，无同步
        @param src 由索引在indexLocal上的下标计算得来的valuesLocal搬运起始点
        @param dst 由索引值计算得来的目的地址起始点，单位是元素个数
    */
    __aicore__ inline void CopyToOut(uint64_t src, uint64_t dst, const LocalTensor<IT>& srcTensor, uint64_t blockSize) {
        CopyUb2Gm<IT>(this->outGm[dst], srcTensor[src], blockSize);
    }

    __aicore__ inline void CopyInIndices(uint64_t start, uint64_t size) {
        CopyGm2Ub<int32_t>(this->indexLocal, this->indexGm[start], size);
        PIPE_MTE2_S();
    }

    __aicore__ inline void CopyInPosIdx(uint64_t start, uint64_t size) {
        CopyGm2Ub<int32_t>(this->posIdxLocal, this->posIdxGm[start], size);
        PIPE_MTE2_S();
    }

    __aicore__ inline void AddToCache(uint64_t src, uint64_t dst, const LocalTensor<CT>& srcTensor, uint64_t blockSize) {
        AddUb2Gm<CT>(this->cacheSpaceGm[dst], srcTensor[src], blockSize);
        PIPE_MTE3_S();
    }

    __aicore__ inline void CopyToCache(uint64_t src, uint64_t dst, const LocalTensor<CT>& srcTensor, uint64_t blockSize) {
        CopyUb2Gm<CT>(this->cacheSpaceGm[dst], srcTensor[src], blockSize);
        PIPE_MTE3_S();
    }

protected:
    GlobalTensor<IT> outGm;
    GlobalTensor<int32_t> indexGm;
    GlobalTensor<int32_t> posIdxGm;
    GlobalTensor<IT> valuesGm;
    GlobalTensor<int32_t> syncSpaceGm;
    GlobalTensor<CT> cacheSpaceGm;

    uint32_t sliceSize = 0;
    uint32_t allIndicesNums = 0;
    uint32_t sliceSizeAligned = 0;

    uint64_t coreStartId = 0; // 核在尾轴上的起始位置
    uint64_t coreEndId = 0; // 核在位置上的截止位置

    uint64_t frontBlocks = 0; // 整块数，再核上计算
    uint64_t frontBlockSize = 0; // 整块大小
    uint64_t lastBlockSize = 0; // 尾块大小

    bool castEnable = (sizeof(IT) != sizeof(CT));
    int32_t headIndexValue = -1;
    int32_t tailIndexValue = -1;
    uint32_t accumulate = 0;
    uint64_t indexBlockLastValue = -1;

    LocalTensor<int32_t> indexLocal;
    LocalTensor<int32_t> posIdxLocal;
    LocalTensor<IT> valuesLocal;
    LocalTensor<IT> outLocal;

    TPipe* pipe;
    TBuf<TPosition::VECCALC> indexUbBuf;
    TBuf<TPosition::VECCALC> posIdxUbBuf;
    TBuf<TPosition::VECCALC> valuesUbBuf;
    TBuf<TPosition::VECCALC> outUbBuf;
};

template<typename IT, typename CT>
class ScatterDataInKernelOp : public ScatterDataBetweenKernelOp<IT, CT> {
public:
    __aicore__ inline void Init(GM_ADDR out, GM_ADDR linear_index, GM_ADDR pos_idx, 
                                GM_ADDR values, GM_ADDR userWorkspace,
                                const IndexPutWithSortTilingData* tilingData, TPipe* tpipe) {
        InitTilingData(tilingData, tpipe);
        this->InitLocalTensor(INDEX_UB_NUMS, TBUF_BLOCK_SIZE * BUFFER_NUM, TBUF_BLOCK_SIZE);
        InitGmTensor(out, linear_index, pos_idx, values, userWorkspace);
    }

    __aicore__ inline void InitTilingData(const IndexPutWithSortTilingData* tilingData, TPipe* tpipe) {
        this->pipe = tpipe;
        this->coreNums = tilingData->coreNums;
        this->sliceSize = tilingData->sliceSize;
        this->sliceSizeAligned = tilingData->sliceSizeAligned;
        this->frontBlocks = tilingData->frontBlocks;
        this->frontBlockSize = tilingData->frontBlockSize;
        this->lastBlockSize = tilingData->lastBlockSize;
        this->accumulate = tilingData->accumulate;

        if (GetBlockIdx() < tilingData->frontCoreNums) {
            this->coreStartIndex = GetBlockIdx() * tilingData->frontCoreIndicesNums;
            this->coreEndIndex = this->coreStartIndex + tilingData->frontCoreIndicesNums;
        } else {
            this->coreStartIndex = tilingData->frontCoreNums * tilingData->frontCoreIndicesNums + 
                  (GetBlockIdx() - tilingData->frontCoreNums) * tilingData->tailCoreIndicesNums;
            this->coreEndIndex = this->coreStartIndex + tilingData->tailCoreIndicesNums;
        }
    }

    __aicore__ inline void InitGmTensor(GM_ADDR out, GM_ADDR linear_index, GM_ADDR pos_idx, 
                                        GM_ADDR values, GM_ADDR userWorkspace) {
        ScatterDataBetweenKernelOp<IT, CT>::InitGmTensor(out, linear_index, pos_idx, values);
        this->syncSpaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(userWorkspace));
        uint32_t syncLength = this->coreNums * SYNC_SIZE;
        this->cacheSpaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ CT*>(userWorkspace) + syncLength);
        InitWorkSpace();
    }

    __aicore__ inline void InitWorkSpace() {
        const uint64_t sizeSync = this->coreNums * SYNC_SIZE;
        Duplicate(this->indexLocal, (int32_t)0, sizeSync);
        PIPE_V_S();
        CopyUb2Gm<int32_t>(this->syncSpaceGm, this->indexLocal, sizeSync);
        PIPE_MTE3_S();

        auto valuesLocalCT = this->valuesLocal.template ReinterpretCast<CT>();
        Duplicate(valuesLocalCT, (CT)0, TBUF_BLOCK_SIZE);
        PIPE_V_S();
        for (uint64_t i = 0; i < BUFFER_NUM; i++) {
            for (uint64_t j = 0; j <= this->frontBlocks; j++) {
                if (j == this->frontBlocks && this->lastBlockSize == 0) {
                    break;
                }
                uint64_t blockSize = (j == this->frontBlocks ? this->lastBlockSize : this->frontBlockSize);
                uint64_t blockStart = j * this->frontBlockSize;
                uint64_t dst = GetBlockIdx() * BUFFER_NUM * this->sliceSize + i * this->sliceSize + blockStart;
                auto coreCacheGm = this->cacheSpaceGm[dst];
                CopyUb2Gm<CT>(coreCacheGm, valuesLocalCT, blockSize);
            }
        }
        PIPE_MTE3_S();
        SyncAll();
    }

    __aicore__ inline void Process() {
        this->GetHeadTailIndexValue();
        // 尾轴切块
        for (uint64_t i = 0; i <= this->frontBlocks; i++) {
            if (i == this->frontBlocks && this->lastBlockSize == 0) {
                break;
            }
            uint64_t blockSize = (i == this->frontBlocks ? this->lastBlockSize : this->frontBlockSize);
            uint64_t blockStart = i * this->frontBlockSize;
            uint64_t indicesNums = this->coreEndIndex - this->coreStartIndex;
            this->ProcessDataBlockWithIndices(blockStart, blockSize, this->coreStartIndex, indicesNums);
        }

        this->CopyOutSyncData();
        SyncAll();
        this->PIPE_CORE_S();
        this->indexBlockLastValue = -1;

        for (uint64_t i = 0; i <= this->frontBlocks; i++) {
            if (i == this->frontBlocks && this->lastBlockSize == 0) {
                break;
            }
            uint64_t blockSize = (i == this->frontBlocks ? this->lastBlockSize : this->frontBlockSize);
            uint64_t blockStart = i * this->frontBlockSize;
            this->ProcessCacheData(blockSize, blockStart);
        }
    }

    /**
        @brief 处理cacheSpaceGm上的数据，每个核需要处理自己的头部数据，和上一个核的尾部数据
    */
    __aicore__ inline void ProcessCacheData(uint64_t blockSize, uint64_t blockStart) {
        // 搬入所有核的头部、尾部索引值
        uint32_t validIndicesNums = this->CopyInCacheIndices();
        this->headIndexValue = -1;
        this->tailIndexValue = -1;
        if (validIndicesNums == 1) { //必须注重分支覆盖率
            if (GetBlockIdx() == 0) {
                this->InitOutLocal(0, blockSize, blockStart);
                this->template ProcessGivenIndex<CT>(0, 1, blockSize, blockStart, this->cacheSpaceGm);
                this->ThrowOutResult(0, blockSize, blockStart, true);
            }
        }

        uint32_t pre = 0;
        uint64_t coreId = 0;
        int32_t preIndexValue = this->indexLocal.GetValue(pre);
        for (uint64_t i = 1; i < validIndicesNums; i++) {
            int32_t indexValue = this->indexLocal.GetValue(i);
            if (indexValue != preIndexValue) {
                GetFreeCore(coreId);
                if (GetBlockIdx() == coreId) {
                    this->InitOutLocal(pre, blockSize, blockStart);
                    this->template ProcessGivenIndex<CT>(pre, i, blockSize, blockStart, this->cacheSpaceGm);
                    this->ThrowOutResult(pre, blockSize, blockStart, true);
                }
                pre = i;
                preIndexValue = indexValue;
            }
            if (i == validIndicesNums - 1) {
                GetFreeCore(coreId);
                if (GetBlockIdx() == coreId) {
                    this->InitOutLocal(pre, blockSize, blockStart);
                    this->template ProcessGivenIndex<CT>(pre, i + 1, blockSize, blockStart, this->cacheSpaceGm);
                    this->ThrowOutResult(pre, blockSize, blockStart, true);
                }
            }
        }
    }

    /**
        @brief 处理所有核的首尾索引值，去除首尾索引值相同的
        最后使posIdxLocal上记录“values”在cacheSpaceGm上的位置，indexLocal记录要写入out的位置，并且indexLocal是有序的
        validIndicesNums记录有多少个索引。
    */
    __aicore__ inline uint32_t CopyInCacheIndices() {
        // 搬入所有核的首尾索引值
        uint32_t dst = INDEX_UB_NUMS / BUFFER_NUM;
        uint64_t size = this->coreNums * SYNC_SIZE;
        CopyGm2Ub<int32_t>(this->indexLocal[dst], this->syncSpaceGm, size);
        PIPE_MTE2_S();

        uint32_t validIndicesNums = 0;
        for (uint64_t i = 0; i < this->coreNums; i++) {
            uint64_t coreSrc = dst + i * SYNC_SIZE;
            int32_t coreHeadIndexValue = this->indexLocal.GetValue(coreSrc + 1);
            int32_t coreTailIndexValue = this->indexLocal.GetValue(coreSrc + 2); // 2 is tailIndexValue
            this->indexLocal.SetValue(validIndicesNums, coreHeadIndexValue);
            this->posIdxLocal.SetValue(validIndicesNums, i * 2); // 2 is double location
            validIndicesNums++;
            if (coreHeadIndexValue != coreTailIndexValue) {
                this->indexLocal.SetValue(validIndicesNums, coreTailIndexValue);
                this->posIdxLocal.SetValue(validIndicesNums, i * 2 + 1);// 2 is double location
                validIndicesNums++;
            }
        }
        return validIndicesNums;
    }

    __aicore__ inline void GetHeadTailIndexValue() {
        this->CopyInIndices(this->coreStartIndex, 1);
        this->headIndexValue = this->indexLocal.GetValue(0);
        this->CopyInIndices(this->coreEndIndex - 1, 1);
        this->tailIndexValue = this->indexLocal.GetValue(0);
    }

    /**
        @brief 核间同步相关信息写入syncGm
    */
    __aicore__ inline void CopyOutSyncData() {
        this->indexLocal.SetValue(0, 1); // 第一阶段完成标识
        this->indexLocal.SetValue(1, this->headIndexValue); // core首索引值
        this->indexLocal.SetValue(2, this->tailIndexValue); // core尾索引值
        PIPE_S_MTE3();
        uint64_t dst = GetBlockIdx() * SYNC_SIZE;
        CopyUb2Gm<int32_t>(this->syncSpaceGm[dst], this->indexLocal, SYNC_NUMS);
        PIPE_MTE3_S();
    }

    /**
        @brief 阻塞等待前一个核第一阶段计算任务的完成
    */
    __aicore__ inline void PIPE_CORE_S() {
        uint64_t frontCoreSrc = (GetBlockIdx() - 1) * SYNC_SIZE;
        if (GetBlockIdx() != 0) {
            while(true) {
                this->CopyInSyncData(frontCoreSrc);
                if (this->indexLocal.GetValue(0) == 1) {
                    break;
                }
            }
        }
    }

    __aicore__ inline void CopyInSyncData(uint64_t src) {
        CopyGm2Ub<int32_t>(this->indexLocal, this->syncSpaceGm[src], 1);
        PIPE_MTE2_S();
    }

    __aicore__ inline void GetFreeCore(uint64_t& coreId) {
        coreId += 1;
        if (coreId == this->coreNums) {
            coreId = 0;
        }
    }

protected:
    uint32_t coreNums = 0;
    uint64_t coreStartIndex = 0;
    uint64_t coreEndIndex = 0; // [start, end) 核分到的索引范围
};
}
#endif // INDEX_PUT_WITH_SORT_SCATTER_DATA