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
 * \file scatter_nd_common_simt_sort_tiling.cpp
 * \brief scatter_nd_common_simt_sort_tiling
 */

#include "scatter_nd_common_simt_sort_tiling.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

using namespace ScatterNdCommon;

static constexpr int64_t SORT_LIMIT = 5;
static constexpr int64_t ALIGN_SIZE = 32;
static constexpr int64_t MIN_HANDLE_SIZE = 128;
static constexpr int64_t INT32_BYTES = 4;
static constexpr int64_t FP32_BYTES = 4;
static constexpr int64_t DCACHE_SIZE = 32768;  // 32k
static constexpr int64_t INDICES_MIN_BLOCK_SIZE = 1024;
static constexpr int64_t TEMPLATE_MODE_SIMT_SORT = 5;
static constexpr uint64_t RESERVE_SIZE = 256;
static constexpr uint64_t DB_BUFFER = 2;
static constexpr uint64_t MIN_TILING_SIZE = 128;

bool ScatterNdCommonSimtSortTiling::IsSimtSortSupportType(ge::DataType updateDtype)
{
    if (updateDtype == ge::DT_FLOAT || updateDtype == ge::DT_FLOAT16
        || updateDtype == ge::DT_INT32 || updateDtype == ge::DT_UINT32
        || updateDtype == ge::DT_INT64 || updateDtype == ge::DT_UINT64
        || updateDtype == ge::DT_BF16 || updateDtype == ge::DT_BOOL) {
        return true;
    }
    return false;
}

bool ScatterNdCommonSimtSortTiling::IsCapable()
{
    if (varInAxis_ == 0) {
        return false;
    }

    // 是否排序
    if (indicesAxis_ / varInAxis_ >= SORT_LIMIT && IsSimtSortSupportType(updateDtype_)) {
        return true;
    }
    return false;
}

// 计算单个索引行相关的UB缓冲总大小
int64_t ScatterNdCommonSimtSortTiling::CalcOneIndexRowAlignSize(int64_t ubBlock)
{
    int64_t indicesDtypeCastSize = indiceCastMode_ ? indiceCastDtypeSize_ : 0;
    int64_t indicesRealTypeSize = indiceCastMode_ ? indiceCastDtypeSize_ : indicesTypeSize_;
    int64_t oneIndexSize = static_cast<int64_t>(rankSize_) * indicesTypeSize_;

    return Ops::Base::CeilAlign(oneIndexSize, ubBlock) +                                  // indicesBuf_: 1个索引行的数据
           Ops::Base::CeilAlign(indicesTypeSize_, ubBlock) +                              // outOfstBuf: 1个一维线性偏移值
           Ops::Base::CeilAlign(indicesDtypeCastSize, ubBlock) +                          // castIndicesQue: 存放indices cast后的数据
           Ops::Base::CeilAlign(indicesRealTypeSize + TWO * ALIGN_SIZE, ubBlock) +        // sortIndicesQue + 64字节padding
           Ops::Base::CeilAlign(INT32_BYTES, ubBlock) +                                   // updateOriginIndexQue: 原始索引(1个)
           Ops::Base::CeilAlign(INT32_BYTES * TWO, ubBlock) +                             // uniqueIdCountQue: 唯一索引计数(1个)
           Ops::Base::CeilAlign(indicesTypeSize_, ubBlock);                               // updateSumIdxQue: 唯一索引位置(1个)
}

// 计算单个update行相关的UB缓冲总大小
int64_t ScatterNdCommonSimtSortTiling::CalcOneUpdateRowAlignSize(int64_t ubBlock, ge::DataType sortTmpType)
{
    return Ops::Base::CeilAlign(varTypeSize_ * afterAxis_, ubBlock) +                      // dataQueue: 完整afterAxis的一行update数据
           Ops::Base::CeilAlign(varTypeSize_ * afterAxis_, ubBlock) +                      // updateSumQue: 完整afterAxis的一行累加数据
           GetSortTmpSize(sortTmpType, 1, false);                                          // sort需要的空间大小
}

// 处理ub空间不足情况: indices 空间 + update 空间 > halfUbSize 时的分块策略
void ScatterNdCommonSimtSortTiling::CalcTilingWhenTight(int64_t halfUbSize, int64_t indicesAlignSize, ge::DataType sortTmpType)
{
    int64_t indicesSize = std::min(INDICES_MIN_BLOCK_SIZE, indicesAxis_ * indicesAlignSize);
    if (indicesAlignSize == 0) {
        return;
    }
    indicesFactor_ = Ops::Base::CeilAlign(indicesSize, ALIGN_SIZE) / indicesAlignSize;
    int64_t sortTmpSize = GetSortTmpSize(sortTmpType, indicesFactor_, false);
    int64_t restSize = halfUbSize - indicesFactor_ * indicesAlignSize - sortTmpSize;
    int64_t alignNum = ALIGN_SIZE / varTypeSize_;
    afterAxisFactor_ = restSize / (indicesFactor_ * (varTypeSize_ + varTypeSize_));   // (update + updateSum)各占一个varTypeSize_大小的空间
    afterAxisFactor_ = Ops::Base::FloorAlign(afterAxisFactor_, alignNum);
}

// 处理ub空间充裕时的分块策略
void ScatterNdCommonSimtSortTiling::CalcTilingWhenSpacious(int64_t halfUbSize, int64_t ubBlock, int64_t indicesAlignSize,
    int64_t updateAlignSize, ge::DataType sortTmpType)
{
    int64_t alignNum = ALIGN_SIZE / varTypeSize_;
    int64_t indicesDtypeCastSize = indiceCastMode_ ? indiceCastDtypeSize_ : 0;
    int64_t indicesRealTypeSize = indiceCastMode_ ? indiceCastDtypeSize_ : indicesTypeSize_;

    afterAxisFactor_ = Ops::Base::CeilAlign(afterAxis_, alignNum);
    indicesFactor_ = halfUbSize / (updateAlignSize + indicesAlignSize);

    // restSize默认为-1, 作为循环触发器
    int64_t restSize = static_cast<int64_t>(-1);
    while (restSize <= 0 && indicesFactor_ > 1) {
        int64_t occupy = Ops::Base::CeilAlign(indicesFactor_ * rankSize_ * indicesTypeSize_, ubBlock) +
                        Ops::Base::CeilAlign(indicesFactor_ * indicesTypeSize_, ubBlock) +
                        Ops::Base::CeilAlign(indicesFactor_ * indicesDtypeCastSize, ubBlock) +
                        Ops::Base::CeilAlign(indicesFactor_ * (indicesRealTypeSize + TWO * ALIGN_SIZE), ubBlock) +
                        Ops::Base::CeilAlign(indicesFactor_ * INT32_BYTES, ubBlock) +
                        Ops::Base::CeilAlign(indicesFactor_ * (INT32_BYTES * TWO), ubBlock) +
                        Ops::Base::CeilAlign(indicesFactor_ * indicesTypeSize_, ubBlock) +
                        indicesFactor_ * Ops::Base::CeilAlign(varTypeSize_ * afterAxisFactor_, ubBlock) +
                        indicesFactor_ * Ops::Base::CeilAlign(varTypeSize_ * afterAxisFactor_, ubBlock) +
                        GetSortTmpSize(sortTmpType, indicesFactor_, false);
        restSize = halfUbSize - occupy;
        if (indicesFactor_ > indicesAxis_) {
            indicesFactor_ = indicesAxis_;
            break;
        }
        --indicesFactor_;
    }
}

void ScatterNdCommonSimtSortTiling::DoOpTilingSplitIndices()
{
    ge::DataType sortTmpType = indiceCastMode_ ?  indiceCastDtype_ : indiceDtype_;

    int64_t halfUbSize = static_cast<int64_t>((ubSize_ - RESERVE_SIZE) / DB_BUFFER);
    halfUbSize = halfUbSize - MIN_HANDLE_SIZE * FP32_BYTES; // maxScore

    // split indices分核
    eachCoreIndexCount_ = Ops::Base::CeilDiv(indicesAxis_, totalCoreNum_);
    usedCoreNumBefore_ = Ops::Base::CeilDiv(indicesAxis_, eachCoreIndexCount_);
    tailCoreIndexCount_ = indicesAxis_ - eachCoreIndexCount_ * (usedCoreNumBefore_ - 1);

    // 同地址优化:搬入多少行indices,就搬入相同行数的updates, strideBuf放在RESERVE_SIZE中
    auto ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    int64_t indicesAlignSize = CalcOneIndexRowAlignSize(ubBlock);
    int64_t updateAlignSize  = CalcOneUpdateRowAlignSize(ubBlock, sortTmpType);
    if (indicesAlignSize + updateAlignSize > halfUbSize) {
        CalcTilingWhenTight(halfUbSize, indicesAlignSize, sortTmpType);
    } else {
        CalcTilingWhenSpacious(halfUbSize, ubBlock, indicesAlignSize, updateAlignSize, sortTmpType);
    }

    updateLoopSize_ = Ops::Base::CeilDiv(afterAxis_, afterAxisFactor_);
    updateTailNum_ = afterAxis_ - (updateLoopSize_ - 1) * afterAxisFactor_;
}

ge::graphStatus ScatterNdCommonSimtSortTiling::DoOpTiling()
{
    GetCastType();
    DoOpTilingSplitIndices();
    SetStride();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdCommonSimtSortTiling::PostTiling()
{
    OP_LOGD("ScatterNdCommonSimtSortTiling::PostTiling begin");
    context_->SetTilingKey(GetTilingKey());
    context_->SetScheduleMode(1);
    context_->SetBlockDim(usedCoreNumBefore_);
    if (indiceShapeSize_ == 0 || updateShapeSize_ == 0) {
        // 输入为空tensor时，设置blockNum为1，在kernel中直接返回
        context_->SetBlockDim(1);
    }

    uint32_t usedLocalMemorySize = static_cast<uint32_t>(ubSize_ - DCACHE_SIZE);
    auto res = context_->SetLocalMemorySize(usedLocalMemorySize);
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS),
        OP_LOGD(context_->GetNodeName(), "SetLocalMemorySize ubSize = %lu failed.", usedLocalMemorySize), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

uint64_t ScatterNdCommonSimtSortTiling::GetTilingKey() const
{
    OP_LOGD("ScatterNdCommonSimtSortTiling::GetTilingKey begin");
    uint64_t addrMode = 1;
    if (indiceShapeSize_ < UINT32_MAX && updateShapeSize_ < UINT32_MAX && outputShapeSize_ < UINT32_MAX) {
        addrMode = 0;
    }
    const uint64_t tilingKey = GET_TPL_TILING_KEY(TEMPLATE_MODE_SIMT_SORT, indiceCastMode_, addrMode);
    OP_LOGD(context_->GetNodeName(), "tilingKey is: [%lu]", tilingKey);
    return tilingKey;
}

void ScatterNdCommonSimtSortTiling::SetTilingData()
{
    ScatterNdCommonSimtSortTilingData *tilingData = context_->GetTilingData<ScatterNdCommonSimtSortTilingData>();
    tilingData->indicesFactor = indicesFactor_;
    tilingData->afterAxis = afterAxis_;
    tilingData->varInAxis = varInAxis_;
    tilingData->afterAxisFactor = afterAxisFactor_;
    tilingData->indexRankSize = rankSize_;
    tilingData->eachCoreAfterAxisCount = eachCoreAfterAxisCount_;
    tilingData->eachCoreIndexCount = eachCoreIndexCount_;
    tilingData->tailCoreIndexCount = tailCoreIndexCount_;
    tilingData->usedCoreNumBefore = usedCoreNumBefore_;
    tilingData->updateLoopSize = updateLoopSize_;
    tilingData->updateTailNum = updateTailNum_;

    for (int32_t i = 0; i < OPTILING_MAX_RANK_COUNT; i++) {
        tilingData->strideList[i] = strideList_[i];
    }

    for (int32_t i = 0; i < OPTILING_MAX_SHAPE_RANK; i++) {
        tilingData->outPutShape[i] = outPutShape_[i];
    }
}

std::string ScatterNdCommonSimtSortTiling::TilingDataToString()
{
    ScatterNdCommonSimtSortTilingData* tilingData = context_->GetTilingData<ScatterNdCommonSimtSortTilingData>();
    std::string str = " indicesFactor:" + std::to_string(tilingData->indicesFactor);
    str += " afterAxis:" + std::to_string(tilingData->afterAxis);
    str += " varInAxis:" + std::to_string(tilingData->varInAxis);
    str += " afterAxisFactor:" + std::to_string(tilingData->afterAxisFactor);
    str += " indexRankSize:" + std::to_string(tilingData->indexRankSize);
    str += " eachCoreAfterAxisCount:" + std::to_string(tilingData->eachCoreAfterAxisCount);
    str += " eachCoreIndexCount:" + std::to_string(tilingData->eachCoreIndexCount);
    str += " tailCoreIndexCount:" + std::to_string(tilingData->tailCoreIndexCount);
    str += " usedCoreNumBefore:" + std::to_string(tilingData->usedCoreNumBefore);
    str += " updateLoopSize:" + std::to_string(tilingData->updateLoopSize);
    str += " updateTailNum:" + std::to_string(tilingData->updateTailNum);

    for (int32_t i = 0; i < OPTILING_MAX_RANK_COUNT; i++) {
        str += " strideList[i]:" + std::to_string(tilingData->strideList[i]);
    }
    return str;
}

void ScatterNdCommonSimtSortTiling::DumpTilingInfo()
{
    OP_LOGI(context_->GetNodeName(), "Tiling info is: %s", TilingDataToString().c_str());
}

} // namespace optiling