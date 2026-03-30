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
 * \file index_put_with_sort_v2_tiling_arch35.cpp
 * \brief IndexPutWithSortV2 regbase tiling file
 */

#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "index_put_with_sort_v2_tiling_arch35.h"
#include "index/index_put_with_sort_v2/op_kernel/arch35/index_put_with_sort_v2_struct.h"
namespace optiling
{
using namespace Ops::NN::OpTiling;

constexpr int64_t INPUT_NUM = 4;
constexpr int64_t OUTPUT_NUM = 1;
constexpr int64_t ATTR_INDEX_0 = 0;
constexpr int64_t ATTR_INDEX_1 = 1;
constexpr int64_t INPUT_INDEX_0 = 0;
constexpr int64_t INPUT_INDEX_1 = 1;
constexpr int64_t INPUT_INDEX_2 = 2;
constexpr int64_t INPUT_INDEX_3 = 3;
constexpr int64_t INPUT_INDEX_4 = 4;
constexpr uint32_t DCACHE_SIZE = 32768;
constexpr uint32_t ASCENDC_TOOLS_WORKSPACE = 16777216;
constexpr int32_t THREAD_NUM_FULL = 512;
constexpr int32_t THREAD_NUM_HALF = 512;
constexpr int64_t TILING_KEY_ACCUMULATE_TRUE = 1;
constexpr int64_t TILING_KEY_ALL_INDEXED_TRUE = 10;
constexpr int64_t TILING_KEY_INDEXED_BLOCK_MODE_TRUE = 100;

ge::graphStatus IndexPutWithSortV2Tiling::GetPlatformInfo()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, maxUbSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexPutWithSortV2Tiling::CheckShapeAllPositive(gert::Shape& shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF(
            shape.GetDim(i) <= 0,
            OP_LOGE(context_->GetNodeName(),
                                            "Dim %lu of input should be positive, but actual %ld.", i, shape.GetDim(i)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexPutWithSortV2Tiling::CheckShapesEqual(gert::Shape& shape0, gert::Shape& shape1)
{
    OP_CHECK_IF(
        shape0.GetDimNum() != shape1.GetDimNum(),
        OP_LOGE(context_->GetNodeName(), "DimNum of shapes are not equal: %lu vs %lu",
                                        shape0.GetDimNum(), shape1.GetDimNum()),
        return ge::GRAPH_FAILED);

    for (size_t i = 0; i < shape0.GetDimNum(); i++) {
        OP_CHECK_IF(
            shape0.GetDim(i) != shape1.GetDim(i),
            OP_LOGE(context_->GetNodeName(), "Dim %lu of shapes are not equal: %ld vs %ld", i,
                                            shape0.GetDim(i), shape1.GetDim(i)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexPutWithSortV2Tiling::CheckInputsShape()
{
    // check self
    auto inputShape = context_->GetInputShape(INPUT_INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    auto storageShape0 = inputShape->GetStorageShape();
    if (CheckShapeAllPositive(storageShape0) != ge::GRAPH_SUCCESS) {
       OP_LOGE(context_->GetNodeName(), "self shape contains zero.");
        return ge::GRAPH_FAILED;
    }
    auto inputXDesc = context_->GetInputDesc(INPUT_INDEX_0);
 	OP_CHECK_NULL_WITH_CONTEXT(context_, inputXDesc);
 	auto inputXDtype = inputXDesc->GetDataType();
 	xDataType_ = inputXDtype;

    // check linear_index
    inputShape = context_->GetInputShape(INPUT_INDEX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    auto storageShape1 = inputShape->GetStorageShape();
    if (CheckShapeAllPositive(storageShape1) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "linear_index shape contains zero.");
        return ge::GRAPH_FAILED;
    }
    indexedDimSize_ = storageShape1.GetShapeSize();
    auto indexed = context_->GetInputDesc(INPUT_INDEX_1);
 	OP_CHECK_NULL_WITH_CONTEXT(context_, indexed);
 	auto indexedDtype = indexed->GetDataType();
 	indicesTypeSize_ = (indexedDtype == ge::DT_INT64) ? B8_SIZE : B4_SIZE;

    // check pos_idx
    inputShape = context_->GetInputShape(INPUT_INDEX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    auto storageShape2 = inputShape->GetStorageShape();
    if (CheckShapeAllPositive(storageShape2) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "pos_idx shape contains zero.");
        return ge::GRAPH_FAILED;
    }

    // check shapes of input1 and input2 are equal
    if (CheckShapesEqual(storageShape1, storageShape2) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Shapes of linear_index and pos_idx are not equal.");
        return ge::GRAPH_FAILED;
    }

    // check values
    inputShape = context_->GetInputShape(INPUT_INDEX_3);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    auto storageShape3 = inputShape->GetStorageShape();
    if (CheckShapeAllPositive(storageShape3) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "values shape contains zero.");
        return ge::GRAPH_FAILED;
    }

    selfDimNum_ = storageShape0.GetDimNum();
    for (uint32_t i = 0; i < selfDimNum_; i++) {
        selfDims_[i] = storageShape0.GetDim(i);
    }
    return ge::GRAPH_SUCCESS;
}


bool IndexPutWithSortV2Tiling::IsIndexedContinous(const int64_t* arr, int64_t size)
{
    int64_t prevIndex = static_cast<int64_t>(-1);
    for (int64_t i = 0; i < size; i++) {
        if (arr[i] == 1) {
            if (prevIndex != static_cast<int64_t>(-1)) {
                // 检查当前1和前一个1是否有gap
                if (i - prevIndex > 1) {
                    return false;
                }
            }
        }
        prevIndex = i;  // 更新前一个1的位置
    }
    return true;
}

ge::graphStatus IndexPutWithSortV2Tiling::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        OP_LOGE("IndexPutWithSortV2", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }

    // check inputs shape
    if (CheckInputsShape() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Inputs shape invalid.");
        return ge::GRAPH_FAILED;
    }
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto indexedSizesPtr = attrs->GetListInt(0);  // CheckNull
    int64_t indexedSizesDimNum = indexedSizesPtr->GetSize();  // check和selfDimNum_是否一致
    for (int64_t i = 0; i < indexedSizesDimNum; i++) {
        indexedSizes_[i] = indexedSizesPtr->GetData()[i];
    }
    isContinous_ = IsIndexedContinous(indexedSizes_, indexedSizesDimNum);
    indexed0_ = indexedSizes_[0];
    accumulate_ = *attrs->GetBool(1);

    for (size_t i = 0; i < selfDimNum_; i++) { // 注意之前要校验selfDimNum_和 indexedSizesDimNum一致
        if (indexedSizes_[i] == 0) {
            nonIdxedDims_[nonIndexedDimNum_] = selfDims_[i];
            nonIndexedDimNum_++;
            nonIndexedDimSize_ *= selfDims_[i];
        } else {
            indexedDimNum_++;
        }
    }
    return ge::GRAPH_SUCCESS;
}

bool IndexPutWithSortV2Tiling::IsCapable()
{
    return true;
}

uint64_t IndexPutWithSortV2Tiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(accumulate_, nonIndexedDimSize_ == 1, indexedBlockMode_ , false, 0);
}

ge::graphStatus IndexPutWithSortV2Tiling::PostTiling()
{
    context_->SetBlockDim(aivCoreNum_);
    context_->SetLocalMemorySize(maxUbSize_ - DCACHE_SIZE);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexPutWithSortV2Tiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexPutWithSortV2Tiling::GetWorkspaceSize()
{
    workspaceSize_ = ASCENDC_TOOLS_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}


void IndexPutWithSortV2Tiling::LogTilingResult()
{
    OP_LOGI(context_->GetNodeName(), "indexedDimSize_: %ld, nonIndexedDimSize_: %ld", indexedDimSize_, nonIndexedDimSize_);
}
 	 
template<typename T>
void CopyArray(T &arrSrc, T &arrDst, uint8_t count)
{
    for (uint8_t i = 0; i < count; i++) {
        arrDst[i] = arrSrc[i];
    }
}

void IndexPutWithSortV2Tiling::SetTilingData()
{
    IndexPutWithSortV2TilingData* tilingData =
    context_->GetTilingData<IndexPutWithSortV2TilingData>();
    tilingData->nonIndexedDimNum = static_cast<int64_t>(nonIndexedDimNum_);
    tilingData->indexedDimSize = indexedDimSize_;
    tilingData->nonIndexedDimSize = nonIndexedDimSize_;
    CopyArray(nonIdxedStride_, tilingData->nonIdxedStride, MAX_DIM_NUM);
    CopyArray(nonIdxedSelfStride_, tilingData->nonIdxedSelfStride, MAX_DIM_NUM);
    CopyArray(nonIdxedValueStride_, tilingData->nonIdxedValueStride, MAX_DIM_NUM);
    tilingData->idxedValueStride = idxedValueStride_;
    tilingData->indexedThreadNum = indexedThreadNum_;
    tilingData->nonIndexedThreadNum = nonIndexedThreadNum_;
}

void IndexPutWithSortV2Tiling::CalcSelfAndValueStride(int64_t* selfStride, int64_t* valueStride)
{
    int64_t valueReshapeDims[nonIndexedDimNum_ + 1U];
    for (int64_t k = 0; k < nonIndexedDimNum_ + 1U; k++){
        valueReshapeDims[k] = 0L;
    }

    if (isContinous_) {
        size_t j = 0;  // value
        bool noWriteIndexd = true;
        for (size_t i = 0; i < selfDimNum_; i++) {
            if (indexedSizes_[i] == 0) {
                valueReshapeDims[j] = selfDims_[i];
                j++;
            } else {
                if (noWriteIndexd) {
                    valueReshapeDims[j] = indexedDimSize_;
                    noWriteIndexd = false;
                    j++;
                }
            }
        }
    } else {
        valueReshapeDims[0] = indexedDimSize_;  // 索引始终在前；
        size_t j = 1;
        for (size_t i = 0; i < selfDimNum_; i++) {
            if (indexedSizes_[i] == 0) {
                valueReshapeDims[j] = selfDims_[i];
                j++;
            }
        }
    }

    // 计算selfStride和valueStride
    selfStride[selfDimNum_ - 1U] = 1;  // 最后一维设为1
    valueStride[nonIndexedDimNum_] = 1;  // 最后一维设为1
    for (int32_t i = static_cast<int32_t>(selfDimNum_) - 2; i >= 0; i--) {
        selfStride[i] = selfDims_[i + 1] * selfStride[i + 1];
    }
    for (int32_t i = static_cast<int32_t>(nonIndexedDimNum_) - 1; i >= 0; i--) {
        valueStride[i] = valueReshapeDims[i + 1] * valueStride[i + 1];
    }
}

void IndexPutWithSortV2Tiling::CalcNonIndexedStride(int64_t* selfStride, int64_t* valueStride)
{
    // 计算nonIdxedStride_
    for (int32_t i = static_cast<int32_t>(nonIndexedDimNum_) - 2; i >= 0; i--) {
        nonIdxedStride_[i] = nonIdxedStride_[i + 1] * nonIdxedDims_[i + 1];
    }

    // 计算value的indexedSizes
    int64_t indexedValueSizes[nonIndexedDimNum_ + 1U];
    for (int64_t k = 0; k < nonIndexedDimNum_ + 1U; k++){
        indexedValueSizes[k] = 0L;
    }

    if (isContinous_) {
        for (size_t i = 0; i < selfDimNum_; i++) {
            if (indexedSizes_[i] == 1) {
                indexedValueSizes[i] = 1;
                break;
            }
        }
    } else {
        indexedValueSizes[0] = 1;
    }

    // 计算nonIdxedSelfStride_和nonIdxedValueStride_
    size_t j = 0UL;
    for (size_t i = 0; i < selfDimNum_; i++) { // 注意之前要校验selfDimNum_和 indexedSizesDimNum一致
        if (indexedSizes_[i] == 0) {
            nonIdxedSelfStride_[j] = selfStride[i];
            j++;
        }
    }

    j = 0UL;
    for (size_t i = 0; i < static_cast<size_t>(nonIndexedDimNum_ + 1U); i++) { // 注意之前要校验selfDimNum_和 indexedSizesDimNum一致
        if (indexedValueSizes[i] == 0) {
            nonIdxedValueStride_[j] = valueStride[i];
            j++;
        }
    }
    // 计算idxedValueStride_，非连续场景就是nonIndexedDimSize，连续场景是索引轴后所有非索引轴的乘积
    if (isContinous_) {
        for (size_t i = 0; i < static_cast<size_t>(nonIndexedDimNum_ + 1U); i++) {
            if (indexedValueSizes[i] == 1) {
                idxedValueStride_ = valueStride[i];  // 相当于索引轴的stride
                break;
            }
        }
    } else {
        idxedValueStride_ = nonIndexedDimSize_;  // 可能不用区分是否continous？
    }
}

void IndexPutWithSortV2Tiling::CalcThreadNum() {
    if (indexedDimSize_ < aivCoreNum_ && nonIndexedDimSize_ > aivCoreNum_) {
        indexedBlockMode_ = false;
    } else {
        indexedBlockMode_ = true;
    }
    bool allIndexed = nonIndexedDimSize_ == 1;
    int32_t threadNumPerBlock = allIndexed ? THREAD_NUM_FULL : THREAD_NUM_HALF;

    if (indexedBlockMode_) {
        if (indexedDimSize_ >= aivCoreNum_ * threadNumPerBlock || allIndexed) {
            // 场景一：索引轴大于总线程数或者全索引，线程全分配索引轴，非索引轴内循环
            indexedThreadNum_ = threadNumPerBlock;
            nonIndexedThreadNum_ = 1;
        } else if (indexedDimSize_ >= aivCoreNum_) {
            // 场景二：索引轴小于总线程数但大于核数，分配部分线程给非索引轴
            indexedThreadNum_ = indexedDimSize_ / aivCoreNum_;
            nonIndexedThreadNum_ = threadNumPerBlock / indexedThreadNum_;
        } else {
            // 场景三：索引轴小于核数，非索引轴小于核数，分配线程给非索引轴
            indexedThreadNum_ = 1;
            nonIndexedThreadNum_ = nonIndexedDimSize_;
        }
    } else {
        if (nonIndexedDimSize_ >= aivCoreNum_ * threadNumPerBlock) {
            // 场景四：非索引轴大于总线程数，线程全分配非索引轴，索引轴内循环
            indexedThreadNum_ = 1;
            nonIndexedThreadNum_ = threadNumPerBlock;
        } else {
            // 场景五：非索引轴小于于总线程数但大于核数，分配部分线程给索引轴
            nonIndexedThreadNum_ = nonIndexedDimSize_ / aivCoreNum_;
            indexedThreadNum_ = threadNumPerBlock / nonIndexedThreadNum_;
        }
    }
}

ge::graphStatus IndexPutWithSortV2Tiling::DoOpTiling()
{
    OP_LOGI(context_->GetNodeName(), "IndexPutWithSortV2Tiling DoOpTiling");
    ge::graphStatus result = ge::GRAPH_SUCCESS;
    int64_t selfStride[selfDimNum_];
    for (size_t k = 0; k < selfDimNum_; k++){
        selfStride[k] = 0L;
    }

    int64_t valueStride[nonIndexedDimNum_ + 1U];
    for (int64_t k = 0; k < nonIndexedDimNum_ + 1U; k++){
        valueStride[k] = 0L;
    }
    CalcSelfAndValueStride(selfStride, valueStride);
    CalcNonIndexedStride(selfStride, valueStride);
    CalcThreadNum();
    if (indexedThreadNum_ <= 0 || nonIndexedThreadNum_ <= 0) {
        OP_LOGE(context_->GetNodeName(), "ThreadNum result less than zero.");
        return ge::GRAPH_FAILED;
    }
    SetTilingData();
    LogTilingResult();
    return result;
}

REGISTER_OPS_TILING_TEMPLATE(IndexPutWithSortV2, IndexPutWithSortV2Tiling, 100);

ge::graphStatus Tiling4IndexPutWithSortV2(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling for IndexPutWithSortV2 is running.");
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4IndexPutWithSortV2([[maybe_unused]] gert::TilingParseContext* context) 
{
    return ge::GRAPH_SUCCESS;
}

struct IndexPutWithSortV2CompileInfo {};
IMPL_OP_OPTILING(IndexPutWithSortV2).Tiling(Tiling4IndexPutWithSortV2).TilingParse<IndexPutWithSortV2CompileInfo>(TilingPrepare4IndexPutWithSortV2);

}  // namespace optiling