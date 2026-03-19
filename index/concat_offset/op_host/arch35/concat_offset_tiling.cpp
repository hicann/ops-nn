/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file concat_offset_tiling.cpp
 * \brief
 */

#include "concat_offset_tiling.h"

namespace optiling
{

const static int32_t INPUT_CONCAT_DIM_INDEX = 0;
const static int32_t INPUT_X_INDEX = 1;
const static int32_t ATTR_N_INDEX = 0;
const static uint32_t LEAST_INPUT_NUM = 2;
constexpr int64_t DIM1 = 1;
constexpr int64_t DIM8 = 8;
constexpr int64_t USE_ONE_CORE = 1;
const static uint64_t SIMT_TILING_KEY = 1000;
constexpr int32_t LOCAL_MEMORY_SIZE = 8192;
constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 16777216;  // 16M

static const std::set<ge::DataType> TENSOR_SUPPORTED_DTYPE = {ge::DT_INT32};

inline static bool IsSupportDtype(const std::set<ge::DataType> &supportDtype, const ge::DataType dtype)
{
  return (supportDtype.count(dtype) != 0);
}

void ConcatOffsetTiling::Reset() {
  opName_ = nullptr;
}

ge::graphStatus ConcatOffsetTiling::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const ConcatOffsetCompileParams*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    totalCoreNum_ = compileInfo->core_num;
    ubSize_ = compileInfo->ubSize;
    OP_TILING_CHECK(
        (totalCoreNum_ <= 0 || ubSize_ <= 0),
        OP_LOGE(
            context_->GetNodeName(), "ConcatOffset GetCompileInfo Failed, core_num:%ld, ubSize:%ld.", totalCoreNum_, ubSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// x: tensor list
ge::graphStatus ConcatOffsetTiling::GetXInfoAndCheck() {
    // check shape
    auto x0TensorShapePtr = context_->GetDynamicInputShape(INPUT_X_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x0TensorShapePtr);
    auto x0TensorShape = x0TensorShapePtr->GetStorageShape();
    // xi must be 1D
    size_t x0DimNum = x0TensorShape.GetDimNum();
    if (x0DimNum != 1U) {
        OP_LOGE(context_->GetNodeName(), "The first tensor in input x must be 1D");
        return ge::GRAPH_FAILED;
    }
    // xi shape size must be 1~8
    perTensorShapeSize_ = x0TensorShape.GetDim(0);
    if (perTensorShapeSize_ < DIM1 || perTensorShapeSize_ > DIM8) {
        OP_LOGE(context_->GetNodeName(), "The shape size of the first tensor in input x must be 1~8");
        return ge::GRAPH_FAILED;
    }
    // check dtype
    auto x0Desc = context_->GetDynamicInputDesc(INPUT_X_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x0Desc);
    auto x0Dtype = x0Desc->GetDataType();
    OP_TILING_CHECK(!IsSupportDtype(TENSOR_SUPPORTED_DTYPE, x0Dtype), OP_LOGE(context_->GetNodeName(),
    "The tensor dtype only support int32 currently, please check."), return ge::GRAPH_FAILED);
    
    // all input tensors dtype and dtype should be equal
    for (int64_t i = 1; i < sizeN_; i++) {
        auto xiTensorShapePtr = context_->GetDynamicInputShape(INPUT_X_INDEX, i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, xiTensorShapePtr);
        auto xiTensorShape = xiTensorShapePtr->GetStorageShape();
        OP_TILING_CHECK(
            xiTensorShape != x0TensorShape,
            OP_LOGE(context_->GetNodeName(), "all input x tensor shapes should be equal"),
            return ge::GRAPH_FAILED);
        
        auto xiDesc = context_->GetDynamicInputDesc(INPUT_X_INDEX, i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, xiDesc);
        auto xiDtype = xiDesc->GetDataType();
        OP_TILING_CHECK(
            x0Dtype != xiDtype, OP_LOGE(context_->GetNodeName(), "all input tensors dtype should be equal"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}


inline ge::graphStatus ConcatOffsetTiling::GetConcatDimInfoAndCheck() 
{
    auto concatDimTensor = context_->GetRequiredInputTensor(INPUT_CONCAT_DIM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, concatDimTensor);
    const int32_t *concatDimValPtr = concatDimTensor->GetData<int32_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, concatDimValPtr);
    concatDim_ = concatDimValPtr[0];
  
    // concat_dim is negative
    if (concatDim_ < 0) {
        concatDim_ = concatDim_ + perTensorShapeSize_;
    }
    // concat_dim out of bound
    if (concatDim_ < 0 || concatDim_ >= perTensorShapeSize_) {
        OP_LOGE(context_->GetNodeName(), "concat_dim (%ld) must be in range rank(xi) (%ld).", concatDim_, perTensorShapeSize_);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus ConcatOffsetTiling::GetAttrInfoAndCheck() {
    // get input num
    auto computeNodeInfo = context_->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, computeNodeInfo);
    auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, anchorInstanceInfo);
    uint32_t inputNum = anchorInstanceInfo->GetInstanceNum();
    OP_TILING_CHECK(
        inputNum < LEAST_INPUT_NUM, OP_LOGE(context_->GetNodeName(), "The number of input tensors must be greater than or equal to 2."),
        return ge::GRAPH_FAILED);

    // get attr N
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t* NPtr = attrs->GetAttrPointer<int64_t>(ATTR_N_INDEX);
    sizeN_ = *NPtr;
    OP_TILING_CHECK(
        sizeN_ != static_cast<int64_t>(inputNum),
        OP_LOGE(
            context_->GetNodeName(), "attr N:%ld should be same as input x tensor num:%ld", 
            sizeN_, static_cast<int64_t>(inputNum)),
        return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConcatOffsetTiling::GetShapeAttrsInfo() {
  opName_ = context_->GetNodeName();
  OP_LOGD(opName_, "ConcatOffsetTiling GetShapeAttrsInfo start running.");
  OP_TILING_CHECK(GetAttrInfoAndCheck() != ge::GRAPH_SUCCESS,
                  OP_LOGE(opName_, "input attr check failed."),
                  return ge::GRAPH_FAILED);
  OP_TILING_CHECK(GetXInfoAndCheck() != ge::GRAPH_SUCCESS,
                  OP_LOGE(opName_, "input x check failed."),
                  return ge::GRAPH_FAILED);
  OP_TILING_CHECK(GetConcatDimInfoAndCheck() != ge::GRAPH_SUCCESS,
                  OP_LOGE(opName_, "input concat_dim check failed."),
                  return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}


ge::graphStatus ConcatOffsetTiling::DoOpTiling() 
{
    needCalNum_ = sizeN_ * perTensorShapeSize_;
    if (needCalNum_ <= threadNum_) {
        threadNum_ = needCalNum_;
    }
    ConcatOffsetTilingData *tilingData = context_->GetTilingData<ConcatOffsetTilingData>();
    tilingData->threadNum = threadNum_;
    tilingData->concatDim = concatDim_;
    tilingData->perTensorShapeSize = perTensorShapeSize_;
    tilingData->needCalNum = needCalNum_;
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus ConcatOffsetTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t ConcatOffsetTiling::GetTilingKey() const {
    return SIMT_TILING_KEY;
}

ge::graphStatus ConcatOffsetTiling::GetWorkspaceSize()
{
    workspaceSize_ = ASCENDC_TOOLS_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConcatOffsetTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "ConcatOffsetTiling PostTiling start running.");

    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;

    context_->SetBlockDim(USE_ONE_CORE);
    
    context_->SetLocalMemorySize(static_cast<uint32_t>(LOCAL_MEMORY_SIZE));
    return ge::GRAPH_SUCCESS;
}

void ConcatOffsetTiling::DumpTilingInfo()
{
    std::ostringstream info;
    ConcatOffsetTilingData *tilingData = context_->GetTilingData<ConcatOffsetTilingData>();
    info << "threadNum: " << tilingData->threadNum << ", ";
    info << "concatDim: " << tilingData->concatDim << ", ";
    info << "perTensorShapeSize: " << tilingData->perTensorShapeSize << ", ";
    info << "needCalNum: " << tilingData->needCalNum;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

ge::graphStatus ConcatOffsetTilingForAscendC(gert::TilingContext* context)
{
    ConcatOffsetTiling tiling(context);
    return tiling.DoTiling();
}

}  // namespace optiling