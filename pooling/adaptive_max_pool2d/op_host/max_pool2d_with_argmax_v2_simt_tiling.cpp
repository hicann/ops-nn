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
 * \file max_pool2d_with_argmax_v2_simt_tiling.cpp
 * \brief
 */

#include <cctype>
#include <algorithm>
#include "log/log.h"
#include "util/math_util.h"
#include "error_util.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "adaptive_max_pool2d_simt_tiling.h"
#include "op_common/op_host/util/platform_util.h"
#include "platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"


using namespace ge;

constexpr uint64_t CAL_KER_THRESHOLD = 10000;
constexpr int64_t N_IDX = 0;
constexpr int64_t C_IDX = 1;
constexpr int64_t H_IDX = 2;
constexpr int64_t W_IDX = 3;

namespace optiling{

static const gert::Shape g_vec_1_shape = {1};

static const gert::Shape& EnsureNotScalar(const gert::Shape &inShape) {
  if (inShape.IsScalar()) {
    return g_vec_1_shape;
  }
  return inShape;
}

bool AdaMaxPool2dTilingSIMT::IsCapable()
{
    return true;
}

ge::graphStatus AdaMaxPool2dTilingSIMT::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const AdaptiveMaxPool2dCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    coreNum_ = compileInfo->coreNum;
    OP_CHECK_IF(coreNum_ <= 0, OP_LOGE(context_, "GetPlatformInfo get corenum <= 0"), return ge::GRAPH_FAILED);
    sysWorkspaceSize_ = compileInfo->sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaMaxPool2dTilingSIMT::CheckPlatformAndGetShapes() {
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto inputX = context_->GetInputShape(FIRPOS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto inputShape = EnsureNotScalar(inputX->GetStorageShape());
    auto outX = context_->GetOutputShape(FIRPOS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outX);
    auto outShape = EnsureNotScalar(outX->GetStorageShape());
    auto indicesX = context_->GetOutputShape(SECPOS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, indicesX);
    if (inputShape.GetDimNum() != NCHW_DIMS) {
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                        "AdaptiveMaxPool2d: input shape dim = %zu, should be equal 4",
                                        inputShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        inputShape.GetDim(N_IDX) < 1 || inputShape.GetDim(C_IDX) < 1 ||
        inputShape.GetDim(H_IDX) < 1 || inputShape.GetDim(W_IDX) < 1,
        OP_LOGE(context_->GetNodeName(), "Invalid shape. Maybe empty tensor."), return ge::GRAPH_FAILED);

    inputData.inputShape =
        array<uint64_t, NCHW_DIMS>{uint64_t(inputShape.GetDim(N_IDX)), uint64_t(inputShape.GetDim(C_IDX)),
                                    uint64_t(inputShape.GetDim(H_IDX)), uint64_t(inputShape.GetDim(W_IDX))};
    inputData.outShape =
        array<uint64_t, NCHW_DIMS>{uint64_t(inputShape.GetDim(N_IDX)), uint64_t(inputShape.GetDim(C_IDX)),
                                    uint64_t(outShape.GetDim(H_IDX)), uint64_t(outShape.GetDim(W_IDX))};

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaMaxPool2dTilingSIMT::CheckDataTypeAndAttrs() {
    auto inputDesc = context_->GetInputDesc(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    dtype = inputDesc->GetDataType();
    if (dtype != ge::DataType::DT_BF16 && dtype != ge::DataType::DT_FLOAT16 && dtype != ge::DataType::DT_FLOAT) {
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), 
            "AdaptiveMaxPool2d: invalid dtype %s, should be BFloat16、Float16 or Float32",
            Ops::Base::ToString(dtype).c_str());
        return ge::GRAPH_FAILED;
    }

    auto indicesX = context_->GetOutputShape(SECPOS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, indicesX);
    auto indicesShape = EnsureNotScalar(indicesX->GetStorageShape());
    if (indicesShape.GetDimNum() != NCHW_DIMS) {
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                        "AdaptiveMaxPool2d: indices shape dim = %zu, should be 4",
                                        indicesShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    array<uint64_t, NCHW_DIMS> indicesArray{
        uint64_t(indicesShape.GetDim(N_IDX)),
        uint64_t(indicesShape.GetDim(C_IDX)),
        uint64_t(indicesShape.GetDim(H_IDX)),
        uint64_t(indicesShape.GetDim(W_IDX))
    };
    if (indicesArray != inputData.outShape) {
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                        "AdaptiveMaxPool2d: indices shape and values shape is different");
        return ge::GRAPH_FAILED;
    }

    auto attrPtr = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrPtr);
    auto outputSizePtr = attrPtr->GetAttrPointer<gert::ContinuousVector>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputSizePtr);
    OP_CHECK_IF(
        outputSizePtr->GetSize() != DOUB,
        OP_LOGE(context_->GetNodeName(), "the size of outputsize only support 2"),
        return ge::GRAPH_FAILED);
    const int64_t* outputSize = static_cast<const int64_t*>(outputSizePtr->GetData());
    OP_CHECK_IF(
        outputSize[0] <= 0 || outputSize[1] <= 0,
        OP_LOGE(context_->GetNodeName(), "the value of outputsize should > 0"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaMaxPool2dTilingSIMT::GetShapeAttrsInfo() {
    auto status = CheckPlatformAndGetShapes();
    if (status != ge::GRAPH_SUCCESS) return status;
    return CheckDataTypeAndAttrs();
}

ge::graphStatus AdaMaxPool2dTilingSIMT::DoOpTiling()
{
    tiling.set_N(inputData.inputShape[N_DIM_]);
    tiling.set_C(inputData.inputShape[C_DIM_]);
    tiling.set_Hi(inputData.inputShape[H_DIM_]);
    tiling.set_Wi(inputData.inputShape[W_DIM_]);
    tiling.set_Ho(inputData.outShape[H_DIM_]);
    tiling.set_Wo(inputData.outShape[W_DIM_]);
    tiling.set_kMaxSizeH(inputData.kernelHMax);
    tiling.set_kMaxSizeW(inputData.kernelWMax);
    tiling.set_coreNums(0);
    tiling.set_useCoreNum(0);
    tiling.set_totalIdx(0);
    tiling.set_blockFactor(0);
    tiling.set_blockTail(0);
    tiling.set_ncFactor(0);
    tiling.set_hoFactor(0);
    tiling.set_woFactor(0);
    tiling.set_ncOuter(0);
    tiling.set_hoOuter(0);
    tiling.set_woOuter(0);
    tiling.set_ncTail(0);
    tiling.set_hoTail(0);
    tiling.set_woTail(0);
    int64_t outputDataCount = tiling.get_N() * tiling.get_C() * tiling.get_Hi() * tiling.get_Wi();
    int64_t threads = std::min(outputDataCount, MAX_THREAD_NUM);
    int64_t blockNum = Ops::Base::CeilDiv(outputDataCount, threads);
    blockNum = std::min(blockNum, static_cast<int64_t>(coreNum_));
    context_->SetBlockDim(blockNum);
    context_->SetTilingKey(GetTilingKey());
    tiling.set_threadNums(threads);
    tiling.set_blockNums(blockNum);
    return ge::GRAPH_SUCCESS;
}

uint64_t AdaMaxPool2dTilingSIMT::GetTilingKey() const
{
    return 0;
}

ge::graphStatus AdaMaxPool2dTilingSIMT::GetWorkspaceSize()
{
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaMaxPool2dTilingSIMT::PostTiling()
{
    OP_CHECK_IF(context_->GetRawTilingData()->GetCapacity() < tiling.GetDataSize(),
                OP_LOGE(context_, "tiling data's[%zu] is larger than capacity[%zu].", tiling.GetDataSize(),
                        context_->GetRawTilingData()->GetCapacity()),
                return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void AdaMaxPool2dTilingSIMT::DumpTilingInfo()
{
    std::string str;
    str += " threadNums:" + std::to_string(tiling.get_threadNums());
    str += " blockNums:" + std::to_string(tiling.get_blockNums());
    str += " nDim:" + std::to_string(tiling.get_N());
    str += " cDim:" + std::to_string(tiling.get_C());
    str += " hInDim:" + std::to_string(tiling.get_Hi());
    str += " wInDim:" + std::to_string(tiling.get_Wi());
    str += " hOutDim:" + std::to_string(tiling.get_Ho());
    str += " wOutDim:" + std::to_string(tiling.get_Wo());
    str += " kMaxSizeH:" + std::to_string(tiling.get_kMaxSizeH());
    str += " kMaxSizeW:" + std::to_string(tiling.get_kMaxSizeW());
}
REGISTER_TILING_TEMPLATE("AdaptiveMaxPool2d", AdaMaxPool2dTilingSIMT, 0);
}  // namespace optiling