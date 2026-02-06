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
 * \file apply_ftrl_tiling.cpp
 * \brief
 */
#include "apply_ftrl_tiling.h"

#include <graph/utils/type_utils.h>

#include <iostream>

#include "error_util.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;
using namespace ApplyFtrlOp;
using namespace ApplyFtrlOpTiling;

namespace optiling {
const size_t SYS_WORKSPACE = 16777216;  // 16M
constexpr static int32_t SCALAR_INPUT_LR_INDEX = 4;
constexpr static int32_t INPUT_NUM = 8;
const static std::map<int32_t, string> TENSOR_INDEX_LIST = {{1, "accum"}, {2, "linear"}, {3, "grad"}};
const static std::map<int32_t, string> SCALAR_INDEX_LIST = {{4, "lr"}, {5, "l1"}, {6, "l2"}, {7, "lr_power"}};

ge::graphStatus ApplyFtrlRegbaseTiling::CheckScalarShape(int32_t inputIdx) {
    auto inputShape = tilingContext_->GetInputShape(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape);
    auto storageShape = inputShape->GetStorageShape();
    OP_CHECK_IF((!storageShape.IsScalar() && storageShape.GetShapeSize() != 1),
                OP_LOGE(tilingContext_, "Check input %d failed.", inputIdx), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyFtrlRegbaseTiling::CheckSameShape(int32_t inputIdx, const gert::Shape& input0Shape) {
    auto inputShape = tilingContext_->GetInputShape(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape);
    auto curStorageShape = inputShape->GetStorageShape();
    if (curStorageShape != input0Shape) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyFtrlRegbaseTiling::CheckSameDtype(int32_t inputIdx, const ge::DataType& input0Dtype) {
    auto inputDesc = tilingContext_->GetInputDesc(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputDesc);
    auto curDtype = inputDesc->GetDataType();
    if (curDtype != input0Dtype) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyFtrlRegbaseTiling::CheckShapeAndType() {
    auto inputShape = tilingContext_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape);
    auto inputStorageShape = inputShape->GetStorageShape();

    auto inputVarDesc = tilingContext_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputVarDesc);
    auto inputDtype = inputVarDesc->GetDataType();
    // check scalar input
    for (const auto& pair : SCALAR_INDEX_LIST) {
        OP_CHECK_IF(
            CheckScalarShape(pair.first) != ge::GRAPH_SUCCESS,
            OP_LOGE(tilingContext_, "the shape of input %s must be rank 0.", pair.second.c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            CheckSameDtype(pair.first, inputDtype) != ge::GRAPH_SUCCESS,
            OP_LOGE(tilingContext_, "the dtype of input %s is different from that of input var.", pair.second.c_str()),
            return ge::GRAPH_FAILED);
    }
    // check tensor input
    for (const auto& pair : TENSOR_INDEX_LIST) {
        OP_CHECK_IF(
            CheckSameShape(pair.first, inputStorageShape) != ge::GRAPH_SUCCESS,
            OP_LOGE(tilingContext_, "the shape of input %s is different from that of input var.", pair.second.c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            CheckSameDtype(pair.first, inputDtype) != ge::GRAPH_SUCCESS,
            OP_LOGE(tilingContext_, "the dtype of input %s is different from that of input var.", pair.second.c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyFtrlRegbaseTiling::DoElewiseTiling() {
    ElewiseBaseTiling eleBaseTiling(tilingContext_);
    auto varDesc = tilingContext_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, varDesc);
    ge::DataType varDType = varDesc->GetDataType();
    ge::graphStatus ret = ge::GRAPH_FAILED;
    if (varDType == ge::DT_FLOAT) {
        ret = eleBaseTiling.DoTiling<ApplyFtrlOp::ApplyFtrlDag<float, float>::OpDag>(tiling_->elewiseTiling);
    } else if (varDType == ge::DT_FLOAT16) {
        ret = eleBaseTiling.DoTiling<ApplyFtrlOp::ApplyFtrlDag<half, float>::OpDag>(tiling_->elewiseTiling);
    } else if (varDType == ge::DT_BF16) {
        ret = eleBaseTiling.DoTiling<ApplyFtrlOp::ApplyFtrlDag<bfloat16_t, float>::OpDag>(tiling_->elewiseTiling);
    } else {
        OP_LOGE(tilingContext_, "input dtype is not support!");
        ret = ge::GRAPH_FAILED;
    }
    return ret;
}

ge::graphStatus ApplyFtrlRegbaseTiling::SetTilingData() {
    OP_LOGD(tilingContext_->GetNodeName(), "Enter SetTilingData");
    size_t* currentWorkspace = tilingContext_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, currentWorkspace);
    currentWorkspace[0] = SYS_WORKSPACE;
    tilingKey = GET_TPL_TILING_KEY(tiling_->elewiseTiling.scheMode);
    tilingContext_->SetTilingKey(tilingKey);
    uint32_t numBlocks = static_cast<uint32_t>(tiling_->elewiseTiling.blockNum);
    OP_CHECK_IF(numBlocks <= 0, OP_LOGE(tilingContext_, "Get numBlocks failed"),
                return ge::GRAPH_FAILED);
    tilingContext_->SetBlockDim(numBlocks);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyFtrlRegbaseTiling::RunTiling() {
    if (tilingContext_ == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(CheckShapeAndType() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext_, "Shape and dtype check failed."),
                return ge::GRAPH_FAILED);

    tiling_ = tilingContext_->GetTilingData<ApplyFtrlRegbaseTilingData>();
    OP_CHECK_IF((tiling_ == nullptr), OP_LOGE(tilingContext_, "Get ApplyFtrlRegbaseTiling from GE context failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(DoElewiseTiling() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext_, "elewiseBaseTiling failed"),
                return ge::GRAPH_FAILED);
    return SetTilingData();
}

ge::graphStatus Tiling4ApplyFtrl(gert::TilingContext* context)
{
    auto nodeName = context;
    OP_LOGD(nodeName, "Tiling4ApplyFtrl running begin");
    ApplyFtrlRegbaseTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepareForApplyFtrl(gert::TilingParseContext* context)
{
    auto nodeName = context;
    OP_LOGD(nodeName, "TilingPrepareForApplyFtrl running begin");
    return ge::GRAPH_SUCCESS;
}

struct ApplyFtrlCompileInfo {
};

IMPL_OP_OPTILING(ApplyFtrl)
    .Tiling(Tiling4ApplyFtrl)
    .TilingParse<ApplyFtrlCompileInfo>(TilingPrepareForApplyFtrl);

}  // namespace optiling