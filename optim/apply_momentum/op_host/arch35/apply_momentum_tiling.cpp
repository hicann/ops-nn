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
 * \file apply_momentum_tiling.cpp
 * \brief
 */
#include "apply_momentum_tiling.h"
#include <graph/utils/type_utils.h>
#include "error_util.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;
using namespace ApplyMomentumOp;

namespace optiling {
const size_t SYS_WORKSPACE = 16777216;  // 16M
const static std::map<int32_t, string> TENSOR_INDEX_LIST = {{1, "accum"}, {3, "grad"}};
const static std::map<int32_t, string> SCALAR_INDEX_LIST = {{2, "lr"}, {4, "momentum"}};

ge::graphStatus ApplyMomentumRegbaseTiling::CheckScalarShape(int32_t inputIdx) {
    auto inputShape = tilingContext_->GetInputShape(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape);
    auto storageShape = inputShape->GetStorageShape();
    OP_CHECK_IF((!storageShape.IsScalar() && storageShape.GetShapeSize() != 1),
                OP_LOGE(tilingContext_, "Check input %d failed.", inputIdx), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyMomentumRegbaseTiling::CheckSameShape(int32_t inputIdx, const gert::Shape& input0Shape) {
    auto inputShape = tilingContext_->GetInputShape(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape);
    auto curStorageShape = inputShape->GetStorageShape();
    if (curStorageShape != input0Shape) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyMomentumRegbaseTiling::CheckSameDtype(int32_t inputIdx, const ge::DataType& input0Dtype) {
    auto inputDesc = tilingContext_->GetInputDesc(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputDesc);
    auto curDtype = inputDesc->GetDataType();
    if (curDtype != input0Dtype) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyMomentumRegbaseTiling::CheckShapeAndType() {
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

ge::graphStatus ApplyMomentumRegbaseTiling::GetAttr() {
    auto attrs = tilingContext_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrs);
    const bool* useNesterovAttr = attrs->GetAttrPointer<bool>(0);
    useNesterov_ = useNesterovAttr != nullptr ? *useNesterovAttr : false;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyMomentumRegbaseTiling::DoElewiseTiling() {
    ElewiseBaseTiling eleBaseTiling(tilingContext_);
    auto varDesc = tilingContext_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, varDesc);
    ge::DataType varDType = varDesc->GetDataType();
    ge::graphStatus ret = ge::GRAPH_FAILED;
    if (useNesterov_) {
        OP_LOGI(tilingContext_->GetNodeName(), "Do elemwise base tiling with attr useNesterov true");
        useNesterovKey_ = 1;
        if (varDType == ge::DT_FLOAT) {
            ret = eleBaseTiling.DoTiling<ApplyMomentumOp::ApplyNesterovMomentumDag<float, float>::OpDag>(tiling_->elewiseTiling);
        } else if (varDType == ge::DT_FLOAT16) {
            ret = eleBaseTiling.DoTiling<ApplyMomentumOp::ApplyNesterovMomentumDag<half, float>::OpDag>(tiling_->elewiseTiling);
        } else if (varDType == ge::DT_BF16) {
            ret = eleBaseTiling.DoTiling<ApplyMomentumOp::ApplyNesterovMomentumDag<bfloat16_t, float>::OpDag>(tiling_->elewiseTiling);
        } else {
            OP_LOGE(tilingContext_, "input dtype is not support!");
            ret = ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGI(tilingContext_->GetNodeName(), "Do elemwise base tiling with attr useNesterov false");
        if (varDType == ge::DT_FLOAT) {
            ret = eleBaseTiling.DoTiling<ApplyMomentumOp::ApplyMomentumDag<float, float>::OpDag>(tiling_->elewiseTiling);
        } else if (varDType == ge::DT_FLOAT16) {
            ret = eleBaseTiling.DoTiling<ApplyMomentumOp::ApplyMomentumDag<half, float>::OpDag>(tiling_->elewiseTiling);
        } else if (varDType == ge::DT_BF16) {
            ret = eleBaseTiling.DoTiling<ApplyMomentumOp::ApplyMomentumDag<bfloat16_t, float>::OpDag>(tiling_->elewiseTiling);
        } else {
            OP_LOGE(tilingContext_, "input dtype is not support!");
            ret = ge::GRAPH_FAILED;
        }
    }
    return ret;
}

ge::graphStatus ApplyMomentumRegbaseTiling::SetTilingData() {
    OP_LOGD(tilingContext_->GetNodeName(), "Enter SetTilingData");
    size_t* currentWorkspace = tilingContext_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, currentWorkspace);
    currentWorkspace[0] = SYS_WORKSPACE;
    OP_LOGI(tilingContext_->GetNodeName(), "scheMode is %ld, useNesterov is %ld", tiling_->elewiseTiling.scheMode, useNesterovKey_);
    tilingKey_ = GET_TPL_TILING_KEY(tiling_->elewiseTiling.scheMode, useNesterovKey_);
    tilingContext_->SetTilingKey(tilingKey_);
    uint32_t blockDim = static_cast<uint32_t>(tiling_->elewiseTiling.blockNum);
    OP_CHECK_IF(blockDim <= 0, OP_LOGE(tilingContext_, "Get blockDim failed"),
                return ge::GRAPH_FAILED);
    tilingContext_->SetBlockDim(blockDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyMomentumRegbaseTiling::RunTiling() {
    if (tilingContext_ == nullptr) {
        OP_LOGE("ApplyMomentum", "Get nullptr while obtaining tilingContext_.");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(GetAttr() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext_, "Get attr failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeAndType() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext_, "Shape and dtype check failed."),
                return ge::GRAPH_FAILED);
    tiling_ = tilingContext_->GetTilingData<ApplyMomentumRegbaseTilingData>();
    OP_CHECK_IF((tiling_ == nullptr), OP_LOGE(tilingContext_, "Get ApplyMomentumRegbaseTiling from GE context failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(DoElewiseTiling() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext_, "elewiseBaseTiling failed"),
                return ge::GRAPH_FAILED);
    return SetTilingData();
}

ge::graphStatus Tiling4ApplyMomentum(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling4ApplyMomentum running begin");
    ApplyMomentumRegbaseTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepareForApplyMomentum(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForApplyMomentum running begin");
    return ge::GRAPH_SUCCESS;
}

struct ApplyMomentumCompileInfo {
};

IMPL_OP_OPTILING(ApplyMomentum)
    .Tiling(Tiling4ApplyMomentum)
    .TilingParse<ApplyMomentumCompileInfo>(TilingPrepareForApplyMomentum);

}  // namespace optiling