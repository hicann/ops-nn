/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mse_loss_tiling_arch35.cpp
 * \brief
 */

#include <cstdint>
#include <vector>
#include <string>

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "loss/mse_loss/op_kernel/arch35/mse_loss_dag.h"
#include "loss/mse_loss/op_kernel/arch35/mse_loss_tiling_key.h"
#include "mse_loss_tiling_arch35.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "atvoss/elewise/elewise_tiling.h"

namespace optiling {
using namespace Ops::Base;
static const int64_t ASCEND_WORKSPACE = static_cast<int64_t>(16 * 1024 * 1024);
static const uint32_t REDUCTION_MEAN = 2;
static const std::map<std::string, uint32_t> STR_2_INT = {{"none", 0}, {"sum", 1}, {"mean", 2}};
static const std::map<ge::DataType, uint32_t> DTYEP_2_INT_KEY{
    {ge::DT_FLOAT16, 10}, {ge::DT_FLOAT, 20}, {ge::DT_BF16, 30}};

ge::graphStatus MseLossTiling::CheckShape()
{
    auto predictStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, predictStorageShape);
    const gert::Shape& inputPShape = Ops::Base::EnsureNotScalar(predictStorageShape->GetStorageShape());
    auto labelStorageShape = tilingContext->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, labelStorageShape);
    const gert::Shape& inputLShape = Ops::Base::EnsureNotScalar(labelStorageShape->GetStorageShape());
    auto inputPDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputPDesc);
    auto inputLDesc = tilingContext->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputLDesc);
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    auto iter = DTYEP_2_INT_KEY.find(this->outputDtype);
    OP_CHECK_IF(
        iter == DTYEP_2_INT_KEY.end(),
        OP_LOGE(
            tilingContext->GetNodeName(),
            "output dtype is not support, should be one of {float32, float16, bfloat16}. "),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inputPDesc->GetDataType() != inputLDesc->GetDataType(),
        OP_LOGE(tilingContext->GetNodeName(), "input predict and label dtype not same"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inputPShape != inputLShape, OP_LOGE(tilingContext->GetNodeName(), "input predict and label shape not same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MseLossTiling::SetTilingData()
{
    OP_LOGD(tilingContext->GetNodeName(), "Enter SetTilingData");
    auto rawTilingData = tilingContext->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, rawTilingData);

    const uint64_t tilingKey = GET_TPL_TILING_KEY(
        key.ReduceTiling.patternID, key.ReduceTiling.loopARCount, key.ReduceTiling.loopInnerARCount, key.Reduction,
        key.Dtype);
    OP_LOGI(
        tilingContext->GetNodeName(),
        "patternID:%u, loopARCount:%u, loopInnerARCount:%u, Tiling Key is:%lu, Reduction is : %u, Dtype is %u",
        key.ReduceTiling.patternID, key.ReduceTiling.loopARCount, key.ReduceTiling.loopInnerARCount, tilingKey,
        key.Reduction, key.Dtype);
    if (static_cast<int32_t>(this->reduction) < 1) {
        size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
        currentWorkspace[0] = static_cast<size_t>(ASCEND_WORKSPACE);
        tilingContext->SetBlockDim(tiling->baseTiling.blockNum);
    }
    tilingContext->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MseLossTiling::TilingEle()
{
    ElewiseBaseTiling eleBaseTiling(tilingContext);
    key.Dtype = DTYEP_2_INT_KEY.at(this->outputDtype);
    if (this->outputDtype == ge::DT_FLOAT16 || this->outputDtype == ge::DT_BF16) {
        // fp16需要cast成fp32处理
        OP_CHECK_IF(
            eleBaseTiling.DoTiling<MseLoss::MseLossOp<half>::OpDag>(tiling->baseTiling) != ge::GRAPH_SUCCESS,
            OP_LOGE(tilingContext->GetNodeName(), "do tiling failed for fp16"), return ge::GRAPH_FAILED);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        OP_CHECK_IF(
            eleBaseTiling.DoTiling<MseLoss::MseLossOp<float>::OpDag>(tiling->baseTiling) != ge::GRAPH_SUCCESS,
            OP_LOGE(tilingContext->GetNodeName(), "do tiling failed for fp32"), return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(tilingContext->GetNodeName(), "current dtype not supported");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MseLossTiling::TilingReduce(const MseLossCompileInfo* compileInfo)
{
    ReduceOpInputParam opInput;
    OP_CHECK_IF(
        (ReduceOpTmpl::GetInputParam(tilingContext, opInput, 0) == ge::GRAPH_FAILED),
        OP_LOGE(tilingContext->GetNodeName(), "ReduceOp get x input failed"), return ge::GRAPH_FAILED);
    opInput.axes.resize(opInput.shape.size());
    for (size_t i = 0; i < opInput.shape.size(); i++) {
        opInput.axes[i] = i;
    }
    if (this->outputDtype == ge::DT_FLOAT16 || this->outputDtype == ge::DT_BF16) {
        if (static_cast<int32_t>(this->reduction) == 1) {
            OP_CHECK_IF(
                (Tiling4ReduceOp<MseLoss::MseLossSumDag<half, float>::OpDag>(
                     tilingContext, opInput, key.ReduceTiling, &compileInfo->opInfo, &tiling->reduceTiling) == ge::GRAPH_FAILED),
                OP_LOGE(tilingContext->GetNodeName(), "MseLoss Tiling failed"), return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(
                (Tiling4ReduceOp<MseLoss::MseLossMeanDag<half, float>::OpDag>(
                     tilingContext, opInput, key.ReduceTiling, &compileInfo->opInfo, &tiling->reduceTiling) == ge::GRAPH_FAILED),
                OP_LOGE(tilingContext->GetNodeName(), "MseLoss Tiling failed"), return ge::GRAPH_FAILED);
        }
    } else {
        if (static_cast<int32_t>(this->reduction) == 1) {
            OP_CHECK_IF(
                (Tiling4ReduceOp<MseLoss::MseLossSumDag<float, float>::OpDag>(
                     tilingContext, opInput, key.ReduceTiling, &compileInfo->opInfo, &tiling->reduceTiling) == ge::GRAPH_FAILED),
                OP_LOGE(tilingContext->GetNodeName(), "MseLoss Tiling failed"), return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(
                (Tiling4ReduceOp<MseLoss::MseLossMeanDag<float, float>::OpDag>(
                     tilingContext, opInput, key.ReduceTiling, &compileInfo->opInfo, &tiling->reduceTiling) == ge::GRAPH_FAILED),
                OP_LOGE(tilingContext->GetNodeName(), "MseLoss Tiling failed"), return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MseLossTiling::RunTiling(const MseLossCompileInfo* compileInfo)
{
    OP_CHECK_IF(
        CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"), return ge::GRAPH_FAILED);
    auto attrs = tilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, attrs);
    const std::string reductionStr = std::string(attrs->GetAttrPointer<char>(0));
    auto iter = STR_2_INT.find(reductionStr);
    OP_CHECK_IF(
        iter == STR_2_INT.end(),
        OP_LOGE(
            tilingContext->GetNodeName(), "reduction %s is not support, should be one of {mean, sum, none}.",
            reductionStr.c_str()),
        return ge::GRAPH_FAILED);
    this->reduction = iter->second;
    key.Reduction = this->reduction;
    tiling = tilingContext->GetTilingData<MseLossTilingData>();
    if (reductionStr == "none") {
        OP_CHECK_IF(
            TilingEle() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext->GetNodeName(), "do tiling failed for elewise"),
            return ge::GRAPH_FAILED);
    } else if (reductionStr == "mean" || reductionStr == "sum") {
        OP_CHECK_IF(
            TilingReduce(compileInfo) != ge::GRAPH_SUCCESS,
            OP_LOGE(tilingContext->GetNodeName(), "do tiling failed for reduce"), return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(tilingContext->GetNodeName(), "reduction %s is not supported", reductionStr.c_str());
        return ge::GRAPH_FAILED;
    }
    return SetTilingData();
}
} // namespace optiling