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
 * \file binary_cross_entropy_tiling_arch35.cpp
 * \brief binary_cross_entropy_tiling source file
 */
#include "atvoss/broadcast/broadcast_tiling.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "atvoss/reduce/reduce_util.h"
#include "binary_cross_entropy_tiling_arch35.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "op_util.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "platform/platform_info.h"
#include "log/log.h"

using namespace Ops::Base;

namespace optiling
{
using namespace BinaryCrossEntropy;
const size_t ASCEND_WORKSPACE = 16777216;  // 16MB

static const size_t INPUT_X_INDEX = 0;
static const size_t INPUT_Y_INDEX = 1;
static const size_t INPUT_WEIGHT_INDEX = 2;
static const size_t OUTPUT_INDEX = 0;
static const size_t REDUCTION_INDEX = 0;
static const map<std::string, uint32_t> STR_2_INT = {{"none", 0}, {"sum", 1}, {"mean", 2}};
static const map<ge::DataType, uint32_t> DTYEP_2_INT_KEY{{ge::DT_FLOAT16, 10}, {ge::DT_FLOAT, 20}, {ge::DT_BF16, 30}};

ge::graphStatus BinaryCrossEntropyTiling::CalcInputDtype()
{
    auto inputXDesc = tilingContext->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputXDesc);
    this->inputXDtype = inputXDesc->GetDataType();
    OP_CHECK_IF(
        this->inputXDtype != ge::DT_FLOAT16 && this->inputXDtype != ge::DT_BF16 && this->inputXDtype != ge::DT_FLOAT,
        OP_LOGE(tilingContext->GetNodeName(), "input X dtype not support"),
        return ge::GRAPH_FAILED);
    auto inputYDesc = tilingContext->GetInputDesc(INPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputYDesc);
    this->inputYDtype = inputYDesc->GetDataType();
    OP_CHECK_IF(
        this->inputYDtype != ge::DT_FLOAT16 && this->inputYDtype != ge::DT_BF16 && this->inputYDtype != ge::DT_FLOAT,
        OP_LOGE(tilingContext->GetNodeName(), "input Y dtype not support"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BinaryCrossEntropyTiling::CalcOutputDtype()
{
    auto outputDesc = tilingContext->GetOutputDesc(OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != ge::DT_FLOAT16 && this->outputDtype != ge::DT_BF16 && this->outputDtype != ge::DT_FLOAT,
        OP_LOGE(tilingContext->GetNodeName(), "output dtype not support"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(this->outputDtype != this->inputXDtype,
                    OP_LOGE(tilingContext->GetNodeName(), "output y dtype not same as inputs"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BinaryCrossEntropyTiling::CheckInputShape()
{
    auto yStorageShape = tilingContext->GetInputShape(INPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, yStorageShape);
    const gert::Shape& inputYShape = Ops::Base::EnsureNotScalar(yStorageShape->GetStorageShape());

    auto xStorageShape = tilingContext->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, xStorageShape);
    const gert::Shape& inputXShape = Ops::Base::EnsureNotScalar(xStorageShape->GetStorageShape());

    OP_CHECK_IF(
        inputYShape != inputXShape,
        OP_LOGE(tilingContext->GetNodeName(), "the shape of X is different from that of Y"),
        return ge::GRAPH_FAILED);
    auto weightStorageShape = tilingContext->GetOptionalInputShape(INPUT_WEIGHT_INDEX);
    if (weightStorageShape != nullptr) {
        const gert::Shape& inputWeightShape = Ops::Base::EnsureNotScalar(weightStorageShape->GetStorageShape());
        OP_CHECK_IF(inputWeightShape != inputXShape,
                        OP_LOGE(tilingContext->GetNodeName(),
                                                        "the shape of Weight is different from that of X"),
                        return ge::GRAPH_FAILED);
        bceTilingKey.hasWeight = static_cast<uint32_t>(1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BinaryCrossEntropyTiling::DoElewiseTiling()
{
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    bceTilingKey.dType = DTYEP_2_INT_KEY.at(this->outputDtype);
    ge::graphStatus eleBaseTilingResult = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16 || this->outputDtype == ge::DT_BF16) {
        if (static_cast<int32_t>(bceTilingKey.hasWeight) == 1) {
            // fp16 or bf16 dtype and optional input weight is not null
            eleBaseTilingResult = elewiseBaseTiling.DoTiling<BCEHasWeightElewise<half>::OpDag>(tiling->eleBaseTiling);
        } else {
            // fp16 or bf16 dtype and optional input weight is null
            eleBaseTilingResult = elewiseBaseTiling.DoTiling<BCEElewise<half>::OpDag>(tiling->eleBaseTiling);
        }
    } else if (this->outputDtype == ge::DT_FLOAT) {
        if (static_cast<int32_t>(bceTilingKey.hasWeight) == 1) {
            // fp32 dtype and optional input weight is not null
            eleBaseTilingResult = elewiseBaseTiling.DoTiling<BCEHasWeightElewise<float>::OpDag>(tiling->eleBaseTiling);
        } else {
            // fp32 dtype and optional input weight is null
            eleBaseTilingResult = elewiseBaseTiling.DoTiling<BCEElewise<float>::OpDag>(tiling->eleBaseTiling);
        }
    } else {
        OP_LOGE(tilingContext->GetNodeName(), "output dtype not support");
        return ge::GRAPH_FAILED;
    }
    return eleBaseTilingResult;
}

ge::graphStatus BinaryCrossEntropyTiling::RunFp16ReduceTiling(ReduceOpInputParam& opInput,
                                                              const BinaryCrossEntropyCompileInfo* compileInfo)
{
    bool hasWeight = static_cast<int32_t>(bceTilingKey.hasWeight) == 1;
    if (this->isReductionSum && !hasWeight) {
        // reducesum and optional input weight is null
        OP_CHECK_IF(
            (Tiling4ReduceOp<BCESumDag<half, float>::OpDag>(tilingContext, opInput, bceTilingKey.reduceTiling,
                                                            &compileInfo->opInfo, &tiling->reduceTiling) == ge::GRAPH_FAILED),
            OP_LOGE(tilingContext->GetNodeName(), "BinaryCrossEntropy tiling failed"),
            return ge::GRAPH_FAILED);
    } else if (this->isReductionSum && hasWeight) {
        // reducesum and optional input weight is not null
        OP_CHECK_IF(
            (Tiling4ReduceOp<BCEHasWeightSumDag<half, float>::OpDag>(tilingContext, opInput, bceTilingKey.reduceTiling,
                                                                     &compileInfo->opInfo,
                                                                     &tiling->reduceTiling) == ge::GRAPH_FAILED),
            OP_LOGE(tilingContext->GetNodeName(), "BinaryCrossEntropy tiling failed"),
            return ge::GRAPH_FAILED);
    } else if (this->isReductionMean && !hasWeight) {
        // reducemean and optional input weight is null
        OP_CHECK_IF(
            (Tiling4ReduceOp<BCEMeanDag<half, float>::OpDag>(tilingContext, opInput, bceTilingKey.reduceTiling,
                                                             &compileInfo->opInfo, &tiling->reduceTiling) == ge::GRAPH_FAILED),
            OP_LOGE(tilingContext->GetNodeName(), "BinaryCrossEntropy tiling failed"),
            return ge::GRAPH_FAILED);
    } else if (this->isReductionMean && hasWeight) {
        // reducemean and optional input weight is not null
        OP_CHECK_IF(
            (Tiling4ReduceOp<BCEHasWeightMeanDag<half, float>::OpDag>(tilingContext, opInput, bceTilingKey.reduceTiling,
                                                                      &compileInfo->opInfo,
                                                                      &tiling->reduceTiling) == ge::GRAPH_FAILED),
            OP_LOGE(tilingContext->GetNodeName(), "BinaryCrossEntropy tiling failed"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BinaryCrossEntropyTiling::RunFp32ReduceTiling(ReduceOpInputParam& opInput,
                                                              const BinaryCrossEntropyCompileInfo* compileInfo)
{
    bool hasWeight = static_cast<int32_t>(bceTilingKey.hasWeight) == 1;
    if (this->isReductionSum && !hasWeight) {
        OP_CHECK_IF(
            // reducesum and optional input weight is null
            (Tiling4ReduceOp<BCESumDag<float, float>::OpDag>(tilingContext, opInput, bceTilingKey.reduceTiling,
                                                             &compileInfo->opInfo, &tiling->reduceTiling) == ge::GRAPH_FAILED),
            OP_LOGE(tilingContext->GetNodeName(), "BinaryCrossEntropy tiling failed"),
            return ge::GRAPH_FAILED);
    } else if (this->isReductionSum && hasWeight) {
        OP_CHECK_IF(
            // reducesum and optional input weight is not null
            (Tiling4ReduceOp<BCEHasWeightSumDag<float, float>::OpDag>(tilingContext, opInput, bceTilingKey.reduceTiling,
                                                                      &compileInfo->opInfo,
                                                                      &tiling->reduceTiling) == ge::GRAPH_FAILED),
            OP_LOGE(tilingContext->GetNodeName(), "BinaryCrossEntropy tiling failed"),
            return ge::GRAPH_FAILED);
    } else if (this->isReductionMean && !hasWeight) {
        OP_CHECK_IF(
            // reducemean and optional input weight is null
            (Tiling4ReduceOp<BCEMeanDag<float, float>::OpDag>(tilingContext, opInput, bceTilingKey.reduceTiling,
                                                              &compileInfo->opInfo, &tiling->reduceTiling) == ge::GRAPH_FAILED),
            OP_LOGE(tilingContext->GetNodeName(), "BinaryCrossEntropy tiling failed"),
            return ge::GRAPH_FAILED);
    } else if (this->isReductionMean && hasWeight) {
        OP_CHECK_IF(
            // reducemean and optional input weight is not null
            (Tiling4ReduceOp<BCEHasWeightMeanDag<float, float>::OpDag>(tilingContext, opInput,
                                                                       bceTilingKey.reduceTiling, &compileInfo->opInfo,
                                                                       &tiling->reduceTiling) == ge::GRAPH_FAILED),
            OP_LOGE(tilingContext->GetNodeName(), "BinaryCrossEntropy tiling failed"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BinaryCrossEntropyTiling::DoReduceTiling(const BinaryCrossEntropyCompileInfo* compileInfo)
{
    ReduceOpInputParam opInput;
    OP_CHECK_IF((ReduceOpTmpl::GetInputParam(tilingContext, opInput, 0) == ge::GRAPH_FAILED),
                    OP_LOGE(tilingContext->GetNodeName(), "ReduceOp get x input failed"),
                    return ge::GRAPH_FAILED);
    opInput.axes.resize(opInput.shape.size());
    for (size_t i = 0; i < opInput.shape.size(); i++) {
        opInput.axes[i] = i;
    }
    if (this->outputDtype == ge::DT_FLOAT16 || this->outputDtype == ge::DT_BF16) {
        OP_CHECK_IF(RunFp16ReduceTiling(opInput, compileInfo) == ge::GRAPH_FAILED,
                        OP_LOGE(tilingContext->GetNodeName(), "get input dtype failed"),
                        return ge::GRAPH_FAILED);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        OP_CHECK_IF(RunFp32ReduceTiling(opInput, compileInfo) == ge::GRAPH_FAILED,
                        OP_LOGE(tilingContext->GetNodeName(), "get input dtype failed"),
                        return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(tilingContext->GetNodeName(), "output dtype not support");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BinaryCrossEntropyTiling::SetTilingData()
{
    OP_LOGD(tilingContext->GetNodeName(), "Enter SetTilingData");
    auto rawTilingData = tilingContext->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, rawTilingData);

    const uint64_t tilingKey = GET_TPL_TILING_KEY(
        bceTilingKey.reduceTiling.patternID, bceTilingKey.reduceTiling.loopARCount,
        bceTilingKey.reduceTiling.loopInnerARCount, bceTilingKey.reduction, bceTilingKey.hasWeight, bceTilingKey.dType);
    OP_LOGI(tilingContext->GetNodeName(),
            "patternID:%u, loopARCount:%u, loopInnerARCount:%u, Tiling Key is:%lu, reduction is : %u, hasWeight is : "
            "%u, Dtype is %u",
            bceTilingKey.reduceTiling.patternID, bceTilingKey.reduceTiling.loopARCount,
            bceTilingKey.reduceTiling.loopInnerARCount, tilingKey, bceTilingKey.reduction, bceTilingKey.hasWeight,
            bceTilingKey.dType);
    if (this->isReductionNone) {
        size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
        currentWorkspace[0] = static_cast<size_t>(ASCEND_WORKSPACE);
        tilingContext->SetBlockDim(tiling->eleBaseTiling.blockNum);
    }
    tilingContext->SetTilingKey(tilingKey);
    OP_LOGD(tilingContext->GetNodeName(), "End SetTilingData");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BinaryCrossEntropyTiling::RunTiling(const BinaryCrossEntropyCompileInfo* compileInfo)
{
    if (tilingContext == nullptr) {
        OP_LOGE("BinaryCrossEntropy", "Tiling context is null");
    }
    OP_LOGD(tilingContext->GetNodeName(), "BinaryCrossEntropyTiling RunTiling enter.");
    // check input dtype
    OP_CHECK_IF(CalcInputDtype() == ge::GRAPH_FAILED,
                    OP_LOGE(tilingContext->GetNodeName(), "get input dtype failed"),
                    return ge::GRAPH_FAILED);
    // check output dtype
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED,
                    OP_LOGE(tilingContext->GetNodeName(), "get output dtype failed"),
                    return ge::GRAPH_FAILED);
    // check input shape
    OP_CHECK_IF(CheckInputShape() == ge::GRAPH_FAILED,
                    OP_LOGE(tilingContext->GetNodeName(), "get output dtype failed"),
                    return ge::GRAPH_FAILED);

    auto attrs = tilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, attrs);

    this->reductionStr = attrs->GetAttrPointer<char>(REDUCTION_INDEX);
    auto iter = STR_2_INT.find(this->reductionStr);
    OP_CHECK_IF(
        iter == STR_2_INT.end(),
        OP_LOGE(tilingContext->GetNodeName(), "reduction is not in none, mean or sum."),
        return ge::GRAPH_FAILED);
    this->isReductionNone = strcmp(this->reductionStr, "none") == 0;
    this->isReductionMean = strcmp(this->reductionStr, "mean") == 0;
    this->isReductionSum = strcmp(this->reductionStr, "sum") == 0;
    bceTilingKey.reduction = iter->second;
    tiling = tilingContext->GetTilingData<BinaryCrossEntropyTilingData>();
    if (this->isReductionNone) {
        OP_LOGD(tilingContext->GetNodeName(), "use elewise pattern");
        OP_CHECK_IF(DoElewiseTiling() == ge::GRAPH_FAILED,
                        OP_LOGE(tilingContext->GetNodeName(), "elewiseBaseTiling failed"),
                        return ge::GRAPH_FAILED);
    } else if (this->isReductionMean || this->isReductionSum) {
        OP_LOGD(tilingContext->GetNodeName(), "use reduce pattern");
        OP_CHECK_IF(DoReduceTiling(compileInfo) == ge::GRAPH_FAILED,
                        OP_LOGE(tilingContext->GetNodeName(), "reduceTiling failed"),
                        return ge::GRAPH_FAILED);
    }
    return SetTilingData();
}

ge::graphStatus Tiling4BinaryCrossEntropy(gert::TilingContext* context)
{
    auto compileInfo = static_cast<const BinaryCrossEntropyCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    BinaryCrossEntropyTiling baseOpTiling(context);
    return baseOpTiling.RunTiling(compileInfo);
}

ge::graphStatus TilingPrepare4BinaryCrossEntropy([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the BinaryCrossEntropy op.
IMPL_OP_OPTILING(BinaryCrossEntropy)
    .Tiling(Tiling4BinaryCrossEntropy)
    .TilingParse<BinaryCrossEntropyCompileInfo>(TilingPrepare4BinaryCrossEntropy);
}  // namespace optiling
