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
 * \file quant_batch_matmul_v4_msd_tiling.cpp
 * \brief
 */
#include "quant_batch_matmul_v4_msd_tiling.h"
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "vector"
#include "error_util.h"
#include "matmul/common/op_host/math_util.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "op_cache_tiling.h"
#include "op_host/tiling_key.h"
#include "../../op_kernel/quant_batch_matmul_v4_tiling_key.h"

using Ops::NN::Optiling::RecursiveSum;
using Ops::NN::TilingPrepareForOpCache;
namespace {

const std::vector<uint64_t> BASEN = {64, 128, 256, 512};    // baseN切分方案
constexpr uint32_t INT4_SIZE = 2; // sizeof(int4)的倒数
constexpr uint32_t UB_CALSIZE = 32U * 256U; // ub上每块buffer的size
constexpr uint32_t UB_BUFFER_NUMBER = 4; // ub上分成4块buffer
constexpr uint32_t UB_RESTBYTES = UB_CALSIZE * sizeof(float) * UB_BUFFER_NUMBER;
constexpr uint32_t CV_PARAll_NUM = 4;
constexpr size_t SUPPORT_DIM = 2;

}

namespace optiling {

ge::graphStatus QuantBatchMatmulV4MsdTiling::GetShapeAttrsInfo()
{
    inputParams_.opName = context_->GetNodeName();
    OPS_LOG_D(inputParams_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    OP_TILING_CHECK(CheckContext() != ge::GRAPH_SUCCESS, CUBE_INNER_ERR_REPORT(inputParams_.opName, "Invalid context_."),
                    return ge::GRAPH_PARAM_INVALID);
    OP_TILING_CHECK(!AnalyzeAttrs() || !AnalyzeDtype() || !AnalyzeInputs(),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Fail to analyze context_ info."),
                    return ge::GRAPH_FAILED);
    OP_LOGD(inputParams_.opName, "Input params: M K N [%ld, %ld, %ld], transposeA[%s], transposeB[%s]",
             inputParams_.mSize, inputParams_.kSize, inputParams_.nSize,
             inputParams_.transA ? "true" : "false", inputParams_.transB ? "true" : "false");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV4MsdTiling::CheckContext()
{
    auto x1Shape = context_->GetInputShape(X1_IDX);
    auto x1Desc = context_->GetInputDesc(X1_IDX);
    auto x2Shape = context_->GetInputShape(X2_IDX);
    auto x2Desc = context_->GetInputDesc(X2_IDX);
    auto x1ScaleShape = context_->GetOptionalInputShape(X1_SCALE_IDX);
    auto x1ScaleDesc = context_->GetOptionalInputDesc(X1_SCALE_IDX);
    auto x2ScaleShape = context_->GetOptionalInputShape(X2_SCALE_IDX);
    auto x2SclaeDesc = context_->GetOptionalInputDesc(X2_SCALE_IDX);
    auto yOffsetShape = context_->GetOptionalInputShape(Y_OFFSET_IDX);
    auto yOffsetDesc = context_->GetOptionalInputDesc(Y_OFFSET_IDX);
    auto outputShape = context_->GetOutputShape(Y_OUTPUT_IDX);
    auto outputDesc = context_->GetOutputDesc(Y_OUTPUT_IDX);
    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Function context_.GetAttrs() failed!"),
                    return ge::GRAPH_FAILED);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Desc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2Desc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1ScaleShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1ScaleDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2ScaleShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2SclaeDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, yOffsetShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, yOffsetDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    return ge::GRAPH_SUCCESS;
}

bool QuantBatchMatmulV4MsdTiling::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    auto x1ScaleShape = GetOptionShape(X1_SCALE_IDX);
    auto x2ScaleShape = GetOptionShape(X2_SCALE_IDX);

    const int64_t *groupSizePtr = attrs->GetAttrPointer<int64_t>(GROUP_SIZE_IDX);
    OP_TILING_CHECK(groupSizePtr == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Group size can not be nullptr."),
                    return false);
    inputParams_.groupSize = static_cast<uint64_t>(*groupSizePtr);

    // check transposeX1
    auto transposeX1 = attrs->GetAttrPointer<bool>(TRANSPOSE_X1_IDX);
    OP_TILING_CHECK(transposeX1 == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "TransposeX1 false can not be nullptr."),
                    return false);
    OP_TILING_CHECK(*transposeX1 != false,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Unsupported value [%d] for transposeX1, "
                                                   "Only transposeX1 = false is supported.", *transposeX1),
                    return false;
                    );
    inputParams_.transA = transposeX1 != nullptr && *transposeX1;
    // check transposeX2
    auto transposeX2 = attrs->GetAttrPointer<bool>(TRANSPOSE_X2_IDX);
    OP_TILING_CHECK(transposeX2 == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "TransposeX2 false can not be nullptr."),
                    return false);
    inputParams_.transB = transposeX2 != nullptr && *transposeX2;

    if (x1ScaleShape.GetDimNum() == 1 && x2ScaleShape.GetDimNum() == 1) {
        antiQuantType = QuantBatchMatmulV4QuantType::K_C;
    } else {
        antiQuantType = QuantBatchMatmulV4QuantType::K_G;
        OP_TILING_CHECK(inputParams_.groupSize != 256,
                        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Group size only support group k 256 on the A8W4 scenario, actual group sizeN is [%lu]", inputParams_.groupSize),
                        return false);
        OP_TILING_CHECK(*transposeX2 != false,
                        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Unsupported value [%d] for transposeX2, "
                                                        "Only transposeX2 = false is supported.", *transposeX2),
                        return false;);
    }

    return true;
}

bool QuantBatchMatmulV4MsdTiling::AnalyzeDtype()
{
    inputParams_.aDtype = ge::DT_INT4;
    auto actualADtype = context_->GetInputDesc(X1_IDX)->GetDataType();
    inputParams_.bDtype = context_->GetInputDesc(X2_IDX)->GetDataType();
    inputParams_.perTokenScaleDtype = context_->GetOptionalInputDesc(X1_SCALE_IDX)->GetDataType();
    inputParams_.scaleDtype = context_->GetOptionalInputDesc(X2_SCALE_IDX)->GetDataType();
    auto yOffsetDtype = context_->GetOptionalInputDesc(Y_OFFSET_IDX)->GetDataType();
    inputParams_.cDtype = context_->GetOutputDesc(Y_OUTPUT_IDX)->GetDataType();
    OP_TILING_CHECK(actualADtype != ge::DT_INT8,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,"Input dtype of x1 should be int8, actual dtype is %s",
                                                   ge::TypeUtils::DataTypeToSerialString(actualADtype).c_str()),return false);
    OP_TILING_CHECK(inputParams_.bDtype != ge::DT_INT4,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,"Input dtype of x2 should be int4, actual dtype is %s",
                                                   ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),return false);
    OP_TILING_CHECK(inputParams_.perTokenScaleDtype != ge::DT_FLOAT,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,"Input dtype of x1Scale should be float32, actual dtype is %s",
                                                   ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str()),return false);
    OP_TILING_CHECK(inputParams_.scaleDtype != ge::DT_UINT64,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,"Input dtype of x2Scale should be uint64, actual dtype is %s",
                                                   ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str()),return false);
    OP_TILING_CHECK(yOffsetDtype != ge::DT_FLOAT,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,"Input dtype of yOffset should be float32, actual dtype is %s",
                                                   ge::TypeUtils::DataTypeToSerialString(yOffsetDtype).c_str()),return false);
    OP_TILING_CHECK(!(inputParams_.cDtype == ge::DT_FLOAT16 || inputParams_.cDtype == ge::DT_BF16),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,"Output dtype  should be float16 or bfloat16, actual dtype is %s",
                                                   ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str()),return false);
    return true;
}

bool QuantBatchMatmulV4MsdTiling::AnalyzeInputs()
{
    inputParams_.aFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(X1_IDX)->GetStorageFormat()));
    OP_TILING_CHECK(inputParams_.aFormat != ge::FORMAT_ND, VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "aFormat Only support Nd"), return false);
    inputParams_.bFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(X2_IDX)->GetStorageFormat()));
    if (antiQuantType == QuantBatchMatmulV4QuantType::K_G) {
        OP_TILING_CHECK(inputParams_.bFormat != ge::FORMAT_ND, VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "bFormat Only support Nd"), return false);
    }
    auto x1Shape = GetShape(X1_IDX);
    auto x1ShapeLen = x1Shape.GetDimNum();
    OP_TILING_CHECK(x1ShapeLen != SUPPORT_DIM,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Input x1 dimension should equal to 2, but x1 dimension: %zu.",
                                                    x1ShapeLen), return false);
    actualMSize = x1Shape.GetDim(0);
    inputParams_.mSize =  actualMSize * INT4_SIZE;
    inputParams_.kSize = x1Shape.GetDim(1);

    if (inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ) {
        auto x2ScaleShape = GetOptionShape(X2_SCALE_IDX);
        auto x2ScaleShapeLen = x2ScaleShape.GetDimNum();
        inputParams_.nSize = x2ScaleShape[x2ScaleShapeLen - 1];
    } else {
        auto x2Shape = GetShape(X2_IDX);
        auto x2ShapeLen = x2Shape.GetDimNum();
        OP_TILING_CHECK(x2ShapeLen != SUPPORT_DIM,
                        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Inputx2 dimension should equal to 2, but x2 dimension: %zu.",
                                                        x2ShapeLen), return false);
        size_t bKSize = 0;
        size_t bNSize = 0;
        if (inputParams_.transB) {
            bKSize = x2Shape.GetDim(1);
            bNSize = x2Shape.GetDim(0);
        } else {
            bKSize = x2Shape.GetDim(0);
            bNSize = x2Shape.GetDim(1);
        }
        OP_TILING_CHECK(inputParams_.kSize != static_cast<uint64_t>(bKSize),
                        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "The size of k dimension in x1[%lu] is not equal to the size of k dimension in x2[%zu].",
                                                        inputParams_.kSize, bKSize), return false);
        inputParams_.nSize = bNSize;                                                        
    }

    return AnalyzeScaleInputs() && AnalyzeYOffsetInputs();
}

bool QuantBatchMatmulV4MsdTiling::AnalyzeScalePerChannel(const gert::Shape &x1ScaleShape, const gert::Shape &x2ScaleShape,
                                                       int64_t x1ScaleShapeLen, int64_t x2ScaleShapeLen) {
    OP_TILING_CHECK(x1ScaleShapeLen != 1,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Input x1Scale dimension should equal to 1, but x1Scale dimension %ld", x1ScaleShapeLen),
                    return false);
    auto x1ScaleShapeSize = x1ScaleShape.GetDim(0);
    OP_TILING_CHECK(static_cast<uint64_t>(x1ScaleShapeSize) != actualMSize,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "x1Scale size should equal to mSize %lu, but x1Scale inner dim is %ld",
                                                   actualMSize, x1ScaleShapeSize),
                    return false);
    OP_TILING_CHECK(x2ScaleShapeLen != 1,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Input x2Scale dimension should equal to 1, but x2Scale dimension is %ld", x2ScaleShapeLen),
                    return false);
    auto x2ScaleShapeSize = x2ScaleShape.GetDim(0);
    OP_TILING_CHECK(static_cast<uint64_t>(x2ScaleShapeSize) != inputParams_.nSize,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "x2Scale size should equal to nSize %zu, but x2Scale outer dim is %ld",
                                                   inputParams_.nSize, x2ScaleShapeSize),
                    return false);
    return true;
}

bool QuantBatchMatmulV4MsdTiling::AnalyzeScalePerGroup(const gert::Shape &x1ScaleShape, const gert::Shape &x2ScaleShape,
                                                       int64_t x1ScaleShapeLen, int64_t x2ScaleShapeLen) {
    OP_TILING_CHECK(x1ScaleShapeLen != SUPPORT_DIM,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Input x1Scale dimension should equal to 2, but x1Scale dimension %ld", x1ScaleShapeLen),
                    return false);
    auto x1ScaleShapeInner = x1ScaleShape.GetDim(0);
    auto x1ScaleShapeOuter = x1ScaleShape.GetDim(1);
    OP_TILING_CHECK(static_cast<uint64_t>(x1ScaleShapeInner) != actualMSize,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "x1Scale inner dim should equal to mSize %lu, but x1Scale inner dim is %ld",
                                                   actualMSize, x1ScaleShapeInner),
                    return false);
    OP_TILING_CHECK(x1ScaleShapeOuter != 1,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "x1Scale outer dim should equal to 1, but x1Scale outer dim is %ld",
                                                    x1ScaleShapeOuter),
                    return false);
    OP_TILING_CHECK(x2ScaleShapeLen != SUPPORT_DIM,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Input x2Scale dimension should equal to 2, but x2Scale dimension is %ld", x2ScaleShapeLen),
                    return false);
    auto x2ScaleShapeInner = x2ScaleShape.GetDim(0);
    auto x2ScaleShapeOuter = x2ScaleShape.GetDim(1);
    OP_TILING_CHECK(static_cast<uint64_t>(x2ScaleShapeInner) * inputParams_.groupSize != inputParams_.kSize,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "x2Scale inner dim should equal to nSize %zu / groupSizeK %zu, but x2Scale inner dim is %ld",
                                                   inputParams_.nSize, inputParams_.groupSize, x2ScaleShapeInner),
                    return false);
    OP_TILING_CHECK(static_cast<uint64_t>(x2ScaleShapeOuter) != inputParams_.nSize,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "x2Scale outer dim should equal to nSize %zu, but x2Scale outer dim is %ld",
                                                   inputParams_.nSize, x2ScaleShapeOuter),
                    return false);
    return true;
}

bool QuantBatchMatmulV4MsdTiling::AnalyzeScaleInputs()
{
    auto x1ScaleShape = GetOptionShape(X1_SCALE_IDX);
    auto x2ScaleShape = GetOptionShape(X2_SCALE_IDX);
    auto x1ScaleShapeLen = x1ScaleShape.GetDimNum();
    auto x2ScaleShapeLen = x2ScaleShape.GetDimNum();
    if (antiQuantType == QuantBatchMatmulV4QuantType::K_C) {
        OP_TILING_CHECK(!AnalyzeScalePerChannel(x1ScaleShape, x2ScaleShape, x1ScaleShapeLen, x2ScaleShapeLen),
                        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "A8W4 perchannel scale check failed"),
                        return false);
    } else {
        OP_TILING_CHECK(!AnalyzeScalePerGroup(x1ScaleShape, x2ScaleShape, x1ScaleShapeLen, x2ScaleShapeLen),
                        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "A8W4 pergroup scale check failed"),
                        return false);
    }
    return true;
}

bool QuantBatchMatmulV4MsdTiling::AnalyzeYOffsetInputs()
{
    auto yOffsetShape = GetOptionShape(Y_OFFSET_IDX);
    auto yOffsetShapeLen = yOffsetShape.GetDimNum();
    OP_TILING_CHECK(yOffsetShapeLen != 1,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Input yOffset dimension should equal to 1, but yOffset dimension %zu",
                                                    yOffsetShapeLen),
                    return false);
    auto yOffsetShapeInner = yOffsetShape.GetDim(0);
    OP_TILING_CHECK(static_cast<uint64_t>(yOffsetShapeInner) != inputParams_.nSize,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Input yOffset size should equal to nSize %zu, but yOffset size is %ld",
                                                    inputParams_.nSize, yOffsetShapeInner),
                    return false);
    return true;
}

ge::graphStatus QuantBatchMatmulV4MsdTiling::DoOpTiling()
{
    InitCompileInfo();
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "The platformInfo Ptr is null!");
    PlatformUtil::ParseRuntimePlatformInfo(compileInfo_, context->GetNodeName(), *platformInfoPtr);
    OP_LOGE_IF(!SetPlatformInfoForTiling(), ge::GRAPH_FAILED, inputParams_.opName, "SetPlatformInfoForTiling fail");
    OP_TILING_CHECK(InstantiateTilingData() == ge::GRAPH_FAILED,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "unable to get pointer of tiling data"),
                    return ge::GRAPH_FAILED);
    if (antiQuantType == QuantBatchMatmulV4QuantType::K_G) {
        OP_TILING_CHECK(DoOpTilingPerGroup() == false,
                        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "unable to get pointer of tiling data"),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(DoOpTilingPerChannel() == false,
                        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "unable to get pointer of tiling data"),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

bool QuantBatchMatmulV4MsdTiling::DoOpTilingPerChannel()
{
    SetTransAttr(trans_);
    isUbQuant_ = false;
    OP_LOGE_IF(!QuantBatchMatmulV3BasicTiling::DoBasicTiling(), ge::GRAPH_FAILED, inputParams_.opName, "DoBasicTiling failed.");
    SetMatmulTilingFromBasicTiling();
    workspaceSize_ = SYS_WORKSPACE_SIZE + CV_PARAll_NUM * basicTiling_.usedCoreNum * basicTiling_.baseM * basicTiling_.baseN * sizeof(int16_t);
    return true;
}

void QuantBatchMatmulV4MsdTiling::SetMatmulTilingFromBasicTiling()
{
    tilingData_->matmulTiling.M = inputParams_.GetTotalMatmulApiMSize(basicTiling_.baseM);
    tilingData_->matmulTiling.N = inputParams_.nSize;
    tilingData_->matmulTiling.Ka = inputParams_.kSize;
    tilingData_->matmulTiling.Kb = inputParams_.kSize;
    tilingData_->matmulTiling.usedCoreNum = basicTiling_.usedCoreNum;
    tilingData_->matmulTiling.singleCoreM = basicTiling_.baseM;
    tilingData_->matmulTiling.singleCoreN = basicTiling_.baseN;
    tilingData_->matmulTiling.singleCoreK = basicTiling_.singleCoreK;
    tilingData_->matmulTiling.baseM = basicTiling_.baseM;
    tilingData_->matmulTiling.baseN = basicTiling_.baseN;
    tilingData_->matmulTiling.baseK = basicTiling_.baseK;
    tilingData_->matmulTiling.depthA1 = basicTiling_.depthA1;
    tilingData_->matmulTiling.depthB1 = basicTiling_.depthB1;
    tilingData_->matmulTiling.stepM = basicTiling_.stepM;
    tilingData_->matmulTiling.stepN = basicTiling_.stepN;
    tilingData_->matmulTiling.stepKa = basicTiling_.stepKa;
    tilingData_->matmulTiling.stepKb = basicTiling_.stepKb;
    tilingData_->matmulTiling.iterateOrder = basicTiling_.iterateOrder;
    tilingData_->matmulTiling.dbL0C = basicTiling_.dbL0c;  // 1: off, 2:on

    tilingData_->groupSize = inputParams_.groupSize;
    tilingData_->kSize = inputParams_.kSize;
    tilingData_->nSize = inputParams_.nSize;
    tilingData_->mSize = actualMSize;
    tilingData_->ubCalSize = UB_CALSIZE;
    uint32_t vBaseMMax = tilingData_->ubCalSize / basicTiling_.baseN;
    uint32_t vBaseMOneVector = basicTiling_.baseM / INT4_SIZE;
    tilingData_->vBaseM =  vBaseMMax < vBaseMOneVector? vBaseMMax : vBaseMOneVector ;
    tilingData_->ubRestBytes = UB_RESTBYTES;
    tilingData_->parallNum = CV_PARAll_NUM;
    tilingData_->coreNum = basicTiling_.usedCoreNum;
}


bool QuantBatchMatmulV4MsdTiling::DoOpTilingPerGroup()
{
    constexpr uint32_t vBaseMSize = 32;
    tilingData_->coreNum = compileInfo_.aicNum;
    tilingData_->groupSize = inputParams_.groupSize;
    tilingData_->kSize = inputParams_.kSize;
    tilingData_->nSize = inputParams_.nSize;
    tilingData_->mSize = actualMSize;
    tilingData_->ubCalSize = UB_CALSIZE;
    tilingData_->vBaseM = vBaseMSize;
    tilingData_->ubRestBytes = UB_RESTBYTES;
    tilingData_->parallNum = CV_PARAll_NUM;
    if(!SetMatmulTilingPerGroup()) {
        return false;
    }
    uint64_t baseSize = tilingData_->matmulTiling.baseM * tilingData_->matmulTiling.baseN;
    workspaceSize_ = SYS_WORKSPACE_SIZE + CV_PARAll_NUM * compileInfo_.aicNum * baseSize * sizeof(int16_t);
    return true;
}

bool QuantBatchMatmulV4MsdTiling::SetMatmulTilingPerGroup()
{
    constexpr uint32_t baseM = 16;  //基本块大小，参考设计文档
    constexpr uint32_t baseK = 256; //基本块大小，参考设计文档
    uint64_t baseMNum = ops::CeilDiv(inputParams_.mSize, static_cast<uint64_t>(baseM));
    uint64_t aicNumN = ops::CeilDiv(static_cast<uint64_t>(compileInfo_.aicNum), baseMNum);
    uint64_t baseN = ops::CeilDiv(inputParams_.nSize, aicNumN);
    for (size_t i =0; i < BASEN.size(); ++i) {
        if (baseN < BASEN[i]) {
            baseN = BASEN[i];
            break;
        }
        if (i == BASEN.size() - 1) {
            baseN = BASEN[i];
        }
    }
    matmul_tiling::PlatformInfo platformInfo;
    InitPlatformInfo(&compileInfo_, platformInfo);
    matmul_tiling::MultiCoreMatmulTiling mm(platformInfo);
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT4, false);
    if (inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ) {
        mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_INT4, inputParams_.transB);      
    } else {
        mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT4, inputParams_.transB);        
    }
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    mm.SetBias(false);
    mm.SetOrgShape(baseM, inputParams_.nSize, inputParams_.kSize);
    mm.SetShape(baseM, baseN, inputParams_.kSize);
    mm.SetFixSplit(baseM, baseN, baseK);
    if (mm.GetTiling(tilingData_->matmulTiling) ==-1) {
        return false;
    }
    constexpr uint32_t stepK = 2;    //l0a l0b上开pingpong
    tilingData_->matmulTiling.dbL0C = 1;
    tilingData_->matmulTiling.stepKa = stepK;
    tilingData_->matmulTiling.stepKb = stepK;
    tilingData_->matmulTiling.depthA1 = 1;
    tilingData_->matmulTiling.depthB1 = 1;
    tilingData_->matmulTiling.stepM = 1;
    tilingData_->matmulTiling.stepN = 1;
    tilingData_->matmulTiling.baseK = baseK / stepK;   //l0a l0b上开pingpong
    return true;
}

ge::graphStatus QuantBatchMatmulV4MsdTiling::InstantiateTilingData()
{
    if (tilingData_ == nullptr) {
        tilingDataManager_ = std::make_unique<QuantBatchMatmulV4MsdTilingData>();
        OP_TILING_CHECK(tilingDataManager_ == nullptr,
                        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "failed to instantiate tilingData_"),
                        return ge::GRAPH_FAILED);
        tilingData_ = tilingDataManager_.get();
    }
    size_t tilingDataSize = sizeof(QuantBatchMatmulV4MsdTilingData);
    OP_TILING_CHECK(tilingData_ == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,"tiling data capacity %zu < actual tiling data size %zu",
                                                   context_->GetRawTilingData()->GetCapacity(),
                                                   tilingDataSize), return ge::GRAPH_SUCCESS;);
    return ge::GRAPH_SUCCESS;
}

void QuantBatchMatmulV4MsdTiling::InitPlatformInfo(const QuantBatchMatmulV3CompileInfo* compileInfoPtr_, matmul_tiling::PlatformInfo& platformInfo) const
{
    platformInfo.socVersion = compileInfoPtr_->socVersion;
    platformInfo.l1Size = compileInfoPtr_->l1Size;
    platformInfo.l0CSize = compileInfoPtr_->l0cSize;
    platformInfo.ubSize = compileInfoPtr_->ubSize;
    platformInfo.l0ASize = compileInfoPtr_->l0aSize;
    platformInfo.l0BSize = compileInfoPtr_->l0bSize;
}

uint64_t QuantBatchMatmulV4MsdTiling::GetTilingKey() const
{
    uint64_t trans =
        (static_cast<uint64_t>(inputParams_.transA) << 1) | static_cast<uint64_t>(inputParams_.transB);
    bool weightNz = inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ;
    return GET_TPL_TILING_KEY(trans, static_cast<uint64_t>(antiQuantType),/*hasAntiQuantOffset*/uint64_t(0),
                            static_cast<uint64_t>(weightNz), static_cast<uint64_t>(KernelTemplateType::MSD_BASIS));
}

const gert::Shape QuantBatchMatmulV4MsdTiling::GetShape(const size_t index) const
{
    return context_->GetInputShape(index)->GetStorageShape();
}

const gert::Shape QuantBatchMatmulV4MsdTiling::GetOptionShape(const size_t index) const
{
    return context_->GetOptionalInputShape(index)->GetStorageShape();
}

ge::graphStatus QuantBatchMatmulV4MsdTiling::DoLibApiTiling()
{
    size_t tilingDataSize = sizeof(QuantBatchMatmulV4MsdTilingData);
    context_->SetBlockDim(tilingData_->coreNum);
    context_->SetScheduleMode(1);   // 独占全核，设置以后会让所有核空闲以后才启动，有多核同步指令需要做此设置避免影响整网其他算子
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(), reinterpret_cast<void *>(tilingData_), tilingDataSize);
    if (ret != EOK){
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

void QuantBatchMatmulV4MsdTiling::PrintTilingData() const
{
    if (CheckLogLevel(OP, DLOG_DEBUG) != 1) {
        return;
    }
    std::stringstream ss;
    ss << " CoreNum: " << static_cast<int32_t>(tilingData_->coreNum) << " vBaseM: " << tilingData_->vBaseM
       << " ubRestBytes: " << tilingData_->ubRestBytes << " parallNum: " << tilingData_->parallNum
       << " ubCalSize: " << tilingData_->ubCalSize << " mSize: " << tilingData_->mSize
       << " kSize: " << tilingData_->kSize << " nSize: " << tilingData_->nSize
       << " groupSize: " << tilingData_->groupSize;
    OPS_LOG_D(inputParams_.opName, "api tiling: %s", ss.str().c_str());
    PrintMatmulTilingData();
}

void QuantBatchMatmulV4MsdTiling::PrintMatmulTilingData() const
{
    if (CheckLogLevel(OP, DLOG_DEBUG) != 1) {
        return;
    }
    auto &matmulTiling = tilingData_->matmulTiling;
    std::stringstream ss;
    ss << " usedCoreNum: " << matmulTiling.usedCoreNum << " M: " << matmulTiling.M << " N: " << matmulTiling.N
       << " Ka: " << matmulTiling.Ka << " Kb: " << matmulTiling.Kb << " singleCoreM: " << matmulTiling.singleCoreM
       << " singleCoreN: " << matmulTiling.singleCoreN << " singleCoreK: " << matmulTiling.singleCoreK
       << " baseM: " << matmulTiling.baseM << " baseN: " << matmulTiling.baseN << " baseK: " << matmulTiling.baseK
       << " depthA1: " << matmulTiling.depthA1 << " depthB1: " << matmulTiling.depthB1
       << " stepM: " << matmulTiling.stepM << " stepN: " << matmulTiling.stepN << " stepka: " << matmulTiling.stepKa
       << " stepkb: " << matmulTiling.stepKb << " isBias: " << matmulTiling.isBias
       << " transLength: " << matmulTiling.transLength << " iterateOrder: " << matmulTiling.iterateOrder
       << " shareMode: " << matmulTiling.shareMode << " dbL0A: " << matmulTiling.dbL0A
       << " dbL0B: " << matmulTiling.dbL0B << " dbL0C: " << matmulTiling.dbL0C
       << " usedL1Size: " << matmulTiling.shareL1Size << " usedL0CSize: " << matmulTiling.shareL0CSize
       << " usedUBSize: " << matmulTiling.shareUbSize << " batchM: " << matmulTiling.batchM
       << " batchN: " << matmulTiling.batchN << " singleBatchM: " << matmulTiling.singleBatchM
       << " singleBatchN: " << matmulTiling.singleBatchN;
    OPS_LOG_D(inputParams_.opName, "matmulTiling: %s", ss.str().c_str());
}

bool QuantBatchMatmulV4MsdTiling::IsCapable()
{
    OP_TILING_CHECK(inputParams_.kSize > 29576 ,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "K should be less than 29576 on the A8W4 scenario,"
                                                    "but now is %lu", inputParams_.kSize),
                    return false);
    if (antiQuantType == QuantBatchMatmulV4QuantType::K_G) {
        OP_TILING_CHECK(inputParams_.kSize % inputParams_.groupSize != 0 ,
                        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "K should be divisible by groupSizek, but now k = %lu,"
                                                        "groupSize is %lu", inputParams_.kSize, inputParams_.groupSize),
                        return false);
        OP_TILING_CHECK(inputParams_.groupSize != 256 ,
                        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "A8W4 pergroup groupSize should be 256, but now groupSize = %lu",
                                                        inputParams_.groupSize),
                        return false);        
    } else {
        OP_TILING_CHECK(inputParams_.groupSize != 0,
                        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "A8W4 perchannel groupSize should be 0, but now groupSize = %lu",
                                                        inputParams_.groupSize),
                        return false);
    }

    return true;
}

ge::graphStatus QuantBatchMatmulV4MsdTiling::PostTiling()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV4MsdTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}
}// namespace optiling