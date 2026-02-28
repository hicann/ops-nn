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
 * \file fused_quant_matmul_asw_tiling.cpp
 * \brief
 */

#include "fused_quant_matmul_asw_tiling.h"

#include "common/op_host/op_tiling/tiling_type.h"
#include "error_util.h"
#include "op_api/op_util.h"

#include "fused_quant_matmul_checker.h"

using namespace FusedQuantMatMulTilingKey;
namespace {

static constexpr int8_t OUTPUT_INFER_FAIL = static_cast<int8_t>(-1);
static constexpr int8_t OUTPUT_INFER_SUCCESS = 1;
constexpr size_t MIN_DIM_NUM_ND = 2;
constexpr size_t MAX_DIM_NUM_ND = 6;
} // namespace

namespace optiling {

const std::unordered_map<std::string, FQMMFusedOpType> FUSED_OP_TYPE_STR_TO_ENUM_MAP = {
    {"", FQMMFusedOpType::NONE},
    {"relu", FQMMFusedOpType::RELU},
    {"swiglu", FQMMFusedOpType::SWIGLU},
};

bool FusedQuantMatMulASWTiling::AnalyzeAttrs() {
    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, CUBE_INNER_ERR_REPORT(inputParams_.opName, "GetAttrs return nullptr."),
                    return false);

    size_t idx = 0;
    auto dtypePtr = attrs->GetAttrPointer<int64_t>(idx++);
    OP_TILING_CHECK(dtypePtr == nullptr, CUBE_INNER_ERR_REPORT(inputParams_.opName, "Attr dtype is nullptr."),
                    return false);
    // skip attr compute_type
    idx++;
    auto transposeX1Ptr = attrs->GetAttrPointer<bool>(idx++);
    auto transposeX2Ptr = attrs->GetAttrPointer<bool>(idx++);
    auto groupSizePtr = attrs->GetAttrPointer<int64_t>(idx++);

    inputParams_.outDtype = *dtypePtr;
    inputParams_.transA = transposeX1Ptr ? *transposeX1Ptr : false;
    inputParams_.transB = transposeX2Ptr ? *transposeX2Ptr : false;
    inputParams_.groupSize = groupSizePtr ? static_cast<uint64_t>(*groupSizePtr) : 0ULL;
    OP_TILING_CHECK(inputParams_.groupSize != 0ULL,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "FusedQuantMatMul only suppot groupSize equal 0."),
                    return false);

    // attr fused_op_type
    auto fusedOpType = attrs->GetAttrPointer<char>(idx++);
    OP_TILING_CHECK(fusedOpType == nullptr,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "fused_op_type should not be null"), return false);
    std::string fusedOpTypeStr(fusedOpType);
    auto it = FUSED_OP_TYPE_STR_TO_ENUM_MAP.find(fusedOpTypeStr);
    OP_TILING_CHECK(
        it == FUSED_OP_TYPE_STR_TO_ENUM_MAP.end(),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "current fused_op_type:[%s] is not supported", fusedOpType),
        return false);
    fusedOpType_ = static_cast<uint64_t>(it->second);

    OP_LOGI(inputParams_.opName, "Init attr param, transA: %s, transB: %s, outDtype: %ld, fused op type: %s",
            inputParams_.transA ? "true" : "false", inputParams_.transB ? "true" : "false", inputParams_.outDtype,
            strcmp(fusedOpType, "") != 0 ? fusedOpType : "none");

    return true;
}

bool FusedQuantMatMulASWTiling::CheckDtype() const {
    FusedQuantMatMulChecker fqmmChecker(context_, inputParams_);
    OP_TILING_CHECK(!fqmmChecker.CheckDtype(), CUBE_INNER_ERR_REPORT(inputParams_.opName, "CheckDtype fail"),
                    return false);
    return true;
}

bool FusedQuantMatMulASWTiling::AnalyzeX2TableAttr(const gert::StorageShape *x2TableShape) {
    if (x2TableShape != nullptr) {
        OP_TILING_CHECK(!compileInfo_.supportMmadS8S4,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName, "x2Table only support soc with MmadS8S4."),
                        return false);

        auto x2TableShapeLen = x2TableShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(x2TableShapeLen != 2,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName, "x2 table shape should be 2 dim"), return false);
        // the x2Table is transposed
        inputParams_.x2TableNSize =
            static_cast<uint64_t>(x2TableShape->GetStorageShape().GetDim(x2TableShapeLen - LAST_SECOND_DIM_INDEX));
        inputParams_.x2TableKSize =
            static_cast<uint64_t>(x2TableShape->GetStorageShape().GetDim(x2TableShapeLen - LAST_FIRST_DIM_INDEX));
    }
    return true;
}

bool FusedQuantMatMulASWTiling::CheckShape(const std::vector<gert::Shape *> &mandtoryShape,
                                           const std::vector<const gert::StorageShape *> &optionalInputShape,
                                           const std::vector<int64_t> &dimValueOfMKN) const {
    FusedQuantMatMulChecker fqmmChecker(context_, inputParams_);
    OP_TILING_CHECK(!fqmmChecker.CheckShape(mandtoryShape, optionalInputShape, dimValueOfMKN),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "CheckShape fail"), return false);
    return true;
}

ge::graphStatus FusedQuantMatMulASWTiling::CheckContext() {
    auto x1Shape = context_->GetInputShape(GetX1Idx());
    auto x1Desc = context_->GetInputDesc(GetX1Idx());
    auto x2Shape = context_->GetInputShape(GetX2Idx());
    auto x2Desc = context_->GetInputDesc(GetX2Idx());
    auto outputShape = context_->GetOutputShape(0);
    auto outputDesc = context_->GetOutputDesc(0);
    auto attrs = context_->GetAttrs();
    auto dtypeAttr = attrs->GetAttrPointer<int64_t>(0);

    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Desc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2Desc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, dtypeAttr);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    OP_TILING_CHECK(context_->GetRawTilingData()->GetCapacity() < tilingDataSize_,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "context tiling data capacity %zu < actual tiling data size %zu.",
                                          context_->GetRawTilingData()->GetCapacity(), tilingDataSize_),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool FusedQuantMatMulASWTiling::AnalyzeDtype() {
    inputParams_.aDtype = context_->GetInputDesc(GetX1Idx())->GetDataType();
    auto x2Desc = context_->GetInputDesc(GetX2Idx());
    inputParams_.bDtype = x2Desc->GetDataType();
    OP_TILING_CHECK(
        !(inputParams_.aDtype == ge::DT_INT8 || inputParams_.bDtype == ge::DT_INT8),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "x1 and x2 of FusedQuantMatmul dtype onply support INT8!"),
        return false);

    auto biasDesc = context_->GetOptionalInputDesc(GetBiasIdx());
    inputParams_.biasDtype = biasDesc != nullptr ? biasDesc->GetDataType() : ge::DT_INT32;

    auto pertokenScaleDesc = context_->GetOptionalInputDesc(GetPertokenIdx());
    inputParams_.perTokenScaleDtype =
        pertokenScaleDesc != nullptr ? pertokenScaleDesc->GetDataType() : inputParams_.perTokenScaleDtype;

    auto x2ScaleDesc = context_->GetOptionalInputDesc(GetScaleIdx());
    inputParams_.scaleDtype =
        x2ScaleDesc != nullptr ? x2ScaleDesc->GetDataType() : inputParams_.scaleDtype;

    // isLut and x2TableDesc only support soc version with MmadS8S4
    auto x2TableDesc = context_->GetOptionalInputDesc(GetX2TableIdx());
    inputParams_.x2TableDtype = x2TableDesc != nullptr ? x2TableDesc->GetDataType() : inputParams_.x2TableDtype;
    inputParams_.isLut = x2TableDesc != nullptr && compileInfo_.supportMmadS8S4;

    inputParams_.cDtype = context_->GetOutputDesc(0)->GetDataType();
    OP_TILING_CHECK(
        !(inputParams_.cDtype == ge::DT_INT8 || inputParams_.cDtype == ge::DT_FLOAT16),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "y of FusedQuantMatmul dtype onply support INT8 or Float16!"),
        return false);

    isUbQuant_ = inputParams_.cDtype == ge::DT_BF16 || pertokenScaleDesc != nullptr;
    SetFormat();

    OP_TILING_CHECK(!CheckDtype(), CUBE_INNER_ERR_REPORT(inputParams_.opName, "CheckDtype failed!"), return false);
    return true;
}

bool FusedQuantMatMulASWTiling::CheckUnsupportedOptionInputs() {
    // 当前算子不支持x1Scale
    auto pertokenShape = GetPertokenShape(GetPertokenIdx());
    OP_TILING_CHECK(pertokenShape,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "The current FusedQuantMatMul not support x1Scale."),
                    return false);
    // 当前算子不支持yScale
    auto yScaleShape = context_->GetOptionalInputShape(GetYScaleIdx());
    OP_TILING_CHECK(yScaleShape,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "The current FusedQuantMatMul not support yScale."),
                    return false);
    // 当前算子不支持x1Offset
    auto x1OffsetShape = context_->GetOptionalInputShape(GetX1OffsetIdx());
    OP_TILING_CHECK(x1OffsetShape,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "The current FusedQuantMatMul not support x1Offset."),
                    return false);
    // 当前算子不支持yOffset
    auto OffYsetShape = context_->GetOptionalInputShape(GetYOffsetIdx());
    OP_TILING_CHECK(OffYsetShape,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "The current FusedQuantMatMul not support YOffset."),
                    return false);
    // 当前算子不支持x2Table
    auto x2TableShape = context_->GetOptionalInputShape(GetX2TableIdx());
    OP_TILING_CHECK(x2TableShape,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "The current FusedQuantMatMul not support X2Table."),
                    return false);
    // 当前算子不支持x3输入
    auto x3Shape = context_->GetOptionalInputShape(GetX3Idx());
    OP_TILING_CHECK(x3Shape, CUBE_INNER_ERR_REPORT(inputParams_.opName, "The current FusedQuantMatMul not support X3."),
                    return false);

    return true;
}

bool FusedQuantMatMulASWTiling::AnalyzeInputs() {
    // 必选输入 x1, x2
    auto x1Shape = GetX1Shape(GetX1Idx());
    auto x2Shape = GetX2Shape(GetX2Idx());

    auto x1ShapeLen = x1Shape.GetDimNum();
    auto x2ShapeLen = x2Shape.GetDimNum();
    if (!isTilingOut_ && !CheckShapeInRangeForMandtoryInputs(x1ShapeLen, x2ShapeLen)) {
        return false;
    }

    auto x1Inner = x1Shape.GetDim(x1ShapeLen - LAST_FIRST_DIM_INDEX);
    auto x1Outer = x1Shape.GetDim(x1ShapeLen - LAST_SECOND_DIM_INDEX);
    auto x2Inner = x2Shape.GetDim(x2ShapeLen - LAST_FIRST_DIM_INDEX);
    auto x2Outer = x2Shape.GetDim(x2ShapeLen - LAST_SECOND_DIM_INDEX);

    // int4pack输入场景修正为dtype为int4
    if (inputParams_.bDtype == ge::DT_INT32) {
        x2Inner *= 8; // int4类型不支持时，在图模式中用1个int32代表8个int4
        inputParams_.bDtype = ge::DT_INT4;
        OP_LOGD(inputParams_.opName, "The conversion of weight from int32 to int4 is completed.");
    }

    inputParams_.mSize = static_cast<uint64_t>(inputParams_.transA ? x1Inner : x1Outer);
    inputParams_.kSize = static_cast<uint64_t>(inputParams_.transA ? x1Outer : x1Inner);
    inputParams_.nSize = static_cast<uint64_t>(inputParams_.transB ? x2Outer : x2Inner);

    const std::vector<int64_t> dimValueOfMKN = {x1Inner, x1Outer, x2Inner, x2Outer};
    const std::vector<gert::Shape *> mandtoryShape = {&x1Shape, &x2Shape};

    // 可选输入
    OP_TILING_CHECK(
        !CheckUnsupportedOptionInputs(),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "The optional input of FusedQuantMatMul is not supported."),
        return false);

    auto biasShape = GetBiasShape(GetBiasIdx());
    inputParams_.hasBias = biasShape != nullptr;
    inputParams_.batchBias = inputParams_.hasBias ? GetBatchSize(biasShape->GetStorageShape()) : 1;
    // 当前算子不支持bias有batch
    OP_TILING_CHECK(
        inputParams_.batchBias > 1U,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "The batchSize of FusedQuantMatMul's bias should equal 1."),
        return false);

    // 当前算子不支持pertoken场景
    inputParams_.isPertoken = false;
    const gert::StorageShape *pertokenShape = nullptr;

    // 当前算子不支持scale为空
    auto scaleShape = context_->GetOptionalInputShape(GetScaleIdx());
    OP_TILING_CHECK(!scaleShape,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "The x2 scale of FusedQuantMatMul can not be null."),
                    return false);

    // 当前不支持x2Table场景, 仅做预埋
    const gert::StorageShape *x2TableShape = nullptr;
    OP_TILING_CHECK(!AnalyzeX2TableAttr(x2TableShape),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "The X2Table of FusedQuantMatMul is illegal."),
                    return false);

    // AnalyzeGroupInfo为pertoken场景下对group的check，由于当前不支持pertoken场景，仅预埋
    if (!AnalyzeGroupInfo(scaleShape->GetStorageShape(), pertokenShape)) {
        return false;
    }
    inputParams_.batchA = GetBatchSize(x1Shape);
    inputParams_.batchB = GetBatchSize(x2Shape);
    AnalyzeBatchInfo(context_->GetInputShape(0)->GetOriginShape(), context_->GetInputShape(1)->GetOriginShape());
    OP_TILING_CHECK(!InferOutBatchDim(x1Shape, x2Shape),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Batch dimension can not be broadcasted."),
                    return false);

    if (!SetQuantMode(scaleShape->GetStorageShape(), pertokenShape)) {
        return false;
    }
    // 当前算子仅支持PerChannel和PerTensor量化场景
    OP_TILING_CHECK(!(inputParams_.isPerChannel || inputParams_.isPerTensor),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "The current FusedQuantMatMul only support PerChannel or PerTensor."),
                    return false);

    const std::vector<const gert::StorageShape *> optionalInputShape = {biasShape, scaleShape};
    if (!isTilingOut_ && !CheckShape(mandtoryShape, optionalInputShape, dimValueOfMKN)) {
        return false;
    }
    OP_TILING_CHECK(
        !CheckOutputShapeAvailable(),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Multiple of output shape dims should be in boundary of INT64_MAX"),
        return false);

    uint64_t fusedDimValue = inputParams_.mSize * inputParams_.batchA;
    int8_t resultCheckFusionBatchA = CheckFusionBatchA(x1Shape, x2Shape, biasShape, fusedDimValue);
    OP_TILING_CHECK(resultCheckFusionBatchA == OUTPUT_INFER_FAIL,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "The fused M [%lu] exceed INT32_MAX [%d] case",
                                          fusedDimValue, INT32_MAX),
                    return false);
    if (resultCheckFusionBatchA == OUTPUT_INFER_SUCCESS) {
        DoBatchFusion(fusedDimValue);
    }
    auto isPerTensorStr = inputParams_.isPerTensor ? "true" : "false";
    auto isPertokenStr = inputParams_.isPertoken ? "true" : "false";
    OP_LOGD(inputParams_.opName, "batchA: %lu, batchB: %lu, batchC: %lu, isPerTensor: %s, isPertoken: %s",
            inputParams_.batchA, inputParams_.batchB, inputParams_.batchC, isPerTensorStr, isPertokenStr);
    return true;
}

bool FusedQuantMatMulASWTiling::CheckShapeInRangeForMandtoryInputs(size_t x1ShapeLen, size_t x2ShapeLen) const {
    OP_TILING_CHECK(x1ShapeLen < MIN_DIM_NUM_ND || x2ShapeLen < MIN_DIM_NUM_ND,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "x1 len and x2 len should be greater than 1, but x1 len: %zu, x2 len: %zu",
                                          x1ShapeLen, x2ShapeLen),
                    return false);
    OP_TILING_CHECK(x1ShapeLen > MAX_DIM_NUM_ND || x2ShapeLen > MAX_DIM_NUM_ND,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "x1 len and x2 len should be less than 7, but x1 len: %zu, x2 len: %zu",
                                          x1ShapeLen, x2ShapeLen),
                    return false);
    return true;
}

uint64_t FusedQuantMatMulASWTiling::GetTilingKey() const {
    auto biasMode = GetBiasMode();
    auto kernelType = GetKernelType();
    OP_LOGD(inputParams_.opName,
            "tilingKey encode with transA: %lu, transB: %lu, BiasMode: %lu, KernelType: %lu, fusedOpType: %lu",
            static_cast<uint64_t>(inputParams_.transA), static_cast<uint64_t>(inputParams_.transB), biasMode,
            kernelType, fusedOpType_);

    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams_.transA), static_cast<uint64_t>(inputParams_.transB),
                              biasMode, kernelType, static_cast<uint64_t>(fusedOpType_));
}
} // namespace optiling