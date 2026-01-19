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
 * \file fused_quant_matmul_unary_tiling.cpp
 * \brief
 */

#include "fused_quant_matmul_unary_tiling.h"

#include "common/op_host/op_tiling/tiling_type.h"
#include "error_util.h"
#include "op_util.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "../../op_kernel/fused_quant_mat_mul_kernel_tilingkey.h"

namespace {
std::string DType2Str(const ge::DataType dataType)
{
    std::string serialString = ge::TypeUtils::DataTypeToSerialString(dataType);
    std::string prefix = "DT_";
    size_t pos = serialString.find(prefix);
    if (pos != std::string::npos) {
        serialString.erase(pos, prefix.length());
    }
    return serialString;
}
} // namespace

namespace optiling {
const std::unordered_map<std::string, FQMMFusedOpType> FUSED_UNARY_OP_TYPE_STR_TO_ENUM_MAP = {
    {"gelu_tanh", FQMMFusedOpType::GELU_TANH},
    {"gelu_erf", FQMMFusedOpType::GELU_ERF},
};

bool FusedQuantMatMulUnaryTiling::CheckUseBasicTiling()
{   
    return true; // 默认走qbmmv3 basic tiling
}

ge::graphStatus FusedQuantMatMulUnaryTiling::CheckContext()
{
    auto x1Shape = context_->GetInputShape(GetX1Idx());
    auto x1Desc = context_->GetInputDesc(GetX1Idx());
    auto x2Shape = context_->GetInputShape(GetX2Idx());
    auto x2Desc = context_->GetInputDesc(GetX2Idx());
    auto pertokenScaleShape = context_->GetOptionalInputShape(GetPertokenIdx());
    auto pertokenScaleDesc = context_->GetOptionalInputDesc(GetPertokenIdx());
    auto scaleShape = context_->GetOptionalInputShape(GetScaleIdx());
    auto scaleDesc = context_->GetOptionalInputDesc(GetScaleIdx());
    auto outputShape = context_->GetOutputShape(0);
    auto outputDesc = context_->GetOutputDesc(0);
    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Function context_->GetAttrs() failed!"),
                    return ge::GRAPH_FAILED);
    auto dtypeAttr = attrs->GetAttrPointer<int64_t>(0);

    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Desc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2Desc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, pertokenScaleShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, pertokenScaleDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, scaleShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, scaleDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, dtypeAttr);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    OP_TILING_CHECK(
        context_->GetRawTilingData()->GetCapacity() < tilingDataSize_,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "context tiling data capacity %zu < actual tiling data size %zu.",
                              context_->GetRawTilingData()->GetCapacity(), tilingDataSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool FusedQuantMatMulUnaryTiling::AnalyzeAttrs()
{
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
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "FusedQuantMatMul only support groupSize equal 0."),
                    return false);
    
    QuantBatchMatmulV3Trans trans = QuantBatchMatmulV3Trans::NO_TRANS;
    SetTransAttr(trans);

    // attr fused_op_type
    auto fusedOpType = attrs->GetAttrPointer<char>(idx++);
    OP_TILING_CHECK(fusedOpType == nullptr,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "fused_op_type should not be null"), return false);
    std::string fusedOpTypeStr(fusedOpType);
    auto it = FUSED_UNARY_OP_TYPE_STR_TO_ENUM_MAP.find(fusedOpTypeStr);
    OP_TILING_CHECK(
        it == FUSED_UNARY_OP_TYPE_STR_TO_ENUM_MAP.end(),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "current fused_op_type:[%s] is not supported, only supports gelu_tanh or gelu_erf", fusedOpType),
        return false);
    fusedOpType_ = static_cast<uint64_t>(it->second);

    OP_LOGI(inputParams_.opName, "Init attr param, transA: %s, transB: %s, outDtype: %ld, fused op type: %s",
            inputParams_.transA ? "true" : "false", inputParams_.transB ? "true" : "false", inputParams_.outDtype,
            fusedOpType);

    return true;
}

bool FusedQuantMatMulUnaryTiling::AnalyzeInputs()
{
    auto x1Shape = context_->GetInputShape(GetX1Idx())->GetOriginShape();
    auto x2Shape = context_->GetInputShape(GetX2Idx())->GetOriginShape();
    auto scaleShape = context_->GetOptionalInputShape(GetScaleIdx());
    auto pertokenShape = context_->GetOptionalInputShape(GetPertokenIdx());
    inputParams_.isPertoken = pertokenShape != nullptr;
    auto biasShape = context_->GetOptionalInputShape(GetBiasIdx());
    inputParams_.hasBias = biasShape != nullptr;
    inputParams_.batchBias = inputParams_.hasBias ? GetBatchSize(biasShape->GetStorageShape()) : 1;
    auto x1ShapeLen = x1Shape.GetDimNum();
    auto x2ShapeLen = x2Shape.GetDimNum();
    if (!CheckShapeInRangeForMandtoryInputs(x1ShapeLen, x2ShapeLen)){
        return false;
    }
    auto x1Inner = x1Shape.GetDim(x1ShapeLen - LAST_FIRST_DIM_INDEX);
    auto x1Outer = x1Shape.GetDim(x1ShapeLen - LAST_SECOND_DIM_INDEX);
    auto x2Inner = x2Shape.GetDim(x2ShapeLen - LAST_FIRST_DIM_INDEX);
    auto x2Outer = x2Shape.GetDim(x2ShapeLen - LAST_SECOND_DIM_INDEX);

    const std::vector<int64_t> dimValueOfMKN = {x1Inner, x1Outer, x2Inner, x2Outer};
    inputParams_.mSize = static_cast<uint64_t>(inputParams_.transA ? x1Inner : x1Outer);
    inputParams_.kSize = static_cast<uint64_t>(inputParams_.transA ? x1Outer : x1Inner);
    inputParams_.nSize = static_cast<uint64_t>(inputParams_.transB ? x2Outer : x2Inner);
    const std::vector<gert::Shape *> mandtoryShape = {&x1Shape, &x2Shape};

    inputParams_.batchA = GetBatchSize(x1Shape);
    inputParams_.batchB = GetBatchSize(x2Shape);
    AnalyzeBatchInfo(context_->GetInputShape(0)->GetOriginShape(), context_->GetInputShape(1)->GetOriginShape());
    OP_TILING_CHECK(!InferOutBatchDim(x1Shape, x2Shape),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Batch dimension can not be broadcasted."), return false);
    if (scaleShape != nullptr && !SetQuantMode(scaleShape->GetStorageShape(), pertokenShape)) {
        return false;
    }
    if (!CheckShape(mandtoryShape, biasShape, pertokenShape, scaleShape, dimValueOfMKN)) {
        return false;
    }
    OP_TILING_CHECK(!CheckOutputShapeAvailable(),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "Multiple of output shape dims should be in boundary of INT64_MAX"),
                                          return false);

    auto isPerTensorStr = inputParams_.isPerTensor ? "true" : "false";
    auto isPertokenStr = inputParams_.isPertoken ? "true" : "false";
    OP_LOGD(inputParams_.opName, "batchA: %lu, batchB: %lu, batchC: %lu, isPerTensor: %s, isPertoken: %s",
            inputParams_.batchA, inputParams_.batchB, inputParams_.batchC, isPerTensorStr, isPertokenStr);
    return true;
}

bool FusedQuantMatMulUnaryTiling::CheckDtype() const
{
    OP_TILING_CHECK(
        !((inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT8) || 
            (inputParams_.aDtype == ge::DT_INT4 && inputParams_.bDtype == ge::DT_INT4)),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "x1 and x2 of FusedQuantMatmul dtype only support Int4 or Int8, that is only \
            supports A8W8 or A4W4 scenario now, actual dtype is x1:%s, x2:%s.", DType2Str(inputParams_.aDtype).c_str(),
            DType2Str(inputParams_.bDtype).c_str()),
        return false);
    
    OP_TILING_CHECK(
        !inputParams_.perTokenScaleDtype == ge::DT_FLOAT,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "x1_scale of FusedQuantMatmul dtype only support Float32 for x1 perToken quant now, actual x1_scale dtype is %s.",
            DType2Str(inputParams_.perTokenScaleDtype).c_str()),
        return false);

    OP_TILING_CHECK(
        !(inputParams_.cDtype == ge::DT_FLOAT16 || inputParams_.cDtype == ge::DT_BF16),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "y of FusedQuantMatmul dtype only support Float16 or BFloat16, actual y dtype is %s.",
            DType2Str(inputParams_.cDtype).c_str()),
        return false);
    
    // 输出是fp16时，x2scale必须是fp32，bias必须是int32/fp16/fp32
    OP_TILING_CHECK(
            inputParams_.cDtype == ge::DT_FLOAT16 && inputParams_.scaleDtype != ge::DT_FLOAT,
            CUBE_INNER_ERR_REPORT(inputParams_.opName, "When y dtype is Float16 in FusedQuantMatmul, x2_scale dtype should be Float32, actual dtype is %s.",
                DType2Str(inputParams_.scaleDtype).c_str()),
            return false);
    OP_TILING_CHECK(
            inputParams_.cDtype == ge::DT_FLOAT16 && context_->GetOptionalInputDesc(GetBiasIdx()) != nullptr &&
            (inputParams_.biasDtype != ge::DT_INT32 && inputParams_.biasDtype != ge::DT_FLOAT16 && inputParams_.biasDtype != ge::DT_FLOAT),
            CUBE_INNER_ERR_REPORT(inputParams_.opName, "When y dtype is Float16 in FusedQuantMatmul and bias is not null, bias dtype should be Int32/Float16/Float32, \
                actual dtype is %s.", DType2Str(inputParams_.biasDtype).c_str()),
            return false);
    
    // 输出是bf16时，x2scale必须是bf16/fp32，bias必须是int32/bf16/fp32
    OP_TILING_CHECK(
            inputParams_.cDtype == ge::DT_BF16 && (inputParams_.scaleDtype != ge::DT_BF16 && inputParams_.scaleDtype != ge::DT_FLOAT),
            CUBE_INNER_ERR_REPORT(inputParams_.opName, "When y dtype is BFloat16 in FusedQuantMatmul, x2_scale dtype should be BFloat16/Float32, actual dtype is %s.",
                DType2Str(inputParams_.scaleDtype).c_str()),
            return false);
    OP_TILING_CHECK(
            inputParams_.cDtype == ge::DT_BF16 && context_->GetOptionalInputDesc(GetBiasIdx()) != nullptr &&
            (inputParams_.biasDtype != ge::DT_INT32 && inputParams_.biasDtype != ge::DT_BF16 && inputParams_.biasDtype != ge::DT_FLOAT),
            CUBE_INNER_ERR_REPORT(inputParams_.opName, "When y dtype is BFloat16 in FusedQuantMatmul and bias is not null, bias dtype should be Int32/BFloat16/Float32, \
                actual dtype is %s.", DType2Str(inputParams_.biasDtype).c_str()),
            return false);
    return true;
}

bool FusedQuantMatMulUnaryTiling::CheckFormat() const
{
    auto x1Desc = context_->GetInputDesc(GetX1Idx());
    auto x1Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x1Desc->GetStorageFormat()));
    OP_TILING_CHECK(
        x1Format != ge::Format::FORMAT_ND,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "X1 format should be ND, actual format is %s.",
            ge::TypeUtils::FormatToSerialString(x1Format).c_str()),
        return false);
    
    auto x2Desc = context_->GetInputDesc(GetX2Idx());
    auto x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x2Desc->GetStorageFormat()));
    OP_TILING_CHECK(
        x2Format != ge::Format::FORMAT_ND && x2Format != ge::Format::FORMAT_FRACTAL_NZ,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "X2 format should be ND/FRACTAL_NZ, actual format is %s.",
            ge::TypeUtils::FormatToSerialString(x2Format).c_str()),
        return false);

    return true;
}

bool FusedQuantMatMulUnaryTiling::CheckShape(const std::vector<gert::Shape *> &mandtoryShape,
                                          const gert::StorageShape *biasShape,
                                          const gert::StorageShape *pertokenShape,
                                          const gert::StorageShape *scaleShape,
                                          const std::vector<int64_t> &dimValueOfMKN) const
{
    auto x1Shape = *mandtoryShape[0]; // using index 0 to get x1Shape
    auto x2Shape = *mandtoryShape[1]; // using index 1 to get x2Shape
    auto x2ScaleShape = scaleShape->GetStorageShape();

    OP_TILING_CHECK(x2ScaleShape.GetDimNum() != 1,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "Only support for x2 scale dimension equals to 1 in FusedQuantMatmul, but actually it is %zu.",
                                          x2ScaleShape.GetDimNum()), return false);

    if (!CheckShapeInRangeForOptionalInputs(biasShape, pertokenShape)){
        return false;
    }
    if (!CheckDimValue(x2ScaleShape, biasShape, pertokenShape, dimValueOfMKN)){
        return false;
    }
    if (!CheckShapeInBoundary(x1Shape, GetX1Idx()) || !CheckShapeInBoundary(x2Shape, GetX2Idx())) {
        return false;
    }
    return true;
}

bool FusedQuantMatMulUnaryTiling::AnalyzeDtype()
{
    inputParams_.aDtype = context_->GetInputDesc(GetX1Idx())->GetDataType();
    auto x2Desc = context_->GetInputDesc(GetX2Idx());
    inputParams_.bDtype = x2Desc->GetDataType();

    auto biasDesc = context_->GetOptionalInputDesc(GetBiasIdx());
    inputParams_.biasDtype = biasDesc != nullptr ? biasDesc->GetDataType() : inputParams_.biasDtype;

    auto pertokenScaleDesc = context_->GetOptionalInputDesc(GetPertokenIdx());
    OP_TILING_CHECK(
        pertokenScaleDesc == nullptr,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "x1_scale of FusedQuantMatmul should not be null for x1 perToken quant."),
        return false);
    inputParams_.perTokenScaleDtype = pertokenScaleDesc->GetDataType();

    auto x2ScaleDesc = context_->GetOptionalInputDesc(GetScaleIdx());
    OP_TILING_CHECK(
        x2ScaleDesc == nullptr,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "x2_scale of FusedQuantMatmul should not be null."),
        return false);
    inputParams_.scaleDtype = x2ScaleDesc->GetDataType();

    auto x2TableDesc = context_->GetOptionalInputDesc(GetX2TableIdx());
    OP_TILING_CHECK(
        x2TableDesc != nullptr,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "current FusedQuantMatmul does not support x2_table, it should be null!"),
        return false);

    inputParams_.cDtype = context_->GetOutputDesc(0)->GetDataType();

    SetFormat();
    // 校验当前x1、x2、x1scale、out的dtype支持情况，根据输出的dtype校验x2scale、bias的dtype
    OP_TILING_CHECK(!CheckDtype(), CUBE_INNER_ERR_REPORT(inputParams_.opName, "CheckDtype failed!"), return false);

    OP_TILING_CHECK(!CheckFormat(), CUBE_INNER_ERR_REPORT(inputParams_.opName, "CheckFormat failed!"), return false);
    return true;
}

uint64_t FusedQuantMatMulUnaryTiling::GetTilingKey() const
{   
    uint64_t trans =
            (static_cast<uint64_t>(inputParams_.transA) << 1) | static_cast<uint64_t>(inputParams_.transB);
    bool isBasicTiling = true;
    uint64_t kernelTemplateType = (static_cast<uint64_t>(isBf16Opt_) << 1) | static_cast<uint64_t>(isBasicTiling);
    uint64_t optionAttrs = 0;
    uint64_t unaryFusedOpType = fusedOpType_;

    return GET_TPL_TILING_KEY(trans, kernelTemplateType, static_cast<uint64_t>(inputParams_.isPertoken), optionAttrs, unaryFusedOpType);
}


} // namespace optiling