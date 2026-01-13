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
 * \file quant_batch_matmul_v3_pergroup_tiling.cc
 * \brief
 */

#include "quant_batch_matmul_v4_pergroup_tiling.h"
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "vector"
#include "error_util.h"
#include "matmul/common/op_host/math_util.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "op_cache_tiling.h"
#include "tiling_base/tiling_key.h"
#include "../../op_kernel/quant_batch_matmul_v4_tiling_key.h"

namespace optiling {

constexpr uint64_t GROUP_MKN_BIT_SIZE = 0xFFFF;
constexpr uint64_t DECODEM = 1024;
constexpr uint64_t L2_REAL_SIZE = 168;  // B4真实的L2Size大小
constexpr uint64_t L2_FAKE_SIZE = 96;   // B4被上层修改后的L2Size大小

const gert::Shape QuantBatchMatmulV4PergroupTiling::GetShape(const size_t index)
{
    return context_->GetInputShape(index)->GetStorageShape();
}

const gert::Shape QuantBatchMatmulV4PergroupTiling::GetOptionShape(const size_t index)
{
    return context_->GetOptionalInputShape(index)->GetStorageShape();
}

ge::graphStatus QuantBatchMatmulV4PergroupTiling::CalcDequantTiling(uint32_t baseM, uint32_t baseN, uint32_t groupSizeK)
{
    // 非对称版本
    // X1Scale * X2Scale @ (X1@X2 - X1@X2Offset)
    // X1Scale BaseM / 2 -> load
    // X2Scale BaseN -> load
    // X2Offset BaseN -> load
    // ub中累加 BaseM * BaseN
    // 计算X1[baseM, groupsizeK] @ X2Offset[baseN]
    // X1@int4 ubCalcM * 256 -> load
    // X1@fp32 ubCalcM * 256
    // rowsum(X1@fp32) ubCalcM *1
    // bcast rowsum ubCalcM * 256
    // mul X2Offset ubCalcM * 256
    // load X1@X2 ubCalcM * 256
    // cast X1@X2 ubCalcM * 256
    // (X1@X2 - X1@X2Offset) : ubCalcM * 256

    uint64_t ubSize = aicoreParams_.ubSize;
    // 常驻L1
    uint64_t elesize = ubSize / sizeof(float);
    uint64_t AivM = ops::CeilDiv(baseM, 2U);
    // K方向累加结果常驻L1
    elesize -= (AivM) * baseN;
    // X1 scale [AivM] brcb [AivM, 8]
    uint64_t align8 = 8UL;
    elesize -= ops::CeilAlign(AivM, align8);
    elesize -= ops::CeilAlign(AivM, align8) * align8;

    // 非对齐前处理方案 A@X2offset 常驻L1
    // N 方向不切
    uint32_t ubCalcN = baseN;
    // bias baseN X2 Scale baseN X2 offset FP32 baseN X2 offset BF16 BASEN
    float_t numUbCalcN = 3.5;
    elesize -= static_cast<uint64_t>(numUbCalcN * ubCalcN);
    elesize -= ops::CeilAlign(AivM, align8);
    // MulX2Scale ubCalcM*baseN mmout ubCalcM*baseN CastFP32 ubCalcM*baseN
    // X1 @ INT4 X1 @ FP32 ubCalcM * groupsizeK
    // X1 @ Half ubCalcM * groupsizeK // 2
    // rowsum ubCalcM
    // bcast ubCalcM * baseN
    // Mul ubCalcM * baseN
    // 2MN + MK + M + 2MN = ubres
    float_t numGroupSizeK = 1.5;
    uint32_t ubCalcM = elesize / (4 * ubCalcN + numGroupSizeK * groupSizeK);
    tilingData_.params.ubCalcN = ubCalcN;
    tilingData_.params.ubCalcM = ubCalcM;
    OP_LOGD(inputParams_.opName, "UbTiling ubCalcM: %u,  ubCalcM: %u", ubCalcM, ubCalcN);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV4PergroupTiling::DoOpTiling()
{
    isUbQuant_ = true;
    InitCompileInfo();
    SetTransAttr(trans_);
    OP_LOGE_IF(!SetPlatformInfoForTiling(), ge::GRAPH_FAILED, inputParams_.opName, "SetPlatformInfoForTiling fail");
    OP_LOGE_IF(!DoBasicTiling(), ge::GRAPH_FAILED, inputParams_.opName, "DoBasicTiling failed.");\

    constexpr uint32_t BASE_M = 128;
    constexpr uint32_t BASE_N = 256;
    constexpr uint32_t DEPTH_A1 = 8;
    constexpr uint32_t DEPTH_B1 = 8;
    constexpr uint32_t STEP_KA = 4;
    constexpr uint32_t STEP_KB = 4;
    constexpr uint32_t DB_L0C = 1;
    constexpr uint32_t DB_L0A = 2;
    constexpr uint32_t DB_L0B = 2;

    basicTiling_.baseM = BASE_M;
    basicTiling_.baseN = BASE_N;
    basicTiling_.baseK = inputParams_.groupSizeK;
    basicTiling_.depthA1 = DEPTH_A1;
    basicTiling_.depthB1 = DEPTH_B1;
    basicTiling_.stepKa = STEP_KA;
    basicTiling_.stepKb = STEP_KB;
    basicTiling_.dbL0c = DB_L0C;
    tilingData_.matmulTiling.dbL0A = DB_L0A;
    tilingData_.matmulTiling.dbL0B = DB_L0B;

    QuantBatchMatmulV3BasicTiling::DoL2CacheTiling();
    if (basicTiling_.mTileCntl2 == 1UL && basicTiling_.nTileCntl2 == 1UL) {
        basicTiling_.mTileBlock = ops::CeilDiv(inputParams_.mSize, basicTiling_.baseM);
        basicTiling_.nTileBlock = ops::CeilDiv(inputParams_.nSize, basicTiling_.baseN);
    }
    basicTiling_.usedCoreNum = std::min(basicTiling_.usedCoreNum, basicTiling_.mTileBlock * basicTiling_.nTileBlock);
    tilingData_.params.groupSizeK = inputParams_.groupSizeK;
    return CalcDequantTiling(basicTiling_.baseM, basicTiling_.baseN, inputParams_.groupSizeK);
}

uint64_t QuantBatchMatmulV4PergroupTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(/*not trans*/0, static_cast<uint64_t>(QuantBatchMatmulV4QuantType::MX),
                              /*hasAntiQuantOffset*/0, /*weightNz*/0,
                              static_cast<uint64_t>(KernelTemplateType::PERGROUP_BASIS));
}

ge::graphStatus QuantBatchMatmulV4PergroupTiling::CheckContext()
{
    // a4w4 pergroup
    // 要求传入x1scale x2scale/offset
    auto x1Shape = context_->GetInputShape(X1_IDX);
    auto x1Desc = context_->GetInputDesc(X1_IDX);
    auto x2Shape = context_->GetInputShape(X2_IDX);
    auto x2Desc = context_->GetInputDesc(X2_IDX);
    auto x1ScaleShape = context_->GetOptionalInputShape(X1_SCALE_IDX);
    auto x1ScaleDesc = context_->GetOptionalInputDesc(X1_SCALE_IDX);
    auto x2ScaleShape = context_->GetOptionalInputShape(X2_SCALE_IDX);
    auto x2ScaleDesc = context_->GetOptionalInputDesc(X2_SCALE_IDX);
    auto x2OffsetShape = context_->GetOptionalInputShape(X2_OFFSET_IDX);
    auto x2OffsetDesc = context_->GetOptionalInputDesc(X2_OFFSET_IDX);
    auto outputShape = context_->GetOutputShape(Y_OUTPUT_IDX);
    auto outputDesc = context_->GetOutputDesc(Y_OUTPUT_IDX);
    auto attrs = context_->GetAttrs();

    OP_TILING_CHECK(
        attrs == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Function context_.GetAttrs() failed!"),
        return ge::GRAPH_FAILED);

    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Desc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2Desc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1ScaleShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1ScaleDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2ScaleShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2ScaleDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2OffsetShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2OffsetDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());

    return ge::GRAPH_SUCCESS;
}

bool QuantBatchMatmulV4PergroupTiling::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    const int64_t* groupSizePtr = attrs->GetAttrPointer<int64_t>(GROUP_SIZE_IDX);
    OP_TILING_CHECK(
        groupSizePtr == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Group size can not be nullptr."),
        return false);
    inputParams_.groupSize = static_cast<uint64_t>(*groupSizePtr);
    inputParams_.groupSizeK = inputParams_.groupSize & GROUP_MKN_BIT_SIZE;
    inputParams_.groupSizeN =
        (inputParams_.groupSize >> 16U) & GROUP_MKN_BIT_SIZE; // 16 is the bit size of MKN group size
    inputParams_.groupSizeM =
        (inputParams_.groupSize >> 32U) & GROUP_MKN_BIT_SIZE; // groupSizeM start at 32 bit of groupSize
    OP_TILING_CHECK(
        inputParams_.groupSizeK != 256,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName,
            "Group size only support group k 256 on the A4W4 scenario, actual group sizeN is [%lu]",
            inputParams_.groupSize),
        return false);
    auto transposeX1 = attrs->GetAttrPointer<bool>(TRANSPOSE_X1_IDX);
    auto transposeX2 = attrs->GetAttrPointer<bool>(TRANSPOSE_X2_IDX);
    inputParams_.transA = transposeX1 != nullptr && *transposeX1;
    inputParams_.transB = transposeX2 != nullptr && *transposeX2;

    return true;
}

bool QuantBatchMatmulV4PergroupTiling::AnalyzeDtype()
{
    inputParams_.aDtype = context_->GetInputDesc(X1_IDX)->GetDataType();
    inputParams_.bDtype = context_->GetInputDesc(X2_IDX)->GetDataType();
    inputParams_.perTokenScaleDtype = context_->GetOptionalInputDesc(X1_SCALE_IDX)->GetDataType();
    inputParams_.scaleDtype = context_->GetOptionalInputDesc(X2_SCALE_IDX)->GetDataType();
    inputParams_.cDtype = context_->GetOutputDesc(Y_OUTPUT_IDX)->GetDataType();
    inputParamsPergroup_.x2OffsetDtype = context_->GetOptionalInputDesc(X2_OFFSET_IDX)->GetDataType();

    OP_TILING_CHECK(
        inputParams_.aDtype != ge::DT_INT4,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "Input dtype of x1 should be int4, actual dtype is %s",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str()),
        return false);
    OP_TILING_CHECK(
        inputParams_.bDtype != ge::DT_INT4,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "Input dtype of x2 should be int4, actual dtype is %s",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
        return false);
    OP_TILING_CHECK(
        inputParams_.perTokenScaleDtype != ge::DT_FLOAT,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "Input dtype of x1Scale should be float32, actual dtype is %s",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str()),
        return false);
    OP_TILING_CHECK(
        inputParams_.scaleDtype != ge::DT_FLOAT,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "Input dtype of x2Scale should be float32, actual dtype is %s",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str()),
        return false);
    OP_TILING_CHECK(
        !(inputParams_.cDtype == ge::DT_FLOAT16 || inputParams_.cDtype == ge::DT_BF16),
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "Output dtype should be float16 or bfloat16, actual dtype is %s",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str()),
        return false);
    OP_TILING_CHECK(
        inputParamsPergroup_.x2OffsetDtype != ge::DT_FLOAT16,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "Input dtype of x2Offset should be float16, actual dtype is %s",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str()),
        return false);

    return true;
}

bool QuantBatchMatmulV4PergroupTiling::CheckInputLen(size_t actualLen, size_t expectedLen, const string& inputParamName) {
    OP_TILING_CHECK(
        actualLen != expectedLen,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "Input %s dimension should equal to %zu, but it is %zu actually", inputParamName.c_str(),
            expectedLen, actualLen),
        return false);
    return true;
}

bool QuantBatchMatmulV4PergroupTiling::CheckInputsLen(
    const gert::Shape& x1Shape, const gert::Shape& x2Shape, const gert::Shape& x1ScaleShape,
    const gert::Shape& x2ScaleShape, const gert::Shape& x2OffsetShape)
{
    auto x1ShapeLen = x1Shape.GetDimNum();
    auto x2ShapeLen = x2Shape.GetDimNum();
    auto x1ScaleShapeLen = x1ScaleShape.GetDimNum();
    auto x2ScaleShapeLen = x2ScaleShape.GetDimNum();
    auto x2OffsetShapeLen = x2OffsetShape.GetDimNum();

    size_t expectedX1Len = 2;
    size_t expectedX2Len = 2;
    size_t expectedX1ScaleLen = 2;
    size_t expectedX2ScaleLen = 2;
    size_t expectedX2OffsetLen = 2;
    CheckInputLen(x1ShapeLen, expectedX1Len, "x1");
    CheckInputLen(x2ShapeLen, expectedX2Len, "x2");
    CheckInputLen(x1ScaleShapeLen, expectedX1ScaleLen, "x1Scale");
    CheckInputLen(x2ScaleShapeLen, expectedX2ScaleLen, "x2Scale");
    CheckInputLen(x2OffsetShapeLen, expectedX2OffsetLen, "x2Offset");

    return true;
}

bool QuantBatchMatmulV4PergroupTiling::CheckInputsShape(
    const gert::Shape& x1Shape, const gert::Shape& x2Shape, const gert::Shape& x1ScaleShape,
    const gert::Shape& x2ScaleShape, const gert::Shape& x2OffsetShape)
{
    CheckInputsLen(x1Shape, x2Shape, x1ScaleShape, x2ScaleShape, x2OffsetShape);
    int64_t m = inputParams_.mSize;
    int64_t k = inputParams_.kSize;
    int64_t n = inputParams_.nSize;
    int64_t groupSizeK = 256;
    int64_t nkgroup = ops::CeilDiv(k, groupSizeK);

    auto x1ScaleInner = x1ScaleShape.GetDim(1);
    auto x1ScaleOuter = x1ScaleShape.GetDim(0);
    OP_TILING_CHECK(
        x1ScaleInner != 1 || x1ScaleOuter != m,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "Input x1Scale's shape should be [%ld, 1], but actual shape is [%ld, %ld]", m,
            x1ScaleOuter, x1ScaleInner),
        return false);

    auto x2ScaleInner = x2ScaleShape.GetDim(1);
    auto x2ScaleOuter = x2ScaleShape.GetDim(0);
    OP_TILING_CHECK(
        x2ScaleInner != n || x2ScaleOuter != nkgroup,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "Input x1Scale's shape should be [%ld, %ld], but actual shape is [%ld, %ld]", nkgroup,
            n, x2ScaleOuter, x2ScaleInner),
        return false);

    auto x2OffsetInner = x2OffsetShape.GetDim(1);
    auto x2OffsetOuter = x2OffsetShape.GetDim(0);
    OP_TILING_CHECK(
        x2OffsetInner != n || x2OffsetOuter != nkgroup,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "Input x1Scale's shape should be [%ld, %ld], but actual shape is [%ld, %ld]", nkgroup,
            n, x2OffsetOuter, x2OffsetInner),
        return false);

    return true;
}

bool QuantBatchMatmulV4PergroupTiling::CheckFormat() {
    auto x1Format = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(X1_IDX)->GetStorageFormat()));
    OP_TILING_CHECK(x1Format != ge::FORMAT_ND, VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "x1Format Only support Nd"), return false);
    auto x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(X2_IDX)->GetStorageFormat()));
    OP_TILING_CHECK(x2Format != ge::FORMAT_ND, VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "x2Format Only support Nd"), return false);
    auto x1ScaleFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetOptionalInputDesc(X1_SCALE_IDX)->GetStorageFormat()));
    OP_TILING_CHECK(x1ScaleFormat != ge::FORMAT_ND, VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "x1ScaleFormat Only support Nd"), return false);
    auto x2ScaleFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetOptionalInputDesc(X2_SCALE_IDX)->GetStorageFormat()));
    OP_TILING_CHECK(x2ScaleFormat != ge::FORMAT_ND, VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "x2ScaleFormat Only support Nd"), return false);
    auto x2OffsetFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetOptionalInputDesc(X2_OFFSET_IDX)->GetStorageFormat()));
    OP_TILING_CHECK(x2OffsetFormat != ge::FORMAT_ND, VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "x2OffsetFormat Only support Nd"), return false);

    return true;
}

bool QuantBatchMatmulV4PergroupTiling::AnalyzeInputs()
{
    auto x1Shape = GetShape(X1_IDX);
    auto x2Shape = GetShape(X2_IDX);
    auto x1ScaleShape = GetOptionShape(X1_SCALE_IDX);
    auto x2ScaleShape = GetOptionShape(X2_SCALE_IDX);
    auto x2OffsetShape = GetOptionShape(X2_OFFSET_IDX);

    auto biasShapePtr = GetBiasShape(BIAS_IDX);
    inputParams_.hasBias = (biasShapePtr != nullptr);
    OP_TILING_CHECK(
        inputParams_.hasBias, CUBE_INNER_ERR_REPORT(inputParams_.opName, "Bias should be null right now."), return false);

    auto x1Inner = x1Shape.GetDim(1);
    auto x1Outer = x1Shape.GetDim(0);
    auto x2Inner = x2Shape.GetDim(1);
    auto x2Outer = x2Shape.GetDim(0);
    const std::vector<int64_t> dimValueOfMKN = {x1Inner, x1Outer, x2Inner, x2Outer};
    inputParams_.mSize = static_cast<uint64_t>(inputParams_.transA ? x1Inner : x1Outer); // transA [k, m]
    inputParams_.kSize = static_cast<uint64_t>(inputParams_.transA ? x1Outer : x1Inner);
    inputParams_.nSize = static_cast<uint64_t>(inputParams_.transB ? x2Outer : x2Inner);
    inputParams_.batchA = GetBatchSize(x1Shape);
    inputParams_.batchB = GetBatchSize(x2Shape);
    inputParamsPergroup_.antiQuantType = QuantBatchMatmulV4QuantType::PER_GROUP;

    int ALIGN1024 = 1024;
    int ALIGN256 = 256;
    OP_TILING_CHECK(
        inputParams_.kSize % ALIGN1024 != 0,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "k should align with 1024."), return false);
    OP_TILING_CHECK(
        inputParams_.nSize % ALIGN256 != 0,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "n should align with 256."), return false);

    AnalyzeBatchInfo(context_->GetInputShape(0)->GetOriginShape(), context_->GetInputShape(1)->GetOriginShape());
    OP_TILING_CHECK(
        !InferOutBatchDim(x1Shape, x2Shape),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Batch dimension can not be broadcasted."), return false);
    OP_TILING_CHECK(
        !CheckOutputShapeAvailable(),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Multiple of output shape dims should be in boundary of INT64_MAX."),
        return false);
    OP_TILING_CHECK(
        !CheckInputsShape(x1Shape, x2Shape, x1ScaleShape, x2ScaleShape, x2OffsetShape),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "There is at least one input with wrong dim."), return false);
    OP_TILING_CHECK(!CheckFormat(), CUBE_INNER_ERR_REPORT(inputParams_.opName, "There is at least one format wrong."), return false);

    return true;
}

bool QuantBatchMatmulV4PergroupTiling::SetPlatformInfoForTiling()
{
    if (!compileInfoInit_) {
        InitCompileInfo();
    }
    OP_LOGE_IF(compileInfo_.aicNum <= 0, false, inputParams_.opName, "coreNum <= 0");
    aicoreParams_.aicNum = compileInfo_.aicNum;
    OP_LOGE_IF(compileInfo_.l2Size <= 0, false, inputParams_.opName, "l2Size <= 0");
    // 纠正L2实际物理大小
    compileInfo_.l2Size =
        compileInfo_.l2Size == L2_FAKE_SIZE * MB_SIZE ? L2_REAL_SIZE * MB_SIZE : compileInfo_.l2Size;
    inputParams_.libApiWorkSpaceSize = compileInfo_.workspaceNum;
    aicoreParams_.ubSize = compileInfo_.ubSize;
    aicoreParams_.l1Size = compileInfo_.l1Size;
    aicoreParams_.l0aSize = compileInfo_.l0aSize;
    aicoreParams_.l0cSize = compileInfo_.l0cSize;
    aicoreParams_.blockDim = 0;
    return true;
}

} // namespace optiling