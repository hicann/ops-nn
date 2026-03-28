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
 * \file quant_batch_matmul_v4_pergroup_arch35_tiling.cpp
 * \brief
 */

#include "quant_batch_matmul_v4_pergroup_arch35_tiling.h"
#include "../../../op_kernel/arch35/quant_batch_matmul_v4_tiling_key.h"
#include <algorithm>

#include "error_util.h"
#include "graph/utils/type_utils.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "platform/platform_infos_def.h"

namespace optiling {

namespace {
constexpr uint64_t GROUP_MKN_BIT_SIZE = 0xFFFF;
constexpr int ALIGN1024 = 1024;
constexpr int ALIGN256 = 256;
constexpr int64_t GROUP_SIZE_K = 256;
constexpr uint64_t L2_REAL_SIZE = 168; // B4真实的L2Size大小
constexpr uint64_t L2_FAKE_SIZE = 96;  // B4被上层修改后的L2Size大小
constexpr const char* INT4_KG_QUANT_MODE = "int4 K-G quantification";
} // namespace

const gert::Shape QuantBatchMatmulV4PergroupArch35Tiling::GetShape(const size_t index)
{
    return context_->GetInputShape(index)->GetStorageShape();
}

const gert::Shape QuantBatchMatmulV4PergroupArch35Tiling::GetOptionShape(const size_t index)
{
    return context_->GetOptionalInputShape(index)->GetStorageShape();
}

ge::graphStatus QuantBatchMatmulV4PergroupArch35Tiling::CalcDequantTiling(
    uint32_t baseM, uint32_t baseN, uint32_t groupSizeK)
{
    uint64_t ubSize = aicoreParams_.ubSize;
    uint64_t elesize = ubSize / sizeof(float);
    uint64_t aivM = ops::CeilDiv(baseM, 2U);
    constexpr uint64_t ALIGN8 = 8UL;

    elesize -= aivM * baseN;
    elesize -= ops::CeilAlign(aivM, ALIGN8);
    elesize -= ops::CeilAlign(aivM, ALIGN8) * ALIGN8;

    uint32_t ubCalcN = baseN;
    constexpr double NUM_UB_CALC_N_FACTOR = 3.5;
    elesize -= static_cast<uint64_t>(NUM_UB_CALC_N_FACTOR * ubCalcN);
    elesize -= ops::CeilAlign(aivM, ALIGN8);

    constexpr double NUM_GROUP_SIZE_K_FACTOR = 1.5;
    uint32_t ubCalcM = elesize / (4U * ubCalcN + static_cast<uint32_t>(NUM_GROUP_SIZE_K_FACTOR * groupSizeK));
    tilingData_.params.ubCalcN = ubCalcN;
    tilingData_.params.ubCalcM = ubCalcM;
    OP_LOGD(inputParams_.opName, "UbTiling ubCalcM: %u, ubCalcN: %u, groupSizeK: %u", ubCalcM, ubCalcN, groupSizeK);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV4PergroupArch35Tiling::DoOpTiling()
{
    isUbQuant_ = true;
    InitCompileInfo();
    SetTransAttr(trans_);
    OP_LOGE_IF(!SetPlatformInfoForTiling(), ge::GRAPH_FAILED, inputParams_.opName, "SetPlatformInfoForTiling fail");

    auto* platformInfoPtr = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    auto savedADtype = inputParams_.aDtype;
    auto savedBDtype = inputParams_.bDtype;

    inputParams_.aDtype = ge::DT_INT8;
    inputParams_.bDtype = ge::DT_INT8;

    OP_LOGE_IF(!DoBasicTiling(), ge::GRAPH_FAILED, inputParams_.opName, "DoBasicTiling failed.");

    constexpr uint32_t BASE_M = 128;
    constexpr uint32_t BASE_N = 256;
    constexpr uint32_t DEPTH_A1 = 8;
    constexpr uint32_t DEPTH_B1 = 8;
    constexpr uint32_t DB_L0C = 1;
    constexpr uint32_t DB_L0A = 2;
    constexpr uint32_t DB_L0B = 1;

    basicTiling_.baseM = BASE_M;
    basicTiling_.baseN = BASE_N;
    basicTiling_.depthA1 = DEPTH_A1;
    basicTiling_.depthB1 = DEPTH_B1;
    basicTiling_.dbL0c = DB_L0C;
    tilingData_.matmulTiling.dbL0A = DB_L0A;
    tilingData_.matmulTiling.dbL0B = DB_L0B;

    uint32_t l1Size = aicoreParams_.l1Size;
    uint32_t stepK = l1Size / (2U * (BASE_M + BASE_N) * inputParams_.groupSizeK);
    stepK = std::max(1U, stepK);
    basicTiling_.stepKa = stepK;
    basicTiling_.stepKb = stepK;
    OP_LOGD(
        inputParams_.opName, "arch35 int8 tiling: groupSizeK=%u, stepK=%u, l1Size=%u", inputParams_.groupSizeK, stepK,
        l1Size);

    basicTiling_.baseK = inputParams_.groupSizeK;
    QuantBatchMatmulV3BasicTiling::DoL2CacheTiling();

    inputParams_.aDtype = savedADtype;
    inputParams_.bDtype = savedBDtype;
    if (basicTiling_.mTileCntl2 == 1UL && basicTiling_.nTileCntl2 == 1UL) {
        basicTiling_.mTileBlock = ops::CeilDiv(inputParams_.mSize, basicTiling_.baseM);
        basicTiling_.nTileBlock = ops::CeilDiv(inputParams_.nSize, basicTiling_.baseN);
    }
    basicTiling_.usedCoreNum = std::min(basicTiling_.usedCoreNum, basicTiling_.mTileBlock * basicTiling_.nTileBlock);
    tilingData_.params.groupSizeK = inputParams_.groupSizeK;
    return CalcDequantTiling(basicTiling_.baseM, basicTiling_.baseN, inputParams_.groupSizeK);
}

ge::graphStatus QuantBatchMatmulV4PergroupArch35Tiling::PostTiling()
{
    ge::graphStatus ret = QuantBatchMatmulV3BasicTiling::PostTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV4PergroupArch35Tiling::CheckContext()
{
    auto x1Shape = context_->GetInputShape(X1_IDX);
    auto x1Desc = context_->GetInputDesc(X1_IDX);
    auto x2Shape = context_->GetInputShape(X2_IDX);
    auto x2Desc = context_->GetInputDesc(X2_IDX);
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
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    OP_TILING_CHECK(
        context_->GetRawTilingData()->GetCapacity() < tilingDataSize_,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName, "context tiling data capacity %zu < actual tiling data size %zu.",
            context_->GetRawTilingData()->GetCapacity(), tilingDataSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV4PergroupArch35Tiling::GetShapeAttrsInfo()
{
    tilingDataSize_ = sizeof(QuantBatchMatmulV3TilingData);
    if (context_ == nullptr) {
        return ge::GRAPH_FAILED;
    }

    inputParams_.opName = context_->GetNodeName();
    OPS_LOG_D(inputParams_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());

    if (CheckContext() != ge::GRAPH_SUCCESS) {
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Invalid context.");
        return ge::GRAPH_FAILED;
    }

    auto x1ScaleShape = context_->GetOptionalInputShape(X1_SCALE_IDX);
    auto x1ScaleDesc = context_->GetOptionalInputDesc(X1_SCALE_IDX);
    auto x2ScaleShape = context_->GetOptionalInputShape(X2_SCALE_IDX);
    auto x2ScaleDesc = context_->GetOptionalInputDesc(X2_SCALE_IDX);
    auto x2OffsetShape = context_->GetOptionalInputShape(X2_OFFSET_IDX);
    auto x2OffsetDesc = context_->GetOptionalInputDesc(X2_OFFSET_IDX);
    if (x1ScaleShape == nullptr || x1ScaleDesc == nullptr || x2ScaleShape == nullptr || x2ScaleDesc == nullptr ||
        x2OffsetShape == nullptr || x2OffsetDesc == nullptr) {
        OP_LOGD(inputParams_.opName, "x1Scale/x2Scale/x2Offset is nullptr, skip this tiling template.");
        return ge::GRAPH_PARAM_INVALID;
    }

    if (!AnalyzeAttrs() || !AnalyzeDtype() || !AnalyzeInputs()) {
        OP_LOGD(inputParams_.opName, "Fail to analyze context info.");
        return ge::GRAPH_PARAM_INVALID;
    }

    return ge::GRAPH_SUCCESS;
}

bool QuantBatchMatmulV4PergroupArch35Tiling::IsCapable()
{
    auto* platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return false;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    if (ascendcPlatform.GetCurNpuArch() != NpuArch::DAV_3510) {
        return false;
    }
    if (!CheckPergroupAttrs()) {
        return false;
    }
    if (!CheckPergroupDtype()) {
        return false;
    }
    if (!CheckPergroupShape()) {
        return false;
    }
    return true;
}

bool QuantBatchMatmulV4PergroupArch35Tiling::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    auto transposeX1 = attrs->GetAttrPointer<bool>(TRANSPOSE_X1_IDX);
    auto transposeX2 = attrs->GetAttrPointer<bool>(TRANSPOSE_X2_IDX);
    const int64_t* groupSizePtr = attrs->GetAttrPointer<int64_t>(GROUP_SIZE_IDX);

    inputParams_.groupSize = groupSizePtr != nullptr ? static_cast<uint64_t>(*groupSizePtr) : 0UL;
    inputParams_.groupSizeK = inputParams_.groupSize & GROUP_MKN_BIT_SIZE;
    inputParams_.groupSizeN = (inputParams_.groupSize >> 16U) & GROUP_MKN_BIT_SIZE;
    inputParams_.groupSizeM = (inputParams_.groupSize >> 32U) & GROUP_MKN_BIT_SIZE;
    inputParams_.transA = transposeX1 != nullptr && *transposeX1;
    inputParams_.transB = transposeX2 != nullptr && *transposeX2;
    return true;
}

bool QuantBatchMatmulV4PergroupArch35Tiling::AnalyzeDtype()
{
    inputParams_.aDtype = context_->GetInputDesc(X1_IDX)->GetDataType();
    inputParams_.bDtype = context_->GetInputDesc(X2_IDX)->GetDataType();

    auto x1ScaleDesc = context_->GetOptionalInputDesc(X1_SCALE_IDX);
    auto x2ScaleDesc = context_->GetOptionalInputDesc(X2_SCALE_IDX);
    auto x2OffsetDesc = context_->GetOptionalInputDesc(X2_OFFSET_IDX);

    inputParams_.perTokenScaleDtype = x1ScaleDesc != nullptr ? x1ScaleDesc->GetDataType() : ge::DT_UNDEFINED;
    inputParams_.scaleDtype = x2ScaleDesc != nullptr ? x2ScaleDesc->GetDataType() : ge::DT_UNDEFINED;
    inputParamsPergroup_.x2OffsetDtype = x2OffsetDesc != nullptr ? x2OffsetDesc->GetDataType() : ge::DT_UNDEFINED;
    inputParams_.cDtype = context_->GetOutputDesc(Y_OUTPUT_IDX)->GetDataType();
    return true;
}

bool QuantBatchMatmulV4PergroupArch35Tiling::AnalyzeInputs()
{
    auto x1Shape = GetShape(X1_IDX);
    auto x2Shape = GetShape(X2_IDX);

    if (x1Shape.GetDimNum() != 2 || x2Shape.GetDimNum() != 2) {
        OP_LOGD(inputParams_.opName, "Input dims of x1/x2 should be 2 for this pergroup template.");
        return false;
    }

    auto x1Inner = x1Shape.GetDim(1);
    auto x1Outer = x1Shape.GetDim(0);
    auto x2Inner = x2Shape.GetDim(1);
    auto x2Outer = x2Shape.GetDim(0);

    inputParams_.mSize = static_cast<uint64_t>(inputParams_.transA ? x1Inner : x1Outer);
    inputParams_.kSize = static_cast<uint64_t>(inputParams_.transA ? x1Outer : x1Inner);
    inputParams_.nSize = static_cast<uint64_t>(inputParams_.transB ? x2Outer : x2Inner);
    inputParams_.batchA = GetBatchSize(x1Shape);
    inputParams_.batchB = GetBatchSize(x2Shape);

    auto biasShapePtr = GetBiasShape(BIAS_IDX);
    inputParams_.hasBias = (biasShapePtr != nullptr);

    AnalyzeBatchInfo(context_->GetInputShape(0)->GetOriginShape(), context_->GetInputShape(1)->GetOriginShape());
    (void)InferOutBatchDim(x1Shape, x2Shape);
    return true;
}

bool QuantBatchMatmulV4PergroupArch35Tiling::SetPlatformInfoForTiling()
{
    if (!compileInfoInit_) {
        InitCompileInfo();
    }
    OP_LOGE_IF(compileInfo_.aicNum <= 0, false, inputParams_.opName, "coreNum <= 0");
    aicoreParams_.aicNum = compileInfo_.aicNum;
    OP_LOGE_IF(compileInfo_.l2Size <= 0, false, inputParams_.opName, "l2Size <= 0");
    compileInfo_.l2Size = compileInfo_.l2Size == L2_FAKE_SIZE * MB_SIZE ? L2_REAL_SIZE * MB_SIZE : compileInfo_.l2Size;
    inputParams_.libApiWorkSpaceSize = compileInfo_.workspaceNum;
    aicoreParams_.ubSize = compileInfo_.ubSize;
    aicoreParams_.l1Size = compileInfo_.l1Size;
    aicoreParams_.l0aSize = compileInfo_.l0aSize;
    aicoreParams_.l0cSize = compileInfo_.l0cSize;
    aicoreParams_.blockDim = 0;
    return true;
}

bool QuantBatchMatmulV4PergroupArch35Tiling::CheckPergroupAttrs() const
{
    auto attrs = context_->GetAttrs();
    const int64_t* groupSizePtr = attrs->GetAttrPointer<int64_t>(GROUP_SIZE_IDX);
    if (groupSizePtr == nullptr) {
        OP_LOGD(inputParams_.opName, "In %s, groupSize is required, but got nullptr.", INT4_KG_QUANT_MODE);
        return false;
    }
    if (inputParams_.groupSizeK != GROUP_SIZE_K) {
        OP_LOGD(
            inputParams_.opName,
            "In %s, expected groupSizeK == %ld (groupSizeK = groupSize & 0xFFFF), but got %lu, full groupSize=%lu.",
            INT4_KG_QUANT_MODE, GROUP_SIZE_K, inputParams_.groupSizeK, inputParams_.groupSize);
        return false;
    }
    if (inputParams_.transA != false ||  inputParams_.transB != true) {
        OP_LOGD(
            inputParams_.opName,
            "In %s, only support transA=false and transB=true, but got transA=%s, transB=%s.",
            INT4_KG_QUANT_MODE, inputParams_.transA ? "true" : "false", inputParams_.transB ? "true" : "false");
        return false;
    }
    return true;
}

bool QuantBatchMatmulV4PergroupArch35Tiling::CheckPergroupDtype() const
{
    if (inputParams_.aDtype != ge::DT_INT4) {
        OP_LOGD(
            inputParams_.opName, "In %s, expected x1 dtype=int4, but got %s.", INT4_KG_QUANT_MODE,
            ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str());
        return false;
    }
    if (inputParams_.bDtype != ge::DT_INT4) {
        OP_LOGD(
            inputParams_.opName, "In %s, expected x2 dtype=int4, but got %s.", INT4_KG_QUANT_MODE,
            ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str());
        return false;
    }
    if (inputParams_.perTokenScaleDtype != ge::DT_FLOAT) {
        OP_LOGD(
            inputParams_.opName, "In %s, expected x1Scale dtype=float32, but got %s.", INT4_KG_QUANT_MODE,
            ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str());
        return false;
    }
    if (inputParams_.scaleDtype != ge::DT_FLOAT) {
        OP_LOGD(
            inputParams_.opName, "In %s, expected x2Scale dtype=float32, but got %s.", INT4_KG_QUANT_MODE,
            ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str());
        return false;
    }
    if (!(inputParams_.cDtype == ge::DT_FLOAT16 || inputParams_.cDtype == ge::DT_BF16)) {
        OP_LOGD(
            inputParams_.opName, "In %s, expected output dtype in {float16, bfloat16}, but got %s.",
            INT4_KG_QUANT_MODE,
            ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str());
        return false;
    }
    if (inputParamsPergroup_.x2OffsetDtype != ge::DT_FLOAT16) {
        OP_LOGD(
            inputParams_.opName, "In %s, expected x2Offset dtype=float16, but got %s.", INT4_KG_QUANT_MODE,
            ge::TypeUtils::DataTypeToSerialString(inputParamsPergroup_.x2OffsetDtype).c_str());
        return false;
    }
    return true;
}

bool QuantBatchMatmulV4PergroupArch35Tiling::CheckPergroupShape()
{
    // This template only handles int4 K-G quantification path on arch35.
    if (!CheckPergroupBasicShapeConstraints()) {
        return false;
    }

    auto x1Shape = GetShape(X1_IDX);
    auto x2Shape = GetShape(X2_IDX);
    auto x1ScaleShape = GetOptionShape(X1_SCALE_IDX);
    auto x2ScaleShape = GetOptionShape(X2_SCALE_IDX);
    auto x2OffsetShape = GetOptionShape(X2_OFFSET_IDX);

    return CheckPergroupDimAndOutput(x1Shape, x2Shape, x1ScaleShape, x2ScaleShape, x2OffsetShape) &&
           CheckPergroupScaleShape(x1ScaleShape, x2ScaleShape, x2OffsetShape) && CheckPergroupInputFormat();
}

bool QuantBatchMatmulV4PergroupArch35Tiling::CheckPergroupBasicShapeConstraints() const
{
    if (inputParams_.hasBias) {
        OP_LOGD(inputParams_.opName, "In %s, expected bias=null, but got non-null bias.", INT4_KG_QUANT_MODE);
        return false;
    }
    if (inputParams_.kSize % ALIGN1024 != 0) {
        OP_LOGD(
            inputParams_.opName, "In %s, expected k aligned to %d, but got k=%lu.", INT4_KG_QUANT_MODE, ALIGN1024,
            inputParams_.kSize);
        return false;
    }
    if (inputParams_.nSize % ALIGN256 != 0) {
        OP_LOGD(
            inputParams_.opName, "In %s, expected n aligned to %d, but got n=%lu.", INT4_KG_QUANT_MODE, ALIGN256,
            inputParams_.nSize);
        return false;
    }
    return true;
}

bool QuantBatchMatmulV4PergroupArch35Tiling::CheckPergroupDimAndOutput(
    const gert::Shape& x1Shape, const gert::Shape& x2Shape, const gert::Shape& x1ScaleShape,
    const gert::Shape& x2ScaleShape, const gert::Shape& x2OffsetShape)
{
    if (x1Shape.GetDimNum() != 2 || x2Shape.GetDimNum() != 2 || x1ScaleShape.GetDimNum() != 2 ||
        x2ScaleShape.GetDimNum() != 2 || x2OffsetShape.GetDimNum() != 2) {
        OP_LOGD(
            inputParams_.opName,
            "In %s, expected dims=2 for x1/x2/x1Scale/x2Scale/x2Offset, but got [%zu, %zu, %zu, %zu, %zu].",
            INT4_KG_QUANT_MODE, x1Shape.GetDimNum(), x2Shape.GetDimNum(), x1ScaleShape.GetDimNum(),
            x2ScaleShape.GetDimNum(), x2OffsetShape.GetDimNum());
        return false;
    }
    if (!InferOutBatchDim(x1Shape, x2Shape)) {
        OP_LOGD(
            inputParams_.opName, "In %s, expected batch dims of x1 and x2 to be broadcastable, but got non-broadcastable.",
            INT4_KG_QUANT_MODE);
        return false;
    }
    if (!CheckOutputShapeAvailable()) {
        OP_LOGD(
            inputParams_.opName, "In %s, expected output shape product within INT64_MAX, but got overflow risk.",
            INT4_KG_QUANT_MODE);
        return false;
    }
    return true;
}

bool QuantBatchMatmulV4PergroupArch35Tiling::CheckPergroupScaleShape(
    const gert::Shape& x1ScaleShape, const gert::Shape& x2ScaleShape, const gert::Shape& x2OffsetShape) const
{
    int64_t m = static_cast<int64_t>(inputParams_.mSize);
    int64_t k = static_cast<int64_t>(inputParams_.kSize);
    int64_t n = static_cast<int64_t>(inputParams_.nSize);
    int64_t nkgroup = ops::CeilDiv(k, GROUP_SIZE_K);

    if (x1ScaleShape.GetDim(0) != m || x1ScaleShape.GetDim(1) != 1) {
        OP_LOGD(
            inputParams_.opName, "In %s, expected x1Scale shape=[%ld, 1], but got [%ld, %ld].", INT4_KG_QUANT_MODE, m,
            x1ScaleShape.GetDim(0), x1ScaleShape.GetDim(1));
        return false;
    }
    if (x2ScaleShape.GetDim(0) != nkgroup || x2ScaleShape.GetDim(1) != n) {
        OP_LOGD(
            inputParams_.opName, "In %s, expected x2Scale shape=[%ld, %ld], but got [%ld, %ld].", INT4_KG_QUANT_MODE,
            nkgroup, n, x2ScaleShape.GetDim(0), x2ScaleShape.GetDim(1));
        return false;
    }
    if (x2OffsetShape.GetDim(0) != nkgroup || x2OffsetShape.GetDim(1) != n) {
        OP_LOGD(
            inputParams_.opName, "In %s, expected x2Offset shape=[%ld, %ld], but got [%ld, %ld].", INT4_KG_QUANT_MODE,
            nkgroup, n, x2OffsetShape.GetDim(0), x2OffsetShape.GetDim(1));
        return false;
    }
    return true;
}

bool QuantBatchMatmulV4PergroupArch35Tiling::CheckPergroupInputFormat() const
{
    auto x1Format = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(X1_IDX)->GetStorageFormat()));
    auto x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(X2_IDX)->GetStorageFormat()));
    auto x1ScaleFormat =
        static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetOptionalInputDesc(X1_SCALE_IDX)->GetStorageFormat()));
    auto x2ScaleFormat =
        static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetOptionalInputDesc(X2_SCALE_IDX)->GetStorageFormat()));
    auto x2OffsetFormat = static_cast<ge::Format>(
        ge::GetPrimaryFormat(context_->GetOptionalInputDesc(X2_OFFSET_IDX)->GetStorageFormat()));

    if (x1Format != ge::FORMAT_ND || x2Format != ge::FORMAT_ND || x1ScaleFormat != ge::FORMAT_ND ||
        x2ScaleFormat != ge::FORMAT_ND || x2OffsetFormat != ge::FORMAT_ND) {
        OP_LOGD(
            inputParams_.opName,
            "In %s, expected storage format=ND for x1/x2/x1Scale/x2Scale/x2Offset, but got [%s, %s, %s, %s, %s].",
            INT4_KG_QUANT_MODE, ge::TypeUtils::FormatToSerialString(x1Format).c_str(),
            ge::TypeUtils::FormatToSerialString(x2Format).c_str(),
            ge::TypeUtils::FormatToSerialString(x1ScaleFormat).c_str(),
            ge::TypeUtils::FormatToSerialString(x2ScaleFormat).c_str(),
            ge::TypeUtils::FormatToSerialString(x2OffsetFormat).c_str());
        return false;
    }
    return true;
}

uint64_t QuantBatchMatmulV4PergroupArch35Tiling::GetTilingKey() const
{
    uint64_t trans = (static_cast<uint64_t>(inputParams_.transA) << 1) | static_cast<uint64_t>(inputParams_.transB);
    matmul_v4::KernelTemplateType kernelType = matmul_v4::KernelTemplateType::LUT_ASW;
    return GET_TPL_TILING_KEY(
        trans, static_cast<uint64_t>(matmul_v4::QuantType::INT4_ASYMMETRICAL), static_cast<uint64_t>(false),
        static_cast<uint64_t>(false), static_cast<uint64_t>(kernelType));
}

ge::graphStatus QuantBatchMatmulV4PergroupArch35Tiling::GetWorkspaceSize()
{
    workspaceSize_ = inputParams_.libApiWorkSpaceSize;
    constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128;
    uint64_t aInt8Size = inputParams_.mSize * inputParams_.kSize;
    uint64_t bInt8Size = inputParams_.kSize * inputParams_.nSize;
    uint64_t convertWorkspaceSize = ops::CeilAlign(aInt8Size, (uint64_t)BASIC_BLOCK_SIZE_128) +
                                    ops::CeilAlign(bInt8Size, (uint64_t)BASIC_BLOCK_SIZE_128);
    workspaceSize_ += convertWorkspaceSize;
    workspaceSize_ += sizeof(int32_t) * static_cast<uint64_t>(tilingData_.matmulTiling.baseM) *
                      tilingData_.matmulTiling.baseN * tilingData_.matmulTiling.usedCoreNum * NUM_DB;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
