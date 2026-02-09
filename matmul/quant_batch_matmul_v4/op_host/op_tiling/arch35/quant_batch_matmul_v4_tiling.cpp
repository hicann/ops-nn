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
 * \file quant_batch_matmul_v4_tiling.cpp
 * \brief
 */

#include "quant_batch_matmul_v4_tiling.h"

#include <memory>
#include <mutex>
#include <numeric>
#include <set>

#include "util/math_util.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "common/inc/error_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "error_util.h"
#include "matmul/common/op_host/math_util.h"
#include "platform/platform_infos_def.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "../../../op_kernel/arch35/quant_batch_matmul_v4_tiling_data.h"

using AscendC::BLOCK_CUBE;
using namespace Ops::NN;

namespace optiling {
constexpr uint64_t B4_IN_B32_NUMS = 8UL;
constexpr uint64_t GROUP_MKN_BIT_SIZE = 0xFFFF;
using namespace matmul_v4;

inline bool IsNotEmptyShape(const gert::StorageShape* storageShape)
{
    return storageShape != nullptr && storageShape->GetStorageShape().GetShapeSize() != 0;
}

inline bool IsFormatNZ(ge::Format format)
{
    return format == ge::FORMAT_FRACTAL_NZ || format == ge::FORMAT_FRACTAL_NZ_C0_4;
}

void QuantBatchMatmulV4TilingBase::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr_ =
        std::unique_ptr<QuantBatchMatmulV4CompileInfo>(new (std::nothrow) QuantBatchMatmulV4CompileInfo());
    OP_TILING_CHECK(compileInfoPtr_ == nullptr,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "failed to instantiate compile info"),
        return);

    compileInfoPtr_->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr_->aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr_->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr_->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr_->l0cSize);
    compileInfoPtr_->workspaceNum = ascendcPlatform.GetLibApiWorkSpaceSize();

    std::string fixpipeL0c2out;
    bool res = platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_fix_pipe_l0c2out", fixpipeL0c2out);
    compileInfoPtr_->supportL0c2Out = res && !fixpipeL0c2out.empty();

    std::string dataMoveL12Bt;
    res = platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_data_move_l12bt", dataMoveL12Bt);
    compileInfoPtr_->supportL12BtBf16 = res && dataMoveL12Bt.find("bf16") != string::npos;

    std::string mmad;
    res = platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_mmad", mmad);
    compileInfoPtr_->supportMmadS8S4 = res && mmad.find("s8s4") != std::string::npos;

    compileInfoPtr_->socVersion = ascendcPlatform.GetSocVersion();

    // gert::GemmCompileInfo gmmcompileInfo;
    TilingPrepareForOpCache(context_);
    OP_LOGD(context_->GetNodeName(), "MatmulAllReduce Init Quant Tiling Compile Info Success");
}

void QuantBatchMatmulV4TilingBase::Reset()
{
    inputParams_.transA = false;
    inputParams_.transB = false;
    inputParams_.hasX1Scale = false;
    inputParams_.hasX2Scale = false;
    inputParams_.hasBias = false;
    inputParams_.hasAntiQuantOffset = false;
    inputParams_.weightNz = false;
    inputParams_.groupSize = 0UL;
    inputParams_.libApiWorkSpaceSize = 0U;
    inputParams_.mSize = 0L;
    inputParams_.kSize = 0L;
    inputParams_.nSize = 0L;
    cubeBaseN_ = static_cast<uint64_t>(BLOCK_CUBE);
    inputParams_.vecInnerAxisAlignUnit = VEC_INNER_AXIS_ALIGN_UINT;
    inputParams_.aDtype = ge::DT_FLOAT16;
    inputParams_.bDtype = ge::DT_INT8;
    inputParams_.cDtype = ge::DT_FLOAT16;
    inputParams_.x1ScaleDtype = ge::DT_BF16;
    inputParams_.x2ScaleDtype = ge::DT_BF16;
    inputParams_.biasDtype = ge::DT_FLOAT16;
    aFormat = ge::FORMAT_ND;
    bFormat = ge::FORMAT_ND;
    cFormat = ge::FORMAT_ND;
    inputParams_.templateDtype = DtypeEnum::FLOAT16;
    inputParams_.antiQuantType = QuantType::PER_GROUP;
    mmInputDtype_ = matmul_tiling::DataType::DT_FLOAT16;
    mmOutputDtype_ = matmul_tiling::DataType::DT_FLOAT16;
    mmBiasDtype_ = matmul_tiling::DataType::DT_FLOAT16;
    mmScaleADtype_ = matmul_tiling::DataType::DT_BF16;
    mmScaleBDtype_ = matmul_tiling::DataType::DT_BF16;
    aivNum_ = 0;
    aicNum_ = 0;
    inputParams_.opName = nullptr;
}

ge::graphStatus QuantBatchMatmulV4TilingBase::GetShapeAttrsInfo()
{
    inputParams_.opName = context_->GetNodeName();
    OPS_LOG_D(inputParams_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    auto compileInfoPtr = compileInfoPtr_ ? compileInfoPtr_.get() :
        reinterpret_cast<const QuantBatchMatmulV4CompileInfo *>(context_->GetCompileInfo());
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context_->GetNodeName(), "compileInfoPtr is null");
    inputParams_.supportL0c2Out = compileInfoPtr->supportL0c2Out;
    inputParams_.supportL12BtBf16 = compileInfoPtr->supportL12BtBf16;
    OPS_LOG_D(inputParams_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    OP_TILING_CHECK(CheckContext() != ge::GRAPH_SUCCESS, VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "invalid context"),
        return ge::GRAPH_FAILED);
    inputParams_.bFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(1)->GetStorageFormat()));
    if (IsFormatNZ(inputParams_.bFormat)) {
        inputParams_.bFormat = ge::FORMAT_FRACTAL_NZ;
    }
    inputParams_.weightNz = inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ;
    OP_TILING_CHECK(!AnalyzeQuantType() || !AnalyzeAttrs() || !AnalyzeInputs() || !AnalyzeDtype(),
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Fail to analyze context info"), return ge::GRAPH_FAILED);
    bool maxDimCheck = inputParams_.kSize > MAX_SHAPE_DIM || inputParams_.nSize > MAX_SHAPE_DIM;
    if (inputParams_.transA) {
        maxDimCheck |= inputParams_.mSize > MAX_SHAPE_DIM;
    }
    OP_TILING_CHECK(inputParams_.supportL0c2Out && maxDimCheck,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
            "only support MKN in range [1, %lu], get actual value[%lu, %lu, %lu]",
            MAX_SHAPE_DIM, inputParams_.mSize, inputParams_.kSize, inputParams_.nSize),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(inputParams_.groupSize > inputParams_.kSize || inputParams_.groupSize % MIN_GROUP_SIZE != 0,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
            "Only support group size greater than %lu, less than K and align to %lu, get K[%lu] and group size[%lu]",
            MIN_GROUP_SIZE, MIN_GROUP_SIZE, inputParams_.kSize, inputParams_.groupSize),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(inputParams_.supportL0c2Out && !inputParams_.supportL12BtBf16 &&
                        inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ &&
                        (inputParams_.bDtype != ge::DT_INT8 || inputParams_.antiQuantType != QuantType::PER_CHANNEL ||
                            inputParams_.cDtype == ge::DT_INT8),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
                        "weight Nz only support weight dtype INT8 per-channel scene, and not support quant scale, "
                        "current input bDtype[%s], antiquantType[%d], cDtype[%s]",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str(),
                        static_cast<int>(inputParams_.antiQuantType),
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
                    return ge::GRAPH_FAILED);
    auto transA_str = inputParams_.transA ? "true" : "false";
    auto transB_str = inputParams_.transB ? "true" : "false";
    auto hasBias_str = inputParams_.hasBias ? "true" : "false";
    OP_LOGD(inputParams_.opName,
        "input params: MKN[%lu, %lu, %lu], transA[%s], transB[%s], bias[%s], group size[%lu]",
        inputParams_.mSize, inputParams_.kSize, inputParams_.nSize, transA_str,
        transB_str, hasBias_str, inputParams_.groupSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV4TilingBase::CheckContext() const
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    auto x1Shape = context_->GetInputShape(idx);
    auto x1Desc = context_->GetInputDesc(idx++);
    auto x2Shape = context_->GetInputShape(idx);
    auto weightDesc = context_->GetInputDesc(idx++);
    auto outputShape = context_->GetOutputShape(0);
    auto outputDesc = context_->GetOutputDesc(0);

    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Desc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, weightDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());

    size_t xDimNum = x1Shape->GetStorageShape().GetDimNum();
    size_t weigthDimNum = x2Shape->GetStorageShape().GetDimNum();
    ge::Format weightFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(weightDesc->GetStorageFormat()));

    OP_TILING_CHECK(inputParams_.supportL0c2Out && xDimNum != VALID_INPUT_DIM_NUM,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName,"only support dim num[%zu] for x, but get [%zu]", VALID_INPUT_DIM_NUM, xDimNum),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(inputParams_.supportL0c2Out &&
            ((!IsFormatNZ(weightFormat) && weigthDimNum != VALID_INPUT_DIM_NUM) ||
             (IsFormatNZ(weightFormat) && weigthDimNum != VALID_WEIGHT_NZ_DIM_NUM)),
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
            "only support weight dim num[%zu] for ND format and dim num[%zu] for NZ format, but get [%zu] dim for [%s]",
            VALID_INPUT_DIM_NUM,
            VALID_WEIGHT_NZ_DIM_NUM,
            weigthDimNum,
            ge::TypeUtils::FormatToSerialString(weightFormat).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool QuantBatchMatmulV4TilingBase::AnalyzeDtype()
{
    inputParams_.aDtype = context_->GetInputDesc(X1_INDEX)->GetDataType();
    inputParams_.bDtype = context_->GetInputDesc(X2_INDEX)->GetDataType();
    if (inputParams_.bDtype == ge::DT_FLOAT) {
        inputParams_.bDtype = ge::DT_FLOAT4_E2M1;
        OP_LOGI(inputParams_.opName, "The conversion of x2 from fp32 to fp4 is completed.");
    }

    auto biasDesc = context_->GetOptionalInputDesc(BIAS_INDEX);
    auto x1ScaleDesc = context_->GetOptionalInputDesc(X1_SCALE_INDEX);
    auto x2ScaleDesc = context_->GetOptionalInputDesc(X2_SCALE_INDEX);
    auto yScaleDesc = context_->GetOptionalInputDesc(Y_SCALE_INDEX);
    inputParams_.cDtype = context_->GetOutputDesc(Y_OUTPUT_INDEX)->GetDataType();
    mmInputDtype_ = GetMatmulTilingDtype(inputParams_.aDtype);
    mmOutputDtype_ = GetMatmulTilingDtype(inputParams_.cDtype);
    inputParams_.templateDtype = inputParams_.cDtype == ge::DT_FLOAT16 ? DtypeEnum::FLOAT16 : DtypeEnum::BFLOAT16;
    // check x1 dtype
    OP_TILING_CHECK(inputParams_.aDtype != ge::DT_FLOAT8_E5M2 && inputParams_.aDtype != ge::DT_FLOAT8_E4M3FN,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
            "Unsupported data type [%s] for X1. Only DT_FLOAT8_E5M2 and DT_FLOAT8_E4M3FN are supported.",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str()),
        return false);
    // check x2 dtype
    OP_TILING_CHECK(inputParams_.bDtype != ge::DT_FLOAT4_E2M1,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
            "Unsupported data type [%s] for X2. Only DT_FLOAT4_E2M1 is supported.",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
        return false);
    OP_TILING_CHECK(
        inputParams_.antiQuantType == QuantType::PER_GROUP && inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ &&
        inputParams_.bDtype != ge::DT_FLOAT4_E2M1,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
            "Unsupported data type [%s] for X2. Only DT_FLOAT4_E2M1 is supported for per_group and NZ format.",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
        return false);
    // check y dtype
    OP_TILING_CHECK(inputParams_.cDtype != ge::DT_BF16,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
        "Unsupported data type [%s] for Y. Only DT_BF16 is supported.",
        ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str()),
        return false);
    if (inputParams_.antiQuantType != QuantType::MX) {
        // check yScale dtype
        OP_TILING_CHECK(yScaleDesc == nullptr,
            VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "yScaleDesc is null"),
            return false);
        OP_TILING_CHECK(yScaleDesc->GetDataType() != ge::DT_UINT64,
            VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
            "Unsupported data type [%s] for yScale. Only DT_UINT64 is supported.",
            ge::TypeUtils::DataTypeToSerialString(yScaleDesc->GetDataType()).c_str()),
            return false);
    }
    return AnalyzeBiasDtype(biasDesc) && AnalyzeX1scaleDtype(x1ScaleDesc) && AnalyzeX2scaleDtype(x2ScaleDesc);
}

bool QuantBatchMatmulV4TilingBase::AnalyzeBiasDtype(const gert::CompileTimeTensorDesc* biasDesc)
{
    if (inputParams_.hasBias && biasDesc != nullptr) {
        inputParams_.biasDtype = biasDesc->GetDataType();
        OP_TILING_CHECK(inputParams_.biasDtype != ge::DT_BF16 && inputParams_.biasDtype != ge::DT_FLOAT16,
            VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
                "Unsupported data type [%s] for Bias. Only DT_BF16 and DT_FLOAT16 is supported.",
                ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str()),
            return false);
        mmBiasDtype_ = GetMatmulTilingDtype(inputParams_.biasDtype);
    }

    return true;
}

bool QuantBatchMatmulV4TilingBase::AnalyzeX1scaleDtype(const gert::CompileTimeTensorDesc* x1ScaleDesc)
{
    if (inputParams_.hasX1Scale && x1ScaleDesc != nullptr) {
        inputParams_.x1ScaleDtype = x1ScaleDesc->GetDataType();
        OP_TILING_CHECK(
            inputParams_.x1ScaleDtype != ge::DT_FLOAT8_E8M0,
            VECTOR_INNER_ERR_REPORT_TILIING(
                inputParams_.opName, "Unsupported data type [%s] for X1 scale. Only DT_FLOAT8_E8M0 is supported.",
                ge::TypeUtils::DataTypeToSerialString(inputParams_.x1ScaleDtype).c_str()),
            return false);
    }
    return true;
}

bool QuantBatchMatmulV4TilingBase::AnalyzeX2scaleDtype(const gert::CompileTimeTensorDesc* x2ScaleDesc)
{
    OP_TILING_CHECK(x2ScaleDesc == nullptr,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "X2 scale can not be null."), return false);
    inputParams_.x2ScaleDtype = x2ScaleDesc->GetDataType();
    OP_TILING_CHECK(
        inputParams_.antiQuantType == QuantType::PER_GROUP && inputParams_.x2ScaleDtype != ge::DT_BF16,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName,
            "In per_group quantization mode, the x2 scale dtype supports only DT_BF16, but the actual value is %s.",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.x2ScaleDtype).c_str()),
        return false);
    OP_TILING_CHECK(inputParams_.x2ScaleDtype != ge::DT_BF16 && inputParams_.x2ScaleDtype != ge::DT_FLOAT8_E8M0,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName,
            "Unsupported data type [%s] for X2 scale. Only DT_BF16 and DT_FLOAT8_E8M0 is supported.",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.x2ScaleDtype).c_str()),
        return false);
    OP_TILING_CHECK(
        inputParams_.antiQuantType == QuantType::MX && inputParams_.x2ScaleDtype != ge::DT_FLOAT8_E8M0,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName,
            "In MX quantization mode, the x2 scale dtype supports only DT_FLOAT8_E8M0, but the actual value is %s.",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.x2ScaleDtype).c_str()),
        return false);
    return true;
}

bool QuantBatchMatmulV4TilingBase::AnalyzeTranspose()
{
    auto attrs = context_->GetAttrs();
    // check transposeX1
    auto transposeX1 = attrs->GetAttrPointer<bool>(TRANSPOSE_X1_INDEX);
    OP_TILING_CHECK(transposeX1 == nullptr,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "TransposeX1 false can not be nullptr"),
        return false);
    OP_TILING_CHECK(*transposeX1 != false,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
        "Unsupported value [%d] for transposeX1. Only transposeX1 = false is supported.",
        *transposeX1),
        return false);
    inputParams_.transA = transposeX1 != nullptr && *transposeX1;
    // check transposeX2
    auto transposeX2 = attrs->GetAttrPointer<bool>(TRANSPOSE_X2_INDEX);
    OP_TILING_CHECK(transposeX2 == nullptr,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "TransposeX2 true can not be nullptr"),
        return false);
    OP_TILING_CHECK(
        inputParams_.bFormat == ge::FORMAT_ND && *transposeX2 != true,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
            "Unsupported value [%d] for transposeX2 in ND format. Only transposeX2 = true is supported in ND format.",
            *transposeX2),
        return false);
    if (inputParams_.antiQuantType == QuantType::MX) {
        OP_TILING_CHECK(
            inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ && *transposeX2 != true,
            VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
                "Unsupported value [%d] for transposeX2 in NZ format. Only transposeX2 = true is supported in NZ format.",
                *transposeX2),
            return false);
    } else {
        OP_TILING_CHECK(
            inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ && *transposeX2 != false,
            VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
                "Unsupported value [%d] for transposeX2 in NZ format. Only transposeX2 = false is supported in NZ format.",
                *transposeX2),
            return false);
    }
    inputParams_.transB = transposeX2 != nullptr && *transposeX2;
    return true;
}

bool QuantBatchMatmulV4TilingBase::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    // check groupSize
    const int64_t *groupSizePtr = attrs->GetAttrPointer<int64_t>(GROUP_SIZE_INDEX);
    OP_TILING_CHECK(groupSizePtr == nullptr,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Group size can not be nullptr"),
        return false);
    OP_TILING_CHECK(inputParams_.bFormat == ge::FORMAT_ND && *groupSizePtr != GROUP_ALIGN_SIZE,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
                                                    "Unsupported value [%ld] for groupSize in x2 ND format. Only "
                                                    "groupSize = %ld is supported in x2 ND format.",
                                                    *groupSizePtr, GROUP_ALIGN_SIZE),
                    return false);
    OP_TILING_CHECK(
        inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ && *groupSizePtr != NZ_GROUP_SIZE_32,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Unsupported value [%ld] for groupSize in x2 FRACTAL_NZ format. Only "
            "groupSize = %ld is supported in x2 ND format.", *groupSizePtr, NZ_GROUP_SIZE_32),
        return false);
    inputParams_.groupSize = static_cast<uint64_t>(*groupSizePtr);
    inputParams_.vecInnerAxisAlignUnit = inputParams_.groupSize;
    return AnalyzeTranspose();;
}

bool QuantBatchMatmulV4TilingBase::AnalyzeX2InputDim(const gert::StorageShape *x2Shape)
{
    auto x2ShapeDimSize = x2Shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(
        inputParams_.bFormat == ge::FORMAT_ND && x2ShapeDimSize != VALID_INPUT_DIM_NUM,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName,
            "Unsupport value [%lu] for x2 shape dim in FORMAT_ND. Only shape size = %lu is supported.", x2ShapeDimSize, VALID_INPUT_DIM_NUM),
        return false);
    OP_TILING_CHECK(inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ && x2ShapeDimSize != VALID_WEIGHT_NZ_DIM_NUM,
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        inputParams_.opName,
                        "Unsupport value [%lu] for x2 shape dim in FORMAT_FRACTAL_NZ. Only shape size = %lu is supported.",
                        x2ShapeDimSize, VALID_WEIGHT_NZ_DIM_NUM),
                    return false);
    return true;
}

bool QuantBatchMatmulV4TilingBase::AnalyzeInputs()
{
    auto x1Shape = context_->GetInputShape(X1_INDEX);
    auto x2Shape = context_->GetInputShape(X2_INDEX);
    auto biasShape = context_->GetOptionalInputShape(BIAS_INDEX);
    auto x1ScaleShape = context_->GetOptionalInputShape(X1_SCALE_INDEX);
    auto x2ScaleShape = context_->GetOptionalInputShape(X2_SCALE_INDEX);
    auto yScaleShape = context_->GetOptionalInputShape(Y_SCALE_INDEX);
    auto yOffsetShape = context_->GetOptionalInputShape(Y_OFFSET_INDEX);
    auto yShape = context_->GetOutputShape(Y_OUTPUT_INDEX)->GetStorageShape();
    OP_TILING_CHECK(x1Shape->GetStorageShape().GetShapeSize() == 0, VECTOR_INNER_ERR_REPORT_TILIING(
        inputParams_.opName, "X1 shape can not be empty. Only support shape size greater than 0, but get [%s]",
        Ops::Base::ToString(x1Shape->GetStorageShape()).c_str()), return false);
    OP_TILING_CHECK(x2Shape->GetStorageShape().GetShapeSize() == 0, VECTOR_INNER_ERR_REPORT_TILIING(
        inputParams_.opName, "X2 shape can not be empty. Only support shape size greater than 0, but get [%s]",
        Ops::Base::ToString(x2Shape->GetStorageShape()).c_str()), return false);
    uint64_t shapeBatch = 1;
    auto outShapeDim = yShape.GetDimNum();
    uint64_t idx = 0;
    while (outShapeDim > MATMUL_SHAPE_DIM_NUM) {
        shapeBatch = shapeBatch * static_cast<uint64_t>(yShape.GetDim(idx++));
        --outShapeDim;
    }
    inputParams_.batchSize = shapeBatch;
    ge::Format aFormatCur = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(0)->GetStorageFormat()));
    OP_TILING_CHECK(aFormatCur != ge::FORMAT_ND, VECTOR_INNER_ERR_REPORT_TILIING(
        inputParams_.opName, "aFormat Only support Nd"), return false);
    return AnalyzeX2InputDim(x2Shape) && AnalyzeShapeSize(x1Shape, x2Shape) && AnalyzeBiasShape(biasShape) &&
           AnalyzeX1ScaleShape(x1ScaleShape) && AnalyzeX2ScaleShape(x2ScaleShape) &&
           AnalyzeYScaleOffsetShape(yScaleShape, yOffsetShape);
}

bool QuantBatchMatmulV4TilingBase::AnalyzeShapeSize(const gert::StorageShape* x1Shape,
                                                    const gert::StorageShape* x2Shape)
{
    auto x1ShapeDimSize = x1Shape->GetStorageShape().GetDimNum();
    inputParams_.mSize = static_cast<uint64_t>(inputParams_.transA ?
        x1Shape->GetStorageShape().GetDim(x1ShapeDimSize - 1) :
        x1Shape->GetStorageShape().GetDim(x1ShapeDimSize - MATMUL_SHAPE_DIM_NUM));
    OP_TILING_CHECK(x2Shape->GetStorageShape().GetShapeSize() == 0,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
            "X2 shape can not be empty. Only support shape size greater than 0, but get [%s]",
            Ops::Base::ToString(x2Shape->GetStorageShape()).c_str()), return false);
    inputParams_.kSize = static_cast<uint64_t>(
        inputParams_.transA ? x1Shape->GetStorageShape().GetDim(x1ShapeDimSize - MATMUL_SHAPE_DIM_NUM)
                            : x1Shape->GetStorageShape().GetDim(x1ShapeDimSize - 1));
    uint64_t kBSize = 0;
    auto x2ShapeDimSize = x2Shape->GetStorageShape().GetDimNum();
    if (x2ShapeDimSize == VALID_INPUT_DIM_NUM) {
        inputParams_.nSize = static_cast<uint64_t>(
            inputParams_.transB ? x2Shape->GetStorageShape().GetDim(x2ShapeDimSize - MATMUL_SHAPE_DIM_NUM)
                                : x2Shape->GetStorageShape().GetDim(x2ShapeDimSize - 1));
        kBSize = static_cast<uint64_t>(inputParams_.transB
                                           ? x2Shape->GetStorageShape().GetDim(x2ShapeDimSize - 1)
                                           : x2Shape->GetStorageShape().GetDim(x2ShapeDimSize - MATMUL_SHAPE_DIM_NUM));
        OP_TILING_CHECK(inputParams_.kSize != kBSize, VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
            "kA[%lu] is not equal kB[%lu]", inputParams_.kSize, kBSize), return false);
    } else if (x2ShapeDimSize == VALID_WEIGHT_NZ_DIM_NUM) {
        uint32_t n0Idx = inputParams_.transB ? DIM_INDEX_2 : DIM_INDEX_3;
        uint32_t n1Idx = inputParams_.transB ? DIM_INDEX_1 : DIM_INDEX_0;
        // (n1, k1, k0, n0) or (k1, n1, n0, k0)
        uint64_t n1 = x2Shape->GetStorageShape().GetDim(n1Idx);
        uint64_t n0 = x2Shape->GetStorageShape().GetDim(n0Idx);
        // 当x2数据类型是float32, 并且最后1维是n0时, 最后一维扩大8倍
        OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(X2_INDEX));
        if (context_->GetInputDesc(X2_INDEX)->GetDataType() == ge::DT_FLOAT && !inputParams_.transB) {
            n0 *= B4_IN_B32_NUMS;
        }

        inputParams_.nSize = n1 * n0;
    }
    OP_TILING_CHECK(inputParams_.mSize < MIN_SHAPE_SIZE,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
            "Unsupported value [%lu] for m, m shouldn't be less than %ld.",
            inputParams_.mSize, MIN_SHAPE_SIZE), return false);
    OP_TILING_CHECK(inputParams_.nSize < MIN_SHAPE_SIZE,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
            "Unsupported value [%lu] for n. Only values greater than or equal to %ld is supported.",
            inputParams_.nSize, MIN_SHAPE_SIZE), return false);
    OP_TILING_CHECK(inputParams_.kSize < MIN_SHAPE_SIZE,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
            "Unsupported value [%lu] for k. Only values greater than or equal to %ld is supported.",
            inputParams_.kSize, MIN_SHAPE_SIZE), return false);
    return true;
}

bool QuantBatchMatmulV4TilingBase::AnalyzeBiasShape(const gert::StorageShape* biasShape)
{
    if (biasShape == nullptr) {
        inputParams_.hasBias = false;
        return true;
    }
    OP_TILING_CHECK(inputParams_.antiQuantType != QuantType::MX,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
                                                    "Only Mx quant scene supports bias, actual quant type is %d",
                                                    static_cast<int>(inputParams_.antiQuantType)),
                    return false);
    inputParams_.hasBias = true;
    auto biasShapeDimNum = static_cast<uint64_t>(biasShape->GetStorageShape().GetDimNum());
    auto biasStorageShape = biasShape->GetStorageShape();
    OP_TILING_CHECK(biasShapeDimNum != VALID_BIAS_MAX_DIM,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "bias's dim should be 2. Actual is %lu",
                                                    biasShapeDimNum),
                    return false);
    OP_TILING_CHECK(biasStorageShape.GetDim(DIM_INDEX_0) != VALID_BIAS_SHAPE_SIZE ||
                        static_cast<size_t>(biasStorageShape.GetDim(DIM_INDEX_1)) != inputParams_.nSize,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "bias shape only support [1, n], input is %s",
                                                    Ops::Base::ToString(biasStorageShape).c_str()),
                    return false);
    return true;
}

bool QuantBatchMatmulV4TilingBase::AnalyzeX1ScaleShape(const gert::StorageShape* x1ScaleShape)
{
    if (x1ScaleShape == nullptr) {
        return true;
    }
    if (inputParams_.antiQuantType == QuantType::MX) { // check mx shape
        auto x1ScaleShapeDimNum = static_cast<uint64_t>(x1ScaleShape->GetStorageShape().GetDimNum());
        auto x1ScaleStorageShape = x1ScaleShape->GetStorageShape();
        OP_TILING_CHECK(x1ScaleShapeDimNum != VALID_X1_SCALE_DIM_NUM,
            VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
                "Expected dimension of X1 scale to be %lu, but actual dimension is %lu.",
                VALID_X1_SCALE_DIM_NUM,
                x1ScaleShapeDimNum),
            return false);
        OP_TILING_CHECK(x1ScaleStorageShape.GetDim(0) != static_cast<int64_t>(inputParams_.mSize) ||
                            x1ScaleStorageShape.GetDim(1) != static_cast<int64_t>(inputParams_.kSize / GROUP_ALIGN_SIZE),
            VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
                "Expected shape of X1 scale to be [%lu, %lu], but actual shape is %s.",
                inputParams_.mSize, inputParams_.kSize / GROUP_ALIGN_SIZE,
                Ops::Base::ToString(x1ScaleStorageShape).c_str()), return false);
    }
    return true;
}

bool QuantBatchMatmulV4TilingBase::AnalyzeX2ScalePerGroupShape(const gert::StorageShape* x2ScaleShape)
{
    OP_TILING_CHECK(
        inputParams_.kSize % inputParams_.groupSize != 0,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "The value of groupNum is not an integer multiple."),
        return false);
    uint64_t groupNum = ops::CeilDiv(inputParams_.kSize, inputParams_.groupSize);
    gert::Shape expectShape;
    if (inputParams_.transB) {
        expectShape.AppendDim(static_cast<int64_t>(inputParams_.nSize));
        expectShape.AppendDim(static_cast<int64_t>(groupNum));
    } else {
        expectShape.AppendDim(static_cast<int64_t>(groupNum));
        expectShape.AppendDim(static_cast<int64_t>(inputParams_.nSize));
    }
    OP_TILING_CHECK(expectShape != x2ScaleShape->GetStorageShape(),
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
            "X2 scale shape %s is invalid, it should be %s, "
            "N, group size[%lu], K[%lu], N[%lu], transpose_weight[%s].",
            Ops::Base::ToString(x2ScaleShape->GetStorageShape()).c_str(),
            Ops::Base::ToString(expectShape).c_str(),
            inputParams_.groupSize, inputParams_.kSize, inputParams_.nSize,
            inputParams_.transB ? "true" : "false"), return false);
    return true;
}

bool QuantBatchMatmulV4TilingBase::AnalyzeX2ScaleShape(const gert::StorageShape* x2ScaleShape)
{
    OP_TILING_CHECK(x2ScaleShape == nullptr,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "X2 scale can not be null"),
        return false);
    auto x2ScaleShapeSize = static_cast<size_t>(x2ScaleShape->GetStorageShape().GetShapeSize());
    if (inputParams_.antiQuantType == QuantType::MX) { // check mx shape
        auto x2ScaleShapeDimNum = static_cast<uint64_t>(x2ScaleShape->GetStorageShape().GetDimNum());
        auto x2ScaleStorageShape = x2ScaleShape->GetStorageShape();
        OP_TILING_CHECK(x2ScaleShapeDimNum != VALID_X2_SCALE_DIM_NUM,
            VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
                "Expected dimension of X2 scale to be %lu, but actual dimension is %lu.",
                VALID_X2_SCALE_DIM_NUM, x2ScaleShapeDimNum), return false);
        OP_TILING_CHECK(x2ScaleStorageShape.GetDim(0) != static_cast<int64_t>(inputParams_.nSize) ||
                            x2ScaleStorageShape.GetDim(1) != static_cast<int64_t>(inputParams_.kSize) / GROUP_ALIGN_SIZE,
            VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
                "Expected shape of X2 scale to be [%lu, %lu], but actual shape is %s.",
                inputParams_.nSize, inputParams_.kSize / GROUP_ALIGN_SIZE,
                Ops::Base::ToString(x2ScaleStorageShape).c_str()), return false);
    } else if (inputParams_.groupSize > 0) {
        return AnalyzeX2ScalePerGroupShape(x2ScaleShape);
    } else if (x2ScaleShapeSize == 1) {
        inputParams_.antiQuantType = QuantType::PER_TENSOR;
    } else {
        OP_TILING_CHECK(x2ScaleShapeSize != inputParams_.nSize, VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
            "X2 scale %s shape size should same as N[%lu].", Ops::Base::ToString(x2ScaleShape->GetStorageShape()).c_str(),
            inputParams_.nSize), return false);
        inputParams_.antiQuantType = QuantType::PER_CHANNEL;
    }
    return true;
}

bool QuantBatchMatmulV4TilingBase::AnalyzeYScaleOffsetShape(
    const gert::StorageShape *yScaleShape, const gert::StorageShape *yOffsetShape) const
{
    OP_TILING_CHECK(!IsNotEmptyShape(yScaleShape) && IsNotEmptyShape(yOffsetShape),
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "not support quant_offset without quant_scale"),
        return false);
    if (!IsNotEmptyShape(yScaleShape)) {
        OP_TILING_CHECK(
            inputParams_.antiQuantType == QuantType::PER_GROUP,
            VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
                "Unsupported quant_scale shape size 0 when per_group and NZ format."),
            return false);
        return true;
    }
    size_t yScaleShapeSize = static_cast<size_t>(yScaleShape->GetStorageShape().GetShapeSize());
    OP_TILING_CHECK(yScaleShapeSize == 0 && inputParams_.cDtype == ge::DT_INT8,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "not support quant_scale shape size 0 when output dtype is int8"),
        return false);
    OP_TILING_CHECK(yScaleShape->GetStorageShape().GetDimNum() > VALID_INPUT_DIM_NUM ||
                        (yScaleShape->GetStorageShape().GetDimNum() == VALID_INPUT_DIM_NUM &&
                            yScaleShape->GetStorageShape().GetDim(0) != 1),
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
            "quant_scale shape only support [1, n] or [n,], input is %s",
            Ops::Base::ToString(yScaleShape->GetStorageShape()).c_str()),
        return false);
    OP_TILING_CHECK(IsNotEmptyShape(yOffsetShape) && yScaleShape->GetStorageShape() != yOffsetShape->GetStorageShape(),
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
            "quant_scale %s and quant_offset %s should have same shape",
            Ops::Base::ToString(yScaleShape->GetStorageShape()).c_str(),
            Ops::Base::ToString(yOffsetShape->GetStorageShape()).c_str()),
        return false);

    return true;
}

bool QuantBatchMatmulV4TilingBase::AnalyzeQuantType()
{
    auto x1ScaleShape = context_->GetOptionalInputShape(X1_SCALE_INDEX);
    auto x2ScaleShape = context_->GetOptionalInputShape(X2_SCALE_INDEX);
    // 判断是否有Scale以及量化方式
    if (x1ScaleShape == nullptr) {
        inputParams_.hasX1Scale = false;
    } else {
        inputParams_.hasX1Scale = true;
        auto x1ScaleDesc = context_->GetOptionalInputDesc(X1_SCALE_INDEX);
        if (x1ScaleDesc != nullptr && x1ScaleDesc->GetDataType() == ge::DT_FLOAT8_E8M0) {
            inputParams_.antiQuantType = QuantType::MX;
        }
    }
    if (x2ScaleShape == nullptr) {
        inputParams_.hasX2Scale = false;
    } else {
        inputParams_.hasX2Scale = true;
        auto x2ScaleDesc = context_->GetOptionalInputDesc(X2_SCALE_INDEX);
        if (x2ScaleDesc != nullptr && x2ScaleDesc->GetDataType() == ge::DT_FLOAT8_E8M0) {
            inputParams_.antiQuantType = QuantType::MX;
        }
    }
    return true;
}

ge::graphStatus QuantBatchMatmulV4TilingBase::GetPlatformInfo()
{
    auto compileInfoPtr = compileInfoPtr_
                              ? compileInfoPtr_.get()
                              : reinterpret_cast<const QuantBatchMatmulV4CompileInfo *>(context_->GetCompileInfo());
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context_->GetNodeName(), "compileInfoPtr is null");

    aivNum_ = compileInfoPtr->aivNum;
    aicNum_ = compileInfoPtr->aicNum;
    aicoreParams_.blockDim = 0;
    aicoreParams_.ubSize = compileInfoPtr->ubSize;
    aicoreParams_.l1Size = compileInfoPtr->l1Size;
    aicoreParams_.l0cSize = compileInfoPtr->l0cSize;
    inputParams_.libApiWorkSpaceSize = compileInfoPtr->workspaceNum;

    OP_LOGI(inputParams_.opName,
        "get platform: aivNum(%u) aicNum(%u) ubSize(%lu) l1Size(%lu) l0cSize(%lu)",
        aivNum_,
        aicNum_,
        aicoreParams_.ubSize,
        aicoreParams_.l1Size,
        aicoreParams_.l0cSize);

    if (inputParams_.bDtype == ge::DT_INT4) {
        OP_TILING_CHECK(!CalcUBSize(1UL, inputParams_.groupSize),
            VECTOR_INNER_ERR_REPORT_TILIING(
                inputParams_.opName, "group size[%lu] cannot full load to UB", inputParams_.groupSize),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

bool QuantBatchMatmulV4TilingBase::GetTilingFromCache()
{
    return false;
}

ge::graphStatus QuantBatchMatmulV4TilingBase::PostTiling()
{
#ifdef A8W4_TILING
    OP_LOGD(inputParams_.opName, "final tiling data size: %zu", tilingDataSize_);

    OP_TILING_CHECK(tilingDataSize_ % sizeof(uint64_t) != 0,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "tiling data size[%zu] not aligned to 8", tilingDataSize_),
        return ge::GRAPH_FAILED);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize_);
    uint32_t usedAicNum = tilingData_->cubeNumBlocksM * tilingData_->cubeNumBlocksN;
    uint32_t usedAivNum = tilingData_->vecNumBlocksK * tilingData_->vecNumBlocksN;
    context_->SetBlockDim(std::max(usedAicNum, CalcTschNumBlocks(usedAivNum, aicNum_, aivNum_)));

    OP_TILING_CHECK(
        !CheckFinalTilingData(), PrintTilingData(false);
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "get invalid tiling data, check above validate rule"),
        return ge::GRAPH_FAILED);
    size_t *workspaces = context_->GetWorkspaceSizes(1);  // set workspace
    workspaces[0] = workspaceSize_;

    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void*>(tilingData_), tilingDataSize_);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    PrintTilingData(true);
#endif
    return ge::GRAPH_SUCCESS;
}

void QuantBatchMatmulV4TilingBase::PrintTilingData(bool debugLevel)
{
    if (debugLevel && CheckLogLevel(OP, DLOG_DEBUG) != 1) {
        return;
    }

    std::stringstream ss;
    ss << "kAlign: " << tilingData_->kAlign << " nAlign: " << tilingData_->nAlign
       << " kSize: " << tilingData_->kSize << " nSize: " << tilingData_->nSize
       << " mSize: " << tilingData_->mSize << " groupSize: " << tilingData_->groupSize
       << " cubeNumBlocksN: " << static_cast<uint32_t>(tilingData_->cubeNumBlocksN)
       << " cubeNumBlocksM: " << static_cast<uint32_t>(tilingData_->cubeNumBlocksM);

    if (debugLevel) {
        OPS_LOG_D(inputParams_.opName, "tiling data: %s", ss.str().c_str());
    }else {
        OPS_LOG_E(inputParams_.opName, "tiling data: %s", ss.str().c_str());
    }
    PrintMatMulTiling();
}

void QuantBatchMatmulV4TilingBase::PrintMatMulTiling() const
{
    std::stringstream ss;
    auto &matmulTiling = tilingData_->matmulTiling;
    ss << "usedCoreNum " << matmulTiling.usedCoreNum << " M " << matmulTiling.M << " N "
       << matmulTiling.N << " Ka " << matmulTiling.Ka << " Kb " << matmulTiling.Kb << " singleCoreM "
       << matmulTiling.singleCoreM << " singleCoreN " << matmulTiling.singleCoreN << " singleCoreK "
       << matmulTiling.singleCoreK << " baseM " << matmulTiling.baseM << " baseN "
       << matmulTiling.baseN << " baseK " << matmulTiling.baseK << " depthA1 " << matmulTiling.depthA1
       << " depthB1 " << matmulTiling.depthB1 << " stepM " << matmulTiling.stepM << " stepN "
       << matmulTiling.stepN << " isBias " << matmulTiling.isBias << " transLength "
       << matmulTiling.transLength << " iterateOrder " << matmulTiling.iterateOrder << " shareMode "
       << matmulTiling.shareMode << " shareL1Size " << matmulTiling.shareL1Size << " shareL0CSize "
       << matmulTiling.shareL0CSize << " shareUbSize " << matmulTiling.shareUbSize << " batchM "
       << matmulTiling.batchM << " batchN " << matmulTiling.batchN << " stepKa "
       << matmulTiling.stepKa << " stepKb " << matmulTiling.stepKb << " dbL0A " << matmulTiling.dbL0A
       << " dbL0B " << matmulTiling.dbL0B << " dbL0C " << matmulTiling.dbL0C;

    OPS_LOG_I(inputParams_.opName, "matmul tiling: %s", ss.str().c_str());
}

ge::graphStatus QuantBatchMatmulV4TilingBase::InstantiateTilingData()
{
    if (tilingData_ == nullptr) {
        tilingDataManager_ = std::make_unique<qbmmv4_tiling::QuantBatchMatmulV4TilingDataParams>();
        OP_TILING_CHECK(tilingDataManager_ == nullptr,
                        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "failed to instantiate tilingData"),
                        return ge::GRAPH_FAILED);
        tilingData_ = tilingDataManager_.get();
    }
    OP_TILING_CHECK(tilingData_ == nullptr,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "failed to instantiate tilingData"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetRawTilingData()->GetCapacity() < tilingDataSize_,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
            "tiling data capacity %zu < actual tiling data size %zu",
            context_->GetRawTilingData()->GetCapacity(),
            tilingDataSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("QuantBatchMatmulV4", QuantBatchMatmulV4RegBase, BASIC_PRIORITY);

}  // namespace optiling
