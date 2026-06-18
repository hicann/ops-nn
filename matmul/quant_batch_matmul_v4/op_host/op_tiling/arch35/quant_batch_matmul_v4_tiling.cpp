/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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
#include "error_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "error_util.h"
#include "matmul/common/op_host/math_util.h"
#include "platform/platform_infos_def.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "../../../op_kernel/arch35/quant_batch_matmul_v4_tiling_data_apt.h"

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
    return format == ge::FORMAT_FRACTAL_NZ ||
           format == ge::FORMAT_FRACTAL_NZ_C0_4 ||
           format == ge::FORMAT_FRACTAL_NZ_C0_32;
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
        OP_LOGE(context_->GetNodeName(), "failed to instantiate compile info"),
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
    OPS_CHECK_NULL_WITH_CONTEXT(context_, compileInfoPtr);
    inputParams_.supportL0c2Out = compileInfoPtr->supportL0c2Out;
    inputParams_.supportL12BtBf16 = compileInfoPtr->supportL12BtBf16;
    OPS_LOG_D(inputParams_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    OP_TILING_CHECK(CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "invalid context"),
        return ge::GRAPH_FAILED);
    inputParams_.bFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(1)->GetStorageFormat()));
    if (IsFormatNZ(inputParams_.bFormat)) {
        inputParams_.bFormat = ge::FORMAT_FRACTAL_NZ;
    }
    inputParams_.weightNz = inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ;
    OP_TILING_CHECK(!AnalyzeQuantType() || !AnalyzeAttrs() || !AnalyzeInputs() || !AnalyzeDtype(),
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Fail to analyze context info"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckInputParams() != ge::GRAPH_SUCCESS, 
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "invalid input parameters"),
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

ge::graphStatus QuantBatchMatmulV4TilingBase::CheckInputParams() const
{
    bool maxDimCheck = inputParams_.kSize > MAX_SHAPE_DIM || inputParams_.nSize > MAX_SHAPE_DIM;
    if (inputParams_.transA) {
        maxDimCheck |= inputParams_.mSize > MAX_SHAPE_DIM;
    }
    OP_TILING_CHECK(inputParams_.supportL0c2Out && maxDimCheck,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
            "only support MKN in range [1, %lu], get actual value[%lu, %lu, %lu]",
            MAX_SHAPE_DIM, inputParams_.mSize, inputParams_.kSize, inputParams_.nSize),
        return ge::GRAPH_FAILED);
    if (inputParams_.antiQuantType == QuantType::MX) {
        OP_TILING_CHECK(inputParams_.groupSize != MX_GROUP_SIZE,
            VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
                "Group size must be 32 for MX quantization, get group size[%lu]", inputParams_.groupSize),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(inputParams_.groupSize > inputParams_.kSize || inputParams_.groupSize % MIN_GROUP_SIZE != 0,
            VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
                "Only support group size greater than %lu, less than K and align to %lu, get K[%lu] and group size[%lu]",
                MIN_GROUP_SIZE, MIN_GROUP_SIZE, inputParams_.kSize, inputParams_.groupSize),
            return ge::GRAPH_FAILED);
    }
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
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "x1", ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                        "The dtype of x1 must be FLOAT8_E5M2 or FLOAT8_E4M3FN"),
                    return false);
    // check x2 dtype
    OP_TILING_CHECK(inputParams_.bDtype != ge::DT_FLOAT4_E2M1,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "x2", ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str(),
                        "The dtype of x2 must be FLOAT4_E2M1"),
                    return false);
    OP_TILING_CHECK(
        inputParams_.antiQuantType == QuantType::PER_GROUP && inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ &&
            inputParams_.bDtype != ge::DT_FLOAT4_E2M1,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "x2", ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str(),
            "When the quant mode is per_group and the format of x2 is FRACTAL_NZ, the dtype of x2 must be FLOAT4_E2M1"),
        return false);
    // check y dtype
    OP_TILING_CHECK(
        inputParams_.cDtype != ge::DT_BF16 && inputParams_.cDtype != ge::DT_FLOAT16,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "y", ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str(),
            "The dtype of y must be BF16 or FLOAT16."),
        return false);
    if (inputParams_.antiQuantType != QuantType::MX) {
        // check yScale dtype
        OP_TILING_CHECK(yScaleDesc == nullptr, OP_LOGE(inputParams_.opName, "yScaleDesc is null"), return false);
        OP_TILING_CHECK(
            yScaleDesc->GetDataType() != ge::DT_UINT64,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                inputParams_.opName, "yScale", ge::TypeUtils::DataTypeToSerialString(yScaleDesc->GetDataType()).c_str(),
                "The dtype of yScale must be UINT64."),
            return false);
    }
    return AnalyzeBiasDtype(biasDesc) && AnalyzeX1scaleDtype(x1ScaleDesc) && AnalyzeX2scaleDtype(x2ScaleDesc);
}

bool QuantBatchMatmulV4TilingBase::AnalyzeBiasDtype(const gert::CompileTimeTensorDesc* biasDesc)
{
    if (inputParams_.hasBias && biasDesc != nullptr) {
        inputParams_.biasDtype = biasDesc->GetDataType();
        OP_TILING_CHECK(
            inputParams_.biasDtype != ge::DT_BF16 && inputParams_.biasDtype != ge::DT_FLOAT16,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "bias",
                                                  ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str(),
                                                  "The dtype of bias must be BF16 or FLOAT16"),
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
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                inputParams_.opName, "x1Scale",
                ge::TypeUtils::DataTypeToSerialString(inputParams_.x1ScaleDtype).c_str(),
                "The dtype of x1Scale must be FLOAT8_E8M0"),
            return false);
    }
    return true;
}

bool QuantBatchMatmulV4TilingBase::AnalyzeX2scaleDtype(const gert::CompileTimeTensorDesc* x2ScaleDesc)
{
    OP_TILING_CHECK(x2ScaleDesc == nullptr,
        OP_LOGE(inputParams_.opName, "X2 scale can not be null."), return false);
    inputParams_.x2ScaleDtype = x2ScaleDesc->GetDataType();
    OP_TILING_CHECK(
        inputParams_.antiQuantType == QuantType::PER_GROUP && inputParams_.x2ScaleDtype != ge::DT_BF16 &&
            inputParams_.x2ScaleDtype != ge::DT_FLOAT16,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "x2Scale", ge::TypeUtils::DataTypeToSerialString(inputParams_.x2ScaleDtype).c_str(),
            "When the quant mode is per_group, the dtype of x2Scale must be BF16 or FLOAT16"),
        return false);
    OP_TILING_CHECK(
        inputParams_.x2ScaleDtype != ge::DT_BF16 && inputParams_.x2ScaleDtype != ge::DT_FLOAT8_E8M0 &&
            inputParams_.x2ScaleDtype != ge::DT_FLOAT16,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "x2Scale",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams_.x2ScaleDtype).c_str(),
                                              "The dtype of x2Scale must be BF16, FLOAT16, or FLOAT8_E8M0"),
        return false);
    OP_TILING_CHECK(
        inputParams_.antiQuantType == QuantType::MX && inputParams_.x2ScaleDtype != ge::DT_FLOAT8_E8M0,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "x2Scale",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams_.x2ScaleDtype).c_str(),
                                              "When the quant mode is MX, the dtype of x2Scale must be FLOAT8_E8M0"),
        return false);
    return true;
}

bool QuantBatchMatmulV4TilingBase::AnalyzeTranspose()
{
    auto attrs = context_->GetAttrs();
    // check transposeX1
    auto transposeX1 = attrs->GetAttrPointer<bool>(TRANSPOSE_X1_INDEX);
    OP_TILING_CHECK(transposeX1 == nullptr,
        OP_LOGE(inputParams_.opName, "TransposeX1 can not be nullptr"),
        return false);
    OP_TILING_CHECK(
        *transposeX1 != false,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "transposeX1", (*transposeX1 ? "true" : "false"),
                                              "The value of transposeX1 must be false"),
        return false);
    inputParams_.transA = transposeX1 != nullptr && *transposeX1;
    // check transposeX2
    auto transposeX2 = attrs->GetAttrPointer<bool>(TRANSPOSE_X2_INDEX);
    OP_TILING_CHECK(transposeX2 == nullptr,
        OP_LOGE(inputParams_.opName, "TransposeX2 can not be nullptr"),
        return false);
    OP_TILING_CHECK(
        inputParams_.bFormat == ge::FORMAT_ND && *transposeX2 != true,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "transposeX2", (*transposeX2 ? "true" : "false"),
                                              "When the format of x2 is ND, the value of transposeX2 must be true"),
        return false);
    if (inputParams_.antiQuantType == QuantType::MX) {
        OP_TILING_CHECK(
            inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ && *transposeX2 != true,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                inputParams_.opName, "transposeX2", (*transposeX2 ? "true" : "false"),
                "When the quant mode is MX and the format of x2 is FRACTAL_NZ, the value of transposeX2 must be true"),
            return false);
    } else {
        OP_TILING_CHECK(inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ && *transposeX2 != false,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            inputParams_.opName, "transposeX2", (*transposeX2 ? "true" : "false"),
                            "When the format of x2 is FRACTAL_NZ, the value of transposeX2 must be false"),
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
        OP_LOGE(inputParams_.opName, "Group size can not be nullptr"),
        return false);
    OP_TILING_CHECK(
        inputParams_.bFormat == ge::FORMAT_ND && *groupSizePtr != GROUP_ALIGN_SIZE,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "groupSize", std::to_string(*groupSizePtr).c_str(),
                                              "When the format of x2 is ND, the value of groupSize must be 32"),
        return false);
    OP_TILING_CHECK(
        inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ && *groupSizePtr != NZ_GROUP_SIZE_32,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "groupSize", std::to_string(*groupSizePtr).c_str(),
                                              "When the format of x2 is FRACTAL_NZ, the value of groupSize must be 32"),
        return false);
    inputParams_.groupSize = static_cast<uint64_t>(*groupSizePtr);
    inputParams_.vecInnerAxisAlignUnit = inputParams_.groupSize;
    return AnalyzeTranspose();;
}

bool QuantBatchMatmulV4TilingBase::AnalyzeX2InputDim(const gert::StorageShape* x2Shape)
{
    auto x2ShapeDimSize = x2Shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(
        inputParams_.bFormat == ge::FORMAT_ND && x2ShapeDimSize != VALID_INPUT_DIM_NUM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "x2", std::to_string(x2ShapeDimSize).c_str(),
                                                 "When the format of x2 is ND, the shape dim of x2 must be 2D"),
        return false);
    OP_TILING_CHECK(
        inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ && x2ShapeDimSize != VALID_WEIGHT_NZ_DIM_NUM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "x2", std::to_string(x2ShapeDimSize).c_str(),
                                                 "When the format of x2 is FRACTAL_NZ, the shape dim of x2 must be 4D"),
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
    OP_TILING_CHECK(
        x1Shape->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            inputParams_.opName, "x1", Ops::Base::ToString(x1Shape->GetStorageShape()).c_str(),
            "The shape size of x1 must be > 0"),
        return false);
    OP_TILING_CHECK(
        x2Shape->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            inputParams_.opName, "x2", Ops::Base::ToString(x2Shape->GetStorageShape()).c_str(),
            "The shape size of x2 must be > 0"),
        return false);
    uint64_t shapeBatch = 1;
    auto outShapeDim = yShape.GetDimNum();
    uint64_t idx = 0;
    while (outShapeDim > MATMUL_SHAPE_DIM_NUM) {
        shapeBatch = shapeBatch * static_cast<uint64_t>(yShape.GetDim(idx++));
        --outShapeDim;
    }
    inputParams_.batchSize = shapeBatch;
    ge::Format aFormatCur = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(0)->GetStorageFormat()));
    OP_TILING_CHECK(aFormatCur != ge::FORMAT_ND, OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
        inputParams_.opName, "x1", "aFormat", "The format of x1 must be ND"), return false);
    return AnalyzeX2InputDim(x2Shape) && AnalyzeShapeSize(x1Shape, x2Shape) && AnalyzeBiasShape(biasShape) &&
           AnalyzeX1ScaleShape(x1ScaleShape) && AnalyzeX2ScaleShape(x2ScaleShape) &&
           AnalyzeYScaleOffsetShape(yScaleShape, yOffsetShape);
}

bool QuantBatchMatmulV4TilingBase::ValidateShapeDimensions()
{
    OP_TILING_CHECK(
        inputParams_.mSize < MIN_SHAPE_SIZE,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(inputParams_.opName, "mSize", std::to_string(inputParams_.mSize).c_str(),
                                               "The value of mSize must be >= 1"),
        return false);
    OP_TILING_CHECK(
        inputParams_.nSize < MIN_SHAPE_SIZE,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(inputParams_.opName, "nSize", std::to_string(inputParams_.nSize).c_str(),
                                               "The value of nSize must be >= 1"),
        return false);
    OP_TILING_CHECK(
        inputParams_.kSize < MIN_SHAPE_SIZE,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(inputParams_.opName, "kSize", std::to_string(inputParams_.kSize).c_str(),
                                               "The value of kSize must be >= 1"),
        return false);
    return true;
}

bool QuantBatchMatmulV4TilingBase::AnalyzeShapeSize(const gert::StorageShape* x1Shape,
                                                     const gert::StorageShape* x2Shape)
{
    auto x1ShapeDimSize = x1Shape->GetStorageShape().GetDimNum();
    inputParams_.mSize = static_cast<uint64_t>(inputParams_.transA ?
        x1Shape->GetStorageShape().GetDim(x1ShapeDimSize - 1) :
        x1Shape->GetStorageShape().GetDim(x1ShapeDimSize - MATMUL_SHAPE_DIM_NUM));
    OP_TILING_CHECK(x2Shape->GetStorageShape().GetShapeSize() == 0,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(inputParams_.opName, "x2",
                                                              Ops::Base::ToString(x2Shape->GetStorageShape()).c_str(),
                                                              "The shape size of x2 must be > 0"),
                    return false);
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
        OP_TILING_CHECK(inputParams_.kSize != kBSize,
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(inputParams_.opName, "x1, x2", "kSize mismatch",
                                                               "The k dimension sizes of x1 and x2 must be equal"),
                        return false);
    } else if (x2ShapeDimSize == VALID_WEIGHT_NZ_DIM_NUM) {
        auto x2OriginShape = x2Shape->GetOriginShape();
        auto x2ShapeDimSize = x2OriginShape.GetDimNum();
        inputParams_.nSize = static_cast<uint64_t>(
            inputParams_.transB ? x2OriginShape.GetDim(x2ShapeDimSize - MATMUL_SHAPE_DIM_NUM) :
                                  x2OriginShape.GetDim(x2ShapeDimSize - 1)); // - 1: 表示尾轴为n轴
        if (context_->GetInputDesc(X2_INDEX)->GetDataType() == ge::DT_FLOAT && !inputParams_.transB) {
            inputParams_.nSize *= B4_IN_B32_NUMS;
        }
    }
    return ValidateShapeDimensions();
}

bool QuantBatchMatmulV4TilingBase::AnalyzeBiasShape(const gert::StorageShape* biasShape)
{
    if (biasShape == nullptr) {
        inputParams_.hasBias = false;
        return true;
    }
    OP_TILING_CHECK(inputParams_.antiQuantType != QuantType::MX,
                    OP_LOGE_FOR_INVALID_CONFIG_WITH_REASON(
                        inputParams_.opName, std::to_string(static_cast<int>(inputParams_.antiQuantType)).c_str(),
                        "antiQuantType", "quantConfig", "When bias exists, the quant mode must be MX"),
                    return false);
    inputParams_.hasBias = true;
    auto biasShapeDimNum = static_cast<uint64_t>(biasShape->GetStorageShape().GetDimNum());
    auto biasStorageShape = biasShape->GetStorageShape();
    OP_TILING_CHECK(
        biasShapeDimNum != VALID_BIAS_MAX_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "bias", std::to_string(biasShapeDimNum).c_str(),
                                                 "The shape dim of bias must be 2D"),
        return false);
    OP_TILING_CHECK(biasStorageShape.GetDim(DIM_INDEX_0) != VALID_BIAS_SHAPE_SIZE ||
                        static_cast<size_t>(biasStorageShape.GetDim(DIM_INDEX_1)) != inputParams_.nSize,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        inputParams_.opName, "bias", Ops::Base::ToString(biasStorageShape).c_str(),
                        "The shape of bias must be [1, " + std::to_string(inputParams_.nSize) + "]"),
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
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "x1Scale",
                                                                 std::to_string(x1ScaleShapeDimNum).c_str(),
                                                                 "The shape dim of x1Scale must be 3D"),
                        return false);
        // x1ScaleStorageShape (m, k / GROUP_ALIGN_SIZE / 2, 2)
        OP_TILING_CHECK(
            x1ScaleStorageShape.GetDim(0) != static_cast<int64_t>(inputParams_.mSize) ||
                x1ScaleStorageShape.GetDim(1) !=
                    ops::CeilDiv(static_cast<int64_t>(inputParams_.kSize), GROUP_ALIGN_SIZE * 2L) ||
                x1ScaleStorageShape.GetDim(2) != 2UL,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                inputParams_.opName, "x1Scale", Ops::Base::ToString(x1ScaleStorageShape).c_str(),
                "The shape of x1Scale must be [" + std::to_string(static_cast<int64_t>(inputParams_.mSize)) + ", " +
                    std::to_string(ops::CeilDiv(static_cast<int64_t>(inputParams_.kSize), GROUP_ALIGN_SIZE * 2L)) +
                    ", 2]"),
            return false);
    }
    return true;
}

bool QuantBatchMatmulV4TilingBase::AnalyzeX2ScalePerGroupShape(const gert::StorageShape* x2ScaleShape)
{
    OP_TILING_CHECK(inputParams_.kSize % inputParams_.groupSize != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        inputParams_.opName, "groupNum", std::to_string(inputParams_.kSize).c_str(),
                        "The value of kSize must be an integer multiple of groupSize"),
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
    std::string shapeReason = std::string("Expected [") + Ops::Base::ToString(expectShape) +
        "], groupSize=" + std::to_string(inputParams_.groupSize) + ", K=" + std::to_string(inputParams_.kSize) +
        ", N=" + std::to_string(inputParams_.nSize) + ", transpose_weight=" +
        (inputParams_.transB ? "true" : "false");
    OP_TILING_CHECK(
        expectShape != x2ScaleShape->GetStorageShape(),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            inputParams_.opName, "x2Scale",
            Ops::Base::ToString(x2ScaleShape->GetStorageShape()).c_str(), shapeReason.c_str()),
        return false);
    return true;
}

bool QuantBatchMatmulV4TilingBase::AnalyzeX2ScaleShape(const gert::StorageShape* x2ScaleShape)
{
    OP_TILING_CHECK(x2ScaleShape == nullptr,
        OP_LOGE(inputParams_.opName, "X2 scale can not be null"),
        return false);
    auto x2ScaleShapeSize = static_cast<size_t>(x2ScaleShape->GetStorageShape().GetShapeSize());
    if (inputParams_.antiQuantType == QuantType::MX) { // check mx shape
        auto x2ScaleShapeDimNum = static_cast<uint64_t>(x2ScaleShape->GetStorageShape().GetDimNum());
        auto x2ScaleStorageShape = x2ScaleShape->GetStorageShape();
        OP_TILING_CHECK(x2ScaleShapeDimNum != VALID_X2_SCALE_DIM_NUM,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "x2Scale",
                                                                 std::to_string(x2ScaleShapeDimNum).c_str(),
                                                                 "The shape dim of x2Scale must be 3D"),
                        return false);
        // x2ScaleStorageShape: (n, k / GROUP_ALIGN_SIZE / 2, 2)
        OP_TILING_CHECK(
            x2ScaleStorageShape.GetDim(0) != static_cast<int64_t>(inputParams_.nSize) ||
                x2ScaleStorageShape.GetDim(1) != ops::CeilDiv<int64_t>(inputParams_.kSize, (GROUP_ALIGN_SIZE * 2)) ||
                x2ScaleStorageShape.GetDim(2) != 2,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                inputParams_.opName, "x2Scale", Ops::Base::ToString(x2ScaleStorageShape).c_str(),
                "The shape of x2Scale must be [" + std::to_string(static_cast<int64_t>(inputParams_.nSize)) + ", " +
                    std::to_string(ops::CeilDiv<int64_t>(inputParams_.kSize, (GROUP_ALIGN_SIZE * 2))) + ", 2]"),
            return false);
    } else if (inputParams_.groupSize > 0) {
        return AnalyzeX2ScalePerGroupShape(x2ScaleShape);
    } else if (x2ScaleShapeSize == 1) {
        inputParams_.antiQuantType = QuantType::PER_TENSOR;
    } else {
        OP_TILING_CHECK(x2ScaleShapeSize != inputParams_.nSize,
                        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(inputParams_.opName, "x2Scale",
                                                                  std::to_string(x2ScaleShapeSize).c_str(),
                                                                  "The shape size of x2Scale must be equal to nSize"),
                        return false);
        inputParams_.antiQuantType = QuantType::PER_CHANNEL;
    }
    return true;
}

bool QuantBatchMatmulV4TilingBase::AnalyzeYScaleOffsetShape(
    const gert::StorageShape *yScaleShape, const gert::StorageShape *yOffsetShape) const
{
    OP_TILING_CHECK(!IsNotEmptyShape(yScaleShape) && IsNotEmptyShape(yOffsetShape),
        OP_LOGE(inputParams_.opName, "not support quant_offset without quant_scale"),
        return false);
    if (!IsNotEmptyShape(yScaleShape)) {
        OP_TILING_CHECK(
            inputParams_.antiQuantType == QuantType::PER_GROUP,
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(inputParams_.opName, "yScale", "0",
                                                      "When the quant mode is per_group and the format of x2 is "
                                                      "FRACTAL_NZ, the shape size of yScale can not be 0"),
            return false);
        return true;
    }
    size_t yScaleShapeSize = static_cast<size_t>(yScaleShape->GetStorageShape().GetShapeSize());
    OP_TILING_CHECK(yScaleShapeSize == 0 && inputParams_.cDtype == ge::DT_INT8,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        inputParams_.opName, "yScale", std::to_string(yScaleShapeSize).c_str(),
                        "When the dtype of y is INT8, the shape size of yScale can not be 0"),
                    return false);
    OP_TILING_CHECK(
        yScaleShape->GetStorageShape().GetDimNum() > VALID_INPUT_DIM_NUM ||
            (yScaleShape->GetStorageShape().GetDimNum() == VALID_INPUT_DIM_NUM &&
             yScaleShape->GetStorageShape().GetDim(0) != 1),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            inputParams_.opName, "yScale", Ops::Base::ToString(yScaleShape->GetStorageShape()).c_str(),
            "The shape of yScale must be [1, n] or [n,]"),
        return false);
    OP_TILING_CHECK(
        IsNotEmptyShape(yOffsetShape) && yScaleShape->GetStorageShape() != yOffsetShape->GetStorageShape(),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            inputParams_.opName, "yScale, yOffset", Ops::Base::ToString(yScaleShape->GetStorageShape()).c_str(),
            "The shape of yScale must be equal to the shape of yOffset"),
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
    OPS_CHECK_NULL_WITH_CONTEXT(context_, compileInfoPtr);

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
        OP_TILING_CHECK(
            !CalcUBSize(1UL, inputParams_.groupSize),
            OP_LOGE(inputParams_.opName, "group size[%lu] cannot full load to UB", inputParams_.groupSize),
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
        OP_LOGE(
            inputParams_.opName, "tiling data size[%zu] not aligned to 8", tilingDataSize_),
        return ge::GRAPH_FAILED);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize_);
    uint32_t usedAicNum = tilingData_->cubeNumBlocksM * tilingData_->cubeNumBlocksN;
    uint32_t usedAivNum = tilingData_->vecNumBlocksK * tilingData_->vecNumBlocksN;
    context_->SetBlockDim(std::max(usedAicNum, CalcTschNumBlocks(usedAivNum, aicNum_, aivNum_)));

    OP_TILING_CHECK(
        !CheckFinalTilingData(), PrintTilingData(false);
        OP_LOGE(inputParams_.opName, "get invalid tiling data, check above validate rule"),
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
    if (debugLevel) {
        OPS_LOG_D(inputParams_.opName, "%ld", DumpTilingDataToLog(debugLevel));
    } else {
        OPS_LOG_E(inputParams_.opName, "%ld", DumpTilingDataToLog(debugLevel));
    }
}

int64_t QuantBatchMatmulV4TilingBase::DumpTilingDataToLog(bool debugLevel)
{
    std::stringstream ss;
    ss << "kAlign: " << tilingData_->kAlign << " nAlign: " << tilingData_->nAlign
       << " kSize: " << tilingData_->kSize << " nSize: " << tilingData_->nSize
       << " mSize: " << tilingData_->mSize << " groupSize: " << tilingData_->groupSize
       << " cubeNumBlocksN: " << static_cast<uint32_t>(tilingData_->cubeNumBlocksN)
       << " cubeNumBlocksM: " << static_cast<uint32_t>(tilingData_->cubeNumBlocksM);
    if (debugLevel) {
        OPS_LOG_D(inputParams_.opName, "tiling data: %s", ss.str().c_str());
    } else {
        OPS_LOG_E(inputParams_.opName, "tiling data: %s", ss.str().c_str());
    }
    PrintMatMulTiling();
    return 0;
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
        OPS_CHECK_NULL_WITH_CONTEXT(context_, tilingDataManager_);
        tilingData_ = tilingDataManager_.get();
    }
    OPS_CHECK_NULL_WITH_CONTEXT(context_, tilingData_);
    OP_TILING_CHECK(context_->GetRawTilingData()->GetCapacity() < tilingDataSize_,
        OP_LOGE(inputParams_.opName,
            "tiling data capacity %zu < actual tiling data size %zu",
            context_->GetRawTilingData()->GetCapacity(),
            tilingDataSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("QuantBatchMatmulV4", QuantBatchMatmulV4RegBase, BASIC_PRIORITY);

}  // namespace optiling
