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
 * \file rotate_quant_tiling.cpp
 * \brief RotateQuant算子arch35 tiling实现
 */
#include "rotate_quant_tiling.h"
#include "error_util.h"
#include "log/log.h"
#include "util/math_util.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "platform/platform_infos_def.h"
#include "platform/soc_spec.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_key.h"
#include "op_host/tiling_templates_registry.h"

namespace {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t ROT_INDEX = 1;
constexpr uint32_t ALPHA_INDEX = 2;
constexpr uint32_t Y_INDEX = 0;
constexpr uint32_t SCALE_INDEX = 1;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t MAX_X_DIM_NUM = 7;
constexpr uint32_t MIN_X_DIM_NUM = 1;
constexpr uint32_t MIN_ROT_DIM_NUM = 2;
constexpr uint32_t MAX_ROT_DIM_NUM = 3;
constexpr uint32_t MIN_M = 16;
constexpr uint32_t INIT_ACCUMULATE_VALUE = 1;
constexpr uint32_t MIN_BASE_M = 1;
constexpr int64_t MX_SCALE_LAST_SIZE = 2;
constexpr int64_t MX_SCALE_CEIL_NUM = 64;
constexpr int64_t SYS_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;
constexpr uint64_t DEFAULT_TILING_KEY = 0;
constexpr uint32_t WORKSPACE_NUM = 1;

constexpr uint32_t YDTYPE_ATTR_INDEX = 0;
constexpr uint32_t AXIS_ATTR_INDEX = 1;
constexpr uint32_t ROUND_MODE_ATTR_INDEX = 2;
constexpr uint32_t SCALE_ALG_ATTR_INDEX = 3;
constexpr uint32_t DST_TYPE_MAX_ATTR_INDEX = 4;
constexpr uint32_t TRANS_ATTR_INDEX = 5;

constexpr int32_t MIN_SCALE_ALG = 0;
constexpr int32_t MAX_SCALE_ALG = 2;
constexpr int32_t SCALE_ALG_PER_CHANNEL = 1;
constexpr float DST_TYPE_MAX_LOWER = 6.0f;
constexpr float DST_TYPE_MAX_UPPER = 12.0f;
constexpr float DST_TYPE_MAX_DEFAULT = 0.0f;
constexpr float INVERSE_UNIT = 1.0f;
constexpr int32_t DEFAULT_AXIS = -1;

constexpr int64_t ROT_K_SIZE_32 = 32;
constexpr int64_t ROT_K_SIZE_64 = 64;
constexpr int64_t ROT_K_SIZE_128 = 128;

constexpr uint32_t ALPHA_DIM_NUM = 1;
constexpr uint32_t ALPHA_DIM_SIZE = 1;

constexpr uint64_t ROT_TYPE_MULTI = 0;
constexpr uint64_t ROT_TYPE_SINGLE = 1;

constexpr uint32_t ELEMENT_BYTES_16BIT = 2;
constexpr uint32_t ELEMENT_BYTES_32BIT = 4;
constexpr uint32_t SINGLE_ROTATION_B_SIZE = 1;
constexpr uint32_t SCHEDULE_MODE_STATIC = 1;
constexpr int32_t TEMPLATE_PRIORITY = 0;

enum class RoundModeList {
    MODE_ROUND = 0,
    MODE_FLOOR = 1,
    MODE_CEIL = 2,
    MODE_TRUNC = 3,
    MODE_RINT = 4,
    MODE_HYBRID = 5,
    MODE_UNDEFINED = -1,
};

RoundModeList GetRoundMode(const std::string& roundMode)
{
    if (roundMode == "rint") {
        return RoundModeList::MODE_RINT;
    } else if (roundMode == "round") {
        return RoundModeList::MODE_ROUND;
    } else if (roundMode == "floor") {
        return RoundModeList::MODE_FLOOR;
    }
    return RoundModeList::MODE_UNDEFINED;
}
} // namespace

namespace Ops {
namespace NN {
namespace RotateQuant {

ge::graphStatus RotateQuantAptTiling::GetPlatformInfo() { return ge::GRAPH_SUCCESS; }

void RotateQuantAptTiling::InitCompileInfo()
{
    if (context_ == nullptr) {
        OP_LOGE("RotateQuantApt", "context_ is null");
        return;
    }
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }

    const auto& ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo_.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfo_.l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfo_.l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfo_.l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfo_.l0CSize);
    compileInfo_.aicNum = ascendcPlatform.GetCoreNumAic();

    if (compileInfo_.aicNum <= 0) {
        OP_LOGE(context_->GetNodeName(), "aicNum <= 0");
        return;
    }
    OP_LOGD(context_->GetNodeName(), "RotateQuantApt InitCompileInfo Success");
}

ge::graphStatus RotateQuantAptTiling::GetShapeAttrsInfo()
{
    tilingDataSize_ = sizeof(RotateQuantAptOpt::RotateQuantAptTilingData);
    if (inputParams_.initFlag) {
        OP_LOGD(inputParams_.opName, "No need to get shape and attrs from tiling context again");
        return ge::GRAPH_SUCCESS;
    }

    inputParams_.opName = context_->GetNodeName();
    OP_LOGI(inputParams_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    OP_TILING_CHECK(
        CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid context."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        AnalyzeAttrs() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid attrs."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        AnalyzeDtype() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid dtypes."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        AnalyzeShapes() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid shapes."), return ge::GRAPH_FAILED);

    OP_LOGD(
        inputParams_.opName, "Input params: MNK[%ld, %ld, %ld], nKRatio[%ld].", inputParams_.M, inputParams_.N,
        inputParams_.K, inputParams_.nKRatio);

    inputParams_.initFlag = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantAptTiling::CheckContext()
{
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(X_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(X_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(ROT_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(ROT_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(Y_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(Y_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(SCALE_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(SCALE_INDEX));
    OP_TILING_CHECK(
        context_->GetRawTilingData()->GetCapacity() < tilingDataSize_,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName, "context tiling data capacity %zu < actual tiling data size %zu.",
            context_->GetRawTilingData()->GetCapacity(), tilingDataSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantAptTiling::ParseAttrValues()
{
    const gert::RuntimeAttrs* attrs = context_->GetAttrs();
    const int64_t* yDtypePtr = attrs->GetAttrPointer<int64_t>(YDTYPE_ATTR_INDEX);
    if (yDtypePtr != nullptr) {
        inputParams_.yDtype = static_cast<ge::DataType>(*yDtypePtr);
    }

    const int64_t* axisPtr = attrs->GetAttrPointer<int64_t>(AXIS_ATTR_INDEX);
    if (axisPtr != nullptr) {
        inputParams_.axis = static_cast<int32_t>(*axisPtr);
    }

    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<char>(ROUND_MODE_ATTR_INDEX));
    std::string roundModeStr = attrs->GetAttrPointer<char>(ROUND_MODE_ATTR_INDEX);
    RoundModeList roundMode = GetRoundMode(roundModeStr);
    OP_CHECK_IF(
        (roundMode == RoundModeList::MODE_UNDEFINED),
        OP_LOGE(
            context_->GetNodeName(), "invalid round_mode:%s; round_mode should be one of {rint, floor, round}",
            roundModeStr.c_str()),
        return ge::GRAPH_FAILED);
    inputParams_.roundMode = static_cast<int32_t>(roundMode);

    const int64_t* scaleAlgPtr = attrs->GetAttrPointer<int64_t>(SCALE_ALG_ATTR_INDEX);
    if (scaleAlgPtr != nullptr) {
        inputParams_.scaleAlg = static_cast<int32_t>(*scaleAlgPtr);
        OP_TILING_CHECK(
            inputParams_.scaleAlg < MIN_SCALE_ALG || inputParams_.scaleAlg > MAX_SCALE_ALG,
            OP_LOGE(
                inputParams_.opName, "The scaleAlg[%d] should be in [%d, %d].", inputParams_.scaleAlg, MIN_SCALE_ALG,
                MAX_SCALE_ALG),
            return ge::GRAPH_FAILED);
    }

    tilingData_.invDstTypeMax = DST_TYPE_MAX_DEFAULT;
    const float* dstTypeMaxPtr = attrs->GetAttrPointer<float>(DST_TYPE_MAX_ATTR_INDEX);
    if (dstTypeMaxPtr != nullptr) {
        inputParams_.dstTypeMax = *dstTypeMaxPtr;
    }

    const bool* transPtr = attrs->GetAttrPointer<bool>(TRANS_ATTR_INDEX);
    if (transPtr != nullptr) {
        inputParams_.trans = *transPtr;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantAptTiling::AnalyzeAttrs()
{
    const gert::RuntimeAttrs* attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGD(inputParams_.opName, "No attrs provided, using default values");
        return ge::GRAPH_SUCCESS;
    }

    OP_TILING_CHECK(
        ParseAttrValues() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Failed to parse attrs."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        inputParams_.trans == true, OP_LOGE(inputParams_.opName, "The trans only support false."),
        return ge::GRAPH_FAILED);

    OP_LOGD(
        inputParams_.opName, "Attrs: yDtype=%d, axis=%d, roundMode=%d, scaleAlg=%d, dstTypeMax=%f, trans=%d",
        inputParams_.yDtype, inputParams_.axis, inputParams_.roundMode, inputParams_.scaleAlg, inputParams_.dstTypeMax,
        inputParams_.trans);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantAptTiling::CheckAlphaInput()
{
    if (context_->GetComputeNodeInfo()->GetInputsNum() > ALPHA_INDEX) {
        auto alphaDesc = context_->GetInputDesc(ALPHA_INDEX);
        if (alphaDesc != nullptr) {
            auto alphaDtype = alphaDesc->GetDataType();
            OP_TILING_CHECK(
                alphaDtype != ge::DT_BF16, OP_LOGE(inputParams_.opName, "alpha dtype should be bf16."),
                return ge::GRAPH_FAILED);
            auto alphaShape = context_->GetInputShape(ALPHA_INDEX)->GetStorageShape();
            OP_TILING_CHECK(
                alphaShape.GetDimNum() != ALPHA_DIM_NUM || alphaShape.GetDim(0) != ALPHA_DIM_SIZE,
                OP_LOGE(inputParams_.opName, "alpha shape should be (1,)."), return ge::GRAPH_FAILED);
            inputParams_.hasAlpha = true;
            OP_LOGD(inputParams_.opName, "hasAlpha detected");
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantAptTiling::AnalyzeDtype()
{
    inputParams_.dataType = context_->GetInputDesc(X_INDEX)->GetDataType();
    auto rotDtype = context_->GetInputDesc(ROT_INDEX)->GetDataType();
    auto yDtype = context_->GetOutputDesc(Y_INDEX)->GetDataType();
    auto scaleDtype = context_->GetOutputDesc(SCALE_INDEX)->GetDataType();

    OP_TILING_CHECK(inputParams_.dataType != ge::DT_FLOAT16 && inputParams_.dataType != ge::DT_BF16,
        OP_LOGE(inputParams_.opName, "x dtype should be fp16 or bf16 for ascend950."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(inputParams_.dataType != rotDtype, 
        OP_LOGE(inputParams_.opName, "x and rotation dtype should be same."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(yDtype != ge::DT_FLOAT4_E2M1 && yDtype != ge::DT_FLOAT8_E4M3FN && yDtype != ge::DT_FLOAT8_E5M2,
        OP_LOGE(inputParams_.opName, "y dtype should be in [FLOAT4_E2M1, FLOAT8_E4M3FN, FLOAT8_E5M2] for ascend950."),
        return ge::GRAPH_FAILED);
    if (yDtype == ge::DT_FLOAT4_E2M1) {
        OP_TILING_CHECK(inputParams_.scaleAlg != MIN_SCALE_ALG && inputParams_.scaleAlg != MAX_SCALE_ALG,
            OP_LOGE(
                inputParams_.opName, "When y's data type is FLOAT4_E2M1, scaleAlg must be %d or %d.", MIN_SCALE_ALG,
                MAX_SCALE_ALG),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(inputParams_.roundMode != static_cast<int32_t>(RoundModeList::MODE_RINT),
            OP_LOGE(inputParams_.opName, "When y's data type is FLOAT8_E4M3/FLOAT8_E5M2, round_mode only support rint."),
            return ge::GRAPH_FAILED);

        OP_TILING_CHECK(inputParams_.scaleAlg != MIN_SCALE_ALG && inputParams_.scaleAlg != SCALE_ALG_PER_CHANNEL,
            OP_LOGE(
                inputParams_.opName, "When y's data type is FLOAT8_E4M3/FLOAT8_E5M2, scaleAlg must be %d or %d.",
                MIN_SCALE_ALG, SCALE_ALG_PER_CHANNEL),
            return ge::GRAPH_FAILED);
    }

    if (yDtype == ge::DT_FLOAT4_E2M1 && inputParams_.scaleAlg == MAX_SCALE_ALG) {
        OP_TILING_CHECK(inputParams_.dstTypeMax < DST_TYPE_MAX_LOWER || inputParams_.dstTypeMax > DST_TYPE_MAX_UPPER,
            OP_LOGE(
                inputParams_.opName,
                "When y's data type is FLOAT4_E2M1 and scaleAlg is %d, dstTypeMax must be in [%f, %f].", MAX_SCALE_ALG,
                DST_TYPE_MAX_LOWER, DST_TYPE_MAX_UPPER),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(inputParams_.dstTypeMax != DST_TYPE_MAX_DEFAULT,
            OP_LOGE(inputParams_.opName, "dstTypeMax must equal %f.", DST_TYPE_MAX_DEFAULT), return ge::GRAPH_FAILED);
    }

    OP_TILING_CHECK(scaleDtype != ge::DT_FLOAT8_E8M0,
        OP_LOGE(inputParams_.opName, "scale dtype should be FLOAT8_E8M0 for ascend950."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAlphaInput() != ge::GRAPH_SUCCESS, 
        OP_LOGE(inputParams_.opName, "Invalid alpha input."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantAptTiling::ValidateRotShape(const gert::Shape& rotShape)
{
    OP_TILING_CHECK(
        rotShape.GetDimNum() < MIN_ROT_DIM_NUM || rotShape.GetDimNum() > MAX_ROT_DIM_NUM,
        OP_LOGE(inputParams_.opName, "Input rot rank[%zu] should be in [2, 3].", rotShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    inputParams_.K = rotShape.GetDim(rotShape.GetDimNum() - 1);
    OP_TILING_CHECK(
        inputParams_.K != ROT_K_SIZE_32 && inputParams_.K != ROT_K_SIZE_64 && inputParams_.K != ROT_K_SIZE_128,
        OP_LOGE(
            inputParams_.opName, "Input rot last dim size must be in [%ld, %ld, %ld].", ROT_K_SIZE_32, ROT_K_SIZE_64,
            ROT_K_SIZE_128),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        inputParams_.N % inputParams_.K != 0,
        OP_LOGE(inputParams_.opName, "N(%ld) must be divisible by K(%ld).", inputParams_.N, inputParams_.K),
        return ge::GRAPH_FAILED);

    inputParams_.nKRatio = inputParams_.N / inputParams_.K;

    if (rotShape.GetDimNum() == MIN_ROT_DIM_NUM) {
        inputParams_.rotType = ROT_TYPE_SINGLE;
        OP_TILING_CHECK(
            rotShape.GetDim(0) != inputParams_.K,
            OP_LOGE(
                inputParams_.opName, "Input rotationMatrix shape should be (%ld, %ld), please check.", inputParams_.K,
                inputParams_.K),
            return ge::GRAPH_FAILED);
    } else {
        inputParams_.rotType = (inputParams_.nKRatio > 1) ? ROT_TYPE_MULTI : ROT_TYPE_SINGLE;
        OP_TILING_CHECK(
            rotShape.GetDim(0) != inputParams_.nKRatio || rotShape.GetDim(1) != inputParams_.K,
            OP_LOGE(
                inputParams_.opName, "Input rotationMatrix shape should be (%ld, %ld, %ld), please check.",
                inputParams_.nKRatio, inputParams_.K, inputParams_.K),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantAptTiling::ValidateScaleShape(const gert::Shape& mxscaleShape, const gert::Shape& scaleShape)
{
    OP_CHECK_IF(
        mxscaleShape != scaleShape,
        OP_LOGE(
            inputParams_.opName, "The shape of output mxscale %s is incorrect, correct shape is %s, please check.",
            Shape2String(scaleShape).c_str(), Shape2String(mxscaleShape).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantAptTiling::AnalyzeShapes()
{
    const auto& xShape = context_->GetInputShape(X_INDEX)->GetStorageShape();
    const auto& rotShape = context_->GetInputShape(ROT_INDEX)->GetStorageShape();

    OP_TILING_CHECK(
        xShape.GetDimNum() < MIN_X_DIM_NUM || xShape.GetDimNum() > MAX_X_DIM_NUM,
        OP_LOGE(
            inputParams_.opName, "Input x rank[%zu] should be in [%d, %d].", MIN_X_DIM_NUM, MAX_X_DIM_NUM,
            xShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        inputParams_.axis != DEFAULT_AXIS && inputParams_.axis != static_cast<int32_t>(xShape.GetDimNum() - 1),
        OP_LOGE(
            inputParams_.opName, "axis should be %d or %d.", DEFAULT_AXIS,
            static_cast<int32_t>(xShape.GetDimNum() - 1)),
        return ge::GRAPH_FAILED);

    inputParams_.M = INIT_ACCUMULATE_VALUE;
    for (size_t i = 0; i < xShape.GetDimNum() - 1; i++) {
        inputParams_.M *= xShape.GetDim(i);
    }
    inputParams_.N = xShape.GetDim(xShape.GetDimNum() - 1);

    OP_TILING_CHECK(
        ValidateRotShape(rotShape) != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid rot shape."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void RotateQuantAptTiling::PrintTilingData()
{
    auto& tiling = tilingData_;
    OP_LOGD(inputParams_.opName, "=== RotateQuantApt Tiling Data ===");
    OP_LOGD(inputParams_.opName, "M:            [%u]", tiling.M);
    OP_LOGD(inputParams_.opName, "N:            [%u]", tiling.N);
    OP_LOGD(inputParams_.opName, "K:            [%u]", tiling.K);
    OP_LOGD(inputParams_.opName, "nKRatio:      [%u]", tiling.nKRatio);
    OP_LOGD(inputParams_.opName, "mL1:          [%u]", tiling.mL1);
    OP_LOGD(inputParams_.opName, "tailML1:      [%u]", tiling.tailML1);
    OP_LOGD(inputParams_.opName, "nL1:          [%u]", tiling.nL1);
    OP_LOGD(inputParams_.opName, "kL1:          [%u]", tiling.kL1);
    OP_LOGD(inputParams_.opName, "baseM:        [%u]", tiling.baseM);
    OP_LOGD(inputParams_.opName, "baseN:        [%u]", tiling.baseN);
    OP_LOGD(inputParams_.opName, "baseK:        [%u]", tiling.baseK);
    OP_LOGD(inputParams_.opName, "tailML1:      [%u]", tiling.tailML1);
    OP_LOGD(inputParams_.opName, "dstTypeMax:   [%f]", tiling.dstTypeMax);
    OP_LOGD(inputParams_.opName, "invDstTypeMax:[%f]", tiling.invDstTypeMax);
    OP_LOGD(inputParams_.opName, "axis:         [%d]", tiling.axis);
    OP_LOGD(inputParams_.opName, "roundMode:    [%d]", tiling.roundMode);
    OP_LOGD(inputParams_.opName, "scaleAlg:     [%d]", tiling.scaleAlg);
    OP_LOGD(inputParams_.opName, "trans:        [%u]", tiling.trans);
    OP_LOGD(inputParams_.opName, "hasAlpha:     [%u]", tiling.hasAlpha);
    OP_LOGD(inputParams_.opName, "=== End Tiling Data ===");
}

ge::graphStatus RotateQuantAptTiling::CalcTilingData()
{
    tilingData_.M = static_cast<uint32_t>(inputParams_.M);
    tilingData_.N = static_cast<uint32_t>(inputParams_.N);
    tilingData_.K = static_cast<uint32_t>(inputParams_.K);
    tilingData_.B =
        (inputParams_.rotType == ROT_TYPE_MULTI) ? static_cast<uint32_t>(inputParams_.nKRatio) : SINGLE_ROTATION_B_SIZE;
    tilingData_.nKRatio = static_cast<uint32_t>(inputParams_.nKRatio);

    tilingData_.dstTypeMax = inputParams_.dstTypeMax;
    if (tilingData_.dstTypeMax > DST_TYPE_MAX_DEFAULT) {
        tilingData_.invDstTypeMax = INVERSE_UNIT / tilingData_.dstTypeMax;
    } else {
        tilingData_.invDstTypeMax = DST_TYPE_MAX_DEFAULT;
    }

    tilingData_.axis = inputParams_.axis;
    tilingData_.roundMode = inputParams_.roundMode;
    tilingData_.scaleAlg = inputParams_.scaleAlg;
    tilingData_.trans = static_cast<uint32_t>(inputParams_.trans);
    tilingData_.hasAlpha = static_cast<uint32_t>(inputParams_.hasAlpha);
    return ge::GRAPH_SUCCESS;
}

bool RotateQuantAptTiling::SetMatmulTiling() 
{
    tilingData_.kL1 = tilingData_.K;
    tilingData_.nL1 = tilingData_.K;
    tilingData_.baseK = tilingData_.kL1;
    tilingData_.baseN = tilingData_.nL1;
    uint32_t baseML0A = compileInfo_.l0ASize / tilingData_.baseK / DOUBLE_BUFFER / ELEMENT_BYTES_16BIT;
    uint32_t baseML0C = compileInfo_.l0CSize / tilingData_.baseN / DOUBLE_BUFFER / ELEMENT_BYTES_32BIT;
    tilingData_.baseM = std::max(std::min(baseML0A, baseML0C), MIN_BASE_M);

    uint32_t totalCoreNum = static_cast<uint32_t>(compileInfo_.aicNum);
    if (tilingData_.B == SINGLE_ROTATION_B_SIZE) {
        uint32_t mL1Max = static_cast<uint32_t>(
            (compileInfo_.l1Size - tilingData_.kL1 * tilingData_.nL1 * ELEMENT_BYTES_16BIT) /
            (tilingData_.kL1 * ELEMENT_BYTES_16BIT));
        uint32_t mL1PerCore = Ops::Base::CeilDiv(tilingData_.M * tilingData_.nKRatio, totalCoreNum);
        mL1PerCore = Ops::Base::CeilAlign(mL1PerCore, MIN_M); // 16对齐
        tilingData_.mL1 = std::min({mL1Max, mL1PerCore, tilingData_.baseM});
        inputParams_.realCoreNum =
            std::min(Ops::Base::CeilDiv(tilingData_.M * tilingData_.nKRatio, tilingData_.mL1), totalCoreNum);
        uint32_t totalEffectiveM = tilingData_.M * tilingData_.nKRatio;
        uint32_t remainderM = totalEffectiveM % tilingData_.mL1;
        tilingData_.tailML1 = (remainderM == 0) ? tilingData_.mL1 : remainderM;
    } else {
        tilingData_.mL1 = tilingData_.baseM;
        uint32_t mL1LoopNum = Ops::Base::CeilDiv(tilingData_.M, tilingData_.mL1);
        if (mL1LoopNum * tilingData_.B < totalCoreNum) {
            // mL1 = baseM情况下如果核数未用满，调整mL1大小
            mL1LoopNum = totalCoreNum / tilingData_.B;
            tilingData_.mL1 = Ops::Base::CeilDiv(tilingData_.M, mL1LoopNum);
            tilingData_.mL1 = Ops::Base::CeilAlign(tilingData_.mL1, MIN_M); // 16对齐
            mL1LoopNum = Ops::Base::CeilDiv(tilingData_.M, tilingData_.mL1);
        }
        uint32_t remainderM = tilingData_.M % tilingData_.mL1;
        tilingData_.tailML1 = (remainderM == 0) ? tilingData_.mL1 : remainderM;
        inputParams_.realCoreNum = std::min(mL1LoopNum * tilingData_.B, totalCoreNum);
    }
    return true; 
}

ge::graphStatus RotateQuantAptTiling::DoOpTiling()
{
    OP_TILING_CHECK(
        CalcTilingData() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "CalcTilingData failed."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        !SetMatmulTiling(), OP_LOGE(inputParams_.opName, "SetMatmulTiling failed."), return ge::GRAPH_FAILED);

    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantAptTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t RotateQuantAptTiling::GetTilingKey() const { return DEFAULT_TILING_KEY; }

ge::graphStatus RotateQuantAptTiling::GetWorkspaceSize()
{
    workspaceSize_ = SYS_WORKSPACE_SIZE +
                     static_cast<size_t>(inputParams_.M) * static_cast<size_t>(inputParams_.N) * ELEMENT_BYTES_32BIT;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantAptTiling::PostTiling()
{
    context_->SetBlockDim(inputParams_.realCoreNum);
    context_->SetScheduleMode(SCHEDULE_MODE_STATIC);
    errno_t ret = memcpy_s(
        context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
        reinterpret_cast<void*>(&tilingData_), tilingDataSize_);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize_);
    size_t* workspaces = context_->GetWorkspaceSizes(WORKSPACE_NUM);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

// 注册tiling类（使用arch特定的注册宏）
REGISTER_TILING_TEMPLATE_WITH_ARCH(
    RotateQuant, RotateQuantAptTiling, static_cast<int32_t>(NpuArch::DAV_3510), TEMPLATE_PRIORITY);

static ge::graphStatus RotateQuantAptTilingFunc(gert::TilingContext* context)
{
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "RotateQuant", "TilingContext is null!");
    return Ops::NN::Optiling::TilingRegistryArch::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingParseForRotateQuantApt(gert::TilingParseContext* context)
{
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "RotateQuant", "TilingParseContext is null!");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RotateQuant)
    .Tiling(RotateQuantAptTilingFunc)
    .TilingParse<RotateQuantAptCompileInfo>(TilingParseForRotateQuantApt);

} // namespace RotateQuant
} // namespace NN
} // namespace Ops