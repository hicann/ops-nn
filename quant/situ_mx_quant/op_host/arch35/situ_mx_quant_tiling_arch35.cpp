/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file situ_mx_quant_tiling_arch35.cpp
 * \brief Tiling implementation for Situ + MX quantization
 */

#include "situ_mx_quant_tiling_arch35.h"
#include "../../op_kernel/arch35/situ_mx_quant_tiling_data.h"
#include "quant/situ_mx_quant/op_kernel/arch35/situ_mx_quant_tiling_key.h"

#include <cmath>
#include <sstream>
#include "platform/platform_info.h"
#include "op_host/tiling_util.h"
#include "util/math_util.h"

using namespace std;
using namespace ge;
using namespace AscendC;
using namespace SituMxQuantOp;

namespace optiling {
// ==================== Constants ====================
constexpr int64_t INDEX_ATTR_BETA = 0;
constexpr int64_t INDEX_ATTR_LINEAR_BETA = 1;
constexpr int64_t INDEX_ATTR_ACTIVATE_LEFT = 2;
constexpr int64_t INDEX_ATTR_AXIS = 3;
constexpr int64_t INDEX_ATTR_DST_TYPE = 4;
constexpr int64_t INDEX_ATTR_ROUND_MODE = 5;

constexpr int64_t BYTES_OF_FP16 = 2;
constexpr int64_t BYTES_OF_BF16 = 2;
constexpr int64_t BYTES_OF_FP8 = 1;
constexpr int64_t BYTES_OF_INT16 = 2;
constexpr int64_t RESERVED_UB_SIZE = 32;
constexpr int64_t RESERVED_UB_FOR_ALIGN = 128;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t CONST_TWO = 2;
constexpr int64_t DTYPE_35 = 35;   // FP8_E5M2
constexpr int64_t DTYPE_36 = 36;   // FP8_E4M3FN
constexpr int64_t DTYPE_40 = 40;   // FP4_E2M1
constexpr int64_t DTYPE_41 = 41;   // FP4_E1M2
constexpr int64_t BASE_DIM1 = 256; // basic block size for last axis

const set<ge::DataType> INPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT16, ge::DT_BF16};
const set<ge::DataType> Y_SUPPORT_DTYPE_SET = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E4M3FN,
                                               ge::DT_FLOAT8_E5M2};
const set<ge::DataType> SCALE_SUPPORT_DTYPE_SET = {ge::DT_FLOAT8_E8M0};

// ==================== Helper Functions ====================
template <typename T>
static string Shape2String(const T& shape)
{
    ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

static RoundModeList GetRoundModeEnum(const string& roundMode)
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

// ==================== Class Methods ====================
ge::graphStatus SituMxQuantRegbaseTiling::GetNpuInfo()
{
    OP_LOGD(context_->GetNodeName(), "GetNpuInfo begin.");
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo_.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo_.totalCoreNum <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get core num."),
                return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo_.ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo_.ubSize <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get UB size."),
                return ge::GRAPH_FAILED);
    OP_LOGI(context_->GetNodeName(), "CompileInfo: totalCoreNum=%ld, ubSize=%ld", compileInfo_.totalCoreNum,
            compileInfo_.ubSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SituMxQuantRegbaseTiling::ParseAttrs()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto* attrBeta = attrs->GetAttrPointer<float>(INDEX_ATTR_BETA);
    attrParam_.beta = (attrBeta != nullptr) ? *attrBeta : 1.0f;
    OP_CHECK_IF((attrParam_.beta <= 0.0f),
                OP_LOGE(context_->GetNodeName(), "beta must be greater than 0, but got %f.", attrParam_.beta),
                return ge::GRAPH_FAILED);

    auto* attrLinearBeta = attrs->GetAttrPointer<float>(INDEX_ATTR_LINEAR_BETA);
    attrParam_.linearBeta = (attrLinearBeta != nullptr) ? *attrLinearBeta : 0.0f;
    attrParam_.hasLinearBeta = (attrParam_.linearBeta > 0.0f);

    auto* attrActivateLeft = attrs->GetAttrPointer<bool>(INDEX_ATTR_ACTIVATE_LEFT);
    attrParam_.activateLeft = (attrActivateLeft != nullptr) ? *attrActivateLeft : false;

    auto* attrAxis = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_AXIS);
    attrParam_.axis = (attrAxis != nullptr) ? static_cast<int64_t>(*attrAxis) : -1;
    OP_CHECK_IF((attrParam_.axis != -1),
                OP_LOGE(context_->GetNodeName(), "Only axis=-1 is supported currently, but got %ld.", attrParam_.axis),
                return ge::GRAPH_FAILED);

    auto* attrDstType = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_DST_TYPE);
    attrParam_.dstType = (attrDstType != nullptr) ? static_cast<int64_t>(*attrDstType) : DTYPE_40;
    OP_CHECK_IF((attrParam_.dstType != DTYPE_35) && (attrParam_.dstType != DTYPE_36) &&
                    (attrParam_.dstType != DTYPE_40) && (attrParam_.dstType != DTYPE_41),
                OP_LOGE(context_->GetNodeName(),
                        "Invalid dstType: %ld, only 35(E5M2), 36(E4M3FN), "
                        "40(E2M1), 41(E1M2) supported.",
                        attrParam_.dstType),
                return ge::GRAPH_FAILED);

    // Parse round_mode
    const char* attrRoundMode = attrs->GetAttrPointer<char>(INDEX_ATTR_ROUND_MODE);
    string roundModeStr = (attrRoundMode != nullptr) ? attrRoundMode : "rint";
    auto roundModeEnum = GetRoundModeEnum(roundModeStr);
    OP_CHECK_IF((roundModeEnum == RoundModeList::MODE_UNDEFINED),
                OP_LOGE(context_->GetNodeName(), "Invalid round_mode: %s", roundModeStr.c_str()),
                return ge::GRAPH_FAILED);
    attrParam_.roundMode = roundModeStr;
    if (roundModeEnum == RoundModeList::MODE_FLOOR) {
        roundMode_ = TPL_FLOOR;
    } else if (roundModeEnum == RoundModeList::MODE_ROUND) {
        roundMode_ = TPL_ROUND;
    } else {
        roundMode_ = TPL_RINT;
    }

    // FP8 output only supports rint
    OP_CHECK_IF(
        (attrParam_.dstType == DTYPE_35 || attrParam_.dstType == DTYPE_36) &&
            (roundModeEnum != RoundModeList::MODE_RINT),
        OP_LOGE(context_->GetNodeName(), "FP8 output requires round_mode=rint, but got %s.", roundModeStr.c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SituMxQuantRegbaseTiling::ValidateInput()
{
    auto xDtype = context_->GetInputDesc(0)->GetDataType();
    OP_CHECK_IF(
        (INPUT_SUPPORT_DTYPE_SET.find(xDtype) == INPUT_SUPPORT_DTYPE_SET.end()),
        OP_LOGE(context_->GetNodeName(), "Input x dtype %d is not supported. Only FLOAT16 and BF16 are supported.",
                static_cast<int>(xDtype)),
        return ge::GRAPH_FAILED);
    inputInfo_.xDtype = xDtype;

    auto xShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    int64_t dimNum = static_cast<int64_t>(xShape->GetStorageShape().GetDimNum());
    OP_LOGI(context_->GetNodeName(), "Input x shape = %s", Shape2String(xShape->GetStorageShape()).c_str());
    int64_t xSize = xShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(
        (dimNum < 1 || xSize == 0),
        OP_LOGE(context_->GetNodeName(), "rank of x must >= 1, but is %ld, and not support empty tensor", dimNum),
        return ge::GRAPH_FAILED);
    inputInfo_.dimNum = dimNum;
    inputInfo_.inputDim2 = xShape->GetStorageShape().GetDim(dimNum - 1);
    OP_CHECK_IF(
        (inputInfo_.inputDim2 % CONST_TWO != 0),
        OP_LOGE(context_->GetNodeName(), "Last dimension must be divisible by 2, but got %ld.", inputInfo_.inputDim2),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SituMxQuantRegbaseTiling::ValidateOutput()
{
    auto yDtype = context_->GetOutputDesc(0)->GetDataType();
    OP_CHECK_IF((Y_SUPPORT_DTYPE_SET.find(yDtype) == Y_SUPPORT_DTYPE_SET.end()),
                OP_LOGE(context_->GetNodeName(), "Output y dtype %d is not supported.", static_cast<int>(yDtype)),
                return ge::GRAPH_FAILED);
    outputInfo_.yDtype = yDtype;
    OP_CHECK_IF((static_cast<int64_t>(outputInfo_.yDtype) != attrParam_.dstType),
                OP_LOGE(context_->GetNodeName(), "attr dst_type(%ld) does not match output y dtype(%ld)",
                        attrParam_.dstType, static_cast<int64_t>(outputInfo_.yDtype)),
                return ge::GRAPH_FAILED);

    auto yScaleDtype = context_->GetOutputDesc(1)->GetDataType();
    OP_CHECK_IF(
        (SCALE_SUPPORT_DTYPE_SET.find(yScaleDtype) == SCALE_SUPPORT_DTYPE_SET.end()),
        OP_LOGE(context_->GetNodeName(), "Output y_scale dtype %d is not supported.", static_cast<int>(yScaleDtype)),
        return ge::GRAPH_FAILED);
    outputInfo_.yScaleDtype = yScaleDtype;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SituMxQuantRegbaseTiling::PreProcess()
{
    const gert::StorageShape* xShape = context_->GetInputShape(0);
    int64_t dimNum = inputInfo_.dimNum;
    const gert::StorageShape* yShape = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape);
    int64_t yDimNum = static_cast<int64_t>(yShape->GetStorageShape().GetDimNum());
    OP_CHECK_IF((yDimNum != dimNum),
                OP_LOGE(context_->GetNodeName(), "rank of yShape(%ld) must equal rank of xShape(%ld)", yDimNum, dimNum),
                return ge::GRAPH_FAILED);

    outputInfo_.outputDim2 = yShape->GetStorageShape().GetDim(yDimNum - 1);

    // FP4 output requires even last dim
    if (attrParam_.dstType == DTYPE_40 || attrParam_.dstType == DTYPE_41) {
        OP_CHECK_IF((outputInfo_.outputDim2 % CONST_TWO != 0),
                    OP_LOGE(context_->GetNodeName(), "When dst_type is FP4, output last dim must be even, but got %ld.",
                            outputInfo_.outputDim2),
                    return ge::GRAPH_FAILED);
    }

    // Collapse leading dims into inputDim1 (batch * rows)
    int64_t inDim1 = 1;
    for (int64_t i = 0; i < dimNum - 1; i++) {
        inDim1 *= xShape->GetStorageShape().GetDim(i);
    }
    inputInfo_.inputDim1 = inDim1;
    outputInfo_.outputDim1 = inDim1; // same as input since only last dim is halved

    OP_LOGI(context_->GetNodeName(), "3D view: dim1=%ld, dim2=%ld(2H), outputDim2=%ld(H)", inputInfo_.inputDim1,
            inputInfo_.inputDim2, outputInfo_.outputDim2);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SituMxQuantRegbaseTiling::CalculateTiling()
{
    tilingResult_.basicDim1 = 1;
    tilingResult_.basicDim2 = BASE_DIM1; // 256
    tilingResult_.dimNBlockNum = Ops::Base::CeilDiv(outputInfo_.outputDim2, tilingResult_.basicDim2);

    // UB capacity calculation
    int64_t availableUB = compileInfo_.ubSize - RESERVED_UB_SIZE - RESERVED_UB_FOR_ALIGN;

    // Input x is always 2 bytes per element (FP16 or BF16)
    int64_t bytesPerInputElement = BYTES_OF_FP16; // FP16 and BF16 are both 2 bytes

    int64_t bytesPerIteration = 0;
    // Input x: 2 halves (gate + up), each basicDim1 * basicDim2 * sizeof(FP16/BF16)
    bytesPerIteration += tilingResult_.basicDim1 * tilingResult_.basicDim2 * CONST_TWO * bytesPerInputElement;
    // Output y: FP4 uses 0.5 byte per element, FP8 uses 1 byte per element
    if (attrParam_.dstType == DTYPE_40 || attrParam_.dstType == DTYPE_41) {
        bytesPerIteration += tilingResult_.basicDim1 * tilingResult_.basicDim2 / CONST_TWO; // FP4
    } else {
        bytesPerIteration += tilingResult_.basicDim1 * tilingResult_.basicDim2 * BYTES_OF_FP8; // FP8
    }
    // Output y_scale: E8M0
    int64_t scaleCount = tilingResult_.basicDim1 * tilingResult_.basicDim2 / BLOCK_SIZE;
    bytesPerIteration += scaleCount * BYTES_OF_FP8;
    // Double buffer
    bytesPerIteration *= DOUBLE_BUFFER;
    // Situ output buffer (FP16/BF16)
    bytesPerIteration += tilingResult_.basicDim1 * tilingResult_.basicDim2 * bytesPerInputElement;
    // maxExp + halfScale (uint16_t each)
    bytesPerIteration += scaleCount * BYTES_OF_INT16 * CONST_TWO;

    int64_t ubTotalBasicBlock = availableUB / bytesPerIteration;
    OP_LOGI(context_->GetNodeName(), "ubTotalBasicBlock is %ld", ubTotalBasicBlock);

    if (ubTotalBasicBlock >= tilingResult_.dimNBlockNum) {
        tilingResult_.maxBasicNumUbDim2 = tilingResult_.dimNBlockNum;
        tilingResult_.maxBasicNumUbDim1 = Ops::Base::FloorDiv(ubTotalBasicBlock, tilingResult_.dimNBlockNum);
    } else {
        tilingResult_.maxBasicNumUbDim2 = (ubTotalBasicBlock > 0) ? ubTotalBasicBlock : 1;
        tilingResult_.maxBasicNumUbDim1 = 1;
    }

    // Core distribution: M × N grid
    int64_t dimM = outputInfo_.outputDim1;
    int64_t bmCores = std::min(dimM, compileInfo_.totalCoreNum);
    int64_t nCores = 1;
    if (bmCores < compileInfo_.totalCoreNum) {
        nCores = compileInfo_.totalCoreNum / bmCores;
        nCores = std::min(nCores, tilingResult_.dimNBlockNum);
    }
    tilingResult_.mCorePerB = bmCores;
    tilingResult_.nCoreNum = nCores;
    tilingResult_.usedCoreNum = bmCores * nCores;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SituMxQuantRegbaseTiling::FillTilingData()
{
    tilingData_ = context_->GetTilingData<SituMxQuantTilingData>();
    OP_CHECK_IF(tilingData_ == nullptr, OP_LOGE(context_->GetNodeName(), "get tilingdata ptr failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((memset_s(tilingData_, sizeof(SituMxQuantTilingData), 0, sizeof(SituMxQuantTilingData)) != EOK),
                OP_LOGE(context_->GetNodeName(), "memset tilingData failed"), return ge::GRAPH_FAILED);

    tilingData_->usedCoreNum = tilingResult_.usedCoreNum;
    tilingData_->inputDim0 = 1;
    tilingData_->inputDim1 = outputInfo_.outputDim1;
    tilingData_->inputDim2 = outputInfo_.outputDim2;
    tilingData_->dimNBlockNum = tilingResult_.dimNBlockNum;
    tilingData_->maxBasicNumUbDim2 = tilingResult_.maxBasicNumUbDim2;
    tilingData_->maxBasicNumUbDim1 = tilingResult_.maxBasicNumUbDim1;
    tilingData_->nCoreNum = tilingResult_.nCoreNum;
    tilingData_->mCorePerB = tilingResult_.mCorePerB;
    tilingData_->frontCoreNum = 0;
    tilingData_->tailCoreBasicNumDim1 = 0;
    tilingData_->activateLeft = attrParam_.activateLeft ? 1 : 0;
    tilingData_->beta = attrParam_.beta;
    tilingData_->linearBeta = attrParam_.linearBeta;
    tilingData_->hasLinearBeta = attrParam_.hasLinearBeta ? 1 : 0;
    return ge::GRAPH_SUCCESS;
}

void SituMxQuantRegbaseTiling::SetTilingKeyAndCore()
{
    hasLinearBeta_ = attrParam_.hasLinearBeta ? TPL_HAS_LINEAR_BETA : TPL_NO_LINEAR_BETA;

    int64_t tilingKey = GET_TPL_TILING_KEY(hasLinearBeta_, roundMode_);
    OP_LOGI(context_->GetNodeName(), "hasLinearBeta=%lu, roundMode=%lu, tilingKey=%ld", hasLinearBeta_, roundMode_,
            tilingKey);
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(tilingData_->usedCoreNum);
}

void SituMxQuantRegbaseTiling::PrintTilingData() const
{
    OP_LOGI(context_->GetNodeName(),
            "TilingData: usedCoreNum=%ld, inputDim1=%ld, inputDim2=%ld, dimNBlockNum=%ld, "
            "maxBasicNumUbDim2=%ld, maxBasicNumUbDim1=%ld, nCoreNum=%ld, mCorePerB=%ld, "
            "beta=%f, linearBeta=%f, hasLinearBeta=%ld",
            tilingData_->usedCoreNum, tilingData_->inputDim1, tilingData_->inputDim2, tilingData_->dimNBlockNum,
            tilingData_->maxBasicNumUbDim2, tilingData_->maxBasicNumUbDim1, tilingData_->nCoreNum,
            tilingData_->mCorePerB, tilingData_->beta, tilingData_->linearBeta, tilingData_->hasLinearBeta);
}

// ==================== Entry Functions ====================
ge::graphStatus Tiling4SituMxQuant(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do Tiling4SituMxQuant");
    SituMxQuantRegbaseTiling tiling(context);

    OP_CHECK_IF(tiling.GetNpuInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "GetNpuInfo failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.ParseAttrs() != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "ParseAttrs failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.ValidateInput() != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "ValidateInput failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.ValidateOutput() != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "ValidateOutput failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.PreProcess() != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "PreProcess failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.CalculateTiling() != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CalculateTiling failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.FillTilingData() != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "FillTilingData failed"),
                return ge::GRAPH_FAILED);
    tiling.SetTilingKeyAndCore();
    tiling.PrintTilingData();

    // Set workspace size
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t usrWorkspaceSize = 0;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    *currentWorkspace = usrWorkspaceSize + sysWorkspaceSize;

    OP_LOGI(context->GetNodeName(), "End to do Tiling4SituMxQuant");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4SituMxQuant(gert::TilingParseContext* context) { return ge::GRAPH_SUCCESS; }

// ==================== Registration ====================
IMPL_OP_OPTILING(SituMxQuant).Tiling(Tiling4SituMxQuant).TilingParse<SituMxQuantCompileInfo>(TilingPrepare4SituMxQuant);

} // namespace optiling
