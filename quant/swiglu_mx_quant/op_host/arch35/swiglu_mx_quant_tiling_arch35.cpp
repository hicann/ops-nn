/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file swiglu_mx_quant_tiling_arch35.cpp
 * \brief Tiling implementation for SwiGLU + MX quantization
 */

#include "swiglu_mx_quant_tiling_arch35.h"
#include "../../op_kernel/arch35/swiglu_mx_quant_tiling_data.h"

#include <cmath>
#include <sstream>
#include "platform/platform_info.h"
#include "util/math_util.h"

using namespace std;
using namespace ge;
using namespace AscendC;

namespace optiling {
// ==================== 常量定义 ====================
constexpr int64_t INDEX_ATTR_ACTIVATE_DIM = 0;
constexpr int64_t INDEX_ATTR_ACTIVATE_LEFT = 1;
constexpr int64_t INDEX_ATTR_SWIGLU_MODE = 2;
constexpr int64_t INDEX_ATTR_CLAMP_LIMIT = 3;
constexpr int64_t INDEX_ATTR_GLU_ALPHA = 4;
constexpr int64_t INDEX_ATTR_GLU_BIAS = 5;
constexpr int64_t INDEX_ATTR_GROUP_MODE = 6;
constexpr int64_t INDEX_ATTR_AXIS = 7;
constexpr int64_t INDEX_ATTR_DST_TYPE = 8;
constexpr int64_t INDEX_ATTR_ROUND_MODE = 9;
constexpr int64_t INDEX_ATTR_SCALE_ALG = 10;
constexpr int64_t INDEX_ATTR_MAX_DTYPE_VALUE = 11;

constexpr int64_t BYTES_OF_INT16 = 2;
constexpr int64_t BYTES_OF_FP16 = 2;
constexpr int64_t BYTES_OF_FP32 = 4;
constexpr int64_t BYTES_OF_FP8 = 1;
constexpr int64_t RESERVED_UB_SIZE = 32;
constexpr int64_t RESERVED_UB_FOR_ALIGN = 128;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t TILING_KEY_BASE = 1000;
constexpr int64_t INPUT_GROUP_INDEX = 1;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t CONST_TWO = 2;
constexpr int64_t CONST_FOUR = 4;
constexpr int64_t DTYPE_35 = 35;               // F8e5m2
constexpr int64_t DTYPE_36 = 36;               // F8e8m0
constexpr int64_t DTYPE_40 = 40;               // F4e2m1
constexpr int64_t DTYPE_41 = 41;               // F4e1m2
constexpr int64_t BASE_LAST_FACTOR_DIM1 = 256; // 尾轴量化时基本块大小是(1, 256)
constexpr int64_t BASE_NOT_LAST_FACTOR_DIM0 = 64;
constexpr int64_t BASE_NOT_LAST_FACTOR_DIM1 = 128; // 非尾轴量化时基本块大小是(64, 128)
constexpr int64_t LIMIT_GRPUP_INDEX = 256;         // group_index的输入shape大小的限制值

// 支持的数据类型集合
const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_SET = { ge::DT_FLOAT16, ge::DT_BF16 };
const std::set<ge::DataType> Y_SUPPORT_DTYPE_SET = { ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E4M3FN,
    ge::DT_FLOAT8_E5M2 };
const std::set<ge::DataType> SCALE_SUPPORT_DTYPE_SET = { ge::DT_FLOAT8_E8M0 };

// ==================== 辅助函数 ====================

template <typename T> static std::string Shape2String(const T &shape)
{
    std::ostringstream oss;
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

static RoundModeList GetRoundModeEnum(const std::string &roundMode)
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

// ==================== SwigluMxQuantRegbaseTiling 类方法实现 ====================

ge::graphStatus SwigluMxQuantRegbaseTiling::GetNpuInfo()
{
    OP_LOGD(context_->GetNodeName(), "GetNpuInfo begin.");

    // Get platform info
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    // Get core num
    compileInfo_.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo_.totalCoreNum <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get core num."),
        return ge::GRAPH_FAILED);

    // Get UB size
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo_.ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo_.ubSize <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get UB size."),
        return ge::GRAPH_FAILED);

    OP_LOGI(context_->GetNodeName(), "CompileInfo: totalCoreNum=%ld, ubSize=%ld", compileInfo_.totalCoreNum,
        compileInfo_.ubSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluMxQuantRegbaseTiling::ParseAttrs()
{
    auto *attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    // Get activate_dim (int64 type)
    auto *attrActivateDim = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_ACTIVATE_DIM);
    attrParam_.activateDim = (attrActivateDim != nullptr) ? static_cast<int64_t>(*attrActivateDim) : -1;
    OP_LOGD(context_->GetNodeName(), "attr activate_dim = %ld", attrParam_.activateDim);

    // Get activate_left (bool type)
    auto *attrActivateLeft = attrs->GetAttrPointer<bool>(INDEX_ATTR_ACTIVATE_LEFT);
    attrParam_.activateLeft = (attrActivateLeft != nullptr) ? *attrActivateLeft : false;
    OP_LOGD(context_->GetNodeName(), "attr activate_left = %d", attrParam_.activateLeft);

    // Get swiglu_mode (int64 type)
    auto *attrSwigluMode = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_SWIGLU_MODE);
    attrParam_.swigluMode = (attrSwigluMode != nullptr) ? static_cast<int64_t>(*attrSwigluMode) : 0;

    // Get clamp_limit (float type)
    auto *attrClampLimit = attrs->GetAttrPointer<float>(INDEX_ATTR_CLAMP_LIMIT);
    attrParam_.clampLimit = (attrClampLimit != nullptr) ? *attrClampLimit : 7.0f;
    OP_CHECK_IF((attrParam_.swigluMode == 1) && (attrParam_.clampLimit <= 0.0f),
        OP_LOGE(context_->GetNodeName(), "swigluMode == 1, clampLimit must > 0, but is %f", attrParam_.clampLimit),
        return ge::GRAPH_FAILED);
    // Get glu_alpha (float type)
    auto *attrGluAlpha = attrs->GetAttrPointer<float>(INDEX_ATTR_GLU_ALPHA);
    attrParam_.gluAlpha = (attrGluAlpha != nullptr) ? *attrGluAlpha : 1.702f;

    // Get glu_bias (float type)
    auto *attrGluBias = attrs->GetAttrPointer<float>(INDEX_ATTR_GLU_BIAS);
    attrParam_.gluBias = (attrGluBias != nullptr) ? *attrGluBias : 1.0f;

    // Get axis (int64 type)
    auto *attrAxis = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_AXIS);
    attrParam_.axis = (attrAxis != nullptr) ? static_cast<int64_t>(*attrAxis) : -1;

    // Get dst_type (int type)
    auto *attrDstType = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_DST_TYPE);
    attrParam_.dstType = (attrDstType != nullptr) ? static_cast<int64_t>(*attrDstType) : DTYPE_40;
    OP_LOGD(context_->GetNodeName(), "attr dst_type = %ld", attrParam_.dstType);
    OP_CHECK_IF((attrParam_.dstType != DTYPE_35) && (attrParam_.dstType != DTYPE_36) &&
        (attrParam_.dstType != DTYPE_40) && (attrParam_.dstType != DTYPE_41),
        OP_LOGE(context_->GetNodeName(), "Invalid dstType: %ld", attrParam_.dstType), return ge::GRAPH_FAILED);

    // Get round_mode (string type, stored as const char*)
    const char *attrRoundMode = attrs->GetAttrPointer<char>(INDEX_ATTR_ROUND_MODE);
    std::string roundModeStr = (attrRoundMode != nullptr) ? attrRoundMode : "rint";
    auto roundMode = GetRoundModeEnum(roundModeStr);
    OP_CHECK_IF((roundMode == RoundModeList::MODE_UNDEFINED),
        OP_LOGE(context_->GetNodeName(), "Invalid round_mode: %s", roundModeStr.c_str()), return ge::GRAPH_FAILED);
    attrParam_.roundMode = static_cast<int64_t>(roundMode);
    OP_LOGD(context_->GetNodeName(), "attr round_mode = %s -> %ld", roundModeStr.c_str(), attrParam_.roundMode);

    // Get scale_alg (int64 type)
    auto *attrScaleAlg = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_SCALE_ALG);
    attrParam_.scaleAlg = (attrScaleAlg != nullptr) ? static_cast<int64_t>(*attrScaleAlg) : 0;
    OP_LOGD(context_->GetNodeName(), "attr scale_alg = %ld", attrParam_.scaleAlg);
    OP_CHECK_IF((attrParam_.scaleAlg != 0) && (attrParam_.scaleAlg != 1) && (attrParam_.scaleAlg != 2),
        OP_LOGE(context_->GetNodeName(), "Invalid scaleAlg: %ld", attrParam_.scaleAlg), return ge::GRAPH_FAILED);
    // Get max_dtype_value (float type)
    auto *attrMaxDtypeValue = attrs->GetAttrPointer<float>(INDEX_ATTR_MAX_DTYPE_VALUE);
    attrParam_.maxDtypeValue = (attrMaxDtypeValue != nullptr) ? *attrMaxDtypeValue : 0.0f;

    // Get group_mode (int64 type)
    auto *attrGroupMode = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_GROUP_MODE);
    attrParam_.groupMode = (attrGroupMode != nullptr) ? static_cast<int64_t>(*attrGroupMode) : 0;

    // blockSize is fixed to 32
    attrParam_.blockSize = BLOCK_SIZE;

    // 校验 activateDim 取值范围：[-dimNum, dimNum-1]
    OP_CHECK_IF((attrParam_.activateDim < -inputInfo_.dimNum || attrParam_.activateDim >= inputInfo_.dimNum),
        OP_LOGE(context_->GetNodeName(), "activate_dim=%ld is out of range [-%ld, %ld].", attrParam_.activateDim,
        inputInfo_.dimNum, inputInfo_.dimNum - 1),
        return ge::GRAPH_FAILED);

    // 校验 axis 取值范围：[-dimNum, dimNum-1]
    OP_CHECK_IF((attrParam_.axis < -inputInfo_.dimNum || attrParam_.axis >= inputInfo_.dimNum),
        OP_LOGE(context_->GetNodeName(), "axis=%ld is out of range [-%ld, %ld].", attrParam_.axis, inputInfo_.dimNum,
        inputInfo_.dimNum - 1),
        return ge::GRAPH_FAILED);

    // 将正索引统一转换为负索引，便于后续判断
    if (attrParam_.activateDim >= 0) {
        attrParam_.activateDim -= inputInfo_.dimNum;
    }
    if (attrParam_.axis >= 0) {
        attrParam_.axis -= inputInfo_.dimNum;
    }

    // Check constraints: only support activate_dim=-1 and axis=-1 currently
    OP_CHECK_IF((attrParam_.activateDim != -1),
        OP_LOGE(context_->GetNodeName(), "Only activate_dim=-1 is supported currently, but got %ld.",
        attrParam_.activateDim),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((attrParam_.axis != -1),
        OP_LOGE(context_->GetNodeName(), "Only axis=-1 is supported currently, but got %ld.", attrParam_.axis),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluMxQuantRegbaseTiling::ValidateInput()
{
    // Check x dtype
    auto xDtype = context_->GetInputDesc(0)->GetDataType();
    OP_CHECK_IF((INPUT_SUPPORT_DTYPE_SET.find(xDtype) == INPUT_SUPPORT_DTYPE_SET.end()),
        OP_LOGE(context_->GetNodeName(), "Input x dtype %d is not supported.", static_cast<int>(xDtype)),
        return ge::GRAPH_FAILED);
    inputInfo_.xDtype = xDtype;

    // Get input shape
    auto xShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    int64_t dimNum = static_cast<int64_t>(xShape->GetStorageShape().GetDimNum());
    OP_LOGI(context_->GetNodeName(), "Input x shape = %s", Shape2String(xShape->GetStorageShape()).c_str());

    OP_CHECK_IF((dimNum < CONST_TWO),
        OP_LOGE(context_->GetNodeName(), "Input dimension must be at least 2, but got %ld.", dimNum),
        return ge::GRAPH_FAILED);

    // 保存维度数量，供属性校验使用
    inputInfo_.dimNum = dimNum;

    // Extract dimensions
    inputInfo_.inputDim2 = xShape->GetStorageShape().GetDim(dimNum - 1);

    // Detect optional input group_index
    inputInfo_.groupIndexNum = 0; // 初始值为 0
    auto groupIndexDesc = context_->GetOptionalInputDesc(INPUT_GROUP_INDEX);
    if (groupIndexDesc != nullptr) {
        auto groupIndexDtype = groupIndexDesc->GetDataType();
        if (groupIndexDtype == ge::DT_INT32) {
            inputInfo_.groupIndexType = 1;
            OP_LOGI(context_->GetNodeName(), "group_index exists with type int32");
        } else if (groupIndexDtype == ge::DT_INT64) {
            inputInfo_.groupIndexType = CONST_TWO;
            OP_LOGI(context_->GetNodeName(), "group_index exists with type int64");
        } else {
            OP_LOGE(context_->GetNodeName(), "group_index has unsupported dtype %d", static_cast<int>(groupIndexDtype));
            return ge::GRAPH_FAILED;
        }

        // 获取 group_index 的 shape 并校验维度必须为 1
        auto groupIndexShape = context_->GetOptionalInputShape(INPUT_GROUP_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, groupIndexShape);
        size_t groupIndexDimNum = groupIndexShape->GetStorageShape().GetDimNum();
        OP_CHECK_IF(groupIndexDimNum != 1,
            OP_LOGE(context_->GetNodeName(), "group_index dimension must be 1, but got %zu", groupIndexDimNum),
            return ge::GRAPH_FAILED);
        inputInfo_.groupIndexNum = groupIndexShape->GetStorageShape().GetDim(0);
        OP_LOGI(context_->GetNodeName(), "group_index exists with shape[0]=%ld", inputInfo_.groupIndexNum);

        // 校验 group_index 的 shape[0] 必须不大于 256
        OP_CHECK_IF(inputInfo_.groupIndexNum > LIMIT_GRPUP_INDEX,
            OP_LOGE(context_->GetNodeName(), "group_index shape[0] must be <= 256, but got %ld.",
            inputInfo_.groupIndexNum),
            return ge::GRAPH_FAILED);
    } else {
        inputInfo_.groupIndexType = 0;
        OP_LOGI(context_->GetNodeName(), "group_index does not exist");
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluMxQuantRegbaseTiling::ValidateOutput()
{
    // Check y dtype
    auto yDtype = context_->GetOutputDesc(0)->GetDataType();
    OP_CHECK_IF((Y_SUPPORT_DTYPE_SET.find(yDtype) == Y_SUPPORT_DTYPE_SET.end()),
        OP_LOGE(context_->GetNodeName(), "Output y dtype %d is not supported.", static_cast<int>(yDtype)),
        return ge::GRAPH_FAILED);
    outputInfo_.yDtype = yDtype;
    OP_CHECK_IF((static_cast<int64_t>(outputInfo_.yDtype) != attrParam_.dstType),
        OP_LOGE(context_->GetNodeName(),
        "attr dst_type is not same as out_y dtype, attr dst_type is %ld, out_y dtype is %ld", attrParam_.dstType,
        static_cast<int64_t>(outputInfo_.yDtype)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((attrParam_.dstType == DTYPE_40 || attrParam_.dstType == DTYPE_41) && (attrParam_.scaleAlg != 0),
        OP_LOGE(context_->GetNodeName(), "output is fp4, scaleAlg must be 0, but is %ld", attrParam_.scaleAlg),
        return ge::GRAPH_FAILED);
    // Check mxscale dtype
    auto mxscaleDtype = context_->GetOutputDesc(1)->GetDataType();
    OP_CHECK_IF((SCALE_SUPPORT_DTYPE_SET.find(mxscaleDtype) == SCALE_SUPPORT_DTYPE_SET.end()),
        OP_LOGE(context_->GetNodeName(), "Output mxscale dtype %d is not supported.", static_cast<int>(mxscaleDtype)),
        return ge::GRAPH_FAILED);
    outputInfo_.mxscaleDtype = mxscaleDtype;

    // 获取输出 y 的 shape，设置 outputDim2 为 y 的 activateDim 维度的值
    auto yShape = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape);
    int64_t yDimNum = static_cast<int64_t>(yShape->GetStorageShape().GetDimNum());
    outputInfo_.outputDim2 = yShape->GetStorageShape().GetDim(yDimNum - 1);
    // 校验：y 的 activateDim 轴的 shape = x 的 activateDim 轴的 shape / 2
    if (attrParam_.activateDim == -1) {
        int64_t expectedOutputDim2 = inputInfo_.inputDim2 / CONST_TWO;
        OP_CHECK_IF((outputInfo_.outputDim2 != expectedOutputDim2),
            OP_LOGE(context_->GetNodeName(),
            "Output y's activateDim dimension size %ld should equal to x's activateDim dimension size / 2, expected "
            "%ld.",
            outputInfo_.outputDim2, expectedOutputDim2),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF((attrParam_.dstType == DTYPE_35 || attrParam_.dstType == DTYPE_36) && (attrParam_.roundMode != 1),
        OP_LOGE(context_->GetNodeName(), "outDtype is fp8, roundMode must be rint, but is %ld", attrParam_.roundMode),
        return ge::GRAPH_FAILED);
    OP_LOGI(context_->GetNodeName(), "Output y outputDim2=%ld", outputInfo_.outputDim2);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluMxQuantRegbaseTiling::PreProcess()
{
    // 获取 x 的完整 shape
    auto xShape = context_->GetInputShape(0);
    int64_t dimNum = inputInfo_.dimNum;

    // 根据 activateDim 合轴计算 inputDim1 和 inputDim2
    if (attrParam_.activateDim == -1) {
        // 对 -1 轴之前的轴做合轴，shape 视为三维 [1, 前序轴乘积, 最后一维]
        // 例如: [a, b, c, d, e, f, g] -> [1, a*b*c*d*e*f, g]
        int64_t mergedDim1 = 1;
        for (int64_t i = 0; i < dimNum - 1; i++) {
            mergedDim1 *= xShape->GetStorageShape().GetDim(i);
        }
        inputInfo_.inputDim1 = mergedDim1;
        // inputDim2 保持为最后一维
    } else { // activateDim == -2
        // 对 -2 轴之前的轴做合轴，shape 视为三维 [前序轴乘积, 激活轴, 最后一维]
        // 例如: [a, b, c, d, e, f, g] -> [a*b*c*d*e, f, g]
        // 预留分支，暂不处理
        return ge::GRAPH_FAILED;
    }

    OP_LOGI(context_->GetNodeName(), "After merge: inputDim1=%ld, inputDim2=%ld, outputDim2=%ld", inputInfo_.inputDim1,
        inputInfo_.inputDim2, outputInfo_.outputDim2);

    // 获取 x 在 activateDim 维度的 shape 值
    int64_t activateDimSize = 0;
    if (attrParam_.activateDim == -1) {
        activateDimSize = inputInfo_.inputDim2;
    } else { // activateDim == -2
        activateDimSize = inputInfo_.inputDim1;
    }

    // 根据 dstType 检查 activateDim 维度的 shape 是否满足对齐要求
    if (attrParam_.dstType == DTYPE_40 || attrParam_.dstType == DTYPE_41) {
        // FP4 类型: activateDim 维度必须能被 4 整除
        OP_CHECK_IF((activateDimSize % CONST_FOUR != 0),
            OP_LOGE(context_->GetNodeName(),
            "When dst_type is FP4, activate_dim dimension size %ld must be divisible by 4.", activateDimSize),
            return ge::GRAPH_FAILED);
    } else {
        // FP8 类型: activateDim 维度必须能被 2 整除
        OP_CHECK_IF((activateDimSize % CONST_TWO != 0),
            OP_LOGE(context_->GetNodeName(),
            "When dst_type is FP8, activate_dim dimension size %ld must be divisible by 2.", activateDimSize),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluMxQuantRegbaseTiling::CalculateTiling()
{
    // Set basic block parameters according to axis
    if (attrParam_.axis == -1) {
        tilingResult_.basicDim2 = BASE_LAST_FACTOR_DIM1;
        tilingResult_.basicDim1 = 1;
    } else { // axis == -2
        tilingResult_.basicDim2 = BASE_NOT_LAST_FACTOR_DIM1;
        tilingResult_.basicDim1 = BASE_NOT_LAST_FACTOR_DIM0;
    }

    // Calculate available UB size
    int64_t availableUB = compileInfo_.ubSize - RESERVED_UB_SIZE - RESERVED_UB_FOR_ALIGN;

    // UB capacity calculation per iteration
    int64_t bytesPerIteration = 0;
    bytesPerIteration += tilingResult_.basicDim1 * tilingResult_.basicDim2 * CONST_TWO * BYTES_OF_FP16; // Input x
    // Output y: FP4 uses 0.5 byte per element, FP8 uses 1 byte per element
    if (attrParam_.dstType == DTYPE_40 || attrParam_.dstType == DTYPE_41) {                 // FP4_E2M1 or FP4_E1M2
        bytesPerIteration += tilingResult_.basicDim1 * tilingResult_.basicDim2 / CONST_TWO; // Output y (FP4)
    } else { // FP8_E4M3FN(36) or FP8_E5M2(35)
        bytesPerIteration += tilingResult_.basicDim1 * tilingResult_.basicDim2 * BYTES_OF_FP8; // Output y (FP8)
    }
    bytesPerIteration += tilingResult_.basicDim1 * tilingResult_.basicDim2 / BLOCK_SIZE * BYTES_OF_FP8; // Output Scale
    bytesPerIteration *= DOUBLE_BUFFER;
    bytesPerIteration +=
        tilingResult_.basicDim1 * tilingResult_.basicDim2 / BLOCK_SIZE * BYTES_OF_FP16; // reciprocal_scale[8]
    bytesPerIteration += tilingResult_.basicDim1 * tilingResult_.basicDim2 / BLOCK_SIZE * BYTES_OF_INT16; // max_exp[8]
    bytesPerIteration += tilingResult_.basicDim1 * tilingResult_.basicDim2 * BYTES_OF_FP16;               // SwiGLU

    // Calculate how many basic blocks can fit in UB
    int64_t ubTotalBasicBlock = availableUB / bytesPerIteration;
    OP_LOGI(context_->GetNodeName(), "ubTotalBasicBlock is %ld", ubTotalBasicBlock);

    // Calculate basic blocks per row
    int64_t basicPerRow = Ops::Base::CeilDiv(outputInfo_.outputDim2, tilingResult_.basicDim2);

    if (ubTotalBasicBlock >= basicPerRow) {
        // Full-load scenario: entire row fits in UB
        tilingResult_.maxBasicNumUbDim2 = basicPerRow;
        tilingResult_.maxBasicNumUbDim1 = Ops::Base::FloorDiv(ubTotalBasicBlock, basicPerRow);
        tilingResult_.ubLoopPerRow = 1;
        tilingResult_.ubTailPerRow = outputInfo_.outputDim2;
        tilingResult_.isFullLoad = 1;
    } else {
        // Non-full-load scenario
        tilingResult_.maxBasicNumUbDim2 = ubTotalBasicBlock;
        tilingResult_.maxBasicNumUbDim1 = 1;
        tilingResult_.ubLoopPerRow = Ops::Base::CeilDiv(basicPerRow, tilingResult_.maxBasicNumUbDim2);
        tilingResult_.ubTailPerRow = outputInfo_.outputDim2 -
            (tilingResult_.ubLoopPerRow - 1) * tilingResult_.maxBasicNumUbDim2 * tilingResult_.basicDim2;
        tilingResult_.isFullLoad = 0;
    }

    // Calculate inter-core split strategy
    int64_t basicPerCol = Ops::Base::CeilDiv(inputInfo_.inputDim1, tilingResult_.basicDim1);
    tilingResult_.usedCoreNum = std::min(basicPerCol, compileInfo_.totalCoreNum);

    // Tail core parameters
    int64_t tailCoreBasicNumDim1 = Ops::Base::FloorDiv(basicPerCol, tilingResult_.usedCoreNum);
    int64_t tailCoreLoopTimes = Ops::Base::CeilDiv(tailCoreBasicNumDim1, tilingResult_.maxBasicNumUbDim1);
    int64_t tailCoreLastLoopBasicNum = tailCoreBasicNumDim1 - (tailCoreLoopTimes - 1) * tilingResult_.maxBasicNumUbDim1;

    // Front core parameters
    tilingResult_.frontCoreNum = basicPerCol % tilingResult_.usedCoreNum;
    tilingResult_.frontCoreBasicNumDim1 = tailCoreBasicNumDim1 + 1;
    tilingResult_.frontCoreLoopTimes =
        Ops::Base::CeilDiv(tilingResult_.frontCoreBasicNumDim1, tilingResult_.maxBasicNumUbDim1);
    tilingResult_.frontCoreLastLoopBasicNum =
        tilingResult_.frontCoreBasicNumDim1 - (tilingResult_.frontCoreLoopTimes - 1) * tilingResult_.maxBasicNumUbDim1;

    tilingResult_.tailCoreBasicNumDim1 = tailCoreBasicNumDim1;
    tilingResult_.tailCoreLoopTimes = tailCoreLoopTimes;
    tilingResult_.tailCoreLastLoopBasicNum = tailCoreLastLoopBasicNum;

    return ge::GRAPH_SUCCESS;
}

int64_t SwigluMxQuantRegbaseTiling::CalculateTilingKey()
{
    // TilingKey = 1000 + groupIndexType * 100 + activateDimIndex * 10 + axisIndex
    int64_t activateDimIndex = (attrParam_.activateDim == -1) ? 0 : 1;
    int64_t axisIndex = (attrParam_.axis == -1) ? 0 : 1;
    return TILING_KEY_BASE + inputInfo_.groupIndexType * 100 + activateDimIndex * 10 + axisIndex;
}

ge::graphStatus SwigluMxQuantRegbaseTiling::FillTilingData()
{
    // Get tiling data pointer
    tilingData_ = context_->GetTilingData<SwigluMxQuantTilingData>();
    OP_CHECK_IF(tilingData_ == nullptr, OP_LOGE(context_->GetNodeName(), "get tilingdata ptr failed"),
        return ge::GRAPH_FAILED);

    // Clear tiling data
    OP_CHECK_IF((memset_s(tilingData_, sizeof(SwigluMxQuantTilingData), 0, sizeof(SwigluMxQuantTilingData)) != EOK),
        OP_LOGE(context_->GetNodeName(), "memset tilingData failed"), return ge::GRAPH_FAILED);

    // Basic parameters
    tilingData_->usedCoreNum = tilingResult_.usedCoreNum;

    // Data shape
    tilingData_->inputDim2 = inputInfo_.inputDim2;
    tilingData_->inputDim1 = inputInfo_.inputDim1;
    tilingData_->outputDim2 = outputInfo_.outputDim2;

    // Memory allocation
    tilingData_->basicDim2 = tilingResult_.basicDim2;
    tilingData_->basicDim1 = tilingResult_.basicDim1;
    tilingData_->maxBasicNumUbDim2 = tilingResult_.maxBasicNumUbDim2;
    tilingData_->maxBasicNumUbDim1 = tilingResult_.maxBasicNumUbDim1;
    tilingData_->ubLoopPerRow = tilingResult_.ubLoopPerRow;
    tilingData_->ubTailPerRow = tilingResult_.ubTailPerRow;

    // Inter-core split
    tilingData_->frontCoreNum = tilingResult_.frontCoreNum;
    tilingData_->frontCoreBasicNumDim1 = tilingResult_.frontCoreBasicNumDim1;
    tilingData_->frontCoreLoopTimes = tilingResult_.frontCoreLoopTimes;
    tilingData_->frontCoreLastLoopBasicNum = tilingResult_.frontCoreLastLoopBasicNum;
    tilingData_->tailCoreBasicNumDim1 = tilingResult_.tailCoreBasicNumDim1;
    tilingData_->tailCoreLoopTimes = tilingResult_.tailCoreLoopTimes;
    tilingData_->tailCoreLastLoopBasicNum = tilingResult_.tailCoreLastLoopBasicNum;

    // SwiGLU parameters
    tilingData_->activateLeft = attrParam_.activateLeft ? 1 : 0;
    tilingData_->swigluMode = attrParam_.swigluMode;
    tilingData_->clampLimit = attrParam_.clampLimit;
    tilingData_->gluAlpha = attrParam_.gluAlpha;
    tilingData_->gluBias = attrParam_.gluBias;

    // Quantization parameters
    tilingData_->roundMode = attrParam_.roundMode;
    tilingData_->scaleAlg = attrParam_.scaleAlg;
    tilingData_->maxDtypeValue = attrParam_.maxDtypeValue;
    tilingData_->groupMode = attrParam_.groupMode;
    tilingData_->groupIndexNum = inputInfo_.groupIndexNum;
    return ge::GRAPH_SUCCESS;
}

void SwigluMxQuantRegbaseTiling::PrintTilingData() const
{
    OP_LOGI(context_->GetNodeName(),
        "TilingData: usedCoreNum=%ld, inputDim2=%ld, inputDim1=%ld, outputDim2=%ld, "
        "basicDim2=%ld, basicDim1=%ld, maxBasicNumUbDim2=%ld, maxBasicNumUbDim1=%ld, "
        "ubLoopPerRow=%ld, ubTailPerRow=%ld, isFullLoad=%ld, "
        "frontCoreNum=%ld, frontCoreBasicNumDim1=%ld, frontCoreLoopTimes=%ld, frontCoreLastLoopBasicNum=%ld, "
        "tailCoreBasicNumDim1=%ld, tailCoreLoopTimes=%ld, tailCoreLastLoopBasicNum=%ld, "
        "scaleAlg=%ld, roundMode=%ld, groupIndexNum=%ld, swigluMode=%ld, "
        "activateLeft=%ld, clampLimit=%f, gluBias=%f, gluAlpha=%f",
        tilingData_->usedCoreNum, tilingData_->inputDim2, tilingData_->inputDim1, tilingData_->outputDim2,
        tilingData_->basicDim2, tilingData_->basicDim1, tilingData_->maxBasicNumUbDim2, tilingData_->maxBasicNumUbDim1,
        tilingData_->ubLoopPerRow, tilingData_->ubTailPerRow, tilingResult_.isFullLoad, tilingData_->frontCoreNum,
        tilingData_->frontCoreBasicNumDim1, tilingData_->frontCoreLoopTimes, tilingData_->frontCoreLastLoopBasicNum,
        tilingData_->tailCoreBasicNumDim1, tilingData_->tailCoreLoopTimes, tilingData_->tailCoreLastLoopBasicNum,
        tilingData_->scaleAlg, tilingData_->roundMode, tilingData_->groupIndexNum, tilingData_->swigluMode,
        tilingData_->activateLeft, tilingData_->clampLimit, tilingData_->gluBias, tilingData_->gluAlpha);
}

ge::graphStatus SwigluMxQuantRegbaseTiling::SetParams()
{
    // Set TilingKey and BlockDim
    int64_t tilingKey = CalculateTilingKey();
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(tilingData_->usedCoreNum);

    // Set workspace
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t usrWorkspaceSize = 0;

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = usrWorkspaceSize + sysWorkspaceSize;

    OP_LOGI(context_->GetNodeName(), "SetParams done. TilingKey=%ld, BlockDim=%ld", tilingKey,
        tilingData_->usedCoreNum);

    return ge::GRAPH_SUCCESS;
}

// ==================== TilingPrepare ====================
ge::graphStatus TilingPrepare4SwigluMxQuant(gert::TilingParseContext *context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4SwigluMxQuant begin.");
    return ge::GRAPH_SUCCESS;
}

// ==================== 主 Tiling 函数 ====================
ge::graphStatus Tiling4SwigluMxQuant(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4SwigluMxQuant begin.");

    SwigluMxQuantRegbaseTiling tilingImpl(context);

    // Phase 0: Get NPU info
    OP_CHECK_IF(tilingImpl.GetNpuInfo() != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Failed to get NPU info."), return ge::GRAPH_FAILED);

    // Phase 1: Validate input (先获取输入，以便 ParseAttrs 可以使用 dimNum 进行校验)
    OP_CHECK_IF(tilingImpl.ValidateInput() != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Input validation failed."), return ge::GRAPH_FAILED);

    // Phase 2: Parse attributes (在 ValidateInput 之后，可以使用 inputInfo_.dimNum 进行属性校验)
    OP_CHECK_IF(tilingImpl.ParseAttrs() != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Failed to parse attributes."), return ge::GRAPH_FAILED);

    // Phase 3: Validate output
    OP_CHECK_IF(tilingImpl.ValidateOutput() != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Output validation failed."), return ge::GRAPH_FAILED);

    // Phase 3.5: Post process (检查 shape 对齐要求)
    OP_CHECK_IF(tilingImpl.PreProcess() != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "Post process failed."),
        return ge::GRAPH_FAILED);

    // Phase 4: Calculate tiling
    OP_CHECK_IF(tilingImpl.CalculateTiling() != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Tiling calculation failed."), return ge::GRAPH_FAILED);

    // Phase 5: Fill tiling data
    OP_CHECK_IF(tilingImpl.FillTilingData() != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Failed to fill tiling data."), return ge::GRAPH_FAILED);

    // Phase 6: Set params (TilingKey, BlockDim, Workspace)
    OP_CHECK_IF(tilingImpl.SetParams() != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "Failed to set params."),
        return ge::GRAPH_FAILED);

    tilingImpl.PrintTilingData();

    OP_LOGI(context->GetNodeName(), "Tiling4SwigluMxQuant done.");

    return ge::GRAPH_SUCCESS;
}

// ==================== 注册 Tiling 接口 ====================
IMPL_OP_OPTILING(SwigluMxQuant)
    .Tiling(Tiling4SwigluMxQuant)
    .TilingParse<SwigluMxQuantCompileInfo>(TilingPrepare4SwigluMxQuant);
} // namespace optiling
