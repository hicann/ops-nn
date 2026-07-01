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
#include "quant/swiglu_mx_quant/op_kernel/arch35/swiglu_mx_quant_tiling_key.h"

#include <cmath>
#include <sstream>
#include "platform/platform_info.h"
#include "op_host/tiling_util.h"
#include "util/math_util.h"

using namespace std;
using namespace ge;
using namespace AscendC;
using namespace SwigluMxQuantOp;

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
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t CONST_TWO = 2;
constexpr int64_t CONST_THREE = 3;
constexpr int64_t CONST_FOUR = 4;
constexpr int64_t DTYPE_35 = 35;   // F8e5m2
constexpr int64_t DTYPE_36 = 36;   // F8e8m0
constexpr int64_t DTYPE_40 = 40;   // F4e2m1
constexpr int64_t DTYPE_41 = 41;   // F4e1m2
constexpr int64_t BASE_DIM1 = 256; // 尾轴量化时基本块大小是(1, 256)
constexpr int64_t BASE_DIM0 = 64;
constexpr int64_t LIMIT_GRPUP_INDEX = 256; // group_index的输入shape大小的限制值

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
    if (roundMode == RoundModeList::MODE_FLOOR) {
        roundMode_ = TPL_FLOOR;
    } else if (roundMode == RoundModeList::MODE_ROUND) {
        roundMode_ = TPL_ROUND;
    } else {
        roundMode_ = TPL_RINT;
    }
    attrParam_.roundMode = static_cast<int64_t>(roundMode);
    OP_LOGD(context_->GetNodeName(), "attr round_mode = %s -> %ld", roundModeStr.c_str(), attrParam_.roundMode);

    // Get scale_alg (int64 type)
    auto *attrScaleAlg = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_SCALE_ALG);
    attrParam_.scaleAlg = (attrScaleAlg != nullptr) ? static_cast<int64_t>(*attrScaleAlg) : 0;
    OP_LOGD(context_->GetNodeName(), "attr scale_alg = %ld", attrParam_.scaleAlg);
    OP_CHECK_IF((attrParam_.scaleAlg != 0) && (attrParam_.scaleAlg != 1) && (attrParam_.scaleAlg != 2),
        OP_LOGE(context_->GetNodeName(), "Invalid scaleAlg: %ld", attrParam_.scaleAlg), return ge::GRAPH_FAILED);

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

    // Check constraints: only support activate_dim=-1, axis=-1 or axis=-2
    OP_CHECK_IF((attrParam_.activateDim != -1) && (attrParam_.activateDim != -2),
        OP_LOGE(context_->GetNodeName(), "Only activate_dim=-1 or -2 is supported currently, but got %ld.",
        attrParam_.activateDim),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((attrParam_.axis != -1 && attrParam_.axis != -2),
        OP_LOGE(context_->GetNodeName(), "axis must be -1 or -2, but got %ld.", attrParam_.axis),
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
    int64_t xSize = xShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF((dimNum < CONST_TWO || xSize == 0),
        OP_LOGE(context_->GetNodeName(), "rank of x must >= 2, but is %ld, and not support empty tensor", dimNum),
        return ge::GRAPH_FAILED);

    // 保存维度数量，供属性校验使用
    inputInfo_.dimNum = dimNum;

    // Extract dimensions
    inputInfo_.inputDim2 = xShape->GetStorageShape().GetDim(dimNum - 1);
    inputInfo_.inputDim1 = xShape->GetStorageShape().GetDim(dimNum - 2);
    // Detect optional input group_index
    inputInfo_.groupIndexNum = 0; // 初始值为 0
    auto groupIndexDesc = context_->GetOptionalInputDesc(1);
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
        auto groupIndexShape = context_->GetOptionalInputShape(1);
        OP_CHECK_NULL_WITH_CONTEXT(context_, groupIndexShape);
        size_t groupIndexDimNum = groupIndexShape->GetStorageShape().GetDimNum();
        OP_CHECK_IF(groupIndexDimNum != 1,
            OP_LOGE(context_->GetNodeName(), "group_index dimension must be 1, but got %zu", groupIndexDimNum),
            return ge::GRAPH_FAILED);
        inputInfo_.groupIndexNum = groupIndexShape->GetStorageShape().GetDim(0);
        OP_LOGI(context_->GetNodeName(), "group_index exists with shape[0]=%ld", inputInfo_.groupIndexNum);

        // 校验 group_index shape必须满足 0 <shape[0] <= 256
        OP_CHECK_IF(inputInfo_.groupIndexNum > LIMIT_GRPUP_INDEX || inputInfo_.groupIndexNum <= 0,
            OP_LOGE(context_->GetNodeName(), "group_index shape[0] must be <= 256 and > 0, but got %ld.",
            inputInfo_.groupIndexNum),
            return ge::GRAPH_FAILED);
    } else {
        inputInfo_.groupIndexType = 0;
        OP_LOGI(context_->GetNodeName(), "group_index does not exist");
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluMxQuantRegbaseTiling::CheckScaleShape(const gert::StorageShape* scaleShape,
    const gert::StorageShape* yShape, int64_t scaleDimNum)
{
    if (attrParam_.axis == -1) {
        int64_t expectedScaleNum = Ops::Base::CeilDiv(outputInfo_.outputDim2, BASE_DIM0);
        int64_t mxScaleNum = scaleShape->GetStorageShape().GetDim(scaleDimNum - CONST_TWO);
        OP_CHECK_IF(mxScaleNum != expectedScaleNum,
            OP_LOGE(context_->GetNodeName(),
            "Output mxScale's axis dimension size is error,mxScaleNum is %ld, expectedScaleNum is %ld", mxScaleNum,
            expectedScaleNum),
            return ge::GRAPH_FAILED);
        for (int64_t i = 0; i < scaleDimNum - CONST_TWO; i++) {
            OP_CHECK_IF(scaleShape->GetStorageShape().GetDim(i) != yShape->GetStorageShape().GetDim(i),
                OP_LOGE(context_->GetNodeName(),
                "axis = -1, scaleShape[i] must be equal yShape[i], but i is %ld, scaleShape[i] is %ld, yShape[i] is "
                "%ld",
                i, scaleShape->GetStorageShape().GetDim(i), yShape->GetStorageShape().GetDim(i)),
                return ge::GRAPH_FAILED);
        }
    } else { // axis == -2
        int64_t expectedScaleM = inputInfo_.groupIndexType == 0 ?
            Ops::Base::CeilDiv(outputInfo_.outputDim1, BASE_DIM0) :
            Ops::Base::FloorDiv(outputInfo_.outputDim1, BASE_DIM0) + inputInfo_.groupIndexNum;
        int64_t mxScaleDim = scaleShape->GetStorageShape().GetDim(scaleDimNum - CONST_THREE);
        OP_CHECK_IF(mxScaleDim != expectedScaleM,
            OP_LOGE(context_->GetNodeName(), "axis=-2: mxScale dim error, got=%ld, expected=%ld", mxScaleDim,
            expectedScaleM),
            return ge::GRAPH_FAILED);
        int64_t scaleN = scaleShape->GetStorageShape().GetDim(scaleDimNum - CONST_TWO);
        OP_CHECK_IF(scaleN != outputInfo_.outputDim2,
            OP_LOGE(context_->GetNodeName(), "axis=-2: mxScale N mismatch, got=%ld, expected=%ld", scaleN,
            outputInfo_.outputDim2),
            return ge::GRAPH_FAILED);
        for (int64_t i = 0; i < scaleDimNum - CONST_THREE; i++) {
            OP_CHECK_IF(scaleShape->GetStorageShape().GetDim(i) != yShape->GetStorageShape().GetDim(i),
                OP_LOGE(context_->GetNodeName(),
                "axis = -2, scaleShape[i] must be equal yShape[i], but i is %ld, scaleShape[i] is %ld, yShape[i] is "
                "%ld",
                i, scaleShape->GetStorageShape().GetDim(i), yShape->GetStorageShape().GetDim(i)),
                return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluMxQuantRegbaseTiling::ValidateScaleOutput()
{
    OP_CHECK_IF((inputInfo_.groupIndexType != 0) && (attrParam_.activateDim != -1 || attrParam_.axis != -1) &&
        (inputInfo_.dimNum != CONST_TWO),
        OP_LOGE(context_->GetNodeName(),
        "When axis = -2 or activate_dim = -2 and group is exist, the rank of x must be 2, but now is %ld ",
        inputInfo_.dimNum),
        return ge::GRAPH_FAILED);
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
    const gert::StorageShape* yShape = context_->GetOutputShape(0);
    const gert::StorageShape* scaleShape = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleShape);
    auto scaleShapeNew = Ops::NN::OpTiling::EnsureNotScalar(scaleShape->GetStorageShape());
    int64_t scaleShapeSize = scaleShapeNew.GetShapeSize();
    int64_t scaleDimNum = static_cast<int64_t>(scaleShape->GetStorageShape().GetDimNum());
    OP_CHECK_IF(scaleShapeSize <= 0 || scaleDimNum < CONST_THREE,
        OP_LOGE(context_->GetNodeName(),
        "out not support empty tensor, rank of scale must >=3, but rank of scale is %ld", scaleDimNum),
        return ge::GRAPH_FAILED);
    int64_t scaleLast = scaleShape->GetStorageShape().GetDim(scaleDimNum - 1);
    OP_CHECK_IF(scaleLast != CONST_TWO,
        OP_LOGE(context_->GetNodeName(), "last dim of scaleShape must be 2, but is %ld", scaleLast),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((attrParam_.dstType == DTYPE_35 || attrParam_.dstType == DTYPE_36) && (attrParam_.roundMode != 1),
        OP_LOGE(context_->GetNodeName(), "outDtype is fp8, roundMode must be rint, but is %ld", attrParam_.roundMode),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckScaleShape(scaleShape, yShape, scaleDimNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "CheckScaleShape check failed"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluMxQuantRegbaseTiling::ValidateYOutput(const gert::StorageShape* xShape, const gert::StorageShape* yShape, int64_t yDimNum)
{
    // 校验：y 的 activateDim 轴的 shape = x 的 activateDim 轴的 shape / 2
    if (attrParam_.activateDim == -1) {
        int64_t expectedOutputDim2 = inputInfo_.inputDim2 / CONST_TWO;
        OP_CHECK_IF((outputInfo_.outputDim2 != expectedOutputDim2),
            OP_LOGE(context_->GetNodeName(),
            "Output y's activateDim dimension size %ld should equal to x's activateDim dimension size / 2, expected "
            "%ld.",
            outputInfo_.outputDim2, expectedOutputDim2),
            return ge::GRAPH_FAILED);
        for (int64_t i = 0; i < yDimNum - 1; i++) {
            OP_CHECK_IF((yShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(i)),
                OP_LOGE(context_->GetNodeName(),
                "activateDim = -1, yShape[i] must be equal xShape[i], but i is %ld, yShape[i] is %ld, xShape[i] is %ld",
                i, yShape->GetStorageShape().GetDim(i), xShape->GetStorageShape().GetDim(i)),
                return ge::GRAPH_FAILED);
        }
    }
    if (attrParam_.activateDim == -2) {
        int64_t expectedOutputDim1 = inputInfo_.inputDim1 / CONST_TWO;
        OP_CHECK_IF((outputInfo_.outputDim1 != expectedOutputDim1),
            OP_LOGE(context_->GetNodeName(),
            "Output y's activateDim dimension size %ld should equal to x's activateDim dimension size / 2, expected "
            "%ld.",
            outputInfo_.outputDim1, expectedOutputDim1),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(outputInfo_.outputDim2 != inputInfo_.inputDim2,
            OP_LOGE(context_->GetNodeName(),
            "activateDim = -2, yShape[-1] must be equal xShape[-1], but yShape[-1] is %ld, xShape[-1] is %ld",
            outputInfo_.outputDim2, inputInfo_.inputDim2),
            return ge::GRAPH_FAILED);
        for (int64_t i = 0; i < yDimNum - CONST_TWO; i++) {
            OP_CHECK_IF((yShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(i)),
                OP_LOGE(context_->GetNodeName(),
                "activateDim = -2, yShape[i] must be equal xShape[i], but i is %ld, yShape[i] is %ld, xShape[i] is %ld",
                i, yShape->GetStorageShape().GetDim(i), xShape->GetStorageShape().GetDim(i)),
                return ge::GRAPH_FAILED);
        }
    }
    // 对齐检查
    bool alignLast = outputInfo_.outputDim2 % CONST_TWO == 0 ? true : false;
    OP_CHECK_IF((attrParam_.dstType == DTYPE_40 || attrParam_.dstType == DTYPE_41) && (!alignLast),
        OP_LOGE(context_->GetNodeName(), "When dst_type is FP4, outputShape last dim must be even number, but is %ld",
        outputInfo_.outputDim2),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluMxQuantRegbaseTiling::PreProcess()
{
    const gert::StorageShape* xShape = context_->GetInputShape(0);
    int64_t dimNum = inputInfo_.dimNum;
    const gert::StorageShape* yShape = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape);
    auto yShapeNew = Ops::NN::OpTiling::EnsureNotScalar(yShape->GetStorageShape());
    int64_t yShapeSize = yShapeNew.GetShapeSize();
    int64_t yDimNum = static_cast<int64_t>(yShape->GetStorageShape().GetDimNum());
    OP_CHECK_IF(yShapeSize <= 0 || yDimNum != dimNum,
        OP_LOGE(context_->GetNodeName(),
        "y not support empty tensor, rank of yShape must equal rank of xShape, but xRank=%ld, yRank=%ld", dimNum,
        yDimNum),
        return ge::GRAPH_FAILED);
    outputInfo_.outputDim2 = yShape->GetStorageShape().GetDim(yDimNum - 1);
    outputInfo_.outputDim1 = yShape->GetStorageShape().GetDim(yDimNum - 2);
    if ((attrParam_.axis == -1) && (attrParam_.activateDim == -1)) {
        int64_t inDim1 = 1;
        int64_t outDim1 = 1;
        for (int64_t i = 0; i < dimNum - 1; i++) {
            inDim1 *= xShape->GetStorageShape().GetDim(i);
            outDim1 *= yShape->GetStorageShape().GetDim(i);
        }
        inputInfo_.inputDim1 = inDim1;
        outputInfo_.outputDim1 = outDim1;
    }
    if ((attrParam_.axis == -2 || attrParam_.activateDim == -2) && inputInfo_.groupIndexType == 0) {
        for (int64_t i = 0; i < dimNum - 2; i++) {
            inputInfo_.inputDim0 *= xShape->GetStorageShape().GetDim(i);
            outputInfo_.outputDim0 *= yShape->GetStorageShape().GetDim(i);
        }
    }
    OP_LOGI(context_->GetNodeName(), "3D view: dim0=%ld, dim1=%ld, dim2=%ld", inputInfo_.inputDim0,
        inputInfo_.inputDim1, inputInfo_.inputDim2);
    OP_CHECK_IF(ValidateYOutput(xShape, yShape, yDimNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "Tiling ValidateYOutput failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluMxQuantRegbaseTiling::ComputeTilingAxisNotLast()
{
    tilingResult_.basicDim1 = BASE_DIM0;
    tilingResult_.basicDim2 = BASE_DIM1;
    int64_t inHalfSize_ = tilingResult_.basicDim1 * tilingResult_.basicDim2;
    int64_t inBufferSize = inHalfSize_ * BYTES_OF_FP16;
    int64_t xUb = inBufferSize * CONST_TWO * CONST_TWO;
    int64_t swigluUb = inBufferSize;
    int64_t yUb = inHalfSize_ * CONST_TWO;
    int64_t scaleUb = tilingResult_.basicDim2 * CONST_THREE * CONST_TWO;
    int64_t scaleInt16Ub = tilingResult_.basicDim2 * CONST_TWO * CONST_TWO;
    int64_t allNeedUb = xUb + swigluUb + yUb + scaleUb + scaleInt16Ub;
    OP_CHECK_IF(allNeedUb > compileInfo_.ubSize, OP_LOGE(context_->GetNodeName(), "ub not enough"),
        return ge::GRAPH_FAILED);
    tilingResult_.dimMBlockNum = Ops::Base::CeilDiv(outputInfo_.outputDim1, tilingResult_.basicDim1);
    tilingResult_.dimNBlockNum = Ops::Base::CeilDiv(outputInfo_.outputDim2, tilingResult_.basicDim2);
    tilingResult_.blockCountPerBatch = tilingResult_.dimMBlockNum * tilingResult_.dimNBlockNum;
    int64_t totalBlocks = inputInfo_.inputDim0 * tilingResult_.blockCountPerBatch;
    tilingResult_.usedCoreNum = std::min(totalBlocks, compileInfo_.totalCoreNum);
    if (inputInfo_.groupIndexType != 0) {
        tilingResult_.usedCoreNum = compileInfo_.totalCoreNum;
    }
    tilingResult_.dimMTail = outputInfo_.outputDim1 - (tilingResult_.dimMBlockNum - 1) * tilingResult_.basicDim1;
    tilingResult_.dimNTail = outputInfo_.outputDim2 - (tilingResult_.dimNBlockNum - 1) * tilingResult_.basicDim2;
    tilingResult_.tailCoreBasicNumDim1 = Ops::Base::FloorDiv(totalBlocks, tilingResult_.usedCoreNum);
    tilingResult_.frontCoreNum = totalBlocks % tilingResult_.usedCoreNum;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluMxQuantRegbaseTiling::ComputeTilingAxisLast()
{
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
    // Output Scale: axis=-1 groups by columns, axis=-2 groups by rows
    int64_t scaleCount = tilingResult_.basicDim1 * tilingResult_.basicDim2 / BLOCK_SIZE;
    bytesPerIteration += scaleCount * BYTES_OF_FP8;
    bytesPerIteration *= DOUBLE_BUFFER;
    bytesPerIteration += scaleCount * BYTES_OF_FP16;                                        // reciprocal_scale
    bytesPerIteration += scaleCount * BYTES_OF_INT16;                                       // max_exp
    bytesPerIteration += tilingResult_.basicDim1 * tilingResult_.basicDim2 * BYTES_OF_FP16; // SwiGLU

    // Calculate how many basic blocks can fit in UB
    int64_t ubTotalBasicBlock = availableUB / bytesPerIteration;
    OP_LOGI(context_->GetNodeName(), "ubTotalBasicBlock is %ld", ubTotalBasicBlock);

    // Calculate basic blocks per row
    if (ubTotalBasicBlock >= tilingResult_.dimNBlockNum) {
        // Full-load scenario: entire row fits in UB
        tilingResult_.maxBasicNumUbDim2 = tilingResult_.dimNBlockNum;
        tilingResult_.maxBasicNumUbDim1 = Ops::Base::FloorDiv(ubTotalBasicBlock, tilingResult_.dimNBlockNum);
    } else {
        // Non-full-load scenario
        tilingResult_.maxBasicNumUbDim2 = ubTotalBasicBlock;
        tilingResult_.maxBasicNumUbDim1 = 1;
    }
    // Calculate blockIdx
    int64_t batch = inputInfo_.inputDim0;
    int64_t dimM = outputInfo_.outputDim1;
    int64_t totalBMRows = batch * dimM;
    // Step 1: prefer whole rows per core, only split N if extra cores available
    int64_t bmCores = std::min(totalBMRows, compileInfo_.totalCoreNum);
    int64_t nCores = 1;
    if (bmCores < compileInfo_.totalCoreNum) {
        nCores = compileInfo_.totalCoreNum / bmCores;
        nCores = std::min(nCores, tilingResult_.dimNBlockNum);
    }
    if (attrParam_.axis == -1 && attrParam_.activateDim == -2) {
        int64_t bCores = std::min(batch, bmCores);
        int64_t mCoresPerB = bmCores / bCores;
        tilingResult_.bCoreNum = bCores;
        tilingResult_.mCorePerB = mCoresPerB;
    } else {
        tilingResult_.bCoreNum = 1;
        tilingResult_.mCorePerB = bmCores;
    }
    tilingResult_.nCoreNum = nCores;
    tilingResult_.usedCoreNum = tilingResult_.bCoreNum * tilingResult_.mCorePerB * nCores;
    if (inputInfo_.groupIndexType != 0) {
        tilingResult_.usedCoreNum = compileInfo_.totalCoreNum;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluMxQuantRegbaseTiling::CalculateTiling()
{
    if (attrParam_.axis == -1) {
        tilingResult_.basicDim1 = 1;
        tilingResult_.basicDim2 = BASE_DIM1;
        tilingResult_.dimNBlockNum = Ops::Base::CeilDiv(outputInfo_.outputDim2, tilingResult_.basicDim2);
        tilingResult_.dimNTail = outputInfo_.outputDim2 - (tilingResult_.dimNBlockNum - 1) * tilingResult_.basicDim2;
        return ComputeTilingAxisLast();
    } else {
        return ComputeTilingAxisNotLast();
    }
}

void SwigluMxQuantRegbaseTiling::SetTilingKeyAndCore()
{
    groupIndexType_ = static_cast<uint64_t>(inputInfo_.groupIndexType);
    axisLast_ = (attrParam_.axis == -1) ? TPL_AXIS_LAST : TPL_AXIS_NOT_LAST;
    activateDimLast_ = (attrParam_.activateDim == -1) ? TPL_ACTIVATE_LAST : TPL_ACTIVATE_NOT_LAST;

    int64_t tilingKey = GET_TPL_TILING_KEY(groupIndexType_, axisLast_, activateDimLast_, roundMode_);
    OP_LOGI(context_->GetNodeName(),
        "!!!! groupIndexType=%lu, axisLast=%lu, activateDimLast=%lu, roundMode_=%ld, tilingKey=%ld", groupIndexType_,
        axisLast_, activateDimLast_, roundMode_, tilingKey);
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(tilingData_->usedCoreNum);
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

    // 3D data shape
    tilingData_->inputDim0 = inputInfo_.inputDim0;
    tilingData_->inputDim1 = outputInfo_.outputDim1;
    tilingData_->inputDim2 = outputInfo_.outputDim2; // axis=-2: N (SwiGLU output)

    // Memory allocation
    tilingData_->maxBasicNumUbDim2 = tilingResult_.maxBasicNumUbDim2;
    tilingData_->maxBasicNumUbDim1 = tilingResult_.maxBasicNumUbDim1;

    // Inter-core split
    tilingData_->frontCoreNum = tilingResult_.frontCoreNum;
    tilingData_->tailCoreBasicNumDim1 = tilingResult_.tailCoreBasicNumDim1;

    // 3D block distribution (axis=-2 path)
    tilingData_->dimNBlockNum = tilingResult_.dimNBlockNum;
    tilingData_->dimNTail = tilingResult_.dimNTail;
    tilingData_->dimMBlockNum = tilingResult_.dimMBlockNum;
    tilingData_->dimMTail = tilingResult_.dimMTail;
    tilingData_->blockCountPerBatch = tilingResult_.blockCountPerBatch;
    // 3D grid core distribution (act=-2, axis=-1 path)
    tilingData_->nCoreNum = tilingResult_.nCoreNum;
    tilingData_->bCoreNum = tilingResult_.bCoreNum;
    tilingData_->mCorePerB = tilingResult_.mCorePerB;

    // SwiGLU parameters
    tilingData_->activateLeft = attrParam_.activateLeft ? 1 : 0;
    tilingData_->swigluMode = attrParam_.swigluMode;
    tilingData_->clampLimit = attrParam_.clampLimit;
    tilingData_->gluAlpha = attrParam_.gluAlpha;
    tilingData_->gluBias = attrParam_.gluBias;

    // Quantization parameters
    tilingData_->scaleAlg = attrParam_.scaleAlg;
    tilingData_->groupMode = attrParam_.groupMode;
    tilingData_->groupIndexNum = inputInfo_.groupIndexNum;
    return ge::GRAPH_SUCCESS;
}

void SwigluMxQuantRegbaseTiling::PrintTilingData() const
{
    OP_LOGI(context_->GetNodeName(),
        "TilingData: usedCoreNum=%ld, inputDim0=%ld, inputDim1=%ld, inputDim2=%ld, "
        "maxBasicNumUbDim2=%ld, maxBasicNumUbDim1=%ld, frontCoreNum=%ld, tailCoreBasicNumDim1=%ld, scaleAlg=%ld, "
        "groupIndexNum=%ld, swigluMode=%ld, "
        "activateLeft=%ld, dimMBlockNum=%ld, dimNBlockNum = %ld, blockCountPerBatch=%ld, dimMTail=%ld, dimNTail=%ld, "
        "bCoreNum=%ld, mCorePerB=%ld, nCoreNum=%ld, clampLimit=%f, gluBias=%f, gluAlpha=%f",
        tilingData_->usedCoreNum, tilingData_->inputDim0, tilingData_->inputDim1, tilingData_->inputDim2,
        tilingData_->maxBasicNumUbDim2, tilingData_->maxBasicNumUbDim1, tilingData_->frontCoreNum,
        tilingData_->tailCoreBasicNumDim1, tilingData_->scaleAlg, tilingData_->groupIndexNum, tilingData_->swigluMode,
        tilingData_->activateLeft, tilingData_->dimMBlockNum, tilingData_->dimNBlockNum,
        tilingData_->blockCountPerBatch, tilingData_->dimMTail, tilingData_->dimNTail, tilingData_->bCoreNum,
        tilingData_->mCorePerB, tilingData_->nCoreNum, tilingData_->clampLimit, tilingData_->gluBias,
        tilingData_->gluAlpha);
}

ge::graphStatus SwigluMxQuantRegbaseTiling::SetParams()
{
    // Set workspace
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t usrWorkspaceSize = 0;

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = usrWorkspaceSize + sysWorkspaceSize;

    OP_LOGI(context_->GetNodeName(), "SetParams done. BlockDim=%ld", tilingData_->usedCoreNum);

    return ge::GRAPH_SUCCESS;
}

// ==================== TilingPrepare ====================
ge::graphStatus TilingPrepare4SwigluMxQuant(gert::TilingParseContext *context)
{
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
    // Phase 3.5: Post process (检查 shape 对齐要求)
    OP_CHECK_IF(tilingImpl.PreProcess() != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "Post process failed."),
        return ge::GRAPH_FAILED);
    // Phase 3: Validate output
    OP_CHECK_IF(tilingImpl.ValidateScaleOutput() != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Output validation failed."), return ge::GRAPH_FAILED);
    // 后续需要把输出shape和输入shape非axis对应的其他维度校验加上去，防止非法的shape情况

    // Phase 4: Calculate tiling
    OP_CHECK_IF(tilingImpl.CalculateTiling() != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Tiling calculation failed."), return ge::GRAPH_FAILED);

    // Phase 5: Fill tiling data
    OP_CHECK_IF(tilingImpl.FillTilingData() != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Failed to fill tiling data."), return ge::GRAPH_FAILED);

    // Phase 6: Set params (TilingKey, BlockDim, Workspace)
    tilingImpl.SetTilingKeyAndCore();
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