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
 * \file ascend_quant_tiling_arch35.cpp
 * \brief
 */

#include "arch35/ascend_quant_tiling_arch35.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "../../op_kernel/arch35/ascend_quant_struct.h"
#include "../../op_kernel/arch35/ascend_quant_tilingdata.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"

using namespace Ops::Base;

namespace optiling {
namespace ascendquantregbase {
using namespace AscendQuantOp;
constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t ATTR_SCALE_INDEX = 0;
constexpr size_t ATTR_OFFSET_INDEX = 1;
constexpr size_t ATTR_SQRT_MODE_INDEX = 2;
constexpr size_t ATTR_ROUND_MODE_INDEX = 3;
constexpr size_t ATTR_DST_TYPE_INDEX = 4;
constexpr size_t OUTPUT_Y_INDEX = 0;

constexpr size_t SYNC_WORKSPACE_SIZE = 16777216;
constexpr int64_t CACHE_SIZE_910D = 128;

constexpr size_t FIRST_DIM = 0;

static constexpr int64_t INT4_NUMS_IN_INT8_SPACE = 2;
constexpr int64_t DEFAULT_BASE_LEN = 128;
constexpr int64_t BUFF_NUM = 2;

const gert::Shape g_vec_1_shape = {1};

inline static const gert::Shape& EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

ge::graphStatus AscendQuantRegbase::DoAscendQuantTiling()
{
    OP_CHECK_IF(
        (GetPlatform() != ge::GRAPH_SUCCESS),
        OP_LOGE(context_->GetNodeName(), "DoAscendQuantTiling GetPlatform Failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (GetOpParam() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "DoAscendQuantTiling GetOpParam Failed."),
        return ge::GRAPH_FAILED);

    CalcTiling();
    CalcTilingKey();
    WriteTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AscendQuantRegbase::GetPlatform()
{
    OP_LOGD("AscendQuantTiling", "Enter arch3510 AscendQuantTiling");
    fe::PlatFormInfos* platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (static_cast<int32_t>(coreNum) <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get core num."),
        return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(
        (static_cast<int64_t>(ubSize) <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get ub size."),
        return ge::GRAPH_FAILED);

    coreNum_ = static_cast<int64_t>(coreNum);
    ubSize_ = ubSize;
    cacheLine_ = CACHE_SIZE_910D;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AscendQuantRegbase::CheckDtype()
{
    auto xInputDesc = context_->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInputDesc);
    xDtype_ = xInputDesc->GetDataType();
    OP_CHECK_IF(
        xDtype_ != ge::DT_FLOAT16 && xDtype_ != ge::DT_FLOAT,
        OP_LOGE(
            context_->GetNodeName(), "input x dtype [%s] not supported, only support [DT_FLOAT16, DT_FLOAT]",
            ge::TypeUtils::DataTypeToSerialString(xDtype_).c_str()),
        return ge::GRAPH_FAILED);

    auto yOutputDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yOutputDesc);
    yDtype_ = yOutputDesc->GetDataType();
    OP_CHECK_IF(
        yDtype_ != ge::DT_INT8 && yDtype_ != ge::DT_INT4 && yDtype_ != ge::DT_HIFLOAT8 &&
            yDtype_ != ge::DT_FLOAT8_E5M2 && yDtype_ != ge::DT_FLOAT8_E4M3FN,
        OP_LOGE(
            context_->GetNodeName(),
            "output y dtype [%s] not supported, only support [DT_INT8, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E5M2, "
            "DT_FLOAT8_E4M3FN]",
            ge::TypeUtils::DataTypeToSerialString(yDtype_).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

RoundMode AscendQuantRegbase::GetRoundMode(std::string& roundMode)
{
    if (dstType_ == ge::DT_FLOAT8_E5M2 || dstType_ == ge::DT_FLOAT8_E4M3FN) {
        if (roundMode == "Round") {
            return RoundMode::MODE_RINT;
        }
        errorMsg_ = "round_mode only supports 'Round' for float8_e5m2/float8_e4m3fn.";
        return RoundMode::MODE_UNDEFINED;
    }

    if (dstType_ == ge::DT_HIFLOAT8) {
        if (roundMode == "Round") {
            return RoundMode::MODE_ROUND;
        } else if (roundMode == "Hybrid") {
            return RoundMode::MODE_HYBRID;
        }
        errorMsg_ = "round_mode only supports 'Round' and 'Hybrid' for hifloat8.";
        return RoundMode::MODE_UNDEFINED;
    }

    if (roundMode == "Round") {
        return RoundMode::MODE_ROUND;
    } else if (roundMode == "Floor") {
        return RoundMode::MODE_FLOOR;
    } else if (roundMode == "Ceil") {
        return RoundMode::MODE_CEIL;
    } else if (roundMode == "Trunc") {
        return RoundMode::MODE_TRUNC;
    }
    errorMsg_ = "round_mode only supports 'Round', 'Floor', 'Ceil' and 'Trunc' for int8/int4.";
    return RoundMode::MODE_UNDEFINED;
}

ge::graphStatus AscendQuantRegbase::CheckAttrs()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    // get scale
    const auto* scale = attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scale);
    scale_ = *scale;
    // get offset
    const auto* offset = attrs->GetAttrPointer<float>(ATTR_OFFSET_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, offset);
    offset_ = *offset;
    // get sqrtMode
    const auto* sqrtMode = attrs->GetAttrPointer<bool>(ATTR_SQRT_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sqrtMode);
    if (*sqrtMode) {
        scale_ = scale_ * scale_;
    }
    // get roundMode
    const char* roundMode = attrs->GetAttrPointer<char>(ATTR_ROUND_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, roundMode);
    // get dstType
    const int32_t* dstType = attrs->GetAttrPointer<int32_t>(ATTR_DST_TYPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dstType);
    dstType_ = *dstType;

    // check dstType and output dtype, must be same
    if (dstType_ != ge::DT_INT8 && dstType_ != ge::DT_INT4 && dstType_ != ge::DT_HIFLOAT8 &&
        dstType_ != ge::DT_FLOAT8_E5M2 && dstType_ != ge::DT_FLOAT8_E4M3FN) {
        OP_LOGE(
            context_->GetNodeName(), "dst type:%s is invalid", ToString(static_cast<ge::DataType>(dstType_)).c_str());
        return ge::GRAPH_FAILED;
    }
    if (dstType_ != yDtype_) {
        OP_LOGE(
            context_->GetNodeName(), "dst type:%s not equal output y dtype:%s",
            ToString(static_cast<ge::DataType>(dstType_)).c_str(), ToString(yDtype_).c_str());
        return ge::GRAPH_FAILED;
    }

    // check round mode
    std::string roundModeStr = roundMode;
    roundMode_ = GetRoundMode(roundModeStr);
    OP_CHECK_IF(
        (roundMode_ == RoundMode::MODE_UNDEFINED),
        OP_LOGE(context_->GetNodeName(), "invalid round mode:%s, %s", roundMode, errorMsg_.c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AscendQuantRegbase::CheckShape(const gert::Shape& xShape, const gert::Shape& yShape) const
{
    size_t xDimNum = xShape.GetDimNum();

    OP_CHECK_IF(
        xDimNum > 8 || xDimNum < 1, OP_LOGE(context_->GetNodeName(), "input x dim num should be in range [1, 8]"),
        return ge::GRAPH_FAILED);

    if (dstType_ == ge::DT_INT4 && (xShape.GetDim(xDimNum - 1) % INT4_NUMS_IN_INT8_SPACE)) {
        OP_LOGE(
            context_->GetNodeName(), "if dst_type represents DT_INT4, x last dim:%ld must be divisible by 2",
            xShape.GetDim(xDimNum - 1));
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        xShape != yShape, OP_LOGE(context_->GetNodeName(), "input x and output y shape not same"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void AscendQuantRegbase::MergeInputShape(const gert::Shape& input)
{
    int64_t shape0 = 1;

    for (size_t idx = 0; idx < input.GetDimNum(); ++idx) {
        shape0 = shape0 * input.GetDim(idx);
    }

    // merge shape to 1 dim
    xInputShape_.SetDimNum(1);
    xInputShape_.SetDim(FIRST_DIM, shape0);
    OP_LOGI(context_->GetNodeName(), "merged shape:%ld", shape0);
}

ge::graphStatus AscendQuantRegbase::GetOpParam()
{
    auto xInput = context_->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInput);
    const gert::Shape& xInputShape = EnsureNotScalar(xInput->GetStorageShape());

    auto yOutput = context_->GetOutputShape(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yOutput);
    const gert::Shape& yOutputShape = EnsureNotScalar(yOutput->GetStorageShape());

    size_t xSizeNum = xInputShape.GetShapeSize();
    if (xSizeNum == 0ULL) {
        OP_LOGE(context_->GetNodeName(), "ascend_quant does not support empty tensor.");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        (CheckDtype() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "check input/output dtype failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CheckAttrs() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "op attrs is invalid."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CheckShape(xInputShape, yOutputShape) != ge::GRAPH_SUCCESS),
        OP_LOGE(context_->GetNodeName(), "input/output shape is invalid."), return ge::GRAPH_FAILED);

    MergeInputShape(xInputShape);

    return ge::GRAPH_SUCCESS;
}

int64_t AscendQuantRegbase::GetCoreNum(int64_t factor, int64_t coreNum) const
{
    int64_t elePerCore = CeilDiv(factor, coreNum);
    int64_t actCore = CeilDiv(factor, elePerCore);
    return actCore;
}

int64_t AscendQuantRegbase::CalcMaxBaseLen(int64_t ubSize) const
{
    // set n == 1 to calc max base
    int64_t xDtypeSize = ge::GetSizeByDataType(xDtype_);
    int64_t yDtypeSize = ge::GetSizeByDataType(ge::DT_INT8);

    int64_t totalBytes = (xDtypeSize + yDtypeSize) * BUFF_NUM;
    return totalBytes == 0 ? DEFAULT_BASE_LEN : ubSize / totalBytes;
}

void AscendQuantRegbase::CalcPerTensorBlockFactor(int64_t size)
{
    // 以一个cache为基本单位，计算block块的宽度
    blockFactor_ = CeilDiv(size, actCoreNum_);
    int64_t shape = xInputShape_.GetDim(FIRST_DIM);
    int64_t dtypeSize = ge::GetSizeByDataType(xDtype_);
    blockFactor_ = blockFactor_ * cacheLine_ / dtypeSize;
    blockTailFactor_ = shape - blockFactor_ * (actCoreNum_ - 1);
    blockTailFactor_ = blockTailFactor_ == 0 ? blockFactor_ : blockTailFactor_;
}

void AscendQuantRegbase::CalcPerTensorUBFactor(int64_t numPerCache)
{
    int64_t availableUb = static_cast<int64_t>(ubSize_) - reserveUb_;
    int64_t maxBase = CalcMaxBaseLen(availableUb); // 一个UB能算的数
    maxBase = FloorAlign(maxBase, numPerCache);    // 用cacheLine对齐
    int64_t blockBase = blockFactor_;              // 合成一个轴时，block块的宽度
    blockBase = CeilAlign(blockBase, numPerCache);
    baseLen_ = std::min(blockBase, maxBase);
}

void AscendQuantRegbase::CalcTiling()
{
    // per tensor模式，所有轴合一
    int64_t shape = xInputShape_.GetDim(FIRST_DIM);
    int64_t dtypeSize = ge::GetSizeByDataType(xDtype_);
    int64_t cacheLineNum = CeilDiv(shape, cacheLine_ / dtypeSize);
    int64_t actCoreNum = GetCoreNum(cacheLineNum, coreNum_);

    actCoreNum_ = actCoreNum;
    int64_t size = cacheLineNum;
    CalcPerTensorBlockFactor(size);
    CalcPerTensorUBFactor(cacheLine_ / dtypeSize);
}

void AscendQuantRegbase::CalcTilingKey()
{
    uint32_t roundModeKey = static_cast<uint32_t>(roundMode_) + 1;
    tilingKey_ = GET_TPL_TILING_KEY(roundModeKey, TPL_EXTRA_BIT_ONE);
}

void AscendQuantRegbase::WriteTilingData()
{
    OP_LOGD(context_->GetNodeName(), "coreNum:%ld, tilingKey:%lu", coreNum_, tilingKey_);
    context_->SetBlockDim(coreNum_);
    context_->SetTilingKey(tilingKey_);

    OP_LOGD(
        context_->GetNodeName(), "scale:%f, offset:%f, roundMode:%d, dstType:%d", scale_, offset_,
        static_cast<int16_t>(roundMode_), dstType_);
    AscendQuantTilingData* tilingData_ = context_->GetTilingData<AscendQuantTilingData>();
    tilingData_->roundMode = static_cast<int64_t>(roundMode_);
    tilingData_->scale = scale_;
    tilingData_->offset = offset_;

    OP_LOGD(
        context_->GetNodeName(), "actCoreNum:%ld, blockFactor:%ld, blockTailFactor:%ld, baseLen:%ld", actCoreNum_,
        blockFactor_, blockTailFactor_, baseLen_);
    tilingData_->numCore = actCoreNum_;
    tilingData_->blockFactor = blockFactor_;
    tilingData_->blockTailFactor = blockTailFactor_;
    tilingData_->baseLen = baseLen_;

    tilingData_->dim0 = xInputShape_.GetDim(FIRST_DIM);

    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = SYNC_WORKSPACE_SIZE;
}
} // namespace ascendquantregbase

} // namespace optiling