/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_quant_max_tiling_arch35.h"
#include <numeric>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/grouped_quant_max_tiling_data.h"
#include "../../op_kernel/arch35/grouped_quant_max_struct.h"
#include "atvoss/broadcast/broadcast_tiling.h"

using namespace Ops::Base;
using namespace GroupedQuantMaxOp;
using namespace gert;

namespace optiling {
constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t INPUT_SCALE_INDEX = 1;
constexpr size_t INPUT_GROUP_LIST_INDEX = 2;
constexpr size_t ATTR_ROUND_MODE_INDEX = 0;
constexpr size_t ATTR_DST_TYPE_INDEX = 1;
constexpr size_t OUTPUT_Y_INDEX = 0;
constexpr size_t OUTPUT_AMAX_INDEX = 1;
constexpr size_t BYTES_OF_INPUT_SCALE_TYPE = 4;
constexpr size_t BYTES_OF_INPUT_GROUP_LIST_TYPE = 8;
constexpr size_t BYTES_OF_TMP_AMAX_TYPE = 4;
constexpr int64_t ALIGN_BYTES = 32;
constexpr size_t FIRST_DIM = 0;
constexpr int64_t DEFAULT_BASE_LEN = 128;
constexpr int64_t BUFF_NUM = 2;
constexpr int64_t SIZE_2MB = 2 * 1024 * 1024;

ge::graphStatus GroupedQuantMaxRegbase::DoTiling()
{
    OP_CHECK_IF((GetPlatform() != ge::GRAPH_SUCCESS),
                OP_LOGE(context_->GetNodeName(), "DoGroupedQuantMaxTiling GetPlatform Failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((GetOpParam() != ge::GRAPH_SUCCESS),
                OP_LOGE(context_->GetNodeName(), "DoGroupedQuantMaxTiling GetOpParam Failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((CalcTiling() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "CalcTiling Failed."),
                return ge::GRAPH_FAILED);
    CalcTilingKey();
    return WriteTilingData();
}

ge::graphStatus GroupedQuantMaxRegbase::GetPlatform()
{
    OP_LOGD("GroupedQuantMaxTiling", "Enter arch3510 GroupedQuantMaxTiling");
    fe::PlatFormInfos* platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((static_cast<int32_t>(coreNum) <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get core num."),
                return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF((static_cast<int64_t>(ubSize) <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get ub size."),
                return ge::GRAPH_FAILED);

    coreNum_ = static_cast<int64_t>(coreNum);
    ubSize_ = ubSize;

    size_t usrSize = SIZE_2MB;
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedQuantMaxRegbase::CheckDtype()
{
    auto xInputDesc = context_->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInputDesc);
    xDtype_ = xInputDesc->GetDataType();
    OP_CHECK_IF(xDtype_ != ge::DT_FLOAT16 && xDtype_ != ge::DT_FLOAT && xDtype_ != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context_->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(xDtype_).c_str(),
                    "The dtype of x must be within the range [DT_FLOAT16, DT_FLOAT, DT_BF16]"),
                return ge::GRAPH_FAILED);

    auto scaleInputDesc = context_->GetInputDesc(INPUT_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleInputDesc);
    scaleDtype_ = scaleInputDesc->GetDataType();
    OP_CHECK_IF(scaleDtype_ != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "scale",
                                                      ge::TypeUtils::DataTypeToSerialString(scaleDtype_).c_str(),
                                                      "The dtype of scale must be DT_FLOAT"),
                return ge::GRAPH_FAILED);

    auto groupListInputDesc = context_->GetInputDesc(INPUT_GROUP_LIST_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, groupListInputDesc);
    groupListDtype_ = groupListInputDesc->GetDataType();
    OP_CHECK_IF(groupListDtype_ != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "groupList",
                                                      ge::TypeUtils::DataTypeToSerialString(groupListDtype_).c_str(),
                                                      "The dtype of groupList must be DT_INT64"),
                return ge::GRAPH_FAILED);

    auto yOutputDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yOutputDesc);
    yDtype_ = yOutputDesc->GetDataType();
    OP_CHECK_IF(yDtype_ != ge::DT_HIFLOAT8 && yDtype_ != ge::DT_FLOAT8_E5M2 && yDtype_ != ge::DT_FLOAT8_E4M3FN,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context_->GetNodeName(), "y", ge::TypeUtils::DataTypeToSerialString(yDtype_).c_str(),
                    "The dtype of y must be within the range [DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN]"),
                return ge::GRAPH_FAILED);

    auto amaxOutputDesc = context_->GetOutputDesc(OUTPUT_AMAX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, amaxOutputDesc);
    amaxDtype_ = amaxOutputDesc->GetDataType();
    OP_CHECK_IF(
        amaxDtype_ != xDtype_,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "amax", Ops::Base::ToString(amaxDtype_).c_str(),
                                              "The dtype of amax must be the same as x"),
        return ge::GRAPH_FAILED);

    dtypeSizeX_ = ge::GetSizeByDataType(xDtype_);
    dtypeSizeY_ = ge::GetSizeByDataType(yDtype_);

    return ge::GRAPH_SUCCESS;
}

RoundMode GroupedQuantMaxRegbase::GetRoundMode(const std::string& roundMode)
{
    if (dstType_ == ge::DT_FLOAT8_E5M2 || dstType_ == ge::DT_FLOAT8_E4M3FN) {
        if (roundMode == "rint") {
            return RoundMode::MODE_RINT;
        }
        errorMsg_ = "round_mode only supports 'rint' for float8_e5m2/float8_e4m3fn.";
        return RoundMode::MODE_UNDEFINED;
    }

    if (dstType_ == ge::DT_HIFLOAT8) {
        if (roundMode == "round") {
            return RoundMode::MODE_ROUND;
        } else if (roundMode == "hybrid") {
            return RoundMode::MODE_HYBRID;
        }
        errorMsg_ = "round_mode only supports 'round' and 'hybrid' for hifloat8.";
        return RoundMode::MODE_UNDEFINED;
    }
    return RoundMode::MODE_UNDEFINED;
}

ge::graphStatus GroupedQuantMaxRegbase::CheckAttrs()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const char* roundMode = attrs->GetAttrPointer<char>(ATTR_ROUND_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, roundMode);
    const int32_t* dstType = attrs->GetAttrPointer<int32_t>(ATTR_DST_TYPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dstType);
    dstType_ = *dstType;

    if (dstType_ != ge::DT_HIFLOAT8 && dstType_ != ge::DT_FLOAT8_E5M2 && dstType_ != ge::DT_FLOAT8_E4M3FN) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), "dstType", ToString(static_cast<ge::DataType>(dstType_)).c_str(),
            "dstType is invalid, must be DT_HIFLOAT8/DT_FLOAT8_E5M2/DT_FLOAT8_E4M3FN");
        return ge::GRAPH_FAILED;
    }
    if (dstType_ != yDtype_) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "dstType",
                                              ToString(static_cast<ge::DataType>(dstType_)).c_str(),
                                              "dstType must be equal to output y dtype");
        return ge::GRAPH_FAILED;
    }

    std::string roundModeStr = roundMode;
    roundMode_ = GetRoundMode(roundModeStr);
    OP_CHECK_IF(
        (roundMode_ == RoundMode::MODE_UNDEFINED),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "roundMode", roundMode, errorMsg_.c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static bool IsShapeEqualsNumGroups(const gert::Shape& shape, int num_groups)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == num_groups;
}

ge::graphStatus GroupedQuantMaxRegbase::CheckShape(const gert::Shape& xShape, const gert::Shape& scaleShape,
                                                   const gert::Shape& groupListShape, const gert::Shape& yShape,
                                                   const gert::Shape& amaxShape) const
{
    size_t xDimNum = xShape.GetDimNum();
    size_t scaleDimNum = scaleShape.GetDimNum();
    size_t groupListDimNum = groupListShape.GetDimNum();

    OP_CHECK_IF(xDimNum > 8 || xDimNum < 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x", std::to_string(xDimNum).c_str(),
                                                         "The shape dim of x must be in range [2D, 8D]"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        scaleDimNum != 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "scale", std::to_string(scaleDimNum).c_str(),
                                                 "The shape dim of scale must be 1D"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(groupListDimNum != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "groupList",
                                                         std::to_string(groupListDimNum).c_str(),
                                                         "The shape dim of groupList must be 1D"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(!IsShapeEqualsNumGroups(scaleShape, groupListShape.GetDim(0)),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), "scale, groupList",
                    (Ops::Base::ToString(scaleShape) + ", " + Ops::Base::ToString(groupListShape)).c_str(),
                    "The shape of scale must be equal to groupList"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(!IsShapeEqualsNumGroups(amaxShape, groupListShape.GetDim(0)),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), "amax, groupList",
                    (Ops::Base::ToString(amaxShape) + ", " + Ops::Base::ToString(groupListShape)).c_str(),
                    "The shape of amax must be equal to groupList"),
                return ge::GRAPH_FAILED);

    auto num_groups = groupListShape.GetDim(0);
    OP_CHECK_IF(num_groups < 1 || num_groups > 384,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "groupList.dim0",
                                                      std::to_string(num_groups).c_str(),
                                                      "The value must be in range [1, 384]"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(xShape != yShape,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), "x, y",
                    (std::to_string(xShape.GetShapeSize()) + ", " + std::to_string(yShape.GetShapeSize())).c_str(),
                    "The shape of x must equal the shape of y"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void GroupedQuantMaxRegbase::MergeInputShape(const gert::Shape& input)
{
    int64_t shape0 = 1;
    int64_t shape1 = 1;

    for (size_t idx = 1; idx < input.GetDimNum(); ++idx) {
        shape1 = shape1 * input.GetDim(idx);
    }

    dim1_ = shape1;

    for (size_t idx = 0; idx < input.GetDimNum(); ++idx) {
        shape0 = shape0 * input.GetDim(idx);
    }

    xInputShape_.SetDimNum(1);
    xInputShape_.SetDim(FIRST_DIM, shape0);
    OP_LOGI(context_->GetNodeName(), "merged shape:%ld", shape0);
}

ge::graphStatus GroupedQuantMaxRegbase::GetOpParam()
{
    auto xInput = context_->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInput);
    const gert::Shape& xInputShape = Ops::Base::EnsureNotScalar(xInput->GetStorageShape());

    auto scaleInput = context_->GetInputShape(INPUT_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleInput);
    const gert::Shape& scaleInputShape = Ops::Base::EnsureNotScalar(scaleInput->GetStorageShape());

    auto groupListInput = context_->GetInputShape(INPUT_GROUP_LIST_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, groupListInput);
    const gert::Shape& groupListInputShape = Ops::Base::EnsureNotScalar(groupListInput->GetStorageShape());

    auto yOutput = context_->GetOutputShape(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yOutput);
    const gert::Shape& yOutputShape = Ops::Base::EnsureNotScalar(yOutput->GetStorageShape());

    auto amaxOutput = context_->GetOutputShape(OUTPUT_AMAX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, amaxOutput);
    const gert::Shape& amaxOutputShape = Ops::Base::EnsureNotScalar(amaxOutput->GetStorageShape());

    int64_t xSizeNum = xInputShape.GetShapeSize();
    if (xSizeNum == 0ULL) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x", "empty",
                                              "grouped_quant_max does not support empty tensor");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF((CheckDtype() != ge::GRAPH_SUCCESS),
                OP_LOGE(context_->GetNodeName(), "check input/output dtype failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF((CheckAttrs() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "op attrs is invalid."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((CheckShape(xInputShape, scaleInputShape, groupListInputShape, yOutputShape, amaxOutputShape) !=
                 ge::GRAPH_SUCCESS),
                OP_LOGE(context_->GetNodeName(), "input/output shape is invalid."), return ge::GRAPH_FAILED);

    MergeInputShape(xInputShape);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedQuantMaxRegbase::CalcUbFactor()
{
    int64_t availableUb = static_cast<int64_t>(ubSize_) - reserveUb_;
    auto groupListShapePtr = context_->GetInputShape(INPUT_GROUP_LIST_INDEX);
    auto groupListShape = groupListShapePtr->GetStorageShape();
    int64_t num_groups = groupListShape.GetShapeSize();
    numGroups_ = num_groups;

    int64_t scaleSize = ((num_groups * BYTES_OF_INPUT_SCALE_TYPE + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES;
    int64_t groupListSize = ((num_groups * BYTES_OF_INPUT_GROUP_LIST_TYPE + ALIGN_BYTES - 1) / ALIGN_BYTES) *
                            ALIGN_BYTES;
    int64_t maxSize = ((num_groups * BYTES_OF_TMP_AMAX_TYPE + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES;
    int64_t groupMaxSize = ((coreNum_ * num_groups * BYTES_OF_TMP_AMAX_TYPE + ALIGN_BYTES - 1) / ALIGN_BYTES) *
                           ALIGN_BYTES;
    int64_t amaxSize = ((num_groups * BYTES_OF_TMP_AMAX_TYPE + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES;
    int64_t zero = ((num_groups * BYTES_OF_TMP_AMAX_TYPE + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES;
    int64_t tmpBuffer = scaleSize + groupListSize + maxSize + groupMaxSize + amaxSize + zero;

    OP_CHECK_IF((tmpBuffer >= availableUb), OP_LOGE(context_->GetNodeName(), "UB allocation overflow."),
                return ge::GRAPH_FAILED);

    ubFactor_ = (availableUb - tmpBuffer) / (BUFF_NUM * (dtypeSizeX_ + dtypeSizeY_));

    int64_t alignElemX = ALIGN_BYTES / dtypeSizeX_;
    int64_t alignElemY = ALIGN_BYTES / dtypeSizeY_;
    int64_t alignElem = std::lcm(alignElemX, alignElemY);
    ubFactor_ = (ubFactor_ / alignElem) * alignElem;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedQuantMaxRegbase::CalcTiling()
{
    int64_t shape = xInputShape_.GetDim(FIRST_DIM);
    int64_t dtypeSize = ge::GetSizeByDataType(xDtype_);
    OP_CHECK_IF((dtypeSize <= 0), OP_LOGE(context_->GetNodeName(), "dtypeSize is invalid: %ld", dtypeSize),
                return ge::GRAPH_FAILED);
    actualCoreNum_ = std::min(shape, coreNum_);
    blockFactor_ = CeilDiv(shape, actualCoreNum_);
    blockTailCoreNum_ = (actualCoreNum_ * blockFactor_) - shape;
    if (blockTailCoreNum_ == 0) {
        blockTailFactor_ = blockFactor_;
    } else {
        blockTailFactor_ = blockFactor_ - 1;
    }

    OP_CHECK_IF((CalcUbFactor() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "CalcUBFactor failed."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void GroupedQuantMaxRegbase::CalcTilingKey()
{
    uint32_t roundModeKey = static_cast<uint32_t>(roundMode_);
    tilingKey_ = GET_TPL_TILING_KEY(roundModeKey);
    OP_LOGD(context_->GetNodeName(), "roundMode is %ld", roundMode_);
    context_->SetTilingKey(tilingKey_);
}

ge::graphStatus GroupedQuantMaxRegbase::WriteTilingData()
{
    OP_LOGD(context_->GetNodeName(), "coreNum:%ld, tilingKey:%lu", coreNum_, tilingKey_);
    context_->SetBlockDim(coreNum_);

    GroupedQuantMaxTilingData* tilingData_ = context_->GetTilingData<GroupedQuantMaxTilingData>();
    tilingData_->roundMode = static_cast<int64_t>(roundMode_);

    OP_LOGD(context_->GetNodeName(),
            "totalCoreNum:%ld, actualCoreNum:%ld, blockTailCoreNum_:%ld, blockFactor:%ld, blockTailFactor:%ld, "
            "ubFactor:%ld, dim1:%ld, numGroups:%ld",
            coreNum_, actualCoreNum_, blockTailCoreNum_, blockFactor_, blockTailFactor_, ubFactor_, dim1_, numGroups_);
    tilingData_->totalCoreNum = coreNum_;
    tilingData_->actualCoreNum = actualCoreNum_;
    tilingData_->blockTailCoreNum = blockTailCoreNum_;
    tilingData_->blockFactor = blockFactor_;
    tilingData_->blockTailFactor = blockTailFactor_;
    tilingData_->ubFactor = ubFactor_;
    tilingData_->dim1 = dim1_;
    tilingData_->numGroups = numGroups_;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForGroupedQuantMax(gert::TilingContext* context)
{
    OP_LOGD("GroupedQuantMaxTiling", "Enter TilingForGroupedQuantMaxTiling");

    OP_CHECK_IF(context == nullptr, OP_LOGE("GroupedQuantMaxTiling", "Tiling context is null."),
                return ge::GRAPH_FAILED);

    GroupedQuantMaxRegbase GroupedQuantMaxTiling(context);
    return GroupedQuantMaxTiling.DoTiling();
}

static ge::graphStatus TilingPrepareForGroupedQuantMax(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("GroupedQuantMaxTiling", "Tiling context is null."),
                return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<GroupedQuantMaxCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->vectorCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    OP_CHECK_IF((compileInfoPtr->vectorCoreNum <= 0 || compileInfoPtr->ubSize <= 0),
                OP_LOGE(context->GetNodeName(), "GroupedQuantMax GetHardwareInfo Failed, vectorCoreNum:%d, ubSize:%lu.",
                        compileInfoPtr->vectorCoreNum, compileInfoPtr->ubSize),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "GetCoreNum:%d, ubSize:%lu", compileInfoPtr->vectorCoreNum, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(GroupedQuantMax)
    .Tiling(TilingForGroupedQuantMax)
    .TilingParse<GroupedQuantMaxCompileInfo>(TilingPrepareForGroupedQuantMax);
} // namespace optiling
