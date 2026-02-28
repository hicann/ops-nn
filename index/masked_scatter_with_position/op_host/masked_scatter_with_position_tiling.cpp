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
 * \file masked_scatter_with_position_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/masked_scatter_with_position_tiling_struct.h"
#include "masked_scatter_with_position_tiling.h"
#include "op_common/op_host/util/platform_util.h"
#include "platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "platform/platform_info.h"

namespace optiling {

static constexpr uint16_t INPUT_X_IDX = 0;
static constexpr uint16_t INPUT_MASK_IDX = 1;
static constexpr uint16_t INPUT_POSITION_IDX = 2;
static constexpr uint16_t INPUT_UPDATES_IDX = 3;
static constexpr uint16_t OUTPUT_Y_IDX = 0;

static constexpr uint16_t PATTERN_BA = 1;
static constexpr uint16_t PATTERN_AB = 2;
static constexpr uint16_t PATTERN_INT32 = 0;
static constexpr uint16_t PATTERN_INT64 = 1;

static constexpr uint32_t TILING_KEY_BASE = 100;
static constexpr int64_t DIGIT_TEN = 10;

static constexpr uint32_t DCACHE_SIZE = 128U * 1024U;
static constexpr int64_t THREAD_NUM = 1024;
static constexpr uint32_t DEFAULT_WORKSPACE_SIZE = 16 * 1024 * 1024;

static const std::set<ge::DataType> SUPPORT_DTYPE = {
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_DOUBLE, ge::DT_UINT8, ge::DT_INT8,
    ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_BOOL, ge::DT_BF16
};

bool MaskedScatterWithPositionTiling::IsCapable()
{
    return true;
}

ge::graphStatus MaskedScatterWithPositionTiling::GetPlatformInfo()  // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
{
    OP_LOGI(opName_, "MaskedScatterWithPositionTiling GetPlatformInfo.");
    auto compileInfo = static_cast<const MaskedScatterWithPositionCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);

    totalCoreNum_ = static_cast<int64_t>(compileInfo->totalCoreNum);
    ubSize_ = compileInfo->ubSize - DCACHE_SIZE;
    OP_CHECK_IF((ubSize_ <= 0), OP_LOGE(opName_, "ub size is invalid."), return ge::GRAPH_FAILED);
    OP_LOGI(opName_, "MaskedScatterWithPositionTiling::GetPlatformInfo ubSize_=%d, totalCoreNum_=%d", ubSize_, totalCoreNum_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedScatterWithPositionTiling::GetShapeAttrsInfo()  // 2、获取INPUT/OUTPUT/ATTR信息
{
    OP_LOGI(opName_, "MaskedScatterWithPositionTiling::GetShapeAttrsInfo begin.");
    OP_CHECK_IF(CheckDataType(),
        OP_LOGE(opName_, "CheckDataType failed!"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckOutputShape(),
        OP_LOGE(opName_, "CheckOutputShape failed!"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedScatterWithPositionTiling::DoOpTiling()  // 3、计算数据切分TilingData
{
    OP_LOGD(opName_, "MaskedScatterWithPositionTiling DoOpTiling.");
    isInt64Mask_ = xEleNums_ > INT32_MAX;
    usedCoreNum_ = Ops::Base::CeilDiv(xEleNums_, THREAD_NUM);
    usedCoreNum_ = std::min(usedCoreNum_, totalCoreNum_);
    MaskedScatterWithPositionTilingData* tilingData_ = context_->GetTilingData<MaskedScatterWithPositionTilingData>();
    tilingData_->xNum = xEleNums_;
    tilingData_->xInner = xInner_;
    tilingData_->updatesEleNums = updatesEleNums_;
    tilingData_->pattern = isBA_? static_cast<int64_t>(0) : static_cast<int64_t>(1);
    tilingData_->computeType = isInt64Mask_? static_cast<int64_t>(1) : static_cast<int64_t>(0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedScatterWithPositionTiling::DoLibApiTiling()  // 4、计算高阶API的TilingData
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MaskedScatterWithPositionTiling::GetTilingKey() const  // 5、计算TilingKey
{
    uint64_t tilingKey = TILING_KEY_BASE * (isBA_? PATTERN_BA : PATTERN_AB) + (isInt64Mask_ ? PATTERN_INT64 : PATTERN_INT32);
    return tilingKey;
}

ge::graphStatus MaskedScatterWithPositionTiling::GetWorkspaceSize()  // 6、计算Workspace 大小
{
    workspaceSize_ = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedScatterWithPositionTiling::PostTiling()  // 7、保存Tiling数据
{
    OP_LOGD(opName_, "MaskedScatterWithPositionTiling PostTiling.");
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    tilingKey_ = GetTilingKey();
    context_->SetTilingKey(tilingKey_);
    context_->SetBlockDim(usedCoreNum_);
    context_->SetLocalMemorySize(ubSize_);
    return ge::GRAPH_SUCCESS;
}

void MaskedScatterWithPositionTiling::DumpTilingInfo()
{
    std::ostringstream info;
    MaskedScatterWithPositionTilingData* tilingData_ =
        context_->GetTilingData<MaskedScatterWithPositionTilingData>();
    info << "tilingKey: " << tilingKey_;
    info << ", usedCoreNum: " << usedCoreNum_;
    info << ", totalCoreNum_: " << totalCoreNum_;
    info << ", maskEleNums_: " << maskEleNums_;
    info << ", isInt64Mask_: " << isInt64Mask_;
    info << ", isBA_: " << isBA_;
    info << ", xOuter_: " << xOuter_;
    info << ", xInner_: " << xInner_;
    info << ", updatesEleNums: " << tilingData_->updatesEleNums;
    info << ", xNum: " << tilingData_->xNum;
    info << ", xInner: " << tilingData_->xInner;
    info << ", pattern: " << tilingData_->pattern;
    info << ", computeType: " << tilingData_->computeType;

    OP_LOGI(opName_, "%s", info.str().c_str());
}

ge::graphStatus MaskedScatterWithPositionTiling::CheckOutputShape()
{
    OP_LOGI(opName_, "MaskedScatterWithPositionTiling::CheckOutputShape begin.");
    auto xShapePtr = context_->GetRequiredInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    xEleNums_ = xShape.GetShapeSize();

    auto updatesShapePtr = context_->GetRequiredInputShape(INPUT_UPDATES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, updatesShapePtr);
    auto updatesShape = updatesShapePtr->GetStorageShape();
    updatesEleNums_ = updatesShape.GetShapeSize();

    auto positionShapePtr = context_->GetRequiredInputShape(INPUT_POSITION_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, positionShapePtr);
    auto positionShape = positionShapePtr->GetStorageShape();

    auto maskShapePtr = context_->GetRequiredInputShape(INPUT_MASK_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maskShapePtr);
    auto maskShape = maskShapePtr->GetStorageShape();
    maskEleNums_ = maskShape.GetShapeSize();

    auto yShapePtr = context_->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();

    OP_CHECK_IF(xShape != yShape, OP_LOGE(opName_,
        "The x and y must have the same shape, please check."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(positionShape != maskShape, OP_LOGE(opName_,
        "The position and mask must have the same shape, please check."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(maskShape.GetDimNum() > xShape.GetDimNum() , OP_LOGE(opName_,
        "The shape dimension of x must be greater than or equal to the shape dimension of mask, please check."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(maskShape.GetDimNum() == 0 , OP_LOGE(opName_,
        "The mask cannot be empty, please check."), return ge::GRAPH_FAILED);
    bool isBAorAB = CanBroadcastBAOrAB(xShape, maskShape);
    OP_CHECK_IF(isBAorAB != true, OP_LOGE(opName_,
        "The x and position shape must be compatible with broadcasting rules of BA/AB, please check."), return ge::GRAPH_FAILED);

    OP_LOGI(opName_, "MaskedScatterWithPositionTiling::CheckOutputShape end.");
    return ge::GRAPH_SUCCESS;
}

bool MaskedScatterWithPositionTiling::CanBroadcastBAOrAB(const gert::Shape xShape, const gert::Shape maskShape)  // 是否为BA
{
    size_t lenA = xShape.GetDimNum();
    size_t lenB = maskShape.GetDimNum();
    bool BA = false;
    bool AB = false;

    std::vector<int64_t> xShapeVec(lenA);
    std::vector<int64_t> maskShapeVec(lenB);
    for (size_t i = 0; i < lenA; i++) {
        xShapeVec[i] = xShape[i];
    }
    for (size_t i = 0; i < lenB; i++) {
        maskShapeVec[i] = maskShape[i];
    }

    if (lenA == lenB) {
        CanBroadcastBAOrABEqual(xShapeVec, maskShapeVec,lenA, BA, AB);
    } else {
        std::vector<int64_t> maskShapeVecPadded(lenA);
        std::fill(maskShapeVecPadded.begin(), maskShapeVecPadded.begin() + (lenA - lenB), 1);
        std::copy(maskShapeVec.begin(), maskShapeVec.end(), maskShapeVecPadded.begin() + (lenA - lenB));
        CanBroadcastBAOrABEqual(xShapeVec, maskShapeVecPadded,lenA, BA, AB);
    }

    if (BA) {
        xOuter_ = xEleNums_ / maskEleNums_;  // mask(1,N)
        xInner_ = maskEleNums_;
        isBA_ = true;
        return true;
    } else if (AB) {
        xOuter_ = maskEleNums_;  // mask(M,1)
        xInner_ = xEleNums_ / maskEleNums_;
        isBA_ = false;
        return true;
    } else {
        return false;
    }
}

void MaskedScatterWithPositionTiling::CanBroadcastBAOrABEqual(const std::vector<int64_t>& xShapeVec, const std::vector<int64_t>& maskShapeVec, const size_t len, bool& BA, bool& AB){
    size_t i = 0;
    while(i < len && xShapeVec[i] == 1 && maskShapeVec[i] == 1){
        i++;
    }

    if (i == len) {  // all 1
        BA = true;
        AB = true;
        return;
    }

    if (maskShapeVec[i] == 1) {
        // BA:   xShapeVec[num1, num2, num3, num4, num5],
        //    maskShapeVec[1   , 1   , num3, num4, num5]
        while (i < len && maskShapeVec[i] == 1) i++;
        while (i < len && maskShapeVec[i] == xShapeVec[i]) i++;
        if (i == len) BA = true;
    } else {
        // AB:   xShapeVec[num1, num2, num3, num4, num5],
        //    maskShapeVec[num1, num2, num3, 1   , 1   ]
        while (i < len && maskShapeVec[i] == xShapeVec[i]) i++;
        while (i < len && maskShapeVec[i] == 1) i++;
        if (i == len) AB = true;
    }
}

ge::graphStatus MaskedScatterWithPositionTiling::CheckDataType()
{
    OP_LOGI(opName_, "MaskedScatterWithPositionTiling::CheckDataType begin.");
    auto inputXPtr = context_->GetRequiredInputDesc(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXPtr);
    dType_ = inputXPtr->GetDataType();
    OP_CHECK_IF(SUPPORT_DTYPE.count(dType_) == 0, OP_LOGE(opName_,
        "The dtype only support float32, float16, double, uint8, int8, int16, int32, int64, bool, bfloat16 currently, please check."),
        return ge::GRAPH_FAILED);

    auto inputMaskPtr = context_->GetRequiredInputDesc(INPUT_MASK_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputMaskPtr);
    auto maskDtype = inputMaskPtr->GetDataType();
    OP_CHECK_IF(maskDtype != ge::DT_BOOL, OP_LOGE(opName_,
        "The dtype of mask only support bool currently, please check."), return ge::GRAPH_FAILED);

    auto inputPositionPtr = context_->GetRequiredInputDesc(INPUT_POSITION_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputPositionPtr);
    auto positionDtype = inputPositionPtr->GetDataType();
    OP_CHECK_IF(positionDtype != ge::DT_INT64, OP_LOGE(opName_,
        "The dtype of position only support int64 currently, please check."), return ge::GRAPH_FAILED);

    auto inputUpdatesPtr = context_->GetRequiredInputDesc(INPUT_UPDATES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputUpdatesPtr);
    auto updatesDtype = inputUpdatesPtr->GetDataType();

    auto outputYPtr = context_->GetOutputDesc(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();

    OP_CHECK_IF(dType_ != updatesDtype || dType_ != yDtype,
        OP_LOGE(opName_,
        "The x, updates and y must have the same dtype, please check."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}


// tiling 分发入口
static ge::graphStatus Tiling4MaskedScatterWithPosition(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4MaskedScatterWithPosition running MaskedScatterWithPosition tiling.");
    MaskedScatterWithPositionTiling tiling(context);

    return tiling.DoTiling();
}

static ge::graphStatus TilingParse4MaskedScatterWithPosition(gert::TilingParseContext* context)
{
    OP_LOGI(context->GetNodeName(), "TilingParse4MaskedScatterWithPosition begin.");
    auto compileInfo = context->GetCompiledInfo<MaskedScatterWithPositionCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->totalCoreNum <= 0), OP_LOGE(context, "Tiling4MaskedScatterWithPosition fail to get core num."),
        return ge::GRAPH_FAILED);

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(
        (compileInfo->ubSize <= 0), OP_LOGE(context, "Tiling4MaskedScatterWithPosition fail to get ub size."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(MaskedScatterWithPosition)
    .Tiling(Tiling4MaskedScatterWithPosition)
    .TilingParse<MaskedScatterWithPositionCompileInfo>(TilingParse4MaskedScatterWithPosition);
} // namespace optiling
