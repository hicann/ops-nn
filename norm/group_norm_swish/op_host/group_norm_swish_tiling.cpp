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
 * \file group_norm_swish_tiling.cpp
 * \brief
 */
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "group_norm_swish_tiling.h"
#include "register/op_impl_registry.h"

namespace optiling {
static const uint64_t INPUT_IDX_X = 0;
static const uint64_t INPUT_IDX_GAMMA = 1;
static const uint64_t INPUT_IDX_BETA = 2;
static const uint64_t X_SHAPE_MIN_LEN = 2;
static const uint64_t ATTR_IDX_NUM_GROUPS = 0;
static const uint64_t ATTR_IDX_EPSILON = 2;
static const uint64_t ATTR_IDX_ACTIVATE_SWISH = 3;
static const uint64_t ATTR_IDX_SWISH_SCALE = 4;
static const uint64_t DIM_0 = 0;
static const uint64_t DIM_1 = 1;
static const uint64_t FLOAT32_BYTES = 4;
static const uint64_t FLOAT16_BYTES = 2;
static const int64_t X_TWO_BYTES_TILING_KEY = 100;
static const int64_t X_FOUR_BYTES_TILING_KEY = 200;
static const int64_t X_GAMMA_SAME_TILING_KEY = 10;
static const int64_t X_GAMMA_DIFF_TILING_KEY = 20;
static const int64_t SHAPE_HW1_TILING_KEY = 1;
static const int64_t SHAPE_SMALL_TILING_KEY = 2;
static const int64_t SHAPE_NORM_TILING_KEY = 3;
static const int64_t SHAPE_LARGE_TILING_KEY = 4;
static const int64_t ProcessSize = 8192;
static const int64_t numPerBlock = 8;
static const uint64_t RESERVED_WORKSPACE_SIZE_ATLAS_A2 = 16 * 1024 * 1024;
static const uint64_t RESERVED_WORKSPACE_SIZE_310P = 2 * 1024 * 1024;

inline static int64_t CeilDiv(int64_t value, int64_t factor)
{
    if (factor == 0) {
        return value;
    } else if (value % factor == 0) {
        return value / factor;
    } else {
        return value / factor + 1;
    }
}

inline static int64_t CeilInt(int64_t value, int64_t factor)
{
    return CeilDiv(value, factor) * factor;
}

void PrintTilingData(gert::TilingContext* context, GroupNormSwishTilingData& tilingData)
{
    OP_LOGD(context, ">>>>>>>>>>>>>>> Start to print GroupNormSwish tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(context, "numGroups is %lu.", tilingData.get_numGroups());
    OP_LOGD(context, "epsilon is %f.", tilingData.get_epsilon());
    OP_LOGD(context, "activateSwish is %lu.", tilingData.get_activateSwish());
    OP_LOGD(context, "swishScale is %f.", tilingData.get_swishScale());
    OP_LOGD(context, "hwNum is %lu.", tilingData.get_hwNum());
    OP_LOGD(context, "shapeC is %lu.", tilingData.get_shapeC());
    OP_LOGD(context, "shapeCAlign is %lu.", tilingData.get_shapeCAlign());
    OP_LOGD(context, "shapeD is %lu.", tilingData.get_shapeD());
    OP_LOGD(context, "numPerGroup is %lu.", tilingData.get_numPerGroup());
    OP_LOGD(context, "groupPerCore is %lu.", tilingData.get_groupPerCore());
    OP_LOGD(context, "groupLastCore is %lu.", tilingData.get_groupLastCore());
    OP_LOGD(context, "groupPerCoreAlign is %lu.", tilingData.get_groupPerCoreAlign());
    OP_LOGD(context, "numPerLoop is %lu.", tilingData.get_numPerLoop());
    OP_LOGD(context, "loopTimes is %lu.", tilingData.get_loopTimes());
    OP_LOGD(context, "loopTimesAlign is %lu.", tilingData.get_loopTimesAlign());
    OP_LOGD(context, "numTailLoop is %lu.", tilingData.get_numTailLoop());
    OP_LOGD(context, "usedCoreNum is %u.", context->GetBlockDim());
    OP_LOGD(context, "tilingKey is %lu.", context->GetTilingKey());
    OP_LOGD(context, ">>>>>>>>>>>>>>> Print GroupNormSwish tiling data end <<<<<<<<<<<<<<<<");
}

static ge::graphStatus CheckInputParams(const gert::TilingContext* context)
{
    // check x
    auto inputX = context->GetInputDesc(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto xDtype = inputX->GetDataType();
    OP_CHECK_IF(
        (xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_FLOAT && xDtype != ge::DT_BF16),
        OP_LOGE(context, "xDtype should be FP16/BF16/FP32, please check."),
        return ge::GRAPH_FAILED);
    auto xShapePtr = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    uint64_t xDims = xShape.GetDimNum();
    OP_CHECK_IF(
        (xDims < X_SHAPE_MIN_LEN),
        OP_LOGE(context, "inputDims can't be smaller than 2."),
        return ge::GRAPH_FAILED);
    uint64_t channel = xShape.GetDim(DIM_1);
    // check gamma and beta
    auto gammaShapePtr = context->GetInputShape(INPUT_IDX_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShapePtr);
    auto gammaShape = gammaShapePtr->GetStorageShape();
    uint64_t gammaSizes = gammaShape.GetDim(DIM_0);
    OP_CHECK_IF(
        (gammaShape.GetDimNum() != 1 || gammaSizes != channel),
        OP_LOGE(
            context,
            "The shape of gamma must be"
            " the same as channel, currently is %lu.",
            gammaSizes),
        return ge::GRAPH_FAILED);
    auto betaShapePtr = context->GetInputShape(INPUT_IDX_BETA);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaShapePtr);
    auto betaShape = betaShapePtr->GetStorageShape();
    uint64_t betaSizes = betaShape.GetDim(DIM_0);
    OP_CHECK_IF(
        (betaShape.GetDimNum() != 1 || betaSizes != channel),
        OP_LOGE(
            context,
            "The shape of beta must be"
            " the same as channel, currently is %lu.",
            betaSizes),
        return ge::GRAPH_FAILED);
    auto gammaDtypePtr = context->GetInputDesc(INPUT_IDX_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaDtypePtr);
    auto gammaDtype = gammaDtypePtr->GetDataType();
    auto betaDtypePtr = context->GetInputDesc(INPUT_IDX_BETA);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaDtypePtr);
    auto betaDtype = betaDtypePtr->GetDataType();
    OP_CHECK_IF(
        (gammaDtype != betaDtype),
        OP_LOGE(
            context,
            "The dtype of gamma and beta must"
            " be consistent."),
        return ge::GRAPH_FAILED);
    if (xDtype == ge::DT_FLOAT) {
        OP_CHECK_IF(
            (gammaDtype != xDtype),
            OP_LOGE(
                context,
                "The dtype of x is float32, gamma and beta must"
                " be consistent."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrParams(const gert::TilingContext* context)
{
    auto xShape = context->GetInputShape(INPUT_IDX_X)->GetStorageShape();
    uint64_t channel = xShape.GetDim(DIM_1);
    // check num_groups
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* numGroups = attrs->GetAttrPointer<int64_t>(ATTR_IDX_NUM_GROUPS);
    OP_CHECK_IF(
        (*numGroups <= 0), OP_LOGE(context, "numGroups must be bigger than 0."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (channel % *numGroups != 0),
        OP_LOGE(context, "channel must be integer multiples of numGroups."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static void SetAttrParams(const gert::TilingContext* context, GroupNormSwishTilingData& tilingData)
{
    auto attrs = context->GetAttrs();
    const int64_t* numGroups = attrs->GetAttrPointer<int64_t>(ATTR_IDX_NUM_GROUPS);
    const float* epsilon = attrs->GetAttrPointer<float>(ATTR_IDX_EPSILON);
    const bool* activateSwish = attrs->GetAttrPointer<bool>(ATTR_IDX_ACTIVATE_SWISH);
    const float* swishScale = attrs->GetAttrPointer<float>(ATTR_IDX_SWISH_SCALE);
    tilingData.set_numGroups(*numGroups);
    tilingData.set_epsilon(*epsilon);
    tilingData.set_activateSwish(*activateSwish);
    tilingData.set_swishScale(*swishScale);
}

static void SetInputParams(const gert::TilingContext* context, GroupNormSwishTilingData& tiling)
{
    auto xShape = context->GetInputShape(INPUT_IDX_X)->GetStorageShape();
    uint64_t hwNum = 1;
    uint64_t xDims = xShape.GetDimNum();
    for (uint64_t i = 2; i < xDims; i++) {
        hwNum = hwNum * xShape.GetDim(i);
    }
    tiling.set_shapeC(xShape.GetDim(DIM_1));
    tiling.set_shapeD(tiling.get_shapeC() / tiling.get_numGroups());
    tiling.set_hwNum(hwNum);
    tiling.set_numPerGroup(tiling.get_shapeD() * hwNum);
}

static void SetBlockTiling(gert::TilingContext* context, GroupNormSwishTilingData& tilingData)
{
    auto xShape = context->GetInputShape(INPUT_IDX_X)->GetStorageShape();
    uint64_t shapeN = xShape.GetDim(DIM_0);
    int64_t totalGroup = shapeN * tilingData.get_numGroups();
    int64_t groupPerCore = CeilDiv(totalGroup, tilingData.get_totalCoreNum());
    int64_t groupPerCoreAlign = CeilInt(groupPerCore, numPerBlock);
    uint32_t usedCoreNum = CeilDiv(totalGroup, groupPerCore);
    int64_t groupLastCore = totalGroup - (usedCoreNum - 1) * groupPerCore;

    tilingData.set_groupPerCore(groupPerCore);
    tilingData.set_groupPerCoreAlign(groupPerCoreAlign);
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    tilingData.set_groupLastCore(groupLastCore);
}

static void SetTilingKey(gert::TilingContext* context, GroupNormSwishTilingData& tilingData)
{
    auto xDtype = context->GetInputDesc(INPUT_IDX_X)->GetDataType();
    uint64_t xDtypeSize = ge::GetSizeByDataType(xDtype);
    auto gammaDtype = context->GetInputDesc(INPUT_IDX_GAMMA)->GetDataType();
    uint64_t gammaDtypeSize = ge::GetSizeByDataType(gammaDtype);
    int64_t shapeCAlign = CeilInt(tilingData.get_shapeC(), 32 / gammaDtypeSize);
    tilingData.set_shapeCAlign(shapeCAlign);
    int64_t tilingKey = 0;
    if (xDtypeSize == FLOAT16_BYTES) {
        tilingKey += X_TWO_BYTES_TILING_KEY;
    } else if (xDtypeSize == FLOAT32_BYTES) {
        tilingKey += X_FOUR_BYTES_TILING_KEY;
    }

    if (xDtypeSize == gammaDtypeSize) {
        tilingKey += X_GAMMA_SAME_TILING_KEY;
    } else {
        tilingKey += X_GAMMA_DIFF_TILING_KEY;
    }

    if (tilingData.get_hwNum() == 1) {
        tilingKey += SHAPE_HW1_TILING_KEY;
    } else {
        if (tilingData.get_numPerGroup() <= ProcessSize) {
            int64_t limit = ProcessSize * 2 / xDtypeSize - 8;
            if (tilingData.get_shapeCAlign() + tilingData.get_groupPerCoreAlign() <= limit) {
                tilingKey += SHAPE_SMALL_TILING_KEY;
            } else {
                tilingKey += SHAPE_LARGE_TILING_KEY;
            }
        } else {
            int64_t limit = ProcessSize * 2 / xDtypeSize - tilingData.get_loopTimesAlign() * 2;
            if (tilingData.get_shapeCAlign() + tilingData.get_groupPerCoreAlign() <= limit) {
                tilingKey += SHAPE_NORM_TILING_KEY;
            } else {
                tilingKey += SHAPE_LARGE_TILING_KEY;
            }
        }
    }
    context->SetTilingKey(static_cast<uint64_t>(tilingKey));
}

static void SetUbTiling(GroupNormSwishTilingData& tilingData)
{
    int64_t numPerLoop = 8192;
    tilingData.set_numPerLoop(numPerLoop);
    tilingData.set_loopTimes(CeilDiv(tilingData.get_numPerGroup(), numPerLoop));
    tilingData.set_loopTimesAlign(CeilInt(tilingData.get_loopTimes(), numPerBlock));

    int numTailLoop = tilingData.get_numPerGroup() % numPerLoop;
    numTailLoop = numTailLoop == 0 ? numPerLoop : numTailLoop;
    tilingData.set_numTailLoop(numTailLoop);
}

static ge::graphStatus Tiling4GroupNormSwish(gert::TilingContext* context)
{
    OP_LOGD(context, "Start running TilingPrepare4GroupNormSwish.");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    GroupNormSwishTilingData tilingData;
    tilingData.set_totalCoreNum(ascendcPlatform.GetCoreNumAiv());
    OP_LOGD(context, "Get total core num:%d", tilingData.get_totalCoreNum());
    OP_CHECK_IF(
        (tilingData.get_totalCoreNum() <= 0),
        OP_LOGE(context, "Failed to get core num."), return ge::GRAPH_FAILED);
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    OP_LOGD(context, "Get total ub size:%lu", static_cast<int64_t>(ubSizePlatForm));
    OP_LOGD(context, "TilingPrepare4GroupNormSwish ends.");

    OP_LOGD(context, "Start running Tiling4GroupNormSwish.");
    // check input && attrs params
    OP_CHECK_IF(
        (CheckInputParams(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "InputParams is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (CheckAttrParams(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "AttrParams is invalid."), return ge::GRAPH_FAILED);
    // set TilingData

    SetAttrParams(context, tilingData);
    SetInputParams(context, tilingData);
    SetBlockTiling(context, tilingData);
    SetUbTiling(tilingData);
    SetTilingKey(context, tilingData);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    size_t sysWorkspaceSize = RESERVED_WORKSPACE_SIZE_ATLAS_A2;
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
        sysWorkspaceSize = RESERVED_WORKSPACE_SIZE_310P;
    }
    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = sysWorkspaceSize;

    PrintTilingData(context, tilingData);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4GroupNormSwish(gert::TilingParseContext* context)
{
    if(context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupNormSwish)
    .Tiling(Tiling4GroupNormSwish)
    .TilingParse<GroupNormSwishCompileInfo>(TilingPrepare4GroupNormSwish);
} // namespace optiling
