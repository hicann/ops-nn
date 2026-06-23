/**
 * Copyright (c) 2026 Huawei Technologies
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_group_quant_tiling_utils.h
 * \brief Tiling utility functions for SwiGLU Group Quant operator
 */

#ifndef SWIGLU_GROUP_QUANT_TILING_UTILS_H
#define SWIGLU_GROUP_QUANT_TILING_UTILS_H

#include "swiglu_group_quant_tiling.h"

namespace optiling {

// Input/Output indices
constexpr uint32_t SWIGLU_GROUP_QUANT_INPUT_X_INDEX = 0;
constexpr uint32_t INPUT_WEIGHT_INDEX = 1;
constexpr uint32_t INPUT_GROUP_INDEX_INDEX = 2;
constexpr uint32_t INPUT_SCALE_INDEX = 3;
constexpr uint32_t SWIGLU_GROUP_QUANT_OUTPUT_Y_INDEX = 0;
constexpr uint32_t OUTPUT_Y_SCALE_INDEX = 1;
constexpr uint32_t OUTPUT_Y_ORIGIN_INDEX = 2;

// Attribute indices
constexpr uint32_t ATTR_DST_TYPE_INDEX = 0;
constexpr uint32_t ATTR_QUANT_MODE_INDEX = 1;
constexpr uint32_t ATTR_BLOCK_SIZE_INDEX = 2;
constexpr uint32_t ATTR_ROUND_SCALE_INDEX = 3;
constexpr uint32_t ATTR_CLAMP_LIMIT_INDEX = 4;
constexpr uint32_t ATTR_DST_TYPE_MAX_FINITE_INDEX = 5;
constexpr uint32_t ATTR_OUTPUT_ORIGIN_INDEX = 6;

// Constants
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t UB_RESERVE = 1024;
constexpr uint32_t DB_BUFFER = 2;
constexpr uint32_t SWI_FACTOR = 2;
constexpr uint32_t SIZE_OF_FLOAT = 4;

constexpr float CLAMP_LIMIT_DEFAULT = 0.0f;
constexpr float DST_TYPE_MAX_FINITE_DEFAULT = 448.0f;

constexpr uint32_t SWIGLU_GROUP_QUANT_MODE_DYNAMIC = 3;

constexpr uint32_t SWIGLU_GROUP_QUANT_ZERO = 0;
constexpr uint32_t SWIGLU_GROUP_QUANT_ONE = 1;
constexpr uint32_t SWIGLU_GROUP_QUANT_TWO = 2;

// Supported dtypes
static const std::set<ge::DataType> SUPPORT_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

// Template utility functions
template <typename T>
inline auto SwigluGroupQuantAlignUp(T num, T div) -> decltype(num)
{
    return (div == 0) ? 0 : (num + div - 1) / div * div;
}

template <typename T>
inline auto SwigluGroupQuantAlignDown(T num, T div) -> decltype(num)
{
    return (div == 0) ? 0 : num / div * div;
}

template <typename T>
inline auto SwigluGroupQuantCeilDiv(T num, T div) -> decltype(num)
{
    return div == 0 ? 0 : (num + div - 1) / div;
}

// Check input dtype
inline ge::graphStatus CheckSwigluGroupQuantInputDtype(const gert::TilingContext *context)
{
    auto xDtype = context->GetInputDesc(SWIGLU_GROUP_QUANT_INPUT_X_INDEX)->GetDataType();
    if (xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16 && xDtype != ge::DT_FLOAT) {
        OP_LOGE(context->GetNodeName(), "input x dtype is only support fp16/bf16/fp32.");
        return ge::GRAPH_FAILED;
    }

    auto weightDesc = context->GetOptionalInputDesc(INPUT_WEIGHT_INDEX);
    if (weightDesc != nullptr) {
        auto weightDtype = weightDesc->GetDataType();
        if (weightDtype != ge::DataType::DT_FLOAT) {
            OP_LOGE(context->GetNodeName(), "input weight dtype is only support fp32.");
            return ge::GRAPH_FAILED;
        }
    }

    auto groupDesc = context->GetOptionalInputDesc(INPUT_GROUP_INDEX_INDEX);
    if (groupDesc != nullptr) {
        auto groupDtype = groupDesc->GetDataType();
        if (groupDtype != ge::DataType::DT_INT64) {
            OP_LOGE(context->GetNodeName(), "group_index dtype is only support int64.");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

// Check output dtype
inline ge::graphStatus CheckSwigluGroupQuantOutputDtype(const gert::TilingContext *context)
{
    auto yDtype = context->GetOutputDesc(SWIGLU_GROUP_QUANT_OUTPUT_Y_INDEX)->GetDataType();
    if (yDtype != ge::DataType::DT_HIFLOAT8) {
        OP_LOGE(context->GetNodeName(), "y dtype must be hifloat8.");
        return ge::GRAPH_FAILED;
    }

    auto yScaleDtype = context->GetOutputDesc(OUTPUT_Y_SCALE_INDEX)->GetDataType();
    if (yScaleDtype != ge::DataType::DT_FLOAT) {
        OP_LOGE(context->GetNodeName(), "y_scale dtype must be fp32.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// Check attributes
inline ge::graphStatus CheckSwigluGroupQuantAttrs(const gert::TilingContext *context, bool &hasClamp, bool &outputOrigin,
    float &clampLimit, float &dstTypeMaxFinite)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    auto quantModePtr = attrs->GetInt(ATTR_QUANT_MODE_INDEX);
    if (quantModePtr != nullptr) {
        uint32_t quantMode = static_cast<uint32_t>(*quantModePtr);
        if (quantMode != SWIGLU_GROUP_QUANT_MODE_DYNAMIC) {
            OP_LOGE(context->GetNodeName(), "quant_mode must be %u (HiF8 Dynamic Quant), but got %u.",
                SWIGLU_GROUP_QUANT_MODE_DYNAMIC, quantMode);
            return ge::GRAPH_FAILED;
        }
    }

    auto clampLimitPtr = attrs->GetFloat(ATTR_CLAMP_LIMIT_INDEX);
    clampLimit = (clampLimitPtr != nullptr) ? *clampLimitPtr : CLAMP_LIMIT_DEFAULT;
    if (clampLimit < 0.0f) {
        OP_LOGE(context->GetNodeName(), "clamp_limit must be >= 0.0, but got %f.", clampLimit);
        return ge::GRAPH_FAILED;
    }
    hasClamp = (clampLimit != 0.0f);

    auto dstTypeMaxFinitePtr = attrs->GetFloat(ATTR_DST_TYPE_MAX_FINITE_INDEX);
    dstTypeMaxFinite = (dstTypeMaxFinitePtr != nullptr) ? *dstTypeMaxFinitePtr : DST_TYPE_MAX_FINITE_DEFAULT;
    if (dstTypeMaxFinite <= 0.0f) {
        OP_LOGE(context->GetNodeName(), "dst_type_max_finite must be > 0, but got %f.", dstTypeMaxFinite);
        return ge::GRAPH_FAILED;
    }

    auto outputOriginPtr = attrs->GetBool(ATTR_OUTPUT_ORIGIN_INDEX);
    outputOrigin = (outputOriginPtr != nullptr) ? *outputOriginPtr : false;

    return ge::GRAPH_SUCCESS;
}

// Check group_index input (optional)
inline ge::graphStatus CheckGroupIndex(const gert::TilingContext *context, uint32_t totalTokens,
    bool &isGroup, uint32_t &groupNum, std::vector<int64_t> &groupTokens)
{
    auto groupIndexShape = context->GetOptionalInputShape(INPUT_GROUP_INDEX_INDEX);
    if (groupIndexShape == nullptr) {
        isGroup = false;
        groupNum = 0;
        groupTokens.clear();
        return ge::GRAPH_SUCCESS;
    }
    isGroup = true;
    size_t giDimNum = groupIndexShape->GetStorageShape().GetDimNum();
    if (giDimNum != SWIGLU_GROUP_QUANT_ONE) {
        OP_LOGE(context->GetNodeName(), "group_index must be 1D, but got %zu dims.", giDimNum);
        return ge::GRAPH_FAILED;
    }
    groupNum = groupIndexShape->GetStorageShape().GetDim(0);
    if (groupNum <= SWIGLU_GROUP_QUANT_ZERO) {
        OP_LOGE(context->GetNodeName(), "group_index length must be > 0, but got %u.", groupNum);
        return ge::GRAPH_FAILED;
    }

    auto groupIndexTensor = context->GetOptionalInputTensor(INPUT_GROUP_INDEX_INDEX);
    if (groupIndexTensor == nullptr) {
        OP_LOGE(context->GetNodeName(), "group_index tensor is null in group mode, need ValueDepend(OPTIONAL).");
        return ge::GRAPH_FAILED;
    }
    auto groupIndexData = groupIndexTensor->GetData<int64_t>();
    if (groupIndexData == nullptr) {
        OP_LOGE(context->GetNodeName(), "group_index tensor data is null, need ValueDepend(OPTIONAL).");
        return ge::GRAPH_FAILED;
    }
    groupTokens.resize(groupNum);
    uint32_t totalGroupTokens = 0;
    for (uint32_t g = 0; g < groupNum; g++) {
        groupTokens[g] = groupIndexData[g];
        if (groupTokens[g] < 0) {
            OP_LOGE(context->GetNodeName(), "group_index[%u] must be >= 0, but got %ld.", g, groupTokens[g]);
            return ge::GRAPH_FAILED;
        }
        totalGroupTokens += static_cast<uint32_t>(groupTokens[g]);
    }
    if (totalGroupTokens > totalTokens) {
        OP_LOGE(context->GetNodeName(), "sum of group_index tokens [%u] must <= x total tokens [%u].",
            totalGroupTokens, totalTokens);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// Check input shape and get shape info
inline ge::graphStatus CheckInputShape(const gert::TilingContext *context, uint32_t &totalTokens, uint32_t &dim2H,
    uint32_t &dimH, bool &hasWeight, bool &isGroup, uint32_t &groupNum, std::vector<int64_t> &groupTokens)
{
    auto xShape = context->GetInputShape(SWIGLU_GROUP_QUANT_INPUT_X_INDEX);
    if (xShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "x shape is null.");
        return ge::GRAPH_FAILED;
    }
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    if (xDimNum < SWIGLU_GROUP_QUANT_ONE) {
        OP_LOGE(context->GetNodeName(), "x dims must be >= 1, but got %zu.", xDimNum);
        return ge::GRAPH_FAILED;
    }
    totalTokens = 1;
    for (size_t i = 0; i < xDimNum - 1; i++) {
        totalTokens *= xShape->GetStorageShape().GetDim(i);
    }
    dim2H = xShape->GetStorageShape().GetDim(xDimNum - 1);
    if (dim2H % SWI_FACTOR != 0) {
        OP_LOGE(context->GetNodeName(), "x last dim must be even, but got %u.", dim2H);
        return ge::GRAPH_FAILED;
    }
    dimH = dim2H / SWI_FACTOR;
    hasWeight = (context->GetOptionalInputShape(INPUT_WEIGHT_INDEX) != nullptr);
    return CheckGroupIndex(context, totalTokens, isGroup, groupNum, groupTokens);
}

// Check y output shape
inline ge::graphStatus CheckYShape(const gert::TilingContext *context, uint32_t xDimNum, uint32_t dimH)
{
    auto xShape = context->GetInputShape(SWIGLU_GROUP_QUANT_INPUT_X_INDEX);
    auto yShape = context->GetOutputShape(SWIGLU_GROUP_QUANT_OUTPUT_Y_INDEX);
    if (yShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "y shape is null.");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &yShapeStorage = yShape->GetStorageShape();
    if (yShapeStorage.GetDimNum() != xDimNum) {
        OP_LOGE(context->GetNodeName(), "y dim num [%zu] must equal x dim num [%u].",
            yShapeStorage.GetDimNum(), xDimNum);
        return ge::GRAPH_FAILED;
    }
    for (uint32_t i = 0; i < xDimNum - 1; i++) {
        if (yShapeStorage.GetDim(i) != xShape->GetStorageShape().GetDim(i)) {
            OP_LOGE(context->GetNodeName(), "y dim[%u] [%ld] must equal x dim[%u] [%ld].",
                i, yShapeStorage.GetDim(i), i, xShape->GetStorageShape().GetDim(i));
            return ge::GRAPH_FAILED;
        }
    }
    if (yShapeStorage.GetDim(xDimNum - 1) != dimH) {
        OP_LOGE(context->GetNodeName(), "y last dim [%ld] must equal dimH [%u].",
            yShapeStorage.GetDim(xDimNum - 1), dimH);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// Check y_scale output shape
inline ge::graphStatus CheckYScaleShape(const gert::TilingContext *context, bool isGroup, uint32_t groupNum)
{
    auto yScaleShape = context->GetOutputShape(OUTPUT_Y_SCALE_INDEX);
    if (yScaleShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "y_scale shape is null.");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &yScaleShapeStorage = yScaleShape->GetStorageShape();
    if (yScaleShapeStorage.GetDimNum() != SWIGLU_GROUP_QUANT_ONE) {
        OP_LOGE(context->GetNodeName(), "y_scale must be 1D, but got %zu dims.", yScaleShapeStorage.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    if (isGroup) {
        if (yScaleShapeStorage.GetDim(0) != groupNum) {
            OP_LOGE(context->GetNodeName(), "y_scale dim[0] [%ld] must equal groupNum [%u] in group mode.",
                yScaleShapeStorage.GetDim(0), groupNum);
            return ge::GRAPH_FAILED;
        }
    } else {
        if (yScaleShapeStorage.GetDim(0) != SWIGLU_GROUP_QUANT_ONE) {
            OP_LOGE(context->GetNodeName(), "y_scale dim[0] [%ld] must be 1 in non-group mode.",
                yScaleShapeStorage.GetDim(0));
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// Check y_origin output shape (optional)
inline ge::graphStatus CheckYOriginShape(const gert::TilingContext *context, bool outputOrigin, uint32_t xDimNum,
    uint32_t dimH, ge::DataType xDtype)
{
    if (!outputOrigin) {
        return ge::GRAPH_SUCCESS;
    }
    auto xShape = context->GetInputShape(SWIGLU_GROUP_QUANT_INPUT_X_INDEX);
    auto yOriginShape = context->GetOutputShape(OUTPUT_Y_ORIGIN_INDEX);
    if (yOriginShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "y_origin shape is null when output_origin=true.");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &yOriginShapeStorage = yOriginShape->GetStorageShape();
    if (yOriginShapeStorage.GetDimNum() != xDimNum) {
        OP_LOGE(context->GetNodeName(), "y_origin dim num [%zu] must equal x dim num [%u].",
            yOriginShapeStorage.GetDimNum(), xDimNum);
        return ge::GRAPH_FAILED;
    }
    for (uint32_t i = 0; i < xDimNum - 1; i++) {
        if (yOriginShapeStorage.GetDim(i) != xShape->GetStorageShape().GetDim(i)) {
            OP_LOGE(context->GetNodeName(), "y_origin dim[%u] [%ld] must equal x dim[%u] [%ld].",
                i, yOriginShapeStorage.GetDim(i), i, xShape->GetStorageShape().GetDim(i));
            return ge::GRAPH_FAILED;
        }
    }
    if (yOriginShapeStorage.GetDim(xDimNum - 1) != dimH) {
        OP_LOGE(context->GetNodeName(), "y_origin last dim [%ld] must equal dimH [%u].",
            yOriginShapeStorage.GetDim(xDimNum - 1), dimH);
        return ge::GRAPH_FAILED;
    }
    auto yOriginDesc = context->GetOutputDesc(OUTPUT_Y_ORIGIN_INDEX);
    if (yOriginDesc != nullptr && yOriginDesc->GetDataType() != xDtype) {
        OP_LOGE(context->GetNodeName(), "y_origin dtype must equal x dtype.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// Check output shape
inline ge::graphStatus CheckOutputShape(const gert::TilingContext *context, uint32_t xDimNum, uint32_t dimH,
    bool isGroup, uint32_t groupNum, bool outputOrigin, ge::DataType xDtype)
{
    if (CheckYShape(context, xDimNum, dimH) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckYScaleShape(context, isGroup, groupNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckYOriginShape(context, outputOrigin, xDimNum, dimH, xDtype) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// Check all parameters
inline ge::graphStatus CheckSwigluGroupQuantOpParams(const gert::TilingContext *context, uint32_t &totalTokens, uint32_t &dim2H,
    uint32_t &dimH, bool &hasWeight, bool &isGroup, uint32_t &groupNum, std::vector<int64_t> &groupTokens,
    bool &hasClamp, bool &outputOrigin, float &clampLimit, float &dstTypeMaxFinite)
{
    if (CheckSwigluGroupQuantInputDtype(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "input dtype check failed.");
        return ge::GRAPH_FAILED;
    }
    if (CheckSwigluGroupQuantOutputDtype(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "output dtype check failed.");
        return ge::GRAPH_FAILED;
    }
    if (CheckSwigluGroupQuantAttrs(context, hasClamp, outputOrigin, clampLimit, dstTypeMaxFinite) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "attrs check failed.");
        return ge::GRAPH_FAILED;
    }
    if (CheckInputShape(context, totalTokens, dim2H, dimH, hasWeight, isGroup, groupNum,
                        groupTokens) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "input shape check failed.");
        return ge::GRAPH_FAILED;
    }

    auto xShape = context->GetInputShape(SWIGLU_GROUP_QUANT_INPUT_X_INDEX);
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    auto xDtype = context->GetInputDesc(SWIGLU_GROUP_QUANT_INPUT_X_INDEX)->GetDataType();
    if (CheckOutputShape(context, xDimNum, dimH, isGroup, groupNum, outputOrigin, xDtype) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "output shape check failed.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// Calculate tile tokens based on UB size
inline void CalculateTileTokens(uint32_t ubSize, uint32_t dimH, bool hasWeight, bool outputOrigin,
    SwigluGroupQuantTilingParam &tilingParam)
{
    uint32_t ubAvailable = ubSize - UB_RESERVE;
    uint32_t ubFactor = 7;
    uint32_t ubPerToken = ubFactor * dimH * SIZE_OF_FLOAT;
    if (hasWeight) {
        ubPerToken += DB_BUFFER * SIZE_OF_FLOAT;
    }

    tilingParam.tileTokens = ubAvailable / ubPerToken;
    if (tilingParam.tileTokens < SWIGLU_GROUP_QUANT_ONE) {
        tilingParam.tileTokens = SWIGLU_GROUP_QUANT_ONE;
    }
}

// Find the lightest core (min token load) to handle group-mode tail [groupTokensSum, totalTokens)
inline void CalcMinLoadCoreIdx(const std::vector<int64_t> &groupTokens, SwigluGroupQuantTilingParam &tilingParam)
{
    uint32_t minLoad = ~static_cast<uint32_t>(0);
    uint32_t minCoreIdx = SWIGLU_GROUP_QUANT_ZERO;
    for (uint32_t core = SWIGLU_GROUP_QUANT_ZERO; core < tilingParam.usedCoreNum; core++) {
        uint32_t coreLoad = SWIGLU_GROUP_QUANT_ZERO;
        for (uint32_t gg = tilingParam.coreGroupStartArr[core];
             gg < tilingParam.coreGroupStartArr[core] + tilingParam.coreGroupCountArr[core]; gg++) {
            coreLoad += static_cast<uint32_t>(groupTokens[gg]);
        }
        if (coreLoad < minLoad) {
            minLoad = coreLoad;
            minCoreIdx = core;
        }
    }
    tilingParam.minLoadCoreIdx = minCoreIdx;
}

// Continuous segment split for load balancing: each core handles a contiguous
// group range [start, start+count). Greedily accumulates groups per core toward
// the target load so the kernel can compute tokenStart by summing group_index.
inline void ContinuousGroupSplit(uint32_t totalCore, uint32_t groupNum,
    const std::vector<int64_t> &groupTokens, SwigluGroupQuantTilingParam &tilingParam)
{
    tilingParam.usedCoreNum = std::min({totalCore, groupNum, static_cast<uint32_t>(MAX_CORE_COUNT)});
    for (uint32_t i = SWIGLU_GROUP_QUANT_ZERO; i < MAX_CORE_COUNT; i++) {
        tilingParam.coreGroupStartArr[i] = SWIGLU_GROUP_QUANT_ZERO;
        tilingParam.coreGroupCountArr[i] = SWIGLU_GROUP_QUANT_ZERO;
    }
    if (tilingParam.usedCoreNum == SWIGLU_GROUP_QUANT_ZERO || groupNum == SWIGLU_GROUP_QUANT_ZERO) {
        return;
    }

    uint32_t totalGroupTokens = SWIGLU_GROUP_QUANT_ZERO;
    for (uint32_t g = SWIGLU_GROUP_QUANT_ZERO; g < groupNum; g++) {
        totalGroupTokens += static_cast<uint32_t>(groupTokens[g]);
    }
    uint32_t targetLoad = SwigluGroupQuantCeilDiv(totalGroupTokens, tilingParam.usedCoreNum);

    uint32_t g = SWIGLU_GROUP_QUANT_ZERO;
    for (uint32_t core = SWIGLU_GROUP_QUANT_ZERO; core < tilingParam.usedCoreNum; core++) {
        uint32_t start = g;
        uint32_t remainingCores = tilingParam.usedCoreNum - core - SWIGLU_GROUP_QUANT_ONE;
        if (core == tilingParam.usedCoreNum - SWIGLU_GROUP_QUANT_ONE) {
            // last core takes all remaining groups
            g = groupNum;
        } else {
            uint32_t coreLoad = SWIGLU_GROUP_QUANT_ZERO;
            while (g < groupNum) {
                uint32_t remainingGroups = groupNum - g;
                // reserve at least one group for each remaining core
                if (remainingGroups <= remainingCores) {
                    break;
                }
                coreLoad += static_cast<uint32_t>(groupTokens[g]);
                g++;
                if (coreLoad >= targetLoad) {
                    break;
                }
            }
        }
        tilingParam.coreGroupStartArr[core] = start;
        tilingParam.coreGroupCountArr[core] = g - start;
    }

    tilingParam.groupTokensSum = totalGroupTokens;
    CalcMinLoadCoreIdx(groupTokens, tilingParam);
}

// Calculate core distribution
inline void CalculateCoreDistribution(uint32_t totalCore, uint32_t totalTokens, bool isGroup, bool outputOrigin,
    uint32_t groupNum, const std::vector<int64_t> &groupTokens, SwigluGroupQuantTilingParam &tilingParam)
{
    if (!isGroup) {
        // Non-group mode: evenly distribute tokens across cores
        tilingParam.usedCoreNum = std::min(totalCore, totalTokens);
        tilingParam.tokensPerCore = SwigluGroupQuantCeilDiv(totalTokens, tilingParam.usedCoreNum);
        for (uint32_t i = SWIGLU_GROUP_QUANT_ZERO; i < MAX_CORE_COUNT; i++) {
            tilingParam.coreGroupStartArr[i] = SWIGLU_GROUP_QUANT_ZERO;
            tilingParam.coreGroupCountArr[i] = SWIGLU_GROUP_QUANT_ZERO;
        }
        tilingParam.groupTokensSum = totalTokens;
        tilingParam.minLoadCoreIdx = SWIGLU_GROUP_QUANT_ZERO;
    } else {
        // Group mode: continuous segment split for per-core group assignment
        ContinuousGroupSplit(totalCore, groupNum, groupTokens, tilingParam);
    }
}

// Set tiling data
inline void SetSwigluGroupQuantTilingData(const SwigluGroupQuantTilingParam &tilingParam, uint32_t totalTokens,
    uint32_t dim2H, uint32_t dimH, bool hasWeight, bool isGroup, bool hasClamp, bool outputOrigin,
    float clampLimit, float dstTypeMaxFinite, SwigluGroupQuantTilingData &tilingData)
{
    tilingData.set_totalTokens(totalTokens);
    tilingData.set_dim2H(dim2H);
    tilingData.set_dimH(dimH);
    tilingData.set_isGroup(isGroup ? SWIGLU_GROUP_QUANT_ONE : SWIGLU_GROUP_QUANT_ZERO);
    tilingData.set_hasWeight(hasWeight ? SWIGLU_GROUP_QUANT_ONE : SWIGLU_GROUP_QUANT_ZERO);
    tilingData.set_hasClamp(hasClamp ? SWIGLU_GROUP_QUANT_ONE : SWIGLU_GROUP_QUANT_ZERO);
    tilingData.set_outputOrigin(outputOrigin ? SWIGLU_GROUP_QUANT_ONE : SWIGLU_GROUP_QUANT_ZERO);
    tilingData.set_clampLimit(clampLimit);
    tilingData.set_dstTypeMaxFinite(dstTypeMaxFinite);
    tilingData.set_tileTokens(tilingParam.tileTokens);
    tilingData.set_usedCoreNum(tilingParam.usedCoreNum);
    tilingData.set_tokensPerCore(tilingParam.tokensPerCore);
    tilingData.set_groupTokensSum(tilingParam.groupTokensSum);
    tilingData.set_minLoadCoreIdx(tilingParam.minLoadCoreIdx);
    uint32_t coreGroupStartArrCopy[MAX_CORE_COUNT] = {0};
    uint32_t coreGroupCountArrCopy[MAX_CORE_COUNT] = {0};
    for (uint32_t i = SWIGLU_GROUP_QUANT_ZERO; i < MAX_CORE_COUNT; i++) {
        coreGroupStartArrCopy[i] = tilingParam.coreGroupStartArr[i];
        coreGroupCountArrCopy[i] = tilingParam.coreGroupCountArr[i];
    }
    tilingData.set_coreGroupStartArr(coreGroupStartArrCopy);
    tilingData.set_coreGroupCountArr(coreGroupCountArrCopy);
}

} // namespace optiling

#endif // SWIGLU_GROUP_QUANT_TILING_UTILS_H
