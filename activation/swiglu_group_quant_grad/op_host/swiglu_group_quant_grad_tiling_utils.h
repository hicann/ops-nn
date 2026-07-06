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
 * \file swiglu_group_quant_grad_tiling_utils.h
 * \brief SwiGLU Group Dynamic Quant Backward tiling utility functions
 */

#ifndef SWIGLU_GROUP_QUANT_GRAD_TILING_UTILS_H
#define SWIGLU_GROUP_QUANT_GRAD_TILING_UTILS_H

#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "swiglu_group_quant_grad_tiling.h"

namespace optiling {

constexpr uint32_t INPUT_GRAD_Y_INDEX = 0;
constexpr uint32_t INPUT_OP_X_INDEX = 1;
constexpr uint32_t INPUT_WEIGHT_INDEX = 2;
constexpr uint32_t INPUT_Y_ORIGIN_INDEX = 3;
constexpr uint32_t INPUT_GROUP_INDEX_INDEX = 4;
constexpr uint32_t OUTPUT_GRAD_X_INDEX = 0;
constexpr uint32_t OUTPUT_GRAD_WEIGHT_INDEX = 1;

constexpr uint32_t TMP_DATA_UB_SIZE = 8 * 1024;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t ALIGN_EIGHT = 8;
constexpr uint32_t FP16_BFP16_32B_ALIGN_NUM = 16;
constexpr uint32_t ZERO = 0;
constexpr uint32_t ONE = 1;
constexpr uint32_t TWO = 2;

constexpr uint32_t UB_BASE_FACTOR = 12;
constexpr uint32_t UB_CLAMP_EXTRA_FACTOR = 8;
constexpr uint32_t UB_WEIGHT_EXTRA_FACTOR = 4;
constexpr uint32_t UB_WEIGHT_EXTRA_TOKENS = 16;

constexpr uint32_t MAX_H = 4096;
constexpr uint32_t MIN_H = 512;

template <typename T>
inline auto AlignUpT(T num, T div) -> decltype(num)
{
    return (div == 0) ? 0 : (num + div - 1) / div * div;
}

template <typename T>
inline auto AlignDownT(T num, T div) -> decltype(num)
{
    return (div == 0) ? 0 : num / div * div;
}

template <typename T>
inline auto CeilDivT(T num, T div) -> decltype(num)
{
    return div == 0 ? 0 : (num + div - 1) / div;
}

inline ge::graphStatus CheckAllInputDtype(const gert::TilingContext* context)
{
    auto gradYDtype = context->GetInputDesc(INPUT_GRAD_Y_INDEX)->GetDataType();
    auto xDtype = context->GetInputDesc(INPUT_OP_X_INDEX)->GetDataType();

    if (gradYDtype != ge::DT_FLOAT16 && gradYDtype != ge::DT_BF16 && gradYDtype != ge::DT_FLOAT) {
        OP_LOGE(context->GetNodeName(), "input grad_y dtype is only support fp16/bf16/fp32.");
        return ge::GRAPH_FAILED;
    }

    if (xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16 && xDtype != ge::DT_FLOAT) {
        OP_LOGE(context->GetNodeName(), "input x dtype is only support fp16/bf16/fp32.");
        return ge::GRAPH_FAILED;
    }

    if (gradYDtype != xDtype) {
        OP_LOGE(context->GetNodeName(), "input grad_y and x dtype must be same.");
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

    auto yOriginDesc = context->GetOptionalInputDesc(INPUT_Y_ORIGIN_INDEX);
    if (yOriginDesc != nullptr) {
        auto yOriginDtype = yOriginDesc->GetDataType();
        if (yOriginDtype != ge::DT_FLOAT16 && yOriginDtype != ge::DT_BF16 && yOriginDtype != ge::DT_FLOAT) {
            OP_LOGE(context->GetNodeName(), "input y_origin dtype is only support fp16/bf16/fp32.");
            return ge::GRAPH_FAILED;
        }
    }

    auto groupIndexDesc = context->GetOptionalInputDesc(INPUT_GROUP_INDEX_INDEX);
    if (groupIndexDesc != nullptr) {
        auto groupIndexDtype = groupIndexDesc->GetDataType();
        if (groupIndexDtype != ge::DataType::DT_INT64) {
            OP_LOGE(context->GetNodeName(), "input group_index dtype is only support int64.");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckAllOutputDtype(const gert::TilingContext* context)
{
    auto gradXDtype = context->GetOutputDesc(OUTPUT_GRAD_X_INDEX)->GetDataType();
    auto xDtype = context->GetInputDesc(INPUT_OP_X_INDEX)->GetDataType();

    if (gradXDtype != xDtype) {
        OP_LOGE(context->GetNodeName(), "output grad_x dtype must be same as input x.");
        return ge::GRAPH_FAILED;
    }

    auto gradWeightDesc = context->GetOutputDesc(OUTPUT_GRAD_WEIGHT_INDEX);
    if (gradWeightDesc != nullptr) {
        auto gradWeightDtype = gradWeightDesc->GetDataType();
        if (gradWeightDtype != ge::DataType::DT_FLOAT) {
            OP_LOGE(context->GetNodeName(), "output grad_weight dtype is only support fp32.");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckAllAttrs(const gert::TilingContext* context, SwigluGroupQuantGradCompileInfo& compileInfo)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        auto clampLimitPtr = attrs->GetFloat(0);
        if (clampLimitPtr != nullptr) {
            compileInfo.clampLimit = *clampLimitPtr;
            if (compileInfo.clampLimit != -1.0f) {
                if (!(compileInfo.clampLimit > 0.0f)) {
                    OP_LOGE(context->GetNodeName(), "attr clamp_limit must be -1 or > 0.0, got %f.",
                            compileInfo.clampLimit);
                    return ge::GRAPH_FAILED;
                }
                compileInfo.hasClampLimit = 1;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckGradYAndXShapeDim(const gert::TilingContext* context)
{
    auto gradYShape = context->GetInputShape(INPUT_GRAD_Y_INDEX);
    auto xShape = context->GetInputShape(INPUT_OP_X_INDEX);

    size_t gradYDimNum = gradYShape->GetStorageShape().GetDimNum();
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();

    if (gradYDimNum < ONE || xDimNum < ONE) {
        OP_LOGE(context->GetNodeName(), "Input shape dimension must >= 1.");
        return ge::GRAPH_FAILED;
    }

    if (gradYDimNum != xDimNum) {
        OP_LOGE(context->GetNodeName(), "grad_y and x shape dimension must be same.");
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < xDimNum; i++) {
        if (i < xDimNum - 1) {
            if (gradYShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(i)) {
                OP_LOGE(context->GetNodeName(), "grad_y and x shape must be same except last dim.");
                return ge::GRAPH_FAILED;
            }
        } else {
            int64_t gradYDimLast = gradYShape->GetStorageShape().GetDim(i);
            int64_t xDimLast = xShape->GetStorageShape().GetDim(i);

            if (xDimLast % TWO != 0) {
                OP_LOGE(context->GetNodeName(), "Input x last dim must be divisible by 2.");
                return ge::GRAPH_FAILED;
            }

            if (gradYDimLast != xDimLast / TWO) {
                OP_LOGE(context->GetNodeName(), "Input grad_y last dim must be half of x last dim.");
                return ge::GRAPH_FAILED;
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckWeightShapeDim(const gert::TilingContext* context,
                                           SwigluGroupQuantGradCompileInfo& compileInfo)
{
    auto weightShape = context->GetOptionalInputShape(INPUT_WEIGHT_INDEX);
    if (weightShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    compileInfo.hasWeight = 1;
    auto gradYShape = context->GetInputShape(INPUT_GRAD_Y_INDEX);
    size_t gradYDimNum = gradYShape->GetStorageShape().GetDimNum();
    size_t weightDimNum = weightShape->GetStorageShape().GetDimNum();

    if (weightDimNum != gradYDimNum) {
        OP_LOGE(context->GetNodeName(), "weight and grad_y shape dimension must be same.");
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < gradYDimNum; i++) {
        if (i < gradYDimNum - 1) {
            if (weightShape->GetStorageShape().GetDim(i) != gradYShape->GetStorageShape().GetDim(i)) {
                OP_LOGE(context->GetNodeName(), "weight and grad_y shape must be same except last dim.");
                return ge::GRAPH_FAILED;
            }
        } else {
            int64_t weightDimLast = weightShape->GetStorageShape().GetDim(i);
            if (weightDimLast != ONE) {
                OP_LOGE(context->GetNodeName(), "weight last dim must be 1.");
                return ge::GRAPH_FAILED;
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckYOriginShapeDim(const gert::TilingContext* context,
                                            SwigluGroupQuantGradCompileInfo& compileInfo)
{
    auto yOriginShape = context->GetOptionalInputShape(INPUT_Y_ORIGIN_INDEX);
    if (yOriginShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    compileInfo.hasYOrigin = 1;
    auto gradYShape = context->GetInputShape(INPUT_GRAD_Y_INDEX);
    size_t gradYDimNum = gradYShape->GetStorageShape().GetDimNum();
    size_t yOriginDimNum = yOriginShape->GetStorageShape().GetDimNum();

    if (yOriginDimNum != gradYDimNum) {
        OP_LOGE(context->GetNodeName(), "y_origin and grad_y shape dimension must be same.");
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < gradYDimNum; i++) {
        if (yOriginShape->GetStorageShape().GetDim(i) != gradYShape->GetStorageShape().GetDim(i)) {
            OP_LOGE(context->GetNodeName(), "y_origin shape must be same as grad_y.");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckGroupIndexShapeDim(const gert::TilingContext* context,
                                               SwigluGroupQuantGradCompileInfo& compileInfo)
{
    auto groupIndexShape = context->GetOptionalInputShape(INPUT_GROUP_INDEX_INDEX);
    if (groupIndexShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    compileInfo.hasGroupIndex = 1;
    size_t groupIndexDimNum = groupIndexShape->GetStorageShape().GetDimNum();
    if (groupIndexDimNum != ONE) {
        OP_LOGE(context->GetNodeName(), "group_index must be 1D tensor.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckInputShape(const gert::TilingContext* context, SwigluGroupQuantGradCompileInfo& compileInfo)
{
    if (CheckGradYAndXShapeDim(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckWeightShapeDim(context, compileInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckYOriginShapeDim(context, compileInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (compileInfo.hasWeight && !compileInfo.hasYOrigin) {
        OP_LOGE(context->GetNodeName(), "When weight exists, y_origin must also exist.");
        return ge::GRAPH_FAILED;
    }

    if (CheckGroupIndexShapeDim(context, compileInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckOutputShape(const gert::TilingContext* context)
{
    auto xShape = context->GetInputShape(INPUT_OP_X_INDEX);
    auto gradXShape = context->GetOutputShape(OUTPUT_GRAD_X_INDEX);

    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t gradXDimNum = gradXShape->GetStorageShape().GetDimNum();

    if (xDimNum != gradXDimNum) {
        OP_LOGE(context->GetNodeName(), "x and grad_x shape dimension must be same.");
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < xDimNum; i++) {
        if (xShape->GetStorageShape().GetDim(i) != gradXShape->GetStorageShape().GetDim(i)) {
            OP_LOGE(context->GetNodeName(), "x and grad_x shape must be same.");
            return ge::GRAPH_FAILED;
        }
    }

    auto weightShape = context->GetOptionalInputShape(INPUT_WEIGHT_INDEX);
    if (weightShape != nullptr) {
        auto gradWeightShape = context->GetOutputShape(OUTPUT_GRAD_WEIGHT_INDEX);

        size_t gradWeightDimNum = gradWeightShape->GetStorageShape().GetDimNum();
        size_t weightDimNum = weightShape->GetStorageShape().GetDimNum();

        if (gradWeightDimNum != weightDimNum) {
            OP_LOGE(context->GetNodeName(), "grad_weight and weight shape dimension must be same.");
            return ge::GRAPH_FAILED;
        }

        for (size_t i = 0; i < weightDimNum; i++) {
            if (gradWeightShape->GetStorageShape().GetDim(i) != weightShape->GetStorageShape().GetDim(i)) {
                OP_LOGE(context->GetNodeName(), "grad_weight shape must be same as weight.");
                return ge::GRAPH_FAILED;
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckOpAllParams(gert::TilingContext* context, SwigluGroupQuantGradCompileInfo& compileInfo)
{
    if (CheckAllInputDtype(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Check input dtype failed.");
        return ge::GRAPH_FAILED;
    }
    if (CheckAllOutputDtype(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Check output dtype failed.");
        return ge::GRAPH_FAILED;
    }
    if (CheckAllAttrs(context, compileInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Check attrs failed.");
        return ge::GRAPH_FAILED;
    }
    if (CheckInputShape(context, compileInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Check input shape failed.");
        return ge::GRAPH_FAILED;
    }
    if (CheckOutputShape(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Check output shape failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

inline uint32_t GetTotalTokens(const gert::StorageShape* shape, gert::TilingContext* context)
{
    uint32_t totalTokens = 1;
    size_t dimNum = shape->GetStorageShape().GetDimNum();
    for (size_t i = 0; i < dimNum - 1; i++) {
        totalTokens *= static_cast<uint32_t>(shape->GetStorageShape().GetDim(i));
    }
    return totalTokens;
}

inline void CalculateTilingParams(const gert::TilingContext* context, SwigluGroupQuantGradCompileInfo& compileInfo,
                                  SwigluGroupQuantGradTilingData& tilingData)
{
    uint32_t dimH = tilingData.get_dimH();

    uint32_t hasClampLimit = compileInfo.hasClampLimit;
    uint32_t hasWeight = compileInfo.hasWeight;

    uint32_t ubFactor = UB_BASE_FACTOR;
    if (hasClampLimit) {
        ubFactor += UB_CLAMP_EXTRA_FACTOR;
    }
    if (hasWeight) {
        ubFactor += UB_WEIGHT_EXTRA_FACTOR;
    }

    uint32_t ubAvailable = compileInfo.ubSize - BLOCK_SIZE - TMP_DATA_UB_SIZE;
    uint32_t ubPerTokenFullH = ubFactor * dimH * sizeof(float);

    if (hasWeight) {
        ubPerTokenFullH += UB_WEIGHT_EXTRA_TOKENS * sizeof(float);
    }

    uint32_t needSplitH = (ubPerTokenFullH > ubAvailable) ? 1 : 0;
    tilingData.set_needSplitH(needSplitH);

    uint32_t tileH = dimH;
    uint32_t tileTokens = 0;

    if (needSplitH == 0) {
        tileTokens = ubAvailable / ubPerTokenFullH;
        tileTokens = std::max(tileTokens, ONE);
    } else {
        uint32_t ubPerTokenMinH = ubFactor * sizeof(float);
        if (hasWeight) {
            ubAvailable -= UB_WEIGHT_EXTRA_TOKENS * sizeof(float);
        }
        tileH = ubAvailable / ubPerTokenMinH;
        tileH = AlignDownT(tileH, FP16_BFP16_32B_ALIGN_NUM);
        tileH = std::max(tileH, FP16_BFP16_32B_ALIGN_NUM);
        tileTokens = ONE;
    }

    uint32_t numHTiles = CeilDivT(dimH, tileH);

    OP_LOGD(context, "ubAvailable %u needSplitH %u tileTokens %u tileH %u compileInfo.totalCore %u \n", ubAvailable,
            needSplitH, tileTokens, tileH, compileInfo.totalCore);
    tilingData.set_tileTokens(tileTokens);
    tilingData.set_tileH(tileH);
    tilingData.set_numHTiles(numHTiles);
}

inline ge::graphStatus SetTilingDataToContext(gert::TilingContext* context, SwigluGroupQuantGradTilingData& tilingData)
{
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus GetCompileInfo(gert::TilingContext* context, SwigluGroupQuantGradCompileInfo& compileInfo)
{
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatform;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    uint32_t ubSize = static_cast<uint32_t>(ubSizePlatform);

    if (totalCoreNum == 0 || ubSize <= 0) {
        OP_LOGE(context->GetNodeName(), "GetCompileInfo failed, coreNum:%u, ubSize:%u.", totalCoreNum, ubSize);
        return ge::GRAPH_FAILED;
    }

    compileInfo.totalCore = totalCoreNum;
    compileInfo.ubSize = ubSize;
    compileInfo.blockNum = BLOCK_SIZE / sizeof(float);

    return ge::GRAPH_SUCCESS;
}

inline void SetBasicTilingData(gert::TilingContext* context, const SwigluGroupQuantGradCompileInfo& compileInfo,
                               SwigluGroupQuantGradTilingData& tilingData)
{
    auto gradYShape = context->GetInputShape(INPUT_GRAD_Y_INDEX);
    auto xShape = context->GetInputShape(INPUT_OP_X_INDEX);
    uint32_t totalTokens = GetTotalTokens(gradYShape, context);
    uint32_t dimH = static_cast<uint32_t>(
        gradYShape->GetStorageShape().GetDim(gradYShape->GetStorageShape().GetDimNum() - 1));
    uint32_t dim2H = static_cast<uint32_t>(xShape->GetStorageShape().GetDim(xShape->GetStorageShape().GetDimNum() - 1));

    tilingData.set_coreNumAll(compileInfo.totalCore);
    tilingData.set_ubSize(compileInfo.ubSize);
    tilingData.set_totalTokens(totalTokens);
    tilingData.set_dim2H(dim2H);
    tilingData.set_dimH(dimH);
    tilingData.set_hasWeight(compileInfo.hasWeight);
    tilingData.set_hasYOrigin(compileInfo.hasYOrigin);
    tilingData.set_hasGroupIndex(compileInfo.hasGroupIndex);
    tilingData.set_hasClampLimit(compileInfo.hasClampLimit);
    tilingData.set_clampLimit(compileInfo.clampLimit);

    auto groupIndexShape = context->GetOptionalInputShape(INPUT_GROUP_INDEX_INDEX);
    if (groupIndexShape != nullptr) {
        uint32_t groupNum = static_cast<uint32_t>(groupIndexShape->GetStorageShape().GetDim(0));
        tilingData.set_groupNum(groupNum);
    } else {
        tilingData.set_groupNum(0);
    }
}

} // namespace optiling

#endif // SWIGLU_GROUP_QUANT_GRAD_TILING_UTILS_H