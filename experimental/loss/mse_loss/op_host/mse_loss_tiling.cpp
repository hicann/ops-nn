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
 * \file mse_loss_tiling.cpp
 * \brief MseLoss 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/mse_loss_tiling_data.h"
#include "../op_kernel/mse_loss_tiling_key.h"
#include <cstring>
#include <limits>

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t ALIGN_ELEM = 64;
constexpr int64_t MAX_VECTOR_REPEAT = 255;
constexpr int64_t MAX_UB_FACTOR = ALIGN_ELEM * MAX_VECTOR_REPEAT;
constexpr int64_t RESERVED_UB_SIZE = 16 * 1024;
constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 16 * 1024 * 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline gert::Shape EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return inShape;
}

static bool IsSameShape(const gert::Shape& lhs, const gert::Shape& rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum should be positive, got %ld", coreNum),
                return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspaceSize(gert::TilingContext* context, size_t workspaceSize)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalNum, ge::DataType& dataType,
                                         int64_t& reduction)
{
    const auto* predictShapeInfo = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, predictShapeInfo);
    const auto predictShape = EnsureNotScalar(predictShapeInfo->GetStorageShape());

    const auto* labelShapeInfo = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, labelShapeInfo);
    const auto labelShape = EnsureNotScalar(labelShapeInfo->GetStorageShape());

    const auto* outputShapeInfo = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShapeInfo);
    const auto outputShape = EnsureNotScalar(outputShapeInfo->GetStorageShape());

    OP_CHECK_IF(!IsSameShape(predictShape, labelShape),
                OP_LOGE(context, "MseLoss requires predict and label to have the same shape"), return ge::GRAPH_FAILED);
    totalNum = predictShape.GetShapeSize();

    const auto* predictDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, predictDesc);
    const auto* labelDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, labelDesc);
    const auto* outputDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    dataType = predictDesc->GetDataType();
    OP_CHECK_IF(labelDesc->GetDataType() != dataType,
                OP_LOGE(context, "MseLoss requires predict and label to have the same dtype"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(outputDesc->GetDataType() != dataType,
                OP_LOGE(context, "MseLoss requires predict and y to have the same dtype"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(dataType != ge::DT_FLOAT16 && dataType != ge::DT_FLOAT && dataType != ge::DT_BF16,
                OP_LOGE(context, "MseLoss invalid dtype %d", static_cast<int32_t>(dataType)), return ge::GRAPH_FAILED);

    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* reductionAttr = attrs->GetAttrPointer<char>(0);
    OP_CHECK_IF(reductionAttr == nullptr, OP_LOGE(context, "failed to get reduction attribute"),
                return ge::GRAPH_FAILED);

    if (std::strcmp(reductionAttr, "none") == 0) {
        reduction = 0;
        OP_CHECK_IF(!IsSameShape(predictShape, outputShape),
                    OP_LOGE(context, "MseLoss none mode requires y to have the same shape as predict"),
                    return ge::GRAPH_FAILED);
    } else if (std::strcmp(reductionAttr, "sum") == 0) {
        reduction = 1;
        OP_CHECK_IF(outputShape.GetShapeSize() != 1, OP_LOGE(context, "MseLoss sum mode requires scalar y"),
                    return ge::GRAPH_FAILED);
    } else if (std::strcmp(reductionAttr, "mean") == 0) {
        reduction = 2;
        OP_CHECK_IF(outputShape.GetShapeSize() != 1, OP_LOGE(context, "MseLoss mean mode requires scalar y"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(context, "MseLoss invalid reduction %s", reductionAttr);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static int64_t CalcUbFactor(uint64_t ubSize, ge::DataType dataType, int64_t totalNum)
{
    if (ubSize <= static_cast<uint64_t>(RESERVED_UB_SIZE)) {
        return 0;
    }
    int64_t availableUb = static_cast<int64_t>(ubSize) - RESERVED_UB_SIZE;
    int64_t bytesPerElem = 0;
    if (dataType == ge::DT_FLOAT) {
        bytesPerElem = 7 * static_cast<int64_t>(sizeof(float));
    } else {
        int64_t ioTypeSize = 2;
        bytesPerElem = 3 * 2 * ioTypeSize + 3 * static_cast<int64_t>(sizeof(float));
    }
    int64_t rawUbFactor = FloorDiv(availableUb, bytesPerElem);
    int64_t ubFactor = FloorAlign(rawUbFactor, ALIGN_ELEM);
    if (ubFactor <= 0) {
        ubFactor = ALIGN_ELEM;
    }
    if (ubFactor > MAX_UB_FACTOR) {
        ubFactor = MAX_UB_FACTOR;
    }
    if (totalNum > 0 && totalNum < ubFactor) {
        ubFactor = CeilDiv(totalNum, ALIGN_ELEM) * ALIGN_ELEM;
    }
    return ubFactor;
}

static int64_t GetGmAlignElem(ge::DataType dataType, int64_t blockSize)
{
    if (dataType == ge::DT_FLOAT) {
        return blockSize / static_cast<int64_t>(sizeof(float));
    }
    return blockSize / 2;
}

struct MseLossTilingParams {
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    int64_t totalNum = 0;
    ge::DataType dataType = ge::DT_FLOAT;
    int64_t reduction = 2;
    int64_t ubBlockSize = 0;
};

static ge::graphStatus GetTilingParams(gert::TilingContext* context, MseLossTilingParams& params)
{
    OP_CHECK_IF(GetPlatformInfo(context, params.ubSize, params.coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetShapeAttrsInfo(context, params.totalNum, params.dataType, params.reduction) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);
    params.ubBlockSize = static_cast<int64_t>(GetUbBlockSize(context));
    OP_CHECK_IF(params.ubBlockSize <= 0,
                OP_LOGE(context, "UB block size should be positive, got %ld", params.ubBlockSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingData(gert::TilingContext* context, const MseLossTilingParams& params,
                                     MseLossTilingData& tiling)
{
    int64_t blockNum = 1;
    if (params.totalNum > 0) {
        int64_t ubFactor = CalcUbFactor(params.ubSize, params.dataType, params.totalNum);
        OP_CHECK_IF(ubFactor <= 0, OP_LOGE(context, "failed to calculate UB factor, ubSize=%lu", params.ubSize),
                    return ge::GRAPH_FAILED);
        int64_t singleCoreThreshold = CalcUbFactor(params.ubSize, params.dataType, std::numeric_limits<int64_t>::max());
        int64_t blockFactor = 1;
        if (params.reduction != 0 && params.totalNum <= singleCoreThreshold) {
            blockFactor = params.totalNum;
        } else {
            blockFactor = CeilDiv(params.totalNum, params.coreNum);
            const int64_t gmAlignElem = GetGmAlignElem(params.dataType, params.ubBlockSize);
            blockFactor = CeilDiv(blockFactor, gmAlignElem) * gmAlignElem;
            blockNum = CeilDiv(params.totalNum, blockFactor);
        }
        tiling.totalNum = params.totalNum;
        tiling.blockFactor = blockFactor;
        tiling.ubFactor = ubFactor;
    } else {
        tiling.totalNum = 0;
        tiling.blockFactor = 1;
        tiling.ubFactor = ALIGN_ELEM;
    }
    tiling.reduction = params.reduction;
    tiling.blockNum = blockNum;
    tiling.workspaceFloatsPerCore = CeilDiv(params.ubBlockSize, static_cast<int64_t>(sizeof(float)));
    tiling.meanScale = (params.totalNum > 0) ? (1.0f / static_cast<float>(params.totalNum)) :
                                               std::numeric_limits<float>::quiet_NaN();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingContext(gert::TilingContext* context, const MseLossTilingParams& params,
                                        const MseLossTilingData& tiling)
{
    context->SetBlockDim(tiling.blockNum);
    if (params.reduction != 0 && tiling.blockNum > 1) {
        context->SetScheduleMode(1);
    }

    const int64_t workspaceBytesPerCore = tiling.workspaceFloatsPerCore * static_cast<int64_t>(sizeof(float));
    size_t workspaceSize = (params.reduction == 0 || tiling.blockNum == 1) ?
                               WS_SYS_SIZE :
                               static_cast<size_t>(ASCENDC_TOOLS_WORKSPACE + tiling.blockNum * workspaceBytesPerCore);
    OP_CHECK_IF(SetWorkspaceSize(context, workspaceSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "SetWorkspaceSize error"), return ge::GRAPH_FAILED);

    uint64_t tilingKey = 0;
    if (params.dataType == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(MSELOSS_TPL_SCH_MODE_0);
    } else if (params.dataType == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(MSELOSS_TPL_SCH_MODE_1);
    } else {
        tilingKey = GET_TPL_TILING_KEY(MSELOSS_TPL_SCH_MODE_2);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MseLossTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("MseLoss", "context is nullptr"), return ge::GRAPH_FAILED);
    MseLossTilingParams params;
    OP_CHECK_IF(GetTilingParams(context, params) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetTilingParams error"),
                return ge::GRAPH_FAILED);

    MseLossTilingData* tiling = context->GetTilingData<MseLossTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(SetTilingData(context, params, *tiling) != ge::GRAPH_SUCCESS, OP_LOGE(context, "SetTilingData error"),
                return ge::GRAPH_FAILED);
    return SetTilingContext(context, params, *tiling);
}

static ge::graphStatus TilingParseForMseLoss([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct MseLossCompileInfo {};

IMPL_OP_OPTILING(MseLoss).Tiling(MseLossTilingFunc).TilingParse<MseLossCompileInfo>(TilingParseForMseLoss);

} // namespace optiling
