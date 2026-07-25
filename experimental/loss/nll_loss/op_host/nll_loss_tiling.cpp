/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <string>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/nll_loss_tiling_data.h"
#include "../op_kernel/nll_loss_tiling_key.h"

namespace optiling {
constexpr uint32_t INPUT_X_IDX = 0u;
constexpr uint32_t INPUT_TARGET_IDX = 1u;
constexpr uint32_t INPUT_WEIGHT_IDX = 2u;
constexpr size_t ATTR_REDUCTION_IDX = 0u;
constexpr size_t ATTR_IGNORE_INDEX_IDX = 1u;

// map x dtype to schMode tiling key.
static ge::graphStatus SelectNllLossTilingKey(ge::DataType dtype, uint64_t& tilingKey)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
            tilingKey = GET_TPL_TILING_KEY(NLLLOSS_TPL_SCH_MODE_0);
            return ge::GRAPH_SUCCESS;
        case ge::DT_FLOAT:
            tilingKey = GET_TPL_TILING_KEY(NLLLOSS_TPL_SCH_MODE_1);
            return ge::GRAPH_SUCCESS;
        case ge::DT_BF16:
            tilingKey = GET_TPL_TILING_KEY(NLLLOSS_TPL_SCH_MODE_2);
            return ge::GRAPH_SUCCESS;
        default:
            return ge::GRAPH_FAILED;
    }
}

static void ParseNllLossAttrs(const gert::RuntimeAttrs* attrs, int64_t& reduction, int64_t& ignoreIndex)
{
    reduction = 1;
    ignoreIndex = -100;
    if (attrs == nullptr) {
        return;
    }
    const char* r = attrs->GetAttrPointer<char>(ATTR_REDUCTION_IDX);
    if (r != nullptr) {
        if (strcmp(r, "none") == 0) {
            reduction = 0;
        } else if (strcmp(r, "sum") == 0) {
            reduction = 2;
        }
    }
    const int64_t* ig = attrs->GetAttrPointer<int64_t>(ATTR_IGNORE_INDEX_IDX);
    if (ig != nullptr) {
        ignoreIndex = *ig;
    }
}

// core split -> UB tile split. Fills usedCoreNum / rowsPerCore / tileRows / useVector.
static void ComputeNllLossSplit(NllLossTilingData* tiling, uint64_t aivCoreNum, uint64_t rowNum, uint64_t classNum,
                                uint64_t xElemSize, uint64_t hasWeight)
{
    constexpr uint64_t COMPUTE_EQV_BYTES = 512u;
    constexpr uint64_t WORK_PER_CORE = 16u * 1024u;
    uint64_t workAmount = rowNum * (classNum * xElemSize + COMPUTE_EQV_BYTES);
    uint64_t usedCoreNum = (workAmount + WORK_PER_CORE - 1u) / WORK_PER_CORE;
    if (usedCoreNum > aivCoreNum) {
        usedCoreNum = aivCoreNum;
    }
    if (usedCoreNum > rowNum) {
        usedCoreNum = rowNum;
    }
    if (usedCoreNum == 0u) {
        usedCoreNum = 1u;
    }
    uint64_t rowsPerCore = (rowNum + usedCoreNum - 1u) / usedCoreNum;
    usedCoreNum = (rowNum + rowsPerCore - 1u) / rowsPerCore;

    uint64_t repRows = (usedCoreNum <= 1u) ? rowNum : rowsPerCore;
    constexpr uint64_t VEC_THRESHOLD = 128u;
    uint64_t useVector = (repRows >= VEC_THRESHOLD) ? 1u : 0u;

    uint64_t ubBudget = 140u * 1024u;
    if (hasWeight == 1u) {
        uint64_t wBytes = classNum * (xElemSize + 4u) + 256u;
        ubBudget = (ubBudget > wBytes + 32u * 1024u) ? (ubBudget - wBytes) : (32u * 1024u);
    }
    uint64_t perRow = classNum * xElemSize + 320u;
    uint64_t tileRows = ubBudget / perRow;
    if (tileRows == 0u) {
        tileRows = 1u;
    }
    if (tileRows > rowsPerCore) {
        tileRows = rowsPerCore;
    }

    tiling->usedCoreNum = usedCoreNum;
    tiling->rowsPerCore = rowsPerCore;
    tiling->tileRows = tileRows;
    tiling->useVector = useVector;
}

static ge::graphStatus NllLossTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context, "NllLoss tiling starts.");

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint64_t aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(aivCoreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);

    auto xDesc = context->GetInputDesc(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    ge::DataType dtype = xDesc->GetDataType();
    uint64_t tilingKey = 0u;
    OP_CHECK_IF(SelectNllLossTilingKey(dtype, tilingKey) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "x dtype must be one of float16, float32, bfloat16."), return ge::GRAPH_FAILED);

    auto xShape = context->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto targetShape = context->GetInputShape(INPUT_TARGET_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetShape);

    auto& xStorage = xShape->GetStorageShape();
    size_t xDimNum = xStorage.GetDimNum();
    OP_CHECK_IF(xDimNum == 0u, OP_LOGE(context, "x rank must be >= 1."), return ge::GRAPH_FAILED);
    uint64_t classNum = static_cast<uint64_t>(xStorage.GetDim(xDimNum - 1));
    uint64_t rowNum = static_cast<uint64_t>(targetShape->GetStorageShape().GetShapeSize());
    OP_CHECK_IF(classNum == 0u, OP_LOGE(context, "class num (x last dim) must not be 0."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(rowNum == 0u, OP_LOGE(context, "row num (target size) must not be 0."), return ge::GRAPH_FAILED);

    int64_t reduction = 1;
    int64_t ignoreIndex = -100;
    ParseNllLossAttrs(context->GetAttrs(), reduction, ignoreIndex);

    uint64_t hasWeight = 0u;
    auto weightShape = context->GetOptionalInputShape(INPUT_WEIGHT_IDX);
    if (weightShape != nullptr && weightShape->GetStorageShape().GetShapeSize() > 0) {
        hasWeight = 1u;
    }

    auto targetDesc = context->GetInputDesc(INPUT_TARGET_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetDesc);
    ge::DataType targetDtype = targetDesc->GetDataType();
    OP_CHECK_IF(targetDtype != ge::DT_INT32 && targetDtype != ge::DT_INT64,
                OP_LOGE(context, "target dtype must be int32 or int64."), return ge::GRAPH_FAILED);
    uint64_t targetIsInt64 = (targetDtype == ge::DT_INT64) ? 1u : 0u;

    NllLossTilingData* tiling = context->GetTilingData<NllLossTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(NllLossTilingData), 0, sizeof(NllLossTilingData)) != EOK,
                OP_LOGE(context, "memset tiling data error"), return ge::GRAPH_FAILED);

    uint64_t xElemSize = (dtype == ge::DT_FLOAT) ? 4u : 2u;
    ComputeNllLossSplit(tiling, aivCoreNum, rowNum, classNum, xElemSize, hasWeight);
    tiling->rowNum = rowNum;
    tiling->classNum = classNum;
    tiling->reduction = reduction;
    tiling->ignoreIndex = ignoreIndex;
    tiling->hasWeight = hasWeight;
    tiling->targetIsInt64 = targetIsInt64;

    context->SetBlockDim(tiling->usedCoreNum);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);

    uint64_t sysWs = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint64_t syncAllWs = 32u * tiling->usedCoreNum;
    uint64_t reduceWs = 2u * 32u * tiling->usedCoreNum;
    currentWorkspace[0] = sysWs + syncAllWs + reduceWs;

    context->SetTilingKey(tilingKey);
    OP_LOGD(
        context,
        "NllLoss tiling: key=%lu N=%lu C=%lu red=%ld ig=%ld hasW=%lu i64=%lu cores=%lu rowsPC=%lu tileR=%lu vec=%lu",
        tilingKey, rowNum, classNum, reduction, ignoreIndex, hasWeight, targetIsInt64, tiling->usedCoreNum,
        tiling->rowsPerCore, tiling->tileRows, tiling->useVector);
    return ge::GRAPH_SUCCESS;
}

struct NllLossCompileInfo {};

static ge::graphStatus TilingParseForNllLoss([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(NllLoss).Tiling(NllLossTilingFunc).TilingParse<NllLossCompileInfo>(TilingParseForNllLoss);

} // namespace optiling
