/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file multi_scale_deformable_attn_function_tiling_arch35.cpp
 * \brief ascend950 (SIMT/regbase) tiling for MultiScaleDeformableAttnFunction
 */

#include "multi_scale_deformable_attn_function_tiling.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/arch35/multi_scale_deformable_attn_function_tiling_data.h"
#include "../op_kernel/arch35/multi_scale_deformable_attn_function_tiling_key.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"

using namespace ge;

namespace {
const std::string OP_NAME = "MultiScaleDeformableAttn";
const uint64_t INPUT_VALUE = 0;
const uint64_t INPUT_SPATIAL_SHAPE = 1;
const uint64_t INPUT_ATTN_WEIGHT = 4;

const uint64_t NUM_KEYS_DIM_TRANSPOSE = 2;
const uint64_t NUM_HEADS_DIM_TRANSPOSE = 1;
const uint64_t REAL_LEVEL_DIM_TRANSPOSE = 2;
const uint64_t NUM_QUERIES_DIM_TRANSPOSE = 4;
const uint64_t NUM_POINTS_DIM_TRANSPOSE = 3;
const uint64_t BATCH_SIZE_DIM = 0;
const uint64_t NUM_KEYS_DIM = 1;
const uint64_t NUM_HEADS_DIM = 2;
const uint64_t EMBED_DIMS_DIM = 3;
const uint64_t NUM_LEVEL_DIM = 0;
const uint64_t REAL_LEVEL_DIM = 3;
const uint64_t NUM_QUERIES_DIM = 1;
const uint64_t NUM_POINTS_DIM = 4;

const uint64_t sysWorkspaceSize = 16 * 1024 * 1024;
constexpr uint64_t SIMT_THREAD_DIM = 512;
constexpr uint64_t CHANNELS_SIMD_THRESHOLD = 64;
} // namespace

namespace optiling {

struct MsdaShapeDims {
    uint64_t batchSize;
    uint64_t numKeys;
    uint64_t numHeads;
    uint64_t embedDims;
    uint64_t numLevels;
    uint64_t numQueries;
    uint64_t numPoints;
    uint64_t realLevels;
};

static MsdaShapeDims ParseShapeDims(const gert::Shape& valueShape, const gert::Shape& spatialShape,
                                    const gert::Shape& attnWeightShape)
{
    MsdaShapeDims d;
    d.batchSize = valueShape.GetDim(BATCH_SIZE_DIM);
    d.embedDims = valueShape.GetDim(EMBED_DIMS_DIM);
    d.numLevels = spatialShape.GetDim(NUM_LEVEL_DIM);
    if (d.embedDims < CHANNELS_SIMD_THRESHOLD) {
        d.numQueries = attnWeightShape.GetDim(NUM_QUERIES_DIM);
        d.realLevels = attnWeightShape.GetDim(REAL_LEVEL_DIM);
        d.numPoints = attnWeightShape.GetDim(NUM_POINTS_DIM);
        d.numHeads = attnWeightShape.GetDim(NUM_HEADS_DIM);
        d.numKeys = valueShape.GetDim(NUM_KEYS_DIM);
    } else {
        d.numQueries = attnWeightShape.GetDim(NUM_QUERIES_DIM_TRANSPOSE);
        d.realLevels = attnWeightShape.GetDim(REAL_LEVEL_DIM_TRANSPOSE);
        d.numPoints = attnWeightShape.GetDim(NUM_POINTS_DIM_TRANSPOSE);
        d.numHeads = attnWeightShape.GetDim(NUM_HEADS_DIM_TRANSPOSE);
        d.numKeys = valueShape.GetDim(NUM_KEYS_DIM_TRANSPOSE);
    }
    return d;
}

static uint64_t SelectRoutingAndSetBlockDim(gert::TilingContext* context, uint64_t embedDims, uint64_t coreNum,
                                            uint64_t batchSize, uint64_t numQueries, uint64_t numHeads)
{
    if (embedDims >= CHANNELS_SIMD_THRESHOLD) {
        context->SetBlockDim(coreNum);
        context->SetScheduleMode(1);
        return MSDA_MODE_GENERIC;
    }
    uint64_t totalOutput = batchSize * numQueries * numHeads * embedDims;
    uint64_t needCoreNum = (totalOutput + SIMT_THREAD_DIM - 1) / SIMT_THREAD_DIM;
    needCoreNum = std::min(needCoreNum, coreNum);
    needCoreNum = std::max(needCoreNum, 1UL);
    context->SetBlockDim(needCoreNum);
    OP_LOGD(context, "950 SIMT: totalOutput=%lu, needCoreNum=%lu, coreNum=%lu", totalOutput, needCoreNum, coreNum);
    return MSDA_MODE_SIMT;
}

static void SetTilingData(MsdaRegBaseTilingData* tiling, const MsdaShapeDims& dims, uint64_t coreNum)
{
    tiling->batchSize = dims.batchSize;
    tiling->numKeys = dims.numKeys;
    tiling->numHeads = dims.numHeads;
    tiling->embedDims = dims.embedDims;
    tiling->numLevels = dims.numLevels;
    tiling->numQueries = dims.numQueries;
    tiling->numPoints = dims.numPoints;
    tiling->coreNum = coreNum;
    tiling->pointLoops = 0;
    tiling->realLevels = dims.realLevels;
}

ge::graphStatus Tiling4MultiScaleDeformableAttnArch35(gert::TilingContext* context)
{
    OP_LOGD(OP_NAME, "Tiling4MultiScaleDeformableAttnArch35 (950 SIMD/SIMT) start.");

    MsdaRegBaseTilingData* tiling = context->GetTilingData<MsdaRegBaseTilingData>();

    auto valueTensorPtr = context->GetInputTensor(INPUT_VALUE);
    auto spatialTensorPtr = context->GetInputTensor(INPUT_SPATIAL_SHAPE);
    auto attnWeightTensorPtr = context->GetInputTensor(INPUT_ATTN_WEIGHT);
    if (valueTensorPtr == nullptr || spatialTensorPtr == nullptr || attnWeightTensorPtr == nullptr) {
        OP_LOGE(context->GetNodeName(), "value/spatialShape/attnWeight tensor is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto valueShape = valueTensorPtr->GetStorageShape();
    auto spatialShape = spatialTensorPtr->GetStorageShape();
    auto attnWeightShape = attnWeightTensorPtr->GetStorageShape();

    auto compileInfo = static_cast<const MultiScaleDeformableAttnFunctionCompileInfo*>(context->GetCompileInfo());
    if (compileInfo == nullptr) {
        OP_LOGE(context->GetNodeName(), "compile info is null");
        return ge::GRAPH_FAILED;
    }
    uint64_t coreNum = compileInfo->totalCoreNum;
    uint64_t deterministicFlag = context->GetDeterministic() == 1 ? 1 : 0;
    if (deterministicFlag == 1) {
        coreNum = 1;
    }
    OP_LOGD(context, "deterministicFlag is %lu, coreNum = %lu", deterministicFlag, coreNum);

    auto dims = ParseShapeDims(valueShape, spatialShape, attnWeightShape);

    uint64_t schMode = SelectRoutingAndSetBlockDim(context, dims.embedDims, coreNum, dims.batchSize, dims.numQueries,
                                                   dims.numHeads);
    uint64_t tilingKey = GET_TPL_TILING_KEY(schMode);
    context->SetTilingKey(tilingKey);

    SetTilingData(tiling, dims, coreNum);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        OP_LOGE(context->GetNodeName(), "currentWorkspace is null");
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = sysWorkspaceSize;

    OP_LOGD(OP_NAME, "Tiling4MultiScaleDeformableAttnArch35 end.");
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
