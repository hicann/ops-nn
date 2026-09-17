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
 * \file multi_scale_deformable_attention_grad_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_util.h"
#include "util/math_util.h"
#include "multi_scale_deformable_attention_grad_tiling.h"

namespace {
constexpr uint32_t WORKSPACE_16MBYTE_SIZE = 16 * 1024 * 1024;
}

namespace optiling {
constexpr static int64_t FP32_MODE = 0;
constexpr static int64_t FP16_MODE = 1;
constexpr static int64_t BF16_MODE = 2;
static uint64_t RESERVE_SAPCE = 8192;
constexpr static int64_t NUM_LEVEL_BUFFER = 3;
constexpr static int64_t NUM_EMBEDDIM_BUFFER = 8;
constexpr static int64_t NUM_EMBEDDIM_BUFFER_REGBASE = 6;
constexpr static int64_t NUM_QUERIE_BUFFER = 15;
constexpr static int64_t NUM_HALF_TWO_BUFFER = 6;
constexpr static int64_t NUM_CHANNEL_BUFFER = 13;
class MultiScaleDeformableAttentionGradTiling {
public:
    explicit MultiScaleDeformableAttentionGradTiling(gert::TilingContext* context) : TilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
    void TilingDataPrint();

private:
    void SetTilingKeyMode(ge::DataType dType_str);
    MultiScaleDeformableAttentionGradTilingData TilingData;
    gert::TilingContext* TilingContext = nullptr;
    uint64_t batch_size = 1;     // 1 size
    uint64_t spatial_size = 100; // 100 size
    uint64_t num_heads = 8;      // 8 size
    uint64_t channels = 32;      // 32 size
    uint64_t num_levels = 1;     // 1 size
    uint64_t num_query = 4;      // 4 size
    uint64_t num_point = 8;      // 8 size
    uint64_t core_used = 48;     // 1024 size
    uint64_t block_bytes = 32;
    uint64_t dtype_size = 4;
    ge::DataType valueDtype = ge::DT_FLOAT;
    uint64_t max_ub_num = 0;
    uint64_t ub_size = 192 * 1024; // 192 * 1024 size
    uint64_t deterministicFlag = 0;
};

void MultiScaleDeformableAttentionGradTiling::SetTilingKeyMode(ge::DataType dType_str)
{
    (void)dType_str;
    TilingContext->SetTilingKey(FP32_MODE);
}

ge::graphStatus MultiScaleDeformableAttentionGradTiling::Init()
{
    OP_LOGD(TilingContext, "Tiling initing.");
    auto value_shape_ptr = TilingContext->GetInputShape(0);
    auto spatial_shapes_ptr = TilingContext->GetInputShape(1);
    auto level_start_index_ptr = TilingContext->GetInputShape(2);
    auto sampling_loc_shape_ptr = TilingContext->GetInputShape(3);
    auto attn_weight_shape_ptr = TilingContext->GetInputShape(4);
    if (value_shape_ptr == nullptr || spatial_shapes_ptr == nullptr || level_start_index_ptr == nullptr ||
        sampling_loc_shape_ptr == nullptr || attn_weight_shape_ptr == nullptr) {
        OP_LOGE(TilingContext->GetNodeName(), "input shape ptr is nullptr");
        return ge::GRAPH_FAILED;
    }
    auto value_shape = value_shape_ptr->GetStorageShape();
    auto spatial_shapes = spatial_shapes_ptr->GetStorageShape();
    auto level_start_index_shape = level_start_index_ptr->GetStorageShape();
    auto sampling_loc_shape = sampling_loc_shape_ptr->GetStorageShape();
    auto attn_weight_shape = attn_weight_shape_ptr->GetStorageShape();
    auto compileInfo = reinterpret_cast<const MultiScaleDeformableAttentionGradCompileInfo*>(
        TilingContext->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(TilingContext, compileInfo);
    uint64_t total_ub_size = compileInfo->ub_size_platform;
    ub_size = total_ub_size - RESERVE_SAPCE;
    uint64_t core_num = compileInfo->total_core_num;

    deterministicFlag = TilingContext->GetDeterministic() == 1 ? 1 : 0;
    OP_LOGD(TilingContext, "deterministicFlag is %lu.", deterministicFlag);
    if (deterministicFlag == 1) {
        core_num = 1;
    }
    OP_LOGD(TilingContext, "core_num is %lu.", core_num);

    uint32_t value_idx = 0U;   // 0: first idex
    uint32_t sample_idx = 2U;  // 2: levels idx
    uint32_t sample_step = 2U; // 2 step;
    batch_size = value_shape.GetDim(value_idx++);
    spatial_size = value_shape.GetDim(value_idx++);
    num_heads = value_shape.GetDim(value_idx++);
    channels = value_shape.GetDim(value_idx);
    OP_LOGD(TilingContext, "batch size is %lu.", batch_size);
    OP_LOGD(TilingContext, "num_head is %lu.", num_heads);
    OP_LOGD(TilingContext, "spatial_size is %lu.", spatial_size);
    num_levels = sampling_loc_shape.GetDim(sample_idx++);
    num_point = sampling_loc_shape.GetDim(sample_idx);
    sample_idx += sample_step;
    num_query = sampling_loc_shape.GetDim(sample_idx);

    uint64_t spatial_num_levels = spatial_shapes.GetDim(0);
    uint64_t level_start_index_num_levels = level_start_index_shape.GetDim(0);
    uint64_t attn_weight_num_levels = attn_weight_shape.GetDim(2);
    if (num_levels != spatial_num_levels || num_levels != level_start_index_num_levels ||
        num_levels != attn_weight_num_levels) {
        OP_LOGE(TilingContext->GetNodeName(),
                "numLevels dimensions must be equal: samplingLocLevels=%lu, spatialShapeLevels=%lu, "
                "levelStartIndexLevels=%lu, attnWeightLevels=%lu",
                num_levels, spatial_num_levels, level_start_index_num_levels, attn_weight_num_levels);
        return ge::GRAPH_FAILED;
    }

    auto dtype_str = TilingContext->GetInputDesc(0)->GetDataType(); // 0 value idex
    valueDtype = dtype_str;
    SetTilingKeyMode(dtype_str);

    uint64_t data_align = block_bytes / dtype_size;
    if (dtype_str != ge::DT_FLOAT) {
        data_align = block_bytes / 2; // FP16/BF16: align Q to 16 elements for overlap cast 32B alignment
    }
    uint64_t num_levels_align = (num_levels + data_align - 1) / data_align * data_align;
    uint64_t staging_cost = 0;
    if (dtype_str != ge::DT_FLOAT) {
        uint64_t input_dtype_size = 2; // half or bfloat16_t
        uint64_t staging_stride = std::max(channels, block_bytes / input_dtype_size);
        staging_cost = 4 * staging_stride * sizeof(float); // 1 input staging buf x 4 slots, allocated as float
    }
    uint64_t num_query_buffer = NUM_QUERIE_BUFFER;
    if (dtype_str != ge::DT_FLOAT) {
        num_query_buffer += NUM_HALF_TWO_BUFFER;
    }
    uint64_t num_embeddim_buffer = compileInfo->isRegBase ? NUM_EMBEDDIM_BUFFER_REGBASE : NUM_EMBEDDIM_BUFFER;
    max_ub_num = (ub_size / dtype_size - NUM_LEVEL_BUFFER * num_levels_align - num_embeddim_buffer * channels -
                  staging_cost / dtype_size) /
                 (num_query_buffer + NUM_CHANNEL_BUFFER * channels);
    max_ub_num = max_ub_num / data_align * data_align;
    uint64_t taskNum = ((num_query + max_ub_num - 1) / max_ub_num) * batch_size * num_heads * num_levels * num_point;
    core_used = std::min(core_num, taskNum);
    if (deterministicFlag == 1 && compileInfo->isRegBase) {
        uint64_t totalBH = batch_size * num_heads;
        core_used = std::min(compileInfo->total_core_num, taskNum);
        core_used = std::min(core_used, totalBH);
        if (core_used == 0) {
            core_used = 1;
        }
    }
    OP_LOGD(TilingContext, "Tiling init finish.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MultiScaleDeformableAttentionGradTiling::RunKernelTiling()
{
    OP_LOGD(TilingContext, "Tiling start.");
    TilingContext->SetBlockDim(core_used);
    TilingContext->SetScheduleMode(1);
    TilingData.set_batchSize(batch_size);
    TilingData.set_numKeys(spatial_size);
    TilingData.set_numHeads(num_heads);
    TilingData.set_embedDims(channels);
    TilingData.set_numLevels(num_levels);
    TilingData.set_numQueries(num_query);
    TilingData.set_numPoints(num_point);
    TilingData.set_maxUbNum(max_ub_num);
    TilingData.set_coreNum(core_used);
    TilingData.set_isDeterministic(deterministicFlag);
    size_t sysWorkspaceSize = WORKSPACE_16MBYTE_SIZE;
    if (valueDtype != ge::DT_FLOAT) {
        sysWorkspaceSize += batch_size * spatial_size * num_heads * channels * sizeof(float);
    }
    size_t* currentWorkspace = TilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;
    TilingData.SaveToBuffer(TilingContext->GetRawTilingData()->GetData(),
                            TilingContext->GetRawTilingData()->GetCapacity());
    TilingContext->GetRawTilingData()->SetDataSize(TilingData.GetDataSize());
    TilingDataPrint();
    OP_LOGD(TilingContext->GetNodeName(), "Tiling end.");
    return ge::GRAPH_SUCCESS;
}

void MultiScaleDeformableAttentionGradTiling::TilingDataPrint()
{
    OP_LOGD(TilingContext, "batch_size:     %lu.", batch_size);
    OP_LOGD(TilingContext, "spatial_size:   %lu.", spatial_size);
    OP_LOGD(TilingContext, "num_heads:      %lu.", num_heads);
    OP_LOGD(TilingContext, "channels:       %lu.", channels);
    OP_LOGD(TilingContext, "num_levels:     %lu.", num_levels);
    OP_LOGD(TilingContext, "num_query:      %lu.", num_query);
    OP_LOGD(TilingContext, "num_point:      %lu.", num_point);
    OP_LOGD(TilingContext, "max_ub_num:     %lu.", max_ub_num);
    OP_LOGD(TilingContext, "core_used:      %lu.", core_used);
    OP_LOGD(TilingContext, "ub_size:        %lu.", ub_size);
}

static ge::graphStatus TilingMultiScaleDeformableAttentionGrad(gert::TilingContext* context)
{
    MultiScaleDeformableAttentionGradTiling tilingObject(context);
    tilingObject.Init();
    return tilingObject.RunKernelTiling();
}

static ge::graphStatus TilingPrepareForMultiScaleDeformableAttentionGrad(gert::TilingParseContext* context)
{
    OP_LOGD("MultiScaleDeformableAttentionGrad:", "TilingPrepareForMultiScaleDeformableAttentionGrad start.");
    auto compileInfo = context->GetCompiledInfo<MultiScaleDeformableAttentionGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->total_core_num = ascendcPlatform.GetCoreNumAiv();
    compileInfo->isRegBase = Ops::NN::OpTiling::IsRegbaseSocVersion(context);
    OP_CHECK_IF((compileInfo->total_core_num <= 0), // 0 negative number
                OP_LOGE(context->GetNodeName(), "Failed to get core num."), return false);

    uint64_t ub_size_platform = 0U; // 0, init
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size_platform);
    compileInfo->ub_size_platform = static_cast<int64_t>(ub_size_platform);
    OP_CHECK_IF((compileInfo->ub_size_platform <= 0), // 0
                OP_LOGE(context->GetNodeName(), "Failed to get ub size"), return false);
    OP_LOGD("MultiScaleDeformableAttentionGrad:", "TilingPrepareForMultiScaleDeformableAttentionGrad end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MultiScaleDeformableAttentionGrad)
    .Tiling(TilingMultiScaleDeformableAttentionGrad)
    .TilingParse<MultiScaleDeformableAttentionGradCompileInfo>(TilingPrepareForMultiScaleDeformableAttentionGrad);
} // namespace optiling
