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
#include "util/math_util.h"
#include "multi_scale_deformable_attention_grad_tiling.h"

namespace {
  constexpr uint32_t WORKSPACE_16MBYTE_SIZE = 16 * 1024 * 1024;
}

namespace optiling {
  constexpr static int64_t FP32_MODE = 0;
  constexpr static int64_t FP16_MODE = 1;
  static uint64_t RESERVE_SAPCE = 1024;
  constexpr static int64_t NUM_LEVEL_BUFFER = 3;
  constexpr static int64_t NUM_EMEBDDIM_BUFFER = 8;
  constexpr static int64_t NUM_QUERIE_BUFFER = 15;
  constexpr static int64_t NUM_CHANNEL_BUFFER = 13;
  class MultiScaleDeformableAttentionGradTiling {
    public:
        explicit MultiScaleDeformableAttentionGradTiling(gert::TilingContext* context) : TilingContext(context){};
        ge::graphStatus Init();
        ge::graphStatus RunKernelTiling();
        void TilingDataPrint();
    private:
        void SetTilingKeyMode(ge::DataType dType_str);
        MultiScaleDeformableAttentionGradTilingData TilingData;
        gert::TilingContext* TilingContext = nullptr;
        uint64_t batch_size = 1; // 1 size
        uint64_t spatial_size = 100; // 100 size
        uint64_t num_heads = 8; // 8 size
        uint64_t channels = 32; // 32 size
        uint64_t num_levels = 1; // 1 size
        uint64_t num_query = 4; // 4 size
        uint64_t num_point = 8; // 8 size
        uint64_t core_used = 48; // 1024 size
        uint64_t block_bytes = 32;
        uint64_t dtype_size = 4;
        uint64_t max_ub_num = 0;
        uint64_t ub_size = 192 * 1024; // 192 * 1024 size
        uint64_t deterministicFlag = 0;
  };

  void MultiScaleDeformableAttentionGradTiling::SetTilingKeyMode(ge::DataType dType_str) {
    switch (dType_str) {
        case ge::DT_FLOAT:
            TilingContext->SetTilingKey(FP32_MODE);
            break;
        case ge::DT_FLOAT16:
            TilingContext->SetTilingKey(FP16_MODE);
            break;
        default:
            TilingContext->SetTilingKey(FP32_MODE);
            break;
    }
  }

  ge::graphStatus MultiScaleDeformableAttentionGradTiling::Init(){
    OP_LOGD(TilingContext, "Tiling initing.");
    auto value_shape = TilingContext->GetInputShape(0)->GetStorageShape();
    auto sampling_loc_shape = TilingContext->GetInputShape(3)->GetStorageShape();
    auto compileInfo = reinterpret_cast<const MultiScaleDeformableAttentionGradCompileInfo*>(TilingContext->GetCompileInfo());
    uint64_t total_ub_size = compileInfo->ub_size_platform;
    ub_size = total_ub_size - RESERVE_SAPCE;
    uint64_t core_num = compileInfo->total_core_num;

    deterministicFlag = TilingContext->GetDeterministic() == 1 ? 1 : 0;
    OP_LOGD(TilingContext, "deterministicFlag is %lu.", deterministicFlag);
    if (deterministicFlag == 1) {
      core_num = 1;
    }
    OP_LOGD(TilingContext, "core_num is %lu.", core_num);

    uint32_t value_idx = 0U; // 0: first idex
    uint32_t sample_idx = 2U; // 2: levels idx
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
    auto dtype_str = TilingContext->GetInputDesc(0)->GetDataType(); // 0 value idex
    SetTilingKeyMode(dtype_str);

    uint64_t data_align = block_bytes / dtype_size;
    uint64_t num_levels_align = (num_levels + data_align - 1) / data_align * data_align;
    max_ub_num = (ub_size / dtype_size - NUM_LEVEL_BUFFER * num_levels_align - NUM_EMEBDDIM_BUFFER * channels) /
                  (NUM_QUERIE_BUFFER + NUM_CHANNEL_BUFFER * channels);
    max_ub_num = max_ub_num / data_align * data_align;
    uint64_t taskNum = ((num_query + max_ub_num - 1) / max_ub_num) * batch_size * num_heads * num_levels * num_point;
    core_used = std::min(core_num, taskNum);
    OP_LOGD(TilingContext, "Tiling init finish.");
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus MultiScaleDeformableAttentionGradTiling::RunKernelTiling(){
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
    size_t sysWorkspaceSize = WORKSPACE_16MBYTE_SIZE;
    size_t *currentWorkspace = TilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;
    TilingData.SaveToBuffer(TilingContext->GetRawTilingData()->GetData(), TilingContext->GetRawTilingData()->GetCapacity());
    TilingContext->GetRawTilingData()->SetDataSize(TilingData.GetDataSize());
    TilingDataPrint();
    OP_LOGD(TilingContext->GetNodeName(), "Tiling end.");
    return ge::GRAPH_SUCCESS;
  }

  void MultiScaleDeformableAttentionGradTiling::TilingDataPrint() {
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

  static ge::graphStatus TilingMultiScaleDeformableAttentionGrad(gert::TilingContext* context) {
    MultiScaleDeformableAttentionGradTiling tilingObject(context);
    tilingObject.Init();
    return tilingObject.RunKernelTiling();
  }

  static ge::graphStatus TilingPrepareForMultiScaleDeformableAttentionGrad(gert::TilingParseContext* context) {
    OP_LOGD("MultiScaleDeformableAttentionGrad:", "TilingPrepareForMultiScaleDeformableAttentionGrad start.");
    auto compileInfo = context->GetCompiledInfo<MultiScaleDeformableAttentionGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->total_core_num = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->total_core_num <= 0), // 0 negative number
                     OP_LOGE(context->GetNodeName(), "Failed to get core num."),
                     return false);

    uint64_t ub_size_platform = 0U; // 0, init
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size_platform);
    compileInfo->ub_size_platform = static_cast<int64_t>(ub_size_platform);
    OP_CHECK_IF((compileInfo->ub_size_platform <= 0), // 0
                     OP_LOGE(context->GetNodeName(), "Failed to get ub size"),
                     return false);
    OP_LOGD("MultiScaleDeformableAttentionGrad:", "TilingPrepareForMultiScaleDeformableAttentionGrad end.");
    return ge::GRAPH_SUCCESS;
  }

  IMPL_OP_OPTILING(MultiScaleDeformableAttentionGrad)
      .Tiling(TilingMultiScaleDeformableAttentionGrad)
      .TilingParse<MultiScaleDeformableAttentionGradCompileInfo>(TilingPrepareForMultiScaleDeformableAttentionGrad);
}  // namespace optiling
