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
 * \file chamfer_distance_grad.cc
 * \brief
 */
#include "chamfer_distance_grad_tiling.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
namespace {
constexpr uint32_t WORKSPACE_16MBYTE_SIZE = 16 * 1024 * 1024;
}

namespace optiling {
constexpr int64_t FP32_MODE = 1;
constexpr int64_t FP16_MODE = 2;
uint32_t BLOCK_BYTES = 32;
uint64_t RESERVE_SAPCE = 1024;
uint32_t FLOAT_DTYPE_BYTES = 4;
uint32_t BFLOAT16_DTYPE_BYTES = 2;
uint32_t FLOAT16_DTYPE_BYTES = 2;
constexpr uint32_t BATCH_MODE = 1;
class ChamferDistanceGradTiling {
public:
    explicit ChamferDistanceGradTiling(gert::TilingContext* context) : TilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
    void TilingDataPrint();

private:
    void SetTilingKeyMode(ge::DataType dType_str);
    ChamferDistanceGradTilingData TilingData;
    gert::TilingContext* TilingContext = nullptr;
    uint32_t batch_size = 1;
    uint32_t num = 100;
    uint64_t ub_size = 192 * 1024;
    uint32_t task_per_core = 1;
    uint32_t core_used = 1;
    uint32_t task_tail_core = 1;
};

void ChamferDistanceGradTiling::SetTilingKeyMode(ge::DataType dType_str)
{
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

ge::graphStatus ChamferDistanceGradTiling::Init()
{
    OP_LOGD(TilingContext, "Tiling initing.");
    auto xyz_shape = TilingContext->GetInputShape(0)->GetStorageShape();
    auto compileInfo = reinterpret_cast<const ChamferDistanceGradCompileInfo*>(TilingContext->GetCompileInfo());
    uint64_t total_ub_size = compileInfo->ubSizePlatForm;
    ub_size = total_ub_size - RESERVE_SAPCE;
    uint32_t core_num = compileInfo->totalCoreNum;
    OP_LOGD(TilingContext, "core_num is %u.", core_num);
    batch_size = xyz_shape.GetDim(0);
    num = xyz_shape.GetDim(1);
    auto dtype_str = TilingContext->GetInputDesc(0)->GetDataType();
    SetTilingKeyMode(dtype_str);
    task_per_core = (batch_size * num - 1) / core_num + 1;
    core_used = (batch_size * num - 1) / task_per_core + 1;
    task_tail_core = batch_size * num - (core_used - 1) * task_per_core;
    OP_LOGD(TilingContext, "Tiling inited.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChamferDistanceGradTiling::RunKernelTiling()
{
    OP_LOGD(TilingContext, "Tiling start.");
    TilingContext->SetBlockDim(core_used);
    TilingData.set_batch_size(batch_size);
    TilingData.set_num(num);
    TilingData.set_ub_size(ub_size);
    TilingData.set_task_per_core(task_per_core);
    TilingData.set_core_used(core_used);
    TilingData.set_task_tail_core(task_tail_core);
    size_t sysWorkspaceSize = WORKSPACE_16MBYTE_SIZE;
    size_t* currentWorkspace = TilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;
    TilingData.SaveToBuffer(
        TilingContext->GetRawTilingData()->GetData(), TilingContext->GetRawTilingData()->GetCapacity());
    TilingContext->GetRawTilingData()->SetDataSize(TilingData.GetDataSize());
    TilingDataPrint();
    TilingContext->SetScheduleMode(BATCH_MODE);
    OP_LOGD(TilingContext, "Tiling end.");
    return ge::GRAPH_SUCCESS;
}

void ChamferDistanceGradTiling::TilingDataPrint()
{
    OP_LOGD(TilingContext, "batch_size:       %d.", batch_size);
    OP_LOGD(TilingContext, "num:  %d.", num);
    OP_LOGD(TilingContext, "ub_size:               %lu.", ub_size);
    OP_LOGD(TilingContext, "task_per_core:      %d.", task_per_core);
    OP_LOGD(TilingContext, "core_used:     %d.", core_used);
    OP_LOGD(TilingContext, "task_tail_core:     %d.", task_tail_core);
}

static ge::graphStatus TilingChamferDistanceGrad(gert::TilingContext* context)
{
    ChamferDistanceGradTiling tilingObject(context);
    tilingObject.Init();
    return tilingObject.RunKernelTiling();
}

static ge::graphStatus TilingPrepareForChamferDistanceGrad(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForChamferDistanceGrad start.");
    auto compileInfo = context->GetCompiledInfo<ChamferDistanceGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->totalCoreNum <= 0), OP_LOGE(context->GetNodeName(), "Failed to get core num."),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."),
        return ge::GRAPH_FAILED);
    OP_LOGD(context, "TilingPrepareForChamferDistanceGrad end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChamferDistanceGrad)
    .Tiling(TilingChamferDistanceGrad)
    .TilingParse<ChamferDistanceGradCompileInfo>(TilingPrepareForChamferDistanceGrad);
} // namespace optiling