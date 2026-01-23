/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_index_add_tiling.cpp
 * \brief
 */
#include "inplace_index_add_tiling.h"
#include <math.h>
#include <string>
#include "inplace_index_add_tiling_arch35.h"
#include "util/platform_util.h"

using namespace ge;

namespace optiling {
static const int64_t SOC_VERSION_MULTI_ATOMIC_ADD = 1;
static const int64_t SUPPORT_ATOMIC_ADD = 1;
static const int64_t ASSIGNED_UB_SIZE = 2048;
static const int64_t ONE_BLOCK_SIZE = 32;
static const int64_t ALIGN_SIZE = 64;
static const int64_t INDEX_ATTR_AXIS = 0;
static const int64_t TENSOR_NUM = 2;

static int64_t GetCeilInt(int64_t value1, int64_t value2) {
  OP_CHECK_IF(value2 == 0,
                  OP_LOGE("InplaceIndexAdd",
                                                  "In the GetCeilInt function, the divisor is 0."),
                  return value1);
  return (value1 + value2 - 1) / value2;
}

static void PrintTilingParams(const gert::TilingContext* context, const InplaceIndexAddTilingData* tiling_params) {
  OP_LOGD(context->GetNodeName(), "block_num=%ld.", tiling_params->block_num);
  OP_LOGD(context->GetNodeName(), "indices_num=%ld.", tiling_params->indices_num);
  OP_LOGD(context->GetNodeName(), "outer_loop=%ld.", tiling_params->outer_loop);
  OP_LOGD(context->GetNodeName(), "full_num_per_block=%ld.", tiling_params->full_num_per_block);
  OP_LOGD(context->GetNodeName(), "tail_num=%ld.", tiling_params->tail_num);
  OP_LOGD(context->GetNodeName(), "axis_and_after_data_num_updates=%ld.",
          tiling_params->axis_and_after_data_num_updates);
  OP_LOGD(context->GetNodeName(), "axis_and_after_data_num_var=%ld.", tiling_params->axis_and_after_data_num_var);
  OP_LOGD(context->GetNodeName(), "update_data_num=%ld.", tiling_params->update_data_num);
  OP_LOGD(context->GetNodeName(), "axis=%ld.", tiling_params->axis);
  OP_LOGD(context->GetNodeName(), "updates_ub_size=%ld.", tiling_params->updates_ub_size);
  OP_LOGD(context->GetNodeName(), "indices_ub_size=%ld.", tiling_params->indices_ub_size);
  OP_LOGD(context->GetNodeName(), "tiling_core_num=%ld.", tiling_params->tiling_core_num);
  OP_LOGD(context->GetNodeName(), "var_shape_num=%ld.", tiling_params->var_shape_num);
  OP_LOGD(context->GetNodeName(), "updates_shape_num=%ld.", tiling_params->updates_shape_num);
}

static void CalRunningCaseGeneral(InplaceIndexAddTilingData* params) {
  params->block_num = 1;
  params->full_num_per_block = params->outer_loop;
}

static void CalRunningCaseBF16(InplaceIndexAddTilingData* params) {
  params->block_num = 1;
  params->full_num_per_block = params->indices_num;
}

static bool CalRunningCaseAtomicAdd(const InplaceIndexAddCompileInfo *compile_info,
                                    InplaceIndexAddTilingData* params) {
  if (params->indices_num < compile_info->core_num) {
    params->block_num = 1;
    params->full_num_per_block = params->indices_num;
    params->tail_num = params->full_num_per_block;
  } else {
    params->full_num_per_block = GetCeilInt(params->indices_num, compile_info->core_num);
    params->block_num = GetCeilInt(params->indices_num, params->full_num_per_block);
    params->tail_num = params->full_num_per_block;
    OP_CHECK_IF(params->block_num == 0,
                  OP_LOGE("InplaceIndexAdd",
                                                  "block num cannot be 0."),
                  return false);
    if (params->indices_num % params->block_num != 0) {
      params->tail_num = params->indices_num % params->full_num_per_block;
    }
  }
  return true;
}

static bool InitRunningInfo(InplaceIndexAddTilingData* params,
                            const gert::Shape &indices_shape_val, int32_t axis_val) {
  params->block_num = 0;
  params->indices_num = indices_shape_val.GetShapeSize();
  OP_CHECK_IF(params->indices_num == 0,
                  OP_LOGE("InplaceIndexAdd",
                                                  "indices num cannot be 0."),
                  return false);
  params->outer_loop = 1;
  params->full_num_per_block = 0;
  params->tail_num = 0;
  params->axis_and_after_data_num_updates = 1;
  params->axis_and_after_data_num_var = 1;
  params->update_data_num = 1;
  params->axis = axis_val;
  params->updates_ub_size = ASSIGNED_UB_SIZE;
  params->indices_ub_size = ASSIGNED_UB_SIZE;
  params->tiling_core_num = 0;
  params->var_shape_num = 0;
  params->updates_shape_num = 0;
  return true;
}

static bool CalRunningInfo(const InplaceIndexAddCompileInfo *compile_info,
                           InplaceIndexAddTilingData* params,
                           const gert::Shape &var_shape_val, const gert::Shape &updates_shape_val,
                           int64_t axis_val, ge::DataType dtype) {
  for (int64_t i = 0; i < axis_val; i++) {
    params->outer_loop = params->outer_loop * var_shape_val.GetDim(i);
  }

  if (axis_val == 0) {
    params->outer_loop = params->outer_loop * var_shape_val.GetDim(0);
  }

  const int64_t updates_dims = updates_shape_val.GetDimNum();
  for (int64_t i = axis_val; i < updates_dims; i++) {
    params->axis_and_after_data_num_updates =
      params->axis_and_after_data_num_updates * updates_shape_val.GetDim(i);
  }

  const int64_t var_dims = var_shape_val.GetDimNum();
  for (int64_t i = axis_val; i < var_dims; i++) {
    params->axis_and_after_data_num_var = params->axis_and_after_data_num_var * var_shape_val.GetDim(i);
  }
  for (int64_t i = axis_val + 1; i < var_dims; i++) {
    params->update_data_num = params->update_data_num * var_shape_val.GetDim(i);
  }

  params->tiling_core_num = compile_info->core_num;
  params->var_shape_num = static_cast<int64_t>(var_shape_val.GetShapeSize());
  params->updates_shape_num = static_cast<int64_t>(updates_shape_val.GetShapeSize());

  int64_t soc_version = compile_info->soc_version;
  int64_t atomic_add = compile_info->atomic_add;
  if (dtype == DT_BF16) {
    CalRunningCaseBF16(params);
  } else if ((soc_version == SOC_VERSION_MULTI_ATOMIC_ADD) && (atomic_add == SUPPORT_ATOMIC_ADD)) {
    OP_CHECK_IF(!CalRunningCaseAtomicAdd(compile_info, params),
                    OP_LOGE("InplaceIndexAdd", "CalRunningCaseAtomicAdd failed."),
                    return false);
    params->updates_ub_size =
      (compile_info->ub_size - params->indices_ub_size * static_cast<int64_t>(sizeof(int64_t))) / (TENSOR_NUM * static_cast<int64_t>(sizeof(float)));
    params->updates_ub_size = GetCeilInt(params->updates_ub_size, ALIGN_SIZE) * ALIGN_SIZE;
  } else {
    CalRunningCaseGeneral(params);
  }
  return true;
}

ge::graphStatus Tiling4InplaceIndexAdd(gert::TilingContext* context) {
  (void) context;
  return InplaceIndexAddTilingForAscendC(context);
  
}

ge::graphStatus TilingPrepareInplaceIndexAddForAscendC(gert::TilingParseContext *context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareInplaceIndexAddForAscendC entering.");
    auto compileInfo = context->GetCompiledInfo<InplaceIndexAddCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0),
        OP_LOGE(context->GetNodeName(), "Failed to core num."), return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0),
        OP_LOGE(context->GetNodeName(), "Failed to get ub size."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4InplaceIndexAdd(gert::TilingParseContext* context) {
  (void) context;
  return TilingPrepareInplaceIndexAddForAscendC(context);
}

// register tiling interface of the InplaceIndexAdd op.
IMPL_OP_OPTILING(InplaceIndexAdd)
    .Tiling(Tiling4InplaceIndexAdd)
    .TilingParse<InplaceIndexAddCompileInfo>(TilingPrepare4InplaceIndexAdd);
}  // namespace optiling
