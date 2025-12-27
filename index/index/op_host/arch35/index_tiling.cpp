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
 * \file index.cc
 * \brief index tiling file
 */

#include "tiling_base/tiling_templates_registry.h"
#include "platform/platform_info.h"
#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "register/op_def_registry.h"
#include "index_tiling.h"
#include "index_tiling_arch35.h"

using namespace ge;
using namespace AscendC;
using Ops::NN::Optiling::TilingRegistry;
namespace {
constexpr int64_t INPUT_IDX_X = 0;
constexpr int64_t INPUT_IDX_INDEXED_SIZES = 1;
constexpr int64_t INPUT_IDX_INDICES = 3;
constexpr int64_t TILING_MODE_FIRST_AXIS = 0;
constexpr int64_t TILING_MODE_NON_FIRST_AXIS = 1;
constexpr int64_t TILING_MODE_NON_FIRST_AXIS_2D = 2;
constexpr int64_t BYTES_PER_BLOCK = 32;
constexpr int64_t DIM_MAX = 10;
constexpr int64_t DATA_LIMIT_MULTI_INDEX = 500000;
const std::string OP_NAME = "Index";
constexpr int64_t TWO_DIMS = 2;
} // namespace

namespace optiling {
static void PrintInfo(const IndexTilingData* params)
{
    for (size_t idx = 0; idx < DIM_MAX; idx++) {
        OP_LOGD(OP_NAME.c_str(), "op [TilingData] : input_batch_num=%ld.", params->batch_num_list[idx]);
    }
    OP_LOGD(OP_NAME.c_str(), "op [TilingData] : core_num_var=%ld.", params->core_num_var);
    OP_LOGD(OP_NAME.c_str(), "op [TilingData] : output_size=%ld.", params->output_size);
    OP_LOGD(OP_NAME.c_str(), "op [TilingData] : output_num_per_batch=%ld.", params->output_num_per_batch);
    OP_LOGD(OP_NAME.c_str(), "op [TilingData] : output_batch_num=%ld.", params->output_batch_num);
    OP_LOGD(OP_NAME.c_str(), "op [TilingData] : input_size=%ld.", params->input_size);
    OP_LOGD(OP_NAME.c_str(), "op [TilingData] : input_shape_dim_num=%ld.", params->input_shape_dim_num);
    OP_LOGD(OP_NAME.c_str(), "op [TilingData] : index_size=%ld.", params->index_size);
    OP_LOGD(OP_NAME.c_str(), "op [TilingData] : masks_num=%ld.", params->masks_num);
    OP_LOGD(OP_NAME.c_str(), "op [TilingData] : tiling_mode=%ld.", params->tiling_mode);
}

static ge::graphStatus IsAiCore(IndexTilingData* params, const IndexCompileInfo* compile_info)
{
    if (compile_info->indices_num < params->masks_num) {
        if (params->tiling_mode == TILING_MODE_NON_FIRST_AXIS_2D) {
            // This is the UB size judgment in the non-first axis case
            if (compile_info->available_size <
                (compile_info->task_align) * (params->batch_num_list[1] + params->index_size)) {
                OP_LOGW(OP_NAME.c_str(), "there is not enough space in the UB, execute the AICPU engine.");
                return ge::GRAPH_FAILED;
            }
        } else if (params->tiling_mode != TILING_MODE_NON_FIRST_AXIS) {
            OP_LOGW(OP_NAME.c_str(), "tail size is too small for non_first_axis case, execute the AICPU engine.");
            return ge::GRAPH_FAILED;
        }
    }

    // full_axis case only supports output size < 50W
    if (compile_info->after_v200 && params->output_size > DATA_LIMIT_MULTI_INDEX &&
        compile_info->indices_num == params->input_shape_dim_num) {
        OP_LOGW(OP_NAME.c_str(), "in multiple index mode, compute size is too much, execute the AICPU engine.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4Index(gert::TilingContext* context)
{
    OP_LOGD(OP_NAME.c_str(), "Tiling4Index rt2.0 is running.");
    auto compile_info = reinterpret_cast<const IndexCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

    OP_LOGD(context->GetNodeName(), "Tiling4Index dsl compile_info is Null, running Simt tiling.");
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareIndexForAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start init IndexSimtTiling.");
    auto ci = context->GetCompiledInfo<IndexCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, ci);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ci->core_num = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((ci->core_num <= 0), OP_LOGE(context->GetNodeName(), "Failed to core num."), return ge::GRAPH_FAILED);
    uint64_t indexUbSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, indexUbSize);
    ci->ubSize = static_cast<int64_t>(indexUbSize);
    OP_CHECK_IF((ci->ubSize <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4Index(gert::TilingParseContext* context)
{
    auto compile_info = context->GetCompiledInfo<IndexCompileInfo>();
    OP_CHECK_IF(compile_info == nullptr, OP_LOGE(OP_NAME, "compile_info is nullptr!"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "AscendC TilingPrepare4Index Simt Mode success!");
    TilingPrepareIndexForAscendC(context);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Index)
    .Tiling(Tiling4Index)
    .TilingInputsDataDependency({1})
    .TilingParse<IndexCompileInfo>(TilingPrepare4Index);
} // namespace optiling
