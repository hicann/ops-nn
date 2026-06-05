/*
 * Copyright (c) 2026 联通（广东）产业互联网有限公司.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "intern_vl_add_rms_norm_tiling.h"
#include "opdev/op_log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {


struct DtypeInfo {
    uint32_t dtype_size;
    uint64_t tiling_key;
};

static bool GetDtypeInfo(ge::DataType dtype, DtypeInfo& dtype_info)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
            dtype_info = {2, 10};
            return true;
        case ge::DT_FLOAT:
            dtype_info = {4, 20};
            return true;
        case ge::DT_BF16:
            dtype_info = {2, 30};
            return true;
        default:
            return false;
    }
}

static float GetEpsilon(gert::TilingContext* context)
{
    float eps = 1e-6f;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() > 0) {
        auto eps_attr = attrs->GetFloat(0);
        if (eps_attr != nullptr) {
            eps = *eps_attr;
        }
    }
    return eps;
}

static uint32_t CalcMaxTileSize(uint32_t dtype_size)
{
    constexpr uint32_t UB_SIZE_LIMIT = 128 * 1024;
    constexpr uint32_t MIN_TILE_SIZE = 128;
    uint32_t ub_bytes_per_elem = 10 * dtype_size + 3 * sizeof(float);
    uint32_t max_tile_size = (UB_SIZE_LIMIT / ub_bytes_per_elem / MIN_TILE_SIZE) * MIN_TILE_SIZE;
    return max_tile_size < MIN_TILE_SIZE ? MIN_TILE_SIZE : max_tile_size;
}

ge::graphStatus InternVlAddRmsNormTiling(gert::TilingContext* context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aiv_core_num = platform.GetCoreNumAiv();
    uint32_t block_num = aiv_core_num == 0 ? 1 : static_cast<uint32_t>(aiv_core_num);

    auto x_shape = context->GetInputShape(0);
    auto dim_num = x_shape->GetOriginShape().GetDimNum();

    if (dim_num < 2) {
        OP_LOGE(static_cast<int32_t>(ge::GRAPH_FAILED),
            "InternVlAddRmsNorm node %s dim_num < 2.", context->GetNodeName());
        return ge::GRAPH_FAILED;
    }

    uint32_t batch = 1;
    for (size_t i = 0; i < dim_num - 1; ++i) {
        batch *= static_cast<uint32_t>(x_shape->GetOriginShape().GetDim(i));
    }

    uint32_t hidden_size = static_cast<uint32_t>(x_shape->GetOriginShape().GetDim(dim_num - 1));

    auto dtype = context->GetInputDesc(0)->GetDataType();
    DtypeInfo dtype_info{};
    if (!GetDtypeInfo(dtype, dtype_info)) {
        OP_LOGE(static_cast<int32_t>(ge::GRAPH_FAILED),
            "InternVlAddRmsNorm node %s does not support dtype %d.",
            context->GetNodeName(), static_cast<int32_t>(dtype));
        return ge::GRAPH_FAILED;
    }
    float eps = GetEpsilon(context);
    uint32_t max_tile_size = CalcMaxTileSize(dtype_info.dtype_size);

    uint32_t tile_size = hidden_size <= max_tile_size ? hidden_size : max_tile_size;
    uint32_t tile_num = (hidden_size + tile_size - 1) / tile_size;

    uint32_t total_tokens = batch;
    uint32_t tokens_per_core = (total_tokens + block_num - 1) / block_num;

    InternVlAddRmsNormTilingData tiling;
    tiling.set_batch(batch);
    tiling.set_hidden_size(hidden_size);
    tiling.set_tile_size(tile_size);
    tiling.set_tile_num(tile_num);
    tiling.set_dtype_size(dtype_info.dtype_size);
    tiling.set_total_size(static_cast<uint64_t>(batch) * hidden_size * dtype_info.dtype_size);
    tiling.set_eps(eps);
    tiling.set_block_num(block_num);
    tiling.set_tokens_per_core(tokens_per_core);

    auto raw_tiling_data = context->GetRawTilingData();
    tiling.SaveToBuffer(raw_tiling_data->GetData(), raw_tiling_data->GetCapacity());
    raw_tiling_data->SetDataSize(tiling.GetDataSize());

    context->SetTilingKey(dtype_info.tiling_key);
    context->SetBlockDim(block_num);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(InternVlAddRmsNorm).Tiling(InternVlAddRmsNormTiling);

} // namespace optiling
