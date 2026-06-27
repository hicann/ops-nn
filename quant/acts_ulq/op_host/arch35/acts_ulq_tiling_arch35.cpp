/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#include "acts_ulq_tiling_arch35.h"
#include "../../op_kernel/arch35/acts_ulq_struct.h"
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "op_common/log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/soc_spec.h"

using namespace ge;

namespace acts_ulq {

bool CheckBroadcastShape(
    const std::vector<std::vector<int64_t>>& padded_in,
    const std::vector<std::vector<int64_t>>& padded_out,
    int64_t max_rank)
{
    for (int64_t d = 0; d < max_rank; d++) {
        int64_t ref = -1;
        for (size_t i = 0; i < padded_in.size(); i++) {
            if (padded_in[i][d] != 1) {
                if (ref == -1) ref = padded_in[i][d];
                else if (padded_in[i][d] != ref) {
                    OP_LOGE("CheckBroadcastShape", "dim %d broadcast incompatible", (int)d);
                    return false;
                }
            }
        }
        for (size_t i = 0; i < padded_out.size(); i++) {
            if (padded_out[i][d] != 1) {
                if (ref == -1) ref = padded_out[i][d];
                else if (padded_out[i][d] != ref) {
                    OP_LOGE("CheckBroadcastShape", "dim %d broadcast incompatible", (int)d);
                    return false;
                }
            }
        }
    }
    return true;
}

bool PadAndSqueeze(
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::vector<std::vector<int64_t>>& output_shapes,
    std::vector<int64_t>& maximum_bro_shape,
    std::vector<std::vector<int64_t>>& normal_input_shapes,
    std::vector<std::vector<int64_t>>& normal_output_shapes)
{
    int64_t num_inputs  = (int64_t)input_shapes.size();
    int64_t num_outputs = (int64_t)output_shapes.size();
    int64_t max_rank = 0;
    for (auto& s : input_shapes)
        max_rank = std::max(max_rank, (int64_t)s.size());
    for (auto& s : output_shapes)
        max_rank = std::max(max_rank, (int64_t)s.size());

    auto pad = [&](const std::vector<int64_t>& s) {
        std::vector<int64_t> p;
        p.assign(max_rank - (int64_t)s.size(), 1);
        p.insert(p.end(), s.begin(), s.end());
        return p;
    };
    std::vector<std::vector<int64_t>> padded_in(num_inputs), padded_out(num_outputs);
    for (int64_t i = 0; i < num_inputs; i++) padded_in[i]  = pad(input_shapes[i]);
    for (int64_t i = 0; i < num_outputs; i++) padded_out[i] = pad(output_shapes[i]);

    maximum_bro_shape.clear();
    normal_input_shapes.assign(num_inputs, std::vector<int64_t>());
    normal_output_shapes.assign(num_outputs, std::vector<int64_t>());
    for (int64_t d = 0; d < max_rank; d++) {
        bool all_one = true;
        int64_t max_dim = 0;
        for (int64_t i = 0; i < num_inputs; i++) {
            if (padded_in[i][d] != 1) all_one = false;
            max_dim = std::max(max_dim, padded_in[i][d]);
        }
        for (int64_t i = 0; i < num_outputs; i++) {
            if (padded_out[i][d] != 1) all_one = false;
            max_dim = std::max(max_dim, padded_out[i][d]);
        }
        if (!all_one) {
            maximum_bro_shape.push_back(max_dim);
            for (int64_t i = 0; i < num_inputs; i++) normal_input_shapes[i].push_back(padded_in[i][d]);
            for (int64_t i = 0; i < num_outputs; i++) normal_output_shapes[i].push_back(padded_out[i][d]);
        }
    }
    if (maximum_bro_shape.empty()) {
        maximum_bro_shape.push_back(1);
        for (int64_t i = 0; i < num_inputs; i++) normal_input_shapes[i].push_back(1);
        for (int64_t i = 0; i < num_outputs; i++) normal_output_shapes[i].push_back(1);
    }
    return true;
}

bool FindSplitAxis(const std::vector<int64_t>& max_bro_shape,
    int64_t ub_per_core, int64_t phys_nodes, SplitResult& out)
{
    int64_t per_buf_bytes = (ub_per_core / phys_nodes) & ~31LL;
    int64_t per_buf_elems = per_buf_bytes / sizeof(float);
    int64_t rank = (int64_t)max_bro_shape.size();
    int64_t inner = 1;
    for (int64_t k = rank - 1; k >= 0; k--) {
        if (max_bro_shape[k] * inner > per_buf_elems) {
            out.a_i = per_buf_elems / inner;
            out.a_o = (max_bro_shape[k] + out.a_i - 1) / out.a_i;
            int64_t rem = max_bro_shape[k] % out.a_i;
            out.a_i_tail = (rem == 0) ? out.a_i : rem;
            out.axis = k;
            return true;
        }
        if (k == 0) {
            out.axis = 0;
            out.a_i = max_bro_shape[0];
            out.a_o = 1;
            out.a_i_tail = max_bro_shape[0];
            return true;
        }
        inner *= max_bro_shape[k];
    }
    return true;
}

bool MultiCoreSplit(const std::vector<int64_t>& max_bro_shape,
    const SplitResult& ub_split, int64_t max_cores, MultiCoreResult& out)
{
    int64_t k = ub_split.axis, outer_prod = 1;
    for (int64_t j = 0; j < k; j++) outer_prod *= max_bro_shape[j];
    out.total_tiles = outer_prod * ub_split.a_o;
    out.num_cores   = (out.total_tiles < max_cores) ? out.total_tiles : max_cores;
    out.tiles_main  = out.total_tiles / out.num_cores;
    out.cores_tail  = out.total_tiles % out.num_cores;
    return true;
}

bool PrecomputeStrides(const std::vector<int64_t>& s, std::vector<int64_t>& strides) {
    int64_t rank = (int64_t)s.size();
    strides.assign(rank, 0);
    for (int64_t d = rank - 1; d >= 0; d--) {
        if (s[d] == 1) { strides[d] = 0; continue; }
        int64_t prod = 1;
        for (int64_t j = d + 1; j < rank; j++) prod *= s[j];
        strides[d] = prod;
    }
    return true;
}

} // namespace acts_ulq

namespace optiling {

using namespace acts_ulq;

static std::string Arr2String(const int64_t* arr, int64_t n)
{
    std::ostringstream oss;
    oss << "[";
    if (n > 0) {
        for (int64_t i = 0; i < n - 1; ++i) oss << arr[i] << ",";
        oss << arr[n - 1];
    }
    oss << "]";
    return oss.str();
}

ActsUlqTiling::ActsUlqTiling(gert::TilingContext* ctx) : ctx_(ctx) {}

ge::graphStatus ActsUlqTiling::GetShapeInfo()
{
    auto compileInfo = static_cast<const ActsUlqCompileInfo*>(ctx_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, compileInfo);
    for (size_t i = 0; i < ctx_->GetComputeNodeInfo()->GetInputsNum(); ++i) {
        auto shape = ctx_->GetInputShape(i); OP_CHECK_NULL_WITH_CONTEXT(ctx_, shape);
        std::vector<int64_t> dims;
        gert::Shape s = shape->GetStorageShape();
        for (size_t d = 0; d < s.GetDimNum(); ++d) dims.push_back(s.GetDim(d));
        raw_input_shapes_.push_back(dims);
    }
    for (size_t i = 0; i < ctx_->GetComputeNodeInfo()->GetOutputsNum(); ++i) {
        auto shape = ctx_->GetOutputShape(i); OP_CHECK_NULL_WITH_CONTEXT(ctx_, shape);
        std::vector<int64_t> dims;
        gert::Shape s = shape->GetStorageShape();
        for (size_t d = 0; d < s.GetDimNum(); ++d) dims.push_back(s.GetDim(d));
        raw_output_shapes_.push_back(dims);
    }

    PadAndSqueeze(raw_input_shapes_, raw_output_shapes_,
                  max_bro_shape_, normal_input_shapes_, normal_output_shapes_);
    rank_ = (int64_t)max_bro_shape_.size();

    OP_CHECK_IF(
        !CheckBroadcastShape(normal_input_shapes_, normal_output_shapes_, rank_),
        OP_LOGE(ctx_->GetNodeName(), "check broadcast shape failed"),
        return ge::GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

template<int64_t R>
ge::graphStatus ActsUlqTiling::DoTilingAndSet()
{
    auto* tiling = ctx_->GetTilingData<ActsUlqTilingData<R>>();
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, tiling);

    auto* compileInfo = static_cast<const ActsUlqCompileInfo*>(
        ctx_->GetCompileInfo());
    int64_t ub_per_core = (int64_t)compileInfo->ubSize;

    const gert::Tensor* inputTensor = ctx_->GetInputTensor(0);
    bool is_fp16 = (inputTensor != nullptr &&
                    inputTensor->GetDataType() == ge::DataType::DT_FLOAT16);
    int64_t phys_nodes = is_fp16 ? kTotalBufsFp16 : kPhysNodes;

    int64_t per_buf_bytes = (ub_per_core / phys_nodes) & ~31LL;

    FindSplitAxis(max_bro_shape_, ub_per_core, phys_nodes, tiling->split);
    MultiCoreSplit(max_bro_shape_, tiling->split, (int64_t)compileInfo->coreNum, tiling->multicore);
    tiling->per_buf_bytes = per_buf_bytes;
    tiling->per_buf_elems = per_buf_bytes / sizeof(float);

    int64_t num_in  = (int64_t)normal_input_shapes_.size();
    int64_t num_out = (int64_t)normal_output_shapes_.size();
    std::vector<std::vector<int64_t>> in_strides(num_in), out_strides(num_out);
    for (int64_t i = 0; i < num_in; i++) PrecomputeStrides(normal_input_shapes_[i], in_strides[i]);
    for (int64_t i = 0; i < num_out; i++) PrecomputeStrides(normal_output_shapes_[i], out_strides[i]);

    tiling->rank = rank_;
    int64_t delta = R - rank_;

    for (int64_t d = 0; d < delta; d++) tiling->max_bro_shape[d] = 1;
    for (int64_t d = 0; d < rank_; d++) tiling->max_bro_shape[d + delta] = max_bro_shape_[d];

    tiling->split.axis += delta;

    tiling->num_inputs  = num_in;
    tiling->num_outputs = num_out;

    for (int64_t i = 0; i < num_in; i++) {
        for (int64_t d = 0; d < delta; d++) {
            tiling->input_shapes[i][d]  = 1;
            tiling->input_strides[i][d] = 0;
        }
        for (int64_t d = 0; d < rank_; d++) {
            tiling->input_shapes[i][d + delta]  = normal_input_shapes_[i][d];
            tiling->input_strides[i][d + delta] = in_strides[i][d];
        }
    }
    for (int64_t i = num_in; i < kMaxInputSlots; i++)
        for (int64_t d = 0; d < R; d++) {
            tiling->input_shapes[i][d]  = 1;
            tiling->input_strides[i][d] = 0;
        }

    for (int64_t i = 0; i < num_out; i++) {
        for (int64_t d = 0; d < delta; d++) {
            tiling->output_shapes[i][d]  = 1;
            tiling->output_strides[i][d] = 0;
        }
        for (int64_t d = 0; d < rank_; d++) {
            tiling->output_shapes[i][d + delta]  = normal_output_shapes_[i][d];
            tiling->output_strides[i][d + delta] = out_strides[i][d];
        }
    }
    for (int64_t i = num_out; i < kMaxOutputSlots; i++)
        for (int64_t d = 0; d < R; d++) {
            tiling->output_shapes[i][d]  = 1;
            tiling->output_strides[i][d] = 0;
        }

    const gert::RuntimeAttrs* attrs = ctx_->GetAttrs();
    if (attrs != nullptr) {
        const bool* fixed_min_ptr = attrs->GetBool(0);
        tiling->fixed_min = (fixed_min_ptr != nullptr) ? static_cast<int64_t>(*fixed_min_ptr) : 0;
        const int64_t* num_bits_ptr = attrs->GetInt(1);
        tiling->num_bits = (num_bits_ptr != nullptr) ? *num_bits_ptr : 8;
    } else {
        tiling->fixed_min = 0;
        tiling->num_bits = 8;
    }

    ctx_->SetBlockDim(tiling->multicore.num_cores);

    return GRAPH_SUCCESS;
}

ge::graphStatus ActsUlqTiling::RunTiling()
{
    ge::graphStatus ret = GetShapeInfo();
    if (ret != GRAPH_SUCCESS) return ret;

    int64_t mapped = (rank_ <= 4) ? 4 : 8;
    if (mapped == 4) {
        ret = DoTilingAndSet<4>();
        ctx_->SetTilingKey(GET_TPL_TILING_KEY(ACTS_ULQ_RANK_4));
    } else {
        ret = DoTilingAndSet<8>();
        ctx_->SetTilingKey(GET_TPL_TILING_KEY(ACTS_ULQ_RANK_8));
    }
    return ret;
}

static ge::graphStatus TilingFuncActsUlq(gert::TilingContext* context)
{
    ActsUlqTiling tiling(context);
    auto ret = tiling.RunTiling();
    if (ret != GRAPH_SUCCESS) return ret;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = sysWorkspaceSize;
    return GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForActsUlq(gert::TilingParseContext* context)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    auto compileInfo = context->GetCompiledInfo<ActsUlqCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto ap = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ap.GetCoreNumAiv();
    ap.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ActsULQ).Tiling(TilingFuncActsUlq).TilingParse<ActsUlqCompileInfo>(TilingPrepareForActsUlq);

} // namespace optiling
