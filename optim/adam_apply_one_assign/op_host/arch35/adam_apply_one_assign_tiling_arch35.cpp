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
 * \file adam_apply_one_assign_tiling_arch35.cpp
 * \brief
 */
#include "adam_apply_one_assign_tiling_arch35.h"
#include "optim/adam_apply_one_assign/op_kernel/arch35/adam_struct.h"
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/soc_spec.h"

using namespace ge;

// ============================================================
// Tiling 函数模块 — namespace adam_apply_one_assign
// ============================================================

namespace adam_apply_one_assign {

// 5.1 PadAndSqueeze

bool PadAndSqueeze(const std::vector<std::vector<int64_t>>& input_shapes,
                   const std::vector<std::vector<int64_t>>& output_shapes, std::vector<int64_t>& maximum_bro_shape,
                   std::vector<std::vector<int64_t>>& normal_input_shapes,
                   std::vector<std::vector<int64_t>>& normal_output_shapes)
{
    int64_t num_inputs = (int64_t)input_shapes.size();
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
    for (int64_t i = 0; i < num_inputs; i++)
        padded_in[i] = pad(input_shapes[i]);
    for (int64_t i = 0; i < num_outputs; i++)
        padded_out[i] = pad(output_shapes[i]);
    maximum_bro_shape.clear();
    normal_input_shapes.assign(num_inputs, std::vector<int64_t>());
    normal_output_shapes.assign(num_outputs, std::vector<int64_t>());
    for (int64_t d = 0; d < max_rank; d++) {
        bool all_one = true;
        int64_t max_dim = 0;
        for (int64_t i = 0; i < num_inputs; i++) {
            if (padded_in[i][d] != 1)
                all_one = false;
            max_dim = std::max(max_dim, padded_in[i][d]);
        }
        for (int64_t i = 0; i < num_outputs; i++) {
            if (padded_out[i][d] != 1)
                all_one = false;
            max_dim = std::max(max_dim, padded_out[i][d]);
        }
        if (!all_one) {
            maximum_bro_shape.push_back(max_dim);
            for (int64_t i = 0; i < num_inputs; i++)
                normal_input_shapes[i].push_back(padded_in[i][d]);
            for (int64_t i = 0; i < num_outputs; i++)
                normal_output_shapes[i].push_back(padded_out[i][d]);
        }
    }
    if (maximum_bro_shape.empty()) {
        maximum_bro_shape.push_back(1);
        for (int64_t i = 0; i < num_inputs; i++)
            normal_input_shapes[i].push_back(1);
        for (int64_t i = 0; i < num_outputs; i++)
            normal_output_shapes[i].push_back(1);
    }
    return true;
}

// 5.2 FindSplitAxis

bool FindSplitAxis(const std::vector<int64_t>& max_bro_shape, int64_t dtype_size, int64_t ub_per_core,
                   int64_t phys_nodes, SplitResult& out)
{
    int64_t per_buf_bytes = (ub_per_core / phys_nodes) & ~31LL; // 32B 对齐, TBuf 硬件要求
    int64_t per_buf_elems = per_buf_bytes / dtype_size;
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

// 5.3 MultiCoreSplit

bool MultiCoreSplit(const std::vector<int64_t>& max_bro_shape, const SplitResult& ub_split, int64_t max_cores,
                    MultiCoreResult& out)
{
    int64_t k = ub_split.axis, outer_prod = 1;
    for (int64_t j = 0; j < k; j++)
        outer_prod *= max_bro_shape[j];
    out.total_tiles = outer_prod * ub_split.a_o;
    out.num_cores = (out.total_tiles < max_cores) ? out.total_tiles : max_cores;
    out.tiles_main = out.total_tiles / out.num_cores;
    out.cores_tail = out.total_tiles % out.num_cores;
    return true;
}

// 5.5 地址偏移计算

bool PrecomputeInputStrides(const std::vector<int64_t>& s, std::vector<int64_t>& strides)
{
    int64_t rank = (int64_t)s.size();
    strides.assign(rank, 0);
    for (int64_t d = rank - 1; d >= 0; d--) {
        if (s[d] == 1) {
            strides[d] = 0;
            continue;
        }
        int64_t prod = 1;
        for (int64_t j = d + 1; j < rank; j++)
            prod *= s[j];
        strides[d] = prod;
    }
    return true;
}

bool PrecomputeOutputStrides(const std::vector<int64_t>& s, std::vector<int64_t>& strides)
{
    int64_t rank = (int64_t)s.size();
    strides.assign(rank, 0);
    for (int64_t d = rank - 1; d >= 0; d--) {
        if (s[d] == 1) {
            strides[d] = 0;
            continue;
        }
        int64_t prod = 1;
        for (int64_t j = d + 1; j < rank; j++)
            prod *= s[j];
        strides[d] = prod;
    }
    return true;
}

} // namespace adam_apply_one_assign

// ============================================================
// AdamTiling — CANN 主线：单次归一化 → 映射 rank → 模板填充
// ============================================================

namespace optiling {

using namespace adam_apply_one_assign;

static std::string Arr2String(const int64_t* arr, int64_t n)
{
    std::ostringstream oss;
    oss << "[";
    if (n > 0) {
        for (int64_t i = 0; i < n - 1; ++i) {
            oss << arr[i] << ",";
        }
        oss << arr[n - 1];
    }
    oss << "]";
    return oss.str();
}

AdamTiling::AdamTiling(gert::TilingContext* ctx) : ctx_(ctx) {}

ge::graphStatus AdamTiling::GetShapeInfo()
{
    auto compileInfo = static_cast<const AdamCompileInfo*>(ctx_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, compileInfo);
    for (size_t i = 0; i < ctx_->GetComputeNodeInfo()->GetInputsNum(); ++i) {
        auto shape = ctx_->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(ctx_, shape);
        std::vector<int64_t> dims;
        gert::Shape s = shape->GetStorageShape();
        for (size_t d = 0; d < s.GetDimNum(); ++d)
            dims.push_back(s.GetDim(d));
        raw_input_shapes_.push_back(dims);
    }
    for (size_t i = 0; i < ctx_->GetComputeNodeInfo()->GetOutputsNum(); ++i) {
        auto shape = ctx_->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(ctx_, shape);
        std::vector<int64_t> dims;
        gert::Shape s = shape->GetStorageShape();
        for (size_t d = 0; d < s.GetDimNum(); ++d)
            dims.push_back(s.GetDim(d));
        raw_output_shapes_.push_back(dims);
    }
    auto inputDesc = ctx_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    if (dtype == ge::DT_FLOAT16)
        dtype_size_ = 2;
    else if (dtype == ge::DT_FLOAT)
        dtype_size_ = 4;
    else {
        OP_LOGE(ctx_->GetNodeName(), "Unsupported dtype");
        return GRAPH_FAILED;
    }

    PadAndSqueeze(raw_input_shapes_, raw_output_shapes_, max_bro_shape_, normal_input_shapes_, normal_output_shapes_);
    rank_ = (int64_t)max_bro_shape_.size();

    // 维测: 输入预处理结果
    OP_LOGI(ctx_->GetNodeName(), "GetShapeInfo done rank %ld dtype %ld ub %lu core %lu", rank_, dtype_size_,
            compileInfo->ubSize, compileInfo->coreNum);

    return GRAPH_SUCCESS;
}

template <int64_t R>
ge::graphStatus AdamTiling::DoTilingAndSet()
{
    auto* tiling = ctx_->GetTilingData<AdamTilingData<R>>();
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, tiling);

    auto* compileInfo = static_cast<const AdamCompileInfo*>(ctx_->GetCompileInfo());
    int64_t ub_per_core = (int64_t)compileInfo->ubSize;
    int64_t per_buf_bytes = (ub_per_core / kPhysNodes) & ~31LL;

    FindSplitAxis(max_bro_shape_, dtype_size_, ub_per_core, kPhysNodes, tiling->split);
    MultiCoreSplit(max_bro_shape_, tiling->split, (int64_t)compileInfo->coreNum, tiling->multicore);
    tiling->per_buf_bytes = per_buf_bytes;

    int64_t num_in = (int64_t)normal_input_shapes_.size();
    int64_t num_out = (int64_t)normal_output_shapes_.size();
    std::vector<std::vector<int64_t>> in_strides(num_in), out_strides(num_out);
    for (int64_t i = 0; i < num_in; i++)
        PrecomputeInputStrides(normal_input_shapes_[i], in_strides[i]);
    for (int64_t i = 0; i < num_out; i++)
        PrecomputeOutputStrides(normal_output_shapes_[i], out_strides[i]);

    tiling->rank = rank_;
    int64_t delta = R - rank_; // 前补维数

    // max_bro_shape: 前补 1，实际值右移
    for (int64_t d = 0; d < delta; d++)
        tiling->max_bro_shape[d] = 1;
    for (int64_t d = 0; d < rank_; d++)
        tiling->max_bro_shape[d + delta] = max_bro_shape_[d];

    // split axis 右平移
    tiling->split.axis += delta;

    tiling->num_inputs = num_in;
    tiling->num_outputs = num_out;

    // 每条 input: 前补 shape=1 stride=0，实际值右移
    for (int64_t i = 0; i < num_in; i++) {
        for (int64_t d = 0; d < delta; d++) {
            tiling->input_shapes[i][d] = 1;
            tiling->input_strides[i][d] = 0;
        }
        for (int64_t d = 0; d < rank_; d++) {
            tiling->input_shapes[i][d + delta] = normal_input_shapes_[i][d];
            tiling->input_strides[i][d + delta] = in_strides[i][d];
        }
    }
    // 未使用 input slot: 全填 1/0
    for (int64_t i = num_in; i < kMaxInputSlots; i++)
        for (int64_t d = 0; d < R; d++) {
            tiling->input_shapes[i][d] = 1;
            tiling->input_strides[i][d] = 0;
        }

    // 每条 output: 前补 shape=1 stride=0，实际值右移
    for (int64_t i = 0; i < num_out; i++) {
        for (int64_t d = 0; d < delta; d++) {
            tiling->output_shapes[i][d] = 1;
            tiling->output_strides[i][d] = 0;
        }
        for (int64_t d = 0; d < rank_; d++) {
            tiling->output_shapes[i][d + delta] = normal_output_shapes_[i][d];
            tiling->output_strides[i][d + delta] = out_strides[i][d];
        }
    }
    // 未使用 output slot: 全填 1/0
    for (int64_t i = num_out; i < kMaxOutputSlots; i++)
        for (int64_t d = 0; d < R; d++) {
            tiling->output_shapes[i][d] = 1;
            tiling->output_strides[i][d] = 0;
        }

    ctx_->SetBlockDim(tiling->multicore.num_cores);

    // 维测: TilingData 全部字段（从 struct 数组打印，含 R 维 padding）
    OP_LOGI(ctx_->GetNodeName(),
            "TilingData: per_buf_bytes=%ld rank=%ld->R=%d "
            "max_bro_shape=%s "
            "split(axis=%ld a_i=%ld a_o=%ld a_i_tail=%ld) "
            "multi(cores=%ld tiles=%ld main=%ld core_tail=%ld) num_in=%ld num_out=%ld",
            tiling->per_buf_bytes, rank_, (int)R, Arr2String(tiling->max_bro_shape, R).c_str(), tiling->split.axis,
            tiling->split.a_i, tiling->split.a_o, tiling->split.a_i_tail, tiling->multicore.num_cores,
            tiling->multicore.total_tiles, tiling->multicore.tiles_main, tiling->multicore.cores_tail, num_in, num_out);

    for (int64_t i = 0; i < num_in; i++)
        OP_LOGI(ctx_->GetNodeName(), "TilingData input[%ld]: shape=%s stride=%s", i,
                Arr2String(tiling->input_shapes[i], R).c_str(), Arr2String(tiling->input_strides[i], R).c_str());
    for (int64_t i = 0; i < num_out; i++)
        OP_LOGI(ctx_->GetNodeName(), "TilingData output[%ld]: shape=%s stride=%s", i,
                Arr2String(tiling->output_shapes[i], R).c_str(), Arr2String(tiling->output_strides[i], R).c_str());

    return GRAPH_SUCCESS;
}

ge::graphStatus AdamTiling::RunTiling()
{
    ge::graphStatus ret = GetShapeInfo();
    if (ret != GRAPH_SUCCESS)
        return ret;

    int64_t mapped = (rank_ <= 4) ? 4 : 8;
    if (mapped == 4) {
        ret = DoTilingAndSet<4>();
        ctx_->SetTilingKey(GET_TPL_TILING_KEY(ADAM_RANK_4));
    } else {
        ret = DoTilingAndSet<8>();
        ctx_->SetTilingKey(GET_TPL_TILING_KEY(ADAM_RANK_8));
    }
    return ret;
}

static ge::graphStatus TilingFuncAdam(gert::TilingContext* context)
{
    AdamTiling adamTiling(context);
    auto ret = adamTiling.RunTiling();
    if (ret != GRAPH_SUCCESS)
        return ret;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = sysWorkspaceSize;
    return GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForAdam(gert::TilingParseContext* context)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    auto compileInfo = context->GetCompiledInfo<AdamCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto ap = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ap.GetCoreNumAiv();
    ap.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AdamApplyOneAssign).Tiling(TilingFuncAdam).TilingParse<AdamCompileInfo>(TilingPrepareForAdam);

} // namespace optiling
