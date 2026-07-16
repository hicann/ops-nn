/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dequantize_tiling_arch35.h"
#include "register/op_impl_registry.h"
#include "op_common/log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/soc_spec.h"
#include <graph/utils/type_utils.h>

using namespace ge;

namespace dequantize_ns {

static constexpr int64_t kMetadataPerNode = 128;

static constexpr float kInvRangeInt8 = 1.0f / 255.0f;                    // 2^8 - 1
static constexpr float kInvRangeUint8 = 1.0f / 255.0f;                   // 2^8 - 1
static constexpr float kInvRangeInt32MinCombined = 1.0f / 4294967295.0f; // 2^32 - 1
static constexpr float kInvRangeInt32Scaled = 1.0f / 2147483647.0f;      // 2^31 - 1
static constexpr float kInvRangeInt8Scaled = 1.0f / 127.0f;              // 2^7 - 1
static constexpr float kBiasInt8 = 128.0f;                               // 2^(8-1)
static constexpr float kBiasUint8 = 0.0f;                                // unsigned, no bias
static constexpr float kBiasInt32 = 2147483648.0f;                       // 2^(32-1)

bool CheckBroadcastShape(const std::vector<std::vector<int64_t>>& padded_in,
                         const std::vector<std::vector<int64_t>>& padded_out, int64_t max_rank)
{
    for (int64_t d = 0; d < max_rank; d++) {
        int64_t ref = -3;
        for (size_t i = 0; i < padded_in.size(); i++) {
            if (padded_in[i][d] != 1) {
                if (ref == -3)
                    ref = padded_in[i][d];
                else if (padded_in[i][d] != ref)
                    return false;
            }
        }
        for (size_t i = 0; i < padded_out.size(); i++) {
            if (padded_out[i][d] != 1) {
                if (ref == -3)
                    ref = padded_out[i][d];
                else if (padded_out[i][d] != ref)
                    return false;
            }
        }
    }
    return true;
}

void SqueezeBroadcastDims(const std::vector<std::vector<int64_t>>& padded_in, int64_t num_inputs,
                          const std::vector<std::vector<int64_t>>& padded_out, int64_t num_outputs, int64_t max_rank,
                          std::vector<int64_t>& max_shape, std::vector<std::vector<int64_t>>& normal_in,
                          std::vector<std::vector<int64_t>>& normal_out)
{
    max_shape.clear();
    normal_in.assign(num_inputs, std::vector<int64_t>());
    normal_out.assign(num_outputs, std::vector<int64_t>());
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
            max_shape.push_back(max_dim);
            for (int64_t i = 0; i < num_inputs; i++)
                normal_in[i].push_back(padded_in[i][d]);
            for (int64_t i = 0; i < num_outputs; i++)
                normal_out[i].push_back(padded_out[i][d]);
        }
    }
    if (max_shape.empty()) {
        max_shape.push_back(1);
        for (int64_t i = 0; i < num_inputs; i++)
            normal_in[i].push_back(1);
        for (int64_t i = 0; i < num_outputs; i++)
            normal_out[i].push_back(1);
    }
}

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

    SqueezeBroadcastDims(padded_in, num_inputs, padded_out, num_outputs, max_rank, maximum_bro_shape,
                         normal_input_shapes, normal_output_shapes);
    return true;
}

bool FindSplitAxis(const std::vector<int64_t>& max_bro_shape, int64_t dtype_size, int64_t ub_per_core,
                   int64_t phys_nodes, SplitResult& out)
{
    int64_t metadata = phys_nodes * kMetadataPerNode;
    int64_t per_buf_bytes = ((ub_per_core - metadata) / phys_nodes) & ~31LL;
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

bool MultiCoreSplit(const std::vector<int64_t>& max_bro_shape, const SplitResult& ub_split, int64_t max_cores,
                    MultiCoreResult& out)
{
    int64_t k = ub_split.axis, outer_prod = 1;
    for (int64_t j = 0; j < k; j++)
        outer_prod *= max_bro_shape[j];
    out.total_tiles = outer_prod * ub_split.a_o;
    if (max_cores <= 0)
        return false;
    out.num_cores = (out.total_tiles < max_cores) ? out.total_tiles : max_cores;
    out.tiles_main = out.total_tiles / out.num_cores;
    out.cores_tail = out.total_tiles % out.num_cores;
    return true;
}

bool PrecomputeStrides(const std::vector<int64_t>& s, std::vector<int64_t>& strides)
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

} // namespace dequantize_ns

namespace optiling {

using namespace dequantize_ns;

static std::string Arr2String(const int64_t* arr, int64_t n)
{
    std::ostringstream oss;
    oss << "[";
    if (n > 0) {
        for (int64_t i = 0; i < n - 1; ++i)
            oss << arr[i] << ",";
        oss << arr[n - 1];
    }
    oss << "]";
    return oss.str();
}

DequantizeTiling::DequantizeTiling(gert::TilingContext* ctx) : ctx_(ctx) {}

ge::graphStatus DequantizeTiling::ReadTensorShapes(bool is_input, std::vector<std::vector<int64_t>>& raw_shapes)
{
    size_t num = is_input ? ctx_->GetComputeNodeInfo()->GetInputsNum() : ctx_->GetComputeNodeInfo()->GetOutputsNum();
    const char* tag = is_input ? "input" : "output";
    for (size_t i = 0; i < num; ++i) {
        auto shape = is_input ? ctx_->GetInputShape(i) : ctx_->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(ctx_, shape);
        std::vector<int64_t> dims;
        gert::Shape s = shape->GetStorageShape();
        if (s.GetDimNum() == 0) {
            dims.push_back(1);
        } else {
            for (size_t d = 0; d < s.GetDimNum(); ++d) {
                int64_t dim = s.GetDim(d);
                OP_CHECK_IF(
                    dim == 0,
                    OP_LOGE(ctx_->GetNodeName(), "%s %zu dim %zu is zero, empty tensor not supported", tag, i, d),
                    return ge::GRAPH_FAILED);
                dims.push_back(dim);
            }
        }
        raw_shapes.push_back(dims);
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus DequantizeTiling::GetShapeInfo()
{
    auto compileInfo = static_cast<const DequantizeCompileInfo*>(ctx_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, compileInfo);

    if (ReadTensorShapes(true, raw_input_shapes_) != GRAPH_SUCCESS)
        return GRAPH_FAILED;
    if (ReadTensorShapes(false, raw_output_shapes_) != GRAPH_SUCCESS)
        return GRAPH_FAILED;

    ge::graphStatus ret = ResolveDtype();
    if (ret != GRAPH_SUCCESS)
        return ret;

    mode_ = "MIN_COMBINED";
    auto attrs = ctx_->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() > 0) {
        const char* mode_attr = attrs->GetStr(0);
        if (mode_attr != nullptr) {
            mode_ = std::string(mode_attr);
        }
    }

    PrecomputeConstants();

    PadAndSqueeze(raw_input_shapes_, raw_output_shapes_, max_bro_shape_, normal_input_shapes_, normal_output_shapes_);
    rank_ = (int64_t)max_bro_shape_.size();

    OP_LOGI(ctx_->GetNodeName(), "GetShapeInfo done rank %ld dtype_size %ld mode %s ub %lu core %lu", rank_,
            dtype_size_, mode_.c_str(), compileInfo->ubSize, compileInfo->coreNum);

    OP_CHECK_IF(!CheckBroadcastShape(normal_input_shapes_, normal_output_shapes_, rank_),
                OP_LOGE(ctx_->GetNodeName(), "check broadcast shape failed"), return ge::GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

ge::graphStatus DequantizeTiling::ResolveDtype()
{
    auto inputDesc = ctx_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    if (dtype == ge::DT_INT8) {
        dtype_size_ = 1;
        dtype_x_ = 0;
    } else if (dtype == ge::DT_UINT8) {
        dtype_size_ = 1;
        dtype_x_ = 1;
    } else if (dtype == ge::DT_INT32) {
        dtype_size_ = 4;
        dtype_x_ = 2;
    } else {
        OP_LOGE(ctx_->GetNodeName(), "Unsupported input dtype");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

void DequantizeTiling::PrecomputeConstants()
{
    if (mode_ == "MIN_COMBINED" || mode_ == "MIN_FIRST") {
        if (dtype_x_ == 0) {
            inv_range_ = kInvRangeInt8;
            bias_ = kBiasInt8;
        } else if (dtype_x_ == 1) {
            inv_range_ = kInvRangeUint8;
            bias_ = kBiasUint8;
        } else {
            inv_range_ = kInvRangeInt32MinCombined;
            bias_ = kBiasInt32;
        }
    } else {
        bias_ = 0.0f;
        if (dtype_x_ == 0) {
            inv_range_ = kInvRangeInt8Scaled;
        } else if (dtype_x_ == 1) {
            inv_range_ = kInvRangeUint8;
        } else {
            inv_range_ = kInvRangeInt32Scaled;
        }
    }
}

template <int64_t R>
void DequantizeTiling::FillInputTilingData(DequantizeTilingData<R>* tiling, int64_t num_in, int64_t delta,
                                           const std::vector<std::vector<int64_t>>& in_strides)
{
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
    for (int64_t i = num_in; i < kMaxInputSlots; i++)
        for (int64_t d = 0; d < R; d++) {
            tiling->input_shapes[i][d] = 1;
            tiling->input_strides[i][d] = 0;
        }
}

template <int64_t R>
void DequantizeTiling::FillOutputTilingData(DequantizeTilingData<R>* tiling, int64_t num_out, int64_t delta,
                                            const std::vector<std::vector<int64_t>>& out_strides)
{
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
    for (int64_t i = num_out; i < kMaxOutputSlots; i++)
        for (int64_t d = 0; d < R; d++) {
            tiling->output_shapes[i][d] = 1;
            tiling->output_strides[i][d] = 0;
        }
}

template <int64_t R>
ge::graphStatus DequantizeTiling::DoTilingAndSet()
{
    auto* tiling = ctx_->GetTilingData<DequantizeTilingData<R>>();
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, tiling);

    auto* compileInfo = static_cast<const DequantizeCompileInfo*>(ctx_->GetCompileInfo());
    int64_t ub_per_core = (int64_t)compileInfo->ubSize;
    int64_t metadata = kPhysNodes * kMetadataPerNode;
    int64_t per_buf_bytes = ((ub_per_core - metadata) / kPhysNodes) & ~31LL;

    FindSplitAxis(max_bro_shape_, sizeof(float), ub_per_core, kPhysNodes, tiling->split);
    if (!MultiCoreSplit(max_bro_shape_, tiling->split, (int64_t)compileInfo->coreNum, tiling->multicore)) {
        OP_LOGE(ctx_->GetNodeName(), "MultiCoreSplit failed: total_tiles is zero");
        return ge::GRAPH_FAILED;
    }
    tiling->per_buf_bytes = per_buf_bytes;

    int64_t num_in = (int64_t)normal_input_shapes_.size();
    int64_t num_out = (int64_t)normal_output_shapes_.size();
    std::vector<std::vector<int64_t>> in_strides(num_in), out_strides(num_out);
    for (int64_t i = 0; i < num_in; i++)
        PrecomputeStrides(normal_input_shapes_[i], in_strides[i]);
    for (int64_t i = 0; i < num_out; i++)
        PrecomputeStrides(normal_output_shapes_[i], out_strides[i]);

    tiling->rank = rank_;
    int64_t delta = R - rank_;

    for (int64_t d = 0; d < delta; d++)
        tiling->max_bro_shape[d] = 1;
    for (int64_t d = 0; d < rank_; d++)
        tiling->max_bro_shape[d + delta] = max_bro_shape_[d];

    tiling->split.axis += delta;

    tiling->num_inputs = num_in;
    tiling->num_outputs = num_out;

    FillInputTilingData<R>(tiling, num_in, delta, in_strides);
    FillOutputTilingData<R>(tiling, num_out, delta, out_strides);

    tiling->bias = bias_;
    tiling->inv_range = inv_range_;

    ctx_->SetBlockDim(tiling->multicore.num_cores);

    OP_LOGI(ctx_->GetNodeName(),
            "TilingData: per_buf_bytes=%ld rank=%ld->R=%d "
            "max_bro_shape=%s bias=%f inv_range=%f "
            "split(axis=%ld a_i=%ld a_o=%ld a_i_tail=%ld) "
            "multi(cores=%ld tiles=%ld main=%ld core_tail=%ld)",
            tiling->per_buf_bytes, rank_, (int)R, Arr2String(tiling->max_bro_shape, R).c_str(), tiling->bias,
            tiling->inv_range, tiling->split.axis, tiling->split.a_i, tiling->split.a_o, tiling->split.a_i_tail,
            tiling->multicore.num_cores, tiling->multicore.total_tiles, tiling->multicore.tiles_main,
            tiling->multicore.cores_tail);

    return GRAPH_SUCCESS;
}

ge::graphStatus DequantizeTiling::RunTiling()
{
    ge::graphStatus ret = GetShapeInfo();
    if (ret != GRAPH_SUCCESS)
        return ret;

    int64_t mapped = (rank_ <= 4) ? 4 : 8;
    int64_t mode_num = 0;
    if (mode_ == "MIN_FIRST")
        mode_num = DEQUANTIZE_MODE_MIN_FIRST;
    else if (mode_ == "SCALED")
        mode_num = DEQUANTIZE_MODE_SCALED;
    else
        mode_num = DEQUANTIZE_MODE_MIN_COMBINED;

    if (mapped == 4) {
        ret = DoTilingAndSet<4>();
        uint64_t tiling_key = 0;
        if (mode_num == DEQUANTIZE_MODE_MIN_COMBINED)
            tiling_key = GET_TPL_TILING_KEY(DEQUANTIZE_MODE_MIN_COMBINED, DEQUANTIZE_RANK_4);
        else if (mode_num == DEQUANTIZE_MODE_MIN_FIRST)
            tiling_key = GET_TPL_TILING_KEY(DEQUANTIZE_MODE_MIN_FIRST, DEQUANTIZE_RANK_4);
        else
            tiling_key = GET_TPL_TILING_KEY(DEQUANTIZE_MODE_SCALED, DEQUANTIZE_RANK_4);
        ctx_->SetTilingKey(tiling_key);
    } else {
        ret = DoTilingAndSet<8>();
        uint64_t tiling_key = 0;
        if (mode_num == DEQUANTIZE_MODE_MIN_COMBINED)
            tiling_key = GET_TPL_TILING_KEY(DEQUANTIZE_MODE_MIN_COMBINED, DEQUANTIZE_RANK_8);
        else if (mode_num == DEQUANTIZE_MODE_MIN_FIRST)
            tiling_key = GET_TPL_TILING_KEY(DEQUANTIZE_MODE_MIN_FIRST, DEQUANTIZE_RANK_8);
        else
            tiling_key = GET_TPL_TILING_KEY(DEQUANTIZE_MODE_SCALED, DEQUANTIZE_RANK_8);
        ctx_->SetTilingKey(tiling_key);
    }
    return ret;
}

static ge::graphStatus TilingFuncDequantize(gert::TilingContext* context)
{
    DequantizeTiling tiling(context);
    auto ret = tiling.RunTiling();
    if (ret != GRAPH_SUCCESS)
        return ret;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = sysWorkspaceSize;
    return GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForDequantize(gert::TilingParseContext* context)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    auto compileInfo = context->GetCompiledInfo<DequantizeCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto ap = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ap.GetCoreNumAiv();
    ap.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Dequantize)
    .Tiling(TilingFuncDequantize)
    .TilingParse<DequantizeCompileInfo>(TilingPrepareForDequantize);

} // namespace optiling
