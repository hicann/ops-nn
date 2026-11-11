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
static constexpr int64_t kMaxRank = 8;
static constexpr int64_t kBroadcastDim = 1;
static constexpr int64_t kRank4 = 4;
static constexpr int64_t kRank8 = 8;
static constexpr int64_t kUbAlignBytes = 32;
static constexpr int64_t kUbAlignMask = ~(kUbAlignBytes - 1);
static constexpr int64_t kOutputDtypeSize = static_cast<int64_t>(sizeof(float));

static constexpr size_t kInputNum = 3;
static constexpr size_t kOutputNum = 1;
static constexpr size_t kRangeInputNum = 2;
static constexpr size_t kXIndex = 0;
static constexpr size_t kMinIndex = 1;
static constexpr size_t kMaxIndex = kMinIndex + kRangeInputNum - 1;
static constexpr size_t kYIndex = 0;
static constexpr size_t kWorkspaceNum = 1;
static constexpr size_t kModeAttrIndex = 0;

static constexpr const char* kInputNames[kInputNum] = {"x", "min_range", "max_range"};
static constexpr const char* kOutputName = "y";
static constexpr const char* kModeMinCombined = "MIN_COMBINED";
static constexpr const char* kModeMinFirst = "MIN_FIRST";
static constexpr const char* kModeScaled = "SCALED";

static constexpr float kDenomInt8 = 255.0f;               // 2^8 - 1
static constexpr float kDenomUint8 = 255.0f;              // 2^8 - 1
static constexpr float kDenomInt32 = 4294967296.0f;       // 2^32
static constexpr float kDenomScaledInt8 = 127.0f;         // 2^(8-1) - 1
static constexpr float kDenomScaledUint8 = 255.0f;        // 2^8 - 1
static constexpr float kDenomScaledInt32 = 2147483648.0f; // 2^31
static constexpr float kBiasInt8 = 128.0f;                // 2^(8-1)
static constexpr float kBiasUint8 = 0.0f;                 // unsigned, no bias
static constexpr float kBiasInt32 = 2147483648.0f;        // 2^(32-1)

static int64_t GetMetadataBytes() { return kPhysNodes * kMetadataPerNode; }

static bool IsSupportedMode(const std::string& mode)
{
    return mode == kModeMinCombined || mode == kModeMinFirst || mode == kModeScaled;
}

static bool ValidateBroadcastRank(const std::vector<std::vector<int64_t>>& in_shapes,
                                  const std::vector<std::vector<int64_t>>& out_shapes, int64_t max_rank,
                                  const char* op_name)
{
    if (in_shapes.empty()) {
        OP_LOGE(op_name, "CheckBroadcastShape failed: no input shapes");
        return false;
    }
    const auto& x_shape = in_shapes[kXIndex];
    if (static_cast<int64_t>(x_shape.size()) != max_rank) {
        OP_LOGE(op_name, "CheckBroadcastShape failed: x shape rank mismatch");
        return false;
    }
    for (size_t i = kMinIndex; i < in_shapes.size(); i++) {
        if (static_cast<int64_t>(in_shapes[i].size()) != max_rank) {
            OP_LOGE(op_name, "CheckBroadcastShape failed: input %zu shape rank mismatch", i);
            return false;
        }
    }
    for (size_t i = 0; i < out_shapes.size(); i++) {
        if (static_cast<int64_t>(out_shapes[i].size()) != max_rank) {
            OP_LOGE(op_name, "CheckBroadcastShape failed: output %zu shape rank mismatch", i);
            return false;
        }
    }
    return true;
}

static bool ValidateBroadcastDims(const std::vector<std::vector<int64_t>>& in_shapes,
                                  const std::vector<std::vector<int64_t>>& out_shapes, int64_t max_rank,
                                  const char* op_name)
{
    const auto& x_shape = in_shapes[kXIndex];
    for (int64_t d = 0; d < max_rank; d++) {
        const int64_t x_dim = x_shape[d];
        for (size_t i = kMinIndex; i < in_shapes.size(); i++) {
            const int64_t dim = in_shapes[i][d];
            if (dim != kBroadcastDim && dim != x_dim) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, kInputNames[i], std::string("dim ") + std::to_string(d) + "=" + std::to_string(dim),
                    "min_range/max_range dim must be 1 or equal to x dim");
                return false;
            }
        }
        for (size_t i = 0; i < out_shapes.size(); i++) {
            const int64_t dim = out_shapes[i][d];
            if (dim != x_dim) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, kOutputName, std::string("dim ") + std::to_string(d) + "=" + std::to_string(dim),
                    "output shape must equal x shape");
                return false;
            }
        }
    }
    return true;
}

bool CheckBroadcastShape(const std::vector<std::vector<int64_t>>& in_shapes,
                         const std::vector<std::vector<int64_t>>& out_shapes, int64_t max_rank, const char* op_name)
{
    return ValidateBroadcastRank(in_shapes, out_shapes, max_rank, op_name) &&
           ValidateBroadcastDims(in_shapes, out_shapes, max_rank, op_name);
}

static void SqueezeBroadcastDims(const std::vector<std::vector<int64_t>>& padded_in, int64_t num_inputs,
                                 const std::vector<std::vector<int64_t>>& padded_out, int64_t num_outputs,
                                 int64_t max_rank, std::vector<int64_t>& max_shape,
                                 std::vector<std::vector<int64_t>>& normal_in,
                                 std::vector<std::vector<int64_t>>& normal_out)
{
    max_shape.clear();
    normal_in.assign(num_inputs, std::vector<int64_t>());
    normal_out.assign(num_outputs, std::vector<int64_t>());
    for (int64_t d = 0; d < max_rank; d++) {
        bool all_one = true;
        int64_t max_dim = kBroadcastDim;
        for (int64_t i = 0; i < num_inputs; i++) {
            if (padded_in[i][d] != kBroadcastDim) {
                all_one = false;
            }
            max_dim = std::max(max_dim, padded_in[i][d]);
        }
        for (int64_t i = 0; i < num_outputs; i++) {
            if (padded_out[i][d] != kBroadcastDim) {
                all_one = false;
            }
            max_dim = std::max(max_dim, padded_out[i][d]);
        }
        if (!all_one) {
            max_shape.push_back(max_dim);
            for (int64_t i = 0; i < num_inputs; i++) {
                normal_in[i].push_back(padded_in[i][d]);
            }
            for (int64_t i = 0; i < num_outputs; i++) {
                normal_out[i].push_back(padded_out[i][d]);
            }
        }
    }
    if (max_shape.empty()) {
        max_shape.push_back(kBroadcastDim);
        for (int64_t i = 0; i < num_inputs; i++) {
            normal_in[i].push_back(kBroadcastDim);
        }
        for (int64_t i = 0; i < num_outputs; i++) {
            normal_out[i].push_back(kBroadcastDim);
        }
    }
}

void PadAndSqueeze(const std::vector<std::vector<int64_t>>& input_shapes,
                   const std::vector<std::vector<int64_t>>& output_shapes, std::vector<int64_t>& maximum_bro_shape,
                   std::vector<std::vector<int64_t>>& normal_input_shapes,
                   std::vector<std::vector<int64_t>>& normal_output_shapes)
{
    int64_t num_inputs = static_cast<int64_t>(input_shapes.size());
    int64_t num_outputs = static_cast<int64_t>(output_shapes.size());
    int64_t max_rank = 0;
    for (const auto& shape : input_shapes) {
        max_rank = std::max(max_rank, static_cast<int64_t>(shape.size()));
    }
    for (const auto& shape : output_shapes) {
        max_rank = std::max(max_rank, static_cast<int64_t>(shape.size()));
    }

    auto pad = [&](const std::vector<int64_t>& shape) {
        std::vector<int64_t> padded;
        padded.assign(max_rank - static_cast<int64_t>(shape.size()), kBroadcastDim);
        padded.insert(padded.end(), shape.begin(), shape.end());
        return padded;
    };
    std::vector<std::vector<int64_t>> padded_in(num_inputs);
    std::vector<std::vector<int64_t>> padded_out(num_outputs);
    for (int64_t i = 0; i < num_inputs; i++) {
        padded_in[i] = pad(input_shapes[i]);
    }
    for (int64_t i = 0; i < num_outputs; i++) {
        padded_out[i] = pad(output_shapes[i]);
    }

    SqueezeBroadcastDims(padded_in, num_inputs, padded_out, num_outputs, max_rank, maximum_bro_shape,
                         normal_input_shapes, normal_output_shapes);
}

bool CalcPerBufferLayout(int64_t ub_per_core, int64_t phys_nodes, int64_t dtype_size, PerBufferLayout& out)
{
    if (phys_nodes <= 0 || dtype_size <= 0) {
        return false;
    }
    int64_t metadata = GetMetadataBytes();
    if (ub_per_core <= metadata) {
        return false;
    }
    int64_t per_buf_bytes = ((ub_per_core - metadata) / phys_nodes) & kUbAlignMask;
    if (per_buf_bytes < dtype_size) {
        return false;
    }
    int64_t per_buf_elems = per_buf_bytes / dtype_size;
    if (per_buf_elems <= 0) {
        return false;
    }
    out.bytes = per_buf_bytes;
    out.elems = per_buf_elems;
    return true;
}

bool FindSplitAxis(const std::vector<int64_t>& max_bro_shape, const PerBufferLayout& layout, SplitResult& out)
{
    if (layout.elems <= 0 || layout.bytes <= 0 || max_bro_shape.empty()) {
        return false;
    }
    int64_t rank = static_cast<int64_t>(max_bro_shape.size());
    int64_t inner = 1;
    for (int64_t k = rank - 1; k >= 0; k--) {
        if (max_bro_shape[k] <= 0) {
            return false;
        }
        if (max_bro_shape[k] * inner > layout.elems) {
            out.a_i = layout.elems / inner;
            if (out.a_i <= 0) {
                return false;
            }
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
    return false;
}

bool MultiCoreSplit(const std::vector<int64_t>& max_bro_shape, const SplitResult& ub_split, int64_t max_cores,
                    MultiCoreResult& out)
{
    if (max_cores <= 0 || ub_split.a_i <= 0 || ub_split.a_o <= 0 || ub_split.axis < 0 ||
        ub_split.axis >= static_cast<int64_t>(max_bro_shape.size())) {
        return false;
    }
    int64_t k = ub_split.axis;
    int64_t outer_prod = 1;
    for (int64_t j = 0; j < k; j++) {
        outer_prod *= max_bro_shape[j];
    }
    out.total_tiles = outer_prod * ub_split.a_o;
    if (out.total_tiles <= 0) {
        return false;
    }
    out.num_cores = (out.total_tiles < max_cores) ? out.total_tiles : max_cores;
    if (out.num_cores <= 0) {
        return false;
    }
    out.tiles_main = out.total_tiles / out.num_cores;
    out.cores_tail = out.total_tiles % out.num_cores;
    return true;
}

void PrecomputeStrides(const std::vector<int64_t>& shape, std::vector<int64_t>& strides)
{
    int64_t rank = static_cast<int64_t>(shape.size());
    strides.assign(rank, 0);
    for (int64_t d = rank - 1; d >= 0; d--) {
        if (shape[d] == kBroadcastDim) {
            strides[d] = 0;
            continue;
        }
        int64_t prod = 1;
        for (int64_t j = d + 1; j < rank; j++) {
            prod *= shape[j];
        }
        strides[d] = prod;
    }
}

} // namespace dequantize_ns

namespace optiling {

using namespace dequantize_ns;

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

DequantizeTiling::DequantizeTiling(gert::TilingContext* ctx) : ctx_(ctx) {}

ge::graphStatus DequantizeTiling::ValidateNodeInfo()
{
    const auto* nodeInfo = ctx_->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, nodeInfo);
    OP_CHECK_IF(nodeInfo->GetInputsNum() != kInputNum,
                OP_LOGE_FOR_INVALID_TENSORNUM(ctx_->GetNodeName(), "inputs", nodeInfo->GetInputsNum(),
                                              std::to_string(kInputNum)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(nodeInfo->GetOutputsNum() != kOutputNum,
                OP_LOGE_FOR_INVALID_TENSORNUM(ctx_->GetNodeName(), "outputs", nodeInfo->GetOutputsNum(),
                                              std::to_string(kOutputNum)),
                return ge::GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

ge::graphStatus DequantizeTiling::ValidateCompileInfo()
{
    auto compileInfo = static_cast<const DequantizeCompileInfo*>(ctx_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, compileInfo);
    OP_CHECK_IF(
        compileInfo->coreNum == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx_->GetNodeName(), "compileInfo.coreNum",
                                              std::to_string(compileInfo->coreNum), "core number must be positive"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<int64_t>(compileInfo->ubSize) <= GetMetadataBytes(),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx_->GetNodeName(), "compileInfo.ubSize",
                                                      std::to_string(compileInfo->ubSize),
                                                      "unified buffer size must be greater than metadata"),
                return ge::GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

ge::graphStatus DequantizeTiling::ReadOneTensorShape(bool is_input, size_t index, std::vector<int64_t>& dims)
{
    auto shape = is_input ? ctx_->GetInputShape(index) : ctx_->GetOutputShape(index);
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, shape);
    const char* name = is_input ? kInputNames[index] : kOutputName;
    dims.clear();
    gert::Shape storageShape = shape->GetStorageShape();
    if (storageShape.GetDimNum() == 0) {
        dims.push_back(kBroadcastDim);
    } else {
        for (size_t d = 0; d < storageShape.GetDimNum(); ++d) {
            int64_t dim = storageShape.GetDim(d);
            OP_CHECK_IF(dim < 0,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(ctx_->GetNodeName(), name,
                                                                 std::to_string(d) + "=" + std::to_string(dim),
                                                                 "dimension must be non-negative"),
                        return ge::GRAPH_FAILED);
            dims.push_back(dim);
        }
    }
    OP_CHECK_IF(dims.size() > static_cast<size_t>(kMaxRank),
                OP_LOGE(ctx_->GetNodeName(), "%s rank %zu exceeds max rank %ld", name, dims.size(), kMaxRank),
                return ge::GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

ge::graphStatus DequantizeTiling::ReadAndValidateShapes()
{
    raw_input_shapes_.resize(kInputNum);
    raw_output_shapes_.resize(kOutputNum);
    for (size_t i = 0; i < kInputNum; ++i) {
        if (ReadOneTensorShape(true, i, raw_input_shapes_[i]) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    }
    for (size_t i = 0; i < kOutputNum; ++i) {
        if (ReadOneTensorShape(false, i, raw_output_shapes_[i]) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    }
    OP_CHECK_IF(raw_output_shapes_[kYIndex] != raw_input_shapes_[kXIndex],
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ctx_->GetNodeName(), kOutputName, "output shape",
                                                      "output shape must equal x shape"),
                return ge::GRAPH_FAILED);
    auto xShape = ctx_->GetInputShape(kXIndex);
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, xShape);
    x_numel_ = xShape->GetStorageShape().GetShapeSize();
    return GRAPH_SUCCESS;
}

ge::graphStatus DequantizeTiling::ResolveInputDtype()
{
    auto inputDesc = ctx_->GetInputDesc(kXIndex);
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    if (dtype == ge::DT_INT8) {
        dtype_x_ = XDtype::INT8;
    } else if (dtype == ge::DT_UINT8) {
        dtype_x_ = XDtype::UINT8;
    } else if (dtype == ge::DT_INT32) {
        dtype_x_ = XDtype::INT32;
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(ctx_->GetNodeName(), kInputNames[kXIndex], Ops::Base::ToString(dtype).c_str(),
                                  "INT8/UINT8/INT32");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus DequantizeTiling::ValidateRangeDtypes()
{
    for (size_t index = kMinIndex; index <= kMaxIndex; ++index) {
        auto rangeDesc = ctx_->GetInputDesc(index);
        OP_CHECK_NULL_WITH_CONTEXT(ctx_, rangeDesc);
        if (rangeDesc->GetDataType() != ge::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE(ctx_->GetNodeName(), kInputNames[index],
                                      Ops::Base::ToString(rangeDesc->GetDataType()).c_str(), "FLOAT32");
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus DequantizeTiling::ValidateOutputDtype()
{
    auto outputDesc = ctx_->GetOutputDesc(kYIndex);
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, outputDesc);
    if (outputDesc->GetDataType() != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(ctx_->GetNodeName(), kOutputName,
                                  Ops::Base::ToString(outputDesc->GetDataType()).c_str(), "FLOAT32");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus DequantizeTiling::ParseMode()
{
    mode_ = kModeMinCombined;
    auto attrs = ctx_->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() > kModeAttrIndex) {
        const char* modeAttr = attrs->GetStr(kModeAttrIndex);
        if (modeAttr != nullptr) {
            mode_ = std::string(modeAttr);
        }
    }
    OP_CHECK_IF(!IsSupportedMode(mode_), OP_LOGE(ctx_->GetNodeName(), "Unsupported mode %s", mode_.c_str()),
                return ge::GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

ge::graphStatus DequantizeTiling::ResolveDtypeAndMode()
{
    if (ResolveInputDtype() != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    if (ValidateRangeDtypes() != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    if (ValidateOutputDtype() != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    if (ParseMode() != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    PrecomputeConstants();
    return GRAPH_SUCCESS;
}

void DequantizeTiling::PrecomputeConstants()
{
    if (mode_ == kModeMinCombined || mode_ == kModeMinFirst) {
        switch (dtype_x_) {
            case XDtype::INT8:
                denom_ = kDenomInt8;
                bias_ = kBiasInt8;
                break;
            case XDtype::UINT8:
                denom_ = kDenomUint8;
                bias_ = kBiasUint8;
                break;
            case XDtype::INT32:
            default:
                denom_ = kDenomInt32;
                bias_ = kBiasInt32;
                break;
        }
        return;
    }

    bias_ = 0.0f;
    switch (dtype_x_) {
        case XDtype::INT8:
            denom_ = kDenomScaledInt8;
            break;
        case XDtype::UINT8:
            denom_ = kDenomScaledUint8;
            break;
        case XDtype::INT32:
        default:
            denom_ = kDenomScaledInt32;
            break;
    }
}

ge::graphStatus DequantizeTiling::BuildAndValidateBroadcast()
{
    PadAndSqueeze(raw_input_shapes_, raw_output_shapes_, max_bro_shape_, normal_input_shapes_, normal_output_shapes_);
    if (x_numel_ == 0) {
        max_bro_shape_ = normal_input_shapes_[kXIndex];
    }
    rank_ = static_cast<int64_t>(max_bro_shape_.size());
    OP_CHECK_IF(!CheckBroadcastShape(normal_input_shapes_, normal_output_shapes_, rank_, ctx_->GetNodeName()),
                OP_LOGE(ctx_->GetNodeName(), "check broadcast shape failed"), return ge::GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

void DequantizeTiling::LogShapeInfo()
{
    auto compileInfo = static_cast<const DequantizeCompileInfo*>(ctx_->GetCompileInfo());
    OP_LOGI(ctx_->GetNodeName(), "GetShapeInfo done rank %ld dtype %ld mode %s ub %lu core %lu", rank_,
            static_cast<int64_t>(dtype_x_), mode_.c_str(), compileInfo->ubSize, compileInfo->coreNum);
}

ge::graphStatus DequantizeTiling::GetShapeInfo()
{
    if (ValidateNodeInfo() != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    if (ValidateCompileInfo() != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    if (ReadAndValidateShapes() != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    if (ResolveDtypeAndMode() != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    if (BuildAndValidateBroadcast() != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    LogShapeInfo();
    return GRAPH_SUCCESS;
}

template <int64_t R>
void DequantizeTiling::FillInputTilingData(DequantizeTilingData<R>* tiling, int64_t num_in, int64_t delta,
                                           const std::vector<std::vector<int64_t>>& in_strides)
{
    for (int64_t i = 0; i < num_in; i++) {
        for (int64_t d = 0; d < delta; d++) {
            tiling->input_shapes[i][d] = kBroadcastDim;
            tiling->input_strides[i][d] = 0;
        }
        for (int64_t d = 0; d < rank_; d++) {
            tiling->input_shapes[i][d + delta] = normal_input_shapes_[i][d];
            tiling->input_strides[i][d + delta] = in_strides[i][d];
        }
    }
    for (int64_t i = num_in; i < kMaxInputSlots; i++) {
        for (int64_t d = 0; d < R; d++) {
            tiling->input_shapes[i][d] = kBroadcastDim;
            tiling->input_strides[i][d] = 0;
        }
    }
}

template <int64_t R>
void DequantizeTiling::FillOutputTilingData(DequantizeTilingData<R>* tiling, int64_t num_out, int64_t delta,
                                            const std::vector<std::vector<int64_t>>& out_strides)
{
    for (int64_t i = 0; i < num_out; i++) {
        for (int64_t d = 0; d < delta; d++) {
            tiling->output_shapes[i][d] = kBroadcastDim;
            tiling->output_strides[i][d] = 0;
        }
        for (int64_t d = 0; d < rank_; d++) {
            tiling->output_shapes[i][d + delta] = normal_output_shapes_[i][d];
            tiling->output_strides[i][d + delta] = out_strides[i][d];
        }
    }
    for (int64_t i = num_out; i < kMaxOutputSlots; i++) {
        for (int64_t d = 0; d < R; d++) {
            tiling->output_shapes[i][d] = kBroadcastDim;
            tiling->output_strides[i][d] = 0;
        }
    }
}

template <int64_t R>
bool DequantizeTiling::PrepareSplit(DequantizeTilingData<R>* tiling, int64_t& per_buf_bytes)
{
    auto compileInfo = static_cast<const DequantizeCompileInfo*>(ctx_->GetCompileInfo());
    if (compileInfo == nullptr) {
        OP_LOGE(ctx_->GetNodeName(), "PrepareSplit failed: compileInfo is nullptr");
        return false;
    }
    PerBufferLayout layout;
    if (!CalcPerBufferLayout(static_cast<int64_t>(compileInfo->ubSize), kPhysNodes, kOutputDtypeSize, layout)) {
        OP_LOGE(ctx_->GetNodeName(), "PrepareSplit failed: invalid UB layout");
        return false;
    }
    if (!FindSplitAxis(max_bro_shape_, layout, tiling->split)) {
        OP_LOGE(ctx_->GetNodeName(), "PrepareSplit failed: FindSplitAxis failed");
        return false;
    }
    if (!MultiCoreSplit(max_bro_shape_, tiling->split, static_cast<int64_t>(compileInfo->coreNum), tiling->multicore)) {
        OP_LOGE(ctx_->GetNodeName(), "PrepareSplit failed: MultiCoreSplit failed");
        return false;
    }
    per_buf_bytes = layout.bytes;
    return true;
}

template <int64_t R>
void DequantizeTiling::FillShapeAndStrideTiling(DequantizeTilingData<R>* tiling)
{
    int64_t num_in = static_cast<int64_t>(normal_input_shapes_.size());
    int64_t num_out = static_cast<int64_t>(normal_output_shapes_.size());
    std::vector<std::vector<int64_t>> in_strides(num_in);
    std::vector<std::vector<int64_t>> out_strides(num_out);
    for (int64_t i = 0; i < num_in; i++) {
        PrecomputeStrides(normal_input_shapes_[i], in_strides[i]);
    }
    for (int64_t i = 0; i < num_out; i++) {
        PrecomputeStrides(normal_output_shapes_[i], out_strides[i]);
    }

    tiling->rank = rank_;
    int64_t delta = R - rank_;
    for (int64_t d = 0; d < delta; d++) {
        tiling->max_bro_shape[d] = kBroadcastDim;
    }
    for (int64_t d = 0; d < rank_; d++) {
        tiling->max_bro_shape[d + delta] = max_bro_shape_[d];
    }
    tiling->split.axis += delta;

    tiling->num_inputs = num_in;
    tiling->num_outputs = num_out;
    FillInputTilingData<R>(tiling, num_in, delta, in_strides);
    FillOutputTilingData<R>(tiling, num_out, delta, out_strides);
}

template <int64_t R>
void DequantizeTiling::FillScalarTiling(DequantizeTilingData<R>* tiling, int64_t per_buf_bytes)
{
    tiling->per_buf_bytes = per_buf_bytes;
    tiling->bias = bias_;
    tiling->denom = denom_;
    tiling->x_numel = x_numel_;
    ctx_->SetBlockDim(tiling->multicore.num_cores);
}

template <int64_t R>
void DequantizeTiling::LogTilingData(const DequantizeTilingData<R>* tiling) const
{
    OP_LOGI(ctx_->GetNodeName(),
            "TilingData: per_buf_bytes=%ld rank=%ld->R=%d "
            "max_bro_shape=%s bias=%f denom=%f x_numel=%ld "
            "split(axis=%ld a_i=%ld a_o=%ld a_i_tail=%ld) "
            "multi(cores=%ld tiles=%ld main=%ld core_tail=%ld)",
            tiling->per_buf_bytes, rank_, static_cast<int>(R), Arr2String(tiling->max_bro_shape, R).c_str(),
            tiling->bias, tiling->denom, tiling->x_numel, tiling->split.axis, tiling->split.a_i, tiling->split.a_o,
            tiling->split.a_i_tail, tiling->multicore.num_cores, tiling->multicore.total_tiles,
            tiling->multicore.tiles_main, tiling->multicore.cores_tail);
}

template <int64_t R>
ge::graphStatus DequantizeTiling::DoTilingAndSet()
{
    auto* tiling = ctx_->GetTilingData<DequantizeTilingData<R>>();
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, tiling);
    int64_t per_buf_bytes = 0;
    if (x_numel_ == 0) {
        auto compileInfo = static_cast<const DequantizeCompileInfo*>(ctx_->GetCompileInfo());
        OP_CHECK_NULL_WITH_CONTEXT(ctx_, compileInfo);
        PerBufferLayout layout;
        OP_CHECK_IF(
            !CalcPerBufferLayout(static_cast<int64_t>(compileInfo->ubSize), kPhysNodes, kOutputDtypeSize, layout),
            OP_LOGE(ctx_->GetNodeName(), "empty tensor tiling failed: invalid UB layout"), return ge::GRAPH_FAILED);
        tiling->split = {0, 0, 1, 0};
        tiling->multicore = {1, 1, 1, 0};
        FillShapeAndStrideTiling(tiling);
        FillScalarTiling(tiling, layout.bytes);
        LogTilingData(tiling);
        return GRAPH_SUCCESS;
    }
    if (!PrepareSplit(tiling, per_buf_bytes)) {
        return ge::GRAPH_FAILED;
    }
    FillShapeAndStrideTiling(tiling);
    FillScalarTiling(tiling, per_buf_bytes);
    LogTilingData(tiling);
    return GRAPH_SUCCESS;
}

int64_t DequantizeTiling::GetModeNum() const
{
    if (mode_ == kModeMinFirst) {
        return DEQUANTIZE_MODE_MIN_FIRST;
    }
    if (mode_ == kModeScaled) {
        return DEQUANTIZE_MODE_SCALED;
    }
    return DEQUANTIZE_MODE_MIN_COMBINED;
}

uint64_t DequantizeTiling::GetTilingKey(int64_t rank) const
{
    const int64_t modeNum = GetModeNum();
    const bool isRank4 = rank == kRank4;
    switch (modeNum) {
        case DEQUANTIZE_MODE_MIN_FIRST:
            return isRank4 ? GET_TPL_TILING_KEY(DEQUANTIZE_MODE_MIN_FIRST, DEQUANTIZE_RANK_4) :
                             GET_TPL_TILING_KEY(DEQUANTIZE_MODE_MIN_FIRST, DEQUANTIZE_RANK_8);
        case DEQUANTIZE_MODE_SCALED:
            return isRank4 ? GET_TPL_TILING_KEY(DEQUANTIZE_MODE_SCALED, DEQUANTIZE_RANK_4) :
                             GET_TPL_TILING_KEY(DEQUANTIZE_MODE_SCALED, DEQUANTIZE_RANK_8);
        case DEQUANTIZE_MODE_MIN_COMBINED:
        default:
            return isRank4 ? GET_TPL_TILING_KEY(DEQUANTIZE_MODE_MIN_COMBINED, DEQUANTIZE_RANK_4) :
                             GET_TPL_TILING_KEY(DEQUANTIZE_MODE_MIN_COMBINED, DEQUANTIZE_RANK_8);
    }
}

template <int64_t R>
ge::graphStatus DequantizeTiling::RunWithRank()
{
    auto ret = DoTilingAndSet<R>();
    if (ret != GRAPH_SUCCESS) {
        return ret;
    }
    ctx_->SetTilingKey(GetTilingKey(R));
    return GRAPH_SUCCESS;
}

ge::graphStatus DequantizeTiling::RunTiling()
{
    if (GetShapeInfo() != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    if (rank_ <= kRank4) {
        return RunWithRank<kRank4>();
    }
    return RunWithRank<kRank8>();
}

static ge::graphStatus TilingFuncDequantize(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin the tiling process for Arch35 architecture");
    DequantizeTiling tiling(context);
    auto ret = tiling.RunTiling();
    if (ret != GRAPH_SUCCESS) {
        return ret;
    }
    constexpr uint32_t kSysWorkspaceSize = 0;
    size_t* workspaces = context->GetWorkspaceSizes(kWorkspaceNum);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = kSysWorkspaceSize;
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
