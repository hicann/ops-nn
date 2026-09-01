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
 * \file bn3_d_training_update_tiling_arch35.cpp
 * \brief
 */
#include "register/op_def_registry.h"         // IMPL_OP_OPTILING
#include "exe_graph/runtime/tiling_context.h" // gert::TilingContext
#include "tiling/platform/platform_ascendc.h" // PlatformAscendC
#include "op_common/log/log.h"                // OP_LOGE, OP_CHECK_NULL_WITH_CONTEXT
#include "graph/types.h"                      // ge::DataType, ge::Format

#include "norm/bn3_d_training_update/op_kernel/arch35/bn3_d_training_update_tiling_struct.h" // BN3DTrainingUpdateTilingData<kRank>
#include "norm/bn3_d_training_update/op_kernel/arch35/bn3_d_training_update_struct.h" // BN3_D_TRAINING_UPDATE_RANK_4/5 + GET_TPL_TILING_KEY
#include "bn3_d_training_update_tiling_common.h"                                      // §5.3 common formulas
#include "bn3_d_training_update_tiling_branch0.h"                                     // ComputeBranch0Tiling (RANK=4)
#include "bn3_d_training_update_tiling_branch1.h"                                     // ComputeBranch1Tiling (RANK=5)
#include "bn3_d_training_update_tiling_arch35.h" // TilingFuncBN3DTrainingUpdate decl

#include <vector>
#include <cstring>

namespace optiling {

namespace {

// ---- §3.1.2 format → channel_axis ----
//   Implemented in the common module (bn3_d_training_update_tiling_common.cpp):
//   OpDef declares only explicit formats — channel axis is directly determined
//   from the format: NCHW→1, NCDHW→1, NHWC→3, NDHWC→4.
//   Unexpected format returns -1 (error).

// ---- §5.2 dtype → dtype_id table  {f32:0, f16:1, bf16:2} ----
//   Also returns the element byte size (sizeof(T)) used by FindSplitAxis.
//   Returns -1 / 0 on unsupported dtype.
int32_t DtypeToId(ge::DataType dt, int64_t& out_elem_bytes)
{
    switch (dt) {
        case ge::DT_FLOAT:
            out_elem_bytes = 4;
            return 0;
        case ge::DT_FLOAT16:
            out_elem_bytes = 2;
            return 1;
        case ge::DT_BF16:
            out_elem_bytes = 2;
            return 2;
        default:
            out_elem_bytes = 0;
            return -1;
    }
}

// ===========================================================================
// DoTilingAndSet<4>/<5> — per-branch TilingData fill.
//   Wires ComputeBranch0Tiling (Task 22, DESIGN-BRANCH-0 §1+§2) /
//   ComputeBranch1Tiling (Task 26, DESIGN-BRANCH-1 §1+§2) into the RANK-templated
//   POD, then writes the host-budget scalar fields (num_rec / bessel_scaler /
//   factor / one_minus_factor / epsilon) that the kernel reads.
//   The two branches take their own input bundles (Bn3dBranch0Inputs /
//   Bn3dBranch1Inputs — same fields, separate types for UT isolation).
// ===========================================================================
template <int64_t RANK>
ge::graphStatus DoTilingAndSet(gert::TilingContext* context, const Bn3dBranch0Inputs& in0, const Bn3dBranch1Inputs& in1,
                               float factor, float epsilon)
{
    auto* tiling = context->GetTilingData<BN3DTrainingUpdateTilingData<RANK>>();
    if (tiling == nullptr) {
        OP_LOGE(context->GetNodeName(), "GetTilingData<BN3DTrainingUpdateTilingData<%ld>> returned null",
                static_cast<long>(RANK));
        return ge::GRAPH_FAILED;
    }
    std::memset(tiling, 0, sizeof(BN3DTrainingUpdateTilingData<RANK>));

    if constexpr (RANK == 4) {
        // Branch-0 §1+§2 array + scalar fields.
        ComputeBranch0Tiling(in0, *tiling);
        // Host-budget float fields (consumed by kernel segment A).
        const int64_t num = in0.num;
        tiling->num_rec = (num > 0) ? (1.0f / static_cast<float>(num)) : 0.0f;
        tiling->bessel_scaler = (num > 1) ? (static_cast<float>(num) / static_cast<float>(num - 1)) : 0.0f;
        tiling->factor = factor;
        tiling->one_minus_factor = 1.0f - factor;
        tiling->epsilon = epsilon;
    } else {
        // Branch-1 (RANK==5) §1+§2 array + scalar fields.
        ComputeBranch1Tiling(in1, *tiling);
        // Host-budget float fields (consumed by kernel segment A).
        const int64_t num = in1.num;
        tiling->num_rec = (num > 0) ? (1.0f / static_cast<float>(num)) : 0.0f;
        tiling->bessel_scaler = (num > 1) ? (static_cast<float>(num) / static_cast<float>(num - 1)) : 0.0f;
        tiling->factor = factor;
        tiling->one_minus_factor = 1.0f - factor;
        tiling->epsilon = epsilon;
    }
    return ge::GRAPH_SUCCESS;
}

} // anonymous namespace

// ===========================================================================
// TilingFuncBN3DTrainingUpdate — DESIGN §5.3 + §5.1 host-side entry
// ===========================================================================
ge::graphStatus TilingFuncBN3DTrainingUpdate(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);

    // ----------------------------------------------------------------------
    // 1. Platform info (num_cores for MultiCoreSplitBn3d)
    // ----------------------------------------------------------------------
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    const int64_t max_cores = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());

    // ----------------------------------------------------------------------
    // 2. Read input/output shapes + dtype + format
    //    7 inputs:  x, sum, square_sum, scale, offset, mean, variance
    //    5 outputs: y, mean, variance, batch_mean, batch_variance
    // ----------------------------------------------------------------------
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;
    input_shapes.reserve(7);
    output_shapes.reserve(5);

    for (size_t i = 0; i < 7; ++i) {
        const gert::StorageShape* shp = context->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, shp);
        const gert::Shape& s = shp->GetStorageShape();
        std::vector<int64_t> dims;
        dims.reserve(s.GetDimNum());
        for (size_t d = 0; d < s.GetDimNum(); ++d) {
            dims.push_back(s.GetDim(d));
        }
        input_shapes.push_back(std::move(dims));
    }
    for (size_t i = 0; i < 5; ++i) {
        const gert::StorageShape* shp = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, shp);
        const gert::Shape& s = shp->GetStorageShape();
        std::vector<int64_t> dims;
        dims.reserve(s.GetDimNum());
        for (size_t d = 0; d < s.GetDimNum(); ++d) {
            dims.push_back(s.GetDim(d));
        }
        output_shapes.push_back(std::move(dims));
    }

    // x (input 0) dtype + format
    auto xDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const ge::DataType x_dtype = xDesc->GetDataType();
    const ge::Format x_fmt = xDesc->GetStorageFormat();

    int64_t elem_bytes = 0;
    const int32_t dtype_id = DtypeToId(x_dtype, elem_bytes);
    if (dtype_id < 0 || elem_bytes == 0) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", Ops::Base::ToString(x_dtype).c_str(),
                                  "FLOAT32, FLOAT16 or BFLOAT16");
        return ge::GRAPH_FAILED;
    }

    // ----------------------------------------------------------------------
    // 3. channel_axis — resolved from x's explicit format (NCHW/NCDHW/NHWC/NDHWC).
    // ----------------------------------------------------------------------
    const int64_t rank_x = static_cast<int64_t>(input_shapes[0].size());
    const int32_t C_from_sum = (input_shapes[1].empty()) ? -1 : static_cast<int32_t>(input_shapes[1][0]);
    const int32_t channel_axis = FormatToChannelAxis(x_fmt, rank_x, input_shapes[0], C_from_sum);
    if (channel_axis < 0) {
        OP_LOGE_FOR_INVALID_FORMAT(context->GetNodeName(), "x", Ops::Base::ToString(x_fmt).c_str(),
                                   "NCHW, NHWC, NCDHW or NDHWC");
        return ge::GRAPH_FAILED;
    }

    // ----------------------------------------------------------------------
    // 4. PadAndSqueezeBn3d → channel-last view + CheckBroadcastShapeBn3d
    // ----------------------------------------------------------------------
    std::vector<int64_t> max_bro_shape;
    std::vector<std::vector<int64_t>> normal_input_shapes;
    std::vector<std::vector<int64_t>> normal_output_shapes;
    if (!PadAndSqueezeBn3d(input_shapes, output_shapes, channel_axis, max_bro_shape, normal_input_shapes,
                           normal_output_shapes)) {
        OP_LOGE(context->GetNodeName(), "PadAndSqueezeBn3d failed");
        return ge::GRAPH_FAILED;
    }
    if (!CheckBroadcastShapeBn3d(normal_input_shapes, normal_output_shapes, rank_x)) {
        OP_LOGE(context->GetNodeName(), "CheckBroadcastShapeBn3d failed (channel dim incompatible)");
        return ge::GRAPH_FAILED;
    }

    // ----------------------------------------------------------------------
    // 5. Host budget fields (DESIGN §5.2):
    //      C       = sum.shape[0]
    //      num     = x.size / C   (reduce-domain element count)
    //      num_rec = 1.0f / num
    //      bessel_scaler = num==1 ? 0.0f : num / (num - 1)
    //      factor / epsilon from attrs; one_minus_factor = 1 - factor
    // ----------------------------------------------------------------------
    const int32_t C = static_cast<int32_t>(input_shapes[1][0]);
    int64_t x_size = 1;
    for (int64_t d : input_shapes[0]) {
        x_size *= d;
    }
    const int64_t num = (C == 0) ? 0 : (x_size / C);
    const float num_rec = (num > 0) ? (1.0f / static_cast<float>(num)) : 0.0f;
    const float bessel_scaler = (num > 1) ? (static_cast<float>(num) / static_cast<float>(num - 1)) : 0.0f;

    float factor = 0.1f;
    float epsilon = 1.0e-5f;
    const auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float* factorPtr = attrs->GetFloat(0);
        const float* epsilonPtr = attrs->GetFloat(1);
        if (factorPtr != nullptr)
            factor = *factorPtr;
        if (epsilonPtr != nullptr)
            epsilon = *epsilonPtr;
    }
    if (epsilon < 0.0f) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "epsilon", std::to_string(epsilon).c_str(),
                                              "The value of attribute epsilon cannot be less than 0");
        return ge::GRAPH_FAILED;
    }
    const float one_minus_factor = 1.0f - factor;

    // ----------------------------------------------------------------------
    // 6. FindSplitAxisBn3d + MultiCoreSplitBn3d (common formulas)
    // ----------------------------------------------------------------------
    // ub_split is still needed by the host-budget section below (ub_split.a_i).
    Bn3dSplitResult ub_split{};
    FindSplitAxisBn3d(max_bro_shape, elem_bytes, kBn3dUbPerCore, kBn3dPhysNodes, ub_split);

    // No single-core cap. The historical cap existed because the per-point
    // scatter wrote y via S-pipe SetValue through the core's write-back L1
    // (no cross-core snoop → 64B-line clobber between neighbouring cores),
    // and an early multi-core DataCopyPad-Compact attempt "failed" — that
    // failure is now attributed to the MTE3_MTE2 drain bug (drain keyed on
    // nSpatial while the whole-n remap extended the loop to sCount), fixed
    // in the kernel. Today every y store is MTE3 DataCopyPad (per-element
    // Compact blocks / grouped 32B blocks / compact C-contiguous), which
    // stores to GM directly and is coherent across cores; multi-core
    // sub-64B RMW stores on shared lines are exercised by case00070 (fp32
    // W-split, stride 23408 elems) and the NHWC compact multi-core cases,
    // all passing. Multi-core is therefore restored unconditionally.
    int64_t eff_max_cores = max_cores;

    // num_cores is set AFTER DoTilingAndSet fills the struct (see §8).
    // ----------------------------------------------------------------------
    // 7. §5.1 rank fork → SetTilingKey + dispatch to DoTilingAndSet<RANK>
    // ----------------------------------------------------------------------
    // Assemble the branch host-input bundles (used by ComputeBranch0Tiling for
    // RANK=4 / ComputeBranch1Tiling for RANK=5). The two bundle types carry
    // the same fields but stay separate so each branch's UT oracle can be
    // compiled against its own type.
    Bn3dBranch0Inputs bin{};
    bin.input_shapes = input_shapes;
    bin.output_shapes = output_shapes;
    bin.channel_axis = channel_axis;
    bin.C = static_cast<int32_t>(C);
    bin.num = num;
    bin.dtype_id = dtype_id;
    bin.elem_bytes = elem_bytes;
    bin.max_cores = static_cast<int64_t>(eff_max_cores);

    Bn3dBranch1Inputs bin1{};
    bin1.input_shapes = input_shapes;
    bin1.output_shapes = output_shapes;
    bin1.channel_axis = channel_axis;
    bin1.C = static_cast<int32_t>(C);
    bin1.num = num;
    bin1.dtype_id = dtype_id;
    bin1.elem_bytes = elem_bytes;
    bin1.max_cores = static_cast<int64_t>(eff_max_cores);

    const int32_t key = ChooseTilingKeyBn3d(rank_x);
    ge::graphStatus branch_ret = ge::GRAPH_FAILED;
    switch (key) {
        case 0:
            context->SetTilingKey(GET_TPL_TILING_KEY(BN3_D_TRAINING_UPDATE_RANK_4));
            branch_ret = DoTilingAndSet<4>(context, bin, bin1, factor, epsilon);
            break;
        case 1:
            context->SetTilingKey(GET_TPL_TILING_KEY(BN3_D_TRAINING_UPDATE_RANK_5));
            branch_ret = DoTilingAndSet<5>(context, bin, bin1, factor, epsilon);
            break;
        default:
            OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", std::to_string(rank_x).c_str(), "4D or 5D");
            return ge::GRAPH_FAILED;
    }
    if (branch_ret != ge::GRAPH_SUCCESS) {
        return branch_ret;
    }

    // Read num_cores from the struct (written by branch tiling's MultiCoreSplitBn3d)
    // so SetBlockDim is guaranteed consistent with the struct's multicore values.
    int64_t struct_num_cores = 0;
    switch (key) {
        case 0: {
            auto* td4 = context->GetTilingData<BN3DTrainingUpdateTilingData<4>>();
            struct_num_cores = (td4 != nullptr) ? td4->multicore.num_cores : 0;
            break;
        }
        case 1: {
            auto* td5 = context->GetTilingData<BN3DTrainingUpdateTilingData<5>>();
            struct_num_cores = (td5 != nullptr) ? td5->multicore.num_cores : 0;
            break;
        }
    }
    const uint32_t num_cores = (struct_num_cores > 0) ? static_cast<uint32_t>(struct_num_cores) : 1u;

    // ----------------------------------------------------------------------
    // 8. SetBlockDim(num_cores) + workspace.
    // ----------------------------------------------------------------------
    context->SetBlockDim(num_cores);
    constexpr int64_t kAlign = 32;
    const int64_t maxCountTile = ub_split.a_i; // single-tile max element count
    const int64_t perEndBytes = ((maxCountTile * static_cast<int64_t>(sizeof(float)) + kAlign - 1) / kAlign) * kAlign;
    const int64_t perCoreBytes = 2 * perEndBytes;
    const int64_t wsBytes = perCoreBytes * static_cast<int64_t>(num_cores);
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = static_cast<size_t>(wsBytes);

    OP_LOGI(context->GetNodeName(),
            "BN3DTrainingUpdate tiling: rank=%ld key=%d dtype_id=%d channel_axis=%d C=%d num=%ld "
            "num_cores=%u ub_split(axis=%ld a_i=%ld a_o=%ld a_i_tail=%ld) total_tiles=%ld",
            static_cast<long>(rank_x), key, dtype_id, channel_axis, C, static_cast<long>(num), num_cores,
            static_cast<long>(ub_split.axis), static_cast<long>(ub_split.a_i), static_cast<long>(ub_split.a_o),
            static_cast<long>(ub_split.a_i_tail), static_cast<long>(struct_num_cores));
    (void)factor;
    (void)epsilon;
    (void)num_rec;
    (void)bessel_scaler;
    (void)one_minus_factor;
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// TilingPrepareForBN3DTrainingUpdate — stub, must exist for auto-tiling runtime.
// ---------------------------------------------------------------------------
ge::graphStatus TilingPrepareForBN3DTrainingUpdate(gert::TilingParseContext* context) { return ge::GRAPH_SUCCESS; }

// ---------------------------------------------------------------------------
// IMPL_OP_OPTILING(BN3DTrainingUpdate) — register the TilingFunc + TilingParse callbacks.
//   Per DESIGN §5.3 "Host 侧注册". Branch DoTilingAndSet<4>/<5> fills the
//   RANK-templated TilingData in Task 22 / Task 26.
// ---------------------------------------------------------------------------
IMPL_OP_OPTILING(BN3DTrainingUpdate)
    .Tiling(TilingFuncBN3DTrainingUpdate)
    .TilingParse<BN3DTrainingUpdateCompileInfo>(TilingPrepareForBN3DTrainingUpdate);

} // namespace optiling
