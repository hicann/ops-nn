/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <cstdint>
#include <string>
#include <vector>

#include "graph/types.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "tiling_context_faker.h"
#include "op_host/op_const_def_nn.h"
#include "../../../op_host/op_tiling/arch35/weight_quant_batch_matmul_v2_checker_for_920r1.h"

using namespace std;

namespace {
enum class ScaleLayout {
    AUTO,           // derive from groupSize and weight/scale dtypes
    PER_TENSOR,     // {1}
    PER_CHANNEL_1D, // {n}
    PER_CHANNEL_2D, // transB ? {n, 1} : {1, n}
    PER_GROUP,      // {ceil(k/groupSize), n}
    MX,             // {n, ceil(k/64), 2}
    HIF4,           // {n, ceil(k/64), 1}
};

struct CheckerCase {
    const char* name = nullptr;
    bool expectOk = true;

    // context integrity knobs
    size_t ioInputs = 7U;
    size_t ioOutputs = 1U;
    bool noTilingData = false;

    // dims: x is {m, k} (or {k, m} when transA), weight origin is {k, n} (or {n, k} when transB)
    int64_t m = 1;
    int64_t k = 128;
    int64_t n = 128;
    int64_t kW = -1; // -1: same as k
    int64_t nW = -1; // -1: same as n
    bool transA = false;
    bool transB = false;

    // formats and storage rank overrides
    ge::Format aFormat = ge::FORMAT_ND;
    ge::Format bFormat = ge::FORMAT_ND;
    int64_t nzC0 = 32;         // dim(3) of NZ storage, 16 means C016 layout
    int64_t nzN1Override = -1; // -1: default ceil(n/16), used by transB-NZ n0*n1 check
    int64_t xOriDimNum = -1;   // -1: default 2
    int64_t xStoDimNum = -1;   // -1: default 2
    int64_t wOriDimNum = -1;   // -1: default 2
    int64_t wStoDimNum = -1;   // -1: ND->2, NZ->4

    // dtypes
    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType wDtype = ge::DT_INT8;
    ge::DataType scaleDtype = ge::DT_FLOAT16;
    ge::DataType offsetDtype = ge::DT_FLOAT16;
    ge::DataType biasDtype = ge::DT_FLOAT16;
    ge::DataType yDtype = ge::DT_FLOAT16;

    // attrs: order is transpose_x, transpose_weight, antiquant_group_size, dtype, inner_precise
    int64_t groupSize = 0;
    int64_t innerPrecise = 0;
    int attrNum = 5;

    // optional inputs
    bool hasOffset = false;
    bool hasQuantScale = false;
    bool hasQuantOffset = false;
    bool hasBias = false;

    // shape layout knobs, empty means derived
    ScaleLayout scaleLayout = ScaleLayout::AUTO;
    vector<int64_t> scaleShapeOverride;
    vector<int64_t> offsetShapeOverride;
    vector<int64_t> biasShapeOverride; // empty: {n}
};

int64_t CeilDivInt(int64_t a, int64_t b) { return (a + b - 1) / b; }

gert::Shape MakeShape(const vector<int64_t>& dims)
{
    gert::Shape shape;
    for (auto dim : dims) {
        shape.AppendDim(dim);
    }
    return shape;
}

vector<int64_t> PadToRank(vector<int64_t> dims, int64_t rank)
{
    while (static_cast<int64_t>(dims.size()) < rank) {
        dims.push_back(1);
    }
    return dims;
}

vector<int64_t> WeightStorageDims(const CheckerCase& c, int64_t kW, int64_t nW)
{
    vector<int64_t> origin = c.transB ? vector<int64_t>{nW, kW} : vector<int64_t>{kW, nW};
    if (c.bFormat == ge::FORMAT_FRACTAL_NZ) {
        const int64_t rank = c.wStoDimNum > 0 ? c.wStoDimNum : 4;
        if (rank != 4) {
            return PadToRank(origin, rank);
        }
        const int64_t n1 = c.nzN1Override > 0 ? c.nzN1Override : CeilDivInt(nW, 16);
        if (c.transB) {
            // n0 = dim(2), n1 = dim(1), n0 * n1 must equal N for non-Mx cases
            return {CeilDivInt(kW, 16), n1, 16, c.nzC0};
        }
        return {CeilDivInt(nW, c.nzC0), CeilDivInt(kW, 16), 16, c.nzC0};
    }
    const int64_t rank = c.wStoDimNum > 0 ? c.wStoDimNum : 2;
    return PadToRank(origin, rank);
}

vector<int64_t> ScaleDims(const CheckerCase& c, int64_t k, int64_t nW)
{
    if (!c.scaleShapeOverride.empty()) {
        return c.scaleShapeOverride;
    }
    const bool isMxFp4 = c.wDtype == ge::DT_FLOAT4_E2M1 && c.scaleDtype == ge::DT_FLOAT8_E8M0;
    const bool isHiF4 = c.wDtype == Ops::NN::DT_HIFLOAT4;
    ScaleLayout layout = c.scaleLayout;
    if (layout == ScaleLayout::AUTO) {
        if (c.groupSize > 0) {
            layout = isMxFp4 ? ScaleLayout::MX : (isHiF4 ? ScaleLayout::HIF4 : ScaleLayout::PER_GROUP);
        } else {
            layout = ScaleLayout::PER_CHANNEL_2D;
        }
    }
    switch (layout) {
        case ScaleLayout::PER_TENSOR:
            return {1};
        case ScaleLayout::PER_CHANNEL_1D:
            return {nW};
        case ScaleLayout::PER_CHANNEL_2D:
            return c.transB ? vector<int64_t>{nW, 1} : vector<int64_t>{1, nW};
        case ScaleLayout::PER_GROUP:
            return {CeilDivInt(k, c.groupSize), nW};
        case ScaleLayout::MX:
            return {nW, CeilDivInt(k, 64), 2};
        case ScaleLayout::HIF4:
            return {nW, CeilDivInt(k, 64), 1};
        default:
            return {1};
    }
}

void RunCheckerCase(const CheckerCase& c)
{
    SCOPED_TRACE(c.name);
    const int64_t kW = c.kW >= 0 ? c.kW : c.k;
    const int64_t nW = c.nW >= 0 ? c.nW : c.n;

    gert::StorageShape xShape;
    gert::StorageShape weightShape;
    gert::StorageShape antiQuantScaleShape;
    gert::StorageShape antiQuantOffsetShape;
    gert::StorageShape quantScaleShape;
    gert::StorageShape quantOffsetShape;
    gert::StorageShape biasShape;
    gert::StorageShape outputShape;

    const vector<int64_t> xDims = c.transA ? vector<int64_t>{c.k, c.m} : vector<int64_t>{c.m, c.k};
    xShape.MutableOriginShape() = MakeShape(PadToRank(xDims, c.xOriDimNum > 0 ? c.xOriDimNum : 2));
    xShape.MutableStorageShape() = MakeShape(PadToRank(xDims, c.xStoDimNum > 0 ? c.xStoDimNum : 2));

    const vector<int64_t> wOrigin = c.transB ? vector<int64_t>{nW, kW} : vector<int64_t>{kW, nW};
    weightShape.MutableOriginShape() = MakeShape(PadToRank(wOrigin, c.wOriDimNum > 0 ? c.wOriDimNum : 2));
    weightShape.MutableStorageShape() = MakeShape(WeightStorageDims(c, kW, nW));

    const vector<int64_t> scaleDims = ScaleDims(c, c.k, nW);
    antiQuantScaleShape.MutableStorageShape() = MakeShape(scaleDims);
    antiQuantScaleShape.MutableOriginShape() = MakeShape(scaleDims);
    const vector<int64_t> offsetDims = c.offsetShapeOverride.empty() ? scaleDims : c.offsetShapeOverride;
    antiQuantOffsetShape.MutableStorageShape() = MakeShape(offsetDims);
    antiQuantOffsetShape.MutableOriginShape() = MakeShape(offsetDims);
    quantScaleShape.MutableStorageShape() = MakeShape({1});
    quantOffsetShape.MutableStorageShape() = MakeShape({1});
    const vector<int64_t> biasDims = c.biasShapeOverride.empty() ? vector<int64_t>{nW} : c.biasShapeOverride;
    biasShape.MutableStorageShape() = MakeShape(biasDims);
    biasShape.MutableOriginShape() = MakeShape(biasDims);
    outputShape.MutableStorageShape() = MakeShape({c.m, nW});
    outputShape.MutableOriginShape() = MakeShape({c.m, nW});

    auto rawTilingData = gert::TilingData::CreateCap(4096);
    ASSERT_NE(rawTilingData, nullptr);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    // the checker never reads compile info but the context builder requires a non-null one
    uint64_t dummyCompileInfo = 0U;

    gert::TilingContextFaker faker;
    faker.NodeIoNum(c.ioInputs, c.ioOutputs);
    faker.CompileInfo(&dummyCompileInfo);
    faker.PlatformInfo(reinterpret_cast<char*>(&platformInfo));
    if (c.ioInputs == 7U) {
        faker.IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
            .InputShapes({&xShape, &weightShape, &antiQuantScaleShape, c.hasOffset ? &antiQuantOffsetShape : nullptr,
                          c.hasQuantScale ? &quantScaleShape : nullptr, c.hasQuantOffset ? &quantOffsetShape : nullptr,
                          c.hasBias ? &biasShape : nullptr});
    }
    if (c.ioOutputs > 0U) {
        faker.OutputShapes({&outputShape});
    }
    if (c.ioInputs >= 1U) {
        faker.NodeInputTd(0, c.xDtype, ge::FORMAT_ND, c.aFormat);
    }
    if (c.ioInputs >= 2U) {
        faker.NodeInputTd(1, c.wDtype, ge::FORMAT_ND, c.bFormat);
    }
    if (c.ioInputs >= 3U) {
        faker.NodeInputTd(2, c.scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    if (c.ioInputs >= 4U) {
        faker.NodeInputTd(3, c.offsetDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    if (c.ioInputs >= 5U) {
        faker.NodeInputTd(4, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    if (c.ioInputs >= 6U) {
        faker.NodeInputTd(5, c.xDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    if (c.ioInputs >= 7U) {
        faker.NodeInputTd(6, c.biasDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    if (c.ioOutputs >= 1U) {
        faker.NodeOutputTd(0, c.yDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }

    vector<pair<string, Ops::NN::AnyValue>> attrs = {
        {"transpose_x", Ops::NN::AnyValue::CreateFrom<bool>(c.transA)},
        {"transpose_weight", Ops::NN::AnyValue::CreateFrom<bool>(c.transB)}};
    if (c.attrNum >= 3) {
        attrs.push_back({"antiquant_group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(c.groupSize)});
    }
    if (c.attrNum >= 4) {
        attrs.push_back({"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)});
    }
    if (c.attrNum >= 5) {
        attrs.push_back({"inner_precise", Ops::NN::AnyValue::CreateFrom<int64_t>(c.innerPrecise)});
    }
    faker.NodeAttrs(attrs);

    if (!c.noTilingData) {
        faker.TilingData(rawTilingData.get());
    }
    auto holder = faker.SetOpType("WeightQuantBatchMatmulV2").Build();
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext, nullptr);

    optiling::WeightQuantBatchMatmulV2CheckerFor920R1 checker(tilingContext);
    const auto expectStatus = c.expectOk ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
    EXPECT_EQ(checker.Check(), expectStatus) << "case: " << c.name;
}

vector<CheckerCase> BuildCheckerCases()
{
    vector<CheckerCase> cases;
    auto add = [&cases](const char* name, bool expectOk, const CheckerCase& c) {
        CheckerCase scenario = c;
        scenario.name = name;
        scenario.expectOk = expectOk;
        cases.push_back(scenario);
    };

    // presets: x FP16 + INT8 weight ND per-channel {1, n} FP16 scale by default
    auto w8NdPerChannel = []() {
        CheckerCase c;
        return c;
    };
    auto w8NdPerGroup = []() {
        CheckerCase c;
        c.groupSize = 64;
        return c;
    }; // scale {2, n} FP16
    auto w8NdPerTensor = []() {
        CheckerCase c;
        c.scaleLayout = ScaleLayout::PER_TENSOR;
        return c;
    };
    auto w4NdPerChannel = []() {
        CheckerCase c;
        c.wDtype = ge::DT_INT4;
        c.scaleDtype = ge::DT_UINT64;
        return c;
    };
    auto w4NdPerGroup = []() {
        CheckerCase c;
        c.wDtype = ge::DT_INT4;
        c.groupSize = 64;
        return c;
    };
    auto w4NzC032PerChannel = []() {
        CheckerCase c;
        c.wDtype = ge::DT_INT4;
        c.bFormat = ge::FORMAT_FRACTAL_NZ;
        return c;
    };
    auto w4NzC016PerGroup = []() {
        CheckerCase c;
        c.wDtype = ge::DT_INT4;
        c.bFormat = ge::FORMAT_FRACTAL_NZ;
        c.nzC0 = 16;
        c.groupSize = 64;
        return c;
    };
    auto w8NzC032PerGroup = []() {
        CheckerCase c;
        c.bFormat = ge::FORMAT_FRACTAL_NZ;
        c.groupSize = 64;
        return c;
    };
    auto w8NzC016PerGroup = []() {
        CheckerCase c;
        c.bFormat = ge::FORMAT_FRACTAL_NZ;
        c.nzC0 = 16;
        c.groupSize = 64;
        return c;
    };
    auto mxFp4 = []() {
        CheckerCase c;
        c.wDtype = ge::DT_FLOAT4_E2M1;
        c.scaleDtype = ge::DT_FLOAT8_E8M0;
        c.groupSize = 32;
        return c;
    };
    auto hiF4 = []() {
        CheckerCase c;
        c.wDtype = Ops::NN::DT_HIFLOAT4;
        c.scaleDtype = Ops::NN::DT_HIFLOAT4_SCALE;
        c.groupSize = 64;
        return c;
    };
    auto fp4PerGroup = []() {
        CheckerCase c;
        c.wDtype = ge::DT_FLOAT4_E2M1;
        c.groupSize = 64;
        return c;
    };
    auto fp8NdPerChannel = []() {
        CheckerCase c;
        c.wDtype = ge::DT_FLOAT8_E4M3FN;
        c.scaleDtype = ge::DT_UINT64;
        return c;
    };
    auto hif8NdPerChannel = []() {
        CheckerCase c;
        c.wDtype = ge::DT_HIFLOAT8;
        return c;
    };
    // keep scaleDtype from the preset, only align x/y/offset/bias with BF16
    auto bf16 = [](CheckerCase c) {
        c.xDtype = ge::DT_BF16;
        c.yDtype = ge::DT_BF16;
        c.offsetDtype = ge::DT_BF16;
        c.biasDtype = ge::DT_BF16;
        return c;
    };

    // =============== CheckContext ===============
    {
        auto c = w8NdPerChannel();
        c.noTilingData = true;
        add("ctx_no_tiling_data", false, c);
    }
    {
        auto c = w8NdPerChannel();
        c.ioInputs = 1;
        add("ctx_no_weight_desc", false, c);
    }
    {
        auto c = w8NdPerChannel();
        c.ioInputs = 2;
        add("ctx_no_scale_desc", false, c);
    }

    // =============== CheckAttr ===============
    {
        auto c = w8NdPerChannel();
        c.transA = true;
        add("attr_transa_true_fail", false, c);
    }
    {
        auto c = w8NdPerChannel();
        c.groupSize = 48;
        add("attr_group_size_48_fail", false, c);
    }
    {
        auto c = w8NdPerChannel();
        c.innerPrecise = 2;
        add("attr_inner_precise_2_fail", false, c);
    }
    {
        auto c = w8NdPerChannel();
        c.innerPrecise = 1;
        add("attr_inner_precise_1_ok", true, c);
    }
    {
        auto c = w8NdPerChannel();
        c.attrNum = 2;
        add("attr_only_transposes_ok", true, c);
    }
    {
        auto c = w8NdPerChannel();
        c.attrNum = 3;
        add("attr_with_group_only_ok", true, c);
    }
    {
        auto c = w8NdPerChannel();
        c.attrNum = 4;
        add("attr_with_dtype_only_ok", true, c);
    }

    // =============== CheckShape: empty tensors ===============
    {
        auto c = w8NdPerChannel();
        c.m = 0;
        add("shape_x_empty_fail", false, c);
    }
    {
        auto c = w8NdPerChannel();
        c.kW = 0;
        add("shape_weight_empty_fail", false, c);
    }
    {
        auto c = w8NdPerChannel();
        c.scaleShapeOverride = {0};
        add("shape_scale_empty_fail", false, c);
    }

    // =============== CheckInputDimNum ===============
    {
        auto c = w8NdPerChannel();
        c.xOriDimNum = 3;
        add("dim_x_origin_3d_fail", false, c);
    }
    {
        auto c = w8NdPerChannel();
        c.xStoDimNum = 3;
        add("dim_x_storage_3d_fail", false, c);
    }
    {
        auto c = w8NdPerChannel();
        c.wOriDimNum = 3;
        add("dim_weight_origin_3d_fail", false, c);
    }
    {
        auto c = w8NdPerChannel();
        c.wStoDimNum = 3;
        add("dim_weight_nd_storage_3d_fail", false, c);
    }
    {
        auto c = w8NzC032PerGroup();
        c.wStoDimNum = 2;
        add("dim_weight_nz_storage_2d_fail", false, c);
    }

    // =============== CheckInputShape: x format ===============
    {
        auto c = w8NdPerChannel();
        c.aFormat = ge::FORMAT_FRACTAL_NZ;
        add("format_x_not_nd_fail", false, c);
    }

    // =============== CheckInputSize ===============
    {
        auto c = w8NdPerChannel();
        c.kW = 144;
        add("size_k_mismatch_fail", false, c);
    }
    {
        auto c = w8NdPerChannel();
        c.bFormat = ge::FORMAT_FRACTAL_NZ;
        c.transB = true;
        c.nzN1Override = 9;
        add("size_nz_transb_n0n1_mismatch_fail", false, c);
    }
    {
        auto c = w8NdPerChannel();
        c.bFormat = ge::FORMAT_FRACTAL_NZ;
        c.transB = true;
        c.nzC0 = 16;
        add("size_nz_c016_transb_fail", false, c);
    }
    {
        auto c = w8NdPerGroup();
        c.transB = true;
        add("ok_w8_nd_per_group_transb", true, c);
    }
    {
        auto c = w8NdPerChannel();
        c.bFormat = ge::FORMAT_FRACTAL_NZ;
        c.transB = true;
        c.scaleDtype = ge::DT_UINT64;
        add("ok_w8_nz_transb_per_channel_uint64", true, c);
    }
    {
        auto c = mxFp4();
        add("ok_mxfp4_nd", true, c);
    }
    {
        auto c = mxFp4();
        c.bFormat = ge::FORMAT_FRACTAL_NZ;
        c.transB = true;
        add("ok_mxfp4_nz_transb_c032", true, c);
    }
    {
        auto c = mxFp4();
        c.bFormat = ge::FORMAT_FRACTAL_NZ;
        c.transB = true;
        c.nzC0 = 16;
        add("mxfp4_nz_transb_c016_fail", false, c);
    }

    // =============== CheckAntiQuantShape ===============
    {
        auto c = w8NdPerChannel();
        c.scaleShapeOverride = {1, 1, 1};
        add("antiq_scale_3d_w8_fail", false, c);
    }
    {
        auto c = mxFp4();
        c.scaleShapeOverride = {1, 128};
        add("antiq_mxfp4_scale_2d_fail", false, c);
    }
    {
        auto c = mxFp4();
        c.scaleShapeOverride = {128, 2, 2, 1};
        add("antiq_mxfp4_scale_4d_fail", false, c);
    }
    {
        auto c = mxFp4();
        c.groupSize = 64;
        add("antiq_mxfp4_group_64_fail", false, c);
    }
    {
        auto c = mxFp4();
        c.scaleShapeOverride = {128, 3, 2};
        add("antiq_mxfp4_scale_shape_mismatch_fail", false, c);
    }
    {
        auto c = hiF4();
        c.groupSize = 128;
        add("antiq_hif4_group_128_fail", false, c);
    }
    {
        auto c = w8NdPerChannel();
        c.groupSize = 32;
        add("antiq_w8_group_32_fail", false, c);
    }
    {
        auto c = w8NdPerChannel();
        c.groupSize = 256;
        add("antiq_w8_group_256_fail", false, c);
    }
    {
        auto c = w4NdPerGroup();
        c.groupSize = 32;
        add("antiq_w4_nd_group_32_fail", false, c);
    }
    {
        auto c = w8NdPerGroup();
        c.k = 96;
        add("antiq_k_not_divisible_fail", false, c);
    }
    {
        auto c = w8NdPerGroup();
        c.scaleShapeOverride = {1, 128};
        add("antiq_scale_shape_mismatch_fail", false, c);
    }
    {
        auto c = w8NdPerGroup();
        c.hasOffset = true;
        c.offsetShapeOverride = {1, 128};
        add("antiq_offset_shape_mismatch_fail", false, c);
    }
    {
        auto c = w8NdPerChannel();
        c.scaleShapeOverride = {129};
        add("antiq_per_channel_1d_size_mismatch_fail", false, c);
    }
    {
        auto c = w8NzC032PerGroup();
        c.hasOffset = true;
        add("antiq_offset_not_allowed_nz_c032_fail", false, c);
    }
    {
        auto c = fp4PerGroup();
        c.bFormat = ge::FORMAT_FRACTAL_NZ;
        c.nzC0 = 16;
        c.hasOffset = true;
        add("antiq_offset_not_allowed_fp4_nz_c016_fail", false, c);
    }
    {
        auto c = fp8NdPerChannel();
        c.hasOffset = true;
        add("antiq_offset_not_allowed_fp8_nd_fail", false, c);
    }
    {
        auto c = w8NdPerChannel();
        c.scaleLayout = ScaleLayout::PER_CHANNEL_1D;
        add("ok_w8_nd_per_channel_1d", true, c);
    }
    {
        auto c = w8NdPerGroup();
        c.hasOffset = true;
        add("ok_w8_nd_per_group_with_offset", true, c);
    }

    // =============== quant scale/offset must be absent ===============
    {
        auto c = w8NdPerChannel();
        c.hasQuantScale = true;
        add("quant_scale_present_fail", false, c);
    }
    {
        auto c = w8NdPerChannel();
        c.hasQuantOffset = true;
        add("quant_offset_present_fail", false, c);
    }

    // =============== CheckBiasShape ===============
    {
        auto c = w8NdPerChannel();
        c.hasBias = true;
        c.biasShapeOverride = {129};
        add("bias_1d_size_fail", false, c);
    }
    {
        auto c = w8NdPerChannel();
        c.hasBias = true;
        c.biasShapeOverride = {0};
        add("bias_1d_zero_ok", true, c);
    }
    {
        auto c = w8NdPerChannel();
        c.hasBias = true;
        add("bias_1d_ok", true, c);
    }
    {
        auto c = w8NdPerChannel();
        c.hasBias = true;
        c.biasShapeOverride = {1, 128};
        add("bias_2d_ok", true, c);
    }
    {
        auto c = w8NdPerChannel();
        c.hasBias = true;
        c.biasShapeOverride = {2, 128};
        add("bias_2d_first_dim_fail", false, c);
    }
    {
        auto c = w8NdPerChannel();
        c.hasBias = true;
        c.biasShapeOverride = {1, 129};
        add("bias_2d_second_dim_fail", false, c);
    }
    {
        auto c = w8NdPerChannel();
        c.hasBias = true;
        c.biasShapeOverride = {1, 128, 1};
        add("bias_3d_fail", false, c);
    }

    // =============== CheckShapeAdvanced: alignment ===============
    {
        auto c = w4NdPerChannel();
        c.k = 127;
        add("align_w4_per_channel_odd_k_fail", false, c);
    }
    {
        auto c = w4NdPerChannel();
        c.n = 127;
        add("align_w4_per_channel_odd_n_fail", false, c);
    }
    {
        auto c = fp4PerGroup();
        c.n = 96;
        add("align_fp4_per_group_n_not_64_fail", false, c);
    }
    {
        auto c = hiF4();
        c.n = 96;
        add("align_hif4_n_not_64_fail", false, c);
    }
    {
        auto c = w8NzC032PerGroup();
        c.n = 100;
        add("align_w8_nz_per_group_n_not_32_fail", false, c);
    }
    {
        auto c = fp8NdPerChannel();
        c.k = 100;
        add("align_fp8_nd_per_channel_k_not_32_fail", false, c);
    }
    {
        auto c = w4NzC016PerGroup();
        c.n = 127;
        add("align_w4_c016_per_group_odd_n_fail", false, c);
    }

    // =============== CheckShapeAdvanced: size limits ===============
    {
        auto c = w8NdPerGroup();
        c.n = 65536;
        add("limit_n_over_65535_fail", false, c);
    }
    {
        auto c = w8NdPerGroup();
        c.k = 65600;
        add("limit_k_over_65535_fail", false, c);
    }
    {
        auto c = w8NdPerTensor();
        c.m = 2147483648LL;
        add("limit_m_over_int32_fail", false, c);
    }
    {
        auto c = w4NdPerChannel();
        c.n = 70000;
        add("limit_w4_nd_per_channel_n_over_65535_ok", true, c);
    }
    {
        auto c = w4NdPerChannel();
        c.n = 2147483648LL;
        add("limit_w4_nd_per_channel_n_over_int32_fail", false, c);
    }
    {
        auto c = w8NdPerChannel();
        c.groupSize = 128;
        add("limit_group_ge_k_fail", false, c);
    }
    {
        auto c = hiF4();
        c.k = 64;
        add("limit_hif4_group_eq_k_ok", true, c);
    }
    {
        auto c = w4NzC032PerChannel();
        c.n = 70000;
        add("limit_w4_nz_per_channel_n_over_65535_ok", true, c);
    }
    {
        auto c = w4NzC016PerGroup();
        c.n = 70000;
        add("limit_w4_c016_per_group_n_over_65535_ok", true, c);
    }
    {
        auto c = w8NzC016PerGroup();
        c.n = 65568;
        add("limit_w8_c016_per_group_n_over_65535_ok", true, c);
    }

    // =============== CheckDtype ===============
    {
        auto c = w8NdPerChannel();
        c.xDtype = ge::DT_INT8;
        add("dtype_x_int8_fail", false, c);
    }
    {
        auto c = w8NdPerChannel();
        c.wDtype = ge::DT_FLOAT;
        add("dtype_weight_float_fail", false, c);
    }
    {
        auto c = w8NdPerGroup();
        c.scaleDtype = ge::DT_BF16;
        add("dtype_per_group_scale_bf16_fail", false, c);
    }
    {
        auto c = w8NdPerChannel();
        c.scaleDtype = ge::DT_BF16;
        add("dtype_per_channel_w8nd_scale_bf16_fail", false, c);
    }
    {
        auto c = w8NdPerTensor();
        c.scaleDtype = ge::DT_BF16;
        add("dtype_per_tensor_w8nd_scale_bf16_fail", false, c);
    }
    {
        auto c = w4NdPerChannel();
        c.scaleDtype = ge::DT_FLOAT16;
        add("dtype_w4_nd_scale_fp16_fail", false, c);
    }
    {
        auto c = w4NzC032PerChannel();
        c.scaleDtype = ge::DT_BF16;
        add("dtype_w4_nz_scale_bf16_fail", false, c);
    }
    {
        auto c = hiF4();
        c.scaleDtype = ge::DT_FLOAT16;
        add("dtype_hif4_scale_fp16_fail", false, c);
    }
    {
        auto c = w8NdPerGroup();
        c.hasOffset = true;
        c.offsetDtype = ge::DT_BF16;
        add("dtype_offset_bf16_fail", false, c);
    }
    {
        auto c = w8NdPerChannel();
        c.yDtype = ge::DT_BF16;
        add("dtype_y_bf16_fail", false, c);
    }
    {
        auto c = w8NdPerChannel();
        add("ok_w8_nd_per_channel", true, c);
    }
    {
        auto c = w8NdPerGroup();
        add("ok_w8_nd_per_group", true, c);
    }
    {
        auto c = w8NdPerTensor();
        add("ok_w8_nd_per_tensor", true, c);
    }
    {
        auto c = w4NdPerChannel();
        add("ok_w4_nd_per_channel_uint64", true, c);
    }
    {
        auto c = w4NdPerChannel();
        c.scaleLayout = ScaleLayout::PER_TENSOR;
        add("ok_w4_nd_per_tensor_uint64", true, c);
    }
    {
        auto c = w4NdPerGroup();
        add("ok_w4_nd_per_group", true, c);
    }
    {
        auto c = w8NzC032PerGroup();
        add("ok_w8_nz_c032_per_group", true, c);
    }
    {
        auto c = w8NzC016PerGroup();
        c.groupSize = 32;
        add("ok_w8_nz_c016_per_group_g32", true, c);
    }
    {
        auto c = w8NzC016PerGroup();
        add("ok_w8_nz_c016_per_group_g64", true, c);
    }
    {
        auto c = w4NzC032PerChannel();
        add("ok_w4_nz_c032_per_channel", true, c);
    }
    {
        auto c = w4NzC016PerGroup();
        add("ok_w4_nz_c016_per_group", true, c);
    }
    {
        auto c = w4NzC032PerChannel();
        c.nzC0 = 16;
        c.hasOffset = true;
        add("ok_w4_nz_c016_per_channel_with_offset", true, c);
    }
    {
        auto c = fp8NdPerChannel();
        add("ok_fp8_nd_per_channel_uint64", true, c);
    }
    {
        auto c = fp8NdPerChannel();
        c.scaleLayout = ScaleLayout::PER_TENSOR;
        add("ok_fp8_nd_per_tensor_uint64", true, c);
    }
    {
        auto c = hif8NdPerChannel();
        add("ok_hif8_nd_per_channel", true, c);
    }
    {
        auto c = hif8NdPerChannel();
        c.bFormat = ge::FORMAT_FRACTAL_NZ;
        c.scaleDtype = ge::DT_UINT64;
        add("ok_hif8_nz_c032_per_channel_uint64", true, c);
    }
    {
        auto c = fp4PerGroup();
        add("ok_fp4_per_group", true, c);
    }
    {
        auto c = fp4PerGroup();
        c.bFormat = ge::FORMAT_FRACTAL_NZ;
        c.nzC0 = 16;
        add("ok_fp4_nz_c016_per_group", true, c);
    }
    {
        auto c = hiF4();
        add("ok_hif4_nd", true, c);
    }

    // =============== CheckBiasDtype ===============
    {
        auto c = w8NdPerGroup();
        c.hasBias = true;
        c.biasDtype = ge::DT_BF16;
        add("bias_dtype_x_fp16_bias_bf16_fail", false, c);
    }
    {
        auto c = bf16(w8NdPerGroup());
        c.scaleDtype = ge::DT_BF16;
        c.hasBias = true;
        c.biasDtype = ge::DT_FLOAT16;
        add("bias_dtype_x_bf16_w8_bias_fp16_fail", false, c);
    }
    {
        auto c = bf16(w8NdPerGroup());
        c.scaleDtype = ge::DT_BF16;
        c.hasBias = true;
        c.biasDtype = ge::DT_FLOAT;
        add("bias_dtype_x_bf16_w8_bias_float_ok", true, c);
    }
    {
        auto c = bf16(w4NdPerChannel());
        c.hasBias = true;
        add("bias_dtype_x_bf16_w4_bias_bf16_ok", true, c);
    }
    {
        auto c = bf16(w4NdPerChannel());
        c.hasBias = true;
        c.biasDtype = ge::DT_FLOAT16;
        add("bias_dtype_x_bf16_w4_bias_fp16_fail", false, c);
    }
    {
        auto c = bf16(fp4PerGroup());
        c.scaleDtype = ge::DT_BF16;
        c.hasBias = true;
        c.biasDtype = ge::DT_FLOAT16;
        add("bias_dtype_x_bf16_fp4_bias_fp16_fail", false, c);
    }
    {
        auto c = bf16(fp8NdPerChannel());
        c.hasBias = true;
        c.biasDtype = ge::DT_FLOAT16;
        add("bias_dtype_x_bf16_fp8_bias_fp16_fail", false, c);
    }
    {
        auto c = bf16(mxFp4());
        c.hasBias = true;
        c.biasDtype = ge::DT_FLOAT16;
        add("bias_dtype_x_bf16_mxfp4_bias_fp16_fail", false, c);
    }
    {
        auto c = bf16(hiF4());
        c.hasBias = true;
        c.biasDtype = ge::DT_FLOAT16;
        add("bias_dtype_x_bf16_hif4_bias_fp16_fail", false, c);
    }
    {
        auto c = bf16(hiF4());
        c.hasBias = true;
        add("bias_dtype_x_bf16_hif4_bias_bf16_ok", true, c);
    }
    {
        auto c = w8NdPerGroup();
        c.hasBias = true;
        add("bias_dtype_x_fp16_w8_bias_fp16_ok", true, c);
    }

    return cases;
}
} // namespace

using namespace ge;
using namespace optiling;

class TestWeightQuantBatchMatmulV2Checker920R1Tiling : public testing::TestWithParam<CheckerCase> {};

TEST_P(TestWeightQuantBatchMatmulV2Checker920R1Tiling, checkerCase) { RunCheckerCase(GetParam()); }

INSTANTIATE_TEST_CASE_P(Checker920R1, TestWeightQuantBatchMatmulV2Checker920R1Tiling,
                        testing::ValuesIn(BuildCheckerCases()));
