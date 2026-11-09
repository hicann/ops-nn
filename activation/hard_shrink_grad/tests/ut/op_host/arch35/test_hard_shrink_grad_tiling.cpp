/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "../../../../op_kernel/arch35/hard_shrink_grad_tiling_data.h"

using namespace ut_util;

namespace optiling {
struct HardShrinkGradCompileInfo {};
} // namespace optiling

namespace {
constexpr const char* COMPILE_INFO = R"({
  "hardware_info": {
    "BT_SIZE": 0,
    "load3d_constraints": "1",
    "Intrinsic_fix_pipe_l0c2out": false,
    "Intrinsic_data_move_l12ub": true,
    "Intrinsic_data_move_l0c2ub": true,
    "Intrinsic_data_move_out2l1_nd2nz": false,
    "UB_SIZE": 245760,
    "L2_SIZE": 33554432,
    "L1_SIZE": 524288,
    "L0A_SIZE": 65536,
    "L0B_SIZE": 65536,
    "L0C_SIZE": 131072,
    "CORE_NUM": 64
  }
})";

struct TilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint32_t blockDim = 0;
    HardShrinkGradTilingData tilingData{};
};

gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

TilingResult RunTilingCase(const std::vector<int64_t>& gradDims, const std::vector<int64_t>& selfDims,
                           const std::vector<int64_t>& outDims, ge::DataType gradDtype, ge::DataType selfDtype,
                           float lambd = 0.5f, bool withAttrs = true, ge::DataType outDtype = ge::DT_UNDEFINED,
                           const int nullShapeSlot = -1)
{
    if (outDtype == ge::DT_UNDEFINED) {
        outDtype = gradDtype;
    }
    gert::StorageShape gradShape = MakeShape(gradDims);
    gert::StorageShape selfShape = MakeShape(selfDims);
    gert::StorageShape outShape = MakeShape(outDims);
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> version = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    GetPlatFormInfos(COMPILE_INFO, socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::HardShrinkGradCompileInfo compileInfo;
    TilingResult result;

    auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl("HardShrinkGrad");
    if (impl == nullptr || impl->tiling_parse == nullptr || impl->tiling == nullptr) {
        ADD_FAILURE() << "HardShrinkGrad tiling callbacks are not registered";
        return result;
    }
    auto parseContextHolder = gert::KernelRunContextFaker()
                                  .KernelIONum(1, 1)
                                  .Inputs({const_cast<char*>(COMPILE_INFO), reinterpret_cast<void*>(&platformInfo)})
                                  .Outputs({&compileInfo})
                                  .Build();
    auto* parseContext = parseContextHolder.GetContext<gert::TilingParseContext>();
    if (parseContext == nullptr || parseContext->GetPlatformInfo() == nullptr ||
        !parseContext->GetPlatformInfo()->Init()) {
        ADD_FAILURE() << "Failed to create the tiling parse context";
        return result;
    }
    auto* parsePlatformInfo = parseContext->GetPlatformInfo();
    parsePlatformInfo->SetPlatformRes("SoCInfo", socInfos);
    parsePlatformInfo->SetPlatformRes("AICoreSpec", aicoreSpec);
    parsePlatformInfo->SetCoreNumByCoreType("AICore");
    parsePlatformInfo->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    parsePlatformInfo->SetPlatformRes("version", version);
    if (impl->tiling_parse(parseContextHolder.GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
        ADD_FAILURE() << "HardShrinkGrad tiling parse failed";
        return result;
    }

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    if (tilingData == nullptr || workspaceHolder == nullptr) {
        ADD_FAILURE() << "Failed to allocate tiling test buffers";
        return result;
    }
    auto* workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    std::vector<std::pair<std::string, Ops::NN::AnyValue>>
        attrs = withAttrs ?
                    std::vector<std::pair<std::string, Ops::NN::AnyValue>>{
                        {"lambd", Ops::NN::AnyValue::CreateFrom<float>(lambd)}} :
                    std::vector<std::pair<std::string, Ops::NN::AnyValue>>{};
    gert::TilingContextFaker faker;
    faker.SetOpType("HardShrinkGrad").NodeIoNum(2, 1).IrInstanceNum({1, 1});
    if (nullShapeSlot == 0) {
        faker.InputShapes(std::vector<void*>{nullptr, &selfShape});
    } else if (nullShapeSlot == 1) {
        faker.InputShapes(std::vector<void*>{&gradShape, nullptr});
    } else {
        faker.InputShapes(std::vector<gert::StorageShape*>{&gradShape, &selfShape});
    }
    if (nullShapeSlot != 2) {
        faker.OutputShapes(std::vector<gert::StorageShape*>{&outShape});
    }
    faker.CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
        .NodeInputTd(0, gradDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, selfDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, outDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs(std::move(attrs))
        .TilingData(tilingData.get())
        .Workspace(workspace);
    auto contextHolder = faker.Build();
    auto* context = contextHolder.GetContext<gert::TilingContext>();
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        ADD_FAILURE() << "Failed to create the tiling context";
        return result;
    }
    auto* tilingPlatformInfo = context->GetPlatformInfo();
    tilingPlatformInfo->SetPlatformRes("SoCInfo", socInfos);
    tilingPlatformInfo->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingPlatformInfo->SetCoreNumByCoreType("AICore");
    tilingPlatformInfo->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    result.status = impl->tiling(context);
    result.blockDim = context->GetBlockDim();
    auto* rawTilingData = context->GetRawTilingData();
    if (result.status == ge::GRAPH_SUCCESS && rawTilingData != nullptr &&
        rawTilingData->GetDataSize() >= sizeof(HardShrinkGradTilingData)) {
        std::memcpy(&result.tilingData, rawTilingData->GetData(), sizeof(HardShrinkGradTilingData));
    }
    return result;
}
} // namespace

TEST(HardShrinkGradTilingTest, Float32LargeShapeDoubleBuffer)
{
    const auto result = RunTilingCase({16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}, ge::DT_FLOAT, ge::DT_FLOAT);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.blockDim, 64U);
    EXPECT_EQ(result.tilingData.totalNum, 2048);
    EXPECT_EQ(result.tilingData.blockFactor, 32);
    EXPECT_EQ(result.tilingData.ubFactor, 2048);
    EXPECT_FLOAT_EQ(result.tilingData.lambd, 0.5f);
}

TEST(HardShrinkGradTilingTest, Float16SmallShapeSingleBuffer)
{
    const auto result = RunTilingCase({16}, {16}, {16}, ge::DT_FLOAT16, ge::DT_FLOAT16);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.blockDim, 16U);
    EXPECT_EQ(result.tilingData.totalNum, 16);
    EXPECT_EQ(result.tilingData.blockFactor, 1);
    EXPECT_EQ(result.tilingData.ubFactor, 128);
}

TEST(HardShrinkGradTilingTest, Bfloat16LargeShapeDoubleBuffer)
{
    const auto result = RunTilingCase({12, 13, 9, 10, 12, 6}, {12, 13, 9, 10, 12, 6}, {12, 13, 9, 10, 12, 6},
                                      ge::DT_BF16, ge::DT_BF16);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.blockDim, 64U);
    EXPECT_EQ(result.tilingData.totalNum, 1010880);
    EXPECT_EQ(result.tilingData.blockFactor, 15795);
    EXPECT_EQ(result.tilingData.ubFactor, 11456);
}

// Empty tensor: totalIdx == 0 finishes with a single-core empty schedule.
TEST(HardShrinkGradTilingTest, EmptyTensorSingleCore)
{
    const auto result = RunTilingCase({0, 9, 1}, {0, 9, 1}, {0, 9, 1}, ge::DT_FLOAT16, ge::DT_FLOAT16);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.blockDim, 1U);
    EXPECT_EQ(result.tilingData.totalNum, 0);
}

// Scalar (0-dim) inputs are normalized to {1} by the host shape guard.
TEST(HardShrinkGradTilingTest, ScalarInputTreatedAsOne)
{
    const auto result = RunTilingCase({}, {}, {}, ge::DT_FLOAT, ge::DT_FLOAT);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.blockDim, 1U);
    EXPECT_EQ(result.tilingData.totalNum, 1);
    EXPECT_EQ(result.tilingData.ubFactor, 64);
}

// Same element count with different ranks is accepted (the kernel computes over
// the flat element layout; only the element counts of the three tensors must agree).
TEST(HardShrinkGradTilingTest, SameNumelRankDiffAccepted)
{
    const auto result = RunTilingCase({2, 3}, {6}, {2, 3}, ge::DT_FLOAT, ge::DT_FLOAT);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingData.totalNum, 6);
}

// Same element count with different per-dimension lengths is accepted (flat
// element-wise layout, same contract as above).
TEST(HardShrinkGradTilingTest, SameNumelDimDiffAccepted)
{
    const auto result = RunTilingCase({2, 3}, {3, 2}, {2, 3}, ge::DT_FLOAT, ge::DT_FLOAT);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingData.totalNum, 6);
}

// Negative: element count mismatch among the three tensors must be rejected.
TEST(HardShrinkGradTilingTest, NumelMismatchRejected)
{
    const auto result = RunTilingCase({2, 3}, {2, 4}, {2, 3}, ge::DT_FLOAT, ge::DT_FLOAT);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

// Negative: dtype outside the supported list (fp16/fp32/bf16) must be rejected.
TEST(HardShrinkGradTilingTest, DtypeUnsupportedRejected)
{
    const auto result = RunTilingCase({2, 3}, {2, 3}, {2, 3}, ge::DT_DOUBLE, ge::DT_DOUBLE);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

// Negative: gradients and features dtypes differ, consistency check must fail.
TEST(HardShrinkGradTilingTest, DtypeMismatchRejected)
{
    const auto result = RunTilingCase({2, 3}, {2, 3}, {2, 3}, ge::DT_FLOAT16, ge::DT_FLOAT);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

// Negative: backprops dtype outside the supported list must be rejected at the tiling gate.
TEST(HardShrinkGradTilingTest, OutputDtypeUnsupportedRejected)
{
    const auto result = RunTilingCase({2, 3}, {2, 3}, {2, 3}, ge::DT_FLOAT, ge::DT_FLOAT, 0.5f, true, ge::DT_BOOL);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

// Negative: backprops dtype must match the gradients dtype (same-type contract).
TEST(HardShrinkGradTilingTest, OutputDtypeMismatchRejected)
{
    const auto result = RunTilingCase({2, 3}, {2, 3}, {2, 3}, ge::DT_BF16, ge::DT_BF16, 0.5f, true, ge::DT_FLOAT);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

// Negative lambd is stored verbatim in the tiling data (literal threshold
// semantics: a negative lambd makes the dead-zone comparison always false).
TEST(HardShrinkGradTilingTest, NegativeLambdPassedThrough)
{
    const auto result = RunTilingCase({16}, {16}, {16}, ge::DT_FLOAT, ge::DT_FLOAT, -1.0f);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(result.tilingData.lambd, -1.0f);
}

// Negative: missing node attrs (no lambd attribute) must fail the null check.
TEST(HardShrinkGradTilingTest, MissingAttrsRejected)
{
    const auto result = RunTilingCase({16}, {16}, {16}, ge::DT_FLOAT, ge::DT_FLOAT, 0.5f, false);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

// Negative: a null gradients input shape must fail the shape null guard.
TEST(HardShrinkGradTilingTest, NullInputShapeGradRejected)
{
    const auto result = RunTilingCase({2, 3}, {2, 3}, {2, 3}, ge::DT_FLOAT, ge::DT_FLOAT, 0.5f, true, ge::DT_UNDEFINED,
                                      0);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

// Negative: a null features input shape must fail the shape null guard.
TEST(HardShrinkGradTilingTest, NullInputShapeSelfRejected)
{
    const auto result = RunTilingCase({2, 3}, {2, 3}, {2, 3}, ge::DT_FLOAT, ge::DT_FLOAT, 0.5f, true, ge::DT_UNDEFINED,
                                      1);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

// Negative: a null backprops output shape must fail the shape null guard.
TEST(HardShrinkGradTilingTest, NullOutputShapeRejected)
{
    const auto result = RunTilingCase({2, 3}, {2, 3}, {2, 3}, ge::DT_FLOAT, ge::DT_FLOAT, 0.5f, true, ge::DT_UNDEFINED,
                                      2);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}
