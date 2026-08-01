/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <map>
#include <string>
#include <vector>

#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "../../../../op_kernel/arch35/deep_norm_grad_tiling_data.h"
#include "../../../../op_host/arch35/deep_norm_grad_tiling_arch35.h"

namespace {

constexpr uint64_t UB_SIZE = 245760;

struct TilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint64_t key = UINT64_MAX;
    uint32_t blockDim = 0;
    float invCols = 0.0f;
};

gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (int64_t dim : dims) {
        shape.MutableStorageShape().AppendDim(dim);
        shape.MutableOriginShape().AppendDim(dim);
    }
    return shape;
}

TilingResult RunTiling(const std::vector<int64_t>& leadingDims, const std::vector<int64_t>& gammaDims,
                       ge::DataType dtype, bool invalidXShape = false)
{
    std::string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": )" + std::to_string(UB_SIZE) +
                                    R"(, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64}})";
    std::map<std::string, std::string> socInfos = {{"Short_SoC_version", "Ascend950"}};
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::DeepNormGradCompileInfo compileInfo;
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("DeepNormGrad");
    if (opImpl == nullptr) {
        return {};
    }

    auto parseHolder = gert::KernelRunContextFaker()
                           .KernelIONum(2, 1)
                           .Inputs(
                               {const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
                           .Outputs({&compileInfo})
                           .Build();
    auto parseContext = parseHolder.GetContext<gert::TilingParseContext>();
    parseContext->GetPlatformInfo()->Init();
    parseContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    parseContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    if (opImpl->tiling_parse(parseHolder.GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
        return {};
    }

    std::vector<int64_t> xDims = leadingDims;
    xDims.insert(xDims.end(), gammaDims.begin(), gammaDims.end());
    std::vector<int64_t> meanDims = leadingDims;
    meanDims.insert(meanDims.end(), gammaDims.size(), 1);
    std::vector<int64_t> badXDims = xDims;
    if (invalidXShape) {
        badXDims.back() += 1;
    }

    auto dyShape = MakeShape(xDims);
    auto xShape = MakeShape(badXDims);
    auto gxShape = MakeShape(xDims);
    auto gammaShape = MakeShape(gammaDims);
    auto meanShape = MakeShape(meanDims);
    auto rstdShape = MakeShape(meanDims);
    auto dxShape = MakeShape(xDims);
    auto dgxShape = MakeShape(xDims);
    auto dbetaShape = MakeShape(gammaDims);
    auto dgammaShape = MakeShape(gammaDims);

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    if (tilingData == nullptr || workspace == nullptr) {
        return {};
    }
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(6, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dyShape, &xShape, &gxShape, &gammaShape, &meanShape, &rstdShape})
                      .OutputShapes({&dxShape, &dgxShape, &dbetaShape, &dgammaShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"alpha", Ops::NN::AnyValue::CreateFrom<float>(0.3f)}})
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();
    auto context = holder.GetContext<gert::TilingContext>();
    context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    TilingResult result;
    result.status = opImpl->tiling(context);
    if (result.status == ge::GRAPH_SUCCESS) {
        result.key = context->GetTilingKey();
        result.blockDim = context->GetBlockDim();
        auto rawTilingData = context->GetRawTilingData();
        if (rawTilingData != nullptr && rawTilingData->GetData() != nullptr) {
            auto tiling = reinterpret_cast<const DeepNormGradTilingDataArch35*>(rawTilingData->GetData());
            result.invCols = tiling->invCols;
        }
    }
    return result;
}

} // namespace

TEST(DeepNormGradTilingArch35, Fp32UnalignedD)
{
    auto result = RunTiling({4, 3}, {1000}, ge::DT_FLOAT);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, 0);
    EXPECT_GT(result.blockDim, 0);
}

TEST(DeepNormGradTilingArch35, Fp16LargeDSplit)
{
    auto result = RunTiling({2}, {65537}, ge::DT_FLOAT16);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, 0);
}

TEST(DeepNormGradTilingArch35, InvColsUsesDoubleBeforeFloatCast)
{
    constexpr uint64_t cols = 16777217ULL;
    auto result = RunTiling({1}, {static_cast<int64_t>(cols)}, ge::DT_FLOAT);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(result.invCols, static_cast<float>(1.0 / static_cast<double>(cols)));
    EXPECT_NE(result.invCols, 1.0f / static_cast<float>(cols));
}

TEST(DeepNormGradTilingArch35, Bf16MultiDimGamma)
{
    auto result = RunTiling({8}, {7, 9, 11}, ge::DT_BF16);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, 0);
}

TEST(DeepNormGradTilingArch35, SmallDManyRows)
{
    auto result = RunTiling({4096}, {9}, ge::DT_FLOAT16);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_GT(result.blockDim, 1);
}

TEST(DeepNormGradTilingArch35, RejectMismatchedXShape)
{
    auto result = RunTiling({4}, {128}, ge::DT_FLOAT, true);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST(DeepNormGradTilingArch35, RejectZeroDyDim)
{
    auto result = RunTiling({4, 0}, {128}, ge::DT_BF16);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}
