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
#include "../../../../op_kernel/arch35/dynamic_quant_update_scatter_tiling_data.h"

namespace optiling {
struct DynamicQuantUpdateScatterCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    bool isRegbase = false;
};
} // namespace optiling

namespace {
constexpr uint64_t UB_SIZE = 245760;
constexpr uint64_t TILING_KEY_REGBASE_NO_SMOOTH = 0;
constexpr uint64_t TILING_KEY_REGBASE_WITH_SMOOTH = 1;

struct TilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint64_t key = UINT64_MAX;
    uint32_t blockDim = 0;
    DynamicQuantUpdateScatterRegbaseTilingData data{};
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

TilingResult RunTiling(const std::vector<int64_t>& varDims, const std::vector<int64_t>& updatesDims,
                       const std::vector<int64_t>& indicesDims, ge::DataType updatesDtype, ge::DataType indicesDtype,
                       int64_t axis, bool hasSmooth = true, const std::vector<int64_t>& varScaleOverride = {},
                       const std::vector<int64_t>& smoothOverride = {})
{
    std::string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": )" + std::to_string(UB_SIZE) +
                                    R"(, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64}})";
    std::map<std::string, std::string> socInfos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::DynamicQuantUpdateScatterCompileInfo compileInfo;
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuantUpdateScatter");
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
    compileInfo.isRegbase = true;

    std::vector<int64_t> varScaleDims = varDims;
    varScaleDims.back() = 1;
    if (!varScaleOverride.empty()) {
        varScaleDims = varScaleOverride;
    }
    std::vector<int64_t> smoothDims = {updatesDims.back()};
    if (!smoothOverride.empty()) {
        smoothDims = smoothOverride;
    }

    auto varShape = MakeShape(varDims);
    auto varScaleShape = MakeShape(varScaleDims);
    auto indicesShape = MakeShape(indicesDims);
    auto updatesShape = MakeShape(updatesDims);
    auto smoothShape = MakeShape(smoothDims);
    auto varOutShape = MakeShape(varDims);
    auto varScaleOutShape = MakeShape(varScaleDims);
    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    if (tilingData == nullptr || workspace == nullptr) {
        return {};
    }

    gert::KernelRunContextHolder holder;
    if (hasSmooth) {
        holder = gert::TilingContextFaker()
                     .NodeIoNum(5, 2)
                     .IrInstanceNum({1, 1, 1, 1, 1})
                     .InputShapes({&varShape, &varScaleShape, &indicesShape, &updatesShape, &smoothShape})
                     .OutputShapes({&varOutShape, &varScaleOutShape})
                     .CompileInfo(&compileInfo)
                     .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                     .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(2, indicesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(3, updatesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(4, updatesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeAttrs({{"reduce", Ops::NN::AnyValue::CreateFrom<std::string>("update")},
                                 {"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(axis)}})
                     .TilingData(tilingData.get())
                     .Workspace(workspace)
                     .Build();
    } else {
        holder = gert::TilingContextFaker()
                     .NodeIoNum(5, 2)
                     .IrInstanceNum({1, 1, 1, 1, 0})
                     .InputShapes({&varShape, &varScaleShape, &indicesShape, &updatesShape})
                     .OutputShapes({&varOutShape, &varScaleOutShape})
                     .CompileInfo(&compileInfo)
                     .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                     .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(2, indicesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(3, updatesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeAttrs({{"reduce", Ops::NN::AnyValue::CreateFrom<std::string>("update")},
                                 {"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(axis)}})
                     .TilingData(tilingData.get())
                     .Workspace(workspace)
                     .Build();
    }

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
        result.data = *reinterpret_cast<const DynamicQuantUpdateScatterRegbaseTilingData*>(
            context->GetRawTilingData()->GetData());
    }
    return result;
}
} // namespace

TEST(DynamicQuantUpdateScatterTilingArch35, Fp16Rank1NoSmoothAndMultipleUpdates)
{
    auto result = RunTiling({8, 2, 4, 128}, {8, 2, 2, 128}, {8}, ge::DT_FLOAT16, ge::DT_INT32, -2, false);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, TILING_KEY_REGBASE_NO_SMOOTH);
    EXPECT_EQ(result.blockDim, 32);
    EXPECT_EQ(result.data.updateAxisShape, 2);
    EXPECT_EQ(result.data.quantReptNum, 1);
}

TEST(DynamicQuantUpdateScatterTilingArch35, Bf16Rank2WithSmooth)
{
    auto result = RunTiling({6, 3, 8, 128}, {4, 3, 2, 128}, {4, 2}, ge::DT_BF16, ge::DT_INT64, -2);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, TILING_KEY_REGBASE_WITH_SMOOTH);
    EXPECT_EQ(result.data.indicesShapeRank, 2);
    EXPECT_EQ(result.data.numHead, 3);
    EXPECT_EQ(result.data.lastCoreBsNum, 1);
}

TEST(DynamicQuantUpdateScatterTilingArch35, MultipleQuantRowsWithNonAlignedOriginalLastDim)
{
    auto result = RunTiling({4, 2, 8, 2, 16}, {4, 2, 3, 2, 16}, {4}, ge::DT_FLOAT16, ge::DT_INT32, -3);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.blockDim, 24);
    EXPECT_EQ(result.data.eachCoreBsNum, 1);
    EXPECT_EQ(result.data.quantReptNum, 2);
    EXPECT_EQ(result.data.varOrigLastDimSize, 16);
    EXPECT_EQ(result.data.sizePerHead, 32);
}

TEST(DynamicQuantUpdateScatterTilingArch35, LargeLastDimUsesMultipleUbTiles)
{
    auto result = RunTiling({2, 1, 4, 131072}, {2, 1, 1, 131072}, {2}, ge::DT_FLOAT16, ge::DT_INT32, -2, false);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_GT(result.data.innerLoopTimes, 1);
    EXPECT_LT(result.data.innerLoopEle, result.data.varOrigLastDimSize);
}

TEST(DynamicQuantUpdateScatterTilingArch35, RejectLastAxis)
{
    auto result = RunTiling({2, 1, 4, 128}, {2, 1, 1, 128}, {2}, ge::DT_FLOAT16, ge::DT_INT32, -1);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST(DynamicQuantUpdateScatterTilingArch35, RejectUnalignedMergedTail)
{
    auto result = RunTiling({2, 1, 4, 3, 10}, {2, 1, 1, 3, 10}, {2}, ge::DT_FLOAT16, ge::DT_INT32, -3);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST(DynamicQuantUpdateScatterTilingArch35, RejectInvalidVarScaleShape)
{
    auto result = RunTiling({2, 1, 4, 128}, {2, 1, 1, 128}, {2}, ge::DT_FLOAT16, ge::DT_INT32, -2, true, {2, 1, 4, 2});
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST(DynamicQuantUpdateScatterTilingArch35, RejectInvalidSmoothShape)
{
    auto result = RunTiling({2, 1, 4, 128}, {2, 1, 1, 128}, {2}, ge::DT_FLOAT16, ge::DT_INT32, -2, true, {}, {2, 64});
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}
