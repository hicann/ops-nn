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

#include "kernel_run_context_facker.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "../../../../op_kernel/group_norm_tiling_data.h"
#include "../../../../op_kernel/group_norm_tiling_key.h"

using namespace ge;
using namespace std;
using namespace ut_util;

namespace {
struct GroupNormCompileInfo {};

constexpr const char* COMPILE_INFO = R"({
    "hardware_info": {
        "BT_SIZE": 0,
        "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false,
        "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true,
        "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 262144,
        "L2_SIZE": 33554432,
        "L1_SIZE": 524288,
        "L0A_SIZE": 65536,
        "L0B_SIZE": 65536,
        "L0C_SIZE": 131072,
        "CORE_NUM": 20
    }
})";

void InitPlatform(fe::PlatFormInfos& platformInfo)
{
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersionInfos = {{"Short_SoC_version", "Ascend910B"}, {"NpuArch", "220"}};
    GetPlatFormInfos(COMPILE_INFO, socInfos, aicoreSpec, intrinsics);

    platformInfo.Init();
    platformInfo.SetPlatformRes("SoCInfo", socInfos);
    platformInfo.SetPlatformRes("AICoreSpec", aicoreSpec);
    platformInfo.SetCoreNumByCoreType("AICore");
    platformInfo.SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    platformInfo.SetPlatformRes("version", socVersionInfos);
}
} // namespace

class TestGroupNormTiling : public testing::Test {};

TEST_F(TestGroupNormTiling, group_norm_tiling_fp16_small_group)
{
    fe::PlatFormInfos platformInfo;
    InitPlatform(platformInfo);
    GroupNormCompileInfo compileInfo;

    string opType("GroupNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(4, 2)
                            .Inputs({const_cast<char*>(COMPILE_INFO), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    gert::StorageShape xShape = {{2, 8, 4, 4}, {2, 8, 4, 4}};
    gert::StorageShape gammaShape = {{8}, {8}};
    gert::StorageShape betaShape = {{8}, {8}};
    gert::StorageShape yShape = {{2, 8, 4, 4}, {2, 8, 4, 4}};
    gert::StorageShape meanShape = {{2, 4}, {2, 4}};
    gert::StorageShape rstdShape = {{2, 4}, {2, 4}};
    auto tilingData = gert::TilingData::CreateCap(sizeof(GroupNormTilingData));
    ASSERT_NE(tilingData, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(1);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &gammaShape, &betaShape})
                      .OutputShapes({&yShape, &meanShape, &rstdShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                           {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001f)}})
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    auto context = holder.GetContext<gert::TilingContext>();
    EXPECT_EQ(tilingFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetTilingKey(), static_cast<uint64_t>(GROUP_NORM_SCH_SMALL_GROUP));
    EXPECT_EQ(context->GetBlockDim(), 8);

    const auto* data = reinterpret_cast<const GroupNormTilingData*>(context->GetRawTilingData()->GetData());
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->n, 2U);
    EXPECT_EQ(data->c, 8U);
    EXPECT_EQ(data->hxw, 16U);
    EXPECT_EQ(data->numGroups, 4U);
    EXPECT_EQ(data->channelsPerGroup, 2U);
    EXPECT_EQ(data->elementsPerGroup, 32U);
    EXPECT_EQ(data->groupNum, 8U);
    EXPECT_EQ(data->hasGamma, 1U);
    EXPECT_EQ(data->hasBeta, 1U);
}

TEST_F(TestGroupNormTiling, group_norm_tiling_fp32_single_large_group)
{
    fe::PlatFormInfos platformInfo;
    InitPlatform(platformInfo);
    GroupNormCompileInfo compileInfo;

    string opType("GroupNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    gert::StorageShape xShape = {{1, 1, 16384}, {1, 1, 16384}};
    gert::StorageShape yShape = {{1, 1, 16384}, {1, 1, 16384}};
    gert::StorageShape meanShape = {{1, 1}, {1, 1}};
    gert::StorageShape rstdShape = {{1, 1}, {1, 1}};
    auto tilingData = gert::TilingData::CreateCap(sizeof(GroupNormTilingData));
    ASSERT_NE(tilingData, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(1);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, nullptr, nullptr})
                      .OutputShapes({&yShape, &meanShape, &rstdShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    auto context = holder.GetContext<gert::TilingContext>();
    EXPECT_EQ(tilingFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetTilingKey(), static_cast<uint64_t>(GROUP_NORM_SCH_GENERAL));
    EXPECT_EQ(context->GetBlockDim(), 1);

    const auto* data = reinterpret_cast<const GroupNormTilingData*>(context->GetRawTilingData()->GetData());
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->n, 1U);
    EXPECT_EQ(data->c, 1U);
    EXPECT_EQ(data->hxw, 16384U);
    EXPECT_EQ(data->numGroups, 1U);
    EXPECT_EQ(data->channelsPerGroup, 1U);
    EXPECT_EQ(data->elementsPerGroup, 16384U);
    EXPECT_EQ(data->groupNum, 1U);
    EXPECT_EQ(data->hasGamma, 0U);
    EXPECT_EQ(data->hasBeta, 0U);
}

TEST_F(TestGroupNormTiling, group_norm_tiling_invalid_group)
{
    fe::PlatFormInfos platformInfo;
    InitPlatform(platformInfo);
    GroupNormCompileInfo compileInfo;

    string opType("GroupNorm");
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    gert::StorageShape xShape = {{2, 8, 4, 4}, {2, 8, 4, 4}};
    gert::StorageShape yShape = {{2, 8, 4, 4}, {2, 8, 4, 4}};
    gert::StorageShape meanShape = {{2, 3}, {2, 3}};
    gert::StorageShape rstdShape = {{2, 3}, {2, 3}};
    auto tilingData = gert::TilingData::CreateCap(sizeof(GroupNormTilingData));
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(1);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, nullptr, nullptr})
                      .OutputShapes({&yShape, &meanShape, &rstdShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeAttrs({{"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(3)}})
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    EXPECT_EQ(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_FAILED);
}
