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
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "../../../../op_graph/experimental_index_proto.h"
#include "../../../../op_kernel/index_tiling_data.h"

using namespace ge;
using namespace std;

class TestIndexTiling : public testing::Test {
};

namespace {
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
        "CORE_NUM": 64
    }
})";

void SetPlatformInfo(gert::TilingContext* context)
{
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(COMPILE_INFO, socInfos, aicoreSpec, intrinsics);
    map<string, string> socVersionInfos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    ASSERT_NE(context->GetPlatformInfo(), nullptr);
    context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    context->GetPlatformInfo()->SetPlatformRes("version", socVersionInfos);
}
}  // namespace

TEST_F(TestIndexTiling, index_aicore_tiling_two_indices_success)
{
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    int32_t compileInfo = 0;
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("IndexAiCore")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    gert::StorageShape x = {{8, 9, 4}, {8, 9, 4}};
    gert::StorageShape indexedSizes = {{3}, {3}};
    gert::StorageShape indexedStrides = {{3}, {3}};
    gert::StorageShape index0 = {{2, 1}, {2, 1}};
    gert::StorageShape index1 = {{1, 3}, {1, 3}};
    gert::StorageShape y = {{2, 3, 4}, {2, 3, 4}};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("IndexAiCore")
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1, 1, 2}, {1})
                      .InputShapes({&x, &indexedSizes, &indexedStrides, &index0, &index1})
                      .OutputShapes({&y})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    auto context = holder.GetContext<gert::TilingContext>();
    SetPlatformInfo(context);
    ASSERT_EQ(tilingFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetTilingKey(), 0);
    EXPECT_EQ(context->GetBlockDim(), 24);
    auto raw = reinterpret_cast<const optiling::IndexTilingData*>(context->GetRawTilingData()->GetData());
    EXPECT_EQ(raw->xRank, 3U);
    EXPECT_EQ(raw->indexedDimNum, 2U);
    EXPECT_EQ(raw->indexRank, 2U);
    EXPECT_EQ(raw->totalNum, 24U);
    EXPECT_EQ(raw->yShape[0], 2U);
    EXPECT_EQ(raw->yShape[1], 3U);
    EXPECT_EQ(raw->yShape[2], 4U);
}

TEST_F(TestIndexTiling, index_aicore_tiling_single_index_caps_core_num)
{
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    int32_t compileInfo = 0;
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("IndexAiCore")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    gert::StorageShape x = {{100, 8}, {100, 8}};
    gert::StorageShape indexedSizes = {{2}, {2}};
    gert::StorageShape indexedStrides = {{2}, {2}};
    gert::StorageShape index0 = {{64}, {64}};
    gert::StorageShape y = {{64, 8}, {64, 8}};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("IndexAiCore")
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1}, {1})
                      .InputShapes({&x, &indexedSizes, &indexedStrides, &index0})
                      .OutputShapes({&y})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    auto context = holder.GetContext<gert::TilingContext>();
    SetPlatformInfo(context);
    ASSERT_EQ(tilingFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetBlockDim(), 64);
    auto raw = reinterpret_cast<const optiling::IndexTilingData*>(context->GetRawTilingData()->GetData());
    EXPECT_EQ(raw->totalNum, 512U);
    EXPECT_EQ(raw->usedCoreNum, 64U);
}

TEST_F(TestIndexTiling, index_aicore_tiling_no_dynamic_index_failed)
{
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    int32_t compileInfo = 0;
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("IndexAiCore")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    gert::StorageShape x = {{8, 9}, {8, 9}};
    gert::StorageShape indexedSizes = {{2}, {2}};
    gert::StorageShape indexedStrides = {{2}, {2}};
    gert::StorageShape y = {{8, 9}, {8, 9}};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("IndexAiCore")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1, 0}, {1})
                      .InputShapes({&x, &indexedSizes, &indexedStrides})
                      .OutputShapes({&y})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    auto context = holder.GetContext<gert::TilingContext>();
    SetPlatformInfo(context);
    EXPECT_EQ(tilingFunc(context), ge::GRAPH_FAILED);
}

TEST_F(TestIndexTiling, index_aicore_tiling_index_shape_not_broadcast_failed)
{
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    int32_t compileInfo = 0;
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("IndexAiCore")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    gert::StorageShape x = {{8, 9}, {8, 9}};
    gert::StorageShape indexedSizes = {{2}, {2}};
    gert::StorageShape indexedStrides = {{2}, {2}};
    gert::StorageShape index0 = {{2}, {2}};
    gert::StorageShape index1 = {{3}, {3}};
    gert::StorageShape y = {{2}, {2}};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("IndexAiCore")
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1, 1, 2}, {1})
                      .InputShapes({&x, &indexedSizes, &indexedStrides, &index0, &index1})
                      .OutputShapes({&y})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    auto context = holder.GetContext<gert::TilingContext>();
    SetPlatformInfo(context);
    EXPECT_EQ(tilingFunc(context), ge::GRAPH_FAILED);
}
