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
 * \file test_npu_scatter_add_bwd_tiling.cpp
 * \brief NpuScatterAddBwd tiling UT
 */
#include <gtest/gtest.h>
#include <map>
#include <string>
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_faker.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "../../../../op_host/arch22/npu_scatter_add_bwd_tiling.h"

using namespace ge;
using namespace std;

class TestNpuScatterAddBwdTiling : public testing::Test {};

namespace {
constexpr const char* COMPILE_INFO = R"({
    "hardware_info": {
        "BT_SIZE": 0,
        "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false,
        "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true,
        "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 196608,
        "L2_SIZE": 33554432,
        "L1_SIZE": 524288,
        "L0A_SIZE": 65536,
        "L0B_SIZE": 65536,
        "L0C_SIZE": 131072,
        "CORE_NUM": 48
    }
})";

void SetPlatformInfo(gert::TilingContext* context)
{
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(COMPILE_INFO, socInfos, aicoreSpec, intrinsics);
    map<string, string> socVersionInfos = {{"Short_SoC_version", "Ascend910B"}, {"NpuArch", "220"}};
    ASSERT_NE(context->GetPlatformInfo(), nullptr);
    context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    context->GetPlatformInfo()->SetPlatformRes("version", socVersionInfos);
}
} // namespace

// bf16，tilingKey = 0
TEST_F(TestNpuScatterAddBwdTiling, npu_scatter_add_bwd_tiling_bf16_success)
{
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    int32_t compileInfo = 0;
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NpuScatterAddBwd")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    gert::StorageShape yGrad = {{4, 16}, {4, 16}};
    gert::StorageShape x = {{8, 16}, {8, 16}};
    gert::StorageShape s = {{8}, {8}};
    gert::StorageShape indices = {{8}, {8}};
    gert::StorageShape xGrad = {{8, 16}, {8, 16}};
    gert::StorageShape sGrad = {{8}, {8}};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("NpuScatterAddBwd")
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1}, {1, 1})
                      .InputShapes({&yGrad, &x, &s, &indices})
                      .OutputShapes({&xGrad, &sGrad})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    auto context = holder.GetContext<gert::TilingContext>();
    SetPlatformInfo(context);
    ASSERT_EQ(tilingFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetTilingKey(), 0U);
    EXPECT_GT(context->GetBlockDim(), 0U);
    auto raw = reinterpret_cast<const optiling::NpuScatterAddBwdTilingData*>(context->GetRawTilingData()->GetData());
    EXPECT_EQ(raw->rowsPerCore, 1U);
    EXPECT_EQ(raw->totalRows, 8U);
    EXPECT_EQ(raw->hiddenState, 16U);
    EXPECT_EQ(raw->alignHiddenState, 16U);
    EXPECT_EQ(raw->usedCoreNum, 48U);
}

// fp16，tilingKey = 1
TEST_F(TestNpuScatterAddBwdTiling, npu_scatter_add_bwd_tiling_fp16_success)
{
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    int32_t compileInfo = 0;
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NpuScatterAddBwd")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    gert::StorageShape yGrad = {{4, 32}, {4, 32}};
    gert::StorageShape x = {{16, 32}, {16, 32}};
    gert::StorageShape s = {{16}, {16}};
    gert::StorageShape indices = {{16}, {16}};
    gert::StorageShape xGrad = {{16, 32}, {16, 32}};
    gert::StorageShape sGrad = {{16}, {16}};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("NpuScatterAddBwd")
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1}, {1, 1})
                      .InputShapes({&yGrad, &x, &s, &indices})
                      .OutputShapes({&xGrad, &sGrad})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    auto context = holder.GetContext<gert::TilingContext>();
    SetPlatformInfo(context);
    ASSERT_EQ(tilingFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetTilingKey(), 1U);
    auto raw = reinterpret_cast<const optiling::NpuScatterAddBwdTilingData*>(context->GetRawTilingData()->GetData());
    EXPECT_EQ(raw->totalRows, 16U);
    EXPECT_EQ(raw->hiddenState, 32U);
    EXPECT_EQ(raw->alignHiddenState, 32U);
}

// x为3维张量，校验失败
TEST_F(TestNpuScatterAddBwdTiling, npu_scatter_add_bwd_tiling_x_dim3_failed)
{
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    int32_t compileInfo = 0;
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NpuScatterAddBwd")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    gert::StorageShape yGrad = {{4, 16}, {4, 16}};
    gert::StorageShape x = {{8, 16, 16}, {8, 16, 16}};
    gert::StorageShape s = {{8}, {8}};
    gert::StorageShape indices = {{8}, {8}};
    gert::StorageShape xGrad = {{8, 16, 16}, {8, 16, 16}};
    gert::StorageShape sGrad = {{8}, {8}};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("NpuScatterAddBwd")
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1}, {1, 1})
                      .InputShapes({&yGrad, &x, &s, &indices})
                      .OutputShapes({&xGrad, &sGrad})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    auto context = holder.GetContext<gert::TilingContext>();
    SetPlatformInfo(context);
    EXPECT_EQ(tilingFunc(context), ge::GRAPH_FAILED);
}

// x为fp32类型，校验失败
TEST_F(TestNpuScatterAddBwdTiling, npu_scatter_add_bwd_tiling_fp32_dtype_failed)
{
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    int32_t compileInfo = 0;
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NpuScatterAddBwd")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    gert::StorageShape yGrad = {{4, 16}, {4, 16}};
    gert::StorageShape x = {{8, 16}, {8, 16}};
    gert::StorageShape s = {{8}, {8}};
    gert::StorageShape indices = {{8}, {8}};
    gert::StorageShape xGrad = {{8, 16}, {8, 16}};
    gert::StorageShape sGrad = {{8}, {8}};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("NpuScatterAddBwd")
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1}, {1, 1})
                      .InputShapes({&yGrad, &x, &s, &indices})
                      .OutputShapes({&xGrad, &sGrad})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    auto context = holder.GetContext<gert::TilingContext>();
    SetPlatformInfo(context);
    EXPECT_EQ(tilingFunc(context), ge::GRAPH_FAILED);
}

// 隐藏维度H超出UB容量限制，校验失败
TEST_F(TestNpuScatterAddBwdTiling, npu_scatter_add_bwd_tiling_hidden_too_large_failed)
{
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    int32_t compileInfo = 0;
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NpuScatterAddBwd")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    gert::StorageShape yGrad = {{4, 100000}, {4, 100000}};
    gert::StorageShape x = {{8, 100000}, {8, 100000}};
    gert::StorageShape s = {{8}, {8}};
    gert::StorageShape indices = {{8}, {8}};
    gert::StorageShape xGrad = {{8, 100000}, {8, 100000}};
    gert::StorageShape sGrad = {{8}, {8}};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("NpuScatterAddBwd")
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1}, {1, 1})
                      .InputShapes({&yGrad, &x, &s, &indices})
                      .OutputShapes({&xGrad, &sGrad})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    auto context = holder.GetContext<gert::TilingContext>();
    SetPlatformInfo(context);
    EXPECT_EQ(tilingFunc(context), ge::GRAPH_FAILED);
}
