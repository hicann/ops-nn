/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "../../../op_kernel/masked_scatter_v2_tiling_data.h"

using namespace ge;
using namespace std;
using namespace ut_util;

namespace {
class MaskedScatterV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MaskedScatterV2Tiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MaskedScatterV2Tiling TearDown" << std::endl; }
};

// [TODO-CONTRIB] 950PR 平台 fake 平台信息键名（AIV 核数等）需与仓内
// platform_infos_def 的 ascend950pr JSON 对齐后调整；此处沿用 A2 版 UT 骨架。
const char* kCompileInfo = R"({
    "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                    "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                    "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                    "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288,
                    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                    "CORE_NUM": 48}
                    })";

struct MaskedScatterV2CompileInfo {};

void SetPlatformInfo(fe::PlatFormInfos* platformInfo, map<string, string>& socInfos, map<string, string>& aicoreSpec,
                     map<string, string>& intrinsics)
{
    platformInfo->SetPlatformRes("SoCInfo", socInfos);
    platformInfo->SetPlatformRes("AICoreSpec", aicoreSpec);
    platformInfo->SetCoreNumByCoreType("AICore");
    platformInfo->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
}

ge::graphStatus RunMaskedScatterV2Tiling(ge::DataType xDtype, ge::DataType maskDtype, ge::DataType updatesDtype,
                                         ge::DataType yDtype, gert::StorageShape& xShape, gert::StorageShape& maskShape,
                                         gert::StorageShape& updatesShape, gert::StorageShape& yShape,
                                         MaskedScatterV2TilingData* outTilingData, size_t* outWorkspaceSize = nullptr)
{
    std::string opType("MaskedScatterV2");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    if (opImpl == nullptr || opImpl->tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(kCompileInfo, socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    SetPlatformInfo(&platformInfo, socInfos, aicoreSpec, intrinsics);
    MaskedScatterV2CompileInfo compileInfo;

    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(kCompileInfo), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();
    SetPlatformInfo(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo(), socInfos, aicoreSpec,
                    intrinsics);

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    if (tilingData == nullptr || workspaceSize == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto holder = gert::TilingContextFaker()
                      .SetOpType("MaskedScatterV2")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &maskShape, &updatesShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, maskDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, updatesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(tilingData.get())
                      .Workspace(workspaceSize)
                      .Build();
    auto tilingContext = holder.GetContext<gert::TilingContext>();
    if (tilingContext == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto ret = opImpl->tiling(tilingContext);
    if (ret == ge::GRAPH_SUCCESS && outWorkspaceSize != nullptr && workspaceSize->GetSize() > 0) {
        *outWorkspaceSize = static_cast<const size_t*>(workspaceSize->GetData())[0];
    }
    if (ret == ge::GRAPH_SUCCESS && outTilingData != nullptr) {
        auto rawTilingData = tilingContext->GetRawTilingData();
        if (rawTilingData != nullptr && rawTilingData->GetData() != nullptr) {
            auto tiling = reinterpret_cast<const MaskedScatterV2TilingData*>(rawTilingData->GetData());
            *outTilingData = *tiling;
        }
    }
    return ret;
}
} // namespace

// 同 shape：tiling 字段与 workspace（SyncAll 计数槽）校验
TEST_F(MaskedScatterV2Tiling, masked_scatter_v2_float32_success)
{
    gert::StorageShape xShape = {{1024, 4096}, {1024, 4096}};
    gert::StorageShape maskShape = {{1024, 4096}, {1024, 4096}};
    gert::StorageShape updatesShape = {{100}, {100}};
    gert::StorageShape yShape = {{1024, 4096}, {1024, 4096}};
    MaskedScatterV2TilingData tilingData{};
    size_t workspaceSize = 0;

    EXPECT_EQ(RunMaskedScatterV2Tiling(ge::DT_FLOAT, ge::DT_BOOL, ge::DT_FLOAT, ge::DT_FLOAT, xShape, maskShape,
                                       updatesShape, yShape, &tilingData, &workspaceSize),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingData.total, 1024 * 4096);
    EXPECT_EQ(tilingData.sourceLen, 100);
    EXPECT_EQ(tilingData.useSync, 1); // 多核
    // workspace = (coreNum*32 + 512) * 4B（SyncAll flags + counts）
    EXPECT_EQ(workspaceSize, (48U * 32U + 512U) * 4U);
}

// 标量（rank0）：EnsureNotScalar 将 xShape/updatesShape 视作 [1]
TEST_F(MaskedScatterV2Tiling, masked_scatter_v2_scalar_ensured)
{
    gert::StorageShape xShape = {{}, {}};
    gert::StorageShape maskShape = {{}, {}};
    gert::StorageShape updatesShape = {{}, {}};
    gert::StorageShape yShape = {{}, {}};
    MaskedScatterV2TilingData tilingData{};
    size_t workspaceSize = 0;

    EXPECT_EQ(RunMaskedScatterV2Tiling(ge::DT_FLOAT, ge::DT_BOOL, ge::DT_FLOAT, ge::DT_FLOAT, xShape, maskShape,
                                       updatesShape, yShape, &tilingData, &workspaceSize),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingData.total, 1);
    EXPECT_EQ(tilingData.sourceLen, 1);
    EXPECT_EQ(tilingData.useSync, 0); // 单 chunk，无需跨核同步
    EXPECT_EQ(tilingData.maskIsBcast, 0);
}

// dim-first 广播（mask=[1, D]）：native 准入（total>=8192、mask>=32B、内层扩展 stride==1）
TEST_F(MaskedScatterV2Tiling, masked_scatter_v2_dim_first_broadcast_admitted)
{
    gert::StorageShape xShape = {{64, 2048}, {64, 2048}};
    gert::StorageShape maskShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape updatesShape = {{100}, {100}};
    gert::StorageShape yShape = {{64, 2048}, {64, 2048}};
    MaskedScatterV2TilingData tilingData{};

    EXPECT_EQ(RunMaskedScatterV2Tiling(ge::DT_FLOAT, ge::DT_BOOL, ge::DT_FLOAT, ge::DT_FLOAT, xShape, maskShape,
                                       updatesShape, yShape, &tilingData),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingData.maskIsBcast, 1);
    EXPECT_EQ(tilingData.maskRank, 2);
    EXPECT_EQ(tilingData.maskSize[0], 64);
    EXPECT_EQ(tilingData.maskSize[1], 2048);
    EXPECT_EQ(tilingData.maskStride[0], 0); // 广播维
    EXPECT_EQ(tilingData.maskStride[1], 1); // 内层连续
}

// dim-last 广播（mask=[B,1]）：native 不准入（内层扩展 stride != 1），tiling 失败并提示预展开
TEST_F(MaskedScatterV2Tiling, masked_scatter_v2_dim_last_broadcast_rejected)
{
    gert::StorageShape xShape = {{64, 2048}, {64, 2048}};
    gert::StorageShape maskShape = {{64, 1}, {64, 1}};
    gert::StorageShape updatesShape = {{100}, {100}};
    gert::StorageShape yShape = {{64, 2048}, {64, 2048}};
    EXPECT_EQ(RunMaskedScatterV2Tiling(ge::DT_FLOAT, ge::DT_BOOL, ge::DT_FLOAT, ge::DT_FLOAT, xShape, maskShape,
                                       updatesShape, yShape, nullptr),
              ge::GRAPH_FAILED);
}

// native 准入回退：total < 8192 的广播
TEST_F(MaskedScatterV2Tiling, masked_scatter_v2_broadcast_too_small_rejected)
{
    gert::StorageShape xShape = {{8, 16}, {8, 16}};
    gert::StorageShape maskShape = {{1, 16}, {1, 16}};
    gert::StorageShape updatesShape = {{10}, {10}};
    gert::StorageShape yShape = {{8, 16}, {8, 16}};
    EXPECT_EQ(RunMaskedScatterV2Tiling(ge::DT_FLOAT, ge::DT_BOOL, ge::DT_FLOAT, ge::DT_FLOAT, xShape, maskShape,
                                       updatesShape, yShape, nullptr),
              ge::GRAPH_FAILED);
}

// mask dtype 非法
TEST_F(MaskedScatterV2Tiling, masked_scatter_v2_mask_dtype_invalid)
{
    gert::StorageShape xShape = {{1024, 4096}, {1024, 4096}};
    gert::StorageShape maskShape = {{1024, 4096}, {1024, 4096}};
    gert::StorageShape updatesShape = {{100}, {100}};
    gert::StorageShape yShape = {{1024, 4096}, {1024, 4096}};
    EXPECT_EQ(RunMaskedScatterV2Tiling(ge::DT_FLOAT, ge::DT_INT8, ge::DT_FLOAT, ge::DT_FLOAT, xShape, maskShape,
                                       updatesShape, yShape, nullptr),
              ge::GRAPH_FAILED);
}

// dtype 不一致
TEST_F(MaskedScatterV2Tiling, masked_scatter_v2_dtype_mismatch)
{
    gert::StorageShape xShape = {{1024, 4096}, {1024, 4096}};
    gert::StorageShape maskShape = {{1024, 4096}, {1024, 4096}};
    gert::StorageShape updatesShape = {{100}, {100}};
    gert::StorageShape yShape = {{1024, 4096}, {1024, 4096}};
    EXPECT_EQ(RunMaskedScatterV2Tiling(ge::DT_FLOAT, ge::DT_BOOL, ge::DT_FLOAT16, ge::DT_FLOAT, xShape, maskShape,
                                       updatesShape, yShape, nullptr),
              ge::GRAPH_FAILED);
}

// dtype 白名单（v2 首批 4 种）
TEST_F(MaskedScatterV2Tiling, masked_scatter_v2_dtype_whitelist)
{
    gert::StorageShape xShape = {{1024, 4096}, {1024, 4096}};
    gert::StorageShape maskShape = {{1024, 4096}, {1024, 4096}};
    gert::StorageShape updatesShape = {{100}, {100}};
    gert::StorageShape yShape = {{1024, 4096}, {1024, 4096}};
    EXPECT_EQ(RunMaskedScatterV2Tiling(ge::DT_UINT8, ge::DT_BOOL, ge::DT_UINT8, ge::DT_UINT8, xShape, maskShape,
                                       updatesShape, yShape, nullptr),
              ge::GRAPH_FAILED);
}
