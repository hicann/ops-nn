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
#include "../../../op_kernel/masked_scatter_tiling_data.h"

using namespace ge;
using namespace std;
using namespace ut_util;

namespace {
class MaskedScatterTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MaskedScatterTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MaskedScatterTiling TearDown" << std::endl; }
};

const char* kCompileInfo = R"({
    "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                    "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                    "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                    "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                    "CORE_NUM": 40}
                    })";

struct MaskedScatterCompileInfo {};

void SetPlatformInfo(fe::PlatFormInfos* platformInfo, map<string, string>& socInfos, map<string, string>& aicoreSpec,
                     map<string, string>& intrinsics)
{
    platformInfo->SetPlatformRes("SoCInfo", socInfos);
    platformInfo->SetPlatformRes("AICoreSpec", aicoreSpec);
    platformInfo->SetCoreNumByCoreType("AICore");
    platformInfo->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
}

ge::graphStatus RunMaskedScatterTiling(ge::DataType xDtype, ge::DataType maskDtype, ge::DataType updatesDtype,
                                       ge::DataType yDtype, gert::StorageShape& xShape, gert::StorageShape& maskShape,
                                       gert::StorageShape& updatesShape, gert::StorageShape& yShape,
                                       MaskedScatterTilingData* outTilingData, size_t* outWorkspaceCount = nullptr,
                                       size_t* outWorkspaceSize = nullptr)
{
    std::string opType("MaskedScatter");
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
    MaskedScatterCompileInfo compileInfo;

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
                      .SetOpType("MaskedScatter")
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
    if (ret == ge::GRAPH_SUCCESS && outWorkspaceCount != nullptr) {
        *outWorkspaceCount = workspaceSize->GetSize();
    }
    if (ret == ge::GRAPH_SUCCESS && outWorkspaceSize != nullptr && workspaceSize->GetSize() > 0) {
        *outWorkspaceSize = static_cast<const size_t*>(workspaceSize->GetData())[0];
    }
    if (ret == ge::GRAPH_SUCCESS && outTilingData != nullptr) {
        auto rawTilingData = tilingContext->GetRawTilingData();
        if (rawTilingData != nullptr && rawTilingData->GetData() != nullptr) {
            auto tiling = reinterpret_cast<const MaskedScatterTilingData*>(rawTilingData->GetData());
            *outTilingData = *tiling;
        }
    }
    return ret;
}
} // namespace

TEST_F(MaskedScatterTiling, masked_scatter_float32_success)
{
    gert::StorageShape xShape = {{2, 4}, {2, 4}};
    gert::StorageShape maskShape = {{2, 4}, {2, 4}};
    gert::StorageShape updatesShape = {{3}, {3}};
    gert::StorageShape yShape = {{2, 4}, {2, 4}};
    MaskedScatterTilingData tilingData{};
    size_t workspaceCount = 0;
    size_t workspaceSize = 1;

    EXPECT_EQ(RunMaskedScatterTiling(ge::DT_FLOAT, ge::DT_BOOL, ge::DT_FLOAT, ge::DT_FLOAT, xShape, maskShape,
                                     updatesShape, yShape, &tilingData, &workspaceCount, &workspaceSize),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingData.numElemX, 8);
    EXPECT_EQ(tilingData.numElemMask, 8);
    EXPECT_EQ(tilingData.numElemUpdates, 3);
    EXPECT_EQ(tilingData.tilingCoreNum, 40);
    EXPECT_EQ(workspaceCount, 1);
    EXPECT_EQ(workspaceSize, 0);
}

TEST_F(MaskedScatterTiling, masked_scatter_mask_dtype_invalid)
{
    gert::StorageShape xShape = {{2, 4}, {2, 4}};
    gert::StorageShape maskShape = {{2, 4}, {2, 4}};
    gert::StorageShape updatesShape = {{3}, {3}};
    gert::StorageShape yShape = {{2, 4}, {2, 4}};
    EXPECT_EQ(RunMaskedScatterTiling(ge::DT_FLOAT, ge::DT_INT8, ge::DT_FLOAT, ge::DT_FLOAT, xShape, maskShape,
                                     updatesShape, yShape, nullptr),
              ge::GRAPH_FAILED);
}

TEST_F(MaskedScatterTiling, masked_scatter_shape_mismatch)
{
    gert::StorageShape xShape = {{2, 4}, {2, 4}};
    gert::StorageShape maskShape = {{2, 3}, {2, 3}};
    gert::StorageShape updatesShape = {{3}, {3}};
    gert::StorageShape yShape = {{2, 4}, {2, 4}};
    EXPECT_EQ(RunMaskedScatterTiling(ge::DT_FLOAT, ge::DT_BOOL, ge::DT_FLOAT, ge::DT_FLOAT, xShape, maskShape,
                                     updatesShape, yShape, nullptr),
              ge::GRAPH_FAILED);
}

TEST_F(MaskedScatterTiling, masked_scatter_dtype_mismatch)
{
    gert::StorageShape xShape = {{2, 4}, {2, 4}};
    gert::StorageShape maskShape = {{2, 4}, {2, 4}};
    gert::StorageShape updatesShape = {{3}, {3}};
    gert::StorageShape yShape = {{2, 4}, {2, 4}};
    EXPECT_EQ(RunMaskedScatterTiling(ge::DT_FLOAT, ge::DT_BOOL, ge::DT_FLOAT16, ge::DT_FLOAT, xShape, maskShape,
                                     updatesShape, yShape, nullptr),
              ge::GRAPH_FAILED);
}
