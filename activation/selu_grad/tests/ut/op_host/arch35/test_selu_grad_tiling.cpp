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
#include "test_cube_util.h"

namespace optiling {
struct SeluGradCompileInfo {};
} // namespace optiling

namespace {
constexpr const char* OP_TYPE = "SeluGrad";
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

ge::graphStatus RunTiling(ge::DataType gradientsDtype, ge::DataType outputsDtype, ge::DataType yDtype)
{
    gert::StorageShape shape = {{16}, {16}};
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> version = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    GetPlatFormInfos(COMPILE_INFO, socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::SeluGradCompileInfo compileInfo;

    auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE);
    if (impl == nullptr || impl->tiling_parse == nullptr || impl->tiling == nullptr) {
        ADD_FAILURE() << "SeluGrad tiling callbacks are not registered";
        return ge::GRAPH_FAILED;
    }

    auto parseContextHolder = gert::KernelRunContextFaker()
                                  .KernelIONum(2, 1)
                                  .Inputs({const_cast<char*>(COMPILE_INFO), reinterpret_cast<void*>(&platformInfo)})
                                  .Outputs({&compileInfo})
                                  .Build();
    auto* parseContext = parseContextHolder.GetContext<gert::TilingParseContext>();
    if (parseContext == nullptr || parseContext->GetPlatformInfo() == nullptr ||
        !parseContext->GetPlatformInfo()->Init()) {
        ADD_FAILURE() << "Failed to create the SeluGrad tiling parse context";
        return ge::GRAPH_FAILED;
    }
    auto* parsePlatformInfo = parseContext->GetPlatformInfo();
    parsePlatformInfo->SetPlatformRes("SoCInfo", socInfos);
    parsePlatformInfo->SetPlatformRes("AICoreSpec", aicoreSpec);
    parsePlatformInfo->SetCoreNumByCoreType("AICore");
    parsePlatformInfo->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    parsePlatformInfo->SetPlatformRes("version", version);
    if (impl->tiling_parse(parseContextHolder.GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
        ADD_FAILURE() << "SeluGrad tiling parse failed";
        return ge::GRAPH_FAILED;
    }

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    if (tilingData == nullptr || workspaceHolder == nullptr) {
        ADD_FAILURE() << "Failed to allocate SeluGrad tiling test buffers";
        return ge::GRAPH_FAILED;
    }
    auto* workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    auto contextHolder = gert::TilingContextFaker()
                             .SetOpType(OP_TYPE)
                             .NodeIoNum(2, 1)
                             .IrInstanceNum({1, 1})
                             .InputShapes({&shape, &shape})
                             .OutputShapes({&shape})
                             .CompileInfo(&compileInfo)
                             .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                             .NodeInputTd(0, gradientsDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, outputsDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .TilingData(tilingData.get())
                             .Workspace(workspace)
                             .Build();
    return impl->tiling(contextHolder.GetContext<gert::TilingContext>());
}
} // namespace

TEST(SeluGradTilingTest, AcceptsMatchingSupportedDtypes)
{
    EXPECT_EQ(RunTiling(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT), ge::GRAPH_SUCCESS);
}

TEST(SeluGradTilingTest, RejectsMismatchedSupportedInputDtypes)
{
    EXPECT_EQ(RunTiling(ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT), ge::GRAPH_FAILED);
}
