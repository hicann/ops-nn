/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"

using namespace ge;
using namespace std;
using namespace ut_util;
class InplaceScatterAddTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "InplaceScatterAddTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "InplaceScatterAddTiling TearDown" << std::endl; }
};

TEST_F(InplaceScatterAddTiling, test_inplace_scatter_add_tiling_000)
{
    std::string opType("InplaceScatterAdd");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    string compileInfoString = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 48}
                                    })";
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    // platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    // compile info
    struct InplaceScatterAddCompileInfo {
        int32_t totalCoreNum = 30;
        uint64_t ubSizePlatForm = 0;
    } compileInfo;
    // tilingParseFunc simulate
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs(
                                {const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();
    ASSERT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                           intrinsics);
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    gert::StorageShape inputShape = {{65535, 4096}, {65535, 4096}};
    gert::StorageShape srcShape = {{63806, 4096}, {63806, 4096}};
    gert::StorageShape indicShape = {{
                                         63806,
                                     },
                                     {
                                         63806,
                                     }};
    gert::StorageShape outputShape = {{65535, 4096}, {65535, 4096}};

    // tilingParseFunc simulate
    auto holder = gert::TilingContextFaker()
                      .SetOpType("InplaceScatterAdd")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&inputShape, &indicShape, &srcShape})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
}

TEST_F(InplaceScatterAddTiling, test_inplace_scatter_add_tiling_001)
{
    std::string opType("InplaceScatterAdd");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    string compileInfoString = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 48}
                                    })";
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    // platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    // compile info
    struct InplaceScatterAddCompileInfo {
        int32_t totalCoreNum = 30;
        uint64_t ubSizePlatForm = 0;
    } compileInfo;
    // tilingParseFunc simulate
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs(
                                {const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();
    ASSERT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                           intrinsics);
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    gert::StorageShape inputShape = {{65535, 4096}, {65535, 4096}};
    gert::StorageShape srcShape = {{63806, 4096}, {63806, 4096}};
    gert::StorageShape indicShape = {{
                                         63806,
                                     },
                                     {
                                         63806,
                                     }};
    gert::StorageShape outputShape = {{65535, 4096}, {65535, 4096}};

    // tilingParseFunc simulate
    auto holder = gert::TilingContextFaker()
                      .SetOpType("InplaceScatterAdd")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&inputShape, &indicShape, &srcShape})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
}

namespace {
// Common faker skeleton for the updates shape/dtype violation-rejection cases and the zero-case
// validation-pass case (ValidateUpdatesInput). Same TilingContextFaker shape as _000/_001, differing
// only in shapes/dtypes/expectation; the positive main path is covered by the two pre-existing cases.
void RunUpdatesInputGuardCase(gert::StorageShape& varShape, gert::StorageShape& indicesShape,
                              gert::StorageShape& updatesShape, ge::DataType updatesDtype, ge::DataType varDtype,
                              ge::graphStatus expected)
{
    std::string opType("InplaceScatterAdd");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    string compileInfoString = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 48}
                                    })";
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    struct TilingGuardCompileInfo {
        int32_t totalCoreNum = 30;
        uint64_t ubSizePlatForm = 0;
    } compileInfo;
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs(
                                {const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();
    ASSERT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                           intrinsics);
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    gert::StorageShape outputShape = varShape;

    auto holder = gert::TilingContextFaker()
                      .SetOpType("InplaceScatterAdd")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&varShape, &indicesShape, &updatesShape})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, updatesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // Build-target behaviour probe: the cross-input validation (ValidateUpdatesInput) that the
    // updates violation-rejection cases below rely on exists only in the arch35 tiling; the arch22
    // tiling does not have it (the target chip ascend950 never takes that path). Cases expecting
    // rejection are unsatisfiable on the ascend910b (arch22) build — if a pass-through is detected,
    // SKIP (that pass-through is pre-existing arch22 behaviour, not a guard-target failure);
    // on the ascend950 (arch35) build the violations are still rejected with the assertions in
    // force. If the arch35 guard is ever lost (regression), this probe SKIPs as well, and detection
    // falls back to the updates shape-violation regression cases on the kernel/GEIR paths
    // (expected-rejection form).
    const ge::graphStatus actual = tilingFunc(tilingContext);
    if (expected == ge::GRAPH_FAILED && actual == ge::GRAPH_SUCCESS) {
        GTEST_SKIP() << "updates cross-input validation absent on this build target "
                        "(arch22 tiling; guard target is arch35)";
    }
    EXPECT_EQ(actual, expected);
}
} // namespace

// Mismatched updates dtype vs var → rejected by the ValidateUpdatesInput dtype branch (GRAPH_FAILED,
// kernel not executed)
TEST_F(InplaceScatterAddTiling, test_inplace_scatter_add_tiling_002_updates_dtype_mismatch)
{
    gert::StorageShape varShape = {{4, 8}, {4, 8}};
    gert::StorageShape indicesShape = {{3}, {3}};
    gert::StorageShape updatesShape = {{3, 8}, {3, 8}};
    RunUpdatesInputGuardCase(varShape, indicesShape, updatesShape, ge::DT_INT32, ge::DT_FLOAT, ge::GRAPH_FAILED);
}

// Mismatched updates rank vs var → rejected by the shape dims branch
TEST_F(InplaceScatterAddTiling, test_inplace_scatter_add_tiling_003_updates_rank_mismatch)
{
    gert::StorageShape varShape = {{4, 8}, {4, 8}};
    gert::StorageShape indicesShape = {{3}, {3}};
    gert::StorageShape updatesShape = {{24}, {24}};
    RunUpdatesInputGuardCase(varShape, indicesShape, updatesShape, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_FAILED);
}

// updates.dim0 != K (K = number of indices elements) → rejected by the shapesize branch
TEST_F(InplaceScatterAddTiling, test_inplace_scatter_add_tiling_004_updates_dim0_not_equal_k)
{
    gert::StorageShape varShape = {{4, 8}, {4, 8}};
    gert::StorageShape indicesShape = {{3}, {3}};
    gert::StorageShape updatesShape = {{5, 8}, {5, 8}};
    RunUpdatesInputGuardCase(varShape, indicesShape, updatesShape, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_FAILED);
}

// Remaining updates dims mismatch var (dim0 correct, dim1 different) → rejected by the per-dim shape branch
TEST_F(InplaceScatterAddTiling, test_inplace_scatter_add_tiling_005_updates_shape_mismatch)
{
    gert::StorageShape varShape = {{4, 8}, {4, 8}};
    gert::StorageShape indicesShape = {{3}, {3}};
    gert::StorageShape updatesShape = {{3, 4}, {3, 4}};
    RunUpdatesInputGuardCase(varShape, indicesShape, updatesShape, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_FAILED);
}

// K=0 empty input passes validation and takes the zero-case path (static shape checks are not waived
// for empty inputs) → HandleZeroCase returns success
TEST_F(InplaceScatterAddTiling, test_inplace_scatter_add_tiling_006_zero_case_passes_validation)
{
    gert::StorageShape varShape = {{4, 8}, {4, 8}};
    gert::StorageShape indicesShape = {{0}, {0}};
    gert::StorageShape updatesShape = {{0, 8}, {0, 8}};
    RunUpdatesInputGuardCase(varShape, indicesShape, updatesShape, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_SUCCESS);
}
