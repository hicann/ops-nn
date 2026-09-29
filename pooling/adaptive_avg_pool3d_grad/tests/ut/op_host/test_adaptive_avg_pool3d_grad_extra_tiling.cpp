/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_adaptive_avg_pool3d_grad_extra_tiling.cpp
 * \brief AdaptiveAvgPool3dGrad arch35 tiling 覆盖增强: big/small kernel 的
 *        SplitUnalignDHW/DynamicAdjustmentDWH/SearchBestTiling 穷举与兜底路径、
 *        BaseV35 默认实现与错误分支
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "../../../op_host/adaptive_avg_pool3d_grad_tiling_arch35.h"

using namespace std;
using namespace ge;

namespace {
struct AdaptiveAvgPool3dGradExtraCompileInfo {
    int32_t totalCoreNum = 0;
    uint32_t sysWorkspaceSize = 0;
    uint64_t ubSizePlatForm = 0;
};
} // namespace

class AdaptiveAvgPool3dGradExtraTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AdaptiveAvgPool3dGradExtraTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AdaptiveAvgPool3dGradExtraTiling TearDown" << std::endl; }
};

/*
 * 通用执行器: 构造 950 平台 TilingContext 并执行 tiling_func
 * withSocInfo=false 时 tiling 上下文不设置 SoCInfo(核数为 0), 触发 GetPlatformInfo 错误分支
 */
static void ExecuteAdaptiveAvgPool3dGradExtraCase(gert::StorageShape& yGradShape, gert::StorageShape& xShape,
                                                  gert::StorageShape& xGradShape, ge::DataType dtype,
                                                  std::string dataFormat, bool withSocInfo,
                                                  ge::graphStatus expectedStatus)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    AdaptiveAvgPool3dGradExtraCompileInfo compile_info;

    std::string op_type("AdaptiveAvgPool3dGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    auto tiling_data = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(tiling_data, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&yGradShape, &xShape})
                      .OutputShapes({&xGradShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtype, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(1, dtype, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, dtype, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeAttrs({{"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(dataFormat)}})
                      .TilingData(tiling_data.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    if (withSocInfo) {
        tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    }
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

    ASSERT_EQ(tiling_func(tiling_context), expectedStatus);
    if (expectedStatus == ge::GRAPH_SUCCESS) {
        // 至少应设置非零 block dim
        ASSERT_GE(tiling_context->GetBlockDim(), 1U);
    }
}

// ===================== BigKernel: TrySplitNC 失败 -> SplitUnalignDHW/DynamicAdjustmentDWH =====================

// kernelD=kernelH=kernelW=64(kprod=262144>=256 且 kernelW=64>32) 走 big_kernel;
// 全量 inner(512^3) 下 UB 严重超限 -> TrySplitNC 两次尝试均失败 -> SplitUnalignDHW 循环
// 经 DynamicAdjustmentDWH 逐轴收缩 D/H/W 直至满足或退化为 1
TEST_F(AdaptiveAvgPool3dGradExtraTiling, adaptive_avg_pool3d_grad_big_kernel_split_unalign_dhw)
{
    gert::StorageShape yGradShape = {{1, 1, 8, 8, 8}, {1, 1, 8, 8, 8}};
    gert::StorageShape xShape = {{1, 1, 512, 512, 512}, {1, 1, 512, 512, 512}};
    gert::StorageShape xGradShape = {{1, 1, 512, 512, 512}, {1, 1, 512, 512, 512}};
    ExecuteAdaptiveAvgPool3dGradExtraCase(yGradShape, xShape, xGradShape, ge::DT_FLOAT, "NCDHW", true,
                                          ge::GRAPH_SUCCESS);
}

// ===================== SmallKernel: TrySplitNC 失败 -> 穷举搜索(ExhaustiveSearchBestTiling) =====================

// kernel=4^3, big 因 kernelW=4<=32 拒绝, small 接管; NC=32<64 使两次 TrySplitNC 的
// IsMeetTargetCoreNum 均不满足 -> 进入穷举搜索(searchDhwSize=32^3<=200000),
// 在 dInner=1/hInner=1/wInner=16/highAxisInner=64 处找到满足 UB 与核数约束的解,
// 覆盖 EvalTilingCandidate/TryRecordBetterTiling 与 found 分支
TEST_F(AdaptiveAvgPool3dGradExtraTiling, adaptive_avg_pool3d_grad_small_kernel_exhaustive_search)
{
    gert::StorageShape yGradShape = {{1, 32, 8, 8, 8}, {1, 32, 8, 8, 8}};
    gert::StorageShape xShape = {{1, 32, 32, 32, 32}, {1, 32, 32, 32, 32}};
    gert::StorageShape xGradShape = {{1, 32, 32, 32, 32}, {1, 32, 32, 32, 32}};
    ExecuteAdaptiveAvgPool3dGradExtraCase(yGradShape, xShape, xGradShape, ge::DT_FLOAT, "NCDHW", true,
                                          ge::GRAPH_SUCCESS);
}

// ===================== SmallKernel: searchDhwSize 超限 -> ApplyCoarseFallback =====================

// searchDhwSize=128^3>200000 跳过穷举 -> ApplyCoarseFallback 兜底(含 SplitUnalignDHW/
// DynamicAdjustmentDWH/ShrinkInnerStrict 收缩链路)
TEST_F(AdaptiveAvgPool3dGradExtraTiling, adaptive_avg_pool3d_grad_small_kernel_coarse_fallback)
{
    gert::StorageShape yGradShape = {{1, 32, 16, 16, 16}, {1, 32, 16, 16, 16}};
    gert::StorageShape xShape = {{1, 32, 128, 128, 128}, {1, 32, 128, 128, 128}};
    gert::StorageShape xGradShape = {{1, 32, 128, 128, 128}, {1, 32, 128, 128, 128}};
    ExecuteAdaptiveAvgPool3dGradExtraCase(yGradShape, xShape, xGradShape, ge::DT_FLOAT, "NCDHW", true,
                                          ge::GRAPH_SUCCESS);
}

// ===================== NDHWC 格式(所有 NCDHW 模板拒绝后落到 simt) =====================

// data_format=NDHWC: ksize_one/big/small 均 IsCapable=false -> simt(TPL_SIMT_KERNEL) 接管,
// 覆盖 SetInputParams 的 NDHWC 维度映射分支
TEST_F(AdaptiveAvgPool3dGradExtraTiling, adaptive_avg_pool3d_grad_ndhwc_simt)
{
    gert::StorageShape yGradShape = {{1, 4, 4, 4, 4}, {1, 4, 4, 4, 4}};
    gert::StorageShape xShape = {{1, 8, 8, 8, 4}, {1, 8, 8, 8, 4}};
    gert::StorageShape xGradShape = {{1, 8, 8, 8, 4}, {1, 8, 8, 8, 4}};
    ExecuteAdaptiveAvgPool3dGradExtraCase(yGradShape, xShape, xGradShape, ge::DT_FLOAT, "NDHWC", true,
                                          ge::GRAPH_SUCCESS);
}

// ===================== BaseV35 错误分支 =====================

// grad 维度含 0 -> CheckInputShape 失败
TEST_F(AdaptiveAvgPool3dGradExtraTiling, adaptive_avg_pool3d_grad_grad_dim_zero)
{
    gert::StorageShape yGradShape = {{1, 4, 0, 4, 4}, {1, 4, 0, 4, 4}};
    gert::StorageShape xShape = {{1, 4, 4, 4, 4}, {1, 4, 4, 4, 4}};
    gert::StorageShape xGradShape = {{1, 4, 4, 4, 4}, {1, 4, 4, 4, 4}};
    ExecuteAdaptiveAvgPool3dGradExtraCase(yGradShape, xShape, xGradShape, ge::DT_FLOAT, "NCDHW", true,
                                          ge::GRAPH_FAILED);
}

// 平台信息缺少 SoCInfo -> GetCoreNumAiv 返回 0 -> GetPlatformInfo coreNum==0 错误分支
TEST_F(AdaptiveAvgPool3dGradExtraTiling, adaptive_avg_pool3d_grad_core_num_zero)
{
    gert::StorageShape yGradShape = {{1, 32, 8, 8, 8}, {1, 32, 8, 8, 8}};
    gert::StorageShape xShape = {{1, 32, 32, 32, 32}, {1, 32, 32, 32, 32}};
    gert::StorageShape xGradShape = {{1, 32, 32, 32, 32}, {1, 32, 32, 32, 32}};
    ExecuteAdaptiveAvgPool3dGradExtraCase(yGradShape, xShape, xGradShape, ge::DT_FLOAT, "NCDHW", false,
                                          ge::GRAPH_FAILED);
}

// ===================== BaseV35 默认虚函数实现(直接实例化) =====================

// AdaptiveAvgPool3dGradTilingBaseV35 的 IsCapable(false)/DoOpTiling/DoLibApiTiling/
// GetWorkspaceSize/PostTiling/GetTilingKey(0) 默认实现: 所有注册子类均覆写,
// 注册表路径不可达(DoTiling 因 IsCapable=false 返回 PARAM_INVALID), 直接实例化逐个验证
TEST_F(AdaptiveAvgPool3dGradExtraTiling, adaptive_avg_pool3d_grad_base_v35_default_impl_direct)
{
    gert::StorageShape yGradShape = {{1, 32, 8, 8, 8}, {1, 32, 8, 8, 8}};
    gert::StorageShape xShape = {{1, 32, 32, 32, 32}, {1, 32, 32, 32, 32}};
    gert::StorageShape xGradShape = {{1, 32, 32, 32, 32}, {1, 32, 32, 32, 32}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    AdaptiveAvgPool3dGradExtraCompileInfo compile_info;
    auto tiling_data = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(tiling_data, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("AdaptiveAvgPool3dGrad")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&yGradShape, &xShape})
                      .OutputShapes({&xGradShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeAttrs({{"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")}})
                      .TilingData(tiling_data.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

    optiling::AdaptiveAvgPool3dGradTilingBaseV35 baseTiling(tiling_context);
    // IsCapable 默认 false -> DoTiling 返回 GRAPH_PARAM_INVALID
    ASSERT_EQ(baseTiling.DoTiling(), ge::GRAPH_PARAM_INVALID);
    // 默认实现逐个调用
    ASSERT_EQ(baseTiling.DoOpTiling(), ge::GRAPH_SUCCESS);
    ASSERT_EQ(baseTiling.DoLibApiTiling(), ge::GRAPH_SUCCESS);
    ASSERT_EQ(baseTiling.GetWorkspaceSize(), ge::GRAPH_SUCCESS);
    ASSERT_EQ(baseTiling.PostTiling(), ge::GRAPH_SUCCESS);
    ASSERT_EQ(baseTiling.GetTilingKey(), 0U);
}
