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
 * \file test_adaptive_avg_pool3d_extra_tiling.cpp
 * \brief AdaptiveAvgPool3d arch35 tiling 覆盖增强: para/simt/big_kernel 模板选择与
 *        AdaptivePool3dBaseTiling 未覆盖分支
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"
#include "pooling/adaptive_pool3d_common/op_host/arch35/adaptive_pool3d_tiling.h"

using namespace std;
using namespace ge;

namespace optiling {
struct AdaptiveAvgPool3dCompileInfo {
    int32_t totalCoreNum = 0;
    uint32_t sysWorkspaceSize = 0;
    uint64_t ubSizePlatForm = 0;
};
} // namespace optiling

class AdaptiveAvgPool3dExtraTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AdaptiveAvgPool3dExtraTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AdaptiveAvgPool3dExtraTiling TearDown" << std::endl; }
};

/*
 * 通用执行器: 构造 950 平台的 TilingContext 并执行 tiling_func
 * expectedStatus: 期望返回值; expectedKey/expectedBlockDim: 期望 tiling key 与 block dim
 */
static void ExecuteAdaptiveAvgPool3dExtraCase(gert::StorageShape& xShape, gert::StorageShape& yShape,
                                              std::vector<int64_t> outputSize, std::string dataFormat,
                                              ge::DataType dataType, ge::DataType outputDataType,
                                              std::map<std::string, std::string> npuArchInfos,
                                              ge::graphStatus expectedStatus, uint64_t expectedKey = 0,
                                              int64_t expectedBlockDim = -1)
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

    optiling::AdaptiveAvgPool3dCompileInfo compile_info;

    std::string op_type("AdaptiveAvgPool3d");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, outputDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(outputSize)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(dataFormat)}})
                      .TilingData(param.get())
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
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", npuArchInfos);

    ASSERT_EQ(tiling_func(tiling_context), expectedStatus);
    if (expectedStatus == ge::GRAPH_SUCCESS) {
        ASSERT_EQ(tiling_context->GetTilingKey(), expectedKey);
    }
    if (expectedBlockDim >= 0) {
        ASSERT_EQ(static_cast<int64_t>(tiling_context->GetBlockDim()), expectedBlockDim);
    }
}

// ===================== Para 模板正向选中 (DoOpTiling/SetTilingData/PostTiling/GetTilingKey 链路) =====================

// para fp16: NC=64>=vfLen/2, kD=kH=3/kW=16(kprod=144<350), wOut=1 退化向量化分支
// (isWOutDegenerateVectorized: kprod>=vfLen 且 wIn>=alignNum), key=4(int32), blockDim=useCoreNum=4
TEST_F(AdaptiveAvgPool3dExtraTiling, adaptive_avg_pool3d_para_fp16_w_out_degenerate)
{
    gert::StorageShape xShape = {{1, 64, 6, 6, 16}, {1, 64, 6, 6, 16}};
    gert::StorageShape yShape = {{1, 64, 2, 2, 1}, {1, 64, 2, 2, 1}};
    std::map<std::string, std::string> npuArch = {{"NpuArch", "3510"}};
    ExecuteAdaptiveAvgPool3dExtraCase(xShape, yShape, {2, 2, 1}, "NCDHW", ge::DT_FLOAT16, ge::DT_FLOAT16, npuArch,
                                      ge::GRAPH_SUCCESS, 4, 4);
}

// para fp32: NC=32>=vfLen/2(32), kD=kH=3/kW=8(kprod=72>=vfLen=64), wOut=1 退化向量化分支,
// 且 SearchOuterSingle 的 do-while 循环继续/退出路径均被走到, key=4, blockDim=4
TEST_F(AdaptiveAvgPool3dExtraTiling, adaptive_avg_pool3d_para_fp32_w_out_degenerate)
{
    gert::StorageShape xShape = {{1, 32, 6, 6, 8}, {1, 32, 6, 6, 8}};
    gert::StorageShape yShape = {{1, 32, 2, 2, 1}, {1, 32, 2, 2, 1}};
    std::map<std::string, std::string> npuArch = {{"NpuArch", "3510"}};
    ExecuteAdaptiveAvgPool3dExtraCase(xShape, yShape, {2, 2, 1}, "NCDHW", ge::DT_FLOAT, ge::DT_FLOAT, npuArch,
                                      ge::GRAPH_SUCCESS, 4, 4);
}

// para fp16 int64 索引: dIn*dOut=2^43>INT32_MAX -> idxTypeMode=INT64, key=16,
// totalOuter=4194304 占满 64 核(useCoreNum==coreNum, SearchOuter 立即返回), blockDim=64;
// dOut=2097152>10000 覆盖 CalKernelSizeOneDimMax 的快捷分支
TEST_F(AdaptiveAvgPool3dExtraTiling, adaptive_avg_pool3d_para_fp16_int64_index)
{
    gert::StorageShape xShape = {{1, 64, 4194304, 6, 16}, {1, 64, 4194304, 6, 16}};
    gert::StorageShape yShape = {{1, 64, 2097152, 2, 1}, {1, 64, 2097152, 2, 1}};
    std::map<std::string, std::string> npuArch = {{"NpuArch", "3510"}};
    ExecuteAdaptiveAvgPool3dExtraCase(xShape, yShape, {2097152, 2, 1}, "NCDHW", ge::DT_FLOAT16, ge::DT_FLOAT16, npuArch,
                                      ge::GRAPH_SUCCESS, 16, 64);
}

// ===================== BigKernel 模板正向选中 =====================

// big_kernel: kD=kH=kW=64(kprod=262144>=350), para 因 kprod>=350 拒绝, gather 因 kprod>6 拒绝;
// totalIdx=1<coreNum -> blockFactor=0 分支, coreNums=totalIdx=1, blockDim=1, key=1(TPL_MODE_1)
TEST_F(AdaptiveAvgPool3dExtraTiling, adaptive_avg_pool3d_big_kernel_single_output)
{
    gert::StorageShape xShape = {{1, 1, 64, 64, 64}, {1, 1, 64, 64, 64}};
    gert::StorageShape yShape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    std::map<std::string, std::string> npuArch = {{"NpuArch", "3510"}};
    ExecuteAdaptiveAvgPool3dExtraCase(xShape, yShape, {1, 1, 1}, "NCDHW", ge::DT_FLOAT16, ge::DT_FLOAT16, npuArch,
                                      ge::GRAPH_SUCCESS, 1, 1);
}

// big_kernel 多核: kD=kH=kW=8(kprod=512>=350), totalIdx=384>=64 -> blockFactor=6>0 分支,
// coreNums=64, blockDim=64, key=1
TEST_F(AdaptiveAvgPool3dExtraTiling, adaptive_avg_pool3d_big_kernel_multi_core)
{
    gert::StorageShape xShape = {{2, 3, 32, 32, 32}, {2, 3, 32, 32, 32}};
    gert::StorageShape yShape = {{2, 3, 4, 4, 4}, {2, 3, 4, 4, 4}};
    std::map<std::string, std::string> npuArch = {{"NpuArch", "3510"}};
    ExecuteAdaptiveAvgPool3dExtraCase(xShape, yShape, {4, 4, 4}, "NCDHW", ge::DT_FLOAT16, ge::DT_FLOAT16, npuArch,
                                      ge::GRAPH_SUCCESS, 1, 64);
}

// big_kernel 输出 dtype 非法: DoOpTiling->CheckOutputDtypeInfo 失败 -> GRAPH_FAILED
TEST_F(AdaptiveAvgPool3dExtraTiling, adaptive_avg_pool3d_big_kernel_invalid_output_dtype)
{
    gert::StorageShape xShape = {{1, 1, 64, 64, 64}, {1, 1, 64, 64, 64}};
    gert::StorageShape yShape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    std::map<std::string, std::string> npuArch = {{"NpuArch", "3510"}};
    ExecuteAdaptiveAvgPool3dExtraCase(xShape, yShape, {1, 1, 1}, "NCDHW", ge::DT_FLOAT16, ge::DT_INT32, npuArch,
                                      ge::GRAPH_FAILED);
}

// ===================== Base GetShapeAttrsInfo 未覆盖分支 =====================

// NpuArch=5102(regbase 但非 3510): base GetShapeAttrsInfo 返回 PARAM_INVALID,
// 4 个模板均不可用 -> 注册表返回 GRAPH_FAILED
TEST_F(AdaptiveAvgPool3dExtraTiling, adaptive_avg_pool3d_arch_5102_not_supported)
{
    gert::StorageShape xShape = {{1, 64, 6, 6, 16}, {1, 64, 6, 6, 16}};
    gert::StorageShape yShape = {{1, 64, 2, 2, 1}, {1, 64, 2, 2, 1}};
    std::map<std::string, std::string> npuArch = {{"NpuArch", "5102"}};
    ExecuteAdaptiveAvgPool3dExtraCase(xShape, yShape, {2, 2, 1}, "NCDHW", ge::DT_FLOAT16, ge::DT_FLOAT16, npuArch,
                                      ge::GRAPH_FAILED);
}

// 输入 dtype 不支持(DT_INT32) -> base GetShapeAttrsInfo 失败
TEST_F(AdaptiveAvgPool3dExtraTiling, adaptive_avg_pool3d_invalid_x_dtype)
{
    gert::StorageShape xShape = {{1, 64, 6, 6, 16}, {1, 64, 6, 6, 16}};
    gert::StorageShape yShape = {{1, 64, 2, 2, 1}, {1, 64, 2, 2, 1}};
    std::map<std::string, std::string> npuArch = {{"NpuArch", "3510"}};
    ExecuteAdaptiveAvgPool3dExtraCase(xShape, yShape, {2, 2, 1}, "NCDHW", ge::DT_INT32, ge::DT_INT32, npuArch,
                                      ge::GRAPH_FAILED);
}

// 输入维度数非法(3 维) -> base GetShapeAttrsInfo 失败
TEST_F(AdaptiveAvgPool3dExtraTiling, adaptive_avg_pool3d_invalid_x_dim)
{
    gert::StorageShape xShape = {{64, 6, 6}, {64, 6, 6}};
    gert::StorageShape yShape = {{2, 2, 1}, {2, 2, 1}};
    std::map<std::string, std::string> npuArch = {{"NpuArch", "3510"}};
    ExecuteAdaptiveAvgPool3dExtraCase(xShape, yShape, {2, 2, 1}, "NCDHW", ge::DT_FLOAT16, ge::DT_FLOAT16, npuArch,
                                      ge::GRAPH_FAILED);
}

// output_size 长度非法(2) -> base GetShapeAttrsInfo 失败(仅支持 0/1/3)
TEST_F(AdaptiveAvgPool3dExtraTiling, adaptive_avg_pool3d_invalid_output_size_len)
{
    gert::StorageShape xShape = {{1, 64, 6, 6, 16}, {1, 64, 6, 6, 16}};
    gert::StorageShape yShape = {{1, 64, 2, 2, 1}, {1, 64, 2, 2, 1}};
    std::map<std::string, std::string> npuArch = {{"NpuArch", "3510"}};
    ExecuteAdaptiveAvgPool3dExtraCase(xShape, yShape, {2, 2}, "NCDHW", ge::DT_FLOAT16, ge::DT_FLOAT16, npuArch,
                                      ge::GRAPH_FAILED);
}

// output_size 含非正值 -> base GetShapeAttrsInfo 失败
TEST_F(AdaptiveAvgPool3dExtraTiling, adaptive_avg_pool3d_invalid_output_size_value)
{
    gert::StorageShape xShape = {{1, 64, 6, 6, 16}, {1, 64, 6, 6, 16}};
    gert::StorageShape yShape = {{1, 64, 2, 2, 1}, {1, 64, 2, 2, 1}};
    std::map<std::string, std::string> npuArch = {{"NpuArch", "3510"}};
    ExecuteAdaptiveAvgPool3dExtraCase(xShape, yShape, {2, 0, 1}, "NCDHW", ge::DT_FLOAT16, ge::DT_FLOAT16, npuArch,
                                      ge::GRAPH_FAILED);
}

// ===================== output_size 退化形式(广播/空) 走到 simt =====================

// output_size 长度 1: h/w 同值广播, para 因最终 UB 使用率不足拒绝 -> simt,
// key=70(NCDHW int32), blockDim=ceil(512/min(512,1024))=1
TEST_F(AdaptiveAvgPool3dExtraTiling, adaptive_avg_pool3d_output_size_one_dim_broadcast)
{
    gert::StorageShape xShape = {{1, 64, 6, 6, 16}, {1, 64, 6, 6, 16}};
    gert::StorageShape yShape = {{1, 64, 2, 2, 2}, {1, 64, 2, 2, 2}};
    std::map<std::string, std::string> npuArch = {{"NpuArch", "3510"}};
    ExecuteAdaptiveAvgPool3dExtraCase(xShape, yShape, {2}, "NCDHW", ge::DT_FLOAT16, ge::DT_FLOAT16, npuArch,
                                      ge::GRAPH_SUCCESS, 70, 1);
}

// output_size 为空: 输出取输入后 3 维, 恒等池化 kernel=1, para 拒绝 -> simt,
// key=70, blockDim=ceil(36864/1024)=36
TEST_F(AdaptiveAvgPool3dExtraTiling, adaptive_avg_pool3d_output_size_none_dim_same_as_input)
{
    gert::StorageShape xShape = {{1, 64, 6, 6, 16}, {1, 64, 6, 6, 16}};
    gert::StorageShape yShape = {{1, 64, 6, 6, 16}, {1, 64, 6, 6, 16}};
    std::map<std::string, std::string> npuArch = {{"NpuArch", "3510"}};
    ExecuteAdaptiveAvgPool3dExtraCase(xShape, yShape, {}, "NCDHW", ge::DT_FLOAT16, ge::DT_FLOAT16, npuArch,
                                      ge::GRAPH_SUCCESS, 70, 36);
}

// ===================== Base 类默认虚函数实现(直接实例化) =====================

// AdaptivePool3dBaseTiling 的 IsCapable(true)/DoOpTiling/DoLibApiTiling/PostTiling/GetTilingKey(0)/
// DumpTilingInfo 默认实现: 所有注册子类均覆写, 注册表路径不可达, 直接实例化验证
TEST_F(AdaptiveAvgPool3dExtraTiling, adaptive_avg_pool3d_base_default_impl_direct)
{
    gert::StorageShape xShape = {{1, 64, 6, 6, 16}, {1, 64, 6, 6, 16}};
    gert::StorageShape yShape = {{1, 64, 2, 2, 1}, {1, 64, 2, 2, 1}};

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
    std::map<std::string, std::string> npu_arch_infos = {{"NpuArch", "3510"}};
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::AdaptiveAvgPool3dCompileInfo compile_info;
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AdaptiveAvgPool3d")
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 1})},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")}})
                      .TilingData(param.get())
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
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", npu_arch_infos);

    optiling::AdaptivePool3dBaseTiling baseTiling(tiling_context);
    ASSERT_EQ(baseTiling.DoTiling(), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tiling_context->GetTilingKey(), 0U);
}
