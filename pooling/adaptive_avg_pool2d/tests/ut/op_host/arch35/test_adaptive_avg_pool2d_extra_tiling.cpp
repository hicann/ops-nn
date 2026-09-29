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
 * \file test_adaptive_avg_pool2d_extra_tiling.cpp
 * \brief Tiling补充测试 - AdaptiveAvgPool2dTiling950ExtraTest
 * 覆盖率增强: split_w/split_h/base/simt 模板 tiling 的未覆盖分支
 */

#include <gtest/gtest.h>
#include <iostream>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"

#include "../../../../op_host/arch35/adaptive_avg_pool2d_tiling.h"
#include "../../../../op_host/arch35/adaptive_avg_pool2d_base_tiling.h"

using namespace ut_util;
using namespace std;
using namespace ge;

// adaptive_avg_pool2d_simt_tiling.h 内部使用 "../op_kernel/..." 相对路径引用 struct 头,
// 该路径仅在 op tiling 编译目标(带 -I op_host)下可解析, UT 测试目标无法直接包含该头文件。
// SplitW/SplitH/Simt 的 SetTilingData 在注册表路径下被编译器内联, 非内联函数体从未被调用,
// 这里通过 tiling 注册表工厂构造真实模板对象后按符号名直接调用(与类定义一致的非虚成员函数)。
extern "C" void _ZN8optiling29AdaptiveAvgPool2dSplitWTiling13SetTilingDataEv(void* self);
extern "C" void _ZN8optiling29AdaptiveAvgPool2dSplitHTiling13SetTilingDataEv(void* self);
extern "C" void _ZN8optiling27AdaptiveAvgPool2DTilingSimt13SetTilingDataEv(void* self);
extern "C" ge::graphStatus _ZN8optiling27AdaptiveAvgPool2DTilingSimt10PostTilingEv(void* self);

static void SetAscend950GlobalPlatformInfoExtra()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;

    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";

    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
}

class AdaptiveAvgPool2dTiling950ExtraTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AdaptiveAvgPool2dTiling950ExtraTest SetUp" << std::endl;
        SetAscend950GlobalPlatformInfoExtra();
    }

    static void TearDownTestCase() { std::cout << "AdaptiveAvgPool2dTiling950ExtraTest TearDown" << std::endl; }
};

static void ExecuteExtraTiling(gert::StorageShape xShape, gert::StorageShape yShape, std::vector<int64_t> outputSize,
                               ge::DataType dtype, uint64_t expectTilingKey,
                               const std::function<void(gert::TilingContext*)>& postCheck = nullptr,
                               const std::string& npuArch = "3510")
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

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", npuArch}};

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::AdaptiveAvgPool2dCompileInfo compile_info;

    std::string op_type("AdaptiveAvgPool2d");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(1, 1)
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

    auto tiling_data = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(tiling_data, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(0, dtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(outputSize)}})
                      .TilingData(tiling_data.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

    auto ret = tiling_func(tiling_context);

    ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
    ASSERT_EQ(tiling_context->GetTilingKey(), expectTilingKey);
    if (postCheck != nullptr) {
        postCheck(tiling_context);
    }
}

static void ExecuteExtraTilingExpectFail(gert::StorageShape xShape, gert::StorageShape yShape,
                                         std::vector<int64_t> outputSize, ge::DataType dtype,
                                         ge::graphStatus expectResult, const std::string& npuArch = "3510")
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

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", npuArch}};

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::AdaptiveAvgPool2dCompileInfo compile_info;

    std::string op_type("AdaptiveAvgPool2d");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(1, 1)
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

    auto tiling_data = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(tiling_data, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(0, dtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(outputSize)}})
                      .TilingData(tiling_data.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

    auto ret = tiling_func(tiling_context);
    ASSERT_EQ(ret, expectResult);
}

// 供直接构造 tiling 类使用的上下文: platformInfo/compileInfo 传 nullptr 表示不注入
struct ExtraTilingContext {
    gert::KernelRunContextHolder holder;
    std::unique_ptr<uint8_t[]> tilingData;
    std::unique_ptr<uint8_t[]> workspace;
    gert::TilingContext* context = nullptr;
};

static ExtraTilingContext BuildDirectTilingContext(gert::StorageShape* xShape, gert::StorageShape* yShape,
                                                   const std::vector<int64_t>& outputSize, ge::DataType dtype,
                                                   void* compileInfo, fe::PlatFormInfos* platformInfo)
{
    ExtraTilingContext result;
    result.tilingData = gert::TilingData::CreateCap(4096);
    result.workspace = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(result.workspace.get());
    result.holder = gert::TilingContextFaker()
                        .SetOpType("AdaptiveAvgPool2d")
                        .NodeIoNum(1, 1)
                        .IrInstanceNum({1})
                        .InputShapes({xShape})
                        .OutputShapes({yShape})
                        .CompileInfo(compileInfo)
                        .PlatformInfo(reinterpret_cast<char*>(platformInfo))
                        .NodeInputTd(0, dtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                        .NodeOutputTd(0, dtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                        .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(outputSize)}})
                        .TilingData(result.tilingData.get())
                        .Workspace(wsSize)
                        .Build();
    result.context = result.holder.GetContext<gert::TilingContext>();
    return result;
}

// ===================== SplitW 正向选中 =====================

// SplitW(优先级2) fp32 正向选中: hIn==hOut(H恒等, 无H上/下采样) + W大kernel下采样
// kWMax = CeilDiv(128,4) 窗口 = 32 >= SPLIT_W_KERNEL_W_LINE(32); wOut=4 <= wIn; wOut > 1; nc=64 >= 32
// UpsampleH(无H上采样)与SplitH(无W上采样/H下采样)均拒绝 -> SplitW 接管, fp32 ncFactor=64 -> key=4
// 覆盖: split_w 的 DoOpTiling/CalUbSplitSize/SetTilingData/PrintTilingData/GetTilingKey/PostTiling
TEST_F(AdaptiveAvgPool2dTiling950ExtraTest, test_split_w_select_fp32_h_identity_w_down)
{
    gert::StorageShape x_shape = {{1, 64, 4, 128}, {1, 64, 4, 128}};
    gert::StorageShape y_shape = {{1, 64, 4, 4}, {1, 64, 4, 4}};
    ExecuteExtraTiling(x_shape, y_shape, {4, 4}, ge::DT_FLOAT, 4, [](gert::TilingContext* ctx) {
        auto* td = ctx->GetTilingData<AdaptiveAvgPool2dOp::AdaptivePool2dSplitWTilingData>();
        ASSERT_NE(td, nullptr);
        EXPECT_EQ(td->hIn, 4);
        EXPECT_EQ(td->wIn, 128);
        EXPECT_EQ(td->hOut, 4);
        EXPECT_EQ(td->wOut, 4);
        EXPECT_EQ(td->useCoreNum, 4);
        EXPECT_EQ(td->blockFactor, 1);
        EXPECT_EQ(td->ncFactor, 64);
        EXPECT_EQ(td->hoFactor, 1);
        EXPECT_EQ(td->hiFactor, 1);
        EXPECT_EQ(td->ncOuter, 1);
        EXPECT_EQ(td->hoOuter, 4);
        EXPECT_EQ(td->ncTail, 64);
        EXPECT_EQ(td->hoTail, 1);
        EXPECT_EQ(td->inputQueSize, 32768);
        EXPECT_EQ(td->resQue1Size, 4096);
        EXPECT_EQ(td->resQue2Size, 0);
    });
}

// SplitW fp16 正向选中: vfLen=128, kH*kW=1*32=32 不大于 TILING_LARGE_KERNEL_AREA(32)
// -> TryHalfNcFactor 不触发, ncFactor 保持 128 != VRegSize/sizeof(float)=64
// -> TPL_NC_FACTOR_128 -> key = TPL_SPLIT_W_KERNEL(4) | 1<<6 = 68
TEST_F(AdaptiveAvgPool2dTiling950ExtraTest, test_split_w_select_fp16_nc_factor_128)
{
    gert::StorageShape x_shape = {{1, 128, 4, 128}, {1, 128, 4, 128}};
    gert::StorageShape y_shape = {{1, 128, 4, 4}, {1, 128, 4, 4}};
    ExecuteExtraTiling(x_shape, y_shape, {4, 4}, ge::DT_FLOAT16, 68, [](gert::TilingContext* ctx) {
        auto* td = ctx->GetTilingData<AdaptiveAvgPool2dOp::AdaptivePool2dSplitWTilingData>();
        ASSERT_NE(td, nullptr);
        EXPECT_EQ(td->useCoreNum, 4);
        EXPECT_EQ(td->ncFactor, 128);
        EXPECT_EQ(td->hoFactor, 1);
        EXPECT_EQ(td->hiFactor, 1);
        EXPECT_EQ(td->inputQueSize, 32768);
        EXPECT_EQ(td->resQue1Size, 8192);
        EXPECT_EQ(td->resQue2Size, 8192);
    });
}

// SplitW 接管 SplitH 因 isDmaPerOutputTooHigh 拒绝的 H 大幅下采样场景:
// hIn=1000 > 90*hOut*wOut=360 -> SplitH 拒绝; kWMax=32 满足 SplitW 门限 -> SplitW 接管
// DoOpTiling 内 ShrinkHiFactor 从 kernelHMax=500 线性收缩到 hiFactor=7 (UB 约束)
TEST_F(AdaptiveAvgPool2dTiling950ExtraTest, test_split_w_select_h_down_dma_per_output_too_high)
{
    gert::StorageShape x_shape = {{1, 64, 1000, 64}, {1, 64, 1000, 64}};
    gert::StorageShape y_shape = {{1, 64, 2, 2}, {1, 64, 2, 2}};
    ExecuteExtraTiling(x_shape, y_shape, {2, 2}, ge::DT_FLOAT, 4, [](gert::TilingContext* ctx) {
        auto* td = ctx->GetTilingData<AdaptiveAvgPool2dOp::AdaptivePool2dSplitWTilingData>();
        ASSERT_NE(td, nullptr);
        EXPECT_EQ(td->useCoreNum, 2);
        EXPECT_EQ(td->ncFactor, 64);
        EXPECT_EQ(td->hoFactor, 1);
        EXPECT_EQ(td->hiFactor, 7);
        EXPECT_EQ(td->hoOuter, 2);
        EXPECT_EQ(td->inputQueSize, 114688);
        EXPECT_EQ(td->resQue1Size, 4096);
    });
}

// SplitW INT64 索引模式: hIn=hOut=46341, maxIdx=hIn*hOut=46341^2 >= INT32_MAX
// -> idxTypeMode=TPL_INT64_UINT64 -> key = 4 | 1<<3 = 12
// kernelHMax=(46341+46340)/46341+1=2, ShrinkHiFactor 保持 hiFactor=2;
// BSearchMaxHoFactor 将 hoFactor 收敛到 26(UB 约束), useCoreNum=64
TEST_F(AdaptiveAvgPool2dTiling950ExtraTest, test_split_w_tiling_key_int64_index_mode)
{
    gert::StorageShape x_shape = {{1, 64, 46341, 128}, {1, 64, 46341, 128}};
    gert::StorageShape y_shape = {{1, 64, 46341, 4}, {1, 64, 46341, 4}};
    ExecuteExtraTiling(x_shape, y_shape, {46341, 4}, ge::DT_FLOAT, 12, [](gert::TilingContext* ctx) {
        auto* td = ctx->GetTilingData<AdaptiveAvgPool2dOp::AdaptivePool2dSplitWTilingData>();
        ASSERT_NE(td, nullptr);
        EXPECT_EQ(td->useCoreNum, 64);
        EXPECT_EQ(td->ncFactor, 64);
        EXPECT_EQ(td->hoFactor, 26);
        EXPECT_EQ(td->hiFactor, 2);
        EXPECT_EQ(td->inputQueSize, 65536);
        EXPECT_EQ(td->resQue1Size, 53248);
    });
}

// ===================== SplitH 未覆盖分支 =====================

// SplitH IsCapable 的 isSmallKernelBetter 短路末段(split_h_tiling.cpp:56):
// H低倍率上采样(hOut=7 > hIn=4 且 7 < 2*4) + nc(32) < vfLen(64) -> 计算 kH*kW < 128
// wOut=8 > wIn=2 且 wIn <= 2 -> UpsampleH 的 isWUpTinyInput 拒绝;
// kH*kW=1 < 128 -> isSmallKernelBetter=true -> SplitH 拒绝 -> SmallKernel(优先级3)接管, key=0
TEST_F(AdaptiveAvgPool2dTiling950ExtraTest, test_split_h_reject_small_kernel_better_h_up_low_mag)
{
    gert::StorageShape x_shape = {{1, 32, 4, 2}, {1, 32, 4, 2}};
    gert::StorageShape y_shape = {{1, 32, 7, 8}, {1, 32, 7, 8}};
    ExecuteExtraTiling(x_shape, y_shape, {7, 8}, ge::DT_FLOAT, 0);
}

// SplitH W上采样非整数倍路径(IsMeetUbSize split_h_tiling.cpp:114-118) + DoOpTiling W↑ hiFactor 搜索:
// wOut=16 > wIn=3 且 16%3 != 0 -> tempSumBufSize 换算分支;
// hIn=128, hOut=8: tryHiF>=57 时 ho=1 仍不满足 UB -> continue(split_h_tiling.cpp:219);
// tryHiF=55/56 时 BSearch 将 hoFactor 压到 1 且 useCoreNum(8) 不低于 baseUseCoreNum(8)
// -> 进入 totalBatches 计分循环(split_h_tiling.cpp:228-240), 首个候选 tryHiF=55 胜出
TEST_F(AdaptiveAvgPool2dTiling950ExtraTest, test_split_h_select_w_up_non_multiple_hi_factor_search)
{
    gert::StorageShape x_shape = {{1, 64, 128, 3}, {1, 64, 128, 3}};
    gert::StorageShape y_shape = {{1, 64, 8, 16}, {1, 64, 8, 16}};
    ExecuteExtraTiling(x_shape, y_shape, {8, 16}, ge::DT_FLOAT, 5, [](gert::TilingContext* ctx) {
        auto* td = ctx->GetTilingData<AdaptiveAvgPool2dOp::AdaptivePool2dSplitHTilingData>();
        ASSERT_NE(td, nullptr);
        EXPECT_EQ(td->useCoreNum, 8);
        EXPECT_EQ(td->ncFactor, 64);
        EXPECT_EQ(td->hoFactor, 1);
        EXPECT_EQ(td->hiFactor, 55);
        EXPECT_EQ(td->inputQueSize, 112640);
        EXPECT_EQ(td->resQue1Size, 4096);
    });
}

// SplitH hIn==1 && kWMax==1 的 slim 快路径(split_h_tiling.cpp:250-267):
// hIn=1 且 wOut=8 > wIn=4(W上采样, kWMax=1); wIn=4 < alignNum(8) -> UpsampleH 的 isHin1TinyW 拒绝
// -> SplitH 接管; slimTotal=2048+4096+2*8192+64=22592 <= UB -> 走 slim 命中分支(hoFactor=hOut=4)
TEST_F(AdaptiveAvgPool2dTiling950ExtraTest, test_split_h_select_slim_path_hin1_kw1_fit)
{
    gert::StorageShape x_shape = {{1, 64, 1, 4}, {1, 64, 1, 4}};
    gert::StorageShape y_shape = {{1, 64, 4, 8}, {1, 64, 4, 8}};
    ExecuteExtraTiling(x_shape, y_shape, {4, 8}, ge::DT_FLOAT, 5, [](gert::TilingContext* ctx) {
        auto* td = ctx->GetTilingData<AdaptiveAvgPool2dOp::AdaptivePool2dSplitHTilingData>();
        ASSERT_NE(td, nullptr);
        EXPECT_EQ(td->hIn, 1);
        EXPECT_EQ(td->wIn, 4);
        EXPECT_EQ(td->hOut, 4);
        EXPECT_EQ(td->wOut, 8);
        EXPECT_EQ(td->useCoreNum, 1);
        EXPECT_EQ(td->ncFactor, 64);
        EXPECT_EQ(td->hoFactor, 4);
        EXPECT_EQ(td->hiFactor, 1);
        EXPECT_EQ(td->hoOuter, 1);
        EXPECT_EQ(td->hoTail, 4);
        EXPECT_EQ(td->inputQueSize, 2048);
        EXPECT_EQ(td->resQue1Size, 8192);
        EXPECT_EQ(td->resQue2Size, 8192);
    });
}

// SplitH slim 快路径溢出分支(split_h_tiling.cpp:270):
// hIn=1, hOut=60, wOut=8: slimTotal=2048+4096+2*122880+64=251968 > UB(245760)
// -> 走 else 分支重新 CalUbSplitSize
TEST_F(AdaptiveAvgPool2dTiling950ExtraTest, test_split_h_select_slim_path_hin1_kw1_overflow)
{
    gert::StorageShape x_shape = {{1, 64, 1, 4}, {1, 64, 1, 4}};
    gert::StorageShape y_shape = {{1, 64, 60, 8}, {1, 64, 60, 8}};
    ExecuteExtraTiling(x_shape, y_shape, {60, 8}, ge::DT_FLOAT, 5, [](gert::TilingContext* ctx) {
        auto* td = ctx->GetTilingData<AdaptiveAvgPool2dOp::AdaptivePool2dSplitHTilingData>();
        ASSERT_NE(td, nullptr);
        EXPECT_EQ(td->ncFactor, 64);
        EXPECT_EQ(td->hoFactor, 1);
        EXPECT_EQ(td->hiFactor, 1);
        EXPECT_EQ(td->inputQueSize, 2048);
        EXPECT_EQ(td->resQue1Size, 4096);
    });
}

// ===================== Base 未覆盖分支 =====================

// GetRealOutDims 的 ONE_DIM 分支(base_tiling.cpp:83-85): output_size 长度为 1, h/w 输出同值
// output_size={4} -> hOut=wOut=4; hIn=8>hOut=4 H下采样 + W下采样 -> SplitH(W↓路径)接管, key=5
TEST_F(AdaptiveAvgPool2dTiling950ExtraTest, test_base_output_size_one_dim_broadcast)
{
    gert::StorageShape x_shape = {{1, 64, 8, 10}, {1, 64, 8, 10}};
    gert::StorageShape y_shape = {{1, 64, 4, 4}, {1, 64, 4, 4}};
    ExecuteExtraTiling(x_shape, y_shape, {4}, ge::DT_FLOAT, 5);
}

// GetRealOutDims 的 NONE_DIM 分支(base_tiling.cpp:88-89): output_size 为空, 输出尺寸取输入后两维
// x={1,64,8,10} -> hOut=8, wOut=10(恒等) -> 无H上采样/H下采样/W上采样, kWMax=1
// -> SmallKernel 接管, key=0
TEST_F(AdaptiveAvgPool2dTiling950ExtraTest, test_base_output_size_none_dim_same_as_input)
{
    gert::StorageShape x_shape = {{1, 64, 8, 10}, {1, 64, 8, 10}};
    gert::StorageShape y_shape = {{1, 64, 8, 10}, {1, 64, 8, 10}};
    ExecuteExtraTiling(x_shape, y_shape, {}, ge::DT_FLOAT, 0);
}

// CalKernelSizeOneDimMax 的 outSize > KERNEL_CALC_COUNT_THERSHOLD(10000) 快速分支(base_tiling.cpp:224):
// wOut=20000 -> kernelWMax=(2+20000-1)/20000+1=2; 输出单行(hOut=1), wOut > SPLIT_H_MAX_WOUT(512)
// -> SplitH 拒绝; kWMax=2 < 32 -> SplitW 拒绝 -> SmallKernel 接管, key=0
TEST_F(AdaptiveAvgPool2dTiling950ExtraTest, test_base_kernel_size_one_dim_max_over_threshold)
{
    gert::StorageShape x_shape = {{1, 64, 2, 2}, {1, 64, 2, 2}};
    gert::StorageShape y_shape = {{1, 64, 1, 20000}, {1, 64, 1, 20000}};
    ExecuteExtraTiling(x_shape, y_shape, {1, 20000}, ge::DT_FLOAT, 0);
}

// CheckNpuArch 非 DAV_3510 拒绝分支(base_tiling.cpp:36): NpuArch=2201
// 所有模板 GetShapeAttrsInfo 均返回 GRAPH_PARAM_INVALID, 注册表遍历完返回 GRAPH_FAILED
TEST_F(AdaptiveAvgPool2dTiling950ExtraTest, test_base_check_npu_arch_unsupported)
{
    gert::StorageShape x_shape = {{1, 64, 4, 4}, {1, 64, 4, 4}};
    gert::StorageShape y_shape = {{1, 64, 2, 2}, {1, 64, 2, 2}};
    ExecuteExtraTilingExpectFail(x_shape, y_shape, {2, 2}, ge::DT_FLOAT, ge::GRAPH_FAILED, "2201");
}

// Base 类默认虚函数实现(base_tiling.cpp:203/205/235/237): IsCapable/DoOpTiling/PostTiling/GetTilingKey
// 所有已注册子类均覆写这些接口, 注册表路径不会命中基类实现, 直接构造 AdaptivePool2dBaseTiling 验证
TEST_F(AdaptiveAvgPool2dTiling950ExtraTest, test_base_class_do_tiling_default_impl)
{
    gert::StorageShape x_shape = {{2, 32, 8, 8}, {2, 32, 8, 8}};
    gert::StorageShape y_shape = {{2, 32, 4, 4}, {2, 32, 4, 4}};

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

    // faker 的 Build 要求 CompileInfo/PlatformInfo 均非空, 传入占位 compileInfo 仅为构建成功
    optiling::AdaptivePool2dCompileInfo placeholderCompileInfo;
    auto ctxHolder = BuildDirectTilingContext(&x_shape, &y_shape, {4, 4}, ge::DT_FLOAT, &placeholderCompileInfo,
                                              &platform_info);
    ASSERT_NE(ctxHolder.context, nullptr);
    ASSERT_NE(ctxHolder.context->GetPlatformInfo(), nullptr);
    ctxHolder.context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    ctxHolder.context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    ctxHolder.context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    ctxHolder.context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    ctxHolder.context->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

    optiling::AdaptivePool2dBaseTiling baseTiling(ctxHolder.context);
    ASSERT_EQ(baseTiling.DoTiling(), ge::GRAPH_SUCCESS);
    ASSERT_EQ(ctxHolder.context->GetTilingKey(), 0U);
    ASSERT_EQ(baseTiling.input_.coreNum, 64U);
    ASSERT_EQ(baseTiling.input_.ubSize, 245760U);
}

// GetPlatformInfo 无平台信息时回退 compileInfo 分支(base_tiling.cpp:189-190)与
// 无平台且无 compileInfo 的错误分支(base_tiling.cpp:185-187):
// metadef OpTilingContextBuilder::BuildTilingContext 对空 platform info 走 GE 断言报错路径
// ("Platform info is nullptr"), TilingContextFaker 无法构造 GetPlatformInfo()==nullptr 的合法
// TilingContext, 该两分支在本 UT 框架下不可达, 详见覆盖报告"无法覆盖代码"清单

// ===================== Simt 未覆盖分支 =====================

// SplitW/SplitH/Simt 的 SetTilingData 非内联函数体覆盖:
// split_w_tiling.cpp:71-74 / split_h_tiling.cpp:283-285 / simt_tiling.cpp:40-48
// 注册表路径下这些函数被内联, 非内联副本从未执行; 通过注册表工厂构造真实模板对象,
// 填充 input_ 后按符号名直接调用 SetTilingData 并校验写入的 tiling data
TEST_F(AdaptiveAvgPool2dTiling950ExtraTest, test_out_of_line_set_tiling_data_impl)
{
    gert::StorageShape x_shape = {{2, 3, 5, 7}, {2, 3, 5, 7}};
    gert::StorageShape y_shape = {{2, 3, 4, 6}, {2, 3, 4, 6}};

    // faker 的 Build 要求 CompileInfo/PlatformInfo 均非空
    optiling::AdaptivePool2dCompileInfo placeholderCompileInfo;
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    auto ctxHolder = BuildDirectTilingContext(&x_shape, &y_shape, {4, 6}, ge::DT_FLOAT16, &placeholderCompileInfo,
                                              &platform_info);
    ASSERT_NE(ctxHolder.context, nullptr);

    const auto& tilingCases = Ops::NN::Optiling::TilingRegistry::GetInstance().GetTilingTemplates("AdaptiveAvgPool2d");
    ASSERT_EQ(tilingCases.count(1), 1U);   // SplitH
    ASSERT_EQ(tilingCases.count(2), 1U);   // SplitW
    ASSERT_EQ(tilingCases.count(100), 1U); // Simt

    // SplitW: SetTilingData 将 input_ 维度写入 AdaptivePool2dSplitWTilingData
    auto splitWObj = tilingCases.at(2)(ctxHolder.context);
    ASSERT_NE(splitWObj, nullptr);
    auto* splitWBase = static_cast<optiling::AdaptivePool2dBaseTiling*>(splitWObj.get());
    splitWBase->input_.nIn = 1;
    splitWBase->input_.cIn = 5;
    splitWBase->input_.hIn = 11;
    splitWBase->input_.wIn = 22;
    splitWBase->input_.hOut = 3;
    splitWBase->input_.wOut = 4;
    _ZN8optiling29AdaptiveAvgPool2dSplitWTiling13SetTilingDataEv(splitWObj.get());
    auto* splitWTd = ctxHolder.context->GetTilingData<AdaptiveAvgPool2dOp::AdaptivePool2dSplitWTilingData>();
    ASSERT_NE(splitWTd, nullptr);
    EXPECT_EQ(splitWTd->hIn, 11);
    EXPECT_EQ(splitWTd->wIn, 22);
    EXPECT_EQ(splitWTd->hOut, 3);
    EXPECT_EQ(splitWTd->wOut, 4);

    // SplitH: SetTilingData 将 input_ 维度写入 AdaptivePool2dSplitHTilingData
    auto splitHObj = tilingCases.at(1)(ctxHolder.context);
    ASSERT_NE(splitHObj, nullptr);
    auto* splitHBase = static_cast<optiling::AdaptivePool2dBaseTiling*>(splitHObj.get());
    splitHBase->input_.nIn = 2;
    splitHBase->input_.cIn = 6;
    splitHBase->input_.hIn = 12;
    splitHBase->input_.wIn = 23;
    splitHBase->input_.hOut = 5;
    splitHBase->input_.wOut = 7;
    _ZN8optiling29AdaptiveAvgPool2dSplitHTiling13SetTilingDataEv(splitHObj.get());
    auto* splitHTd = ctxHolder.context->GetTilingData<AdaptiveAvgPool2dOp::AdaptivePool2dSplitHTilingData>();
    ASSERT_NE(splitHTd, nullptr);
    EXPECT_EQ(splitHTd->hIn, 12);
    EXPECT_EQ(splitHTd->wIn, 23);
    EXPECT_EQ(splitHTd->hOut, 5);
    EXPECT_EQ(splitHTd->wOut, 7);

    // Simt: SetTilingData 将 input_ 维度写入 AdaptivePool2DSimtTilingData
    auto simtObj = tilingCases.at(100)(ctxHolder.context);
    ASSERT_NE(simtObj, nullptr);
    auto* simtBase = static_cast<optiling::AdaptivePool2dBaseTiling*>(simtObj.get());
    simtBase->input_.nIn = 2;
    simtBase->input_.cIn = 3;
    simtBase->input_.hIn = 5;
    simtBase->input_.wIn = 7;
    simtBase->input_.hOut = 4;
    simtBase->input_.wOut = 6;
    _ZN8optiling27AdaptiveAvgPool2DTilingSimt13SetTilingDataEv(simtObj.get());
    auto* simtTd = ctxHolder.context->GetTilingData<AdaptiveAvgPool2dOp::AdaptivePool2DSimtTilingData>();
    ASSERT_NE(simtTd, nullptr);
    EXPECT_EQ(simtTd->nDim, 2);
    EXPECT_EQ(simtTd->cDim, 3);
    EXPECT_EQ(simtTd->hInDim, 5);
    EXPECT_EQ(simtTd->wInDim, 7);
    EXPECT_EQ(simtTd->hOutDim, 4);
    EXPECT_EQ(simtTd->wOutDim, 6);
}

// Simt PostTiling 的 SetLocalMemorySize 失败分支(simt_tiling.cpp:91-93):
// TilingContextFaker 构建的上下文固定携带全部 9 个 tiling 输出槽, SetLocalMemorySize 恒成功;
// 这里将上下文 output_size 截断到 4(kOutputTilingData=3 可用, kOutputLocalMemorySize=7 越界返回空),
// 使 SetLocalMemorySize 取输出槽失败 -> PostTiling 返回 GRAPH_FAILED
TEST_F(AdaptiveAvgPool2dTiling950ExtraTest, test_simt_post_tiling_local_memory_size_failed)
{
    gert::StorageShape x_shape = {{2, 3, 5, 7}, {2, 3, 5, 7}};
    gert::StorageShape y_shape = {{2, 3, 4, 6}, {2, 3, 4, 6}};

    // faker 的 Build 要求 CompileInfo/PlatformInfo 均非空
    optiling::AdaptivePool2dCompileInfo placeholderCompileInfo;
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    auto ctxHolder = BuildDirectTilingContext(&x_shape, &y_shape, {4, 6}, ge::DT_FLOAT, &placeholderCompileInfo,
                                              &platform_info);
    ASSERT_NE(ctxHolder.context, nullptr);
    ctxHolder.context->context_.output_size = 4; // kOutputTilingData(3) 保留, kOutputLocalMemorySize(7) 越界

    const auto& tilingCases = Ops::NN::Optiling::TilingRegistry::GetInstance().GetTilingTemplates("AdaptiveAvgPool2d");
    ASSERT_EQ(tilingCases.count(100), 1U);
    auto simtObj = tilingCases.at(100)(ctxHolder.context);
    ASSERT_NE(simtObj, nullptr);

    auto* simtBase = static_cast<optiling::AdaptivePool2dBaseTiling*>(simtObj.get());
    simtBase->input_.ubSize = 245760;
    ASSERT_EQ(_ZN8optiling27AdaptiveAvgPool2DTilingSimt10PostTilingEv(simtObj.get()), ge::GRAPH_FAILED);
}

// Simt 大kernel多线程分支(simt_tiling.cpp:70): FloorDiv(kH,kW)=64>=8 且 kH*kW=64>=64
// 且 outputSize=8256*1*8=66048 > MAX_THREAD*coreNum=65536 -> threads_=MAX_THREAD(1024)
// 路由: hOut=1<=1 且 kH*kW=64<128 -> SplitH 的 isSingleRowSmallKernel 拒绝;
// SmallKernel 因 UB 不足拒绝; SplitC hOut<=1 拒绝; BigKernel kernelMinHW=64<256 拒绝 -> Simt 接管
TEST_F(AdaptiveAvgPool2dTiling950ExtraTest, test_simt_max_thread_h_down_big_kernel_h)
{
    gert::StorageShape x_shape = {{1, 8256, 64, 8}, {1, 8256, 64, 8}};
    gert::StorageShape y_shape = {{1, 8256, 1, 8}, {1, 8256, 1, 8}};
    ExecuteExtraTiling(x_shape, y_shape, {1, 8}, ge::DT_FLOAT, 2, [](gert::TilingContext* ctx) {
        auto* td = ctx->GetTilingData<AdaptiveAvgPool2dOp::AdaptivePool2DSimtTilingData>();
        ASSERT_NE(td, nullptr);
        EXPECT_EQ(td->nDim, 1);
        EXPECT_EQ(td->cDim, 8256);
        EXPECT_EQ(td->hInDim, 64);
        EXPECT_EQ(td->wInDim, 8);
        EXPECT_EQ(td->hOutDim, 1);
        EXPECT_EQ(td->wOutDim, 8);
        EXPECT_EQ(td->threads, 1024);
    });
}
