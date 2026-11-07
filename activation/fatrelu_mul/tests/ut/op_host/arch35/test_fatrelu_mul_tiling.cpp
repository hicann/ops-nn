/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* arch35 (ascend950) tiling UT：正常 shape 零回归 + int64 乘积溢出拒绝（T10 CODE.06） */

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <gtest/gtest.h> // NOLINT
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/fatrelu_mul_tiling_arch35.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class FatreluMulTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FatreluMulTilingArch35 SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "FatreluMulTilingArch35 TearDown" << std::endl; }
};

static void FatreluMulTilingArch35TestCase(std::initializer_list<int64_t> xShape,
                                           std::initializer_list<int64_t> threShape,
                                           std::initializer_list<int64_t> yShape, ge::DataType dtype,
                                           ge::graphStatus expectedStatus, int64_t expectedBatchSize = -1)
{
    gert::StorageShape x_shape = {xShape, xShape};
    gert::StorageShape thre_shape = {threShape, threShape};
    gert::StorageShape y_shape = {yShape, yShape};
    std::map<std::string, std::string> soc_infos;
    std::map<std::string, std::string> aicore_spec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    std::string compile_info_string = R"({
      "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64
      }
    })";
    std::string op_type("FatreluMul");

    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::FatreluMulCompileInfo compile_info;

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("FatreluMul")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("FatreluMul")->tiling_parse;

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
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&x_shape, &thre_shape})
                      .OutputShapes({&y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);

    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), expectedStatus);
    if (expectedStatus != ge::GRAPH_SUCCESS || expectedBatchSize < 0) {
        return;
    }
    // 行模型断言（rank 透明性见证）：batchSize = ∏(前 n-1 维)
    auto rawTilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(rawTilingData, nullptr);
    ASSERT_NE(rawTilingData->GetData(), nullptr);
    const auto* tilingData = reinterpret_cast<const FatreluMulTilingData*>(rawTilingData->GetData());
    EXPECT_EQ(tilingData->batch_size, expectedBatchSize);
}

TEST_F(FatreluMulTilingArch35, test_tiling_fp32_normal_001)
{
    FatreluMulTilingArch35TestCase({2, 8}, {1}, {2, 4}, ge::DT_FLOAT, ge::GRAPH_SUCCESS);
}

TEST_F(FatreluMulTilingArch35, test_tiling_fp16_normal_002)
{
    FatreluMulTilingArch35TestCase({2, 8}, {1}, {2, 4}, ge::DT_FLOAT16, ge::GRAPH_SUCCESS);
}

TEST_F(FatreluMulTilingArch35, test_tiling_bf16_normal_003)
{
    FatreluMulTilingArch35TestCase({2, 8}, {1}, {2, 4}, ge::DT_BF16, ge::GRAPH_SUCCESS);
}

// 空 Tensor（前导维 0）短路合法
TEST_F(FatreluMulTilingArch35, test_tiling_empty_batch_004)
{
    FatreluMulTilingArch35TestCase({0, 8}, {1}, {0, 4}, ge::DT_FLOAT, ge::GRAPH_SUCCESS);
}

// 末维 N=0（A5/arch35 合法空末维）短路
TEST_F(FatreluMulTilingArch35, test_tiling_empty_last_dim_005)
{
    FatreluMulTilingArch35TestCase({4, 0}, {1}, {4, 0}, ge::DT_FLOAT, ge::GRAPH_SUCCESS);
}

// 大而合法的 shape：乘积不溢出，不得误拒
TEST_F(FatreluMulTilingArch35, test_tiling_large_legal_006)
{
    FatreluMulTilingArch35TestCase({8, 65536}, {1}, {8, 32768}, ge::DT_FLOAT, ge::GRAPH_SUCCESS);
}

// rank=1 低于下界 2：tiling 校验拒绝（既有防线，零回归锚点）
TEST_F(FatreluMulTilingArch35, test_tiling_rank1_rejected_007)
{
    FatreluMulTilingArch35TestCase({8}, {1}, {4}, ge::DT_FLOAT, ge::GRAPH_FAILED);
}

// batchSize 累加乘积溢出：x=[2^40, 2^23, 8]，前导维乘积 = 2^63 > INT64_MAX，
// 溢出前保护应拒绝（fatrelu_mul_tiling_arch35.cpp ExpandRowModel 乘积循环）
TEST_F(FatreluMulTilingArch35, test_tiling_batch_product_overflow_008)
{
    constexpr int64_t dim2pow40 = 1LL << 40;
    constexpr int64_t dim2pow23 = 1LL << 23;
    FatreluMulTilingArch35TestCase({dim2pow40, dim2pow23, 8}, {1}, {dim2pow40, dim2pow23, 4}, ge::DT_FLOAT,
                                   ge::GRAPH_FAILED);
}

// 多核切分总字节数乘积溢出：x=[4, 2^62]，batchSize*2*halfDim*dtypeBytes = 2^66 > INT64_MAX，
// 溢出前保护应拒绝（fatrelu_mul_tiling_arch35.cpp MultiCoreSplit 乘积）
TEST_F(FatreluMulTilingArch35, test_tiling_total_bytes_overflow_009)
{
    constexpr int64_t dim2pow62 = 1LL << 62;
    constexpr int64_t dim2pow61 = 1LL << 61;
    FatreluMulTilingArch35TestCase({4, dim2pow62}, {1}, {4, dim2pow61}, ge::DT_FLOAT, ge::GRAPH_FAILED);
}

// 前导维本身达 2^62：batchSize*2 即溢出，应拒绝
TEST_F(FatreluMulTilingArch35, test_tiling_batch_dim_overflow_010)
{
    constexpr int64_t dim2pow62 = 1LL << 62;
    FatreluMulTilingArch35TestCase({dim2pow62, 2}, {1}, {dim2pow62, 1}, ge::DT_FLOAT16, ge::GRAPH_FAILED);
}

// rank=9 越旧上界 8：上界已放开（rank(x) ≥ 2，行模型 rank 透明），应放行；
// batchSize = ∏(前 8 维) = 2^8 = 256（行模型展开断言，防逐维索引回退）
TEST_F(FatreluMulTilingArch35, test_tiling_rank9_fwd_011)
{
    FatreluMulTilingArch35TestCase({2, 2, 2, 2, 2, 2, 2, 2, 8}, {1}, {2, 2, 2, 2, 2, 2, 2, 2, 4}, ge::DT_FLOAT,
                                   ge::GRAPH_SUCCESS, 256);
}

// threshold rank=9 且 numel=1：rank 不限（单元素为唯一语义约束），应放行；
// batchSize = 前导维乘积 = 2（行模型断言）
TEST_F(FatreluMulTilingArch35, test_tiling_thr_rank9_numel1_fwd_012)
{
    FatreluMulTilingArch35TestCase({2, 8}, {1, 1, 1, 1, 1, 1, 1, 1, 1}, {2, 4}, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 2);
}

// threshold numel=2：非单元素，tiling 校验拒绝（numel==1 唯一约束的拒绝路径）
TEST_F(FatreluMulTilingArch35, test_tiling_thr_numel2_rejected_013)
{
    FatreluMulTilingArch35TestCase({2, 8}, {2}, {2, 4}, ge::DT_FLOAT, ge::GRAPH_FAILED);
}

// threshold 0-d 标量（rank=0，numel=1）：合法输入，应放行
TEST_F(FatreluMulTilingArch35, test_tiling_thr_scalar0d_fwd_014)
{
    FatreluMulTilingArch35TestCase({2, 8}, {}, {2, 4}, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 2);
}
