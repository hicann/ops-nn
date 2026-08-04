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
 * \file test_in_training_reduce_v2_tiling.cpp
 * \brief arch35 核心路径 UT —— Tiling（AR_FULL_REDUCE / R 全载）
 *   契约（spec.yaml / DESIGN §5.1/§6.3）：
 *     - TilingKey == 200000（AR_FULL_REDUCE）
 *     - IsCapable 对 NCHW / ND / NCDHW 返回 true（一期支持）
 *     - 输入 1（x）/ 输出 2（sum, square_sum），无 attr
 *     - 典型 shape [4,16,32,32] fp32 tiling 成功，产出非空 TilingData
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "ut_op_util.h"
#include "platform/platform_infos_def.h"
#include "test_in_training_reduce_v2_tiling.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class INTrainingReduceV2TilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "INTrainingReduceV2TilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "INTrainingReduceV2TilingTest TearDown" << std::endl; }
};

namespace {
// 公共 compile_info（Ascend950，CORE_NUM=64，UB=245760）
static const char* kCompileInfoStr = R"({
   "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                     "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                     "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                     "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                     "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                     "CORE_NUM": 64, "socVersion": "Ascend950"}
            })";

// 运行一次 in_training_reduce_v2 tiling，返回 tiling_key（失败返回 UINT64_MAX）。
// x_shape：输入 shape；out_shape：sum/square_sum shape（[N,C,1,1] 或 5D [N,C,1,1,1]）；
// fmt：输入 format（NCHW/ND/NCDHW）。
// 注意：shape 以非 const 引用传入——gert::TilingContextFaker 的 InputShapes/OutputShapes 需要
// 非 const 的 StorageShape*，若用 const& 则 &x_shape 得 const 指针，导致 const void*→void* 转换失败。
static uint64_t RunTiling(gert::StorageShape& x_shape, gert::StorageShape& out_shape, ge::DataType dtype,
                          ge::Format fmt)
{
    std::map<std::string, std::string> soc_version_infos = {{"NpuArch", "3510"}};
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(kCompileInfoStr, soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::INTrainingReduceV2CompileInfo compile_info;

    std::string op_type("INTrainingReduceV2");
    if (gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()) == nullptr) {
        return UINT64_MAX;
    }
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tiling parse
    std::string compile_info_string(kCompileInfoStr);
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(3, 3)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    if (!kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init()) {
        return UINT64_MAX;
    }
    auto parse_ctx = kernel_holder.GetContext<gert::TilingParseContext>();
    parse_ctx->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    parse_ctx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    parse_ctx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parse_ctx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    parse_ctx->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    if (tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
        return UINT64_MAX;
    }

    // tiling：1 输入 x / 2 输出 sum,square_sum，无 attr
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    if (param == nullptr) {
        return UINT64_MAX;
    }
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&x_shape})
                      .OutputShapes({&out_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtype, fmt, fmt)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    if (tiling_context->GetPlatformInfo() == nullptr) {
        return UINT64_MAX;
    }
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    if (tiling_func(tiling_context) != ge::GRAPH_SUCCESS) {
        return UINT64_MAX;
    }
    return tiling_context->GetTilingKey();
}
} // namespace

// ---------------------------------------------------------------------------
// AR_FULL_REDUCE：4D NCHW 典型 shape [4,16,32,32] → TilingKey 200000
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_nchw_200000_001)
{
    gert::StorageShape x_shape = {{4, 16, 32, 32}, {4, 16, 32, 32}};
    gert::StorageShape out_shape = {{4, 16, 1, 1}, {4, 16, 1, 1}};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCHW);
    ASSERT_EQ(key, 200000U);
}

// ---------------------------------------------------------------------------
// AR_FULL_REDUCE：fp16 输入同样走 200000（dtype 由编译期宏分派，不进 key）
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_nchw_fp16_200000_002)
{
    gert::StorageShape x_shape = {{2, 8, 16, 16}, {2, 8, 16, 16}};
    gert::StorageShape out_shape = {{2, 8, 1, 1}, {2, 8, 1, 1}};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    ASSERT_EQ(key, 200000U);
}

// ---------------------------------------------------------------------------
// AR_FULL_REDUCE：5D NCDHW [2,3,4,5,6] → [2,3,1,1,1]，IsCapable true，200000
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_ncdhw_5d_200000_004)
{
    gert::StorageShape x_shape = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    gert::StorageShape out_shape = {{2, 3, 1, 1, 1}, {2, 3, 1, 1, 1}};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCDHW);
    ASSERT_EQ(key, 200000U);
}

// ---------------------------------------------------------------------------
// AR_FULL_REDUCE：ND format 也被 IsCapable 接受，200000
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_nd_200000_005)
{
    gert::StorageShape x_shape = {{4, 16, 32, 32}, {4, 16, 32, 32}};
    gert::StorageShape out_shape = {{4, 16, 1, 1}, {4, 16, 1, 1}};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_ND);
    ASSERT_EQ(key, 200000U);
}

// ---------------------------------------------------------------------------
// AR_FULL_REDUCE：退化保留维 N=1/C=1 + 大 R（[1,1,64,64]），tiling 成功
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_keepdim_200000_006)
{
    gert::StorageShape x_shape = {{1, 1, 64, 64}, {1, 1, 64, 64}};
    gert::StorageShape out_shape = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCHW);
    ASSERT_EQ(key, 200000U);
}

// ---------------------------------------------------------------------------
// 除零守卫（代码检视 HIGH）：ND format 下 r = GetShapeSize()/a1/a0 在 a1/a0
// 后置校验之前执行，故 N=0（a1=0）必须被前置守卫拦截，返回 GRAPH_FAILED（而非
// 触发整数除零崩溃）。RunTiling 失败时返回 UINT64_MAX。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_nd_n0_guard_graph_failed_007)
{
    gert::StorageShape x_shape = {{0, 16, 32, 32}, {0, 16, 32, 32}};
    gert::StorageShape out_shape = {{0, 16, 1, 1}, {0, 16, 1, 1}};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_ND);
    ASSERT_EQ(key, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// 除零守卫（代码检视 HIGH）：ND format 下 C=0（a0=0）同样在除法前被守卫拦截，
// 返回 GRAPH_FAILED。此分支仅由新增前置守卫覆盖（后置 a0<=0 检查在除法之后）。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_nd_c0_guard_graph_failed_008)
{
    gert::StorageShape x_shape = {{4, 0, 32, 32}, {4, 0, 32, 32}};
    gert::StorageShape out_shape = {{4, 0, 1, 1}, {4, 0, 1, 1}};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_ND);
    ASSERT_EQ(key, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// sub-R 分块路径：R 超大（超单次 UB 容量）时触发 DoSubRTiling，
// TilingKey 仍为 200000（同 key + isSubRTiling 标志区分）。
// fp32 [1,1,316,316] → R=99856，单行全载超 UB → sub-R 分块。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_sub_r_nchw_fp32_200000_009)
{
    gert::StorageShape x_shape = {{1, 1, 316, 316}, {1, 1, 316, 316}};
    gert::StorageShape out_shape = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCHW);
    ASSERT_EQ(key, 200000U);
}

// ---------------------------------------------------------------------------
// sub-R 分块路径：fp16 输入，5D NCDHW，R 超大 → sub-R 分块。
// [1,1,100,100,100] → R=1,000,000，fp16 elemSize=2 → sub-R。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_sub_r_ncdhw_fp16_200000_010)
{
    gert::StorageShape x_shape = {{1, 1, 100, 100, 100}, {1, 1, 100, 100, 100}};
    gert::StorageShape out_shape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT16, ge::FORMAT_NCDHW);
    ASSERT_EQ(key, 200000U);
}

// ---------------------------------------------------------------------------
// sub-R 分块路径：ND format，3D [1,1,1000000] → R=1,000,000，fp32 → sub-R。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_sub_r_nd_fp32_200000_011)
{
    gert::StorageShape x_shape = {{1, 1, 1000000}, {1, 1, 1000000}};
    gert::StorageShape out_shape = {{1, 1, 1}, {1, 1, 1}};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_ND);
    ASSERT_EQ(key, 200000U);
}
