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
 * \file test_bn3d_training_reduce_tiling.cpp
 * \brief arch35 核心路径 UT —— Tiling（DENSE_CHANNEL）
 *
 * 契约（DESIGN / bn3d_training_reduce_tiling_dense_channel_arch35.cpp）：
 *   - TilingKey 有三个值：
 *       100000 DENSE_CHANNEL     —— NCDHW / NCHW，每通道输出 1 个标量（numC0=0）；
 *       200000 NDC1HWC0_CHANNEL  —— NDC1HWC0，每通道输出 C0 个标量（numC0=C0）；
 *       300000 SPLIT_REDUCE      —— channel-first 低通道超大 R，通道内多核归约；
 *   - shape 归一化按 **storage format** 走同一个 R1-A-R0 模型：
 *       NCDHW 收 rank 2~5、NCHW 只收 rank 4 → R1=dim0(N)、A=dim1(C)、R0=product(dim2:)；
 *       NDC1HWC0 固定 rank 6 [N,D,C1,H,W,C0] → R1=N*D、A=C1、R0=H*W*C0；
 *       其余 storage format 一律失败；
 *   - C == 0 走 no-work 分支：usedCoreNum=0 且 blockDim=1（不下发 blockDim=0）；
 *   - UB 放得下整行 R0 时 isSubR=0，否则 isSubR=1 且 nTile 恒为 1。
 *
 * 本组用例不止断言 TilingKey——两条路线内部还各有 R0 全载 / sub-R 分块 / 空通道
 * 等分支，仅断言 key 区分不出来，故逐字段校验 TilingData 与 blockDim。
 */

#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "ut_op_util.h"
#include "platform/platform_infos_def.h"
#include "test_bn3d_training_reduce_tiling.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class BN3DTrainingReduceTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BN3DTrainingReduceTilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "BN3DTrainingReduceTilingTest TearDown" << std::endl; }
};

namespace {
constexpr uint64_t TILINGKEY_DENSE_CHANNEL = 100000U;
constexpr uint64_t TILINGKEY_NDC1HWC0_CHANNEL = 200000U;
constexpr uint64_t TILINGKEY_SPLIT_REDUCE = 300000U;

// 公共 compile_info。UB_SIZE / CORE_NUM 取真机 Ascend950PR 的实际值
// （platform_config/Ascend950PR_957*.ini 中 ub_size=253952），使 UT 里的
// R0 全载 / sub-R 分界与真机一致。
static const char* kCompileInfoStr = R"({
   "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                     "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                     "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                     "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                     "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                     "CORE_NUM": 64, "socVersion": "Ascend950"}
            })";

struct TilingResult {
    uint64_t key{UINT64_MAX};
    uint32_t blockDim{0};
    optiling::BN3DTrainingReduceDenseChannelTilingData td{};
    uint32_t vectorLength{0}; // tiling_parse 解析出的向量寄存器字节宽度
    bool ok{false};
};

// 跑一次 bn3d_training_reduce tiling。
// x_shape：输入 storage shape；out_shape：sum/square_sum shape（[C]）；
// storage_fmt：输入 storage format（决定 shape 归一化）；
// origin_fmt：输入 origin format（tiling 侧仅记录，语义校验在 InferShape）。
// 注意：shape 以非 const 引用传入——gert::TilingContextFaker 的 InputShapes/OutputShapes
// 需要非 const 的 StorageShape*。
static TilingResult RunTiling(gert::StorageShape& x_shape, gert::StorageShape& out_shape, ge::DataType dtype,
                              ge::Format storage_fmt, ge::Format origin_fmt, ge::DataType out_dtype = ge::DT_FLOAT)
{
    TilingResult result;

    std::map<std::string, std::string> soc_version_infos = {{"NpuArch", "3510"}};
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(kCompileInfoStr, soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::BN3DTrainingReduceCompileInfo compile_info;

    std::string op_type("BN3DTrainingReduce");
    if (gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()) == nullptr) {
        return result;
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
        return result;
    }
    auto parse_ctx = kernel_holder.GetContext<gert::TilingParseContext>();
    parse_ctx->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    parse_ctx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    parse_ctx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parse_ctx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    parse_ctx->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    if (tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
        return result;
    }
    result.vectorLength = compile_info.vectorLength;

    // tiling：1 输入 x / 2 输出 sum,square_sum，无 attr
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    if (param == nullptr) {
        return result;
    }
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&x_shape})
                      .OutputShapes({&out_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtype, origin_fmt, storage_fmt)
                      .NodeOutputTd(0, out_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, out_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    if (tiling_context->GetPlatformInfo() == nullptr) {
        return result;
    }
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    if (tiling_func(tiling_context) != ge::GRAPH_SUCCESS) {
        return result;
    }

    result.key = tiling_context->GetTilingKey();
    result.blockDim = tiling_context->GetBlockDim();
    auto raw = tiling_context->GetRawTilingData();
    if (raw != nullptr && raw->GetDataSize() >= sizeof(optiling::BN3DTrainingReduceDenseChannelTilingData)) {
        result.td = *reinterpret_cast<const optiling::BN3DTrainingReduceDenseChannelTilingData*>(raw->GetData());
    }
    result.ok = true;
    return result;
}

static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& originDims,
                                           const std::vector<int64_t>& storageDims)
{
    gert::StorageShape shape;
    for (const int64_t dim : originDims) {
        shape.MutableOriginShape().AppendDim(dim);
    }
    for (const int64_t dim : storageDims) {
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims) { return MakeStorageShape(dims, dims); }
} // namespace

// ---------------------------------------------------------------------------
// storage NCDHW rank 5 fp32：R0 全载路径。
// [2,3,4,4,8] → N=2, C=3, R0=4*4*8=128。UB 充裕故 nTile 取满 N，isSubR=0。
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_dense_channel_ncdhw_rank5_fp32_001)
{
    gert::StorageShape x_shape = {{2, 3, 4, 4, 8}, {2, 3, 4, 4, 8}};
    gert::StorageShape out_shape = {{3}, {3}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);

    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.key, TILINGKEY_DENSE_CHANNEL);
    EXPECT_EQ(r.td.numN, 2U);
    EXPECT_EQ(r.td.numC, 3U);
    EXPECT_EQ(r.td.numR0, 128U);
    EXPECT_EQ(r.td.numC0, 0U); // channel-first：每通道 1 个标量，不折叠
    EXPECT_EQ(r.td.isSubR, 0U);
    EXPECT_EQ(r.td.nTile, 2U); // R0 全载且 UB 充裕 → nTile 取满 N
    // 通道独占分核：C=3 < 核数 → 每核 1 个通道，用 3 个核。
    EXPECT_EQ(r.td.cPerCore, 1U);
    EXPECT_EQ(r.td.usedCoreNum, 3U);
    EXPECT_EQ(r.blockDim, 3U);
    // UB 内行步长向上对齐到 VL_FP32。
    ASSERT_GT(r.vectorLength, 0U);
    const uint64_t vlFp32 = r.vectorLength / sizeof(float);
    EXPECT_GE(r.td.r0Align, r.td.numR0);
    EXPECT_EQ(r.td.r0Align % vlFp32, 0U);
}

// ---------------------------------------------------------------------------
// storage NCDHW rank 5 fp16 / bf16：dtype 由编译期宏分派，不进 TilingKey。
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_dense_channel_ncdhw_fp16_002)
{
    gert::StorageShape x_shape = {{2, 3, 4, 4, 8}, {2, 3, 4, 4, 8}};
    gert::StorageShape out_shape = {{3}, {3}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);

    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.key, TILINGKEY_DENSE_CHANNEL);
    EXPECT_EQ(r.td.numR0, 128U);
    EXPECT_EQ(r.td.isSubR, 0U);
}

TEST_F(BN3DTrainingReduceTilingTest, tiling_dense_channel_ncdhw_bf16_003)
{
    gert::StorageShape x_shape = {{2, 3, 4, 4, 8}, {2, 3, 4, 4, 8}};
    gert::StorageShape out_shape = {{3}, {3}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_BF16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);

    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.key, TILINGKEY_DENSE_CHANNEL);
    EXPECT_EQ(r.td.isSubR, 0U);
}

// ---------------------------------------------------------------------------
// storage NCDHW rank 2：下边界，R0 为空乘积 1。
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_dense_channel_ncdhw_rank2_004)
{
    gert::StorageShape x_shape = {{4, 5}, {4, 5}};
    gert::StorageShape out_shape = {{5}, {5}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);

    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.key, TILINGKEY_DENSE_CHANNEL);
    EXPECT_EQ(r.td.numN, 4U);
    EXPECT_EQ(r.td.numC, 5U);
    EXPECT_EQ(r.td.numR0, 1U);
    EXPECT_EQ(r.td.isSubR, 0U);
}

// ---------------------------------------------------------------------------
// storage NCHW rank 4：NCHW 只收 rank 4。
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_dense_channel_nchw_rank4_005)
{
    gert::StorageShape x_shape = {{2, 3, 8, 8}, {2, 3, 8, 8}};
    gert::StorageShape out_shape = {{3}, {3}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCHW);

    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.key, TILINGKEY_DENSE_CHANNEL);
    EXPECT_EQ(r.td.numN, 2U);
    EXPECT_EQ(r.td.numC, 3U);
    EXPECT_EQ(r.td.numR0, 64U);
}

// ---------------------------------------------------------------------------
// storage NCHW rank 5：NCHW 只收 rank 4 → GRAPH_FAILED
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_nchw_rank5_failed_006)
{
    gert::StorageShape x_shape = {{2, 3, 4, 4, 8}, {2, 3, 4, 4, 8}};
    gert::StorageShape out_shape = {{3}, {3}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW);

    EXPECT_FALSE(r.ok);
}

// ---------------------------------------------------------------------------
// storage NCDHW rank 6：越上边界 → GRAPH_FAILED
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_ncdhw_rank6_failed_007)
{
    gert::StorageShape x_shape = {{2, 3, 4, 5, 6, 7}, {2, 3, 4, 5, 6, 7}};
    gert::StorageShape out_shape = {{3}, {3}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);

    EXPECT_FALSE(r.ok);
}

// ---------------------------------------------------------------------------
// storage NCDHW rank 1：越下边界 → GRAPH_FAILED
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_ncdhw_rank1_failed_008)
{
    gert::StorageShape x_shape = {{8}, {8}};
    gert::StorageShape out_shape = {{8}, {8}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);

    EXPECT_FALSE(r.ok);
}

// ---------------------------------------------------------------------------
// storage FORMAT_ND → GRAPH_FAILED（ND 不是受支持的 storage format）
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_storage_nd_failed_009)
{
    gert::StorageShape x_shape = {{2, 3, 4, 4, 8}, {2, 3, 4, 4, 8}};
    gert::StorageShape out_shape = {{3}, {3}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);

    EXPECT_FALSE(r.ok);
}

// ---------------------------------------------------------------------------
// storage NDC1HWC0 rank 6：走 NDC1HWC0_CHANNEL 路线。
// [2,4,3,8,8,16] → R1=N*D=8、A=C1=3、R0=H*W*C0=8*8*16=1024、numC0=16。
// 与 NCDHW 共用同一套 R1-A-R0 归一化，差别只在每通道输出 C0 个标量。
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_ndc1hwc0_rank6_fp16_010)
{
    gert::StorageShape x_shape = {{2, 4, 3, 8, 8, 16}, {2, 4, 3, 8, 8, 16}};
    gert::StorageShape out_shape = {{1, 1, 3, 1, 1, 16}, {1, 1, 3, 1, 1, 16}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0);

    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.key, TILINGKEY_NDC1HWC0_CHANNEL);
    EXPECT_EQ(r.td.numN, 8U); // R1 = N * D
    EXPECT_EQ(r.td.numC, 3U); // A  = C1
    EXPECT_EQ(r.td.numR0, 1024U);
    EXPECT_EQ(r.td.numC0, 16U);
    EXPECT_EQ(r.td.isSubR, 0U);
    // 通道独占分核：C1=3 → 3 个核。
    EXPECT_EQ(r.td.cPerCore, 1U);
    EXPECT_EQ(r.td.usedCoreNum, 3U);
    EXPECT_EQ(r.blockDim, 3U);
    // C0 折叠要求 C0 整除 VL_FP32，否则 tiling 必须直接失败（见 010d）。
    ASSERT_GT(r.vectorLength, 0U);
    const uint64_t vlFp32 = r.vectorLength / sizeof(float);
    EXPECT_EQ(vlFp32 % r.td.numC0, 0U);
}

// ---------------------------------------------------------------------------
// storage NDC1HWC0 C0=32 fp32：C0 不是只有 16 一种取值，折叠宽度随 C0 变。
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_ndc1hwc0_c0_32_fp32_010a)
{
    gert::StorageShape x_shape = {{2, 2, 3, 4, 4, 32}, {2, 2, 3, 4, 4, 32}};
    gert::StorageShape out_shape = {{1, 1, 3, 1, 1, 32}, {1, 1, 3, 1, 1, 32}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0);

    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.key, TILINGKEY_NDC1HWC0_CHANNEL);
    EXPECT_EQ(r.td.numN, 4U);
    EXPECT_EQ(r.td.numC, 3U);
    EXPECT_EQ(r.td.numR0, 512U); // 4 * 4 * 32
    EXPECT_EQ(r.td.numC0, 32U);
    EXPECT_EQ(r.td.isSubR, 0U);
}

// ---------------------------------------------------------------------------
// storage NDC1HWC0 sub-R：R0 = 64*64*16 = 65536 个 fp32 = 256KB > UB 预算，
// 必须分块。分块偏移必须是 C0 的整数倍，否则 C0 折叠的 lane→c0 映射会错位；
// r0Factor 对齐到 VL_FP32 且 C0 | VL_FP32，故该性质自动成立，这里把它钉住。
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_ndc1hwc0_subr_010b)
{
    gert::StorageShape x_shape = {{2, 2, 2, 64, 64, 16}, {2, 2, 2, 64, 64, 16}};
    gert::StorageShape out_shape = {{1, 1, 2, 1, 1, 16}, {1, 1, 2, 1, 1, 16}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0);

    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.key, TILINGKEY_NDC1HWC0_CHANNEL);
    EXPECT_EQ(r.td.numN, 4U);
    EXPECT_EQ(r.td.numC, 2U);
    EXPECT_EQ(r.td.numR0, 65536U);
    EXPECT_EQ(r.td.numC0, 16U);
    EXPECT_EQ(r.td.isSubR, 1U);
    EXPECT_EQ(r.td.nTile, 1U); // sub-R 下 nTile 恒为 1
    ASSERT_GT(r.td.numChunks, 0U);
    ASSERT_GT(r.td.r0Factor, 0U);
    // 分块自洽：(numChunks - 1) 个整块 + 1 个 tail == R0。
    EXPECT_EQ((r.td.numChunks - 1U) * r.td.r0Factor + r.td.tailLen, r.td.numR0);
    // 折叠前提：块偏移 ≡ 0 (mod C0)。
    EXPECT_EQ(r.td.r0Factor % r.td.numC0, 0U);
    EXPECT_EQ(r.td.tailLen % r.td.numC0, 0U);
}

// ---------------------------------------------------------------------------
// storage NDC1HWC0 但 rank != 6 → GRAPH_FAILED（NDC1HWC0 的 rank 是固定的）。
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_ndc1hwc0_rank5_failed_010c)
{
    gert::StorageShape x_shape = {{2, 4, 3, 8, 16}, {2, 4, 3, 8, 16}};
    gert::StorageShape out_shape = {{1, 1, 3, 1, 1, 16}, {1, 1, 3, 1, 1, 16}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0);

    EXPECT_FALSE(r.ok);
}

// ---------------------------------------------------------------------------
// storage NDC1HWC0 且 C0 不整除 VL_FP32 → GRAPH_FAILED。
// C0 折叠的前提是「VL 宽累加器的 lane L 恒对应 c0 = L % C0」，C0 ∤ VL_FP32 时
// 该映射不成立，tiling 必须拒绝而不是产出一份会算错的 TilingData。
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_ndc1hwc0_c0_not_divide_vl_failed_010d)
{
    gert::StorageShape x_shape = {{2, 2, 3, 4, 4, 24}, {2, 2, 3, 4, 4, 24}};
    gert::StorageShape out_shape = {{1, 1, 3, 1, 1, 24}, {1, 1, 3, 1, 1, 24}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0);

    EXPECT_FALSE(r.ok);
}

// ---------------------------------------------------------------------------
// C0 整除 VL_FP32 但不满 UB block（C0 ∈ {1,2,4}，C0 * 4B < 32B）：必须在 Host 侧拒绝。
//
// 回归背景：这三个取值曾只被"C0 | VL_FP32"一条约束放行，Host 算得出 tiling，
// 但 Kernel 的 C0 折叠按 k * C0 / slot * C0（单位 fp32）递进，偏移落不到 32B 边界，
// 真机 500 条 TTK 回归里 C0 ∈ {1,2,4} 的 24 条全部 VEC_ERROR，C0 >= 8 的 119 条全过。
// 这里逐个钉死，防止日后有人只看到"整除"那一条就把校验放松回去。
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_ndc1hwc0_c0_below_ub_block_failed_010g)
{
    for (int64_t c0 : {1, 2, 4}) {
        gert::StorageShape x_shape = {{2, 2, 3, 4, 4, c0}, {2, 2, 3, 4, 4, c0}};
        gert::StorageShape out_shape = {{1, 1, 3, 1, 1, c0}, {1, 1, 3, 1, 1, c0}};
        auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0);

        EXPECT_FALSE(r.ok) << "C0 = " << c0 << " 不满一个 UB block，必须拒绝";
    }
}

// C0 == 8：恰好等于一个 UB block（8 * 4B = 32B），是受支持的最小取值，必须放行。
TEST_F(BN3DTrainingReduceTilingTest, tiling_ndc1hwc0_c0_8_min_supported_010h)
{
    gert::StorageShape x_shape = {{2, 3, 4, 4, 4, 8}, {2, 3, 4, 4, 4, 8}};
    gert::StorageShape out_shape = {{1, 1, 4, 1, 1, 8}, {1, 1, 4, 1, 1, 8}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0);

    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.key, TILINGKEY_NDC1HWC0_CHANNEL);
    EXPECT_EQ(r.td.numC0, 8U);
    EXPECT_EQ(r.td.numR0, 128U); // 4 * 4 * 8
}

// ---------------------------------------------------------------------------
// storage NDC1HWC0 且 C1 == 0：空通道 no-work 分支，与 NCDHW 的 C==0 同语义。
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_ndc1hwc0_empty_channel_010e)
{
    gert::StorageShape x_shape = {{2, 2, 0, 4, 4, 16}, {2, 2, 0, 4, 4, 16}};
    gert::StorageShape out_shape = {{1, 1, 0, 1, 1, 16}, {1, 1, 0, 1, 1, 16}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0);

    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.usedCoreNum, 0U);
    EXPECT_EQ(r.blockDim, 1U); // 不下发 blockDim=0
}

// ---------------------------------------------------------------------------
// storage NDC1HWC0 且 C0 == 0：C1 * C0 同样是 0 个有效通道，走 no-work 分支。
// C0 是 NDC1HWC0 独有的第二条"通道维"，C1=0 通不代表 C0=0 也通，故单列一条。
// 若这里没走 no-work，r0 = H*W*C0 会是 0，SolveUbSplit 里 r0Align 归零后
// `inputBudget / rowBytesDoubleBuf` 就成了整型除零。
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_ndc1hwc0_zero_c0_010f)
{
    gert::StorageShape x_shape = {{2, 2, 3, 4, 4, 0}, {2, 2, 3, 4, 4, 0}};
    gert::StorageShape out_shape = {{1, 1, 3, 1, 1, 0}, {1, 1, 3, 1, 1, 0}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0);

    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.usedCoreNum, 0U);
    EXPECT_EQ(r.blockDim, 1U);
}

// ---------------------------------------------------------------------------
// 输入 dtype 不在 {fp16, fp32, bf16} → GRAPH_FAILED
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_invalid_input_dtype_failed_011)
{
    gert::StorageShape x_shape = {{2, 3, 4, 4, 8}, {2, 3, 4, 4, 8}};
    gert::StorageShape out_shape = {{3}, {3}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_INT32, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);

    EXPECT_FALSE(r.ok);
}

// ---------------------------------------------------------------------------
// 输出 dtype 非 fp32 → GRAPH_FAILED（两个输出恒 fp32，与输入 dtype 无关）
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_invalid_output_dtype_failed_012)
{
    gert::StorageShape x_shape = {{2, 3, 4, 4, 8}, {2, 3, 4, 4, 8}};
    gert::StorageShape out_shape = {{3}, {3}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::DT_FLOAT16);

    EXPECT_FALSE(r.ok);
}

// ---------------------------------------------------------------------------
// C == 0：no-work 分支。两个输出为空，不启动归约。
// usedCoreNum 必须为 0，blockDim 必须为 1——不能下发未经验证的 blockDim=0。
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_empty_channel_no_work_013)
{
    gert::StorageShape x_shape = {{2, 0, 4, 4, 8}, {2, 0, 4, 4, 8}};
    gert::StorageShape out_shape = {{0}, {0}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);

    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.key, TILINGKEY_DENSE_CHANNEL);
    EXPECT_EQ(r.td.numC, 0U);
    EXPECT_EQ(r.td.usedCoreNum, 0U);
    EXPECT_EQ(r.blockDim, 1U);
}

// ---------------------------------------------------------------------------
// C > 0 但 N == 0：归约集合为空，属于不支持的输入 → GRAPH_FAILED
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_n0_with_c_positive_failed_014)
{
    gert::StorageShape x_shape = {{0, 3, 4, 4, 8}, {0, 3, 4, 4, 8}};
    gert::StorageShape out_shape = {{3}, {3}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);

    EXPECT_FALSE(r.ok);
}

// ---------------------------------------------------------------------------
// C > 0 但某个空间归约维为 0 → GRAPH_FAILED
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_spatial_dim0_with_c_positive_failed_015)
{
    gert::StorageShape x_shape = {{2, 3, 4, 0, 8}, {2, 3, 4, 0, 8}};
    gert::StorageShape out_shape = {{3}, {3}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);

    EXPECT_FALSE(r.ok);
}

// ---------------------------------------------------------------------------
// 多核分通道 + nTile > 1 的跨 N 跳搬：[8,256,64]。
// C=256 > 核数 64 → 每核 4 个通道，64 核用满；R0=64 很小，UB 放得下全部 8 行。
// 与真机上跑通的 8x256x64 用例同形。
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_multicore_ntile_gt1_016)
{
    gert::StorageShape x_shape = {{8, 256, 64}, {8, 256, 64}};
    gert::StorageShape out_shape = {{256}, {256}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);

    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.key, TILINGKEY_DENSE_CHANNEL);
    EXPECT_EQ(r.td.numN, 8U);
    EXPECT_EQ(r.td.numC, 256U);
    EXPECT_EQ(r.td.numR0, 64U);
    EXPECT_EQ(r.td.cPerCore, 4U);
    EXPECT_EQ(r.td.usedCoreNum, 64U);
    EXPECT_EQ(r.blockDim, 64U);
    EXPECT_EQ(r.td.isSubR, 0U);
    EXPECT_EQ(r.td.nTile, 8U); // 一次 DataCopyPad 搬完全部 N 行
    // blockCount 是 uint16_t，nTile 不得越界。
    EXPECT_LE(r.td.nTile, 65535U);
}

// ---------------------------------------------------------------------------
// sub-R 分块：[2,4,65536] fp32。
// 单行 R0 需 65536*4*2 = 524288 B（含双 buffer），超出 UB 预算（UB=253952）→ isSubR=1，
// 此时 nTile 恒为 1，且分块参数须自洽。与真机上跑通的 2x4x65536 用例同形。
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_sub_r_fp32_017)
{
    gert::StorageShape x_shape = {{2, 4, 65536}, {2, 4, 65536}};
    gert::StorageShape out_shape = {{4}, {4}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);

    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.key, TILINGKEY_DENSE_CHANNEL);
    EXPECT_EQ(r.td.numR0, 65536U);
    EXPECT_EQ(r.td.isSubR, 1U);
    EXPECT_EQ(r.td.nTile, 1U);

    // 分块参数自洽：r0Factor 是 VL_FP32 整数倍；分块能覆盖且不超出 R0；尾块长度正确。
    ASSERT_GT(r.vectorLength, 0U);
    const uint64_t vlFp32 = r.vectorLength / sizeof(float);
    ASSERT_GT(r.td.r0Factor, 0U);
    EXPECT_EQ(r.td.r0Factor % vlFp32, 0U);
    ASSERT_GT(r.td.numChunks, 0U);
    EXPECT_GE(r.td.numChunks * r.td.r0Factor, r.td.numR0);
    EXPECT_LT((r.td.numChunks - 1U) * r.td.r0Factor, r.td.numR0);
    EXPECT_EQ(r.td.tailLen, r.td.numR0 - (r.td.numChunks - 1U) * r.td.r0Factor);
    EXPECT_LE(r.td.tailLen, r.td.r0Factor);
}

// ---------------------------------------------------------------------------
// sub-R 分块：fp16 元素占 2B，需要更大的 R0 才越过同一条 UB 分界。
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_sub_r_fp16_018)
{
    gert::StorageShape x_shape = {{2, 4, 262144}, {2, 4, 262144}};
    gert::StorageShape out_shape = {{4}, {4}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);

    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.key, TILINGKEY_DENSE_CHANNEL);
    EXPECT_EQ(r.td.isSubR, 1U);
    EXPECT_EQ(r.td.nTile, 1U);
    EXPECT_EQ(r.td.tailLen, r.td.numR0 - (r.td.numChunks - 1U) * r.td.r0Factor);
}

// ---------------------------------------------------------------------------
// R0 非 VL 对齐：[3,37,100]，r0Align 向上对齐后大于 R0，尾块由 mask 处理；
// C=37 不整除核数 → cPerCore=1、usedCoreNum=37。与真机上跑通的 3x37x100 同形。
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_r0_unaligned_tail_019)
{
    gert::StorageShape x_shape = {{3, 37, 100}, {3, 37, 100}};
    gert::StorageShape out_shape = {{37}, {37}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);

    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.key, TILINGKEY_DENSE_CHANNEL);
    EXPECT_EQ(r.td.numR0, 100U);
    EXPECT_EQ(r.td.cPerCore, 1U);
    EXPECT_EQ(r.td.usedCoreNum, 37U);
    EXPECT_EQ(r.blockDim, 37U);

    ASSERT_GT(r.vectorLength, 0U);
    const uint64_t vlFp32 = r.vectorLength / sizeof(float);
    EXPECT_GT(r.td.r0Align, r.td.numR0); // 100 不是 VL_FP32 的整数倍，必然被抬高
    EXPECT_EQ(r.td.r0Align % vlFp32, 0U);
}

// ---------------------------------------------------------------------------
// 【UT 桩限制，勿再尝试】本组用例无法区分 tiling 走的是 storage format 还是 origin format。
//
// ParseShapeByFormat 归一化的是 storage shape，因此按 storage format 解释；
// 但 TilingContextFaker 虽然对 gert::Tensor 同时调了 SetOriginFormat / SetStorageFormat
// （tests/ut/common/tiling_context_faker.cpp:86），闭源的 OpTilingContextBuilder 在构造
// ComputeNodeInfo 的 CompileTimeTensorDesc 时并未把 storage format 透传：
// 实测 origin=NCHW + storage=NCDHW + rank 5 时 tiling 失败，说明算子侧
// GetStorageFormat() 读到的是 origin（NCHW，只允许 rank 4）。
//
// 故上面所有 format 相关用例一律令 origin == storage，测的是"该 format + 该 rank 是否放行"，
// 不承诺覆盖"按哪个 format 分派"。后者已在真机侧确认，不必再在 UT 里较劲：
//   - GE 图模式：PreRunAfterBuild dump 中该节点 origin 与 storage 均为 NCDHW，两种读法
//     结果一致，通路不受影响；
//   - TTK kernel 模式（真机 Ascend950PR）：NDC1HWC0 用例实测选中
//     BN3DTrainingReduce_*_NDC1HWC0_high_performance 且 TilingKey=200000，NCDHW/NCHW 用例
//     选中 TilingKey=100000 —— 分派确实由 format 驱动且落在正确的一侧。
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// cRound 上限：C 很大时每轮暂存的通道结果数受 C_ROUND_CAP=512 约束，
// 且不超过该核实际负责的通道数 cPerCore。
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_c_round_capped_020)
{
    gert::StorageShape x_shape = {{2, 65536, 8}, {2, 65536, 8}};
    gert::StorageShape out_shape = {{65536}, {65536}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);

    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.key, TILINGKEY_DENSE_CHANNEL);
    EXPECT_EQ(r.td.numC, 65536U);
    EXPECT_EQ(r.td.cPerCore, 1024U); // 65536 / 64
    EXPECT_EQ(r.td.usedCoreNum, 64U);
    EXPECT_EQ(r.td.cRound, 512U); // 被 C_ROUND_CAP 截断
    EXPECT_LE(r.td.cRound, r.td.cPerCore);
}

// A == 1 时 R1 折进 R0：折叠后 numN 恒为 1、numR0 == 原 R1 * R0，归约集合不变。
// 折叠的动因是精度——R0 < VL_FP32 时 Kernel 每次向量加只有 R0 个 lane 有效，
// R0 == 1 即退化为 lane 0 上链长为 R1 的串行标量累加，R1 到百万量级会超 stat_rel_err
// 阈值（实测 (3900000,1,1) 的 Σx² 相对误差 2.7e-3 / 阈值 1.22e-4）。
TEST_F(BN3DTrainingReduceTilingTest, tiling_single_channel_r1_folded_into_r0_021)
{
    gert::StorageShape x_shape = {{3900000, 1, 1}, {3900000, 1, 1}};
    gert::StorageShape out_shape = {{1}, {1}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);

    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.key, TILINGKEY_SPLIT_REDUCE);
    EXPECT_EQ(r.td.numC, 1U);
    EXPECT_EQ(r.td.numN, 1U);        // R1 已折走
    EXPECT_EQ(r.td.numR0, 3900000U); // R0' = R1 * R0
    EXPECT_EQ(r.td.isSubR, 1U);      // 单行放不下 UB → 满宽 sub-R 分块
    EXPECT_GT(r.td.numChunks, 1U);
    EXPECT_EQ(r.td.cPerCore, 64U); // split key 下表示每通道核数
    EXPECT_EQ(r.td.usedCoreNum, 64U);
    EXPECT_EQ(r.td.numAccSlots, 1U);
    EXPECT_EQ(r.blockDim, 64U);
}

// C 不能整除核数时，每通道核数须向下对齐到 8，使连续 FP32 partial 搬运为 32B 整数倍。
// 否则 C=3、21 核/通道的 84B 尾块会把下一通道 partial 带入横向归约。
TEST_F(BN3DTrainingReduceTilingTest, tiling_split_reduce_aligns_partial_copy_021b)
{
    gert::StorageShape x_shape = {{2, 3, 1, 1, 655360}, {2, 3, 1, 1, 655360}};
    gert::StorageShape out_shape = {{3}, {3}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_BF16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);

    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.key, TILINGKEY_SPLIT_REDUCE);
    EXPECT_EQ(r.td.cPerCore, 16U); // floor(64 / 3) 再向下对齐到 8
    EXPECT_EQ(r.td.usedCoreNum, 48U);
    EXPECT_EQ(r.blockDim, 48U);
}

// A == 1 且 R0 本身够大时同样折叠，且不改变"单行放不下就走 sub-R"的既有判定。
TEST_F(BN3DTrainingReduceTilingTest, tiling_single_channel_fold_keeps_full_load_022)
{
    gert::StorageShape x_shape = {{8, 1, 64}, {8, 1, 64}};
    gert::StorageShape out_shape = {{1}, {1}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);

    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.numN, 1U);
    EXPECT_EQ(r.td.numR0, 512U); // 8 * 64
    EXPECT_EQ(r.td.isSubR, 0U);  // 512 个 fp32 一行放得下 → 仍走全载
    EXPECT_EQ(r.td.nTile, 1U);   // R1' == 1
}

// A > 1 时不得折叠：通道数据在 GM 上被其他通道隔开，折叠会把别的通道算进来。
TEST_F(BN3DTrainingReduceTilingTest, tiling_multi_channel_not_folded_023)
{
    gert::StorageShape x_shape = {{64, 3, 1}, {64, 3, 1}};
    gert::StorageShape out_shape = {{3}, {3}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);

    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.numC, 3U);
    EXPECT_EQ(r.td.numN, 64U); // 原样保留
    EXPECT_EQ(r.td.numR0, 1U);
}

// NDC1HWC0 且 C1 == 1 时也折叠，且 R0' 仍是 C0 的整数倍（lane ↔ c0 映射不被破坏）。
TEST_F(BN3DTrainingReduceTilingTest, tiling_ndc1hwc0_c1_1_folded_024)
{
    gert::StorageShape x_shape = {{4, 8, 1, 2, 2, 16}, {4, 8, 1, 2, 2, 16}};
    gert::StorageShape out_shape = {{1, 1, 1, 1, 1, 16}, {1, 1, 1, 1, 1, 16}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0);

    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.key, TILINGKEY_NDC1HWC0_CHANNEL);
    EXPECT_EQ(r.td.numN, 1U);
    EXPECT_EQ(r.td.numR0, 2048U);           // R1(4*8) * R0(2*2*16) = 32 * 64
    EXPECT_EQ(r.td.numR0 % r.td.numC0, 0U); // 折叠后仍是 C0 的整数倍
}

// ---------------------------------------------------------------------------
// 单维超 INT32_MAX：shape 的任一维都按 int64 处理，不得在 Host 侧被截成 32 位。
// 归一化后的 R1 / A / R0 与 TilingData 全部是 64 位；Kernel 侧 numN_ / numC_ /
// numR0_ 同为 uint64，GM 下标 r1*(A*R0) + a*R0 + r0 也在 uint64 里算。
// 32 位窄化分两类，均有 Host 侧看护：
//   * Kernel 的 r0Factor / numChunks / tailLen 与 full-R 的 r0Align / numR0：Host 校验
//     UINT32 及 DataCopy blockLen 字节上限；
//   * 硬件 ABI 的 DataCopyExtParams.srcStride：超 UINT32_MAX 时 Host 把 nTile 压到 1，
//     blockCount == 1 时该字段不参与寻址；blockCount(uint16) 则由 nTile<=65535 保证。
// 若哪天有人把 numN / numR0 改成 32 位，下面三条会立刻失败。
// ---------------------------------------------------------------------------

// N 超 INT32_MAX。取 C = 2 以避开 A == 1 时的 R1 折进 R0（见 021）。
TEST_F(BN3DTrainingReduceTilingTest, tiling_dim_n_over_int32_025)
{
    constexpr int64_t kNOverInt32 = 3000000000L; // > 2^31 - 1
    gert::StorageShape x_shape = {{kNOverInt32, 2, 1}, {kNOverInt32, 2, 1}};
    gert::StorageShape out_shape = {{2}, {2}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);

    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.key, TILINGKEY_DENSE_CHANNEL);
    EXPECT_EQ(r.td.numN, static_cast<uint64_t>(kNOverInt32)); // 未被截成 uint32
    EXPECT_EQ(r.td.numC, 2U);
    EXPECT_EQ(r.td.numR0, 1U);
    EXPECT_LE(r.td.nTile, 65535U); // blockCount 是 uint16
}

// R0 超 INT32_MAX：走 sub-R 分块，numR0 / tailLen 须原样保留 64 位，
// 且 numChunks = ceil(numR0 / r0Factor) 必须仍落在 uint32 内（Kernel 侧是 uint32）。
TEST_F(BN3DTrainingReduceTilingTest, tiling_dim_r0_over_int32_026)
{
    constexpr int64_t kR0OverInt32 = 3000000000L;
    gert::StorageShape x_shape = {{1, 2, kR0OverInt32}, {1, 2, kR0OverInt32}};
    gert::StorageShape out_shape = {{2}, {2}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);

    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.numC, 2U);
    EXPECT_EQ(r.td.numR0, static_cast<uint64_t>(kR0OverInt32));
    EXPECT_EQ(r.td.isSubR, 1U); // 单行放不下 UB
    ASSERT_GT(r.td.r0Factor, 0U);
    EXPECT_EQ(r.td.numChunks, static_cast<uint64_t>((kR0OverInt32 + r.td.r0Factor - 1) / r.td.r0Factor));
    EXPECT_LE(r.td.numChunks, 0xFFFFFFFFULL);
    EXPECT_EQ(r.td.numR0 - (r.td.numChunks - 1) * r.td.r0Factor, r.td.tailLen);
}

// NDC1HWC0 的 R1 = N * D：两维各自在 int32 内，乘积超出——须在 int64 里相乘。
TEST_F(BN3DTrainingReduceTilingTest, tiling_ndc1hwc0_r1_product_over_int32_027)
{
    constexpr int64_t kN = 2000000000L;
    gert::StorageShape x_shape = {{kN, 2, 2, 4, 4, 16}, {kN, 2, 2, 4, 4, 16}};
    gert::StorageShape out_shape = {{1, 1, 2, 1, 1, 16}, {1, 1, 2, 1, 1, 16}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0);

    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.key, TILINGKEY_NDC1HWC0_CHANNEL);
    EXPECT_EQ(r.td.numN, static_cast<uint64_t>(kN) * 2ULL); // N * D，超 uint32
    EXPECT_EQ(r.td.numC, 2U);                               // C1
    EXPECT_EQ(r.td.numR0, 256U);                            // H * W * C0
}

// 元素总数溢出 int64 时必须报错，绝不静默回绕（CheckedMul 看护）。
TEST_F(BN3DTrainingReduceTilingTest, tiling_total_elements_overflow_int64_failed_028)
{
    constexpr int64_t kHuge = 1L << 40;
    gert::StorageShape x_shape = {{kHuge, kHuge, kHuge}, {kHuge, kHuge, kHuge}};
    gert::StorageShape out_shape = {{kHuge}, {kHuge}};
    auto r = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);

    EXPECT_FALSE(r.ok);
}

// ---------------------------------------------------------------------------
// 空 Tensor 契约矩阵：C==0 时 rank 2~5、其他零轴/极大轴均走 no-work；C>0 时
// N/各空间轴任一为 0 均拒收。另覆盖 NCHW、内部 NDC1HWC0、rank1、负维以及
// 动态图 storage shape 左补 1 的还原路径。逐项枚举用于防止未来轴映射/早退顺序回归。
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_empty_tensor_contract_matrix_029)
{
    struct EmptyCase {
        const char* name;
        std::vector<int64_t> originDims;
        std::vector<int64_t> storageDims;
        std::vector<int64_t> outputDims;
        ge::Format format;
        bool expectNoWork;
    };

    constexpr int64_t kHuge = 1LL << 40;
    const std::vector<EmptyCase> cases = {
        {"ncdhw_rank2_c0", {2, 0}, {2, 0}, {0}, ge::FORMAT_NCDHW, true},
        {"ncdhw_rank3_c0", {2, 0, 7}, {2, 0, 7}, {0}, ge::FORMAT_NCDHW, true},
        {"ncdhw_rank4_c0", {2, 0, 7, 8}, {2, 0, 7, 8}, {0}, ge::FORMAT_NCDHW, true},
        {"ncdhw_rank5_c0", {2, 0, 7, 8, 9}, {2, 0, 7, 8, 9}, {0}, ge::FORMAT_NCDHW, true},
        {"ncdhw_all_zero", {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0}, ge::FORMAT_NCDHW, true},
        {"ncdhw_c0_short_circuits_products",
         {kHuge, 0, kHuge, kHuge, kHuge},
         {kHuge, 0, kHuge, kHuge, kHuge},
         {0},
         ge::FORMAT_NCDHW,
         true},
        {"ncdhw_n0", {0, 3, 4, 5, 6}, {0, 3, 4, 5, 6}, {3}, ge::FORMAT_NCDHW, false},
        {"ncdhw_d0", {2, 3, 0, 5, 6}, {2, 3, 0, 5, 6}, {3}, ge::FORMAT_NCDHW, false},
        {"ncdhw_h0", {2, 3, 4, 0, 6}, {2, 3, 4, 0, 6}, {3}, ge::FORMAT_NCDHW, false},
        {"ncdhw_w0", {2, 3, 4, 5, 0}, {2, 3, 4, 5, 0}, {3}, ge::FORMAT_NCDHW, false},
        {"ncdhw_multi_zero", {0, 3, 0, 0, 0}, {0, 3, 0, 0, 0}, {3}, ge::FORMAT_NCDHW, false},
        {"nchw_h0", {2, 3, 0, 5}, {2, 3, 0, 5}, {3}, ge::FORMAT_NCHW, false},
        {"nchw_w0", {2, 3, 4, 0}, {2, 3, 4, 0}, {3}, ge::FORMAT_NCHW, false},
        {"ndc1hwc0_n0", {0, 2, 3, 4, 5, 16}, {0, 2, 3, 4, 5, 16}, {3}, ge::FORMAT_NDC1HWC0, false},
        {"ndc1hwc0_d0", {2, 0, 3, 4, 5, 16}, {2, 0, 3, 4, 5, 16}, {3}, ge::FORMAT_NDC1HWC0, false},
        {"ndc1hwc0_h0", {2, 2, 3, 0, 5, 16}, {2, 2, 3, 0, 5, 16}, {3}, ge::FORMAT_NDC1HWC0, false},
        {"ndc1hwc0_w0", {2, 2, 3, 4, 0, 16}, {2, 2, 3, 4, 0, 16}, {3}, ge::FORMAT_NDC1HWC0, false},
        {"ndc1hwc0_c1_zero_with_reduction_zeros",
         {0, 0, 0, 0, 0, 16},
         {0, 0, 0, 0, 0, 16},
         {1, 1, 0, 1, 1, 16},
         ge::FORMAT_NDC1HWC0,
         true},
        {"ndc1hwc0_c0_zero_with_reduction_zeros",
         {0, 0, 3, 0, 0, 0},
         {0, 0, 3, 0, 0, 0},
         {1, 1, 3, 1, 1, 0},
         ge::FORMAT_NDC1HWC0,
         true},
        {"ncdhw_rank1_zero", {0}, {0}, {0}, ge::FORMAT_NCDHW, false},
        {"ncdhw_c0_but_negative_axis", {2, 0, -1}, {2, 0, -1}, {0}, ge::FORMAT_NCDHW, false},
        {"left_padded_storage_c0", {2, 0, 4, 4, 8}, {1, 1, 2, 0, 4, 4, 8}, {0}, ge::FORMAT_NCDHW, true},
        {"left_pad_non_one_not_stripped", {2, 0, 4, 4, 8}, {1, 2, 2, 0, 4, 4, 8}, {0}, ge::FORMAT_NCDHW, false},
        {"left_pad_tail_mismatch_not_stripped", {2, 0, 4, 4, 8}, {1, 1, 2, 0, 4, 4, 9}, {0}, ge::FORMAT_NCDHW, false},
    };

    for (const auto& testCase : cases) {
        SCOPED_TRACE(testCase.name);
        auto xShape = MakeStorageShape(testCase.originDims, testCase.storageDims);
        auto outShape = MakeStorageShape(testCase.outputDims);
        const auto result = RunTiling(xShape, outShape, ge::DT_FLOAT, testCase.format, testCase.format);

        EXPECT_EQ(result.ok, testCase.expectNoWork);
        if (result.ok) {
            EXPECT_EQ(result.td.numC, 0U);
            EXPECT_EQ(result.td.usedCoreNum, 0U);
            EXPECT_EQ(result.blockDim, 1U);
        }
    }
}

// ---------------------------------------------------------------------------
// 极端 shape 的派生算术与平台约束必须可预测：越过 int64/Kernel 计数边界时明确失败；
// 跨 N stride 超出硬件字段时安全回退单行；N 极大时槽数启发式仍为常数级，不能线性求 sqrt。
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceTilingTest, tiling_extreme_shape_arithmetic_guards_030)
{
    struct ExtremeCase {
        const char* name;
        std::vector<int64_t> dims;
        ge::Format format;
        bool expectSuccess;
        bool expectSingleRow;
    };

    const std::vector<ExtremeCase> cases = {
        {"row_double_buffer_bytes_overflow", {1, 2, 1LL << 61}, ge::FORMAT_NCDHW, false, false},
        {"r0_ceil_align_overflow", {1, 1, std::numeric_limits<int64_t>::max()}, ge::FORMAT_NCDHW, false, false},
        {"subr_num_chunks_over_uint32", {1, 2, 1LL << 48}, ge::FORMAT_NCDHW, false, false},
        {"src_stride_uint32_boundary", {8, 1073741824LL, 1}, ge::FORMAT_NCDHW, true, false},
        {"src_stride_over_uint32", {8, 1073741825LL, 1}, ge::FORMAT_NCDHW, true, true},
        // N 必须大于 1，才能证明 nTile 原本可取多行、确由 stride 溢出分支回退到 1。
        {"src_stride_bytes_over_int64", {2, (1LL << 61) + 1, 1}, ge::FORMAT_NCDHW, true, true},
        {"huge_n_bounded_slot_heuristic", {1LL << 61, 2, 1}, ge::FORMAT_NCDHW, true, false},
        {"spatial_product_over_int64", {1, 2, 1LL << 40, 1LL << 40}, ge::FORMAT_NCDHW, false, false},
        {"total_elements_second_multiply_overflow", {1LL << 30, 1LL << 30, 8}, ge::FORMAT_NCDHW, false, false},
        // C0 必须先满足 C0 | VL；极端但不可整除的 C0 在字节乘法前即应被拒收。
        {"ndc1hwc0_extreme_c0_not_divide_vl", {1, 1, 1, 1, 1, 1LL << 62}, ge::FORMAT_NDC1HWC0, false, false},
    };

    for (const auto& testCase : cases) {
        SCOPED_TRACE(testCase.name);
        auto xShape = MakeStorageShape(testCase.dims);
        auto outShape = MakeStorageShape(std::vector<int64_t>{testCase.dims.size() > 1 ? testCase.dims[1] : 0});
        const auto result = RunTiling(xShape, outShape, ge::DT_FLOAT, testCase.format, testCase.format);

        EXPECT_EQ(result.ok, testCase.expectSuccess);
        if (result.ok) {
            EXPECT_LE(result.td.nTile, 65535U);
            if (testCase.expectSingleRow) {
                EXPECT_EQ(result.td.nTile, 1U);
            } else {
                EXPECT_GT(result.td.nTile, 1U);
            }
        }
    }
}
