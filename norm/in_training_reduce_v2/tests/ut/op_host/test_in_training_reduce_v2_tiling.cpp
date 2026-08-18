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
#include "../../../op_kernel/arch35/in_training_reduce_v2_tiling_data.h"
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
// td_out 非空时，额外把下发的 TilingData 拷回给调用方，供 sub-R 用例校验 rFactor / numChunks。
static uint64_t RunTiling(gert::StorageShape& x_shape, gert::StorageShape& out_shape, ge::DataType dtype,
                          ge::Format fmt, INTrainingReduceV2ARFullReduceTilingData* td_out = nullptr)
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
    if (td_out != nullptr) {
        auto raw = tiling_context->GetRawTilingData();
        if (raw == nullptr || raw->GetDataSize() < sizeof(*td_out)) {
            return UINT64_MAX;
        }
        *td_out = *reinterpret_cast<const INTrainingReduceV2ARFullReduceTilingData*>(raw->GetData());
    }
    return tiling_context->GetTilingKey();
}

// 复算 Kernel InitSubR() 的 UB 占用，用于在 UT 里守住「Tiling 下发的参数确实塞得进 UB」。
// 公式与 op_host 的 CalcSubRUbBytes / op_kernel 的 InitSubR 三方一致。
static uint64_t SubRUbBytes(const INTrainingReduceV2ARFullReduceTilingData& td, uint64_t elemSize)
{
    constexpr uint64_t kBlockSize = 32;   // platform::GetUbBlockSize()
    constexpr uint64_t kVlFp32 = 64;      // 256B vector length / sizeof(float)
    constexpr uint64_t kDoubleBuffer = 2; // DOUBLE_BUFFER_NUM
    auto ceilAlign = [](uint64_t v, uint64_t a) { return (v + a - 1) / a * a; };
    uint64_t slots = td.chunksPerGroup + (td.numGroups > 1 ? 1U : 0U);
    uint64_t inBytes = ceilAlign(td.rFactor * elemSize, kBlockSize) * kDoubleBuffer;
    uint64_t partialBytes = ceilAlign(slots, kVlFp32) * sizeof(float) * 2;
    uint64_t outBytes = ceilAlign(sizeof(float), kBlockSize) * kDoubleBuffer * 2;
    return inBytes + partialBytes + outBytes;
}

constexpr uint64_t kUsableUb = 245760 - 512; // UB_SIZE - RESERVE_FOR_ALIGN

// sub-R 一组自洽性断言：分块 / 分组参数彼此对得上，且 Kernel 按这组参数申请的 UB 不越界。
// 任何 R（含超 uint32）都必须满足，用它取代逐用例手写断言。
static void ExpectSubRSelfConsistent(const INTrainingReduceV2ARFullReduceTilingData& td, uint64_t elemSize)
{
    constexpr uint64_t kVlFp32 = 64;
    ASSERT_EQ(td.isSubRTiling, 1U);
    ASSERT_EQ(td.rFactor % kVlFp32, 0U);
    ASSERT_GT(td.chunksPerGroup, 0U);
    ASSERT_GT(td.numGroups, 0U);
    ASSERT_EQ(td.numChunks, (td.numR + td.rFactor - 1) / td.rFactor);
    ASSERT_EQ(td.tailLen, td.numR - (td.numChunks - 1) * td.rFactor);
    ASSERT_EQ(td.numGroups, (td.numChunks + td.chunksPerGroup - 1) / td.chunksPerGroup);
    ASSERT_EQ(td.tailChunks, td.numChunks - (td.numGroups - 1) * td.chunksPerGroup);
    ASSERT_LE(td.tailChunks, td.chunksPerGroup);
    ASSERT_LE(SubRUbBytes(td, elemSize), kUsableUb);
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

// ---------------------------------------------------------------------------
// 空 tensor：规约（空间）轴为 0 时 Tiling 明确失败。
// 本迭代不含 REDUCE_EMPTY 模板，图原型 / README 均已声明不支持空 tensor；
// 这条用例把「不支持」钉成可回归的行为，防止后续默默变成越界下发。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_nchw_r0_empty_rejected_012)
{
    gert::StorageShape x_shape = {{2, 3, 0, 4}, {2, 3, 0, 4}};
    gert::StorageShape out_shape = {{2, 3, 1, 1}, {2, 3, 1, 1}};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCHW);
    ASSERT_EQ(key, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// sub-R 联合求解：R 大到部分和缓存撑破 usable/8 的初始预留时，rFactor 必须回缩，
// 使 Kernel 侧实际 UB 申请仍落在 UB 内。
// fp32 ND [1,1,104439809]：按老实现 rFactor=26752 / numChunks=3905 → 申请 245888B，
// 超 245760B 的 UB 128B；联合求解后 rFactor 回缩、总占用回到 usable 以内。
//
// ⚠ 这里的 245760 是本 UT compile_info 里那份 UB_SIZE（仓内 arch35 UT 沿用的旧模板，
//   activation/ norm/ 下几十个算子都是这个值），**不是 Ascend950 的真实 UB**——
//   平台 ini 里 ub_size=253952。所以本用例验的是"给定 UB 下联合求解会回缩"这个逻辑，
//   而不是真实芯片上的越界起点（真实起点：fp32 R=109707265、fp16 R=219668481）。
//   UB_SIZE 是全仓级别的清理项，不在本算子范围内改。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_sub_r_partial_buf_fits_ub_013)
{
    gert::StorageShape x_shape = {{1, 1, 104439809}, {1, 1, 104439809}};
    gert::StorageShape out_shape = {{1, 1, 1}, {1, 1, 1}};
    INTrainingReduceV2ARFullReduceTilingData td{};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_ND, &td);
    ASSERT_EQ(key, 200000U);
    ASSERT_EQ(td.numR, 104439809U);
    ExpectSubRSelfConsistent(td, sizeof(float));
    // 快路径：单次全折叠塞得进 UB，不分组 —— 参数与分组改造前逐位一致
    ASSERT_EQ(td.numGroups, 1U);
    ASSERT_EQ(td.chunksPerGroup, td.numChunks);
}

// ---------------------------------------------------------------------------
// 分组折叠：R 大到单组放不下时，改由多组折叠承接，而不是拒绝。
// fp32 R=2e9：快路径无解（最优点 rFactor≈sqrt(R) 时总占用远超 UB），落到分组路径。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_sub_r_grouped_fold_2e9_014)
{
    gert::StorageShape x_shape = {{1, 1, 2000000000}, {1, 1, 2000000000}};
    gert::StorageShape out_shape = {{1, 1, 1}, {1, 1, 1}};
    INTrainingReduceV2ARFullReduceTilingData td{};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_ND, &td);
    ASSERT_EQ(key, 200000U);
    ASSERT_EQ(td.numR, 2000000000U);
    ExpectSubRSelfConsistent(td, sizeof(float));
    ASSERT_GT(td.numGroups, 1U); // 必须真的走了分组
}

// ---------------------------------------------------------------------------
// sub-R 联合求解的**迭代轮数**：R 逼近 UB 容量上限时，rFactor 已被部分和缓存挤得很小，
// 每轮只能再缩一点点，收敛轮数急剧上升（这里需要 20 轮）。SUB_R_SOLVE_MAX_ITER 原为 8，
// 落在这一段的 R 明明有可行解却会被判成 "cannot fit UB" 并拒绝下发 —— 失败方向安全，
// 但仍是错判。本用例锁死"有解就必须求出来"，上限调回 8 时它会失败。
// fp32 ND [1,1,233500000]：收敛解 rFactor=15936 / numChunks=14653 → 244864B ≤ 245248B。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_sub_r_solve_needs_many_iters_016)
{
    gert::StorageShape x_shape = {{1, 1, 233500000}, {1, 1, 233500000}};
    gert::StorageShape out_shape = {{1, 1, 1}, {1, 1, 1}};
    INTrainingReduceV2ARFullReduceTilingData td{};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_ND, &td);
    ASSERT_EQ(key, 200000U); // 不是 UINT64_MAX —— 有解就不能拒绝
    ASSERT_EQ(td.numR, 233500000U);
    ExpectSubRSelfConsistent(td, sizeof(float));
    ASSERT_EQ(td.numGroups, 1U); // 仍在快路径内，不应被分组抢走
}

// ---------------------------------------------------------------------------
// R 超 UINT32_MAX：分组折叠落地后 numR 全程 uint64，不再有 uint32 上限。
// 这条正是上游反馈（changwei#26）里 InTrainingReduceV2_L1_upboundary_highpre_047
// 用的 R=2^32；改造前被 CheckSubRNarrowable() 拒绝，现在必须能出 tiling。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_sub_r_r_over_uint32_015)
{
    gert::StorageShape x_shape = {{1, 1, 4294967296L}, {1, 1, 4294967296L}};
    gert::StorageShape out_shape = {{1, 1, 1}, {1, 1, 1}};
    INTrainingReduceV2ARFullReduceTilingData td{};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_ND, &td);
    ASSERT_EQ(key, 200000U);
    ASSERT_EQ(td.numR, 4294967296UL);
    ExpectSubRSelfConsistent(td, sizeof(float));
    ASSERT_GT(td.numGroups, 1U);
}

// ---------------------------------------------------------------------------
// R 远超 UINT32_MAX（2^34，fp16）：验证上限确实解除而不是抬高了一档。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_sub_r_r_far_over_uint32_fp16_017)
{
    gert::StorageShape x_shape = {{1, 1, 17179869184L}, {1, 1, 17179869184L}};
    gert::StorageShape out_shape = {{1, 1, 1}, {1, 1, 1}};
    INTrainingReduceV2ARFullReduceTilingData td{};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT16, ge::FORMAT_ND, &td);
    ASSERT_EQ(key, 200000U);
    ASSERT_EQ(td.numR, 17179869184UL);
    ExpectSubRSelfConsistent(td, sizeof(uint16_t));
    ASSERT_GT(td.numGroups, 1U);
}

// ---------------------------------------------------------------------------
// sub-R 路径的 N*C 容量闸：N*C 溢出 uint32 须拒绝。
// 生效范围仅限 sub-R —— CheckSubRNarrowable() 只被 DoSubRTiling() 调用，R 小走
// full-load 路径时 N*C 不受任何约束（故本用例的 R 必须取大值才能触发）。
// 这不是 Kernel 收窄造成的：TilingData 与两条 Kernel 路径的 numN/numC/numR 现已
// 全部是 64 位（sub_r.h 的 uint64_t、ar_full_reduce.h 的 int64_t）。保留它是因为
// 撞闸需同时满足 R > 2.5e8 且 N*C > 2^32，即总元素数 > 1e18，实际不可达；放行只会
// 得到一条既跑不到也无法验证的路径。
// 对比 canndev：canndev 对 N*C 与总元素数均无上限（para_check.check_shape 的
// max_size=SHAPE_SIZE_LIMIT 是死参数，函数体从未使用；实际只强制 rank<=8、单维<=2^63-1）。
// N=65536, C=65536 → N*C=2^32 > UINT32_MAX；R 取大值以确保走 sub-R 路径。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_sub_r_nc_over_uint32_rejected_018)
{
    gert::StorageShape x_shape = {{65536, 65536, 100000}, {65536, 65536, 100000}};
    gert::StorageShape out_shape = {{65536, 65536, 1}, {65536, 65536, 1}};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_ND);
    ASSERT_EQ(key, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// N*C 溢出 uint32 但 R 很小：走 full-load 路径，必须被接受。
// 与 018 成对，共同界定容量闸的生效范围 —— CheckSubRNarrowable() 只被
// DoSubRTiling() 调用，而 IsCapable() 仅在 cInner < 1（单行 R 全载放不下 UB）
// 时才进 DoSubRTiling()。本例 N=65536, C=65536（N*C=2^32 > UINT32_MAX）、R=4：
//   rAlign = CeilAlign(4*4, 32)/4 = 8
//   cInner = (245760-512) / (8*4*2 + 4*2*2 + 32*2) = 245248/144 = 1703 >= 1
// 故走 full-load，N*C 不受任何约束。tiling 全程 64 位，无截断。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_nc_over_uint32_small_r_accepted_019)
{
    gert::StorageShape x_shape = {{65536, 65536, 4}, {65536, 65536, 4}};
    gert::StorageShape out_shape = {{65536, 65536, 1}, {65536, 65536, 1}};
    INTrainingReduceV2ARFullReduceTilingData td{};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_ND, &td);
    ASSERT_NE(key, UINT64_MAX);
    // 走 full-load，不是 sub-R
    ASSERT_EQ(td.isSubRTiling, 0U);
    // N / C / R 原样下发，未被收窄
    ASSERT_EQ(td.numN, 65536UL);
    ASSERT_EQ(td.numC, 65536UL);
    ASSERT_EQ(td.numR, 4UL);
    // N*C 用 64 位乘出来仍然正确（32 位会回绕成 0）
    ASSERT_EQ(td.numN * td.numC, 4294967296UL);
}
