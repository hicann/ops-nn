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
 * \file test_acts_ulq_input_grad_tiling.cpp
 * \brief ActsULQInputGrad Host 侧 Tiling 单元测试（arch35 / DAV_3510）
 *
 * 覆盖：
 *   - 4 dtype 组合（K1 fp16+bool / K2 fp16+fp16 / K3 fp32+bool / K4 fp32+fp32），
 *     各 dtype 走不同 bufferDivisor → ubFormer 分支，逐字段验证多核 + UB 双切分计算。
 *   - 多核切分（大 shape 打满 64 核）+ UB 双切分（首/尾块多次 UB 循环）。
 *   - 多核 + 首尾块 blockFormer/blockTail 不等。
 *   - rank-0 标量 [] → EnsureNotScalar → {1}。
 *   - 空 Tensor（含 0 维）短路分支。
 *   - 异常：三输入 shape 不一致 / 混合浮点非法组合 / 两 mask dtype 不一致 /
 *           y_grad dtype 非法 / mask dtype 非法 / x_grad dtype 与 y_grad 不一致。
 *
 * 说明（与前向 acts_ulq 框架的差异）：
 *   - 本算子 Host Tiling 的 coreNum/ubSize 在运行时经 PlatformAscendC 从平台信息读取
 *     （faker 注入 CORE_NUM=64 / UB_SIZE=262144），不经 CompileInfo；
 *   - TilingKey 整数由 ASCENDC_TPL_SEL_PARAM 模板编码生成，不像前向手工赋 0/1/2/3。
 *   因此正向用例采用「读回 TilingData struct 字段逐一 EXPECT」的字段级断言（本框架允许，
 *   见 tiling_case_executor.h::ExecuteTiling），期望值全部依据
 *   op_host/arch35/acts_ulq_input_grad_tiling_arch35.cpp 实际公式手工推导，
 *   不硬编码模板生成的 TilingKey 整数；异常用例沿用前向 ExecuteTestCase(GRAPH_FAILED) 写法。
 *
 *   构造 TilingContextPara 时必须传入非空 compileInfo 指针（否则 BuildTilingContext 失败、
 *   context 为空 → 调用 tiling func 崩溃）。本算子 tiling func 不读取 CompileInfo，
 *   故传入一个哑元 CompileInfo 地址即可（与前向 acts_ulq 传 &ci 同理）。
 */

#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_kernel/arch35/acts_ulq_input_grad_tiling_data.h"

using namespace std;
using namespace ge;

namespace {
constexpr char kOpType[] = "ActsULQInputGrad";

// 哑元 CompileInfo：本算子 tiling 只用 PlatformInfo，不读 CompileInfo，
// 但 BuildTilingContext 要求 compileInfo 非空，故传入其地址。
struct ActsULQInputGradCompileInfoStub {};
ActsULQInputGradCompileInfoStub g_compileInfoStub;

// 读回 TilingData 并逐字段断言（正向用例统一入口）
void ExpectTilingFields(const gert::TilingContextPara& ctx, int64_t dim0, int32_t coreNum, int64_t blockFormer,
                        int64_t blockNum, int64_t ubFormer, int64_t ubLoopOfFormerBlock, int64_t ubTailOfFormerBlock,
                        int64_t ubLoopOfTailBlock, int64_t ubTailOfTailBlock)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));
    ASSERT_GE(info.tilingDataSize, sizeof(ActsULQInputGradTilingData));
    const auto* td = reinterpret_cast<const ActsULQInputGradTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->dim0, dim0);
    EXPECT_EQ(td->coreNum, coreNum);
    EXPECT_EQ(td->blockFormer, blockFormer);
    EXPECT_EQ(td->blockNum, blockNum);
    EXPECT_EQ(td->ubFormer, ubFormer);
    EXPECT_EQ(td->ubLoopOfFormerBlock, ubLoopOfFormerBlock);
    EXPECT_EQ(td->ubTailOfFormerBlock, ubTailOfFormerBlock);
    EXPECT_EQ(td->ubLoopOfTailBlock, ubLoopOfTailBlock);
    EXPECT_EQ(td->ubTailOfTailBlock, ubTailOfTailBlock);
    // SetBlockDim(usedCoreNum=blockNum)；workspace 声明 1 槽且为 0
    EXPECT_EQ(static_cast<int64_t>(info.blockNum), blockNum);
    ASSERT_EQ(info.workspaceSizes.size(), 1u);
    EXPECT_EQ(info.workspaceSizes[0], 0u);
}
} // namespace

class ActsUlqInputGradTilingTest : public testing::Test {};

// -------------------- 4 dtype 组合正常用例（K1-K4） --------------------

// K1: y_grad=fp16, mask=bool，shape [128]
TEST_F(ActsUlqInputGradTilingTest, k1_fp16_bool_1d_128)
{
    gert::TilingContextPara ctx(kOpType,
                                {{{{{128}, {128}}, DT_FLOAT16, FORMAT_ND},
                                  {{{128}, {128}}, DT_BOOL, FORMAT_ND},
                                  {{{128}, {128}}, DT_BOOL, FORMAT_ND}}},
                                {{{{{128}, {128}}, DT_FLOAT16, FORMAT_ND}}}, &g_compileInfoStub);
    // totalIdx=128; elemBits=(2*2+2*1)*8=48; coreNum=CeilDiv(128*48,32768)=1
    // blockFormer=CeilAlign(128,512)=512; blockNum=1; ubFormer(K1)=20992
    // ubLoopFmr=1, ubTailFmr=512; blockTail=128, ubLoopTail=1, ubTailTail=128
    ExpectTilingFields(ctx, 128, 1, 512, 1, 20992, 1, 512, 1, 128);
}

// K2: y_grad=fp16, mask=fp16，shape [4,8]
TEST_F(ActsUlqInputGradTilingTest, k2_fp16_fp16_2d_4x8)
{
    gert::TilingContextPara ctx(kOpType,
                                {{{{{4, 8}, {4, 8}}, DT_FLOAT16, FORMAT_ND},
                                  {{{4, 8}, {4, 8}}, DT_FLOAT16, FORMAT_ND},
                                  {{{4, 8}, {4, 8}}, DT_FLOAT16, FORMAT_ND}}},
                                {{{{{4, 8}, {4, 8}}, DT_FLOAT16, FORMAT_ND}}}, &g_compileInfoStub);
    // totalIdx=32; elemBits=(4+4)*8=64; coreNum=1; blockFormer=512; blockNum=1
    // ubFormer(K2)=15872; ubLoopFmr=1,ubTailFmr=512; blockTail=32,ubLoopTail=1,ubTailTail=32
    ExpectTilingFields(ctx, 32, 1, 512, 1, 15872, 1, 512, 1, 32);
}

// K3: y_grad=fp32, mask=bool，shape [32,3,5,5]
TEST_F(ActsUlqInputGradTilingTest, k3_fp32_bool_4d_32x3x5x5)
{
    gert::TilingContextPara ctx(kOpType,
                                {{{{{32, 3, 5, 5}, {32, 3, 5, 5}}, DT_FLOAT, FORMAT_ND},
                                  {{{32, 3, 5, 5}, {32, 3, 5, 5}}, DT_BOOL, FORMAT_ND},
                                  {{{32, 3, 5, 5}, {32, 3, 5, 5}}, DT_BOOL, FORMAT_ND}}},
                                {{{{{32, 3, 5, 5}, {32, 3, 5, 5}}, DT_FLOAT, FORMAT_ND}}}, &g_compileInfoStub);
    // totalIdx=2400; elemBits=(2*4+2*1)*8=80; coreNum=CeilDiv(2400*80,32768)=6
    // blockFormer=CeilAlign(CeilDiv(2400,6)=400,512)=512; blockNum=CeilDiv(2400,512)=5
    // ubFormer(K3)=12544; ubLoopFmr=1,ubTailFmr=512
    // blockTail=2400-4*512=352; ubLoopTail=1,ubTailTail=352
    ExpectTilingFields(ctx, 2400, 6, 512, 5, 12544, 1, 512, 1, 352);
}

// K4: y_grad=fp32, mask=fp32，shape [32,3,5,5]
TEST_F(ActsUlqInputGradTilingTest, k4_fp32_fp32_4d_32x3x5x5)
{
    gert::TilingContextPara ctx(kOpType,
                                {{{{{32, 3, 5, 5}, {32, 3, 5, 5}}, DT_FLOAT, FORMAT_ND},
                                  {{{32, 3, 5, 5}, {32, 3, 5, 5}}, DT_FLOAT, FORMAT_ND},
                                  {{{32, 3, 5, 5}, {32, 3, 5, 5}}, DT_FLOAT, FORMAT_ND}}},
                                {{{{{32, 3, 5, 5}, {32, 3, 5, 5}}, DT_FLOAT, FORMAT_ND}}}, &g_compileInfoStub);
    // totalIdx=2400; elemBits=(8+8)*8=128; coreNum=CeilDiv(2400*128,32768)=10
    // blockFormer=CeilAlign(CeilDiv(2400,10)=240,512)=512; blockNum=5
    // ubFormer(K4)=7936; ubLoopFmr=1,ubTailFmr=512; blockTail=352,ubLoopTail=1,ubTailTail=352
    ExpectTilingFields(ctx, 2400, 10, 512, 5, 7936, 1, 512, 1, 352);
}

// -------------------- 多核 + UB 双切分（首/尾块多次 UB 循环） --------------------

// K4 大 shape [4096,1024]，打满 64 核 + 每块 9 次 UB 循环
TEST_F(ActsUlqInputGradTilingTest, k4_multicore_ub_multiloop)
{
    gert::TilingContextPara ctx(kOpType,
                                {{{{{4096, 1024}, {4096, 1024}}, DT_FLOAT, FORMAT_ND},
                                  {{{4096, 1024}, {4096, 1024}}, DT_FLOAT, FORMAT_ND},
                                  {{{4096, 1024}, {4096, 1024}}, DT_FLOAT, FORMAT_ND}}},
                                {{{{{4096, 1024}, {4096, 1024}}, DT_FLOAT, FORMAT_ND}}}, &g_compileInfoStub);
    // totalIdx=4194304; elemBits=128; coreNum=CeilDiv(4194304*128,32768)=16384 → 截断 64
    // blockFormer=CeilAlign(CeilDiv(4194304,64)=65536,512)=65536; blockNum=64
    // ubFormer(K4)=7936; ubLoopFmr=CeilDiv(65536,7936)=9; ubTailFmr=65536-8*7936=2048
    // blockTail=4194304-63*65536=65536; ubLoopTail=9; ubTailTail=2048
    ExpectTilingFields(ctx, 4194304, 64, 65536, 64, 7936, 9, 2048, 9, 2048);
}

// K4 多核 + 首尾块大小不等（blockFormer != blockTail）
TEST_F(ActsUlqInputGradTilingTest, k4_multicore_former_tail_diff)
{
    gert::TilingContextPara ctx(kOpType,
                                {{{{{100000}, {100000}}, DT_FLOAT, FORMAT_ND},
                                  {{{100000}, {100000}}, DT_FLOAT, FORMAT_ND},
                                  {{{100000}, {100000}}, DT_FLOAT, FORMAT_ND}}},
                                {{{{{100000}, {100000}}, DT_FLOAT, FORMAT_ND}}}, &g_compileInfoStub);
    // totalIdx=100000; elemBits=128; coreNum=CeilDiv(100000*128,32768)=391 → 截断 64
    // blockFormer=CeilAlign(CeilDiv(100000,64)=1563,512)=2048; blockNum=CeilDiv(100000,2048)=49
    // ubFormer(K4)=7936; ubLoopFmr=1,ubTailFmr=2048
    // blockTail=100000-48*2048=1696; ubLoopTail=1,ubTailTail=1696
    ExpectTilingFields(ctx, 100000, 64, 2048, 49, 7936, 1, 2048, 1, 1696);
}

// -------------------- rank-0 标量 → EnsureNotScalar → {1} --------------------

TEST_F(ActsUlqInputGradTilingTest, scalar_rank0_ensure_not_scalar)
{
    gert::TilingContextPara ctx(
        kOpType, {{{{{}, {}}, DT_FLOAT16, FORMAT_ND}, {{{}, {}}, DT_BOOL, FORMAT_ND}, {{{}, {}}, DT_BOOL, FORMAT_ND}}},
        {{{{{}, {}}, DT_FLOAT16, FORMAT_ND}}}, &g_compileInfoStub);
    // 标量 [] → {1}: totalIdx=1; elemBits=48; coreNum=1; blockFormer=512; blockNum=1
    // ubFormer(K1)=20992; ubLoopFmr=1,ubTailFmr=512; blockTail=1,ubLoopTail=1,ubTailTail=1
    ExpectTilingFields(ctx, 1, 1, 512, 1, 20992, 1, 512, 1, 1);
}

// -------------------- 空 Tensor 短路分支（dim0=0） --------------------

TEST_F(ActsUlqInputGradTilingTest, empty_tensor_shortcut)
{
    gert::TilingContextPara ctx(kOpType,
                                {{{{{2, 0}, {2, 0}}, DT_FLOAT, FORMAT_ND},
                                  {{{2, 0}, {2, 0}}, DT_FLOAT, FORMAT_ND},
                                  {{{2, 0}, {2, 0}}, DT_FLOAT, FORMAT_ND}}},
                                {{{{{2, 0}, {2, 0}}, DT_FLOAT, FORMAT_ND}}}, &g_compileInfoStub);
    // totalIdx=0 → 短路：SetBlockDim(1)，TilingData 除 ubFormer 外全 0（memset），SUCCESS。
    // ubFormer 显式设为 1（非 0）：kernel Process 计算 loopCount=(blockLength_+ubFormer_-1)/ubFormer_，
    // 若 ubFormer_=0 会除零崩溃；设 1 时 blockLength_=0 → loopCount=0，循环不执行，安全返回。
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));
    const auto* td = reinterpret_cast<const ActsULQInputGradTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->dim0, 0);
    EXPECT_EQ(td->coreNum, 0);
    EXPECT_EQ(td->blockFormer, 0);
    EXPECT_EQ(td->blockNum, 0);
    EXPECT_EQ(td->ubFormer, 1); // 防除零：空 Tensor 短路显式设 ubFormer=1
    EXPECT_EQ(info.blockNum, 1u);
}

// -------------------- 异常用例（GRAPH_FAILED） --------------------

// 三输入 shape 不一致
TEST_F(ActsUlqInputGradTilingTest, err_shape_mismatch)
{
    gert::TilingContextPara ctx(kOpType,
                                {{{{{128}, {128}}, DT_FLOAT16, FORMAT_ND},
                                  {{{64}, {64}}, DT_BOOL, FORMAT_ND},
                                  {{{128}, {128}}, DT_BOOL, FORMAT_ND}}},
                                {{{{{128}, {128}}, DT_FLOAT16, FORMAT_ND}}}, &g_compileInfoStub);
    ExecuteTestCase(ctx, ge::GRAPH_FAILED);
}

// 混合浮点非法组合：y_grad=fp16, mask=fp32（v2.0 已移除）
TEST_F(ActsUlqInputGradTilingTest, err_mixed_float_combo)
{
    gert::TilingContextPara ctx(kOpType,
                                {{{{{128}, {128}}, DT_FLOAT16, FORMAT_ND},
                                  {{{128}, {128}}, DT_FLOAT, FORMAT_ND},
                                  {{{128}, {128}}, DT_FLOAT, FORMAT_ND}}},
                                {{{{{128}, {128}}, DT_FLOAT16, FORMAT_ND}}}, &g_compileInfoStub);
    ExecuteTestCase(ctx, ge::GRAPH_FAILED);
}

// 两 mask dtype 不一致：min=bool, max=fp16
TEST_F(ActsUlqInputGradTilingTest, err_two_masks_dtype_mismatch)
{
    gert::TilingContextPara ctx(kOpType,
                                {{{{{128}, {128}}, DT_FLOAT16, FORMAT_ND},
                                  {{{128}, {128}}, DT_BOOL, FORMAT_ND},
                                  {{{128}, {128}}, DT_FLOAT16, FORMAT_ND}}},
                                {{{{{128}, {128}}, DT_FLOAT16, FORMAT_ND}}}, &g_compileInfoStub);
    ExecuteTestCase(ctx, ge::GRAPH_FAILED);
}

// y_grad dtype 非法：int32
TEST_F(ActsUlqInputGradTilingTest, err_ygrad_dtype_unsupported)
{
    gert::TilingContextPara ctx(kOpType,
                                {{{{{128}, {128}}, DT_INT32, FORMAT_ND},
                                  {{{128}, {128}}, DT_BOOL, FORMAT_ND},
                                  {{{128}, {128}}, DT_BOOL, FORMAT_ND}}},
                                {{{{{128}, {128}}, DT_INT32, FORMAT_ND}}}, &g_compileInfoStub);
    ExecuteTestCase(ctx, ge::GRAPH_FAILED);
}

// mask dtype 非法：int32（两 mask 一致但不在 bool/fp16/fp32 支持集）
TEST_F(ActsUlqInputGradTilingTest, err_mask_dtype_unsupported)
{
    gert::TilingContextPara ctx(kOpType,
                                {{{{{128}, {128}}, DT_FLOAT16, FORMAT_ND},
                                  {{{128}, {128}}, DT_INT32, FORMAT_ND},
                                  {{{128}, {128}}, DT_INT32, FORMAT_ND}}},
                                {{{{{128}, {128}}, DT_FLOAT16, FORMAT_ND}}}, &g_compileInfoStub);
    ExecuteTestCase(ctx, ge::GRAPH_FAILED);
}

// x_grad dtype 与 y_grad 不一致
TEST_F(ActsUlqInputGradTilingTest, err_xgrad_dtype_mismatch)
{
    gert::TilingContextPara ctx(kOpType,
                                {{{{{128}, {128}}, DT_FLOAT16, FORMAT_ND},
                                  {{{128}, {128}}, DT_BOOL, FORMAT_ND},
                                  {{{128}, {128}}, DT_BOOL, FORMAT_ND}}},
                                {{{{{128}, {128}}, DT_FLOAT, FORMAT_ND}}}, &g_compileInfoStub);
    ExecuteTestCase(ctx, ge::GRAPH_FAILED);
}
