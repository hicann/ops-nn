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
 * \file test_apply_adam_d_tiling.cpp
 * \brief ApplyAdamD Tiling UT（全覆盖：6 TilingKey 分支 + 多核/UB 切分 + 边界 + 异常）
 *
 * 算子接口：10 输入(var,m,v,beta1_power,beta2_power,lr,beta1,beta2,epsilon,grad) /
 *           3 输出(var,m,v) / 2 属性(use_locking,use_nesterov)。
 *
 * TilingKey 由 ASCENDC_TPL_SEL_PARAM(context, dtypeKey, useNesterov) 生成
 * （FastEncodeTilingKeyDirect: 低位 D_T = C_DT 枚举值[8bw]，高位 USE_NESTEROV 在 {0,1} 的索引[8bw]）：
 *   C_DT_FLOAT=0, C_DT_FLOAT16=1, C_DT_BF16=27；nesterov=1 -> +(1<<8)=256
 *   => 6 组合 key: fp32/0=0  fp16/0=1  bf16/0=27  fp32/1=256  fp16/1=257  bf16/1=283
 *   （与 docs/INTEGRATION_REPORT.md §2 实测二进制后缀一致）
 *
 * TilingData = {int64 totalNum, int64 blockFactor, int64 ubFactor}（apply_adam_d_tiling_data.h）。
 * 平台: coreNum=64, ubSize=262144(=256KB), RESERVED_UB=48KB -> availableUb=212992。
 *   fp32 : alignElem=32/4=8 , bytePerElem=64 -> ubFactor=FloorAlign(212992/64=3328,256)=3328
 *   16bit: alignElem=32/2=16, bytePerElem=52 -> ubFactor=FloorAlign(212992/52=4096,256)=4096
 *
 * 覆盖场景：
 *   T001..T006  6 TilingKey 组合（3 dtype × 2 useNesterov），小 shape [2,3]
 *   T007        rank1 单元素 [1]           -> totalNum=1, blockFactor=1, 单核
 *   T008        空 tensor [0]              -> totalNum=0, blockFactor=0, 单核（kernel 早退）
 *   T009        rank0 标量 []              -> EnsureNotScalar->[1], totalNum=1
 *   T010        大 shape [4096,64] fp32    -> 多核(64) + 每核多 loop(blockFactor>ubFactor)
 *   T011        高 rank4 [2,3,4,5] fp32    -> min-work-per-core 限核（n=120<2048 -> 单核, blockFactor=120）
 *   T012        大 shape [256,256] bf16/1  -> bf16 多核 + min-work-per-core(=2048) 限核（32 核）+ key=283
 *
 * 多核切分策略（op_host/apply_adam_d_tiling.cpp ComputeMultiCoreSplit）：
 *   wantCore = clamp(CeilDiv(totalNum, MIN_ELEM_PER_CORE=2048), 1, coreNum)
 *   blockFactor = CeilAlign(CeilDiv(totalNum, wantCore), alignElem); usedCoreNum = CeilDiv(totalNum, blockFactor)
 *   小 shape 限核以削减启动/同步开销；大 shape(>= coreNum*2048) 仍占满 coreNum。
 *   T013        unsupported dtype DT_INT32 -> GRAPH_FAILED
 *   T014        unsupported dtype DT_INT64 -> GRAPH_FAILED
 *
 * 输入契约校验负向用例（M1 修复：op_host CheckInputContract，对齐 spec.error_codes）：
 *   T015..T017  m / v / grad 与 var 非同形（含同 size 不同 rank）-> shape_mismatch -> GRAPH_FAILED
 *   T018/T019   标量 beta1_power / epsilon GetShapeSize()!=1      -> shape_mismatch -> GRAPH_FAILED
 *   T020/T021   grad / lr dtype 与 var 不一致                     -> dtype_not_supported -> GRAPH_FAILED
 *   T022        合法基线（同形 + 标量[1] + 同 dtype）             -> 通过（负向辅助器对合法输入零误伤）
 */

#include <cstdint>
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "apply_adam_d_tiling_data.h"  // struct ApplyAdamDTilingData（op_kernel，host↔kernel 同名）

namespace ApplyAdamDUT {

using namespace std;
using namespace ge;
using namespace gert;

static const std::string OP_NAME = "ApplyAdamD";

// 平台 / Tiling 常量（与 op_host/apply_adam_d_tiling.cpp 一致）
static constexpr uint64_t CORE_NUM = 64;
static constexpr uint64_t UB_SIZE = 262144;        // 256KB
static constexpr int64_t RESERVED_UB = 48 * 1024;  // 49152
static constexpr int64_t AVAILABLE_UB = static_cast<int64_t>(UB_SIZE) - RESERVED_UB;  // 212992
static constexpr int64_t UB_FACTOR_FP32 = 3328;    // FloorAlign(212992/64,256)
static constexpr int64_t UB_FACTOR_16BIT = 4096;   // FloorAlign(212992/52,256)

class AddExampleTilingCompileInfo {};

class ApplyAdamDTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ApplyAdamDTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "ApplyAdamDTilingTest TearDown." << std::endl;
    }
};

// 构造 10 输入 + 3 输出 + 2 bool attr 的 Tiling 上下文。
//   var/m/v/grad 取 varShape；6 个标量取 [1]；全部同 dtype。
//   attr 顺序: use_locking(idx0), use_nesterov(idx1) —— 与 op_def 注册顺序一致。
static gert::TilingContextPara MakeTilingPara(const std::initializer_list<int64_t>& varShape,
                                              ge::DataType dtype,
                                              bool useLocking,
                                              bool useNesterov,
                                              void* compileInfo)
{
    gert::StorageShape tensorShape = {varShape, varShape};
    gert::StorageShape scalarShape = {{1}, {1}};

    std::vector<TilingContextPara::TensorDescription> inputs({
        {tensorShape, dtype, ge::FORMAT_ND},  // var
        {tensorShape, dtype, ge::FORMAT_ND},  // m
        {tensorShape, dtype, ge::FORMAT_ND},  // v
        {scalarShape, dtype, ge::FORMAT_ND},  // beta1_power
        {scalarShape, dtype, ge::FORMAT_ND},  // beta2_power
        {scalarShape, dtype, ge::FORMAT_ND},  // lr
        {scalarShape, dtype, ge::FORMAT_ND},  // beta1
        {scalarShape, dtype, ge::FORMAT_ND},  // beta2
        {scalarShape, dtype, ge::FORMAT_ND},  // epsilon
        {tensorShape, dtype, ge::FORMAT_ND},  // grad
    });
    std::vector<TilingContextPara::TensorDescription> outputs({
        {tensorShape, dtype, ge::FORMAT_ND},  // var
        {tensorShape, dtype, ge::FORMAT_ND},  // m
        {tensorShape, dtype, ge::FORMAT_ND},  // v
    });
    std::vector<TilingContextPara::OpAttr> attrs({
        {"use_locking",  Ops::Math::AnyValue::CreateFrom<bool>(useLocking)},
        {"use_nesterov", Ops::Math::AnyValue::CreateFrom<bool>(useNesterov)},
    });
    return gert::TilingContextPara(OP_NAME, inputs, outputs, attrs, compileInfo, CORE_NUM, UB_SIZE, 4096);
}

// 构造 10 输入张量描述（合法基线：var/m/v/grad=tensorShape，6 标量=scalarShape，全部同 dtype）。
// 负向用例在返回后按 index 覆盖个别输入的 {shape, dtype} 以构造契约违反场景。
static std::vector<TilingContextPara::TensorDescription> BuildLegalInputs(
    const gert::StorageShape& tensorShape, const gert::StorageShape& scalarShape, ge::DataType dtype)
{
    return std::vector<TilingContextPara::TensorDescription>({
        {tensorShape, dtype, ge::FORMAT_ND},  // 0 var
        {tensorShape, dtype, ge::FORMAT_ND},  // 1 m
        {tensorShape, dtype, ge::FORMAT_ND},  // 2 v
        {scalarShape, dtype, ge::FORMAT_ND},  // 3 beta1_power
        {scalarShape, dtype, ge::FORMAT_ND},  // 4 beta2_power
        {scalarShape, dtype, ge::FORMAT_ND},  // 5 lr
        {scalarShape, dtype, ge::FORMAT_ND},  // 6 beta1
        {scalarShape, dtype, ge::FORMAT_ND},  // 7 beta2
        {scalarShape, dtype, ge::FORMAT_ND},  // 8 epsilon
        {tensorShape, dtype, ge::FORMAT_ND},  // 9 grad
    });
}

// 由（可能已被覆盖的）输入向量构造 Tiling 上下文。输出固定取 var 的 shape/dtype（tiling 不校验输出）。
static gert::TilingContextPara MakeParaFromInputs(
    const std::vector<TilingContextPara::TensorDescription>& inputs,
    const gert::StorageShape& outShape, ge::DataType outDtype, bool useNesterov, void* compileInfo)
{
    std::vector<TilingContextPara::TensorDescription> outputs({
        {outShape, outDtype, ge::FORMAT_ND},  // var
        {outShape, outDtype, ge::FORMAT_ND},  // m
        {outShape, outDtype, ge::FORMAT_ND},  // v
    });
    std::vector<TilingContextPara::OpAttr> attrs({
        {"use_locking",  Ops::Math::AnyValue::CreateFrom<bool>(false)},
        {"use_nesterov", Ops::Math::AnyValue::CreateFrom<bool>(useNesterov)},
    });
    return gert::TilingContextPara(OP_NAME, inputs, outputs, attrs, compileInfo, CORE_NUM, UB_SIZE, 4096);
}

// 期望 tiling 拒绝（GRAPH_FAILED -> ExecuteTiling 返回 false）。
static void ExpectTilingReject(const gert::TilingContextPara& para, const char* why)
{
    TilingInfo info;
    bool ok = ExecuteTiling(para, info);
    EXPECT_FALSE(ok) << "Tiling should fail: " << why;
}

// 校验成功路径的 TilingKey + TilingData + 多核/workspace
static void CheckTilingSuccess(const gert::TilingContextPara& para,
                               int64_t expectKey,
                               int64_t expectTotalNum,
                               int64_t expectBlockFactor,
                               int64_t expectUbFactor,
                               size_t expectBlockNum)
{
    TilingInfo info;
    bool ok = ExecuteTiling(para, info);
    ASSERT_TRUE(ok) << "Tiling should succeed";

    EXPECT_EQ(info.tilingKey, expectKey) << "TilingKey mismatch";
    EXPECT_EQ(info.blockNum, expectBlockNum) << "usedCoreNum(blockDim) mismatch";

    ASSERT_GE(info.tilingDataSize, sizeof(ApplyAdamDTilingData))
        << "TilingData size insufficient";
    auto* td = reinterpret_cast<ApplyAdamDTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->totalNum, expectTotalNum) << "totalNum mismatch";
    EXPECT_EQ(td->blockFactor, expectBlockFactor) << "blockFactor mismatch";
    EXPECT_EQ(td->ubFactor, expectUbFactor) << "ubFactor mismatch";

    // workspace：仅系统预留 0
    ASSERT_EQ(info.workspaceSizes.size(), 1u);
    EXPECT_EQ(info.workspaceSizes[0], 0);
}

// ---- T001..T006: 6 TilingKey 组合（小 shape [2,3]，totalNum=6 < alignElem -> blockFactor=totalNum，单核）----
TEST_F(ApplyAdamDTilingTest, T001_Fp32_NoNesterov_Key0)
{
    AddExampleTilingCompileInfo ci;
    auto para = MakeTilingPara({2, 3}, ge::DT_FLOAT, false, false, &ci);
    CheckTilingSuccess(para, /*key*/ 0, /*total*/ 6, /*block*/ 6, UB_FACTOR_FP32, /*cores*/ 1);
}

TEST_F(ApplyAdamDTilingTest, T002_Fp16_NoNesterov_Key1)
{
    AddExampleTilingCompileInfo ci;
    auto para = MakeTilingPara({2, 3}, ge::DT_FLOAT16, false, false, &ci);
    CheckTilingSuccess(para, /*key*/ 1, 6, 6, UB_FACTOR_16BIT, 1);
}

TEST_F(ApplyAdamDTilingTest, T003_Bf16_NoNesterov_Key27)
{
    AddExampleTilingCompileInfo ci;
    auto para = MakeTilingPara({2, 3}, ge::DT_BF16, false, false, &ci);
    CheckTilingSuccess(para, /*key*/ 27, 6, 6, UB_FACTOR_16BIT, 1);
}

TEST_F(ApplyAdamDTilingTest, T004_Fp32_Nesterov_Key256)
{
    AddExampleTilingCompileInfo ci;
    auto para = MakeTilingPara({2, 3}, ge::DT_FLOAT, false, true, &ci);
    CheckTilingSuccess(para, /*key*/ 256, 6, 6, UB_FACTOR_FP32, 1);
}

TEST_F(ApplyAdamDTilingTest, T005_Fp16_Nesterov_Key257)
{
    AddExampleTilingCompileInfo ci;
    auto para = MakeTilingPara({2, 3}, ge::DT_FLOAT16, false, true, &ci);
    CheckTilingSuccess(para, /*key*/ 257, 6, 6, UB_FACTOR_16BIT, 1);
}

TEST_F(ApplyAdamDTilingTest, T006_Bf16_Nesterov_Key283)
{
    AddExampleTilingCompileInfo ci;
    auto para = MakeTilingPara({2, 3}, ge::DT_BF16, false, true, &ci);
    CheckTilingSuccess(para, /*key*/ 283, 6, 6, UB_FACTOR_16BIT, 1);
}

// use_locking=true 不应改变数值/分支（仅图层语义）；与 use_locking=false 同 key/同 tiling
TEST_F(ApplyAdamDTilingTest, T006b_Fp32_UseLockingTrue_SameAsFalse)
{
    AddExampleTilingCompileInfo ci;
    auto para = MakeTilingPara({2, 3}, ge::DT_FLOAT, true, false, &ci);
    CheckTilingSuccess(para, /*key*/ 0, 6, 6, UB_FACTOR_FP32, 1);
}

// ---- T007: rank1 单元素 ----
TEST_F(ApplyAdamDTilingTest, T007_Rank1_SingleElement)
{
    AddExampleTilingCompileInfo ci;
    auto para = MakeTilingPara({1}, ge::DT_FLOAT, false, false, &ci);
    // totalNum=1 < alignElem(8) -> blockFactor=1, 单核
    CheckTilingSuccess(para, 0, /*total*/ 1, /*block*/ 1, UB_FACTOR_FP32, 1);
}

// ---- T008: 空 tensor ----
TEST_F(ApplyAdamDTilingTest, T008_EmptyTensor)
{
    AddExampleTilingCompileInfo ci;
    auto para = MakeTilingPara({0}, ge::DT_FLOAT, false, false, &ci);
    // totalNum=0 -> blockFactor=0, usedCoreNum=1（kernel Process 早退）
    CheckTilingSuccess(para, 0, /*total*/ 0, /*block*/ 0, UB_FACTOR_FP32, 1);
}

// ---- T009: rank0 标量（EnsureNotScalar -> [1]）----
TEST_F(ApplyAdamDTilingTest, T009_Rank0_Scalar)
{
    AddExampleTilingCompileInfo ci;
    auto para = MakeTilingPara({}, ge::DT_FLOAT, false, false, &ci);
    // 0-dim -> EnsureNotScalar 视作 [1] -> totalNum=1
    CheckTilingSuccess(para, 0, /*total*/ 1, /*block*/ 1, UB_FACTOR_FP32, 1);
}

// ---- T010: 大 shape，多核 + 每核多 UB loop（blockFactor=4096 > ubFactor=3328）----
TEST_F(ApplyAdamDTilingTest, T010_LargeShape_MultiCore_MultiLoop_Fp32)
{
    AddExampleTilingCompileInfo ci;
    auto para = MakeTilingPara({4096, 64}, ge::DT_FLOAT, false, false, &ci);
    // totalNum=262144; perCoreRaw=CeilDiv(262144,64)=4096; blockFactor=CeilAlign(4096,8)=4096;
    // usedCoreNum=CeilDiv(262144,4096)=64; ubFactor=3328 -> 每核 loopCount=ceil(4096/3328)=2
    CheckTilingSuccess(para, 0, /*total*/ 262144, /*block*/ 4096, UB_FACTOR_FP32, 64);
}

// ---- T011: 高 rank 小 shape，min-work-per-core 限核（n=120 < 2048 -> 单核）----
TEST_F(ApplyAdamDTilingTest, T011_HighRank4_MinWorkPerCore_SingleCore_Fp32)
{
    AddExampleTilingCompileInfo ci;
    auto para = MakeTilingPara({2, 3, 4, 5}, ge::DT_FLOAT, false, false, &ci);
    // totalNum=120 >= alignElem(8); wantCore=clamp(CeilDiv(120,2048)=1,1,64)=1;
    // perCoreRaw=CeilDiv(120,1)=120; blockFactor=CeilAlign(120,8)=120; usedCoreNum=CeilDiv(120,120)=1
    CheckTilingSuccess(para, 0, /*total*/ 120, /*block*/ 120, UB_FACTOR_FP32, 1);
}

// ---- T012: bf16 大 shape 多核 + min-work-per-core 限核（32 核）+ nesterov（key=283）----
TEST_F(ApplyAdamDTilingTest, T012_LargeShape_MultiCore_Bf16_Nesterov)
{
    AddExampleTilingCompileInfo ci;
    auto para = MakeTilingPara({256, 256}, ge::DT_BF16, false, true, &ci);
    // totalNum=65536; alignElem=16; wantCore=clamp(CeilDiv(65536,2048)=32,1,64)=32;
    // perCoreRaw=CeilDiv(65536,32)=2048; blockFactor=CeilAlign(2048,16)=2048;
    // usedCoreNum=CeilDiv(65536,2048)=32; ubFactor=4096
    CheckTilingSuccess(para, /*key*/ 283, /*total*/ 65536, /*block*/ 2048, UB_FACTOR_16BIT, 32);
}

// ---- T013/T014: 不支持 dtype -> Tiling 失败 ----
TEST_F(ApplyAdamDTilingTest, T013_Err_UnsupportedDtypeInt32)
{
    AddExampleTilingCompileInfo ci;
    auto para = MakeTilingPara({32, 32}, ge::DT_INT32, false, false, &ci);
    TilingInfo info;
    bool ok = ExecuteTiling(para, info);
    EXPECT_FALSE(ok) << "Tiling should fail for DT_INT32";
}

TEST_F(ApplyAdamDTilingTest, T014_Err_UnsupportedDtypeInt64)
{
    AddExampleTilingCompileInfo ci;
    auto para = MakeTilingPara({16, 16}, ge::DT_INT64, false, false, &ci);
    TilingInfo info;
    bool ok = ExecuteTiling(para, info);
    EXPECT_FALSE(ok) << "Tiling should fail for DT_INT64";
}

// ============================================================================
// 输入契约校验负向用例（M1 修复：op_host CheckInputContract）
// spec.yaml error_codes: shape_mismatch（同形 / 标量 size==1）、dtype_not_supported（跨输入同 dtype）。
// 合法基线: var/m/v/grad=[2,3], 6 标量=[1], 全部 fp32 -> 应通过；下方各破坏一处契约 -> 应被 tiling 拒绝。
// ============================================================================

// ---- T015..T017: 张量同形违反（m / v / grad 与 var shape 不一致）-> shape_mismatch ----
TEST_F(ApplyAdamDTilingTest, T015_Err_ShapeMismatch_M)
{
    AddExampleTilingCompileInfo ci;
    gert::StorageShape ts = {{2, 3}, {2, 3}};
    gert::StorageShape ss = {{1}, {1}};
    auto inputs = BuildLegalInputs(ts, ss, ge::DT_FLOAT);
    inputs[1] = {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND};  // m -> [2,4] != var [2,3]
    ExpectTilingReject(MakeParaFromInputs(inputs, ts, ge::DT_FLOAT, false, &ci), "m shape != var");
}

TEST_F(ApplyAdamDTilingTest, T016_Err_ShapeMismatch_V_Rank)
{
    AddExampleTilingCompileInfo ci;
    gert::StorageShape ts = {{2, 3}, {2, 3}};
    gert::StorageShape ss = {{1}, {1}};
    auto inputs = BuildLegalInputs(ts, ss, ge::DT_FLOAT);
    inputs[2] = {{{6}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND};  // v -> [6] (same size, diff rank) != var [2,3]
    ExpectTilingReject(MakeParaFromInputs(inputs, ts, ge::DT_FLOAT, false, &ci), "v rank != var");
}

TEST_F(ApplyAdamDTilingTest, T017_Err_ShapeMismatch_Grad)
{
    AddExampleTilingCompileInfo ci;
    gert::StorageShape ts = {{2, 3}, {2, 3}};
    gert::StorageShape ss = {{1}, {1}};
    auto inputs = BuildLegalInputs(ts, ss, ge::DT_FLOAT);
    inputs[9] = {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND};  // grad -> [3,2] != var [2,3]
    ExpectTilingReject(MakeParaFromInputs(inputs, ts, ge::DT_FLOAT, false, &ci), "grad shape != var");
}

// ---- T018/T019: 标量 GetShapeSize()!=1 -> shape_mismatch ----
TEST_F(ApplyAdamDTilingTest, T018_Err_ScalarSizeNotOne_Beta1Power)
{
    AddExampleTilingCompileInfo ci;
    gert::StorageShape ts = {{2, 3}, {2, 3}};
    gert::StorageShape ss = {{1}, {1}};
    auto inputs = BuildLegalInputs(ts, ss, ge::DT_FLOAT);
    inputs[3] = {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND};  // beta1_power -> [2] (size 2)
    ExpectTilingReject(MakeParaFromInputs(inputs, ts, ge::DT_FLOAT, false, &ci), "beta1_power size != 1");
}

TEST_F(ApplyAdamDTilingTest, T019_Err_ScalarSizeNotOne_Epsilon)
{
    AddExampleTilingCompileInfo ci;
    gert::StorageShape ts = {{2, 3}, {2, 3}};
    gert::StorageShape ss = {{1}, {1}};
    auto inputs = BuildLegalInputs(ts, ss, ge::DT_FLOAT);
    inputs[8] = {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND};  // epsilon -> [3] (size 3)
    ExpectTilingReject(MakeParaFromInputs(inputs, ts, ge::DT_FLOAT, false, &ci), "epsilon size != 1");
}

// ---- T020/T021: 跨输入 dtype 不一致（var 与某输入 dtype 不同）-> dtype_not_supported ----
TEST_F(ApplyAdamDTilingTest, T020_Err_DtypeMismatch_GradVsVar)
{
    AddExampleTilingCompileInfo ci;
    gert::StorageShape ts = {{2, 3}, {2, 3}};
    gert::StorageShape ss = {{1}, {1}};
    auto inputs = BuildLegalInputs(ts, ss, ge::DT_FLOAT);
    inputs[9] = {ts, ge::DT_FLOAT16, ge::FORMAT_ND};  // grad fp16 != var fp32
    ExpectTilingReject(MakeParaFromInputs(inputs, ts, ge::DT_FLOAT, false, &ci), "grad dtype != var");
}

TEST_F(ApplyAdamDTilingTest, T021_Err_DtypeMismatch_ScalarVsVar)
{
    AddExampleTilingCompileInfo ci;
    gert::StorageShape ts = {{2, 3}, {2, 3}};
    gert::StorageShape ss = {{1}, {1}};
    auto inputs = BuildLegalInputs(ts, ss, ge::DT_FLOAT16);
    inputs[5] = {ss, ge::DT_FLOAT, ge::FORMAT_ND};  // lr fp32 != var fp16
    ExpectTilingReject(MakeParaFromInputs(inputs, ts, ge::DT_FLOAT16, false, &ci), "lr dtype != var");
}

// ---- T022: 同样输入向量但不破坏任何契约（全 fp32 同形）-> 应通过（确认负向辅助器对合法输入零误伤）----
TEST_F(ApplyAdamDTilingTest, T022_Contract_LegalBaseline_Passes)
{
    AddExampleTilingCompileInfo ci;
    gert::StorageShape ts = {{2, 3}, {2, 3}};
    gert::StorageShape ss = {{1}, {1}};
    auto inputs = BuildLegalInputs(ts, ss, ge::DT_FLOAT);
    auto para = MakeParaFromInputs(inputs, ts, ge::DT_FLOAT, false, &ci);
    // 与 T001 等价：fp32/no-nesterov, totalNum=6, blockFactor=6, 单核, key=0
    CheckTilingSuccess(para, /*key*/ 0, /*total*/ 6, /*block*/ 6, UB_FACTOR_FP32, /*cores*/ 1);
}

}  // namespace ApplyAdamDUT
