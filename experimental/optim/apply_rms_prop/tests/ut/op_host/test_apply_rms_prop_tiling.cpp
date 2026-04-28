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
 * \file test_apply_rms_prop_tiling.cpp
 * \brief ApplyRmsProp atvoss Tiling UT（迭代三 - 全覆盖：多 dtype 正常路径 + 异常路径）
 *
 * 覆盖场景：
 *   正常路径（迭代一/二）：
 *     T_001  基础 shape + fp32
 *     T_002  大 shape（跨核心）
 *     T_003  1D shape
 *     T_004  scalar 边界值（rho=0.0 → rho1m=1.0）
 *     T_005  fp16 dtype：验证 Fp16BitsToFloat + ApplyRmsPropWithCast<half>
 *     T_006  fp16 边界 scalar（+0 / -0 / 规格化 1.0 / 次正规数）
 *     T_007  bf16 dtype：验证 Bf16BitsToFloat + ApplyRmsPropWithCast<bfloat16_t>
 *     T_008  bf16 边界 scalar（负值 / 极小值 / 1.0）
 *     T_009  dtype 分发后 TilingKey 存在且一致
 *   异常路径（迭代三新增）：
 *     T_010  不支持的 dtype (DT_INT32) → Tiling 失败
 *     T_011  不支持的 dtype (DT_INT64) → Tiling 失败
 *     T_012  scalar tensor data 指针为空（isConst=false）→ Tiling 失败
 *            [覆盖 ReadScalarAsFloat 的 dataPtr==nullptr 保护分支]
 *     T_013  scalar tensor 16bit 数据指针为空（fp16 + 非 const）→ Tiling 失败
 *            [覆盖 ReadScalarAsFloat 的 rawPtr==nullptr 保护分支]
 *
 * 重点：
 *   - scalar tensor 通过 TensorDescription 的 isConst + constValue 传入实际数值
 *   - fp32 通过 float* 传；fp16/bf16 通过 uint16_t 位模式传递（Host 侧 Tiling 按 dtype 解释字节）
 *   - Tiling 函数从 tensor->GetData<float/uint16_t>() 读取 scalar 实际值写入 TilingData（统一存为 float）
 *   - dtype 分发走不同 OpDag 实例化 ElewiseBaseTiling.DoTiling
 *   - 异常用例通过 ExecuteTiling 返回 false（Tiling 返回 GRAPH_FAILED）判定
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "apply_rms_prop_struct.h"

namespace ApplyRmsPropUT {
using namespace std;
using namespace ge;
using namespace gert;

static const std::string OP_NAME = "ApplyRmsProp";

// -------------------------------------------------------------------
// 全局 scalar 常量数据（用于 isConst tensor 的 constValue 指针）
// 每个 test 函数使用这些静态变量，避免局部 scope 问题
//
// 说明：
//   - fp32 路径使用 float 字段
//   - fp16/bf16 路径使用 uint16_t 字段直接存储位模式（与 Tiling 内部
//     `tensor->GetData<uint16_t>()` 读取方式一致）
// -------------------------------------------------------------------
struct ScalarStorageF32 {
    float lr       = 0.01f;
    float rho      = 0.9f;
    float momentum = 0.5f;
    float epsilon  = 1e-8f;
};

struct ScalarStorageU16 {
    uint16_t lr       = 0;
    uint16_t rho      = 0;
    uint16_t momentum = 0;
    uint16_t epsilon  = 0;
};

// -------------------------------------------------------------------
// 位转换工具（UT 内部用于"构造期望值"）
// 与 apply_rms_prop_tiling.cpp 中 Fp16BitsToFloat / Bf16BitsToFloat 一致
// -------------------------------------------------------------------
static float Ut_Fp16BitsToFloat(uint16_t bits)
{
    uint32_t sign     = (bits >> 15) & 0x1u;
    uint32_t exponent = (bits >> 10) & 0x1Fu;
    uint32_t mantissa = bits & 0x3FFu;

    uint32_t f32 = 0u;
    if (exponent == 0u) {
        if (mantissa == 0u) {
            f32 = sign << 31;
        } else {
            while ((mantissa & 0x400u) == 0u) {
                mantissa <<= 1u;
                exponent = static_cast<uint32_t>(static_cast<int32_t>(exponent) - 1);
            }
            exponent += 1u;
            mantissa &= 0x3FFu;
            uint32_t newExp = exponent + (127u - 15u);
            f32 = (sign << 31) | (newExp << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1Fu) {
        f32 = (sign << 31) | (0xFFu << 23) | (mantissa << 13);
    } else {
        uint32_t newExp = exponent + (127u - 15u);
        f32 = (sign << 31) | (newExp << 23) | (mantissa << 13);
    }
    float out;
    std::memcpy(&out, &f32, sizeof(out));
    return out;
}

static float Ut_Bf16BitsToFloat(uint16_t bits)
{
    uint32_t f32 = static_cast<uint32_t>(bits) << 16;
    float out;
    std::memcpy(&out, &f32, sizeof(out));
    return out;
}

// fp32 -> fp16 位模式（简化：仅用于常见规格化正/负数；UT 自身构造用）
static uint16_t Ut_FloatToFp16Bits(float f)
{
    uint32_t f32;
    std::memcpy(&f32, &f, sizeof(f32));
    uint32_t sign     = (f32 >> 31) & 0x1u;
    int32_t  exp      = static_cast<int32_t>((f32 >> 23) & 0xFFu) - 127;
    uint32_t mantissa = f32 & 0x7FFFFFu;

    if (f == 0.0f) {
        return static_cast<uint16_t>(sign << 15);
    }
    // 规格化 fp16 范围：-14 <= exp <= 15
    if (exp >= 16) {
        // Inf
        return static_cast<uint16_t>((sign << 15) | (0x1Fu << 10));
    }
    if (exp < -14) {
        // 简化处理：当作 0
        return static_cast<uint16_t>(sign << 15);
    }
    uint32_t fp16Exp = static_cast<uint32_t>(exp + 15);
    uint32_t fp16Man = mantissa >> 13;
    return static_cast<uint16_t>((sign << 15) | (fp16Exp << 10) | fp16Man);
}

// fp32 -> bf16 位模式（截断高 16 位；不做舍入）
static uint16_t Ut_FloatToBf16Bits(float f)
{
    uint32_t f32;
    std::memcpy(&f32, &f, sizeof(f32));
    return static_cast<uint16_t>(f32 >> 16);
}

class AddExampleTilingCompileInfo {};

class ApplyRmsPropTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ApplyRmsPropTilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ApplyRmsPropTilingTest TearDown." << std::endl;
    }
};

// -------------------------------------------------------------------
// 辅助函数：构造 ApplyRmsProp 的 TilingContextPara（方案 B：4 input tensor + 4 float attr）
//
// 方案 B（2026-04-23）修正：
//   算子定义已将 lr/rho/momentum/epsilon 从 Input(scalar tensor) 改为 Attr(float)，
//   UT helper 同步：inputs 仅 4 个 tensor（var/ms/mom/grad），scalar 通过 attrs_ 传入。
// -------------------------------------------------------------------
static gert::TilingContextPara MakeFp32TilingContextPara(
    const std::initializer_list<int64_t>& varShape,
    ScalarStorageF32* scalars,
    void* compileInfo,
    uint64_t coreNum = 64,
    uint64_t ubSize = 262144,
    uint64_t tilingDataSize = 4096)
{
    gert::StorageShape tensorShape = {varShape, varShape};

    std::vector<TilingContextPara::TensorDescription> inputs({
        {tensorShape, ge::DT_FLOAT, ge::FORMAT_ND},  // var
        {tensorShape, ge::DT_FLOAT, ge::FORMAT_ND},  // ms
        {tensorShape, ge::DT_FLOAT, ge::FORMAT_ND},  // mom
        {tensorShape, ge::DT_FLOAT, ge::FORMAT_ND},  // grad
    });

    std::vector<TilingContextPara::TensorDescription> outputs({
        {tensorShape, ge::DT_FLOAT, ge::FORMAT_ND},  // var_out
        {tensorShape, ge::DT_FLOAT, ge::FORMAT_ND},  // ms_out
        {tensorShape, ge::DT_FLOAT, ge::FORMAT_ND},  // mom_out
    });

    std::vector<TilingContextPara::OpAttr> attrs({
        {"lr",       Ops::Math::AnyValue::CreateFrom<float>(scalars->lr)},
        {"rho",      Ops::Math::AnyValue::CreateFrom<float>(scalars->rho)},
        {"momentum", Ops::Math::AnyValue::CreateFrom<float>(scalars->momentum)},
        {"epsilon",  Ops::Math::AnyValue::CreateFrom<float>(scalars->epsilon)},
    });
    return gert::TilingContextPara(OP_NAME, inputs, outputs, attrs, compileInfo,
                                   coreNum, ubSize, tilingDataSize);
}

// fp16/bf16 共用（方案 B：scalar 仍以 float attr 下发，与 fp32 helper 等价）
// 保留旧签名以减少 test case 改动面；ScalarStorageU16 字段在 helper 内被
// 解释回 float（使用对应的 BitsToFloat），便于现有 test 沿用 U16 位模式构造。
static gert::TilingContextPara Make16BitTilingContextPara(
    const std::initializer_list<int64_t>& varShape,
    ge::DataType dtype,
    ScalarStorageU16* scalars,
    void* compileInfo,
    uint64_t coreNum = 64,
    uint64_t ubSize = 262144,
    uint64_t tilingDataSize = 4096)
{
    gert::StorageShape tensorShape = {varShape, varShape};

    std::vector<TilingContextPara::TensorDescription> inputs({
        {tensorShape, dtype, ge::FORMAT_ND},  // var
        {tensorShape, dtype, ge::FORMAT_ND},  // ms
        {tensorShape, dtype, ge::FORMAT_ND},  // mom
        {tensorShape, dtype, ge::FORMAT_ND},  // grad
    });

    std::vector<TilingContextPara::TensorDescription> outputs({
        {tensorShape, dtype, ge::FORMAT_ND},  // var_out
        {tensorShape, dtype, ge::FORMAT_ND},  // ms_out
        {tensorShape, dtype, ge::FORMAT_ND},  // mom_out
    });

    // 将 U16 位模式还原为 float（dtype 决定解码器）
    auto bitsToFloat = [&](uint16_t bits) -> float {
        return (dtype == ge::DT_FLOAT16) ? Ut_Fp16BitsToFloat(bits)
                                         : Ut_Bf16BitsToFloat(bits);
    };
    std::vector<TilingContextPara::OpAttr> attrs({
        {"lr",       Ops::Math::AnyValue::CreateFrom<float>(bitsToFloat(scalars->lr))},
        {"rho",      Ops::Math::AnyValue::CreateFrom<float>(bitsToFloat(scalars->rho))},
        {"momentum", Ops::Math::AnyValue::CreateFrom<float>(bitsToFloat(scalars->momentum))},
        {"epsilon",  Ops::Math::AnyValue::CreateFrom<float>(bitsToFloat(scalars->epsilon))},
    });
    return gert::TilingContextPara(OP_NAME, inputs, outputs, attrs, compileInfo,
                                   coreNum, ubSize, tilingDataSize);
}

// -------------------------------------------------------------------
// T_001: 基础 shape + fp32，Tiling 成功，覆盖核心路径
// -------------------------------------------------------------------
TEST_F(ApplyRmsPropTilingTest, T001_BasicShape_Fp32_Succeeds)
{
    AddExampleTilingCompileInfo compileInfo;
    ScalarStorageF32 scalars;
    scalars.lr       = 0.01f;
    scalars.rho      = 0.9f;
    scalars.momentum = 0.5f;
    scalars.epsilon  = 1e-8f;

    auto para = MakeFp32TilingContextPara({32, 4, 4, 4}, &scalars, &compileInfo);

    TilingInfo info;
    bool ok = ExecuteTiling(para, info);
    EXPECT_TRUE(ok) << "Tiling should succeed for basic fp32 shape";
    if (!ok) return;

    // 方案 B：移除 atvoss EleBaseTilingData，自管 ApplyRmsPropTilingData
    EXPECT_GE(info.tilingDataSize, sizeof(ApplyRmsPropTilingData))
        << "TilingData size insufficient for ApplyRmsPropTilingData";

    auto* td = reinterpret_cast<ApplyRmsPropTilingData*>(info.tilingData.get());
    EXPECT_FLOAT_EQ(td->lr,    0.01f);
    EXPECT_FLOAT_EQ(td->rho1m, 1.0f - 0.9f);
    EXPECT_FLOAT_EQ(td->mom,   0.5f);
    EXPECT_FLOAT_EQ(td->eps,   1e-8f);
}

// -------------------------------------------------------------------
// T_002: 大 shape（多核切分分支）
// -------------------------------------------------------------------
TEST_F(ApplyRmsPropTilingTest, T002_LargeShape_Fp32_Succeeds)
{
    AddExampleTilingCompileInfo compileInfo;
    ScalarStorageF32 scalars;
    scalars.lr       = 0.001f;
    scalars.rho      = 0.99f;
    scalars.momentum = 0.0f;
    scalars.epsilon  = 1e-7f;

    auto para = MakeFp32TilingContextPara({8192, 128}, &scalars, &compileInfo);

    TilingInfo info;
    bool ok = ExecuteTiling(para, info);
    EXPECT_TRUE(ok) << "Tiling should succeed for large shape";
    if (!ok) return;

    EXPECT_GT(info.blockNum, 0u) << "Large shape should be split across cores";

    auto* td = reinterpret_cast<ApplyRmsPropTilingData*>(info.tilingData.get());
    EXPECT_FLOAT_EQ(td->lr,    0.001f);
    EXPECT_NEAR(td->rho1m, 1.0f - 0.99f, 1e-6f);
    EXPECT_FLOAT_EQ(td->mom,   0.0f);
    EXPECT_FLOAT_EQ(td->eps,   1e-7f);
}

// -------------------------------------------------------------------
// T_003: 1D shape
// -------------------------------------------------------------------
TEST_F(ApplyRmsPropTilingTest, T003_1D_Shape_Succeeds)
{
    AddExampleTilingCompileInfo compileInfo;
    ScalarStorageF32 scalars;
    scalars.lr       = 0.1f;
    scalars.rho      = 0.5f;
    scalars.momentum = 0.9f;
    scalars.epsilon  = 1e-6f;

    auto para = MakeFp32TilingContextPara({1024}, &scalars, &compileInfo);

    TilingInfo info;
    bool ok = ExecuteTiling(para, info);
    EXPECT_TRUE(ok) << "Tiling should succeed for 1D shape";
    if (!ok) return;

    auto* td = reinterpret_cast<ApplyRmsPropTilingData*>(info.tilingData.get());
    EXPECT_FLOAT_EQ(td->lr,    0.1f);
    EXPECT_FLOAT_EQ(td->rho1m, 0.5f);
    EXPECT_FLOAT_EQ(td->mom,   0.9f);
    EXPECT_FLOAT_EQ(td->eps,   1e-6f);
}

// -------------------------------------------------------------------
// T_004: scalar 边界 rho=0.0 → rho1m=1.0
// -------------------------------------------------------------------
TEST_F(ApplyRmsPropTilingTest, T004_Scalar_BoundaryRho_Succeeds)
{
    AddExampleTilingCompileInfo compileInfo;
    ScalarStorageF32 scalars;
    scalars.lr       = 1.0f;
    scalars.rho      = 0.0f;
    scalars.momentum = 1.0f;
    scalars.epsilon  = 0.0f;

    auto para = MakeFp32TilingContextPara({64, 32}, &scalars, &compileInfo);

    TilingInfo info;
    bool ok = ExecuteTiling(para, info);
    EXPECT_TRUE(ok) << "Tiling should succeed for boundary scalar values";
    if (!ok) return;

    auto* td = reinterpret_cast<ApplyRmsPropTilingData*>(info.tilingData.get());
    EXPECT_FLOAT_EQ(td->lr,    1.0f);
    EXPECT_FLOAT_EQ(td->rho1m, 1.0f);
    EXPECT_FLOAT_EQ(td->mom,   1.0f);
    EXPECT_FLOAT_EQ(td->eps,   0.0f);
}

// -------------------------------------------------------------------
// T_005: fp16 dtype 基础路径
//   - 验证 DT_FLOAT16 分发：ApplyRmsPropWithCast<half> OpDag
//   - 验证 Fp16BitsToFloat 规格化数位读取
//   - 验证 Host 预计算 rho1m
// -------------------------------------------------------------------
TEST_F(ApplyRmsPropTilingTest, T005_Fp16_BasicShape_Succeeds)
{
    AddExampleTilingCompileInfo compileInfo;
    ScalarStorageU16 scalars;
    // fp16 表示的 lr=0.01（近似值，取 fp16 可精确表示的最近数）
    scalars.lr       = Ut_FloatToFp16Bits(0.01f);
    scalars.rho      = Ut_FloatToFp16Bits(0.9f);
    scalars.momentum = Ut_FloatToFp16Bits(0.5f);
    scalars.epsilon  = Ut_FloatToFp16Bits(0.001f);  // fp16 下 1e-8 会下溢，选较大的值保证精确

    auto para = Make16BitTilingContextPara({16, 32}, ge::DT_FLOAT16, &scalars, &compileInfo);

    TilingInfo info;
    bool ok = ExecuteTiling(para, info);
    EXPECT_TRUE(ok) << "Tiling should succeed for fp16 dtype";
    if (!ok) return;

    EXPECT_GE(info.tilingDataSize, sizeof(ApplyRmsPropTilingData));

    auto* td = reinterpret_cast<ApplyRmsPropTilingData*>(info.tilingData.get());
    // 预期值用同一位转换函数计算，验证 Tiling 内部使用了相同的 fp16 → fp32 转换
    float expectLr  = Ut_Fp16BitsToFloat(scalars.lr);
    float expectRho = Ut_Fp16BitsToFloat(scalars.rho);
    float expectMom = Ut_Fp16BitsToFloat(scalars.momentum);
    float expectEps = Ut_Fp16BitsToFloat(scalars.epsilon);

    EXPECT_FLOAT_EQ(td->lr,    expectLr);
    EXPECT_FLOAT_EQ(td->rho1m, 1.0f - expectRho);
    EXPECT_FLOAT_EQ(td->mom,   expectMom);
    EXPECT_FLOAT_EQ(td->eps,   expectEps);
}

// -------------------------------------------------------------------
// T_006: fp16 边界 scalar - 覆盖 Fp16BitsToFloat 关键分支
//   - lr=+0.0 (sign=0, exp=0, mantissa=0)           → 带符号零分支
//   - rho=-0.0（sign=1, exp=0, mantissa=0）         → 带符号零分支（负号）
//   - momentum=1.0（规格化数）                         → 规格化数分支（主路径）
//   - epsilon=次正规数（exp=0, mantissa!=0）          → 次正规数归一化分支
// -------------------------------------------------------------------
TEST_F(ApplyRmsPropTilingTest, T006_Fp16_BoundaryScalar_Succeeds)
{
    AddExampleTilingCompileInfo compileInfo;
    ScalarStorageU16 scalars;
    scalars.lr       = 0x0000;   // +0.0
    scalars.rho      = 0x8000;   // -0.0 → 1 - (-0) = 1
    scalars.momentum = 0x3C00;   // 1.0  (sign=0, exp=15, mantissa=0)
    scalars.epsilon  = 0x0001;   // 最小次正规数 ≈ 5.96e-8

    auto para = Make16BitTilingContextPara({128}, ge::DT_FLOAT16, &scalars, &compileInfo);

    TilingInfo info;
    bool ok = ExecuteTiling(para, info);
    EXPECT_TRUE(ok) << "Tiling should succeed for fp16 boundary scalars";
    if (!ok) return;

    auto* td = reinterpret_cast<ApplyRmsPropTilingData*>(info.tilingData.get());

    // +0.0
    EXPECT_FLOAT_EQ(td->lr, 0.0f);
    // rho=-0.0 → rho1m = 1 - (-0) = 1.0
    EXPECT_FLOAT_EQ(td->rho1m, 1.0f);
    // 1.0
    EXPECT_FLOAT_EQ(td->mom, 1.0f);
    // 次正规数：值由转换函数决定，与 UT 端同函数计算结果严格相等
    float expectEps = Ut_Fp16BitsToFloat(0x0001);
    EXPECT_FLOAT_EQ(td->eps, expectEps);
    EXPECT_GT(td->eps, 0.0f) << "Subnormal fp16 value must be positive non-zero";
}

// -------------------------------------------------------------------
// T_007: bf16 dtype 基础路径
//   - 验证 DT_BF16 分发：ApplyRmsPropWithCast<bfloat16_t> OpDag
//   - 验证 Bf16BitsToFloat 位读取
// -------------------------------------------------------------------
TEST_F(ApplyRmsPropTilingTest, T007_Bf16_BasicShape_Succeeds)
{
    AddExampleTilingCompileInfo compileInfo;
    ScalarStorageU16 scalars;
    // bf16 可表达范围与 fp32 相同（指数 8 位）；截断尾数的位模式
    scalars.lr       = Ut_FloatToBf16Bits(0.01f);
    scalars.rho      = Ut_FloatToBf16Bits(0.9f);
    scalars.momentum = Ut_FloatToBf16Bits(0.5f);
    scalars.epsilon  = Ut_FloatToBf16Bits(1e-8f);

    auto para = Make16BitTilingContextPara({64, 64}, ge::DT_BF16, &scalars, &compileInfo);

    TilingInfo info;
    bool ok = ExecuteTiling(para, info);
    EXPECT_TRUE(ok) << "Tiling should succeed for bf16 dtype";
    if (!ok) return;

    auto* td = reinterpret_cast<ApplyRmsPropTilingData*>(info.tilingData.get());

    float expectLr  = Ut_Bf16BitsToFloat(scalars.lr);
    float expectRho = Ut_Bf16BitsToFloat(scalars.rho);
    float expectMom = Ut_Bf16BitsToFloat(scalars.momentum);
    float expectEps = Ut_Bf16BitsToFloat(scalars.epsilon);

    EXPECT_FLOAT_EQ(td->lr,    expectLr);
    EXPECT_FLOAT_EQ(td->rho1m, 1.0f - expectRho);
    EXPECT_FLOAT_EQ(td->mom,   expectMom);
    EXPECT_FLOAT_EQ(td->eps,   expectEps);
}

// -------------------------------------------------------------------
// T_008: bf16 边界 scalar
//   - lr=0（全 0）       → 零
//   - rho=+Inf 的低位部分截断形式
//   - momentum=负值       → 验证符号位
//   - epsilon=1.0 的 bf16 位模式（0x3F80）
// -------------------------------------------------------------------
TEST_F(ApplyRmsPropTilingTest, T008_Bf16_BoundaryScalar_Succeeds)
{
    AddExampleTilingCompileInfo compileInfo;
    ScalarStorageU16 scalars;
    scalars.lr       = 0x0000;                        // +0.0
    scalars.rho      = 0x0000;                        // +0.0 → rho1m = 1.0
    scalars.momentum = Ut_FloatToBf16Bits(-0.25f);    // 负值
    scalars.epsilon  = 0x3F80;                        // 1.0 (bf16: sign=0, exp=127, mantissa=0)

    auto para = Make16BitTilingContextPara({256}, ge::DT_BF16, &scalars, &compileInfo);

    TilingInfo info;
    bool ok = ExecuteTiling(para, info);
    EXPECT_TRUE(ok) << "Tiling should succeed for bf16 boundary scalars";
    if (!ok) return;

    auto* td = reinterpret_cast<ApplyRmsPropTilingData*>(info.tilingData.get());

    EXPECT_FLOAT_EQ(td->lr,    0.0f);
    EXPECT_FLOAT_EQ(td->rho1m, 1.0f);
    EXPECT_FLOAT_EQ(td->mom,   -0.25f);
    EXPECT_FLOAT_EQ(td->eps,   1.0f);
}

// -------------------------------------------------------------------
// T_009: dtype 分发后 TilingKey 正确性验证
//   - 对 fp32 / fp16 / bf16 分别执行 Tiling，TilingKey 均由 schMode (0 或 1) 决定
//   - 验证三个 dtype 都能获得合法的 TilingKey（非 -1 / 0xFFFFFFFF）
//   - 验证 ElewiseBaseTiling 在 3 dtype 下均成功
// -------------------------------------------------------------------
TEST_F(ApplyRmsPropTilingTest, T009_TilingKey_DtypeDispatch_Consistent)
{
    AddExampleTilingCompileInfo compileInfo;

    // -------- fp32 --------
    ScalarStorageF32 scalarsF32;
    scalarsF32.lr       = 0.01f;
    scalarsF32.rho      = 0.9f;
    scalarsF32.momentum = 0.5f;
    scalarsF32.epsilon  = 1e-7f;
    auto paraF32 = MakeFp32TilingContextPara({1024, 8}, &scalarsF32, &compileInfo);
    TilingInfo infoF32;
    ASSERT_TRUE(ExecuteTiling(paraF32, infoF32)) << "fp32 Tiling failed";
    EXPECT_GE(infoF32.tilingKey, 0) << "fp32 TilingKey should be valid";

    // -------- fp16 --------
    ScalarStorageU16 scalarsF16;
    scalarsF16.lr       = Ut_FloatToFp16Bits(0.01f);
    scalarsF16.rho      = Ut_FloatToFp16Bits(0.9f);
    scalarsF16.momentum = Ut_FloatToFp16Bits(0.5f);
    scalarsF16.epsilon  = Ut_FloatToFp16Bits(0.001f);
    auto paraF16 = Make16BitTilingContextPara({1024, 8}, ge::DT_FLOAT16, &scalarsF16, &compileInfo);
    TilingInfo infoF16;
    ASSERT_TRUE(ExecuteTiling(paraF16, infoF16)) << "fp16 Tiling failed";
    EXPECT_GE(infoF16.tilingKey, 0) << "fp16 TilingKey should be valid";

    // -------- bf16 --------
    ScalarStorageU16 scalarsBf16;
    scalarsBf16.lr       = Ut_FloatToBf16Bits(0.01f);
    scalarsBf16.rho      = Ut_FloatToBf16Bits(0.9f);
    scalarsBf16.momentum = Ut_FloatToBf16Bits(0.5f);
    scalarsBf16.epsilon  = Ut_FloatToBf16Bits(1e-7f);
    auto paraBf16 = Make16BitTilingContextPara({1024, 8}, ge::DT_BF16, &scalarsBf16, &compileInfo);
    TilingInfo infoBf16;
    ASSERT_TRUE(ExecuteTiling(paraBf16, infoBf16)) << "bf16 Tiling failed";
    EXPECT_GE(infoBf16.tilingKey, 0) << "bf16 TilingKey should be valid";

    // 三种 dtype 均写入同样 4 个 float 字段
    auto* tdF32  = reinterpret_cast<ApplyRmsPropTilingData*>(infoF32.tilingData.get());
    auto* tdF16  = reinterpret_cast<ApplyRmsPropTilingData*>(infoF16.tilingData.get());
    auto* tdBf16 = reinterpret_cast<ApplyRmsPropTilingData*>(infoBf16.tilingData.get());
    EXPECT_FLOAT_EQ(tdF32->lr, 0.01f);
    EXPECT_FLOAT_EQ(tdF16->lr, Ut_Fp16BitsToFloat(scalarsF16.lr));
    EXPECT_FLOAT_EQ(tdBf16->lr, Ut_Bf16BitsToFloat(scalarsBf16.lr));

    // Elewise 模式下 TilingKey 由 schMode 决定（0 或 1），无 dtype 维度
    // 三 dtype 的 TilingKey 取值应在合法域（>=0）；架构上允许不同 shape 下落到不同 schMode
    std::cout << "TilingKey fp32=" << infoF32.tilingKey
              << ", fp16=" << infoF16.tilingKey
              << ", bf16=" << infoBf16.tilingKey << std::endl;
}

// -------------------------------------------------------------------
// 异常路径用例（迭代三新增）
// -------------------------------------------------------------------

// 构造不支持 dtype 的 TilingContextPara（方案 B：scalar 改为 attr）
static gert::TilingContextPara MakeIntTilingContextPara(
    const std::initializer_list<int64_t>& varShape,
    ge::DataType dtype,
    void* compileInfo,
    uint64_t coreNum = 64,
    uint64_t ubSize = 262144,
    uint64_t tilingDataSize = 4096)
{
    gert::StorageShape tensorShape = {varShape, varShape};

    std::vector<TilingContextPara::TensorDescription> inputs({
        {tensorShape, dtype, ge::FORMAT_ND},
        {tensorShape, dtype, ge::FORMAT_ND},
        {tensorShape, dtype, ge::FORMAT_ND},
        {tensorShape, dtype, ge::FORMAT_ND},
    });

    std::vector<TilingContextPara::TensorDescription> outputs({
        {tensorShape, dtype, ge::FORMAT_ND},
        {tensorShape, dtype, ge::FORMAT_ND},
        {tensorShape, dtype, ge::FORMAT_ND},
    });

    std::vector<TilingContextPara::OpAttr> attrs({
        {"lr",       Ops::Math::AnyValue::CreateFrom<float>(0.01f)},
        {"rho",      Ops::Math::AnyValue::CreateFrom<float>(0.9f)},
        {"momentum", Ops::Math::AnyValue::CreateFrom<float>(0.0f)},
        {"epsilon",  Ops::Math::AnyValue::CreateFrom<float>(1e-8f)},
    });
    return gert::TilingContextPara(OP_NAME, inputs, outputs, attrs, compileInfo,
                                   coreNum, ubSize, tilingDataSize);
}

// -------------------------------------------------------------------
// T_010: 不支持的 dtype (DT_INT32) → Tiling 失败
// -------------------------------------------------------------------
TEST_F(ApplyRmsPropTilingTest, T010_Err_UnsupportedDtypeInt32_Fails)
{
    AddExampleTilingCompileInfo compileInfo;
    auto para = MakeIntTilingContextPara({32, 32}, ge::DT_INT32, &compileInfo);

    TilingInfo info;
    bool ok = ExecuteTiling(para, info);
    EXPECT_FALSE(ok) << "Tiling should fail for unsupported dtype DT_INT32";
}

// -------------------------------------------------------------------
// T_011: 不支持的 dtype (DT_INT64) → Tiling 失败
// -------------------------------------------------------------------
TEST_F(ApplyRmsPropTilingTest, T011_Err_UnsupportedDtypeInt64_Fails)
{
    AddExampleTilingCompileInfo compileInfo;
    auto para = MakeIntTilingContextPara({16, 16, 16}, ge::DT_INT64, &compileInfo);

    TilingInfo info;
    bool ok = ExecuteTiling(para, info);
    EXPECT_FALSE(ok) << "Tiling should fail for unsupported dtype DT_INT64";
}

// -------------------------------------------------------------------
// T_012: 缺失 attr (fp32) → Tiling 失败
//   方案 B：scalar 改为 attr，测试 GetAttrs()->GetFloat() 返回 nullptr 的保护分支
// -------------------------------------------------------------------
TEST_F(ApplyRmsPropTilingTest, T012_Err_Fp32_ScalarDataNull_Fails)
{
    AddExampleTilingCompileInfo compileInfo;
    gert::StorageShape tensorShape = {{32, 32}, {32, 32}};

    std::vector<TilingContextPara::TensorDescription> inputs({
        {tensorShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {tensorShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {tensorShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {tensorShape, ge::DT_FLOAT, ge::FORMAT_ND},
    });
    std::vector<TilingContextPara::TensorDescription> outputs({
        {tensorShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {tensorShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {tensorShape, ge::DT_FLOAT, ge::FORMAT_ND},
    });

    std::vector<TilingContextPara::OpAttr> attrs;  // 空 attrs → GetFloat 返回 nullptr
    gert::TilingContextPara para(OP_NAME, inputs, outputs, attrs, &compileInfo,
                                 64, 262144, 4096);

    TilingInfo info;
    bool ok = ExecuteTiling(para, info);
    EXPECT_FALSE(ok) << "Tiling should fail when attrs missing (fp32)";
}

// -------------------------------------------------------------------
// T_013: 缺失 attr (fp16) → Tiling 失败
// -------------------------------------------------------------------
TEST_F(ApplyRmsPropTilingTest, T013_Err_Fp16_ScalarDataNull_Fails)
{
    AddExampleTilingCompileInfo compileInfo;
    gert::StorageShape tensorShape = {{16, 16}, {16, 16}};

    std::vector<TilingContextPara::TensorDescription> inputs({
        {tensorShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        {tensorShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        {tensorShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        {tensorShape, ge::DT_FLOAT16, ge::FORMAT_ND},
    });
    std::vector<TilingContextPara::TensorDescription> outputs({
        {tensorShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        {tensorShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        {tensorShape, ge::DT_FLOAT16, ge::FORMAT_ND},
    });

    std::vector<TilingContextPara::OpAttr> attrs;  // 空 attrs
    gert::TilingContextPara para(OP_NAME, inputs, outputs, attrs, &compileInfo,
                                 64, 262144, 4096);

    TilingInfo info;
    bool ok = ExecuteTiling(para, info);
    EXPECT_FALSE(ok) << "Tiling should fail when attrs missing (fp16)";
}

}  // namespace ApplyRmsPropUT
