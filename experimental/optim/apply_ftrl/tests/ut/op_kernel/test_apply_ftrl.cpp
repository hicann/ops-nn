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
 * \file test_apply_ftrl.cpp
 * \brief ApplyFtrl Kernel UT（迭代一 - 核心路径：主线 TilingKey FLOAT16,0,1）
 *
 * 在 CPU 孪生（tikicpulib）上用 ICPU_RUN_KF 直跑主线 op_kernel/apply_ftrl_kernel.h 的
 * ApplyFtrl<half, PAD_TAIL=0, HAS_L1=1>（完整 sign+软阈值+L1 gate 路径），
 * 与按 spec.yaml math_semantics.formula 实现的 fp64 CPU golden 逐元素比对。
 *
 * 核心路径选取（规避 probe 已定位的尾块 DataCopyPad>32B 风险，参 probe/PROBE_SUMMARY.md）：
 *   - totalElements = 256（64 的倍数）→ 单核单 tile，currentNum==alignedNum，rightPad=0，无 pad lane。
 *   - 良态值域：accum>0（Ln 良定义）、lr!=0、linear_t 远离 0 且 |linear_t|>l1（gate 全过、规避近零脆弱）。
 *
 * 三输出：var→var_out；accum/linear 原地写回各自输入 GM（Ref Tensor）。
 *
 * ST shape 桶覆盖意图（迁移自已移除的 tests/st/testcases/aclnnApplyFtrl_l{0,1,2}_test_cases.csv，B1）：
 *   黑盒用例集覆盖的代表性 shape 桶/边界在本 UT + tests/st/aclnnApplyFtrl/atk_ApplyFtrl.json + op_host
 *   tiling UT 中复现 ——
 *     · 标量 0-D `()` vs 1 元素 1-D `[1]`：atk_ApplyFtrl.json 各 case 的 lr/l1/l2/lrPower 两形态混排；
 *     · 32B 对齐 vs 非对齐尾块：本 UT T006/T008/T009/T010/T012/T013（N=100）+ T019（N=200 多 tile 尾）；
 *     · 大/多维 shape 桶（如 [34,374]、[512,1,24]、rank-8 [32,2,2,1,32,2,1,7]、[1,288,26,1,5]）：
 *       逐元素无跨核归约，数值语义与小 shape 同构 → 由 op_host tiling UT T026(rank-8)/T021-T023/T031
 *       覆盖多核切分，本 UT 覆盖单核计算，二者合并等价于大 shape 桶；
 *     · 空 Tensor [0,3]：op_host tiling UT T004/T020/T030 短路；
 *     · 值域（accum∈[0.5,4]、lr∈[0.5,2]、l1/l2∈[0,0.5]、lr_power∈[0.3,0.8]）：FillGoodDomain 良态值域同口径，
 *       极端值（NaN grad / accum_new=0 / lr=0）见 T014–T016。
 */

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <vector>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#endif

// CPU 孪生（ASCENDC_TPL_KERNEL + ASCENDC_DEBUG）下 tiling_key.h 的 ASCENDC_TPL_DATATYPE_*
// 会展开为引用 C_DT_FLOAT16/C_DT_BF16/C_DT_FLOAT 的 ParamStruct，需先引入这些枚举（轻量纯枚举头）。
#include "graph/c_types.h"

// GET_TILING_DATA_WITH_STRUCT 在真实 kernel 编译时由 tiling 代码生成（get_op_tiling.py）产出，
// standalone CPU 孪生 UT 无该 codegen。ApplyFtrlTilingData 为纯 C++ struct，按字节拷贝即可
// 还原（等价于生成版的 InitTilingData<T> memcpy 语义）。REGISTER_TILING_DEFAULT 由 asc 头提供。
#ifndef GET_TILING_DATA_WITH_STRUCT
#define GET_TILING_DATA_WITH_STRUCT(StructT, name, arg)                  \
    StructT name;                                                        \
    (void)memcpy(reinterpret_cast<void*>(&(name)),                       \
                 reinterpret_cast<const void*>(arg), sizeof(StructT))
#endif

// 主线 kernel（带入 ApplyFtrl<T,PAD_TAIL,HAS_L1> 模板与 ApplyFtrlTilingData）。
#include "../../../op_kernel/apply_ftrl.cpp"

using namespace std;

namespace {

constexpr int64_t N = 256;  // 64 的倍数 → 单 tile 无尾块 pad

// ---- spec.yaml math_semantics.formula 的 fp64 CPU golden（逐元素）----
struct GoldenOut {
    double var;
    double accum;   // accum_new
    double linear;  // linear_t
};

static GoldenOut ComputeGolden(double var, double accum, double linear, double grad,
                               double lr, double l1, double l2, double lr_power)
{
    double accum_new = accum + grad * grad;
    double pow_new = std::pow(accum_new, -lr_power);
    double pow_old = std::pow(accum, -lr_power);
    double linear_t = linear + grad - ((pow_new - pow_old) / lr) * var;
    double quadratic = pow_new / lr + 2.0 * l2;
    double sgn = (linear_t > 0.0) ? 1.0 : ((linear_t < 0.0) ? -1.0 : 0.0);
    double x_res = sgn * l1 - linear_t;
    double var_new = (std::fabs(linear_t) > l1) ? (x_res / quadratic) : 0.0;
    return {var_new, accum_new, linear_t};
}

// fp16 往返（kernel 实际读入的是 fp16 值，golden 用同一 fp16-rounded 值计算，
// 只比对 kernel 的计算+输出舍入误差）。
static inline float ToFp16f(float v) { return static_cast<float>(static_cast<half>(v)); }

class ApplyFtrlKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "ApplyFtrlKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "ApplyFtrlKernelTest TearDown" << endl; }
};

TEST_F(ApplyFtrlKernelTest, T001_CorePath_Fp16_HasL1_Aligned)
{
    const size_t tElems = static_cast<size_t>(N);
    const size_t tBytes = tElems * sizeof(half);
    const size_t sBytes = sizeof(half);  // scalar: 1 element

    // ---- GM 分配 ----
    uint8_t* var = (uint8_t*)AscendC::GmAlloc(tBytes);
    uint8_t* accum = (uint8_t*)AscendC::GmAlloc(tBytes);
    uint8_t* linear = (uint8_t*)AscendC::GmAlloc(tBytes);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(tBytes);
    uint8_t* lr = (uint8_t*)AscendC::GmAlloc(sBytes);
    uint8_t* l1 = (uint8_t*)AscendC::GmAlloc(sBytes);
    uint8_t* l2 = (uint8_t*)AscendC::GmAlloc(sBytes);
    uint8_t* lr_power = (uint8_t*)AscendC::GmAlloc(sBytes);
    uint8_t* var_out = (uint8_t*)AscendC::GmAlloc(tBytes);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ApplyFtrlTilingData));

    // ---- scalar（良态：lr!=0，l1>0 触发完整 sign+gate）----
    const float lrF = 0.1f, l1F = 0.01f, l2F = 0.05f, lrPowerF = 0.5f;
    *reinterpret_cast<half*>(lr) = static_cast<half>(lrF);
    *reinterpret_cast<half*>(l1) = static_cast<half>(l1F);
    *reinterpret_cast<half*>(l2) = static_cast<half>(l2F);
    *reinterpret_cast<half*>(lr_power) = static_cast<half>(lrPowerF);

    // ---- 张量输入（确定性、良态值域；linear+grad>0 且 |linear_t|>l1，gate 全过）----
    auto* varH = reinterpret_cast<half*>(var);
    auto* accumH = reinterpret_cast<half*>(accum);
    auto* linearH = reinterpret_cast<half*>(linear);
    auto* gradH = reinterpret_cast<half*>(grad);
    std::vector<float> varIn(tElems), accumIn(tElems), linearIn(tElems), gradIn(tElems);
    for (size_t i = 0; i < tElems; ++i) {
        varIn[i] = ToFp16f(-0.8f + 1.6f * static_cast<float>(i % 11) / 10.0f);  // [-0.8, 0.8]
        accumIn[i] = ToFp16f(0.5f + 1.0f * static_cast<float>(i % 7) / 6.0f);   // [0.5, 1.5] > 0
        linearIn[i] = ToFp16f(0.4f + 0.5f * static_cast<float>(i % 5) / 4.0f);  // [0.4, 0.9] > 0
        gradIn[i] = ToFp16f(-0.2f + 0.4f * static_cast<float>(i % 3) / 2.0f);   // [-0.2, 0.2]
        varH[i] = static_cast<half>(varIn[i]);
        accumH[i] = static_cast<half>(accumIn[i]);
        linearH[i] = static_cast<half>(linearIn[i]);
        gradH[i] = static_cast<half>(gradIn[i]);
    }

    // ---- TilingData：单核单 tile，无尾块 ----
    auto* td = reinterpret_cast<ApplyFtrlTilingData*>(tiling);
    td->totalElements = N;
    td->blockFactor = N;
    td->ubFactor = N;

    // ---- 运行主线 kernel：ApplyFtrl<half, PAD_TAIL=0, HAS_L1=1> ----
    auto kernel = [](GM_ADDR var, GM_ADDR accum, GM_ADDR linear, GM_ADDR grad, GM_ADDR lr,
                     GM_ADDR l1, GM_ADDR l2, GM_ADDR lr_power, GM_ADDR var_out, GM_ADDR ws,
                     GM_ADDR tl) {
        ::apply_ftrl<half, 0U, 1U>(var, accum, linear, grad, lr, l1, l2, lr_power, var_out, ws, tl);
    };
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1, var, accum, linear, grad, lr, l1, l2, lr_power, var_out, workspace, tiling);

    // ---- 比对（var_out / accum 原地 / linear 原地）----
    const double lrD = ToFp16f(lrF), l1D = ToFp16f(l1F), l2D = ToFp16f(l2F),
                 lrPowerD = ToFp16f(lrPowerF);
    // fp16：尾数 10bit，相对精度 ~2^-11≈4.9e-4；叠加 fp32 内算 + 末尾舍入，取保守组合容差。
    const double rtol = 4e-3, atol = 4e-3;

    auto* varOutH = reinterpret_cast<half*>(var_out);
    auto* accumOutH = reinterpret_cast<half*>(accum);    // 原地写回
    auto* linearOutH = reinterpret_cast<half*>(linear);  // 原地写回

    int gatePass = 0;
    for (size_t i = 0; i < tElems; ++i) {
        GoldenOut g = ComputeGolden(varIn[i], accumIn[i], linearIn[i], gradIn[i], lrD, l1D, l2D,
                                    lrPowerD);
        if (std::fabs(g.linear) > l1D) gatePass++;

        double ov = static_cast<float>(varOutH[i]);
        double oa = static_cast<float>(accumOutH[i]);
        double ol = static_cast<float>(linearOutH[i]);

        EXPECT_LE(std::fabs(ov - g.var), atol + rtol * std::fabs(g.var))
            << "var mismatch at i=" << i << " got=" << ov << " golden=" << g.var;
        EXPECT_LE(std::fabs(oa - g.accum), atol + rtol * std::fabs(g.accum))
            << "accum mismatch at i=" << i << " got=" << oa << " golden=" << g.accum;
        EXPECT_LE(std::fabs(ol - g.linear), atol + rtol * std::fabs(g.linear))
            << "linear mismatch at i=" << i << " got=" << ol << " golden=" << g.linear;

        // invariant: accum_out 非负（spec invariants.accum_out_nonneg）
        EXPECT_GE(oa, 0.0) << "accum_out must be non-negative at i=" << i;
    }
    EXPECT_EQ(gatePass, static_cast<int>(tElems)) << "data should drive the full sign+gate path";

    AscendC::GmFree(var);
    AscendC::GmFree(accum);
    AscendC::GmFree(linear);
    AscendC::GmFree(grad);
    AscendC::GmFree(lr);
    AscendC::GmFree(l1);
    AscendC::GmFree(l2);
    AscendC::GmFree(lr_power);
    AscendC::GmFree(var_out);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// ===================================================================
// 迭代二：通用 dtype × HAS_L1 × PAD_TAIL kernel 运行器
//   覆盖代表性 TilingKey：fp16 / fp32 / bf16 各 1 + HAS_L1=0 编译期快路 + l1==0 运行期快路
//   + PAD_TAIL=1 两级尾块对齐。三输出（var→var_out；accum/linear 原地）对 spec fp64 golden 比对。
//   近零脆弱规避（PROBE_SUMMARY #5）：数据良态（|linear_t|>l1、accum>0），且组合容差
//   |got-golden| <= atol + rtol*|golden| 的 atol 项对近零元素自动走绝对容差。
// ===================================================================

// float <-> T 往返（kernel 实际读入的是 T 舍入值；golden 用同一舍入值计算，只比对计算+输出舍入误差）。
template <typename T>
static inline T FromF(float v)
{
    if constexpr (std::is_same<T, float>::value) {
        return v;
    } else {
        return static_cast<T>(v);  // half / bfloat16_t 均支持 float 构造（CPU 孪生）
    }
}
template <typename T>
static inline float ToF(T v) { return static_cast<float>(v); }
template <typename T>
static inline float RoundF(float v) { return ToF<T>(FromF<T>(v)); }

// 通用运行器。l1GmF 写入 GM（kernel 读到的 l1）；l1GoldenF 用于 golden（HAS_L1=0 时 kernel
// 静态丢弃 l1 分支，golden 须以 l1=0 计算 → 以 l1GmF != l1GoldenF 证明 l1 被忽略）。
template <typename T, uint32_t PAD_TAIL, uint32_t HAS_L1>
static void RunFtrlCase(int64_t N, float lrF, float l1GmF, float l2F, float lrPowerF,
                        float l1GoldenF, double rtol, double atol)
{
    const size_t tElems = static_cast<size_t>(N);
    const size_t tBytes = tElems * sizeof(T);
    const size_t sBytes = sizeof(T);

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(tBytes);
    uint8_t* accum = (uint8_t*)AscendC::GmAlloc(tBytes);
    uint8_t* linear = (uint8_t*)AscendC::GmAlloc(tBytes);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(tBytes);
    uint8_t* lr = (uint8_t*)AscendC::GmAlloc(sBytes);
    uint8_t* l1 = (uint8_t*)AscendC::GmAlloc(sBytes);
    uint8_t* l2 = (uint8_t*)AscendC::GmAlloc(sBytes);
    uint8_t* lr_power = (uint8_t*)AscendC::GmAlloc(sBytes);
    uint8_t* var_out = (uint8_t*)AscendC::GmAlloc(tBytes);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ApplyFtrlTilingData));

    *reinterpret_cast<T*>(lr) = FromF<T>(lrF);
    *reinterpret_cast<T*>(l1) = FromF<T>(l1GmF);
    *reinterpret_cast<T*>(l2) = FromF<T>(l2F);
    *reinterpret_cast<T*>(lr_power) = FromF<T>(lrPowerF);

    auto* varT = reinterpret_cast<T*>(var);
    auto* accumT = reinterpret_cast<T*>(accum);
    auto* linearT = reinterpret_cast<T*>(linear);
    auto* gradT = reinterpret_cast<T*>(grad);
    std::vector<float> varIn(tElems), accumIn(tElems), linearIn(tElems), gradIn(tElems);
    for (size_t i = 0; i < tElems; ++i) {
        varIn[i] = RoundF<T>(-0.8f + 1.6f * static_cast<float>(i % 11) / 10.0f);  // [-0.8, 0.8]
        accumIn[i] = RoundF<T>(0.5f + 1.0f * static_cast<float>(i % 7) / 6.0f);   // [0.5, 1.5] > 0
        linearIn[i] = RoundF<T>(0.4f + 0.5f * static_cast<float>(i % 5) / 4.0f);  // [0.4, 0.9] > 0
        gradIn[i] = RoundF<T>(-0.2f + 0.4f * static_cast<float>(i % 3) / 2.0f);   // [-0.2, 0.2]
        varT[i] = FromF<T>(varIn[i]);
        accumT[i] = FromF<T>(accumIn[i]);
        linearT[i] = FromF<T>(linearIn[i]);
        gradT[i] = FromF<T>(gradIn[i]);
    }

    auto* td = reinterpret_cast<ApplyFtrlTilingData*>(tiling);
    td->totalElements = N;
    td->blockFactor = N;                       // 单核
    td->ubFactor = ((N + 63) / 64) * 64;       // 64 元素(256B)对齐，单 tile 容纳 N

    auto kernel = [](GM_ADDR var, GM_ADDR accum, GM_ADDR linear, GM_ADDR grad, GM_ADDR lr,
                     GM_ADDR l1, GM_ADDR l2, GM_ADDR lr_power, GM_ADDR var_out, GM_ADDR ws,
                     GM_ADDR tl) {
        ::apply_ftrl<T, PAD_TAIL, HAS_L1>(var, accum, linear, grad, lr, l1, l2, lr_power, var_out,
                                          ws, tl);
    };
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1, var, accum, linear, grad, lr, l1, l2, lr_power, var_out, workspace,
                tiling);

    const double lrD = RoundF<T>(lrF), l1D = RoundF<T>(l1GoldenF), l2D = RoundF<T>(l2F),
                 lrPowerD = RoundF<T>(lrPowerF);
    auto* varOutT = reinterpret_cast<T*>(var_out);
    auto* accumOutT = reinterpret_cast<T*>(accum);    // 原地写回
    auto* linearOutT = reinterpret_cast<T*>(linear);  // 原地写回

    for (size_t i = 0; i < tElems; ++i) {
        GoldenOut g = ComputeGolden(varIn[i], accumIn[i], linearIn[i], gradIn[i], lrD, l1D, l2D,
                                    lrPowerD);
        double ov = ToF<T>(varOutT[i]);
        double oa = ToF<T>(accumOutT[i]);
        double ol = ToF<T>(linearOutT[i]);

        EXPECT_LE(std::fabs(ov - g.var), atol + rtol * std::fabs(g.var))
            << "var mismatch at i=" << i << " got=" << ov << " golden=" << g.var;
        EXPECT_LE(std::fabs(oa - g.accum), atol + rtol * std::fabs(g.accum))
            << "accum mismatch at i=" << i << " got=" << oa << " golden=" << g.accum;
        EXPECT_LE(std::fabs(ol - g.linear), atol + rtol * std::fabs(g.linear))
            << "linear mismatch at i=" << i << " got=" << ol << " golden=" << g.linear;
        EXPECT_GE(oa, 0.0) << "accum_out must be non-negative at i=" << i;  // spec invariant
    }

    AscendC::GmFree(var);
    AscendC::GmFree(accum);
    AscendC::GmFree(linear);
    AscendC::GmFree(grad);
    AscendC::GmFree(lr);
    AscendC::GmFree(l1);
    AscendC::GmFree(l2);
    AscendC::GmFree(lr_power);
    AscendC::GmFree(var_out);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// 组合容差（元素级，CPU 孪生 fp32 内算 + 末尾 cast；远松于 spec MERE 阈值的多算子叠加裕量，
// 但足以捕获公式错误 → 错误公式误差 >> 这些阈值）。近零元素由 atol 项兜底（PROBE #5）。
constexpr double kRtolFp16 = 5e-3, kAtolFp16 = 5e-3;
constexpr double kRtolBf16 = 2e-2, kAtolBf16 = 2e-2;
constexpr double kRtolFp32 = 1e-3, kAtolFp32 = 1e-4;

// T002: fp32 完整路径（HAS_L1=1, l1>0），对齐 N=256。fp32 ReinterpretCast 直算。
TEST_F(ApplyFtrlKernelTest, T002_CorePath_Fp32_HasL1_Aligned)
{
    RunFtrlCase<float, 0U, 1U>(256, 0.1f, 0.01f, 0.05f, 0.5f, /*l1Golden=*/0.01f, kRtolFp32,
                               kAtolFp32);
}

// T003: bf16 完整路径（HAS_L1=1, l1>0），对齐 N=256。标量借道 Cast（DESIGN s3.5）。
TEST_F(ApplyFtrlKernelTest, T003_CorePath_Bf16_HasL1_Aligned)
{
    RunFtrlCase<bfloat16_t, 0U, 1U>(256, 0.1f, 0.01f, 0.05f, 0.5f, /*l1Golden=*/0.01f, kRtolBf16,
                                    kAtolBf16);
}

// T004: HAS_L1=0 编译期快路（var=-linear_t/quadratic）。GM 写入 l1=0.5（非零），golden 以 l1=0
//   计算 → 证明 HAS_L1=0 静态丢弃了 sign+软阈值+gate 分支、完全忽略 l1。fp32, N=256。
TEST_F(ApplyFtrlKernelTest, T004_FastPath_Fp32_HasL1_0_IgnoresL1)
{
    RunFtrlCase<float, 0U, 0U>(256, 0.1f, /*l1Gm=*/0.5f, 0.05f, 0.5f, /*l1Golden=*/0.0f, kRtolFp32,
                               kAtolFp32);
}

// T005: HAS_L1=1 但 l1==0 → 运行期快路（var=-linear_t/quadratic）。fp16, N=256。
TEST_F(ApplyFtrlKernelTest, T005_FastPath_Fp16_RuntimeL1Zero)
{
    RunFtrlCase<half, 0U, 1U>(256, 0.1f, /*l1Gm=*/0.0f, 0.05f, 0.5f, /*l1Golden=*/0.0f, kRtolFp16,
                              kAtolFp16);
}

// T006: PAD_TAIL=1 非 32B 对齐尾块（N=100）→ 两级尾块对齐（32B DMA pad + fp32 域 Duplicate 补齐
//   [nDma,nComp)）。仅前 100 个真实元素写回并比对。fp32, HAS_L1=1。
TEST_F(ApplyFtrlKernelTest, T006_Tail_Fp32_HasL1_TwoLevelAlign)
{
    RunFtrlCase<float, 1U, 1U>(100, 0.1f, 0.01f, 0.05f, 0.5f, /*l1Golden=*/0.01f, kRtolFp32,
                               kAtolFp32);
}

// ===================================================================
// 迭代三：op_kernel 全覆盖补齐
//   (A) 12 TilingKey 组合（dtype × PAD_TAIL × HAS_L1）补齐剩余 7 格（复用 RunFtrlCase）；
//   (B) 极端值（NaN 传播 / accum_new=0&lr_power>0 / lr=0）Inf/NaN 位置比对；
//   (C) 确定性（重复执行字节一致）；
//   (D) 三输出原地写回 ABI（var→var_out；accum/linear 原地；var 输入端口不被改写）；
//   (E) 多 tile 主循环（blockLen > ubFactor，含非对齐尾 tile）。
// 三输出比对用 spec 组合 rtol+atol（近零绝对容差，PROBE_SUMMARY #5）。
// ===================================================================

// ---- (A) 12 组合补齐 ----
//   已覆盖（迭代二，作为模板实例化）：(half,0,1)T001 (float,0,1)T002 (bf16,0,1)T003
//                                   (float,0,0)T004 (float,1,1)T006。
//   PAD_TAIL=1 组合用 N=100（非 32B 对齐）真实驱动两级尾块对齐；PAD_TAIL=0 用 N=256。
//   HAS_L1=0 组合：GM 写 l1=0.5 但 golden 以 l1=0 计算 → 证明静态丢弃 sign+gate+忽略 l1。

TEST_F(ApplyFtrlKernelTest, T007_Combo_Fp16_Pad0_L0)
{
    RunFtrlCase<half, 0U, 0U>(256, 0.1f, /*l1Gm=*/0.5f, 0.05f, 0.5f, /*l1Golden=*/0.0f, kRtolFp16,
                              kAtolFp16);
}
TEST_F(ApplyFtrlKernelTest, T008_Combo_Fp16_Pad1_L1)
{
    RunFtrlCase<half, 1U, 1U>(100, 0.1f, 0.01f, 0.05f, 0.5f, /*l1Golden=*/0.01f, kRtolFp16,
                              kAtolFp16);
}
TEST_F(ApplyFtrlKernelTest, T009_Combo_Fp16_Pad1_L0)
{
    RunFtrlCase<half, 1U, 0U>(100, 0.1f, /*l1Gm=*/0.5f, 0.05f, 0.5f, /*l1Golden=*/0.0f, kRtolFp16,
                              kAtolFp16);
}
TEST_F(ApplyFtrlKernelTest, T010_Combo_Fp32_Pad1_L0)
{
    RunFtrlCase<float, 1U, 0U>(100, 0.1f, /*l1Gm=*/0.5f, 0.05f, 0.5f, /*l1Golden=*/0.0f, kRtolFp32,
                               kAtolFp32);
}
TEST_F(ApplyFtrlKernelTest, T011_Combo_Bf16_Pad0_L0)
{
    RunFtrlCase<bfloat16_t, 0U, 0U>(256, 0.1f, /*l1Gm=*/0.5f, 0.05f, 0.5f, /*l1Golden=*/0.0f,
                                    kRtolBf16, kAtolBf16);
}
TEST_F(ApplyFtrlKernelTest, T012_Combo_Bf16_Pad1_L1)
{
    RunFtrlCase<bfloat16_t, 1U, 1U>(100, 0.1f, 0.01f, 0.05f, 0.5f, /*l1Golden=*/0.01f, kRtolBf16,
                                    kAtolBf16);
}
TEST_F(ApplyFtrlKernelTest, T013_Combo_Bf16_Pad1_L0)
{
    RunFtrlCase<bfloat16_t, 1U, 0U>(100, 0.1f, /*l1Gm=*/0.5f, 0.05f, 0.5f, /*l1Golden=*/0.0f,
                                    kRtolBf16, kAtolBf16);
}

// ---- 低层启动器：任意输入数组 + 标量，运行主线 kernel，回收三输出 + var 输入端口现值 ----
//   varInputAfter 用于验证 var 输入 GM 未被改写（var 经 var_out 输出，非原地写回输入）。
//   ubFactorOverride>0 时覆盖默认单 tile ubFactor，用于多 tile 主循环测试。
template <typename T, uint32_t PAD_TAIL, uint32_t HAS_L1>
static void LaunchFtrl(int64_t N, const std::vector<float>& varIn, const std::vector<float>& accumIn,
                       const std::vector<float>& linearIn, const std::vector<float>& gradIn,
                       float lrF, float l1F, float l2F, float lrPowerF, std::vector<T>& varOut,
                       std::vector<T>& accumOut, std::vector<T>& linearOut,
                       std::vector<T>& varInputAfter, int64_t ubFactorOverride = -1)
{
    const size_t tElems = static_cast<size_t>(N);
    const size_t tBytes = tElems * sizeof(T);
    const size_t sBytes = sizeof(T);

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(tBytes);
    uint8_t* accum = (uint8_t*)AscendC::GmAlloc(tBytes);
    uint8_t* linear = (uint8_t*)AscendC::GmAlloc(tBytes);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(tBytes);
    uint8_t* lr = (uint8_t*)AscendC::GmAlloc(sBytes);
    uint8_t* l1 = (uint8_t*)AscendC::GmAlloc(sBytes);
    uint8_t* l2 = (uint8_t*)AscendC::GmAlloc(sBytes);
    uint8_t* lr_power = (uint8_t*)AscendC::GmAlloc(sBytes);
    uint8_t* var_out = (uint8_t*)AscendC::GmAlloc(tBytes);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ApplyFtrlTilingData));

    *reinterpret_cast<T*>(lr) = FromF<T>(lrF);
    *reinterpret_cast<T*>(l1) = FromF<T>(l1F);
    *reinterpret_cast<T*>(l2) = FromF<T>(l2F);
    *reinterpret_cast<T*>(lr_power) = FromF<T>(lrPowerF);

    auto* varT = reinterpret_cast<T*>(var);
    auto* accumT = reinterpret_cast<T*>(accum);
    auto* linearT = reinterpret_cast<T*>(linear);
    auto* gradT = reinterpret_cast<T*>(grad);
    for (size_t i = 0; i < tElems; ++i) {
        varT[i] = FromF<T>(varIn[i]);
        accumT[i] = FromF<T>(accumIn[i]);
        linearT[i] = FromF<T>(linearIn[i]);
        gradT[i] = FromF<T>(gradIn[i]);
    }

    auto* td = reinterpret_cast<ApplyFtrlTilingData*>(tiling);
    td->totalElements = N;
    td->blockFactor = N;  // single core
    td->ubFactor = (ubFactorOverride > 0) ? ubFactorOverride : (((N + 63) / 64) * 64);

    auto kernel = [](GM_ADDR var, GM_ADDR accum, GM_ADDR linear, GM_ADDR grad, GM_ADDR lr,
                     GM_ADDR l1, GM_ADDR l2, GM_ADDR lr_power, GM_ADDR var_out, GM_ADDR ws,
                     GM_ADDR tl) {
        ::apply_ftrl<T, PAD_TAIL, HAS_L1>(var, accum, linear, grad, lr, l1, l2, lr_power, var_out,
                                          ws, tl);
    };
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1, var, accum, linear, grad, lr, l1, l2, lr_power, var_out, workspace,
                tiling);

    varOut.resize(tElems);
    accumOut.resize(tElems);
    linearOut.resize(tElems);
    varInputAfter.resize(tElems);
    auto* varOutT = reinterpret_cast<T*>(var_out);
    auto* accumInplaceT = reinterpret_cast<T*>(accum);    // 原地写回端口
    auto* linearInplaceT = reinterpret_cast<T*>(linear);  // 原地写回端口
    auto* varInputT = reinterpret_cast<T*>(var);          // var 输入端口（应保持不变）
    for (size_t i = 0; i < tElems; ++i) {
        varOut[i] = varOutT[i];
        accumOut[i] = accumInplaceT[i];
        linearOut[i] = linearInplaceT[i];
        varInputAfter[i] = varInputT[i];
    }

    AscendC::GmFree(var);
    AscendC::GmFree(accum);
    AscendC::GmFree(linear);
    AscendC::GmFree(grad);
    AscendC::GmFree(lr);
    AscendC::GmFree(l1);
    AscendC::GmFree(l2);
    AscendC::GmFree(lr_power);
    AscendC::GmFree(var_out);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// 良态值域输入填充（与 RunFtrlCase 同款；accum>0、|linear_t|>l1、gate 全过）。
static void FillGoodDomain(int64_t N, std::vector<float>& v, std::vector<float>& a,
                           std::vector<float>& l, std::vector<float>& g)
{
    v.resize(N);
    a.resize(N);
    l.resize(N);
    g.resize(N);
    for (int64_t i = 0; i < N; ++i) {
        v[i] = -0.8f + 1.6f * static_cast<float>(i % 11) / 10.0f;  // [-0.8, 0.8]
        a[i] = 0.5f + 1.0f * static_cast<float>(i % 7) / 6.0f;     // [0.5, 1.5] > 0
        l[i] = 0.4f + 0.5f * static_cast<float>(i % 5) / 4.0f;     // [0.4, 0.9] > 0
        g[i] = -0.2f + 0.4f * static_cast<float>(i % 3) / 2.0f;    // [-0.2, 0.2]
    }
}

// 按 IEEE 类别（finite / Inf+符号 / NaN）逐元素比对（极端值场景，PROBE_SUMMARY #4）。
static void ExpectClass(double got, double golden, double rtol, double atol, const char* tag,
                        size_t i)
{
    if (std::isnan(golden)) {
        EXPECT_TRUE(std::isnan(got)) << tag << " expected NaN at i=" << i << " got=" << got;
    } else if (std::isinf(golden)) {
        EXPECT_TRUE(std::isinf(got)) << tag << " expected Inf at i=" << i << " got=" << got;
        if (std::isinf(got)) {
            EXPECT_EQ(std::signbit(got), std::signbit(golden))
                << tag << " Inf sign mismatch at i=" << i;
        }
    } else {
        EXPECT_TRUE(std::isfinite(got)) << tag << " expected finite at i=" << i << " got=" << got;
        if (std::isfinite(got)) {
            EXPECT_LE(std::fabs(got - golden), atol + rtol * std::fabs(golden))
                << tag << " mismatch at i=" << i << " got=" << got << " golden=" << golden;
        }
    }
}

// ---- (B) 极端值 ----

// T014: grad 含 NaN → accum_new/linear 沿元素污染为 NaN；var 经 gate(|NaN|>l1=false) 归 0（finite）。
//   逐元素与 fp64 golden 类别一致（spec extreme_inputs「NaN grad 传播」）。fp32, N=64。
TEST_F(ApplyFtrlKernelTest, T014_Extreme_NanGrad_Propagates_Fp32)
{
    const int64_t N = 64;
    std::vector<float> varIn, accumIn, linearIn, gradIn;
    FillGoodDomain(N, varIn, accumIn, linearIn, gradIn);
    const std::vector<size_t> nanIdx = {3, 17, 50};
    for (size_t k : nanIdx) gradIn[k] = std::nanf("");

    const float lr = 0.1f, l1 = 0.01f, l2 = 0.05f, lrPower = 0.5f;
    std::vector<float> vO, aO, lO, vAfter;
    LaunchFtrl<float, 0U, 1U>(N, varIn, accumIn, linearIn, gradIn, lr, l1, l2, lrPower, vO, aO, lO,
                              vAfter);

    int nanCount = 0;
    for (size_t i = 0; i < static_cast<size_t>(N); ++i) {
        GoldenOut g =
            ComputeGolden(varIn[i], accumIn[i], linearIn[i], gradIn[i], lr, l1, l2, lrPower);
        ExpectClass(aO[i], g.accum, kRtolFp32, kAtolFp32, "accum", i);
        ExpectClass(lO[i], g.linear, kRtolFp32, kAtolFp32, "linear", i);
        ExpectClass(vO[i], g.var, kRtolFp32, kAtolFp32, "var", i);
        if (std::isnan(g.accum)) {
            ++nanCount;
            EXPECT_TRUE(std::isnan(aO[i])) << "accum NaN must propagate at i=" << i;
            EXPECT_TRUE(std::isnan(lO[i])) << "linear NaN must propagate at i=" << i;
        } else {
            EXPECT_GE(aO[i], 0.0) << "accum_out non-negative invariant at i=" << i;
        }
    }
    EXPECT_EQ(nanCount, static_cast<int>(nanIdx.size())) << "NaN must land exactly at injected idx";
}

// T015: accum=0 & grad=0 & lr_power=0.5 → accum_new=0, pow=0^(-0.5)=Inf, Inf-Inf=NaN → linear=NaN；
//   accum_out=0(finite), var 经 gate 归 0(finite)（spec extreme_inputs「accum_new=0 & lr_power>0」）。fp32, N=64。
TEST_F(ApplyFtrlKernelTest, T015_Extreme_AccumZero_LrPowerPos_Fp32)
{
    const int64_t N = 64;
    std::vector<float> varIn, accumIn, linearIn, gradIn;
    FillGoodDomain(N, varIn, accumIn, linearIn, gradIn);
    for (int64_t i = 0; i < N; ++i) {
        accumIn[i] = 0.0f;
        gradIn[i] = 0.0f;
    }
    const float lr = 0.1f, l1 = 0.01f, l2 = 0.05f, lrPower = 0.5f;  // lr_power > 0
    std::vector<float> vO, aO, lO, vAfter;
    LaunchFtrl<float, 0U, 1U>(N, varIn, accumIn, linearIn, gradIn, lr, l1, l2, lrPower, vO, aO, lO,
                              vAfter);
    for (size_t i = 0; i < static_cast<size_t>(N); ++i) {
        EXPECT_TRUE(std::isfinite(aO[i])) << "accum_new=0 stays finite at i=" << i;
        EXPECT_NEAR(aO[i], 0.0, 1e-6) << "accum_new must be 0 at i=" << i;
        EXPECT_TRUE(std::isnan(lO[i])) << "0^(-lr_power) singularity must yield NaN linear at i=" << i;
        EXPECT_TRUE(std::isfinite(vO[i])) << "var gated to finite 0 at i=" << i;
    }
}

// T016: lr=0（除零）→ invLr=1/0=Inf 污染 linear/var（非 finite）；accum_new=accum+g² 不受 lr 影响仍 finite。
//   （spec extreme_inputs「lr=0」produces_nan）。fp32, N=64。
TEST_F(ApplyFtrlKernelTest, T016_Extreme_LrZero_Fp32)
{
    const int64_t N = 64;
    std::vector<float> varIn, accumIn, linearIn, gradIn;
    FillGoodDomain(N, varIn, accumIn, linearIn, gradIn);
    const float lr = 0.0f, l1 = 0.01f, l2 = 0.05f, lrPower = 0.5f;
    std::vector<float> vO, aO, lO, vAfter;
    LaunchFtrl<float, 0U, 1U>(N, varIn, accumIn, linearIn, gradIn, lr, l1, l2, lrPower, vO, aO, lO,
                              vAfter);
    for (size_t i = 0; i < static_cast<size_t>(N); ++i) {
        double accumNew = static_cast<double>(accumIn[i]) + static_cast<double>(gradIn[i]) * gradIn[i];
        EXPECT_TRUE(std::isfinite(aO[i])) << "accum_new is lr-independent, must stay finite at i=" << i;
        EXPECT_LE(std::fabs(aO[i] - accumNew), kAtolFp32 + kRtolFp32 * std::fabs(accumNew))
            << "accum_new at i=" << i;
        EXPECT_GE(aO[i], 0.0) << "accum_out non-negative invariant at i=" << i;
        EXPECT_FALSE(std::isfinite(lO[i])) << "lr=0 must make linear non-finite (Inf/NaN) at i=" << i;
    }
}

// ---- (C) 确定性：重复执行字节一致（spec determinism.bitwise_reproducible）----
TEST_F(ApplyFtrlKernelTest, T017_Determinism_ByteIdentical_Fp16)
{
    const int64_t N = 256;
    std::vector<float> varIn, accumIn, linearIn, gradIn;
    FillGoodDomain(N, varIn, accumIn, linearIn, gradIn);
    const float lr = 0.1f, l1 = 0.01f, l2 = 0.05f, lrPower = 0.5f;

    std::vector<half> vO1, aO1, lO1, vA1, vO2, aO2, lO2, vA2;
    LaunchFtrl<half, 0U, 1U>(N, varIn, accumIn, linearIn, gradIn, lr, l1, l2, lrPower, vO1, aO1, lO1,
                             vA1);
    LaunchFtrl<half, 0U, 1U>(N, varIn, accumIn, linearIn, gradIn, lr, l1, l2, lrPower, vO2, aO2, lO2,
                             vA2);
    const size_t nbytes = static_cast<size_t>(N) * sizeof(half);
    EXPECT_EQ(std::memcmp(vO1.data(), vO2.data(), nbytes), 0) << "var_out must be bitwise reproducible";
    EXPECT_EQ(std::memcmp(aO1.data(), aO2.data(), nbytes), 0) << "accum_out must be bitwise reproducible";
    EXPECT_EQ(std::memcmp(lO1.data(), lO2.data(), nbytes), 0) << "linear_out must be bitwise reproducible";
}

// ---- (D) 三输出原地写回 ABI（DESIGN s3.4）----
//   var → var_out（独立端口）；accum/linear → 各自输入 GM 原地；var 输入端口不被改写。
TEST_F(ApplyFtrlKernelTest, T018_InPlaceWriteBack_ThreeOutputs_Fp32)
{
    const int64_t N = 64;
    std::vector<float> varIn, accumIn, linearIn, gradIn;
    FillGoodDomain(N, varIn, accumIn, linearIn, gradIn);
    const float lr = 0.1f, l1 = 0.01f, l2 = 0.05f, lrPower = 0.5f;

    std::vector<float> vO, aO, lO, vAfter;
    LaunchFtrl<float, 0U, 1U>(N, varIn, accumIn, linearIn, gradIn, lr, l1, l2, lrPower, vO, aO, lO,
                              vAfter);
    int accumChanged = 0;
    for (size_t i = 0; i < static_cast<size_t>(N); ++i) {
        GoldenOut g =
            ComputeGolden(varIn[i], accumIn[i], linearIn[i], gradIn[i], lr, l1, l2, lrPower);
        // var 输入端口字节级未变（var 经 var_out 输出，不回写 var 输入）。
        EXPECT_EQ(vAfter[i], varIn[i]) << "var input GM must NOT be overwritten at i=" << i;
        // accum/linear 输入端口被原地改写为新值。
        EXPECT_LE(std::fabs(aO[i] - g.accum), kAtolFp32 + kRtolFp32 * std::fabs(g.accum))
            << "accum in-place = accum_new at i=" << i;
        EXPECT_LE(std::fabs(lO[i] - g.linear), kAtolFp32 + kRtolFp32 * std::fabs(g.linear))
            << "linear in-place = linear_t at i=" << i;
        EXPECT_LE(std::fabs(vO[i] - g.var), kAtolFp32 + kRtolFp32 * std::fabs(g.var))
            << "var_out = var result at i=" << i;
        if (gradIn[i] != 0.0f && std::fabs(aO[i] - varIn[i]) > 0.0f) {
            // accum_new = accum + g^2 (> accum when g != 0) -> in-place write actually changed accum.
            if (std::fabs(aO[i] - accumIn[i]) > 1e-6) ++accumChanged;
        }
    }
    EXPECT_GT(accumChanged, 0) << "accum input GM must be updated in place (accum_new != accum)";
}

// ---- (E) 多 tile 主循环：blockLen(200) > ubFactor(64) → 4 tiles（末 tile=8，非对齐尾）----
TEST_F(ApplyFtrlKernelTest, T019_MultiTile_TailMidLoop_Fp32)
{
    const int64_t N = 200;
    std::vector<float> varIn, accumIn, linearIn, gradIn;
    FillGoodDomain(N, varIn, accumIn, linearIn, gradIn);
    const float lr = 0.1f, l1 = 0.01f, l2 = 0.05f, lrPower = 0.5f;

    std::vector<float> vO, aO, lO, vAfter;
    LaunchFtrl<float, 1U, 1U>(N, varIn, accumIn, linearIn, gradIn, lr, l1, l2, lrPower, vO, aO, lO,
                              vAfter, /*ubFactorOverride=*/64);
    for (size_t i = 0; i < static_cast<size_t>(N); ++i) {
        GoldenOut g =
            ComputeGolden(varIn[i], accumIn[i], linearIn[i], gradIn[i], lr, l1, l2, lrPower);
        EXPECT_LE(std::fabs(vO[i] - g.var), kAtolFp32 + kRtolFp32 * std::fabs(g.var))
            << "var mismatch (multi-tile) at i=" << i;
        EXPECT_LE(std::fabs(aO[i] - g.accum), kAtolFp32 + kRtolFp32 * std::fabs(g.accum))
            << "accum mismatch (multi-tile) at i=" << i;
        EXPECT_LE(std::fabs(lO[i] - g.linear), kAtolFp32 + kRtolFp32 * std::fabs(g.linear))
            << "linear mismatch (multi-tile) at i=" << i;
        EXPECT_GE(aO[i], 0.0) << "accum_out non-negative invariant at i=" << i;
    }
}

}  // namespace
