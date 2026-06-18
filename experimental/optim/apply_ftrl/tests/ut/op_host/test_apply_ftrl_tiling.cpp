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
 * \file test_apply_ftrl_tiling.cpp
 * \brief ApplyFtrl Tiling UT（迭代一 - 核心路径：主线 TilingKey FLOAT16,0,1）
 *
 * 验证 op_host/apply_ftrl_tiling.cpp 的单一 Elementwise 切分策略（DESIGN s3.3）：
 *   - 多核均分：blockFactor = CeilAlign(CeilDiv(total, coreNum), ubBlockSize=32)，
 *     usedCoreNum = CeilDiv(total, blockFactor)，相邻核 GM 不重叠（32 元素 DMA 粒度对齐）。
 *   - UB 切分：先扣 8KB Select 框架 tmp（MED-1），再按 UB_FP32_SLOTS=24 预算，
 *     上限 TILE_ELEM_NUM_TARGET=2048，向下对齐 64 元素（256B，MED-2）。
 *   - 空 Tensor 短路：totalElements==0 → blockFactor=ubFactor=0，SetBlockDim(1)。
 *   - dtype 校验：仅 bf16/fp16/fp32（其它 → GRAPH_FAILED，对应 spec dtype_not_supported）。
 *   - shape 一致性（MED-3）：var/accum/linear/grad 必须同 shape（否则 GRAPH_FAILED，
 *     对应 spec boundary shape_mismatch）。
 *   - TilingKey：ASCENDC_TPL_SEL_PARAM(dTypeVar, padTail, hasL1)；PAD_TAIL 由
 *     (total % 32 != 0) 决定 → 非对齐尾块 TilingKey 与对齐路径不同。
 *
 * 真值源：docs/spec.yaml（dtype/shape/boundary）+ docs/DESIGN.md（s3.1/s3.3/s3.8）。
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "apply_ftrl_tiling_data.h"

namespace ApplyFtrlUT {
using namespace std;
using namespace ge;
using namespace gert;

static const std::string OP_NAME = "ApplyFtrl";

// 与 op_host/apply_ftrl_tiling.cpp 同步的常量（仅用于 UT 期望值推导）。
constexpr int64_t kUbBlockSize = 32;            // GetUbBlockSize() 返回 32（元素粒度）
constexpr int64_t kCmpAlignElem = 64;           // 256B / sizeof(fp32) = 64
constexpr int64_t kTileTarget = 2048;           // TILE_ELEM_NUM_TARGET
constexpr int64_t kSelectTmpBytes = 8192;       // Select 模式1/2 框架 tmp（MED-1）
constexpr int64_t kUbFp32Slots = 24;            // UB_FP32_SLOTS
constexpr int64_t kMinElemPerCore = 512;        // MIN_ELEM_PER_CORE（PERF: 自适应 block-dim 下限）

// -------------------------------------------------------------------
// TilingKey 字段布局（迭代二 - 全分支覆盖）
//   ASCENDC_TPL_SEL_PARAM(context, dTypeVar, padTail, hasL1) 按 DECL 顺序（D_T_VAR,
//   PAD_TAIL, HAS_L1）自**低位**向高位每个 param 占 8 bit 打包（低→高 = DECL 序）：
//       key = (HAS_L1 << 16) | (PAD_TAIL << 8) | C_DataType(input0.dtype)
//   实测：fp16(cdt=1) 因 dtype 码==HAS_L1==1 而对称（0x010001/0x010101），fp32/bf16 才显出真实布局：
//         fp32 空(cdt=0,pad=0,l1=1)=0x010000=65536；bf16 空(cdt=27)=0x01001B=65563。
//   - D_T_VAR 低字节 = C_DataType 编码（graph/c_types.h：C_DT_FLOAT=0 / C_DT_FLOAT16=1 / C_DT_BF16=27）。
//   - PAD_TAIL 中字节 = (totalElements % 32 != 0)。
//   - HAS_L1 高字节：host **恒置 1**（tiling 期读不到 Device GM 标量 l1；kernel 在 l1==0 时
//     走运行期快路，HAS_L1=0 二进制仍产出供 op_kernel UT 直驱）。HAS_L1=0 分支由
//     op_kernel UT 经模板实例化覆盖（见 tests/ut/op_kernel/test_apply_ftrl.cpp）。
// -------------------------------------------------------------------
constexpr int64_t kCDtFloat = 0;     // C_DT_FLOAT   (fp32)
constexpr int64_t kCDtFloat16 = 1;   // C_DT_FLOAT16
constexpr int64_t kCDtBf16 = 27;     // C_DT_BF16

static inline int64_t KeyDtype(int64_t k) { return k & 0xFF; }
static inline int64_t KeyPadTail(int64_t k) { return (k >> 8) & 0xFF; }
static inline int64_t KeyHasL1(int64_t k) { return (k >> 16) & 0xFF; }
static inline int64_t ExpectKey(int64_t cdt, int64_t pad, int64_t hasL1)
{
    return (hasL1 << 16) | (pad << 8) | cdt;
}

struct ApplyFtrlCompileInfoStub {};

class ApplyFtrlTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ApplyFtrlTilingTest SetUp." << std::endl; }
    static void TearDownTestCase() { std::cout << "ApplyFtrlTilingTest TearDown." << std::endl; }
};

// 构造 ApplyFtrl 的 TilingContextPara：8 输入(var/accum/linear/grad + 4 scalar) + 1 输出(var)。
// tiling 仅读取 input0..3 的 shape 与 input0 的 dtype；scalar 仅占位（shape {1}）。
static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape ss;
    ss.MutableOriginShape().SetDimNum(0);
    ss.MutableStorageShape().SetDimNum(0);
    for (auto d : dims) {
        ss.MutableOriginShape().AppendDim(d);
        ss.MutableStorageShape().AppendDim(d);
    }
    return ss;
}

static gert::TilingContextPara MakeTilingPara(
    const std::vector<int64_t>& varShape,
    const std::vector<int64_t>& accumShape,
    const std::vector<int64_t>& linearShape,
    const std::vector<int64_t>& gradShape,
    ge::DataType dtype,
    void* compileInfo,
    uint64_t coreNum = 48,
    uint64_t ubSize = 262144,
    uint64_t tilingDataSize = 4096)
{
    auto mk = [](const std::vector<int64_t>& s) { return MakeStorageShape(s); };
    gert::StorageShape scalarShape = mk({1});

    std::vector<TilingContextPara::TensorDescription> inputs({
        {mk(varShape),    dtype, ge::FORMAT_ND},  // var
        {mk(accumShape),  dtype, ge::FORMAT_ND},  // accum
        {mk(linearShape), dtype, ge::FORMAT_ND},  // linear
        {mk(gradShape),   dtype, ge::FORMAT_ND},  // grad
        {scalarShape,     dtype, ge::FORMAT_ND},  // lr
        {scalarShape,     dtype, ge::FORMAT_ND},  // l1
        {scalarShape,     dtype, ge::FORMAT_ND},  // l2
        {scalarShape,     dtype, ge::FORMAT_ND},  // lr_power
    });
    std::vector<TilingContextPara::TensorDescription> outputs({
        {mk(varShape),    dtype, ge::FORMAT_ND},  // var (declared output)
    });
    return gert::TilingContextPara(OP_NAME, inputs, outputs, compileInfo, coreNum, ubSize, tilingDataSize);
}

static gert::TilingContextPara MakeUniformTilingPara(
    const std::vector<int64_t>& shape, ge::DataType dtype, void* compileInfo,
    uint64_t coreNum = 48, uint64_t ubSize = 262144)
{
    return MakeTilingPara(shape, shape, shape, shape, dtype, compileInfo, coreNum, ubSize);
}

static const ApplyFtrlTilingData* AsTd(const TilingInfo& info)
{
    return reinterpret_cast<const ApplyFtrlTilingData*>(info.tilingData.get());
}

// 通用核心路径不变式（成功用例共用）。
static void CheckCoreInvariants(const TilingInfo& info, int64_t expectTotal)
{
    ASSERT_GE(info.tilingDataSize, sizeof(ApplyFtrlTilingData));
    const auto* td = AsTd(info);
    EXPECT_EQ(td->totalElements, expectTotal);

    // blockFactor 按 32 元素（DMA 粒度）对齐，相邻核 GM 不重叠。
    EXPECT_GT(td->blockFactor, 0);
    EXPECT_EQ(td->blockFactor % kUbBlockSize, 0) << "blockFactor must be 32-element aligned";

    // ubFactor 为 256B/64 元素倍数（MED-2），且 <= TILE 上限。
    EXPECT_GT(td->ubFactor, 0);
    EXPECT_EQ(td->ubFactor % kCmpAlignElem, 0) << "ubFactor must be 64-element (256B) aligned";
    EXPECT_LE(td->ubFactor, kTileTarget);

    // 多核均分：blockNum 个核覆盖全部元素，且无完全空转的尾核。
    EXPECT_GE(info.blockNum, 1u);
    EXPECT_GE(static_cast<int64_t>(info.blockNum) * td->blockFactor, expectTotal);
    EXPECT_LT((static_cast<int64_t>(info.blockNum) - 1) * td->blockFactor, expectTotal);

    // TilingKey 已设置（ASCENDC_TPL_SEL_PARAM）。
    EXPECT_GE(info.tilingKey, 0);
}

// 迭代二：大 shape（多核 + ubFactor 命中 TILE 上限 2048）combo 校验：
//   核心不变式 + 精确 TilingKey + 逐字段解码（dtype/PAD_TAIL/HAS_L1）。
static void CheckLargeCombo(const TilingInfo& info, int64_t expectTotal, int64_t expectCdt,
                           int64_t expectPad)
{
    CheckCoreInvariants(info, expectTotal);
    EXPECT_EQ(AsTd(info)->ubFactor, kTileTarget) << "large shape should hit TILE_ELEM_NUM_TARGET=2048";
    EXPECT_GT(info.blockNum, 1u) << "large shape should span multiple cores";

    // 精确 TilingKey：host 恒置 HAS_L1=1。
    EXPECT_EQ(info.tilingKey, ExpectKey(expectCdt, expectPad, 1))
        << "TilingKey must encode (dtype,PAD_TAIL,HAS_L1=1)";
    EXPECT_EQ(KeyDtype(info.tilingKey), expectCdt) << "D_T_VAR byte must encode input0 dtype";
    EXPECT_EQ(KeyPadTail(info.tilingKey), expectPad) << "PAD_TAIL byte must match (total%32!=0)";
    EXPECT_EQ(KeyHasL1(info.tilingKey), 1) << "host must always emit HAS_L1=1";
    std::cout << "[combo] cdt=" << expectCdt << " pad=" << expectPad
              << " key=" << info.tilingKey << " ubFactor=" << AsTd(info)->ubFactor
              << " blockFactor=" << AsTd(info)->blockFactor << " blockNum=" << info.blockNum
              << std::endl;
}

// -------------------------------------------------------------------
// T001: 核心路径 fp16，大 shape → 多核均分 + UB 多次迭代；ubFactor 取 TILE 上限 2048。
//   total=524288（32 对齐 → PAD_TAIL=0），ubSize 默认 256KB（8KB 扣除后预算 >2048）。
// -------------------------------------------------------------------
TEST_F(ApplyFtrlTilingTest, T001_CorePath_Fp16_MultiCore_Succeeds)
{
    ApplyFtrlCompileInfoStub ci;
    auto para = MakeUniformTilingPara({2048, 256}, ge::DT_FLOAT16, &ci);  // 524288

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info)) << "core-path fp16 tiling should succeed";

    CheckCoreInvariants(info, 524288);
    // 预算充足（256KB）→ ubFactor 命中 TILE 上限 2048（与 coreNum 无关，blockFactor 远大于 2048）。
    EXPECT_EQ(AsTd(info)->ubFactor, 2048) << "ubFactor should hit TILE_ELEM_NUM_TARGET=2048";
    // total 是 32 的倍数 → 多核（blockNum 应 > 1）。
    EXPECT_GT(info.blockNum, 1u) << "large shape should span multiple cores";
}

// -------------------------------------------------------------------
// T002: 8KB Select tmp 扣除验证（MED-1）。
//   人为收紧 ubSize=32768：ubFactor 由 fp32 预算决定 =
//     FloorAlign(FloorDiv((32768-8192)/4, 24), 64) = 256。
//   若不扣 8KB，则为 FloorAlign(FloorDiv(32768/4, 24), 64) = 320。
//   断言 ubFactor==256 即证明 8KB 已被扣除，且按 64 元素对齐。
// -------------------------------------------------------------------
TEST_F(ApplyFtrlTilingTest, T002_UbSplit_8KBSelectTmpDeducted)
{
    // 收紧 ubSize 使 fp32 UB 预算成为约束（而非 TILE 上限 2048），从而能观测 8KB 扣除。
    // 取 49152：8KB 扣除后预算 = FloorAlign(FloorDiv((49152-8192)/4, 24), 64) = 384；
    //           不扣 8KB 则为 FloorAlign(FloorDiv(49152/4, 24), 64) = 512。
    // 二者相差两个 64 块 → ubFactor 落在 384（< 512）即证明 8KB 已扣除且按 64 元素对齐。
    constexpr uint64_t ubSize = 49152;
    ApplyFtrlCompileInfoStub ci;
    auto para = MakeUniformTilingPara({2048, 256}, ge::DT_FLOAT16, &ci, /*coreNum=*/48, ubSize);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info)) << "tiling should succeed with tight UB";
    CheckCoreInvariants(info, 524288);

    const int64_t predWith8KB =
        ((((static_cast<int64_t>(ubSize) - kSelectTmpBytes) / 4) / kUbFp32Slots) / kCmpAlignElem) *
        kCmpAlignElem;  // 384
    const int64_t predNo8KB =
        (((static_cast<int64_t>(ubSize) / 4) / kUbFp32Slots) / kCmpAlignElem) * kCmpAlignElem;  // 512
    ASSERT_LT(predWith8KB, predNo8KB);  // 守护：扣与不扣确实不同，断言才有判别力

    const int64_t ubFactor = AsTd(info)->ubFactor;
    std::cout << "[T002] ubSize=" << ubSize << " ubFactor=" << ubFactor
              << " predWith8KB=" << predWith8KB << " predNo8KB=" << predNo8KB << std::endl;

    EXPECT_EQ(ubFactor % kCmpAlignElem, 0) << "ubFactor must be 64-element (256B) aligned";
    // 8KB 扣除被证明：ubFactor 与「扣 8KB」预算一致（容许 1 个 64 块的边界取整），
    // 且严格小于「不扣 8KB」预算。
    EXPECT_LE(ubFactor, predWith8KB);
    EXPECT_GE(ubFactor, predWith8KB - kCmpAlignElem);
    EXPECT_LT(ubFactor, predNo8KB) << "8KB Select tmp must be reserved before UB split";
}

// -------------------------------------------------------------------
// T003: 小 shape → ubFactor 256B/64 对齐下限（64-floor）。
//   total=48（32 对齐尾块 → PAD_TAIL=1）；blockFactor=32，按 64 元素 floor 后下限抬到 64。
//   验证 ubFactor 即使 > blockFactor 也至少为 64（Compare/Select 256B count 约束）。
// -------------------------------------------------------------------
TEST_F(ApplyFtrlTilingTest, T003_SmallShape_UbFactor64Floor)
{
    ApplyFtrlCompileInfoStub ci;
    auto para = MakeUniformTilingPara({48}, ge::DT_FLOAT16, &ci, /*coreNum=*/8);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info)) << "small shape tiling should succeed";

    const auto* td = AsTd(info);
    EXPECT_EQ(td->totalElements, 48);
    EXPECT_EQ(td->blockFactor % kUbBlockSize, 0);
    EXPECT_GE(td->ubFactor, kCmpAlignElem) << "ubFactor must floor up to 64 (256B) even for tiny shapes";
    EXPECT_EQ(td->ubFactor % kCmpAlignElem, 0);
}

// -------------------------------------------------------------------
// T004: 空 Tensor 短路（spec boundary「空 Tensor 直通」）。
//   shape={0,3} → totalElements=0 → blockFactor=ubFactor=0，SetBlockDim(1)。
// -------------------------------------------------------------------
TEST_F(ApplyFtrlTilingTest, T004_EmptyTensor_ShortCircuit)
{
    ApplyFtrlCompileInfoStub ci;
    auto para = MakeUniformTilingPara({0, 3}, ge::DT_FLOAT16, &ci);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info)) << "empty tensor tiling should succeed (short-circuit)";

    const auto* td = AsTd(info);
    EXPECT_EQ(td->totalElements, 0);
    EXPECT_EQ(td->blockFactor, 0);
    EXPECT_EQ(td->ubFactor, 0);
    EXPECT_EQ(info.blockNum, 1u) << "empty tensor uses a single idle core";
}

// -------------------------------------------------------------------
// T005: fp32 路径 core path（同一 UB 预算公式，与 dtype 无关）。
// -------------------------------------------------------------------
TEST_F(ApplyFtrlTilingTest, T005_CorePath_Fp32_Succeeds)
{
    ApplyFtrlCompileInfoStub ci;
    auto para = MakeUniformTilingPara({4096, 256}, ge::DT_FLOAT, &ci);  // 1048576

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info)) << "fp32 tiling should succeed";

    CheckCoreInvariants(info, 1048576);
    EXPECT_EQ(AsTd(info)->ubFactor, 2048);
}

// -------------------------------------------------------------------
// T006: bf16 路径 core path（dtype 分发）。
// -------------------------------------------------------------------
TEST_F(ApplyFtrlTilingTest, T006_CorePath_Bf16_Succeeds)
{
    ApplyFtrlCompileInfoStub ci;
    auto para = MakeUniformTilingPara({2048, 256}, ge::DT_BF16, &ci);  // 524288

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info)) << "bf16 tiling should succeed";

    CheckCoreInvariants(info, 524288);
    EXPECT_EQ(AsTd(info)->ubFactor, 2048);
}

// -------------------------------------------------------------------
// T007: PAD_TAIL 影响 TilingKey（对齐 vs 非对齐尾块 → 不同 key）。
//   {256}: 256%32==0 → PAD_TAIL=0；{255}: 255%32==31 → PAD_TAIL=1。
// -------------------------------------------------------------------
TEST_F(ApplyFtrlTilingTest, T007_PadTail_TilingKeyDiffers)
{
    ApplyFtrlCompileInfoStub ci;

    auto paraAligned = MakeUniformTilingPara({256}, ge::DT_FLOAT16, &ci);
    TilingInfo infoAligned;
    ASSERT_TRUE(ExecuteTiling(paraAligned, infoAligned));

    auto paraTail = MakeUniformTilingPara({255}, ge::DT_FLOAT16, &ci);
    TilingInfo infoTail;
    ASSERT_TRUE(ExecuteTiling(paraTail, infoTail));

    EXPECT_GE(infoAligned.tilingKey, 0);
    EXPECT_GE(infoTail.tilingKey, 0);
    EXPECT_NE(infoAligned.tilingKey, infoTail.tilingKey)
        << "PAD_TAIL must be encoded in the TilingKey (aligned vs non-aligned tail)";
    std::cout << "TilingKey aligned(PAD_TAIL=0)=" << infoAligned.tilingKey
              << ", tail(PAD_TAIL=1)=" << infoTail.tilingKey << std::endl;
}

// -------------------------------------------------------------------
// 异常路径
// -------------------------------------------------------------------

// T008: var/accum/linear/grad shape 不一致 → GRAPH_FAILED（MED-3 / spec shape_mismatch）。
TEST_F(ApplyFtrlTilingTest, T008_Err_AccumShapeMismatch_Fails)
{
    ApplyFtrlCompileInfoStub ci;
    auto para = MakeTilingPara({2, 3}, {2, 4}, {2, 3}, {2, 3}, ge::DT_FLOAT16, &ci);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info)) << "shape mismatch (accum) must fail tiling";
}

TEST_F(ApplyFtrlTilingTest, T009_Err_LinearShapeMismatch_Fails)
{
    ApplyFtrlCompileInfoStub ci;
    auto para = MakeTilingPara({16, 16}, {16, 16}, {16, 8}, {16, 16}, ge::DT_FLOAT, &ci);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info)) << "shape mismatch (linear) must fail tiling";
}

TEST_F(ApplyFtrlTilingTest, T010_Err_GradShapeMismatch_Fails)
{
    ApplyFtrlCompileInfoStub ci;
    auto para = MakeTilingPara({1024}, {1024}, {1024}, {512}, ge::DT_BF16, &ci);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info)) << "shape mismatch (grad) must fail tiling";
}

// T011: 不支持的 dtype (DT_INT32) → GRAPH_FAILED（spec dtype_not_supported）。
TEST_F(ApplyFtrlTilingTest, T011_Err_UnsupportedDtypeInt32_Fails)
{
    ApplyFtrlCompileInfoStub ci;
    auto para = MakeUniformTilingPara({32, 32}, ge::DT_INT32, &ci);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info)) << "unsupported dtype DT_INT32 must fail tiling";
}

// ===================================================================
// 迭代二：TilingKey 全分支组合覆盖（dtype × PAD_TAIL，HAS_L1 由 host 恒置 1）
//   目标：tiling 对每个 (dtype, 对齐/非对齐) 组合选出正确的 blockDim / ubFactor / TilingKey。
//   覆盖矩阵：D_T_VAR ∈ {FLOAT16, FLOAT, BF16} × PAD_TAIL ∈ {0,1}（6 组可达组合）。
//   tail shape 取 N%32!=0 的大 1-D（同样多核 + ubFactor 命中 2048），对齐 shape 取 N%32==0。
//   注：HAS_L1=0 不可由 tiling 产出（host 恒置 1），由 op_kernel UT 直驱覆盖。
// ===================================================================

// ---- FLOAT16 ----
TEST_F(ApplyFtrlTilingTest, T012_Key_Fp16_Aligned)
{
    ApplyFtrlCompileInfoStub ci;
    auto para = MakeUniformTilingPara({2048, 256}, ge::DT_FLOAT16, &ci);  // 524288, %32==0
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckLargeCombo(info, 524288, kCDtFloat16, /*pad=*/0);
    EXPECT_EQ(info.tilingKey, 65537);  // 0x010001 实测锚点
}

TEST_F(ApplyFtrlTilingTest, T013_Key_Fp16_Tail)
{
    ApplyFtrlCompileInfoStub ci;
    auto para = MakeUniformTilingPara({524287}, ge::DT_FLOAT16, &ci);  // 524287%32==31
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckLargeCombo(info, 524287, kCDtFloat16, /*pad=*/1);
    EXPECT_EQ(info.tilingKey, 65793);  // 0x010101 实测锚点
}

// ---- FLOAT (fp32) ----
TEST_F(ApplyFtrlTilingTest, T014_Key_Fp32_Aligned)
{
    ApplyFtrlCompileInfoStub ci;
    auto para = MakeUniformTilingPara({4096, 256}, ge::DT_FLOAT, &ci);  // 1048576, %32==0
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckLargeCombo(info, 1048576, kCDtFloat, /*pad=*/0);
    EXPECT_EQ(info.tilingKey, 65536);  // 0x010000 (HAS_L1=1, PAD_TAIL=0, C_DT_FLOAT=0)
}

TEST_F(ApplyFtrlTilingTest, T015_Key_Fp32_Tail)
{
    ApplyFtrlCompileInfoStub ci;
    auto para = MakeUniformTilingPara({1048575}, ge::DT_FLOAT, &ci);  // 1048575%32==31
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckLargeCombo(info, 1048575, kCDtFloat, /*pad=*/1);
    EXPECT_EQ(info.tilingKey, 65792);  // 0x010100 (HAS_L1=1, PAD_TAIL=1, C_DT_FLOAT=0)
}

// ---- BF16 ----
TEST_F(ApplyFtrlTilingTest, T016_Key_Bf16_Aligned)
{
    ApplyFtrlCompileInfoStub ci;
    auto para = MakeUniformTilingPara({2048, 256}, ge::DT_BF16, &ci);  // 524288, %32==0
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckLargeCombo(info, 524288, kCDtBf16, /*pad=*/0);
    EXPECT_EQ(info.tilingKey, 65563);  // 0x01001B (HAS_L1=1, PAD_TAIL=0, C_DT_BF16=27)
}

TEST_F(ApplyFtrlTilingTest, T017_Key_Bf16_Tail)
{
    ApplyFtrlCompileInfoStub ci;
    auto para = MakeUniformTilingPara({524287}, ge::DT_BF16, &ci);  // 524287%32==31
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckLargeCombo(info, 524287, kCDtBf16, /*pad=*/1);
    EXPECT_EQ(info.tilingKey, 65819);  // 0x01011B (HAS_L1=1, PAD_TAIL=1, C_DT_BF16=27)
}

// T018: 6 组 (dtype × PAD_TAIL) TilingKey 两两互异（证明 key 正确编码 dtype + 对齐位）。
TEST_F(ApplyFtrlTilingTest, T018_Key_AllSixCombosDistinct)
{
    ApplyFtrlCompileInfoStub ci;
    struct Combo { std::vector<int64_t> shape; ge::DataType dt; const char* name; };
    const std::vector<Combo> combos = {
        {{2048, 256}, ge::DT_FLOAT16, "fp16_aligned"},
        {{524287},    ge::DT_FLOAT16, "fp16_tail"},
        {{4096, 256}, ge::DT_FLOAT,   "fp32_aligned"},
        {{1048575},   ge::DT_FLOAT,   "fp32_tail"},
        {{2048, 256}, ge::DT_BF16,    "bf16_aligned"},
        {{524287},    ge::DT_BF16,    "bf16_tail"},
    };
    std::vector<int64_t> keys;
    for (const auto& c : combos) {
        auto para = MakeUniformTilingPara(c.shape, c.dt, &ci);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(para, info)) << "combo " << c.name << " must tile";
        keys.push_back(info.tilingKey);
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        for (size_t j = i + 1; j < keys.size(); ++j) {
            EXPECT_NE(keys[i], keys[j])
                << "combos " << combos[i].name << " and " << combos[j].name
                << " must map to distinct TilingKeys (" << keys[i] << " vs " << keys[j] << ")";
        }
    }
}

// T019: host 恒置 HAS_L1=1（覆盖大/小/对齐/非对齐/空 各 dtype）。
//   验证 tiling 永远不会产出 HAS_L1=0 的 key（HAS_L1=0 仅由 op_kernel UT 直驱）。
TEST_F(ApplyFtrlTilingTest, T019_HasL1_AlwaysOneFromHost)
{
    ApplyFtrlCompileInfoStub ci;
    struct Case { std::vector<int64_t> shape; ge::DataType dt; };
    const std::vector<Case> cases = {
        {{2048, 256}, ge::DT_FLOAT16}, {{255}, ge::DT_FLOAT16},
        {{4096, 256}, ge::DT_FLOAT},   {{1048575}, ge::DT_FLOAT},
        {{2048, 256}, ge::DT_BF16},    {{524287}, ge::DT_BF16},
        {{0, 3}, ge::DT_FLOAT16},      {{0, 3}, ge::DT_FLOAT},  {{0, 3}, ge::DT_BF16},
    };
    for (const auto& c : cases) {
        auto para = MakeUniformTilingPara(c.shape, c.dt, &ci);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(para, info));
        EXPECT_EQ(KeyHasL1(info.tilingKey), 1)
            << "host tiling must always force HAS_L1=1 (dtype=" << static_cast<int>(c.dt) << ")";
    }
}

// T020: 空 Tensor 的 TilingKey 逐 dtype 正确（PAD_TAIL=0, HAS_L1=1, 对应 dtype 码）。
//   空路径单独 SetBlockDim(1)，blockFactor=ubFactor=0；key 仍按 dtype 编码（DESIGN s3.3 空短路）。
TEST_F(ApplyFtrlTilingTest, T020_EmptyTensor_TilingKey_PerDtype)
{
    ApplyFtrlCompileInfoStub ci;
    struct Case { ge::DataType dt; int64_t cdt; };
    const std::vector<Case> cases = {
        {ge::DT_FLOAT16, kCDtFloat16}, {ge::DT_FLOAT, kCDtFloat}, {ge::DT_BF16, kCDtBf16},
    };
    for (const auto& c : cases) {
        auto para = MakeUniformTilingPara({0, 3}, c.dt, &ci);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(para, info)) << "empty tensor tiling should succeed";
        const auto* td = AsTd(info);
        EXPECT_EQ(td->totalElements, 0);
        EXPECT_EQ(td->blockFactor, 0);
        EXPECT_EQ(td->ubFactor, 0);
        EXPECT_EQ(info.blockNum, 1u);
        EXPECT_EQ(info.tilingKey, ExpectKey(c.cdt, /*pad=*/0, /*hasL1=*/1))
            << "empty tensor key must be (dtype,PAD_TAIL=0,HAS_L1=1)";
    }
}

// ===================================================================
// 迭代三：全覆盖补齐
//   (1) ubFactor 精确计算边界（256B/64 元素对齐、blockFactor 上限裁剪、非对齐尾块）；
//   (2) 错误路径加宽（多种非支持 dtype、shape 不一致 rank 差异）；
//   (3) 维度边界（rank=8 合法 / rank>8 op_host 不强校验 / rank-0）；
//   (4) 标量端口形态（op_host 不读 scalar shape）；
//   (5) PAD_TAIL 精确边界（32 元素对齐 vs 非对齐）+ 多轴空 Tensor。
// 真值源：op_host/apply_ftrl_tiling.cpp（ubFactor 公式）+ DESIGN s3.3/s8 + spec boundary。
// ===================================================================

// 允许自定义 scalar 端口 shape（验证 op_host 不读 lr/l1/l2/lr_power 形态）。
static gert::TilingContextPara MakeTilingParaCustomScalar(
    const std::vector<int64_t>& tensorShape, const std::vector<int64_t>& scalarShape,
    ge::DataType dtype, void* compileInfo, uint64_t coreNum = 48, uint64_t ubSize = 262144)
{
    auto mk = [](const std::vector<int64_t>& s) { return MakeStorageShape(s); };
    std::vector<TilingContextPara::TensorDescription> inputs({
        {mk(tensorShape), dtype, ge::FORMAT_ND},  // var
        {mk(tensorShape), dtype, ge::FORMAT_ND},  // accum
        {mk(tensorShape), dtype, ge::FORMAT_ND},  // linear
        {mk(tensorShape), dtype, ge::FORMAT_ND},  // grad
        {mk(scalarShape), dtype, ge::FORMAT_ND},  // lr
        {mk(scalarShape), dtype, ge::FORMAT_ND},  // l1
        {mk(scalarShape), dtype, ge::FORMAT_ND},  // l2
        {mk(scalarShape), dtype, ge::FORMAT_ND},  // lr_power
    });
    std::vector<TilingContextPara::TensorDescription> outputs({
        {mk(tensorShape), dtype, ge::FORMAT_ND},  // var (declared output)
    });
    return gert::TilingContextPara(OP_NAME, inputs, outputs, compileInfo, coreNum, ubSize);
}

// 复刻 op_host/apply_ftrl_tiling.cpp 的 ubFactor 公式（仅用于 UT 期望值推导）。
//   usableUb = ubSize - 8192; cap = min(2048, FloorAlign(FloorDiv(usableUb/4, 24), 64));
//   ubFactor = FloorAlign(min(cap, blockFactor), 64); 不足 64 抬到 64。
static int64_t ExpectUbFactor(int64_t blockFactor, uint64_t ubSize = 262144)
{
    int64_t usableUb = static_cast<int64_t>(ubSize) - kSelectTmpBytes;
    int64_t cap = ((usableUb / 4) / kUbFp32Slots / kCmpAlignElem) * kCmpAlignElem;
    if (cap > kTileTarget) cap = kTileTarget;
    int64_t uf = (cap < blockFactor) ? cap : blockFactor;
    uf = (uf / kCmpAlignElem) * kCmpAlignElem;  // FloorAlign 64
    if (uf < kCmpAlignElem) uf = kCmpAlignElem;
    return uf;
}
static int64_t ExpectBlockFactor(int64_t total, int64_t coreNum)
{
    int64_t per = (total + coreNum - 1) / coreNum;                       // CeilDiv
    int64_t bf = ((per + kUbBlockSize - 1) / kUbBlockSize) * kUbBlockSize;  // CeilAlign 32
    // PERF: 自适应 block-dim 下限——blockFactor 抬到 MIN_ELEM_PER_CORE(DMA 对齐)，
    // 仅对小/中 shape 生效（降低用核数）；大 shape 朴素 blockFactor 已超过下限，不变。
    int64_t minBf = ((kMinElemPerCore + kUbBlockSize - 1) / kUbBlockSize) * kUbBlockSize;
    if (bf < minBf) bf = minBf;
    return bf;
}

// -------------------------------------------------------------------
// T021: ubFactor 被 blockFactor 上限裁剪，且 blockFactor 恰为 64 倍数 → ubFactor==blockFactor。
//   total=640, coreNum=1 → blockFactor=640（32 对齐，且 640%64==0）；
//   预算 cap=2048 > 640 → ubFactor=FloorAlign(640,64)=640。
// -------------------------------------------------------------------
TEST_F(ApplyFtrlTilingTest, T021_UbFactor_CappedByBlockFactor_64Aligned)
{
    ApplyFtrlCompileInfoStub ci;
    auto para = MakeUniformTilingPara({640}, ge::DT_FLOAT, &ci, /*coreNum=*/1);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    const auto* td = AsTd(info);
    EXPECT_EQ(td->totalElements, 640);
    EXPECT_EQ(td->blockFactor, ExpectBlockFactor(640, 1));  // 640
    EXPECT_EQ(td->blockFactor, 640);
    EXPECT_EQ(td->ubFactor, ExpectUbFactor(td->blockFactor));  // 640
    EXPECT_EQ(td->ubFactor, 640);
    EXPECT_EQ(td->ubFactor % kCmpAlignElem, 0);
    EXPECT_EQ(info.blockNum, 1u);
}

// -------------------------------------------------------------------
// T022: blockFactor 为 32 对齐但非 64 对齐 → ubFactor 经 256B FloorAlign 降到其下方的 64 倍数。
//   total=1500, coreNum=1 → blockFactor=CeilAlign(1500,32)=1504（1504%64==32，非 64 对齐）；
//   cap=2048 > 1504 → ubFactor=FloorAlign(1504,64)=1472 < blockFactor（证明 256B 计算对齐裁剪，MED-2）。
// -------------------------------------------------------------------
TEST_F(ApplyFtrlTilingTest, T022_UbFactor_BlockFactorFlooredTo64)
{
    ApplyFtrlCompileInfoStub ci;
    auto para = MakeUniformTilingPara({1500}, ge::DT_FLOAT16, &ci, /*coreNum=*/1);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    const auto* td = AsTd(info);
    EXPECT_EQ(td->totalElements, 1500);
    EXPECT_EQ(td->blockFactor, 1504);
    EXPECT_EQ(td->blockFactor % kUbBlockSize, 0);
    EXPECT_NE(td->blockFactor % kCmpAlignElem, 0) << "blockFactor must be 32- but not 64-aligned here";
    EXPECT_EQ(td->ubFactor, ExpectUbFactor(td->blockFactor));  // 1472
    EXPECT_EQ(td->ubFactor, 1472);
    EXPECT_LT(td->ubFactor, td->blockFactor) << "256B (64-elem) floor must reduce ubFactor below 32-aligned blockFactor";
    EXPECT_EQ(td->ubFactor % kCmpAlignElem, 0);
}

// -------------------------------------------------------------------
// T023: 小 shape，blockFactor 被自适应下限 MIN_ELEM_PER_CORE(512) 抬高（PERF 优化）。
//   total=200, coreNum=1 → 朴素 blockFactor=CeilAlign(200,32)=224 < 512 → 抬到 512；
//   ubFactor=FloorAlign(min(cap,512),64)=512；blockNum=ceil(200/512)=1。
//   （原"ubFactor floored 到 192"的小 shape 路径已被下限覆盖；非 64 对齐裁剪由 T022 覆盖。）
// -------------------------------------------------------------------
TEST_F(ApplyFtrlTilingTest, T023_SmallShape_BlockFactorFlooredToMinElemPerCore)
{
    ApplyFtrlCompileInfoStub ci;
    auto para = MakeUniformTilingPara({200}, ge::DT_BF16, &ci, /*coreNum=*/1);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    const auto* td = AsTd(info);
    EXPECT_EQ(td->totalElements, 200);
    EXPECT_EQ(td->blockFactor, ExpectBlockFactor(200, 1));  // 512 (floored)
    EXPECT_EQ(td->blockFactor, kMinElemPerCore);            // 512
    EXPECT_EQ(td->ubFactor, ExpectUbFactor(td->blockFactor));  // 512
    EXPECT_EQ(td->ubFactor, 512);
    EXPECT_EQ(td->ubFactor % kCmpAlignElem, 0);
    EXPECT_EQ(info.blockNum, 1u);
}

// -------------------------------------------------------------------
// T031: 自适应 block-dim 下限（PERF 优化核心）——小/中 shape 抬高 blockFactor 至
//   MIN_ELEM_PER_CORE(512) 以降低用核数、摊薄固定头开销（kernel 启动 + 4 标量 GM 读 +
//   buffer init + 多核调度）；纯多核切分决策，不改 FTRL 数值/dtype/三输出。大 shape 不受影响。
//   (a) total=1024, coreNum=48：朴素均分 blockFactor=CeilAlign(22,32)=32 → 32 核；下限抬到 512 → 仅 2 核。
//   (b) total=131072, coreNum=48：朴素 blockFactor=2752 ≥512 → 下限不触发，仍 48 核（无回退证据）。
// -------------------------------------------------------------------
TEST_F(ApplyFtrlTilingTest, T031_AdaptiveBlockDimFloor_SmallShapeFewerCores_LargeUnaffected)
{
    ApplyFtrlCompileInfoStub ci;

    // (a) 小 shape：下限生效 → 用核数从 32 降到 2。
    auto paraSmall = MakeUniformTilingPara({1024}, ge::DT_FLOAT16, &ci, /*coreNum=*/48);
    TilingInfo infoS;
    ASSERT_TRUE(ExecuteTiling(paraSmall, infoS));
    const auto* tdS = AsTd(infoS);
    EXPECT_EQ(tdS->blockFactor, kMinElemPerCore) << "小 shape blockFactor 抬到 MIN_ELEM_PER_CORE";
    EXPECT_EQ(tdS->blockFactor, ExpectBlockFactor(1024, 48));
    EXPECT_EQ(infoS.blockNum, 2u) << "1024 / 512 = 2 核（朴素均分会用 32 核）";
    CheckCoreInvariants(infoS, 1024);

    // (b) 中/大 shape：下限不触发 → 仍用满核（证明无回退）。
    auto paraMed = MakeUniformTilingPara({131072}, ge::DT_FLOAT, &ci, /*coreNum=*/48);
    TilingInfo infoM;
    ASSERT_TRUE(ExecuteTiling(paraMed, infoM));
    const auto* tdM = AsTd(infoM);
    EXPECT_GT(tdM->blockFactor, kMinElemPerCore) << "中 shape 朴素 blockFactor 已超过下限";
    EXPECT_EQ(tdM->blockFactor, ExpectBlockFactor(131072, 48));
    EXPECT_EQ(infoM.blockNum, 48u) << "中/大 shape 仍用满核（无回退）";
    CheckCoreInvariants(infoM, 131072);
}

// -------------------------------------------------------------------
// T024: 多种非支持 dtype 均 → GRAPH_FAILED（spec dtype_not_supported；tiling 仅放行 bf16/fp16/fp32）。
// -------------------------------------------------------------------
TEST_F(ApplyFtrlTilingTest, T024_Err_UnsupportedDtypes_Multiple)
{
    ApplyFtrlCompileInfoStub ci;
    const std::vector<ge::DataType> badDtypes = {
        ge::DT_INT8, ge::DT_INT32, ge::DT_INT64, ge::DT_DOUBLE, ge::DT_UINT8,
    };
    for (auto dt : badDtypes) {
        auto para = MakeUniformTilingPara({32, 32}, dt, &ci);
        TilingInfo info;
        EXPECT_FALSE(ExecuteTiling(para, info))
            << "unsupported dtype " << static_cast<int>(dt) << " must fail tiling";
    }
}

// -------------------------------------------------------------------
// T025: shape 不一致 — rank（DimNum）差异（非逐维差异）→ GRAPH_FAILED（MED-3 DimNum 校验分支）。
//   var={2,3}(rank2) vs accum={6}(rank1，元素数相同但 DimNum 不同)。
// -------------------------------------------------------------------
TEST_F(ApplyFtrlTilingTest, T025_Err_ShapeMismatch_RankDiff)
{
    ApplyFtrlCompileInfoStub ci;
    auto para = MakeTilingPara({2, 3}, {6}, {2, 3}, {2, 3}, ge::DT_FLOAT, &ci);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info)) << "rank-different accum (same numel) must still fail (DimNum check)";

    auto para2 = MakeTilingPara({4, 4}, {4, 4}, {4, 4}, {16}, ge::DT_FLOAT16, &ci);
    TilingInfo info2;
    EXPECT_FALSE(ExecuteTiling(para2, info2)) << "rank-different grad must fail";
}

// -------------------------------------------------------------------
// T026: rank=8（spec rank_range 上界）合法 → 正常 tiling，totalElements 正确。
// -------------------------------------------------------------------
TEST_F(ApplyFtrlTilingTest, T026_Rank8_MaxValid_Succeeds)
{
    ApplyFtrlCompileInfoStub ci;
    auto para = MakeUniformTilingPara({2, 3, 1, 1, 1, 1, 1, 4}, ge::DT_FLOAT16, &ci);  // 24
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info)) << "rank-8 (max valid rank) must tile";
    const auto* td = AsTd(info);
    EXPECT_EQ(td->totalElements, 24);
    EXPECT_GT(td->ubFactor, 0);
    EXPECT_GT(td->blockFactor, 0);
}

// -------------------------------------------------------------------
// T027: rank>8（rank-9）— op_host（tiling）不做 rank<=8 运行时强校验（按 DESIGN s8：rank 上界由
//   aclnn CheckParams / GE 框架兜底，算子内不校验，与"调用方保证前置条件"契约一致）。
//   四张量 shape 一致 → tiling 仍成功，totalElements 为各维乘积。文档化行为，非缺陷。
// -------------------------------------------------------------------
TEST_F(ApplyFtrlTilingTest, T027_RankOver8_NotEnforcedByOpHost)
{
    ApplyFtrlCompileInfoStub ci;
    auto para = MakeUniformTilingPara({2, 3, 1, 1, 1, 1, 1, 1, 1}, ge::DT_FLOAT, &ci);  // rank9, numel6
    TilingInfo info;
    EXPECT_TRUE(ExecuteTiling(para, info))
        << "op_host tiling does not enforce rank<=8 (aclnn/GE responsibility per DESIGN s8)";
    if (info.tilingDataSize >= sizeof(ApplyFtrlTilingData)) {
        EXPECT_EQ(AsTd(info)->totalElements, 6);
    }
}

// -------------------------------------------------------------------
// T028: 标量端口非标量形态 — op_host tiling 只读 input0..3（var/accum/linear/grad），不读
//   lr/l1/l2/lr_power 的 shape；故 scalar 端口给非标量 shape 不影响 tiling（DESIGN s3.5，
//   scalar 校验由 aclnn 兜底）。文档化 op_host 行为。
// -------------------------------------------------------------------
TEST_F(ApplyFtrlTilingTest, T028_NonScalarScalarInputs_NotEnforcedByOpHost)
{
    ApplyFtrlCompileInfoStub ci;
    auto para = MakeTilingParaCustomScalar({16, 16}, {2, 3}, ge::DT_FLOAT16, &ci);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info))
        << "op_host tiling ignores scalar-port shape (aclnn-layer concern)";
    CheckCoreInvariants(info, 256);
}

// -------------------------------------------------------------------
// T029: PAD_TAIL 精确边界（ubBlockSize 恒 32 元素）。
//   N=32 → 32%32==0 → PAD_TAIL=0；N=33 → 33%32==1 → PAD_TAIL=1（fp32，cdt=0）。
// -------------------------------------------------------------------
TEST_F(ApplyFtrlTilingTest, T029_PadTail_Boundary32Exact)
{
    ApplyFtrlCompileInfoStub ci;

    auto paraA = MakeUniformTilingPara({32}, ge::DT_FLOAT, &ci);
    TilingInfo infoA;
    ASSERT_TRUE(ExecuteTiling(paraA, infoA));
    EXPECT_EQ(KeyPadTail(infoA.tilingKey), 0) << "N=32 is 32-aligned -> PAD_TAIL=0";
    EXPECT_EQ(infoA.tilingKey, ExpectKey(kCDtFloat, 0, 1));

    auto paraB = MakeUniformTilingPara({33}, ge::DT_FLOAT, &ci);
    TilingInfo infoB;
    ASSERT_TRUE(ExecuteTiling(paraB, infoB));
    EXPECT_EQ(KeyPadTail(infoB.tilingKey), 1) << "N=33 is non-32-aligned -> PAD_TAIL=1";
    EXPECT_EQ(infoB.tilingKey, ExpectKey(kCDtFloat, 1, 1));

    EXPECT_NE(infoA.tilingKey, infoB.tilingKey);
}

// -------------------------------------------------------------------
// T030: 多轴空 Tensor（{2,0,3}，0 元素）→ 短路（blockFactor=ubFactor=0, blockDim=1）。
// -------------------------------------------------------------------
TEST_F(ApplyFtrlTilingTest, T030_EmptyTensor_MultiAxis)
{
    ApplyFtrlCompileInfoStub ci;
    auto para = MakeUniformTilingPara({2, 0, 3}, ge::DT_BF16, &ci);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    const auto* td = AsTd(info);
    EXPECT_EQ(td->totalElements, 0);
    EXPECT_EQ(td->blockFactor, 0);
    EXPECT_EQ(td->ubFactor, 0);
    EXPECT_EQ(info.blockNum, 1u);
    EXPECT_EQ(info.tilingKey, ExpectKey(kCDtBf16, 0, 1)) << "empty tensor key = (dtype,PAD_TAIL=0,HAS_L1=1)";
}

}  // namespace ApplyFtrlUT
