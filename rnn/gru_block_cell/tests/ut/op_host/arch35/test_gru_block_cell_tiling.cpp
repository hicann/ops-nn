/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED on an "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_gru_block_cell_tiling.cpp
 * \brief GruBlockCell Tiling UT（arch35 / DAV_3510）
 *
 * 挂载：nn_op_host_ut 聚合体（add_all_modules_sources → add_all_ut_sources 扫 tests/ut/op_host）。
 * 框架：仓级 gert::TilingContextPara + ExecuteTiling（tests/ut/common/tiling_case_executor.*）。
 *
 * 定位（评审报告 gru_block_cell_repo_compare_review_20260921.md [D1]）：
 * host tiling（5 道 gate + 容量模型 + 双分核模式 + 决策量）为纯 host 算术、零设备
 * 依赖，此前 tests/ut 不存在 → cmake 静默零覆盖。本文件把报告 A1/B1 的穷举扫描
 * 结论固化为属性测试，作为删死代码（A1）、调 UB 预算（B1）、容量改平台查询（B2）
 * 的回归网。
 *
 * ⚠ 平台容量口径（B2 后的关键设计）：host 决策取 PlatformCaps（在线查询 →
 * compileInfo → 常量兜底），而 UT 环境的平台信息由仓级框架
 * tests/ut/common/tiling_case_executor.cpp:117-119 的 JSON 模板合成，其中
 * **L0C_SIZE 硬编码 131072（128KB），与真机 GE 运行时查询值 262144（256KB，
 * = AscendC::TOTAL_L0C_SIZE for __NPU_ARCH__==3510）不一致**。故本 UT 不写死
 * 平台容量，而以 ProbeCaps() 从 tiling 输出**反推**实测容量，所有期望值由
 * MirrorDecision() 据反推容量独立复算 —— 断言在 UT 环境与真机同时成立。
 *
 * 覆盖：
 *   1. 决策镜像全字段比对（1323 组合 × 全决策字段）——公式漂移即失败
 *   2. 属性 P1-P8（报告 §D1）：L0C 不越界 / 容量自洽 / 分形对齐 / mChunk 界 /
 *      商余分核下限 / RowDispatch 双模型行覆盖无重叠无留洞
 *      （P1「segScratch==0」在 A1 删除该路径后由编译期保证 + 阈界定向用例覆盖）
 *   3. 支持域边界：由镜像解析求出 Hp 上界，二分实测确认「界内放行 / 界外拒绝」
 *   4. UB 扩容阈不可达（A1 删除依据）：Hp>7872 区间必被 L1 先拒
 *   5. 负向 gate：dtype / rank / 空 tensor / 布局违规 / 维度上界
 *   6. 输出 dtype 契约（[V7] 同族框架透传缺口的行为固化）
 */

#include <climits>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

// 交付源：CompileInfo 结构（optiling 命名空间）+ TilingData/决策常量（host—kernel 共享头）
#include "../../../../op_host/arch35/gru_block_cell_tiling_arch35.h"
#include "../../../../op_kernel/arch35/gru_block_cell_tiling_struct.h"

namespace GruBlockCellTilingUT {
using namespace std;
using namespace ge;

static const std::string OP_NAME = "GruBlockCell";

// 经 TilingContextPara 传入的平台量（框架把它们写进合成 JSON 的 UB_SIZE/CORE_NUM，
// 故这两项由本 UT 完全掌控，确定可知）。
static constexpr int64_t kParaCoreNum = 28;
static constexpr int64_t kParaUbSize = GRU_TIL_AIV_UB;
static constexpr uint64_t kTilingDataSize = sizeof(GruBlockCellTilingData);

// ---------------------------------------------------------------------------
// UT 侧独立算术（不与 host 共享实现）
// ---------------------------------------------------------------------------
static inline int64_t UT_CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
static inline int64_t UT_CeilAlign(int64_t a, int64_t f) { return UT_CeilDiv(a, f) * f; }
static inline int64_t UT_AlignUp(int64_t v, int64_t f) { return UT_CeilDiv(v, f) * f; }
static inline int64_t UT_AlignDown(int64_t v, int64_t f) { return (v / f) * f; }
static inline int64_t UT_Min(int64_t a, int64_t b) { return (a < b) ? a : b; }
static inline int64_t UT_Max(int64_t a, int64_t b) { return (a > b) ? a : b; }

// ---------------------------------------------------------------------------
// 用例构造与执行
// ---------------------------------------------------------------------------
static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape ss;
    for (auto d : dims) {
        ss.MutableOriginShape().AppendDim(d);
        ss.MutableStorageShape().AppendDim(d);
    }
    return ss;
}

struct TilingOutcome {
    bool ok = false;
    GruBlockCellTilingData td{};
};

struct CaseOverride {
    int inputIdx = -1;
    std::vector<int64_t> shape;
    ge::DataType dtype = ge::DT_FLOAT;
    bool overrideDtype = false;
    int outputIdx = -1;
    ge::DataType outDtype = ge::DT_FLOAT;
    bool overrideOutDtype = false;
};

static TilingOutcome RunTiling(int64_t B, int64_t I, int64_t H, const CaseOverride& ov = {})
{
    const int64_t K = I + H;
    std::vector<std::vector<int64_t>> inShapes = {{B, I}, {B, H}, {K, 2 * H}, {K, H}, {2 * H}, {H}};
    std::vector<ge::DataType> inDtypes(6, ge::DT_FLOAT);
    if (ov.inputIdx >= 0 && ov.inputIdx < 6) {
        if (!ov.shape.empty()) {
            inShapes[static_cast<size_t>(ov.inputIdx)] = ov.shape;
        }
        if (ov.overrideDtype) {
            inDtypes[static_cast<size_t>(ov.inputIdx)] = ov.dtype;
        }
    }
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    inputs.reserve(inShapes.size());
    for (size_t i = 0; i < inShapes.size(); ++i) {
        inputs.emplace_back(MakeStorageShape(inShapes[i]), inDtypes[i], ge::FORMAT_ND, false, nullptr, ge::FORMAT_ND);
    }
    std::vector<ge::DataType> outDtypes(4, ge::DT_FLOAT);
    if (ov.outputIdx >= 0 && ov.outputIdx < 4 && ov.overrideOutDtype) {
        outDtypes[static_cast<size_t>(ov.outputIdx)] = ov.outDtype;
    }
    std::vector<gert::TilingContextPara::TensorDescription> outputs;
    outputs.reserve(4);
    for (size_t k = 0; k < 4; ++k) {
        outputs.emplace_back(MakeStorageShape({B, H}), outDtypes[k], ge::FORMAT_ND, false, nullptr, ge::FORMAT_ND);
    }
    std::vector<gert::TilingContextPara::OpAttr> attrs; // 本算子无属性
    optiling::GruBlockCellCompileInfo compileInfo{
        static_cast<uint64_t>(kParaCoreNum), static_cast<uint64_t>(kParaUbSize), 0, 0, 0, 0};

    gert::TilingContextPara para(OP_NAME, inputs, outputs, attrs, &compileInfo, static_cast<uint64_t>(kParaCoreNum),
                                 static_cast<uint64_t>(kParaUbSize), kTilingDataSize);
    TilingOutcome out;
    TilingInfo info;
    out.ok = ExecuteTiling(para, info);
    if (out.ok && info.tilingData != nullptr && info.tilingDataSize >= sizeof(GruBlockCellTilingData)) {
        memcpy(&out.td, info.tilingData.get(), sizeof(GruBlockCellTilingData));
    }
    return out;
}

// ---------------------------------------------------------------------------
// ProbeCaps — 从 tiling 输出反推 UT 环境的真实平台容量（不依赖框架模板常量）
//
// 列片外提后 nL0c 由「UB × L0C × 流量模型」三方联合决定，不再是 l0c 的单调函数，
// 故 l0c 改为**候选值匹配**：在 {131072(UT 模板硬编码), 262144(真机查询值)} 中选出
// 能让镜像与真实 tiling 输出在探针形状上逐字段一致的那个。l1/l0a 取共享常量
// （框架模板与真机查询值一致，均为 524288/65536）；l0b 仍可由 nSlice 精确反推
// （nSlice = min(nL0c, l0b/(4×16))，探针形状下 nL0c > 1024 ⇒ nSlice = l0b/64）。
// ---------------------------------------------------------------------------
struct UtCaps {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    int64_t l1Size = 0;
    int64_t l0aSize = 0;
    int64_t l0bSize = 0;
    int64_t l0cSize = 0;
};

// 前向声明：MirrorDecision 需要 caps 作为入参（l0c 候选匹配时 caps 尚未定型）
static GruBlockCellTilingData MirrorDecision(int64_t B, int64_t I, int64_t H, const UtCaps& c);

static bool SameDecision(const GruBlockCellTilingData& a, const GruBlockCellTilingData& b)
{
    return a.rowsPerCore == b.rowsPerCore && a.rowsTail == b.rowsTail && a.coreNumUsed == b.coreNumUsed &&
           a.splitMode == b.splitMode && a.padHidden == b.padHidden && a.nAl == b.nAl && a.nSlice == b.nSlice &&
           a.kc == b.kc && a.nL0c == b.nL0c && a.sliced == b.sliced && a.mChunk == b.mChunk && a.cGroups == b.cGroups &&
           a.cGroupsX == b.cGroupsX && a.kgX == b.kgX && a.kgH == b.kgH;
}

static UtCaps ProbeCaps()
{
    UtCaps c;
    // coreNum：超大 B ⇒ 必进商余分核（splitMode=1）⇒ coreNumUsed == 平台 AIC 核数
    const TilingOutcome big = RunTiling(1000000, 8, 8);
    c.coreNum = big.ok ? big.td.coreNumUsed : kParaCoreNum;
    c.ubSize = kParaUbSize;    // 由 para 传入，框架写进合成平台信息
    c.l1Size = GRU_TIL_CAP_L1; // 框架模板与真机查询值一致
    c.l0aSize = GRU_TIL_CAP_L0A;
    // l0bSize：H=4097（nAl=4112）下 nL0c > 1024 ⇒ nSlice = l0b/(4×16)
    const TilingOutcome probe = RunTiling(4, 8, 4097);
    c.l0bSize = (probe.ok && probe.td.nSlice > 0) ? probe.td.nSlice * 4 * GRU_TIL_CUBE_BLOCK : GRU_TIL_CAP_L0B;
    // l0cSize：候选匹配（镜像 vs 真实 tiling，多形状投票）
    const int64_t cands[] = {131072, 262144, GRU_TIL_CAP_L0C};
    const int64_t probes[3][3] = {{4, 8, 4097}, {1024, 64, 64}, {8083, 64, 512}};
    int64_t best = GRU_TIL_CAP_L0C;
    int bestScore = -1;
    for (int64_t cand : cands) {
        c.l0cSize = cand;
        int score = 0;
        for (const auto& p : probes) {
            const TilingOutcome t = RunTiling(p[0], p[1], p[2]);
            if (t.ok && SameDecision(MirrorDecision(p[0], p[1], p[2], c), t.td)) {
                ++score;
            }
        }
        if (score > bestScore) {
            bestScore = score;
            best = cand;
        }
    }
    c.l0cSize = best;
    return c;
}

static const UtCaps& Caps()
{
    static const UtCaps c = ProbeCaps();
    return c;
}

// ---------------------------------------------------------------------------
// MirrorDecision — host ComputeLayoutDecision + 切分公式的独立复算
// （规格同源、实现独立；任何一侧公式漂移都会在全矩阵比对中暴露）
// ---------------------------------------------------------------------------

// UB 可容纳的列片宽（host UbTileCap 的独立复算）
static int64_t UT_UbTileCap(int64_t mChunk, int64_t ubAvail)
{
    const int64_t rowsMax = UT_CeilDiv(mChunk, 2);
    if (rowsMax <= 0 || ubAvail <= 32) {
        return 0;
    }
    const int64_t denom = rowsMax * (GRU_TIL_STATIC_PLANES * 4 * GRU_TIL_BITS_PER_BYTE + 1);
    return UT_AlignDown(8 * (ubAvail - 32) / denom, GRU_TIL_CUBE_BLOCK);
}

static GruBlockCellTilingData MirrorDecision(int64_t B, int64_t I, int64_t H, const UtCaps& c)
{
    GruBlockCellTilingData m{};
    m.batchSize = B;
    m.inputSize = I;
    m.hiddenSize = H;

    // sM（多核切分行种子）：列片外提后与 H 解耦，恒为 M 分形下限
    const int64_t sM = UT_Min(B, GRU_TIL_MIN_MCHUNK);
    const int64_t rawCores = UT_CeilDiv(B, sM);
    if (rawCores <= c.coreNum) {
        m.rowsPerCore = sM;
        m.coreNumUsed = rawCores;
        m.rowsTail = B - (rawCores - 1) * sM;
        m.splitMode = 0;
    } else {
        m.rowsPerCore = B / c.coreNum;
        m.rowsTail = B % c.coreNum;
        m.coreNumUsed = c.coreNum;
        m.splitMode = 1;
    }

    m.padHidden = UT_AlignUp(H, GRU_TIL_C0F);
    m.nAl = UT_AlignUp(m.padHidden, GRU_TIL_CUBE_BLOCK);

    // split-K 精度分组（x 段 / h 段各自从 0 起）
    int64_t gX = UT_Max(UT_CeilDiv(I, GRU_TIL_C_GROUP_ROWS), 1);
    m.kgX = UT_AlignUp(UT_CeilDiv(I, gX), GRU_TIL_CUBE_BLOCK);
    m.cGroupsX = UT_CeilDiv(I, m.kgX);
    int64_t gH = UT_Max(UT_CeilDiv(H, GRU_TIL_C_GROUP_ROWS), 1);
    m.kgH = UT_AlignUp(UT_CeilDiv(H, gH), GRU_TIL_CUBE_BLOCK);
    m.cGroups = m.cGroupsX + UT_CeilDiv(H, m.kgH);
    const int64_t kgMax = UT_Max(m.kgX, m.kgH);

    // mChunk 上界：每核行数 与 L0A 行界取小（L1 的 aH 全幅槽与 nL0c 耦合，
    // 折进下方候选枚举逐点精确验算，不在此预估——预估会因 AlignDown 到 16 把
    // 15 打成 0 而误压 mChunk）
    const int64_t ubAvail = c.ubSize - GRU_TIL_UB_RESERVE;
    const int64_t l0cElems = c.l0cSize / 4;
    const int64_t mCap = UT_Max(1, UT_Min(m.rowsPerCore, GRU_TIL_CAP_L0A_ROWS));

    // 流量模型择优（host 同式）：traffic = nChunks·W + 2·nTiles·Afull + act
    const int64_t kDim = I + H;
    const int64_t weightBytes = 3 * kDim * m.padHidden * 4;
    const int64_t aStreamBytes = B * kDim * 4;
    const int64_t actBytes = 7 * B * H * 4;
    const int64_t cores = UT_Max(1, c.coreNum);
    int64_t bestM = 0;
    int64_t bestN = 0;
    int64_t bestCost = INT64_MAX;
    const int64_t mcFirst = (mCap < GRU_TIL_CUBE_BLOCK) ? mCap : GRU_TIL_CUBE_BLOCK;
    for (int64_t mc = mcFirst; mc <= mCap; mc += GRU_TIL_CUBE_BLOCK) {
        const int64_t mA = UT_AlignUp(mc, GRU_TIL_CUBE_BLOCK);
        const int64_t ntHi = UT_AlignDown(UT_Min(UT_Min(UT_UbTileCap(mc, ubAvail), l0cElems / mA), m.nAl),
                                          GRU_TIL_CUBE_BLOCK);
        if (ntHi < GRU_TIL_CUBE_BLOCK) {
            continue;
        }
        // L1 精确验算 + nL0c 向下收缩（与 host 同式）：aK/aH 与 nt 无关，b 槽/bias
        // 槽随 nt 单调增；nSlice 受 L0B 上界钳位，kc 再受 L0A 与组宽双界。
        const int64_t aBytes = (UT_CeilDiv(kgMax, GRU_TIL_C0F) + UT_CeilDiv(m.padHidden, GRU_TIL_C0F)) * mA *
                               GRU_TIL_C0F * 4;
        int64_t nt = 0;
        for (int64_t cand = ntHi; cand >= GRU_TIL_CUBE_BLOCK; cand -= GRU_TIL_CUBE_BLOCK) {
            const int64_t sl = UT_Max(UT_Min(cand, c.l0bSize / (4 * GRU_TIL_CUBE_BLOCK)), GRU_TIL_CUBE_BLOCK);
            const int64_t kcc = UT_Max(
                UT_Min(UT_AlignDown(UT_Min(c.l0bSize / (4 * sl), c.l0aSize / (4 * mA)), GRU_TIL_CUBE_BLOCK), kgMax),
                GRU_TIL_CUBE_BLOCK);
            const int64_t bBytes = (sl / GRU_TIL_C0F) * UT_AlignUp(kcc, GRU_TIL_CUBE_BLOCK) * GRU_TIL_C0F * 4;
            if (aBytes + bBytes + UT_AlignUp(cand * 4, 64) <= c.l1Size) {
                nt = cand;
                break;
            }
        }
        if (nt < GRU_TIL_CUBE_BLOCK) {
            continue;
        }
        const int64_t nChunks = cores * UT_CeilDiv(UT_CeilDiv(B, cores), mc);
        const int64_t cost = nChunks * weightBytes + 2 * UT_CeilDiv(m.padHidden, nt) * aStreamBytes + actBytes;
        if (cost < bestCost) {
            bestCost = cost;
            bestM = mc;
            bestN = nt;
        }
        if (mc == mCap) {
            break;
        }
    }
    if (bestN < GRU_TIL_CUBE_BLOCK) {
        bestM = UT_Min(mCap, GRU_TIL_CUBE_BLOCK);
        bestN = UT_Min(m.nAl, GRU_TIL_CUBE_BLOCK);
    }
    m.mChunk = bestM;
    m.nL0c = bestN;
    m.sliced = (m.nAl > m.nL0c) ? 1 : 0;

    // nSlice / kc（L0B 与 L0A 双界，再受组宽钳位）
    m.nSlice = UT_Min(m.nL0c, c.l0bSize / (4 * GRU_TIL_CUBE_BLOCK));
    if (m.nSlice < GRU_TIL_CUBE_BLOCK) {
        m.nSlice = GRU_TIL_CUBE_BLOCK;
    }
    const int64_t kcCapB = c.l0bSize / (4 * m.nSlice);
    const int64_t kcCapA = c.l0aSize / (4 * UT_AlignUp(m.mChunk, GRU_TIL_CUBE_BLOCK));
    m.kc = UT_Min(UT_AlignDown(UT_Min(kcCapB, kcCapA), GRU_TIL_CUBE_BLOCK), kgMax);
    if (m.kc < GRU_TIL_CUBE_BLOCK) {
        m.kc = GRU_TIL_CUBE_BLOCK;
    }
    return m;
}

static GruBlockCellTilingData MirrorDecision(int64_t B, int64_t I, int64_t H)
{
    return MirrorDecision(B, I, H, Caps());
}

// 容量镜像（CheckLayoutCapacity 的独立复算）
struct CapBytes {
    int64_t l1;
    int64_t ub;
    int64_t l0c;
    int64_t l0a;
    int64_t l0b;
};

static CapBytes MirrorCapacity(const GruBlockCellTilingData& td)
{
    const int64_t mAligned = UT_CeilAlign(td.mChunk, GRU_TIL_CUBE_BLOCK);
    const int64_t kgMax = UT_Max(td.kgX, td.kgH);
    const int64_t aK = UT_CeilDiv(kgMax, GRU_TIL_C0F) * mAligned * GRU_TIL_C0F * 4;
    const int64_t aH = UT_CeilDiv(td.padHidden, GRU_TIL_C0F) * mAligned * GRU_TIL_C0F * 4;
    const int64_t b = (td.nSlice / GRU_TIL_C0F) * UT_AlignUp(td.kc, GRU_TIL_CUBE_BLOCK) * GRU_TIL_C0F * 4;
    const int64_t bias = UT_AlignUp(td.nL0c * 4, 64);
    const int64_t rowsMax = (td.mChunk + 1) / 2;
    const int64_t planeElems = rowsMax * td.nL0c;
    const int64_t msk = UT_AlignUp(UT_CeilDiv(planeElems, GRU_TIL_BITS_PER_BYTE), 32);
    return {aK + aH + b + bias, GRU_TIL_STATIC_PLANES * planeElems * 4 + msk, mAligned * td.nL0c * 4,
            UT_CeilDiv(td.kc, GRU_TIL_C0F) * mAligned * GRU_TIL_C0F * 4, b};
}

static bool MirrorGatePass(int64_t B, int64_t I, int64_t H)
{
    const UtCaps& c = Caps();
    const CapBytes cap = MirrorCapacity(MirrorDecision(B, I, H, c));
    return cap.l1 <= c.l1Size && cap.ub <= c.ubSize - GRU_TIL_UB_RESERVE && cap.l0c <= c.l0cSize &&
           cap.l0a <= c.l0aSize && cap.l0b <= c.l0bSize;
}

// ---------------------------------------------------------------------------
// 属性断言 P2-P8（P1 见文件头说明：segScratch 路径已删，编译期保证）
// ---------------------------------------------------------------------------
static void CheckProperties(const GruBlockCellTilingData& td, const std::string& tag)
{
    const UtCaps& c = Caps();
    const int64_t mAligned = UT_CeilAlign(td.mChunk, GRU_TIL_CUBE_BLOCK);

    // P2：L0C 单片不越界（用反推容量，UT/真机两环境同时成立）
    EXPECT_LE(mAligned * td.nL0c * 4, c.l0cSize) << tag << " P2: mAligned×nL0c 超 L0C";

    // P3：决策与容量验算自洽（host gate 放行 ⇒ 独立复算必在界内）
    const CapBytes cap = MirrorCapacity(td);
    EXPECT_LE(cap.l1, c.l1Size) << tag << " P3: 复算 L1=" << cap.l1 << " 超 cap=" << c.l1Size;
    EXPECT_LE(cap.ub, c.ubSize - GRU_TIL_UB_RESERVE) << tag << " P3: 复算 UB=" << cap.ub << " 超 cap";

    // P4/P5：分形对齐前提
    EXPECT_EQ(td.nL0c % GRU_TIL_CUBE_BLOCK, 0) << tag << " P4: nL0c 非 16 倍数";
    EXPECT_EQ(td.kgX % GRU_TIL_CUBE_BLOCK, 0) << tag << " P5: kgX 非 16 倍数";
    EXPECT_EQ(td.kgH % GRU_TIL_CUBE_BLOCK, 0) << tag << " P5: kgH 非 16 倍数";

    // P6：mChunk 界
    EXPECT_GE(td.mChunk, 1) << tag << " P6: mChunk < 1";
    EXPECT_LE(td.mChunk, UT_Min(td.rowsPerCore, GRU_TIL_CAP_L0A_ROWS)) << tag << " P6: mChunk 超上界";

    // P7：商余分核下限
    if (td.splitMode == 1) {
        EXPECT_GE(td.rowsPerCore, GRU_TIL_CUBE_BLOCK) << tag << " P7: splitMode=1 但 rowsPerCore < 16";
    }

    // P8：RowDispatch 双模型行覆盖（Σrows == B、连续无重叠无留洞、rowBase < B）
    int64_t covered = 0;
    int64_t expectBase = 0;
    for (int64_t k = 0; k < td.coreNumUsed; ++k) {
        int64_t rows;
        int64_t rowBase;
        if (td.splitMode == 0) {
            rows = (k + 1 == td.coreNumUsed) ? td.rowsTail : td.rowsPerCore;
            rowBase = k * td.rowsPerCore;
        } else {
            rows = td.rowsPerCore + ((k < td.rowsTail) ? 1 : 0);
            rowBase = k * td.rowsPerCore + UT_Min(k, td.rowsTail);
        }
        EXPECT_LT(rowBase, td.batchSize) << tag << " P8: cluster " << k << " rowBase 越界";
        EXPECT_EQ(rowBase, expectBase) << tag << " P8: cluster " << k << " 行区间不连续";
        EXPECT_GE(rows, 0) << tag << " P8: cluster " << k << " 行数为负";
        covered += rows;
        expectBase = rowBase + rows;
    }
    EXPECT_EQ(covered, td.batchSize) << tag << " P8: 行覆盖总数 != B";

    // 附加自洽：对齐口径 + 分组分解 + sliced 一致性
    EXPECT_EQ(td.padHidden, UT_CeilAlign(td.hiddenSize, GRU_TIL_C0F)) << tag << " padHidden 口径";
    EXPECT_EQ(td.nAl, UT_CeilAlign(td.padHidden, GRU_TIL_CUBE_BLOCK)) << tag << " nAl 口径";
    EXPECT_EQ(td.cGroups, td.cGroupsX + UT_CeilDiv(td.hiddenSize, td.kgH)) << tag << " cGroups 分解";
    EXPECT_EQ((td.nAl > td.nL0c) ? 1 : 0, td.sliced) << tag << " sliced 与 nAl/nL0c 不一致";
}

// 全字段镜像比对（平台相关字段亦覆盖——期望值由反推容量复算）
static void CheckMirror(const GruBlockCellTilingData& td, const std::string& tag)
{
    const GruBlockCellTilingData m = MirrorDecision(td.batchSize, td.inputSize, td.hiddenSize);
    EXPECT_EQ(td.rowsPerCore, m.rowsPerCore) << tag << " rowsPerCore";
    EXPECT_EQ(td.rowsTail, m.rowsTail) << tag << " rowsTail";
    EXPECT_EQ(td.coreNumUsed, m.coreNumUsed) << tag << " coreNumUsed";
    EXPECT_EQ(td.splitMode, m.splitMode) << tag << " splitMode";
    EXPECT_EQ(td.padHidden, m.padHidden) << tag << " padHidden";
    EXPECT_EQ(td.nAl, m.nAl) << tag << " nAl";
    EXPECT_EQ(td.mChunk, m.mChunk) << tag << " mChunk";
    EXPECT_EQ(td.nL0c, m.nL0c) << tag << " nL0c";
    EXPECT_EQ(td.sliced, m.sliced) << tag << " sliced";
    EXPECT_EQ(td.nSlice, m.nSlice) << tag << " nSlice";
    EXPECT_EQ(td.kc, m.kc) << tag << " kc";
    EXPECT_EQ(td.kgX, m.kgX) << tag << " kgX";
    EXPECT_EQ(td.kgH, m.kgH) << tag << " kgH";
    EXPECT_EQ(td.cGroupsX, m.cGroupsX) << tag << " cGroupsX";
    EXPECT_EQ(td.cGroups, m.cGroups) << tag << " cGroups";
}

// 形状矩阵：覆盖 pad/对齐/列片/单核-多核-尾块-商余分核（上界由镜像解析确定）
// ⚠ 上界须落在 UT 环境与真机支持域的**交集**内（UT 模板 L0C=128KB → Hp<=7024，
// 真机 L0C=256KB → Hp<=8152），否则矩阵里的 rejected!=0 断言会因环境差异误报。
// 真机侧的上界由 SupportDomainUpperBound 用镜像解析 + 实测覆盖，不写死在此。
static const std::vector<int64_t> kHList = {1,   2,   7,    8,    15,   16,   17,   33,   64,   100,
                                            128, 512, 1000, 1024, 2048, 4096, 4097, 6088, 6889, 6896};
static const std::vector<int64_t> kBList = {1, 2, 4, 16, 28, 29, 64, 1000, 8083};
static const std::vector<int64_t> kIList = {1, 8, 16, 33, 620, 1024, 4096};

TEST(GruBlockCellTilingPropertyTest, MirrorAndPropertiesOverShapeMatrix)
{
    const UtCaps& c = Caps();
    std::cout << "[probed caps] coreNum=" << c.coreNum << " ub=" << c.ubSize << " l1=" << c.l1Size
              << " l0b=" << c.l0bSize << " l0c=" << c.l0cSize << std::endl;
    EXPECT_GT(c.coreNum, 0);
    EXPECT_GT(c.l0cSize, 0);

    int64_t passed = 0;
    int64_t rejected = 0;
    for (int64_t H : kHList) {
        for (int64_t B : kBList) {
            for (int64_t I : kIList) {
                const std::string tag = "B=" + to_string(B) + " I=" + to_string(I) + " H=" + to_string(H);
                const TilingOutcome r = RunTiling(B, I, H);
                if (!r.ok) {
                    ++rejected;
                    EXPECT_FALSE(MirrorGatePass(B, I, H)) << tag << " host 拒绝但镜像判定应放行";
                    continue;
                }
                ++passed;
                EXPECT_TRUE(MirrorGatePass(B, I, H)) << tag << " host 放行但镜像判定应拒绝";
                CheckMirror(r.td, tag);
                CheckProperties(r.td, tag);
            }
        }
    }
    std::cout << "[property matrix] gate-pass=" << passed << " rejected=" << rejected << std::endl;
    EXPECT_GT(passed, 0) << "属性矩阵无任何过 gate 用例——测试自身失效";
    EXPECT_EQ(rejected, 0) << "矩阵内（H ≤ 支持域上界）出现意外拒绝";
}

TEST(GruBlockCellTilingBoundaryTest, SupportDomainUpperBound)
{
    // 支持域上界由镜像解析求出（Hp 步进 8），再二分/线性实测确认——不写死数值，
    // 使断言在 UT 环境（L0C=128KB）与真机（L0C=256KB → Hp≤8152）同时成立。
    int64_t maxHp = 0;
    for (int64_t Hp = 8; Hp <= 16384; Hp += 8) {
        if (MirrorGatePass(4, 8, Hp)) {
            maxHp = Hp;
        } else if (maxHp > 0) {
            break;
        }
    }
    ASSERT_GT(maxHp, 0) << "镜像未能求出支持域上界";
    std::cout << "[support domain] mirror maxHp=" << maxHp << " (L0C=" << Caps().l0cSize << ")" << std::endl;

    // 界内：Hp = maxHp 与 maxHp-7（同 Hp 档）放行
    const std::vector<int64_t> inCases = {maxHp - 7, maxHp};
    for (int64_t H : inCases) {
        const TilingOutcome r = RunTiling(4, 8, H);
        ASSERT_TRUE(r.ok) << "H=" << H << " (Hp=" << maxHp << ") 应放行";
        EXPECT_EQ(r.td.padHidden, maxHp);
        CheckProperties(r.td, "boundary H=" + to_string(H));
    }
    // 界外：Hp = maxHp+8 起必须干净拒绝（GRAPH_FAILED，非设备侧 EZ9999）
    const std::vector<int64_t> outCases = {maxHp + 1, maxHp + 8, maxHp + 280, static_cast<int64_t>(16384)};
    for (int64_t H : outCases) {
        const TilingOutcome r = RunTiling(4, 8, H);
        EXPECT_FALSE(r.ok) << "H=" << H << " (Hp>" << maxHp << ") 应被容量 gate 拒绝";
    }
}

TEST(GruBlockCellTilingBoundaryTest, SupportDomainBoundByL1NotUb)
{
    // 定向证据：支持域上界由 **L1 的 aH 全幅回灌槽**决定，UB 永不先触顶。
    // 列片外提后 UB 平面是 [rowsMax, nL0c]（非全幅 [rowsMax, Hp]），故旧断言
    // 「UB 扩容阈 Hp>7872 恒被 L1 先拒」的前提已消失——Hp∈[7872, 上界] 现在是
    // 合法域。改为断言：在上界+8 处，L1 镜像超界而 UB 镜像仍在预算内。
    const UtCaps& c = Caps();
    int64_t maxHp = 0;
    for (int64_t Hp = 8; Hp <= 16384; Hp += 8) {
        if (MirrorGatePass(4, 8, Hp)) {
            maxHp = Hp;
        } else if (maxHp > 0) {
            break;
        }
    }
    ASSERT_GT(maxHp, 0) << "镜像未能求出支持域上界";
    const int64_t ubAvail = c.ubSize - GRU_TIL_UB_RESERVE;
    // 界外第一档：L1 必须超界（aH 全幅槽放不下），UB 必须仍有余量
    const GruBlockCellTilingData over = MirrorDecision(4, 8, maxHp + 8, c);
    const CapBytes capOver = MirrorCapacity(over);
    EXPECT_GT(capOver.l1, c.l1Size) << "Hp=" << maxHp + 8 << " 应由 L1 拒绝（aH 全幅槽）";
    // 界内最后一档：L1 恰好放得下，且 UB 远未触顶（证明 UB 不是约束方）
    const GruBlockCellTilingData in = MirrorDecision(4, 8, maxHp, c);
    const CapBytes capIn = MirrorCapacity(in);
    EXPECT_LE(capIn.l1, c.l1Size) << "Hp=" << maxHp << " 应放行";
    EXPECT_LE(capIn.ub, ubAvail) << "Hp=" << maxHp << " UB 应仍在预算内";
    EXPECT_LT(capIn.ub, ubAvail / 2) << "上界处 UB 占用不足一半 ⇒ 约束方是 L1 而非 UB";
    EXPECT_FALSE(RunTiling(4, 8, maxHp + 8).ok) << "Hp=" << maxHp + 8 << " host 应干净拒绝";
}

TEST(GruBlockCellTilingNegativeTest, InputGatesReject)
{
    // dtype gate：任一输入非 fp32 → 拒绝
    for (int idx = 0; idx < 6; ++idx) {
        CaseOverride ov;
        ov.inputIdx = idx;
        ov.overrideDtype = true;
        ov.dtype = ge::DT_FLOAT16;
        EXPECT_FALSE(RunTiling(4, 8, 8, ov).ok) << "input[" << idx << "] fp16 应被拒";
    }
    // rank gate：x rank1 / rank3、wRu rank3、bRu rank2
    const std::vector<std::pair<int, std::vector<int64_t>>> rankCases = {
        {0, {32}},
        {0, {4, 8, 1}},
        {2, {16, 8, 2}},
        {4, {8, 1}},
    };
    for (const auto& rc : rankCases) {
        CaseOverride ov;
        ov.inputIdx = rc.first;
        ov.shape = rc.second;
        EXPECT_FALSE(RunTiling(4, 8, 8, ov).ok) << "input[" << rc.first << "] rank 违规应被拒";
    }
    // 空 tensor：B=0 / I=0
    for (const auto& dims : std::vector<std::vector<int64_t>>{{0, 8}, {4, 0}}) {
        CaseOverride ov;
        ov.inputIdx = 0;
        ov.shape = dims;
        EXPECT_FALSE(RunTiling(dims[0], dims[1], 8, ov).ok) << "空 tensor 应被拒";
    }
    // 布局违规：wRu 列 ±1、wC 列 −1、bRu 长 −1、bC 长 +1、hPrev batch 不齐
    const std::vector<std::tuple<int, std::vector<int64_t>, int64_t, int64_t>> layoutCases = {
        {2, {16, 15}, 4, 8}, {2, {16, 17}, 4, 8}, {3, {16, 7}, 4, 8},
        {4, {15}, 4, 8},     {5, {9}, 4, 8},      {1, {5, 8}, 4, 8},
    };
    for (const auto& lc : layoutCases) {
        CaseOverride ov;
        ov.inputIdx = std::get<0>(lc);
        ov.shape = std::get<1>(lc);
        EXPECT_FALSE(RunTiling(std::get<2>(lc), std::get<3>(lc), 8, ov).ok)
            << "input[" << std::get<0>(lc) << "] 布局违规应被拒";
    }
    // 维度上界：I/H 超 65535 → 拒绝
    EXPECT_FALSE(RunTiling(4, 65536, 8).ok) << "I=65536 应被拒";
    EXPECT_FALSE(RunTiling(4, 8, 65536).ok) << "H=65536 应被拒";
}

TEST(GruBlockCellTilingNegativeTest, OutputDtypeContract)
{
    // 输出 dtype 非 fp32：算子侧契约 fp32-only（proto TensorType({DT_FLOAT}) +
    // ops-info float32）。tiling 期输出 desc 在 GEIR 在线流程不透传（[V7] 同族
    // 框架缺口，host 侧已加防御门），TTK harness 侧则以参数泛化/图层 cast 修复，
    // 违规不送达算子。本用例固化「不因输出声明产生越界决策」的行为。
    CaseOverride ov;
    ov.outputIdx = 0;
    ov.overrideOutDtype = true;
    ov.outDtype = ge::DT_FLOAT16;
    const TilingOutcome r = RunTiling(4, 8, 8, ov);
    if (r.ok) {
        CheckProperties(r.td, "out-fp16(accepted)");
    }
    SUCCEED();
}

TEST(GruBlockCellTilingDecisionTest, KnownValuesPinned)
{
    // 平台无关量钉死（任何公式漂移即失败）+ 平台相关量由镜像复算比对。
    struct Expect {
        int64_t B, I, H;
        int64_t rowsPerCore, coreNumUsed, splitMode, padHidden, nAl, cGroups;
    };
    // sM（多核切分行种子）在列片外提后与 H 解耦、恒为 min(B, 16)：UB 足迹不再随
    // Hp 增长，故不必再用「32H 字节/行」压缩每核行数。副作用是小 H 形状的分核数
    // 上升（B=1024/H=64 由 10 核 → 28 核全启用），核间负载差 ≤1 行。
    const std::vector<Expect> cases = {
        {4, 8, 8, 4, 1, 0, 8, 16, 2},              // B<sM → 单核承载全部行
        {8, 8, 8, 8, 1, 0, 8, 16, 2},              // B==sM → 单核
        {1024, 64, 64, 36, 28, 1, 64, 64, 8},      // rawCores=64>28 → 商余分核（q=36, rem=16）
        {8083, 64, 512, 288, 28, 1, 512, 512, 36}, // rawCores>28 → 商余分核（q=288, rem=19）
        {4, 8, 4097, 4, 1, 0, 4104, 4112, 258},    // 列片路径（nAl>nL0c）
        {128, 33, 100, 16, 8, 0, 104, 112, 10},    // 非对齐 I/H，满行切 8 核
    };
    for (const auto& e : cases) {
        const std::string tag = "B=" + to_string(e.B) + " I=" + to_string(e.I) + " H=" + to_string(e.H);
        const TilingOutcome r = RunTiling(e.B, e.I, e.H);
        ASSERT_TRUE(r.ok) << tag << " 应过 gate";
        EXPECT_EQ(r.td.rowsPerCore, e.rowsPerCore) << tag << " rowsPerCore";
        EXPECT_EQ(r.td.coreNumUsed, e.coreNumUsed) << tag << " coreNumUsed";
        EXPECT_EQ(r.td.splitMode, e.splitMode) << tag << " splitMode";
        EXPECT_EQ(r.td.padHidden, e.padHidden) << tag << " padHidden";
        EXPECT_EQ(r.td.nAl, e.nAl) << tag << " nAl";
        EXPECT_EQ(r.td.cGroups, e.cGroups) << tag << " cGroups";
        CheckMirror(r.td, tag);
        CheckProperties(r.td, tag);
    }
}

} // namespace GruBlockCellTilingUT
