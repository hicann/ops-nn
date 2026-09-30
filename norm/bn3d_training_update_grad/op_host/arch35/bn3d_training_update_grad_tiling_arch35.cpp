/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// bn3d_training_update_grad_package/op_host/arch35/bn3d_training_update_grad_tiling_arch35.cpp
// =============================================================================
//
// Host-side TilingFunc for BN3DTrainingUpdateGrad (arch35 / Ascend950).
//
//   Tiling flow: validation order + 四步合轴 + UB double-split + multicore +
//   group + fill/SetTilingKey. TilingKey set {0=base,1=group,2=empty};
//   routing empty>group>base via ShouldUseGroup. The tiling gtest UT is the
//   pinned oracle.
//
//   Platform facts (coreNum / ubSize) are read from context->GetPlatformInfo()
//   via platform_ascendc::PlatformAscendC (the aclnn/OpTilingContextBuilder path
//   does NOT invoke TilingParse, so CompileInfo may be absent — never relied on).
//
//   Implementation is heap-allocation-free (fixed stack arrays only; rank<=8,
//   axisNum<=MAX_PATTERN_RANK) — no std::vector — to keep the tiling deterministic
//   and avoid perturbing the process allocator.
//
//   Validation order (any failure -> GRAPH_FAILED with an OP_LOGE
//   root-cause anchor visible on stdout via ASCEND_SLOG_PRINT_TO_STDOUT):
//     dtype -> format -> rank/shape -> stats-channel -> empty short-circuit.
// =============================================================================

#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"                // IMPL_OP_OPTILING, OP_ADD
#include "op_common/log/log.h"                       // OP_LOGE / OP_CHECK_NULL_WITH_CONTEXT
#include "op_common/op_host/util/platform_util.h"    // platform_ascendc::PlatformAscendC
#include "graph/types.h"                             // ge::Format, ge::GetPrimaryFormat, ge::DataType
#include "bn3d_training_update_grad_tiling_arch35.h" // shared TilingData struct bridge

namespace optiling {

namespace {

// ---- arithmetic helpers ---------------------------------------------------
inline int64_t CeilDiv(int64_t a, int64_t b) { return (b == 0) ? 0 : (a + b - 1) / b; }
inline int64_t CeilAlign(int64_t v, int64_t f) { return (f == 0) ? v : CeilDiv(v, f) * f; }
inline int64_t FloorAlign(int64_t v, int64_t f) { return (f == 0) ? v : (v / f) * f; }

// ---- platform / paradigm constants ----------------------------------------
constexpr int64_t kBlockSize = 32;   // UB block bytes (Ascend950)
constexpr int64_t kMaxDtypeSize = 4; // compute dtype 恒 fp32 → max(sizeof(D_T),4)=4
constexpr int64_t kCacheLineSize = 512; // arch35 cache line bytes（Step1 联合爬坡上界；GAP：非 oracle 断言项）
constexpr int64_t kSysWorkspace = 16 * 1024 * 1024; // 框架系统 workspace 兜底（非 oracle 断言项）
constexpr int kMaxRank = 8;

const char* NodeName(gert::TilingContext* context)
{
    const char* n = (context == nullptr) ? nullptr : context->GetNodeName();
    return (n == nullptr) ? "BN3DTrainingUpdateGrad" : n;
}

// origin format → channel(A) axis 物理位置
int LocateChannelAxis(int32_t fmt, int rank)
{
    if (fmt == static_cast<int32_t>(ge::FORMAT_NCDHW) || fmt == static_cast<int32_t>(ge::FORMAT_NCHW)) {
        return 1;
    }
    if (fmt == static_cast<int32_t>(ge::FORMAT_NHWC)) {
        return rank - 1;
    }
    return 1; // 非枚举 format 已在 format 白名单校验拒绝，此处不可达
}

// 读 shape 到定长栈数组（避免堆分配；rank>8 记为非法，返回 n=-1）
struct ShapeBuf {
    int64_t d[kMaxRank];
    int n;
};
ShapeBuf ReadShape(const gert::Shape& s)
{
    ShapeBuf b{};
    const size_t nn = s.GetDimNum();
    if (nn > static_cast<size_t>(kMaxRank)) {
        b.n = -1;
        return b;
    }
    b.n = static_cast<int>(nn);
    for (int i = 0; i < b.n; ++i) {
        b.d[i] = s.GetDim(static_cast<size_t>(i));
    }
    return b;
}

bool ShapeEqual(const ShapeBuf& a, const ShapeBuf& b)
{
    if (a.n != b.n) {
        return false;
    }
    for (int i = 0; i < a.n; ++i) {
        if (a.d[i] != b.d[i]) {
            return false;
        }
    }
    return true;
}

// batch_mean/batch_variance 通道维 == C 且非通道维恒 1
bool MatchesC(const ShapeBuf& ss, int64_t C)
{
    int64_t nonOne = 0, nonOneVal = 1;
    for (int i = 0; i < ss.n; ++i) {
        if (ss.d[i] != 1) {
            ++nonOne;
            nonOneVal = ss.d[i];
        }
    }
    if (nonOne == 0) {
        return C == 1;
    }
    return nonOne == 1 && nonOneVal == C;
}

// -------------------------------------------------------------------------
// 四步合轴 → axisNum/axisShape[4]/aTotal/isTailR。
// 与 UT oracle OracleFuse 同算法（axisNum/axisShape/aTotal 为 pinned 断言项）。定长数组实现。
// -------------------------------------------------------------------------
struct FusedAxes {
    int32_t axisNum = 0;
    int64_t axisShape[MAX_PATTERN_RANK] = {1, 1, 1, 1};
    int64_t aTotal = 1;
    bool isTailR = false;
    bool ok = true;
};

FusedAxes FuseAxes(const ShapeBuf& g, int chAxis)
{
    constexpr int kCap = 16;
    int64_t sz[kCap];
    bool isR[kCap];
    int cnt = 0;

    // 原始 A/R 分类
    int64_t rawSz[kMaxRank];
    bool rawR[kMaxRank];
    for (int i = 0; i < g.n; ++i) {
        rawSz[i] = g.d[i];
        rawR[i] = (i != chAxis);
    }

    // Step1: 去 1 维（删 size==1 的非 reduce/A 轴）
    for (int i = 0; i < g.n; ++i) {
        if (!rawR[i] && rawSz[i] == 1) {
            continue;
        }
        sz[cnt] = rawSz[i];
        isR[cnt] = rawR[i];
        ++cnt;
    }
    // Step2: 合并相邻同类轴
    int64_t sz2[kCap];
    bool isR2[kCap];
    int cnt2 = 0;
    for (int i = 0; i < cnt; ++i) {
        if (cnt2 > 0 && isR2[cnt2 - 1] == isR[i]) {
            sz2[cnt2 - 1] *= sz[i];
        } else {
            sz2[cnt2] = sz[i];
            isR2[cnt2] = isR[i];
            ++cnt2;
        }
    }
    if (cnt2 == 0) {
        sz2[0] = 1;
        isR2[0] = false;
        cnt2 = 1;
    }
    // Step3: 补 leading A=1
    if (isR2[0]) {
        for (int i = cnt2; i > 0; --i) {
            sz2[i] = sz2[i - 1];
            isR2[i] = isR2[i - 1];
        }
        sz2[0] = 1;
        isR2[0] = false;
        ++cnt2;
    }
    // Step4: 补 R 增广（纯 A）
    bool hasR = false;
    for (int i = 0; i < cnt2; ++i) {
        if (isR2[i]) {
            hasR = true;
        }
    }
    if (!hasR) {
        if (cnt2 == 1 && sz2[0] == 1) {
            sz2[1] = 1;
            isR2[1] = true;
            cnt2 = 2;
        } else {
            for (int i = cnt2; i > 0; --i) {
                sz2[i + 1] = sz2[i - 1];
                isR2[i + 1] = isR2[i - 1];
            }
            sz2[0] = 1;
            isR2[0] = false;
            sz2[1] = 1;
            isR2[1] = true;
            cnt2 += 2;
        }
    }

    FusedAxes f;
    f.axisNum = cnt2;
    if (cnt2 < 1 || cnt2 > MAX_PATTERN_RANK) {
        f.ok = false;
        return f;
    }
    for (int i = 0; i < MAX_PATTERN_RANK; ++i) {
        f.axisShape[i] = (i < cnt2) ? sz2[i] : 1;
    }
    f.aTotal = 1;
    for (int i = 0; i < cnt2; i += 2) {
        f.aTotal *= sz2[i];
    }
    f.isTailR = (cnt2 % 2 == 0);
    return f;
}

// EMPTY_R tiling（D_T=fp32）
ge::graphStatus FillEmptyR(gert::TilingContext* context, int64_t C, int64_t coreNum, int64_t ubSize)
{
    auto* e = context->GetTilingData<BN3DTrainingUpdateGradEmptyTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, e);

    const int64_t maxDtypeSize = kMaxDtypeSize;
    const int64_t aTotal = C;                                    // ∏(非 reduce 轴) = 通道 C
    const int64_t maxBufSize = std::min<int64_t>(ubSize, 65536); // maxBufSize = min(ubSize/P_post, 65536)
    const int64_t maxUbFactor = maxBufSize / maxDtypeSize;
    const int64_t minAPerCore = CeilDiv(4096, maxDtypeSize); // = 1024
    int64_t aUbFactor = std::min<int64_t>(std::max<int64_t>(minAPerCore, CeilDiv(aTotal, coreNum)), maxUbFactor);
    aUbFactor = std::min<int64_t>(aUbFactor, aTotal);
    const int64_t aLoopCntTotal = CeilDiv(aTotal, aUbFactor);
    const int64_t aSmallCoreLoopCnt = aLoopCntTotal / coreNum;
    const int64_t aBigCoreCnt = aLoopCntTotal % coreNum;
    const int64_t aBigCoreLoopCnt = aSmallCoreLoopCnt + (aBigCoreCnt > 0 ? 1 : 0);
    int64_t usedCoreNum = (aSmallCoreLoopCnt > 0) ? coreNum : aBigCoreCnt;
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }
    const int64_t postBufSize = CeilAlign(std::max<int64_t>(aUbFactor * maxDtypeSize, kBlockSize), kBlockSize);

    e->usedCoreNum = static_cast<int32_t>(usedCoreNum);
    e->aTotal = aTotal;
    e->aUbFactor = aUbFactor;
    e->aBigCoreCnt = static_cast<int32_t>(aBigCoreCnt);
    e->aBigCoreLoopCnt = aBigCoreLoopCnt;
    e->aSmallCoreLoopCnt = aSmallCoreLoopCnt;
    e->postBufSize = postBufSize;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    context->SetTilingKey(2); // (isGroup=0, isEmptyTensor=1) → key 2
    size_t* ws = context->GetWorkspaceSizes(1);
    if (ws != nullptr) {
        ws[0] = static_cast<size_t>(kSysWorkspace);
    }
    return ge::GRAPH_SUCCESS;
}

// EMPTY_A tiling（零工作量，usedCoreNum=0，其余 0，SetBlockDim(1)）
ge::graphStatus FillEmptyA(gert::TilingContext* context)
{
    auto* e = context->GetTilingData<BN3DTrainingUpdateGradEmptyTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, e);
    e->usedCoreNum = 0;
    e->aTotal = 0;
    e->aUbFactor = 0;
    e->aBigCoreCnt = 0;
    e->aBigCoreLoopCnt = 0;
    e->aSmallCoreLoopCnt = 0;
    e->postBufSize = 0;
    context->SetBlockDim(1); // 框架要求 blockDim ≥ 1，严禁 SetBlockDim(0)
    context->SetTilingKey(2);
    size_t* ws = context->GetWorkspaceSizes(1);
    if (ws != nullptr) {
        ws[0] = static_cast<size_t>(kSysWorkspace);
    }
    return ge::GRAPH_SUCCESS;
}

// -------------------------------------------------------------------------
// 平台参数（coreNum / ubSize）：GetPlatformInfo + PlatformAscendC 直读，
// 不经 TilingParse compileInfo（HOST-4）。
// -------------------------------------------------------------------------
struct PlatformFacts {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

ge::graphStatus GetPlatformFacts(gert::TilingContext* context, const char* node, PlatformFacts& out)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ap = platform_ascendc::PlatformAscendC(platformInfo);
    int64_t coreNum = static_cast<int64_t>(ap.GetCoreNumAiv());
    uint64_t ubU = 0;
    ap.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubU);
    // platform 取值合法性校验（HOST-3）：coreNum 有符号 <=0 拦截、ubSize 无符号 ==0 拦截。
    if (coreNum <= 0) {
        OP_LOGE(node, "invalid platform: coreNum must be > 0, got %ld", coreNum);
        return ge::GRAPH_FAILED;
    }
    if (ubU == 0) {
        OP_LOGE(node, "invalid platform: ubSize must be > 0");
        return ge::GRAPH_FAILED;
    }
    out.coreNum = coreNum;
    out.ubSize = static_cast<int64_t>(ubU);
    return ge::GRAPH_SUCCESS;
}

// -------------------------------------------------------------------------
// 输入获取：4 输入的 Tensor / Desc / StorageShape 判空 + dtype 读取 + shape 快照。
// ⛔ TOPK-2: Dtype 必须经 GetInputDesc(i)->GetDataType() 获取（GetInputTensor 在部分
// tiling context 下可能返回无效 dtype）。ReadShape 为纯计算（无日志/无返回），先于
// 校验执行不改变可观测行为。
// -------------------------------------------------------------------------
struct InputFacts {
    const gert::Tensor* gradsT = nullptr; // origin format 读取用
    const gert::Tensor* xT = nullptr;     // origin format 读取用
    ge::DataType gDtype = ge::DT_UNDEFINED;
    ge::DataType xDtype = ge::DT_UNDEFINED;
    ge::DataType mDtype = ge::DT_UNDEFINED;
    ge::DataType vDtype = ge::DT_UNDEFINED;
    ShapeBuf gradsShape{};
    ShapeBuf xShape{};
    ShapeBuf meanShape{};
    ShapeBuf varShape{};
};

ge::graphStatus GetInputFacts(gert::TilingContext* context, InputFacts& out)
{
    const gert::Tensor* gradsT = context->GetInputTensor(0);
    const gert::Tensor* xT = context->GetInputTensor(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradsT);
    OP_CHECK_NULL_WITH_CONTEXT(context, xT);

    auto gradsD = context->GetInputDesc(0);
    auto xD = context->GetInputDesc(1);
    auto meanD = context->GetInputDesc(2);
    auto varD = context->GetInputDesc(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradsD);
    OP_CHECK_NULL_WITH_CONTEXT(context, xD);
    OP_CHECK_NULL_WITH_CONTEXT(context, meanD);
    OP_CHECK_NULL_WITH_CONTEXT(context, varD);

    const gert::StorageShape* gradsSs = context->GetInputShape(0);
    const gert::StorageShape* xSs = context->GetInputShape(1);
    const gert::StorageShape* meanSs = context->GetInputShape(2);
    const gert::StorageShape* varSs = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradsSs);
    OP_CHECK_NULL_WITH_CONTEXT(context, xSs);
    OP_CHECK_NULL_WITH_CONTEXT(context, meanSs);
    OP_CHECK_NULL_WITH_CONTEXT(context, varSs);

    out.gradsT = gradsT;
    out.xT = xT;
    out.gDtype = gradsD->GetDataType();
    out.xDtype = xD->GetDataType();
    out.mDtype = meanD->GetDataType();
    out.vDtype = varD->GetDataType();
    out.gradsShape = ReadShape(gradsSs->GetStorageShape());
    out.xShape = ReadShape(xSs->GetStorageShape());
    out.meanShape = ReadShape(meanSs->GetStorageShape());
    out.varShape = ReadShape(varSs->GetStorageShape());
    return ge::GRAPH_SUCCESS;
}

// -------------------------------------------------------------------------
// (1)~(4) 输入校验：dtype → format 白名单/一致性/秩一致 → rank/shape → 统计量通道。
// 拦截序（dtype→format→rank→stats）与 OP_LOGE 文案逐字保留；通过后产出
// rank / chAxis / C 供空 tensor 短路与合轴使用。
// -------------------------------------------------------------------------
struct CheckedShape {
    int rank = 0;
    int chAxis = 1;
    int64_t C = 1;
};

ge::graphStatus CheckInputs(const char* node, const InputFacts& in, CheckedShape& out)
{
    // ===== (1) dtype 校验（grads/x∈{fp16,fp32,bf16} 且同 dtype；stats 恒 fp32）=====
    const bool gxOk = (in.gDtype == ge::DT_FLOAT16 || in.gDtype == ge::DT_FLOAT || in.gDtype == ge::DT_BF16);
    if (!gxOk || in.gDtype != in.xDtype) {
        OP_LOGE(node, "invalid dtype: grads/x must be one of {float16,float32,bfloat16} and share the same dtype");
        return ge::GRAPH_FAILED;
    }
    if (in.mDtype != ge::DT_FLOAT || in.vDtype != ge::DT_FLOAT) {
        OP_LOGE(node, "invalid dtype: batch_mean/batch_variance must be float32");
        return ge::GRAPH_FAILED;
    }

    // ===== (2) format 校验（读 origin format，白名单仅 NCDHW/NCHW/NHWC）=====
    // ⛔ 用 GetOriginFormat()（不用 GetStorageFormat()）：OpDef 声明 grads/x storage=ND
    // 且 DynamicFormatFlag(false)，逻辑通道位仅由 origin format 反映（对齐 CANN 先例
    // bn_training_update_grad 读 ori_format）。ND/NDHWC/其它 → 通道轴不可判别，拒绝。
    if (in.gradsShape.n < 0 || in.xShape.n < 0 || in.meanShape.n < 0 || in.varShape.n < 0) {
        OP_LOGE(node, "invalid shape: rank exceeds 8");
        return ge::GRAPH_FAILED;
    }
    const int rank = in.gradsShape.n;
    const int32_t fmt = ge::GetPrimaryFormat(static_cast<int32_t>(in.gradsT->GetOriginFormat()));

    // format 白名单：rank>=2 时 origin format 仅 NCDHW/NCHW/NHWC；ND/NDHWC/其它拒绝
    // （unsupported_format；rank<=1 单轴无歧义豁免）。
    if (rank >= 2 && fmt != static_cast<int32_t>(ge::FORMAT_NCDHW) && fmt != static_cast<int32_t>(ge::FORMAT_NCHW) &&
        fmt != static_cast<int32_t>(ge::FORMAT_NHWC)) {
        OP_LOGE(
            node,
            "unsupported format: grads format must be NCDHW/NCHW/NHWC; ND/other rejected (channel axis undecidable)");
        return ge::GRAPH_FAILED;
    }

    // ===== (2b) grads/x format 一致性（grads 与 x 同 dtype/shape/format）=====
    // grads 与 x 必须同 origin format；否则通道轴口径不一致，属非法输入（例：grads=NHWC 而 x=NCHW）。
    // 仅 rank>=2 生效（与 format 白名单一致；rank≤1 单轴无歧义豁免）。
    const int32_t xFmt = ge::GetPrimaryFormat(static_cast<int32_t>(in.xT->GetOriginFormat()));
    if (rank >= 2 && fmt != xFmt) {
        OP_LOGE(node, "unsupported format: grads and x must share the same format");
        return ge::GRAPH_FAILED;
    }

    // ===== (2c) format/rank 一致性（NCHW/NHWC→rank4, NCDHW→rank5）=====
    // 具名空间 format 其秩由 format 唯一确定，秩不符即拒绝（NCHW/NHWC 必 rank4、NCDHW 必 rank5）。
    // 叠加上面白名单，grads rank 被强制收敛到 {4,5}：rank2/3/6/7/8 无法命中任一具名 format
    // （rank!=4 的 NCHW/NHWC 与 rank!=5 的 NCDHW 均在此拒绝，log 含 "format"）。
    if (rank >= 2 && (fmt == static_cast<int32_t>(ge::FORMAT_NCHW) || fmt == static_cast<int32_t>(ge::FORMAT_NHWC)) &&
        rank != 4) {
        OP_LOGE(node, "unsupported format: NCHW/NHWC require rank 4");
        return ge::GRAPH_FAILED;
    }
    if (rank >= 2 && fmt == static_cast<int32_t>(ge::FORMAT_NCDHW) && rank != 5) {
        OP_LOGE(node, "unsupported format: NCDHW require rank 5");
        return ge::GRAPH_FAILED;
    }

    // ===== (3) rank/shape 校验 =====
    if (rank < 1) {
        OP_LOGE(node, "invalid shape: rank-0 scalar grads is not supported");
        return ge::GRAPH_FAILED;
    }
    if (!ShapeEqual(in.gradsShape, in.xShape)) {
        OP_LOGE(node, "invalid shape: grads and x must have identical shape");
        return ge::GRAPH_FAILED;
    }

    // ===== (4) 统计量通道一致性（step 2.5） =====
    const int chAxis = LocateChannelAxis(fmt, rank);
    const int64_t C = (chAxis >= 0 && chAxis < rank) ? in.gradsShape.d[chAxis] : 1;
    if (!MatchesC(in.meanShape, C) || !MatchesC(in.varShape, C)) {
        OP_LOGE(node, "batch_mean/batch_variance channel dim mismatch: must equal grads channel dim C, non-channel "
                      "dims must be 1");
        return ge::GRAPH_FAILED;
    }

    out.rank = rank;
    out.chAxis = chAxis;
    out.C = C;
    return ge::GRAPH_SUCCESS;
}

// -------------------------------------------------------------------------
// Step1 ComputeAUbFactor（联合爬坡，最内向外）：定 aSplitIdx / aUbFactor，
// 并按 tail-A padding 口径求 innerAProdAlign（严格内层 A 轴乘积）与 aUnit。
// -------------------------------------------------------------------------
struct AUbSplit {
    int aSplitIdx = 0;
    int64_t aUbFactor = 1;
    int64_t innerAProdAlign = 1;
    int64_t aUnit = 1; // aUbFactor * innerAProdAlign
};

AUbSplit ComputeAUbFactor(const int64_t axisShape[], int32_t axisNum, int64_t cachelineTmp, int64_t bsElem)
{
    AUbSplit s;
    {
        int64_t product = 1;
        int stopIdx = -1;
        for (int idx = axisNum - 1; idx >= 0; --idx) {
            int64_t axisSize = axisShape[idx];
            int64_t eff = (idx == axisNum - 1) ? CeilAlign(axisSize, bsElem) : axisSize;
            if (product * eff > cachelineTmp) {
                stopIdx = idx;
                break;
            }
            product *= eff;
        }
        if (stopIdx < 0) {
            s.aSplitIdx = 0;
            s.aUbFactor = axisShape[0];
        } else if (stopIdx % 2 == 0) { // 停止轴为 A
            s.aSplitIdx = stopIdx;
            s.aUbFactor = std::min<int64_t>(cachelineTmp / std::max<int64_t>(product, 1), axisShape[stopIdx]);
            if (s.aUbFactor < 1) {
                s.aUbFactor = 1;
            }
        } else { // 停止轴为 R → 左移到左邻 A，每 chunk 取 1
            s.aSplitIdx = stopIdx - 1;
            s.aUbFactor = 1;
        }
        if (s.aSplitIdx < 0) {
            s.aSplitIdx = 0;
        }
        if (s.aUbFactor < 1) {
            s.aUbFactor = 1;
        }
    }
    // innerAProdAlign = ∏(严格内层 A 轴)，其中 tail-A 时最内 A（=通道，UB burst 尾轴）
    // 必须 CeilAlign 到 block（bsElem）——与 kernel BuildUBAxes tail-A 分支对最内 A 的
    // padding 一致（DataCopyPad 要求每 burst 32B 对齐；否则 laneA/reduce srcShape 与
    // UB 实排布行 stride 不符 → RA ReduceSum 越界 507035）。tail-R 时最内是 R 轴、内层 A
    // 稠密，本 CeilAlign 分支不触发（k 恒 != axisNum-1），保持原行为。
    const bool isTailAForPad = (axisNum % 2 == 1);
    for (int k = s.aSplitIdx + 2; k < axisNum; k += 2) {
        int64_t v = axisShape[k];
        if (isTailAForPad && k == axisNum - 1) {
            v = CeilAlign(v, bsElem);
        }
        s.innerAProdAlign *= v;
    }
    s.aUnit = s.aUbFactor * s.innerAProdAlign;
    return s;
}

// -------------------------------------------------------------------------
// Step2 ComputeRUbFactor（UB 预算反解 r_i_max，R 端 climb）：
// rSplitIdx / rUbFactor / rUbFactorAlign / innerRProdAlign + pre/postBufSize。
// -------------------------------------------------------------------------
struct RUbSplit {
    int rSplitIdx = 0;
    int64_t rUbFactor = 0;
    int64_t rUbFactorAlign = 0;
    int64_t innerRProdAlign = 1;
    int64_t preBufSize = 0;
    int64_t postBufSize = 0;
};

ge::graphStatus ComputeRUbFactor(const int64_t axisShape[], int32_t axisNum, bool isTailR, int64_t ubAvailable,
                                 int64_t aUnit, int64_t bsElem, const char* node, RUbSplit& out)
{
    const int64_t bytesPerRElem = kNPreTile * aUnit * kMaxDtypeSize;
    const int64_t r_i_max = ubAvailable / std::max<int64_t>(bytesPerRElem, 1);
    if (r_i_max < bsElem) {
        OP_LOGE(node, "UB budget exhausted: r_i_max below one block; cannot fit any R tile");
        return ge::GRAPH_FAILED;
    }
    // R 轴（奇下标），innermost-first 定长数组
    int rIdx[MAX_PATTERN_RANK];
    int rN = 0;
    for (int idx = axisNum - 1; idx >= 1; --idx) {
        if (idx % 2 == 1) {
            rIdx[rN++] = idx;
        }
    }
    int rSplitIdx = (rN > 0) ? rIdx[0] : (axisNum >= 2 ? 1 : 0);
    int64_t innerRProdAlign = 1;
    for (int t = 0; t < rN; ++t) {
        int idx = rIdx[t];
        int64_t axisSize = axisShape[idx];
        int64_t eff = (t == 0) ? CeilAlign(axisSize, bsElem) : axisSize;
        if (eff * innerRProdAlign > r_i_max) {
            rSplitIdx = idx;
            break;
        }
        // innerRProdAlign is PADDED (innermost burst-tail R CeilAligned to block);
        // use eff (== CeilAlign at t==0, actual otherwise) so it matches the kernel's physical
        // UB row stride (BuildUBAxes pads the burst-tail R to block).
        if (t + 1 < rN) {
            innerRProdAlign *= eff;
            rSplitIdx = rIdx[t + 1];
        } else {
            rSplitIdx = idx;
            break;
        }
    }
    const int innermostR = (rN > 0) ? rIdx[0] : -1;
    int64_t rUbFactor = std::min<int64_t>(r_i_max / std::max<int64_t>(innerRProdAlign, 1), axisShape[rSplitIdx]);
    if (rUbFactor < 1) {
        OP_LOGE(node, "UB budget exhausted: rUbFactor collapsed to zero");
        return ge::GRAPH_FAILED;
    }
    int64_t rUbFactorAlign = rUbFactor;
    const bool isBurstTailR = isTailR && (rSplitIdx == innermostR);
    if (isBurstTailR) {
        if (rUbFactor < axisShape[rSplitIdx]) {
            rUbFactor = FloorAlign(rUbFactor, bsElem);
            if (rUbFactor < 1) {
                OP_LOGE(node, "UB budget exhausted: burst-tail R FloorAlign collapsed to zero");
                return ge::GRAPH_FAILED;
            }
            rUbFactorAlign = rUbFactor;
        } else {
            int64_t align = CeilAlign(rUbFactor, bsElem);
            if (align * innerRProdAlign > r_i_max) {
                rUbFactor = FloorAlign(rUbFactor, bsElem);
                if (rUbFactor < 1) {
                    OP_LOGE(node, "UB budget exhausted: burst-tail R align retreat collapsed to zero");
                    return ge::GRAPH_FAILED;
                }
                rUbFactorAlign = rUbFactor;
            } else {
                rUbFactorAlign = align;
            }
        }
    }

    out.rSplitIdx = rSplitIdx;
    out.rUbFactor = rUbFactor;
    out.rUbFactorAlign = rUbFactorAlign;
    out.innerRProdAlign = innerRProdAlign;
    out.preBufSize = CeilAlign(aUnit * rUbFactorAlign * innerRProdAlign * kMaxDtypeSize, kBlockSize);
    out.postBufSize = CeilAlign(aUnit * kMaxDtypeSize, kBlockSize);
    return ge::GRAPH_SUCCESS;
}

// -------------------------------------------------------------------------
// (8) 多核切分（base 口径）：A 端大核/小核负载均衡 + R 端总循环数。
// -------------------------------------------------------------------------
struct MulticoreSplit {
    int64_t aSplitChunkCnt = 1;
    int64_t aLoopCntTotal = 1;
    int64_t aSmallCoreLoopCnt = 0;
    int64_t aBigCoreCnt = 0;
    int64_t aBigCoreLoopCnt = 0;
    int64_t usedCoreNumBase = 1;
    int64_t rLoopCntTotal = 1;
};

MulticoreSplit ComputeMulticoreSplit(const int64_t axisShape[], int aSplitIdx, int rSplitIdx, int64_t aUbFactor,
                                     int64_t rUbFactor, int64_t coreNum)
{
    MulticoreSplit m;
    int64_t outerAProd = 1;
    for (int i = 0; i < aSplitIdx; i += 2) {
        outerAProd *= axisShape[i];
    }
    m.aSplitChunkCnt = CeilDiv(axisShape[aSplitIdx], aUbFactor);
    m.aLoopCntTotal = outerAProd * m.aSplitChunkCnt;
    m.aSmallCoreLoopCnt = m.aLoopCntTotal / coreNum;
    m.aBigCoreCnt = m.aLoopCntTotal % coreNum;
    m.aBigCoreLoopCnt = m.aSmallCoreLoopCnt + (m.aBigCoreCnt > 0 ? 1 : 0);
    m.usedCoreNumBase = (m.aSmallCoreLoopCnt > 0) ? coreNum : m.aBigCoreCnt;
    if (m.usedCoreNumBase < 1) {
        m.usedCoreNumBase = 1;
    }
    int64_t outerRProd = 1;
    for (int i = 1; i < rSplitIdx; i += 2) {
        outerRProd *= axisShape[i];
    }
    m.rLoopCntTotal = outerRProd * CeilDiv(axisShape[rSplitIdx], rUbFactor);
    return m;
}

// -------------------------------------------------------------------------
// (9) 路由：ShouldUseGroup（A 端 outer 过少且 R 端可切）→ group 多块借 R；
// ComputeGroupSplit 红线不通过则 base 兜底。
// -------------------------------------------------------------------------
struct GroupSplit {
    bool useGroup = false;
    int64_t numBlocks = 0;
    int64_t rGroupCnt = 0;
};

GroupSplit ComputeGroupSplit(int64_t aOuter, int64_t rOuter, int64_t coreNum)
{
    GroupSplit g;
    if (aOuter <= coreNum / 2 && rOuter > 1) { // ShouldUseGroup = true
        const int64_t totalOuter = aOuter * rOuter;
        const int64_t perCoreNum = CeilDiv(totalOuter, coreNum);
        int64_t nb = CeilDiv(totalOuter, std::max<int64_t>(perCoreNum, 1));
        const int64_t ca = CeilAlign(nb, aOuter);
        nb = (ca <= coreNum) ? ca : FloorAlign(nb, aOuter);
        if (nb > 0 && aOuter > 0 && nb <= coreNum) { // ComputeGroupSplit 红线通过
            g.useGroup = true;
            g.numBlocks = nb;
            g.rGroupCnt = nb / aOuter;
        }
    }
    return g;
}

// -------------------------------------------------------------------------
// (10) 填 TilingData + SetBlockDim + SetTilingKey + workspace。
// -------------------------------------------------------------------------
ge::graphStatus FillTilingData(gert::TilingContext* context, int32_t axisNum, const int64_t axisShape[],
                               const int64_t axisStride[], int64_t aTotal, const AUbSplit& a, const RUbSplit& r,
                               const MulticoreSplit& m, const GroupSplit& g)
{
    auto* td = context->GetTilingData<BN3DTrainingUpdateGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);

    td->axisNum = axisNum;
    for (int i = 0; i < MAX_PATTERN_RANK; ++i) {
        td->axisShape[i] = axisShape[i];
        td->axisStride[i] = axisStride[i];
    }
    td->aLoopCntTotal = m.aLoopCntTotal;
    td->aSplitChunkCnt = m.aSplitChunkCnt;
    td->aBigCoreLoopCnt = m.aBigCoreLoopCnt;
    td->aSmallCoreLoopCnt = m.aSmallCoreLoopCnt;
    td->aBigCoreCnt = static_cast<int32_t>(m.aBigCoreCnt);

    const int64_t usedCoreNum = g.useGroup ? g.numBlocks : m.usedCoreNumBase;
    td->usedCoreNum = static_cast<int32_t>(usedCoreNum);
    td->aSplitIdx = static_cast<int32_t>(a.aSplitIdx);
    td->rSplitIdx = static_cast<int32_t>(r.rSplitIdx);
    td->aUbFactor = a.aUbFactor;
    td->rUbFactor = r.rUbFactor;
    td->rUbFactorAlign = r.rUbFactorAlign;
    td->innerAProdAlign = a.innerAProdAlign;
    td->innerRProdAlign = r.innerRProdAlign;
    td->rLoopCntTotal = m.rLoopCntTotal;
    td->preBufSize = r.preBufSize;
    td->postBufSize = r.postBufSize;
    td->cacheBufUbSize = kCacheBufUbSize;
    td->rGroupCnt = g.useGroup ? g.rGroupCnt : 0;
    td->aTotal = aTotal;

    // attr epsilon 透传（GetFloat(0)，nullptr→默认 0.0001）。
    // kernel 用其算 rstd=1/sqrt(var+epsilon)，逐用例生效。
    float epsilon = 0.0001f;
    const auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float* epsPtr = attrs->GetAttrPointer<float>(0);
        if (epsPtr != nullptr) {
            epsilon = *epsPtr;
        }
    }
    td->epsilon = epsilon;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    context->SetTilingKey(g.useGroup ? 1 : 0); // group=(1,0)=1 / base=(0,0)=0

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (g.useGroup) {
        context->SetScheduleMode(1); // SyncAll 要求
    }
    if (currentWorkspace != nullptr) {
        currentWorkspace[0] = g.useGroup ?
                                  static_cast<size_t>(g.rGroupCnt * aTotal * static_cast<int64_t>(sizeof(float)) +
                                                      kSysWorkspace) :
                                  static_cast<size_t>(kSysWorkspace);
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace

// ---------------------------------------------------------------------------
// TilingFuncBN3DTrainingUpdateGrad — real host tiling entry point.
// ---------------------------------------------------------------------------
static ge::graphStatus TilingFuncBN3DTrainingUpdateGrad(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED; // nullptr guard (A10)
    }
    const char* node = NodeName(context);
    OP_LOGI(node, "Enter TilingFuncBN3DTrainingUpdateGrad");

    // ---- platform facts (GetPlatformInfo + PlatformAscendC; TilingParse-free path) ----
    PlatformFacts pf;
    if (GetPlatformFacts(context, node, pf) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const int64_t coreNum = pf.coreNum;
    const int64_t ubSize = pf.ubSize;

    // ---- inputs: Tensor/Desc/StorageShape 判空 + dtype/shape 快照（TOPK-2 见 GetInputFacts）----
    InputFacts in;
    if (GetInputFacts(context, in) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // ===== (1)~(4) 输入校验（dtype → format → rank/shape → stats-channel，拦截序不变）=====
    CheckedShape cs;
    if (CheckInputs(node, in, cs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const int rank = cs.rank;
    const int chAxis = cs.chAxis;
    const int64_t C = cs.C;

    // ===== (5) 空 tensor 短路（合轴之前，原始轴；EMPTY_A 优先于 EMPTY_R）=====
    bool emptyR = false;
    for (int i = 0; i < rank; ++i) {
        if (i != chAxis && in.gradsShape.d[i] == 0) {
            emptyR = true;
        }
    }
    if (C == 0) {
        return FillEmptyA(context);
    }
    if (emptyR) {
        return FillEmptyR(context, C, coreNum, ubSize);
    }

    // ===== (6) 四步合轴 =====
    FusedAxes f = FuseAxes(in.gradsShape, chAxis);
    if (!f.ok) {
        OP_LOGE(node, "invalid shape: fused axisNum out of [1, MAX_PATTERN_RANK]");
        return ge::GRAPH_FAILED;
    }
    const int32_t axisNum = f.axisNum;
    int64_t axisShape[MAX_PATTERN_RANK];
    for (int i = 0; i < MAX_PATTERN_RANK; ++i) {
        axisShape[i] = f.axisShape[i];
    }
    const bool isTailR = f.isTailR;
    const int64_t aTotal = f.aTotal;

    // GM stride（row-major；未用位填 0）
    int64_t axisStride[MAX_PATTERN_RANK] = {0, 0, 0, 0};
    {
        int64_t acc = 1;
        for (int i = axisNum - 1; i >= 0; --i) {
            axisStride[i] = acc;
            acc *= axisShape[i];
        }
    }

    // ===== (7) UB 切分 =====
    const int64_t dtypeSize = (in.gDtype == ge::DT_FLOAT) ? 4 : 2;
    const int64_t bsElem = kBlockSize / dtypeSize; // fp32=8, fp16/bf16=16
    const int64_t cachelineTmp = FloorAlign(kCacheLineSize / dtypeSize, bsElem);
    const int64_t ubAvailable = ubSize - kCacheBufUbSize;
    if (ubAvailable <= 0) {
        OP_LOGE(node, "UB budget exhausted: ubSize below fixed cacheBuf (16KB); no room for pre tiles");
        return ge::GRAPH_FAILED;
    }

    // Step1 ComputeAUbFactor（联合爬坡，最内向外）→ Step2 ComputeRUbFactor（UB 预算反解，R 端 climb）
    const AUbSplit aUb = ComputeAUbFactor(axisShape, axisNum, cachelineTmp, bsElem);
    RUbSplit rUb;
    if (ComputeRUbFactor(axisShape, axisNum, isTailR, ubAvailable, aUb.aUnit, bsElem, node, rUb) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // ===== (8) 多核切分（base 口径）=====
    const MulticoreSplit mc = ComputeMulticoreSplit(axisShape, aUb.aSplitIdx, rUb.rSplitIdx, aUb.aUbFactor,
                                                    rUb.rUbFactor, coreNum);

    const int64_t aOuter = mc.aLoopCntTotal;
    const int64_t rOuter = mc.rLoopCntTotal;

    // ===== (9) 路由：ShouldUseGroup → group；否则 base 兜底 =====
    const GroupSplit grp = ComputeGroupSplit(aOuter, rOuter, coreNum);

    // ===== (10) 填 TilingData + SetBlockDim + SetTilingKey =====
    return FillTilingData(context, axisNum, axisShape, axisStride, aTotal, aUb, rUb, mc, grp);
}

// TilingParse is required by the op-tiling compile interface
// (TbeOptilingPyInterfaceNew, used by binary/kernel compilation). It is a no-op
// here: platform facts (coreNum / ubSize) are NOT obtained via compile info —
// they are read directly inside TilingFuncBN3DTrainingUpdateGrad via
// GetPlatformInfo() (HOST-4: platform params obtained in the Tiling method,
// never via TilingParse compileInfo).
ge::graphStatus TilingPrepareForBN3DTrainingUpdateGrad(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BN3DTrainingUpdateGrad)
    .Tiling(TilingFuncBN3DTrainingUpdateGrad)
    .TilingParse<BN3DTrainingUpdateGradCompileInfo>(TilingPrepareForBN3DTrainingUpdateGrad);

} // namespace optiling
