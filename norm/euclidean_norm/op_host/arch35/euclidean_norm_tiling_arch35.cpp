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
 * \file euclidean_norm_tiling.cpp
 * \brief EuclideanNorm 算子 host tiling 实现。
 *
 * 完全套用 normal 模板：
 *   1) 读 axes 张量 + keep_dims attr  → 建 A/R 类型表
 *   2) 标准 4 步 pattern 预处理：去 1 / 合轴 / 补 leading A / 补 R 增广
 *   3) 双切分：先定 aSplit + aUbFactor，再用剩余 UB 反解 rSplit + rUbFactor；R 全载时回头扩 A
 *   4) 多核切分（fused aLoop）：aLoopCntTotal = ∏(outer A) × aSplitChunkCnt，
 *      按 coreNum 均衡分大小核
 *   5) 算 rLoopCntTotal、UB sizes，填 tilingdata + tilingkey + blockDim + workspace=0
 *
 * Reducer = sum；EuclideanNorm 在 tiling 阶段与 reduce_sum 完全相同。
 */

#include <algorithm>
#include <set>
#include <vector>
#include <cstring>

#include "euclidean_norm_tiling_arch35.h"

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../../op_kernel/arch35/euclidean_norm_tiling_data.h"
#include "../../op_kernel/arch35/euclidean_norm_tiling_key.h"

namespace optiling {

// 输入 / 输出 / 属性的固定索引
static constexpr size_t kInputXIdx = 0;
static constexpr size_t kInputAxesIdx = 1;
static constexpr size_t kOutputYIdx = 0;
static constexpr size_t kAttrKeepDimsIdx = 0;

// pattern 上限
static constexpr int32_t kMinAxisNum = 2;
static constexpr int32_t kMaxAxisNum = MAX_PATTERN_RANK;

// UB 分配预算常量（与设计文档对应）
static constexpr int64_t kCacheBufBytes = 16 * 1024; // cacheBuf 固定 16 KB
static constexpr int64_t kFp32Bytes = 4;             // ComputeT = fp32
static constexpr int64_t kDbFactor = 2;              // 主链路 DB
static constexpr int64_t kNPreReuse = 1;             // x 单输入，自身复用槽
static constexpr int64_t kNPreParallel = 0;          // 无并行槽
static constexpr int64_t kNOut = 1;                  // y 单输出
static constexpr int64_t kTmpBufCopies = 2;          // 主+尾共 2 份 fp32

// 空 tensor 分类；优先级 EMPTY_A > EMPTY_R > NORMAL
enum class EuclideanNormEmptyKind {
    NORMAL = 0,
    EMPTY_A, // ∃ A 轴 axisShape==0 → output 空 tensor、kernel 零操作
    EMPTY_R, // ∃ R 轴 axisShape==0 且 ∀ A 轴 axisShape>0 → 每 entry = sqrt(0) = 0
};

// 内部工作上下文：把 tiling 过程中演化的字段统一聚合，
// 避免函数签名传 20 个参数。
struct EuclideanNormCtx {
    // 平台
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    int64_t blockSize = 32;
    int64_t cacheLineSize = 256;

    // 输入
    ge::DataType xDtype = ge::DT_UNDEFINED;
    int64_t dtypeSize = 0;           // sizeof(D_T)
    std::vector<int64_t> xShape;     // x 各维 size
    std::vector<int64_t> reduceAxes; // 归一化、排序、去重后的 reduce 轴下标集合

    // 空 tensor 分类
    EuclideanNormEmptyKind emptyKind = EuclideanNormEmptyKind::NORMAL;
    int64_t aTotalEmpty = 0; // EMPTY_R 下 ∏(非 reduce 轴 size)；空集时 = 1

    // pattern 描述（合轴 / 去 1 / 补 R 增广后）
    std::vector<int64_t> axisShape;
    std::vector<bool> isReduceAxis; // axisShape[i] 对应是否 R 轴
    int32_t axisNum = 0;            // 等于 axisShape.size()
    bool isTailR = false;

    // 切分
    int32_t aSplitAxisIdx = 0;
    int32_t rSplitAxisIdx = 0;
    int64_t aUbFactor = 0;
    int64_t aUbFactorAlign = 0;
    int64_t rUbFactor = 0;      // valid R count
    int64_t rUbFactorAlign = 0; // padded R count（见 tilingdata 注释）
    int64_t innerAProd = 0;
    int64_t innerAProdAlign = 0;
    int64_t innerRProd = 0;
    int64_t innerRProdAlign = 0;

    // 多核（fused aLoop，按 coreNum 均衡分大小核）
    int64_t aLoopCntTotal = 0;
    int64_t aSplitChunkCnt = 0;
    int64_t aBigCoreLoopCnt = 0;
    int64_t aSmallCoreLoopCnt = 0;
    int32_t aBigCoreCnt = 0;
    int32_t usedCoreNum = 0;

    // 外层 R loop
    int64_t rLoopCntTotal = 0;

    // UB sizes（字节）
    int64_t preReduceUbSize = 0;
    int64_t postReduceUbSize = 0;
    int64_t tmpBufUbSize = 0;

    // ─── group 模板 ───
    bool isGroup = false;
    int64_t rGroupCnt = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// 1) 平台 / shape / dtype 信息收集
// ─────────────────────────────────────────────────────────────────────────────
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, EuclideanNormCtx& ctx)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ctx.coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(ctx.coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);

    uint64_t ub = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    ctx.ubSize = static_cast<int64_t>(ub);
    OP_CHECK_IF(ctx.ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    ctx.blockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context));
    ctx.cacheLineSize = static_cast<int64_t>(Ops::Base::GetCacheLineSize(context));
    OP_CHECK_IF(
        ctx.blockSize == 0 || ctx.cacheLineSize == 0,
        OP_LOGE(context, "blockSize=%ld or cacheLineSize=%ld is 0", ctx.blockSize, ctx.cacheLineSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAndDtype(gert::TilingContext* context, EuclideanNormCtx& ctx)
{
    auto xShapePtr = context->GetInputShape(kInputXIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    const gert::Shape& xS = xShapePtr->GetStorageShape();
    const size_t rank = xS.GetDimNum();
    ctx.xShape.clear();
    if (rank == 0) {
        // Scalar (0-D) normalized to 1-element 1-D tensor for tiling.
        ctx.xShape.push_back(1);
    } else {
        for (size_t i = 0; i < rank; ++i) {
            OP_CHECK_IF(
                xS.GetDim(i) < 0,
                OP_LOGE(context, "x dim[%zu]=%ld is negative (dynamic shape not allowed at tiling)", i, xS.GetDim(i)),
                return ge::GRAPH_FAILED);
            ctx.xShape.push_back(xS.GetDim(i));
        }
    }

    auto xDesc = context->GetInputDesc(kInputXIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    ctx.xDtype = xDesc->GetDataType();
    const std::set<ge::DataType> kSupported = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32};
    OP_CHECK_IF(
        kSupported.count(ctx.xDtype) == 0,
        OP_LOGE(context, "unsupported x dtype %d (expect fp16/bf16/fp32/int32)", static_cast<int>(ctx.xDtype)),
        return ge::GRAPH_FAILED);
    ctx.dtypeSize = static_cast<int64_t>(ge::GetSizeByDataType(ctx.xDtype));
    return ge::GRAPH_SUCCESS;
}

// ─────────────────────────────────────────────────────────────────────────────
// 2) axes 张量解析 + A/R 类型表
// ─────────────────────────────────────────────────────────────────────────────
static ge::graphStatus ParseAxesTensor(gert::TilingContext* context, EuclideanNormCtx& ctx)
{
    const gert::Tensor* axesTensor = context->GetInputTensor(kInputAxesIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, axesTensor);
    const int64_t axesNum = axesTensor->GetShapeSize();
    const int64_t xRank = static_cast<int64_t>(ctx.xShape.size());
    ctx.reduceAxes.clear();

    if (axesNum == 0) {
        // axes=[] → axes=[0..xRank-1]，full reduce（对标 TF/PyTorch；与 NumPy axis=() identity 不同）
        ctx.reduceAxes.reserve(xRank);
        for (int64_t i = 0; i < xRank; ++i) {
            ctx.reduceAxes.push_back(i);
        }
        return ge::GRAPH_SUCCESS;
    }

    // op_def 中 axes 支持 DT_INT32 / DT_INT64（IndexNumberType），两条解析分支必须并存。
    std::set<int64_t> seen;
    auto pushOne = [&](int64_t v, int64_t idx) -> ge::graphStatus {
        OP_CHECK_IF(
            v < -xRank || v >= xRank, OP_LOGE(context, "axes[%ld]=%ld out of range [-%ld, %ld)", idx, v, xRank, xRank),
            return ge::GRAPH_FAILED);
        const int64_t norm = (v < 0) ? (v + xRank) : v;
        OP_CHECK_IF(
            !seen.insert(norm).second, OP_LOGE(context, "duplicate axis %ld in axes (input idx=%ld)", norm, idx),
            return ge::GRAPH_FAILED);
        ctx.reduceAxes.push_back(norm);
        return ge::GRAPH_SUCCESS;
    };

    const ge::DataType dt = axesTensor->GetDataType();
    if (dt == ge::DT_INT32) {
        const int32_t* data = axesTensor->GetData<int32_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, data);
        for (int64_t i = 0; i < axesNum; ++i) {
            if (pushOne(static_cast<int64_t>(data[i]), i) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        }
    } else if (dt == ge::DT_INT64) {
        const int64_t* data = axesTensor->GetData<int64_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, data);
        for (int64_t i = 0; i < axesNum; ++i) {
            if (pushOne(data[i], i) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        }
    } else {
        OP_LOGE(context, "axes dtype %d is not int32/int64", static_cast<int>(dt));
        return ge::GRAPH_FAILED;
    }
    std::sort(ctx.reduceAxes.begin(), ctx.reduceAxes.end());
    return ge::GRAPH_SUCCESS;
}

// 把 x 的 shape 与 reduceAxes 合成「轴列表 + R 标记」（pattern 预处理的初态）
static void BuildInitialAxisList(EuclideanNormCtx& ctx)
{
    const size_t rank = ctx.xShape.size();
    ctx.axisShape = ctx.xShape;
    ctx.isReduceAxis.assign(rank, false);
    for (int64_t a : ctx.reduceAxes) {
        ctx.isReduceAxis[static_cast<size_t>(a)] = true;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3) Pattern 预处理 4 步（顺序不可换）
// ─────────────────────────────────────────────────────────────────────────────
// 3.1 去 1：所有 size==1 的轴整根删除；非空保留：若 ∏shape==1 则保留 1 根 A=1 占位
static void DropSizeOneAxes(EuclideanNormCtx& ctx)
{
    std::vector<int64_t> newShape;
    std::vector<bool> newIsR;
    for (size_t i = 0; i < ctx.axisShape.size(); ++i) {
        if (ctx.axisShape[i] != 1) {
            newShape.push_back(ctx.axisShape[i]);
            newIsR.push_back(ctx.isReduceAxis[i]);
        }
    }
    if (newShape.empty()) {
        // ∏shape==1：保留 1 根 A=1 占位
        newShape.push_back(1);
        newIsR.push_back(false);
    }
    ctx.axisShape = std::move(newShape);
    ctx.isReduceAxis = std::move(newIsR);
}

// 3.2 合轴：连续 A 合并为 1 根，连续 R 合并为 1 根
static void FuseAxis(EuclideanNormCtx& ctx)
{
    std::vector<int64_t> fusedShape;
    std::vector<bool> fusedIsR;
    for (size_t i = 0; i < ctx.axisShape.size(); ++i) {
        if (!fusedShape.empty() && fusedIsR.back() == ctx.isReduceAxis[i]) {
            fusedShape.back() *= ctx.axisShape[i];
        } else {
            fusedShape.push_back(ctx.axisShape[i]);
            fusedIsR.push_back(ctx.isReduceAxis[i]);
        }
    }
    ctx.axisShape = std::move(fusedShape);
    ctx.isReduceAxis = std::move(fusedIsR);
}

// 3.3 补 leading A：若第 0 轴是 R 轴，前置一根大小为 1 的 A 轴
static void PadLeadingOneA(EuclideanNormCtx& ctx)
{
    if (!ctx.axisShape.empty() && ctx.isReduceAxis.front()) {
        ctx.axisShape.insert(ctx.axisShape.begin(), 1);
        ctx.isReduceAxis.insert(ctx.isReduceAxis.begin(), false);
    }
}

// 3.4 补 R 增广：若仍无 R 轴（纯 A 退化）
//   A=1（全 1 退化）：末尾补 R=1 → AR（tail-R），避免不必要的 TailA 路径
//   A>1（纯 A 非全 1）：前置 [A=1, R=1] → ARA（tail-A），禁止末尾补 R=1
static void PadRIfPureA(EuclideanNormCtx& ctx)
{
    bool hasR = false;
    for (bool b : ctx.isReduceAxis) {
        if (b) {
            hasR = true;
            break;
        }
    }
    if (!hasR) {
        if (ctx.axisShape.size() == 1 && ctx.axisShape[0] == 1) {
            ctx.axisShape.push_back(1);
            ctx.isReduceAxis.push_back(true);
        } else {
            ctx.axisShape.insert(ctx.axisShape.begin(), {1, 1});
            ctx.isReduceAxis.insert(ctx.isReduceAxis.begin(), {false, true});
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3.0 空 tensor 分类——在 4 步预处理之前做：
//   ∃ A 轴 axisShape==0 → EMPTY_A
//   ∃ R 轴 axisShape==0 且 ∀ A 轴 axisShape>0 → EMPTY_R（aTotal = ∏ A axisShape；空 A 集合 → 1）
//   其它 → NORMAL
//
// 触发后由 ComputeEmpty*Tiling 单独填 tilingdata，跳过整套 normal 路径。
// ─────────────────────────────────────────────────────────────────────────────
static void ClassifyEmptyTensor(EuclideanNormCtx& ctx)
{
    bool hasZeroA = false;
    bool hasZeroR = false;
    int64_t aTotal = 1;
    for (size_t i = 0; i < ctx.xShape.size(); ++i) {
        const bool isR = ctx.isReduceAxis[i];
        const int64_t sz = ctx.xShape[i];
        if (!isR) {
            if (sz == 0) {
                hasZeroA = true;
            } else {
                aTotal *= sz;
            }
        } else {
            if (sz == 0) {
                hasZeroR = true;
            }
        }
    }
    if (hasZeroA) {
        ctx.emptyKind = EuclideanNormEmptyKind::EMPTY_A;
        ctx.aTotalEmpty = 0; // EMPTY_A 不用 aTotal，但保持字段一致
    } else if (hasZeroR) {
        ctx.emptyKind = EuclideanNormEmptyKind::EMPTY_R;
        ctx.aTotalEmpty = aTotal; // 全 A 轴乘积（无 A 轴时 = 1）
    } else {
        ctx.emptyKind = EuclideanNormEmptyKind::NORMAL;
        ctx.aTotalEmpty = 0;
    }
}

// 4 步执行 + axisNum / isTailR 校验（BuildInitialAxisList 已在主入口的空 tensor 判定前完成）
static ge::graphStatus PreprocessPattern(gert::TilingContext* context, EuclideanNormCtx& ctx)
{
    DropSizeOneAxes(ctx);
    FuseAxis(ctx);
    PadLeadingOneA(ctx);
    PadRIfPureA(ctx);

    ctx.axisNum = static_cast<int32_t>(ctx.axisShape.size());
    OP_CHECK_IF(
        ctx.axisNum < kMinAxisNum || ctx.axisNum > kMaxAxisNum,
        OP_LOGE(
            context, "axisNum=%d out of [%d, %d] after pattern preprocessing", ctx.axisNum, kMinAxisNum, kMaxAxisNum),
        return ge::GRAPH_FAILED);

    // pattern 必形如 A R A R … 严格交替、由 A 起头
    for (int32_t i = 0; i < ctx.axisNum; ++i) {
        const bool wantR = (i % 2 == 1);
        OP_CHECK_IF(
            ctx.isReduceAxis[static_cast<size_t>(i)] != wantR,
            OP_LOGE(
                context, "axis[%d] type mismatch after preprocessing (isR=%d, want=%d)", i,
                static_cast<int>(ctx.isReduceAxis[i]), static_cast<int>(wantR)),
            return ge::GRAPH_FAILED);
    }
    ctx.isTailR = (ctx.axisNum % 2 == 0);
    return ge::GRAPH_SUCCESS;
}

// ─────────────────────────────────────────────────────────────────────────────
// 4) Axis stride（按 element）—— 给 kernel 算 GM offset 用
//    stride[i] = ∏_{j > i} axisShape[j]
// ─────────────────────────────────────────────────────────────────────────────
static void ComputeAxisStrides(EuclideanNormCtx& ctx, int64_t outStride[])
{
    int64_t acc = 1;
    for (int32_t i = ctx.axisNum - 1; i >= 0; --i) {
        outStride[i] = acc;
        acc *= ctx.axisShape[static_cast<size_t>(i)];
    }
}

// 找最内的 A 轴下标
static int32_t LastAAxisIdx(const EuclideanNormCtx& ctx)
{
    for (int32_t i = ctx.axisNum - 1; i >= 0; --i) {
        if (!ctx.isReduceAxis[static_cast<size_t>(i)]) {
            return i;
        }
    }
    return -1;
}

// 找最内的 R 轴下标
static int32_t LastRAxisIdx(const EuclideanNormCtx& ctx)
{
    for (int32_t i = ctx.axisNum - 1; i >= 0; --i) {
        if (ctx.isReduceAxis[static_cast<size_t>(i)]) {
            return i;
        }
    }
    return -1;
}

// 上一根同类型轴下标（type=true 找 R，false 找 A）；不存在返回 -1
static int32_t PrevSameTypeAxis(const EuclideanNormCtx& ctx, int32_t from, bool wantR)
{
    for (int32_t i = from - 1; i >= 0; --i) {
        if (ctx.isReduceAxis[static_cast<size_t>(i)] == wantR) {
            return i;
        }
    }
    return -1;
}

// ─────────────────────────────────────────────────────────────────────────────
// 5) UB 切分 Step 1：定 aSplitAxisIdx 与 aUbFactor
// ─────────────────────────────────────────────────────────────────────────────
static void ComputeAUbFactor(EuclideanNormCtx& ctx)
{
    const int32_t lastA = LastAAxisIdx(ctx);
    const int64_t maxInnerAInitElem = ctx.cacheLineSize / ctx.dtypeSize;

    // 5.1.1 atmp（UB 内 A 维度目标元素数）
    int64_t atmp = 0;
    if (ctx.isTailR) {
        // 偶数轴 pattern：tail R，下界保证多核分得动
        int64_t totalA = 1;
        for (int32_t i = 0; i < ctx.axisNum; ++i) {
            if (!ctx.isReduceAxis[static_cast<size_t>(i)]) {
                totalA *= ctx.axisShape[static_cast<size_t>(i)];
            }
        }
        atmp = std::min(maxInnerAInitElem, Ops::Base::CeilDiv(totalA, ctx.coreNum));
        atmp = std::max<int64_t>(atmp, 1);
    } else {
        atmp = maxInnerAInitElem;
    }

    // 5.1.2 从最内 A 向外 climb
    //   每 climb 一格，把该层 A 整根吸进 inner bundle（innerAProd × axisShape, innerAProdAlign × ...）
    //   tail A 时最内 A 是 burst 尾轴，按 block 对齐累乘到 innerAProdAlign。
    ctx.innerAProd = 1;
    ctx.innerAProdAlign = 1;
    ctx.aSplitAxisIdx = lastA;
    const int64_t bsAElem = ctx.blockSize / ctx.dtypeSize;

    while (true) {
        const int64_t aSize = ctx.axisShape[static_cast<size_t>(ctx.aSplitAxisIdx)];
        if (aSize * ctx.innerAProdAlign >= atmp) {
            break;
        }
        const int32_t prev = PrevSameTypeAxis(ctx, ctx.aSplitAxisIdx, /*wantR=*/false);
        if (prev < 0) {
            break;
        }
        // 整根纳入 inner bundle
        if (ctx.aSplitAxisIdx == lastA && !ctx.isTailR) {
            ctx.innerAProdAlign *= Ops::Base::CeilAlign(aSize, bsAElem);
        } else {
            ctx.innerAProdAlign *= aSize;
        }
        ctx.innerAProd *= aSize;
        ctx.aSplitAxisIdx = prev;
    }

    const int64_t aSize = ctx.axisShape[static_cast<size_t>(ctx.aSplitAxisIdx)];
    ctx.aUbFactor = std::min(Ops::Base::CeilDiv(atmp, ctx.innerAProdAlign), aSize);
    ctx.aUbFactor = std::max<int64_t>(ctx.aUbFactor, 1);

    // aUbFactorAlign：仅"切在最内 A 且 tail A"时该轴本身按 block 对齐；
    //   其余情况 aUbFactor 在中间维度上，padding 已被 innerAProdAlign 吸收
    if (ctx.aSplitAxisIdx == lastA && !ctx.isTailR) {
        ctx.aUbFactorAlign = Ops::Base::CeilAlign(ctx.aUbFactor, bsAElem);
    } else {
        ctx.aUbFactorAlign = ctx.aUbFactor;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6) UB 切分 Step 2：定 rSplitAxisIdx 与 rUbFactor（基于 UB 预算反解）
//
//    bytesPerRElem = (DB × N_pre_reuse + N_pre_parallel) × sizeof(D_T) + 2 × sizeof(fp32)
//    outBytes      = N_out × sizeof(D_T) × aUnit
//    r_i_max       = (ubAvailable − outBytes) / (aUnit × bytesPerRElem)
// ─────────────────────────────────────────────────────────────────────────────
static int64_t ComputeRiMax(const EuclideanNormCtx& ctx)
{
    const int64_t ubAvailable = ctx.ubSize - kCacheBufBytes;
    const int64_t aUnit = ctx.aUbFactorAlign * ctx.innerAProdAlign;
    const int64_t bytesPerRElem = (kDbFactor * kNPreReuse + kNPreParallel) * ctx.dtypeSize + kTmpBufCopies * kFp32Bytes;
    const int64_t outBytes = kNOut * ctx.dtypeSize * aUnit;
    if (aUnit <= 0 || bytesPerRElem <= 0) {
        return -1;
    }
    const int64_t numer = ubAvailable - outBytes;
    if (numer <= 0) {
        return -1;
    }
    return numer / (aUnit * bytesPerRElem);
}

static ge::graphStatus ComputeRUbFactor(gert::TilingContext* context, EuclideanNormCtx& ctx)
{
    const int32_t lastR = LastRAxisIdx(ctx);
    OP_CHECK_IF(lastR < 0, OP_LOGE(context, "no R axis after preprocessing"), return ge::GRAPH_FAILED);

    const int64_t rIMax = ComputeRiMax(ctx);
    OP_CHECK_IF(
        rIMax < 1,
        OP_LOGE(
            context, "r_i_max < 1 (UB cannot hold one R element): aUFA=%ld iAPA=%ld", ctx.aUbFactorAlign,
            ctx.innerAProdAlign),
        return ge::GRAPH_FAILED);

    // 从最内 R 向外 climb
    ctx.innerRProd = 1;
    ctx.innerRProdAlign = 1;
    ctx.rSplitAxisIdx = lastR;
    const int64_t bsAElem = ctx.blockSize / ctx.dtypeSize;

    while (true) {
        const int64_t rSize = ctx.axisShape[static_cast<size_t>(ctx.rSplitAxisIdx)];
        if (rSize * ctx.innerRProdAlign > rIMax) {
            break;
        }
        const int32_t prev = PrevSameTypeAxis(ctx, ctx.rSplitAxisIdx, /*wantR=*/true);
        if (prev < 0) {
            break;
        }
        if (ctx.rSplitAxisIdx == lastR && ctx.isTailR) {
            ctx.innerRProdAlign *= Ops::Base::CeilAlign(rSize, bsAElem);
        } else {
            ctx.innerRProdAlign *= rSize;
        }
        ctx.innerRProd *= rSize;
        ctx.rSplitAxisIdx = prev;
    }

    const int64_t rAxisSize = ctx.axisShape[static_cast<size_t>(ctx.rSplitAxisIdx)];
    ctx.rUbFactor = std::min(rIMax / ctx.innerRProdAlign, rAxisSize);
    ctx.rUbFactor = std::max<int64_t>(ctx.rUbFactor, 1);

    // 仅「切在最内 R 且 tail R 且实际切 chunk」时 rUbFactor 自身 FloorAlign（守卫条件）
    if (ctx.isTailR && ctx.rSplitAxisIdx == lastR && ctx.rUbFactor < rAxisSize) {
        ctx.rUbFactor = Ops::Base::FloorAlign(ctx.rUbFactor, bsAElem);
        OP_CHECK_IF(
            ctx.rUbFactor == 0,
            OP_LOGE(context, "rUbFactor floor-aligned to 0 (block=%ld elem rIMax=%ld)", bsAElem, rIMax),
            return ge::GRAPH_FAILED);
    }

    // rUbFactorAlign：仅「切在最内 R 且 tail R」时 rUbFactor 自身是 burst tail 行宽 → 需按 block CeilAlign
    //   - 全载分支（rUbFactor==rAxisSize）跳过 FloorAlign 时也走这里，让 padded 行宽 ≥ valid
    //   - FloorAlign 已生效时 CeilAlign 结果 == 原值，幂等
    //   - 其它情况 rUbFactor 是「外侧 R 轴」的 chunk count，不需要 block 对齐
    if (ctx.isTailR && ctx.rSplitAxisIdx == lastR) {
        ctx.rUbFactorAlign = Ops::Base::CeilAlign(ctx.rUbFactor, bsAElem);
    } else {
        ctx.rUbFactorAlign = ctx.rUbFactor;
    }
    return ge::GRAPH_SUCCESS;
}

// ─────────────────────────────────────────────────────────────────────────────
// 7) UB 切分 Step 3：R 全载后扩 A（rUbFactor==axisShape[rSplit] 且 rSplit 已是最外 R）
// ─────────────────────────────────────────────────────────────────────────────
static bool RIsFullyLoaded(const EuclideanNormCtx& ctx)
{
    const int32_t rSplit = ctx.rSplitAxisIdx;
    if (ctx.rUbFactor != ctx.axisShape[static_cast<size_t>(rSplit)]) {
        return false;
    }
    // 更外侧不再有 R 轴
    for (int32_t i = rSplit - 1; i >= 0; --i) {
        if (ctx.isReduceAxis[static_cast<size_t>(i)]) {
            return false;
        }
    }
    return true;
}

// 反解：固定 rUbFactorAlign / innerRProdAlign，求 aTotal 上界
//   rPaddedElems 必须用 padded R 计数，否则 burst tail 非对齐时 UB 实际占用 > 预算
static int64_t SolveAFromUbBudget(const EuclideanNormCtx& ctx)
{
    const int64_t ubAvailable = ctx.ubSize - kCacheBufBytes;
    const int64_t bytesPerRElem = (kDbFactor * kNPreReuse + kNPreParallel) * ctx.dtypeSize + kTmpBufCopies * kFp32Bytes;
    const int64_t rPaddedElems = ctx.rUbFactorAlign * ctx.innerRProdAlign;
    // 不等式：aTotal × (rPaddedElems × bytesPerRElem + N_out × sizeof(D_T)) ≤ ubAvailable
    const int64_t coeff = rPaddedElems * bytesPerRElem + kNOut * ctx.dtypeSize;
    if (coeff <= 0) {
        return -1;
    }
    return ubAvailable / coeff;
}

static ge::graphStatus ExpandAIfRFullyLoaded(gert::TilingContext* context, EuclideanNormCtx& ctx)
{
    if (!RIsFullyLoaded(ctx)) {
        return ge::GRAPH_SUCCESS;
    }

    // R 全载 → 用 UB 剩余预算反解新的 aTotal 上界（aTotal = aUbFactorAlign × innerAProdAlign）
    int64_t totalA = 1;
    for (int32_t i = 0; i < ctx.axisNum; ++i) {
        if (!ctx.isReduceAxis[static_cast<size_t>(i)]) {
            totalA *= ctx.axisShape[static_cast<size_t>(i)];
        }
    }
    // cacheBuf 硬约束：
    //   cacheCount × aTotal × sizeof(float) ≤ cacheBufUbSize
    // R 全载是进入本函数的前提（RIsFullyLoaded），此时 cacheCount = 1（rLoopCnt=1 → log2(1)+1=1）
    // 因此 aTotal ≤ cacheBufUbSize / sizeof(float)，直接卡在 atmpNew 上
    const int64_t cacheLaneLimit = kCacheBufBytes / kFp32Bytes;
    int64_t atmpNew = std::min(SolveAFromUbBudget(ctx), totalA);
    atmpNew = std::min(atmpNew, cacheLaneLimit);
    const int64_t curATotal = ctx.aUbFactorAlign * ctx.innerAProdAlign;
    if (atmpNew <= curATotal) {
        return ge::GRAPH_SUCCESS;
    }

    // 重做 climb：从最内 A 向外吸
    const int32_t lastA = LastAAxisIdx(ctx);
    const int64_t bsAElem = ctx.blockSize / ctx.dtypeSize;
    ctx.innerAProd = 1;
    ctx.innerAProdAlign = 1;
    ctx.aSplitAxisIdx = lastA;
    while (true) {
        const int64_t aSize = ctx.axisShape[static_cast<size_t>(ctx.aSplitAxisIdx)];
        if (aSize * ctx.innerAProdAlign >= atmpNew) {
            break;
        }
        const int32_t prev = PrevSameTypeAxis(ctx, ctx.aSplitAxisIdx, /*wantR=*/false);
        if (prev < 0) {
            break;
        }
        if (ctx.aSplitAxisIdx == lastA && !ctx.isTailR) {
            ctx.innerAProdAlign *= Ops::Base::CeilAlign(aSize, bsAElem);
        } else {
            ctx.innerAProdAlign *= aSize;
        }
        ctx.innerAProd *= aSize;
        ctx.aSplitAxisIdx = prev;
    }

    // FloorAlign 仅在「切在最内 A 且 tail A」（即 aSplit 在 burst 尾轴）时需要 block 对齐
    //   其余情况 aUbFactorAlign 是中间维度，无 block 对齐约束
    const int64_t aSize = ctx.axisShape[static_cast<size_t>(ctx.aSplitAxisIdx)];
    const int64_t aTotalUb = std::min(atmpNew / ctx.innerAProdAlign, aSize);
    const int32_t lastAIdx = LastAAxisIdx(ctx);
    if (!ctx.isTailR && ctx.aSplitAxisIdx == lastAIdx) {
        const int64_t aFloor = Ops::Base::FloorAlign(aTotalUb, bsAElem);
        OP_CHECK_IF(
            aFloor == 0,
            OP_LOGE(context, "ExpandA(tail A, LastA): FloorAlign to 0 (aTotalUb=%ld bs=%ld)", aTotalUb, bsAElem),
            return ge::GRAPH_FAILED);
        ctx.aUbFactor = aFloor;
        ctx.aUbFactorAlign = Ops::Base::CeilAlign(aFloor, bsAElem);
    } else {
        ctx.aUbFactor = aTotalUb;
        ctx.aUbFactorAlign = aTotalUb;
    }
    return ge::GRAPH_SUCCESS;
}

// ─────────────────────────────────────────────────────────────────────────────
// 8) 多核切分（fused aLoop）：把 aSplit 左侧所有 A 整根 + aSplit 的 outer chunk 数
//    fuse 成一根线性计数 aLoopCntTotal，按 coreNum 均衡分到大小核。
// ─────────────────────────────────────────────────────────────────────────────
static void ComputeFusedALoopSplit(EuclideanNormCtx& ctx)
{
    int64_t outerAProd = 1;
    for (int32_t i = 0; i < ctx.aSplitAxisIdx; i += 2) { // 严格 A R A R...，A 轴在偶位
        outerAProd *= ctx.axisShape[static_cast<size_t>(i)];
    }
    const int64_t aSplitAxisSize = ctx.axisShape[static_cast<size_t>(ctx.aSplitAxisIdx)];
    ctx.aSplitChunkCnt = Ops::Base::CeilDiv(aSplitAxisSize, ctx.aUbFactor);
    ctx.aLoopCntTotal = outerAProd * ctx.aSplitChunkCnt;

    ctx.aSmallCoreLoopCnt = ctx.aLoopCntTotal / ctx.coreNum;                 // floor
    ctx.aBigCoreCnt = static_cast<int32_t>(ctx.aLoopCntTotal % ctx.coreNum); // 大核个数
    ctx.aBigCoreLoopCnt = ctx.aSmallCoreLoopCnt + (ctx.aBigCoreCnt > 0 ? 1 : 0);
    ctx.usedCoreNum = (ctx.aSmallCoreLoopCnt > 0) ? static_cast<int32_t>(ctx.coreNum) : ctx.aBigCoreCnt;
    ctx.usedCoreNum = std::max(ctx.usedCoreNum, 1); // aLoopCntTotal==0 时兜底（实际不应触发）
}

// ─────────────────────────────────────────────────────────────────────────────
// 9) 算 rLoopCntTotal、UB sizes
// ─────────────────────────────────────────────────────────────────────────────
static void ComputeRLoopCnt(EuclideanNormCtx& ctx)
{
    int64_t outerR = 1;
    for (int32_t i = 0; i < ctx.rSplitAxisIdx; ++i) {
        if (ctx.isReduceAxis[static_cast<size_t>(i)]) {
            outerR *= ctx.axisShape[static_cast<size_t>(i)];
        }
    }
    const int64_t rChunks = Ops::Base::CeilDiv(ctx.axisShape[static_cast<size_t>(ctx.rSplitAxisIdx)], ctx.rUbFactor);
    ctx.rLoopCntTotal = outerR * rChunks;
}

static void ComputeUbSizes(EuclideanNormCtx& ctx)
{
    // unit 必须用 padded R（rUbFactorAlign），否则 burst tail 非对齐时 kernel 的 totalElems
    //   = aUbFactorAlign × innerAProdAlign × rUbFactorAlign × innerRProdAlign
    // 会大于 preReduceUbSize / tmpBufUbSize 算出的容量，造成 UB OOB。
    const int64_t unit = ctx.aUbFactorAlign * ctx.innerAProdAlign * ctx.rUbFactorAlign * ctx.innerRProdAlign;
    const int64_t aTotal = ctx.aUbFactorAlign * ctx.innerAProdAlign;
    ctx.preReduceUbSize = Ops::Base::CeilAlign(unit * ctx.dtypeSize, ctx.blockSize);
    ctx.tmpBufUbSize = Ops::Base::CeilAlign(unit * kFp32Bytes, ctx.blockSize);
    int64_t outRaw = aTotal * ctx.dtypeSize;
    // A=1 退化：outBuf 至少 N_out × blockSize
    outRaw = std::max(outRaw, kNOut * ctx.blockSize);
    ctx.postReduceUbSize = Ops::Base::CeilAlign(outRaw, ctx.blockSize);
}

// ─────────────────────────────────────────────────────────────────────────────
// 10) Group 模板触发判定与 A×R 2D 多核切分
// ─────────────────────────────────────────────────────────────────────────────
static bool ShouldUseGroup(const EuclideanNormCtx& ctx)
{
    if (ctx.aLoopCntTotal > ctx.coreNum / 2) {
        return false;
    }
    if (ctx.rLoopCntTotal <= 1) {
        return false;
    }
    return true;
}

static void ComputeGroupSplit(EuclideanNormCtx& ctx)
{
    int64_t totalOuter = ctx.aLoopCntTotal * ctx.rLoopCntTotal;
    int64_t perCoreNum = Ops::Base::CeilDiv(totalOuter, ctx.coreNum);
    int64_t numBlocks = Ops::Base::CeilDiv(totalOuter, perCoreNum);

    // numBlocks > aOuter 恒成立（group 触发条件 aOuter ≤ coreNum/2 && rOuter ≥ 2 保证）；
    // 对齐到 aOuter 倍数——保 2D 网格整齐
    if (Ops::Base::CeilAlign(numBlocks, ctx.aLoopCntTotal) <= ctx.coreNum) {
        numBlocks = Ops::Base::CeilAlign(numBlocks, ctx.aLoopCntTotal);
    } else {
        numBlocks = Ops::Base::FloorAlign(numBlocks, ctx.aLoopCntTotal);
    }

    ctx.usedCoreNum = static_cast<int32_t>(numBlocks);
    ctx.rGroupCnt = numBlocks / ctx.aLoopCntTotal; // = rGroups，workspace R 维
    ctx.isGroup = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// 11) workspace 大小设置
// ─────────────────────────────────────────────────────────────────────────────
static ge::graphStatus SetWorkspaceSize(gert::TilingContext* context, const EuclideanNormCtx& ctx)
{
    size_t* ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    size_t usrSize = 0;
    if (ctx.isGroup) {
        int64_t aTotal = 1;
        for (int32_t i = 0; i < ctx.axisNum; ++i) {
            if (!ctx.isReduceAxis[static_cast<size_t>(i)]) {
                aTotal *= ctx.axisShape[static_cast<size_t>(i)];
            }
        }
        usrSize = static_cast<size_t>(ctx.rGroupCnt) * static_cast<size_t>(aTotal) * sizeof(float);
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    ws[0] = usrSize + sysWorkspaceSize;
    OP_LOGI(context, "set ws size:%lu, usrSize:%lu", ws[0], usrSize);
    return ge::GRAPH_SUCCESS;
}

// ─────────────────────────────────────────────────────────────────────────────
// 12) 填 tilingdata + 打日志
// ─────────────────────────────────────────────────────────────────────────────
static ge::graphStatus FillAndLogTilingData(gert::TilingContext* context, const EuclideanNormCtx& ctx)
{
    EuclideanNormTilingData* td = context->GetTilingData<EuclideanNormTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);
    OP_CHECK_IF(
        memset_s(td, sizeof(EuclideanNormTilingData), 0, sizeof(EuclideanNormTilingData)) != EOK,
        OP_LOGE(context, "memset tilingdata error"), return ge::GRAPH_FAILED);

    int64_t axisStride[MAX_PATTERN_RANK] = {0};
    EuclideanNormCtx tmpForStride = ctx;
    ComputeAxisStrides(tmpForStride, axisStride);

    td->axisNum = ctx.axisNum;
    for (int32_t i = 0; i < MAX_PATTERN_RANK; ++i) {
        td->axisShape[i] = (i < ctx.axisNum) ? ctx.axisShape[static_cast<size_t>(i)] : 1;
        td->axisStride[i] = (i < ctx.axisNum) ? axisStride[i] : 0;
    }
    td->aLoopCntTotal = ctx.aLoopCntTotal;
    td->aSplitChunkCnt = ctx.aSplitChunkCnt;
    td->aBigCoreLoopCnt = ctx.aBigCoreLoopCnt;
    td->aSmallCoreLoopCnt = ctx.aSmallCoreLoopCnt;
    td->aBigCoreCnt = ctx.aBigCoreCnt;
    td->usedCoreNum = ctx.usedCoreNum;
    td->aSplitAxisIdx = ctx.aSplitAxisIdx;
    td->rSplitAxisIdx = ctx.rSplitAxisIdx;
    td->aUbFactor = ctx.aUbFactor;
    td->aUbFactorAlign = ctx.aUbFactorAlign;
    td->rUbFactor = ctx.rUbFactor;
    td->rUbFactorAlign = ctx.rUbFactorAlign;
    td->innerAProd = ctx.innerAProd;
    td->innerAProdAlign = ctx.innerAProdAlign;
    td->innerRProd = ctx.innerRProd;
    td->innerRProdAlign = ctx.innerRProdAlign;
    td->rLoopCntTotal = ctx.rLoopCntTotal;
    td->preReduceUbSize = ctx.preReduceUbSize;
    td->postReduceUbSize = ctx.postReduceUbSize;
    td->tmpBufUbSize = ctx.tmpBufUbSize;
    td->cacheBufUbSize = kCacheBufBytes;
    td->rGroupCnt = ctx.rGroupCnt;

    OP_LOGI(
        context, "EuclideanNorm tiling: dtype=%d axisNum=%d isTailR=%d usedCoreNum=%d isGroup=%d",
        static_cast<int>(ctx.xDtype), ctx.axisNum, static_cast<int>(ctx.isTailR), ctx.usedCoreNum,
        static_cast<int>(ctx.isGroup));
    OP_LOGI(
        context, "  axisShape=[%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld]", td->axisShape[0], td->axisShape[1],
        td->axisShape[2], td->axisShape[3], td->axisShape[4], td->axisShape[5], td->axisShape[6], td->axisShape[7],
        td->axisShape[8]);
    OP_LOGI(
        context, "  aLoopCntTotal=%ld aSplitChunkCnt=%ld | bigCore=%ld*%d smallCore=%ld*%d", td->aLoopCntTotal,
        td->aSplitChunkCnt, td->aBigCoreLoopCnt, td->aBigCoreCnt, td->aSmallCoreLoopCnt,
        td->usedCoreNum - td->aBigCoreCnt);
    OP_LOGI(
        context, "  aSplit=%d aUbFactor=%ld aUbFactorAlign=%ld", td->aSplitAxisIdx, td->aUbFactor, td->aUbFactorAlign);
    OP_LOGI(
        context,
        "  rSplit=%d rUbFactor=%ld rUbFactorAlign=%ld | innerAProd=%ld iAPAlign=%ld innerRProd=%ld iRPAlign=%ld",
        td->rSplitAxisIdx, td->rUbFactor, td->rUbFactorAlign, td->innerAProd, td->innerAProdAlign, td->innerRProd,
        td->innerRProdAlign);
    OP_LOGI(
        context, "  rLoopCntTotal=%ld | UB: preRed=%ld postRed=%ld tmp=%ld cache=%ld", td->rLoopCntTotal,
        td->preReduceUbSize, td->postReduceUbSize, td->tmpBufUbSize, td->cacheBufUbSize);
    if (ctx.isGroup) {
        OP_LOGI(context, "  group: rGroupCnt=%ld usedCoreNum=%d", td->rGroupCnt, td->usedCoreNum);
    }
    return ge::GRAPH_SUCCESS;
}

// ─────────────────────────────────────────────────────────────────────────────
// 11) 空 tensor 模板 tiling 计算
//
// EMPTY_A：output 是空 tensor，usedCoreNum=0、kernel 早退；TilingData 其余字段填 0、不读。
// EMPTY_R：每 output entry = sqrt(0) = 0；按规则算 aUbFactor + 多核切分，
//          把 aTotal 放进 axisShape[0]，N_tmp=0（本算子退化为直接写 D_T outBuf）。
// ─────────────────────────────────────────────────────────────────────────────
static void ComputeEmptyRTiling(EuclideanNormCtx& ctx)
{
    // max_ub_factor —— 仅 N_out=1 路 D_T，64KB 单 buf 封顶
    constexpr int64_t kMaxSingleUbBytes = 64 * 1024;
    const int64_t maxUbFactor = kMaxSingleUbBytes / ctx.dtypeSize;
    // 也受总 UB 约束（empty 模板 ubAvailable=ubSize；按 N_out × sizeof(D_T) 拆 chunk；
    // 通常 ubSize > 64KB 时 maxUbFactor 由 64KB 主导）
    const int64_t ubNatural = ctx.ubSize / ctx.dtypeSize;
    const int64_t maxUbFactorClamped = std::min(maxUbFactor, ubNatural);

    // aUbFactor + 多核
    constexpr int64_t kMinBytesPerCore = 4096;
    const int64_t minAPerCore = Ops::Base::CeilDiv(kMinBytesPerCore, ctx.dtypeSize);
    const int64_t aTotal = ctx.aTotalEmpty; // >= 1（边界情况）

    int64_t aUbFactor = std::max(minAPerCore, Ops::Base::CeilDiv(aTotal, ctx.coreNum));
    aUbFactor = std::min(aUbFactor, maxUbFactorClamped);
    aUbFactor = std::min(aUbFactor, aTotal);
    aUbFactor = std::max<int64_t>(aUbFactor, 1);

    const int64_t aLoopCntTotal = Ops::Base::CeilDiv(aTotal, aUbFactor);
    const int64_t aSmallCoreLoopCnt = aLoopCntTotal / ctx.coreNum; // floor
    const int32_t aBigCoreCnt = static_cast<int32_t>(aLoopCntTotal % ctx.coreNum);
    const int64_t aBigCoreLoopCnt = aSmallCoreLoopCnt + (aBigCoreCnt > 0 ? 1 : 0);
    const int32_t usedCoreNum = (aSmallCoreLoopCnt > 0) ? static_cast<int32_t>(ctx.coreNum) : std::max(aBigCoreCnt, 1);

    ctx.aUbFactor = aUbFactor;
    ctx.aLoopCntTotal = aLoopCntTotal;
    ctx.aSplitChunkCnt = aLoopCntTotal; // empty 无 outer A，二者相等
    ctx.aSmallCoreLoopCnt = aSmallCoreLoopCnt;
    ctx.aBigCoreLoopCnt = aBigCoreLoopCnt;
    ctx.aBigCoreCnt = aBigCoreCnt;
    ctx.usedCoreNum = usedCoreNum;

    // postReduceUbSize：CeilAlign(aUbFactor × sizeof(D_T), blockSize)；kernel TBuf 预算
    int64_t outRaw = aUbFactor * ctx.dtypeSize;
    outRaw = std::max(outRaw, kNOut * ctx.blockSize);
    ctx.postReduceUbSize = Ops::Base::CeilAlign(outRaw, ctx.blockSize);
    // EMPTY_R 不申请 preInBuf / tmpBuf / cacheBuf
    ctx.preReduceUbSize = 0;
    ctx.tmpBufUbSize = 0;
}

// 填 EMPTY_A / EMPTY_R 的 TilingData
// EMPTY_A：usedCoreNum=0；其余字段全 0
// EMPTY_R：复用 normal struct，重映射字段；axisShape[0] = aTotal、其余 axisShape[1..8] = 0
static ge::graphStatus FillEmptyTilingData(gert::TilingContext* context, const EuclideanNormCtx& ctx)
{
    EuclideanNormTilingData* td = context->GetTilingData<EuclideanNormTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);
    OP_CHECK_IF(
        memset_s(td, sizeof(EuclideanNormTilingData), 0, sizeof(EuclideanNormTilingData)) != EOK,
        OP_LOGE(context, "memset tilingdata error"), return ge::GRAPH_FAILED);

    if (ctx.emptyKind == EuclideanNormEmptyKind::EMPTY_A) {
        // usedCoreNum=0 → kernel 全核早退；其余字段 0，kernel 不读
        OP_LOGI(
            context, "EuclideanNorm EMPTY_A: dtype=%d usedCoreNum=0 (kernel early-exit)", static_cast<int>(ctx.xDtype));
        return ge::GRAPH_SUCCESS;
    }

    // EMPTY_R 字段映射
    td->axisShape[0] = ctx.aTotalEmpty; // aTotal 走 axisShape[0]
    td->usedCoreNum = ctx.usedCoreNum;
    td->aLoopCntTotal = ctx.aLoopCntTotal;
    td->aSplitChunkCnt = ctx.aSplitChunkCnt;
    td->aBigCoreLoopCnt = ctx.aBigCoreLoopCnt;
    td->aSmallCoreLoopCnt = ctx.aSmallCoreLoopCnt;
    td->aBigCoreCnt = ctx.aBigCoreCnt;
    td->aUbFactor = ctx.aUbFactor;
    td->postReduceUbSize = ctx.postReduceUbSize;
    // 其余字段（axisNum / axisShape[1..8] / axisStride / aSplit / rSplit / aUbFactorAlign /
    //   rUbFactor / rUbFactorAlign / innerAProd* / innerRProd* / rLoopCntTotal /
    //   preReduceUbSize / tmpBufUbSize / cacheBufUbSize）保持 0、kernel 不读

    OP_LOGI(
        context,
        "EuclideanNorm EMPTY_R: dtype=%d aTotal=%ld aUbFactor=%ld usedCoreNum=%d | "
        "bigCore=%ld*%d smallCore=%ld*%d | postRed=%ld",
        static_cast<int>(ctx.xDtype), ctx.aTotalEmpty, td->aUbFactor, td->usedCoreNum, td->aBigCoreLoopCnt,
        td->aBigCoreCnt, td->aSmallCoreLoopCnt, td->usedCoreNum - td->aBigCoreCnt, td->postReduceUbSize);
    return ge::GRAPH_SUCCESS;
}

// ─────────────────────────────────────────────────────────────────────────────
// 12) 空 tensor 路径处理；返回 GRAPH_SUCCESS 表示已处理完毕调用方应直接 return
// ─────────────────────────────────────────────────────────────────────────────
static ge::graphStatus HandleEmptyTensor(gert::TilingContext* context, EuclideanNormCtx& ctx)
{
    if (ctx.emptyKind == EuclideanNormEmptyKind::EMPTY_R) {
        ComputeEmptyRTiling(ctx);
    } else {
        ctx.usedCoreNum = 0;
    }
    OP_CHECK_IF(FillEmptyTilingData(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    const uint64_t tilingKey = GET_TPL_TILING_KEY(
        static_cast<uint64_t>(0U),  // templateType
        static_cast<uint64_t>(1U),  // isEmptyTensor
        static_cast<uint64_t>(0U)); // isTailR
    context->SetTilingKey(tilingKey);
    const uint32_t blockDim = static_cast<uint32_t>(std::max(ctx.usedCoreNum, 1));
    context->SetBlockDim(blockDim);

    OP_CHECK_IF(SetWorkspaceSize(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tiling 主入口
// ─────────────────────────────────────────────────────────────────────────────
static ge::graphStatus EuclideanNormTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context, "Begin EuclideanNormTilingFunc");
    EuclideanNormCtx ctx;

    OP_CHECK_IF(GetPlatformInfo(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetShapeAndDtype(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseAxesTensor(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    BuildInitialAxisList(ctx);
    ClassifyEmptyTensor(ctx);

    if (ctx.emptyKind != EuclideanNormEmptyKind::NORMAL) {
        return HandleEmptyTensor(context, ctx);
    }

    OP_CHECK_IF(PreprocessPattern(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    ComputeAUbFactor(ctx);
    OP_CHECK_IF(ComputeRUbFactor(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(ExpandAIfRFullyLoaded(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    ComputeFusedALoopSplit(ctx);
    ComputeRLoopCnt(ctx);

    // Group 模板触发判定
    if (ShouldUseGroup(ctx)) {
        ComputeGroupSplit(ctx);
        OP_CHECK_IF(
            context->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "Failed to set ScheduleMode!"), return ge::GRAPH_FAILED);
    }
    ComputeUbSizes(ctx);

    OP_CHECK_IF(FillAndLogTilingData(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    const uint64_t tilingKey = GET_TPL_TILING_KEY(
        static_cast<uint64_t>(ctx.isGroup ? 1U : 0U), static_cast<uint64_t>(0U),
        static_cast<uint64_t>(ctx.isTailR ? 1U : 0U));
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(static_cast<uint32_t>(std::max(ctx.usedCoreNum, 1)));

    OP_CHECK_IF(SetWorkspaceSize(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForEuclideanNorm([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EuclideanNorm)
    .Tiling(EuclideanNormTilingFunc)
    .TilingParse<EuclideanNormCompileInfo>(TilingParseForEuclideanNorm)
    .TilingInputsDataDependency({1}); // axes（索引 1）值依赖：tiling 阶段需读取张量内容做合轴决策

} // namespace optiling
