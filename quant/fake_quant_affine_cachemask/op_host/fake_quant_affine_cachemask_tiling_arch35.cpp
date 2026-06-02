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
 * \file fake_quant_affine_cachemask_tiling.cpp
 * \brief arch35 详细 Tiling for FakeQuantAffineCachemask
 *
 * 套用 static-quant-tiling.md reference：
 *   - SelectMode (scale.numel + axis)
 *   - MergeShape (PT: 2D, PC: 2D, PH: 3D)
 *   - CalcPT/PC/PH 三函数拆分，返回 TilingResult
 *   - PC 二路竞争 + 二选一 UB
 *   - PH 三路竞争 + 三级退化 UB (dim2 > 32B 才允许切 dim2)
 *   - PC_NDDMA：dim1 < 32B && blockAxis==0 时升级 mode（kernel 走 PC 同实现）
 *   - WriteTilingData
 *   - TilingKey: D_T_X (binary) × MODE × HAS_ZP
 */

#include <algorithm>
#include <securec.h>
#include "register/op_def_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "../op_kernel/arch35/fake_quant_affine_cachemask_tiling_data.h"
#include "../op_kernel/arch35/fake_quant_affine_cachemask_tiling_key.h"
#include "fake_quant_affine_cachemask_tiling_arch35.h"

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;

// ---------- 常量 ----------
constexpr uint32_t WS_SYS_SIZE     = 0U;
constexpr size_t   WORKSPACE_NUM   = 1;
constexpr int64_t  RESERVED_UB     = 4096;     // 给 stack / event / SPR / 中间小 buf
constexpr int64_t  BUFF_NUM        = 2;        // 双缓冲
constexpr int64_t  CACHE_LINE      = 128;      // arch35
constexpr int64_t  BLOCK_SIZE      = 32;       // DMA 最小对齐单元
constexpr int64_t  VL_BYTES        = 256;      // VECTOR_REG_WIDTH (arch3510)
constexpr int64_t  PC_NDDMA_THR    = 32;       // dim1 < 32B 时 PC → PC_NDDMA

constexpr int64_t MODE_PT       = 0;
constexpr int64_t MODE_PC       = 1;
constexpr int64_t MODE_PC_NDDMA = 2;
constexpr int64_t MODE_PH       = 3;

// fake-quant 单元素字节模型（含 mask + 中间 fp32/int32 + DB）
// 行相关：x(in) + y(out) + mask(uint8) → 双缓冲乘 2
// 行无关（PC/PH）：scale(fp32) + zp(fp32) 段加载，按 baseLen 计入
// 中间 buf：fp32_tmp + qFp32 + maskFp32 + t2 + maskHalf + int32_tmp
//
// 保守估计：fp32 输入路径 行相关 ≈ (4+4+1)*2 = 18B；中间 6 个 buf ≈ 6*4 = 24B；
//          总 ≈ 42B；fp16/bf16 类似（x/y 2B*2 + 中间不变）≈ 32B
constexpr int64_t BYTES_X(int64_t xBytes) { return (xBytes + xBytes + 1) * BUFF_NUM; }
constexpr int64_t BYTES_INTERMEDIATE = 4 * 6;  // 6 fp32 中间 buf
constexpr int64_t BYTES_SCALE_ZP_FP32 = 4 + 4; // PC/PH 行无关 scale + zp

// ---------- shape utilities ----------
static const gert::Shape g_vec_1_shape = {1};
static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    return in_shape.GetDimNum() == 0 ? g_vec_1_shape : in_shape;
}

static int64_t XDtypeBytes(ge::DataType dt)
{
    if (dt == ge::DT_FLOAT)   return 4;
    if (dt == ge::DT_FLOAT16) return 2;
    if (dt == ge::DT_BF16)    return 2;
    return 4;
}

// ---------- 平台信息 ----------
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "coreNum=0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0, OP_LOGE(context, "ubSize=0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ---------- 形状属性 ----------
struct ShapeInfo {
    gert::Shape  xShape;
    int64_t      scaleNum = 0;
    int64_t      zpNum    = 0;
    int64_t      axisNorm = -1;
    int64_t      totalNum = 0;
    ge::DataType xDtype;
    ge::DataType zpDtype;
};

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, ShapeInfo* info)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    info->xShape = EnsureNotScalar(inputX->GetStorageShape());
    info->totalNum = info->xShape.GetShapeSize();

    auto inputScale = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputScale);
    info->scaleNum = EnsureNotScalar(inputScale->GetStorageShape()).GetShapeSize();

    auto inputZp = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputZp);
    info->zpNum = EnsureNotScalar(inputZp->GetStorageShape()).GetShapeSize();

    auto xDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    info->xDtype = xDesc->GetDataType();

    auto zpDesc = context->GetInputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, zpDesc);
    info->zpDtype = zpDesc->GetDataType();

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* axisPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, axisPtr);

    int64_t rank = static_cast<int64_t>(info->xShape.GetDimNum());
    int64_t axisVal = *axisPtr;
    if (axisVal < 0) axisVal += rank;
    if (axisVal < 0) axisVal = 0;
    if (axisVal >= rank) axisVal = rank - 1;
    info->axisNorm = axisVal;

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    OP_CHECK_IF(supportedDtype.count(info->xDtype) == 0,
                OP_LOGE(context, "x dtype not supported: %d", static_cast<int>(info->xDtype)),
                return ge::GRAPH_FAILED);
    // zp 只允许 INT32 或与 x 同 dtype
    OP_CHECK_IF(!(info->zpDtype == ge::DT_INT32 || info->zpDtype == info->xDtype),
                OP_LOGE(context, "zp dtype must be INT32 or same as x: zp=%d x=%d",
                        static_cast<int>(info->zpDtype), static_cast<int>(info->xDtype)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ---------- SelectMode ----------
static int64_t SelectMode(const ShapeInfo& info)
{
    if (info.scaleNum <= 1) return MODE_PT;
    int64_t rank = static_cast<int64_t>(info.xShape.GetDimNum());
    if (info.axisNorm == rank - 1) return MODE_PC;
    return MODE_PH;
}

// ---------- MergeShape ----------
static void MergeShape(const ShapeInfo& info, int64_t mode,
                       int64_t* dim0, int64_t* dim1, int64_t* dim2)
{
    int64_t rank = static_cast<int64_t>(info.xShape.GetDimNum());
    if (mode == MODE_PT) {
        *dim0 = 1;
        *dim1 = info.totalNum;
        *dim2 = 1;
        return;
    }
    if (mode == MODE_PC || mode == MODE_PC_NDDMA) {
        *dim1 = info.xShape.GetDim(rank - 1);
        if (*dim1 <= 0) *dim1 = 1;
        *dim0 = (*dim1 > 0) ? (info.totalNum / (*dim1)) : 1;
        *dim2 = 1;
        return;
    }
    // PH
    int64_t pre = 1, cur = info.xShape.GetDim(info.axisNorm), post = 1;
    for (int64_t i = 0; i < info.axisNorm; ++i)  pre  *= info.xShape.GetDim(i);
    for (int64_t i = info.axisNorm + 1; i < rank; ++i) post *= info.xShape.GetDim(i);
    *dim0 = pre  <= 0 ? 1 : pre;
    *dim1 = cur  <= 0 ? 1 : cur;
    *dim2 = post <= 0 ? 1 : post;
}

// ---------- TilingResult ----------
struct TilingResult {
    int64_t mode            = 0;
    int64_t numCore         = 0;
    int64_t blockAxis       = 0;
    int64_t blockUnion      = 1;
    int64_t blockFactor     = 0;
    int64_t blockTailFactor = 0;
    int64_t ubAxis          = 0;
    int64_t baseN           = 1;
    int64_t baseLen         = 0;
};

// 公用：按 dataCount 与字节模型求 maxBase
static int64_t CalcMaxBaseLen(int64_t ubSize, int64_t xBytes, bool perChannelLike)
{
    int64_t bytesPerElem = BYTES_X(xBytes) + BYTES_INTERMEDIATE;
    if (perChannelLike) {
        bytesPerElem += BYTES_SCALE_ZP_FP32;   // scale/zp 按 baseLen 段加载（不乘 baseN）
    }
    int64_t available = ubSize - RESERVED_UB;
    if (available <= 0) available = ubSize / 2;
    int64_t maxBase = available / bytesPerElem;
    if (maxBase < BLOCK_SIZE) maxBase = BLOCK_SIZE;
    return maxBase;
}

// 公用：给定 base（行宽），计算还能放多少行
static int64_t CalcMaxN(int64_t ubSize, int64_t base, int64_t xBytes)
{
    int64_t bytesPerNRow  = base * BYTES_X(xBytes);                         // 行相关 × baseN
    int64_t bytesScaleSeg = base * BYTES_SCALE_ZP_FP32;                     // 行无关常驻
    int64_t bytesIntermed = base * BYTES_INTERMEDIATE;                      // 中间 buf（按 baseLen）
    int64_t fixed         = RESERVED_UB + bytesScaleSeg + bytesIntermed;
    int64_t available     = ubSize - fixed;
    if (available <= 0) return 1;
    int64_t maxN = available / bytesPerNRow;
    if (maxN < 1) maxN = 1;
    return maxN;
}

// ---------- PT Tiling ----------
//   blockAxis=1（永远切 dim1）；ubAxis=1；baseN=1；
//   blockFactor 按 cacheLine 块对齐
static TilingResult CalcPTTiling(int64_t dim1, int64_t coreNum, int64_t ubSize, int64_t xBytes)
{
    TilingResult r;
    r.mode = MODE_PT;
    r.blockAxis = 1;
    r.ubAxis    = 1;
    r.baseN     = 1;
    r.blockUnion = 1;

    if (xBytes <= 0) return TilingResult{};
    int64_t cacheLineElems = CACHE_LINE / (xBytes > 0 ? xBytes : 1);
    if (cacheLineElems <= 0) cacheLineElems = 1;

    int64_t cacheLineNum = CeilDiv(dim1, cacheLineElems);
    int64_t actCore      = CeilDiv(cacheLineNum, CeilDiv(cacheLineNum, coreNum));
    if (actCore <= 0) actCore = 1;
    r.numCore = actCore;
    r.blockFactor = CeilDiv(cacheLineNum, actCore) * cacheLineElems;
    r.blockTailFactor = dim1 - r.blockFactor * (actCore - 1);
    if (r.blockTailFactor <= 0) r.blockTailFactor = r.blockFactor;

    // UB
    int64_t maxBase = FloorAlign(CalcMaxBaseLen(ubSize, xBytes, /*perChannelLike=*/false),
                                  cacheLineElems);
    if (maxBase < cacheLineElems) maxBase = cacheLineElems;
    int64_t blockBase = CeilAlign(r.blockFactor, cacheLineElems);
    r.baseLen = std::min(blockBase, maxBase);
    if (r.baseLen <= 0) r.baseLen = cacheLineElems;
    return r;
}

// ---------- PC Tiling ----------
//   blockAxis ∈ {0, 1}（二路竞争）；ubAxis ∈ {0, 1}；baseN ∈ {1, maxN}
//   blockAxis=0 切合轴（行）；blockAxis=1 切尾轴（列，按 cacheLine 块）
static TilingResult CalcPCTiling(int64_t dim0, int64_t dim1, int64_t coreNum,
                                  int64_t ubSize, int64_t xBytes)
{
    TilingResult r;
    r.mode = MODE_PC;
    if (xBytes <= 0) return TilingResult{};
    int64_t cacheLineElems = CACHE_LINE / (xBytes > 0 ? xBytes : 1);
    if (cacheLineElems <= 0) cacheLineElems = 1;

    int64_t cacheLineNum = CeilDiv(dim1, cacheLineElems);
    int64_t actCore0     = CeilDiv(dim0, CeilDiv(dim0, coreNum));         // 切 dim0
    int64_t actCore1     = CeilDiv(cacheLineNum, CeilDiv(cacheLineNum, coreNum)); // 切 dim1
    if (actCore0 <= 0) actCore0 = 1;
    if (actCore1 <= 0) actCore1 = 1;

    if (actCore0 >= actCore1) {
        r.blockAxis = 0;
        r.numCore   = actCore0;
        r.blockFactor     = CeilDiv(dim0, actCore0);
        r.blockTailFactor = dim0 - r.blockFactor * (actCore0 - 1);
    } else {
        r.blockAxis = 1;
        r.numCore   = actCore1;
        r.blockFactor     = CeilDiv(cacheLineNum, actCore1) * cacheLineElems;
        r.blockTailFactor = dim1 - r.blockFactor * (actCore1 - 1);
    }
    if (r.blockTailFactor <= 0) r.blockTailFactor = r.blockFactor;
    r.blockUnion = 1;

    // UB 二选一
    int64_t maxBase = FloorAlign(CalcMaxBaseLen(ubSize, xBytes, /*perChannelLike=*/true),
                                  cacheLineElems);
    if (maxBase < cacheLineElems) maxBase = cacheLineElems;
    int64_t blockBase = (r.blockAxis == 0) ? dim1 : r.blockFactor;
    blockBase = CeilAlign(blockBase, cacheLineElems);

    if (blockBase <= maxBase / 2 && blockBase > 0) {
        // 多行模式
        int64_t maxN = CalcMaxN(ubSize, blockBase, xBytes);
        int64_t blockInner = (r.blockAxis == 0) ? r.blockFactor : dim0;
        r.ubAxis  = 0;
        r.baseN   = std::min(maxN, blockInner);
        if (r.baseN < 1) r.baseN = 1;
        r.baseLen = blockBase;
    } else {
        // 单行退化
        r.ubAxis  = 1;
        r.baseN   = 1;
        r.baseLen = std::min(blockBase, maxBase);
        if (r.baseLen <= 0) r.baseLen = cacheLineElems;
    }
    return r;
}

// ---------- PC → PC_NDDMA 降级 ----------
//   触发：dim1 < 32B (即 dim1 * xBytes < BLOCK_SIZE) && blockAxis==0
//   降级行为：mode 改 PC_NDDMA；UB 不做 cacheLine 对齐
static void DemoteToPCNddma(TilingResult* r, int64_t dim1, int64_t xBytes, int64_t ubSize)
{
    if (!(r->blockAxis == 0 && dim1 * xBytes < PC_NDDMA_THR)) return;
    r->mode = MODE_PC_NDDMA;
    // UB 不对齐 cacheLine（NDDMA 自带跨距）
    int64_t maxBase = CalcMaxBaseLen(ubSize, xBytes, /*perChannelLike=*/true);
    int64_t blockBase = dim1; // 切合轴时 blockBase = dim1
    if (blockBase <= 0) blockBase = 1;
    if (blockBase <= maxBase / 2) {
        int64_t maxN = CalcMaxN(ubSize, blockBase, xBytes);
        int64_t blockInner = r->blockFactor;
        r->ubAxis  = 0;
        r->baseN   = std::min(maxN, blockInner);
        if (r->baseN < 1) r->baseN = 1;
        r->baseLen = blockBase;
    } else {
        r->ubAxis  = 1;
        r->baseN   = 1;
        r->baseLen = std::min(blockBase, maxBase);
        if (r->baseLen <= 0) r->baseLen = 1;
    }
}

// ---------- PH Tiling ----------
//   blockAxis ∈ {0, 1, 2}（三路竞争；dim2 ≤ 32B 时禁切 dim2）
//   UB 三级退化：整块 / 整行批 / 部分列
static int64_t GetCoreNumDoubleCut(int64_t outerDim, int64_t innerDim, int64_t totalCore)
{
    if (outerDim <= 0) return 0;
    int64_t yCore = totalCore / (outerDim > 0 ? outerDim : 1);
    if (yCore == 0) return 0;
    return outerDim * CeilDiv(innerDim, CeilDiv(innerDim, yCore));
}

static void SelectPHBlockAxis(TilingResult* r, int64_t dim0, int64_t dim1, int64_t dim2,
                               int64_t coreNum, int64_t cacheLineNum, int64_t xBytes)
{
    int64_t core0 = (dim0 > 0) ? CeilDiv(dim0, CeilDiv(dim0, coreNum)) : 0;
    int64_t core1 = GetCoreNumDoubleCut(dim0, dim1, coreNum);
    int64_t core2 = (dim2 * xBytes > BLOCK_SIZE)
                      ? GetCoreNumDoubleCut(dim0 * dim1, cacheLineNum, coreNum)
                      : 0;

    r->blockAxis = 0;
    r->numCore   = (core0 > 0) ? core0 : 1;
    if (core1 > r->numCore) {
        r->blockAxis = 1;
        r->numCore   = core1;
    }
    if (core2 > r->numCore && dim2 * xBytes > BLOCK_SIZE) {
        r->blockAxis = 2;
        r->numCore   = core2;
    }
    if (r->numCore <= 0) r->numCore = 1;
}

static void CalcPHBlockParams(TilingResult* r, int64_t dim0, int64_t dim1, int64_t dim2,
                               int64_t cacheLineNum, int64_t cacheLineElems)
{
    if (r->blockAxis == 0) {
        r->blockUnion = 1;
        r->blockFactor     = CeilDiv(dim0, r->numCore);
        r->blockTailFactor = dim0 - r->blockFactor * (r->numCore - 1);
    } else if (r->blockAxis == 1) {
        r->blockUnion = (dim0 > 0) ? (r->numCore / dim0) : 1;
        if (r->blockUnion <= 0) r->blockUnion = 1;
        r->blockFactor     = CeilDiv(dim1, r->blockUnion);
        r->blockTailFactor = dim1 - r->blockFactor * (r->blockUnion - 1);
    } else {
        int64_t outer = dim0 * dim1;
        r->blockUnion = (outer > 0) ? (r->numCore / outer) : 1;
        if (r->blockUnion <= 0) r->blockUnion = 1;
        int64_t bf_cl = CeilDiv(cacheLineNum, r->blockUnion);
        r->blockFactor     = bf_cl * cacheLineElems;
        r->blockTailFactor = dim2 - r->blockFactor * (r->blockUnion - 1);
    }
    if (r->blockTailFactor <= 0) r->blockTailFactor = r->blockFactor;
}

static void CalcPHUbTiling(TilingResult* r, int64_t dim1, int64_t dim2,
                            int64_t ubSize, int64_t xBytes,
                            int64_t cacheLineElems)
{
    int64_t shape2  = CeilAlign(dim2, cacheLineElems);
    if (shape2 <= 0) shape2 = 1;
    int64_t maxBase = FloorAlign(CalcMaxBaseLen(ubSize, xBytes, /*perChannelLike=*/true),
                                  cacheLineElems);
    if (maxBase < cacheLineElems) maxBase = cacheLineElems;
    int64_t blockSize = CeilAlign(r->blockFactor, cacheLineElems);

    if (r->blockAxis == 0) {
        if (dim1 * shape2 <= maxBase) {
            r->ubAxis = 0; r->baseN = dim1;                       r->baseLen = shape2;
        } else if (shape2 <= maxBase) {
            r->ubAxis = 1; r->baseN = std::max<int64_t>(maxBase / (shape2 > 0 ? shape2 : 1), 1); r->baseLen = shape2;
        } else {
            r->ubAxis = 2; r->baseN = 1;                          r->baseLen = maxBase;
        }
    } else if (r->blockAxis == 1) {
        if (shape2 <= maxBase) {
            r->ubAxis = 1;
            r->baseN  = std::min<int64_t>(r->blockFactor, std::max<int64_t>(maxBase / (shape2 > 0 ? shape2 : 1), 1));
            r->baseLen = shape2;
        } else {
            r->ubAxis = 2; r->baseN = 1; r->baseLen = maxBase;
        }
    } else {
        r->ubAxis = 2; r->baseN = 1; r->baseLen = std::min(blockSize, maxBase);
    }
    if (r->baseLen <= 0) r->baseLen = cacheLineElems;
    if (r->baseN  <= 0) r->baseN  = 1;

    if ((dim2 * xBytes) % BLOCK_SIZE != 0) {
        r->baseN = 1;
    }
}

static TilingResult CalcPHTiling(int64_t dim0, int64_t dim1, int64_t dim2,
                                  int64_t coreNum, int64_t ubSize, int64_t xBytes)
{
    TilingResult r;
    r.mode = MODE_PH;
    if (xBytes <= 0) return TilingResult{};
    int64_t cacheLineElems = CACHE_LINE / (xBytes > 0 ? xBytes : 1);
    if (cacheLineElems <= 0) cacheLineElems = 1;

    int64_t cacheLineNum = CeilDiv(dim2, cacheLineElems);

    SelectPHBlockAxis(&r, dim0, dim1, dim2, coreNum, cacheLineNum, xBytes);
    CalcPHBlockParams(&r, dim0, dim1, dim2, cacheLineNum, cacheLineElems);
    CalcPHUbTiling(&r, dim1, dim2, ubSize, xBytes, cacheLineElems);
    return r;
}

// ---------- WriteTilingData ----------
static void WriteTilingData(FakeQuantAffineCachemaskTilingDataArch35* td, const TilingResult& r,
                            int64_t dim0, int64_t dim1, int64_t dim2,
                            int64_t hasZeroPoint, int64_t axisNorm,
                            int64_t quantMin, int64_t quantMax)
{
    td->numCore         = r.numCore;
    td->mode            = r.mode;
    td->dim0            = dim0;
    td->dim1            = dim1;
    td->dim2            = dim2;
    td->blockAxis       = r.blockAxis;
    td->blockUnion      = r.blockUnion;
    td->blockFactor     = r.blockFactor;
    td->blockTailFactor = r.blockTailFactor;
    td->ubAxis          = r.ubAxis;
    td->baseN           = r.baseN;
    td->baseLen         = r.baseLen;
    td->hasZeroPoint    = hasZeroPoint;
    td->axis            = axisNorm;
    td->quantMin        = quantMin;
    td->quantMax        = quantMax;
}

// ---------- Workspace ----------
static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// ---------- Dispatch ----------
static bool DispatchTiling(int64_t mode, int64_t dim0, int64_t dim1, int64_t dim2,
                            int64_t coreNum, int64_t ubSize, int64_t xBytes,
                            TilingResult* res)
{
    switch (mode) {
        case MODE_PT:
            *res = CalcPTTiling(dim1, coreNum, ubSize, xBytes);
            return true;
        case MODE_PC:
            *res = CalcPCTiling(dim0, dim1, coreNum, ubSize, xBytes);
            DemoteToPCNddma(res, dim1, xBytes, ubSize);
            return true;
        case MODE_PH:
            *res = CalcPHTiling(dim0, dim1, dim2, coreNum, ubSize, xBytes);
            return true;
        default:
            return false;
    }
}

// ---------- 空张量处理 ----------
static void HandleEmptyTensor(gert::TilingContext* context,
                               FakeQuantAffineCachemaskTilingDataArch35* tiling,
                               int64_t mode, int64_t dim0, int64_t dim1, int64_t dim2,
                               int64_t hasZeroPoint, int64_t axisNorm,
                               int64_t quantMin, int64_t quantMax,
                               ge::DataType xDtype, ge::DataType zpDtype)
{
    WriteTilingData(tiling, TilingResult{}, dim0, dim1, dim2,
                    hasZeroPoint, axisNorm, quantMin, quantMax);
    tiling->numCore = 1;
    tiling->mode    = mode;
    context->SetBlockDim(1);
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(xDtype),
                          static_cast<uint32_t>(zpDtype),
                          static_cast<uint32_t>(mode),
                          static_cast<uint32_t>(hasZeroPoint));
}

// ---------- 入口 ----------
static ge::graphStatus FakeQuantAffineCachemaskTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo failed"), return ge::GRAPH_FAILED);

    ShapeInfo info;
    OP_CHECK_IF(GetShapeAttrsInfo(context, &info) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAttrsInfo failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize failed"), return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* qMinPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t* qMaxPtr = attrs->GetAttrPointer<int64_t>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, qMinPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, qMaxPtr);

    int64_t mode = SelectMode(info);
    int64_t dim0 = 1, dim1 = 0, dim2 = 1;
    MergeShape(info, mode, &dim0, &dim1, &dim2);

    int64_t hasZeroPoint = (info.zpNum > 0) ? 1 : 0;
    int64_t xBytes = XDtypeBytes(info.xDtype);

    auto* tiling = context->GetTilingData<FakeQuantAffineCachemaskTilingDataArch35>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(*tiling), 0, sizeof(*tiling)) != EOK,
                OP_LOGE(context, "memset tiling failed"), return ge::GRAPH_FAILED);

    if (info.totalNum == 0) {
        HandleEmptyTensor(context, tiling, mode, dim0, dim1, dim2,
                          hasZeroPoint, info.axisNorm, *qMinPtr, *qMaxPtr,
                          info.xDtype, info.zpDtype);
        return ge::GRAPH_SUCCESS;
    }

    TilingResult res;
    int64_t ubSizeI = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(!DispatchTiling(mode, dim0, dim1, dim2, coreNum, ubSizeI, xBytes, &res),
                OP_LOGE(context, "unsupported mode %ld", static_cast<long>(mode)),
                return ge::GRAPH_FAILED);

    WriteTilingData(tiling, res, dim0, dim1, dim2,
                    hasZeroPoint, info.axisNorm, *qMinPtr, *qMaxPtr);

    context->SetBlockDim(static_cast<uint32_t>(res.numCore));
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(info.xDtype),
                          static_cast<uint32_t>(info.zpDtype),
                          static_cast<uint32_t>(res.mode),
                          static_cast<uint32_t>(hasZeroPoint));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForFakeQuantAffineCachemask(
    [[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4FakeQuantAffineCachemaskArch35(gert::TilingContext* context)
{
    return FakeQuantAffineCachemaskTilingFunc(context);
}

} // namespace optiling
