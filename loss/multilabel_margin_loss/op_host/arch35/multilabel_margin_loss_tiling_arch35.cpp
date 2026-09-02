/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */
/**
 * \file multilabel_margin_loss_tiling_arch35.cpp
 * \brief MultilabelMarginLoss arch35(ascend950/regbase)独立 tiling。
 *
 * 组织方式对齐仓上双代先例 loss/chamfer_distance_grad:唯一的 IMPL_OP_OPTILING 留在根 tiling,
 * 根 tiling 只做一次 soc 分派;A5 的全部计算(dtype 守护 / 分核 / UB 预算 / workspace)都在本文件,
 * A2 的 tiling 与 tiling data 一个字节不动。
 *
 * 本文件承担 host 侧 buffer 实算:kernel 不再自行推导任何 UB 尺寸,全部取自下发的 ubFactor。
 */
#include <cstring>
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "securec.h"
#include "../../op_kernel/arch35/multilabel_margin_loss_tiling_data_arch35.h"

namespace optiling {
namespace {

constexpr uint32_t BATCH_MODE = 1;
constexpr int32_t REDUCTION_INVALID = -1;
constexpr int32_t RED_NONE = 0;

// mean/sum 每核独占槽位:8 个 float = 32B,保证相邻核不共享 GM 写块(故不需要原子加)。
constexpr uint32_t WS_CORE_STRIDE = 8U;
constexpr uint32_t MERGE_VL_FP32 = 64U; // 跨核合并的矢量车道数(fp32)
// UB 分块粒度的对齐单位:同时满足 float 的 8 元素(32B)与 fp16/bf16 的 16 元素对齐。
constexpr uint32_t TILE_ALIGN = 16U;
// 分块下限:低于此值分块收益为负,且需保证 mean/sum 的单元素暂存可用。
constexpr uint32_t TILE_MIN = 16U;
// 分块长度上限,取保守值。约束来自单条向量指令的 repeat 上限(MAX_REPEAT_TIMES=255 ×
// ONE_REPEAT_BYTES=256B,float 折合 16320 元素):FinalizeOutput 对整块做一次 Adds/Cast +
// DataCopyPad。取 8192 相对该上限留一倍余量,且为 16/64 的整数倍,便于对齐。
constexpr uint32_t MAX_VEC_ELEMS = 8192U;
// UB 余量:给编译器分配对齐与框架预留留出裕度,宁可高估(高估只会缩小 ubFactor,是安全方向)。
constexpr uint32_t UB_RESERVE_BYTES = 8192U;

inline uint32_t CeilAlign(uint32_t v, uint32_t align) { return ((v + align - 1U) / align) * align; }

inline uint32_t FloorAlign(uint32_t v, uint32_t align) { return (v / align) * align; }

int32_t ParseReductionAttr(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return REDUCTION_INVALID;
    }
    const char* reductionStr = attrs->GetAttrPointer<char>(0);
    if (reductionStr == nullptr) {
        return REDUCTION_INVALID;
    }
    if (strcmp(reductionStr, "none") == 0) {
        return 0;
    }
    if (strcmp(reductionStr, "mean") == 0) {
        return 1;
    }
    if (strcmp(reductionStr, "sum") == 0) {
        return 2;
    }
    return REDUCTION_INVALID;
}

// Tiling 是独立入口(单算子直测不经 aclnn/GE),dtype 守护必须在此自己再做一遍。
// 契约(对齐 def 的 6 组合):x/y ∈ {FLOAT,FLOAT16,BF16} 且 y==x;target=INT32;
// is_target = INT32(GE 图)或 ==x(aclnn 跟随 self)。
ge::graphStatus CheckDtypeValid(gert::TilingContext* context)
{
    auto xDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    auto targetDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetDesc);
    auto yDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    auto isTargetDesc = context->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, isTargetDesc);

    ge::DataType xDtype = xDesc->GetDataType();
    if (xDtype != ge::DT_FLOAT && xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(),
                                  "FLOAT, FLOAT16 or BF16");
        return ge::GRAPH_FAILED;
    }
    if (yDesc->GetDataType() != xDtype) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "y",
                                  ge::TypeUtils::DataTypeToSerialString(yDesc->GetDataType()).c_str(),
                                  "equal to x dtype");
        return ge::GRAPH_FAILED;
    }
    if (targetDesc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "target",
                                  ge::TypeUtils::DataTypeToSerialString(targetDesc->GetDataType()).c_str(), "INT32");
        return ge::GRAPH_FAILED;
    }
    ge::DataType isTgtDtype = isTargetDesc->GetDataType();
    if (isTgtDtype != ge::DT_INT32 && isTgtDtype != xDtype) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "is_target",
                                  ge::TypeUtils::DataTypeToSerialString(isTgtDtype).c_str(),
                                  "INT32 (GE graph) or equal to x dtype (aclnn)");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 按 C 缩放的 UB 开销(每行一份,与分块无关):
//   inputQueue(T) + targetQueue(int32) + isTargetOutQueue(IsTgtT) + xRowBuf/isPosBuf/reduceBuf/workBuf(float×4)
// 各 buffer 在 kernel 侧按 32B / 向量寄存器宽度对齐,这里按对齐后上界估。
// floatBufNum: 全行路径 4 个 float 行缓冲(xRowBuf/isPosBuf/reduceBuf/workBuf);
// C 分块路径多一个 isTgtFBuf(掩码段, 由 is_target 回读并转 float), 故传 5。
uint32_t PerRowUbBytes(uint32_t C, uint32_t tSize, uint32_t isTgtSize, uint32_t floatBufNum = 4U)
{
    const uint32_t FLOAT_BUF_NUM = floatBufNum;
    constexpr uint32_t VEC_REG_BYTES = 256U; // 向量寄存器宽度上界,float 路径按它对齐
    uint32_t rowT = CeilAlign(C * tSize, 32U);
    uint32_t rowI32 = CeilAlign(C * sizeof(int32_t), 32U);
    uint32_t rowIsTgt = CeilAlign(C * isTgtSize, 32U);
    uint32_t rowF = CeilAlign(C * sizeof(float), VEC_REG_BYTES);
    return rowT + rowI32 + rowIsTgt + rowF * FLOAT_BUF_NUM;
}

// 解析 shape 与 reduction 属性。rank<=2: (N,C), rank==1 视作 (1,C)。
ge::graphStatus ParseShapeAndReduction(gert::TilingContext* context, uint32_t& N, uint32_t& C, int32_t& reduction)
{
    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    size_t dimNum = xShape->GetStorageShape().GetDimNum();
    if (dimNum >= 2) {
        N = static_cast<uint32_t>(xShape->GetStorageShape().GetDim(0));
        C = static_cast<uint32_t>(xShape->GetStorageShape().GetDim(1));
    } else if (dimNum == 1) {
        N = 1;
        C = static_cast<uint32_t>(xShape->GetStorageShape().GetDim(0));
    }

    reduction = ParseReductionAttr(context);
    if (reduction == REDUCTION_INVALID) {
        OP_LOGE(context->GetNodeName(), "The reduction attribute must be 'none', 'mean', or 'sum'.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 跨核合并按整轮 MERGE_VL_FP32 车道读, 所以核0 回读 partial 的 UB 用量必须按整轮算
// (只按 usedCoreNum 个跨步槽算会少估, 且让 kernel 读出张量边界)。
uint32_t CalcPartialUbElems(uint32_t usedCoreNum)
{
    uint32_t partialUbElems = static_cast<uint32_t>(MERGE_VL_FP32) *
                              ((usedCoreNum * WS_CORE_STRIDE + static_cast<uint32_t>(MERGE_VL_FP32) - 1U) /
                               static_cast<uint32_t>(MERGE_VL_FP32));
    return (partialUbElems == 0U) ? static_cast<uint32_t>(MERGE_VL_FP32) : partialUbElems;
}

// x 与 is_target 的元素字节数, UB 记账用。
ge::graphStatus GetElemSizes(gert::TilingContext* context, uint32_t& tSize, uint32_t& isTgtSize)
{
    auto xDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    auto isTargetDesc = context->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, isTargetDesc);
    tSize = static_cast<uint32_t>(ge::GetSizeByDataType(xDesc->GetDataType()));
    isTgtSize = static_cast<uint32_t>(ge::GetSizeByDataType(isTargetDesc->GetDataType()));
    return ge::GRAPH_SUCCESS;
}

// 整行装不下就按 C 方向分块, 而不是拒收: 内核对 C 的支持面不应由"一行必须整条驻留 UB"决定
// (issue #32: A2 由 TBE DSL 承担切分, 对 C 无上限; 我方原实现在 C≈8000 就 GRAPH_FAILED)。
ge::graphStatus SolveCFactor(gert::TilingContext* context, uint64_t ubSize, uint32_t fixedBytes, uint32_t C,
                             uint32_t tSize, uint32_t isTgtSize, uint32_t perTileBytes, uint32_t& cFactor,
                             uint32_t& perRowBytes, uint64_t& used)
{
    constexpr uint32_t SPLIT_FLOAT_BUFS = 5U; // 分块路径的 float 行缓冲个数
    // 行方向分块 buffer 的单元素开销(rowLossBuf/gatherBuf/gatherOutBuf), 解 cFactor 时必须先给它留出
    // 至少 TILE_MIN 份, 否则 C 段把 UB 吃满 → 后面 ubFactor 解出 0, tiling 反而失败。
    uint64_t tileReserve = static_cast<uint64_t>(TILE_MIN) * perTileBytes;
    uint64_t fixedAll = static_cast<uint64_t>(fixedBytes) + UB_RESERVE_BYTES + tileReserve;
    uint64_t budget = (ubSize > fixedAll) ? (ubSize - fixedAll) : 0U;
    uint32_t perElem = tSize + static_cast<uint32_t>(sizeof(int32_t)) + isTgtSize +
                       SPLIT_FLOAT_BUFS * static_cast<uint32_t>(sizeof(float));
    cFactor = FloorAlign(static_cast<uint32_t>(budget / perElem), TILE_ALIGN);
    // 解析解未计对齐余量, 逐档回退到真正装得下为止
    while (cFactor >= TILE_MIN &&
           static_cast<uint64_t>(PerRowUbBytes(cFactor, tSize, isTgtSize, SPLIT_FLOAT_BUFS)) + fixedAll >= ubSize) {
        cFactor -= TILE_ALIGN;
    }
    if (cFactor < TILE_MIN) {
        OP_LOGE(context->GetNodeName(),
                "UB budget exhausted: even one tile of %u elements does not fit (fixed %u B + reserve %u B, "
                "UB %lu B, C=%u).",
                TILE_MIN, fixedBytes, UB_RESERVE_BYTES, static_cast<unsigned long>(ubSize), C);
        return ge::GRAPH_FAILED;
    }
    perRowBytes = PerRowUbBytes(cFactor, tSize, isTgtSize, SPLIT_FLOAT_BUFS);
    used = static_cast<uint64_t>(perRowBytes) + fixedBytes + UB_RESERVE_BYTES;
    OP_LOGI(context->GetNodeName(), "C=%u exceeds one-row UB budget; split C into tiles of %u elements.", C, cFactor);
    return ge::GRAPH_SUCCESS;
}

// 分块 buffer 的单元素开销:rowLossBuf(float) + gatherBuf(float) + gatherOutBuf(T)。
ge::graphStatus SolveUbFactor(gert::TilingContext* context, uint64_t ubSize, uint64_t used, uint32_t perTileBytes,
                              uint32_t N, uint32_t C, int32_t reduction, uint32_t& ubFactor)
{
    // 除零守卫: perTileBytes = 2*sizeof(float)+tSize 恒 > 0, used < ubSize 由调用方保证,
    // 这里显式挡一道, 免得后续改动把前提破坏了还静默算下去。
    OP_CHECK_IF(perTileBytes == 0U || used >= ubSize,
                OP_LOGE(context->GetNodeName(), "invalid UB budget: perTileBytes=%u, used=%lu, ubSize=%lu",
                        perTileBytes, static_cast<unsigned long>(used), static_cast<unsigned long>(ubSize)),
                return ge::GRAPH_FAILED);
    ubFactor = FloorAlign(static_cast<uint32_t>((ubSize - used) / perTileBytes), TILE_ALIGN);
    if (ubFactor > MAX_VEC_ELEMS) {
        ubFactor = MAX_VEC_ELEMS; // 已是 16 的倍数(255×64=16320)
    }
    if (ubFactor < TILE_MIN) {
        OP_LOGE(context->GetNodeName(), "UB too small for tiling: ubFactor=%u < %u (C=%u).", ubFactor, TILE_MIN, C);
        return ge::GRAPH_FAILED;
    }
    // 不超过实际所需:reduction=none 需要覆盖 N 行,mean/sum 只需 1 个元素。
    uint32_t needElems = (reduction == RED_NONE) ? ((N == 0U) ? 1U : N) : 1U;
    uint32_t needAligned = CeilAlign(needElems, TILE_ALIGN);
    if (needAligned < TILE_MIN) {
        needAligned = TILE_MIN;
    }
    if (ubFactor > needAligned) {
        ubFactor = needAligned;
    }
    return ge::GRAPH_SUCCESS;
}

// 写 tiling 数据与 workspace 大小。
// Float 工作区:reduction=none 每行一个槽;mean/sum 每核一个 32B 独占槽(不用原子加,
// 核0 按固定的 blockIdx 顺序 Kahan 合并 -> 结果可复现且更准)。
ge::graphStatus WriteTilingAndWorkspace(gert::TilingContext* context,
                                        const platform_ascendc::PlatformAscendC& ascendcPlatform, uint32_t N,
                                        uint32_t C, uint32_t usedCoreNum, int32_t reduction, uint32_t ubFactor,
                                        uint32_t cFactor, uint32_t partialUbElems)
{
    auto* tiling = context->GetTilingData<MultilabelMarginLossArch35TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    if (memset_s(tiling, sizeof(MultilabelMarginLossArch35TilingData), 0,
                 sizeof(MultilabelMarginLossArch35TilingData)) != EOK) {
        OP_LOGE(context->GetNodeName(), "memset_s tiling data failed.");
        return ge::GRAPH_FAILED;
    }
    // 除零守卫: usedCoreNum 由入口按 (N==0)?1:min(N,coreNum) 算出恒 >= 1, 显式挡一道。
    OP_CHECK_IF(usedCoreNum == 0U, OP_LOGE(context->GetNodeName(), "usedCoreNum is 0"), return ge::GRAPH_FAILED);
    tiling->N = N;
    tiling->C = C;
    tiling->basePerCore = N / usedCoreNum;
    tiling->pivot = N % usedCoreNum;
    tiling->usedCoreNum = usedCoreNum;
    tiling->reduction = reduction;
    tiling->ubFactor = ubFactor;
    tiling->wsCoreStride = WS_CORE_STRIDE;
    tiling->partialUbElems = partialUbElems;
    tiling->cFactor = cFactor;

    uint32_t wsElems = (reduction == RED_NONE) ? ((N == 0U) ? 1U : N) : (usedCoreNum * WS_CORE_STRIDE);
    size_t accBytes = ((static_cast<size_t>(wsElems) * sizeof(float) + 31U) / 32U) * 32U;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = accBytes + ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
} // namespace

// A5 tiling 入口。根 tiling 在 regbase 分支上调用本函数(前置声明见根文件)。
ge::graphStatus DoMultilabelMarginLossTiling950(gert::TilingContext* context)
{
    if (CheckDtypeValid(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint32_t N = 1;
    uint32_t C = 1;
    int32_t reduction = REDUCTION_INVALID;
    if (ParseShapeAndReduction(context, N, C, reduction) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0U) {
        coreNum = 1U;
    }

    // 空 batch(N==0):保持 N==0 让 kernel 处理零行,但用一个 block 维持合法 grid。
    uint32_t usedCoreNum = (N == 0U) ? 1U : ((N < coreNum) ? N : coreNum);
    context->SetBlockDim(usedCoreNum);
    // kernel 用 SyncAll 做跨核归约,batch 模式让所有核同时启动,否则 SyncAll 可能概率性死锁。
    context->SetScheduleMode(BATCH_MODE);

    // ---- UB 预算:host 侧实算 ubFactor,kernel 不再自行推导任何 buffer 尺寸 ----
    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0U) {
        OP_LOGE(context->GetNodeName(), "Failed to get UB size from platform.");
        return ge::GRAPH_FAILED;
    }

    uint32_t tSize = 0U;
    uint32_t isTgtSize = 0U;
    if (GetElemSizes(context, tSize, isTgtSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    uint32_t partialUbElems = CalcPartialUbElems(usedCoreNum);
    uint32_t fixedBytes = partialUbElems * static_cast<uint32_t>(sizeof(float)) + 32U;
    uint32_t perRowBytes = PerRowUbBytes(C, tSize, isTgtSize);
    uint32_t perTileBytes = static_cast<uint32_t>(sizeof(float)) * 2U + tSize;

    uint64_t used = static_cast<uint64_t>(perRowBytes) + fixedBytes + UB_RESERVE_BYTES;
    uint32_t cFactor = C;
    if (used >= ubSize && SolveCFactor(context, ubSize, fixedBytes, C, tSize, isTgtSize, perTileBytes, cFactor,
                                       perRowBytes, used) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint32_t ubFactor = 0U;
    if (SolveUbFactor(context, ubSize, used, perTileBytes, N, C, reduction, ubFactor) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return WriteTilingAndWorkspace(context, ascendcPlatform, N, C, usedCoreNum, reduction, ubFactor, cFactor,
                                   partialUbElems);
}

struct MultilabelMarginLossCompileInfo {};

// 本文件只在 ascend950 编译(COMPUTE_UNIT ascend950 + TILING_DIR arch35),直接走 A5 实算,
// 不做 soc 分派。A2(ascend910b 等)不编译本仓 tiling,回归 canndev 自带 A2 tiling。
// 参照 loss/ctc_loss_v2 的组织方式:arch35 文件自包含 A5 实现 + 注册,无分派、无 A2 回退。
static ge::graphStatus MultilabelMarginLossTilingFunc(gert::TilingContext* context)
{
    return DoMultilabelMarginLossTiling950(context);
}

static ge::graphStatus TilingParseForMultilabelMarginLoss([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MultilabelMarginLoss)
    .Tiling(MultilabelMarginLossTilingFunc)
    .TilingParse<MultilabelMarginLossCompileInfo>(TilingParseForMultilabelMarginLoss);
} // namespace optiling
