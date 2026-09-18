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
 * \file scatter_reduce_common_tiling.cpp
 * \brief Shared tiling for non-nd scatter reduce. Splits index entries across cores; per-index slices
 *        stay on one core (cross-core writes to the same var row are handled by the atomic kernel).
 */
#include "scatter_reduce_common_tiling.h"
#include "../../op_kernel/arch35/scatter_reduce_common_struct.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "tiling/tiling_api.h"

namespace optiling {

constexpr size_t VAR_IDX = 0;
constexpr size_t INDICES_IDX = 1;
constexpr size_t UPDATES_IDX = 2;
constexpr uint64_t SYS_WORKSPACE = 16UL * 1024UL * 1024UL;

ge::graphStatus TilingPrepareForScatterReduce(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<ScatterReduceCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    return ge::GRAPH_SUCCESS;
}

// coreNum 来源：kernel 路径用解析好的 compileInfo；aclnn runtime 不带 compileInfo，回退直接读 platform。
static ge::graphStatus ResolveCoreNum(gert::TilingContext* context, uint64_t& coreNum)
{
    coreNum = 1;
    auto compileInfo = reinterpret_cast<const ScatterReduceCompileInfo*>(context->GetCompileInfo());
    if (compileInfo != nullptr) {
        coreNum = (compileInfo->coreNum == 0) ? 1 : compileInfo->coreNum;
        return ge::GRAPH_SUCCESS;
    }
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = 1;
    }
    return ge::GRAPH_SUCCESS;
}

// Input validation for the A5 graph path (aclnn path validates in CheckParams; the arch35 graph tiling
// must independently enforce the same input contract). Mirrors the mature scatter_add tiling checks:
//   1) var and updates must share the same dtype (the kernel reads both as one element type);
//   2) updates.shape == indices.shape + var.shape[1:].
// Empty tensors pass structurally (indicesNum==0 / varFirstDim==0 are handled as no-op by the kernel).
static ge::graphStatus CheckScatterReduceShapes(gert::TilingContext* context)
{
    const char* opName = context->GetNodeName();
    auto varShapePtr = context->GetInputShape(VAR_IDX);
    auto indicesShapePtr = context->GetInputShape(INDICES_IDX);
    auto updatesShapePtr = context->GetInputShape(UPDATES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, updatesShapePtr);
    const auto& varShape = varShapePtr->GetStorageShape();
    const auto& indicesShape = indicesShapePtr->GetStorageShape();
    const auto& updatesShape = updatesShapePtr->GetStorageShape();
    uint64_t varDimNum = static_cast<uint64_t>(varShape.GetDimNum());
    uint64_t indicesDimNum = static_cast<uint64_t>(indicesShape.GetDimNum());
    uint64_t updatesDimNum = static_cast<uint64_t>(updatesShape.GetDimNum());
    // 各输入维度上限校验（对齐 A2 para_check 的 max_rank=8）。aclnn 框架仅对影响输出的张量在其路径拦截，
    // 此处在算子侧补齐，兼顾 GE 图路径。
    constexpr uint64_t MAX_SUPPORT_DIM = 8;
    OP_CHECK_IF(
        varDimNum > MAX_SUPPORT_DIM || indicesDimNum > MAX_SUPPORT_DIM || updatesDimNum > MAX_SUPPORT_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            opName, "var, indices, updates",
            (std::to_string(varDimNum) + ", " + std::to_string(indicesDimNum) + ", " + std::to_string(updatesDimNum)),
            "the dim number of var, indices and updates must be less than or equal to 8"),
        return ge::GRAPH_FAILED);
    if (varDimNum == 0) {
        return ge::GRAPH_SUCCESS; // scalar var is degenerate; keep permissive (kernel treats varFirstDim as 1)
    }
    if (updatesDimNum != indicesDimNum + varDimNum - 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, "updates", std::to_string(updatesDimNum).c_str(),
                                                 "updatesDimNum must equal indicesDimNum + varDimNum - 1");
        return ge::GRAPH_FAILED;
    }
    for (uint64_t i = 0; i < indicesDimNum; i++) {
        if (updatesShape.GetDim(i) != indicesShape.GetDim(i)) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                opName, "updates, indices",
                (std::to_string(updatesShape.GetDim(i)) + ", " + std::to_string(indicesShape.GetDim(i))).c_str(),
                "updatesShape should equal indicesShape in the first indicesDimNum dimensions");
            return ge::GRAPH_FAILED;
        }
    }
    for (uint64_t i = 1; i < varDimNum; i++) {
        if (updatesShape.GetDim(i + indicesDimNum - 1) != varShape.GetDim(i)) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                opName, "updates, var",
                (std::to_string(updatesShape.GetDim(i + indicesDimNum - 1)) + ", " + std::to_string(varShape.GetDim(i)))
                    .c_str(),
                "updatesShape should equal varShape except for the first dimension");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckScatterReduceInputs(gert::TilingContext* context)
{
    const char* opName = context->GetNodeName();
    auto varDesc = context->GetInputDesc(VAR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, varDesc);
    auto updatesDesc = context->GetInputDesc(UPDATES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, updatesDesc);
    auto varDtype = varDesc->GetDataType();
    auto updatesDtype = updatesDesc->GetDataType();
    if (updatesDtype != varDtype) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName, "updates, var",
                                               (ge::TypeUtils::DataTypeToSerialString(updatesDtype) + ", " +
                                                ge::TypeUtils::DataTypeToSerialString(varDtype))
                                                   .c_str(),
                                               "expected updates dtype to be equal to var dtype");
        return ge::GRAPH_FAILED;
    }
    return CheckScatterReduceShapes(context);
}

// 按 UB 实测容量反解 phase-3 的列切分上限。
// 每列存活的 UB: accUb + rowAccUb(2 份 ACC) + varUb + updQue 深度 8(共 9 份 PARAMS_T)
// + tmp(int32) + MUL 浮点路径 Mul 前暂存掩码的 preMask(每列不足 1B, 按 1B 计)。
// phase-1 的排序缓冲在 pipe_.Reset() 后已释放, phase-3 可用整块 UB; 只需扣掉每个 buffer
// 各自向上按 block 对齐的损失。列上限由容量反解, 而非先定死再判超限 —— sliceSize 再大
// 也只是多切几个 chunk, 不存在因某轴过大而不支持。
static uint64_t ResolveUbChunkMax(const gert::TilingContext* context)
{
    constexpr uint64_t BLOCK_BYTES = static_cast<uint64_t>(ScatterReduceCommon::UB_BLOCK_BYTES);
    constexpr uint64_t ACC_BUF_NUM = 2UL;                   // accUb, rowAccUb
    constexpr uint64_t PARAM_BUF_NUM = 9UL;                 // varUb + updQue 深度 8
    constexpr uint64_t TMP_BUF_BYTES = sizeof(int32_t);     // tmp 固定为 int32
    constexpr uint64_t ACC_WIDEN_BYTES = sizeof(float);     // 见 kernel 侧 AccT: 窄类型提升到 float
    constexpr uint64_t SUBWORD_BYTES_MAX = sizeof(int16_t); // 2B 及以下视为窄类型
    constexpr uint64_t BUF_COUNT = 5UL;                     // InitBuffer 次数, 与 kernel 侧一致

    uint64_t ubSize = 0;
    platform_ascendc::PlatformAscendC(context->GetPlatformInfo())
        .GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto varDesc = context->GetInputDesc(0);
    // GetSizeByDataType 返回 int, 未知 dtype 时返回 -1; 直接转 uint64 会回绕成 2^64-1,
    // 后面按列字节数反解容量会整体失真。非正返回按窄类型兜底(与 varDesc 缺失同一处理),
    // dtype 合法性本身由 CheckScatterReduceInputs 把关。
    const int32_t dtypeSize = (varDesc == nullptr) ? -1 : ge::GetSizeByDataType(varDesc->GetDataType());
    const uint64_t dtypeBytes = (dtypeSize <= 0) ? ACC_WIDEN_BYTES : static_cast<uint64_t>(dtypeSize);
    const uint64_t accBytes = (dtypeBytes <= SUBWORD_BYTES_MAX) ? ACC_WIDEN_BYTES : dtypeBytes;
    const uint64_t colBytes = ACC_BUF_NUM * accBytes + PARAM_BUF_NUM * dtypeBytes + TMP_BUF_BYTES;
    const uint64_t alignReserve = BUF_COUNT * BLOCK_BYTES;
    const uint64_t usable = (ubSize > alignReserve) ? (ubSize - alignReserve) : 0UL;
    const uint64_t chunkMax = (colBytes == 0UL) ? 0UL : usable / colBytes / BLOCK_BYTES * BLOCK_BYTES;
    return (chunkMax < BLOCK_BYTES) ? BLOCK_BYTES : chunkMax; // 保底一个对齐粒度
}

// 排序分片按 UB **实测容量反解**, 不写死: 每个元素在 UB 上同时占 key + sorted + 载荷(uint32)
// + 全局位置(int64) 四份; bSorted 另有 shiftOff 个元素的头部余量, 每个 InitBuffer 留 32B 对齐。
// 写死分片会退化成"手算 buffer -> 跟 UB 比 -> 不够就拍个更小的数", 且要 host/kernel 两处同步。
static uint64_t ResolveSortTile(const gert::TilingContext* context, uint64_t keyBytes)
{
    constexpr uint64_t BLOCK_BYTES = static_cast<uint64_t>(ScatterReduceCommon::UB_BLOCK_BYTES);
    constexpr uint64_t BUF_COUNT = 4UL;              // bKey / bSorted / bOrigin / bPos64
    constexpr uint64_t SORT_PAD = 2UL * BLOCK_BYTES; // 与 kernel 侧 SORT_PAD 一致
    uint64_t ubSize = 0;
    platform_ascendc::PlatformAscendC(context->GetPlatformInfo())
        .GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint64_t elemBytes = keyBytes * 2UL + sizeof(uint32_t) + sizeof(int64_t);
    const uint64_t reserve = BUF_COUNT * SORT_PAD + BLOCK_BYTES;
    const uint64_t usable = (ubSize > reserve) ? (ubSize - reserve) : 0UL;
    // AscendC::Sort 对 calCount **没有上限**(接口只要求 32B 对齐、张量不重叠), 其临时缓冲尺寸由
    // GetSortTmpSize 给出。所以分片大小完全由 UB 容量决定, 不存在什么"单次调用经验上限":
    // 先按不含 tmp 的估计起步, 再把 tmp 算进去逐步收缩, 直到真正装得下。
    uint64_t tile = (elemBytes == 0UL) ? BLOCK_BYTES : usable / elemBytes / BLOCK_BYTES * BLOCK_BYTES;
    platform_ascendc::PlatformAscendC plat(context->GetPlatformInfo());
    while (tile > BLOCK_BYTES) {
        const uint64_t tmpBytes = static_cast<uint64_t>(
            AscendC::GetSortTmpSize(plat, static_cast<uint32_t>(tile), static_cast<uint32_t>(keyBytes)));
        if (tile * elemBytes + tmpBytes + reserve <= ubSize) {
            break;
        }
        const uint64_t next = tile * 3UL / 4UL / BLOCK_BYTES * BLOCK_BYTES;
        tile = (next >= tile) ? (tile - BLOCK_BYTES) : next; // 严格递减, 循环必然终止
    }
    return (tile < BLOCK_BYTES) ? BLOCK_BYTES : tile; // 保底一个对齐粒度: 装不下就继续切, 不拒收
}

// 单核标量路径的选路判定: 该路径把 indices/updates/var 整体放进 UB(见 scatter_reduce_common_scalar.h
// 的 idxBuf/updBuf/varBuf/accBuf), 因此判据必须是"这几块真的放得下", 而不是拿 (M + dim0) 去跟一个
// 写死的数比 —— 那样既不管 dtype 宽度, 也把某台机器的 UB 焊进了算子。放不下就走 sort 路径, 不拒收。
static bool ResolveScalarPath(const gert::TilingContext* context, uint64_t indicesNum, uint64_t varFirstDim,
                              uint64_t keyBytes)
{
    constexpr uint64_t BLOCK_BYTES = static_cast<uint64_t>(ScatterReduceCommon::UB_BLOCK_BYTES);
    constexpr uint64_t BUF_COUNT = 4UL;           // idxBuf / updBuf / varBuf / accBuf
    constexpr uint64_t ACC_BYTES = sizeof(float); // 窄 dtype 会额外开 float 累加缓冲, 按最坏情形计
    uint64_t ubSize = 0;
    platform_ascendc::PlatformAscendC(context->GetPlatformInfo())
        .GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto varDesc = context->GetInputDesc(VAR_IDX);
    const int32_t varSzI = (varDesc == nullptr) ? -1 : ge::GetSizeByDataType(varDesc->GetDataType());
    const uint64_t varSz = (varSzI <= 0) ? ACC_BYTES : static_cast<uint64_t>(varSzI);
    const uint64_t need = indicesNum * (keyBytes + varSz) + varFirstDim * (varSz + ACC_BYTES) + BUF_COUNT * BLOCK_BYTES;
    return need <= ubSize;
}

ge::graphStatus ScatterReduceCommonTiling(gert::TilingContext* context)
{
    if (CheckScatterReduceInputs(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto varShapePtr = context->GetInputShape(VAR_IDX);
    auto indicesShapePtr = context->GetInputShape(INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShapePtr);
    auto& varShape = varShapePtr->GetStorageShape();
    auto& indicesShape = indicesShapePtr->GetStorageShape();

    uint64_t varFirstDim = (varShape.GetDimNum() == 0) ? 1 : varShape.GetDim(0);
    // 排序 key 直接用索引真值(AscendC::Sort 的 key 支持 int64), 不再折算低位 + 第二趟分区。
    // 因此 var 首维没有任何上限, 与 A2(910B) 一致: 不分桶, 也就没有桶数带来的边界。
    uint64_t varTotal = varShape.GetShapeSize();
    uint64_t sliceSize = (varFirstDim == 0) ? 0 : varTotal / varFirstDim;
    uint64_t indicesNum = indicesShape.GetShapeSize();

    // kernel 侧 Init 的布局: sortedIdx[mAlign x keySz] | originPos[mAlign x 4]
    //                        | scratchKeys[mAlign x keySz] | scratchPos[mAlign x 4] | partials
    // key 是索引真值, 两个 key 区按索引 dtype 宽度计 —— 必须与 kernel 同源, 否则区间重叠。
    auto indicesDesc = context->GetInputDesc(INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesDesc);
    const int32_t keySzI = ge::GetSizeByDataType(indicesDesc->GetDataType());
    OP_CHECK_IF(keySzI <= 0, OP_LOGE(context->GetNodeName(), "get indices dtype size fail."), return ge::GRAPH_FAILED);
    const uint64_t keySz = static_cast<uint64_t>(keySzI);
    const uint64_t sortTile = ResolveSortTile(context, keySz);

    uint64_t coreNum = 1;
    if (ResolveCoreNum(context, coreNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // split index entries across cores; each core handles whole slices, read directly from GM in the VF
    uint64_t blockNum = (indicesNum == 0) ? 1 : (indicesNum < coreNum ? indicesNum : coreNum);
    uint64_t perCoreIndices = (blockNum == 0) ? 0 : (indicesNum + blockNum - 1) / blockNum;
    blockNum = (perCoreIndices == 0) ? 1 : (indicesNum + perCoreIndices - 1) / perCoreIndices;
    uint64_t tailCoreIndices = indicesNum - (blockNum - 1) * perCoreIndices;

    auto* td = context->GetTilingData<ScatterReduceCommon::ScatterReduceSimtTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);
    td->blockNum = blockNum;
    td->blockTilingSize = perCoreIndices * sliceSize;
    td->tailBlockTilingSize = tailCoreIndices * sliceSize;
    td->sliceSize = sliceSize;
    td->varFirstDim = varFirstDim;
    td->scalarPath = ResolveScalarPath(context, indicesNum, varFirstDim, keySz) ? 1UL : 0UL;
    td->sortTile = sortTile;
    td->ubChunkMax = ResolveUbChunkMax(context);

    context->SetBlockDim(blockNum);
    context->SetTilingKey(0);
    // user scratch for the sort-based MUL path: sortedIdx + originPos + scratchKeys + scratchPos (per index entry)
    // + per-core partial slots. The kernel's Init allocates these using mAlign = (M + 7)/8*8 + 128 for padding
    // and capacity margin on the parallel merge-sort. Must match kernel's layout exactly to avoid overlap.
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    // per-core partial slot stride. Host uses sliceSize; kernel uses sAlign = align8(min(sliceSize, CHUNK_MAX)),
    // so when sliceSize > CHUNK_MAX the host stride >= kernel stride -- this OVER-allocates (never under), which
    // is safe. Keep sliceSize here (not CHUNK) so host can never under-allocate the partial region.
    uint64_t sliceAlign = (sliceSize + 7UL) / 8UL * 8UL;
    // Each sort buffer must hold padM = P2 * runLen0 (the merge-sort pads every run to runLen0), NOT just
    // indicesNum. P2 is raised past coreNum until runLen0 <= sortTile, so padM - indicesNum can reach up to
    // P2-1 (thousands at large M) -- far beyond the old fixed +128 margin, which under-allocated and let the
    // sort write out of bounds (VEC_ERROR) for indicesNum above ~1M. Replicate the kernel's exact P2/runLen0
    // (scatter_reduce_common_sort.h SortIndices) so the host sizes each region to the true padM.
    uint64_t padM = indicesNum;
    if (indicesNum > sortTile) {
        uint64_t p2 = 1UL;
        while (p2 * 2UL <= blockNum) {
            p2 *= 2UL;
        }
        while ((indicesNum + p2 - 1UL) / p2 > sortTile) {
            p2 *= 2UL;
        }
        uint64_t runLen0 = (indicesNum + p2 - 1UL) / p2;
        padM = p2 * runLen0;
    }
    uint64_t mAlign = ((padM + 7UL) / 8UL * 8UL) + 128UL; // 8B align + 128B margin on top of the true padM
    // kernel layout: sortedIdx[mAlign] | originPos[mAlign] | scratchKeys[mAlign] | scratchPos[mAlign] |
    // partials[blockNum][sAlign]
    workspaces[0] = SYS_WORKSPACE + mAlign * 2UL * (keySz + 8UL) +
                    static_cast<uint64_t>(blockNum) * sliceAlign * 4UL; // per-core partial slots
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
