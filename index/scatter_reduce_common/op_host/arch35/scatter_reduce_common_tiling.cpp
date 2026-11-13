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
// 排序 scratch 的固定预算(字节)。排序只是加速手段, 不允许它决定算子支持的规模上限:
// 超过一片就切分, workspace 恒定。1GiB 下单片约 3.3e7 条(每条目 32B)。
constexpr uint64_t WS_SORT_BUDGET = 1UL << 30;
// 排序后每条位置记录的字节数(内核 originPosGm_ 为 int64), 用于并行轴的搬运量模型。
constexpr uint64_t POS_ENTRY_BYTES = sizeof(int64_t);

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

// ascend950 图路径的入参校验(aclnn 侧在 CheckParams 另有一套, 此处独立成立同一契约):
// var/updates 同 dtype、updates.shape == indices.shape + var.shape[1:]、维数上限 8。
static ge::graphStatus CheckShapeDimNumLimit(const char* opName, uint64_t varDimNum, uint64_t indicesDimNum,
                                             uint64_t updatesDimNum)
{
    constexpr uint64_t MAX_SUPPORT_DIM = 8;
    OP_CHECK_IF(
        varDimNum > MAX_SUPPORT_DIM || indicesDimNum > MAX_SUPPORT_DIM || updatesDimNum > MAX_SUPPORT_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            opName, "var, indices, updates",
            (std::to_string(varDimNum) + ", " + std::to_string(indicesDimNum) + ", " + std::to_string(updatesDimNum)),
            "the dim number of var, indices and updates must be less than or equal to 8"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckShapeRankEquality(const char* opName, const gert::Shape& varShape,
                                              const gert::Shape& indicesShape, const gert::Shape& updatesShape)
{
    const uint64_t varDimNum = static_cast<uint64_t>(varShape.GetDimNum());
    const uint64_t updatesDimNum = static_cast<uint64_t>(updatesShape.GetDimNum());
    uint64_t indicesDimNum = static_cast<uint64_t>(indicesShape.GetDimNum());
    if (indicesDimNum == 1 && indicesShape.GetDim(0) == 1 && varDimNum - updatesDimNum == 1) {
        indicesDimNum = 0;
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

// var 是原地写回目标, 无法经 AutoContiguous 物化。stride 非连续(含 stride-0 广播)超出 kernel
// 的连续布局假设会静默写错, 故在 tiling 层拒绝; 偏移但 stride 连续的视图不受影响。
static ge::graphStatus CheckVarRefContiguous(gert::TilingContext* context, const char* opName,
                                             const gert::Shape& varShape)
{
    auto varViewStride = context->GetInputStride(VAR_IDX);
    if (context->InputIsView(VAR_IDX) && varViewStride != nullptr && varViewStride->GetDimNum() != 0) {
        int64_t expectedStride = 1;
        for (int64_t dim = static_cast<int64_t>(varShape.GetDimNum()) - 1; dim >= 0; --dim) {
            const int64_t dimSize = static_cast<int64_t>(varShape.GetDim(dim));
            if (dimSize > 1 && varViewStride->GetStride(dim) != expectedStride) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "var", "broadcast or non-contiguous strides",
                                                      "var must have contiguous strides (in-place update target)");
                return ge::GRAPH_FAILED;
            }
            expectedStride *= (dimSize > 0 ? dimSize : 1);
        }
    }
    return ge::GRAPH_SUCCESS;
}

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
    const uint64_t varDimNum = static_cast<uint64_t>(varShape.GetDimNum());
    // 校验顺序即语义顺序, 不可重排: 维数上限 -> 0 维早退 -> 维值上限 -> 秩等式 -> varRef stride。
    // 各 helper 内部已打印具体拒绝原因(含维值/形状), 此处的 OP_LOGE 只标出失败环节。
    OP_CHECK_IF(CheckShapeDimNumLimit(opName, varDimNum, static_cast<uint64_t>(indicesShape.GetDimNum()),
                                      static_cast<uint64_t>(updatesShape.GetDimNum())) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName, "CheckShapeDimNumLimit failed"), return ge::GRAPH_FAILED);
    if (varDimNum == 0) {
        return ge::GRAPH_SUCCESS; // scalar var is degenerate; keep permissive (kernel treats varFirstDim as 1)
    }
    OP_CHECK_IF(CheckShapeRankEquality(opName, varShape, indicesShape, updatesShape) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName, "CheckShapeRankEquality failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckVarRefContiguous(context, opName, varShape) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName, "CheckVarRefContiguous failed"), return ge::GRAPH_FAILED);
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
static uint64_t ResolveUbChunkMax(const gert::TilingContext* context, uint64_t mode)
{
    constexpr uint64_t BLOCK_BYTES = static_cast<uint64_t>(ScatterReduceCommon::UB_BLOCK_BYTES);
    constexpr uint64_t ACC_WIDEN_BYTES = sizeof(float);     // 见 kernel 侧 AccT: 窄类型提升到 float
    constexpr uint64_t SUBWORD_BYTES_MAX = sizeof(int16_t); // 2B 及以下视为窄类型
    constexpr uint64_t BUF_COUNT = 5UL; // InitBuffer 次数上限, 每块留一个 block 的对齐余量

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
    const bool isFloat = (varDesc != nullptr) &&
                         ((varDesc->GetDataType() == ge::DT_FLOAT) || (varDesc->GetDataType() == ge::DT_FLOAT16) ||
                          (varDesc->GetDataType() == ge::DT_BF16));
    const uint64_t colBytes = ScatterReduceCommon::SortFoldPlan(dtypeBytes, accBytes, mode, isFloat).colBytes;
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

// 原子路径的 UB 规格反解。内核此前写死 ATOMIC_TILE=4096 / SLOT_NUM=256 / 两个 KB 级预算常量 ——
// 那既不管 dtype 宽度, 也把某台机器的 UB 焊进算子, 与 ResolveScalarPath/ResolveSortTile 的做法相悖。
// 这里按"这几块缓冲真的放得下"反解: idx[tile] + updates[tile*slice] + 槽[slots*slotSpan] + pos[tile]
// 其中槽宽 slotSpan 向上取整到 granule(原子 DMA 的 UB 源地址需 32B 对齐)。
// 放不下就把 tile 折半、槽数折半, 直到装下; 保底各留一份, 绝不因装不下而拒收。
//
// 🔴 uint8 走 x^0x80 保序映射时, **账本里还有两项不能漏**(漏了就是 L1_105 那次整卡 aivec UB 越界):
//   1) FlipSignBitPerByte 的三步合成需要两块与 updates 等宽的临时缓冲 xfA/xfB ⇒ updates 项要乘 3;
//   2) FlipVarInPlace 内部自开三块 XF_TILE=16KB 的缓冲(共 48KB), 且它在主缓冲之后才 InitBuffer,
//      所以这 48KB 是实打实叠加在上面所有项之上的。
// 当时 ss=11 + int64 索引算出来正好溢出 4KB(ss<=10 才装得下), 于是 ss=8/ss=7 的用例全过、
// 只有 ss=11 崩 —— 这种"刚好卡在某个 ss 上"的现象就是 UB 账本漏项的指纹。
struct AtomicUbSpec {
    uint64_t tile;
    uint64_t slots;
    // ↓ 与上面两项同源算出的每块缓冲字节数, 原样下发给内核的 InitBuffer。
    // 内核不得再自己乘一遍 —— 那就是"一笔账两套算式"(G2-kernel-initbuffer-selfsized)。
    uint64_t idxBytes;
    uint64_t updBytes;
    uint64_t posBytes;
    uint64_t slotSpan;
};

// 原子路径的 tile/slots 起始预算: 按 UB 实测逐级折半收缩, 谁占得多先削谁(保持两者比例合理)
static constexpr uint64_t ATOMIC_TILE_INIT = 4096UL;
static constexpr uint64_t ATOMIC_SLOTS_INIT = 256UL;

static void ShrinkTileAndSlots(uint64_t ubSize, uint64_t perTile, uint64_t slotBytes, uint64_t reserve, uint64_t& tile,
                               uint64_t& slots)
{
    while (tile > 1UL || slots > 1UL) {
        if (tile * perTile + slots * slotBytes + reserve <= ubSize) {
            break;
        }
        if (slots > 1UL && slots * slotBytes > tile * perTile) {
            slots /= 2UL;
        } else if (tile > 1UL) {
            tile /= 2UL;
        } else {
            slots /= 2UL;
        }
    }
    tile = (tile == 0UL) ? 1UL : tile;
    slots = (slots == 0UL) ? 1UL : slots;
}

static AtomicUbSpec ResolveAtomicUbSpec(const gert::TilingContext* context, uint64_t sliceSize, uint64_t keyBytes,
                                        uint64_t varBytes, bool xform)
{
    constexpr uint64_t BLK = static_cast<uint64_t>(ScatterReduceCommon::UB_BLOCK_BYTES);
    // 三块 XF_FLIP_TILE_BYTES: 与内核共用 struct.h 的那个常量, 不再各写一份字面量
    constexpr uint64_t XF_FLIP_RESERVE = 3UL * static_cast<uint64_t>(ScatterReduceCommon::XF_FLIP_TILE_BYTES);
    uint64_t ubSize = 0;
    platform_ascendc::PlatformAscendC(context->GetPlatformInfo())
        .GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint64_t granElems = (varBytes == 0UL) ? 1UL : BLK / varBytes;
    const uint64_t slotSpan = (sliceSize + 2UL * granElems + granElems - 1UL) / granElems * granElems;
    const uint64_t slotBytes = slotSpan * varBytes;
    // updates 份数: 普通 dtype 只有 updUb 一份; uint8 另有 xfA/xfB 两块等宽临时缓冲。
    const uint64_t updCopies = xform ? 3UL : 1UL;
    const uint64_t perTile = keyBytes + updCopies * sliceSize * varBytes + sizeof(int32_t);
    const uint64_t reserve = 8UL * BLK + (xform ? XF_FLIP_RESERVE : 0UL); // 对齐余量 + 翻转缓冲
    uint64_t tile = ATOMIC_TILE_INIT;
    uint64_t slots = ATOMIC_SLOTS_INIT;
    ShrinkTileAndSlots(ubSize, perTile, slotBytes, reserve, tile, slots);
    // 连"一个 tile 元素 + 一个槽"都装不下(切片过宽) ⇒ slots=0 作为"原子路径不可用"的信号,
    // 内核见 slots==0 即返回 false 回落排序。判定留在 host 是因为只有这里知道真实 UB,
    // 内核侧原先那个写死的 SLOT_BUDGET 门槛(32KB)既不是真 UB 也不含上面两项, 已删除。
    if (tile * perTile + slots * slotBytes + reserve > ubSize) {
        return AtomicUbSpec{tile, 0UL, tile * keyBytes, tile * sliceSize * varBytes, tile * sizeof(int32_t), slotSpan};
    }
    // 每块缓冲的字节数由与预算同一组中间量算出: idx=tile*keyBytes、upd=tile*sliceSize*varBytes、
    // pos=tile*sizeof(int32_t)。uint8 的 xfA/xfB 与 upd 等宽(预算里体现为 updCopies=3), 内核按
    // updBytes 开那两块即可, 不必另发字段。
    return AtomicUbSpec{tile, slots, tile * keyBytes, tile * sliceSize * varBytes, tile * sizeof(int32_t), slotSpan};
}

// 分桶路径的桶内行数反解: 阶段3 要把"一个桶覆盖的 var 行窗口"整段读进 UB 折叠后连续写回,
// 所以桶内行数 = UB 可用窗口 / 每行字节。行数越大桶越少、游标竞争越小, 故取 UB 允许的最大值。
// 分桶路径的 UB 可用窗口(字节)。阶段4 同时需要 var 窗口与该桶的 (row,value) 对缓冲, 各给一半。
// bucketRows 与下发给内核的 ubUsable **必须共用这一个算式** —— 内核不能拿取整后的 bucketRows 反推。
static uint64_t ResolveUbUsable(const gert::TilingContext* context)
{
    constexpr uint64_t BLK = static_cast<uint64_t>(ScatterReduceCommon::UB_BLOCK_BYTES);
    uint64_t ubSize = 0;
    platform_ascendc::PlatformAscendC(context->GetPlatformInfo())
        .GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    return (ubSize > 16UL * BLK) ? (ubSize - 16UL * BLK) / 2UL : 0UL;
}

static uint64_t ResolveBucketRows(const gert::TilingContext* context, uint64_t sliceSize, uint64_t varBytes)
{
    const uint64_t rowBytes = sliceSize * varBytes;
    if (rowBytes == 0UL) {
        return 0UL;
    }
    const uint64_t usable = ResolveUbUsable(context);
    uint64_t rows = usable / rowBytes;
    if (rows == 0UL) {
        return 0UL;
    }
    // **向下取到 2 的幂**: 内核要向量化地由行号算桶号(bucket = row / bucketRows)。2 的幂使这步退化为
    // 一次 ShiftRights(Reg 层有移位、无整除), 于是"算桶号"进得了 regbase 的向量通路; 否则只能逐命中
    // 标量整除。代价是窗口最多浪费不到一半行数, 换来的是把 O(命中数) 的标量工作搬进向量。
    uint64_t pow2 = 1UL;
    while (pow2 * 2UL <= rows) {
        pow2 *= 2UL;
    }
    rows = pow2;
    return rows;
}

// 选路判定。入口条件只有两类: host 侧的**容量判定**与算子**语义/结构**条件, 不含任何经验阈值,
// 也不含"命中密度"之类运行期才知道的量 —— 后者曾被我拿实测曲线拟过一个 1/8 的阈值, 已撤。
//
// 之所以能不要密度判据: 分桶路径在两端都不差。阶段3 只处理 count>0 的桶, 命中为 0 时整个阶段
// 空转, 开销退化成"扫描一遍", 与原子路径等价; 命中密集时写回按行区间连续且无原子, 正是该档所需。
// 又因为折叠阶段"同一 var 行只属一个核", 不需要数据 dtype 有硬件原子(只有 int32 计数器用 AtomicAdd),
// 所以 uint8/int64 等无原子 dtype 也能走, 不必再做保序映射。
static uint64_t ResolvePathId(const gert::TilingContext* context, bool scalarPath, uint64_t sliceSize,
                              uint64_t varBytes, uint64_t bucketRows)
{
    (void)context;
    (void)varBytes;
    if (scalarPath && sliceSize == 1UL) {
        return 0UL; // 单核 scalar: 整个 var+indices 装得进单核 UB(容量判定)
    }
    // ⚠️ 分桶内核已接线(2026-10-28), 但选路**依然在内核侧**: BucketReduce 要到运行期才知道桶数/
    // 暂存深度/tile/workspace 是否都装得下, host 复刻不出同一套判定。所以这里仍**不能**返回 3 ——
    // 那会报出一条内核可能根本没走的路(曾实测 L1_054 报 pathId=3 而实际走原子切片), 比没有这个字段
    // 更危险。要让 pathId 名实相符, 必须先把那几道容量门整体搬到 host 并由 simt.h 按 pathId 分发;
    // 在那之前这个字段只表示"原子/排序"二选一。bucketRows 与 ubUsable 照常下发, 供分桶内核取用。
    (void)bucketRows;
    // 行太宽, 任何 UB 窗口都装不下一行的切片 -> 退回逐命中原子写(仅 MAX/MIN 且 dtype 有硬件原子时可用,
    // 该能力由内核侧 IsAtomicMaxMinSupported 以模板常量判定), 否则由内核落到排序。
    return (sliceSize == 1UL) ? 1UL : 2UL;
}

// 形态与分块的派生量, 由 ResolveSimtSplit 一次解析, 后续三段只读。
struct SimtSplit {
    size_t varRank = 0;
    uint64_t varFirstDim = 1;
    uint64_t varTotal = 0;
    uint64_t sliceSize = 0;
    uint64_t indicesNum = 0;
    uint64_t keySz = 0;
    uint64_t sortTile = 0;
    uint64_t blockNum = 1;
    uint64_t perCoreIndices = 0;
    uint64_t tailCoreIndices = 0;
};

static ge::graphStatus ResolveSimtSplit(gert::TilingContext* context, SimtSplit& s)
{
    auto varShapePtr = context->GetInputShape(VAR_IDX);
    auto indicesShapePtr = context->GetInputShape(INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShapePtr);
    auto& varShape = varShapePtr->GetStorageShape();
    auto& indicesShape = indicesShapePtr->GetStorageShape();

    s.varRank = varShape.GetDimNum();
    s.varFirstDim = (varShape.GetDimNum() == 0) ? 1 : varShape.GetDim(0);
    // 排序 key 直接用索引真值(AscendC::Sort 的 key 支持 int64), 不再折算低位 + 第二趟分区。
    // 因此 var 首维没有任何上限, 与 ascend910b 一致: 不分桶, 也就没有桶数带来的边界。
    s.varTotal = varShape.GetShapeSize();
    s.sliceSize = (s.varFirstDim == 0) ? 0 : s.varTotal / s.varFirstDim;
    s.indicesNum = (indicesShape.GetDimNum() == 0) ? 1 : indicesShape.GetShapeSize(); // 0 维 = 单标量索引

    // kernel 侧 Init 的布局: sortedIdx[mAlign x keySz] | originPos[mAlign x 4]
    //                        | scratchKeys[mAlign x keySz] | scratchPos[mAlign x 4] | partials
    // key 是索引真值, 两个 key 区按索引 dtype 宽度计 —— 必须与 kernel 同源, 否则区间重叠。
    auto indicesDesc = context->GetInputDesc(INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesDesc);
    const int32_t keySzI = ge::GetSizeByDataType(indicesDesc->GetDataType());
    OP_CHECK_IF(keySzI <= 0, OP_LOGE(context->GetNodeName(), "get indices dtype size fail."), return ge::GRAPH_FAILED);
    s.keySz = static_cast<uint64_t>(keySzI);
    s.sortTile = ResolveSortTile(context, s.keySz);

    uint64_t coreNum = 1;
    if (ResolveCoreNum(context, coreNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // split index entries across cores; each core handles whole slices, read directly from GM in the VF
    s.blockNum = (s.indicesNum == 0) ? 1 : (s.indicesNum < coreNum ? s.indicesNum : coreNum);
    s.perCoreIndices = (s.blockNum == 0) ? 0 : (s.indicesNum + s.blockNum - 1) / s.blockNum;
    s.blockNum = (s.perCoreIndices == 0) ? 1 : (s.indicesNum + s.perCoreIndices - 1) / s.perCoreIndices;
    s.tailCoreIndices = s.indicesNum - (s.blockNum - 1) * s.perCoreIndices;
    return ge::GRAPH_SUCCESS;
}

// 形态可达 2^31, 字节量会越过 uint64; 饱和运算保证比较单调, 不回绕成反向结论。
static uint64_t MulSat(uint64_t a, uint64_t b)
{
    if (a == 0UL || b == 0UL) {
        return 0UL;
    }
    return (a > UINT64_MAX / b) ? UINT64_MAX : a * b;
}

static uint64_t AddSat(uint64_t a, uint64_t b) { return (a > UINT64_MAX - b) ? UINT64_MAX : (a + b); }

// 列轴切分: 先分核(每核一段连续列, 均衡由定义保证), UB 容量只决定段内切几刀。
// 参与核数受 blockNum 约束 —— 排序阶段核数由 M 定, 折叠阶段不能超过它。
struct ColSplit {
    uint64_t cores;
    uint64_t colsPerCore;
    uint64_t chunk;
};

static ColSplit ResolveColSplit(const SimtSplit& s, uint64_t ubChunkMax)
{
    const uint64_t cores = (s.sliceSize < s.blockNum) ? s.sliceSize : s.blockNum;
    if (cores == 0UL || ubChunkMax == 0UL) {
        return ColSplit{1UL, s.sliceSize, s.sliceSize};
    }
    const uint64_t colsPerCore = (s.sliceSize + cores - 1UL) / cores;
    const uint64_t tiles = (colsPerCore + ubChunkMax - 1UL) / ubChunkMax;
    uint64_t chunk = (tiles == 0UL) ? colsPerCore : (colsPerCore + tiles - 1UL) / tiles;
    const uint64_t align = static_cast<uint64_t>(ScatterReduceCommon::UB_BLOCK_BYTES) / sizeof(float);
    chunk = (chunk + align - 1UL) / align * align; // 对齐到一个 block 的元素数
    if (chunk > ubChunkMax) {
        chunk = ubChunkMax;
    }
    if (chunk > colsPerCore) {
        chunk = colsPerCore; // 对齐上取不得超过本核的列数
    }
    return ColSplit{cores, colsPerCore, (chunk == 0UL) ? 1UL : chunk};
}

// 列轴单核搬运字节: 本核列段的 updates + var 读回写, 外加全量 originPos(每核都要扫完 M 条位置)。
static uint64_t ColumnAxisBytes(const SimtSplit& s, uint64_t varBytes, const ColSplit& cs)
{
    const uint64_t colsPerCore = cs.colsPerCore;
    const uint64_t upd = MulSat(MulSat(s.indicesNum, colsPerCore), varBytes);
    const uint64_t var = MulSat(MulSat(2UL, MulSat(s.varFirstDim, colsPerCore)), varBytes);
    return AddSat(AddSat(upd, var), MulSat(s.indicesNum, POS_ENTRY_BYTES));
}

// 位置轴单核搬运字节: 本核位置的 updates + 均摊 var, 外加偏积往返(每核每列一份, owner 读回 blockNum 份)。
static uint64_t PositionAxisBytes(const SimtSplit& s, uint64_t varBytes, uint64_t accBytes)
{
    const uint64_t upd = MulSat(MulSat(s.perCoreIndices, s.sliceSize), varBytes);
    const uint64_t var = MulSat(MulSat(2UL, MulSat(s.varFirstDim, s.sliceSize)), varBytes) / s.blockNum;
    const uint64_t ws = MulSat(MulSat(AddSat(s.blockNum, 1UL), s.sliceSize), accBytes);
    return AddSat(AddSat(AddSat(upd, var), ws), MulSat(s.perCoreIndices, POS_ENTRY_BYTES));
}

// 排序路径并行轴: 取单核搬运字节(makespan 下界)较小的一条。两轴的向量折叠工作量相同, VOP 项相消,
// 差别只在 GM 流量与核利用率; 位置轴独有的集合同步只会抬高它自己, 故省略该项仍是保守比较。
static uint64_t ResolveParallelAxis(const SimtSplit& s, uint64_t varBytes, uint64_t ubChunkMax)
{
    constexpr uint64_t AXIS_POSITION = 0UL;
    constexpr uint64_t AXIS_COLUMN = 1UL;
    if (s.sliceSize < 2UL || s.indicesNum == 0UL || s.blockNum == 0UL || ubChunkMax == 0UL) {
        return AXIS_POSITION; // 无列可切或空形态: 不进排序列折叠
    }
    if (s.indicesNum <= s.blockNum) {
        return AXIS_COLUMN; // 位置轴可行性约束(非性能取舍): 每核不足 2 个位置时跨核分段合并无出口
    }
    const uint64_t accBytes = (varBytes < sizeof(float)) ? sizeof(float) : varBytes; // 窄类型提升到 float
    const uint64_t colBytes = ColumnAxisBytes(s, varBytes, ResolveColSplit(s, ubChunkMax));
    return (colBytes <= PositionAxisBytes(s, varBytes, accBytes)) ? AXIS_COLUMN : AXIS_POSITION;
}

static ge::graphStatus FillSimtTilingData(gert::TilingContext* context, const SimtSplit& s, uint64_t mode,
                                          ScatterReduceCommon::ScatterReduceSimtTilingData* td, uint64_t& varBytes)
{
    td->blockNum = s.blockNum;
    td->blockTilingSize = s.perCoreIndices * s.sliceSize;
    td->tailBlockTilingSize = s.tailCoreIndices * s.sliceSize;
    td->sliceSize = s.sliceSize;
    td->varFirstDim = s.varFirstDim;
    td->scalarPath = ResolveScalarPath(context, s.indicesNum, s.varFirstDim, s.keySz) ? 1UL : 0UL;
    td->sortTile = s.sortTile;
    td->ubChunkMax = ResolveUbChunkMax(context, mode);

    // 原子/分桶规格与选路: 全部按实测 UB 与 dtype 宽度反解, 不含经验阈值。
    auto varDescForSpec = context->GetInputDesc(VAR_IDX);
    const int32_t varSzSpecI = (varDescForSpec == nullptr) ? -1 : ge::GetSizeByDataType(varDescForSpec->GetDataType());
    varBytes = (varSzSpecI <= 0) ? sizeof(float) : static_cast<uint64_t>(varSzSpecI);
    // uint8 无硬件原子, 内核借 x^0x80 保序映射走 int8 原子(XFORM), 那条路额外吃 2 份 updates 宽的
    // 临时缓冲 + 48KB 翻转缓冲, 必须进 UB 账本。判据与内核侧 IsAtomicMaxMinUsableRow1D 的 uint8
    // 分支同源 —— 两边改一边就会再出一次 UB 越界。
    const bool varIsU8 = (varDescForSpec != nullptr) && (varDescForSpec->GetDataType() == ge::DT_UINT8);
    const AtomicUbSpec aspec = ResolveAtomicUbSpec(context, s.sliceSize, s.keySz, varBytes, varIsU8);
    td->atomicTile = aspec.tile;
    td->atomicSlots = aspec.slots;
    td->atomicIdxBytes = aspec.idxBytes;
    td->atomicUpdBytes = aspec.updBytes;
    td->atomicPosBytes = aspec.posBytes;
    td->atomicSlotSpan = aspec.slotSpan;
    td->bucketRows = ResolveBucketRows(context, s.sliceSize, varBytes);
    td->ubUsable = ResolveUbUsable(context);
    td->pathId = ResolvePathId(context, td->scalarPath != 0UL, s.sliceSize, varBytes, td->bucketRows);
    td->parallelAxis = ResolveParallelAxis(s, varBytes, td->ubChunkMax);
    const ColSplit cs = ResolveColSplit(s, td->ubChunkMax);
    td->colsPerCore = cs.colsPerCore;
    td->colChunk = cs.chunk;
    return ge::GRAPH_SUCCESS;
}

// 排序 scratch 按固定预算反解片长, 不随 M 膨胀 —— 否则 workspace 预算就成了更新条数上限。
// 内核把全量 M 切片, 每片独立排序+折叠并复用同一块 scratch。
static ge::graphStatus ResolveSimtWorkspace(gert::TilingContext* context, const SimtSplit& s,
                                            ScatterReduceCommon::ScatterReduceSimtTilingData* td)
{
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    // per-core partial slot stride. Host uses sliceSize; kernel uses sAlign = align8(min(sliceSize, CHUNK_MAX)),
    // so when sliceSize > CHUNK_MAX the host stride >= kernel stride -- this OVER-allocates (never under), which
    // is safe. Keep sliceSize here (not CHUNK) so host can never under-allocate the partial region.
    const uint64_t sliceAlign = (s.sliceSize + 7UL) / 8UL * 8UL;
    const uint64_t perEntry = 2UL * (s.keySz + 8UL); // 四个 scratch 子区每条目合计字节
    uint64_t chunkM = WS_SORT_BUDGET / perEntry;
    if (chunkM > s.indicesNum) {
        chunkM = s.indicesNum;
    }
    if (chunkM == 0UL) {
        chunkM = 1UL;
    }
    td->sortChunkM = chunkM;
    // 与内核 Init 同源: 按片长复刻 padM/对齐/余量(merge-sort 把每个 run 补齐到 runLen0,
    // 故区尺寸要按 padChunk = P2*runLen0 而非 chunkM, 否则排序越界写)。
    uint64_t padChunk = chunkM;
    if (chunkM > s.sortTile) {
        uint64_t p2 = 1UL;
        while (p2 * 2UL <= s.blockNum) {
            p2 *= 2UL;
        }
        while ((chunkM + p2 - 1UL) / p2 > s.sortTile) {
            p2 *= 2UL;
        }
        padChunk = p2 * ((chunkM + p2 - 1UL) / p2);
    }
    const uint64_t mAlignChunk = ((padChunk + 7UL) / 8UL * 8UL) + 128UL; // 8B 对齐 + 128B 余量
    workspaces[0] = SYS_WORKSPACE + mAlignChunk * perEntry +
                    static_cast<uint64_t>(s.blockNum) * sliceAlign * 4UL; // per-core partial slots
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterReduceCommonTiling(gert::TilingContext* context, uint64_t mode)
{
    const char* opName = context->GetNodeName();
    OP_CHECK_IF(CheckScatterReduceInputs(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName, "CheckScatterReduceInputs failed"), return ge::GRAPH_FAILED);
    SimtSplit s;
    OP_CHECK_IF(ResolveSimtSplit(context, s) != ge::GRAPH_SUCCESS, OP_LOGE(opName, "ResolveSimtSplit failed"),
                return ge::GRAPH_FAILED);
    auto* td = context->GetTilingData<ScatterReduceCommon::ScatterReduceSimtTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);
    uint64_t varBytes = sizeof(float);
    OP_CHECK_IF(FillSimtTilingData(context, s, mode, td, varBytes) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName, "FillSimtTilingData failed"), return ge::GRAPH_FAILED);

    OP_LOGI(context->GetNodeName(),
            "scatter_reduce probe: varRank=%zu varFirstDim=%lu varTotal=%lu sliceSize=%lu indicesNum=%lu keySz=%lu "
            "varBytes=%lu blockNum=%lu scalarPath=%lu sortTile=%lu atomicTile=%lu atomicSlots=%lu bucketRows=%lu "
            "ubUsable=%lu "
            "pathId=%lu",
            s.varRank, s.varFirstDim, s.varTotal, s.sliceSize, s.indicesNum, s.keySz, varBytes, s.blockNum,
            td->scalarPath, s.sortTile, td->atomicTile, td->atomicSlots, td->bucketRows, td->ubUsable, td->pathId);

    context->SetBlockDim(s.blockNum);
    context->SetTilingKey(0);
    return ResolveSimtWorkspace(context, s, td);
}
} // namespace optiling
