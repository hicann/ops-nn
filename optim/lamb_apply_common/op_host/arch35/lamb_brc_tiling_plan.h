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
 * \file lamb_brc_tiling_plan.h
 * \brief LAMB 族「多入多出 + 任意 numpy 广播」逐元素算子的共用形状规划(host 侧)
 *
 * 折叠 -> 分类 -> 由 ubSize 反解分片 -> 选平铺/分块。全程无经验阈值:
 * 分片长度是 ubSize 除以槽位总字节解出来的; 行块轴是"能装下"这个条件由内向外推出来的,
 * 退化到最内层必然成立, 因此不存在因尺寸拒收的分支。
 */

#ifndef LAMB_BRC_TILING_PLAN_H
#define LAMB_BRC_TILING_PLAN_H

#include <algorithm>
#include <vector>
#include "../../op_kernel/arch35/lamb_brc_tiling_data.h"
#include "exe_graph/runtime/tiling_context.h"
#include "log/log.h"

namespace optiling {

// ---------------------------------------------------------------------------------------------
// BuildLambBrcPlan 的分步实现: 读形状 -> 折叠轴 -> 分类输入 -> 解分片 -> 平铺分核 /
// 分块(选行块轴 -> 分核 -> 算步长)。拆开是为了每段都短到能一眼读完、各自可单独复核。
// ---------------------------------------------------------------------------------------------

// 读出各输入形状, 右对齐补维, 并校验可广播性。
template <uint32_t NIN>
ge::graphStatus LambBrcReadShapes(gert::TilingContext* context, size_t rank, const std::vector<int64_t>& rawOut,
                                  std::vector<std::vector<int64_t>>& rawIn)
{
    for (uint32_t i = 0; i < NIN; i++) {
        auto sp = context->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, sp);
        const gert::Shape& s = sp->GetStorageShape();
        size_t r = s.GetDimNum();
        OP_CHECK_IF(r > rank, OP_LOGE(context->GetNodeName(), "input %u rank %zu exceeds output rank %zu", i, r, rank),
                    return ge::GRAPH_FAILED);
        for (size_t d = 0; d < r; d++) { // 右对齐补维
            rawIn[i][rank - r + d] = s.GetDim(d);
        }
        for (size_t d = 0; d < rank; d++) {
            OP_CHECK_IF(rawIn[i][d] != 1 && rawIn[i][d] != rawOut[d],
                        OP_LOGE(context->GetNodeName(), "input %u dim %zu (%ld) not broadcastable to %ld", i, d,
                                rawIn[i][d], rawOut[d]),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

// 折叠: 丢掉输出为 1 的轴, 再合并广播状态一致的相邻轴。
template <uint32_t NIN>
void LambBrcCollapse(const std::vector<int64_t>& rawOut, const std::vector<std::vector<int64_t>>& rawIn, size_t rank,
                     std::vector<int64_t>& co, std::vector<std::vector<int64_t>>& ci)
{
    // 折叠: 丢掉输出为 1 的轴, 再合并广播状态一致的相邻轴。
    for (size_t d = 0; d < rank; d++) {
        if (rawOut[d] == 1) {
            continue;
        }
        bool merge = !co.empty();
        for (uint32_t i = 0; i < NIN && merge; i++) {
            merge = ((ci[i].back() == 1) == (rawIn[i][d] == 1));
        }
        if (merge) {
            co.back() *= rawOut[d];
            for (uint32_t i = 0; i < NIN; i++) {
                ci[i].back() *= rawIn[i][d];
            }
        } else {
            co.push_back(rawOut[d]);
            for (uint32_t i = 0; i < NIN; i++) {
                ci[i].push_back(rawIn[i][d]);
            }
        }
    }
    if (co.empty()) { // 全 1 形状
        co.push_back(1);
        for (uint32_t i = 0; i < NIN; i++) {
            ci[i].push_back(1);
        }
    }
}

// 给每个输入定性: 标量 / 与输出同形 / 需要广播。返回是否存在需要广播的输入。
template <uint32_t NIN, uint32_t NOUT>
bool LambBrcClassify(const std::vector<int64_t>& co, const std::vector<std::vector<int64_t>>& ci, uint32_t rc,
                     LambBrcTilingData<NIN, NOUT>& td)
{
    bool anyBrc = false;
    for (uint32_t i = 0; i < NIN; i++) {
        uint64_t numel = 1;
        bool brc = false;
        for (uint32_t d = 0; d < rc; d++) {
            td.inShape[i * LAMB_BRC_MAX_DIM + d] = static_cast<uint32_t>(ci[i][d]);
            numel *= static_cast<uint64_t>(ci[i][d]);
            if (ci[i][d] != co[d]) {
                brc = true;
            }
        }
        if (numel == 1) {
            td.inKind[i] = LAMB_BRC_KIND_SCALAR;
        } else if (!brc) {
            td.inKind[i] = LAMB_BRC_KIND_SAME;
        } else {
            td.inKind[i] = LAMB_BRC_KIND_BRC;
            anyBrc = true;
        }
    }
    return anyBrc;
}

// 由 ubSize 反解分片长度。
template <uint32_t NIN, uint32_t NOUT>
uint32_t LambBrcSolveTileLen(uint64_t ubSize, uint32_t dtSize)
{
    // 分片长度: UB 总字节 / 槽位数 / 元素字节, 向下对齐到 256B(保证每槽首址对齐)。
    constexpr uint32_t slotNum = LambBrcTilingData<NIN, NOUT>::SlotNum();
    uint32_t alignElems = LAMB_BRC_VREG_BYTES / dtSize;
    // 从 ubSize 反解, 再向下对齐到一个向量寄存器; 解不出整数倍时兜底取一个对齐单位 ——
    // 不做"装不下就拒收"的判断, 装不下就继续切, 这是切分问题不是能力问题。
    uint64_t bytesPerElem = static_cast<uint64_t>(slotNum) * dtSize + 4U * sizeof(float);
    uint32_t tileLen = static_cast<uint32_t>((ubSize / bytesPerElem / alignElems) * alignElems);
    if (tileLen == 0) {
        tileLen = alignElems;
    }
    return tileLen;
}

// 无广播输入: 按元素平铺分核。
template <uint32_t NIN, uint32_t NOUT>
void LambBrcPlanFlat(uint64_t coreNum, uint32_t dtSize, uint64_t total, uint32_t tileLen,
                     LambBrcTilingData<NIN, NOUT>& td)
{
    // 分核粒度必须让每个核的输出起始落在 32B 边界上: 否则相邻核的 DataCopyPad 会写到
    // 同一个 32 字节块里互相覆盖。
    const uint64_t gmAlign = LAMB_BRC_GM_BLOCK_BYTES / dtSize;

    td.tilingKey = LAMB_BRC_KEY_FLAT;
    uint64_t tiles = (total + tileLen - 1) / tileLen;
    uint64_t cores = std::min<uint64_t>(coreNum, std::max<uint64_t>(tiles, 1));
    uint64_t perCore = (total + cores - 1) / cores;
    perCore = ((perCore + gmAlign - 1) / gmAlign) * gmAlign; // 对齐到 32B
    td.perCoreElems = perCore;
    td.usedCoreNum = static_cast<uint32_t>((total + perCore - 1) / perCore);
}

// 分块: 选行块轴(splitAxis)与块长(blockLen)。
template <uint32_t NIN, uint32_t NOUT>
void LambBrcChooseSplit(const std::vector<int64_t>& co, const std::vector<std::vector<int64_t>>& ci, uint32_t rc,
                        uint32_t tileLen, LambBrcTilingData<NIN, NOUT>& td, uint32_t& splitAxis, uint64_t& blockLen)
{
    splitAxis = rc;
    blockLen = 1;
    // 块内只允许"尾轴广播"这一种形态(那条 2D Broadcast 路径是验证过的)。任何非尾轴的广播轴
    // 都把 splitAxis 推到它之后, 使其变成行维、由 effStride=0 处理 —— 否则块内要在任意元素偏移
    // 上做 UB->UB 复制, 而向量指令对起始地址有对齐要求。
    uint32_t splitFloor = 0;
    for (uint32_t i = 0; i < NIN; i++) {
        if (td.inKind[i] != LAMB_BRC_KIND_BRC) {
            continue;
        }
        for (uint32_t d = 0; d + 1 < rc; d++) { // 只看非尾轴
            if (ci[i][d] == 1 && co[d] != 1 && d + 1 > splitFloor) {
                splitFloor = d + 1;
            }
        }
    }

    // 分块: 由内向外找最小的前导行维个数(不低于 splitFloor), 使一个行块装得下分片。
    for (int32_t k = static_cast<int32_t>(rc); k >= static_cast<int32_t>(splitFloor); k--) {
        uint64_t len = 1;
        for (uint32_t d = static_cast<uint32_t>(k); d < rc; d++) {
            len *= static_cast<uint64_t>(co[d]);
        }
        if (len > tileLen) {
            break;
        }
        splitAxis = static_cast<uint32_t>(k);
        blockLen = len;
    }
    // 最内轴自己就装不下一个分片时(blockLen 被逼到 1), 不新增轴, 而是在最内轴内部切段:
    // 行索引的最低位变成"段号", 块长取最内轴不超过分片长度的最大因子。轴数不变, 不设上限。
    if (blockLen == 1 && rc > 0 && static_cast<uint64_t>(co[rc - 1]) > 1) {
        int64_t d = co[rc - 1];
        int64_t chunk = 1;
        for (int64_t c = static_cast<int64_t>(tileLen); c >= 1; c--) {
            if (d % c == 0) {
                chunk = c;
                break;
            }
        }
        // 最内轴为大质数时取不到 >1 的因子, 此时保持逐元素: 结果仍正确(行批处理保证搬出连续),
        // 只是 DMA 次数多, 属性能退化, 不拒收。
        if (chunk > 1) {
            splitAxis = rc - 1;
            blockLen = static_cast<uint64_t>(chunk);
            td.innerChunk = static_cast<uint32_t>(chunk);
            td.innerChunkCnt = static_cast<uint32_t>(d / chunk);
        }
    }
}

// 分块模式的分核: 行批大小与每核行数。
template <uint32_t NIN, uint32_t NOUT>
void LambBrcCoreSplit(uint64_t coreNum, uint32_t dtSize, uint64_t total, uint32_t tileLen, uint32_t splitAxis,
                      uint64_t blockLen, LambBrcTilingData<NIN, NOUT>& td)
{
    const uint64_t gmAlign = LAMB_BRC_GM_BLOCK_BYTES / dtSize;
    td.tilingKey = LAMB_BRC_KEY_BLOCK;
    td.splitAxis = splitAxis;
    td.blockLen = static_cast<uint32_t>(blockLen);
    // 一个分片内批量攒多行再一次性搬出: 逐行搬出会退化成 4/2 字节的 DataCopyPad,
    // 同一个 32 字节块内的多次写互相覆盖。
    // 行批处理要求每行在 UB 里的起始偏移 j*blockLen 落在 32 字节边界上, 否则 DataCopyPad 的
    // UB 侧地址不对齐。blockLen 不是 32 字节整数倍时退回每次一行(偏移恒为 0)。
    // 跨核不冲突由 rowsPerCore 的 32 字节对齐保证, 与本项无关。
    bool blockAligned = ((blockLen * dtSize) % LAMB_BRC_GM_BLOCK_BYTES) == 0;
    td.rowsPerTile = blockAligned ? static_cast<uint32_t>(std::max<uint64_t>(tileLen / blockLen, 1)) : 1U;
    td.totalRows = total / blockLen;
    // 每核行数取到 rowsPerGroup 的整数倍, 使 rowsPerCore*blockLen 是 32B 的整数倍。
    uint64_t g = blockLen % gmAlign;
    uint64_t a = gmAlign;
    while (g != 0) {
        uint64_t t2 = a % g;
        a = g;
        g = t2;
    } // gcd(blockLen, gmAlign)
    uint64_t rowsPerGroup = gmAlign / a;
    uint64_t cores = std::min<uint64_t>(coreNum, std::max<uint64_t>(td.totalRows, 1));
    uint64_t rowsPerCore = (td.totalRows + cores - 1) / cores;
    rowsPerCore = ((rowsPerCore + rowsPerGroup - 1) / rowsPerGroup) * rowsPerGroup;
    td.rowsPerCore = rowsPerCore;
    td.usedCoreNum = static_cast<uint32_t>((td.totalRows + rowsPerCore - 1) / rowsPerCore);
}

// 每个输入的块内长度与行维步长(广播轴步长为 0)。
template <uint32_t NIN, uint32_t NOUT>
void LambBrcStrides(const std::vector<std::vector<int64_t>>& ci, uint32_t rc, uint32_t splitAxis,
                    LambBrcTilingData<NIN, NOUT>& td)
{
    for (uint32_t i = 0; i < NIN; i++) {
        uint64_t srcBlock = 1;
        for (uint32_t d = splitAxis; d < rc; d++) {
            srcBlock *= static_cast<uint64_t>(ci[i][d]);
        }
        if (td.innerChunkCnt > 1) {
            // 块只覆盖最内轴的一段: 该输入在这一轴上要么是 1(广播), 要么与输出等长。
            srcBlock = (ci[i][rc - 1] == 1) ? 1 : static_cast<uint64_t>(td.innerChunk);
        }
        td.srcBlockLen[i] = srcBlock;
        for (uint32_t j = 0; j < splitAxis; j++) {
            if (ci[i][j] == 1) {
                td.effStride[i * LAMB_BRC_MAX_DIM + j] = 0; // 广播轴不推进源地址
                continue;
            }
            uint64_t st = 1;
            for (uint32_t d = j + 1; d < rc; d++) {
                st *= static_cast<uint64_t>(ci[i][d]);
            }
            td.effStride[i * LAMB_BRC_MAX_DIM + j] = st;
        }
    }
}

template <uint32_t NIN, uint32_t NOUT>
ge::graphStatus BuildLambBrcPlan(gert::TilingContext* context, uint64_t coreNum, uint64_t ubSize, uint32_t dtSize,
                                 LambBrcTilingData<NIN, NOUT>& td)
{
    auto outShapePtr = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShapePtr);
    const gert::Shape& outShape = outShapePtr->GetStorageShape();
    size_t rank = outShape.GetDimNum();
    std::vector<int64_t> rawOut(rank);
    for (size_t d = 0; d < rank; d++) {
        rawOut[d] = outShape.GetDim(d);
    }
    std::vector<std::vector<int64_t>> rawIn(NIN, std::vector<int64_t>(rank, 1));
    OP_CHECK_IF(LambBrcReadShapes<NIN>(context, rank, rawOut, rawIn) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "read input shapes failed"), return ge::GRAPH_FAILED);

    std::vector<int64_t> co;
    std::vector<std::vector<int64_t>> ci(NIN);
    LambBrcCollapse<NIN>(rawOut, rawIn, rank, co, ci);
    uint32_t rc = static_cast<uint32_t>(co.size());
    td.collapsedRank = rc;
    uint64_t total = 1;
    for (uint32_t d = 0; d < rc; d++) {
        td.outShape[d] = static_cast<uint32_t>(co[d]);
        total *= static_cast<uint64_t>(co[d]);
    }
    td.totalNum = total;

    bool anyBrc = LambBrcClassify<NIN, NOUT>(co, ci, rc, td);
    uint32_t tileLen = LambBrcSolveTileLen<NIN, NOUT>(ubSize, dtSize);
    td.tileLen = tileLen;
    if (!anyBrc) {
        LambBrcPlanFlat<NIN, NOUT>(coreNum, dtSize, total, tileLen, td);
        return ge::GRAPH_SUCCESS;
    }

    uint32_t splitAxis = rc;
    uint64_t blockLen = 1;
    LambBrcChooseSplit<NIN, NOUT>(co, ci, rc, tileLen, td, splitAxis, blockLen);
    LambBrcCoreSplit<NIN, NOUT>(coreNum, dtSize, total, tileLen, splitAxis, blockLen, td);
    LambBrcStrides<NIN, NOUT>(ci, rc, splitAxis, td);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
#endif // LAMB_BRC_TILING_PLAN_H
