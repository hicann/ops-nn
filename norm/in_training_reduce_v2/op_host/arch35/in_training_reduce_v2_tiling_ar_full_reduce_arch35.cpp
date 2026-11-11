/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file in_training_reduce_v2_tiling_ar_full_reduce_arch35.cpp
 * \brief AR full_reduce（R 全载）切核 / UB 切分。裁剪自 instance_norm ar_full_reduce：
 *        删 gamma/beta/y/mean/var/rstd/meanFp32 项、删 avgFactor/epsilon 字段。
 */
#include <vector>
#include <algorithm>
#include <limits>
#include "in_training_reduce_v2_tiling.h"

using namespace ge;

namespace optiling {
constexpr int64_t TILINGKEY_AR_FULL_REDUCE = 200000;
constexpr int64_t TILINGKEY_SPLIT_R = 300000;
constexpr int64_t RESERVE_FOR_ALIGN = 512;
constexpr int64_t FP32_BYTE = 4;
constexpr int64_t FP16_BYTE = 2;
constexpr uint32_t ULONG_BIT_LEN = 64;
constexpr uint64_t BINARY_FOLD_RADIX = 2;
constexpr uint64_t DOUBLE_BUFFER_COUNT = 2;
constexpr uint64_t ACC_COMPONENTS = 2;
constexpr uint64_t REDUCE_OUTPUT_COUNT = 2;
constexpr uint64_t BALANCED_BUDGET_PARTS = 2;
constexpr uint64_t MAX_FULL_AR_VECTOR_SEGMENTS = 256;
constexpr uint64_t MAX_PACKED_R_VECTOR_SEGMENTS = 8;
constexpr uint64_t MEDIUM_R_CORE_ROW_DIVISOR = 2;
constexpr uint64_t MIN_SPLIT_R_ELEMENTS = 65536;
constexpr uint64_t MIN_SPLIT_R_CORE_ELEMENTS = 16384;
// DAV_3510 DataCopyExtParams::blockLen is a 21-bit byte count.
constexpr uint64_t MAX_DMA_BLOCK_BYTES = (1UL << 21) - 1;
// 部分和缓存初始预留比例：numChunks 未知时先按 usable/8 估一版 rFactor，估偏了由联合求解收敛。
constexpr uint64_t PARTIAL_RESERVE_DIV = 8;
// 限制单组快路径的联合求解轮数。双分量 partial 增大时逐步缩小 rFactor；
// 达到上限或单组无法容纳时，转入下方固定预算的分组路径并重新校验完整 UB 占用。
constexpr uint32_t SUB_R_SOLVE_MAX_ITER = 32;

bool INTrainingReduceV2ARFullReduceTiling::IsCapable()
{
    td_ = {};
    // NHWC/NDHWC C=1 时 [N,R,1] 与 [N,1,R] 物理连续布局相同，复用 AR 的连续归约。
    const bool contiguousChannelLast = (format == FORMAT_NHWC || format == FORMAT_NDHWC) && a0 == 1;
    if (format != FORMAT_NCHW && format != FORMAT_NCDHW && format != FORMAT_ND && !contiguousChannelLast) {
        return false;
    }
    // Empty reductions are rejected; nonempty reductions use full-load or sub-R tiling.
    if (r <= 0) {
        return false;
    }
    if (aicoreParams_.ubSize <= static_cast<uint64_t>(RESERVE_FOR_ALIGN) || aicoreParams_.blockDim == 0 ||
        ubBlockSize <= 0 || vlfp32 <= 0) {
        OP_LOGE(GetNodeName(), "invalid platform capacity: ubSize=%lu, blockDim=%lu, blockSize=%ld, vl=%ld",
                aicoreParams_.ubSize, aicoreParams_.blockDim, ubBlockSize, vlfp32);
        return false;
    }

    // 数据类型的元素宽度
    int64_t elemSize = FP32_BYTE;
    if (dataType == ge::DT_FLOAT16) {
        elemSize = FP16_BYTE;
    }
    // Medium reductions also benefit from parallel output lanes when enough
    // rows provide at least half a vector per core. Keep sparse-row cases on
    // the horizontal VF path.
    const bool packedMediumR = r <= vlfp32 * MAX_PACKED_R_VECTOR_SEGMENTS &&
                               totalRows_ / aicoreParams_.blockDim >=
                                   static_cast<uint64_t>(vlfp32) / MEDIUM_R_CORE_ROW_DIVISOR;
    if ((r <= vlfp32 || packedMediumR) && DoPackedRowTiling(static_cast<uint64_t>(elemSize))) {
        return true;
    }
    if (r == 1) {
        OP_LOGE(GetNodeName(), "R=1 input/output buffers exceed UB capacity");
        return false;
    }

    const uint64_t rValue = static_cast<uint64_t>(r);
    const uint64_t elemSizeValue = static_cast<uint64_t>(elemSize);
    const uint64_t blockSize = static_cast<uint64_t>(ubBlockSize);
    const uint64_t alignElements = blockSize / elemSizeValue;
    // 按元素数做 32B 对齐，避免先计算 R * elemSize 给接口引入不必要的
    // uint64 字节数上限。CeilAlign 在对齐值不可表示时饱和到 UINT64_MAX，不会回绕。
    const uint64_t rAlign = Ops::Base::CeilAlign(rValue, alignElements);
    // 计算二分累加折叠点：小于 rAlign 的最大 2 的幂
    uint64_t binAddQuotient = rAlign == 0 ? 1 : (1UL << (ULONG_BIT_LEN - 1 - __builtin_clzl(rAlign)));
    binAddQuotient = (binAddQuotient == rAlign) ? binAddQuotient / BINARY_FOLD_RADIX : binAddQuotient;
    if (DoSplitRTiling(rAlign, binAddQuotient, elemSizeValue)) {
        return true;
    }
    // Bound the per-row VF loop; larger rows retain the bounded sub-R path.
    if (Ops::Base::CeilDiv(rValue, static_cast<uint64_t>(vlfp32)) > MAX_FULL_AR_VECTOR_SEGMENTS) {
        return DoSubRTiling(rAlign, binAddQuotient, elemSize);
    }

    // 单行 UB 占用（删 gamma/beta/y/mean/var/rstd/meanFp32）：
    //   inQueueX_:            rAlign * elemSize        # 双 buf → * 2
    //   outQueueSum_:         sizeof(float)            # 双 buf → * 2
    //   outQueueSquareSum_:   sizeof(float)            # 双 buf → * 2
    // 高低部分和保留在寄存器中，不再分配逐向量水平折叠的scratch。
    const uint64_t usable = aicoreParams_.ubSize - RESERVE_FOR_ALIGN;
    if (rAlign > usable / elemSizeValue) {
        return DoSubRTiling(rAlign, binAddQuotient, elemSize);
    }
    const uint64_t alignedInputBytes = rAlign * elemSizeValue;
    const uint64_t outputBytesPerRow = sizeof(float) * DOUBLE_BUFFER_COUNT * REDUCE_OUTPUT_COUNT;
    if (alignedInputBytes > usable / DOUBLE_BUFFER_COUNT ||
        outputBytesPerRow > usable - alignedInputBytes * DOUBLE_BUFFER_COUNT) {
        // 单行 R 全载超单次 UB 容量 → sub-R 分块路径，同 TilingKey + isSubRTiling 标志。
        return DoSubRTiling(rAlign, binAddQuotient, elemSize);
    }
    const uint64_t bytesPerRow = alignedInputBytes * DOUBLE_BUFFER_COUNT + outputBytesPerRow;
    // Unmasked vector loads on the final row must stay within this input allocation.
    const uint64_t lastVectorBytes = Ops::Base::CeilAlign(rValue, static_cast<uint64_t>(vlfp32)) * elemSizeValue;
    const uint64_t inputTailBytes = lastVectorBytes - alignedInputBytes;
    uint64_t cInner = usable / bytesPerRow;
    // 可全载的行数不超过 C
    cInner = std::min(cInner, static_cast<uint64_t>(a0));
    // Filling UB with every channel can leave only N cores active. Partition
    // independent output rows first; each row keeps the same reduction order.
    const uint64_t tilesPerInstance = std::max(uint64_t{1}, aicoreParams_.blockDim / static_cast<uint64_t>(a1));
    const uint64_t balancedRows = Ops::Base::CeilDiv(static_cast<uint64_t>(a0), tilesPerInstance);
    cInner = std::min(cInner, balancedRows);
    // Both the VF row loop and DMA blockCount use uint16_t.
    cInner = std::min(cInner, static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()));
    uint64_t inputBufferBytes = 0;
    uint64_t outputBufferBytes = 0;
    while (cInner > 0) {
        inputBufferBytes = alignedInputBytes * cInner + inputTailBytes;
        outputBufferBytes = Ops::Base::CeilAlign(cInner * sizeof(float), blockSize);
        const uint64_t exactUbBytes = inputBufferBytes * DOUBLE_BUFFER_COUNT +
                                      outputBufferBytes * DOUBLE_BUFFER_COUNT * REDUCE_OUTPUT_COUNT;
        if (exactUbBytes <= aicoreParams_.ubSize) {
            break;
        }
        --cInner;
    }
    if (cInner == 0) {
        return DoSubRTiling(rAlign, binAddQuotient, elemSize);
    }
    const uint64_t a0Value = static_cast<uint64_t>(a0);
    const uint64_t a1Value = static_cast<uint64_t>(a1);
    uint64_t cOuter = a0Value / cInner + ((a0Value % cInner) != 0 ? 1 : 0);
    uint64_t cTail = a0Value - (cOuter - 1) * cInner;
    OP_CHECK_IF(cOuter != 0 && a1Value > std::numeric_limits<uint64_t>::max() / cOuter,
                OP_LOGE(GetNodeName(), "full-load tile count overflows uint64, N=%lu, cOuter=%lu", a1Value, cOuter),
                return false);
    uint64_t totalTileCnt = cOuter * a1Value;
    uint64_t perCoreCnt = Ops::Base::CeilDiv(totalTileCnt, aicoreParams_.blockDim);
    blockNum_ = Ops::Base::CeilDiv(totalTileCnt, perCoreCnt);

    td_.numN = a1;
    td_.numC = a0;
    td_.numR = r;
    td_.rAlign = rAlign;
    td_.cInner = static_cast<int64_t>(cInner);
    td_.cOuter = static_cast<int64_t>(cOuter);
    td_.cTail = static_cast<int64_t>(cTail);
    td_.binaryAddQuotient = binAddQuotient;
    td_.perCoreCnt = static_cast<int64_t>(perCoreCnt);
    td_.isSubRTiling = 0;
    td_.rFactor = 0;
    td_.numChunks = 0;
    td_.tailLen = 0;
    td_.chunksPerGroup = 0;
    td_.numGroups = 0;
    td_.tailChunks = 0;
    td_.totalRows = static_cast<int64_t>(totalRows_);
    td_.totalElements = static_cast<int64_t>(totalElements_);
    td_.totalTiles = static_cast<int64_t>(totalTileCnt);
    td_.inputBufferBytes = inputBufferBytes;
    td_.outputBufferBytes = outputBufferBytes;
    td_.scratchBufferBytes = 0;
    td_.partialBufferBytes = 0;
    return true;
}

// Borrow idle cores along R, retaining unrounded high/low components until the
// row owner performs the final compensated reduction. Larger grouped cases
// fall back to the existing sub-R implementation without a growing workspace.
bool INTrainingReduceV2ARFullReduceTiling::DoSplitRTiling(uint64_t rAlign, uint64_t binAddQuotient, uint64_t elemSize)
{
    const uint64_t rValue = static_cast<uint64_t>(r);
    const uint64_t vectorElements = static_cast<uint64_t>(vlfp32);
    if (rValue < MIN_SPLIT_R_ELEMENTS || totalRows_ >= aicoreParams_.blockDim) {
        return false;
    }
    uint64_t parts = std::min(
        {aicoreParams_.blockDim / totalRows_, vectorElements, Ops::Base::CeilDiv(rValue, MIN_SPLIT_R_CORE_ELEMENTS)});
    if (parts < 2) {
        return false;
    }
    const uint64_t coreElements = Ops::Base::CeilAlign(Ops::Base::CeilDiv(rValue, parts), vectorElements);
    parts = Ops::Base::CeilDiv(rValue, coreElements);
    if (parts < 2) {
        return false;
    }
    const uint64_t usable = aicoreParams_.ubSize - RESERVE_FOR_ALIGN;
    const uint64_t blockSize = static_cast<uint64_t>(ubBlockSize);
    const uint64_t partialBytes = vectorElements * sizeof(float) * ACC_COMPONENTS;
    const uint64_t workspaceRowBytes = parts * blockSize;
    const uint64_t outputBytes = REDUCE_OUTPUT_COUNT * blockSize;
    const uint64_t fixedBytes = REDUCE_OUTPUT_COUNT * partialBytes + workspaceRowBytes + outputBytes;
    if (fixedBytes >= usable) {
        return false;
    }
    const uint64_t maxLoopElements = std::numeric_limits<uint16_t>::max() * vectorElements;
    const uint64_t maxCopyElements = MAX_DMA_BLOCK_BYTES / elemSize / vectorElements * vectorElements;
    const uint64_t inputElements = (usable - fixedBytes) / DOUBLE_BUFFER_COUNT / elemSize / vectorElements *
                                   vectorElements;
    const uint64_t rFactor = std::min({inputElements, coreElements, maxLoopElements, maxCopyElements});
    if (rFactor == 0 || Ops::Base::CeilDiv(coreElements, rFactor) > vectorElements) {
        return false;
    }
    blockNum_ = static_cast<int64_t>(totalRows_ * parts);
    td_.numN = a1;
    td_.numC = a0;
    td_.numR = r;
    td_.rAlign = rAlign;
    td_.cInner = static_cast<int64_t>(coreElements);
    td_.cOuter = static_cast<int64_t>(parts);
    td_.cTail = static_cast<int64_t>(rValue - (parts - 1) * coreElements);
    td_.binaryAddQuotient = binAddQuotient;
    td_.perCoreCnt = 1;
    td_.isSubRTiling = IN_TRAINING_REDUCE_V2_SPLIT_R_MODE;
    td_.rFactor = rFactor;
    td_.numChunks = Ops::Base::CeilDiv(coreElements, rFactor);
    td_.tailLen = coreElements - (td_.numChunks - 1) * rFactor;
    td_.chunksPerGroup = vectorElements;
    td_.numGroups = 1;
    td_.tailChunks = td_.numChunks;
    td_.totalRows = static_cast<int64_t>(totalRows_);
    td_.totalElements = static_cast<int64_t>(totalElements_);
    td_.totalTiles = blockNum_;
    td_.inputBufferBytes = rFactor * elemSize;
    td_.outputBufferBytes = outputBytes;
    td_.scratchBufferBytes = workspaceRowBytes;
    td_.partialBufferBytes = partialBytes;
    return true;
}

// Each vector lane owns one output row. Medium-R rows with an even number
// of UB blocks add one block between rows to spread gathered lanes across banks.
// Short reductions retain contiguous DMA; R=1 keeps the elementwise fast path.
bool INTrainingReduceV2ARFullReduceTiling::DoPackedRowTiling(uint64_t elemSize)
{
    const uint64_t usable = aicoreParams_.ubSize - RESERVE_FOR_ALIGN;
    const uint64_t vectorElements = static_cast<uint64_t>(vlfp32);
    const uint64_t rowBytes = static_cast<uint64_t>(r) * elemSize;
    const uint64_t blockSize = static_cast<uint64_t>(ubBlockSize);
    const bool padRows = r > vlfp32 && rowBytes % (BINARY_FOLD_RADIX * blockSize) == 0;
    const uint64_t ubRowBytes = rowBytes + (padRows ? blockSize : 0);
    const uint64_t bytesPerRow = DOUBLE_BUFFER_COUNT * (ubRowBytes + REDUCE_OUTPUT_COUNT * sizeof(float));
    const uint64_t maxTileRows = usable / bytesPerRow / vectorElements * vectorElements;
    // Small mocked UB capacities can still use the original full-AR path.
    if (maxTileRows == 0) {
        return false;
    }
    const uint64_t balancedRows = Ops::Base::CeilAlign(Ops::Base::CeilDiv(totalRows_, aicoreParams_.blockDim),
                                                       vectorElements);
    const uint64_t maxLoopRows = std::numeric_limits<uint16_t>::max() * vectorElements;
    // Padded DMA uses one block per row; contiguous DMA uses one byte-count block.
    const uint64_t maxCopyRows = padRows ? std::numeric_limits<uint16_t>::max() / vectorElements * vectorElements :
                                           MAX_DMA_BLOCK_BYTES / rowBytes / vectorElements * vectorElements;
    const uint64_t tileRows = std::min({maxTileRows, balancedRows, maxLoopRows, maxCopyRows});
    const uint64_t totalTiles = Ops::Base::CeilDiv(totalRows_, tileRows);
    const uint64_t tilesPerCore = Ops::Base::CeilDiv(totalTiles, aicoreParams_.blockDim);
    blockNum_ = Ops::Base::CeilDiv(totalTiles, tilesPerCore);

    td_.numN = a1;
    td_.numC = a0;
    td_.numR = r;
    td_.rAlign = 0; // Packed rows use masked gathers.
    td_.rFactor = padRows ? ubRowBytes / elemSize : 0;
    td_.cInner = static_cast<int64_t>(tileRows);
    td_.cOuter = static_cast<int64_t>(totalTiles);
    td_.cTail = static_cast<int64_t>(totalRows_ - (totalTiles - 1) * tileRows);
    td_.perCoreCnt = static_cast<int64_t>(tilesPerCore);
    td_.totalRows = static_cast<int64_t>(totalRows_);
    td_.totalElements = static_cast<int64_t>(totalElements_);
    td_.totalTiles = static_cast<int64_t>(totalTiles);
    // Full vector loads, including the last partial vector, stay inside UB.
    td_.inputBufferBytes = tileRows * ubRowBytes;
    td_.outputBufferBytes = tileRows * sizeof(float);
    return true;
}

// sub-R 路径 Kernel 侧 UB 精确占用。逐项对应 InitSubR()：
//   inQueueX_          : CeilAlign(rFactor * elemSize, BLOCK_SIZE) * DOUBLE_BUFFER_NUM
//   sum/sqPartialBuf_  : CeilAlign(slots, VL_FP32) * sizeof(float) * ACC_COMPONENTS，两块
//                        slots = chunksPerGroup + (numGroups > 1 ? 1 : 0)
//                        —— 多出的那一格是组间 carry 槽，高低两分量各保留一格；单组无需 carry。
//   out 双 queue       : CeilAlign(sizeof(float), BLOCK_SIZE) * DOUBLE_BUFFER_NUM，两条
// 改 InitSubR() 的 buffer 规划时必须同步改这里，否则 Host 的容量校验会失真。
uint64_t INTrainingReduceV2ARFullReduceTiling::CalcSubRUbBytes(uint64_t rFactor, uint64_t chunksPerGroup,
                                                               uint64_t numGroups, int64_t elemSize) const
{
    uint64_t blockSize = static_cast<uint64_t>(ubBlockSize);
    uint64_t slots = chunksPerGroup + ((numGroups > 1) ? 1UL : 0UL);
    uint64_t inBytes = Ops::Base::CeilAlign(rFactor * static_cast<uint64_t>(elemSize), blockSize) * DOUBLE_BUFFER_COUNT;
    uint64_t partialBytes = Ops::Base::CeilAlign(slots, static_cast<uint64_t>(vlfp32)) * sizeof(float) *
                            REDUCE_OUTPUT_COUNT * ACC_COMPONENTS;
    uint64_t outBytes = Ops::Base::CeilAlign(sizeof(float), blockSize) * DOUBLE_BUFFER_COUNT * REDUCE_OUTPUT_COUNT;
    return inBytes + partialBytes + outBytes;
}

// sub-R 分块 tiling：R 超单次 UB 容量时，按 rFactor 分块搬入，
// 每块归约累进独立 fp32 累加器，跨块固定顺序树归约。按行(N*C)切核。
//
// 两段式：
//   快路径 —— 与分组改造前完全相同的求解。所有原本就能通过的 shape 都落在这里，
//              numGroups 恒为 1，tiling 参数与 Kernel 行为逐位不变。
//   慢路径 —— 快路径放不下（即改造前会被拒绝的 R）时启用分组折叠：部分和缓存固定为
//              chunksPerGroup 槽 + 1 个 carry 槽，UB 占用与 R 完全解耦，R 不再有上限。
bool INTrainingReduceV2ARFullReduceTiling::DoSubRTiling(uint64_t rAlign, uint64_t binAddQuotient, int64_t elemSize)
{
    uint64_t usable = aicoreParams_.ubSize - RESERVE_FOR_ALIGN;
    // 输出双 buffer：sum + square_sum，各 CeilAlign(4,32)=32B，×2 buf
    uint64_t outReserve = Ops::Base::CeilAlign(sizeof(float), static_cast<uint64_t>(ubBlockSize)) *
                          DOUBLE_BUFFER_COUNT * REDUCE_OUTPUT_COUNT;
    // 部分和缓存（Σx/Σx² 各 ceil(numChunks/VL)*VL 个 fp32）的初始预留：此刻 numChunks 还依赖
    // 待定的 rFactor，先按 1/8 usable 估一版，估偏了由下面的联合求解修正。
    uint64_t partialReserve = usable / PARTIAL_RESERVE_DIV;
    OP_CHECK_IF(outReserve >= usable || partialReserve >= usable - outReserve,
                OP_LOGE(GetNodeName(), "sub-R fixed buffers leave no input budget: usable=%lu out=%lu partial=%lu",
                        usable, outReserve, partialReserve),
                return false);
    uint64_t inputBudget = usable - outReserve - partialReserve;
    // 输入块双 buffer：rFactor * elemSize * 2
    uint64_t rFactor = inputBudget / (static_cast<uint64_t>(elemSize) * DOUBLE_BUFFER_COUNT);
    rFactor = rFactor / vlfp32 * vlfp32; // VL_FP32 对齐（下取整）
    if (rFactor < static_cast<uint64_t>(vlfp32)) {
        rFactor = vlfp32;
    }
    uint64_t rCeilVL = Ops::Base::CeilAlign(static_cast<uint64_t>(r), static_cast<uint64_t>(vlfp32));
    if (rFactor > rCeilVL) {
        rFactor = rCeilVL;
    }
    uint64_t numChunks = Ops::Base::CeilDiv(static_cast<uint64_t>(r), rFactor);

    // 联合求解（快路径）：partialReserve 只是估值，R 大时实际部分和缓存会撑破预留。
    // 这里用 Kernel 侧的精确公式复核总占用，超了就按实际 partial 占用重算 rFactor ——
    // rFactor 变小 ⇒ 输入 buffer 省下的字节多于 numChunks 增加所需的槽位，故迭代单调收敛。
    // 收敛不到不再直接拒绝，而是落到下面的分组折叠。
    for (uint32_t iter = 0; iter < SUB_R_SOLVE_MAX_ITER; ++iter) {
        if (CalcSubRUbBytes(rFactor, numChunks, 1UL, elemSize) <= usable) {
            break;
        }
        uint64_t partialBytes = Ops::Base::CeilAlign(numChunks, static_cast<uint64_t>(vlfp32)) * sizeof(float) *
                                REDUCE_OUTPUT_COUNT * ACC_COMPONENTS;
        if (usable <= outReserve + partialBytes) {
            break; // 部分和缓存本身已占满 UB，交给分组折叠
        }
        uint64_t nextBudget = usable - outReserve - partialBytes;
        uint64_t nextRFactor = nextBudget / (static_cast<uint64_t>(elemSize) * DOUBLE_BUFFER_COUNT) / vlfp32 * vlfp32;
        if (nextRFactor < static_cast<uint64_t>(vlfp32) || nextRFactor >= rFactor) {
            break; // 无法继续收缩，交给分组折叠
        }
        rFactor = nextRFactor;
        numChunks = Ops::Base::CeilDiv(static_cast<uint64_t>(r), rFactor);
    }

    uint64_t chunksPerGroup = numChunks;
    uint64_t numGroups = 1;
    if (CalcSubRUbBytes(rFactor, chunksPerGroup, numGroups, elemSize) > usable) {
        // ---- 慢路径：固定平衡点 + 分组折叠 ----
        // 令 S = 每分量对齐后的槽位数（含 carry），有 16·S + 2·e·rFactor + outReserve <= usable，
        // 目标是最大化单组覆盖量 chunksPerGroup·rFactor。取两项各占一半预算的平衡点。
        // chunksPerGroup 取 S-1，使 CeilAlign(chunksPerGroup+1, VL) 恰为 S、不额外多吃一个 VL。
        uint64_t budget = usable - outReserve;
        uint64_t slots = budget / BALANCED_BUDGET_PARTS / (sizeof(float) * REDUCE_OUTPUT_COUNT * ACC_COMPONENTS) /
                         vlfp32 * vlfp32;
        rFactor = budget / BALANCED_BUDGET_PARTS / (static_cast<uint64_t>(elemSize) * DOUBLE_BUFFER_COUNT) / vlfp32 *
                  vlfp32;
        if (slots < static_cast<uint64_t>(vlfp32)) {
            slots = vlfp32;
        }
        if (rFactor < static_cast<uint64_t>(vlfp32)) {
            rFactor = vlfp32;
        }
        chunksPerGroup = slots - 1;
        numChunks = Ops::Base::CeilDiv(static_cast<uint64_t>(r), rFactor);
        numGroups = Ops::Base::CeilDiv(numChunks, chunksPerGroup);
        // 对齐取整可能让平衡点略微超预算，逐 VL 收缩部分和槽位兜底（正常不会进循环）。
        while (chunksPerGroup > static_cast<uint64_t>(vlfp32) &&
               CalcSubRUbBytes(rFactor, chunksPerGroup, numGroups, elemSize) > usable) {
            chunksPerGroup -= vlfp32;
            numGroups = Ops::Base::CeilDiv(numChunks, chunksPerGroup);
        }
    }
    OP_CHECK_IF(CalcSubRUbBytes(rFactor, chunksPerGroup, numGroups, elemSize) > usable,
                OP_LOGE(GetNodeName(),
                        "sub-R tiling cannot fit UB: r=%ld needs %luB > %luB usable (rFactor=%lu chunksPerGroup=%lu "
                        "numGroups=%lu).",
                        r, CalcSubRUbBytes(rFactor, chunksPerGroup, numGroups, elemSize), usable, rFactor,
                        chunksPerGroup, numGroups),
                return false);

    uint64_t tailLen = static_cast<uint64_t>(r) - (numChunks - 1) * rFactor;
    uint64_t tailChunks = numChunks - (numGroups - 1) * chunksPerGroup;

    // 按行(N*C)切核：每行一个 (n,c) 的 R 个空间元素独立规约
    uint64_t perCoreCnt = Ops::Base::CeilDiv(totalRows_, aicoreParams_.blockDim);
    if (perCoreCnt < 1) {
        perCoreCnt = 1;
    }
    blockNum_ = Ops::Base::CeilDiv(totalRows_, perCoreCnt);

    OP_CHECK_IF(rFactor > std::numeric_limits<uint32_t>::max() || tailLen > std::numeric_limits<uint32_t>::max() ||
                    chunksPerGroup > std::numeric_limits<uint32_t>::max() ||
                    tailChunks > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(GetNodeName(),
                        "sub-R kernel parameter exceeds uint32: rFactor=%lu tailLen=%lu chunksPerGroup=%lu "
                        "tailChunks=%lu",
                        rFactor, tailLen, chunksPerGroup, tailChunks),
                return false);

    td_.numN = a1;
    td_.numC = a0;
    td_.numR = r;
    td_.rAlign = rAlign;
    td_.cInner = 1;
    td_.cOuter = 1;
    td_.cTail = a0;
    td_.binaryAddQuotient = binAddQuotient;
    td_.perCoreCnt = static_cast<int64_t>(perCoreCnt);
    td_.isSubRTiling = 1;
    td_.rFactor = rFactor;
    td_.numChunks = numChunks;
    td_.tailLen = tailLen;
    td_.chunksPerGroup = chunksPerGroup;
    td_.numGroups = numGroups;
    td_.tailChunks = tailChunks;
    td_.totalRows = static_cast<int64_t>(totalRows_);
    td_.totalElements = static_cast<int64_t>(totalElements_);
    td_.totalTiles = static_cast<int64_t>(totalRows_);
    td_.inputBufferBytes = Ops::Base::CeilAlign(rFactor * static_cast<uint64_t>(elemSize),
                                                static_cast<uint64_t>(ubBlockSize));
    td_.outputBufferBytes = Ops::Base::CeilAlign(sizeof(float), static_cast<uint64_t>(ubBlockSize));
    td_.scratchBufferBytes = 0;
    const uint64_t partialSlots = chunksPerGroup + ((numGroups > 1) ? 1UL : 0UL);
    td_.partialBufferBytes = Ops::Base::CeilAlign(partialSlots, static_cast<uint64_t>(vlfp32)) * sizeof(float) *
                             ACC_COMPONENTS;
    return true;
}

ge::graphStatus INTrainingReduceV2ARFullReduceTiling::DoOpTiling() { return ge::GRAPH_SUCCESS; }

uint64_t INTrainingReduceV2ARFullReduceTiling::GetTilingKey() const
{
    return td_.isSubRTiling == IN_TRAINING_REDUCE_V2_SPLIT_R_MODE ? TILINGKEY_SPLIT_R : TILINGKEY_AR_FULL_REDUCE;
}

ge::graphStatus INTrainingReduceV2ARFullReduceTiling::PostTiling()
{
    OP_CHECK_IF(context_->SetBlockDim(static_cast<uint32_t>(blockNum_)) != ge::GRAPH_SUCCESS,
                OP_LOGE(GetNodeName(), "Set blockDim failed"), return ge::GRAPH_FAILED);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    if (td_.isSubRTiling == IN_TRAINING_REDUCE_V2_SPLIT_R_MODE) {
        // All participating cores must be scheduled together for hardware SyncAll.
        OP_CHECK_IF(context_->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                    OP_LOGE(GetNodeName(), "Failed to set batch schedule mode for split-R"), return ge::GRAPH_FAILED);
        currentWorkspace[0] += static_cast<uint64_t>(td_.totalTiles) * static_cast<uint64_t>(ubBlockSize);
    }
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    OP_CHECK_IF(rawTilingData->GetCapacity() < sizeof(td_),
                OP_LOGE(GetNodeName(), "actual tiling data size %zu > context tiling data size %zu", sizeof(td_),
                        rawTilingData->GetCapacity()),
                return ge::GRAPH_FAILED);
    auto capSize = rawTilingData->GetCapacity();
    void* ptrData = rawTilingData->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrData);
    void* ptrStruct = static_cast<void*>(&td_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrStruct);
    OP_CHECK_IF(memcpy_s(ptrData, capSize, ptrStruct, sizeof(td_)) != 0,
                OP_LOGE(GetNodeName(), "Failed to copy tiling data."), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(td_));
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(INTrainingReduceV2, INTrainingReduceV2ARFullReduceTiling,
                             IN_TRAINING_REDUCE_V2_AR_FULL_REDUCE_PRIORITY);
} // namespace optiling
