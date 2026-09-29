/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_tiling_arch35.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_ARCH35_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_ARCH35_H_

#include <limits>
#include "register/op_impl_registry.h"
#include "add_rms_norm_tiling.h"
#include "op_common/op_host/util/platform_util.h"

namespace optiling {
namespace addRmsNormRegbase {
constexpr uint32_t UINT64_BIT_LEN = std::numeric_limits<uint64_t>::digits;
constexpr uint32_t DTYPE_KEY_FP16 = 1;
constexpr uint32_t DTYPE_KEY_FP32 = 2;
constexpr uint32_t DTYPE_KEY_BF16 = 3;
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t GAMMA_INDEX = 2;
constexpr uint32_t FLOAT_BYTE_SIZE = sizeof(float);
constexpr uint32_t UB_USED = 1024;
constexpr uint32_t UB_RESERVE_FOR_RSTDALIGN = 1024;
constexpr uint32_t MODE_NORMAL = 1000;
constexpr uint32_t MODE_SPLIT_D = 2000;
constexpr uint32_t MODE_SPLIT_AR = 3000;
constexpr uint32_t MODE_TRANSPOSE = 4000;
constexpr uint32_t MODE_REDUCE_EMPTY = 5000;
constexpr int32_t BATCH_INVARIANT_LEVEL = 3;
constexpr uint32_t BATCH_MODE_SCHEDULE = 1;
constexpr uint32_t QUE_NUM = 5;
constexpr uint32_t QUE_MODE_NORMAL_NUM = 4;
constexpr uint64_t ALING_FACTOR_256 = 256;
constexpr uint64_t ALING_FACTOR_512 = 512;
constexpr uint32_t RETAINED_SIZE = 5120; // 256 * 5 * 4;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2;
constexpr uint32_t MULTI_FACTOR_2 = 2;
constexpr uint32_t NUM_2 = 2;
constexpr uint32_t NDDMA_BETTER_STAGE = 512;
// Trans 仅保留每核超过 8 个 FP32 VL 的大 A 性能区。
constexpr uint32_t TRANS_PHYSICAL_LARGE_MIN_VL_PER_CORE = 8;
// SplitAR 至少保留两路 R 向并行，同一常量也约束 A 的准入上界。
constexpr uint32_t SPLIT_AR_MIN_PARALLEL_R_BLOCKS = 2;
// 静态双缓冲的性能准入：每核跨 A 累计至少 8 次 UB tile 迭代。
constexpr uint32_t SPLIT_AR_MIN_DB_TILES_PER_CORE = 8;
constexpr uint32_t AR_CACHE_LEVEL_COUNT = std::numeric_limits<uint64_t>::digits;
constexpr uint64_t DEFAULT_USER_WORKSPACE_SIZE = 256;

const std::map<ge::DataType, uint32_t> dTypeByteMap = {
    {ge::DT_FLOAT16, sizeof(uint16_t)},
    {ge::DT_FLOAT, sizeof(float)},
    {ge::DT_BF16, sizeof(uint16_t)},
};

template <typename T>
auto CeilDiv(T x, T y) -> T
{
    return y == 0 ? x : x / y + static_cast<T>(x % y != 0);
}

ge::graphStatus SetRegbaseWorkspaceSize(gert::TilingContext* context, uint64_t userWorkspaceSize)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    // 系统 workspace 大小由平台接口提供。
    const uint64_t systemWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    constexpr uint64_t maxWorkspaceSize = std::numeric_limits<size_t>::max();
    OP_CHECK_IF(systemWorkspaceSize > maxWorkspaceSize || userWorkspaceSize > maxWorkspaceSize - systemWorkspaceSize,
                OP_LOGE(context, "The total workspace size exceeds the size_t range."), return ge::GRAPH_FAILED);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(systemWorkspaceSize + userWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

bool IsTransposeCapable(uint64_t numRow, uint64_t numCol, uint32_t curElementByte, uint64_t ubBlockSize)
{
    // 这里只判断短 R 转置能力，A 向性能门槛由后续函数判断。
    if (numRow == 0 || numCol == 0 || curElementByte == 0 || ubBlockSize == 0) {
        return false;
    }
    uint64_t maxR = ubBlockSize / curElementByte;
    return maxR != 0 && numCol <= maxR;
}

bool IsPhysicalTransposePreferred(uint64_t blockFactor, uint64_t vlfp32)
{
    if (blockFactor == 0 || vlfp32 == 0 ||
        vlfp32 > std::numeric_limits<uint64_t>::max() / TRANS_PHYSICAL_LARGE_MIN_VL_PER_CORE) {
        return false;
    }
    uint64_t minRowsPerCore = vlfp32 * TRANS_PHYSICAL_LARGE_MIN_VL_PER_CORE;
    return blockFactor > minRowsPerCore;
}

uint64_t GetBinAddQuotient(uint64_t reduceLen)
{
    // 与 RFullLoad 保持一致：2 次幂长度取一半，其余取最高 2 次幂。
    uint64_t foldPoint = reduceLen == 0 ? 1 : (1ULL << (UINT64_BIT_LEN - 1 - __builtin_clzll(reduceLen)));
    return foldPoint == reduceLen ? foldPoint / NUM_2 : foldPoint;
}

void GetInterTileBinaryTiling(uint64_t reduceLen, uint64_t ubFactor, uint64_t& basicBlockLoop, uint64_t& mainFoldCount)
{
    // 推导跨 UB tile 二分折叠的主循环和尾块数。
    uint64_t blockCountCeil = CeilDiv(reduceLen, ubFactor);
    uint64_t blockCountFloor = reduceLen / ubFactor;
    basicBlockLoop = blockCountCeil <= 1 ? 0 : 1ULL << (UINT64_BIT_LEN - 1 - __builtin_clzll(blockCountCeil - 1));
    mainFoldCount = blockCountFloor - basicBlockLoop;
}

void SetDtypeKey(ge::DataType dataType, uint32_t& dtypeKey)
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            dtypeKey = DTYPE_KEY_FP16;
            break;
        case ge::DT_BF16:
            dtypeKey = DTYPE_KEY_BF16;
            break;
        default:
            dtypeKey = DTYPE_KEY_FP32;
            break;
    }
}

uint32_t ComputeTotalBufSize(uint32_t bufferNum, ge::DataType dtype, uint32_t dtypeSize, uint32_t length, bool split,
                             uint64_t vlfp32)
{
    // queBuferSize: 计算搬运需要空间大小
    uint32_t queBufSize = bufferNum * length * dtypeSize * QUE_NUM + vlfp32 * bufferNum * FLOAT_BYTE_SIZE;
    uint32_t tmpBufSzie = 0; // tmpBufSzie: UB内需要临时空间大小
    if (split) {
        // 切分场景下
        tmpBufSzie = (dtype == ge::DT_FLOAT) ? 0 : length * FLOAT_BYTE_SIZE * NUM_2;
    } else {
        // 普通场景下：如果是float16及bfloat16数据类型，需要一块：转FP32
        tmpBufSzie = length * FLOAT_BYTE_SIZE;
    }
    return queBufSize + tmpBufSzie + RETAINED_SIZE;
}

ge::graphStatus TilingReduceEmpty(gert::TilingContext* context, uint64_t numRow, uint32_t numCore, uint64_t ubSize,
                                  uint64_t ubBlockSize, float epsilon)
{
    // 空归约仅需写 A 个 rstd，按每核至少 32 KiB 输出控制启动开销。
    constexpr uint64_t singleCoreMinBytes = 32UL * 1024UL;
    constexpr uint64_t floatBytes = sizeof(float);
    OP_CHECK_IF(numCore == 0, OP_LOGE(context, "The number of AIV cores cannot be zero."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(numRow > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                OP_LOGE(context, "numRow exceeds the int64_t range."), return ge::GRAPH_FAILED);
    uint64_t elemsPerBlock = ubBlockSize / floatBytes;
    OP_CHECK_IF(elemsPerBlock == 0, OP_LOGE(context, "The UB block size is smaller than one float element."),
                return ge::GRAPH_FAILED);
    uint64_t rowsPerMinBlock = singleCoreMinBytes / floatBytes;
    uint64_t blockNum = std::min(CeilDiv(numRow, rowsPerMinBlock), static_cast<uint64_t>(numCore));
    blockNum = std::max(blockNum, 1UL);
    uint64_t usedCoreNum = blockNum;
    uint64_t rowsPerCore = CeilDiv(numRow, usedCoreNum);
    uint64_t rowsPerTailCore = numRow - (usedCoreNum - 1) * rowsPerCore;
    uint64_t tailCoreStartIndex = usedCoreNum - 1;
    uint64_t perLoopMax = ubSize / floatBytes;
    uint64_t chunk = std::min(perLoopMax, rowsPerCore);
    // 常规循环按 UB 数据块对齐，最后一轮由 DataCopyPad 精确处理尾部。
    uint64_t rowsPerLoop = chunk / elemsPerBlock * elemsPerBlock;
    if (rowsPerLoop == 0) {
        rowsPerLoop = rowsPerCore;
    }

    AddRMSNormRegbaseReduceEmptyTilingData tiling;
    tiling.set_numRow(numRow);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_rowsPerCore(rowsPerCore);
    tiling.set_rowsPerTailCore(rowsPerTailCore);
    tiling.set_tailCoreStartIndex(tailCoreStartIndex);
    tiling.set_rowsPerLoop(rowsPerLoop);
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    OP_LOGI(context,
            "TilingData(ReduceEmpty) numRow: %lu, usedCoreNum: %lu, rowsPerCore: %lu, rowsPerTailCore: %lu, "
            "tailCoreStartIndex: %lu, rowsPerLoop: %lu, epsilon: %f",
            numRow, usedCoreNum, rowsPerCore, rowsPerTailCore, tailCoreStartIndex, rowsPerLoop, epsilon);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->SetTilingKey(MODE_REDUCE_EMPTY);
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingTranspose(gert::TilingContext* context, uint64_t numRow, uint64_t numCol, uint32_t numCore,
                                uint64_t ubSize, uint32_t curElementByte, uint64_t ubBlockSize, uint64_t vlfp32,
                                float epsilon, float avgFactor)
{
    // UB 中转为 [R,A]，GM 搬运仍使用真实 R。
    uint64_t rTileBase = ubBlockSize / curElementByte;
    uint64_t aTileAlign = ubBlockSize / sizeof(uint16_t);
    if (rTileBase == 0 || aTileAlign == 0) {
        return ge::GRAPH_FAILED;
    }
    uint64_t rAligned = CeilDiv(numCol, rTileBase) * rTileBase;
    // 按 Kernel buffer 清单反推单个 A tile 上限。
    uint64_t fixedBufSize = rAligned * (curElementByte + FLOAT_BYTE_SIZE); // gammaQueue + gamma fp32
    uint64_t perARowBufSize = rAligned * (curElementByte * DOUBLE_BUFFER_NUM * QUE_MODE_NORMAL_NUM + FLOAT_BYTE_SIZE) +
                              FLOAT_BYTE_SIZE * DOUBLE_BUFFER_NUM;
    if (ubSize <= fixedBufSize) {
        return ge::GRAPH_FAILED;
    }
    uint64_t maxTileALen = (ubSize - fixedBufSize) / perARowBufSize;
    maxTileALen = maxTileALen / aTileAlign * aTileAlign;
    // repeatTimes 为 uint8_t，超出部分由 Kernel 外层循环处理。
    maxTileALen = std::min(maxTileALen, static_cast<uint64_t>(std::numeric_limits<uint8_t>::max()) * aTileAlign);
    if (maxTileALen < aTileAlign) {
        return ge::GRAPH_FAILED;
    }

    // 每核约分配两个 FP32 VL 的 A 行。
    uint64_t targetRowsPerCore = vlfp32 * NUM_2;
    uint64_t usedCoreNum64 = CeilDiv(numRow, targetRowsPerCore);
    usedCoreNum64 = std::max(1UL, std::min({usedCoreNum64, numRow, static_cast<uint64_t>(numCore)}));
    uint32_t usedCoreNum = static_cast<uint32_t>(usedCoreNum64);
    uint64_t rowsPerCoreMax = CeilDiv(numRow, static_cast<uint64_t>(usedCoreNum));
    // R=一个 B16 block 时保留至少两个 tile 的流水机会。
    uint64_t targetTilesPerCore = (numCol == aTileAlign && rowsPerCoreMax > targetRowsPerCore) ? NUM_2 : 1;
    targetTilesPerCore = std::max(targetTilesPerCore, CeilDiv(rowsPerCoreMax, maxTileALen));
    uint64_t tileALen = CeilDiv(rowsPerCoreMax, targetTilesPerCore);
    tileALen = CeilDiv(tileALen, aTileAlign) * aTileAlign;
    // 短 R 时保留至少两个 FP32 VL 的 A 轴展开空间。
    if (numCol < aTileAlign && tileALen <= vlfp32 && maxTileALen >= vlfp32 * NUM_2) {
        tileALen = vlfp32 * NUM_2;
    }
    tileALen = std::min(tileALen, maxTileALen);
    uint64_t aOuter = CeilDiv(numRow, tileALen);
    uint64_t tileATail = numRow - tileALen * (aOuter - 1);
    uint64_t tilesPerCore = CeilDiv(rowsPerCoreMax, tileALen);

    AddRMSNormRegbaseTransTilingData tiling;
    tiling.set_numRow(numRow);
    tiling.set_numCol(numCol);
    tiling.set_rAligned(rAligned);
    tiling.set_rTileBase(rTileBase);
    tiling.set_totalTiles(aOuter);
    tiling.set_tilesPerCore(tilesPerCore);
    tiling.set_tileALen(tileALen);
    tiling.set_tileATail(tileATail);
    tiling.set_epsilon(epsilon);
    tiling.set_avgFactor(avgFactor);
    context->SetBlockDim(usedCoreNum);
    OP_LOGI(context,
            "TransTiling numCore: %u, numRow: %lu, numCol: %lu, rAligned: %lu, rTileBase: %lu, "
            "tileALen: %lu, tileATail: %lu, totalTiles: %lu, tilesPerCore: %lu, usedCoreNum: %u, "
            "targetRowsPerCore: %lu",
            numCore, numRow, numCol, rAligned, rTileBase, tileALen, tileATail, aOuter, tilesPerCore, usedCoreNum,
            targetRowsPerCore);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->SetTilingKey(MODE_TRANSPOSE);
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingSplitD(gert::TilingContext* context, uint64_t numRow, uint64_t numCol, uint32_t numCore,
                             uint64_t ubSize, ge::DataType dataType, uint32_t curElementByte, uint64_t vlfp32,
                             float epsilon, float avgFactor)
{
    constexpr uint64_t uint32Max = std::numeric_limits<uint32_t>::max();
    OP_CHECK_IF(numRow == 0 || numCol == 0 || numCore == 0 || curElementByte == 0,
                OP_LOGE(context, "SplitD received an invalid zero dimension or platform parameter."),
                return ge::GRAPH_FAILED);
    // key=2000 的旧 TilingData 和 Kernel 均使用 uint32_t；超出其表示范围时明确失败，
    // 避免静默截断或 Host 侧倍增循环回绕。正常模式的超大 R 由 uint64_t 的 SplitAR 路径承接。
    OP_CHECK_IF(numRow > uint32Max || numCol > uint32Max,
                OP_LOGE(context, "SplitD only supports numRow and numCol within the uint32_t range."),
                return ge::GRAPH_FAILED);
    uint64_t alignElements = ALING_FACTOR_512 / curElementByte;
    OP_CHECK_IF(alignElements == 0, OP_LOGE(context, "The 512-byte alignment cannot hold one input element."),
                return ge::GRAPH_FAILED);
    uint64_t numColAlign = CeilDiv(numCol, alignElements) * alignElements;
    OP_CHECK_IF(numColAlign > uint32Max,
                OP_LOGE(context, "The aligned SplitD reduction length exceeds the uint32_t range."),
                return ge::GRAPH_FAILED);
    uint64_t rowFactor = vlfp32;
    uint64_t ubFactor = 1U;
    uint64_t ubLoop = 1U;
    uint64_t colBuferLength{0};
    uint64_t blockFactor = CeilDiv(numRow, static_cast<uint64_t>(numCore));
    uint64_t useCoreNum = CeilDiv(numRow, blockFactor);
    while (ubFactor <= uint32Max / MULTI_FACTOR_2 &&
           ComputeTotalBufSize(DOUBLE_BUFFER_NUM, dataType, curElementByte,
                               static_cast<uint32_t>(ubFactor * MULTI_FACTOR_2), true, vlfp32) < ubSize) {
        ubFactor *= MULTI_FACTOR_2;
    }
    if (ubFactor > numCol) {
        ubFactor = numCol;
    }
    uint64_t loopCapacity = numCol / ubFactor;
    while (ubLoop <= loopCapacity / MULTI_FACTOR_2) {
        ubLoop *= MULTI_FACTOR_2;
    }
    colBuferLength = ubFactor;
    OP_CHECK_IF(blockFactor > uint32Max || rowFactor > uint32Max || ubFactor > uint32Max || ubLoop > uint32Max ||
                    colBuferLength > uint32Max || useCoreNum > numCore,
                OP_LOGE(context, "SplitD tiling parameters exceed their supported range."), return ge::GRAPH_FAILED);
    uint32_t isNddma = numCol >= NDDMA_BETTER_STAGE ? 0U : 1U;

    AddRMSNormRegbaseTilingData tiling;
    tiling.set_numRow(static_cast<uint32_t>(numRow));
    tiling.set_numCol(static_cast<uint32_t>(numCol));
    tiling.set_numColAlign(static_cast<uint32_t>(numColAlign));
    tiling.set_blockFactor(static_cast<uint32_t>(blockFactor));
    tiling.set_rowFactor(static_cast<uint32_t>(rowFactor));
    tiling.set_ubFactor(static_cast<uint32_t>(ubFactor));
    tiling.set_epsilon(epsilon);
    tiling.set_avgFactor(avgFactor);
    tiling.set_ubLoop(static_cast<uint32_t>(ubLoop));
    tiling.set_colBuferLength(static_cast<uint32_t>(colBuferLength));
    tiling.set_multiNNum(0);
    tiling.set_isNddma(isNddma);
    OP_LOGI(context,
            "TilingData(SplitD) numCore: %u, ubSize: %lu, numRow: %lu, numCol: %lu, numColAlign: %lu, "
            "colBuferLength: %lu, blockFactor: %lu, rowFactor: %lu, ubFactor: %lu, "
            "epsilon: %f, avgFactor: %f, ubLoop: %u, isNddma: %u.",
            numCore, ubSize, numRow, numCol, numColAlign, colBuferLength, blockFactor, rowFactor, ubFactor,
            tiling.get_epsilon(), tiling.get_avgFactor(), tiling.get_ubLoop(), tiling.get_isNddma());

    context->SetBlockDim(static_cast<uint32_t>(useCoreNum));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->SetTilingKey(MODE_SPLIT_D);
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingSplitAR(gert::TilingContext* context, uint64_t numRow, uint64_t numCol, uint32_t numCore,
                              uint64_t ubSize, uint32_t curElementByte, uint32_t dataPerBlock, uint64_t ubBlockSize,
                              uint64_t vlfp32, uint64_t binaryAddElementMaxLen, float epsilon, float avgFactor)
{
    if (numRow == 0 || numCore == 0) {
        return ge::GRAPH_FAILED;
    }
    // 每核固定一段 R，并在核内遍历全部 A 行。
    uint64_t blockDimCap = numCore;
    // 核间 partial 由一个 VL 完成归并，rBlockNum 不超过 VL_FP32。
    uint64_t maxRBlockNum = std::min(blockDimCap, vlfp32);
    if (maxRBlockNum == 0) {
        return ge::GRAPH_FAILED;
    }
    uint64_t numRowAlign = Ops::Base::CeilAlign(numRow, ubBlockSize / FLOAT_BYTE_SIZE);
    // 固定 UB tile 循环处理任意大 R，归并源区尾部按 VL 对齐。
    uint64_t combElementCount = Ops::Base::CeilAlign(maxRBlockNum * numRowAlign, vlfp32);
    uint64_t combBufSize = combElementCount * FLOAT_BYTE_SIZE;
    uint64_t fixedBufSize = numRowAlign * FLOAT_BYTE_SIZE * NUM_2 + // partQueue + rstdQueue
                            ubBlockSize + combBufSize +             // foldPartBuf + combQueue
                            ubBlockSize * AR_CACHE_LEVEL_COUNT +    // cacheBuf: 每层一个 UB block
                            vlfp32 * FLOAT_BYTE_SIZE;               // workBuf 额外区
    if (ubSize <= fixedBufSize) {
        return ge::GRAPH_FAILED;
    }

    uint64_t perElementBufSize = static_cast<uint64_t>(curElementByte) * QUE_NUM * DOUBLE_BUFFER_NUM +
                                 static_cast<uint64_t>(FLOAT_BYTE_SIZE) * NUM_2;
    uint64_t ubFactorMax = (ubSize - fixedBufSize) / perElementBufSize;
    // 单 tile 不超过块内二分归约支持的最大长度。
    ubFactorMax = std::min(ubFactorMax, binaryAddElementMaxLen);
    ubFactorMax = ubFactorMax / dataPerBlock * dataPerBlock;
    if (ubFactorMax == 0) {
        return ge::GRAPH_FAILED;
    }
    uint64_t ubFactor = 1ULL << (UINT64_BIT_LEN - 1 - __builtin_clzll(ubFactorMax));
    OP_CHECK_IF(ubFactor > std::numeric_limits<uint64_t>::max() / SPLIT_AR_MIN_DB_TILES_PER_CORE,
                OP_LOGE(context, "The SplitAR DB tile threshold overflows uint64."), return ge::GRAPH_FAILED);
    uint64_t perCoreWorkMin = ubFactor * SPLIT_AR_MIN_DB_TILES_PER_CORE;

    // 单行至少容纳两份“每核 8 tile”的工作量。
    uint64_t rBlocksPerRowByWork = numCol / perCoreWorkMin;
    if (rBlocksPerRowByWork < SPLIT_AR_MIN_PARALLEL_R_BLOCKS) {
        return ge::GRAPH_FAILED;
    }
    uint64_t rBlockNum = maxRBlockNum;
    if (rBlocksPerRowByWork < CeilDiv(maxRBlockNum, numRow)) {
        rBlockNum = rBlocksPerRowByWork * numRow;
    }
    uint64_t rBlockFactor = CeilDiv(numCol, rBlockNum);
    rBlockFactor = Ops::Base::CeilAlign(rBlockFactor, static_cast<uint64_t>(dataPerBlock));
    rBlockNum = CeilDiv(numCol, rBlockFactor);
    if (rBlockNum < SPLIT_AR_MIN_PARALLEL_R_BLOCKS || rBlockNum > maxRBlockNum) {
        return ge::GRAPH_FAILED;
    }
    uint64_t tailR = numCol - (rBlockNum - 1) * rBlockFactor;
    uint64_t useCoreNum = rBlockNum;

    uint64_t blockTailLen = rBlockFactor % ubFactor;
    blockTailLen = blockTailLen == 0 ? ubFactor : blockTailLen;
    uint64_t tailBlockTailLen = tailR % ubFactor;
    tailBlockTailLen = tailBlockTailLen == 0 ? ubFactor : tailBlockTailLen;
    uint64_t binAddQuotient = GetBinAddQuotient(ubFactor);
    uint64_t blockTailBinAddQuotient = GetBinAddQuotient(blockTailLen);
    uint64_t tailBinAddQuotient = GetBinAddQuotient(tailBlockTailLen);
    uint64_t basicBlockLoop = 0;
    uint64_t mainFoldCount = 0;
    uint64_t tailBasicBlockLoop = 0;
    uint64_t tailMainFoldCount = 0;
    GetInterTileBinaryTiling(rBlockFactor, ubFactor, basicBlockLoop, mainFoldCount);
    GetInterTileBinaryTiling(tailR, ubFactor, tailBasicBlockLoop, tailMainFoldCount);
    uint64_t actualTileCount = CeilDiv(rBlockFactor, ubFactor);
    uint64_t totalUbNeed = ubFactor * perElementBufSize + fixedBufSize;

    // workspace 按 [rBlockNum,numRowAlign] 存放各核对全部 A 行的 partial。
    OP_CHECK_IF(numRowAlign != 0 && useCoreNum > std::numeric_limits<uint64_t>::max() / numRowAlign,
                OP_LOGE(context, "The SplitAR workspace element count overflows uint64."), return ge::GRAPH_FAILED);
    const uint64_t usrWsElementCount = useCoreNum * numRowAlign;
    OP_CHECK_IF(usrWsElementCount > std::numeric_limits<uint64_t>::max() / sizeof(float),
                OP_LOGE(context, "The SplitAR user workspace size overflows uint64."), return ge::GRAPH_FAILED);
    const uint64_t usrWsSize = usrWsElementCount * sizeof(float);
    OP_CHECK_IF(context->SetScheduleMode(BATCH_MODE_SCHEDULE) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "Failed to set the schedule mode."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetRegbaseWorkspaceSize(context, usrWsSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "Failed to set the SplitAR workspace size."), return ge::GRAPH_FAILED);

    AddRMSNormRegbaseSplitARTilingData tiling;
    tiling.set_numRow(numRow);
    tiling.set_numCol(numCol);
    tiling.set_rBlockFactor(rBlockFactor);
    tiling.set_rBlockNum(rBlockNum);
    tiling.set_tailR(tailR);
    tiling.set_ubFactor(ubFactor);
    tiling.set_binAddQuotient(binAddQuotient);
    tiling.set_blockTailBinAddQuotient(blockTailBinAddQuotient);
    tiling.set_tailBinAddQuotient(tailBinAddQuotient);
    tiling.set_basicBlockLoop(basicBlockLoop);
    tiling.set_mainFoldCount(mainFoldCount);
    tiling.set_tailBasicBlockLoop(tailBasicBlockLoop);
    tiling.set_tailMainFoldCount(tailMainFoldCount);
    tiling.set_numRowAlign(numRowAlign);
    tiling.set_epsilon(epsilon);
    tiling.set_avgFactor(avgFactor);
    OP_LOGI(context,
            "TilingData(SplitAR) numCore: %u, numRow: %lu, numCol: %lu, maxRBlockNum: %lu, "
            "blockDimCap: %lu, perCoreWorkMin: %lu, "
            "rBlockFactor: %lu, rBlockNum: %lu, "
            "tailR: %lu, "
            "ubFactor: %lu, useCoreNum: %lu, "
            "bufferNum: %u, actualTileCount: %lu, binAddQuotient: %lu, "
            "blockTailBinAddQuotient: %lu, "
            "tailBinAddQuotient: %lu, basicBlockLoop: %lu, mainFoldCount: %lu, "
            "tailBasicBlockLoop: %lu, tailMainFoldCount: %lu, numRowAlign: %lu, "
            "totalUbNeed: %lu, wsSize: %lu",
            numCore, numRow, numCol, maxRBlockNum, blockDimCap, perCoreWorkMin, rBlockFactor, rBlockNum, tailR,
            ubFactor, useCoreNum, DOUBLE_BUFFER_NUM, actualTileCount, binAddQuotient, blockTailBinAddQuotient,
            tailBinAddQuotient, basicBlockLoop, mainFoldCount, tailBasicBlockLoop, tailMainFoldCount, numRowAlign,
            totalUbNeed, usrWsSize);
    context->SetBlockDim(static_cast<uint32_t>(useCoreNum));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->SetTilingKey(MODE_SPLIT_AR);
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingAddRmsNormRegbase(gert::TilingContext* context)
{
    OP_LOGD(context, " TilingAddRmsNormRegbase");
    auto ptrCompileInfo = reinterpret_cast<const AddRmsNormCompileInfo*>(context->GetCompileInfo());
    uint32_t numCore;
    uint64_t ubSize;
    if (ptrCompileInfo == nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        numCore = ascendcPlatform.GetCoreNumAiv();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    } else {
        numCore = ptrCompileInfo->totalCoreNum;
        ubSize = ptrCompileInfo->totalUbSize;
    }
    OP_CHECK_IF(numCore == 0, OP_LOGE(context, "The number of AIV cores cannot be zero."), return ge::GRAPH_FAILED);
    const gert::Shape xShape = context->GetInputShape(X_INDEX)->GetStorageShape();

    const gert::Shape gammaShape = context->GetInputShape(GAMMA_INDEX)->GetStorageShape();
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* epsilon = attrs->GetFloat(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, epsilon);
    OP_CHECK_IF(*epsilon < 0, OP_LOGE(context, "epsilon must be nonnegative, but got %f.", *epsilon),
                return ge::GRAPH_FAILED);
    int64_t gammaSize = gammaShape.GetShapeSize();
    int64_t xSize = xShape.GetShapeSize();
    OP_CHECK_IF(gammaSize < 0 || xSize < 0,
                OP_LOGE(context, "The input shape size exceeds the int64_t range or contains an invalid dim."),
                return ge::GRAPH_FAILED);
    uint64_t numCol = static_cast<uint64_t>(gammaSize);
    float avgFactor = (numCol == 0U) ? 0.0f : 1.0f / static_cast<float>(numCol);
    size_t xDimNum = xShape.GetDimNum();
    size_t gammaDimNum = gammaShape.GetDimNum();
    OP_CHECK_IF(xDimNum < gammaDimNum, OP_LOGE(context, "The rank of x cannot be smaller than the rank of gamma."),
                return ge::GRAPH_FAILED);
    uint64_t numRow = 1;
    for (size_t i = 0; i < xDimNum - gammaDimNum; i++) {
        int64_t dim = xShape.GetDim(i);
        OP_CHECK_IF(dim < 0, OP_LOGE(context, "The input shape contains an invalid dim."), return ge::GRAPH_FAILED);
        uint64_t dimValue = static_cast<uint64_t>(dim);
        OP_CHECK_IF(dimValue != 0 && numRow > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / dimValue,
                    OP_LOGE(context, "The outer dimension product exceeds the int64_t range."),
                    return ge::GRAPH_FAILED);
        numRow *= dimValue;
    }
    for (size_t i = 0; i < xDimNum; i++) {
        OP_LOGD(context, " TilingAddRmsNormRegbase x shape:%ld", xShape.GetDim(i));
    }
    for (size_t i = 0; i < gammaDimNum; i++) {
        OP_LOGD(context, " TilingAddRmsNormRegbase gamma shape:%ld", gammaShape.GetDim(i));
    }
    auto dataType = context->GetInputDesc(0)->GetDataType();
    uint32_t dtypeKey = DTYPE_KEY_FP16;
    OP_CHECK_IF(SetRegbaseWorkspaceSize(context, DEFAULT_USER_WORKSPACE_SIZE) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "Failed to set the default workspace size."), return ge::GRAPH_FAILED);
    uint64_t numColAlign = 0;
    uint64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    uint64_t ubfp32 = ubBlockSize / sizeof(float);
    uint64_t vlfp32 = Ops::Base::GetVRegSize(context) / sizeof(float);
    OP_CHECK_IF(ubBlockSize == 0 || ubfp32 == 0 || vlfp32 < NUM_2,
                OP_LOGE(context, "Invalid UB block size or vector length."), return ge::GRAPH_FAILED);
    uint64_t binaryAddElementMaxLen = vlfp32 * vlfp32 * NUM_2 * NUM_2;

    // 空张量(reduce empty)优先: 归一化维 R==0 且外维 A>0, 只有 rstd 输出需兜底
    if (numCol == 0 && numRow > 0) {
        if (TilingReduceEmpty(context, numRow, numCore, ubSize, ubBlockSize, *epsilon) == ge::GRAPH_SUCCESS) {
            return ge::GRAPH_SUCCESS;
        }
    }
    uint64_t blockFactor;
    uint64_t rowFactor = 0;

    bool enableBatchInvariant = (context->GetDeterministicLevel() == BATCH_INVARIANT_LEVEL);
    OP_LOGD(context, " TilingAddRmsNormRegbase enableBatchInvariant: %d", enableBatchInvariant);

    OP_CHECK_IF(ubSize <= UB_USED, OP_LOGE(context, "The available UB size is insufficient."), return ge::GRAPH_FAILED);
    ubSize = ubSize - UB_USED;
    SetDtypeKey(dataType, dtypeKey);

    blockFactor = 1UL;
    uint64_t tileNum = CeilDiv(numRow, static_cast<uint64_t>(numCore));
    blockFactor *= tileNum;
    uint32_t useCoreNum = static_cast<uint32_t>(CeilDiv(numRow, blockFactor));
    context->SetBlockDim(useCoreNum);

    auto dtypeByteIterator = dTypeByteMap.find(dataType);
    OP_CHECK_IF(dtypeByteIterator == dTypeByteMap.end(), OP_LOGE(context, "Unsupported input data type."),
                return ge::GRAPH_FAILED);
    uint32_t curElementByte = dtypeByteIterator->second;
    uint32_t dataPerBlock = static_cast<uint32_t>(ubBlockSize / curElementByte);
    OP_CHECK_IF(dataPerBlock == 0, OP_LOGE(context, "The UB block cannot hold one input element."),
                return ge::GRAPH_FAILED);
    // 先按元素数做向上对齐，避免 R==INT64_MAX 时 R*sizeof(T) 中间值溢出。
    numColAlign = CeilDiv(numCol, static_cast<uint64_t>(dataPerBlock)) * dataPerBlock;

    uint64_t binAddQuotient = 0;
    // 只在 R 可全载时计算整行 UB 用量；大 R 直接留给分块模板，避免字节数乘法溢出。
    if (numColAlign <= binaryAddElementMaxLen) {
        binAddQuotient = GetBinAddQuotient(numColAlign);
        uint64_t binAddBufferOneline = Ops::Base::CeilAlign(CeilDiv(binAddQuotient, vlfp32), ubfp32);
        uint64_t gammaBytes = numColAlign * curElementByte;
        if (ubSize > UB_RESERVE_FOR_RSTDALIGN + gammaBytes) {
            uint64_t tmpSize = ubSize - UB_RESERVE_FOR_RSTDALIGN - gammaBytes;
            uint64_t perRowBytes = gammaBytes * DOUBLE_BUFFER_NUM * QUE_MODE_NORMAL_NUM + numColAlign * sizeof(float) +
                                   sizeof(float) * (DOUBLE_BUFFER_NUM + 1) + binAddBufferOneline * sizeof(float);
            if (perRowBytes != 0) {
                rowFactor = tmpSize / perRowBytes;
            }
        }
    }
    // 物理短 R 且 A 超过大 A 门槛时选择 Trans。
    if (!enableBatchInvariant && IsTransposeCapable(numRow, numCol, curElementByte, ubBlockSize) &&
        IsPhysicalTransposePreferred(blockFactor, vlfp32)) {
        if (TilingTranspose(context, numRow, numCol, numCore, ubSize, curElementByte, ubBlockSize, vlfp32, *epsilon,
                            avgFactor) == ge::GRAPH_SUCCESS) {
            return ge::GRAPH_SUCCESS;
        }
    }
    if (rowFactor >= 1) {
        // R能够全载
        rowFactor = std::min(rowFactor, blockFactor);
        AddRMSNormRegbaseRFullLoadTilingData tiling;
        tiling.set_numRow(numRow);
        tiling.set_numCol(numCol);
        tiling.set_numColAlign(numColAlign);
        tiling.set_blockFactor(blockFactor);
        tiling.set_rowFactor(rowFactor);
        tiling.set_binAddQuotient(binAddQuotient);
        tiling.set_epsilon(*epsilon);
        tiling.set_avgFactor(avgFactor);
        OP_LOGI(context,
                "TilingData numCore: %u, ubSize: %lu, numRow: %lu, numCol: %lu, numColAlign: %lu, "
                "blockFactor: %lu, rowFactor: %lu, binAddQuotient: %lu, "
                "epsilon: %f, avgFactor: %f",
                numCore, ubSize, tiling.get_numRow(), tiling.get_numCol(), tiling.get_numColAlign(),
                tiling.get_blockFactor(), tiling.get_rowFactor(), tiling.get_binAddQuotient(), tiling.get_epsilon(),
                tiling.get_avgFactor());
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->SetTilingKey(MODE_NORMAL);
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }
    // RFullLoad 无法全载后，仅在 A 和 R 工作量满足性能门槛时尝试 SplitAR。
    if (!enableBatchInvariant && numRow > 0 && numRow <= numCore / SPLIT_AR_MIN_PARALLEL_R_BLOCKS) {
        if (TilingSplitAR(context, numRow, numCol, numCore, ubSize, curElementByte, dataPerBlock, ubBlockSize, vlfp32,
                          binaryAddElementMaxLen, *epsilon, avgFactor) == ge::GRAPH_SUCCESS) {
            return ge::GRAPH_SUCCESS;
        }
    }
    // 切 R fallback
    return TilingSplitD(context, numRow, numCol, numCore, ubSize, dataType, curElementByte, vlfp32, *epsilon,
                        avgFactor);
}
} // namespace addRmsNormRegbase
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_ARCH35_H_
