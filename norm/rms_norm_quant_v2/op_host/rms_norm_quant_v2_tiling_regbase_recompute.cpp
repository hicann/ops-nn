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
 * \file rms_norm_quant_v2_tiling_regbase_split_r.cpp
 * \brief
 */

#include "norm/norm_common/op_host/norm_tiling_check_common.h"
#include "rms_norm_quant_v2_tiling.h"
#include "util/math_util.h"
#include "op_api/op_util.h"

using namespace Ops::Base;
using namespace ge;

namespace optiling {
constexpr int64_t R_MAX_VALUE = 16384;
constexpr uint32_t CONST_ZERO = 0;
constexpr uint32_t CONST_ONE = 1;
constexpr uint32_t CONST_TWO = 2;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t RETAINED_SIZE_256 = 256;
constexpr uint32_t ULONG_BIT_LEN = 64;

bool RmsNormQuantV2RegbaseTilingRecompute::IsCapable()
{
    // ub 间二分累加算法，需要 r 至少可以折叠一次
    return tilingParams.r >= baseN;
}

int64_t RmsNormQuantV2RegbaseTilingRecompute::GetPowerSplit(int64_t numN)
{
    uint32_t ubLoops = Ops::Base::CeilDiv(tilingParams.r, static_cast<int64_t>(numN));
    int64_t powerSplit = ubLoops == 0 ? 1 : (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(ubLoops)));
    powerSplit = (powerSplit == ubLoops) ? powerSplit / CONST_TWO : powerSplit;
    if (ubLoops == 1) {
        powerSplit = 1;
    }
    return powerSplit;
}

int64_t RmsNormQuantV2RegbaseTilingRecompute::GetCacheID(const int64_t idx)
{
    return __builtin_popcountll(idx ^ (idx + CONST_ONE)) - CONST_ONE;
}

ge::graphStatus RmsNormQuantV2RegbaseTilingRecompute::DoOpTiling()
{
    uint32_t vlfp32 = tilingParams.vecLength;
    int64_t binaryAddElemtMaxLen = vlfp32 * vlfp32 * CONST_TWO * CONST_TWO; // 整块ub二分累加支持的最大长度

    int64_t betaNum = tilingParams.hasBeta ? CONST_ONE : CONST_ZERO;
    int64_t scalesNum = tilingParams.hasScales2 ? CONST_TWO : CONST_ONE;
    int64_t zeroPointsNum =
        (tilingParams.hasZeroPoints1 ? CONST_ONE : CONST_ZERO) + (tilingParams.hasZeroPoints2 ? CONST_ONE : CONST_ZERO);
    int64_t yNum = tilingParams.hasY2 ? CONST_TWO : CONST_ONE;

    // ub 内存分配情况：
    // 1. 搬入搬出需要的 ub
    //  x, xFold, gamma, beta:          baseN * tilingParams.xDtypeSize * (betaNum + 3)
    //  sacles1, scales2:               baseN * tilingParams.scaleDtypeSize * scalesNum
    //  zero_points1, zero_points2      baseN * tilingParams.zeroPointDtypeSize * zeroPointsNum
    //  y1, y2:                         baseN * yNum * sizeof(uint8)
    // 2. 临时缓存:
    // rstdBuf_:                        Aligned(baseM_ * sizeof(float), BLOCK_SIZE)
    // cacheBuf_:                       Aligned(resultCacheID_ * BLOCK_SIZE, BLOCK_SIZE)
    // binaryAddBuf_:                   Aligned(binAddQuotient_ * sizeof(float), BLOCK_SIZE)
    // xFp32TmpBuf_:                    Aligned(baseN_ * sizeof(float), BLOCK_SIZE)
    int64_t rstdBufSize = CeilDiv(static_cast<int64_t>(baseM * sizeof(float)), tilingParams.ubBlockSize) * tilingParams.ubBlockSize;
    // 当 rstdFlag=1 时，rstd 使用 outQueue(Double Buffer) 而非 TBuf，需要额外一倍空间
    if (tilingParams.rstdFlag != 0) {
        rstdBufSize = rstdBufSize * DOUBLE_BUFFER;
    }
    int64_t binaryAddBufSize = tilingParams.vecLength * CONST_TWO * sizeof(float);

    auto GetMaxBaseN = [=](int64_t initialN) -> int64_t {
        int64_t powerSplit = GetPowerSplit(initialN);
        int64_t cacheBuffSize = (GetCacheID(powerSplit - 1) + 1) * tilingParams.ubBlockSize;
        while (powerSplit > 1 && 2 * initialN <= binaryAddElemtMaxLen &&
            static_cast<uint64_t>(2 * initialN * ((tilingParams.xDtypeSize * (betaNum + 3) + tilingParams.scaleDtypeSize * scalesNum +
                    tilingParams.zeroPointDtypeSize * zeroPointsNum + yNum) * DOUBLE_BUFFER + sizeof(float))) <=
                static_cast<uint64_t>(tilingParams.maxUbSize - RETAINED_SIZE_256 - (rstdBufSize + cacheBuffSize + binaryAddBufSize))) {
            initialN = 2 * initialN;
            powerSplit = GetPowerSplit(initialN);
            cacheBuffSize = (GetCacheID(powerSplit - 1) + 1) * tilingParams.ubBlockSize;
        }
        return initialN;
    };

    // 计算 baseN 的最大值
    baseN = GetMaxBaseN(baseN);
    // A切多核
    tilingParams.blockFactor = Ops::Base::CeilDiv(tilingParams.a, tilingParams.totalCoreNum);
    tilingParams.usedCoreNum = Ops::Base::CeilDiv(tilingParams.a, tilingParams.blockFactor);
    tilingParams.blockTail = tilingParams.a - (tilingParams.usedCoreNum - 1) * tilingParams.blockFactor;
    // 切r
    uint32_t ubLoops = Ops::Base::CeilDiv(tilingParams.r, static_cast<int64_t>(baseN));
    // 计算 二分累加 分界点
    int64_t binAddQuotient = baseN == 0 ? 1 : (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(baseN)));
    binAddQuotient = (binAddQuotient == baseN) ? binAddQuotient / CONST_TWO : binAddQuotient;
    // ub间二分累加
    int64_t powerSplit = GetPowerSplit(baseN);
    uint32_t mainFoldCount = powerSplit * baseN > tilingParams.r ? 0 : (tilingParams.r - powerSplit * baseN) / baseN;
    uint32_t foldTail = tilingParams.r % baseN; // 折叠后  非ub整块的长度
    
    tilingData.numM = tilingParams.a;
    tilingData.numN = tilingParams.r;
    tilingData.baseM = baseM;
    tilingData.baseN = baseN;
    tilingData.mPerCore = tilingParams.blockFactor;
    tilingData.mLastCore = tilingParams.blockTail;
    tilingData.nUbLoops = ubLoops;
    tilingData.binAddQuotient = binAddQuotient; // ub 整块二分折叠点
    tilingData.powerSplit = powerSplit;       // 小于 ubLoops 的最大二次幂
    tilingData.mainFoldCount = mainFoldCount; // 折叠后的  折叠瓣 的 ub整块数
    tilingData.foldTail = foldTail;
    tilingData.optionMask = tilingParams.optionMask;
    tilingData.divMode = tilingParams.divMode;
    tilingData.dstDtype = tilingParams.dstDtype;
    tilingData.epsilon = tilingParams.epsilon;
    tilingData.avgFactor = tilingParams.avgFactor;
    tilingData.rstdFlag = tilingParams.rstdFlag;
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

void RmsNormQuantV2RegbaseTilingRecompute::PrintTilingData()
{
    OP_LOGI(
        nodeName.c_str(),
        "TilingData numM: %ld, numN: %ld, baseM: %ld, baseN: %ld, "
        "mPerCore: %ld, mLastCore: %ld, nUbLoops: %ld, "
        "binAddQuotient: %ld, powerSplit: %ld, mainFoldCount: %ld, foldTail: %ld, "
        "optionMask: %lu, divMode: %lu, dstDtype: %lu, "
        "epsilon: %f, avgFactor: %f, rstdFlag: %u.",
        tilingData.numM, tilingData.numN, tilingData.baseM, tilingData.baseN,
        tilingData.mPerCore, tilingData.mLastCore, tilingData.nUbLoops,
        tilingData.binAddQuotient, tilingData.powerSplit, tilingData.mainFoldCount,
        tilingData.foldTail, tilingData.optionMask, tilingData.divMode, tilingData.dstDtype,
        tilingData.epsilon, tilingData.avgFactor, tilingData.rstdFlag);
}

ge::graphStatus RmsNormQuantV2RegbaseTilingRecompute::PostTiling()
{
    OP_LOGD(nodeName.c_str(), "Tiling usedCoreNum is %ld.", tilingParams.usedCoreNum);
    context_->SetBlockDim(tilingParams.usedCoreNum);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    size_t usrWorkspaceSize = tilingParams.workspaceSize;
    size_t sysWorkSpaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = usrWorkspaceSize + sysWorkSpaceSize;

    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(
        sizeof(tilingData) > rawTilingData->GetCapacity(),
        OP_LOGE(
            context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu", sizeof(tilingData),
            rawTilingData->GetCapacity()),
        return ge::GRAPH_FAILED);
    auto capSize = rawTilingData->GetCapacity();
    void* ptrData = rawTilingData->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrData);
    void* ptrStruct = static_cast<void*>(&tilingData);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrStruct);
    OP_CHECK_IF(
        memcpy_s(ptrData, capSize, ptrStruct, sizeof(tilingData)) != 0,
        OP_LOGE(context_->GetNodeName(), "Set tiling data is failed!"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(tilingData));
    return ge::GRAPH_SUCCESS;
}
uint64_t RmsNormQuantV2RegbaseTilingRecompute::GetTilingKey() const
{
    uint64_t tilingKey = RMSNORMQUANTV2_REGBASE_RECOMPUTE;
    OP_LOGI(nodeName.c_str(), "TilingKey is %lu.", tilingKey);
    return tilingKey;
}

REGISTER_OPS_TILING_TEMPLATE(RmsNormQuantV2, RmsNormQuantV2RegbaseTilingRecompute, 200);
REGISTER_OPS_TILING_TEMPLATE(RmsNormQuantV3, RmsNormQuantV2RegbaseTilingRecompute, 200);
} // namespace optiling
