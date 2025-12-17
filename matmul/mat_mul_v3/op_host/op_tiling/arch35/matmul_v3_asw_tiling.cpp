/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file matmul_v3_asw_tiling.cc
 * \brief
 */
#include "matmul_v3_asw_tiling.h"
#include "matmul_v3_tiling_strategy.h"
#include "./matmul_tiling_registry.h"
#include "matmul/common/op_host/math_util.h"

using Ops::NN::MathUtil;
namespace optiling {
namespace matmul_v3_advanced {
using namespace strategy;

MM_REGISTER_TILING_TEMPLATE(MatMulV3, MatMulV3AswTiling, ASCEND910_95, BASE);

void MatMulV3AswTiling::CalcTailBasicBlock()
{
    uint64_t mCnt = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    uint64_t nCnt = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    uint64_t mnCnt = mCnt * nCnt;
    uint64_t tailCnt = mnCnt <= compileInfo_.aicNum ? 0UL : mnCnt % compileInfo_.aicNum;
    runInfo_.tailInfo.mCnt = 1UL;
    runInfo_.tailInfo.nCnt = 1UL;
    if (tailCnt != 0UL) {
        while ((runInfo_.tailInfo.mCnt + 1UL) * runInfo_.tailInfo.nCnt * tailCnt <= compileInfo_.aicNum) {
            runInfo_.tailInfo.mCnt += 1UL;
            if (runInfo_.tailInfo.mCnt * (runInfo_.tailInfo.nCnt + 1UL) * tailCnt <= compileInfo_.aicNum) {
                runInfo_.tailInfo.nCnt += 1UL;
            }
        }
    }
}

void MatMulV3AswTiling::GetOuterAxisTailCnt(const bool nLoadBalance, uint64_t& baseTailSplitCnt, uint64_t& tailMain)
{
    uint64_t aicNum = compileInfo_.aicNum;
    uint64_t x = args_.mValue;
    uint64_t y = args_.nValue;
    uint64_t baseX = runInfo_.baseM;
    uint64_t baseY = runInfo_.baseN;
    if (nLoadBalance) {
        x = args_.nValue;
        y = args_.mValue;
        baseX = runInfo_.baseN;
        baseY = runInfo_.baseM;
    }

    uint64_t xCnt = MathUtil::CeilDivision(x, baseX);
    uint64_t yCnt = MathUtil::CeilDivision(y, baseY);
    uint64_t xTail = x % baseX;

    uint64_t aswWindowLen = GetAswWindowLen();
    uint64_t totalWindows = MathUtil::CeilDivision(xCnt * yCnt, aicNum);
    uint64_t mainWindows = MathUtil::CeilDivision((xCnt - 1UL) * yCnt + yCnt % aicNum, aicNum);
    // 未做负载均衡的轴是核数的倍数且做负载均衡的轴是窗口的因子或轴的倍数，说明部分核只做主块
    if (yCnt % aicNum == 0UL && (xCnt % aswWindowLen == 0UL || aswWindowLen % xCnt == 0UL)) {
        mainWindows = totalWindows;
    }
    uint64_t tailWindows = totalWindows - mainWindows;
    uint64_t perfRes = mainWindows * baseX + tailWindows * xTail;

    uint64_t baseTailCntMax = 1UL;
    baseTailCntMax = std::min((baseX - xTail) / BASIC_BLOCK_SIZE_16, xCnt);

    for (uint64_t mergeLen = 1UL; mergeLen < baseTailCntMax; ++mergeLen) {
        uint64_t newTailMain =
            MathUtil::Align(MathUtil::CeilDivision((mergeLen * baseX + xTail), mergeLen + 1UL), BASIC_BLOCK_SIZE_16);
        uint64_t newTailLast = mergeLen * (baseX - newTailMain) + xTail;
        uint64_t newMainRound = 0UL;
        uint64_t newTailRound = 0UL;
        if (mergeLen < xCnt - 1UL) {
            // 按照最差的场景计算合并后的主轮，所以性能不会最优
            newMainRound =
                MathUtil::CeilDivision((xCnt - 1UL - mergeLen) * yCnt + (mergeLen + 1UL) * yCnt % aicNum, aicNum);
        }
        if (mergeLen > 0UL) {
            newTailRound =
                std::min(MathUtil::CeilDivision(mergeLen * yCnt + yCnt % aicNum, aicNum), totalWindows - newMainRound);
        }
        uint64_t curPerf = newMainRound * baseX + newTailRound * newTailMain +
                           (totalWindows - newMainRound - newTailRound) * newTailLast;
        // m轴尽量多分基本块，n轴少分基本块
        if (curPerf < perfRes || (!nLoadBalance && curPerf == perfRes)) {
            perfRes = curPerf;
            tailMain = newTailMain;
            baseTailSplitCnt = mergeLen + 1UL;
        }
    }
}

void MatMulV3AswTiling::OptimizeEdgeBasicBlock()
{
    uint64_t mCore = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    uint64_t nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    if (mCore * nCore < compileInfo_.aicNum || mCore == 1UL || nCore == 1UL) {
        return;
    }
    uint64_t mBaseTail = args_.mValue % runInfo_.baseM;
    uint64_t nBaseTail = args_.nValue % runInfo_.baseN;

    bool balanceAfterFixp = args_.kValue <= BASIC_BLOCK_SIZE_256 && args_.nValue % BLOCK_BYTE_SIZE == 0UL;
    if (mBaseTail > 0UL && !args_.isATrans && (nBaseTail == 0UL || mBaseTail <= nBaseTail || balanceAfterFixp)) {
        GetOuterAxisTailCnt(false, runInfo_.mBaseTailSplitCnt, runInfo_.tailInfo.mTailMain);
    } else if (nBaseTail > 0UL && args_.isBTrans && !balanceAfterFixp) {
        GetOuterAxisTailCnt(true, runInfo_.nBaseTailSplitCnt, runInfo_.tailInfo.nTailMain);
    }
}

void MatMulV3AswTiling::FormulateBasicBlock()
{
    uint64_t mCore = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    uint64_t nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    if (mCore * nCore >= compileInfo_.aicNum) {
        runInfo_.baseM = std::min(ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16), runInfo_.baseM);
        runInfo_.baseN = std::min(ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16), runInfo_.baseN);
        return;
    }
    if (mCore <= nCore) {
        runInfo_.baseM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mCore), BASIC_BLOCK_SIZE_16);
        mCore = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
        nCore = runInfo_.usedCoreNum / mCore;
        runInfo_.baseN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nCore), BASIC_BLOCK_SIZE_16);
    } else {
        runInfo_.baseN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nCore), BASIC_BLOCK_SIZE_16);
        nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
        mCore = runInfo_.usedCoreNum / nCore;
        runInfo_.baseM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mCore), BASIC_BLOCK_SIZE_16);
    }

    while (runInfo_.baseN >= runInfo_.baseM * NUM_TWO && nCore < runInfo_.usedCoreNum / NUM_TWO) {
        nCore = nCore * NUM_TWO;
        mCore = runInfo_.usedCoreNum / nCore;
        runInfo_.baseM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mCore), BASIC_BLOCK_SIZE_16);
        runInfo_.baseN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nCore), BASIC_BLOCK_SIZE_16);
        mCore = MathUtil::CeilDivision(args_.mValue, static_cast<uint64_t>(runInfo_.baseM));
        nCore = MathUtil::CeilDivision(args_.nValue, static_cast<uint64_t>(runInfo_.baseN));
    }

    while (runInfo_.baseM >= runInfo_.baseN * NUM_TWO && mCore < runInfo_.usedCoreNum / NUM_TWO) {
        mCore = mCore * NUM_TWO;
        nCore = runInfo_.usedCoreNum / mCore;
        runInfo_.baseM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mCore), BASIC_BLOCK_SIZE_16);
        runInfo_.baseN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nCore), BASIC_BLOCK_SIZE_16);
        mCore = MathUtil::CeilDivision(args_.mValue, static_cast<uint64_t>(runInfo_.baseM));
        nCore = MathUtil::CeilDivision(args_.nValue, static_cast<uint64_t>(runInfo_.baseN));
    }
    mCore = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    runInfo_.usedCoreNum = mCore * nCore;
    uint64_t kValueAlign = ops::CeilAlign(static_cast<uint64_t>(args_.kValue), BASIC_BLOCK_SIZE_16);
    uint64_t kValueMax = ops::FloorAlign(
        L0A_SIZE_2 / DB_SIZE / args_.aDtypeSize / std::max(runInfo_.baseM, runInfo_.baseN), BASIC_BLOCK_SIZE_16);
    runInfo_.baseK = std::min(kValueAlign, kValueMax);
}

ge::graphStatus MatMulV3AswTiling::DoOpTiling()
{
    MatMulV3TilingHelper::ResetBase(compileInfo_, args_, runInfo_);
    OptimizeEdgeBasicBlock();
    FormulateBasicBlock();
    CalcTailBasicBlock();
    MatMulV3TilingHelper::CalL1Tiling(compileInfo_, args_, runInfo_);
    if (MatMulV3TilingHelper::CheckIfDoubleAswt(compileInfo_, args_, 1UL)) {
        aswtModel_ = MatMulV3Model::DOUBLE_ASWT;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t MatMulV3AswTiling::GetTilingKey() const
{
    return MatMulV3TilingKey()
        .SetTrans(args_.isATrans, args_.isBTrans)
        .SetModel(aswtModel_)
        .SetL0C2Out(MatMulV3TilingHelper::GetL0C2Out(compileInfo_, args_, runInfo_))
        .GetTilingKey();
}
} // namespace matmul_v3_advanced
} // namespace optiling
