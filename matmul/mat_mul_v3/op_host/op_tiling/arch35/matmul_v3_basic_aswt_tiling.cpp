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
 * \file matmul_v3_basic_aswt_tiling.cpp
 * \brief
 */
#include "matmul_v3_basic_aswt_tiling.h"
#include "matmul_v3_tiling_strategy.h"
#include "./matmul_tiling_registry.h"
#include "matmul/common/op_host/math_util.h"

using Ops::NN::MathUtil;

namespace optiling {
namespace matmul_v3_advanced {
constexpr uint64_t FP32_SPLIT_K_THRESHOLD = 8192UL;
using namespace strategy;

// 注册BASIC_FULL_LOAD作为基础API实现的全载模板策略
MM_REGISTER_TILING_TEMPLATE(MatMulV3, MatMulV3BasicAswtTiling, ASCEND950, BASIC_ASWT);

bool MatMulV3BasicAswtTiling::IsCapable()
{
    // FP32大K场景需要核内切K 基础API暂未实现
    if (args_.aDtypeSize == DATA_SIZE_FP32 && !args_.isHf32 && args_.bFormat == ge::FORMAT_ND &&
        args_.kValue > FP32_SPLIT_K_THRESHOLD) {
        OP_LOGD(args_.opName, "fp32 big k is not supported for basic api");
        return false;
    }
    OP_LOGI(args_.opName, "MatMulV3 tiling enable state basic api");
    return true;
}

void MatMulV3BasicAswtTiling::FullLoadPre()
{
    uint64_t mCnt = MathUtil::CeilDivision(args_.mValue, runInfo_.singleCoreM);
    uint64_t nCnt = MathUtil::CeilDivision(args_.nValue, runInfo_.singleCoreN);
    isSingleRound_ = mCnt * nCnt <= compileInfo_.aicNum;
    biasSize_ = args_.hasBias ? runInfo_.baseN * runInfo_.stepN * GetSizeByDataType(args_.biasType) : 0;
    return;
}

uint64_t MatMulV3BasicAswtTiling::GetAFullLoadBasicNL1() const {
    return std::min(ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16), runInfo_.baseN * DB_SIZE);
}

// A全载切换基础API的条件
bool MatMulV3BasicAswtTiling::CheckAL1FullLoad91095(uint64_t kAlignedValue, uint64_t mAlignedValue)
{
    uint64_t al1Size = kAlignedValue * mAlignedValue * args_.aDtypeSize;
    // 单核上只有一轮，走basic api模板， 头开销较小，无需走全载模板
    if (isSingleRound_) {
        return false;
    }
    // L2命中率高则不走全载模板
    uint64_t maxStepM = MatMulV3BaseTiling::GetAswWindowLen() > 1UL ? MatMulV3BaseTiling::GetAswWindowLen() - 1UL : 1;
    if (args_.mValue >= maxStepM * runInfo_.baseM) {
        return false;
    }
    uint64_t biasSize = args_.hasBias ? GetAFullLoadBasicNL1() * GetSizeByDataType(args_.biasType) : 0;
    // 3/4L1 = 384M
    if (al1Size + biasSize > compileInfo_.l1Size * 3UL / 4UL) {
        return false;
    }
    uint64_t mCnt = MathUtil::CeilDivision(args_.mValue, runInfo_.singleCoreM);
    uint64_t nCnt = MathUtil::CeilDivision(args_.nValue, runInfo_.singleCoreN);
    uint64_t aL1FullMTE2Size = args_.mValue * compileInfo_.aicNum + args_.nValue * mCnt;
    uint64_t baseMTE2Size = args_.mValue * nCnt + args_.nValue * mCnt;
    // 1.2f 表示切基础API全载后至少减少20%的总数据搬运量
    if (args_.mValue > BASIC_BLOCK_SIZE_256 && static_cast<float>(baseMTE2Size) < 1.2f * aL1FullMTE2Size) {
        return false;
    }
    return true;
}

bool MatMulV3BasicAswtTiling::CheckAL1FullLoad()
{
    if (l0C2Out_ != MatMulV3L0C2Out::ON_THE_FLY) {
        return false;
    }
    if (args_.nValue < CACHELINE) {
        return false;
    }
    uint64_t mAlignedValue = ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16);
    uint64_t kAlignedValue = ops::CeilAlign(args_.kValue, BLOCK_BYTE_SIZE / args_.aDtypeSize);
    if (args_.isATrans) {
        // m为内轴时强制16对齐
        mAlignedValue = ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16);
        kAlignedValue = ops::CeilAlign(args_.kValue, BASIC_BLOCK_SIZE_16);
    }
    // check AL1FullLoad
    return CheckAL1FullLoad91095(kAlignedValue, mAlignedValue);
}

void MatMulV3BasicAswtTiling::CalcTailBasicBlockBL1Full()
{
    uint64_t mCnt = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    uint64_t tailCnt = mCnt <= compileInfo_.aicNum ? 0UL : mCnt % compileInfo_.aicNum;
    runInfo_.tailInfo.mCnt = 1UL;
    runInfo_.tailInfo.nCnt = 1UL;
    if (tailCnt == 0UL) {
        return;
    }
    while ((runInfo_.tailInfo.mCnt + 1UL) * tailCnt <= compileInfo_.aicNum) {
        runInfo_.tailInfo.mCnt += 1UL;
    }
}

// overide CalcTailBasicBlock
void MatMulV3BasicAswtTiling::CalcTailBasicBlockAL1Full()
{
    uint64_t nCnt = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    uint64_t tailCnt = nCnt <= compileInfo_.aicNum ? 0UL : nCnt % compileInfo_.aicNum;
    runInfo_.tailInfo.mCnt = 1UL;
    runInfo_.tailInfo.nCnt = 1UL;

    if (tailCnt == 0UL) {
        return;
    }
    while ((runInfo_.tailInfo.nCnt + 1UL) * tailCnt <= compileInfo_.aicNum) {
        runInfo_.tailInfo.nCnt += 1UL;
    }
}

void MatMulV3BasicAswtTiling::AdjustAL1Tiling91095Basic([[maybe_unused]] uint64_t biasBatchDimAll /* args */) {
    //biasBatchDimAll 没有被使用，但是编译器不发出警告
    uint64_t kAlignedValue = ops::CeilAlign(args_.kValue, BASIC_BLOCK_SIZE_16);
    uint64_t mAlignedValue = ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16);
    // aL1 LoadSize
    uint64_t aL1Size = kAlignedValue * mAlignedValue * args_.aDtypeSize;
    // bl1 LoadSize
    uint64_t bL1loadSize = runInfo_.baseK * runInfo_.depthB1 * runInfo_.baseN * args_.bDtypeSize;
    uint64_t baseBiasSize = args_.hasBias ? GetAFullLoadBasicNL1() * GetSizeByDataType(args_.biasType) : 0;
    // Check L1 load size
    while (bL1loadSize > compileInfo_.l1Size - baseBiasSize - aL1Size) {
        // 第一轮优先调整stepK为2， 进一步的调整baseN
        // 如果stepKa = 1 && stepKb = 1, 则调整baseN减半， stepKa=stepKb=1
        // 如果stepKa > 1 && stepKb = 1, 则baseN不变， stepKb -> 2  stepKa = 1
        // 如果stepKa = 2 && stepKb = 2, 则baseN减半, stepKa=stepkb=2
        // 如果stepKa > 2 && stepKb = 2, 则baseN不变， stepKb -> 2 stepKa > 2
        runInfo_.baseN = (runInfo_.stepKb == std::min(runInfo_.stepKa, 2UL)) ? runInfo_.baseN >> 1 : runInfo_.baseN;
        runInfo_.stepKb = std::min(runInfo_.stepKa, 2UL); // 最小为2保证baseK * 2 * adtype = 256B
        runInfo_.depthB1 = DB_SIZE * runInfo_.stepKb; // stepN = DB_SIZE
        runInfo_.baseN = ops::CeilAlign(runInfo_.baseN, BASIC_BLOCK_SIZE_16);
        bL1loadSize = runInfo_.depthB1 * runInfo_.baseN * runInfo_.baseK * args_.aDtypeSize;
    }
    uint64_t dtypeSize = GetSizeByDataType(ge::DT_FLOAT);
    runInfo_.singleCoreM = args_.mValue;
    runInfo_.singleCoreN = runInfo_.baseN;
    runInfo_.dbL0C = runInfo_.baseM * runInfo_.baseN * dtypeSize * DB_SIZE <= compileInfo_.l0CSize ? DB_SIZE : 1UL;
    // 1.n为外轴且是256倍数
    // 2.n为内轴且是256*2的倍数
    // 则baseN减半不影响MTE2搬运效率
    if ((args_.isBTrans && (runInfo_.baseN * args_.bDtypeSize) % BASIC_BLOCK_SIZE_256 == 0) ||
        (runInfo_.baseN * args_.bDtypeSize) % (BASIC_BLOCK_SIZE_256 * NUM_TWO) == 0) {
        runInfo_.baseN =
            (runInfo_.dbL0C > 1UL) ? runInfo_.baseN : ops::CeilAlign(runInfo_.baseN >> 1, BASIC_BLOCK_SIZE_16);
    }
    // b矩阵开启4buffer
    uint64_t bL14Buffer = runInfo_.baseK * runInfo_.stepKb * runInfo_.baseN * args_.aDtypeSize * BASIC_L1_BUFFER_NUM;
    runInfo_.l1BufferNum =
        bL14Buffer + aL1Size + baseBiasSize * BASIC_L1_BUFFER_NUM > compileInfo_.l1Size ? DB_SIZE : BASIC_L1_BUFFER_NUM;
    runInfo_.singleCoreN = runInfo_.baseN;
    runInfo_.dbL0C = runInfo_.baseM * runInfo_.baseN * dtypeSize * DB_SIZE <= compileInfo_.l0CSize ? DB_SIZE : 1UL;
}

void MatMulV3BasicAswtTiling::DoAL1FullLoad(uint64_t bBatchDimAll, uint64_t biasBatchDimAll)
{
    MatMulV3TilingHelper::ResetFullLoadLoadBalance(runInfo_);
    OP_LOGI(args_.opName, "MatMulV3 tiling enable state is DoAL1FullLoad.");
    // adjust tiling common
    MatMulV3TilingHelper::AdjustAL1TilingCommon(bBatchDimAll, compileInfo_, args_, runInfo_);
    // 复用A全载的标记位, 无需再使用函数指针隔离, 不支持的平台无需设置走基础API的标记
    AdjustAL1Tiling91095Basic(biasBatchDimAll);
    CalcTailBasicBlockAL1Full();
    fullLoad_ = MatMulV3FullLoad::A_FULL_LOAD;
    return;
}

bool MatMulV3BasicAswtTiling::CheckBL1FullLoad91095(uint64_t kAlignedValue, uint64_t nAlignedValue)
{
    uint64_t bl1Size = kAlignedValue * nAlignedValue * args_.bDtypeSize;
    // 单核上只有一轮，走basic api模板， 头开销较小，无需走全载模板
    if (isSingleRound_) {
        return false;
    }
    // L2命中率高， 不走全载模板
    uint64_t maxStepN = MatMulV3BaseTiling::GetAswWindowLen() > 1UL ? MatMulV3BaseTiling::GetAswWindowLen() - 1UL : 1;
    if (args_.nValue >= maxStepN * runInfo_.baseN) {
        return false;
    }
    uint64_t biasSize = args_.hasBias ? nAlignedValue * GetSizeByDataType(args_.biasType) : 0;
    // 3/4L1 = 384M
    if (bl1Size + biasSize > compileInfo_.l1Size * 3UL / 4UL) {
        return false;
    }
    uint64_t mCnt = MathUtil::CeilDivision(args_.mValue, runInfo_.singleCoreM);
    uint64_t nCnt = MathUtil::CeilDivision(args_.nValue, runInfo_.singleCoreN);
    uint64_t bL1FullMTE2Size = args_.nValue * compileInfo_.aicNum + args_.mValue * nCnt;
    uint64_t baseMTE2Size = args_.mValue * nCnt + args_.nValue * mCnt;
    // 1.2表示全载后至少减少了20%的总数据搬运量
    if (args_.nValue > BASIC_BLOCK_SIZE_256 && static_cast<float>(baseMTE2Size) < 1.2f * bL1FullMTE2Size) {
        return false;
    }
    return true;
}

bool MatMulV3BasicAswtTiling::CheckBL1FullLoad()
{
    if (args_.mValue < CACHELINE) {
        return false;
    }
    uint64_t kAlignedValue = ops::CeilAlign(args_.kValue, BASIC_BLOCK_SIZE_16);
    uint64_t nAlignedValue = ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16);
    if (args_.isBTrans) {
        kAlignedValue = ops::CeilAlign(args_.kValue, BLOCK_BYTE_SIZE / args_.bDtypeSize);
        nAlignedValue = ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16);
    }
    return CheckBL1FullLoad91095(kAlignedValue, nAlignedValue);
}

void MatMulV3BasicAswtTiling::AdjustBL1Tiling91095Basic([[maybe_unused]] uint64_t biasBatchDimAll /* args */)
{
    //biasBatchDimAll 没有被使用，但是编译器不发出警告
    uint64_t kAlignedValue = ops::CeilAlign(args_.kValue, BASIC_BLOCK_SIZE_16);
    uint64_t nAlignedValue = ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16);
    uint64_t bL1Size = kAlignedValue * nAlignedValue * args_.bDtypeSize;
    uint64_t aL1LoadSize = runInfo_.baseK * runInfo_.depthA1 * runInfo_.baseM * args_.aDtypeSize;
    uint64_t baseBiasSize = args_.hasBias ? nAlignedValue * GetSizeByDataType(args_.biasType) : 0;
    // Check L1 load size
    while (aL1LoadSize > compileInfo_.l1Size - baseBiasSize - bL1Size) {
        // 第一轮有限调整stepK为2， 进一步的调整baseM
        runInfo_.baseM = (runInfo_.stepKa == std::min(runInfo_.stepKb, 2UL)) ? runInfo_.baseM >> 1 : runInfo_.baseM;
        runInfo_.stepKa = std::min(runInfo_.stepKb, 2UL); // 最小为2保证baseK * 2 * adtype = 256B
        runInfo_.depthA1 = DB_SIZE * runInfo_.stepKa;
        runInfo_.baseM = ops::CeilAlign(runInfo_.baseM, BASIC_BLOCK_SIZE_16);
        aL1LoadSize = runInfo_.depthA1 * runInfo_.baseM * runInfo_.baseK * args_.aDtypeSize;
    }
    uint64_t dtypeSize = GetSizeByDataType(ge::DT_FLOAT);
    runInfo_.singleCoreN = args_.nValue;
    runInfo_.singleCoreM = runInfo_.baseM;
    // 内轴*dtype是256 * 2的整数倍表明tiling减半不影响MTE2搬运效率
    if (!args_.isATrans || (runInfo_.baseM * args_.aDtypeSize) % (BASIC_BLOCK_SIZE_256 * NUM_TWO) == 0) {
            runInfo_.baseM = runInfo_.baseM * runInfo_.baseN * dtypeSize * DB_SIZE <= compileInfo_.l0CSize ?
                                runInfo_.baseM : ops::CeilAlign(runInfo_.baseM >> 1, BASIC_BLOCK_SIZE_16);
    }
    // 重新计算是否满足aL1开启4buffer loc开启db
    uint64_t aL14Buffer = static_cast<uint64_t>(runInfo_.baseK) * runInfo_.stepKa * runInfo_.baseM * args_.aDtypeSize *
                          BASIC_L1_BUFFER_NUM;
    runInfo_.l1BufferNum =
        aL14Buffer + bL1Size + baseBiasSize * BASIC_L1_BUFFER_NUM > compileInfo_.l1Size ? DB_SIZE : BASIC_L1_BUFFER_NUM;
    runInfo_.singleCoreM = runInfo_.baseM;
    runInfo_.dbL0C = runInfo_.baseM * runInfo_.baseN * dtypeSize * DB_SIZE <= compileInfo_.l0CSize ? DB_SIZE : 1UL;
    runInfo_.mixInfo.ubDB = runInfo_.baseM * runInfo_.baseN * dtypeSize <= compileInfo_.ubSize ? DB_SIZE : 1UL;
}

void MatMulV3BasicAswtTiling::DoBL1FullLoad(uint64_t aBatchDimAll, uint64_t biasBatchDimAll)
{
    // 负载均衡屏蔽全载模板
    MatMulV3TilingHelper::ResetFullLoadLoadBalance(runInfo_);
    OP_LOGI(args_.opName, "MatMulV3 tiling enable state is DoBL1FullLoad.");
    MatMulV3TilingHelper::AdjustBL1TilingCommon(aBatchDimAll, compileInfo_, args_, runInfo_);
    AdjustBL1Tiling91095Basic(biasBatchDimAll);
    CalcTailBasicBlockBL1Full();
    fullLoad_ = MatMulV3FullLoad::B_FULL_LOAD;
    return;
}

ge::graphStatus MatMulV3BasicAswtTiling::DoOpTiling()
{
    MatMulV3AswTiling::DoNormOpTiling();
    l0C2Out_ = MatMulV3TilingHelper::GetL0C2Out(compileInfo_, args_, runInfo_);
    FullLoadPre();
    if (CheckAL1FullLoad()) {
        DoAL1FullLoad();
    } else if (CheckBL1FullLoad()) {
        DoBL1FullLoad();
    } else if (l0C2Out_ == MatMulV3L0C2Out::ON_THE_FLY) {
        // 非全载 非fixpipe 负载均衡实现
        MatMulV3AswTiling::DoOpTiling();
        uint64_t remainSizeForAL1BL1 =
            args_.hasBias ? (compileInfo_.l1Size - BIAS_TABLE_NUM * DATA_SIZE_FP32) : compileInfo_.l1Size;
        runInfo_.stepKa =
            remainSizeForAL1BL1 / NUM_TWO / ((runInfo_.baseM + runInfo_.baseN) * runInfo_.baseK) / args_.aDtypeSize;
        runInfo_.stepKb = runInfo_.stepKa; // has bias, adjust stepK to suitable value
        runInfo_.depthA1 = runInfo_.stepKa * DB_SIZE;
        runInfo_.depthB1 = runInfo_.stepKb * DB_SIZE;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t MatMulV3BasicAswtTiling::GetTilingKey() const
{
    MatMulV3TilingKey tmp = MatMulV3TilingKey();
    MatMulV3TilingKey& tilingKey = tilingKeyObj == nullptr ? tmp : *tilingKeyObj;
    return tilingKey.SetTrans(args_.isATrans, args_.isBTrans)
        .SetFullLoad(fullLoad_)
        .SetModel(MatMulV3Model::BASIC)
        .SetL0C2Out(l0C2Out_)
        .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
        .GetTilingKey();
}

ge::graphStatus MatMulV3BasicAswtTiling::GetTilingData(TilingResult& tiling) const
{
    return GetTilingDataImpl<MatMulV3BasicTilingData>(tiling);
}
} // namespace matmul_v3_advanced
} // namespace optiling
