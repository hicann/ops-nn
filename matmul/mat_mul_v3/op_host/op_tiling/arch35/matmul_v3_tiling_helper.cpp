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
 * \file matmul_v3_tiling_helper.cc
 * \brief
 */

#include "matmul_v3_tiling_helper.h"
#include "matmul/common/op_host/math_util.h"
#include "platform/platform_infos_def.h"

using Ops::NN::MathUtil;
namespace {
using namespace optiling;
using namespace optiling::matmul_v3_advanced;

// ------------------------------ CalL1Tiling -------------------------------------------//
void CalL1TilingDefault(const MatmulV3CompileInfo &compileInfo, const MatMulV3Args &args, MatMulV3RunInfo &runInfo)
{
    uint64_t totalL1Size = compileInfo.l1Size;
    if (args.hasScale) {
        totalL1Size -= runInfo.baseN * sizeof(uint64_t);
    }
    uint64_t reserveBTSize = args.hasBias ? BIAS_TABLE_NUM * DATA_SIZE_FP32 : 0UL;
    runInfo.depthA1 = totalL1Size / NUM_TWO / runInfo.baseM / runInfo.baseK / args.aDtypeSize;  // 2: half of l1
    runInfo.depthB1 = totalL1Size / NUM_TWO / runInfo.baseN / runInfo.baseK / args.bDtypeSize;  // 2: half of l1

    uint64_t depthASize = runInfo.depthA1 * runInfo.baseM * runInfo.baseK * args.aDtypeSize;
    uint64_t depthBSize = runInfo.depthB1 * runInfo.baseN * runInfo.baseK * args.bDtypeSize;
    if (depthASize + depthBSize > totalL1Size - reserveBTSize) {
        if (runInfo.baseM <= runInfo.baseN) {
            runInfo.depthA1 = std::max(runInfo.depthA1 / NUM_TWO, 1UL);  // 2: adjust deptch for l1 buffer
        } else {
            runInfo.depthB1 = std::max(runInfo.depthB1 / NUM_TWO, 1UL);  // 2: adjust deptch for l1 buffer
        }
    }
    runInfo.stepKa = std::max(runInfo.depthA1 / DB_SIZE, 1UL);
    runInfo.stepKb = std::max(runInfo.depthB1 / DB_SIZE, 1UL);
    // 对齐且基本块为[256, 256]则stepK改为2
    if (runInfo.baseM == BASIC_BLOCK_SIZE_256 && runInfo.baseN == BASIC_BLOCK_SIZE_256 &&
        args.mValue % BASIC_BLOCK_SIZE_16 == 0 && args.nValue % BASIC_BLOCK_SIZE_16 == 0 &&
        args.kValue % BASIC_BLOCK_SIZE_16 == 0 && runInfo.singleCoreK <= BASIC_BLOCK_SIZE_256) {
        runInfo.stepKa = std::min(runInfo.stepKa, 2UL);
        runInfo.stepKb = std::min(runInfo.stepKb, 2UL);
    }
    // 调整stepKa和stepKb为整数倍关系
    if (runInfo.stepKa >= runInfo.stepKb) {
        runInfo.stepKa = runInfo.stepKa / runInfo.stepKb * runInfo.stepKb;
    } else {
        runInfo.stepKb = runInfo.stepKb / runInfo.stepKa * runInfo.stepKa;
    }
    // 默认开启double buffer
    runInfo.depthA1 = runInfo.stepKa * DB_SIZE;  // depth % (stepKa * stepM) == 0
    runInfo.depthB1 = runInfo.stepKb * DB_SIZE;  // depth % (stepKb * stepN) == 0
    runInfo.singleCoreM = runInfo.baseM;
    runInfo.singleCoreN = runInfo.baseN;
    return;
}

void CalL1Tiling310P(const MatmulV3CompileInfo &compileInfo, const MatMulV3Args &args, MatMulV3RunInfo &runInfo)
{
    runInfo.stepM = 1UL;
    runInfo.stepN = 1UL;
    runInfo.depthA1 = 16UL;  // 16 is full use l1 space
    runInfo.depthB1 = 16UL;  // 16 is full use l1 space
    runInfo.singleCoreM = runInfo.baseM;
    runInfo.singleCoreN = runInfo.baseN;

    if (args.aFormat == ge::FORMAT_ND || args.bFormat == ge::FORMAT_ND) {
        runInfo.depthA1 = 6UL;  // 6为经验值
        runInfo.depthB1 = 6UL;  // 6为经验值
    }
    runInfo.stepKa = runInfo.depthA1 / DB_SIZE;
    runInfo.stepKb = runInfo.depthB1 / DB_SIZE;
    if ((args.aFormat == ge::FORMAT_FRACTAL_NZ) && (args.bFormat == ge::FORMAT_FRACTAL_NZ)) {
        // ka全载
        if (args.mValue <= runInfo.baseM) {
            runInfo.baseM = args.mValue;
            if (runInfo.baseM * args.kValue * args.aDtypeSize < (compileInfo.l1Size / NUM_TWO)) {
                runInfo.depthA1 = MathUtil::CeilDivision(args.kValue, runInfo.baseK);
                runInfo.stepKa = runInfo.depthA1;
            }
        }
        // kb全载
        if (args.nValue <= runInfo.baseN) {
            runInfo.baseN = args.nValue;
            if (runInfo.baseN * args.kValue * args.bDtypeSize < (compileInfo.l1Size / NUM_TWO)) {
                runInfo.depthB1 = MathUtil::CeilDivision(args.kValue, runInfo.baseK);
                runInfo.stepKb = runInfo.depthB1;
            }
        }
    }
    return;
}

using CalL1TilingFunc = void (*)(const MatmulV3CompileInfo &, const MatMulV3Args &, MatMulV3RunInfo &);

const static std::map<NpuArch, CalL1TilingFunc> CalL1TilingFuncMap = {
    {NpuArch::DAV_2002, CalL1Tiling310P},
};

// ------------------------------ ResetBase -------------------------------------------//
void ResetBaseDefault(const MatmulV3CompileInfo &compileInfo, const MatMulV3Args &args, MatMulV3RunInfo &runInfo)
{
    runInfo.usedCoreNum = compileInfo.aicNum;
    runInfo.baseM = BASIC_BLOCK_SIZE_128;
    runInfo.baseN = BASIC_BLOCK_SIZE_256;  // 256 is better base
    runInfo.baseK = BASIC_BLOCK_K_128_BYTE / args.aDtypeSize;
    runInfo.stepM = BASE_STEP;
    runInfo.stepN = BASE_STEP;
    runInfo.iterateOrder = ITER_COL_FIRST;
    runInfo.dbL0C = DB_OFF_SIZE;
    runInfo.singleCoreK = args.kValue;
    runInfo.singleCoreM = runInfo.baseM;
    runInfo.singleCoreN = runInfo.baseN;
    runInfo.mBaseTailSplitCnt = INIT_SPLIT_CNT;
    runInfo.nBaseTailSplitCnt = INIT_SPLIT_CNT;
    runInfo.tailInfo.mTailMain = INIT_SPLIT_VALUE;
    runInfo.tailInfo.nTailMain = INIT_SPLIT_VALUE;
}

void ResetBaseDav3510(const MatmulV3CompileInfo &compileInfo, const MatMulV3Args &args, MatMulV3RunInfo &runInfo)
{
    ResetBaseDefault(compileInfo, args, runInfo);
    runInfo.baseM = BASIC_BLOCK_SIZE_256;
    runInfo.singleCoreM = runInfo.baseM;
}

using ResetBaseFunc = void (*)(const MatmulV3CompileInfo &, const MatMulV3Args &, MatMulV3RunInfo &);

const static std::map<NpuArch, ResetBaseFunc> ResetBaseFuncMap = {
    {NpuArch::DAV_3510, ResetBaseDav3510},
};

// ------------------------------ GetL0C2Out -------------------------------------------//
MatMulV3L0C2Out GetL0C2OutDefault(const MatmulV3CompileInfo & /* compileInfo */, const MatMulV3Args & /* args */,
                              const MatMulV3RunInfo & /* runInfo */)
{
    return MatMulV3L0C2Out::ON_THE_FLY;
}

MatMulV3L0C2Out GetL0C2OutDav3510(const MatmulV3CompileInfo &compileInfo, const MatMulV3Args &args,
                                const MatMulV3RunInfo &runInfo)
{
    bool isValidMKN = args.kValue <= BASIC_BLOCK_SIZE_256 && args.mValue >= BASIC_BLOCK_SIZE_256;
    uint64_t mCnt = MathUtil::CeilDivision(args.mValue, runInfo.singleCoreM);
    uint64_t nCnt = MathUtil::CeilDivision(args.nValue, runInfo.singleCoreN);
    // make sure the fixpipe stream is large enough to be bound
    bool isMultiRound = mCnt * nCnt >= NUM_TWO * compileInfo.aicNum;
    uint64_t cDtypeSize = ge::GetSizeByDataType(args.cType);
    // 128: SMALL_SHAPE_LOWER_THRES
    bool isUnalignedN = args.nValue * cDtypeSize % 128UL != 0 && args.nValue * cDtypeSize > BASIC_BLOCK_SIZE_256;
    bool fixpipeBound = isValidMKN && isMultiRound && isUnalignedN;
    if (!fixpipeBound || (compileInfo.aivNum != (compileInfo.aicNum * NUM_TWO))) {
        return MatMulV3L0C2Out::ON_THE_FLY;
    }
    if (args.aType == ge::DT_FLOAT16 || args.aType == ge::DT_BF16) {
        return MatMulV3L0C2Out::ND_FIXPIPE_1_1;
    }
    return MatMulV3L0C2Out::ND_FIXPIPE_1_2;
}

using GetL0C2OutFunc = MatMulV3L0C2Out (*)(const MatmulV3CompileInfo &, const MatMulV3Args &, const MatMulV3RunInfo &);

const static std::map<NpuArch, GetL0C2OutFunc> GetL0C2OutFuncMap = {
    {NpuArch::DAV_3510, GetL0C2OutDav3510},
};


// ------------------------------ GetStepSmallK -------------------------------------------//
uint64_t GetStepSmallKDefault(const MatMulV3Args& /* args */, const MatMulV3RunInfo& runInfo, bool isBL1FullLoad)
{
    return isBL1FullLoad ? runInfo.stepKa : runInfo.stepKb;
}

uint64_t GetStepSmallKDav3510(const MatMulV3Args& args, const MatMulV3RunInfo& runInfo, bool isBL1FullLoad)
{
    uint64_t stepBigK = runInfo.stepKa;
    uint64_t stepSmallK = runInfo.stepKb;
    uint64_t dtypeSize = args.aDtypeSize;
    ge::DataType inputType = args.aType;
    bool isTrans = args.isBTrans;
    if (isBL1FullLoad) {
        stepBigK = runInfo.stepKb;
        stepSmallK = runInfo.stepKa;
        dtypeSize = args.bDtypeSize;
        inputType = args.bType;
        isTrans = args.isATrans;
    }

    static const double SMALL_TAIL = 0.25;
    bool isSmallTail = static_cast<double>(stepBigK % stepSmallK) / stepSmallK <= SMALL_TAIL;
    isSmallTail = (isSmallTail && !isTrans) || runInfo.baseK * dtypeSize >= BASIC_BLOCK_SIZE_256;
    // A/B全载场景，stepK big为全载矩阵的stepK, 调整stepK small为2, 减少mte2耗时, 提高搬运带宽
    if ((inputType == ge::DT_FLOAT && !args.isHf32)) {
        stepSmallK = 1UL;
    } else if (isSmallTail) {
        stepSmallK = 2UL;
    }
    return stepSmallK;
}

using GetStepSmallKFunc = uint64_t (*)(const MatMulV3Args&, const MatMulV3RunInfo&, bool);

// 全载模板修改stepK
const static std::map<NpuArch, GetStepSmallKFunc> GetStepSmallKFuncMap = {
    {NpuArch::DAV_3510, GetStepSmallKDav3510},
};
}  // namespace

uint64_t GetMaxBaseWithLimit(
    const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args, const MatMulV3RunInfo& runInfo,
    uint64_t baseMNBufferLimit, uint64_t baseAlignUnit, bool isRightMatrix, bool isMemoryBound)
{
    uint64_t shapeValue = isRightMatrix ? args.nValue : args.mValue;
    // baseK限制，cube bound场景需要掩盖fixp，约束baseK至少128B，其他场景不做约束
    uint64_t kAlignValue = ops::CeilAlign(args.kValue, BASIC_BLOCK_SIZE_16);
    uint64_t kLimitValue = isMemoryBound ? BASIC_BLOCK_SIZE_16 : BASIC_BLOCK_K_128_BYTE / args.aDtypeSize;
    uint64_t minKL0 = std::min(kLimitValue, kAlignValue) * args.aDtypeSize;
    // L0 buffer限制
    uint64_t maxBaseMNWithBuffer = baseMNBufferLimit / DATA_SIZE_FP32 / BASIC_BLOCK_SIZE_16;
    uint64_t maxBaseBlock = std::min(compileInfo.l0ASize / DB_SIZE / minKL0, maxBaseMNWithBuffer);
    // bias table约束
    if (isRightMatrix && args.hasBias) {
        maxBaseBlock = std::min(maxBaseBlock, compileInfo.btSize / DB_SIZE / DATA_SIZE_FP32);
    }
    // K内轴时，要求kL1至少256B对齐,目前batchmatmul固定内轴512B对齐
    uint64_t kAlignUnit =
        !args.isATrans || args.isBTrans ?
            (isMemoryBound && args.batchInfo == nullptr ? BASIC_BLOCK_K_256_BYTE : BASIC_BLOCK_K_512_BYTE) /
                args.aDtypeSize :
            BASIC_BLOCK_SIZE_16;
    uint64_t maxBaseMNWithKInner =
        compileInfo.l1Size / (NUM_TWO * DB_SIZE * args.aDtypeSize * std::min(kAlignUnit, kAlignValue));
    maxBaseBlock = std::min(maxBaseBlock, maxBaseMNWithKInner);
    // 输入shape约束
    maxBaseBlock =
        std::min(ops::CeilAlign(shapeValue, baseAlignUnit), ops::FloorAlign(maxBaseBlock, baseAlignUnit));
    return maxBaseBlock;
}

double GetBalanceRateWithTail(const MatMulV3Args& args, uint64_t usedCoreNum, uint64_t baseM, uint64_t baseN)
{
    // 考虑尾轮优化负载均衡率，仅针对cubebound场景生效
    uint64_t batch = args.batchInfo == nullptr ? 1 : args.batchInfo->batchA;
    uint64_t totalRound =
        batch * MathUtil::CeilDivision(args.mValue, baseM) * MathUtil::CeilDivision(args.nValue, baseN);
    uint64_t mainRound = MathUtil::CeilDivision(totalRound, usedCoreNum) - 1;
    uint64_t totalTailSplit = ops::FloorDiv(usedCoreNum, (totalRound - usedCoreNum * mainRound));
    if (mainRound == 0 || ops::FloorDiv(baseM * baseN, totalTailSplit) < MIN_TATL_BLOCK_SIZE ||
        args.batchInfo != nullptr) {
        return (static_cast<double>(batch) * args.mValue * args.nValue / usedCoreNum) /
               ((mainRound + 1) * baseM * baseN);
    }
    uint64_t tailSplitSqrt = static_cast<uint64_t>(std::sqrt(totalTailSplit));
    uint64_t offset = ops::FloorDiv(totalTailSplit - tailSplitSqrt * tailSplitSqrt, tailSplitSqrt) + 1;
    double tailRound = 1.0 / (tailSplitSqrt * (tailSplitSqrt + offset - 1));
    return (static_cast<double>(args.mValue) * args.nValue / usedCoreNum) / ((mainRound + tailRound) * baseM * baseN);
}

void GetBaseK(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args, MatMulV3RunInfo& runInfo)
{
    uint64_t kValueAlign = ops::CeilAlign(static_cast<uint64_t>(args.kValue), BASIC_BLOCK_SIZE_16);
    uint64_t maxBaseK = compileInfo.l0ASize / DB_SIZE / args.aDtypeSize / std::max(runInfo.baseM, runInfo.baseN);
    if (kValueAlign <= maxBaseK) {
        runInfo.baseK = kValueAlign;
        return;
    }
    if (args.isATrans && !args.isBTrans) {
        runInfo.baseK = ops::FloorAlign(maxBaseK, BASIC_BLOCK_SIZE_16);
        return;
    }
    if (maxBaseK * args.aDtypeSize >= BASIC_BLOCK_K_256_BYTE) {
        runInfo.baseK = ops::FloorAlign(maxBaseK, BASIC_BLOCK_K_256_BYTE / args.aDtypeSize);
        return;
    }
    std::vector<uint64_t> baseKCandidate = {128, 64, 32, 16};
    for (uint64_t baseK : baseKCandidate) {
        if (maxBaseK >= baseK) {
            runInfo.baseK = baseK;
            return;
        }
    }
}

namespace optiling {
namespace matmul_v3_advanced {
void MatMulV3TilingHelper::ResetBase(const MatmulV3CompileInfo &compileInfo, const MatMulV3Args &args,
                                     MatMulV3RunInfo &runInfo)
{
    auto iter = (ResetBaseFuncMap.find(compileInfo.npuArch) == ResetBaseFuncMap.end())
                    ? ResetBaseDefault
                    : ResetBaseFuncMap.at(compileInfo.npuArch);
    iter(compileInfo, args, runInfo);
}

void MatMulV3TilingHelper::CalL1Tiling(const MatmulV3CompileInfo &compileInfo, const MatMulV3Args &args,
                                       MatMulV3RunInfo &runInfo)
{
    auto iter = (CalL1TilingFuncMap.find(compileInfo.npuArch) == CalL1TilingFuncMap.end())
                    ? CalL1TilingDefault
                    : CalL1TilingFuncMap.at(compileInfo.npuArch);
    iter(compileInfo, args, runInfo);
}

MatMulV3L0C2Out MatMulV3TilingHelper::GetL0C2Out(const MatmulV3CompileInfo &compileInfo, const MatMulV3Args &args,
                                                 const MatMulV3RunInfo &runInfo)
{
    auto iter = (GetL0C2OutFuncMap.find(compileInfo.npuArch) == GetL0C2OutFuncMap.end())
                    ? GetL0C2OutDefault
                    : GetL0C2OutFuncMap.at(compileInfo.npuArch);
    return iter(compileInfo, args, runInfo);
}

uint64_t MatMulV3TilingHelper::GetStepSmallK(
    bool isBL1FullLoad, const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args, MatMulV3RunInfo& runInfo)
{
    auto iter = (GetStepSmallKFuncMap.find(compileInfo.npuArch) == GetStepSmallKFuncMap.end()) ?
                    GetStepSmallKDefault :
                    GetStepSmallKFuncMap.at(compileInfo.npuArch);
    return iter(args, runInfo, isBL1FullLoad);
}

void MatMulV3TilingHelper::AdjustBL1TilingCommon(
    uint64_t aBatchDimAll, const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args, MatMulV3RunInfo& runInfo)
{
    // fine tune tiling basen
    uint64_t nAlignedValue = ops::CeilAlign(args.nValue, BASIC_BLOCK_SIZE_16);
    uint64_t bl0Size = nAlignedValue * runInfo.baseK * args.bDtypeSize * DB_SIZE;
    runInfo.baseN = bl0Size <= compileInfo.l0BSize ? nAlignedValue : std::min(nAlignedValue, runInfo.baseN);
    runInfo.stepN = MathUtil::CeilDivision(args.nValue, runInfo.baseN);
    runInfo.stepKb = MathUtil::CeilDivision(args.kValue, runInfo.baseK);
    // fine tune stepK for fullload
    runInfo.stepKa = GetStepSmallK(true, compileInfo, args, runInfo);
    // take full use of cores
    if (aBatchDimAll * MathUtil::CeilDivision(args.mValue, runInfo.baseM) < compileInfo.aicNum) {
        runInfo.baseM =
            ops::CeilAlign(MathUtil::CeilDivision(aBatchDimAll * args.mValue, compileInfo.aicNum), BASIC_BLOCK_SIZE_16);
    }
    runInfo.depthA1 = DB_SIZE * runInfo.stepKa;
    runInfo.depthB1 = runInfo.stepN * runInfo.stepKb;
}

void MatMulV3TilingHelper::AdjustAL1TilingCommon(
    uint64_t bBatchDimAll, const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args, MatMulV3RunInfo& runInfo)
{
    uint64_t mAlignedValue = ops::CeilAlign(args.mValue, BASIC_BLOCK_SIZE_16);
    uint64_t al0Size = mAlignedValue * runInfo.baseK * args.aDtypeSize * DB_SIZE;
    runInfo.baseM = al0Size <= compileInfo.l0ASize ? mAlignedValue : std::min(mAlignedValue, runInfo.baseM);
    runInfo.stepM = MathUtil::CeilDivision(args.mValue, runInfo.baseM);
    runInfo.stepKa = MathUtil::CeilDivision(args.kValue, runInfo.baseK);
    runInfo.stepKb = GetStepSmallK(false, compileInfo, args, runInfo);
    // take full use of cores
    if (bBatchDimAll * MathUtil::CeilDivision(args.nValue, runInfo.baseN) < compileInfo.aicNum) {
        runInfo.baseN =
            ops::CeilAlign(MathUtil::CeilDivision(bBatchDimAll * args.nValue, compileInfo.aicNum), BASIC_BLOCK_SIZE_16);
    }
    runInfo.depthB1 = DB_SIZE * runInfo.stepKb;
    runInfo.depthA1 = runInfo.stepM * runInfo.stepKa;
}

void MatMulV3TilingHelper::ResetFullLoadLoadBalance(MatMulV3RunInfo& runInfo)
{
    // 全载模板需重置负载均衡计算
    runInfo.mBaseTailSplitCnt = 1UL;
    runInfo.nBaseTailSplitCnt = 1UL;
    runInfo.tailInfo.mTailMain = 0UL;
    runInfo.tailInfo.nTailMain = 0UL;
}

bool MatMulV3TilingHelper::IsSelfNonContiguous(const gert::TilingContext* context)
{
    auto selfShape = context->GetInputShape(0)->GetOriginShape();
    auto mat2Shape = context->GetInputShape(1)->GetOriginShape();
    auto selfStorageShape = context->GetInputShape(0)->GetStorageShape();
    size_t selfDimNum = selfShape.GetDimNum();
    size_t mat2DimNum = mat2Shape.GetDimNum();
    // createView with TensorV2 & storageShape 1d ->  non contiguous
    return (
        context->InputIsView(0) && selfStorageShape.GetDimNum() == NUM_ONE && selfDimNum == NUM_THREE &&
        mat2DimNum == NUM_TWO);
}

void MatMulV3TilingHelper::GetRebalanceBlock(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args,
                                             MatMulV3RunInfo& runInfo, const gert::TilingContext* context)
{
    // 获取规格，计算相关性能指标
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context->GetNodeName(), "platformInfo is null");
        return;
    }
    double hbmBW = GetHbmBW(compileInfo, args, platformInfo);
    double l2BW = GetL2BW(compileInfo, args, platformInfo);
    double singleCoreComputePower = GetCoreFreq(compileInfo, args, platformInfo) * NUM_EIGHT;
    // balanceRateEdge用于判断是否取得最优解，进行减枝
    double balanceRateEdge = 0.9;
    double cmr = (static_cast<double>(args.mValue) + args.nValue) / (static_cast<double>(args.mValue) * args.nValue);
    double computePower = singleCoreComputePower * compileInfo.aicNum;

    // 切K场景，要求输出size同时小于L0C和UB
    uint64_t baseMNBufferLimit = runInfo.usedCoreNum == compileInfo.aicNum ?
                                     compileInfo.l0CSize :
                                     std::min(compileInfo.l0CSize, compileInfo.ubSize);

    uint64_t batchNum = args.batchInfo == nullptr ? 1 : args.batchInfo->batchA;
    double l2CacheUsage =
        std::max(static_cast<double>(batchNum * (args.mValue + args.nValue) * args.kValue * args.aDtypeSize) /
                     compileInfo.l2Size,
                 1.0);
    runInfo.cubeBoundEdge =
        (l2BW / computePower) + l2CacheUsage * (1 - l2BW / hbmBW) * cmr - (1 + l2BW / hbmBW) / args.kValue;
    uint64_t baseMBest = std::min(ops::CeilAlign(args.mValue, BASIC_BLOCK_SIZE_16), BASIC_BLOCK_SIZE_256);
    uint64_t baseNBest =
        std::max(BASIC_BLOCK_SIZE_16,
                 std::min(ops::CeilAlign(args.nValue, BASIC_BLOCK_SIZE_16),
                          ops::FloorAlign(baseMNBufferLimit / DATA_SIZE_FP32 / baseMBest, BASIC_BLOCK_SIZE_16)));
    double cubeBoundParamBest = (1.0 / baseMBest) + (1.0 / baseNBest);
    bool isMemoryBound = cubeBoundParamBest > runInfo.cubeBoundEdge;
    uint64_t innerAlignUnit = isMemoryBound ? BASIC_BLOCK_SIZE_128 : BASIC_BLOCK_SIZE_64;

    // fixpipe bound场景下，要求baseN是256B对齐，发挥搬出带宽
    uint64_t fixpBoundEdge = (args.mValue * args.nValue * hbmBW) / ((args.mValue + args.nValue) * l2BW);
    uint64_t baseMAlignUnit = args.isATrans ? innerAlignUnit / args.aDtypeSize : BASIC_BLOCK_SIZE_16;
    uint64_t baseNAlignUnit = (args.kValue < fixpBoundEdge) ?
                                  (BASIC_BLOCK_K_256_BYTE / args.bDtypeSize) :
                                  (args.isBTrans ? BASIC_BLOCK_SIZE_16 : innerAlignUnit / args.bDtypeSize);

    // 计算候选解集的上界
    uint64_t maxBaseM =
        GetMaxBaseWithLimit(compileInfo, args, runInfo, baseMNBufferLimit, baseMAlignUnit, false, isMemoryBound);
    uint64_t maxBaseN =
        GetMaxBaseWithLimit(compileInfo, args, runInfo, baseMNBufferLimit, baseNAlignUnit, true, isMemoryBound);

    runInfo.baseM = std::max(BASIC_BLOCK_SIZE_16, std::min(maxBaseM, BASIC_BLOCK_SIZE_256));
    runInfo.baseN = std::max(
        BASIC_BLOCK_SIZE_16,
        std::min(maxBaseN, ops::FloorAlign(baseMNBufferLimit / DATA_SIZE_FP32 / runInfo.baseM, baseNAlignUnit)));
    runInfo.cubeBoundParam = (1.0 / runInfo.baseM) + (1.0 / runInfo.baseN);
    runInfo.cubeBoundEdge = runInfo.cubeBoundEdge * 0.85;
    double balanceRate = GetBalanceRateWithTail(args, runInfo.usedCoreNum, runInfo.baseM, runInfo.baseN);

    for (uint64_t curBaseM = maxBaseM; curBaseM >= 1 && curBaseM <= maxBaseM; curBaseM -= baseMAlignUnit) {
        uint64_t curMaxBaseN =
            std::min(maxBaseN, ops::FloorAlign(baseMNBufferLimit / DATA_SIZE_FP32 / curBaseM, baseNAlignUnit));
        for (uint64_t curBaseN = curMaxBaseN; curBaseN >= 1 && curBaseN <= curMaxBaseN; curBaseN -= baseNAlignUnit) {
            double curCubeBoundParam = (1.0 / curBaseM) + (1.0 / curBaseN);
            double curBalanceRate = GetBalanceRateWithTail(args, runInfo.usedCoreNum, curBaseM, curBaseN);
            // 当前最优解满足负载均衡阈值时，本轮解集无法在计算访存拿到收益时过滤本轮解集
            if (balanceRate >= balanceRateEdge && curCubeBoundParam > runInfo.cubeBoundParam &&
                curCubeBoundParam > runInfo.cubeBoundEdge) {
                continue;
            }
            // 当前解满足cubebound并且负载均衡率更高
            bool cubeBoundCond = curCubeBoundParam <= runInfo.cubeBoundEdge && curBalanceRate > balanceRate;
            // 综合评选负载均衡和计算访存能力
            bool balanceCond = ((curCubeBoundParam / curBalanceRate) < (runInfo.cubeBoundParam / balanceRate)) ||
                               ((curCubeBoundParam / curBalanceRate == runInfo.cubeBoundParam / balanceRate) &&
                                curBalanceRate > balanceRate);
            if (cubeBoundCond || balanceCond) {
                runInfo.baseM = curBaseM;
                runInfo.baseN = curBaseN;
                runInfo.cubeBoundParam = curCubeBoundParam;
                balanceRate = curBalanceRate;
            }
        }
    }
    runInfo.baseM = std::min(ops::CeilAlign(args.mValue, BASIC_BLOCK_SIZE_16), runInfo.baseM);
    runInfo.baseN = std::min(ops::CeilAlign(args.nValue, BASIC_BLOCK_SIZE_16), runInfo.baseN);
    GetBaseK(compileInfo, args, runInfo);
    uint64_t mCore = MathUtil::CeilDivision(args.mValue, runInfo.baseM);
    uint64_t nCore = MathUtil::CeilDivision(args.nValue, runInfo.baseN);
    runInfo.usedCoreNum = std::min(batchNum * mCore * nCore, runInfo.usedCoreNum);
    runInfo.dbL0C = runInfo.baseM * runInfo.baseN * DATA_SIZE_FP32 * DB_SIZE <= compileInfo.l0CSize ? DB_SIZE : 1UL;
    runInfo.mixInfo.ubDB = runInfo.baseM * runInfo.baseN * DATA_SIZE_FP32 <= compileInfo.ubSize ? DB_SIZE : 1UL;
}

double MatMulV3TilingHelper::GetHbmBW(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args,
                                      fe::PlatFormInfos* platformInfo)
{
    std::string coreCntStr = "";
    std::string ddrRateStr = "";
    platformInfo->GetPlatformRes("SoCInfo", "ai_core_cnt", coreCntStr);
    platformInfo->GetPlatformRes("AICoreMemoryRates", "ddr_rate", ddrRateStr);
    return GetCoreFreq(compileInfo, args, platformInfo) * std::atoi(coreCntStr.c_str()) *
           std::atoi(ddrRateStr.c_str()) / KB_SIZE;
}

double MatMulV3TilingHelper::GetL2BW(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args,
                                     fe::PlatFormInfos* platformInfo)
{
    std::string coreCntStr = "";
    std::string l2RateStr = "";
    platformInfo->GetPlatformRes("SoCInfo", "ai_core_cnt", coreCntStr);
    platformInfo->GetPlatformRes("AICoreMemoryRates", "l2_rate", l2RateStr);
    return GetCoreFreq(compileInfo, args, platformInfo) * std::atoi(coreCntStr.c_str()) * std::atoi(l2RateStr.c_str()) /
           KB_SIZE;
}

double MatMulV3TilingHelper::GetCoreFreq(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args,
                                         fe::PlatFormInfos* platformInfo)
{
    std::string freqStr = "";
    platformInfo->GetPlatformRes("AICoreSpec", "cube_freq", freqStr);
    return std::atoi(freqStr.c_str()) / static_cast<double>(THOUSAND_NUM);
}
}  // namespace matmul_v3_advanced
}  // namespace optiling