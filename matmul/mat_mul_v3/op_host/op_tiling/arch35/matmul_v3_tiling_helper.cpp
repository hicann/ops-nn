/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_v3_tiling_helper.cc
 * \brief
 */

#include "matmul_v3_tiling_helper.h"
#include "matmul/common/op_host/math_util.h"

using Ops::NN::MathUtil;
namespace {
using namespace optiling;
using namespace optiling::matmul_v3_advanced;

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
}

void ResetBase91095(const MatmulV3CompileInfo &compileInfo, const MatMulV3Args &args, MatMulV3RunInfo &runInfo)
{
    ResetBaseDefault(compileInfo, args, runInfo);
    runInfo.baseM = BASIC_BLOCK_SIZE_256;
    runInfo.singleCoreM = runInfo.baseM;
}

using ResetBaseFunc = void (*)(const MatmulV3CompileInfo &, const MatMulV3Args &, MatMulV3RunInfo &);

const static std::map<platform_ascendc::SocVersion, ResetBaseFunc> ResetBaseFuncMap = {
    {platform_ascendc::SocVersion::ASCEND910_95, ResetBase91095},
};

// ------------------------------ CalL1Tiling -------------------------------------------//
void CalL1TilingDefault(const MatmulV3CompileInfo &compileInfo, const MatMulV3Args &args, MatMulV3RunInfo &runInfo)
{
    uint64_t totalL1Size = compileInfo.l1Size;
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
    if (runInfo.baseM == BASIC_BLOCK_SIZE_256 && runInfo.baseN == BASIC_BLOCK_SIZE_256 &&
        args.mValue % BASIC_BLOCK_SIZE_16 == 0 && args.nValue % BASIC_BLOCK_SIZE_16 == 0 &&
        args.kValue % BASIC_BLOCK_SIZE_16 == 0 && runInfo.singleCoreK <= BASIC_BLOCK_SIZE_256) {
        runInfo.stepKa = std::min(runInfo.stepKa, 2UL);
        runInfo.stepKb = std::min(runInfo.stepKb, 2UL);
    }
    if (runInfo.stepKa >= runInfo.stepKb) {
        runInfo.stepKa = runInfo.stepKa / runInfo.stepKb * runInfo.stepKb;
    } else {
        runInfo.stepKb = runInfo.stepKb / runInfo.stepKa * runInfo.stepKa;
    }
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

const static std::map<platform_ascendc::SocVersion, CalL1TilingFunc> CalL1TilingFuncMap = {
    {platform_ascendc::SocVersion::ASCEND310P, CalL1Tiling310P},
};

// ------------------------------ GetL0C2Out -------------------------------------------//
MatMulV3L0C2Out GetL0C2OutDefault(const MatmulV3CompileInfo & /* compileInfo */, const MatMulV3Args & /* args */,
                              const MatMulV3RunInfo & /* runInfo */)
{
    return MatMulV3L0C2Out::ON_THE_FLY;
}

MatMulV3L0C2Out GetL0C2Out91095(const MatmulV3CompileInfo &compileInfo, const MatMulV3Args &args,
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
    if (!fixpipeBound) {
        return MatMulV3L0C2Out::ON_THE_FLY;
    }
    if (args.aType == ge::DT_FLOAT16 || args.aType == ge::DT_BF16) {
        return MatMulV3L0C2Out::ND_FIXPIPE_1_1;
    }
    return MatMulV3L0C2Out::ND_FIXPIPE_1_2;
}

using GetL0C2OutFunc = MatMulV3L0C2Out (*)(const MatmulV3CompileInfo &, const MatMulV3Args &, const MatMulV3RunInfo &);

const static std::map<platform_ascendc::SocVersion, GetL0C2OutFunc> GetL0C2OutFuncMap = {
    {platform_ascendc::SocVersion::ASCEND910_95, GetL0C2Out91095},
};

// ------------------------------ CheckIfDoubleAswt -------------------------------------------//
bool CheckIfDoubleAswtDefault(const MatMulV3Args & /* args */, const uint64_t /* batchC */)
{
    return false;
}

bool CheckIfDoubleAswt91095(const MatMulV3Args &args, const uint64_t batchC)
{
    constexpr uint64_t halfL2Size = 64UL * 1024UL * 1024UL;  // 64mb
    constexpr uint64_t cubeBoundRatio = 512UL;
    if (batchC * args.mValue * args.nValue * args.aDtypeSize < halfL2Size) {  // check matC exceed half L2
        return false;
    }
    if ((args.mValue * args.nValue / (args.mValue + args.nValue)) > cubeBoundRatio) {  // check if cube bound or streamk
        return false;
    }
    if (args.kValue > (args.mValue >> 1) ||
        args.kValue > (args.nValue >> 1)) {  // check if matA or matb occupies most of L2
        return false;
    }
    return true;
}

using CheckIfDoubleAswtFunc = bool (*)(const MatMulV3Args &, const uint64_t);

const static std::map<platform_ascendc::SocVersion, CheckIfDoubleAswtFunc> CheckIfDoubleAswtFuncMap = {
    {platform_ascendc::SocVersion::ASCEND910_95, CheckIfDoubleAswt91095},
};
}  // namespace

namespace optiling {
namespace matmul_v3_advanced {
void MatMulV3TilingHelper::ResetBase(const MatmulV3CompileInfo &compileInfo, const MatMulV3Args &args,
                                     MatMulV3RunInfo &runInfo)
{
    auto iter = (ResetBaseFuncMap.find(compileInfo.socVersion) == ResetBaseFuncMap.end())
                    ? ResetBaseDefault
                    : ResetBaseFuncMap.at(compileInfo.socVersion);
    iter(compileInfo, args, runInfo);
}

void MatMulV3TilingHelper::CalL1Tiling(const MatmulV3CompileInfo &compileInfo, const MatMulV3Args &args,
                                       MatMulV3RunInfo &runInfo)
{
    auto iter = (CalL1TilingFuncMap.find(compileInfo.socVersion) == CalL1TilingFuncMap.end())
                    ? CalL1TilingDefault
                    : CalL1TilingFuncMap.at(compileInfo.socVersion);
    iter(compileInfo, args, runInfo);
}

MatMulV3L0C2Out MatMulV3TilingHelper::GetL0C2Out(const MatmulV3CompileInfo &compileInfo, const MatMulV3Args &args,
                                                 const MatMulV3RunInfo &runInfo)
{
    auto iter = (GetL0C2OutFuncMap.find(compileInfo.socVersion) == GetL0C2OutFuncMap.end())
                    ? GetL0C2OutDefault
                    : GetL0C2OutFuncMap.at(compileInfo.socVersion);
    return iter(compileInfo, args, runInfo);
}

bool MatMulV3TilingHelper::CheckIfDoubleAswt(const MatmulV3CompileInfo &compileInfo, const MatMulV3Args &args,
                                             const uint64_t batchC)
{
    auto iter = (CheckIfDoubleAswtFuncMap.find(compileInfo.socVersion) == CheckIfDoubleAswtFuncMap.end())
                    ? CheckIfDoubleAswtDefault
                    : CheckIfDoubleAswtFuncMap.at(compileInfo.socVersion);
    return iter(args, batchC);
}
}  // namespace matmul_v3_advanced
}  // namespace optiling