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

using Ops::NN::MathUtil;
namespace {
using namespace optiling;
using namespace optiling::matmul_v3_advanced;

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
    if (!fixpipeBound) {
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

bool MatMulV3TilingHelper::IsSelfNonContiguous(gert::TilingContext* context)
{
    auto selfShape = context->GetInputShape(0)->GetOriginShape();
    auto mat2Shape = context->GetInputShape(1)->GetOriginShape();
    auto selfStorageShape = context->GetInputShape(0)->GetStorageShape();
    size_t selfDimNum = selfShape.GetDimNum();
    size_t mat2DimNum = mat2Shape.GetDimNum();
    // createView with TensorV2 & storageShape 1d ->  non contiguous
    return (context->InputIsView(0) && selfStorageShape.GetDimNum() == 1 && selfDimNum == 3 && mat2DimNum == 2);
}
}  // namespace matmul_v3_advanced
}  // namespace optiling