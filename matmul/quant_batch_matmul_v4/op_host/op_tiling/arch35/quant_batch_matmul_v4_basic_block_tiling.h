/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v4_basic_block_tiling.h
 * \brief
 */

#ifndef QUANT_BATCH_MATMUL_V4_BASIC_BLOCK_TILING_H
#define QUANT_BATCH_MATMUL_V4_BASIC_BLOCK_TILING_H

#include <algorithm>
#include <limits>
#include <map>
#include <tuple>
#include <vector>

#include "error_util.h"

using std::ignore;
using std::make_tuple;
using std::map;
using std::max;
using std::min;
using std::numeric_limits;
using std::sort;
using std::tie;
using std::tuple;
using std::vector;

namespace optiling {
namespace matmul_v4 {

constexpr double BYTE_BITS = 8;
constexpr int64_t BITS_16 = 16;
constexpr int64_t BITS_4 = 4;
constexpr int64_t BLOCK_CUBE = 16;
constexpr int64_t BUFF_NUM_1 = 1;
constexpr int64_t BUFF_NUM_2 = 2;
constexpr int64_t BUFF_NUM_3 = 3;
constexpr int64_t BUFF_NUM_4 = 4;
constexpr int64_t STEPK_CUSTOM = 2;
constexpr int64_t BASE_MN_LIMIT_BUFF_1 = 65536;
constexpr int64_t BASE_MN_LIMIT_BUFF_2 = 32768;
constexpr int64_t BASE_MK_LIMIT = 32768;
constexpr int64_t BASE_BLOCK_MAX = 512;
constexpr int64_t MAX_BL1_SIZE = 98304; // 96 * 1024
constexpr int64_t MAX_BUB_ELEM_SIZE = 24576;
constexpr int64_t MTE2_AIV_RATE = 16;
constexpr int64_t MTE2_AIC_RATE = 32;
constexpr int64_t VF_PROCESS_ELEM = 128;
constexpr int64_t CUBE_ELEM_FP8 = 8192;
constexpr int64_t VF_RATE_MXA8W4 = 48;
constexpr int64_t VF_RATE_FPA8W4 = 22;
constexpr int64_t VF_MASK_SIZE = 32;
constexpr int64_t NZ_BASIC_BLOCK_ALIGN_SIZE = 64;
constexpr int64_t NZ_SINGLE_N_ALIGN_SIZE = 32;
constexpr int64_t BYTE_SIZE = 8;
constexpr int64_t SCALE_FACTOR_MAX = 2;

struct PlatformParam {
    int64_t blockNum = 0;
    int64_t aicNum = 0;
    int64_t ubSize = 0;
    int64_t l1Size = 0;
    int64_t l0aSize = 0;
    int64_t l0bSize = 0;
    int64_t l0cSize = 0;
    int64_t cacheLine = 0;
    int64_t minCacheLine = 0;

    double frequency = 0.0;
    double hbmBW = 0.0;
    double l2BW = 0.0;
};

struct BasicBlock {
    int64_t baseM = 0;
    int64_t baseN = 0;
    int64_t baseK = 0;
    double mte2BW = 0.0;
    double mte2MinBW = 0.0;
    double mte2BWRatio = 0.0;
    double mte2TailBWRatio = 0.0;
};

struct L1TilingParam {
    int64_t iterateOrder = 0;  // 0: orderM, for m for n; 1: orderN for n for m
    int64_t stepM = 0;
    int64_t stepN = 0;
    int64_t stepKa = 0;
    int64_t stepKb = 0;
    int64_t A1BufferNum = 0;
    int64_t B1BufferNum = 0;
    int64_t scaleFactor = 0;
};

struct BasicBlockParam {
    int64_t mSize = 0;
    int64_t nSize = 0;
    int64_t kSize = 0;
    int64_t singleM = 0;
    int64_t singleN = 0;
    int64_t singleK = 0;
    int64_t mDim = 0;
    int64_t nDim = 0;
    int64_t kDim = 0;
    int64_t mte2DataSize = 0;
    int64_t fixpDataSize = 0;
    int64_t groupSize = 0;
    int64_t cycleVF = 0;
    int64_t cycleBMte2 = 0;
    int64_t cycleAMte2 = 0;
    int64_t cycleCube = 0;
    int64_t mxType = 0;

    L1TilingParam l1Param;
    BasicBlock basicBlock;

    BasicBlockParam() : singleM(1), singleN(1), singleK(1) {}
};

struct StepKParam {
    int64_t stepKMax;
    int64_t stepKaTmp;
    int64_t stepKbTmp;
};

class QuantBatchMatmulV4BasicBlockTiling {
public:
    QuantBatchMatmulV4BasicBlockTiling()
    {
        Init();
    }
    ~QuantBatchMatmulV4BasicBlockTiling() = default;

    void Init();
    void Reset();
    void SetPlatformParam(const PlatformParam &param);
    void SetShape(int64_t mSize, int64_t nSize, int64_t kSize, int64_t groupSize);
    void SetAttr(const char *opName, bool transA, bool transB, bool hasBias, bool weightNzFlag);
    void SetQuantType(bool isMxType);
    void SetDtypeBits(const int64_t aDtypeBits, const int64_t bDtypeBits,
        const int64_t biasDtypeBits, const int64_t yScaleDtypeBits);
    double GetMinMte2BW(int64_t baseM, int64_t baseN, int64_t mDim, int64_t nDim) const;
    double GetMte2BW(int64_t baseM, int64_t baseN, int64_t mDim, int64_t nDim) const;
    double GetMte2BWRatio(int64_t baseM, int64_t baseN, int64_t mDim, int64_t nDim) const;
    int64_t GetCycleVF(const BasicBlockParam &basicBlockParam) const;
    int64_t GetCycleBMte2(const BasicBlockParam &basicBlockParam) const;
    int64_t GetCycleAMte2(const BasicBlockParam &basicBlockParam) const;
    int64_t GetCycleCube(const BasicBlockParam &basicBlockParam) const;
    bool GetBasicBlockTiling();
    const BasicBlockParam &GetTilingResult() const
    {
        return basicBlockParam_;
    }

protected:
    void InitBasicBlockParam();
    void InitL1TilingParam();
    void InitPlarformParam();
    void GetNdBasicBlockBaseKSolution(const int64_t baseM, const int64_t baseN);
    void GetNzBasicBlockBaseKSolution(const int64_t baseM, const int64_t baseN);
    void GetBasicBlockSolution();
    bool ValidateInputParam() const;
    void GetMte2DataSize(BasicBlockParam &basicBlockParam, int64_t &mte2DataSize, int64_t &iterateOrder) const;
    void GetMte2DataSizeGroup(BasicBlockParam &basicBlockParam, int64_t &mte2DataSize, int64_t &iterateOrder) const;
    void GetMte2DataSizeMx(BasicBlockParam &basicBlockParam, int64_t &mte2DataSize) const;
    void UpdateL1Param(L1TilingParam &l1TilingParam, int64_t &mte2DataSize, double &mte2Cost, bool &updateFlag);
    bool GetL1TilingResult(L1TilingParam &l1TilingParam, int64_t &mte2DataSize, double &mte2Cost);
    void UpdateL1ParamWithScaleFactor(L1TilingParam& l1TilingParam, int64_t& mte2DataSize, double& mte2Cost, bool& ret, const StepKParam& stepKParams);
    bool CheckL1TilingInvalid(int64_t stepKa, int64_t stepKb, int64_t stepKMax);
    bool GetFinalResult();
    bool ValidateTilingResult() const;
    void AddOptionalSolution();
    void PrintFinalResult(const BasicBlockParam &param, bool enable) const;
    int64_t GetL1LoadSize(const BasicBlock &basicBlock, const L1TilingParam &l1Param) const;

    /*
        解的排序函数，优先级:
        1）单核上mte2数据量更小的
        2）mte2总数据量最小的
        3）负载均衡
        4）选择理论cycle小的解
        5）优先选择mte2Cost较小的tiling
    */
    static bool CompareOptionalResult(const BasicBlockParam &param1, const BasicBlockParam &param2)
    {
        if (param1.mxType == 1) {
            // 取单核上 mte2DataSize更小的
            int64_t blockNum1 = param1.nDim * param1.mDim;
            int64_t blockNum2 = param2.nDim * param2.mDim;

            if (param1.mte2DataSize / blockNum1 != param2.mte2DataSize / blockNum2) {
                return param1.mte2DataSize / blockNum1 < param2.mte2DataSize / blockNum2;
            }

            // 取mte2DataSize总数据量更小的
            if (param1.mte2DataSize != param2.mte2DataSize) {
                return param1.mte2DataSize < param2.mte2DataSize;
            }
        }

        if (param1.mDim * param1.nDim != param2.mDim * param2.nDim) {
            return param1.mDim * param1.nDim > param2.mDim * param2.nDim;
        }

        // 1）选cycle最小的解
        const int64_t bCycle1 = max(param1.cycleVF, param1.cycleBMte2);
        const int64_t bCycle2 = max(param2.cycleVF, param2.cycleBMte2);
        if (max(max(bCycle1, param1.cycleAMte2), param1.cycleCube) !=
            max(max(bCycle2, param2.cycleAMte2), param2.cycleCube)) {
            return max(max(bCycle2, param2.cycleAMte2), param2.cycleCube) >
                max(max(bCycle1, param1.cycleAMte2), param1.cycleCube);
        }
        int64_t nBl1TailSize1 = param1.singleN % max(param1.l1Param.stepN * param1.basicBlock.baseN, BLOCK_CUBE);
        nBl1TailSize1 = nBl1TailSize1 == 0 ? param1.l1Param.stepN * param1.basicBlock.baseN : nBl1TailSize1;
        int64_t nBl1TailSize2 = param2.singleN % max(param2.l1Param.stepN * param2.basicBlock.baseN, BLOCK_CUBE);
        nBl1TailSize2 = nBl1TailSize2 == 0 ? param2.l1Param.stepN * param2.basicBlock.baseN : nBl1TailSize2;
        const int64_t bubLoadSize1 = CeilAlign(CeilDiv(nBl1TailSize1, BUFF_NUM_2), BLOCK_CUBE) * param1.l1Param.stepKb *
                                param1.basicBlock.baseK;
        const int64_t bubLoadSize2 = CeilAlign(CeilDiv(nBl1TailSize2, BUFF_NUM_2), BLOCK_CUBE) * param2.l1Param.stepKb *
                                param2.basicBlock.baseK;
        // 2）选Bub的载入量较大的解，即优先保证B矩阵的MTE2效率
        if (bubLoadSize1 != bubLoadSize2) {
            return bubLoadSize1 > bubLoadSize2;
        }
        // 3）其次，保证基本块选取合理，保证mad效率；经测试，mte2BWRatio越大的解，L0切分越好（越接近cube
        // bound基本块）。 另外，测试发现L0切分较好的情况下，mte2重复载入量也较小。
        double epsilon = 1e-9;
        if (std::abs(param1.basicBlock.mte2BWRatio - param2.basicBlock.mte2BWRatio) > epsilon) {
            return param1.basicBlock.mte2BWRatio > param2.basicBlock.mte2BWRatio;
        }
        int64_t al1TailSize1 = param1.singleK % max(param1.l1Param.stepKa * param1.basicBlock.baseK, BLOCK_CUBE);
        al1TailSize1 = al1TailSize1 == 0 ? param1.l1Param.stepKa * param1.basicBlock.baseK : al1TailSize1;
        int64_t al1TailSize2 = param2.singleK % max(param2.l1Param.stepKa * param2.basicBlock.baseK, BLOCK_CUBE);
        al1TailSize2 = al1TailSize2 == 0 ? param2.l1Param.stepKa * param2.basicBlock.baseK : al1TailSize2;
        // 4）微调AL1切分，此处希望单次AL1载入量越大越好，尾块越小越好
        if (al1TailSize1 != al1TailSize2) {
            return al1TailSize1 > al1TailSize2;
        } else {
            const int64_t al1LoadSize1 =
                param1.l1Param.stepM * param1.basicBlock.baseM * param1.l1Param.stepKa * param1.basicBlock.baseK;
            const int64_t al1LoadSize2 =
                param2.l1Param.stepM * param2.basicBlock.baseM * param2.l1Param.stepKa * param2.basicBlock.baseK;
            return al1LoadSize1 > al1LoadSize2;
        }
    }

    static double GetMte2Cost(int64_t mte2DataSize, double mte2BW)
    {
        double mte2Cost = numeric_limits<double>::max();
        if (mte2BW > 0) {
            mte2Cost = static_cast<double>(mte2DataSize) / mte2BW;
        }
        return mte2Cost;
    }

    static int64_t CeilDiv(int64_t num1, int64_t num2)
    {
        if (num2 == 0) {
            return 0;
        }
        return (num1 + num2 - 1) / num2;
    }

    static int64_t CeilAlign(int64_t num1, int64_t num2)
    {
        if (num2 == 0) {
            return 0;
        }
        return (num1 + num2 - 1) / num2 * num2;
    }

    const char *opName_;

    bool transA_;
    bool transB_;
    bool hasBias_;
    bool weightNzFlag_;
    bool isMxType_;

    double aByteSize_;
    double bByteSize_;
    double biasByteSize_;
    double yScaleByteSize_;

    int64_t aDtypeBits_;
    int64_t bDtypeBits_;
    int64_t biasDtypeBits_;

    int64_t vFRate_;

    PlatformParam platformParam_;
    vector<BasicBlockParam> optionalResults_;
    BasicBlockParam basicBlockParam_;
};

}  // namespace matmul_v4
}  // namespace optiling

#endif  // QUANT_BATCH_MATMUL_V4_BASIC_BLOCK_TILING_H
