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
 * \file conv_base_blockdim_decision.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CONV_BASE_BLOCK_DIM_DECISION_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CONV_BASE_BLOCK_DIM_DECISION_H

#include "conv_template_utils.h"
using namespace std;
namespace optiling {
namespace conv_ops_tiling {

uint32_t Gcd(uint32_t i, uint32_t j);

uint64_t ConvCeilDiv(uint64_t a, uint64_t b);
void ConvCalcCommFactor(const uint64_t num, const uint32_t numMax, vector<uint32_t> &reslist);
uint32_t ConvAlignB(uint32_t a, uint32_t b);
uint64_t ConvInferHiL1(uint64_t inputHoL1, uint64_t hi, uint64_t singlekH, uint32_t dilationH, uint32_t strideH);
uint64_t ConvInferWiL1(uint64_t inputWoL1, uint64_t wi, uint64_t singlekW, uint32_t dilationW, uint32_t strideW);
int64_t ConvComputeHo(int64_t hi, int64_t hk, int64_t padTop, int64_t padBottom, int64_t dilationH, int64_t strideH);
int64_t ConvComputeWo(int64_t wi, int64_t wk, int64_t padLeft, int64_t padRight, int64_t dilationW, int64_t strideW);
int64_t ConvComputeDo(int64_t di, int64_t dk, int64_t padHead, int64_t padTail, int64_t dilationD, int64_t strideD);
void ConvBlockDimFactorMix(uint32_t orgDim, vector<uint32_t> &inputRange, const vector<uint32_t> &mixRange);
void InitblockDimConstParas(ConvOpsConstParams& convOpsConstParams,
                            const ConvAscendcDescInfo& descInfo, const ConvAscendcShapesInfo& shapeInfo);

class __attribute__((visibility("default"))) ConvBaseDeci {
public:
    ConvBaseDeci(){};
    void SetMKN(uint32_t m0, uint32_t k0, uint32_t n0);
    void SetAiCoreNum(uint32_t aicoreNum);
    ge::graphStatus GetBlockDimInfo(ConvAscendcTilingInfo& tilingInfo);
    void ConvBaseInitAttrInfo(const ConvAscendcAttrInfo& attrInfo);
    void GetConvBaseCoreInfo(ConvOpsConstParams& convOpsConstParams);
    void InitblockDimConstParas();
    bool CheckInstrLimitsHWmode();
    bool CheckInstrLimitsMmode();
    uint64_t CalcMinUsedL1SizeInMsplitMode(uint64_t kAL1min, uint64_t kBL1min);
    uint64_t CalcMinUsedL1SizeInHWsplitMode(uint64_t kAL1min, uint64_t kBL1min, uint64_t wiAL1min);
    BlockDimRes BlockDimDecisionMsplitMode();
    BlockDimRes BlockDimDecisionHWsplitMode();
    ge::graphStatus CheckL1SizeLimitsInMSplitMode();
    ge::graphStatus CheckL1SizeLimitsInHWsplitMode();
    int32_t BlockDimDecision(BlockDimRes& blockDimRes);
private:
    void SetTilingInfo(ConvAscendcTilingInfo& tilingInfo);
    void ConvBaseInit(const ConvAscendcShapesInfo& shapeInfo,
                      const ConvAscendcDescInfo& descInfo, const ConvAscendcTilingFlag& flagInfo);
    void ConvBaseInitPlatformInfo (const ConvAscendcPlatformInfo& platformInfo);
    void ConvBaseInitNodeInfo (const string& nodeName, const string& nodeType);
    void GetBlockDimRangeCommon();
    void GetBlockDimRangeMsplitMode();
    void GetBlockDimInitMsplitMode();
    void CoreBlockDimDecisionMsplitMode();
    void BlockDimDecisionBackTrackMsplitMode(const vector<vector<uint32_t>> &inputRanges,
                                             uint32_t rangeIdx, vector<uint32_t> &record);
    uint64_t CalcTotalCostMsplitMode(uint32_t batchDim, uint32_t mDim,
                                     uint32_t nDim, uint32_t doDim, uint32_t groupDim);
    bool CmpCoreUtilize(const uint32_t curCoreUtilize, const uint32_t minCostCoreUtilize,
                        const uint32_t batchDim, const uint32_t doDim);
    bool CmpCoreUtilizeMsplitMode(uint32_t batchDim, uint32_t hoDim, uint32_t nDim, uint32_t doDim, uint32_t groupDim);
    bool CmpCoreUtilizeHWsplitMode(const vector<uint32_t> &record);
    bool SkipScaleBiasL1Size();
    uint64_t CalcCostHWsplitMode(const BlockDimRes &blockDimRes, const uint64_t ci1, const uint64_t ci0,
                                 const uint64_t co1);
    uint64_t CalcTotalCostHWsplitMode(const BlockDimRes &blockDimRes);
    void SeperateHoRangeHWsplitMode();
    void GetBlockDimRangeHWsplitMode();
    void GetBlockDimInitHWsplitMode();
    void SetBlockDimHWsplitMode(const vector<uint32_t> &record, const uint64_t curCost,
                                BlockDimRes &blockDimRes) const;
    void SetBlockDimMsplitMode(const vector<uint32_t> &record, uint64_t curCost);
    void BlockDimDecisionBackTrackHWsplitMode(const vector<vector<uint32_t>> &inputRanges,
                                              uint32_t rangeIdx, vector<uint32_t> &record);
    void CoreBlockDimDecisionHWsplitMode();
    void CheckCoreUsedupHWsplitMode();

    uint64_t GetMinBurstNum();
    uint32_t GetWeightBandWidthCoeff();

    ge::graphStatus SelectBlockDimMode();
    void GetBlockDimRes();
public:
    ConvAscendcShapesInfo shapeInfo_;
    ConvAscendcAttrInfo attrInfo_;
    ConvAscendcDescInfo descInfo_;
    ConvAscendcTilingFlag flagInfo_;
    ConvAscendcFeatureFlag featureFlagInfo_;
    ConvOriGroupInfo oriGroupInfo_;
    ConvOptGroupInfo optGroupInfo_;
    ConvAscendcPlatformInfo platformInfo_;
    ConvAscendcNodeInfo nodeInfo_;
    FixpipeInfo fixpipeInfo_;

    uint64_t l2Rate_ = 1;

    uint32_t aicoreNum_ = 0;
    uint32_t m0_ = 1;
    uint32_t k0_ = 1;
    uint32_t n0_ = 1;
    BlockDimRes blockDimRes_;
    BlockDimRange blockDimRanges_;
    vector<uint32_t> blockDimInit_;
    ConvOpsConstParams convOpsConstParams_;
};
}
}
#endif