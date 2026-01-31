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
 * \file conv_base_blockdim_decision.cpp
 * \brief
 */
#include "conv_base_blockdim_decision.h"
#include "log/log.h"
#include <cmath>
#include <set>

namespace optiling {
namespace conv_ops_tiling {

uint32_t Gcd(uint32_t i, uint32_t j)
{
    return j > 0 ? Gcd(j, i % j) : i;
}

uint64_t ConvCeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

uint32_t ConvAlignB(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return 0;
    }
    return ((a + b - 1) / b) * b;
}

uint64_t ConvInferHiL1(uint64_t inputHoL1, uint64_t hi, uint64_t singlekH, uint32_t dilationH, uint32_t strideH)
{
    uint64_t khDilated = (singlekH - 1) * dilationH + 1;
    uint64_t tmpHiL1 = (inputHoL1 - 1) * strideH + khDilated;
    if (tmpHiL1 > hi) {
        tmpHiL1 = hi;
    }

    return tmpHiL1;
}

uint64_t ConvInferWiL1(uint64_t inputWoL1, uint64_t wi, uint64_t singlekW, uint32_t dilationW, uint32_t strideW)
{
    uint64_t kwDilated = (singlekW - 1) * dilationW + 1;
    uint64_t tmpWiL1 = (inputWoL1 - 1) * strideW + kwDilated;
    if (tmpWiL1 > wi) {
        tmpWiL1 = wi;
    }

    return tmpWiL1;
}

int64_t ConvComputeHo(int64_t hi, int64_t hk, int64_t padTop, int64_t padBottom, int64_t dilationH, int64_t strideH)
{
    if (strideH == 0) {
        return 1;
    }
    int64_t cmpHo = (hi + padTop + padBottom - dilationH * (hk - 1) - 1) / strideH + 1;
    return cmpHo;
}

int64_t ConvComputeWo(int64_t wi, int64_t wk, int64_t padLeft, int64_t padRight, int64_t dilationW, int64_t strideW)
{
    if (strideW == 0) {
        return 1;
    }
    int64_t cmpWo = (wi + padLeft + padRight - dilationW * (wk - 1) - 1) / strideW + 1;
    return cmpWo;
}

int64_t ConvComputeDo(int64_t di, int64_t dk, int64_t padHead, int64_t padTail, int64_t dilationD, int64_t strideD)
{
    if (strideD == 0) {
        return 1;
    }
    int64_t cmpDo = (di + padHead + padTail - dilationD * (dk - 1) - 1) / strideD + 1;
    return cmpDo;
}

bool IsWeightNZFormat(ge::Format weightFormat)
{
    return weightFormat == ge::Format::FORMAT_FRACTAL_Z || weightFormat == ge::Format::FORMAT_FRACTAL_Z_C04;
}

ge::graphStatus ConvBaseDeci::GetBlockDimInfo(ConvAscendcTilingInfo& tilingInfo)
{
    ConvBaseInit(tilingInfo.shapeInfo, tilingInfo.descInfo, tilingInfo.flagInfo);
    ConvBaseInitAttrInfo(tilingInfo.attrInfo);
    ConvBaseInitPlatformInfo(tilingInfo.platformInfo);
    ConvBaseInitNodeInfo(tilingInfo.nodeInfo.nodeName, tilingInfo.nodeInfo.nodeType);
    InitblockDimConstParas();
    GetConvBaseCoreInfo(tilingInfo.convOpsConstParams);
    if (BlockDimDecision(tilingInfo.blockDimRes) != 0) {
        return ge::GRAPH_FAILED;
    }
    SetTilingInfo(tilingInfo);
    return ge::GRAPH_SUCCESS;
}

void ConvBaseDeci::SetTilingInfo(ConvAscendcTilingInfo& tilingInfo)
{
    tilingInfo.flagInfo = flagInfo_;
}

void ConvBaseDeci::ConvBaseInit(const ConvAscendcShapesInfo& shapeInfo,
                                const ConvAscendcDescInfo& descInfo, const ConvAscendcTilingFlag& flagInfo)
{
    shapeInfo_ = shapeInfo;
    descInfo_ = descInfo;
    flagInfo_ = flagInfo;
}

void ConvBaseDeci::ConvBaseInitPlatformInfo(const ConvAscendcPlatformInfo& platformInfo)
{
    platformInfo_ = platformInfo;
    aicoreNum_ = platformInfo_.aicoreNum;
}

void ConvBaseDeci::ConvBaseInitNodeInfo(const string& nodeName, const string& nodeType)
{
    nodeInfo_.nodeName = nodeName;
    nodeInfo_.nodeType = nodeType;
}

void ConvBaseDeci::ConvBaseInitAttrInfo(const ConvAscendcAttrInfo& attrInfo)
{
    attrInfo_ = attrInfo;
}

void ConvBaseDeci::GetConvBaseCoreInfo(ConvOpsConstParams& convOpsConstParams)
{
    convOpsConstParams = convOpsConstParams_;
}

void ConvBaseDeci::SetAiCoreNum(uint32_t aicoreNum)
{
    aicoreNum_ = aicoreNum;
}

void ConvBaseDeci::InitblockDimConstParas()
{
    convOpsConstParams_.m0 = CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.fMapDtype), MKN_M_IDX);
    convOpsConstParams_.k0 = CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.weightDtype), MKN_K_IDX);
    convOpsConstParams_.n0 = CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.fMapDtype), MKN_N_IDX);
    convOpsConstParams_.ci1 = ConvCeilDiv(shapeInfo_.ci, convOpsConstParams_.k0);
    convOpsConstParams_.co1 = ConvCeilDiv(shapeInfo_.co, convOpsConstParams_.n0);
	SetAiCoreNum(platformInfo_.aicoreNum);
    SetMKN(CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.fMapDtype), MKN_M_IDX),
                     CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.fMapDtype), MKN_K_IDX),
                     CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.fMapDtype), MKN_N_IDX));
}

ge::graphStatus ConvBaseDeci::SelectBlockDimMode()
{
    flagInfo_.mSplitModeFlag = false;
    if (CheckL1SizeLimitsInMSplitMode() == ge::GRAPH_SUCCESS && CheckInstrLimitsMmode()) {
        flagInfo_.mSplitModeFlag = true;
        return ge::GRAPH_SUCCESS;
    }
    if (CheckL1SizeLimitsInHWsplitMode() == ge::GRAPH_SUCCESS && CheckInstrLimitsHWmode()) {
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

bool ConvBaseDeci::CheckInstrLimitsMmode()
{
    if (descInfo_.fMapFormat == ge::Format::FORMAT_NDHWC) {
        uint64_t loadAL1SrcNdMatixStride = shapeInfo_.ci * shapeInfo_.hi * shapeInfo_.wi * attrInfo_.dilationD;
        if (loadAL1SrcNdMatixStride > MAX_40_BIT_NUM) {
            stringstream ss;
            ss << nodeInfo_.nodeType.c_str() << " AscendC: Fmap can't enable m split mode due to DN2NZ's limits: ci("
               << shapeInfo_.ci << ") * hi(" << shapeInfo_.hi << ") * wi(" << shapeInfo_.wi << ") * di("
               << attrInfo_.dilationD << ") > " << MAX_40_BIT_NUM << ".";
            OP_LOGD(nodeInfo_.nodeName, "%s", ss.str().c_str());
            return false;
        }
    }
    return true;
}
 
bool ConvBaseDeci::CheckInstrLimitsHWmode()
{
    if (descInfo_.fMapFormat == ge::Format::FORMAT_NDHWC) {
        uint64_t fixpipeDstNdStride = shapeInfo_.wo * shapeInfo_.co;
        if (fixpipeDstNdStride > MAX_32_BIT_NUM) {
            OP_LOGE(nodeInfo_.nodeName,
                    "%s AscendC: Output shape not satisfy Fixpipe's limits: wout(%lu) * cout(%lu) > %lu.",
                    nodeInfo_.nodeType.c_str(), shapeInfo_.wo, shapeInfo_.co, MAX_32_BIT_NUM);
            return false;
        }
    }
    return true;
}

void ConvBaseDeci::SetMKN(uint32_t m0, uint32_t k0, uint32_t n0)
{
    m0_ = m0;
    k0_ = k0;
    n0_ = n0;
}

void ConvBaseDeci::GetBlockDimRes()
{
    if (flagInfo_.mSplitModeFlag) {
        blockDimRes_ = BlockDimDecisionMsplitMode();
        OP_LOGD(nodeInfo_.nodeName,
            "%s AscendC: batchDim / mDim / nDim / doDim / groupDim / minCost: %u, %u, %u, %u, %u, %lu.",
            nodeInfo_.nodeType.c_str(),
            blockDimRes_.batchDim,
            blockDimRes_.mDim,
            blockDimRes_.nDim,
            blockDimRes_.doDim,
            blockDimRes_.groupDim,
            blockDimRes_.minCost);
    } else {
        blockDimRes_ = BlockDimDecisionHWsplitMode();
        OP_LOGD(nodeInfo_.nodeName,
            "%s AscendC: batchDim / hoDim / nDim / doDim / groupDim / minCost: %u, %u, %u, %u, %u, %lu.",
            nodeInfo_.nodeType.c_str(),
            blockDimRes_.batchDim,
            blockDimRes_.hoDim,
            blockDimRes_.nDim,
            blockDimRes_.doDim,
            blockDimRes_.groupDim,
            blockDimRes_.minCost);
    }
}

int32_t ConvBaseDeci::BlockDimDecision(BlockDimRes& blockDimRes)
{
    if (SelectBlockDimMode() != ge::GRAPH_SUCCESS) {
        return -1;
    }
    GetBlockDimRes();
    blockDimRes = blockDimRes_;
    return 0;
}

uint64_t ConvBaseDeci::CalcTotalCostMsplitMode(uint32_t batchDim, uint32_t mDim,
                                           uint32_t nDim, uint32_t doDim, uint32_t groupDim)
{
    uint64_t curCi = flagInfo_.convGroupType != ConvGroupType::NORMAL_CONV ?
        (flagInfo_.convGroupType == ConvGroupType::ORI_GROUP_CONV ?
         oriGroupInfo_.ciPerGroup : optGroupInfo_.cinOpt) : shapeInfo_.ci;
    uint64_t ci1 = ConvCeilDiv(curCi, k0_);

    uint64_t curCo = flagInfo_.convGroupType != ConvGroupType::NORMAL_CONV ?
        (flagInfo_.convGroupType == ConvGroupType::ORI_GROUP_CONV ?
         oriGroupInfo_.coPerGroup : optGroupInfo_.coutOpt) : shapeInfo_.co;
    uint64_t co1 = ConvCeilDiv(curCo, n0_);

    uint64_t curGroups = flagInfo_.convGroupType == ConvGroupType::OPT_GROUP_CONV ?
        optGroupInfo_.groupOpt : attrInfo_.groups;

    uint64_t loadFeatureMapCost = ConvCeilDiv(shapeInfo_.batch, batchDim) * ConvCeilDiv(curGroups, groupDim) *
                                  ConvCeilDiv(shapeInfo_.dout, doDim) *
                                  ConvCeilDiv(ConvAlignB(shapeInfo_.hi * shapeInfo_.wi, m0_), mDim) *
                                  shapeInfo_.kd * ci1 * k0_;

    uint64_t loadWeightCost = ConvCeilDiv(curGroups, groupDim) *
                              shapeInfo_.kd * ci1 * shapeInfo_.kh * shapeInfo_.kw * k0_ *
                              ConvCeilDiv(shapeInfo_.batch, batchDim);
    // N dim full load to UB in opt group mode.
    if (flagInfo_.convGroupType != ConvGroupType::OPT_GROUP_CONV) {
        loadWeightCost *= ConvCeilDiv(co1 * n0_, nDim);
    }

    uint64_t loadOutputCost = ConvCeilDiv(shapeInfo_.batch, batchDim) * ConvCeilDiv(curGroups, groupDim) *
                              ConvCeilDiv(co1 * n0_, nDim) *
                              ConvCeilDiv(shapeInfo_.dout, doDim) *
                              ConvCeilDiv(ConvAlignB(shapeInfo_.ho * shapeInfo_.wo, m0_), mDim);

    uint64_t cubeCalcCost = ConvCeilDiv(shapeInfo_.batch, batchDim) * ConvCeilDiv(curGroups, groupDim) *
                            ConvCeilDiv(co1, nDim) * ConvCeilDiv(shapeInfo_.dout, doDim) *
                            shapeInfo_.kd * ci1 * shapeInfo_.kh * shapeInfo_.kw *
                            ConvCeilDiv(ConvCeilDiv(shapeInfo_.ho * shapeInfo_.wo, m0_), mDim);

    uint32_t curBwCoeff = GetWeightBandWidthCoeff();

    return (loadFeatureMapCost + (loadWeightCost * curBwCoeff) + loadOutputCost) / MIN_L2_BAND_WIDTH + cubeCalcCost;
}

bool ConvBaseDeci::SkipScaleBiasL1Size()
{
    return descInfo_.fMapFormat == ge::FORMAT_NCDHW &&
           !flagInfo_.quantFlag &&
           descInfo_.fMapDtype == ge::DataType::DT_INT8;
}

uint64_t ConvBaseDeci::CalcMinUsedL1SizeInMsplitMode(uint64_t kAL1min, uint64_t kBL1min)
{
    uint64_t fMapDtypeSize = dtypeSizeTab.at(descInfo_.fMapDtype);
    uint64_t biasDtypeSize = dtypeSizeTab.at(descInfo_.biasDtype);
    uint64_t weightDtypeSize = dtypeSizeTab.at(descInfo_.weightDtype);

    // ensure 32-byte alignment
    uint64_t nBL1min = convOpsConstParams_.n0;
    uint64_t biasUsedL1Size = (flagInfo_.hasBias && !SkipScaleBiasL1Size()) ?
        ConvAlignB(nBL1min * biasDtypeSize, C0_SIZE) : 0;
    uint64_t scaleUsedL1Size = !SkipScaleBiasL1Size() ?
        ConvAlignB(nBL1min * fixpipeInfo_.channelWiseCoeff * FP16_DTYPE_SIZE, C0_SIZE) : 0;
    uint64_t weightUsedL1Size = ConvAlignB(kBL1min * nBL1min * weightDtypeSize, C0_SIZE);
    uint64_t hoAL1min = std::min(convOpsConstParams_.m0 / shapeInfo_.wo + CONST_VALUE_2, shapeInfo_.ho);
    uint64_t hiAL1min = ConvInferHiL1(hoAL1min, shapeInfo_.hi, shapeInfo_.kh, attrInfo_.dilationH, attrInfo_.strideH);
    uint64_t fmapUsedL1Size =
        ConvAlignB(hiAL1min * shapeInfo_.wi * kAL1min * fMapDtypeSize, C0_SIZE);

    uint64_t minL1LoadSize = biasUsedL1Size + fmapUsedL1Size + weightUsedL1Size + scaleUsedL1Size;
    return minL1LoadSize;
}

uint64_t ConvBaseDeci::CalcMinUsedL1SizeInHWsplitMode(uint64_t kAL1min, uint64_t kBL1min, uint64_t wiAL1min)
{
    uint64_t fMapDtypeSize = dtypeSizeTab.at(descInfo_.fMapDtype);
    uint64_t biasDtypeSize = dtypeSizeTab.at(descInfo_.biasDtype);
    uint64_t weightDtypeSize = dtypeSizeTab.at(descInfo_.weightDtype);
    uint64_t nBL1min = convOpsConstParams_.n0;
    uint64_t biasUsedL1Size = (flagInfo_.hasBias && !SkipScaleBiasL1Size()) ?
        ConvAlignB(nBL1min * biasDtypeSize, C0_SIZE) : 0;
    uint64_t scaleUsedL1Size = !SkipScaleBiasL1Size() ?
        ConvAlignB(nBL1min * fixpipeInfo_.channelWiseCoeff * FP16_DTYPE_SIZE, C0_SIZE) : 0;
    uint64_t weightUsedL1Size = ConvAlignB(kBL1min * nBL1min * weightDtypeSize, C0_SIZE);

    uint64_t fmapUsedL1Size = 0;
    uint64_t hoAL1min = std::min(shapeInfo_.wo < convOpsConstParams_.m0 ?
                                 ConvCeilDiv(convOpsConstParams_.m0, shapeInfo_.wo) : 1, shapeInfo_.ho);
    uint64_t hiAL1min = ConvInferHiL1(hoAL1min, shapeInfo_.hi, shapeInfo_.kh, attrInfo_.dilationH, attrInfo_.strideH);
    fmapUsedL1Size = ConvAlignB(hiAL1min * wiAL1min * kAL1min * fMapDtypeSize, C0_SIZE);

    uint64_t minL1LoadSize = biasUsedL1Size + scaleUsedL1Size + fmapUsedL1Size + weightUsedL1Size;
    return minL1LoadSize;
}

ge::graphStatus ConvBaseDeci::CheckL1SizeLimitsInHWsplitMode()
{
    // require hiL1 * wiL1 >= m0
    if (featureFlagInfo_ == ConvAscendcFeatureFlag::IS_DMA_FLAG) {
        return ge::GRAPH_SUCCESS;
    }
    uint64_t woAL1min = convOpsConstParams_.m0;
    uint64_t wiAL1min = ConvInferWiL1(woAL1min, shapeInfo_.wi, shapeInfo_.kw, attrInfo_.dilationW, attrInfo_.strideW);
    uint64_t usdL1SizeUnderMinHWtiling = CalcMinUsedL1SizeInHWsplitMode(convOpsConstParams_.k0,
        shapeInfo_.kh * shapeInfo_.kw * convOpsConstParams_.k0, wiAL1min);
    if (usdL1SizeUnderMinHWtiling > platformInfo_.l1Size) {
        OP_LOGE(nodeInfo_.nodeName,
            "%s AscendC: MinL1LoadSize > L1size, current L1size: %lu, maxL1Size: %lu",
                nodeInfo_.nodeType.c_str(), usdL1SizeUnderMinHWtiling, platformInfo_.l1Size);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
 
ge::graphStatus ConvBaseDeci::CheckL1SizeLimitsInMSplitMode()
{
    uint64_t usdL1SizeUnderMinMtiling = CalcMinUsedL1SizeInMsplitMode(convOpsConstParams_.k0,
        shapeInfo_.kh * shapeInfo_.kw * convOpsConstParams_.k0);
    if (usdL1SizeUnderMinMtiling > platformInfo_.l1Size) {
        OP_LOGD(nodeInfo_.nodeName,
            "%s AscendC: MinL1LoadSizeInMmode > L1size, current L1size: %lu, maxL1Size: %lu",
            nodeInfo_.nodeType.c_str(), usdL1SizeUnderMinMtiling, platformInfo_.l1Size);
        return ge::GRAPH_FAILED;
    }
    // load3dv2 win max value
    if (shapeInfo_.wi > LOAD3DV2_WIN_LIMIT_VALUE) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

BlockDimRes ConvBaseDeci::BlockDimDecisionMsplitMode()
{
    GetBlockDimRangeMsplitMode();
    GetBlockDimInitMsplitMode();
    CoreBlockDimDecisionMsplitMode();
    return blockDimRes_;
}

void ConvBaseDeci::GetBlockDimRangeMsplitMode()
{
    GetBlockDimRangeCommon();
    uint64_t m1 = ConvCeilDiv(shapeInfo_.ho * shapeInfo_.wo, m0_); // mRange
    ConvCalcCommFactor(m1, aicoreNum_, blockDimRanges_.mRange);
    ConvBlockDimFactorMix(m1, blockDimRanges_.mRange, blockDimRanges_.aicNumRange);
}

void ConvBaseDeci::GetBlockDimInitMsplitMode()
{
    blockDimInit_.resize(BLOCKDIM_MSPLIT_DEC_NUM, 1);
    blockDimInit_[BLOCKDIM_MSPLIT_BATCH_IDX] = blockDimRanges_.batchRange[0];
    blockDimInit_[BLOCKDIM_MSPLIT_M_IDX] = blockDimRanges_.mRange[0];
    blockDimInit_[BLOCKDIM_MSPLIT_N_IDX] = blockDimRanges_.nRange[0];
    blockDimInit_[BLOCKDIM_MSPLIT_DO_IDX] = blockDimRanges_.doRange[0];
    blockDimInit_[BLOCKDIM_MSPLIT_GROUP_IDX] = blockDimRanges_.groupRange[0];
}

void ConvBaseDeci::CoreBlockDimDecisionMsplitMode()
{
    blockDimRes_.batchDim = 1;
    blockDimRes_.mDim = 1;
    blockDimRes_.nDim = 1;
    blockDimRes_.doDim = 1;
    blockDimRes_.groupDim = 1;
    blockDimRes_.minCost = CalcTotalCostMsplitMode(blockDimRes_.batchDim, blockDimRes_.mDim, blockDimRes_.nDim,
                                                   blockDimRes_.doDim, blockDimRes_.groupDim);
    vector<vector<uint32_t>> allRanges(BLOCKDIM_MSPLIT_DEC_NUM, vector<uint32_t>(1, 1));
    allRanges[BLOCKDIM_MSPLIT_BATCH_IDX] = blockDimRanges_.batchRange;
    allRanges[BLOCKDIM_MSPLIT_M_IDX] = blockDimRanges_.mRange;
    allRanges[BLOCKDIM_MSPLIT_N_IDX] = blockDimRanges_.nRange;
    allRanges[BLOCKDIM_MSPLIT_DO_IDX] = blockDimRanges_.doRange;
    allRanges[BLOCKDIM_MSPLIT_GROUP_IDX] = blockDimRanges_.groupRange;
    vector<uint32_t> dimsRecord;
    BlockDimDecisionBackTrackMsplitMode(allRanges, BLOCKDIM_MSPLIT_BATCH_IDX, dimsRecord);
}


void ConvBaseDeci::BlockDimDecisionBackTrackMsplitMode(const vector<vector<uint32_t>> &inputRanges,
                                                       uint32_t rangeIdx, vector<uint32_t> &record)
{
    if (record.size() == inputRanges.size()) {
        uint32_t curBlockDim = record[BLOCKDIM_MSPLIT_BATCH_IDX] * record[BLOCKDIM_MSPLIT_M_IDX] *
                               record[BLOCKDIM_MSPLIT_N_IDX] * record[BLOCKDIM_MSPLIT_DO_IDX] *
                               record[BLOCKDIM_MSPLIT_GROUP_IDX];
        if (curBlockDim > aicoreNum_) {
            return;
        }
        bool update_flag = false;
        uint64_t curCost = CalcTotalCostMsplitMode(record[BLOCKDIM_MSPLIT_BATCH_IDX],
                                                   record[BLOCKDIM_MSPLIT_M_IDX], record[BLOCKDIM_MSPLIT_N_IDX],
                                                   record[BLOCKDIM_MSPLIT_DO_IDX], record[BLOCKDIM_MSPLIT_GROUP_IDX]);
        if (curCost < blockDimRes_.minCost) {
            update_flag = true;
        } else if (curCost == blockDimRes_.minCost) {
            update_flag = CmpCoreUtilizeMsplitMode(record[BLOCKDIM_MSPLIT_BATCH_IDX], record[BLOCKDIM_MSPLIT_M_IDX],
                record[BLOCKDIM_MSPLIT_N_IDX], record[BLOCKDIM_MSPLIT_DO_IDX], record[BLOCKDIM_MSPLIT_GROUP_IDX]);
        }
        if (update_flag) {
            SetBlockDimMsplitMode(record, curCost);
        }
        return;
    }

    if (rangeIdx >= inputRanges.size() || rangeIdx >= blockDimInit_.size()) {
        return;
    }

    for (uint32_t i = 0; i < inputRanges[rangeIdx].size(); i++) {
        record.push_back(inputRanges[rangeIdx][i]);
        BlockDimDecisionBackTrackMsplitMode(inputRanges, rangeIdx + 1, record);
        record.pop_back();
    }
}

// hw split mode
BlockDimRes ConvBaseDeci::BlockDimDecisionHWsplitMode()
{
    GetBlockDimRangeHWsplitMode();
    GetBlockDimInitHWsplitMode();
    CoreBlockDimDecisionHWsplitMode();
    CheckCoreUsedupHWsplitMode();
    return blockDimRes_;
}

uint64_t ConvBaseDeci::CalcCostHWsplitMode(const BlockDimRes &blockDimRes,
    const uint64_t ci1, const uint64_t ci0, const uint64_t co1)
{
    const uint64_t curGroups = flagInfo_.convGroupType == ConvGroupType::OPT_GROUP_CONV ?
        optGroupInfo_.groupOpt : attrInfo_.groups;

    const uint64_t loadFeatureMapCost = ConvCeilDiv(shapeInfo_.batch, blockDimRes.batchDim) *
        ConvCeilDiv(curGroups, blockDimRes.groupDim) * ConvCeilDiv(shapeInfo_.dout, blockDimRes.doDim) *
        shapeInfo_.kd * ci1 * ConvCeilDiv(shapeInfo_.hi, blockDimRes.hoDim) *
        ConvCeilDiv(shapeInfo_.wi, blockDimRes.woDim) * ci0;

    const uint64_t weightKSize = flagInfo_.enableC04Flag ?
        static_cast<uint64_t>(ConvAlignB(ci1 * shapeInfo_.kh * shapeInfo_.kw, static_cast<uint64_t>(k0_))) :
        ci1 * shapeInfo_.kh * shapeInfo_.kw * static_cast<uint64_t>(k0_);
    uint64_t loadWeightCost = ConvCeilDiv(shapeInfo_.batch, blockDimRes.batchDim) *
        ConvCeilDiv(curGroups, blockDimRes.groupDim) * shapeInfo_.kd * weightKSize;

    // N dim full load to UB in opt group mode.
    if (flagInfo_.convGroupType != ConvGroupType::OPT_GROUP_CONV) {
        loadWeightCost *= ConvCeilDiv(co1 * n0_, blockDimRes.nDim);
    }

    const uint64_t loadOutputCost = ConvCeilDiv(shapeInfo_.batch, blockDimRes.batchDim) *
        ConvCeilDiv(curGroups, blockDimRes.groupDim) * ConvCeilDiv(shapeInfo_.dout, blockDimRes.doDim) *
        ConvCeilDiv(co1 * n0_, blockDimRes.nDim) * ConvCeilDiv(shapeInfo_.ho, blockDimRes.hoDim) *
        ConvCeilDiv(shapeInfo_.wo, blockDimRes.woDim);

    const uint64_t cubeCalcCost = ConvCeilDiv(shapeInfo_.batch, blockDimRes.batchDim) *
        ConvCeilDiv(curGroups, blockDimRes.groupDim) * ConvCeilDiv(co1, blockDimRes.nDim) *
        ConvCeilDiv(shapeInfo_.dout, blockDimRes.doDim) * shapeInfo_.kd * ci1 * shapeInfo_.kh * shapeInfo_.kw *
        ConvCeilDiv(ConvCeilDiv(shapeInfo_.ho, blockDimRes.hoDim) * ConvCeilDiv(shapeInfo_.wo, blockDimRes.woDim),
        m0_);

    uint32_t curBwCoeff = GetWeightBandWidthCoeff();
    return (loadFeatureMapCost + (loadWeightCost * curBwCoeff) + loadOutputCost) / MIN_L2_BAND_WIDTH + cubeCalcCost;
}

uint64_t ConvBaseDeci::CalcTotalCostHWsplitMode(const BlockDimRes &blockDimRes)
{
    uint64_t ci0 = k0_;
    uint64_t curCi = flagInfo_.convGroupType != ConvGroupType::NORMAL_CONV ?
        (flagInfo_.convGroupType == ConvGroupType::ORI_GROUP_CONV ?
         oriGroupInfo_.ciPerGroup : optGroupInfo_.cinOpt) : shapeInfo_.ci;
    uint64_t ci1 = ConvCeilDiv(curCi, ci0);

    uint64_t curCo = flagInfo_.convGroupType != ConvGroupType::NORMAL_CONV ?
        (flagInfo_.convGroupType == ConvGroupType::ORI_GROUP_CONV ?
         oriGroupInfo_.coPerGroup : optGroupInfo_.coutOpt) : shapeInfo_.co;
    uint64_t co1 = ConvCeilDiv(curCo, n0_);

    if (flagInfo_.enableC04Flag) {
        ci1 = C04_CI1_SIZE;
        ci0 = C04_CIN_SIZE;
    }
    
    return CalcCostHWsplitMode(blockDimRes, ci1, ci0, co1);
}

uint64_t ConvBaseDeci::GetMinBurstNum()
{
    return MIN_L2_BAND_WIDTH / dtypeSizeTab.at(descInfo_.fMapDtype);
}

uint32_t ConvBaseDeci::GetWeightBandWidthCoeff()
{
    if (descInfo_.fMapFormat == ge::Format::FORMAT_NCDHW || descInfo_.fMapFormat == ge::Format::FORMAT_NCHW) {
        if (IsWeightNZFormat(descInfo_.weightFormat)) {
            return BW_COEFF_UB;
        }
        if (flagInfo_.convGroupType != ConvGroupType::OPT_GROUP_CONV) {
            return BW_COEFF;
        }
    }
    return BW_COEFF_UB;
}

void ConvBaseDeci::SeperateHoRangeHWsplitMode()
{
    std::vector<uint32_t> tmpHoRange = blockDimRanges_.hoRange;
    uint64_t minValue = GetMinBurstNum() / shapeInfo_.wo;
    uint32_t firstValidIdx = tmpHoRange.size();
    for (uint32_t i = 0; i < tmpHoRange.size(); i++) {
        if (ConvCeilDiv(shapeInfo_.ho, tmpHoRange[i]) >= minValue) {
            firstValidIdx = i;
            break;
        }
    }
    blockDimRanges_.hoRange.clear();
    blockDimRanges_.hoSpareRange.clear();
    blockDimRanges_.hoRange.assign(tmpHoRange.begin() + firstValidIdx, tmpHoRange.end());
    blockDimRanges_.hoSpareRange.assign(tmpHoRange.begin(), tmpHoRange.begin() + firstValidIdx);
}

void ConvBaseDeci::GetBlockDimRangeCommon()
{
    // aicoreRange
    ConvCalcCommFactor(aicoreNum_, aicoreNum_, blockDimRanges_.aicNumRange);
    // batchRange
    ConvCalcCommFactor(shapeInfo_.batch, aicoreNum_, blockDimRanges_.batchRange);

    if (shapeInfo_.batch >= BATCH_AICORE_COF * aicoreNum_) {
        blockDimRanges_.batchRange = blockDimRanges_.aicNumRange;
    } else {
        ConvBlockDimFactorMix(shapeInfo_.batch, blockDimRanges_.batchRange, blockDimRanges_.aicNumRange);
    }
    // nRange
    uint64_t curCo = shapeInfo_.co;
    if (flagInfo_.convGroupType == ConvGroupType::ORI_GROUP_CONV) {
        curCo = oriGroupInfo_.coPerGroup;
    } else if (flagInfo_.convGroupType == ConvGroupType::OPT_GROUP_CONV) {
        curCo = optGroupInfo_.coutOpt;
    }
    ConvCalcCommFactor(ConvCeilDiv(curCo, n0_), aicoreNum_, blockDimRanges_.nRange);
    ConvBlockDimFactorMix(ConvCeilDiv(curCo, n0_), blockDimRanges_.nRange, blockDimRanges_.aicNumRange);

    // doRange
    if (descInfo_.fMapFormat == ge::Format::FORMAT_NCDHW || descInfo_.fMapFormat == ge::Format::FORMAT_NDHWC) {
        ConvCalcCommFactor(shapeInfo_.dout, aicoreNum_, blockDimRanges_.doRange);
        ConvBlockDimFactorMix(shapeInfo_.dout, blockDimRanges_.doRange, blockDimRanges_.aicNumRange);
    } else {
        blockDimRanges_.doRange.assign(1, 1);
    }
    // groupRange
    uint64_t curGroups = flagInfo_.convGroupType == ConvGroupType::OPT_GROUP_CONV ?
        optGroupInfo_.groupOpt : attrInfo_.groups;
    ConvCalcCommFactor(curGroups, aicoreNum_, blockDimRanges_.groupRange);
    ConvBlockDimFactorMix(curGroups, blockDimRanges_.groupRange, blockDimRanges_.aicNumRange);
}

void ConvBaseDeci::GetBlockDimRangeHWsplitMode()
{
    GetBlockDimRangeCommon();
    ConvCalcCommFactor(shapeInfo_.ho, aicoreNum_, blockDimRanges_.hoRange);
    ConvBlockDimFactorMix(shapeInfo_.ho, blockDimRanges_.hoRange, blockDimRanges_.aicNumRange);
    if (featureFlagInfo_ == ConvAscendcFeatureFlag::IS_CONV1D_FLAG) {
        ConvCalcCommFactor(shapeInfo_.wo, aicoreNum_, blockDimRanges_.woRange);
        ConvBlockDimFactorMix(shapeInfo_.wo, blockDimRanges_.woRange, blockDimRanges_.aicNumRange);
        return;
    }
    blockDimRanges_.woRange.emplace_back(1);

    if (shapeInfo_.wo < GetMinBurstNum()) {
        SeperateHoRangeHWsplitMode();
    }
}

void ConvCalcCommFactor(const uint64_t num, const uint32_t numMax, std::vector<uint32_t> &reslist)
{
    uint32_t sqrtMax = static_cast<uint32_t>(sqrt(num));
    for (uint32_t i = 1; i <= sqrtMax; ++i) {
        if (num % i == 0) {
            if (i <= numMax) {
                reslist.emplace_back(i);
            }
            uint32_t right = num / i;
            if (right != i && right <= numMax) {
                reslist.emplace_back(right);
            }
        }
    }
    sort(reslist.begin(), reslist.end());
}

void ConvBlockDimFactorMix(uint32_t orgDim, std::vector<uint32_t> &inputRange, const std::vector<uint32_t> &mixRange)
{
    std::vector<uint32_t> tmpSelectMixRange;
    for (auto v : mixRange) {
        if (v <= orgDim) {
            tmpSelectMixRange.push_back(v);
        }
    }
    std::set<uint32_t>tmpRanges(inputRange.begin(), inputRange.end());
    tmpRanges.insert(tmpSelectMixRange.begin(), tmpSelectMixRange.end());
    inputRange.assign(tmpRanges.begin(), tmpRanges.end());
}

void InitblockDimConstParas(ConvOpsConstParams& convOpsConstParams,
                            const ConvAscendcDescInfo& descInfo,
                            const ConvAscendcShapesInfo& shapeInfo)
{
    convOpsConstParams.m0 = CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo.fMapDtype), MKN_M_IDX);
    convOpsConstParams.k0 = CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo.weightDtype), MKN_K_IDX);
    convOpsConstParams.n0 = CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo.fMapDtype), MKN_N_IDX);
    convOpsConstParams.ci1 = ConvCeilDiv(shapeInfo.ci, convOpsConstParams.k0);
    convOpsConstParams.co1 = ConvCeilDiv(shapeInfo.co, convOpsConstParams.n0);
}

bool ConvBaseDeci::CmpCoreUtilize(const uint32_t curCoreUtilize, const uint32_t minCostCoreUtilize,
    const uint32_t batchDim, const uint32_t doDim)
{
    if (curCoreUtilize < minCostCoreUtilize) {
        return false;
    } else if (curCoreUtilize == minCostCoreUtilize) {
        // for same cost, preference: batch > dout
        bool updateFlag = std::make_tuple(batchDim, doDim) > std::make_tuple(blockDimRes_.batchDim, blockDimRes_.doDim);
        return updateFlag;
    }
    return true;
}

bool ConvBaseDeci::CmpCoreUtilizeMsplitMode(uint32_t batchDim, uint32_t mDim,
                                            uint32_t nDim, uint32_t doDim, uint32_t groupDim)
{
    const uint32_t curCoreUtilize = batchDim * mDim * nDim * doDim * groupDim / aicoreNum_;
    const uint32_t minCostCoreUtilize = blockDimRes_.batchDim * blockDimRes_.mDim * blockDimRes_.nDim *
                                  blockDimRes_.doDim * blockDimRes_.groupDim / aicoreNum_;
    return CmpCoreUtilize(curCoreUtilize, minCostCoreUtilize, batchDim, doDim);
}

bool ConvBaseDeci::CmpCoreUtilizeHWsplitMode(const vector<uint32_t> &record)
{
    auto batchDim = record[BLOCKDIM_HWSPLIT_BATCH_IDX];
    auto hoDim = record[BLOCKDIM_HWSPLIT_HO_IDX];
    auto woDim = record[BLOCKDIM_HWSPLIT_WO_IDX];
    auto nDim = record[BLOCKDIM_HWSPLIT_N_IDX];
    auto doDim = record[BLOCKDIM_HWSPLIT_DO_IDX];
    auto groupDim = record[BLOCKDIM_HWSPLIT_GROUP_IDX];
    const uint32_t curCoreUtilize = batchDim * hoDim * woDim * nDim * doDim * groupDim / aicoreNum_;
    const uint32_t minCostCoreUtilize = blockDimRes_.batchDim * blockDimRes_.hoDim * blockDimRes_.woDim *
                                        blockDimRes_.nDim * blockDimRes_.doDim * blockDimRes_.groupDim / aicoreNum_;
    return CmpCoreUtilize(curCoreUtilize, minCostCoreUtilize, batchDim, doDim);
}

void ConvBaseDeci::SetBlockDimHWsplitMode(const vector<uint32_t> &record, const uint64_t curCost,
    BlockDimRes &blockDimRes) const
{
    blockDimRes.batchDim = record[BLOCKDIM_HWSPLIT_BATCH_IDX];
    blockDimRes.hoDim = record[BLOCKDIM_HWSPLIT_HO_IDX];
    blockDimRes.woDim = record[BLOCKDIM_HWSPLIT_WO_IDX];
    blockDimRes.nDim = record[BLOCKDIM_HWSPLIT_N_IDX];
    blockDimRes.doDim = record[BLOCKDIM_HWSPLIT_DO_IDX];
    blockDimRes.groupDim = record[BLOCKDIM_HWSPLIT_GROUP_IDX];
    blockDimRes.minCost = curCost;
}

void ConvBaseDeci::SetBlockDimMsplitMode(const vector<uint32_t> &record, uint64_t curCost)
{
    blockDimRes_.batchDim = record[BLOCKDIM_MSPLIT_BATCH_IDX];
    blockDimRes_.mDim = record[BLOCKDIM_MSPLIT_M_IDX];
    blockDimRes_.nDim = record[BLOCKDIM_MSPLIT_N_IDX];
    blockDimRes_.doDim = record[BLOCKDIM_MSPLIT_DO_IDX];
    blockDimRes_.groupDim = record[BLOCKDIM_MSPLIT_GROUP_IDX];
    blockDimRes_.minCost = curCost;
}

void ConvBaseDeci::GetBlockDimInitHWsplitMode()
{
    blockDimInit_.resize(BLOCKDIM_HWSPLIT_DEC_NUM, 1);
    blockDimInit_[BLOCKDIM_HWSPLIT_BATCH_IDX] = blockDimRanges_.batchRange[0];
    blockDimInit_[BLOCKDIM_HWSPLIT_HO_IDX] = blockDimRanges_.hoRange.empty() ? 1 : blockDimRanges_.hoRange[0];
    blockDimInit_[BLOCKDIM_HWSPLIT_WO_IDX] = blockDimRanges_.woRange.empty() ? 1 : blockDimRanges_.woRange[0];
    blockDimInit_[BLOCKDIM_HWSPLIT_N_IDX] = blockDimRanges_.nRange[0];
    blockDimInit_[BLOCKDIM_HWSPLIT_DO_IDX] = blockDimRanges_.doRange[0];
    blockDimInit_[BLOCKDIM_HWSPLIT_GROUP_IDX] = blockDimRanges_.groupRange[0];
}

void ConvBaseDeci::BlockDimDecisionBackTrackHWsplitMode(const vector<vector<uint32_t>> &inputRanges,
                                                        uint32_t rangeIdx, vector<uint32_t> &record)
{
    if (record.size() == inputRanges.size()) {
        BlockDimRes blockDimResTemp;
        SetBlockDimHWsplitMode(record, 0, blockDimResTemp);
        const uint32_t curBlockDim = blockDimResTemp.batchDim * blockDimResTemp.hoDim *
            blockDimResTemp.woDim * blockDimResTemp.nDim * blockDimResTemp.doDim *
            blockDimResTemp.groupDim;
        if (curBlockDim > aicoreNum_) {
            return;
        }

        bool updateFlag = false;
        const uint64_t curCost = CalcTotalCostHWsplitMode(blockDimResTemp);
        if (curCost < blockDimRes_.minCost) {
            updateFlag = true;
        } else if (curCost == blockDimRes_.minCost) {
            updateFlag = CmpCoreUtilizeHWsplitMode(record);
        }
        if (updateFlag) {
            SetBlockDimHWsplitMode(record, curCost, blockDimRes_);
        }
        return;
    }

    if (rangeIdx >= inputRanges.size() || rangeIdx >= blockDimInit_.size()) {
        return;
    }

    for (uint32_t i = 0; i < inputRanges[rangeIdx].size(); i++) {
        record.push_back(inputRanges[rangeIdx][i]);
        BlockDimDecisionBackTrackHWsplitMode(inputRanges, rangeIdx + 1, record);
        record.pop_back();
    }
}

void ConvBaseDeci::CoreBlockDimDecisionHWsplitMode()
{
    blockDimRes_.batchDim = static_cast<uint32_t>(1);
    blockDimRes_.hoDim = static_cast<uint32_t>(1);
    blockDimRes_.woDim = static_cast<uint32_t>(1);
    blockDimRes_.nDim = static_cast<uint32_t>(1);
    blockDimRes_.doDim = static_cast<uint32_t>(1);
    blockDimRes_.groupDim = static_cast<uint32_t>(1);
    blockDimRes_.minCost = CalcTotalCostHWsplitMode(blockDimRes_);

    vector<vector<uint32_t>> allRanges(BLOCKDIM_HWSPLIT_DEC_NUM, vector<uint32_t>(1, 1));
    vector<uint32_t> tmpHoRange = {static_cast<uint32_t>(1)};
    allRanges[BLOCKDIM_HWSPLIT_BATCH_IDX] = blockDimRanges_.batchRange;
    allRanges[BLOCKDIM_HWSPLIT_HO_IDX] = blockDimRanges_.hoRange.empty() ? tmpHoRange : blockDimRanges_.hoRange;
    allRanges[BLOCKDIM_HWSPLIT_WO_IDX] = blockDimRanges_.woRange;
    allRanges[BLOCKDIM_HWSPLIT_N_IDX] = blockDimRanges_.nRange;
    allRanges[BLOCKDIM_HWSPLIT_DO_IDX] = blockDimRanges_.doRange;
    allRanges[BLOCKDIM_HWSPLIT_GROUP_IDX] = blockDimRanges_.groupRange;
    vector<uint32_t> dimsRecord;
    BlockDimDecisionBackTrackHWsplitMode(allRanges, BLOCKDIM_HWSPLIT_BATCH_IDX, dimsRecord);
}

void ConvBaseDeci::CheckCoreUsedupHWsplitMode()
{
    // woDim is not considered because it is only used in Conv1d scene and hoDim is always 1
    if (blockDimRes_.batchDim * blockDimRes_.hoDim * blockDimRes_.nDim *
        blockDimRes_.doDim * blockDimRes_.groupDim == aicoreNum_) {
        return;
    }

    for (auto newHoDim : blockDimRanges_.hoSpareRange) {
        if (blockDimRes_.batchDim * blockDimRes_.groupDim * newHoDim *
            blockDimRes_.nDim * blockDimRes_.doDim <= aicoreNum_) {
            blockDimRes_.hoDim = std::max(blockDimRes_.hoDim, newHoDim);
        }
    }
}

}
}