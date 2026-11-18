/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file conv_basic_block.cpp
 * \brief
 */

#include "conv_basic_block.h"

namespace conv_tiling {

using optiling::conv_ops_tiling::CONST_VALUE_3;
using optiling::conv_ops_tiling::CONST_VALUE_4;
using optiling::conv_ops_tiling::CONST_VALUE_5;
using optiling::conv_ops_tiling::CONST_VALUE_6;
using optiling::conv_ops_tiling::CONST_VALUE_7;
using optiling::conv_ops_tiling::ConvAscendcFeatureFlag;
using optiling::conv_ops_tiling::ConvCalcCommFactor;
using optiling::conv_ops_tiling::CUBE_MKN_MAP;
using optiling::conv_ops_tiling::IsDAV920Arch;
using optiling::conv_ops_tiling::L0C_SIZE;
using optiling::conv_ops_tiling::MKN_N_IDX;
using optiling::conv_ops_tiling::SWEET_POINT_AI_FP16;
using optiling::conv_ops_tiling::SWEET_POINT_AI_FP16_137;
using optiling::conv_ops_tiling::SWEET_POINT_AI_FP16_89;

void GetInitBasicBlockMN(const ConvAscendcShapesInfo& shapeInfo, const ConvAscendcDescInfo& descInfo, uint64_t curCo,
                         uint64_t& basicBlockM, uint64_t& basicBlockN)
{
    uint64_t howo = shapeInfo.ho * shapeInfo.wo;
    if (curCo <= BASICBLOCK_BOUNDARY_VALUE_64) {
        if (descInfo.fMapDtype == ge::DT_FLOAT16 && descInfo.weightDtype == ge::DT_INT8) {
            basicBlockM = BASICBLOCK_INIT_VALUE_512;
            basicBlockN = BASICBLOCK_INIT_VALUE_64;
        } else {
            basicBlockM = BASICBLOCK_INIT_VALUE_1024;
            basicBlockN = BASICBLOCK_INIT_VALUE_64;
        }
    } else if (curCo > BASICBLOCK_BOUNDARY_VALUE_64 && curCo <= BASICBLOCK_BOUNDARY_VALUE_128) {
        basicBlockM = BASICBLOCK_INIT_VALUE_512;
        basicBlockN = BASICBLOCK_INIT_VALUE_128;
    } else if (howo <= BASICBLOCK_BOUNDARY_VALUE_64) {
        basicBlockM = BASICBLOCK_INIT_VALUE_64;
        basicBlockN = BASICBLOCK_INIT_VALUE_1024;
    } else if (howo > BASICBLOCK_BOUNDARY_VALUE_64 && howo <= BASICBLOCK_BOUNDARY_VALUE_128) {
        basicBlockM = BASICBLOCK_INIT_VALUE_128;
        basicBlockN = BASICBLOCK_INIT_VALUE_512;
    } else {
        basicBlockM = BASICBLOCK_INIT_VALUE_256;
        basicBlockN = BASICBLOCK_INIT_VALUE_256;
    }
}

void Get512InitBasicBlockMN(const ConvAscendcShapesInfo& shapeInfo, const ConvAscendcAttrInfo& attrInfo,
                            uint64_t& basicBlockM, uint64_t& basicBlockN)
{
    uint64_t howo = shapeInfo.ho * shapeInfo.wo;
    uint64_t khkw = shapeInfo.kh * shapeInfo.kw;
    uint64_t shsw = attrInfo.strideH * attrInfo.strideW;
    if (shapeInfo.co <= BASICBLOCK_BOUNDARY_VALUE_128) {
        basicBlockM = BASICBLOCK_INIT_VALUE_1024;
        basicBlockN = BASICBLOCK_INIT_VALUE_128;
    } else if (shapeInfo.co > BASICBLOCK_BOUNDARY_VALUE_128 && shapeInfo.co <= BASICBLOCK_BOUNDARY_VALUE_256) {
        basicBlockM = BASICBLOCK_INIT_VALUE_512;
        basicBlockN = BASICBLOCK_INIT_VALUE_256;
    } else if (howo <= BASICBLOCK_BOUNDARY_VALUE_128) {
        basicBlockM = BASICBLOCK_INIT_VALUE_128;
        basicBlockN = BASICBLOCK_INIT_VALUE_1024;
    } else if (howo > BASICBLOCK_BOUNDARY_VALUE_128 && howo <= BASICBLOCK_BOUNDARY_VALUE_256) {
        basicBlockM = BASICBLOCK_INIT_VALUE_256;
        basicBlockN = BASICBLOCK_INIT_VALUE_512;
    } else if (khkw > shsw) {
        basicBlockM = BASICBLOCK_INIT_VALUE_512;
        basicBlockN = BASICBLOCK_INIT_VALUE_256;
    } else if (khkw < shsw) {
        basicBlockM = BASICBLOCK_INIT_VALUE_256;
        basicBlockN = BASICBLOCK_INIT_VALUE_512;
    } else {
        basicBlockM = BASICBLOCK_INIT_VALUE_256;
        basicBlockN = BASICBLOCK_INIT_VALUE_256;
    }
}

bool ConvBasicBlock::CmpFirstAdjustMnTile(int64_t availableL1Size, int64_t& mTile, int64_t& nTile, uint64_t basicBlockM,
                                          uint64_t basicBlockN)
{
    int64_t fMapDtypeSize = static_cast<int64_t>(dtypeSizeTab.at(params_.descInfo.fMapDtype));
    int64_t maxHiWiL1 = availableL1Size / fMapDtypeSize / params_.convOpsConstParams.k0 / CONST_VALUE_2;
    if (maxHiWiL1 <= 0) {
        return false;
    }
    int64_t maxhiL1 = maxHiWiL1 / static_cast<int64_t>(params_.shapeInfo.wi);
    if (maxhiL1 <= static_cast<int64_t>(CONST_VALUE_2)) {
        return false;
    }
    int64_t hoMax = (maxhiL1 -
                     static_cast<int64_t>(params_.attrInfo.dilationH) *
                         (static_cast<int64_t>(params_.shapeInfo.kh) - 1) -
                     1) /
                        static_cast<int64_t>(params_.attrInfo.strideH) +
                    1 - CONST_VALUE_2;
    if (hoMax <= 0) {
        return false;
    }
    int64_t maxHoWoL1 = hoMax * static_cast<int64_t>(params_.shapeInfo.wo);
    int64_t cmpM = params_.shapeInfo.ho * params_.shapeInfo.wo;
    int64_t weightDtypeSize = dtypeSizeTab.at(params_.descInfo.weightDtype);
    int64_t cmpN = availableL1Size / weightDtypeSize / CONST_VALUE_2 / params_.convOpsConstParams.k0 /
                   params_.shapeInfo.kh / params_.shapeInfo.kw;
    if (static_cast<int64_t>(ConvAlignB(cmpM, params_.convOpsConstParams.m0)) <=
        min(static_cast<int64_t>(basicBlockM), maxHoWoL1)) {
        mTile = ConvAlignB(cmpM, params_.convOpsConstParams.m0);
    } else {
        mTile = min(min(cmpM, maxHoWoL1), static_cast<int64_t>(basicBlockM));
    }
    nTile = min(min(static_cast<int64_t>(params_.shapeInfo.co), cmpN), static_cast<int64_t>(basicBlockN));
    if (params_.flagInfo.convGroupType == ConvGroupType::ORI_GROUP_CONV) {
        nTile = static_cast<uint32_t>(nTile) / params_.attrInfo.groups;
    } else if (params_.flagInfo.convGroupType == ConvGroupType::OPT_GROUP_CONV) {
        nTile = static_cast<uint32_t>(nTile) / params_.attrInfo.groups *
                static_cast<uint32_t>(params_.optGroupInfo.enlarge);
    }

    mTile = mTile / params_.convOpsConstParams.m0 * params_.convOpsConstParams.m0;
    nTile = nTile / params_.convOpsConstParams.n0 * params_.convOpsConstParams.n0;
    if (mTile < static_cast<int64_t>(params_.convOpsConstParams.m0) ||
        nTile < static_cast<int64_t>(params_.convOpsConstParams.n0)) {
        return false;
    }
    return true;
}

void ConvBasicBlock::EnableInnerBatch(int64_t availableL1Size)
{
    if (!strategy_->SupportsInnerBatch()) {
        return;
    }
    if (params_.flagInfo.enableC04Flag || params_.flagInfo.convGroupType != ConvGroupType::NORMAL_CONV ||
        params_.featureFlagInfo == ConvAscendcFeatureFlag::IS_DMA_FLAG) {
        return;
    }
    uint64_t basicBlockM = params_.l0cSize / CONST_VALUE_2 / CONST_VALUE_2 / bbInfo_.nTile /
                           params_.convOpsConstParams.m0 * params_.convOpsConstParams.m0;
    if (params_.shapeInfo.co <= BASICBLOCK_BOUNDARY_VALUE_64 && params_.descInfo.fMapDtype == ge::DT_FLOAT16 &&
        params_.descInfo.weightDtype == ge::DT_INT8) {
        basicBlockM = basicBlockM > BASICBLOCK_INIT_VALUE_512 ? BASICBLOCK_INIT_VALUE_512 : basicBlockM;
    } else {
        basicBlockM = basicBlockM > BASICBLOCK_INIT_VALUE_1024 ? BASICBLOCK_INIT_VALUE_1024 : basicBlockM;
    }
    uint64_t innerBatchTemp = (availableL1Size / dtypeSizeTab.at(params_.descInfo.fMapDtype) / CONST_VALUE_2 /
                               CONST_VALUE_2 / params_.convOpsConstParams.k0) /
                              (params_.shapeInfo.wi * params_.shapeInfo.hi);
    uint32_t innerBatch = max(static_cast<uint64_t>(1),
                              min(static_cast<uint64_t>(params_.shapeInfo.batch), innerBatchTemp));
    uint64_t mTileInnerBatch = min(
        innerBatch * ConvCeilDiv(params_.shapeInfo.ho * params_.shapeInfo.wo, params_.convOpsConstParams.m0) *
            params_.convOpsConstParams.m0,
        basicBlockM);
    innerBatch = mTileInnerBatch /
                 (ConvCeilDiv(params_.shapeInfo.ho * params_.shapeInfo.wo, params_.convOpsConstParams.m0) *
                  params_.convOpsConstParams.m0);
    if (innerBatch <= CONST_VALUE_1) {
        return;
    }
    uint64_t nCut = ConvCeilDiv(params_.shapeInfo.co, bbInfo_.nTile);
    uint64_t fCut = ConvCeilDiv(
        ConvCeilDiv(params_.shapeInfo.batch *
                        ConvCeilDiv(params_.shapeInfo.ho * params_.shapeInfo.wo, params_.convOpsConstParams.m0) *
                        params_.convOpsConstParams.m0,
                    bbInfo_.mTile),
        innerBatch);
    if (fCut * nCut <= params_.aicoreNum && ConvCeilDiv(params_.shapeInfo.batch, params_.aicoreNum) == CONST_VALUE_1) {
        return;
    }
    params_.enableInnerBatch = true;
    params_.innerBatch = innerBatch;
}

bool ConvBasicBlock::SelectMModeAlgorithm()
{
    if (strategy_->ShouldSkipBasicBlock(params_)) {
        return false;
    }

    uint64_t basicBlockM = 0;
    uint64_t basicBlockN = 0;
    uint64_t orgCo = params_.flagInfo.convGroupType != ConvGroupType::NORMAL_CONV ?
                         (params_.flagInfo.convGroupType == ConvGroupType::ORI_GROUP_CONV ?
                              params_.oriGroupInfo.coPerGroup :
                              params_.optGroupInfo.coutOpt) :
                         params_.shapeInfo.co;
    uint64_t initCo = IsDAV920Arch(params_.platformInfo.npuArch) ? orgCo : params_.shapeInfo.co;
    if (params_.l0cSize > L0C_SIZE) {
        Get512InitBasicBlockMN(params_.shapeInfo, params_.attrInfo, basicBlockM, basicBlockN);
    } else {
        GetInitBasicBlockMN(params_.shapeInfo, params_.descInfo, initCo, basicBlockM, basicBlockN);
    }

    if (!strategy_->ShouldUseBasicBlockMN(params_, basicBlockM, basicBlockN)) {
        if (params_.l0cSize <= L0C_SIZE) {
            return false;
        }
        GetInitBasicBlockMN(params_.shapeInfo, params_.descInfo, initCo, basicBlockM, basicBlockN);
        if (!strategy_->ShouldUseBasicBlockMN(params_, basicBlockM, basicBlockN)) {
            return false;
        }
    }

    uint64_t biasSize = 0;
    if (params_.flagInfo.hasBias) {
        biasSize = dtypeSizeTab.at(params_.descInfo.biasDtype) * orgCo;
    }
    uint64_t scaleSize = static_cast<uint64_t>(params_.fixpipeInfo.channelWiseCoeff * FP16_DTYPE_SIZE *
                                               params_.shapeInfo.co);
    int64_t availableL1Size = static_cast<int64_t>(params_.l1Size) - static_cast<int64_t>(biasSize) -
                              static_cast<int64_t>(scaleSize);

    int64_t mTile = 0, nTile = 0;
    if (availableL1Size <= 0 || !CmpFirstAdjustMnTile(availableL1Size, mTile, nTile, basicBlockM, basicBlockN)) {
        return false;
    }

    bbInfo_.mTile = static_cast<uint32_t>(mTile);
    bbInfo_.nTile = static_cast<uint32_t>(nTile);
    OP_LOGD(params_.tilingBase->nodeType.c_str(), "mTile %u, nTile %u", bbInfo_.mTile, bbInfo_.nTile);
    EnableInnerBatch(availableL1Size);
    return true;
}

void ConvBasicBlock::BasicBlock()
{
    BasicBlockGroupDecision();
    BasicBlockL0Init();
    BasicBlockBoundMode();
    BasicBlockFWDimDecision();
    BasicBlockBatchMDimDecision();
}

void ConvBasicBlock::BasicBlockGroupDecision()
{
    if (params_.attrInfo.groups == 1) {
        bbInfo_.groupDim = 1;
        bbInfo_.fwDim = params_.aicoreNum;
        return;
    }
    unordered_set<pair<uint32_t, uint32_t>, BBPairHash> candidates = BasicBlockGroupDimCandidates();
    uint32_t minCost = numeric_limits<uint32_t>::max();
    pair<uint32_t, uint32_t> optDecision;
    uint32_t fwCut = static_cast<uint32_t>(strategy_->GetFwCutForGroup(params_, bbInfo_));
    uint32_t groupCut = params_.flagInfo.convGroupType == ConvGroupType::ORI_GROUP_CONV ? params_.attrInfo.groups :
                                                                                          params_.optGroupInfo.groupOpt;
    for (const auto& candidate : candidates) {
        uint32_t fwDim = candidate.first;
        uint32_t groupDim = candidate.second;
        uint32_t cost = ConvCeilDiv(groupCut, groupDim) * ConvCeilDiv(fwCut, fwDim);
        if (cost < minCost || (cost == minCost && fwDim < optDecision.first)) {
            minCost = cost;
            optDecision = candidate;
        }
    }
    bbInfo_.fwDim = optDecision.first;
    bbInfo_.groupDim = optDecision.second;
}

void ConvBasicBlock::BasicBlockL0Init()
{
    strategy_->CalcL0InitParams(params_, bbInfo_, static_cast<uint64_t>(params_.l1Size));
}

void ConvBasicBlock::BasicBlockBoundMode()
{
    uint64_t inAI = strategy_->CalcInputAI(params_, bbInfo_);
    uint64_t outAI = strategy_->CalcOutputAI(params_);
    uint64_t realAI = min(inAI, outAI);
    uint64_t sweetPointAI = !IsDAV920Arch(params_.platformInfo.npuArch) ? SWEET_POINT_AI_FP16 : SWEET_POINT_AI_FP16_89;
    if (params_.l0cSize > L0C_SIZE) {
        sweetPointAI = SWEET_POINT_AI_FP16_137;
    }
    if (params_.flagInfo.quantFlag) {
        sweetPointAI *= CONST_VALUE_2;
    } else if (params_.descInfo.fMapDtype == ge::DT_FLOAT && params_.attrInfo.hf32Mode == 1) {
        sweetPointAI /= CONST_VALUE_2;
    } else if (params_.descInfo.fMapDtype == ge::DT_FLOAT) {
        sweetPointAI /= CONST_VALUE_4;
    }
    bbInfo_.boundType = realAI > sweetPointAI ? BoundType::CUBE_BOUND : BoundType::MEMORY_BOUND;
}

unordered_set<pair<uint32_t, uint32_t>, BBPairHash> ConvBasicBlock::BasicBlockGroupDimCandidates()
{
    uint32_t cores = params_.aicoreNum;
    uint64_t fwCut = strategy_->GetFwCutForGroup(params_, bbInfo_);
    uint32_t groupCut = params_.flagInfo.convGroupType == ConvGroupType::ORI_GROUP_CONV ? params_.attrInfo.groups :
                                                                                          params_.optGroupInfo.groupOpt;
    uint32_t calCut1 = static_cast<uint32_t>(sqrt(static_cast<double>(cores * groupCut) / static_cast<double>(fwCut)));
    uint32_t calCut2 = static_cast<uint32_t>(
        sqrt(static_cast<double>(static_cast<uint64_t>(cores) * fwCut) / static_cast<double>(groupCut)));
    return GenerateCandidates(cores, fwCut, groupCut, calCut1, calCut2, params_.tilingBase->nodeType);
}

unordered_set<pair<uint32_t, uint32_t>, BBPairHash> ConvBasicBlock::BasicBlockFWDimCandidates()
{
    uint32_t cores = bbInfo_.fwDim;
    uint64_t fCut = strategy_->GetFwCut(params_, bbInfo_);
    if (params_.enableInnerBatch) {
        fCut = ConvCeilDiv(
            ConvCeilDiv(params_.shapeInfo.batch *
                            ConvCeilDiv(params_.shapeInfo.ho * params_.shapeInfo.wo, params_.convOpsConstParams.m0) *
                            params_.convOpsConstParams.m0,
                        bbInfo_.mTile),
            params_.innerBatch);
    }
    uint32_t nCut = bbInfo_.nCut;
    uint32_t calCut1 = static_cast<uint32_t>(sqrt(static_cast<float>(cores * bbInfo_.nKernelHKernelW) /
                                                  static_cast<float>(bbInfo_.mIn * params_.innerBatch)));
    uint32_t calCut2 = static_cast<uint32_t>(
        sqrt(static_cast<double>(static_cast<uint64_t>(cores) * fCut * bbInfo_.mIn * params_.innerBatch) /
             static_cast<double>(nCut) / static_cast<double>(bbInfo_.nKernelHKernelW)));
    auto candidates = GenerateCandidates(cores, fCut, nCut, calCut1, calCut2, params_.tilingBase->nodeType);
    uint64_t curCo = params_.flagInfo.convGroupType != ConvGroupType::NORMAL_CONV ?
                         (params_.flagInfo.convGroupType == ConvGroupType::ORI_GROUP_CONV ?
                              params_.oriGroupInfo.coPerGroup :
                              params_.optGroupInfo.coutOpt) :
                         params_.shapeInfo.co;
    uint32_t n0 = CUBE_MKN_MAP.GetMKN(dtypeMap.at(params_.descInfo.fMapDtype), MKN_N_IDX);
    uint32_t maxFDim = min(cores, static_cast<uint32_t>(curCo)) / nCut;
    for (auto i = maxFDim; i > 0; i--) {
        if (IsDAV920Arch(params_.platformInfo.npuArch) && IsPrime(i)) {
            continue;
        }
        uint32_t nDim = min(cores / i, static_cast<uint32_t>(ConvCeilDiv(curCo, n0)));
        if (IsDAV920Arch(params_.platformInfo.npuArch) &&
            static_cast<double>(i) * static_cast<double>(nDim) < FW_DIM_CANDIDATE_RATIO * static_cast<double>(cores)) {
            continue;
        }
        pair<uint32_t, uint32_t> tmp(i, nDim);
        if (candidates.find(tmp) == candidates.end()) {
            candidates.emplace(tmp);
        }
    }
    candidates.emplace(cores, 1);
    return candidates;
}

void ConvBasicBlock::BasicBlockFWDimDecision()
{
    auto candidates = BasicBlockFWDimCandidates();
    vector<FWDimCandidate> evaluated;
    auto oriInfo = make_tuple(bbInfo_.mTile, bbInfo_.nTile, bbInfo_.mCut, bbInfo_.nCut, bbInfo_.mIn);
    for (const auto& candidate : candidates) {
        EvalFWDimCandidate(candidate, oriInfo, evaluated);
    }
    if (evaluated.empty()) {
        return;
    }
    if (!IsDAV920Arch(params_.platformInfo.npuArch)) {
        int32_t bestIdx = BasicBlockSortFWDimScores(evaluated);
        if (bestIdx < 0 || static_cast<size_t>(bestIdx) >= evaluated.size()) {
            return;
        }
        SetBasicBlockInfo(evaluated[bestIdx]);
        return;
    }
    if (bbInfo_.boundType == BoundType::CUBE_BOUND) {
        int32_t bestIdx = BasicBlockSortFWDimScores(evaluated);
        if (bestIdx < 0 || static_cast<size_t>(bestIdx) >= evaluated.size()) {
            return;
        }
        SetBasicBlockInfo(evaluated[bestIdx]);
        strategy_->ExecuteBatchMDimDecision(params_, bbInfo_);
        return;
    }
    auto fwCandidates = CollectMemoryBoundCandidates(evaluated);
    if (fwCandidates.empty()) {
        int32_t index = BasicBlockSortFWDimScores(evaluated);
        if (index >= 0 && static_cast<size_t>(index) < evaluated.size()) {
            SetBasicBlockInfo(evaluated[index]);
            strategy_->ExecuteBatchMDimDecision(params_, bbInfo_);
        }
        return;
    }
    ExecuteMemoryBoundFWDimDecision(fwCandidates);
}

bool ConvBasicBlock::EvalFWDimCandidate(const pair<uint32_t, uint32_t>& candidate,
                                        const tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint64_t>& oriInfo,
                                        vector<FWDimCandidate>& evaluated)
{
    bbInfo_.fDim = candidate.first;
    bbInfo_.nDim = candidate.second;
    bbInfo_.mTile = get<0>(oriInfo);
    bbInfo_.nTile = get<1>(oriInfo);
    bbInfo_.mCut = get<CONST_VALUE_2>(oriInfo);
    bbInfo_.nCut = get<CONST_VALUE_3>(oriInfo);
    bbInfo_.mIn = get<CONST_VALUE_4>(oriInfo);
    bbInfo_.l1LoadScore = static_cast<float>(-1.0);
    bbInfo_.biasFullLoad = false;
    bbInfo_.fixpFullLoad = false;
    if (!strategy_->ShouldAdjustMnTile(params_, bbInfo_)) {
        return false;
    }
    strategy_->SetCandidateSingleOutputShape(params_, bbInfo_, candidate.second);
    if (!params_.tilingBase->EvalCandidateCoreBinding(bbInfo_)) {
        return false;
    }
    uint64_t basicBlockNum1 = strategy_->CalcBasicBlockNum1(params_, bbInfo_);
    uint64_t basicBlockNum2 = ConvCeilDiv(bbInfo_.nCut, bbInfo_.nDim);
    uint32_t fwDimDiff = abs(static_cast<int32_t>(bbInfo_.fDim - bbInfo_.nDim));
    if (bbInfo_.l1LoadScore < 0) {
        return false;
    }
    FWDimCandidate fwCand;
    fwCand.fwDim = candidate;
    fwCand.strategy = make_tuple(bbInfo_.iterateMNOrder, bbInfo_.kAl1FullLoad, bbInfo_.kBl1FullLoad,
                                 bbInfo_.mAl1FullLoad, bbInfo_.nBl1FullLoad, bbInfo_.biasFullLoad, bbInfo_.fixpFullLoad,
                                 bbInfo_.l1LoadScore);
    fwCand.tileInfo = make_tuple(bbInfo_.mTile, bbInfo_.nTile, bbInfo_.mCut, bbInfo_.nCut, bbInfo_.mIn);
    fwCand.total = basicBlockNum1 * basicBlockNum2;
    fwCand.util = bbInfo_.coreUtilization;
    fwCand.fwDimDiff = fwDimDiff;
    fwCand.l1Score = bbInfo_.l1LoadScore;
    evaluated.push_back(fwCand);
    return true;
}

vector<ConvBasicBlock::FWDimCandidate> ConvBasicBlock::CollectMemoryBoundCandidates(
    const vector<FWDimCandidate>& evaluated) const
{
    vector<FWDimCandidate> fwCandidates;
    for (const auto& cand : evaluated) {
        if (cand.util > MEMORY_BOUND_UTIL_THRESHOLD) {
            fwCandidates.push_back(cand);
        }
    }
    return fwCandidates;
}

void ConvBasicBlock::BasicBlockBatchMDimDecision()
{
    if (!IsDAV920Arch(params_.platformInfo.npuArch) || bbInfo_.boundType == BoundType::CUBE_BOUND) {
        strategy_->ExecuteBatchMDimDecision(params_, bbInfo_);
    }
}

bool ConvBasicBlock::CompareMemoryBoundCandidate(const MemBoundCmpData& cur, const MemBoundCmpData& best)
{
    if (cur.mDataSize != best.mDataSize) {
        return cur.mDataSize < best.mDataSize;
    }
    if (abs(cur.l1Score - best.l1Score) >= BB_EPSILON) {
        return cur.l1Score > best.l1Score;
    }
    if (cur.fwDimDiff != best.fwDimDiff) {
        return cur.fwDimDiff < best.fwDimDiff;
    }
    return cur.total < best.total;
}

void ConvBasicBlock::ExecuteMemoryBoundFWDimDecision(const vector<FWDimCandidate>& fwCandidates)
{
    uint64_t singleM = params_.shapeInfo.ho * params_.shapeInfo.wo;
    uint64_t singleCo = params_.flagInfo.convGroupType != ConvGroupType::NORMAL_CONV ?
                            (params_.flagInfo.convGroupType == ConvGroupType::ORI_GROUP_CONV ?
                                 params_.oriGroupInfo.coPerGroup :
                                 params_.optGroupInfo.coutOpt) :
                            static_cast<uint64_t>(params_.shapeInfo.co);
    uint64_t singleBatch = params_.shapeInfo.batch;
    uint64_t singleDo = params_.shapeInfo.dout;

    int32_t bestFwIdx = -1;
    MemBoundCmpData best;
    best.mDataSize = numeric_limits<uint64_t>::max();
    best.l1Score = -1.0;
    best.fwDimDiff = numeric_limits<uint32_t>::max();
    best.total = numeric_limits<uint64_t>::max();

    for (size_t fwIdx = 0; fwIdx < fwCandidates.size(); fwIdx++) {
        const auto& fwCand = fwCandidates[fwIdx];
        SetBasicBlockInfo(fwCand);
        strategy_->ExecuteBatchMDimDecision(params_, bbInfo_);
        MemBoundCmpData cur;
        cur.mDataSize = ConvAlignB(ConvCeilDiv(singleM, static_cast<uint64_t>(bbInfo_.mDim)),
                                   params_.convOpsConstParams.m0) *
                        ConvAlignB(ConvCeilDiv(singleCo, static_cast<uint64_t>(bbInfo_.nDim)),
                                   params_.convOpsConstParams.n0) *
                        ConvCeilDiv(singleBatch, static_cast<uint64_t>(bbInfo_.batchDim)) *
                        ConvCeilDiv(singleDo, static_cast<uint64_t>(bbInfo_.doDim));
        cur.l1Score = fwCand.l1Score;
        cur.fwDimDiff = fwCand.fwDimDiff;
        cur.total = fwCand.total;
        if (!CompareMemoryBoundCandidate(cur, best)) {
            continue;
        }
        bestFwIdx = static_cast<int32_t>(fwIdx);
        best = cur;
    }

    if (bestFwIdx >= 0) {
        const auto& bestCand = fwCandidates[bestFwIdx];
        SetBasicBlockInfo(bestCand);
        strategy_->ExecuteBatchMDimDecision(params_, bbInfo_);
    }
}

int32_t ConvBasicBlock::BasicBlockSortFWDimScores(vector<FWDimCandidate>& evaluated)
{
    if (evaluated.empty()) {
        return -1;
    }
    if (!IsDAV920Arch(params_.platformInfo.npuArch)) {
        vector<tuple<uint64_t, float, uint32_t, double>> scores;
        scores.reserve(evaluated.size());
        for (const auto& cand : evaluated) {
            scores.emplace_back(cand.total, cand.util, cand.fwDimDiff, cand.l1Score);
        }
        vector<int32_t> range(evaluated.size());
        iota(range.begin(), range.end(), 0);
        if (bbInfo_.boundType == BoundType::CUBE_BOUND) {
            auto comp = [&scores](int32_t i, int32_t j) {
                return get<0>(scores[i]) > get<0>(scores[j]) ||
                       (get<0>(scores[i]) == get<0>(scores[j]) && get<1>(scores[i]) < get<1>(scores[j])) ||
                       (get<0>(scores[i]) == get<0>(scores[j]) &&
                        abs(get<1>(scores[i]) - get<1>(scores[j])) < BB_EPSILON &&
                        get<CONST_VALUE_2>(scores[i]) > get<CONST_VALUE_2>(scores[j]));
            };
            return *max_element(range.begin(), range.end(), comp);
        }
        auto comp = [&scores](int32_t i, int32_t j) {
            return get<1>(scores[i]) < get<1>(scores[j]) ||
                   (abs(get<1>(scores[i]) - get<1>(scores[j])) < BB_EPSILON &&
                    get<CONST_VALUE_3>(scores[i]) < get<CONST_VALUE_3>(scores[j])) ||
                   (abs(get<1>(scores[i]) - get<1>(scores[j])) < BB_EPSILON &&
                    abs(get<CONST_VALUE_3>(scores[i]) - get<CONST_VALUE_3>(scores[j])) < BB_EPSILON &&
                    get<CONST_VALUE_2>(scores[i]) > get<CONST_VALUE_2>(scores[j])) ||
                   (abs(get<1>(scores[i]) - get<1>(scores[j])) < BB_EPSILON &&
                    abs(get<CONST_VALUE_3>(scores[i]) - get<CONST_VALUE_3>(scores[j])) < BB_EPSILON &&
                    get<CONST_VALUE_2>(scores[i]) == get<CONST_VALUE_2>(scores[j]) &&
                    get<0>(scores[i]) > get<0>(scores[j]));
        };
        return *max_element(range.begin(), range.end(), comp);
    }
    if (bbInfo_.boundType != BoundType::CUBE_BOUND) {
        return SelectMemoryBoundFallback(evaluated);
    }
    return SelectCubeBoundBest(evaluated);
}

int32_t ConvBasicBlock::SelectMemoryBoundFallback(const vector<FWDimCandidate>& evaluated) const
{
    static constexpr double SCORE_TOLERANCE = 0.1;
    vector<int32_t> range(evaluated.size());
    iota(range.begin(), range.end(), 0);
    auto comp = [&evaluated](int32_t i, int32_t j) {
        return evaluated[i].util < evaluated[j].util - SCORE_TOLERANCE ||
               (abs(evaluated[i].util - evaluated[j].util) < SCORE_TOLERANCE &&
                evaluated[i].l1Score < evaluated[j].l1Score) ||
               (abs(evaluated[i].util - evaluated[j].util) < SCORE_TOLERANCE &&
                abs(evaluated[i].l1Score - evaluated[j].l1Score) < BB_EPSILON &&
                evaluated[i].fwDimDiff > evaluated[j].fwDimDiff) ||
               (abs(evaluated[i].util - evaluated[j].util) < SCORE_TOLERANCE &&
                abs(evaluated[i].l1Score - evaluated[j].l1Score) < BB_EPSILON &&
                evaluated[i].fwDimDiff == evaluated[j].fwDimDiff && evaluated[i].total > evaluated[j].total);
    };
    return *max_element(range.begin(), range.end(), comp);
}

int32_t ConvBasicBlock::SelectCubeBoundBest(vector<FWDimCandidate>& evaluated)
{
    size_t topK = min(CUBE_BOUND_TOPK, evaluated.size());
    partial_sort(evaluated.begin(), evaluated.begin() + topK, evaluated.end(),
                 [](const FWDimCandidate& a, const FWDimCandidate& b) { return a.total < b.total; });
    vector<uint64_t> mDataSizes = CalcMDataSizes(topK, evaluated);
    size_t best = PickBestByMDataSize(evaluated, mDataSizes);
    return static_cast<int32_t>(best);
}

vector<uint64_t> ConvBasicBlock::CalcMDataSizes(size_t topK, vector<FWDimCandidate>& evaluated)
{
    uint64_t singleM = params_.shapeInfo.ho * params_.shapeInfo.wo;
    uint64_t singleCo = params_.flagInfo.convGroupType != ConvGroupType::NORMAL_CONV ?
                            (params_.flagInfo.convGroupType == ConvGroupType::ORI_GROUP_CONV ?
                                 params_.oriGroupInfo.coPerGroup :
                                 params_.optGroupInfo.coutOpt) :
                            static_cast<uint64_t>(params_.shapeInfo.co);
    uint64_t singleBatch = params_.shapeInfo.batch;
    uint64_t singleDo = params_.shapeInfo.dout;
    vector<uint64_t> mDataSizes(topK, numeric_limits<uint64_t>::max());
    for (size_t i = 0; i < topK; i++) {
        SetBasicBlockInfo(evaluated[i]);
        strategy_->ExecuteBatchMDimDecision(params_, bbInfo_);
        mDataSizes[i] = ConvAlignB(ConvCeilDiv(singleM, static_cast<uint64_t>(bbInfo_.mDim)),
                                   params_.convOpsConstParams.m0) *
                        ConvAlignB(ConvCeilDiv(singleCo, static_cast<uint64_t>(bbInfo_.nDim)),
                                   params_.convOpsConstParams.n0) *
                        ConvCeilDiv(singleBatch, static_cast<uint64_t>(bbInfo_.batchDim)) *
                        ConvCeilDiv(singleDo, static_cast<uint64_t>(bbInfo_.doDim));
    }
    return mDataSizes;
}

size_t ConvBasicBlock::PickBestByMDataSize(const vector<FWDimCandidate>& evaluated,
                                           const vector<uint64_t>& mDataSizes) const
{
    static constexpr double SCORE_TOLERANCE = 0.1;
    size_t topK = mDataSizes.size();
    uint64_t minMDataSize = *min_element(mDataSizes.begin(), mDataSizes.end());
    size_t best = topK;
    for (size_t i = 0; i < topK; i++) {
        if (mDataSizes[i] != minMDataSize) {
            continue;
        }
        if (best == topK) {
            best = i;
            continue;
        }
        if (evaluated[i].total < evaluated[best].total) {
            best = i;
            continue;
        }
        if (evaluated[i].total > evaluated[best].total) {
            continue;
        }
        if (evaluated[i].util < evaluated[best].util - SCORE_TOLERANCE) {
            continue;
        }
        if (evaluated[best].util < evaluated[i].util - SCORE_TOLERANCE) {
            best = i;
            continue;
        }
        if (evaluated[i].fwDimDiff < evaluated[best].fwDimDiff) {
            best = i;
        }
    }
    return best;
}

void ConvBasicBlock::SetBasicBlockInfo(const FWDimCandidate& candidate)
{
    bbInfo_.fDim = candidate.fwDim.first;
    bbInfo_.nDim = candidate.fwDim.second;
    bbInfo_.iterateMNOrder = get<0>(candidate.strategy);
    bbInfo_.kAl1FullLoad = get<1>(candidate.strategy);
    bbInfo_.kBl1FullLoad = get<CONST_VALUE_2>(candidate.strategy);
    bbInfo_.mAl1FullLoad = get<CONST_VALUE_3>(candidate.strategy);
    bbInfo_.nBl1FullLoad = get<CONST_VALUE_4>(candidate.strategy);
    bbInfo_.biasFullLoad = get<CONST_VALUE_5>(candidate.strategy);
    bbInfo_.fixpFullLoad = get<CONST_VALUE_6>(candidate.strategy);
    bbInfo_.l1LoadScore = get<CONST_VALUE_7>(candidate.strategy);
    bbInfo_.mTile = get<0>(candidate.tileInfo);
    bbInfo_.nTile = get<1>(candidate.tileInfo);
    bbInfo_.mCut = get<CONST_VALUE_2>(candidate.tileInfo);
    bbInfo_.nCut = get<CONST_VALUE_3>(candidate.tileInfo);
    bbInfo_.mIn = get<CONST_VALUE_4>(candidate.tileInfo);
}

unordered_set<pair<uint32_t, uint32_t>, BBPairHash> ConvBasicBlock::GenerateCandidates(uint32_t cores, uint64_t cut1,
                                                                                       uint32_t cut2, uint32_t calCut1,
                                                                                       uint32_t calCut2,
                                                                                       const string& nodeType)
{
    unordered_set<pair<uint32_t, uint32_t>, BBPairHash> candidates;
    GenerateSingleCandidateCase(cores, static_cast<float>(calCut1), cut1, cut2, candidates, nodeType);
    GenerateSingleCandidateCase(cores, static_cast<float>(calCut2), cut1, cut2, candidates, nodeType);
    uint32_t gcd1 = static_cast<uint32_t>(Gcd(static_cast<uint64_t>(cores), cut1));
    uint32_t gcd2 = Gcd(cores, cut2);
    if (gcd1 == 0 || gcd2 == 0 || min(cut2, cores) == 0 || min(cut1, static_cast<uint64_t>(cores)) == 0) {
        OP_LOGW(nodeType.c_str(), "AscendC: candidates is invalid, which is 0.");
        candidates.emplace(0, 0);
        return candidates;
    }
    candidates.emplace(gcd1, min(cut2, cores / gcd1));
    candidates.emplace(static_cast<uint32_t>(min(cut1, static_cast<uint64_t>(cores / gcd2))), gcd2);
    if (gcd1 == 1) {
        candidates.emplace(static_cast<uint32_t>(min(cut1, static_cast<uint64_t>(cores))),
                           cores / static_cast<uint32_t>(min(cut1, static_cast<uint64_t>(cores))));
    }
    if (gcd2 == 1) {
        candidates.emplace(cores / min(cut2, cores), min(cut2, cores));
    }
    if (gcd1 == cores || gcd2 == cores) {
        float calCut3 = static_cast<float>(sqrt(static_cast<double>(cores)));
        GenerateSingleCandidateCase(cores, calCut3, cut1, cut2, candidates, nodeType);
    }
    return candidates;
}

void ConvBasicBlock::GenerateSingleCandidateCase(uint32_t cores, float candidateCut, uint64_t cut1, uint32_t cut2,
                                                 unordered_set<pair<uint32_t, uint32_t>, BBPairHash>& candidates,
                                                 const string& nodeType)
{
    vector<uint32_t> factors;
    ConvCalcCommFactor(cores, cores, factors);
    auto upper = upper_bound(factors.begin(), factors.end(), candidateCut);
    auto lower = lower_bound(factors.begin(), factors.end(), candidateCut);
    if (upper != factors.end()) {
        uint32_t dim = static_cast<uint32_t>(min(static_cast<uint64_t>(*upper), cut1));
        GenerateSingleCandidateCaseCommon(cores, dim, cut2, candidates, nodeType);
    }
    if (lower != factors.end()) {
        uint32_t dim = static_cast<uint32_t>(min(static_cast<uint64_t>(*lower), cut1));
        GenerateSingleCandidateCaseCommon(cores, dim, cut2, candidates, nodeType);
    }
}

void ConvBasicBlock::GenerateSingleCandidateCaseCommon(uint32_t cores, uint32_t dim, uint32_t cut2,
                                                       unordered_set<pair<uint32_t, uint32_t>, BBPairHash>& candidates,
                                                       const string& nodeType)
{
    if (dim == 0) {
        OP_LOGD(nodeType.c_str(), "AscendC: numBlocks is equal to 0.");
        return;
    }
    if ((cores / dim) > cut2) {
        return;
    }
    pair<uint32_t, uint32_t> tmp(dim, min(cores / dim, cut2));
    if (candidates.find(tmp) == candidates.end()) {
        candidates.emplace(tmp);
    }
}

bool ConvBasicBlock::IsPrime(uint32_t n)
{
    if (n < CONST_VALUE_2) {
        return false;
    }
    if (n < CONST_VALUE_4) {
        return true;
    }
    if (n % CONST_VALUE_2 == 0) {
        return false;
    }
    uint32_t sqrtN = static_cast<uint32_t>(sqrt(static_cast<double>(n)));
    for (uint32_t i = CONST_VALUE_3; i <= sqrtN; i += CONST_VALUE_2) {
        if (n % i == 0) {
            return false;
        }
    }
    return true;
}

} // namespace conv_tiling
