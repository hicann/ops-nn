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
 * \file conv2d_v2_base_tiling.cpp
 * \brief
 */

#include <numeric>
#include <algorithm>
#include <limits>
#include <cmath>

#include "conv2d_v2_base_tiling.h"

namespace optiling {
namespace conv_ops_tiling {
constexpr float epsilon = 1e-6f;

void Conv2dBaseTiling::BasicBlockL0Init()
{
    uint64_t biasSize = 0;
    uint64_t orgCo = flagInfo_.convGroupType != ConvGroupType::NORMAL_CONV ?
        (flagInfo_.convGroupType == ConvGroupType::ORI_GROUP_CONV ?
        oriGroupInfo_.coPerGroup : optGroupInfo_.coutOpt) : shapeInfo_.co;

    if (flagInfo_.hasBias) {
        biasSize = dtypeSizeTab.at(descInfo_.biasDtype) * orgCo;
    }
    uint64_t scaleSize = static_cast<uint64_t>(fixpipeInfo_.channelWiseCoeff * FP16_DTYPE_SIZE * orgCo);
    uint64_t availableL1Size = opInfo_->l1Size - biasSize - scaleSize;

    uint64_t hoAL1 = ConvCeilDiv(conv2dBasicBlockInfo_.mTile, shapeInfo_.wo) + CONST_VALUE_2;
    uint64_t hiAL1 = ConvInferHiL1(hoAL1, shapeInfo_.hi, shapeInfo_.kh,
                attrInfo_.dilationH, attrInfo_.strideH);

    conv2dBasicBlockInfo_.mIn = hiAL1 * shapeInfo_.wi;
    conv2dBasicBlockInfo_.nKernelHKernelW = conv2dBasicBlockInfo_.nTile * shapeInfo_.kh * shapeInfo_.kw;

    // CONST_VALUE_2 means divide L1 space equally for weight and fmap
    conv2dBasicBlockInfo_.cinL1 = availableL1Size / dtypeSizeTab.at(descInfo_.fMapDtype) / CONST_VALUE_2 /
                (conv2dBasicBlockInfo_.mIn * conv2dApiTiling_.innerBatch + shapeInfo_.kh * shapeInfo_.kw *
                conv2dBasicBlockInfo_.nTile) / convOpsConstParams_.k0 * convOpsConstParams_.k0;
    conv2dBasicBlockInfo_.cinL1 = conv2dBasicBlockInfo_.cinL1 < convOpsConstParams_.k0 ?
        convOpsConstParams_.k0 : conv2dBasicBlockInfo_.cinL1;
    conv2dBasicBlockInfo_.mCut = ConvCeilDiv(shapeInfo_.wo * shapeInfo_.ho,
                conv2dBasicBlockInfo_.mTile);
    conv2dBasicBlockInfo_.nCut = ConvCeilDiv(orgCo, conv2dBasicBlockInfo_.nTile);

    conv2dBasicBlockInfo_.batch = shapeInfo_.batch;
    conv2dBasicBlockInfo_.aicoreNum = opInfo_->aicoreNum;
}

void Conv2dBaseTiling::BasicBlock()
{
    BasicBlockGroupDecision();
    BasicBlockL0Init();
    BasicBlockBoundMode();
    BasicBlockFWDimDecision();
    BasicBlockBatchMDimDecision();
    blockDimRes.batchDim = conv2dBasicBlockInfo_.batchDim;
    blockDimRes.mDim = conv2dBasicBlockInfo_.mDim;
    blockDimRes.nDim = conv2dBasicBlockInfo_.nDim;
    blockDimRes.groupDim = conv2dBasicBlockInfo_.groupDim;
}

void Conv2dBaseTiling::BasicBlockGroupDecision()
{
    if (attrInfo_.groups == 1) {
        conv2dBasicBlockInfo_.groupDim = static_cast<uint32_t>(1);
        conv2dBasicBlockInfo_.fwDim = opInfo_->aicoreNum;
        return;
    }

    unordered_set<pair<uint32_t, uint32_t>, pair_hash> candidates = BasicBlockGroupDimCandidates();
    uint32_t minCost = numeric_limits<uint32_t>::max();
    pair<uint32_t, uint32_t> optDecision;

    uint32_t fwCut = shapeInfo_.batch * ConvCeilDiv(shapeInfo_.ho * shapeInfo_.wo, conv2dBasicBlockInfo_.mTile)
                        * ConvCeilDiv(shapeInfo_.co, conv2dBasicBlockInfo_.nTile);
    uint32_t groupCut = flagInfo_.convGroupType == ConvGroupType::ORI_GROUP_CONV ?
                        attrInfo_.groups : optGroupInfo_.groupOpt;

    for (const auto &candidate: candidates) {
        uint32_t fwDim = candidate.first;
        uint32_t groupDim = candidate.second;
        uint32_t cost = ConvCeilDiv(groupCut, groupDim) * ConvCeilDiv(fwCut, fwDim);
        if (cost < minCost || (cost == minCost && fwDim < optDecision.first)) {
            minCost = cost;
            optDecision = candidate;
        }
    }

    conv2dBasicBlockInfo_.fwDim = optDecision.first;
    conv2dBasicBlockInfo_.groupDim = optDecision.second;
}

void Conv2dBaseTiling::BasicBlockBoundMode()
{
    uint64_t inAI = CONST_VALUE_2 * shapeInfo_.batch * shapeInfo_.wo * shapeInfo_.ho * shapeInfo_.co * shapeInfo_.kh *
            shapeInfo_.kw / (shapeInfo_.batch * shapeInfo_.wi * shapeInfo_.hi * conv2dBasicBlockInfo_.nCut +
            shapeInfo_.kh * shapeInfo_.kw * shapeInfo_.co * conv2dBasicBlockInfo_.mCut) /
            dtypeSizeTab.at(descInfo_.fMapDtype);
    uint64_t outAI = CONST_VALUE_2 * shapeInfo_.ci * shapeInfo_.kh * shapeInfo_.kw / dtypeSizeTab.at(descInfo_.outDtype);
    uint64_t realAI = min(inAI, outAI);

    uint64_t sweetPointAI = SWEET_POINT_AI_FP16;
    if (flagInfo_.quantFlag) {
        sweetPointAI *= CONST_VALUE_2;
    } else if (descInfo_.fMapDtype == ge::DT_FLOAT && (attrInfo_.hf32Mode == 1)) {
        sweetPointAI /= CONST_VALUE_2;
    } else if (descInfo_.fMapDtype == ge::DT_FLOAT) {
        sweetPointAI /= CONST_VALUE_4;
    }

    conv2dBasicBlockInfo_.boundType = realAI > sweetPointAI ? BoundType::CUBE_BOUND : BoundType::MEMORY_BOUND;
}

unordered_set<pair<uint32_t, uint32_t>, pair_hash> Conv2dBaseTiling::BasicBlockFWDimCandidates()
{
    const uint32_t cores = conv2dBasicBlockInfo_.fwDim;
    uint64_t fCut = shapeInfo_.batch * conv2dBasicBlockInfo_.mCut;
    if (conv2dApiTiling_.enableInnerBatch) {
        fCut = ConvCeilDiv(ConvCeilDiv(shapeInfo_.batch *  ConvCeilDiv(shapeInfo_.ho * shapeInfo_.wo, convOpsConstParams_.m0) *
            convOpsConstParams_.m0, conv2dBasicBlockInfo_.mTile), conv2dApiTiling_.innerBatch);
    }
    const uint32_t nCut = conv2dBasicBlockInfo_.nCut;
    uint32_t n0 = CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.fMapDtype), MKN_N_IDX);

    uint64_t curCo = flagInfo_.convGroupType != ConvGroupType::NORMAL_CONV ?
        (flagInfo_.convGroupType == ConvGroupType::ORI_GROUP_CONV ?
         oriGroupInfo_.coPerGroup : optGroupInfo_.coutOpt) : shapeInfo_.co;
    const uint32_t calCut1 = static_cast<uint32_t>(sqrt(static_cast<float>(cores * conv2dBasicBlockInfo_.nKernelHKernelW) /
        static_cast<float>(conv2dBasicBlockInfo_.mIn * conv2dApiTiling_.innerBatch)));
    const uint32_t calCut2 = static_cast<uint32_t>(sqrt(static_cast<double>(static_cast<uint64_t>(cores) * fCut *
        conv2dBasicBlockInfo_.mIn * conv2dApiTiling_.innerBatch) / static_cast<double>(conv2dBasicBlockInfo_.nCut) /
        static_cast<double>(conv2dBasicBlockInfo_.nKernelHKernelW)));
    unordered_set<pair<uint32_t, uint32_t>, pair_hash> candidates =
        GenerateCandidates(cores, fCut, nCut, calCut1, calCut2);

    uint32_t maxFDim = min(static_cast<uint32_t>(cores), static_cast<uint32_t>(curCo)) / nCut;
    for (auto i = maxFDim; i > 0; i--) {
        pair<uint32_t, uint32_t> tmpCandidate(i, min((cores / i), static_cast<uint32_t>(ConvCeilDiv(curCo, n0))));
        if (find(candidates.begin(), candidates.end(), tmpCandidate) == candidates.end()) {
            candidates.emplace(tmpCandidate);
        }
    }

    candidates.emplace(cores, static_cast<uint32_t>(1));
    return candidates;
}

void Conv2dBaseTiling::BasicBlockFWDimDecision()
{
    unordered_set<pair<uint32_t, uint32_t>, pair_hash> candidates = BasicBlockFWDimCandidates();

    // score consists of: 1. number of basic blocks 2. core utilization 3. diff between fdim and ndim 4. l1 load score
    vector<tuple<uint64_t, float, uint32_t, double>> scores;
    vector<tuple<IterateMNOrder, bool, bool, bool, bool, bool, bool, float>> strategies;
    vector<tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint64_t>> tileInfo;
    vector<pair<uint32_t, uint32_t>> ordered_candidates;

    uint64_t n0 = static_cast<uint64_t>(CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.fMapDtype), MKN_N_IDX));

    tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint64_t> oriInfo = {
        conv2dBasicBlockInfo_.mTile, conv2dBasicBlockInfo_.nTile,
        conv2dBasicBlockInfo_.mCut, conv2dBasicBlockInfo_.nCut, conv2dBasicBlockInfo_.mIn};
    for (const auto &candidate: candidates) {
        conv2dBasicBlockInfo_.fDim = candidate.first;
        conv2dBasicBlockInfo_.nDim = candidate.second;
        conv2dBasicBlockInfo_.mTile = get<0>(oriInfo);
        conv2dBasicBlockInfo_.nTile = get<1>(oriInfo);
        conv2dBasicBlockInfo_.mCut = get<CONST_VALUE_2>(oriInfo);
        conv2dBasicBlockInfo_.nCut = get<CONST_VALUE_3>(oriInfo);
        conv2dBasicBlockInfo_.mIn = get<CONST_VALUE_4>(oriInfo);
        conv2dBasicBlockInfo_.l1LoadScore = static_cast<float>(-1.0);
        conv2dBasicBlockInfo_.biasFullLoad = false;
        conv2dBasicBlockInfo_.fixpFullLoad = false;
        if (conv2dBasicBlockInfo_.mCut * shapeInfo_.batch >= opInfo_->aicoreNum && conv2dBasicBlockInfo_.nTile <=
            COUT_LIMIT_128 && attrInfo_.groups == CONST_VALUE_1 && shapeInfo_.kh == CONST_VALUE_1 &&
            shapeInfo_.kw == CONST_VALUE_1 && conv2dBasicBlockInfo_.nDim > CONST_VALUE_1) {
            continue;
        }
        uint64_t curCo = flagInfo_.convGroupType != ConvGroupType::NORMAL_CONV ?
            (flagInfo_.convGroupType == ConvGroupType::ORI_GROUP_CONV ?
            oriGroupInfo_.coPerGroup : optGroupInfo_.coutOpt) : shapeInfo_.co;
        uint64_t singleCoreCo = ConvAlignB(ConvCeilDiv(ConvAlignB(curCo, n0), candidate.second), n0);
        int64_t singleCoreBatch = ConvCeilDiv(shapeInfo_.batch, conv2dBasicBlockInfo_.batchDim);
        conv2dApiTiling_.SetSingleOutputShape(static_cast<int64_t>(singleCoreCo), 0, singleCoreBatch);
        if (!conv2dApiTiling_.GetCoreBindingDecisionFactor(conv2dBasicBlockInfo_)) {
            continue;
        }
        uint64_t basicBlockNum1 = !conv2dApiTiling_.enableInnerBatch ? ConvCeilDiv(shapeInfo_.batch *
            conv2dBasicBlockInfo_.mCut, conv2dBasicBlockInfo_.fDim) :
            ConvCeilDiv(ConvCeilDiv(ConvCeilDiv(shapeInfo_.batch * ConvCeilDiv(shapeInfo_.ho *
            shapeInfo_.wo, convOpsConstParams_.m0) * convOpsConstParams_.m0, conv2dBasicBlockInfo_.mTile),
            conv2dApiTiling_.innerBatch), conv2dBasicBlockInfo_.fDim);
        uint64_t basicBlockNum2 = ConvCeilDiv(conv2dBasicBlockInfo_.nCut, conv2dBasicBlockInfo_.nDim);

        uint32_t fwDimDiff = abs(static_cast<int32_t>(conv2dBasicBlockInfo_.fDim - conv2dBasicBlockInfo_.nDim));
        if (conv2dBasicBlockInfo_.l1LoadScore < 0) {
            continue;
        }
        ordered_candidates.emplace_back(candidate.first, candidate.second);
        scores.emplace_back(basicBlockNum1 * basicBlockNum2, conv2dBasicBlockInfo_.coreUtilization,
                            fwDimDiff, conv2dBasicBlockInfo_.l1LoadScore);
        strategies.emplace_back(conv2dBasicBlockInfo_.iterateMNOrder, conv2dBasicBlockInfo_.kAl1FullLoad,
                                conv2dBasicBlockInfo_.kBl1FullLoad, conv2dBasicBlockInfo_.mAl1FullLoad,
                                conv2dBasicBlockInfo_.nBl1FullLoad, conv2dBasicBlockInfo_.biasFullLoad,
                                conv2dBasicBlockInfo_.fixpFullLoad, conv2dBasicBlockInfo_.l1LoadScore);
        tileInfo.emplace_back(conv2dBasicBlockInfo_.mTile, conv2dBasicBlockInfo_.nTile, conv2dBasicBlockInfo_.mCut,
                            conv2dBasicBlockInfo_.nCut, conv2dBasicBlockInfo_.mIn);
    }
    int32_t index = BasicBlockSortFWDimScores(scores);
    SetConv2dBasicBlockInfo(ordered_candidates, strategies, tileInfo, index);
}

void Conv2dBaseTiling::BasicBlockBatchMDimDecision()
{
    const unordered_set<pair<uint32_t, uint32_t>, pair_hash> candidates = BasicBlockBatchMDimCandidates();
    uint32_t minCost = numeric_limits<uint32_t>::max();
    pair<uint32_t, uint32_t> optDecision;

    const uint32_t mCut = conv2dBasicBlockInfo_.mCut;
    const uint32_t batchCut = ConvCeilDiv(shapeInfo_.batch, conv2dApiTiling_.innerBatch);

    for (const auto &candidate: candidates) {
        uint32_t mDim = candidate.first;
        uint32_t batchDim = candidate.second;
        uint32_t cost1 = ConvCeilDiv(batchCut, batchDim) * ConvCeilDiv(mCut, mDim);
        float cost2 = 1.0f / static_cast<float>(batchDim);
        float cost = cost1 + cost2;
        if (cost < minCost) {
            minCost = static_cast<uint32_t>(cost);
            optDecision = candidate;
        }
    }

    conv2dBasicBlockInfo_.mDim = optDecision.first;
    conv2dBasicBlockInfo_.batchDim = optDecision.second;
}

void Conv2dBaseTiling::SetConv2dBasicBlockInfo(const vector<pair<uint32_t, uint32_t>> &ordered_candidates,
    const vector<tuple<IterateMNOrder, bool, bool, bool, bool, bool, bool, float>> &strategies,
    const vector<tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint64_t>> &tileInfo, const int32_t &index)
{
    conv2dBasicBlockInfo_.fDim = ordered_candidates[index].first;
    conv2dBasicBlockInfo_.nDim = ordered_candidates[index].second;
    conv2dBasicBlockInfo_.iterateMNOrder = get<0>(strategies[index]);
    conv2dBasicBlockInfo_.kAl1FullLoad = get<1>(strategies[index]);
    conv2dBasicBlockInfo_.kBl1FullLoad = get<CONST_VALUE_2>(strategies[index]);
    conv2dBasicBlockInfo_.mAl1FullLoad = get<CONST_VALUE_3>(strategies[index]);
    conv2dBasicBlockInfo_.nBl1FullLoad = get<CONST_VALUE_4>(strategies[index]);
    conv2dBasicBlockInfo_.biasFullLoad = get<CONST_VALUE_5>(strategies[index]);
    conv2dBasicBlockInfo_.fixpFullLoad = get<CONST_VALUE_6>(strategies[index]);
    conv2dBasicBlockInfo_.l1LoadScore = get<CONST_VALUE_7>(strategies[index]);

    conv2dBasicBlockInfo_.mTile = get<0>(tileInfo[index]);
    conv2dBasicBlockInfo_.nTile = get<1>(tileInfo[index]);
    conv2dBasicBlockInfo_.mCut = get<CONST_VALUE_2>(tileInfo[index]);
    conv2dBasicBlockInfo_.nCut = get<CONST_VALUE_3>(tileInfo[index]);
    conv2dBasicBlockInfo_.mIn = get<CONST_VALUE_4>(tileInfo[index]);
}

void Conv2dBaseTiling::GenerateSingleCandidateCase(const uint32_t cores, const float candidateCut, const uint64_t cut1,
    const uint32_t cut2, unordered_set<pair<uint32_t, uint32_t>, pair_hash> &candidates) const
{
    vector<uint32_t> factors;
    ConvCalcCommFactor(cores, cores, factors);
    auto upper = upper_bound(factors.begin(), factors.end(), candidateCut);
    auto lower = lower_bound(factors.begin(), factors.end(), candidateCut);

    if (upper != factors.end()) {
        uint32_t dim = static_cast<uint32_t>(min(static_cast<uint64_t>(*upper), cut1));
        GenerateSingleCandidateCaseCommon(cores, dim, cut2, candidates);
    }
    if (lower != factors.end()) {
        uint32_t dim = static_cast<uint32_t>(min(static_cast<uint64_t>(*lower), cut1));
        GenerateSingleCandidateCaseCommon(cores, dim, cut2, candidates);
    }
}

void Conv2dBaseTiling::GenerateSingleCandidateCaseCommon(const uint32_t cores, const uint32_t dim,
    const uint32_t cut2, unordered_set<pair<uint32_t, uint32_t>, pair_hash> &candidates) const
{
    if (dim == 0) {
        OP_LOGD(context_->GetNodeName(), "%s AscendC: blockDim is equal to 0", context_->GetNodeType());
        return;
    }

    if ((cores / dim) > cut2) {
        return; // when (cores / dim) < cut2 may trigger changes in loading strategies.
    }

    pair<uint32_t, uint32_t> tmpCandidate(dim, min(cores / dim, cut2));
    if (find(candidates.begin(), candidates.end(), tmpCandidate) == candidates.end()) {
        candidates.emplace(tmpCandidate);
    }
}

unordered_set<pair<uint32_t, uint32_t>, pair_hash> Conv2dBaseTiling::GenerateCandidates(const uint32_t cores,
    const uint64_t cut1, const uint32_t cut2, const uint32_t calCut1, const uint32_t calCut2) const
{
    unordered_set<pair<uint32_t, uint32_t>, pair_hash> candidates;

    GenerateSingleCandidateCase(cores, calCut1, cut1, cut2, candidates);
    GenerateSingleCandidateCase(cores, calCut2, cut1, cut2, candidates);

    const uint32_t gcd1 = static_cast<uint32_t>(Gcd(static_cast<uint64_t>(cores), cut1));
    const uint32_t gcd2 = Gcd(cores, cut2);
    if (gcd1 == 0 || gcd2 == 0 || min(cut2, cores) == 0 || min(cut1, static_cast<uint64_t>(cores)) == 0) {
        OP_LOGW(context_->GetNodeName(), "%s AscendC: candidates is invalid, which is 0", context_->GetNodeType());
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
        float calCut3 = static_cast<float>(sqrt(cores));
        GenerateSingleCandidateCase(cores, calCut3, cut1, cut2, candidates);
    }

    return candidates;
}

int32_t Conv2dBaseTiling::BasicBlockSortFWDimScores(vector<tuple<uint64_t, float, uint32_t, double>>& scores)
{
    vector<int32_t> range(scores.size());
    iota(range.begin(), range.end(), 0);

    int32_t index = -1;
    if (conv2dBasicBlockInfo_.boundType == BoundType::CUBE_BOUND) {
        auto comp = [&scores](int32_t i, int32_t j)
            {return get<0>(scores[i]) > get<0>(scores[j]) ||
                    (get<0>(scores[i]) == get<0>(scores[j]) && get<1>(scores[i]) < get<1>(scores[j])) ||
                    (get<0>(scores[i]) == get<0>(scores[j]) &&
                    std::abs(get<1>(scores[i]) - get<1>(scores[j])) < epsilon &&
                    get<CONST_VALUE_2>(scores[i]) > get<CONST_VALUE_2>(scores[j]));};
        index = *max_element(range.begin(), range.end(), comp);
    } else {
        auto comp = [&scores](int32_t i, int32_t j)
            {return get<1>(scores[i]) < get<1>(scores[j]) ||
                    (std::abs(get<1>(scores[i]) - get<1>(scores[j])) < epsilon &&
                    get<CONST_VALUE_3>(scores[i]) < get<CONST_VALUE_3>(scores[j])) ||
                    (std::abs(get<1>(scores[i]) - get<1>(scores[j])) < epsilon &&
                    std::abs(get<CONST_VALUE_3>(scores[i]) - get<CONST_VALUE_3>(scores[j])) < epsilon &&
                    get<CONST_VALUE_2>(scores[i]) > get<CONST_VALUE_2>(scores[j])) ||
                    (std::abs(get<1>(scores[i]) - get<1>(scores[j])) < epsilon &&
                    std::abs(get<CONST_VALUE_3>(scores[i]) - get<CONST_VALUE_3>(scores[j])) < epsilon &&
                    get<CONST_VALUE_2>(scores[i]) == get<CONST_VALUE_2>(scores[j]) &&
                    get<0>(scores[i]) > get<0>(scores[j]));};
        index = *max_element(range.begin(), range.end(), comp);
    }
    return index;
}

unordered_set<pair<uint32_t, uint32_t>, pair_hash> Conv2dBaseTiling::BasicBlockBatchMDimCandidates()
{
    const uint32_t cores = conv2dBasicBlockInfo_.fDim;
    const uint64_t mCut = static_cast<uint64_t>(conv2dBasicBlockInfo_.mCut);
    const uint32_t batchCut =  ConvCeilDiv(shapeInfo_.batch, conv2dApiTiling_.innerBatch);

    const uint32_t calCut1 = static_cast<uint32_t>(sqrt(static_cast<double>(cores * batchCut) /
                                                        static_cast<double>(mCut)));
    const uint32_t calCut2 = static_cast<uint32_t>(sqrt(static_cast<double>(static_cast<uint64_t>(cores) * mCut) /
                                                        static_cast<double>(batchCut)));

    return GenerateCandidates(cores, mCut, batchCut, calCut1, calCut2);
}

unordered_set<pair<uint32_t, uint32_t>, pair_hash> Conv2dBaseTiling::BasicBlockGroupDimCandidates()
{
    const uint32_t cores = opInfo_->aicoreNum;
    const uint64_t fwCut = shapeInfo_.batch * ConvCeilDiv(shapeInfo_.ho * shapeInfo_.wo, conv2dBasicBlockInfo_.mTile) *
        ConvCeilDiv(shapeInfo_.co ,conv2dBasicBlockInfo_.nTile);
    const uint32_t groupCut = flagInfo_.convGroupType == ConvGroupType::ORI_GROUP_CONV ?
        attrInfo_.groups : optGroupInfo_.groupOpt;

    const uint32_t calCut1 = static_cast<uint32_t>(sqrt(static_cast<double>(cores * groupCut) /
                                                        static_cast<double>(fwCut)));
    const uint32_t calCut2 = static_cast<uint32_t>(sqrt(static_cast<double>(static_cast<uint64_t>(cores) * fwCut) /
                                                        static_cast<double>(groupCut)));
    return GenerateCandidates(cores, fwCut, groupCut, calCut1, calCut2);
}

void Conv2dBaseTiling::GetInitBasicBlockMN(uint64_t& basicBlockM, uint64_t& basicBlockN)
{
    uint64_t howo = shapeInfo_.ho * shapeInfo_.wo;
    if (shapeInfo_.co <= BASICBLOCK_BOUNDARY_VALUE_64) {
        basicBlockM = BASICBLOCK_INIT_VALUE_1024;
        basicBlockN = BASICBLOCK_INIT_VALUE_64;
    } else if (shapeInfo_.co > BASICBLOCK_BOUNDARY_VALUE_64
        && shapeInfo_.co <= BASICBLOCK_BOUNDARY_VALUE_128) {
        basicBlockM = BASICBLOCK_INIT_VALUE_512;
        basicBlockN = BASICBLOCK_INIT_VALUE_128;
    } else if (howo <= BASICBLOCK_BOUNDARY_VALUE_64) {
        basicBlockM = BASICBLOCK_INIT_VALUE_64;
        basicBlockN = BASICBLOCK_INIT_VALUE_1024;
    } else if (howo > BASICBLOCK_BOUNDARY_VALUE_64
        && howo <= BASICBLOCK_BOUNDARY_VALUE_128) {
        basicBlockM = BASICBLOCK_INIT_VALUE_128;
        basicBlockN = BASICBLOCK_INIT_VALUE_512;
    } else {
        basicBlockM = BASICBLOCK_INIT_VALUE_256;
        basicBlockN = BASICBLOCK_INIT_VALUE_256;
    }
}

void Conv2dBaseTiling::EnableInnerBatchBasicBlock(int64_t availableL1Size) {
    if (flagInfo_.enableC04Flag || flagInfo_.convGroupType != ConvGroupType::NORMAL_CONV ||
        featureFlagInfo_ == ConvAscendcFeatureFlag::IS_DMA_FLAG) {
        return;
    }
    uint64_t basicBlockM = L0C_SIZE / CONST_VALUE_2 / CONST_VALUE_2 / conv2dBasicBlockInfo_.nTile /
        convOpsConstParams_.m0 * convOpsConstParams_.m0;
    basicBlockM = basicBlockM > BASICBLOCK_INIT_VALUE_1024 ? BASICBLOCK_INIT_VALUE_1024 : basicBlockM;
    uint64_t innerBatchTemp = (availableL1Size / dtypeSizeTab.at(descInfo_.fMapDtype) / CONST_VALUE_2 /
        CONST_VALUE_2 / convOpsConstParams_.k0) / (shapeInfo_.wi * shapeInfo_.hi);
    uint32_t innerBatch = max(static_cast<uint64_t>(1), min(shapeInfo_.batch, innerBatchTemp));
    uint64_t mTileInnerBatch  = min(innerBatch * ConvCeilDiv(shapeInfo_.ho * shapeInfo_.wo, convOpsConstParams_.m0) *
        convOpsConstParams_.m0, static_cast<uint64_t>(basicBlockM));
    innerBatch = mTileInnerBatch / (ConvCeilDiv(shapeInfo_.ho * shapeInfo_.wo, convOpsConstParams_.m0) *
        convOpsConstParams_.m0);
    if (innerBatch <= CONST_VALUE_1) {
        return;
    }
    uint64_t nCut = ConvCeilDiv(shapeInfo_.co, conv2dBasicBlockInfo_.nTile);
    uint64_t fCut = ConvCeilDiv(ConvCeilDiv(shapeInfo_.batch *  ConvCeilDiv(shapeInfo_.ho *
        shapeInfo_.wo, convOpsConstParams_.m0) * convOpsConstParams_.m0, conv2dBasicBlockInfo_.mTile), innerBatch);
    if (fCut * nCut <= opInfo_->aicoreNum && ConvCeilDiv(shapeInfo_.batch, opInfo_->aicoreNum) == CONST_VALUE_1) {
        return;
    }
    conv2dApiTiling_.enableInnerBatch = true;
    conv2dApiTiling_.innerBatch = innerBatch;
    return;
}
 
bool Conv2dBaseTiling::CmpFirstAdjustMnTile(int64_t availableL1Size, int64_t& mTile, int64_t& nTile,
    uint64_t basicBlockM, uint64_t basicBlockN)
{
    int64_t fMapDtypeSize = dtypeSizeTab.at(descInfo_.fMapDtype);
    int64_t maxHiWiL1 = availableL1Size / fMapDtypeSize / convOpsConstParams_.k0 /
                        static_cast<int64_t>(CONST_VALUE_2);
    if (maxHiWiL1 <= 0) {
        return false;
    }
    int64_t maxhiL1 = maxHiWiL1 / static_cast<int64_t>(shapeInfo_.wi);
    if (maxhiL1 <= static_cast<int64_t>(CONST_VALUE_2)) {
        return false;
    }
    int64_t hoMax = 0;
    hoMax = (maxhiL1 - static_cast<int64_t>(attrInfo_.dilationH) * (static_cast<int64_t>(shapeInfo_.kh) - 1) - 1) /
        static_cast<int64_t>(attrInfo_.strideH) + 1 - CONST_VALUE_2;
    if (hoMax <= 0) {
          return false;
    }
    int64_t maxHoWoL1 = hoMax * static_cast<int64_t>(shapeInfo_.wo);
    int64_t cmpM = shapeInfo_.ho * shapeInfo_.wo;
    int64_t weightDtypeSize = dtypeSizeTab.at(descInfo_.weightDtype);
    int64_t cmpN = availableL1Size / weightDtypeSize / CONST_VALUE_2 / convOpsConstParams_.k0 /
                    shapeInfo_.kh / shapeInfo_.kw;
    if (static_cast<int64_t>(ConvAlignB(cmpM, convOpsConstParams_.m0)) <= min(static_cast<int64_t>(basicBlockM), maxHoWoL1)) {
        mTile = ConvAlignB(cmpM, convOpsConstParams_.m0);
    } else {
        mTile = min(min(cmpM, maxHoWoL1), static_cast<int64_t>(basicBlockM));
    }
    nTile = min(min(static_cast<int64_t>(shapeInfo_.co), cmpN), static_cast<int64_t>(basicBlockN));
    if (flagInfo_.convGroupType == ConvGroupType::ORI_GROUP_CONV) {
        nTile = static_cast<uint32_t>(nTile) / attrInfo_.groups;
    } else if (flagInfo_.convGroupType == ConvGroupType::OPT_GROUP_CONV) {
        nTile = static_cast<uint32_t>(nTile) / attrInfo_.groups * static_cast<uint32_t>(optGroupInfo_.enlarge);
    }

    mTile = mTile / convOpsConstParams_.m0 * convOpsConstParams_.m0;
    nTile = nTile / convOpsConstParams_.n0 * convOpsConstParams_.n0;
    if (mTile < static_cast<int64_t>(convOpsConstParams_.m0) || nTile < static_cast<int64_t>(convOpsConstParams_.n0)) {
        return false;
    }
    return true;
}
}
}