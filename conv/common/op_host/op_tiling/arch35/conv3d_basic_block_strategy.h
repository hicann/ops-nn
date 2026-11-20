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
 * \file conv3d_basic_block_strategy.h
 * \brief
 */

#ifndef CONV3D_BASIC_BLOCK_STRATEGY_H
#define CONV3D_BASIC_BLOCK_STRATEGY_H

#include "conv_basic_block_strategy.h"
#include "conv_base_numblocks_decision.h"

namespace conv_tiling {

struct Tuple3Hash {
    template <class T>
    size_t operator()(const tuple<T, T, T>& t) const
    {
        return hash<T>()(get<0>(t)) ^ hash<T>()(get<1>(t)) ^ hash<T>()(get<CONST_VALUE_2>(t));
    }
};

class Conv3DBasicBlockStrategy : public ConvBasicBlockStrategy {
public:
    uint32_t GetDoCut(const BBCommonParams& params) const override
    {
        return static_cast<uint32_t>(params.shapeInfo.dout);
    }

    bool ShouldSkipBasicBlock(const BBCommonParams& params) const override
    {
        auto npuArch = params.platformInfo.npuArch;
        if (npuArch != NpuArch::DAV_9201) {
            return true;
        }
        return ConvBasicBlockStrategy::ShouldSkipBasicBlock(params);
    }

    uint64_t GetFwCut(const BBCommonParams& params, const BasicBlockInfo& bbInfo) const override
    {
        return params.shapeInfo.batch * params.shapeInfo.dout * bbInfo.mCut;
    }

    uint64_t GetFwCutForGroup(const BBCommonParams& params, const BasicBlockInfo& bbInfo) const override
    {
        return params.shapeInfo.batch * params.shapeInfo.dout *
               ConvCeilDiv(params.shapeInfo.ho * params.shapeInfo.wo, bbInfo.mTile) *
               ConvCeilDiv(params.shapeInfo.co, bbInfo.nTile);
    }

    uint64_t CalcInputAI(const BBCommonParams& params, const BasicBlockInfo& bbInfo) const override
    {
        return CONST_VALUE_2 * params.shapeInfo.batch * params.shapeInfo.dout * params.shapeInfo.wo *
               params.shapeInfo.ho * params.shapeInfo.co * params.shapeInfo.kd * params.shapeInfo.kh *
               params.shapeInfo.kw /
               (params.shapeInfo.batch * params.shapeInfo.di * params.shapeInfo.wi * params.shapeInfo.hi * bbInfo.nCut +
                params.shapeInfo.kd * params.shapeInfo.kh * params.shapeInfo.kw * params.shapeInfo.co * bbInfo.mCut) /
               dtypeSizeTab.at(params.descInfo.fMapDtype);
    }

    uint64_t CalcOutputAI(const BBCommonParams& params) const override
    {
        uint64_t dtypeSize = dtypeSizeTab.at(params.descInfo.outDtype);
        uint64_t ciAlign = C0_SIZE / dtypeSize;
        uint64_t alignedCi = ConvAlignB(params.shapeInfo.ci, ciAlign);
        return CONST_VALUE_2 * alignedCi * params.shapeInfo.kd * params.shapeInfo.kh * params.shapeInfo.kw / dtypeSize;
    }

    bool ShouldAdjustMnTile(const BBCommonParams& params, const BasicBlockInfo& bbInfo) const override
    {
        if (bbInfo.mCut * params.shapeInfo.batch * bbInfo.doCut >= params.aicoreNum && bbInfo.nTile <= COUT_LIMIT_128 &&
            params.attrInfo.groups == 1 && params.shapeInfo.kh == 1 && params.shapeInfo.kw == 1 && bbInfo.nDim > 1) {
            return false;
        }
        return true;
    }

    void CalcL0InitParams(const BBCommonParams& params, BasicBlockInfo& bbInfo, uint64_t availableL1Size) const override
    {
        CalcL0InitParamsCommon(params, bbInfo, availableL1Size);
        bbInfo.nKernelHKernelW *= params.shapeInfo.kd;
        bbInfo.doCut = params.shapeInfo.dout;
        bbInfo.useStrictAdjustM = true;
    }

    uint64_t CalcBasicBlockNum1(const BBCommonParams& params, const BasicBlockInfo& bbInfo) const override
    {
        return ConvCeilDiv(params.shapeInfo.batch * bbInfo.mCut * bbInfo.doCut, bbInfo.fDim);
    }

    void ExecuteBatchMDimDecision(const BBCommonParams& params, BasicBlockInfo& bbInfo) const override
    {
        auto candidates = Generate3DBatchMDimCandidates(params, bbInfo);
        float minCost = numeric_limits<float>::max();
        tuple<uint32_t, uint32_t, uint32_t> optDecision = {1, 1, 1};
        uint32_t mCutAlignM0 = static_cast<uint32_t>(
            ConvCeilDiv(params.shapeInfo.ho * params.shapeInfo.wo, params.convOpsConstParams.m0));
        uint32_t batchCut = ConvCeilDiv(params.shapeInfo.batch, params.innerBatch);
        uint32_t doCut = bbInfo.doCut;
        for (const auto& candidate : candidates) {
            uint32_t batchDim = get<0>(candidate);
            uint32_t doDim = get<1>(candidate);
            uint32_t mDim = get<CONST_VALUE_2>(candidate);
            uint32_t cost1 = ConvCeilDiv(batchCut, batchDim) * ConvCeilDiv(doCut, doDim) *
                             ConvCeilDiv(mCutAlignM0, mDim);
            float cost2 = 1.0f / static_cast<float>(batchDim * doDim);
            float cost = static_cast<float>(cost1) + cost2;
            if (cost < minCost) {
                minCost = cost;
                optDecision = candidate;
            }
        }
        bbInfo.batchDim = get<0>(optDecision);
        bbInfo.doDim = get<1>(optDecision);
        bbInfo.mDim = get<CONST_VALUE_2>(optDecision);
    }

    void SetCandidateSingleOutputShape(const BBCommonParams& params, BasicBlockInfo& bbInfo,
                                       uint32_t nDim) const override
    {
        uint64_t curCo = params.flagInfo.convGroupType != ConvGroupType::NORMAL_CONV ?
                             (params.flagInfo.convGroupType == ConvGroupType::ORI_GROUP_CONV ?
                                  params.oriGroupInfo.coPerGroup :
                                  params.optGroupInfo.coutOpt) :
                             params.shapeInfo.co;
        uint64_t n0 = static_cast<uint64_t>(CUBE_MKN_MAP.GetMKN(dtypeMap.at(params.descInfo.fMapDtype), MKN_N_IDX));
        uint64_t singleCoreCo = ConvAlignB(ConvCeilDiv(ConvAlignB(curCo, n0), nDim), n0);
        int64_t singleCoreDo = static_cast<int64_t>(ConvCeilDiv(params.shapeInfo.dout, bbInfo.doDim));
        int64_t singleCoreBatch = static_cast<int64_t>(ConvCeilDiv(params.shapeInfo.batch, bbInfo.batchDim));
        params.tilingBase->SetSingleOutputShapeCoDoMBatch(static_cast<int64_t>(singleCoreCo), singleCoreDo, 0,
                                                          singleCoreBatch);
    }

private:
    unordered_set<tuple<uint32_t, uint32_t, uint32_t>, Tuple3Hash> Generate3DBatchMDimCandidates(
        const BBCommonParams& params, const BasicBlockInfo& bbInfo) const
    {
        unordered_set<tuple<uint32_t, uint32_t, uint32_t>, Tuple3Hash> candidates;
        uint32_t cores = bbInfo.fDim;
        uint32_t mDimUpper = static_cast<uint32_t>(
            ConvCeilDiv(params.shapeInfo.ho * params.shapeInfo.wo, params.convOpsConstParams.m0));
        uint32_t batchCut = ConvCeilDiv(params.shapeInfo.batch, params.innerBatch);
        uint32_t doCut = static_cast<uint32_t>(params.shapeInfo.dout);
        for (uint32_t batchDim = 1; batchDim <= min(batchCut, cores); batchDim++) {
            if (cores % batchDim != 0)
                continue;
            uint32_t remain1 = cores / batchDim;

            for (uint32_t doDim = 1; doDim <= min(doCut, remain1); doDim++) {
                if (remain1 % doDim != 0)
                    continue;
                uint32_t mDim = remain1 / doDim;
                if (mDim > mDimUpper)
                    continue;
                candidates.emplace(batchDim, doDim, mDim);
            }
        }
        if (candidates.empty()) {
            if (batchCut <= cores)
                candidates.emplace(batchCut, 1, 1);
            if (doCut <= cores)
                candidates.emplace(1, doCut, 1);
            candidates.emplace(1, 1, bbInfo.fDim);
        }
        return candidates;
    }
};

} // namespace conv_tiling

#endif
