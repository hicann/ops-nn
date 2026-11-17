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
 * \file conv2d_v2_base_tiling_aswt.cpp
 * \brief
 */
#include "conv2d_v2_base_tiling.h"

namespace optiling {
namespace conv_ops_tiling {

static uint64_t minAdaptBaseM = UINT64_MAX;
static uint64_t minAdaptBaseN = UINT64_MAX;
static uint64_t depthRecursion = 0;

void Conv2dBaseTiling::FindMinAdaptTile(vector<tailRegion>& tailRegions, int64_t adaptBaseM, int64_t adaptBaseN)
{
    if (adaptBaseM < convBase_.m0_ || adaptBaseN < convBase_.n0_ || depthRecursion > MAX_DEPTH_RECURSION)
        return;
    depthRecursion++;
    uint64_t curCoreNums = 0;
    for (auto& tailRegion : tailRegions) {
        curCoreNums = curCoreNums +
                      tailRegion.standardBatch * ConvCeilDiv(tailRegion.oriM, adaptBaseM) *
                          ConvCeilDiv(tailRegion.oriN, adaptBaseN) +
                      tailRegion.tailBatch * ConvCeilDiv(tailRegion.oriM, adaptBaseM) *
                          ConvCeilDiv(tailRegion.tailOriN, adaptBaseN);
    }
    if (curCoreNums <= opInfo_->aicoreNum &&
        static_cast<uint64_t>(adaptBaseM * adaptBaseN) < minAdaptBaseM * minAdaptBaseN) {
        minAdaptBaseM = adaptBaseM;
        minAdaptBaseN = adaptBaseN;
    }
    if (curCoreNums < opInfo_->aicoreNum) {
        FindMinAdaptTile(tailRegions, adaptBaseM - convBase_.m0_, adaptBaseN);
        FindMinAdaptTile(tailRegions, adaptBaseM, adaptBaseN - convBase_.n0_);
    }
}

void Conv2dBaseTiling::GetTailRegions(vector<tailRegion>& tailRegions, uint64_t oriM, uint64_t baseM, uint64_t baseN,
                                      uint64_t mWindow)
{
    uint64_t mCuts = ConvCeilDiv(oriM, baseM);
    uint64_t nCuts = ConvCeilDiv(shapeInfo_.co, baseN);
    uint64_t tailBaseM = oriM % baseM == 0 ? baseM : oriM % baseM;

    uint64_t allCuts = mCuts * nCuts * shapeInfo_.batch;
    uint64_t allStandardCuts = (allCuts / opInfo_->aicoreNum) * opInfo_->aicoreNum;
    if (allStandardCuts == allCuts)
        return;

    uint64_t mWindowsAllCuts = mWindow * nCuts * shapeInfo_.batch;
    uint64_t tailMWindowsCuts = allStandardCuts % mWindowsAllCuts;
    uint64_t tailMCuts = tailMWindowsCuts % mWindow;
    uint64_t tailMWindow = mCuts % mWindow == 0 ? mWindow : mCuts % mWindow;
    bool noExistRegion2 = (ConvCeilDiv(allStandardCuts, mWindowsAllCuts) == ConvCeilDiv(mCuts, mWindow));
    if (noExistRegion2) {
        tailMCuts = tailMWindowsCuts % tailMWindow;
    }

    if (tailMCuts != 0) {
        uint64_t tailRegion0OrgM = tailMCuts * baseM;
        uint64_t standardNCuts = ConvCeilDiv(tailMWindowsCuts, mWindow);
        if (noExistRegion2) {
            standardNCuts = ConvCeilDiv(tailMWindowsCuts, tailMWindow);
        }
        uint64_t tailRegion0AllNCuts = shapeInfo_.batch * nCuts - standardNCuts;
        uint64_t tailRegion0StandardBatch = tailRegion0AllNCuts / nCuts;
        uint64_t tailRegion0TailNCuts = tailRegion0AllNCuts % nCuts;
        tailRegions[TAIL_REGION0] = {tailRegion0StandardBatch, tailRegion0OrgM, shapeInfo_.co, 0,
                                     tailRegion0TailNCuts * baseN};
        if (tailRegion0TailNCuts != 0)
            tailRegions[TAIL_REGION0].tailBatch = 1;
    }

    if (tailMWindowsCuts != 0) {
        uint64_t tailRegion1OrgM = (mWindow - tailMCuts) * baseM;
        uint64_t standardNCuts = tailMWindowsCuts / mWindow;
        if (noExistRegion2) {
            tailRegion1OrgM = (tailMWindow - tailMCuts) * baseM;
            standardNCuts = tailMWindowsCuts / tailMWindow;
        }
        if ((allStandardCuts / mWindowsAllCuts) * mWindow + mWindow >= mCuts) {
            tailRegion1OrgM = tailRegion1OrgM - baseM + tailBaseM;
        }
        uint64_t tailRegion1AllNCuts = shapeInfo_.batch * nCuts - standardNCuts;
        uint64_t tailRegion1StandardBatch = tailRegion1AllNCuts / nCuts;
        uint64_t tailRegion1TailNCuts = tailRegion1AllNCuts % nCuts;
        tailRegions[TAIL_REGION1] = {tailRegion1StandardBatch, tailRegion1OrgM, shapeInfo_.co, 0,
                                     tailRegion1TailNCuts * baseN};
        if (tailRegion1TailNCuts != 0)
            tailRegions[TAIL_REGION1].tailBatch = 1;
    }

    if (noExistRegion2)
        return;
    int64_t tailRegion2OrgM = (mCuts - ConvCeilDiv(allStandardCuts, mWindowsAllCuts) * mWindow) * baseM - baseM +
                              tailBaseM;
    tailRegions[TAIL_REGION2] = {shapeInfo_.batch, static_cast<uint64_t>(tailRegion2OrgM), shapeInfo_.co, 0, 0};
}

void Conv2dBaseTiling::SetAWSTTilingdata(vector<tailRegion>& tailRegions, uint64_t mWindow)
{
    uint32_t tailRegionCuts[TAIL_REGIONS];
    uint32_t tailRegionOrgM[TAIL_REGIONS];
    uint32_t tailRegionOrgN[TAIL_REGIONS];
    uint32_t tailRegionBatchTailNs[TAIL_REGIONS];

    for (int j = 0; j < TAIL_REGIONS; ++j) {
        tailRegionCuts[j] = tailRegions[j].standardBatch * ConvCeilDiv(tailRegions[j].oriM, minAdaptBaseM) *
                                ConvCeilDiv(tailRegions[j].oriN, minAdaptBaseN) +
                            tailRegions[j].tailBatch * ConvCeilDiv(tailRegions[j].oriM, minAdaptBaseM) *
                                ConvCeilDiv(tailRegions[j].tailOriN, minAdaptBaseN);
        tailRegionOrgM[j] = tailRegions[j].oriM;
        tailRegionOrgN[j] = tailRegions[j].oriN;
        tailRegionBatchTailNs[j] = tailRegions[j].tailOriN % minAdaptBaseN;
        if (tailRegionBatchTailNs[j] == 0)
            tailRegionBatchTailNs[j] = minAdaptBaseN;
    }

    tilingData_.set_adaptBaseM(minAdaptBaseM);
    tilingData_.set_adaptBaseN(minAdaptBaseN);
    tilingData_.set_mWindows(mWindow);
    tilingData_.set_tailRegionCuts(tailRegionCuts);
    tilingData_.set_tailRegionOrgM(tailRegionOrgM);
    tilingData_.set_tailRegionOrgN(tailRegionOrgN);
    tilingData_.set_tailRegionBatchTailNs(tailRegionBatchTailNs);
}

void Conv2dBaseTiling::PrintASWTTilingInfo()
{
    std::stringstream ss;
    ss << "mWindows: " << tilingData_.get_mWindows() << ", adaptBaseM: " << tilingData_.get_adaptBaseM()
       << ", adaptBaseN: " << tilingData_.get_adaptBaseN();

    for (int i = 0; i < TAIL_REGIONS; ++i) {
        ss << ", tailRegionCuts[" << i << "]: " << tilingData_.get_tailRegionCuts()[i] << ", tailRegionOrgM[" << i
           << "]: " << tilingData_.get_tailRegionOrgM()[i] << ", tailRegionOrgN[" << i
           << "]: " << tilingData_.get_tailRegionOrgN()[i];
        if (i < TAIL_REGIONS - 1) {
            ss << ", tailRegionBatchTailNs[" << i << "]: " << tilingData_.get_tailRegionBatchTailNs()[i];
        }
    }
    OP_LOGD(context_->GetNodeType(), "%s AscendC: ASWT tilingdata: %s", paramInfo_.nodeType.c_str(), ss.str().c_str());
}

uint64_t Conv2dBaseTiling::ASWTDecisionMWindow(uint64_t mCuts) const
{
    uint64_t mWindow = static_cast<uint64_t>(sqrt(static_cast<double>(opInfo_->aicoreNum)));
    while (opInfo_->aicoreNum % mWindow != 0) {
        mWindow++;
    }

    return min(mWindow, mCuts);
}

void Conv2dBaseTiling::ASWTDecision()
{
    uint64_t mCuts = ConvCeilDiv(shapeInfo_.wo * shapeInfo_.ho, tilingData_.get_hoL0());
    uint64_t outputTileNum = shapeInfo_.batch * ConvCeilDiv(shapeInfo_.co, tilingData_.get_nL0()) * mCuts;
    uint64_t usedCoreNum = numBlocksRes.batchDim * numBlocksRes.mDim * numBlocksRes.nDim * numBlocksRes.groupDim;
    if ((opInfo_->npuArch != NpuArch::DAV_9201 && opInfo_->npuArch != NpuArch::DAV_9202) || flagInfo_.quantFlag ||
        flagInfo_.extendConvFlag || flagInfo_.enableC04Flag || !flagInfo_.mSplitModeFlag || attrInfo_.groups != 1 ||
        (tilingData_.get_innerBatch() > 1) || tilingData_.get_hoL1() != tilingData_.get_hoL0() ||
        tilingData_.get_nBL1() != tilingData_.get_nL0() || outputTileNum <= opInfo_->aicoreNum ||
        usedCoreNum >= opInfo_->aicoreNum || featureFlagInfo_ == ConvAscendcFeatureFlag::IS_DMA_FLAG ||
        (tilingData_.get_bUbKStep() > 0 && tilingData_.get_bUbNStep() > 0)) {
        return;
    }

    flagInfo_.isASWT = true;
    convBase_.SetMKN(CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.fMapDtype), MKN_M_IDX),
                     CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.fMapDtype), MKN_K_IDX),
                     CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.fMapDtype), MKN_N_IDX));
    vector<tailRegion> tailRegions(TAIL_REGIONS, {0, 0, 0, 0, 0});
    uint64_t mWindow = ASWTDecisionMWindow(mCuts);
    minAdaptBaseM = tilingData_.get_hoL0();
    minAdaptBaseN = tilingData_.get_nL0();
    depthRecursion = 0;

    GetTailRegions(tailRegions, shapeInfo_.ho * shapeInfo_.wo, tilingData_.get_hoL0(), tilingData_.get_nL0(), mWindow);
    FindMinAdaptTile(tailRegions, tilingData_.get_hoL0(), tilingData_.get_nL0());

    SetAWSTTilingdata(tailRegions, mWindow);
    PrintASWTTilingInfo();
}

} // namespace conv_ops_tiling
} // namespace optiling
