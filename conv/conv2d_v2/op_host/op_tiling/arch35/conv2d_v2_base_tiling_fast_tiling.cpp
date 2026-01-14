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
 * \file conv2d_v2_base_tiling_fast_tiling.cpp
 * \brief
 */
#include "conv2d_v2_base_tiling.h"
 
namespace optiling {
namespace conv_ops_tiling {
ge::graphStatus Conv2dBaseTiling::PrepareTiling()
{
    if (CheckInputDesc() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckParamsDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckFmapShape() != ge::GRAPH_SUCCESS || CheckWeightShape() != ge::GRAPH_SUCCESS ||
        CheckBiasShape() != ge::GRAPH_SUCCESS || CheckOutputShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckExtendDualOutputSpecial() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ShapeAttrSynthesisCheck(oriShapeAttrInfo_, context_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckInstructionLimits() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (Conv2DInfoInitAndCheck() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::GetTilingFromFastTiling()
{
    Conv2dOpTilingSetShape();
    convBase_.UpdateFlagInfo(flagInfo_);
    OP_LOGD(context_->GetNodeName(), "%s AscendC: get tiling from fast_tiling.", paramInfo_.nodeType.c_str());
    if (flagInfo_.mSplitModeFlag) { // M split mode
        if (flagInfo_.mBasicBlockFlag) {
            BasicBlock();
            OP_LOGD(context_->GetNodeName(),
                "%s AscendC: get tiling from Mmode basic block algorithm, belong to fast_tiling.",
                paramInfo_.nodeType.c_str());
            OP_LOGD(context_->GetNodeName(),
                "%s AscendC: batchDim / mDim / nDim / groupDim: %u, %u, %u, %u.", paramInfo_.nodeType.c_str(),
                blockDimRes.batchDim, blockDimRes.mDim, blockDimRes.nDim, blockDimRes.groupDim);
        } else {
            blockDimRes = convBase_.BlockDimDecisionMsplitMode();
            OP_LOGD(context_->GetNodeName(),
                "%s AscendC: get tiling from Mmode formulaic algorithm, belong to fast_tiling.",
                paramInfo_.nodeType.c_str());
            OP_LOGD(context_->GetNodeName(),
                "%s AscendC: batchDim / mDim / nDim / doDim / groupDim / minCost: %u, %u, %u, %u, %u, %lu.",
                paramInfo_.nodeType.c_str(), blockDimRes.batchDim, blockDimRes.mDim, blockDimRes.nDim,
                blockDimRes.doDim, blockDimRes.groupDim, blockDimRes.minCost);
        }
    } else { // HW split mode
        blockDimRes = convBase_.BlockDimDecisionHWsplitMode();
        OP_LOGD(context_->GetNodeName(),
            "%s AscendC: get tiling from HWmode formulaic algorithm, belong to fast_tiling.",
            paramInfo_.nodeType.c_str());
        OP_LOGD(context_->GetNodeName(),
            "%s AscendC: batchDim / hoDim / woDim / nDim / groupDim / minCost: %u, %u, %u, %u, %u, %lu.",
            paramInfo_.nodeType.c_str(), blockDimRes.batchDim, blockDimRes.hoDim, blockDimRes.woDim,
            blockDimRes.nDim, blockDimRes.groupDim, blockDimRes.minCost);
    }
    if (GetConv2dOpsTiling() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::Conv2DInfoInitAndCheck()
{
    convBase_.ConvBaseInitAttrInfo(attrInfo_);
    convBase_.ConvBaseInitOpInfo(opInfo_);
    convBase_.updatePlatformInfoFromOpInfo();
    convBase_.ConvBaseInitFixpipeInfo(fixpipeInfo_);
    convBase_.InitblockDimConstParas();
    convBase_.GetConvBaseCoreInfo(convOpsConstParams_);
    // check if enable c04 mode
    flagInfo_.enableC04Flag = IsEnableC04();

    convBase_.ConvBaseInitFeatureFlag(featureFlagInfo_);

    if (GetTilingSplitMode() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::GetC04TilingSplitMode()
{
    flagInfo_.mSplitModeFlag = false;
    bool forceHWSplitModeFlag = featureFlagInfo_ == ConvAscendcFeatureFlag::IS_CONV1D_FLAG &&
                                shapeInfo_.wo > ENABLE_MMODE_CONV1D_WO_LIMIT_128;
    if (!forceHWSplitModeFlag && convBase_.CheckC04L1SizeLimitsInMsplitMode() == ge::GRAPH_SUCCESS) {
        flagInfo_.mSplitModeFlag = true;
        return ge::GRAPH_SUCCESS;
    }
    if (convBase_.CheckC04L1SizeLimitsInHWSplitMode() != ge::GRAPH_SUCCESS) {
        flagInfo_.enableC04Flag = false;
        return convBase_.CheckL1SizeLimitsInHWsplitMode();
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::GetTilingSplitMode()
{
    if (flagInfo_.enableC04Flag) {
        return GetC04TilingSplitMode();
    }

    bool flagHWCheckFirst = false;
    uint64_t basicBlockM = 0;
    uint64_t basicBlockN = 0;
    uint64_t orgWo = (shapeInfo_.wi + attrInfo_.padLeft + attrInfo_.padRight - attrInfo_.dilationW *
                     (shapeInfo_.kw - 1) - 1) / attrInfo_.strideW + 1;
    GetInitBasicBlockMN(basicBlockM, basicBlockN);
    if ((orgWo > BASICBLOCK_INIT_VALUE_1024) ||
        ((orgWo > CONST_VALUE_2 * basicBlockM) && ((orgWo % basicBlockM) * CONST_VALUE_2 > basicBlockM))) {
            flagHWCheckFirst = true;
            OP_LOGD(context_->GetNodeName(),
                "%s AscendC: get tiling by check HWMode first with orgWo %lu, basicBlockM %lu, basicBlockN %lu.",
                paramInfo_.nodeType.c_str(), orgWo, basicBlockM, basicBlockN);
    }

    flagInfo_.mSplitModeFlag = false;
    if (!flagHWCheckFirst && (featureFlagInfo_ == ConvAscendcFeatureFlag::IS_LOAD3D_FLAG || (featureFlagInfo_ ==
        ConvAscendcFeatureFlag::IS_CONV1D_FLAG && orgWo <= ENABLE_MMODE_CONV1D_WO_LIMIT_128)) &&
        convBase_.CheckL1SizeLimitsInMSplitMode() == ge::GRAPH_SUCCESS) { // M split mode check, conv1d/dma skip M split
        flagInfo_.mSplitModeFlag = true;
        SelectMModeAlgorithm();
        return ge::GRAPH_SUCCESS;
    }
    if (CheckNHWCDataCopyLimits() == ge::GRAPH_SUCCESS &&
        convBase_.CheckL1SizeLimitsInHWsplitMode() == ge::GRAPH_SUCCESS) { // HW split mode check
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

void Conv2dBaseTiling::SelectMModeAlgorithm()
{
    flagInfo_.mBasicBlockFlag = false;
    uint64_t basicBlockM = 0;
    uint64_t basicBlockN = 0;
    GetInitBasicBlockMN(basicBlockM, basicBlockN);

    uint64_t orgCo = flagInfo_.convGroupType != ConvGroupType::NORMAL_CONV ?
        (flagInfo_.convGroupType == ConvGroupType::ORI_GROUP_CONV ?
        oriGroupInfo_.coPerGroup : optGroupInfo_.coutOpt) : shapeInfo_.co;
    uint64_t mCut = ConvCeilDiv(shapeInfo_.wo * shapeInfo_.ho, basicBlockM);
    uint64_t nCut = ConvCeilDiv(orgCo, basicBlockN);
    uint64_t group = attrInfo_.groups;
    if (flagInfo_.convGroupType == ConvGroupType::OPT_GROUP_CONV) {
        group = optGroupInfo_.groupOpt;
    }
    if (mCut * nCut * shapeInfo_.batch * group <= opInfo_->aicoreNum) {
        return;
    }
    int64_t biasSize = 0;
    int64_t scaleSize = 0;
    if (flagInfo_.hasBias) {
        biasSize = dtypeSizeTab.at(descInfo_.biasDtype) * shapeInfo_.co;
    }
    scaleSize = static_cast<int64_t>(fixpipeInfo_.channelWiseCoeff * FP16_DTYPE_SIZE * shapeInfo_.co);
    int64_t availableL1Size = L1_SIZE - biasSize - scaleSize;
    int64_t mTile = 0;
    int64_t nTile = 0;
    if (availableL1Size <= 0 || !CmpFirstAdjustMnTile(availableL1Size, mTile, nTile, basicBlockM, basicBlockN)) {
        return;
    }
    flagInfo_.mBasicBlockFlag = true;
    conv2dBasicBlockInfo_.mTile = static_cast<uint32_t>(mTile);
    conv2dBasicBlockInfo_.nTile = static_cast<uint32_t>(nTile);
    OP_LOGD(context_->GetNodeName(),
            "mTile %u, nTile %u", conv2dBasicBlockInfo_.mTile, conv2dBasicBlockInfo_.nTile);
    EnableInnerBatchBasicBlock(availableL1Size);
    return;
}

void Conv2dBaseTiling::Conv2dApiTilingSetShape()
{
    uint64_t curCo = flagInfo_.convGroupType != ConvGroupType::NORMAL_CONV ?
        (flagInfo_.convGroupType == ConvGroupType::ORI_GROUP_CONV ?
         oriGroupInfo_.coPerGroup : optGroupInfo_.coutOpt) : shapeInfo_.co;
    uint64_t singleCoreCo = ConvAlignB(ConvCeilDiv(ConvAlignB(curCo, convOpsConstParams_.n0), blockDimRes.nDim),
                            convOpsConstParams_.n0);
    uint64_t singleCoreHo = ConvCeilDiv(shapeInfo_.ho, blockDimRes.hoDim);
    uint64_t singleCoreWo = ConvCeilDiv(shapeInfo_.wo, blockDimRes.woDim);
    uint64_t singleCoreBatch = ConvCeilDiv(shapeInfo_.batch, blockDimRes.batchDim);
    uint64_t singleCoreM = ConvCeilDiv(ConvAlignB(shapeInfo_.ho * shapeInfo_.wo, convOpsConstParams_.m0),
                           blockDimRes.mDim);
    if (flagInfo_.mSplitModeFlag) {
        conv2dApiTiling_.SetSingleOutputShape(static_cast<int64_t>(singleCoreCo), static_cast<int64_t>(singleCoreM),
                                              static_cast<int64_t>(singleCoreBatch));
    } else {
        conv2dApiTiling_.SetSingleOutputShape(static_cast<int64_t>(singleCoreCo), static_cast<int64_t>(singleCoreHo),
            static_cast<int64_t>(singleCoreWo), static_cast<int64_t>(singleCoreBatch));
    }

    conv2dApiTiling_.SetOrgWeightShape(static_cast<int64_t>(shapeInfo_.co), static_cast<int64_t>(shapeInfo_.kh),
                                       static_cast<int64_t>(shapeInfo_.kw));
    uint64_t singleGroups = 0;
    uint64_t singleGroupOpt = 0;
    uint64_t enlarge = 0;
    if (flagInfo_.convGroupType == ConvGroupType::OPT_GROUP_CONV) {
        singleGroups = optGroupInfo_.enlarge;
        singleGroupOpt = ConvCeilDiv(optGroupInfo_.groupOpt, blockDimRes.groupDim);
        enlarge = optGroupInfo_.enlarge;
        conv2dApiTiling_.SetOptGroupParams(static_cast<int32_t>(enlarge), static_cast<int64_t>(singleGroups),
                                           static_cast<int64_t>(singleGroupOpt));
    }
    std::stringstream ss;
    ss << "orgCo: " << shapeInfo_.co << ", orgkH: " << shapeInfo_.kh << ", orgkW: " << shapeInfo_.kw
       << ", orgHi: " << shapeInfo_.hi << ", orgWi: " << shapeInfo_.wi << ", singleCo: " << singleCoreCo
       << ", singleHo: " << singleCoreHo << ", singleWo: " << shapeInfo_.wo << ", singleM: " << singleCoreM
       << ", singleGroups: " << singleGroups << ", singleGroupOpt: " << singleGroupOpt << ", enlarge: " << enlarge;
    OP_LOGD(context_->GetNodeName(), "%s AscendC: api got: %s", paramInfo_.nodeType.c_str(), ss.str().c_str());
}

ge::graphStatus Conv2dBaseTiling::GetConv2dApiTiling()
{
    Conv2dApiTilingSetShape();

    if (flagInfo_.mBasicBlockFlag) {
        if (!conv2dApiTiling_.GetTiling(conv2dBasicBlockInfo_, tilingData_.conv2dApiTiling)) {
            OP_LOGE(context_->GetNodeName(), "%s AscendC: get api tiling wrong", paramInfo_.nodeType.c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        if (conv2dApiTiling_.GetTiling(tilingData_.conv2dApiTiling) == -1) {
            OP_LOGE(context_->GetNodeName(), "%s AscendC: get api tiling wrong", paramInfo_.nodeType.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

void Conv2dBaseTiling::Conv2dOpTilingSetAttr()
{
    int8_t outputOrder = flagInfo_.mSplitModeFlag ? 1: 0;
    conv2dApiTiling_.SetOutputOrder(outputOrder);
    conv2dApiTiling_.SetQuantScale(flagInfo_.quantFlag || fixpipeInfo_.channelWiseCoeff > 0);
    conv2dApiTiling_.SetFixpipeParams(fixpipeInfo_);
    conv2dApiTiling_.SetOffsetx(static_cast<int8_t>(attrInfo_.offsetx));
    conv2dApiTiling_.SetC04Flag(flagInfo_.enableC04Flag);
    conv2dApiTiling_.SetRoundMode(static_cast<int8_t>(attrInfo_.roundMode));
    bool isHF32 = (attrInfo_.hf32Mode == 1);
    if (isHF32) {
        bool hf32TransMode = false;
        conv2dApiTiling_.SetHF32(isHF32, hf32TransMode);
    }
 
    conv2dApiTiling_.SetGroups(static_cast<int32_t>(attrInfo_.groups));
}

void Conv2dBaseTiling::Conv2dOpTilingSetShape()
{
    uint64_t orgCo = flagInfo_.convGroupType != ConvGroupType::NORMAL_CONV && flagInfo_.mBasicBlockFlag ?
        (flagInfo_.convGroupType == ConvGroupType::ORI_GROUP_CONV ?
        oriGroupInfo_.coPerGroup : optGroupInfo_.coutOpt) : shapeInfo_.co;
    conv2dApiTiling_.SetOrgWeightShape(static_cast<int64_t>(orgCo), static_cast<int64_t>(shapeInfo_.kh),
                                       static_cast<int64_t>(shapeInfo_.kw));
    conv2dApiTiling_.SetOrgFmapShape(static_cast<int64_t>(shapeInfo_.ci), static_cast<int64_t>(shapeInfo_.hi),
                                     static_cast<int64_t>(shapeInfo_.wi));
    conv2dApiTiling_.SetPadding(static_cast<int32_t>(attrInfo_.padTop), static_cast<int32_t>(attrInfo_.padBottom),
                                static_cast<int32_t>(attrInfo_.padLeft), static_cast<int32_t>(attrInfo_.padRight));
    conv2dApiTiling_.SetDilation(static_cast<int32_t>(attrInfo_.dilationH), static_cast<int32_t>(attrInfo_.dilationW));
    conv2dApiTiling_.SetStride(static_cast<int32_t>(attrInfo_.strideH), static_cast<int32_t>(attrInfo_.strideW));
 
    conv2dApiTiling_.SetWeightType(TPosition::GM, formatMap.at(descInfo_.weightFormat),
                                   dtypeMap.at(descInfo_.weightDtype));
    conv2dApiTiling_.SetFmapType(TPosition::GM, formatMap.at(descInfo_.fMapFormat), dtypeMap.at(descInfo_.fMapDtype));
    conv2dApiTiling_.SetOutputType(TPosition::CO1, formatMap.at(descInfo_.outFormat), dtypeMap.at(descInfo_.outDtype));
 
    if (flagInfo_.hasBias) {
        conv2dApiTiling_.SetBiasType(TPosition::GM, formatMap.at(descInfo_.biasFormat),
                                     dtypeMap.at(descInfo_.biasDtype));
    }

    Conv2dOpTilingSetAttr();

    uint64_t singleCoreCi = flagInfo_.convGroupType != ConvGroupType::NORMAL_CONV ?
        (flagInfo_.convGroupType == ConvGroupType::ORI_GROUP_CONV ?
         oriGroupInfo_.ciPerGroup : optGroupInfo_.cinOpt) : shapeInfo_.ci;
    conv2dApiTiling_.SetSingleWeightShape(static_cast<int64_t>(singleCoreCi), static_cast<int64_t>(shapeInfo_.kh),
        static_cast<int64_t>(shapeInfo_.kw));
    stringstream ss;
    ss << "singleCoreCi: " << static_cast<int64_t>(singleCoreCi)
       << ", singleCorekh: " << static_cast<int64_t>(shapeInfo_.kh)
       << ", singleCorekw: " << static_cast<int64_t>(shapeInfo_.kw);
    OP_LOGD(context_->GetNodeName(), "%s AscendC: api got: %s", paramInfo_.nodeType.c_str(), ss.str().c_str());
 
    if (flagInfo_.convGroupType == ConvGroupType::OPT_GROUP_CONV) {
        // set enlarge for BasicBlock
        conv2dApiTiling_.SetOptGroupParams(static_cast<int32_t>(optGroupInfo_.enlarge), attrInfo_.groups,
            optGroupInfo_.groupOpt);
    }
    convBase_.SetAiCoreNum(opInfo_->aicoreNum);
    convBase_.SetMKN(CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.fMapDtype), MKN_M_IDX),
                     CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.fMapDtype), MKN_K_IDX),
                     CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.fMapDtype), MKN_N_IDX));
}

ge::graphStatus Conv2dBaseTiling::GetConv2dOpsTiling()
{
    tilingData_.conv2dRunInfo.set_hin(static_cast<uint64_t>(shapeInfo_.hi));
    tilingData_.conv2dRunInfo.set_win(static_cast<uint64_t>(shapeInfo_.wi));
    tilingData_.conv2dRunInfo.set_hout(static_cast<uint64_t>(shapeInfo_.ho));
    tilingData_.conv2dRunInfo.set_wout(static_cast<uint64_t>(shapeInfo_.wo));
    tilingData_.conv2dRunInfo.set_batch(static_cast<uint32_t>(shapeInfo_.batch));
    tilingData_.conv2dRunInfo.set_cin(static_cast<uint32_t>(shapeInfo_.ci));
    tilingData_.conv2dRunInfo.set_cout(static_cast<uint32_t>(shapeInfo_.co));
    tilingData_.conv2dRunInfo.set_kh(static_cast<uint32_t>(shapeInfo_.kh));
    tilingData_.conv2dRunInfo.set_kw(static_cast<uint32_t>(shapeInfo_.kw));
    tilingData_.conv2dRunInfo.set_strideH(static_cast<uint32_t>(attrInfo_.strideH));
    tilingData_.conv2dRunInfo.set_strideW(static_cast<uint32_t>(attrInfo_.strideW));
    tilingData_.conv2dRunInfo.set_dilationH(static_cast<uint32_t>(attrInfo_.dilationH));
    tilingData_.conv2dRunInfo.set_dilationW(static_cast<uint32_t>(attrInfo_.dilationW));
    tilingData_.conv2dRunInfo.set_padTop(static_cast<uint32_t>(attrInfo_.padTop));
    tilingData_.conv2dRunInfo.set_padLeft(static_cast<uint32_t>(attrInfo_.padLeft));
    tilingData_.conv2dRunInfo.set_hasBias(static_cast<uint8_t>(flagInfo_.hasBias));
    tilingData_.conv2dRunInfo.set_batchDim(static_cast<uint32_t>(blockDimRes.batchDim));
    tilingData_.conv2dRunInfo.set_nDim(static_cast<uint32_t>(blockDimRes.nDim));
    tilingData_.conv2dRunInfo.set_groupDim(static_cast<uint32_t>(blockDimRes.groupDim));
    tilingData_.conv2dRunInfo.set_groups(static_cast<uint32_t>(attrInfo_.groups));
    if (flagInfo_.convGroupType == ConvGroupType::OPT_GROUP_CONV) {
        tilingData_.conv2dRunInfo.set_cinOpt(static_cast<uint32_t>(optGroupInfo_.cinOpt));
        tilingData_.conv2dRunInfo.set_coutOpt(static_cast<uint32_t>(optGroupInfo_.coutOpt));
        tilingData_.conv2dRunInfo.set_groupOpt(static_cast<uint32_t>(optGroupInfo_.groupOpt));
        tilingData_.conv2dRunInfo.set_enlarge(static_cast<uint32_t>(optGroupInfo_.enlarge));
    }

    if (flagInfo_.mSplitModeFlag) {
        tilingData_.conv2dRunInfo.set_hoDim(static_cast<uint32_t>(blockDimRes.mDim));
        tilingData_.conv2dRunInfo.set_woDim(static_cast<uint32_t>(1));
    } else {
        tilingData_.conv2dRunInfo.set_hoDim(static_cast<uint32_t>(blockDimRes.hoDim));
        tilingData_.conv2dRunInfo.set_woDim(static_cast<uint32_t>(blockDimRes.woDim));
    }
    return ge::GRAPH_SUCCESS;
}
}
}