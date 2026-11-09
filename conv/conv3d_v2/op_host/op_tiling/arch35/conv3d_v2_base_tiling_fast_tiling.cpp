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
 * \file conv3d_v2_base_tiling_fast_tiling.cpp
 * \brief
 */
#include "conv3d_v2_base_tiling.h"
#include "conv/common/op_host/op_tiling/arch35/conv_basic_block.h"
#include "conv/common/op_host/op_tiling/arch35/conv3d_basic_block_strategy.h"
#include "conv/common/op_host/op_tiling/arch35/conv3d_dimension_strategy.h"

namespace optiling {
namespace conv_ops_tiling {
ge::graphStatus Conv3dBaseTilingV2::Conv3DInfoInitAndCheck()
{
    convBase_.ConvBaseInitAttrInfo(attrInfo_);
    convBase_.ConvBaseInitOpInfo(opInfo_);
    convBase_.updatePlatformInfoFromOpInfo();
    convBase_.ConvBaseInitFixpipeInfo(fixpipeInfo_);
    convBase_.InitNumBlocksConstParas();
    convBase_.GetConvBaseCoreInfo(convOpsConstParams_);
    convBase_.ConvBaseInitNodeInfo(paramInfo_.nodeType);

    if (CheckL1SizeLimits() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    convBase_.ConvBaseInitFeatureFlag(ConvAscendcFeatureFlag::IS_LOAD3D_FLAG);
    flagInfo_.convGroupType = GetGroupsInfo();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTilingV2::PrepareTiling()
{
    if (CheckInputDesc() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckParamsDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckFmapShape() != ge::GRAPH_SUCCESS || CheckWeightShape() != ge::GRAPH_SUCCESS ||
        CheckOutputShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ShapeAttrSynthesisCheck(oriShapeAttrInfo_, paramInfo_, context_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckInstructionLimits() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (Conv3DInfoInitAndCheck() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTilingV2::CheckL1SizeLimits()
{
    flagInfo_.mSplitModeFlag = false;
    flagInfo_.isKernelSplit = false;
    // M split mode check
    if (convBase_.CheckL1SizeLimitsInMSplitMode() == ge::GRAPH_SUCCESS && convBase_.CheckInstrLimitsMmode()) {
        flagInfo_.mSplitModeFlag = true;
        return ge::GRAPH_SUCCESS;
    }
    // HW split mode check
    if (convBase_.CheckL1SizeLimitsInHWsplitMode() == ge::GRAPH_SUCCESS && convBase_.CheckInstrLimitsHWmode()) {
        return ge::GRAPH_SUCCESS;
    }
    // HW split & khkw split mode check
    if (descInfo_.weightFormat != ge::FORMAT_FRACTAL_Z_3D &&
        convBase_.CheckKernelSplitL1SizeLimitsInHWSplitMode() == ge::GRAPH_SUCCESS &&
        convBase_.CheckInstrLimitsHWmode()) {
        flagInfo_.isKernelSplit = true;
        return ge::GRAPH_SUCCESS;
    }

    return ge::GRAPH_FAILED;
}

ge::graphStatus Conv3dBaseTilingV2::GetConv3dOpsTiling()
{
    tilingData_.din = static_cast<uint32_t>(shapeInfo_.di);
    tilingData_.hin = static_cast<uint64_t>(shapeInfo_.hi);
    tilingData_.win = static_cast<uint64_t>(shapeInfo_.wi);
    tilingData_.dout = static_cast<uint32_t>(shapeInfo_.dout);
    tilingData_.hout = static_cast<uint64_t>(shapeInfo_.ho);
    tilingData_.wout = static_cast<uint64_t>(shapeInfo_.wo);
    tilingData_.batch = static_cast<uint32_t>(shapeInfo_.batch);
    tilingData_.cin = static_cast<uint32_t>(shapeInfo_.ci);
    tilingData_.cout = static_cast<uint32_t>(shapeInfo_.co);
    tilingData_.kd = static_cast<uint32_t>(shapeInfo_.kd);
    tilingData_.kh = static_cast<uint32_t>(shapeInfo_.kh);
    tilingData_.kw = static_cast<uint32_t>(shapeInfo_.kw);
    tilingData_.batchDim = static_cast<uint32_t>(numBlocksRes.batchDim);
    tilingData_.nDim = static_cast<uint32_t>(numBlocksRes.nDim);
    tilingData_.doDim = static_cast<uint32_t>(numBlocksRes.doDim);
    tilingData_.groupDim = static_cast<uint32_t>(numBlocksRes.groupDim);
    tilingData_.strideH = static_cast<uint32_t>(attrInfo_.strideH);
    tilingData_.strideD = static_cast<uint32_t>(attrInfo_.strideD);
    tilingData_.dilationH = static_cast<uint32_t>(attrInfo_.dilationH);
    tilingData_.dilationD = static_cast<uint32_t>(attrInfo_.dilationD);
    tilingData_.padHead = static_cast<uint32_t>(attrInfo_.padHead);
    tilingData_.padTop = static_cast<uint32_t>(attrInfo_.padTop);
    tilingData_.hasBias = static_cast<uint8_t>(flagInfo_.hasBias);
    tilingData_.groups = static_cast<uint32_t>(attrInfo_.groups);
    if (flagInfo_.convGroupType == ConvGroupType::OPT_GROUP_CONV) {
        tilingData_.cinOpt = static_cast<uint32_t>(optGroupInfo_.cinOpt);
        tilingData_.coutOpt = static_cast<uint32_t>(optGroupInfo_.coutOpt);
        tilingData_.groupOpt = static_cast<uint32_t>(optGroupInfo_.groupOpt);
        tilingData_.enlarge = static_cast<uint32_t>(optGroupInfo_.enlarge);
    }

    if (flagInfo_.mSplitModeFlag) {
        tilingData_.hoDim = static_cast<uint32_t>(numBlocksRes.mDim);
    } else {
        tilingData_.hoDim = static_cast<uint32_t>(numBlocksRes.hoDim);
    }

    return ge::GRAPH_SUCCESS;
}

void Conv3dBaseTilingV2::InitApiTilingCommon()
{
    conv3dApiTiling_.SetOrgFmapShape(static_cast<int64_t>(shapeInfo_.ci), static_cast<int64_t>(shapeInfo_.di),
                                     static_cast<int64_t>(shapeInfo_.hi), static_cast<int64_t>(shapeInfo_.wi));
    conv3dApiTiling_.SetOrgWeightShape(static_cast<int64_t>(shapeInfo_.co), static_cast<int64_t>(shapeInfo_.kd),
                                       static_cast<int64_t>(shapeInfo_.kh), static_cast<int64_t>(shapeInfo_.kw));
    uint64_t singleCoreCi = flagInfo_.convGroupType != ConvGroupType::NORMAL_CONV ?
                                (flagInfo_.convGroupType == ConvGroupType::ORI_GROUP_CONV ? oriGroupInfo_.ciPerGroup :
                                                                                            optGroupInfo_.cinOpt) :
                                shapeInfo_.ci;
    conv3dApiTiling_.SetSingleWeightShape(static_cast<int64_t>(singleCoreCi), static_cast<int64_t>(shapeInfo_.kd),
                                          static_cast<int64_t>(shapeInfo_.kh), static_cast<int64_t>(shapeInfo_.kw));
    conv3dApiTiling_.SetPadding(static_cast<int64_t>(attrInfo_.padHead), static_cast<int64_t>(attrInfo_.padTail),
                                static_cast<int64_t>(attrInfo_.padTop), static_cast<int64_t>(attrInfo_.padBottom),
                                static_cast<int64_t>(attrInfo_.padLeft), static_cast<int64_t>(attrInfo_.padRight));
    conv3dApiTiling_.SetDilation(static_cast<int64_t>(attrInfo_.dilationH), static_cast<int64_t>(attrInfo_.dilationW),
                                 static_cast<int64_t>(attrInfo_.dilationD));
    conv3dApiTiling_.SetStride(static_cast<int64_t>(attrInfo_.strideH), static_cast<int64_t>(attrInfo_.strideW),
                               static_cast<int64_t>(attrInfo_.strideD));
    conv3dApiTiling_.SetGroups(static_cast<int32_t>(attrInfo_.groups));
    int8_t outputOrder = flagInfo_.mSplitModeFlag ? 1 : 0;
    conv3dApiTiling_.SetOutputOrder(outputOrder);
    conv3dApiTiling_.SetKernelSplit(flagInfo_.isKernelSplit);
    conv3dApiTiling_.SetQuantConvFlag(flagInfo_.quantFlag);
    conv3dApiTiling_.SetWeightType(TPosition::GM, formatMap[descInfo_.weightFormat], dtypeMap[descInfo_.weightDtype]);
    conv3dApiTiling_.SetFmapType(TPosition::GM, formatMap[descInfo_.fMapFormat], dtypeMap[descInfo_.fMapDtype]);
    conv3dApiTiling_.SetOutputType(TPosition::CO1, formatMap[descInfo_.outFormat], dtypeMap[descInfo_.outDtype]);
    if (flagInfo_.quantFlag || flagInfo_.isConv3dDequant) {
        conv3dApiTiling_.SetScaleType(TPosition::GM, formatMap[descInfo_.scaleFormat], dtypeMap[descInfo_.scaleDtype]);
    }
    conv3dApiTiling_.SetFixpipeParams(fixpipeInfo_);
    conv3dApiTiling_.SetOffsetx(static_cast<int8_t>(attrInfo_.offsetx));
    conv3dApiTiling_.SetRoundMode(static_cast<int8_t>(attrInfo_.roundMode));
    conv3dApiTiling_.SetNodeType(paramInfo_.nodeType);
    if (flagInfo_.convGroupType == ConvGroupType::OPT_GROUP_CONV) {
        // set enlarge for BasicBlock
        conv3dApiTiling_.SetOptGroupParams(static_cast<int32_t>(optGroupInfo_.enlarge), attrInfo_.groups,
                                           optGroupInfo_.groupOpt);
    }
    if (flagInfo_.hasBias) {
        conv3dApiTiling_.SetBiasType(TPosition::GM, formatMap[descInfo_.biasFormat], dtypeMap[descInfo_.biasDtype]);
    }
}

void Conv3dBaseTilingV2::NumBlocksDecision()
{
    InitApiTilingCommon();
    convBase_.UpdateFlagInfo(flagInfo_);
    if (flagInfo_.mSplitModeFlag) { // M split mode
        auto strategy = std::make_shared<conv_tiling::Conv3DBasicBlockStrategy>();
        conv_tiling::BBCommonParams params = {shapeInfo_,
                                              descInfo_,
                                              attrInfo_,
                                              convOpsConstParams_,
                                              flagInfo_,
                                              ConvAscendcFeatureFlag::IS_LOAD3D_FLAG,
                                              apiInputPlatformInfo_,
                                              opInfo_->aicoreNum,
                                              opInfo_->l1Size,
                                              opInfo_->l0cSize,
                                              oriGroupInfo_,
                                              optGroupInfo_,
                                              fixpipeInfo_,
                                              &conv3dApiTiling_,
                                              false,
                                              1};
        conv_tiling::ConvBasicBlock flow(params, conv3dBasicBlockInfo_, strategy);
        flagInfo_.mBasicBlockFlag = flow.SelectMModeAlgorithm();
        if (flagInfo_.mBasicBlockFlag) {
            flow.BasicBlock();
            numBlocksRes.batchDim = conv3dBasicBlockInfo_.batchDim;
            numBlocksRes.mDim = conv3dBasicBlockInfo_.mDim;
            numBlocksRes.nDim = conv3dBasicBlockInfo_.nDim;
            numBlocksRes.doDim = conv3dBasicBlockInfo_.doDim;
            numBlocksRes.groupDim = conv3dBasicBlockInfo_.groupDim;
            numBlocksRes.minCost = 0;
            OP_LOGD(context_->GetNodeType(),
                    "%s AscendC: get tiling from Mmode basic block algorithm, belongs to fast_tiling.",
                    paramInfo_.nodeType.c_str());
            OP_LOGD(context_->GetNodeType(),
                    "%s AscendC: batchDim / mDim / nDim / doDim / groupDim: %u, %u, %u, %u, %u.",
                    paramInfo_.nodeType.c_str(), numBlocksRes.batchDim, numBlocksRes.mDim, numBlocksRes.nDim,
                    numBlocksRes.doDim, numBlocksRes.groupDim);
            return;
        }
        numBlocksRes = convBase_.NumBlocksDecisionMsplitMode();
        OP_LOGD(paramInfo_.nodeType.c_str(),
                "AscendC: get tiling from Mmode formulaic algorithm, belongs to fast_tiling.");
        OP_LOGD(paramInfo_.nodeType.c_str(),
                "AscendC: batchDim / mDim / nDim / doDim / groupDim / minCost: %u, %u, %u, %u, %u, %lu.",
                numBlocksRes.batchDim, numBlocksRes.mDim, numBlocksRes.nDim, numBlocksRes.doDim, numBlocksRes.groupDim,
                numBlocksRes.minCost);
    } else { // HW split mode
        numBlocksRes = convBase_.NumBlocksDecisionHWsplitMode();
        OP_LOGD(paramInfo_.nodeType.c_str(),
                "AscendC: get tiling from HWmode formulaic algorithm, belongs to fast_tiling.");
        OP_LOGD(paramInfo_.nodeType.c_str(),
                "AscendC: batchDim / hoDim / nDim / doDim / groupDim / minCost: %u, %u, %u, %u, %u, %lu.",
                numBlocksRes.batchDim, numBlocksRes.hoDim, numBlocksRes.nDim, numBlocksRes.doDim, numBlocksRes.groupDim,
                numBlocksRes.minCost);
    }
}

ge::graphStatus Conv3dBaseTilingV2::GetConv3dApiTiling()
{
    Conv3dOpTilingSetShape();
    if (flagInfo_.mBasicBlockFlag) {
        OP_LOGD(context_->GetNodeType(),
                "%s AscendC: [Phase2] batchDim[%u], mDim[%u], nDim[%u], groupDim[%u], doDim[%u], mCut[%u], nCut[%u], "
                "doCut[%u], mTile[%u], nTile[%u], mIn[%u].",
                paramInfo_.nodeType.c_str(), numBlocksRes.batchDim, numBlocksRes.mDim, numBlocksRes.nDim,
                numBlocksRes.groupDim, numBlocksRes.doDim, conv3dBasicBlockInfo_.mCut, conv3dBasicBlockInfo_.nCut,
                conv3dBasicBlockInfo_.doCut, conv3dBasicBlockInfo_.mTile, conv3dBasicBlockInfo_.nTile,
                conv3dBasicBlockInfo_.mIn);
        auto apiStrategy = std::make_shared<conv_tiling::Conv3DAPIStrategy>();
        if (!conv3dApiTiling_.GetTiling(conv3dBasicBlockInfo_, tilingData_, apiStrategy)) {
            OP_LOGE(context_->GetNodeType(), "%s AscendC: get BBmode api tiling wrong.", paramInfo_.nodeType.c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        if (conv3dApiTiling_.GetTiling(tilingData_) == -1) {
            OP_LOGE(context_->GetNodeType(), "%s AscendC: get api tiling wrong.", paramInfo_.nodeType.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

void Conv3dBaseTilingV2::Conv3dOpTilingSetShape()
{
    uint64_t curCo = flagInfo_.convGroupType != ConvGroupType::NORMAL_CONV ?
                         (flagInfo_.convGroupType == ConvGroupType::ORI_GROUP_CONV ? oriGroupInfo_.coPerGroup :
                                                                                     optGroupInfo_.coutOpt) :
                         shapeInfo_.co;
    int64_t singleCoreCo = ConvAlignB(ConvCeilDiv(ConvAlignB(curCo, convOpsConstParams_.n0), numBlocksRes.nDim),
                                      convOpsConstParams_.n0);
    int64_t singleCoreDo = ConvCeilDiv(shapeInfo_.dout, numBlocksRes.doDim);
    int64_t singleCoreBatch = ConvCeilDiv(shapeInfo_.batch, numBlocksRes.batchDim);
    int64_t singleCoreHo = 0;
    int64_t singleCoreMo = 0;
    if (flagInfo_.mSplitModeFlag) {
        singleCoreMo = ConvCeilDiv(ConvAlignB(shapeInfo_.ho * shapeInfo_.wo, convOpsConstParams_.m0),
                                   numBlocksRes.mDim);
        conv3dApiTiling_.SetSingleOutputShape(singleCoreCo, singleCoreDo, singleCoreMo, singleCoreBatch);
    } else {
        singleCoreHo = ConvCeilDiv(shapeInfo_.ho, numBlocksRes.hoDim);
        conv3dApiTiling_.SetSingleOutputShape(singleCoreCo, singleCoreDo, singleCoreHo,
                                              static_cast<int64_t>(shapeInfo_.wo), singleCoreBatch);
    }

    uint64_t singleGroups = 0;
    uint64_t singleGroupOpt = 0;
    uint64_t enlarge = 0;
    if (flagInfo_.convGroupType == ConvGroupType::OPT_GROUP_CONV) {
        singleGroups = optGroupInfo_.enlarge;
        singleGroupOpt = ConvCeilDiv(optGroupInfo_.groupOpt, numBlocksRes.groupDim);
        enlarge = optGroupInfo_.enlarge;
        conv3dApiTiling_.SetOptGroupParams(static_cast<int32_t>(enlarge), static_cast<int64_t>(singleGroups),
                                           static_cast<int64_t>(singleGroupOpt));
    }
    // hf32Mode
    bool hf32TransMode = false;
    bool isHF32 = (attrInfo_.hf32Mode == 1);
    if (isHF32) {
        conv3dApiTiling_.SetHF32(isHF32, hf32TransMode);
    }

    stringstream ss;
    ss << "AscendC: api got: orgCo: " << shapeInfo_.co << ", orgkH: " << shapeInfo_.kh << ", orgkW: " << shapeInfo_.kw
       << ", orgCi: " << shapeInfo_.ci << ", orgHi: " << shapeInfo_.hi << ", orgWi: " << shapeInfo_.wi
       << ", singleCo: " << singleCoreCo << ", singleHo: " << singleCoreHo << ", singleM: " << singleCoreMo
       << ", singleGroups: " << singleGroups << ", singleGroupOpt: " << singleGroupOpt << ", enlarge: " << enlarge
       << ", hf32Enable: " << attrInfo_.hf32Mode << ", hf32TransMode: " << static_cast<uint32_t>(hf32TransMode);
    OP_LOGD(paramInfo_.nodeType.c_str(), "%s", ss.str().c_str());
}
} // namespace conv_ops_tiling
} // namespace optiling
