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
 * \file conv2d_v2_base_tiling_repo.cpp
 * \brief
 */
#include "conv2d_v2_base_tiling.h"
#include "runtime_kb_api.h"
#include <memory>


namespace optiling {
namespace conv_ops_tiling {
bool Conv2dBaseTiling::GetTilingFromRepo()
{
    OP_LOGD(context_->GetNodeName(), "Entering GetTilingFromRepo.");
    string nodeType = context_->GetNodeType();
    string tmpSocVer;
    if (nodeType == "QuantConv2D") {
        auto compileInfoPtr = static_cast<const Conv2DTilingParseInfo*>(context_->GetCompileInfo());
        OPS_CHECK_NULL_WITH_CONTEXT(context_, compileInfoPtr);
        auto quantConvOpInfoTmp = *compileInfoPtr;
        tmpSocVer = quantConvOpInfoTmp.socVersion;
    } else {
        auto compileInfoPtr = static_cast<const ConvTilingParseInfo*>(context_->GetCompileInfo());
        OPS_CHECK_NULL_WITH_CONTEXT(context_, compileInfoPtr);
        auto convOpInfoTmp = *compileInfoPtr;
        tmpSocVer = convOpInfoTmp.socVersion;
    }
    return QueryTilingBank(tmpSocVer, opInfo_->aicoreNum);
}

bool Conv2dBaseTiling::QueryTilingBank(std::string socHardWareVersion, uint32_t aicoreNum)
{
    shared_ptr<tuningtiling::TuningTilingDef> tuningTiling = nullptr;
    shared_ptr<void> inputArgs = nullptr;
    size_t inputArgsSize = 0;
    GetTilingInputArgs(inputArgs, inputArgsSize);
    uint32_t ret = Ops::NN::QueryBank(inputArgs.get(), inputArgsSize, "Conv2DV2", socHardWareVersion,
        aicoreNum, tuningTiling);
    if (ret != 0 || tuningTiling == nullptr) {
        return false;
    }
    return TranslateRepoTiling(tuningTiling);
}

void Conv2dBaseTiling::GetTilingInputArgs(shared_ptr<void> &inputArgs, size_t &size)
{
    shared_ptr<tuningtiling::Conv2DV2InputArgs> conv2DInput = std::make_shared<tuningtiling::Conv2DV2InputArgs>();

    conv2DInput->aDtype = descInfo_.fMapDtype;
    conv2DInput->bDtype = descInfo_.weightDtype;
    conv2DInput->cDtype = descInfo_.outDtype;
    conv2DInput->biasDtype = descInfo_.biasDtype;
    conv2DInput->aShapeN = shapeInfo_.batch;
    conv2DInput->aShapeH = shapeInfo_.hi;
    conv2DInput->aShapeW = shapeInfo_.wi;
    conv2DInput->bShapeN = shapeInfo_.co;
    conv2DInput->bShapeC = shapeInfo_.ci / attrInfo_.groups;
    conv2DInput->bShapeH = shapeInfo_.kh;
    conv2DInput->bShapeW = shapeInfo_.kw;
    conv2DInput->cShapeH = shapeInfo_.ho;
    conv2DInput->cShapeW = shapeInfo_.wo;
    conv2DInput->aFormat = descInfo_.fMapFormat;
    conv2DInput->bFormat = descInfo_.weightFormat;
    conv2DInput->cFormat = descInfo_.outFormat;
    conv2DInput->groups = attrInfo_.groups;
    conv2DInput->strideH = attrInfo_.strideH;
    conv2DInput->strideW = attrInfo_.strideW;
    conv2DInput->dilationH = attrInfo_.dilationH;
    conv2DInput->dilationW = attrInfo_.dilationW;
    conv2DInput->padTop = attrInfo_.padTop;
    conv2DInput->padBottom = attrInfo_.padBottom;
    conv2DInput->padLeft = attrInfo_.padLeft;
    conv2DInput->padRight = attrInfo_.padRight;
    conv2DInput->biasFlag = flagInfo_.hasBias;
    conv2DInput->hf32Flag = (attrInfo_.hf32Mode == 1);
    std::array<bool, UINT64_BIT_COUNT> bitValues{};
    std::array<uint8_t, UINT64_BYTE_COUNT> byteValues{};
    if (flagInfo_.extendConvFlag) {
        byteValues = {fixpipeInfo_.dualOutput, fixpipeInfo_.quantMode0, fixpipeInfo_.reluMode0, fixpipeInfo_.clipMode0,
                      fixpipeInfo_.quantMode1, fixpipeInfo_.reluMode1, fixpipeInfo_.clipMode1};
    }
    convBase_.SetBitsFromBool(conv2DInput->reserverdParam1, bitValues);
    convBase_.SetBytesFromUint8(conv2DInput->reserverdParam2, byteValues);
    convBase_.SetBytesFromUint32(conv2DInput->reserverdParam3, 0, 0);
    conv2DInput->reserverdParam4 = 0;
    conv2DInput->reserverdParam5 = 0;
    conv2DInput->reserverdParam6 = 0;
    inputArgs = conv2DInput;
    size = sizeof(tuningtiling::Conv2DV2InputArgs);
    PrintInputArgs(conv2DInput);
}

bool Conv2dBaseTiling::TranslateRepoTiling(tuningtiling::TuningTilingDefPtr &tuningTiling)
{
    auto convRepoTiling = static_pointer_cast<tuningtiling::Conv2DV2TunnerTiling>(tuningTiling);
    if (convRepoTiling == nullptr) {
        return false;
    }

    TranslateApiTiling(convRepoTiling);
    TranslateRunInfo(convRepoTiling);
    flagInfo_.enableC04Flag = convRepoTiling->isC04Flag;
    flagInfo_.mSplitModeFlag = static_cast<bool>(convRepoTiling->mMode);
    TranslateApiTilingAux(convRepoTiling);

    return true;
}

void Conv2dBaseTiling::TranslateApiTiling(shared_ptr<tuningtiling::Conv2DV2TunnerTiling> convRepoTiling)
{
    tilingData_.convApiTiling.set_groups(convRepoTiling->groups);
    tilingData_.convApiTiling.set_orgCi(convRepoTiling->orgCi);
    tilingData_.convApiTiling.set_orgHi(convRepoTiling->orgHi);
    tilingData_.convApiTiling.set_orgWi(convRepoTiling->orgWi);
    tilingData_.convApiTiling.set_orgCo(convRepoTiling->orgCo);
    tilingData_.convApiTiling.set_orgHo(convRepoTiling->orgHo);
    tilingData_.convApiTiling.set_orgWo(convRepoTiling->orgWo);
    tilingData_.convApiTiling.set_kernelH(convRepoTiling->kernelH);
    tilingData_.convApiTiling.set_kernelW(convRepoTiling->kernelW);
    tilingData_.convApiTiling.set_singleCoreCi(convRepoTiling->singleCoreCi);
    tilingData_.convApiTiling.set_singleCoreCo(convRepoTiling->singleCoreCo);
    tilingData_.convApiTiling.set_singleCoreHo(convRepoTiling->singleCoreHo);
    tilingData_.convApiTiling.set_singleCoreWo(convRepoTiling->singleCoreWo);
    tilingData_.convApiTiling.set_singleCoreBatch(ConvCeilDiv(shapeInfo_.batch, convRepoTiling->batchDim));
    tilingData_.convApiTiling.set_hoL1(convRepoTiling->hoL1);
    tilingData_.convApiTiling.set_woL1(convRepoTiling->woL1);
    tilingData_.convApiTiling.set_kAL1(convRepoTiling->kAL1);
    tilingData_.convApiTiling.set_kBL1(convRepoTiling->kBL1);
    tilingData_.convApiTiling.set_nBL1(convRepoTiling->nBL1);
    tilingData_.convApiTiling.set_hoL0(convRepoTiling->hoL0);
    tilingData_.convApiTiling.set_woL0(convRepoTiling->woL0);
    tilingData_.convApiTiling.set_kL0(convRepoTiling->kL0);
    tilingData_.convApiTiling.set_nL0(convRepoTiling->nL0);
    tilingData_.convApiTiling.set_pBufferFlag(convRepoTiling->pBufferFlag);
    tilingData_.convApiTiling.set_strideH(convRepoTiling->strideH);
    tilingData_.convApiTiling.set_strideW(convRepoTiling->strideW);
    tilingData_.convApiTiling.set_dilationH(convRepoTiling->dilationH);
    tilingData_.convApiTiling.set_dilationW(convRepoTiling->dilationW);
    tilingData_.convApiTiling.set_padTop(convRepoTiling->padTop);
    tilingData_.convApiTiling.set_padBottom(convRepoTiling->padBottom);
    tilingData_.convApiTiling.set_padLeft(convRepoTiling->padLeft);
    tilingData_.convApiTiling.set_padRight(convRepoTiling->padRight);
    tilingData_.convApiTiling.set_iterateMNOrder(convRepoTiling->iterateMNOrder);
    tilingData_.convApiTiling.set_biasFullLoadFlag(convRepoTiling->biasFullLoadFlag);
    tilingData_.convApiTiling.set_fixpParamsFullLoadFlag(convRepoTiling->fixpParamsFullLoadFlag);
    tilingData_.convApiTiling.set_offsetx(attrInfo_.offsetx);
    tilingData_.convApiTiling.set_hf32Enable(convRepoTiling->hf32Enable);
    tilingData_.convApiTiling.set_hf32TransMode(convRepoTiling->hf32TransMode);
    tilingData_.convApiTiling.set_roundMode(attrInfo_.roundMode);
    if (convRepoTiling->enlarge > 0) {
        tilingData_.convApiTiling.set_enlarge(convRepoTiling->enlarge);
    } else {
        tilingData_.convApiTiling.set_enlarge(optGroupInfo_.enlarge);
    }
    tilingData_.convApiTiling.set_innerBatch(convRepoTiling->innerBatch);
}

void Conv2dBaseTiling::TranslateApiTilingAux(shared_ptr<tuningtiling::Conv2DV2TunnerTiling> convRepoTiling)
{
    uint64_t kernelHW = convRepoTiling->kernelH * convRepoTiling->kernelW;
    uint64_t khkwL1 = convRepoTiling->khL1 * convRepoTiling->kwL1;
    uint64_t kernelValueInKSize = featureFlagInfo_ == ConvAscendcFeatureFlag::IS_DMA_FLAG ? khkwL1 : kernelHW;
    uint64_t cinAInCore = convRepoTiling->kAL1 / kernelValueInKSize;
    uint64_t orgHiWi = convRepoTiling->orgHi * convRepoTiling->orgWi;
    uint64_t mStep = 0;
    if (GetOutputOrderVal() == 0) {
        mStep = ConvAlignB(convRepoTiling->hoL0 * convRepoTiling->woL0, convOpsConstParams_.m0);
    } else {
        mStep = ConvAlignB(convRepoTiling->hoL0, convOpsConstParams_.m0);
    }

    uint64_t fmapKStride = mStep / convOpsConstParams_.m0;
    uint64_t weightKStride = ConvCeilDiv(convRepoTiling->multiNBL1 * convRepoTiling->nL0, convOpsConstParams_.n0);
    uint64_t cinBInCore = convRepoTiling->kBL1 / kernelValueInKSize;
    uint64_t kATail = (convRepoTiling->singleCoreCi * kernelValueInKSize) % convRepoTiling->kAL1;
    kATail = kATail == 0 ? convRepoTiling->kAL1 : kATail;
    uint64_t kBTail = (convRepoTiling->singleCoreCi * kernelValueInKSize) % convRepoTiling->kBL1;
    kBTail = kBTail == 0 ? convRepoTiling->kBL1 : kBTail;

    if (featureFlagInfo_ == ConvAscendcFeatureFlag::IS_DMA_FLAG) {
        tilingData_.convApiTiling.set_khL1(convRepoTiling->khL1);
        tilingData_.convApiTiling.set_kwL1(convRepoTiling->kwL1);
    } else {
        tilingData_.convApiTiling.set_khL1(0);
        tilingData_.convApiTiling.set_kwL1(0);
    }
    tilingData_.convApiTiling.set_kernelHxkernelW(kernelHW);
    tilingData_.convApiTiling.set_kernelHxkernelWxkernelD(kernelHW);
    tilingData_.convApiTiling.set_multiNBL1(ConvCeilDiv(convRepoTiling->nBL1, convRepoTiling->nL0));
    tilingData_.convApiTiling.set_cinAInCore(cinAInCore);
    tilingData_.convApiTiling.set_cinATailInCore(kATail / kernelValueInKSize);
    tilingData_.convApiTiling.set_orgHixWi(orgHiWi);
    tilingData_.convApiTiling.set_cinOffsetBlockInGM(orgHiWi * cinAInCore);
    tilingData_.convApiTiling.set_mStep(mStep);
    tilingData_.convApiTiling.set_fmapKStride(fmapKStride);
    tilingData_.convApiTiling.set_nStep(ConvCeilDiv(convRepoTiling->nL0, convOpsConstParams_.n0));
    tilingData_.convApiTiling.set_kStep(convRepoTiling->kL0 / convOpsConstParams_.k0);
    tilingData_.convApiTiling.set_weightKStride(weightKStride);
    tilingData_.convApiTiling.set_coutOffsetBlock((convRepoTiling->orgCi / convRepoTiling->groups) * kernelHW);
    tilingData_.convApiTiling.set_cinBInCore(cinBInCore);
    tilingData_.convApiTiling.set_cinBTailInCore(kBTail / kernelValueInKSize);
    tilingData_.convApiTiling.set_nL1DivBlockSize(convRepoTiling->nBL1 / convOpsConstParams_.n0);
    tilingData_.convApiTiling.set_aL1SpaceSize(CalcAL1SpaceSize(convRepoTiling));
    tilingData_.convApiTiling.set_hasBias(static_cast<uint8_t>(flagInfo_.hasBias));
    tilingData_.convApiTiling.set_hasScale(static_cast<uint8_t>(flagInfo_.quantFlag || flagInfo_.extendConvFlag));
    uint64_t singleGroups = 0;
    uint64_t singleGroupOpt = 0;
    if (flagInfo_.convGroupType == ConvGroupType::OPT_GROUP_CONV) {
        singleGroups = convRepoTiling->enlarge > 0 ? convRepoTiling->enlarge : optGroupInfo_.enlarge;
        singleGroupOpt = convRepoTiling->enlarge > 0 ?
            ConvCeilDiv(ConvCeilDiv(convRepoTiling->groups, convRepoTiling->enlarge), convRepoTiling->groupDim) :
            ConvCeilDiv(optGroupInfo_.groupOpt, convRepoTiling->groupDim);
    }

    tilingData_.convApiTiling.set_singleCoreGroups(singleGroups);
    tilingData_.convApiTiling.set_singleCoreGroupOpt(singleGroupOpt);

    SetUbTiling(convRepoTiling);
    SetFixpipeTiling();
    SetUnionDataXt(convRepoTiling);
}

void Conv2dBaseTiling::SetUbTiling(shared_ptr<tuningtiling::Conv2DV2TunnerTiling> convRepoTiling)
{
    if (convRepoTiling->isC04Flag != 0) {
        c04Info_.curNBL1 = convRepoTiling->nBL1;
        c04Info_.orgkH = convRepoTiling->kernelH;
        c04Info_.orgkW = convRepoTiling->kernelW;
        c04Info_.pbBL1 = static_cast<uint8_t>((convRepoTiling->pBufferFlag >> WEIGHT_L1_PB_OFFSET) & 0x1);
        c04Info_.n0 = convOpsConstParams_.n0;
        c04Info_.k0 = convOpsConstParams_.k0;
        c04Info_.weightDtype = dtypeMap.at(descInfo_.weightDtype);
        tilingData_.convApiTiling.set_bUbNStep(static_cast<uint32_t>(conv2dApiTiling_.CalcC04UbLoadNsize(c04Info_)));
        tilingData_.convApiTiling.set_bUbKStep(0);
    } else if (convRepoTiling->isWeightUbTransFlag != 0) {
        conv_tiling::ConvWeightUbTransParams params = {convRepoTiling->nBL1, convRepoTiling->kBL1,
            static_cast<uint64_t>(convRepoTiling->kernelH), static_cast<uint64_t>(convRepoTiling->kernelW),
            convOpsConstParams_.k0, convOpsConstParams_.n0, dtypeMap.at(descInfo_.weightDtype)};
        conv2dApiTiling_.GetWeightUBTiling(params);
        tilingData_.convApiTiling.set_bUbNStep(params.bUbNStep);
        tilingData_.convApiTiling.set_bUbKStep(params.bUbKStep);
    } else {
        tilingData_.convApiTiling.set_bUbNStep(0);
        tilingData_.convApiTiling.set_bUbKStep(0);
    }

    if (featureFlagInfo_ == ConvAscendcFeatureFlag::IS_DMA_FLAG) {
        conv_tiling::ConvDmaParams params = {convRepoTiling->hoL1, convRepoTiling->woL1,
            static_cast<uint64_t>(convRepoTiling->khL1), static_cast<uint64_t>(convRepoTiling->kwL1),
            convOpsConstParams_.k0, dtypeMap.at(descInfo_.fMapDtype)};
        conv2dApiTiling_.GetDmaUbTiling(params);
        tilingData_.convApiTiling.set_khUb(params.khUb);
        tilingData_.convApiTiling.set_kwUb(params.kwUb);
    } else {
        tilingData_.convApiTiling.set_khUb(0);
        tilingData_.convApiTiling.set_kwUb(0);
    }
}

void Conv2dBaseTiling::SetFixpipeTiling()
{
    tilingData_.convApiTiling.set_dualOutput(fixpipeInfo_.dualOutput);
    tilingData_.convApiTiling.set_quantMode0(fixpipeInfo_.quantMode0);
    tilingData_.convApiTiling.set_reluMode0(fixpipeInfo_.reluMode0);
    tilingData_.convApiTiling.set_clipMode0(fixpipeInfo_.clipMode0);
    tilingData_.convApiTiling.set_quantMode1(fixpipeInfo_.quantMode1);
    tilingData_.convApiTiling.set_reluMode1(fixpipeInfo_.reluMode1);
    tilingData_.convApiTiling.set_clipMode1(fixpipeInfo_.clipMode1);
}

void Conv2dBaseTiling::TranslateRunInfo(shared_ptr<tuningtiling::Conv2DV2TunnerTiling> convRepoTiling)
{
    tilingData_.convRunInfo.set_batch(shapeInfo_.batch);
    tilingData_.convRunInfo.set_hin(convRepoTiling->orgHi);
    tilingData_.convRunInfo.set_win(convRepoTiling->orgWi);
    tilingData_.convRunInfo.set_batchDim(convRepoTiling->batchDim);
    tilingData_.convRunInfo.set_hoDim(convRepoTiling->hoDim);
    tilingData_.convRunInfo.set_woDim(featureFlagInfo_ == ConvAscendcFeatureFlag::IS_CONV1D_FLAG ?
            convRepoTiling->woDim : 1);
    tilingData_.convRunInfo.set_nDim(convRepoTiling->nDim);
    tilingData_.convRunInfo.set_cin(convRepoTiling->orgCi);
    tilingData_.convRunInfo.set_cout(convRepoTiling->orgCo);
    tilingData_.convRunInfo.set_kh(convRepoTiling->kernelH);
    tilingData_.convRunInfo.set_kw(convRepoTiling->kernelW);
    tilingData_.convRunInfo.set_hout(convRepoTiling->orgHo);
    tilingData_.convRunInfo.set_wout(convRepoTiling->orgWo);
    tilingData_.convRunInfo.set_strideH(convRepoTiling->strideH);
    tilingData_.convRunInfo.set_strideW(convRepoTiling->strideW);
    tilingData_.convRunInfo.set_dilationH(convRepoTiling->dilationH);
    tilingData_.convRunInfo.set_dilationW(convRepoTiling->dilationW);
    tilingData_.convRunInfo.set_padTop(convRepoTiling->padTop);
    tilingData_.convRunInfo.set_padLeft(convRepoTiling->padLeft);
    tilingData_.convRunInfo.set_hasBias(flagInfo_.hasBias);
    tilingData_.convRunInfo.set_groups(convRepoTiling->groups);
    if (convRepoTiling->enlarge > 0) {
        tilingData_.convRunInfo.set_cinOpt(oriGroupInfo_.ciPerGroup * convRepoTiling->enlarge);
        tilingData_.convRunInfo.set_coutOpt(oriGroupInfo_.coPerGroup * convRepoTiling->enlarge);
        tilingData_.convRunInfo.set_groupOpt(ConvCeilDiv(convRepoTiling->groups, convRepoTiling->enlarge));
        tilingData_.convRunInfo.set_enlarge(convRepoTiling->enlarge);
    } else {
        tilingData_.convRunInfo.set_cinOpt(optGroupInfo_.cinOpt);
        tilingData_.convRunInfo.set_coutOpt(optGroupInfo_.coutOpt);
        tilingData_.convRunInfo.set_groupOpt(optGroupInfo_.groupOpt);
        tilingData_.convRunInfo.set_enlarge(optGroupInfo_.enlarge);
    }
    tilingData_.convRunInfo.set_groupDim(convRepoTiling->groupDim);
}

uint32_t Conv2dBaseTiling::CalcAL1SpaceSize(shared_ptr<tuningtiling::Conv2DV2TunnerTiling> convRepoTiling)
{
    uint64_t aL1SpaceSize = 0;
    uint64_t fmapSize = dtypeSizeTab.at(descInfo_.fMapDtype);

    if (featureFlagInfo_ == ConvAscendcFeatureFlag::IS_DMA_FLAG) {
        aL1SpaceSize = convRepoTiling->hoL1 * ConvAlignB(convRepoTiling->woL1, convOpsConstParams_.m0) *
            convRepoTiling->kAL1 * fmapSize;
        return static_cast<uint32_t>(aL1SpaceSize);
    }

    if (convRepoTiling->mMode == 1) {
        if (static_cast<bool>(convRepoTiling->isC04Flag) && convRepoTiling->innerBatch > 1) {
            aL1SpaceSize = C04_CIN_SIZE * convRepoTiling->orgHi * convRepoTiling->orgWi;
        } else {
            uint64_t mL1Max = convRepoTiling->hoL1 < convRepoTiling->singleCoreHo ? convRepoTiling->hoL1 : convRepoTiling->singleCoreHo;
            uint64_t hoL1Max = std::min(mL1Max / convRepoTiling->orgWo + 2, convRepoTiling->orgHo);
            uint64_t hiAL1Max = ConvInferHiL1(hoL1Max, convRepoTiling->orgHi, convRepoTiling->kernelH, convRepoTiling->dilationH,
                convRepoTiling->strideH);
            aL1SpaceSize = tilingData_.convApiTiling.get_cinAInCore() * hiAL1Max * convRepoTiling->orgWi;
        }
    } else {
        uint64_t hiAL1Max = ConvInferHiL1(convRepoTiling->hoL1, convRepoTiling->orgHi, convRepoTiling->kernelH, convRepoTiling->dilationH,
            convRepoTiling->strideH);
        uint64_t wiAL1Max = 0;
        if (static_cast<bool>(convRepoTiling->isC04Flag) && convRepoTiling->orgWo == convRepoTiling->woL1) {
            wiAL1Max = convRepoTiling->orgWi;
            aL1SpaceSize = ConvAlignB(hiAL1Max * wiAL1Max, C0_SIZE / (fmapSize * C04_CIN_SIZE)) * C04_CIN_SIZE;
        } else {
            wiAL1Max = ConvInferWiL1(convRepoTiling->woL1, convRepoTiling->orgWi, convRepoTiling->kernelW, convRepoTiling->dilationW,
                convRepoTiling->strideW);
            aL1SpaceSize = tilingData_.convApiTiling.get_cinAInCore() * hiAL1Max * wiAL1Max;
        }
    }
    aL1SpaceSize = ConvAlignB(aL1SpaceSize * fmapSize , C0_SIZE) * convRepoTiling->innerBatch;

    return static_cast<uint32_t>(aL1SpaceSize);
}

void Conv2dBaseTiling::SetUnionDataXt(shared_ptr<tuningtiling::Conv2DV2TunnerTiling> convRepoTiling)
{
    conv_tiling::UnionDataXt unionDataXt;
    unionDataXt.bf.kernelW = static_cast<uint64_t>(convRepoTiling->kernelW);
    unionDataXt.bf.kernelW_highest_bit = (static_cast<uint64_t>(convRepoTiling->kernelW) & 0x100) >> 8;
    unionDataXt.bf.kernelH = static_cast<uint64_t>(convRepoTiling->kernelH);
    unionDataXt.bf.kernelH_highest_bit = (static_cast<uint64_t>(convRepoTiling->kernelH) & 0x100) >> 8;
    unionDataXt.bf.dilationH = static_cast<uint64_t>(convRepoTiling->dilationH);
    unionDataXt.bf.dilationW = static_cast<uint64_t>(convRepoTiling->dilationW);
    unionDataXt.bf.strideH = static_cast<uint64_t>(convRepoTiling->strideH) & 0x3f;
    unionDataXt.bf.strideW = static_cast<uint64_t>(convRepoTiling->strideW) & 0x3f;
    tilingData_.convApiTiling.set_unionDataXt(unionDataXt.n);
}

void Conv2dBaseTiling::PrintInputArgs(shared_ptr<tuningtiling::Conv2DV2InputArgs> conv2DInput)
{
    stringstream ss;
    ss << paramInfo_.nodeType.c_str() << "ops tiling input args: aDtype: " << conv2DInput->aDtype
    << ", bDtype: " << conv2DInput->bDtype
    << ", cDtype: " << conv2DInput->cDtype
    << ", biasDtype: " << conv2DInput->biasDtype
    << ", aShapeN: " << conv2DInput->aShapeN
    << ", aShapeH: " << conv2DInput->aShapeH
    << ", aShapeW: " << conv2DInput->aShapeW
    << ", bShapeN: " << conv2DInput->bShapeN
    << ", bShapeC: " << conv2DInput->bShapeC
    << ", bShapeH: " << conv2DInput->bShapeH
    << ", bShapeW: " << conv2DInput->bShapeW
    << ", cShapeH: " << conv2DInput->cShapeH
    << ", cShapeW: " << conv2DInput->cShapeW
    << ", aFormat: " << conv2DInput->aFormat
    << ", bFormat: " << conv2DInput->bFormat
    << ", cFormat: " << conv2DInput->cFormat
    << ", groups: " << conv2DInput->groups
    << ", strideH: " << conv2DInput->strideH
    << ", strideW: " << conv2DInput->strideW
    << ", dilationH: " << conv2DInput->dilationH
    << ", dilationW: " << conv2DInput->dilationW
    << ", padTop: " << conv2DInput->padTop
    << ", padBottom: " << conv2DInput->padBottom
    << ", padLeft: " << conv2DInput->padLeft
    << ", padRight: " << conv2DInput->padRight
    << ", biasFlag: " << conv2DInput->biasFlag
    << ", hf32Flag: " << conv2DInput->hf32Flag
    << ", reserverdParam1: " << conv2DInput->reserverdParam1
    << ", reserverdParam2: " << conv2DInput->reserverdParam2
    << ", reserverdParam3: " << conv2DInput->reserverdParam3
    << ", reserverdParam4: " << conv2DInput->reserverdParam4
    << ", reserverdParam5: " << conv2DInput->reserverdParam5
    << ", reserverdParam6: " << conv2DInput->reserverdParam6;
    OP_LOGD(context_->GetNodeName(), "%s", ss.str().c_str());
}

}
}