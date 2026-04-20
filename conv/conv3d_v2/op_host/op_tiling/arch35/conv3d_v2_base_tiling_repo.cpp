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
 * \file conv3d_V2_base_tiling_repo.cpp
 * \brief
 */
#include "conv3d_v2_base_tiling.h"
#include "runtime_kb_api.h"
#include <memory>

namespace optiling {
namespace conv_ops_tiling {
bool Conv3dBaseTilingV2::GetTilingFromRepo()
{
    std::shared_ptr<tuningtiling::TuningTilingDef> tuningTiling = nullptr;
    std::shared_ptr<void> inputArgs = nullptr;
    std::size_t inputArgsSize = 0;
    GetTilingInputArgs(inputArgs, inputArgsSize);
    auto compileInfoPtr = static_cast<const ConvTilingParseInfo*>(context_->GetCompileInfo());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, compileInfoPtr);
    uint32_t ret = Ops::NN::QueryBank(inputArgs.get(), inputArgsSize, "Conv3DV2", compileInfoPtr->socVersion,
        opInfo_->aicoreNum, tuningTiling);
    if (ret != 0 || tuningTiling == nullptr) {
        return false;
    }
    return TranslateRepoTiling(tuningTiling);
}

void Conv3dBaseTilingV2::GetTilingInputArgs(std::shared_ptr<void> &inputArgs, size_t &size)
{
    std::shared_ptr<tuningtiling::Conv3DV2InputArgs> conv3DInput =
                std::make_shared<tuningtiling::Conv3DV2InputArgs>();

    conv3DInput->aDtype = descInfo_.fMapDtype;
    conv3DInput->bDtype = descInfo_.weightDtype;
    conv3DInput->cDtype = descInfo_.outDtype;
    conv3DInput->biasDtype = descInfo_.biasDtype;
    conv3DInput->aShapeN = shapeInfo_.batch;
    conv3DInput->aShapeD = shapeInfo_.di;
    conv3DInput->aShapeH = shapeInfo_.hi;
    conv3DInput->aShapeW = shapeInfo_.wi;
    conv3DInput->bShapeN = shapeInfo_.co;
    conv3DInput->bShapeC = shapeInfo_.ci / attrInfo_.groups;
    conv3DInput->bShapeD = shapeInfo_.kd;
    conv3DInput->bShapeH = shapeInfo_.kh;
    conv3DInput->bShapeW = shapeInfo_.kw;
    conv3DInput->cShapeD = shapeInfo_.dout;
    conv3DInput->cShapeH = shapeInfo_.ho;
    conv3DInput->cShapeW = shapeInfo_.wo;
    conv3DInput->aFormat = descInfo_.fMapFormat;
    conv3DInput->bFormat = descInfo_.weightFormat;
    conv3DInput->cFormat = descInfo_.outFormat;
    conv3DInput->groups = attrInfo_.groups;
    conv3DInput->strideD = attrInfo_.strideD;
    conv3DInput->strideH = attrInfo_.strideH;
    conv3DInput->strideW = attrInfo_.strideW;
    conv3DInput->dilationD = attrInfo_.dilationD;
    conv3DInput->dilationH = attrInfo_.dilationH;
    conv3DInput->dilationW = attrInfo_.dilationW;
    conv3DInput->padHead = attrInfo_.padHead;
    conv3DInput->padTail = attrInfo_.padTail;
    conv3DInput->padTop = attrInfo_.padTop;
    conv3DInput->padBottom = attrInfo_.padBottom;
    conv3DInput->padLeft = attrInfo_.padLeft;
    conv3DInput->padRight = attrInfo_.padRight;
    conv3DInput->biasFlag = flagInfo_.hasBias;
    std::array<bool, UINT64_BIT_COUNT> bitValues{};
    std::array<uint8_t, UINT64_BYTE_COUNT> byteValues{};
    if (attrInfo_.hf32Mode == 1) {
        bitValues = {true};
    }
    convBase_.SetBitsFromBool(conv3DInput->reserverdParam1, bitValues);
    convBase_.SetBytesFromUint8(conv3DInput->reserverdParam2, byteValues);
    convBase_.SetBytesFromUint32(conv3DInput->reserverdParam3, 0, 0);
    conv3DInput->reserverdParam4 = 0;
    conv3DInput->reserverdParam5 = 0;
    conv3DInput->reserverdParam6 = 0;

    inputArgs = conv3DInput;
    size = sizeof(tuningtiling::Conv3DV2InputArgs);
    PrintInputArgs(conv3DInput);
}

bool Conv3dBaseTilingV2::TranslateRepoTiling(tuningtiling::TuningTilingDefPtr &tuningTiling)
{
    auto convRepoTiling = std::static_pointer_cast<tuningtiling::Conv3DV2TunnerTiling>(tuningTiling);
    if (convRepoTiling == nullptr) {
        return false;
    }

    TranslateApiTiling(convRepoTiling);
    TranslateRunInfo(convRepoTiling);
    flagInfo_.mSplitModeFlag = static_cast<bool>(convRepoTiling->mMode);
    TranslateApiTilingAux(convRepoTiling);

    return true;
}

void Conv3dBaseTilingV2::TranslateApiTiling(std::shared_ptr<tuningtiling::Conv3DV2TunnerTiling> convRepoTiling)
{
    tilingData_.convApiTiling.groups = convRepoTiling->groups;
    tilingData_.convApiTiling.orgCi = convRepoTiling->orgCi;
    tilingData_.convApiTiling.orgDi = convRepoTiling->orgDi;
    tilingData_.convApiTiling.orgHi = convRepoTiling->orgHi;
    tilingData_.convApiTiling.orgWi = convRepoTiling->orgWi;
    tilingData_.convApiTiling.orgDo = convRepoTiling->orgDo;
    tilingData_.convApiTiling.orgCo = convRepoTiling->orgCo;
    tilingData_.convApiTiling.orgHo = convRepoTiling->orgHo;
    tilingData_.convApiTiling.orgWo = convRepoTiling->orgWo;
    tilingData_.convApiTiling.kernelD = convRepoTiling->kernelD;
    tilingData_.convApiTiling.kernelH = convRepoTiling->kernelH;
    tilingData_.convApiTiling.kernelW = convRepoTiling->kernelW;
    tilingData_.convApiTiling.singleCoreBatch = ConvCeilDiv(shapeInfo_.batch, convRepoTiling->batchDim);
    tilingData_.convApiTiling.singleCoreDo = convRepoTiling->singleCoreDo;
    tilingData_.convApiTiling.singleCoreCo = convRepoTiling->singleCoreCo;
    tilingData_.convApiTiling.singleCoreHo = convRepoTiling->singleCoreHo;
    tilingData_.convApiTiling.singleCoreCi = convRepoTiling->singleCoreCi;
    tilingData_.convApiTiling.singleCoreWo = convRepoTiling->singleCoreWo;
    tilingData_.convApiTiling.hoL0 = convRepoTiling->hoL0;
    tilingData_.convApiTiling.kL0 = convRepoTiling->kL0;
    tilingData_.convApiTiling.nL0 = convRepoTiling->nL0;
    tilingData_.convApiTiling.woL0 = convRepoTiling->woL0;
    tilingData_.convApiTiling.kAL1 = convRepoTiling->kAL1;
    tilingData_.convApiTiling.kBL1 = convRepoTiling->kBL1;
    tilingData_.convApiTiling.nBL1 = convRepoTiling->nBL1;
    tilingData_.convApiTiling.hoL1 = convRepoTiling->hoL1;
    tilingData_.convApiTiling.woL1 = convRepoTiling->woL1;
    tilingData_.convApiTiling.strideD = convRepoTiling->strideD;
    tilingData_.convApiTiling.strideH = convRepoTiling->strideH;
    tilingData_.convApiTiling.strideW = convRepoTiling->strideW;
    tilingData_.convApiTiling.dilationD = convRepoTiling->dilationD;
    tilingData_.convApiTiling.dilationH = convRepoTiling->dilationH;
    tilingData_.convApiTiling.dilationW = convRepoTiling->dilationW;
    tilingData_.convApiTiling.padHead = convRepoTiling->padHead;
    tilingData_.convApiTiling.padTail = convRepoTiling->padTail;
    tilingData_.convApiTiling.padTop = convRepoTiling->padTop;
    tilingData_.convApiTiling.padBottom = convRepoTiling->padBottom;
    tilingData_.convApiTiling.padLeft = convRepoTiling->padLeft;
    tilingData_.convApiTiling.padRight = convRepoTiling->padRight;
    tilingData_.convApiTiling.pBufferFlag = convRepoTiling->pBufferFlag;
    tilingData_.convApiTiling.iterateMNOrder = convRepoTiling->iterateMNOrder;
    tilingData_.convApiTiling.biasFullLoadFlag = convRepoTiling->biasFullLoadFlag;
    tilingData_.convApiTiling.fixpParamsFullLoadFlag = convRepoTiling->fixpParamsFullLoadFlag;
    tilingData_.convApiTiling.enlarge = optGroupInfo_.enlarge;
}

void Conv3dBaseTilingV2::TranslateRunInfo(std::shared_ptr<tuningtiling::Conv3DV2TunnerTiling> convRepoTiling)
{
    tilingData_.convRunInfo.batch = shapeInfo_.batch;
    tilingData_.convRunInfo.cin = convRepoTiling->orgCi;
    tilingData_.convRunInfo.din = convRepoTiling->orgDi;
    tilingData_.convRunInfo.hin = convRepoTiling->orgHi;
    tilingData_.convRunInfo.win = convRepoTiling->orgWi;
    tilingData_.convRunInfo.cout = convRepoTiling->orgCo;
    tilingData_.convRunInfo.kd = convRepoTiling->kernelD;
    tilingData_.convRunInfo.kh = convRepoTiling->kernelH;
    tilingData_.convRunInfo.kw = convRepoTiling->kernelW;
    tilingData_.convRunInfo.dout = convRepoTiling->orgDo;
    tilingData_.convRunInfo.hout = convRepoTiling->orgHo;
    tilingData_.convRunInfo.wout = convRepoTiling->orgWo;
    tilingData_.convRunInfo.batchDim = convRepoTiling->batchDim;
    tilingData_.convRunInfo.hoDim = convRepoTiling->hoDim;
    tilingData_.convRunInfo.nDim = convRepoTiling->nDim;
    tilingData_.convRunInfo.doDim = convRepoTiling->doDim;
    tilingData_.convRunInfo.groupDim = convRepoTiling->groupDim;
    tilingData_.convRunInfo.strideH = convRepoTiling->strideH;
    tilingData_.convRunInfo.strideD = convRepoTiling->strideD;
    tilingData_.convRunInfo.dilationH = convRepoTiling->dilationH;
    tilingData_.convRunInfo.dilationD = convRepoTiling->dilationD;
    tilingData_.convRunInfo.padHead= convRepoTiling->padHead;
    tilingData_.convRunInfo.padTop =convRepoTiling->padTop;
    tilingData_.convRunInfo.hasBias = flagInfo_.hasBias;
    tilingData_.convRunInfo.groups = convRepoTiling->groups;
    tilingData_.convRunInfo.cinOpt = optGroupInfo_.cinOpt;
    tilingData_.convRunInfo.coutOpt = optGroupInfo_.coutOpt;
    tilingData_.convRunInfo.groupOpt = optGroupInfo_.groupOpt;
    tilingData_.convRunInfo.enlarge = optGroupInfo_.enlarge;
}

uint32_t Conv3dBaseTilingV2::CalcAL1SpaceSize(shared_ptr<tuningtiling::Conv3DV2TunnerTiling> convRepoTiling) {
    uint64_t aL1SpaceSize = 0;
    uint64_t fmapSize = dtypeSizeTab.at(descInfo_.fMapDtype);

    uint64_t dilatedKernelH = (convRepoTiling->kernelH - 1) * convRepoTiling->dilationH + 1;
    if (convRepoTiling->mMode == 1) {
        uint64_t mL1Max = convRepoTiling->hoL1 < convRepoTiling->singleCoreHo ? convRepoTiling->hoL1 : convRepoTiling->singleCoreHo;
        uint64_t hoL1Max = std::min(mL1Max / convRepoTiling->orgWo + CONST_VALUE_2,
            convRepoTiling->orgHo);
        uint64_t hiAL1Max = (hoL1Max - 1) * convRepoTiling->strideH + dilatedKernelH;
        hiAL1Max = hiAL1Max > convRepoTiling->orgHi ? convRepoTiling->orgHi : hiAL1Max;

        aL1SpaceSize = tilingData_.convApiTiling.cinAInCore * hiAL1Max * convRepoTiling->orgWi;
    } else {
        uint64_t hiAL1Max = (convRepoTiling->hoL1 - 1) * convRepoTiling->strideH + dilatedKernelH;
        hiAL1Max = hiAL1Max > convRepoTiling->orgHi ? convRepoTiling->orgHi : hiAL1Max;

        uint64_t wiAL1Max = 0;
        if ((convRepoTiling->isC04Flag == 1) && convRepoTiling->singleCoreWo == convRepoTiling->woL1) {
            wiAL1Max = convRepoTiling->orgWi;

            aL1SpaceSize = ConvAlignB(hiAL1Max * wiAL1Max, C0_SIZE /
                (fmapSize * C04_CIN_SIZE)) * C04_CIN_SIZE;
        } else {
            uint64_t dilatedKernelW = (convRepoTiling->kernelW - 1) * convRepoTiling->dilationW + 1;
            wiAL1Max = (convRepoTiling->woL1 - 1) * convRepoTiling->strideW + dilatedKernelW;
            wiAL1Max = wiAL1Max > convRepoTiling->orgWi ? convRepoTiling->orgWi : wiAL1Max;

            aL1SpaceSize = tilingData_.convApiTiling.cinAInCore * hiAL1Max * wiAL1Max;
        }
    }
    aL1SpaceSize = ConvAlignB(aL1SpaceSize * fmapSize, C0_SIZE);

    return static_cast<uint32_t>(aL1SpaceSize);
}

void Conv3dBaseTilingV2::TranslateApiTilingAux(shared_ptr<tuningtiling::Conv3DV2TunnerTiling> convRepoTiling) {
    uint32_t kernelHxkernelW = convRepoTiling->kernelH * convRepoTiling->kernelW;
    uint32_t cinAInCore = convRepoTiling->kAL1 / kernelHxkernelW;
    uint32_t kAL1Tail =
        (ConvAlignB(convRepoTiling->singleCoreCi, convOpsConstParams_.k0) * kernelHxkernelW * convRepoTiling->kernelD) %
        convRepoTiling->kAL1;
    kAL1Tail = kAL1Tail == 0 ? convRepoTiling->kAL1 : kAL1Tail;
    uint32_t kBL1Tail = (convRepoTiling->singleCoreCi * kernelHxkernelW) % convRepoTiling->kBL1;
    kBL1Tail = kBL1Tail == 0 ? convRepoTiling->kBL1 : kBL1Tail;
    uint32_t cinATailInCore = kAL1Tail / kernelHxkernelW;
    uint32_t cinBTailInCore = kBL1Tail / kernelHxkernelW;
    uint32_t orgHixWi = convRepoTiling->orgHi * convRepoTiling->orgWi;
    uint32_t cinOffsetBlockInGM = convRepoTiling->kAL1 / kernelHxkernelW * orgHixWi;
    uint32_t mStep = (!flagInfo_.mSplitModeFlag) ?
        ConvAlignB(convRepoTiling->hoL0 * convRepoTiling->woL0, convOpsConstParams_.m0):
        ConvAlignB(convRepoTiling->hoL0, convOpsConstParams_.m0);
    uint32_t fmapKStride = mStep / convOpsConstParams_.m0;
    uint32_t nStep = ConvCeilDiv(convRepoTiling->nL0, convOpsConstParams_.n0);
    uint32_t kStep = convRepoTiling->kL0 / convOpsConstParams_.k0;
    uint32_t weightKStride = ConvCeilDiv(convRepoTiling->nBL1, convOpsConstParams_.n0);
    uint32_t coutOffsetBlock = (convRepoTiling->orgCi / convRepoTiling->groups) * kernelHxkernelW;
    tilingData_.convApiTiling.orgHixWi = orgHixWi;
    tilingData_.convApiTiling.kernelHxkernelW = kernelHxkernelW;
    tilingData_.convApiTiling.kernelHxkernelWxkernelD = kernelHxkernelW * convRepoTiling->kernelD;
    tilingData_.convApiTiling.multiNBL1 = ConvCeilDiv(convRepoTiling->nBL1, convRepoTiling->nL0);
    tilingData_.convApiTiling.cinAInCore = cinAInCore;
    tilingData_.convApiTiling.cinATailInCore = cinATailInCore;
    tilingData_.convApiTiling.cinBInCore = convRepoTiling->kBL1 / kernelHxkernelW;
    tilingData_.convApiTiling.cinBTailInCore = cinBTailInCore;
    tilingData_.convApiTiling.mStep = mStep;
    tilingData_.convApiTiling.kStep = kStep;
    tilingData_.convApiTiling.nStep = nStep;
    tilingData_.convApiTiling.fmapKStride = fmapKStride;
    tilingData_.convApiTiling.weightKStride = weightKStride;
    tilingData_.convApiTiling.cinOffsetBlockInGM = cinOffsetBlockInGM;
    tilingData_.convApiTiling.coutOffsetBlock = coutOffsetBlock;
    tilingData_.convApiTiling.nL1DivBlockSize = convRepoTiling->nBL1 / convOpsConstParams_.n0;
    tilingData_.convApiTiling.aL1SpaceSize = CalcAL1SpaceSize(convRepoTiling);
    tilingData_.convApiTiling.roundMode = attrInfo_.roundMode;
    tilingData_.convApiTiling.hasBias = flagInfo_.hasBias;
    tilingData_.convApiTiling.hasScale = flagInfo_.quantFlag;
    tilingData_.convApiTiling.offsetx = attrInfo_.offsetx;
    tilingData_.convApiTiling.hf32Enable = convRepoTiling->hf32Enable;
    tilingData_.convApiTiling.hf32TransMode = convRepoTiling->hf32TransMode;
    uint64_t singleGroups = flagInfo_.convGroupType == ConvGroupType::OPT_GROUP_CONV ?
        optGroupInfo_.enlarge : 0;
    uint64_t singleGroupOpt = flagInfo_.convGroupType == ConvGroupType::OPT_GROUP_CONV ?
        ConvCeilDiv(optGroupInfo_.groupOpt, convRepoTiling->groupDim) : 0;
    tilingData_.convApiTiling.singleCoreGroups = singleGroups;
    tilingData_.convApiTiling.singleCoreGroupOpt = singleGroupOpt;
    SetUnionDataXt(convRepoTiling);
}

void Conv3dBaseTilingV2::SetUnionDataXt(shared_ptr<tuningtiling::Conv3DV2TunnerTiling> convRepoTiling)
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
    tilingData_.convApiTiling.unionDataXt = unionDataXt.n;
}

void Conv3dBaseTilingV2::PrintInputArgs(shared_ptr<tuningtiling::Conv3DV2InputArgs> conv3DInput)
{
    stringstream ss;
    ss << paramInfo_.nodeType.c_str() << "ops tiling input args: aDtype: " << conv3DInput->aDtype
    << ", bDtype: " << conv3DInput->bDtype
    << ", cDtype: " << conv3DInput->cDtype
    << ", biasDtype: " << conv3DInput->biasDtype
    << ", aShapeN: " << conv3DInput->aShapeN
    << ", aShapeD: " << conv3DInput->aShapeD
    << ", aShapeH: " << conv3DInput->aShapeH
    << ", aShapeW: " << conv3DInput->aShapeW
    << ", bShapeN: " << conv3DInput->bShapeN
    << ", bShapeC: " << conv3DInput->bShapeC
    << ", bShapeD: " << conv3DInput->bShapeD
    << ", bShapeH: " << conv3DInput->bShapeH
    << ", bShapeW: " << conv3DInput->bShapeW
    << ", cShapeD: " << conv3DInput->cShapeD
    << ", cShapeH: " << conv3DInput->cShapeH
    << ", cShapeW: " << conv3DInput->cShapeW
    << ", aFormat: " << conv3DInput->aFormat
    << ", bFormat: " << conv3DInput->bFormat
    << ", cFormat: " << conv3DInput->cFormat
    << ", groups: " << conv3DInput->groups
    << ", strideD: " << conv3DInput->strideD
    << ", strideH: " << conv3DInput->strideH
    << ", strideW: " << conv3DInput->strideW
    << ", dilationD: " << conv3DInput->dilationD
    << ", dilationH: " << conv3DInput->dilationH
    << ", dilationW: " << conv3DInput->dilationW
    << ", padHead: " << conv3DInput->padHead
    << ", padTail: " << conv3DInput->padTail
    << ", padTop: " << conv3DInput->padTop
    << ", padBottom: " << conv3DInput->padBottom
    << ", padLeft: " << conv3DInput->padLeft
    << ", padRight: " << conv3DInput->padRight
    << ", biasFlag: " << conv3DInput->biasFlag
    << ", reserverdParam1: " << conv3DInput->reserverdParam1
    << ", reserverdParam2: " << conv3DInput->reserverdParam2
    << ", reserverdParam3: " << conv3DInput->reserverdParam3
    << ", reserverdParam4: " << conv3DInput->reserverdParam4
    << ", reserverdParam5: " << conv3DInput->reserverdParam5
    << ", reserverdParam6: " << conv3DInput->reserverdParam6;
    OP_LOGD(context_->GetNodeName(), "%s", ss.str().c_str());
}

}
}