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

void Conv3dBaseTilingV2::GetTilingInputArgs(std::shared_ptr<void>& inputArgs, size_t& size)
{
    std::shared_ptr<tuningtiling::Conv3DV2InputArgs> conv3DInput = std::make_shared<tuningtiling::Conv3DV2InputArgs>();

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

bool Conv3dBaseTilingV2::TranslateRepoTiling(tuningtiling::TuningTilingDefPtr& tuningTiling)
{
    auto convRepoTiling = std::static_pointer_cast<tuningtiling::Conv3DV2TunnerTiling>(tuningTiling);
    if (convRepoTiling == nullptr) {
        return false;
    }

    TranslateApiTiling(convRepoTiling);
    TranslateRunInfo(convRepoTiling);
    flagInfo_.mSplitModeFlag = static_cast<bool>(convRepoTiling->mMode);
    flagInfo_.isKernelSplit = (convRepoTiling->khL1 > 0 && convRepoTiling->khL1 < convRepoTiling->kernelH) ||
                              (convRepoTiling->kwL1 > 0 && convRepoTiling->kwL1 < convRepoTiling->kernelW);
    flagInfo_.convGroupType = GetGroupsInfo();
    TranslateApiTilingAux(convRepoTiling);

    return true;
}

void Conv3dBaseTilingV2::TranslateApiTiling(std::shared_ptr<tuningtiling::Conv3DV2TunnerTiling> convRepoTiling)
{
    tilingData_.groups = convRepoTiling->groups;
    tilingData_.orgCi = convRepoTiling->orgCi;
    tilingData_.orgDi = convRepoTiling->orgDi;
    tilingData_.orgHi = convRepoTiling->orgHi;
    tilingData_.orgWi = convRepoTiling->orgWi;
    tilingData_.orgDo = convRepoTiling->orgDo;
    tilingData_.orgCo = convRepoTiling->orgCo;
    tilingData_.orgHo = convRepoTiling->orgHo;
    tilingData_.orgWo = convRepoTiling->orgWo;
    tilingData_.kernelD = convRepoTiling->kernelD;
    tilingData_.kernelH = convRepoTiling->kernelH;
    tilingData_.kernelW = convRepoTiling->kernelW;
    tilingData_.singleCoreBatch = ConvCeilDiv(shapeInfo_.batch, convRepoTiling->batchDim);
    tilingData_.singleCoreDo = convRepoTiling->singleCoreDo;
    tilingData_.singleCoreCo = convRepoTiling->singleCoreCo;
    tilingData_.singleCoreHo = convRepoTiling->singleCoreHo;
    tilingData_.singleCoreCi = convRepoTiling->singleCoreCi;
    tilingData_.singleCoreWo = convRepoTiling->singleCoreWo;
    tilingData_.hoL0 = convRepoTiling->hoL0;
    tilingData_.kL0 = convRepoTiling->kL0;
    tilingData_.nL0 = convRepoTiling->nL0;
    tilingData_.woL0 = convRepoTiling->woL0;
    tilingData_.kAL1 = convRepoTiling->kAL1;
    tilingData_.kBL1 = convRepoTiling->kBL1;
    tilingData_.nBL1 = convRepoTiling->nBL1;
    tilingData_.hoL1 = convRepoTiling->hoL1;
    tilingData_.woL1 = convRepoTiling->woL1;
    tilingData_.strideD = convRepoTiling->strideD;
    tilingData_.strideH = convRepoTiling->strideH;
    tilingData_.strideW = convRepoTiling->strideW;
    tilingData_.dilationD = convRepoTiling->dilationD;
    tilingData_.dilationH = convRepoTiling->dilationH;
    tilingData_.dilationW = convRepoTiling->dilationW;
    tilingData_.padHead = convRepoTiling->padHead;
    tilingData_.padTail = convRepoTiling->padTail;
    tilingData_.padTop = convRepoTiling->padTop;
    tilingData_.padBottom = convRepoTiling->padBottom;
    tilingData_.padLeft = convRepoTiling->padLeft;
    tilingData_.padRight = convRepoTiling->padRight;
    tilingData_.pBufferFlag = convRepoTiling->pBufferFlag;
    tilingData_.iterateMNOrder = convRepoTiling->iterateMNOrder;
    tilingData_.biasFullLoadFlag = convRepoTiling->biasFullLoadFlag;
    tilingData_.fixpParamsFullLoadFlag = convRepoTiling->fixpParamsFullLoadFlag;
    tilingData_.enlarge = optGroupInfo_.enlarge;
    tilingData_.khL1 = convRepoTiling->khL1;
    tilingData_.kwL1 = convRepoTiling->kwL1;
}

void Conv3dBaseTilingV2::TranslateRunInfo(std::shared_ptr<tuningtiling::Conv3DV2TunnerTiling> convRepoTiling)
{
    tilingData_.batch = shapeInfo_.batch;
    tilingData_.cin = convRepoTiling->orgCi;
    tilingData_.din = convRepoTiling->orgDi;
    tilingData_.hin = convRepoTiling->orgHi;
    tilingData_.win = convRepoTiling->orgWi;
    tilingData_.cout = convRepoTiling->orgCo;
    tilingData_.kd = convRepoTiling->kernelD;
    tilingData_.kh = convRepoTiling->kernelH;
    tilingData_.kw = convRepoTiling->kernelW;
    tilingData_.dout = convRepoTiling->orgDo;
    tilingData_.hout = convRepoTiling->orgHo;
    tilingData_.wout = convRepoTiling->orgWo;
    tilingData_.batchDim = convRepoTiling->batchDim;
    tilingData_.hoDim = convRepoTiling->hoDim;
    tilingData_.nDim = convRepoTiling->nDim;
    tilingData_.doDim = convRepoTiling->doDim;
    tilingData_.groupDim = convRepoTiling->groupDim;
    tilingData_.hasBias = flagInfo_.hasBias;
    tilingData_.cinOpt = optGroupInfo_.cinOpt;
    tilingData_.coutOpt = optGroupInfo_.coutOpt;
    tilingData_.groupOpt = optGroupInfo_.groupOpt;
}

uint32_t Conv3dBaseTilingV2::CalcAL1SpaceSize(shared_ptr<tuningtiling::Conv3DV2TunnerTiling> convRepoTiling)
{
    uint64_t aL1SpaceSize = 0;
    uint64_t fmapSize = dtypeSizeTab.at(descInfo_.fMapDtype);

    uint64_t dilatedKernelH = (convRepoTiling->kernelH - 1) * convRepoTiling->dilationH + 1;
    if (convRepoTiling->mMode == 1) {
        uint64_t mL1Max = convRepoTiling->hoL1 < convRepoTiling->singleCoreHo ? convRepoTiling->hoL1 :
                                                                                convRepoTiling->singleCoreHo;
        uint64_t hoL1Max = std::min(mL1Max / convRepoTiling->orgWo + CONST_VALUE_2, convRepoTiling->orgHo);
        uint64_t hiAL1Max = (hoL1Max - 1) * convRepoTiling->strideH + dilatedKernelH;
        hiAL1Max = hiAL1Max > convRepoTiling->orgHi ? convRepoTiling->orgHi : hiAL1Max;

        aL1SpaceSize = tilingData_.cinAInCore * hiAL1Max * convRepoTiling->orgWi;
    } else {
        uint64_t hiAL1Max = (convRepoTiling->hoL1 - 1) * convRepoTiling->strideH + dilatedKernelH;
        hiAL1Max = hiAL1Max > convRepoTiling->orgHi ? convRepoTiling->orgHi : hiAL1Max;

        uint64_t wiAL1Max = 0;
        if ((convRepoTiling->isC04Flag == 1) && convRepoTiling->singleCoreWo == convRepoTiling->woL1) {
            wiAL1Max = convRepoTiling->orgWi;

            aL1SpaceSize = ConvAlignB(hiAL1Max * wiAL1Max, C0_SIZE / (fmapSize * C04_CIN_SIZE)) * C04_CIN_SIZE;
        } else {
            uint64_t dilatedKernelW = (convRepoTiling->kernelW - 1) * convRepoTiling->dilationW + 1;
            wiAL1Max = (convRepoTiling->woL1 - 1) * convRepoTiling->strideW + dilatedKernelW;
            wiAL1Max = wiAL1Max > convRepoTiling->orgWi ? convRepoTiling->orgWi : wiAL1Max;

            aL1SpaceSize = tilingData_.cinAInCore * hiAL1Max * wiAL1Max;
        }
    }
    aL1SpaceSize = ConvAlignB(aL1SpaceSize * fmapSize, C0_SIZE);

    return static_cast<uint32_t>(aL1SpaceSize);
}

void Conv3dBaseTilingV2::TranslateApiTilingAux(shared_ptr<tuningtiling::Conv3DV2TunnerTiling> convRepoTiling)
{
    uint32_t kernelHxkernelW = convRepoTiling->kernelH * convRepoTiling->kernelW;
    uint32_t kernelValueInKSize = flagInfo_.isKernelSplit ? convRepoTiling->khL1 * convRepoTiling->kwL1 :
                                                            kernelHxkernelW;
    uint32_t cinAInCore = convRepoTiling->kAL1 / kernelValueInKSize;
    uint32_t kAL1Tail = (ConvAlignB(convRepoTiling->singleCoreCi, convOpsConstParams_.k0) * kernelValueInKSize *
                         convRepoTiling->kernelD) %
                        convRepoTiling->kAL1;
    kAL1Tail = kAL1Tail == 0 ? convRepoTiling->kAL1 : kAL1Tail;
    uint32_t kBL1Tail = (convRepoTiling->singleCoreCi * kernelValueInKSize) % convRepoTiling->kBL1;
    kBL1Tail = kBL1Tail == 0 ? convRepoTiling->kBL1 : kBL1Tail;
    uint32_t cinATailInCore = kAL1Tail / kernelValueInKSize;
    uint32_t cinBTailInCore = kBL1Tail / kernelValueInKSize;
    uint32_t orgHixWi = convRepoTiling->orgHi * convRepoTiling->orgWi;
    uint32_t cinOffsetBlockInGM = convRepoTiling->kAL1 / kernelValueInKSize * orgHixWi;
    uint32_t mStep = (!flagInfo_.mSplitModeFlag) ?
                         ConvAlignB(convRepoTiling->hoL0 * convRepoTiling->woL0, convOpsConstParams_.m0) :
                         ConvAlignB(convRepoTiling->hoL0, convOpsConstParams_.m0);
    uint32_t fmapKStride = mStep / convOpsConstParams_.m0;
    uint32_t nStep = ConvCeilDiv(convRepoTiling->nL0, convOpsConstParams_.n0);
    uint32_t kStep = convRepoTiling->kL0 / convOpsConstParams_.k0;
    uint32_t weightKStride = ConvCeilDiv(convRepoTiling->nBL1, convOpsConstParams_.n0);
    uint32_t coutOffsetBlock = (convRepoTiling->orgCi / convRepoTiling->groups) * kernelHxkernelW;
    tilingData_.orgHixWi = orgHixWi;
    tilingData_.kernelHxkernelW = kernelHxkernelW;
    tilingData_.kernelHxkernelWxkernelD = kernelHxkernelW * convRepoTiling->kernelD;
    tilingData_.multiNBL1 = ConvCeilDiv(convRepoTiling->nBL1, convRepoTiling->nL0);
    tilingData_.cinAInCore = cinAInCore;
    tilingData_.cinATailInCore = cinATailInCore;
    tilingData_.cinBInCore = convRepoTiling->kBL1 / kernelValueInKSize;
    tilingData_.cinBTailInCore = cinBTailInCore;
    tilingData_.mStep = mStep;
    tilingData_.kStep = kStep;
    tilingData_.nStep = nStep;
    tilingData_.fmapKStride = fmapKStride;
    tilingData_.weightKStride = weightKStride;
    tilingData_.cinOffsetBlockInGM = cinOffsetBlockInGM;
    tilingData_.coutOffsetBlock = coutOffsetBlock;
    tilingData_.nL1DivBlockSize = convRepoTiling->nBL1 / convOpsConstParams_.n0;
    tilingData_.aL1SpaceSize = CalcAL1SpaceSize(convRepoTiling);
    tilingData_.roundMode = attrInfo_.roundMode;
    tilingData_.hasBias = flagInfo_.hasBias;
    tilingData_.hasScale = flagInfo_.quantFlag;
    tilingData_.offsetx = attrInfo_.offsetx;
    tilingData_.hf32Enable = convRepoTiling->hf32Enable;
    tilingData_.hf32TransMode = convRepoTiling->hf32TransMode;
    uint64_t singleGroups = flagInfo_.convGroupType == ConvGroupType::OPT_GROUP_CONV ? optGroupInfo_.enlarge : 0;
    uint64_t singleGroupOpt = flagInfo_.convGroupType == ConvGroupType::OPT_GROUP_CONV ?
                                  ConvCeilDiv(optGroupInfo_.groupOpt, convRepoTiling->groupDim) :
                                  0;
    tilingData_.singleCoreGroups = singleGroups;
    tilingData_.singleCoreGroupOpt = singleGroupOpt;
    SetUnionDataXt(convRepoTiling);
}

void Conv3dBaseTilingV2::SetUnionDataXt(shared_ptr<tuningtiling::Conv3DV2TunnerTiling> convRepoTiling)
{
    conv_tiling::UnionDataXt unionDataXt;
    uint64_t tmpKh = flagInfo_.isKernelSplit ? static_cast<uint64_t>(convRepoTiling->khL1) :
                                               static_cast<uint64_t>(convRepoTiling->kernelH);
    uint64_t tmpKw = flagInfo_.isKernelSplit ? static_cast<uint64_t>(convRepoTiling->kwL1) :
                                               static_cast<uint64_t>(convRepoTiling->kernelW);
    unionDataXt.bf.kernelW = tmpKw;
    unionDataXt.bf.kernelW_highest_bit = (tmpKw & 0x100) >> 8; // 8 kernelW lower 8 bit
    unionDataXt.bf.kernelH = tmpKh;
    unionDataXt.bf.kernelH_highest_bit = (tmpKh & 0x100) >> 8; // 8 kernelH lower 8 bit
    unionDataXt.bf.dilationH = static_cast<uint64_t>(convRepoTiling->dilationH);
    unionDataXt.bf.dilationW = static_cast<uint64_t>(convRepoTiling->dilationW);
    unionDataXt.bf.strideH = static_cast<uint64_t>(convRepoTiling->strideH) & 0x3f;
    unionDataXt.bf.strideW = static_cast<uint64_t>(convRepoTiling->strideW) & 0x3f;
    tilingData_.unionDataXt = unionDataXt.n;
}

void Conv3dBaseTilingV2::PrintInputArgs(shared_ptr<tuningtiling::Conv3DV2InputArgs> conv3DInput)
{
    stringstream ss;
    ss << paramInfo_.nodeType.c_str() << "ops tiling input args: aDtype: " << conv3DInput->aDtype
       << ", bDtype: " << conv3DInput->bDtype << ", cDtype: " << conv3DInput->cDtype
       << ", biasDtype: " << conv3DInput->biasDtype << ", aShapeN: " << conv3DInput->aShapeN
       << ", aShapeD: " << conv3DInput->aShapeD << ", aShapeH: " << conv3DInput->aShapeH
       << ", aShapeW: " << conv3DInput->aShapeW << ", bShapeN: " << conv3DInput->bShapeN
       << ", bShapeC: " << conv3DInput->bShapeC << ", bShapeD: " << conv3DInput->bShapeD
       << ", bShapeH: " << conv3DInput->bShapeH << ", bShapeW: " << conv3DInput->bShapeW
       << ", cShapeD: " << conv3DInput->cShapeD << ", cShapeH: " << conv3DInput->cShapeH
       << ", cShapeW: " << conv3DInput->cShapeW << ", aFormat: " << conv3DInput->aFormat
       << ", bFormat: " << conv3DInput->bFormat << ", cFormat: " << conv3DInput->cFormat
       << ", groups: " << conv3DInput->groups << ", strideD: " << conv3DInput->strideD
       << ", strideH: " << conv3DInput->strideH << ", strideW: " << conv3DInput->strideW
       << ", dilationD: " << conv3DInput->dilationD << ", dilationH: " << conv3DInput->dilationH
       << ", dilationW: " << conv3DInput->dilationW << ", padHead: " << conv3DInput->padHead
       << ", padTail: " << conv3DInput->padTail << ", padTop: " << conv3DInput->padTop
       << ", padBottom: " << conv3DInput->padBottom << ", padLeft: " << conv3DInput->padLeft
       << ", padRight: " << conv3DInput->padRight << ", biasFlag: " << conv3DInput->biasFlag
       << ", reserverdParam1: " << conv3DInput->reserverdParam1 << ", reserverdParam2: " << conv3DInput->reserverdParam2
       << ", reserverdParam3: " << conv3DInput->reserverdParam3 << ", reserverdParam4: " << conv3DInput->reserverdParam4
       << ", reserverdParam5: " << conv3DInput->reserverdParam5
       << ", reserverdParam6: " << conv3DInput->reserverdParam6;
    OP_LOGD(context_->GetNodeName(), "%s", ss.str().c_str());
}

} // namespace conv_ops_tiling
} // namespace optiling