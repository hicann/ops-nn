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
const bool Conv3dBaseTilingV2::GetTilingFromRepo()
{
    std::shared_ptr<tuningtiling::TuningTilingDef> tuningTiling = nullptr;
    auto compileInfoPtr =
        static_cast<const ConvTilingParseInfo*>(context_->GetCompileInfo());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, compileInfoPtr);
    auto opInfo_tmp = *compileInfoPtr;
    std::shared_ptr<void> inputArgs = nullptr;
    std::size_t inputArgsSize = 0;
    GetTilingInputArgs(inputArgs, inputArgsSize);

    uint32_t ret = Ops::NN::QueryBank(inputArgs.get(), inputArgsSize, "Conv3DV2", opInfo_tmp.socVersion,
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
    convBase_.SetBitsFromBool(conv3DInput->reserverdParam1, bitValues);
    convBase_.SetBytesFromUint8(conv3DInput->reserverdParam2, byteValues);
    convBase_.SetBytesFromUint32(conv3DInput->reserverdParam3, 0, 0);
    conv3DInput->reserverdParam4 = 0;
    conv3DInput->reserverdParam5 = 0;
    conv3DInput->reserverdParam6 = 0;

    inputArgs = conv3DInput;
    size = sizeof(tuningtiling::Conv3DV2InputArgs);
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
    tilingData_.conv3dApiTiling.set_groups(convRepoTiling->groups);
    tilingData_.conv3dApiTiling.set_orgCi(convRepoTiling->orgCi);
    tilingData_.conv3dApiTiling.set_orgDi(convRepoTiling->orgDi);
    tilingData_.conv3dApiTiling.set_orgHi(convRepoTiling->orgHi);
    tilingData_.conv3dApiTiling.set_orgWi(convRepoTiling->orgWi);
    tilingData_.conv3dApiTiling.set_orgDo(convRepoTiling->orgDo);
    tilingData_.conv3dApiTiling.set_orgCo(convRepoTiling->orgCo);
    tilingData_.conv3dApiTiling.set_orgHo(convRepoTiling->orgHo);
    tilingData_.conv3dApiTiling.set_orgWo(convRepoTiling->orgWo);
    tilingData_.conv3dApiTiling.set_kernelD(convRepoTiling->kernelD);
    tilingData_.conv3dApiTiling.set_kernelH(convRepoTiling->kernelH);
    tilingData_.conv3dApiTiling.set_kernelW(convRepoTiling->kernelW);
    tilingData_.conv3dApiTiling.set_singleCoreBatch(ConvCeilDiv(shapeInfo_.batch, convRepoTiling->batchDim));
    tilingData_.conv3dApiTiling.set_singleCoreDo(convRepoTiling->singleCoreDo);
    tilingData_.conv3dApiTiling.set_singleCoreCo(convRepoTiling->singleCoreCo);
    tilingData_.conv3dApiTiling.set_singleCoreHo(convRepoTiling->singleCoreHo);
    tilingData_.conv3dApiTiling.set_singleCoreCi(convRepoTiling->singleCoreCi);
    tilingData_.conv3dApiTiling.set_singleCoreWo(convRepoTiling->singleCoreWo);
    tilingData_.conv3dApiTiling.set_hoL0(convRepoTiling->hoL0);
    tilingData_.conv3dApiTiling.set_kL0(convRepoTiling->kL0);
    tilingData_.conv3dApiTiling.set_nL0(convRepoTiling->nL0);
    tilingData_.conv3dApiTiling.set_woL0(convRepoTiling->woL0);
    tilingData_.conv3dApiTiling.set_kAL1(convRepoTiling->kAL1);
    tilingData_.conv3dApiTiling.set_kBL1(convRepoTiling->kBL1);
    tilingData_.conv3dApiTiling.set_nBL1(convRepoTiling->nBL1);
    tilingData_.conv3dApiTiling.set_hoL1(convRepoTiling->hoL1);
    tilingData_.conv3dApiTiling.set_woL1(convRepoTiling->woL1);
    tilingData_.conv3dApiTiling.set_strideD(convRepoTiling->strideD);
    tilingData_.conv3dApiTiling.set_strideH(convRepoTiling->strideH);
    tilingData_.conv3dApiTiling.set_strideW(convRepoTiling->strideW);
    tilingData_.conv3dApiTiling.set_dilationD(convRepoTiling->dilationD);
    tilingData_.conv3dApiTiling.set_dilationH(convRepoTiling->dilationH);
    tilingData_.conv3dApiTiling.set_dilationW(convRepoTiling->dilationW);
    tilingData_.conv3dApiTiling.set_padHead(convRepoTiling->padHead);
    tilingData_.conv3dApiTiling.set_padTail(convRepoTiling->padTail);
    tilingData_.conv3dApiTiling.set_padTop(convRepoTiling->padTop);
    tilingData_.conv3dApiTiling.set_padBottom(convRepoTiling->padBottom);
    tilingData_.conv3dApiTiling.set_padLeft(convRepoTiling->padLeft);
    tilingData_.conv3dApiTiling.set_padRight(convRepoTiling->padRight);
    tilingData_.conv3dApiTiling.set_pBufferFlag(convRepoTiling->pBufferFlag);
    tilingData_.conv3dApiTiling.set_iterateMNOrder(convRepoTiling->iterateMNOrder);
    tilingData_.conv3dApiTiling.set_biasFullLoadFlag(convRepoTiling->biasFullLoadFlag);
    tilingData_.conv3dApiTiling.set_fixpParamsFullLoadFlag(convRepoTiling->fixpParamsFullLoadFlag);
    tilingData_.conv3dApiTiling.set_enlarge(optGroupInfo_.enlarge);
}

void Conv3dBaseTilingV2::TranslateRunInfo(std::shared_ptr<tuningtiling::Conv3DV2TunnerTiling> convRepoTiling)
{
    tilingData_.conv3dRunInfo.set_batch(shapeInfo_.batch);
    tilingData_.conv3dRunInfo.set_cin(convRepoTiling->orgCi);
    tilingData_.conv3dRunInfo.set_din(convRepoTiling->orgDi);
    tilingData_.conv3dRunInfo.set_hin(convRepoTiling->orgHi);
    tilingData_.conv3dRunInfo.set_win(convRepoTiling->orgWi);
    tilingData_.conv3dRunInfo.set_cout(convRepoTiling->orgCo);
    tilingData_.conv3dRunInfo.set_kd(convRepoTiling->kernelD);
    tilingData_.conv3dRunInfo.set_kh(convRepoTiling->kernelH);
    tilingData_.conv3dRunInfo.set_kw(convRepoTiling->kernelW);
    tilingData_.conv3dRunInfo.set_dout(convRepoTiling->orgDo);
    tilingData_.conv3dRunInfo.set_hout(convRepoTiling->orgHo);
    tilingData_.conv3dRunInfo.set_wout(convRepoTiling->orgWo);
    tilingData_.conv3dRunInfo.set_batchDim(convRepoTiling->batchDim);
    tilingData_.conv3dRunInfo.set_hoDim(convRepoTiling->hoDim);
    tilingData_.conv3dRunInfo.set_nDim(convRepoTiling->nDim);
    tilingData_.conv3dRunInfo.set_doDim(convRepoTiling->doDim);
    tilingData_.conv3dRunInfo.set_groupDim(convRepoTiling->groupDim);
    tilingData_.conv3dRunInfo.set_strideH(convRepoTiling->strideH);
    tilingData_.conv3dRunInfo.set_strideD(convRepoTiling->strideD);
    tilingData_.conv3dRunInfo.set_dilationH(convRepoTiling->dilationH);
    tilingData_.conv3dRunInfo.set_dilationD(convRepoTiling->dilationD);
    tilingData_.conv3dRunInfo.set_padHead(convRepoTiling->padHead);
    tilingData_.conv3dRunInfo.set_padTop(convRepoTiling->padTop);
    tilingData_.conv3dRunInfo.set_hasBias(flagInfo_.hasBias);
    tilingData_.conv3dRunInfo.set_groups(convRepoTiling->groups);
    tilingData_.conv3dRunInfo.set_cinOpt(optGroupInfo_.cinOpt);
    tilingData_.conv3dRunInfo.set_coutOpt(optGroupInfo_.coutOpt);
    tilingData_.conv3dRunInfo.set_groupOpt(optGroupInfo_.groupOpt);
    tilingData_.conv3dRunInfo.set_enlarge(optGroupInfo_.enlarge);
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

        aL1SpaceSize = tilingData_.conv3dApiTiling.get_cinAInCore() * hiAL1Max * convRepoTiling->orgWi;
    } else {
        uint64_t hiAL1Max = (convRepoTiling->hoL1 - 1) * convRepoTiling->strideH + dilatedKernelH;
        hiAL1Max = hiAL1Max > convRepoTiling->orgHi ? convRepoTiling->orgHi : hiAL1Max;

        uint64_t wiAL1Max = 0;
        if (convRepoTiling->isC04Flag && convRepoTiling->singleCoreWo == convRepoTiling->woL1) {
            wiAL1Max = convRepoTiling->orgWi;

            aL1SpaceSize = ConvAlignB(hiAL1Max * wiAL1Max, C0_SIZE /
                (fmapSize * C04_CIN_SIZE)) * C04_CIN_SIZE;
        } else {
            uint64_t dilatedKernelW = (convRepoTiling->kernelW - 1) * convRepoTiling->dilationW + 1;
            wiAL1Max = (convRepoTiling->woL1 - 1) * convRepoTiling->strideW + dilatedKernelW;
            wiAL1Max = wiAL1Max > convRepoTiling->orgWi ? convRepoTiling->orgWi : wiAL1Max;

            aL1SpaceSize = tilingData_.conv3dApiTiling.get_cinAInCore() * hiAL1Max * wiAL1Max;
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
    uint32_t mStep = (GetOutputOrderVal() == 0) ?
        ConvAlignB(convRepoTiling->hoL0 * convRepoTiling->woL0, convOpsConstParams_.m0):
        ConvAlignB(convRepoTiling->hoL0, convOpsConstParams_.m0);
    uint32_t fmapKStride = mStep / convOpsConstParams_.m0;
    uint32_t nStep = ConvCeilDiv(convRepoTiling->nL0, convOpsConstParams_.n0);
    uint32_t kStep = convRepoTiling->kL0 / convOpsConstParams_.k0;
    uint32_t weightKStride = ConvCeilDiv(convRepoTiling->nBL1, convOpsConstParams_.n0);
    uint32_t coutOffsetBlock = (convRepoTiling->orgCi / convRepoTiling->groups) * kernelHxkernelW;
    tilingData_.conv3dApiTiling.set_orgHixWi(orgHixWi);
    tilingData_.conv3dApiTiling.set_kernelHxkernelW(kernelHxkernelW);
    tilingData_.conv3dApiTiling.set_kernelHxkernelWxkernelD(kernelHxkernelW * convRepoTiling->kernelD);
    tilingData_.conv3dApiTiling.set_multiNBL1(ConvCeilDiv(convRepoTiling->nBL1, convRepoTiling->nL0));
    tilingData_.conv3dApiTiling.set_cinAInCore(cinAInCore);
    tilingData_.conv3dApiTiling.set_cinATailInCore(cinATailInCore);
    tilingData_.conv3dApiTiling.set_cinBInCore(convRepoTiling->kBL1 / kernelHxkernelW);
    tilingData_.conv3dApiTiling.set_cinBTailInCore(cinBTailInCore);
    tilingData_.conv3dApiTiling.set_mStep(mStep);
    tilingData_.conv3dApiTiling.set_kStep(kStep);
    tilingData_.conv3dApiTiling.set_nStep(nStep);
    tilingData_.conv3dApiTiling.set_fmapKStride(fmapKStride);
    tilingData_.conv3dApiTiling.set_weightKStride(weightKStride);
    tilingData_.conv3dApiTiling.set_cinOffsetBlockInGM(cinOffsetBlockInGM);
    tilingData_.conv3dApiTiling.set_coutOffsetBlock(coutOffsetBlock);
    tilingData_.conv3dApiTiling.set_nL1DivBlockSize(convRepoTiling->nBL1 / convOpsConstParams_.n0);
    tilingData_.conv3dApiTiling.set_aL1SpaceSize(CalcAL1SpaceSize(convRepoTiling));
    tilingData_.conv3dApiTiling.set_roundMode(attrInfo_.roundMode);
    tilingData_.conv3dApiTiling.set_hasBias(flagInfo_.hasBias);
    tilingData_.conv3dApiTiling.set_hasScale(flagInfo_.quantFlag);
    tilingData_.conv3dApiTiling.set_offsetx(attrInfo_.offsetx);
    tilingData_.conv3dApiTiling.set_hf32Enable(convRepoTiling->hf32Enable);
    tilingData_.conv3dApiTiling.set_hf32TransMode(convRepoTiling->hf32TransMode);
    uint64_t singleGroups = flagInfo_.convGroupType == ConvGroupType::OPT_GROUP_CONV ?
        optGroupInfo_.enlarge : 0;
    uint64_t singleGroupOpt = flagInfo_.convGroupType == ConvGroupType::OPT_GROUP_CONV ?
        ConvCeilDiv(optGroupInfo_.groupOpt, convRepoTiling->groupDim) : 0;
    tilingData_.conv3dApiTiling.set_singleCoreGroups(singleGroups);
    tilingData_.conv3dApiTiling.set_singleCoreGroupOpt(singleGroupOpt);
}

}
}