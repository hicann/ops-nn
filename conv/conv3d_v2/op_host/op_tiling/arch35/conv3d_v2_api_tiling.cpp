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
 * \file conv3d_v2_tiling.cpp
 * \brief
 */
#include <set>
#include <utility>
#include <cstdint>
#include <unordered_set>
#include <unordered_map>
#include "common/op_host/op_tiling/arch35/conv_api_tiling_algorithm_HWmode.h"
#include "common/op_host/op_tiling/arch35/conv_api_tiling_algorithm_Mmode.h"
#include "conv3d_v2_tiling.h"
#include "conv3d_v2_api_tiling.h"
#include "conv3d_v2/op_kernel/conv3d_v2_tiling_data.h"
#include "../conv3d_api_tiling_utils.h"
#include "conv/common/op_host/op_tiling/arch35/conv_base.h"

using namespace conv_tiling_algo_m;
using namespace conv_tiling_algo_hw;
using namespace std;

using optiling::conv_ops_tiling::ConvCeilDiv;
using optiling::conv_ops_tiling::ConvAlignB;

namespace conv_tiling {
namespace {
constexpr int64_t RET_FAIL = -1;
}

void Conv3dTiling::InitFlag()
{
    this->isScaleBiasInUb = descInfo.fMapType.format == ConvFormat::NCDHW &&
                            descInfo.outputType.format == ConvFormat::NDHWC &&
                            descInfo.scaleType.dtype == ConvDtype::FLOAT32;
    // conv3d int8: scale and bias will be stored in UB but not in L1
    if (this->isScaleBiasInUb) {
        this->hasBias = false;
        this->hasScale = false;
    }
}

int64_t Conv3dTiling::GetTiling(Ops::NN::Conv3dV2::TConv3DTiling &tiling)
{
    if (!CheckInputParam()) {
        TILING_LOG_ERROR("conv3d api tiling check params failed.");
        return RET_FAIL;
    }
    InitFlag();
    InferCubeInfo();
    Infer5hdShape();

    platformInfo.abL1mte2BandWidthCof = GetBandWidthCof(platformInfo.socVersion);
    int64_t ret = Compute();
    if (ret == RET_FAIL) {
        TILING_LOG_ERROR("conv3d api tiling compute failed.");
        return RET_FAIL;
    }
    ret = CheckTilingRes();
    if (ret == RET_FAIL) {
        TILING_LOG_ERROR("conv3d api tiling check tiling result failed.");
        return RET_FAIL;
    }
    SetTilingData(tiling);
    PrintTilingData();
    return ret;
}

void Conv3dTiling::SetOutputOrder(int8_t outOrder)
{
    this->outputOrder = outOrder;
}

int64_t Conv3dTiling::Compute()
{
    switch (outputOrder) {
        case static_cast<int8_t>(OutputOrder::M):
            algoPtr = std::make_shared<ConvTilingAlgorithmMmode>(this);
            break;
        case static_cast<int8_t>(OutputOrder::HW):
            algoPtr = std::make_shared<ConvTilingAlgorithmHWmode>(this);
            break;
        default:
            TILING_LOG_ERROR("Unsupported output order %d, only support Mmode(%d) and HWmode(%d).",
                outputOrder, static_cast<int>(OutputOrder::M), static_cast<int>(OutputOrder::HW));
            return RET_FAIL;
    }
    TILING_LOG_DEBUG("output order: %d", outputOrder);
    int64_t ret = algoPtr->Process();
    return ret;
}

void Conv3dTiling::Infer5hdShape()
{
    this->shapeInfo.singleCi1 = CeilDiv(this->shapeInfo.singleCi, this->cubeInfo.k0);
    this->shapeInfo.singleCo1 = CeilDiv(this->shapeInfo.singleCo, this->cubeInfo.n0);
    if (outputOrder == static_cast<int8_t>(OutputOrder::M)) {
        this->shapeInfo.singleM1 = CeilDiv(this->shapeInfo.singleM, this->cubeInfo.m0);
    }
}

bool Conv3dTiling::CheckInputFormat()
{
    std::set<std::pair<ConvFormat, ConvFormat>> conv3dSupportFormatSet;
    if (platformInfo.socVersion == platform_ascendc::SocVersion::ASCEND910_95) {
        conv3dSupportFormatSet = {
            {ConvFormat::NCDHW, ConvFormat::NCDHW}, {ConvFormat::NDHWC, ConvFormat::DHWCN}
        };
    } else if (platformInfo.socVersion == platform_ascendc::SocVersion::ASCEND910_55) {
        conv3dSupportFormatSet = {
            {ConvFormat::NCDHW, ConvFormat::NCDHW}
        };
    }

    if (conv3dSupportFormatSet.find({descInfo.fMapType.format, descInfo.weightType.format}) ==
        conv3dSupportFormatSet.end()) {
        TILING_LOG_ERROR("unSupported format combo: fmap format: %s, weight format: %s.",
                         FORMAT_TO_STR.at(descInfo.fMapType.format).c_str(),
                         FORMAT_TO_STR.at(descInfo.weightType.format).c_str());
        return false;
    }

    return true;
}

uint32_t Conv3dTiling::CalcAL1SpaceSize(Ops::NN::Conv3dV2::TConv3DTiling& tiling)
{
    uint64_t aL1SpaceSize = 0;
    uint64_t fmapSize = DTYPE_SIZE_TAB.at(descInfo.fMapType.dtype);

    uint64_t dilatedKernelH = (tiling.kernelH - 1) * tiling.dilationH + 1;
    if (outputOrder == static_cast<int8_t>(OutputOrder::M)) {
        uint64_t mL1Max = tiling.hoL1 < tiling.singleCoreHo ? tiling.hoL1 : tiling.singleCoreHo;
        uint64_t hoL1Max = std::min(mL1Max / tiling.orgWo + CONST_VALUE_2, tiling.orgHo);
        uint64_t hiAL1Max = (hoL1Max - 1) * tiling.strideH + dilatedKernelH;
        hiAL1Max = hiAL1Max > tiling.orgHi ? tiling.orgHi : hiAL1Max;

        aL1SpaceSize = tiling.cinAInCore * hiAL1Max * tiling.orgWi;
    } else {
        uint64_t hiAL1Max = (tiling.hoL1 - 1) * tiling.strideH + dilatedKernelH;
        hiAL1Max = hiAL1Max > tiling.orgHi ? tiling.orgHi : hiAL1Max;

        uint64_t wiAL1Max = 0;
        if (isC04Flag && tiling.singleCoreWo == tiling.woL1) {
            wiAL1Max = tiling.orgWi;

            aL1SpaceSize = AlignB(hiAL1Max * wiAL1Max, C0_SIZE / (fmapSize * C04_CIN_SIZE)) * C04_CIN_SIZE;
        } else {
            uint64_t dilatedKernelW = (tiling.kernelW - 1) * tiling.dilationW + 1;
            wiAL1Max = (tiling.woL1 - 1) * tiling.strideW + dilatedKernelW;
            wiAL1Max = wiAL1Max > tiling.orgWi ? tiling.orgWi : wiAL1Max;

            aL1SpaceSize = tiling.cinAInCore * hiAL1Max * wiAL1Max;
        }
    }
    aL1SpaceSize = AlignB(aL1SpaceSize * fmapSize, C0_SIZE);

    return static_cast<uint32_t>(aL1SpaceSize);
}

void Conv3dTiling::SetScalarParams(Ops::NN::Conv3dV2::TConv3DTiling& tiling)
{
    // calculate the follow params in tiling process, for scalar optimization in kernel
    uint32_t kernelHxkernelW = tiling.kernelH * tiling.kernelW;
    uint32_t cinAInCore = tiling.kAL1 / kernelHxkernelW;
    uint32_t kAL1Tail = (AlignB(tiling.singleCoreCi, cubeInfo.k0) * tiling.kernelD * kernelHxkernelW) %
                        tiling.kAL1;
    kAL1Tail = kAL1Tail == 0 ? tiling.kAL1 : kAL1Tail;
    uint32_t kBL1Tail = (tiling.singleCoreCi * kernelHxkernelW) % tiling.kBL1;
    kBL1Tail = kBL1Tail == 0 ? tiling.kBL1 : kBL1Tail;
    uint32_t cinATailInCore = kAL1Tail / kernelHxkernelW;
    uint32_t cinBTailInCore = kBL1Tail / kernelHxkernelW;
    uint64_t orgHixWi = tiling.orgHi * tiling.orgWi;
    uint32_t cinOffsetBlockInGM = tiling.kAL1 / kernelHxkernelW * orgHixWi;
    uint32_t mStep = 0;
    if (outputOrder == static_cast<int8_t>(OutputOrder::M)) {
        mStep = AlignB(tiling.hoL0, cubeInfo.m0);
    } else {
        mStep = AlignB(tiling.hoL0 * tiling.woL0, cubeInfo.m0);
    }
    uint32_t fmapKStride = mStep / cubeInfo.m0;
    uint32_t nStep = CeilDiv(tiling.nL0, cubeInfo.n0);
    uint32_t kStep = tiling.kL0 / cubeInfo.k0;
    uint32_t weightKStride = CeilDiv(tiling.nBL1, cubeInfo.n0);
    uint32_t coutOffsetBlock = (tiling.orgCi / tiling.groups) * kernelHxkernelW;
    uint32_t cinBInCore = tiling.kBL1 / kernelHxkernelW;
    uint32_t nL1DivBlockSize = tiling.nBL1 / cubeInfo.n0;

    tiling.kernelHxkernelW  = kernelHxkernelW;
    tiling.kernelHxkernelWxkernelD = tiling.kernelD * tiling.kernelHxkernelW;
    // l0TilingInfo.nL0 != 0 is checked before
    tiling.multiNBL1 = static_cast<uint32_t>(CeilDiv(tiling.nBL1, tiling.nL0));
    tiling.cinAInCore = cinAInCore;
    tiling.cinATailInCore = cinATailInCore;
    tiling.orgHixWi = orgHixWi;
    tiling.cinOffsetBlockInGM = cinOffsetBlockInGM;
    tiling.mStep = mStep;
    tiling.fmapKStride = fmapKStride;
    tiling.nStep = nStep;
    tiling.kStep = kStep;
    tiling.weightKStride = weightKStride;
    tiling.coutOffsetBlock = coutOffsetBlock;
    tiling.cinBInCore = cinBInCore;
    tiling.cinBTailInCore = cinBTailInCore;
    tiling.nL1DivBlockSize = nL1DivBlockSize;
    tiling.aL1SpaceSize = CalcAL1SpaceSize(tiling);
}

void Conv3dTiling::SetAttrsTilingData(Ops::NN::Conv3dV2::TConv3DTiling& tiling)
{
    tiling.strideH = static_cast<uint32_t>(this->attrInfo.strideH);
    tiling.strideW = static_cast<uint32_t>(this->attrInfo.strideW);
    tiling.strideD = static_cast<uint32_t>(this->attrInfo.strideD);
    tiling.dilationH = static_cast<uint32_t>(this->attrInfo.dilationH);
    tiling.dilationW = static_cast<uint32_t>(this->attrInfo.dilationW);
    tiling.dilationD = static_cast<uint32_t>(this->attrInfo.dilationD);
    tiling.padHead = static_cast<uint32_t>(this->attrInfo.padHead);
    tiling.padTail = static_cast<uint32_t>(this->attrInfo.padTail);
    tiling.padTop = static_cast<uint32_t>(this->attrInfo.padTop);
    tiling.padBottom = static_cast<uint32_t>(this->attrInfo.padBottom);
    tiling.padLeft = static_cast<uint32_t>(this->attrInfo.padLeft);
    tiling.padRight = static_cast<uint32_t>(this->attrInfo.padRight);

    tiling.groups = static_cast<uint32_t>(this->attrInfo.groups);

    if (this->optGroupFlag) {
        tiling.singleCoreGroups = static_cast<uint32_t>(shapeInfo.singleGroups);
        tiling.singleCoreGroupOpt = static_cast<uint32_t>(shapeInfo.singleGroupOpt);
        tiling.enlarge = static_cast<uint32_t>(shapeInfo.enlarge);
    }

    tiling.offsetx = this->attrInfo.offsetx;
    tiling.roundMode = this->attrInfo.roundMode;
    if (this->isScaleBiasInUb) {
        tiling.hasBias = 1;
        tiling.hasScale = 1;  
    } else {
        tiling.hasBias = static_cast<uint8_t>(this->hasBias);
        tiling.hasScale = static_cast<uint8_t>(this->hasScale);
    }
}

void Conv3dTiling::SetTilingData(Ops::NN::Conv3dV2::TConv3DTiling& tiling)
{
    if (outputOrder == static_cast<int8_t>(OutputOrder::M)) {
        tiling.singleCoreHo = static_cast<uint64_t>(shapeInfo.singleM);
        tiling.hoL1 = static_cast<uint32_t>(l1TilingInfo.mAL1);
        tiling.hoL0 = static_cast<uint32_t>(l0TilingInfo.mL0);
    } else {
        tiling.singleCoreHo = static_cast<uint64_t>(shapeInfo.singleHo);
        tiling.singleCoreWo = static_cast<uint64_t>(shapeInfo.singleWo);
        tiling.hoL1 = static_cast<uint32_t>(l1TilingInfo.hoAL1);
        tiling.woL1 = static_cast<uint32_t>(l1TilingInfo.woAL1);
        tiling.hoL0 = static_cast<uint32_t>(l0TilingInfo.hoL0);
        tiling.woL0 = static_cast<uint32_t>(l0TilingInfo.woL0);
    }
    tiling.mUB = static_cast<uint32_t>(ubTilingInfo.mUb);
    tiling.nUB = static_cast<uint32_t>(ubTilingInfo.nUb);
    tiling.orgDo = static_cast<uint64_t>(this->shapeInfo.orgDo);
    tiling.orgHo = static_cast<uint64_t>(this->shapeInfo.orgHo);
    tiling.orgWo = static_cast<uint64_t>(this->shapeInfo.orgWo);
    tiling.orgDi = static_cast<uint64_t>(this->shapeInfo.orgDi);
    tiling.orgHi = static_cast<uint64_t>(this->shapeInfo.orgHi);
    tiling.orgWi = static_cast<uint64_t>(this->shapeInfo.orgWi);
    tiling.singleCoreBatch = static_cast<uint64_t>(this->shapeInfo.singleBatch);
    tiling.singleCoreDo = static_cast<uint64_t>(this->shapeInfo.singleDo);
    tiling.singleCoreCi = static_cast<uint32_t>(shapeInfo.singleCi);
    tiling.singleCoreCo = static_cast<uint32_t>(this->shapeInfo.singleCo);
    tiling.orgCo = static_cast<uint32_t>(this->shapeInfo.orgCo);
    tiling.orgCi = static_cast<uint32_t>(this->shapeInfo.orgCi);
    tiling.kernelD = static_cast<uint32_t>(this->shapeInfo.orgkD);
    tiling.kernelH = static_cast<uint32_t>(this->shapeInfo.orgkH);
    tiling.kernelW = static_cast<uint32_t>(this->shapeInfo.orgkW);

    SetAttrsTilingData(tiling);

    tiling.kAL1 = static_cast<uint32_t>(this->l1TilingInfo.kAL1);
    tiling.kBL1 = static_cast<uint32_t>(this->l1TilingInfo.kBL1);
    tiling.nBL1 = static_cast<uint32_t>(this->l1TilingInfo.nBL1);
    tiling.kL0 = static_cast<uint32_t>(this->l0TilingInfo.kL0);
    tiling.nL0 = static_cast<uint32_t>(this->l0TilingInfo.nL0);
    tiling.pBufferFlag = static_cast<uint32_t>(this->dbValue.pBufferFlag);

    tiling.iterateMNOrder = static_cast<uint8_t>(this->l1TilingInfo.iterateMNOrder);
    tiling.biasFullLoadFlag = static_cast<uint8_t>(this->l1TilingInfo.biasFullLoadFlag);
    tiling.fixpParamsFullLoadFlag = static_cast<uint8_t>(this->l1TilingInfo.fixpParamsFullLoadFlag);
    tiling.hf32Enable = static_cast<uint8_t>(this->hf32Enable);
    tiling.hf32TransMode = static_cast<uint8_t>(this->hf32TransMode);
    SetScalarParams(tiling);
}

void Conv3dTiling::PrintTilingData() const
{
    TILING_LOG_DEBUG("orgDo: %ld", this->shapeInfo.orgDo);
    TILING_LOG_DEBUG("orgCo: %ld", this->shapeInfo.orgCo);
    TILING_LOG_DEBUG("orgHo: %ld", this->shapeInfo.orgHo);
    TILING_LOG_DEBUG("orgWo: %ld", this->shapeInfo.orgWo);
    TILING_LOG_DEBUG("orgCi: %ld", this->shapeInfo.orgCi);
    TILING_LOG_DEBUG("orgDi: %ld", this->shapeInfo.orgDi);
    TILING_LOG_DEBUG("orgHi: %ld", this->shapeInfo.orgHi);
    TILING_LOG_DEBUG("orgWi: %ld", this->shapeInfo.orgWi);
    TILING_LOG_DEBUG("orgkD: %ld", this->shapeInfo.orgkD);
    TILING_LOG_DEBUG("orgkH: %ld", this->shapeInfo.orgkH);
    TILING_LOG_DEBUG("orgkW: %ld", this->shapeInfo.orgkW);
    TILING_LOG_DEBUG("singleCo: %ld", this->shapeInfo.singleCo);
    TILING_LOG_DEBUG("singleDo: %ld", this->shapeInfo.singleDo);
    TILING_LOG_DEBUG("singleM: %ld", this->shapeInfo.singleM);
    TILING_LOG_DEBUG("strideH: %d", this->attrInfo.strideH);
    TILING_LOG_DEBUG("strideW: %d", this->attrInfo.strideW);
    TILING_LOG_DEBUG("strideD: %d", this->attrInfo.strideD);
    TILING_LOG_DEBUG("dilationH: %d", this->attrInfo.dilationH);
    TILING_LOG_DEBUG("dilationW: %d", this->attrInfo.dilationW);
    TILING_LOG_DEBUG("dilationD: %d", this->attrInfo.dilationD);
    TILING_LOG_DEBUG("padHead: %d", this->attrInfo.padHead);
    TILING_LOG_DEBUG("padTail: %d", this->attrInfo.padTail);
    TILING_LOG_DEBUG("padTop: %d", this->attrInfo.padTop);
    TILING_LOG_DEBUG("padBottom: %d", this->attrInfo.padBottom);
    TILING_LOG_DEBUG("padLeft: %d", this->attrInfo.padLeft);
    TILING_LOG_DEBUG("padRight: %d", this->attrInfo.padRight);
    TILING_LOG_DEBUG("kAL1: %lu", this->l1TilingInfo.kAL1);
    TILING_LOG_DEBUG("kBL1: %lu", this->l1TilingInfo.kBL1);
    TILING_LOG_DEBUG("hoAL1: %lu", this->l1TilingInfo.hoAL1);
    TILING_LOG_DEBUG("woAL1: %lu", this->l1TilingInfo.woAL1);
    TILING_LOG_DEBUG("mAL1: %lu", this->l1TilingInfo.mAL1);
    TILING_LOG_DEBUG("nBL1: %lu", this->l1TilingInfo.nBL1);
    TILING_LOG_DEBUG("hoL0: %lu", this->l0TilingInfo.hoL0);
    TILING_LOG_DEBUG("woL0: %lu", this->l0TilingInfo.woL0);
    TILING_LOG_DEBUG("mL0: %lu", this->l0TilingInfo.mL0);
    TILING_LOG_DEBUG("kL0: %lu", this->l0TilingInfo.kL0);
    TILING_LOG_DEBUG("nL0: %lu", this->l0TilingInfo.nL0);
    TILING_LOG_DEBUG("pBufferFlag: %lu", this->dbValue.pBufferFlag);
    TILING_LOG_DEBUG("iterateMNOrder: %u", static_cast<uint8_t>(this->l1TilingInfo.iterateMNOrder));
    TILING_LOG_DEBUG("biasFullLoadFlag: %u", static_cast<uint8_t>(this->l1TilingInfo.biasFullLoadFlag));
    TILING_LOG_DEBUG("fixpParamsFullLoadFlag: %u", static_cast<uint8_t>(this->l1TilingInfo.fixpParamsFullLoadFlag));
    TILING_LOG_DEBUG("hf32Enable: %u", static_cast<uint8_t>(this->hf32Enable));
    TILING_LOG_DEBUG("hf32TransMode: %u", static_cast<uint8_t>(this->hf32TransMode));
    TILING_LOG_DEBUG("al1FullLoad: %u", static_cast<uint8_t>(this->l1TilingInfo.al1FullLoad));
    TILING_LOG_DEBUG("bl1FullLoad: %u", static_cast<uint8_t>(this->l1TilingInfo.bl1FullLoad));
    TILING_LOG_DEBUG("offsetx: %d", this->attrInfo.offsetx);
    TILING_LOG_DEBUG("groups: %d", this->attrInfo.groups);
    TILING_LOG_DEBUG("roundMode: %d", this->attrInfo.roundMode);
    return;
}

bool Conv3dTiling::CheckFeaMapShape()
{
    if (shapeInfo.orgHi <= 0 || shapeInfo.orgWi <= 0 || shapeInfo.orgDi <= 0) {
        TILING_LOG_ERROR("Input illegal orgHi: %ld, orgWi: %ld, orgDi: %ld, which must > 0.",
            shapeInfo.orgHi, shapeInfo.orgWi, shapeInfo.orgDi);
        return false;
    }
    if (attrInfo.strideH == 0 || attrInfo.strideW == 0 || attrInfo.strideD == 0) {
        TILING_LOG_ERROR("div zero when calculating output shape, strideH: %d, strideW: %d, strideD: %d",
            attrInfo.strideH, attrInfo.strideW, attrInfo.strideD);
        return false;
    }

    shapeInfo.orgHo = (shapeInfo.orgHi + attrInfo.padTop + attrInfo.padBottom -
                       attrInfo.dilationH * (shapeInfo.orgkH - 1) - 1) / attrInfo.strideH + 1;
    shapeInfo.orgWo = (shapeInfo.orgWi + attrInfo.padLeft + attrInfo.padRight -
                       attrInfo.dilationW * (shapeInfo.orgkW - 1) - 1) / attrInfo.strideW + 1;
    shapeInfo.orgDo = (shapeInfo.orgDi + attrInfo.padHead + attrInfo.padTail -
                       attrInfo.dilationD * (shapeInfo.orgkD - 1) - 1) / attrInfo.strideD + 1;
    if (shapeInfo.orgHo <= 0 || shapeInfo.orgWo <= 0 || shapeInfo.orgDo <= 0) {
        TILING_LOG_ERROR("Calculated output orgHo: %ld, orgWo: %ld, orgDo: %ld, which must > 0",
            shapeInfo.orgHo, shapeInfo.orgWo, shapeInfo.orgDo);
        return false;
    }

    if ((outputOrder == static_cast<int8_t>(OutputOrder::M)) && (shapeInfo.singleM <= 0)) {
        TILING_LOG_ERROR("Input illegal singleM: %ld, which must > 0.", shapeInfo.singleM);
        return false;
    }

    int64_t ciPerg = this->optGroupFlag ? shapeInfo.singleCi / shapeInfo.enlarge : shapeInfo.singleCi;
    if (ciPerg * attrInfo.groups != shapeInfo.orgCi) {
        TILING_LOG_ERROR("only support groups * singleCi = orgCi, current groups: %d, singleCi: %ld, orgCi: %ld",
                         attrInfo.groups, ciPerg, shapeInfo.orgCi);
        return false;
    }
    if (shapeInfo.singleDo <= 0) {
        TILING_LOG_ERROR("Input illegal singleDo: %ld, which must > 0.", shapeInfo.singleDo);
        return false;
    }
    if (shapeInfo.orgCi <= 0 || static_cast<uint64_t>(shapeInfo.orgCi) > MAX_31_BIT_NUM) {
        TILING_LOG_ERROR("Input illegal orgCi: %ld, which must in range [1, %lu].", shapeInfo.orgCi, MAX_31_BIT_NUM);
        return false;
    }

    return true;
}

bool Conv3dTiling::CheckWeightShape()
{
    if (shapeInfo.orgkH <= 0 || static_cast<uint64_t>(shapeInfo.orgkH) > MAX_31_BIT_NUM) {
        TILING_LOG_ERROR("Input illegal kH: %ld, which must in range [1, %lu].", shapeInfo.orgkH, MAX_31_BIT_NUM);
        return false;
    }
    if (shapeInfo.orgkW <= 0 || static_cast<uint64_t>(shapeInfo.orgkW) > MAX_31_BIT_NUM) {
        TILING_LOG_ERROR("Input illegal kW: %ld, which must in range [1, %lu].", shapeInfo.orgkW, MAX_31_BIT_NUM);
        return false;
    }
    if (shapeInfo.orgkD <= 0 || static_cast<uint64_t>(shapeInfo.orgkD) > MAX_31_BIT_NUM) {
        TILING_LOG_ERROR("Input illegal kD: %ld, which must in range [1, %lu].", shapeInfo.orgkD, MAX_31_BIT_NUM);
        return false;
    }
    if (shapeInfo.orgCo <= 0 || static_cast<uint64_t>(shapeInfo.orgCo) > MAX_31_BIT_NUM) {
        TILING_LOG_ERROR("Input illegal orgCo: %ld, which must in range [1, %lu].", shapeInfo.orgCo, MAX_31_BIT_NUM);
        return false;
    }
    if (shapeInfo.singleCo <= 0 || static_cast<uint64_t>(shapeInfo.singleCo) > MAX_31_BIT_NUM) {
        TILING_LOG_ERROR("Input illegal SingleCo: %ld, which must in range [1, %lu].",
            shapeInfo.singleCo, MAX_31_BIT_NUM);
        return false;
    }

    if (shapeInfo.singlekH != shapeInfo.orgkH) {
        TILING_LOG_ERROR("Only support singlekH = orgkH, current singlekH: %ld, orgkH: %ld,",
            shapeInfo.singlekH, shapeInfo.orgkH);
        return false;
    }
    if (shapeInfo.singlekW != shapeInfo.orgkW) {
        TILING_LOG_ERROR("Only support singlekW = orgkW, current singlekW: %ld, orgkW: %ld",
            shapeInfo.singlekW, shapeInfo.orgkW);
        return false;
    }
    if (shapeInfo.singlekD != shapeInfo.orgkD) {
        TILING_LOG_ERROR("Only support singlekD = orgkD, current singlekD: %ld, orgkD: %ld",
            shapeInfo.singlekD, shapeInfo.orgkD);
        return false;
    }

    return true;
}

bool Conv3dTiling::CheckInputShape()
{
    if (!CheckFeaMapShape()) {
        return false;
    }
    if (!CheckWeightShape()) {
        return false;
    }

    return true;
}

bool Conv3dTiling::CheckSoc()
{
    static unordered_set<platform_ascendc::SocVersion> supportedSoc = {platform_ascendc::SocVersion::ASCEND910B,
                                                                       platform_ascendc::SocVersion::ASCEND910_95,
                                                                       platform_ascendc::SocVersion::ASCEND910_55};
    if (supportedSoc.find(this->platformInfo.socVersion) == supportedSoc.end()) {
        TILING_LOG_ERROR("current Soc Version is not support");
        return false;
    }
    return true;
}

bool Conv3dTiling::CheckInputParam()
{
    if (!CheckSoc()) {
        return false;
    }
    if (!CheckAlgorithmLimit()) {
        return false;
    }
    if (!CheckAttr()) {
        return false;
    }
    if (!CheckInputShape()) {
        return false;
    }
    if (!CheckInputFormat()) {
        return false;
    }
    if (!CheckDtype()) {
        return false;
    }
    if (!CheckInstructionLimits()) {
        return false;
    }
    return true;
}

bool Conv3dTiling::CheckAlgorithmLimit()
{
    if (isC04Flag) {
        TILING_LOG_ERROR("conv3d temporarily unSupport C04 format.");
        return false;
    }
    return true;
}

bool Conv3dTiling::CheckAttr()
{
    if (!CheckPadStrideDilation()) {
        return false;
    }
    if (!CheckQuantUniqueAttr()) {
        return false;
    }
    if (!CheckGroups()) {
        return false;
    }
    return true;
}

bool Conv3dTiling::CheckPadStrideDilation()
{
    bool padInvalidFlag = (this->attrInfo.padLeft < 0 || this->attrInfo.padRight < 0 ||
        this->attrInfo.padTop < 0 || this->attrInfo.padBottom < 0 ||
        this->attrInfo.padHead < 0 || this->attrInfo.padTail < 0);
    if (padInvalidFlag) {
        TILING_LOG_ERROR(
            "Illlegal attrs have set: padTop=%d, padBottom=%d, padLeft=%d, padRight=%d, padHead=%d, padTail=%d,\
            which must >= 0.", this->attrInfo.padTop, this->attrInfo.padBottom, this->attrInfo.padLeft,
            this->attrInfo.padRight, this->attrInfo.padHead, this->attrInfo.padTail);
        return false;
    }

    if (this->attrInfo.strideH <= 0 || this->attrInfo.strideW <= 0 || this->attrInfo.strideD <= 0) {
        TILING_LOG_ERROR("Illegal attrs have set: strideH=%d, strideW=%d, strideD=%d, which must > 0.",
                         this->attrInfo.strideH, this->attrInfo.strideW, this->attrInfo.strideD);
        return false;
    }

    if (this->attrInfo.dilationH <= 0 || this->attrInfo.dilationW <= 0 || this->attrInfo.dilationD <= 0) {
        TILING_LOG_ERROR("Illegal attrs have set: dilationH=%d, dilationW=%d, dilationD=%d, which must > 0.",
                         this->attrInfo.dilationH, this->attrInfo.dilationW, this->attrInfo.dilationD);
        return false;
    }

    if (attrInfo.groups <= 0) {
        TILING_LOG_ERROR("Illegal attrs have set: groups=%d which must > 0.", attrInfo.groups);
        return false;
    }

    if (this->optGroupFlag) {
        if (shapeInfo.singleGroups <= 0) {
            TILING_LOG_ERROR("Illegal attrs have set: singleGroups=%ld which must > 0.", shapeInfo.singleGroups);
            return false;
        }

        if (shapeInfo.enlarge <= 0) {
            TILING_LOG_ERROR("Illegal attrs have set: enlarge=%d which must > 0.", shapeInfo.enlarge);
            return false;
        }

        if (shapeInfo.singleGroupOpt <= 0) {
            TILING_LOG_ERROR("Input illegal singleGroupOpt: %ld, which must > 0.", shapeInfo.singleGroupOpt);
            return false;
        }
    }

    return true;
}

bool Conv3dTiling::CheckDataCopyLimits()
{
    if (descInfo.fMapType.format == ConvFormat::NCDHW) {
        uint64_t loadAL1loop1SrcStrideLimits = MAX_40_BIT_NUM;
        int64_t tmpOrgDi = (shapeInfo.orgDi <= 0) ? 1 : shapeInfo.orgDi;
        uint64_t loadAL1loop1SrcStride = shapeInfo.orgHi * shapeInfo.orgWi * tmpOrgDi *
                                        DTYPE_SIZE_TAB.at(descInfo.fMapType.dtype);
        if (loadAL1loop1SrcStride > loadAL1loop1SrcStrideLimits) {
            TILING_LOG_ERROR(
                "Fmap shape not satisfy DataCopy's limits: din(%ld)*hin(%ld)*win(%ld)*typesize(%u)=%lu, must <= %lu",
                tmpOrgDi, shapeInfo.orgHi, shapeInfo.orgWi,
                DTYPE_SIZE_TAB.at(descInfo.fMapType.dtype),
                loadAL1loop1SrcStride, loadAL1loop1SrcStrideLimits);
            return false;
        }
    }
    if (descInfo.fMapType.format == ConvFormat::NDHWC && outputOrder == static_cast<int8_t>(OutputOrder::M)) {
        uint64_t loadAL1SrcNdMatixStride = shapeInfo.orgHi * shapeInfo.orgWi * shapeInfo.orgCi * attrInfo.dilationD;
        if (loadAL1SrcNdMatixStride > MAX_40_BIT_NUM) {
            TILING_LOG_ERROR(
                "Fmap shape not satisfy DataCopy's limits: cin(%ld)*hin(%ld)*win(%ld)*dilationD(%d)=%lu, must <= %lu",
                shapeInfo.orgCi, shapeInfo.orgHi, shapeInfo.orgWi,
                attrInfo.dilationD,
                loadAL1SrcNdMatixStride, MAX_40_BIT_NUM);
            return false;
        }
    }
    return true;
}

bool Conv3dTiling::CheckFixpipeLimits()
{
    if (descInfo.fMapType.format == ConvFormat::NCDHW) {
        uint64_t fixpipeLoop2DstStrideLimit = MAX_32_BIT_NUM;
        int64_t tmpOrgDo = (shapeInfo.orgDo <= 0) ? 1 : shapeInfo.orgDo;
        uint64_t fixpipeLoop2DstStride = static_cast<uint64_t>(shapeInfo.orgHo) * shapeInfo.orgWo * tmpOrgDo;
        if (fixpipeLoop2DstStride > fixpipeLoop2DstStrideLimit) {
            TILING_LOG_ERROR(
                "Output shape not satisfy Fixpipe's limits: dout(%ld)*hout(%ld)*wout(%ld)=%lu, must <= %lu",
                tmpOrgDo, shapeInfo.orgHo, shapeInfo.orgWo,
                fixpipeLoop2DstStride, fixpipeLoop2DstStrideLimit);
            return false;
        }
    }
    if (descInfo.fMapType.format == ConvFormat::NDHWC && outputOrder == static_cast<int8_t>(OutputOrder::HW)) {
        uint64_t fixpipeLoop3DstStride = shapeInfo.orgCo * shapeInfo.orgWo;
        if (fixpipeLoop3DstStride > MAX_32_BIT_NUM) {
            TILING_LOG_ERROR(
                "Output shape not satisfy Fixpipe's limits: cout(%ld)*wout(%ld)=%lu, must <= %lu",
                shapeInfo.orgCo, shapeInfo.orgWo,
                fixpipeLoop3DstStride, MAX_32_BIT_NUM);
            return false;
        }
    }
    return true;
}

bool Conv3dTiling::CheckInstructionLimits()
{
    if (!CheckLoad3DLimits()) {
        return false;
    }
    if (!CheckDataCopyLimits()) {
        return false;
    }
    if (!CheckFixpipeLimits()) {
        return false;
    }
    return true;
}

void Conv3dTiling::SetOrgWeightShape(int64_t orgCo, int64_t orgKd, int64_t orgKh, int64_t orgKw)
{
    this->shapeInfo.orgkH = orgKh;
    this->shapeInfo.orgkW = orgKw;
    this->shapeInfo.orgkD = orgKd;
    this->shapeInfo.orgCo = orgCo;
}

void Conv3dTiling::SetSingleWeightShape(int64_t singleCi, int64_t singleKd,
    int64_t singleKh, int64_t singleKw)
{
    this->shapeInfo.singlekH = singleKh;
    this->shapeInfo.singlekW = singleKw;
    this->shapeInfo.singlekD = singleKd;
    this->shapeInfo.singleCi = singleCi;
}

void Conv3dTiling::SetOrgFmapShape(int64_t orgCi, int64_t orgDi, int64_t orgHi, int64_t orgWi)
{
    this->shapeInfo.orgCi = orgCi;
    this->shapeInfo.orgHi = orgHi;
    this->shapeInfo.orgWi = orgWi;
    this->shapeInfo.orgDi = orgDi;
}

void Conv3dTiling::SetSingleOutputShape(int64_t singleCo, int64_t singleDo, int64_t singleM, int64_t singleBatch)
{
    this->shapeInfo.singleCo = singleCo;
    this->shapeInfo.singleDo = singleDo;
    this->shapeInfo.singleM = singleM;
    this->shapeInfo.singleBatch = singleBatch;
}

void Conv3dTiling::SetSingleOutputShape(int64_t singleCo, int64_t singleDo, int64_t singleHo, int64_t singleWo, int64_t singleBatch)
{
    this->shapeInfo.singleCo = singleCo;
    this->shapeInfo.singleDo = singleDo;
    this->shapeInfo.singleHo = singleHo;
    this->shapeInfo.singleWo = singleWo;
    this->shapeInfo.singleBatch = singleBatch;
}

void Conv3dTiling::SetWeightType(TPosition pos, ConvFormat format, ConvDtype dtype)
{
    this->descInfo.weightType.pos = pos;
    this->descInfo.weightType.format = format;
    this->descInfo.weightType.dtype = dtype;
}

void Conv3dTiling::SetFmapType(TPosition pos, ConvFormat format, ConvDtype dtype)
{
    this->descInfo.fMapType.pos = pos;
    this->descInfo.fMapType.format = format;
    this->descInfo.fMapType.dtype = dtype;
}

void Conv3dTiling::SetPadding(int64_t padHead, int64_t padTail, int64_t padTop, int64_t padBottom,
    int64_t padLeft, int64_t padRight)
{
    this->attrInfo.padHead = padHead;
    this->attrInfo.padTail = padTail;
    this->attrInfo.padTop = padTop;
    this->attrInfo.padBottom = padBottom;
    this->attrInfo.padLeft = padLeft;
    this->attrInfo.padRight = padRight;
}

void Conv3dTiling::SetDilation(int64_t dilationH, int64_t dilationW, int64_t dilationD)
{
    this->attrInfo.dilationH = dilationH;
    this->attrInfo.dilationW = dilationW;
    this->attrInfo.dilationD = dilationD;
}

void Conv3dTiling::SetStride(int64_t strideH, int64_t strideW, int64_t strideD)
{
    this->attrInfo.strideH = strideH;
    this->attrInfo.strideW = strideW;
    this->attrInfo.strideD = strideD;
}

void Conv3dTiling::SetBiasType(TPosition pos, ConvFormat format, ConvDtype dtype)
{
    this->hasBias = true;
    this->descInfo.biasType.pos = pos;
    this->descInfo.biasType.format = format;
    this->descInfo.biasType.dtype = dtype;
}

void Conv3dTiling::SetOutputType(TPosition pos, ConvFormat format, ConvDtype dtype)
{
    this->descInfo.outputType.pos = pos;
    this->descInfo.outputType.format = format;
    this->descInfo.outputType.dtype = dtype;
}

void Conv3dTiling::SetGroups(int32_t groups)
{
    attrInfo.groups = groups;
}

void Conv3dTiling::SetOptGroupParams(int32_t enlarge, int64_t singleGroups, int64_t singleGroupOpt)
{
    this->optGroupFlag = true;
    shapeInfo.enlarge = enlarge;
    shapeInfo.singleGroups = singleGroups;
    shapeInfo.singleGroupOpt = singleGroupOpt;
}

void Conv3dTiling::CalcOptGroupParams(const optiling::conv_ops_tiling::ConvOriGroupInfo& oriGroupInfo,
                                      optiling::conv_ops_tiling::ConvOptGroupInfo& optGroupInfo) const
{
    if (oriGroupInfo.groups <= 0) {
        TILING_LOG_ERROR("Illegal params : groups=%lu which must > 0.", oriGroupInfo.groups);
        return;
    }

    if (oriGroupInfo.ciPerGroup <= 0) {
        TILING_LOG_ERROR("Illegal params : ciPerGroup=%lu which must > 0.", oriGroupInfo.ciPerGroup);
        return;
    }

    if (oriGroupInfo.coPerGroup <= 0) {
        TILING_LOG_ERROR("Illegal params : coPerGroup=%lu which must > 0.", oriGroupInfo.coPerGroup);
        return;
    }

    uint32_t k0 = CUBE_MKN_TAB.GetMKN(oriGroupInfo.weightDtype, MKN_K_INDEX);
    uint32_t n0 = CUBE_MKN_TAB.GetMKN(oriGroupInfo.weightDtype, MKN_N_INDEX);

    optGroupInfo.enlarge = std::min(Lcm(Lcm(oriGroupInfo.ciPerGroup, k0) / oriGroupInfo.ciPerGroup,
        Lcm(oriGroupInfo.coPerGroup, n0) / oriGroupInfo.coPerGroup), static_cast<uint64_t>(oriGroupInfo.groups));
    optGroupInfo.groupOpt = CeilDiv(oriGroupInfo.groups, optGroupInfo.enlarge);
    optGroupInfo.cinOpt = oriGroupInfo.ciPerGroup * optGroupInfo.enlarge;
    optGroupInfo.coutOpt = oriGroupInfo.coPerGroup * optGroupInfo.enlarge;
}

void Conv3dTiling::SetHF32(bool hf32EnableFlag, bool hf32TransModeFlag = false)
{
    this->hf32Enable = hf32EnableFlag;
    this->hf32TransMode = hf32TransModeFlag;
}

void Conv3dTiling::SetQuantConvFlag(bool quantConvEnable)
{
    this->quantConvFlag = quantConvEnable;
}

void Conv3dTiling::SetScaleType(TPosition pos, ConvFormat format, ConvDtype dtype)
{
    this->hasScale = true;
    this->descInfo.scaleType.pos = pos;
    this->descInfo.scaleType.dtype = dtype;
    this->descInfo.scaleType.format = format;
}

void Conv3dTiling::SetFixpipeParams(const optiling::conv_ops_tiling::FixpipeInfo& fixpipeInfo)
{
    shapeInfo.quantMode0 = fixpipeInfo.quantMode0;
    shapeInfo.reluMode0 = fixpipeInfo.reluMode0;
    shapeInfo.clipMode0 = fixpipeInfo.clipMode0;
    shapeInfo.quantMode1 = fixpipeInfo.quantMode1;
    shapeInfo.reluMode1 = fixpipeInfo.reluMode1;
    shapeInfo.clipMode1 = fixpipeInfo.clipMode1;
    shapeInfo.dualOutput = fixpipeInfo.dualOutput;
    shapeInfo.channelWiseCoeff = fixpipeInfo.channelWiseCoeff;
}

void Conv3dTiling::SetOffsetx(int8_t offsetx)
{
    attrInfo.offsetx = offsetx;
}

void Conv3dTiling::SetRoundMode(int8_t roundMode)
{
    attrInfo.roundMode = roundMode;
}

void Conv3dTiling::SetShape(optiling::conv_ops_tiling::ConvAscendcTilingFlag flagInfo,
                            optiling::conv_ops_tiling::ConvAscendcShapesInfo convShapeInfo,
                            optiling::conv_ops_tiling::ConvOpsConstParams convOpsConstParams,
                            optiling::conv_ops_tiling::BlockDimRes blockDimRes)
{
    SetOrgWeightShape(static_cast<int64_t>(convShapeInfo.co), static_cast<int64_t>(convShapeInfo.kd),
                                       static_cast<int64_t>(convShapeInfo.kh), static_cast<int64_t>(convShapeInfo.kw));
    SetOrgFmapShape(static_cast<int64_t>(convShapeInfo.ci), static_cast<int64_t>(convShapeInfo.di),
                                     static_cast<int64_t>(convShapeInfo.hi), static_cast<int64_t>(convShapeInfo.wi));

    uint64_t singleCoreCi = convShapeInfo.ci;                          
    SetSingleWeightShape(static_cast<int64_t>(singleCoreCi), static_cast<int64_t>(convShapeInfo.kd),
                                          static_cast<int64_t>(convShapeInfo.kh), static_cast<int64_t>(convShapeInfo.kw));

    uint64_t curCo = convShapeInfo.co;
    int64_t singleCoreCo = ConvCeilDiv(ConvAlignB(curCo, convOpsConstParams.n0), blockDimRes.nDim);
    int64_t singleCoreDo = ConvCeilDiv(convShapeInfo.dout, blockDimRes.doDim);
    int64_t singleCoreBatch = ConvCeilDiv(convShapeInfo.batch, blockDimRes.batchDim);
    int64_t singleCoreHo = 0;
    int64_t singleCoreMo = 0;
    if (flagInfo.mSplitModeFlag) {
        singleCoreMo = ConvCeilDiv(ConvAlignB(convShapeInfo.ho * convShapeInfo.wo, convOpsConstParams.m0), blockDimRes.mDim);
        SetSingleOutputShape(singleCoreCo, singleCoreDo, singleCoreMo, singleCoreBatch);
    } else {
        singleCoreHo = ConvCeilDiv(convShapeInfo.ho, blockDimRes.hoDim);
        SetSingleOutputShape(singleCoreCo, singleCoreDo, singleCoreHo,
            static_cast<int64_t>(convShapeInfo.wo), singleCoreBatch);
    }
}

int64_t Conv3dTiling::GetTilingData(optiling::conv_ops_tiling::ConvAscendcAttrInfo convAttrInfo, 
                                    optiling::conv_ops_tiling::ConvAscendcDescInfo convDescInfo, 
                                    optiling::conv_ops_tiling::ConvAscendcTilingFlag flagInfo,
                                    optiling::conv_ops_tiling::ConvAscendcShapesInfo convShapeInfo,
                                    optiling::conv_ops_tiling::ConvOpsConstParams convOpsConstParams,
                                    optiling::conv_ops_tiling::BlockDimRes blockDimRes,
                                    Ops::NN::Conv3dV2::Conv3DV2TilingData& tilingData)
{
    SetShape(flagInfo, convShapeInfo, convOpsConstParams, blockDimRes);
    bool hf32TransModeEnable = false;
    bool isHF32 = (convAttrInfo.hf32Mode == 1);
    if (isHF32) {
        SetHF32(isHF32, hf32TransModeEnable);
    }

    int8_t outputOrderFlag = flagInfo.mSplitModeFlag ? 1: 0;
    SetOutputOrder(outputOrderFlag);
    SetPadding(static_cast<int64_t>(convAttrInfo.padHead), static_cast<int64_t>(convAttrInfo.padTail),
                                static_cast<int64_t>(convAttrInfo.padTop), static_cast<int64_t>(convAttrInfo.padBottom),
                                static_cast<int64_t>(convAttrInfo.padLeft), static_cast<int64_t>(convAttrInfo.padRight));
    SetDilation(static_cast<int64_t>(convAttrInfo.dilationH), static_cast<int64_t>(convAttrInfo.dilationW),
                                 static_cast<int64_t>(convAttrInfo.dilationD));
    SetStride(static_cast<int64_t>(convAttrInfo.strideH), static_cast<int64_t>(convAttrInfo.strideW),
                               static_cast<int64_t>(convAttrInfo.strideD));
    SetGroups(static_cast<int32_t>(convAttrInfo.groups));

    SetWeightType(TPosition::GM, optiling::conv_ops_tiling::formatMap[convDescInfo.weightFormat],
                                   optiling::conv_ops_tiling::dtypeMap[convDescInfo.weightDtype]);
    SetFmapType(TPosition::GM, optiling::conv_ops_tiling::formatMap[convDescInfo.fMapFormat],
                                 optiling::conv_ops_tiling::dtypeMap[convDescInfo.fMapDtype]);
    SetOutputType(TPosition::CO1, optiling::conv_ops_tiling::formatMap[convDescInfo.outFormat],
                                   optiling::conv_ops_tiling::dtypeMap[convDescInfo.outDtype]);
    SetQuantConvFlag(flagInfo.quantFlag);
    SetOffsetx(static_cast<int8_t>(convAttrInfo.offsetx));
    SetRoundMode(static_cast<int8_t>(convAttrInfo.roundMode));

    if (flagInfo.hasBias) {
        SetBiasType(TPosition::GM, optiling::conv_ops_tiling::formatMap[convDescInfo.biasFormat],
                                     optiling::conv_ops_tiling::dtypeMap[convDescInfo.biasDtype]);
    }

    if (GetTiling(tilingData.conv3dApiTiling) == -1) {
        return -1;
    }

    return 0;
}

} // namespace conv_tiling