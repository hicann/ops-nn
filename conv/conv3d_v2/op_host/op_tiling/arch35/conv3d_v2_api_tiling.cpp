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

using namespace conv_tiling_algo_m;
using namespace conv_tiling_algo_hw;
using namespace std;

namespace conv_tiling {
namespace {
constexpr int64_t RET_FAIL = -1;
}

int64_t Conv3dTiling::GetTiling(optiling::TConv3DTiling &tiling)
{
    if (!CheckInputParam()) {
        TILING_LOG_ERROR("conv3d api tiling check params failed.");
        return RET_FAIL;
    }
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

void Conv3dTiling::SetOutputOrder(int8_t outputOrder)
{
    this->outputOrder = outputOrder;
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

uint32_t Conv3dTiling::CalcAL1SpaceSize(optiling::TConv3DTiling& tiling)
{
    uint64_t aL1SpaceSize = 0;
    uint64_t fmapSize = DTYPE_SIZE_TAB.at(descInfo.fMapType.dtype);

    uint64_t dilatedKernelH = (tiling.get_kernelH() - 1) * tiling.get_dilationH() + 1;
    if (outputOrder == static_cast<int8_t>(OutputOrder::M)) {
        uint64_t mL1Max = tiling.get_hoL1() < tiling.get_singleCoreHo() ? tiling.get_hoL1() : tiling.get_singleCoreHo();
        uint64_t hoL1Max = std::min(mL1Max / tiling.get_orgWo() + CONST_VALUE_2, tiling.get_orgHo());
        uint64_t hiAL1Max = (hoL1Max - 1) * tiling.get_strideH() + dilatedKernelH;
        hiAL1Max = hiAL1Max > tiling.get_orgHi() ? tiling.get_orgHi() : hiAL1Max;

        aL1SpaceSize = tiling.get_cinAInCore() * hiAL1Max * tiling.get_orgWi();
    } else {
        uint64_t hiAL1Max = (tiling.get_hoL1() - 1) * tiling.get_strideH() + dilatedKernelH;
        hiAL1Max = hiAL1Max > tiling.get_orgHi() ? tiling.get_orgHi() : hiAL1Max;

        uint64_t wiAL1Max = 0;
        if (isC04Flag && tiling.get_singleCoreWo() == tiling.get_woL1()) {
            wiAL1Max = tiling.get_orgWi();

            aL1SpaceSize = AlignB(hiAL1Max * wiAL1Max, C0_SIZE / (fmapSize * C04_CIN_SIZE)) * C04_CIN_SIZE;
        } else {
            uint64_t dilatedKernelW = (tiling.get_kernelW() - 1) * tiling.get_dilationW() + 1;
            wiAL1Max = (tiling.get_woL1() - 1) * tiling.get_strideW() + dilatedKernelW;
            wiAL1Max = wiAL1Max > tiling.get_orgWi() ? tiling.get_orgWi() : wiAL1Max;

            aL1SpaceSize = tiling.get_cinAInCore() * hiAL1Max * wiAL1Max;
        }
    }
    aL1SpaceSize = AlignB(aL1SpaceSize * fmapSize, C0_SIZE);

    return static_cast<uint32_t>(aL1SpaceSize);
}

void Conv3dTiling::SetScalarParams(optiling::TConv3DTiling& tiling)
{
    // calculate the follow params in tiling process, for scalar optimization in kernel
    uint32_t kernelHxkernelW = tiling.get_kernelH() * tiling.get_kernelW();
    uint32_t cinAInCore = tiling.get_kAL1() / kernelHxkernelW;
    uint32_t kAL1Tail = (AlignB(tiling.get_singleCoreCi(), cubeInfo.k0) * tiling.get_kernelD() * kernelHxkernelW) %
                        tiling.get_kAL1();
    kAL1Tail = kAL1Tail == 0 ? tiling.get_kAL1() : kAL1Tail;
    uint32_t kBL1Tail = (tiling.get_singleCoreCi() * kernelHxkernelW) % tiling.get_kBL1();
    kBL1Tail = kBL1Tail == 0 ? tiling.get_kBL1() : kBL1Tail;
    uint32_t cinATailInCore = kAL1Tail / kernelHxkernelW;
    uint32_t cinBTailInCore = kBL1Tail / kernelHxkernelW;
    uint64_t orgHixWi = tiling.get_orgHi() * tiling.get_orgWi();
    uint32_t cinOffsetBlockInGM = tiling.get_kAL1() / kernelHxkernelW * orgHixWi;
    uint32_t mStep = 0;
    if (outputOrder == static_cast<int8_t>(OutputOrder::M)) {
        mStep = AlignB(tiling.get_hoL0(), cubeInfo.m0);
    } else {
        mStep = AlignB(tiling.get_hoL0() * tiling.get_woL0(), cubeInfo.m0);
    }
    uint32_t fmapKStride = mStep / cubeInfo.m0;
    uint32_t nStep = CeilDiv(tiling.get_nL0(), cubeInfo.n0);
    uint32_t kStep = tiling.get_kL0() / cubeInfo.k0;
    uint32_t weightKStride = CeilDiv(tiling.get_nBL1(), cubeInfo.n0);
    uint32_t coutOffsetBlock = (tiling.get_orgCi() / tiling.get_groups()) * kernelHxkernelW;
    uint32_t cinBInCore = tiling.get_kBL1() / kernelHxkernelW;
    uint32_t nL1DivBlockSize = tiling.get_nBL1() / cubeInfo.n0;

    tiling.set_kernelHxkernelW(kernelHxkernelW);
    tiling.set_kernelHxkernelWxkernelD(tiling.get_kernelD() * kernelHxkernelW);
    // l0TilingInfo.nL0 != 0 is checked before
    tiling.set_multiNBL1(static_cast<uint32_t>(CeilDiv(tiling.get_nBL1(), tiling.get_nL0())));
    tiling.set_cinAInCore(cinAInCore);
    tiling.set_cinATailInCore(cinATailInCore);
    tiling.set_orgHixWi(orgHixWi);
    tiling.set_cinOffsetBlockInGM(cinOffsetBlockInGM);
    tiling.set_mStep(mStep);
    tiling.set_fmapKStride(fmapKStride);
    tiling.set_nStep(nStep);
    tiling.set_kStep(kStep);
    tiling.set_weightKStride(weightKStride);
    tiling.set_coutOffsetBlock(coutOffsetBlock);
    tiling.set_cinBInCore(cinBInCore);
    tiling.set_cinBTailInCore(cinBTailInCore);
    tiling.set_nL1DivBlockSize(nL1DivBlockSize);
    tiling.set_aL1SpaceSize(CalcAL1SpaceSize(tiling));
}

void Conv3dTiling::SetAttrsTilingData(optiling::TConv3DTiling& tiling)
{
    tiling.set_strideH(static_cast<uint32_t>(this->attrInfo.strideH));
    tiling.set_strideW(static_cast<uint32_t>(this->attrInfo.strideW));
    tiling.set_strideD(static_cast<uint32_t>(this->attrInfo.strideD));
    tiling.set_dilationH(static_cast<uint32_t>(this->attrInfo.dilationH));
    tiling.set_dilationW(static_cast<uint32_t>(this->attrInfo.dilationW));
    tiling.set_dilationD(static_cast<uint32_t>(this->attrInfo.dilationD));
    tiling.set_padHead(static_cast<uint32_t>(this->attrInfo.padHead));
    tiling.set_padTail(static_cast<uint32_t>(this->attrInfo.padTail));
    tiling.set_padTop(static_cast<uint32_t>(this->attrInfo.padTop));
    tiling.set_padBottom(static_cast<uint32_t>(this->attrInfo.padBottom));
    tiling.set_padLeft(static_cast<uint32_t>(this->attrInfo.padLeft));
    tiling.set_padRight(static_cast<uint32_t>(this->attrInfo.padRight));

    tiling.set_groups(static_cast<uint32_t>(this->attrInfo.groups));

    if (this->optGroupFlag) {
        tiling.set_singleCoreGroups(static_cast<uint32_t>(shapeInfo.singleGroups));
        tiling.set_singleCoreGroupOpt(static_cast<uint32_t>(shapeInfo.singleGroupOpt));
        tiling.set_enlarge(static_cast<uint32_t>(shapeInfo.enlarge));
    }

    tiling.set_offsetx(this->attrInfo.offsetx);
    tiling.set_roundMode(this->attrInfo.roundMode);
}

void Conv3dTiling::SetTilingData(optiling::TConv3DTiling& tiling)
{
    if (outputOrder == static_cast<int8_t>(OutputOrder::M)) {
        tiling.set_singleCoreHo(static_cast<uint64_t>(shapeInfo.singleM));
        tiling.set_hoL1(static_cast<uint32_t>(l1TilingInfo.mAL1));
        tiling.set_hoL0(static_cast<uint32_t>(l0TilingInfo.mL0));
    } else {
        tiling.set_singleCoreHo(static_cast<uint64_t>(shapeInfo.singleHo));
        tiling.set_singleCoreWo(static_cast<uint64_t>(shapeInfo.singleWo));
        tiling.set_hoL1(static_cast<uint32_t>(l1TilingInfo.hoAL1));
        tiling.set_woL1(static_cast<uint32_t>(l1TilingInfo.woAL1));
        tiling.set_hoL0(static_cast<uint32_t>(l0TilingInfo.hoL0));
        tiling.set_woL0(static_cast<uint32_t>(l0TilingInfo.woL0));
    }
    tiling.set_orgDo(static_cast<uint64_t>(this->shapeInfo.orgDo));
    tiling.set_orgHo(static_cast<uint64_t>(this->shapeInfo.orgHo));
    tiling.set_orgWo(static_cast<uint64_t>(this->shapeInfo.orgWo));
    tiling.set_orgDi(static_cast<uint64_t>(this->shapeInfo.orgDi));
    tiling.set_orgHi(static_cast<uint64_t>(this->shapeInfo.orgHi));
    tiling.set_orgWi(static_cast<uint64_t>(this->shapeInfo.orgWi));
    tiling.set_singleCoreBatch(static_cast<uint64_t>(this->shapeInfo.singleBatch));
    tiling.set_singleCoreDo(static_cast<uint64_t>(this->shapeInfo.singleDo));
    tiling.set_singleCoreCi(static_cast<uint32_t>(shapeInfo.singleCi));
    tiling.set_singleCoreCo(static_cast<uint32_t>(this->shapeInfo.singleCo));
    tiling.set_orgCo(static_cast<uint32_t>(this->shapeInfo.orgCo));
    tiling.set_orgCi(static_cast<uint32_t>(this->shapeInfo.orgCi));
    tiling.set_kernelD(static_cast<uint32_t>(this->shapeInfo.orgkD));
    tiling.set_kernelH(static_cast<uint32_t>(this->shapeInfo.orgkH));
    tiling.set_kernelW(static_cast<uint32_t>(this->shapeInfo.orgkW));

    SetAttrsTilingData(tiling);

    tiling.set_kAL1(static_cast<uint32_t>(this->l1TilingInfo.kAL1));
    tiling.set_kBL1(static_cast<uint32_t>(this->l1TilingInfo.kBL1));
    tiling.set_nBL1(static_cast<uint32_t>(this->l1TilingInfo.nBL1));
    tiling.set_kL0(static_cast<uint32_t>(this->l0TilingInfo.kL0));
    tiling.set_nL0(static_cast<uint32_t>(this->l0TilingInfo.nL0));
    tiling.set_pBufferFlag(static_cast<uint32_t>(this->dbValue.pBufferFlag));

    tiling.set_hasBias(static_cast<uint8_t>(this->hasBias));
    tiling.set_hasScale(static_cast<uint8_t>(this->hasQuantScale));
    tiling.set_iterateMNOrder(static_cast<uint8_t>(this->l1TilingInfo.iterateMNOrder));
    tiling.set_biasFullLoadFlag(static_cast<uint8_t>(this->l1TilingInfo.biasFullLoadFlag));
    tiling.set_fixpParamsFullLoadFlag(static_cast<uint8_t>(this->l1TilingInfo.fixpParamsFullLoadFlag));
    tiling.set_hf32Enable(static_cast<uint8_t>(this->hf32Enable));
    tiling.set_hf32TransMode(static_cast<uint8_t>(this->hf32TransMode));
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

void Conv3dTiling::CalcOptGroupParams(const ConvOriGroupInfo& oriGroupInfo, ConvOptGroupInfo& optGroupInfo) const
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

void Conv3dTiling::SetHF32(bool hf32Enable, bool hf32TransMode = false)
{
    this->hf32Enable = hf32Enable;
    this->hf32TransMode = hf32TransMode;
}

void Conv3dTiling::SetQuantScale(bool hasScale)
{
    this->hasQuantScale = hasScale;
    if (hasScale) {
        this->descInfo.quantScaleType.dtype = ConvDtype::INT64;
    }
}

void Conv3dTiling::SetFixpipeParams(const FixpipeInfo& fixpipeInfo)
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

} // namespace conv_tiling