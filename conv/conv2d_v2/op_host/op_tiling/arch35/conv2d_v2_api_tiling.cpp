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
 * \file conv2d_v2_api_tiling.cpp
 * \brief
 */

#include <cstdint>
#include <algorithm>
#include <set>
#include <utility>
#include "common/op_host/op_tiling/arch35/conv_api_tiling_algorithm_HWmode.h"
#include "common/op_host/op_tiling/arch35/conv_api_tiling_algorithm_Mmode.h"
#include "common/op_host/op_tiling/arch35/conv_api_tiling_algorithm_BBmode.h"
#include "conv2d_v2_api_tiling.h"

using namespace conv_tiling_algo_m;
using namespace conv_tiling_algo_hw;
using namespace conv_tiling_algo_bb;
using namespace std;
#define UNUSED __attribute__((unused))

namespace conv_tiling {
int64_t Conv2dTiling::GetTiling(optiling::TConv2DTiling &tiling)
{
    if (!CheckParams()) {
        TILING_LOG_ERROR("conv2d api tiling check params failed.");
        return -1;
    }
    InferCubeInfo();
    Infer5hdShape();

    platformInfo.abL1mte2BandWidthCof = GetBandWidthCof(platformInfo.socVersion);
    int64_t ret = Compute();
    if (ret == -1) {
        TILING_LOG_ERROR("conv2d api tiling compute failed.");
        return -1;
    }

    ret = CheckTilingRes();
    if (ret == -1) {
        TILING_LOG_ERROR("conv2d api tiling check tiling result failed.");
        return -1;
    }

    SetTilingData(tiling);
    PrintTilingData();
    return 0;
}

bool Conv2dTiling::GetTiling(Conv2DBasicBlockInfo& conv2DBasicBlockInfo, optiling::TConv2DTiling &tiling)
{
    if (!CheckTilingAlgorithmType(conv2DBasicBlockInfo, BB_PHASE_2)) {
        return false;
    }

    InferCubeInfo();
    Infer5hdShape();
 
    std::shared_ptr<ConvTilingAlgorithmBBmode> algoBBPtr = std::make_shared<ConvTilingAlgorithmBBmode>(this,
        conv2DBasicBlockInfo);
    int64_t ret = algoBBPtr->Process();
    if (ret == -1) {
        TILING_LOG_ERROR("conv2d api tiling Process failed.");
        return false;
    }
    ret = CheckTilingRes();
    if (ret == -1) {
        TILING_LOG_ERROR("conv2d api tiling check tiling result failed.");
        return false;
    }
    SetTilingData(tiling);
    PrintTilingData();
    return true;
}

void Conv2dTiling::SetDefaultDdim()
{
    shapeInfo.orgkD = 1;
    shapeInfo.orgDi = 1;
    shapeInfo.singleDo = 1;
    shapeInfo.singlekD = 1;
    attrInfo.padHead = 0;
    attrInfo.padTail = 0;
    attrInfo.dilationD = 1;
    attrInfo.strideD = 1;
}

void Conv2dTiling::Infer5hdShape()
{
    shapeInfo.singleCi1 = isC04Flag ? CONST_VALUE_1 : CeilDiv(shapeInfo.singleCi, cubeInfo.k0);
    shapeInfo.singleCo1 = CeilDiv(shapeInfo.singleCo, cubeInfo.n0);
    if (outputOrder == static_cast<int8_t>(OutputOrder::M)) {
        shapeInfo.singleM1 = CeilDiv(shapeInfo.singleM, cubeInfo.m0);
    }
}

int64_t Conv2dTiling::Compute()
{
    SetDefaultDdim();
    switch (outputOrder) {
        case static_cast<int8_t>(OutputOrder::M):
            algoPtr = std::make_shared<ConvTilingAlgorithmMmode>(this);
            break;
        case static_cast<int8_t>(OutputOrder::HW):
            algoPtr = std::make_shared<ConvTilingAlgorithmHWmode>(this);
            break;
        default:
            TILING_LOG_ERROR("Unsupported output order %d, only support Mmode(%d) and HWmode(%d).",
                static_cast<int>(outputOrder), static_cast<int>(OutputOrder::M), static_cast<int>(OutputOrder::HW));
            return -1;
    }
    TILING_LOG_DEBUG("output order: %d", outputOrder);
    int64_t ret = algoPtr->Process();

    return ret;
}

bool Conv2dTiling::CheckParams()
{
    if (!CheckAlgorithmLimit()) {
        return false;
    }
    if (!CheckAttr()) {
        return false;
    }
    if (!CheckShape()) {
        return false;
    }
    if (!CheckFormat()) {
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

bool Conv2dTiling::CheckParamsBBModePhase1()
{
    if (!CheckAlgorithmLimit()) {
        return false;
    }
    if (!CheckAttrBBModePhase1()) {
        return false;
    }
    if (!CheckShapeBBModePhase1()) {
        return false;
    }
    if (!CheckFormat()) {
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

uint64_t Conv2dTiling::CalcWeightUBSize(ConvWeightUbTransParams& params, uint64_t ci1Ub, uint64_t co1Ub)
{
    uint64_t kernelHxkernelW = params.orgkH * params.orgkW;
    uint64_t weightDtypeSize = DTYPE_SIZE_TAB.at(params.weightDtype);

    return (ci1Ub * kernelHxkernelW * params.k0 * co1Ub * params.n0 * WEIGHT_UB_BUFF_NUMS * weightDtypeSize +
            VGATHER_REGISTER_SIZE);
}

void Conv2dTiling::GetWeightUBTiling(ConvWeightUbTransParams& params)
{
    uint64_t kernelHxkernelW = params.orgkH * params.orgkW;

    uint64_t co1BL1 = params.curNBL1 / params.n0;
    std::vector<uint64_t> co1UbRange;
    CalcCommFactor(co1BL1, co1BL1, co1UbRange);
    uint32_t co1UbIndex = 0;
    while (co1UbIndex < co1UbRange.size()) {
        if (CalcWeightUBSize(params, 1, co1UbRange[co1UbIndex]) > platformInfo.ubSize) {
            --co1UbIndex;
            break;
        }

        if (co1UbIndex == co1UbRange.size() - 1) {
            break;
        }

        ++co1UbIndex;
    }
    params.bUbNStep = static_cast<uint32_t>(co1UbRange[co1UbIndex] * params.n0);

    uint64_t ci1BL1 = params.curKBL1 / kernelHxkernelW / params.k0;
    std::vector<uint64_t> ci1UbRange;
    CalcCommFactor(ci1BL1, ci1BL1, ci1UbRange);
    uint32_t ci1UbIndex = 0;
    while (ci1UbIndex < ci1UbRange.size()) {
        if (CalcWeightUBSize(params, ci1UbRange[ci1UbIndex], co1UbRange[co1UbIndex]) > platformInfo.ubSize) {
            --ci1UbIndex;
            break;
        }

        if (ci1UbIndex == ci1UbRange.size() - 1) {
            break;
        }

        ++ci1UbIndex;
    }
    params.bUbKStep = static_cast<uint32_t>(ci1UbRange[ci1UbIndex] * kernelHxkernelW * params.k0);
}

uint64_t Conv2dTiling::CalcDmaUBSize(ConvDmaParams& params, uint64_t khUb, uint64_t kwUb)
{
    uint64_t fmapDtypeSize = DTYPE_SIZE_TAB.at(params.fmapDtype);
 
    return (params.curHoL1 * params.curWoL1 * khUb * kwUb * params.k0 * fmapDtypeSize);
}
 
void Conv2dTiling::GetDmaUbTiling(ConvDmaParams& params)
{
    std::vector<uint64_t> kwUbRange;
    CalcCommFactor(params.kwL1, params.kwL1, kwUbRange);
    uint32_t kwUbIdx = 0;
    while (kwUbIdx < kwUbRange.size()) {
        if (CalcDmaUBSize(params, 1, kwUbRange[kwUbIdx]) > platformInfo.ubSize) {
            --kwUbIdx;
            break;
        }
 
        if (kwUbIdx == kwUbRange.size() - 1) {
            break;
        }
 
        ++kwUbIdx;
    }
    params.kwUb = static_cast<uint32_t>(kwUbRange[kwUbIdx]);
 
    std::vector<uint64_t> khUbRange;
    CalcCommFactor(params.khL1, params.khL1, khUbRange);
    uint32_t khUbIdx = 0;
    while (khUbIdx < khUbRange.size()) {
        if (CalcDmaUBSize(params, khUbRange[khUbIdx], params.kwUb) > platformInfo.ubSize) {
            --khUbIdx;
            break;
        }
 
        if (khUbIdx == khUbRange.size() - 1) {
            break;
        }
 
        ++khUbIdx;
    }
    params.khUb = static_cast<uint32_t>(khUbRange[khUbIdx]);
}

void Conv2dTiling::SetUbTiling(optiling::TConv2DTiling& tiling)
{
    if (isDmaFlag) {
        ConvDmaParams params = {l1TilingInfo.hoAL1, l1TilingInfo.woAL1, l1TilingInfo.khL1,
            l1TilingInfo.kwL1, cubeInfo.k0, descInfo.fMapType.dtype};
        GetDmaUbTiling(params);
        tiling.set_khUb(params.khUb);
        tiling.set_kwUb(params.kwUb);
        return;
    }

    if (isC04Flag) {
        ConvC04Info c04Info = {l1TilingInfo.nBL1, static_cast<uint64_t>(shapeInfo.orgkH),
            static_cast<uint64_t>(shapeInfo.orgkW), cubeInfo.k0, cubeInfo.n0, dbValue.pbBL1, descInfo.weightType.dtype};
        tiling.set_bUbNStep(CalcC04UbLoadNsize(c04Info));
        tiling.set_bUbKStep(0);
        return;
    }

    uint64_t kernelHxkernelW = shapeInfo.orgkH * shapeInfo.orgkW;
    uint64_t weightKSize = AlignB(shapeInfo.orgCi, cubeInfo.k0) * kernelHxkernelW;
    bool weightUbSizeLimit = kernelHxkernelW * cubeInfo.k0 * cubeInfo.n0 * VEC_NUM_PER_CUBE_910D *
        DTYPE_SIZE_TAB.at(descInfo.weightType.dtype) <= (platformInfo.ubSize - VGATHER_REGISTER_SIZE);
    bool weightUbFlag = !this->extendConvFlag && attrInfo.groups == 1 && descInfo.weightType.format == ConvFormat::NCHW &&
        (descInfo.weightType.dtype == ConvDtype::FLOAT16 || descInfo.weightType.dtype == ConvDtype::BFLOAT16 ||
        this->hf32Enable) &&
        !(kernelHxkernelW == 1) && !(kernelHxkernelW % VGATHER_PERF_LIMIT == 0) && l1TilingInfo.kBL1 != weightKSize &&
        dbValue.pbBL1 == DOUBLE_BUFFER_NUM && weightUbSizeLimit;
    if (weightUbFlag) {
        ConvWeightUbTransParams params = {l1TilingInfo.nBL1, l1TilingInfo.kBL1,
            static_cast<uint64_t>(shapeInfo.orgkH), static_cast<uint64_t>(shapeInfo.orgkW),
            cubeInfo.k0, cubeInfo.n0, descInfo.weightType.dtype};
        GetWeightUBTiling(params);
        tiling.set_bUbNStep(params.bUbNStep);
        tiling.set_bUbKStep(params.bUbKStep);
        return;
    }

    tiling.set_bUbNStep(0);
    tiling.set_bUbKStep(0);
}

void Conv2dTiling::SetScalarParams(optiling::TConv2DTiling& tiling)
{
    // calculate the follow params in tiling process, for scalar optimization in kernel
    uint32_t kernelHxkernelW = tiling.get_kernelH() * tiling.get_kernelW();
    uint32_t khL1xhwL1 = tiling.get_khL1() * tiling.get_kwL1();
    uint32_t kernelValueInKSize = isDmaFlag ? khL1xhwL1 : kernelHxkernelW;
    uint32_t cinAInCore = isC04Flag ? C04_CIN_SIZE : tiling.get_kAL1() / kernelValueInKSize;
    uint32_t kAL1Tail = (tiling.get_singleCoreCi() * kernelValueInKSize) % tiling.get_kAL1();
    kAL1Tail = kAL1Tail == 0 ? tiling.get_kAL1() : kAL1Tail;
    uint32_t kBL1Tail = (tiling.get_singleCoreCi() * kernelValueInKSize) % tiling.get_kBL1();
    kBL1Tail = kBL1Tail == 0 ? tiling.get_kBL1() : kBL1Tail;
    uint32_t cinATailInCore = kAL1Tail / kernelValueInKSize;
    uint32_t cinBTailInCore = kBL1Tail / kernelValueInKSize;
    uint64_t orgHixWi = tiling.get_orgHi() * tiling.get_orgWi();
    uint32_t cinOffsetBlockInGM = tiling.get_kAL1() / kernelValueInKSize * orgHixWi;
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
    uint32_t cinBInCore = tiling.get_kBL1() / kernelValueInKSize;
    uint32_t nL1DivBlockSize = tiling.get_nBL1() / cubeInfo.n0;

    tiling.set_kernelHxkernelW(kernelHxkernelW);
    tiling.set_kernelHxkernelWxkernelD(kernelHxkernelW);
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

void Conv2dTiling::SetExtendConv2DParams(optiling::TConv2DTiling& tiling)
{
    // set extendConv2d fixpipe mode to tilingdata
    tiling.set_quantMode0(shapeInfo.quantMode0);
    tiling.set_reluMode0(shapeInfo.reluMode0);
    tiling.set_clipMode0(shapeInfo.clipMode0);
    tiling.set_quantMode1(shapeInfo.quantMode1);
    tiling.set_reluMode1(shapeInfo.reluMode1);
    tiling.set_clipMode1(shapeInfo.clipMode1);
    tiling.set_dualOutput(shapeInfo.dualOutput);
}

void Conv2dTiling::SetAttrsTilingData(optiling::TConv2DTiling& tiling)
{
    tiling.set_strideH(static_cast<uint32_t>(attrInfo.strideH));
    tiling.set_strideW(static_cast<uint32_t>(attrInfo.strideW));
    tiling.set_dilationH(static_cast<uint32_t>(attrInfo.dilationH));
    tiling.set_dilationW(static_cast<uint32_t>(attrInfo.dilationW));
    tiling.set_padTop(static_cast<uint32_t>(attrInfo.padTop));
    tiling.set_padBottom(static_cast<uint32_t>(attrInfo.padBottom));
    tiling.set_padLeft(static_cast<uint32_t>(attrInfo.padLeft));
    tiling.set_padRight(static_cast<uint32_t>(attrInfo.padRight));

    tiling.set_groups(static_cast<uint32_t>(attrInfo.groups));

    if (this->optGroupFlag) {
        tiling.set_singleCoreGroups(static_cast<uint32_t>(shapeInfo.singleGroups));
        tiling.set_singleCoreGroupOpt(static_cast<uint32_t>(shapeInfo.singleGroupOpt));
        tiling.set_enlarge(static_cast<uint32_t>(shapeInfo.enlarge));
    }

    tiling.set_offsetx(attrInfo.offsetx);
    tiling.set_roundMode(attrInfo.roundMode);
}

void Conv2dTiling::SetTilingData(optiling::TConv2DTiling& tiling)
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

    tiling.set_kAL1(static_cast<uint32_t>(l1TilingInfo.kAL1));
    tiling.set_kBL1(static_cast<uint32_t>(l1TilingInfo.kBL1));
    // l0TilingInfo.nL0 != 0 is checked before
    tiling.set_multiNBL1(static_cast<uint32_t>(CeilDiv(l1TilingInfo.nBL1, l0TilingInfo.nL0)));
    tiling.set_nBL1(static_cast<uint32_t>(l1TilingInfo.nBL1));
    tiling.set_kL0(static_cast<uint32_t>(l0TilingInfo.kL0));
    tiling.set_nL0(static_cast<uint32_t>(l0TilingInfo.nL0));

    tiling.set_orgHi(static_cast<uint64_t>(shapeInfo.orgHi));
    tiling.set_orgWi(static_cast<uint64_t>(shapeInfo.orgWi));
    tiling.set_orgHo(static_cast<uint64_t>(shapeInfo.orgHo));
    tiling.set_orgWo(static_cast<uint64_t>(shapeInfo.orgWo));
    tiling.set_singleCoreBatch(static_cast<uint64_t>(shapeInfo.singleBatch));

    tiling.set_orgCi(static_cast<uint32_t>(shapeInfo.orgCi));
    tiling.set_orgCo(static_cast<uint32_t>(shapeInfo.orgCo));
    tiling.set_singleCoreCi(static_cast<uint32_t>(shapeInfo.singleCi));
    tiling.set_singleCoreCo(static_cast<uint32_t>(shapeInfo.singleCo));
    tiling.set_kernelH(static_cast<uint32_t>(shapeInfo.orgkH));
    tiling.set_kernelW(static_cast<uint32_t>(shapeInfo.orgkW));
    tiling.set_khL1(static_cast<uint32_t>(l1TilingInfo.khL1));
    tiling.set_kwL1(static_cast<uint32_t>(l1TilingInfo.kwL1));

    SetAttrsTilingData(tiling);

    tiling.set_pBufferFlag(static_cast<uint32_t>(dbValue.pBufferFlag));
    tiling.set_hf32Enable(this->hf32Enable);
    tiling.set_hf32TransMode(this->hf32TransMode);
    tiling.set_hasBias(static_cast<uint8_t>(hasBias));
    tiling.set_hasScale(static_cast<uint8_t>(hasQuantScale));
    tiling.set_iterateMNOrder(static_cast<uint8_t>(l1TilingInfo.iterateMNOrder));
    tiling.set_biasFullLoadFlag(static_cast<uint8_t>(l1TilingInfo.biasFullLoadFlag));
    tiling.set_fixpParamsFullLoadFlag(static_cast<uint8_t>(l1TilingInfo.fixpParamsFullLoadFlag));
    tiling.set_innerBatch(static_cast<uint32_t>(this->innerBatch));

    SetUbTiling(tiling);
    SetScalarParams(tiling);
    SetExtendConv2DParams(tiling);
}

uint32_t Conv2dTiling::CalcAL1SpaceSize(optiling::TConv2DTiling& tiling)
{
    uint64_t aL1SpaceSize = 0;
    uint64_t fmapSize = DTYPE_SIZE_TAB.at(descInfo.fMapType.dtype);

    if (isDmaFlag) {
        aL1SpaceSize = tiling.get_hoL1() * tiling.get_woL1() * tiling.get_kAL1() * fmapSize;
        return static_cast<uint32_t>(aL1SpaceSize);
    }

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
        if (isC04Flag && tiling.get_orgWo() == tiling.get_woL1()) {
            wiAL1Max = tiling.get_orgWi();

            aL1SpaceSize = AlignB(hiAL1Max * wiAL1Max, C0_SIZE / (fmapSize * C04_CIN_SIZE)) * C04_CIN_SIZE;
        } else {
            uint64_t dilatedKernelW = (tiling.get_kernelW() - 1) * tiling.get_dilationW() + 1;
            wiAL1Max = (tiling.get_woL1() - 1) * tiling.get_strideW() + dilatedKernelW;
            wiAL1Max = wiAL1Max > tiling.get_orgWi() ? tiling.get_orgWi() : wiAL1Max;

            aL1SpaceSize = tiling.get_cinAInCore() * hiAL1Max * wiAL1Max;
        }
    }
    aL1SpaceSize = AlignB(aL1SpaceSize * fmapSize * tiling.get_innerBatch(), C0_SIZE);

    return static_cast<uint32_t>(aL1SpaceSize);
}

void Conv2dTiling::PrintTilingData() const
{
    TILING_LOG_DEBUG("orgHi: %ld", shapeInfo.orgHi);
    TILING_LOG_DEBUG("orgWi: %ld", shapeInfo.orgWi);
    TILING_LOG_DEBUG("orgHo: %ld", shapeInfo.orgHo);
    TILING_LOG_DEBUG("orgWo: %ld", shapeInfo.orgWo);
    if (outputOrder == static_cast<int8_t>(OutputOrder::M)) {
        TILING_LOG_DEBUG("singleCoreM: %ld", shapeInfo.singleM);
        TILING_LOG_DEBUG("mL1: %lu", l1TilingInfo.mAL1);
        TILING_LOG_DEBUG("hoL0: %lu", l0TilingInfo.mL0);
    } else {
        TILING_LOG_DEBUG("singleCoreHo: %ld", shapeInfo.singleHo);
        TILING_LOG_DEBUG("singleCoreWo: %ld", shapeInfo.singleWo);
        TILING_LOG_DEBUG("hoL1: %lu", l1TilingInfo.hoAL1);
        TILING_LOG_DEBUG("woL1: %lu", l1TilingInfo.woAL1);
        TILING_LOG_DEBUG("hoL0: %lu", l0TilingInfo.hoL0);
        TILING_LOG_DEBUG("woL0: %lu", l0TilingInfo.woL0);
    }
    TILING_LOG_DEBUG("groups: %d", attrInfo.groups);
    TILING_LOG_DEBUG("orgCi: %ld", shapeInfo.orgCi);
    TILING_LOG_DEBUG("orgCo: %ld", shapeInfo.orgCo);
    TILING_LOG_DEBUG("kernelH: %ld", shapeInfo.orgkH);
    TILING_LOG_DEBUG("kernelW: %ld", shapeInfo.orgkW);
    TILING_LOG_DEBUG("singleCoreCi: %ld", shapeInfo.singleCi);
    TILING_LOG_DEBUG("singleCoreCo: %ld", shapeInfo.singleCo);
    TILING_LOG_DEBUG("kAL1: %lu", l1TilingInfo.kAL1);
    TILING_LOG_DEBUG("kBL1: %lu", l1TilingInfo.kBL1);
    TILING_LOG_DEBUG("nBL1: %lu", l1TilingInfo.nBL1);
    TILING_LOG_DEBUG("kL0: %lu", l0TilingInfo.kL0);
    TILING_LOG_DEBUG("nL0: %lu", l0TilingInfo.nL0);
    TILING_LOG_DEBUG("pBufferFlag: %lu", dbValue.pBufferFlag);
    TILING_LOG_DEBUG("strideH: %d", attrInfo.strideH);
    TILING_LOG_DEBUG("strideW: %d", attrInfo.strideW);
    TILING_LOG_DEBUG("dilationH: %d", attrInfo.dilationH);
    TILING_LOG_DEBUG("dilationW: %d", attrInfo.dilationW);
    TILING_LOG_DEBUG("padTop: %d", attrInfo.padTop);
    TILING_LOG_DEBUG("padBottom: %d", attrInfo.padBottom);
    TILING_LOG_DEBUG("padLeft: %d", attrInfo.padLeft);
    TILING_LOG_DEBUG("padRight: %d", attrInfo.padRight);
    TILING_LOG_DEBUG("iterateMNOrder: %u", static_cast<uint8_t>(l1TilingInfo.iterateMNOrder));
    TILING_LOG_DEBUG("biasFullLoadFlag: %u", l1TilingInfo.biasFullLoadFlag);
    TILING_LOG_DEBUG("fixpParamsFullLoadFlag: %u", l1TilingInfo.fixpParamsFullLoadFlag);
    TILING_LOG_DEBUG("offsetx: %d", attrInfo.offsetx);
    TILING_LOG_DEBUG("hf32Mode: %u", hf32Enable);
    TILING_LOG_DEBUG("hf32TransMode: %u", hf32TransMode);
    TILING_LOG_DEBUG("enableInnerBatch: %u", enableInnerBatch);
    TILING_LOG_DEBUG("innerBatch: %d", innerBatch);
}

void Conv2dTiling::SetOutputOrder(int8_t outOrder)
{
    this->outputOrder = outOrder;
}

void Conv2dTiling::SetOrgWeightShape(int64_t orgCo, int64_t orgkH, int64_t orgkW)
{
    shapeInfo.orgkH = orgkH;
    shapeInfo.orgkW = orgkW;
    shapeInfo.orgCo = orgCo;
}

void Conv2dTiling::SetSingleWeightShape(int64_t singleCi, int64_t singlekH, int64_t singlekW)
{
    shapeInfo.singlekH = singlekH;
    shapeInfo.singlekW = singlekW;
    shapeInfo.singleCi = singleCi;
}

void Conv2dTiling::SetOrgFmapShape(int64_t orgCi, int64_t orgHi, int64_t orgWi)
{
    shapeInfo.orgCi = orgCi;
    shapeInfo.orgHi = orgHi;
    shapeInfo.orgWi = orgWi;
}

void Conv2dTiling::SetSingleOutputShape(int64_t singleCo, int64_t singleHo, int64_t singleWo, int64_t singleBatch)
{
    shapeInfo.singleCo = singleCo;
    shapeInfo.singleHo = singleHo;
    shapeInfo.singleWo = singleWo;
    shapeInfo.singleBatch = singleBatch;
}

void Conv2dTiling::SetSingleOutputShape(int64_t singleCo, int64_t singleM, int64_t singleBatch)
{
    shapeInfo.singleCo = singleCo;
    shapeInfo.singleM = singleM;
    shapeInfo.singleBatch = singleBatch;
}

void Conv2dTiling::SetWeightType(TPosition pos, ConvFormat format, ConvDtype dtype)
{
    descInfo.weightType.pos = pos;
    descInfo.weightType.format = format;
    descInfo.weightType.dtype = dtype;
}

void Conv2dTiling::SetFmapType(TPosition pos, ConvFormat format, ConvDtype dtype)
{
    descInfo.fMapType.pos = pos;
    descInfo.fMapType.format = format;
    descInfo.fMapType.dtype = dtype;
}

void Conv2dTiling::SetPadding(int32_t padTop, int32_t padBottom, int32_t padLeft, int32_t padRight)
{
    attrInfo.padTop = padTop;
    attrInfo.padBottom = padBottom;
    attrInfo.padLeft = padLeft;
    attrInfo.padRight = padRight;
}

void Conv2dTiling::SetDilation(int32_t dilationH, int32_t dilationW)
{
    attrInfo.dilationH = dilationH;
    attrInfo.dilationW = dilationW;
}

void Conv2dTiling::SetStride(int32_t strideH, int32_t strideW)
{
    attrInfo.strideH = strideH;
    attrInfo.strideW = strideW;
}

void Conv2dTiling::SetBiasType(TPosition pos, ConvFormat format, ConvDtype dtype)
{
    this->hasBias = true;
    descInfo.biasType.pos = pos;
    descInfo.biasType.format = format;
    descInfo.biasType.dtype = dtype;
}

void Conv2dTiling::SetOutputType(TPosition pos, ConvFormat format, ConvDtype dtype)
{
    descInfo.outputType.pos = pos;
    descInfo.outputType.format = format;
    descInfo.outputType.dtype = dtype;
}
 
void Conv2dTiling::SetHF32(bool hf32EnableFlag, bool hf32TransModeFlag = false)
{
    this->hf32Enable = hf32EnableFlag;
    this->hf32TransMode = hf32TransModeFlag;
}

void Conv2dTiling::SetQuantScale(bool hasScale)
{
    this->hasQuantScale = hasScale;
    if (hasScale) {
        this->descInfo.quantScaleType.dtype = ConvDtype::INT64;
    }
}

void Conv2dTiling::SetExtendConvFlag(bool extendConvEnable)
{
    this->extendConvFlag = extendConvEnable;
}

void Conv2dTiling::SetFixpipeParams(FixpipeInfo& fixpipeInfo)
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

void Conv2dTiling::SetOffsetx(int8_t offsetx)
{
    attrInfo.offsetx = offsetx;
}

void Conv2dTiling::SetC04Flag(bool isC04Enable)
{
    this->isC04Flag = isC04Enable;
}

void Conv2dTiling::SetRoundMode(int8_t roundMode)
{
    attrInfo.roundMode = roundMode;
}

void Conv2dTiling::SetGroups(int32_t groups)
{
    attrInfo.groups = groups;
}

void Conv2dTiling::SetOptGroupParams(int32_t enlarge, int64_t singleGroups, int64_t singleGroupOpt)
{
    this->optGroupFlag = true;
    shapeInfo.enlarge = enlarge;
    shapeInfo.singleGroups = singleGroups;
    shapeInfo.singleGroupOpt = singleGroupOpt;
}

void Conv2dTiling::CalcOptGroupParams(const ConvOriGroupInfo& oriGroupInfo, ConvOptGroupInfo& optGroupInfo) const
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

uint64_t Conv2dTiling::CalcC04UbLoadMaxNsize(const ConvC04Info &c04Info) const
{
    uint64_t fullLoadK = AlignB(c04Info.orgkH * c04Info.orgkW * C04_CIN_SIZE, c04Info.k0);
    uint32_t weightDtypeSize = DTYPE_SIZE_TAB.at(c04Info.weightDtype);
    uint64_t nMaxStep = ((platformInfo.ubSize - VGATHER_REGISTER_SIZE) / fullLoadK /
        C04_VEC_USE_BUFF_NUM / weightDtypeSize / c04Info.n0) * c04Info.n0;
    return nMaxStep;
}

uint32_t Conv2dTiling::GetVecNumBySoc() const
{
    if (platformInfo.socVersion == platform_ascendc::SocVersion::ASCEND910_95) {
        return VEC_NUM_PER_CUBE_910D;
    } else {
        return 0;
    }
}

uint64_t Conv2dTiling::CalcC04UbLoadNsize(const ConvC04Info &c04Info) const
{
    if (c04Info.curNBL1 < c04Info.n0) {
        TILING_LOG_DEBUG("Ubsize is not enough to load even a n0 size block, c04 mode cannot be enabled.");
        return 0;
    }

    uint64_t bUbNStep = 0;
    uint64_t nMaxStep = CalcC04UbLoadMaxNsize(c04Info);
    if (c04Info.pbBL1 == 1) {
        bUbNStep = c04Info.curNBL1 > nMaxStep ? nMaxStep : c04Info.curNBL1;
    } else {
        uint32_t vecNum = GetVecNumBySoc();
        if (vecNum == 0) {
            TILING_LOG_DEBUG("can't get vector num per cube by soc version");
            return 0;
        }
        uint64_t nStepVec0 = CeilDiv(c04Info.curNBL1 / c04Info.n0, vecNum) * c04Info.n0;
        // nStepVec1 <= nStepVec0, so bUbNStep of vec1 still alloc same size buffer of vec0;
        bUbNStep = nStepVec0 > nMaxStep ? nMaxStep : nStepVec0;
    }
    return bUbNStep;
}

bool Conv2dTiling::CheckAttrBeforeCoreBind()
{
    if (attrInfo.padLeft < 0 || attrInfo.padRight < 0 ||
        attrInfo.padTop < 0 || attrInfo.padBottom < 0) {
        TILING_LOG_ERROR(
            "Illlegal attrs have set: padTop=%d, padBottom=%d, padLeft=%d, padRight=%d, which must >= 0.",
            attrInfo.padTop, attrInfo.padBottom, attrInfo.padLeft, attrInfo.padRight);
        return false;
    }

    if (attrInfo.strideH <= 0 || attrInfo.strideW <= 0) {
        TILING_LOG_ERROR("Illegal attrs have set: strideH=%d, strideW=%d, which must > 0.",
                         attrInfo.strideH, attrInfo.strideW);
        return false;
    }

    if (attrInfo.dilationH <= 0 || attrInfo.dilationW <= 0) {
        TILING_LOG_ERROR("Illegal attrs have set: dilationH=%d, dilationW=%d, which must > 0.",
                         attrInfo.dilationH, attrInfo.dilationW);
        return false;
    }
    return true;
}

bool Conv2dTiling::CheckAttrBBModePhase1()
{
    if (!CheckAttrBeforeCoreBind()) {
        return false;
    }
    if (!CheckGroupsBBModePhase1()) {
        return false;
    }
    return true;
}

bool Conv2dTiling::CheckAttr()
{
    if (!CheckAttrBeforeCoreBind()) {
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

bool Conv2dTiling::CheckGroupsBBModePhase1()
{
    if (attrInfo.groups <= 0) {
        TILING_LOG_ERROR("Illegal attrs have set: groups=%d which must > 0.", attrInfo.groups);
        return false;
    }

    if (this->optGroupFlag) {
        if (shapeInfo.enlarge <= 0) {
            TILING_LOG_ERROR("Illegal attrs have set: enlarge=%d which must > 0.", shapeInfo.enlarge);
            return false;
        }
    }

    return true;
}

bool Conv2dTiling::CheckFmapShapeBeforeCoreBind()
{
    if (shapeInfo.orgHi <= 0 || shapeInfo.orgWi <= 0) {
        TILING_LOG_ERROR("Input illegal orgHi: %ld, orgWi: %ld, which must > 0.", shapeInfo.orgHi, shapeInfo.orgWi);
        return false;
    }
    if (shapeInfo.orgCi <= 0 || static_cast<uint64_t>(shapeInfo.orgCi) > MAX_31_BIT_NUM) {
        TILING_LOG_ERROR("Input illegal orgCi: %ld, which must in range [1, %lu].", shapeInfo.orgCi, MAX_31_BIT_NUM);
        return false;
    }
    int64_t ciPerg = this->optGroupFlag ? shapeInfo.singleCi / shapeInfo.enlarge : shapeInfo.singleCi;
    if (ciPerg * attrInfo.groups != shapeInfo.orgCi) {
        TILING_LOG_ERROR("only support groups * singleCi = orgCi, current groups: %d, singleCi: %ld, orgCi: %ld",
            attrInfo.groups, ciPerg, shapeInfo.orgCi);
        return false;
    }

    if (attrInfo.strideH == 0 || attrInfo.strideW == 0) {
        TILING_LOG_ERROR("div zero when calculating output shape, strideH: %d, strideW: %d.",
            attrInfo.strideH, attrInfo.strideW);
        return false;
    }
    shapeInfo.orgHo = (shapeInfo.orgHi + attrInfo.padTop + attrInfo.padBottom -
                       attrInfo.dilationH * (shapeInfo.orgkH - 1) - 1) / attrInfo.strideH + 1;
    shapeInfo.orgWo = (shapeInfo.orgWi + attrInfo.padLeft + attrInfo.padRight -
                       attrInfo.dilationW * (shapeInfo.orgkW - 1) - 1) / attrInfo.strideW + 1;
    if (shapeInfo.orgHo <= 0 || shapeInfo.orgWo <= 0) {
        TILING_LOG_ERROR("Calculated output orgHo: %ld, orgWo: %ld, which must > 0", shapeInfo.orgHo, shapeInfo.orgWo);
        return false;
    }

    return true;
}

bool Conv2dTiling::CheckFmapShapeBBModePhase1()
{
    bool ret = CheckFmapShapeBeforeCoreBind();
    return ret;
}

bool Conv2dTiling::CheckFeaMapShape()
{
    if (outputOrder == static_cast<int8_t>(OutputOrder::HW) &&
        (shapeInfo.singleHo <= 0 || shapeInfo.singleWo <= 0)) {
        TILING_LOG_ERROR("Input illegal singleHo: %ld, singleWo: %ld, which must > 0.",
            shapeInfo.singleHo, shapeInfo.singleWo);
        return false;
    }
    if (outputOrder == static_cast<int8_t>(OutputOrder::M) && shapeInfo.singleM <= 0) {
        TILING_LOG_ERROR("Input illegal singleM: %ld, which must > 0.", shapeInfo.singleM);
        return false;
    }
    bool ret = CheckFmapShapeBeforeCoreBind();
    return ret;
}

bool Conv2dTiling::CheckWeightShapeBeforeCoreBind()
{
    if (shapeInfo.orgkW <= 0 || static_cast<uint64_t>(shapeInfo.orgkW) > MAX_31_BIT_NUM) {
        TILING_LOG_ERROR("Input illegal kW: %ld, which must in range [1, %lu].", shapeInfo.orgkW, MAX_31_BIT_NUM);
        return false;
    }

    if (shapeInfo.orgkH <= 0 || static_cast<uint64_t>(shapeInfo.orgkH) > MAX_31_BIT_NUM) {
        TILING_LOG_ERROR("Input illegal kH: %ld, which must in range [1, %lu].", shapeInfo.orgkH, MAX_31_BIT_NUM);
        return false;
    }

    if (shapeInfo.orgCo <= 0 || static_cast<uint64_t>(shapeInfo.orgCo) > MAX_31_BIT_NUM) {
        TILING_LOG_ERROR("Input illegal orgCo: %ld, which must in range [1, %lu].", shapeInfo.orgCo, MAX_31_BIT_NUM);
        return false;
    }

    if (shapeInfo.singlekW != shapeInfo.orgkW) {
        TILING_LOG_ERROR("Only support singlekW = orgkW, current singlekW: %ld, orgkW: %ld",
            shapeInfo.singlekW, shapeInfo.orgkW);
        return false;
    }

    if (shapeInfo.singlekH != shapeInfo.orgkH) {
        TILING_LOG_ERROR("Only support singlekH = orgkH, current singlekH: %ld, orgkH: %ld,",
            shapeInfo.singlekH, shapeInfo.orgkH);
        return false;
    }

    return true;
}

bool Conv2dTiling::CheckWeightShapeBBModePhase1()
{
    bool ret = CheckWeightShapeBeforeCoreBind();
    return ret;
}

bool Conv2dTiling::CheckWeightShape()
{
    if (shapeInfo.singleCo <= 0 || static_cast<uint64_t>(shapeInfo.singleCo) > MAX_31_BIT_NUM) {
        TILING_LOG_ERROR("Input illegal SingleCo: %ld, which must in range [1, %lu].",
            shapeInfo.singleCo, MAX_31_BIT_NUM);
        return false;
    }

    bool ret = CheckWeightShapeBeforeCoreBind();
    return ret;
}

bool Conv2dTiling::CheckShapeBBModePhase1()
{
    if (!CheckFmapShapeBBModePhase1()) {
        return false;
    }

    if (!CheckWeightShapeBBModePhase1()) {
        return false;
    }

    return true;
}

bool Conv2dTiling::CheckShape()
{
    if (!CheckFeaMapShape()) {
        return false;
    }

    if (!CheckWeightShape()) {
        return false;
    }

    return true;
}

bool Conv2dTiling::CheckFormat()
{
    std::set<std::pair<ConvFormat, ConvFormat>> conv2dSupportFormatSet = {
        {ConvFormat::NCHW, ConvFormat::NCHW},
        {ConvFormat::NHWC, ConvFormat::HWCN},
        {ConvFormat::NCHW, ConvFormat::FRACTAL_Z},
        {ConvFormat::NCHW, ConvFormat::FRACTAL_Z_C04},
        {ConvFormat::NHWC, ConvFormat::FRACTAL_Z},
        {ConvFormat::NHWC, ConvFormat::FRACTAL_Z_C04}
    };

    if (conv2dSupportFormatSet.find({descInfo.fMapType.format, descInfo.weightType.format}) ==
        conv2dSupportFormatSet.end()) {
        TILING_LOG_ERROR("unSupported format combo: fmap format: %s, weight format: %s.",
                         FORMAT_TO_STR.at(descInfo.fMapType.format).c_str(),
                         FORMAT_TO_STR.at(descInfo.weightType.format).c_str());
        return false;
    }

    return true;
}

bool Conv2dTiling::CheckAlgorithmLimit()
{
    if (attrInfo.groups > 1 && isC04Flag) {
        // group conv2d currently not support c04 mode
        TILING_LOG_ERROR("not support C04 mode in group Conv.");
        return false;
    }

    return true;
}

bool Conv2dTiling::CheckBBmodeLimits(Conv2DBasicBlockInfo& conv2DBasicBlockInfo)
{
    int64_t groups = attrInfo.groups;
    if (this->optGroupFlag) {
        groups = shapeInfo.singleGroupOpt; // actually not single, it is ori groupopt
    }
    if (!enableInnerBatch && CeilDiv(shapeInfo.orgWo * shapeInfo.orgHo, conv2DBasicBlockInfo.mTile) *
        CeilDiv(shapeInfo.orgCo, conv2DBasicBlockInfo.nTile) * conv2DBasicBlockInfo.batch * groups <=
        conv2DBasicBlockInfo.aicoreNum) {
        return false;
    }
    if (enableInnerBatch && CeilDiv(CeilDiv(conv2DBasicBlockInfo.batch *  CeilDiv(shapeInfo.orgWo *
        shapeInfo.orgHo, cubeInfo.m0) * cubeInfo.m0, conv2DBasicBlockInfo.mTile), innerBatch) *
        CeilDiv(shapeInfo.orgCo, conv2DBasicBlockInfo.nTile) * groups <= conv2DBasicBlockInfo.aicoreNum &&
        CeilDiv(conv2DBasicBlockInfo.batch, conv2DBasicBlockInfo.aicoreNum) == CONST_VALUE_1) {
        return false;
    }
    int64_t biasSize = 0;
    int64_t scaleSize = 0;
    if (hasBias) {
        biasSize = DTYPE_SIZE_TAB.at(descInfo.biasType.dtype) * conv2DBasicBlockInfo.nTile;
    }
    if (hasQuantScale) {
        scaleSize = shapeInfo.channelWiseCoeff *
                    DTYPE_SIZE_TAB.at(descInfo.quantScaleType.dtype) * conv2DBasicBlockInfo.nTile;
    }
    TILING_LOG_DEBUG("scaleSize:%ld, hasQuantScale:%d, channelWiseCoeff:%f", scaleSize, hasQuantScale,
                    shapeInfo.channelWiseCoeff);
    int64_t availableL1Size = platformInfo.l1Size - biasSize - scaleSize;
    int64_t fMapDtypeSize = static_cast<int64_t>(DTYPE_SIZE_TAB.at(descInfo.fMapType.dtype));
    int64_t maxHiWiL1 = availableL1Size / fMapDtypeSize /
        static_cast<int64_t>(CONST_VALUE_2) / static_cast<int64_t>(cubeInfo.k0) / innerBatch;
    if (maxHiWiL1 <= 0) {
        return false;
    }
    int64_t maxhiL1 = maxHiWiL1 / shapeInfo.orgWi;
    if (maxhiL1 <= static_cast<int64_t>(CONST_VALUE_2)) {
        return false;
    }
    if (conv2DBasicBlockInfo.mTile < cubeInfo.m0 || conv2DBasicBlockInfo.nTile < cubeInfo.n0) {
        return false;
    }
    return true;
}

bool Conv2dTiling::CheckDataCopyLimits()
{
    uint64_t loadAL1loop1SrcStrideLimits = MAX_40_BIT_NUM;
    uint64_t loadAL1loop1SrcStride = shapeInfo.orgHi * shapeInfo.orgWi * DTYPE_SIZE_TAB.at(descInfo.fMapType.dtype);
    if (descInfo.fMapType.format == ConvFormat::NCHW && loadAL1loop1SrcStride > loadAL1loop1SrcStrideLimits) {
        TILING_LOG_ERROR(
            "Fmap shape not satisfy DataCopy's limits: hin(%ld)*win(%ld)*typesize(%u)=%lu, must <= %lu",
            shapeInfo.orgHi, shapeInfo.orgWi, DTYPE_SIZE_TAB.at(descInfo.fMapType.dtype),
            loadAL1loop1SrcStride, loadAL1loop1SrcStrideLimits);
        return false;
    }
    return true;
}

bool Conv2dTiling::CheckFixpipeLimits()
{
    uint64_t fixpipeLoop2DstStrideLimit = MAX_32_BIT_NUM;
    uint64_t fixpipeLoop2DstStride = static_cast<uint64_t>(shapeInfo.orgHo) * shapeInfo.orgWo;
    if (descInfo.fMapType.format == ConvFormat::NCHW && fixpipeLoop2DstStride > fixpipeLoop2DstStrideLimit) {
        TILING_LOG_ERROR(
            "Output shape not satisfy Fixpipe's limits: hout(%ld)*wout(%ld)=%lu, which must <= %lu",
            shapeInfo.orgHo, shapeInfo.orgWo, fixpipeLoop2DstStride, fixpipeLoop2DstStrideLimit);
        return false;
    }

    // loop3_dst_stride limits check
    uint64_t fixpipeLoop3DstStrideLimit = MAX_32_BIT_NUM;
    uint64_t fixpipeLoop3DstStride = shapeInfo.orgWo * shapeInfo.orgCo;
    if (descInfo.fMapType.format == ConvFormat::NHWC && outputOrder == static_cast<int8_t>(OutputOrder::HW) &&
        fixpipeLoop3DstStride > fixpipeLoop3DstStrideLimit) {
        TILING_LOG_ERROR(
            "Output shape not satisfy Fixpipe's limits: wout(%lu)*cout(%lu)=%lu, which must <= %lu",
            shapeInfo.orgWo, shapeInfo.orgCo, fixpipeLoop3DstStride, fixpipeLoop3DstStrideLimit);
        return false;
    }

    return true;
}

bool Conv2dTiling::CheckL1SizeLimitsKernelFullLoad()
{
    InferCubeInfo();
    uint64_t fMapDtypeSize = DTYPE_SIZE_TAB.at(descInfo.fMapType.dtype);
    uint64_t weightDtypeSize = DTYPE_SIZE_TAB.at(descInfo.weightType.dtype);
    uint64_t nBL1min = cubeInfo.n0;
    uint64_t biasUsedL1Size = hasBias ? AlignB(nBL1min * DTYPE_SIZE_TAB.at(descInfo.biasType.dtype), C0_SIZE) : 0;
    uint64_t scaleUsedL1Size = AlignB(nBL1min * shapeInfo.channelWiseCoeff * FP16_DTYPE_SIZE, C0_SIZE);
    uint64_t kBL1min = cubeInfo.k0 * shapeInfo.orgkH * shapeInfo.orgkW;
    uint64_t weightUsedL1Size = AlignB(kBL1min * nBL1min * weightDtypeSize, C0_SIZE);
    uint64_t fmapUsedL1Size = 0;
    uint64_t hoAL1min = shapeInfo.orgWo < cubeInfo.m0 ? CeilDiv(cubeInfo.m0, shapeInfo.orgWo) : 1;
    uint64_t khDilated = (shapeInfo.orgkH - 1) * attrInfo.dilationH + 1;
    uint64_t hiAL1min = Conv2DInferHiL1(hoAL1min, khDilated, shapeInfo.orgHi, attrInfo.strideH);
    uint64_t kAL1min = cubeInfo.k0;
    uint64_t woAL1min = cubeInfo.m0;
    uint64_t kwDilated = (shapeInfo.orgkW - 1) * attrInfo.dilationW + 1;
    uint64_t wiAL1min = Conv2DInferHiL1(woAL1min, kwDilated, shapeInfo.orgWi, attrInfo.strideW);
    fmapUsedL1Size = AlignB(hiAL1min * wiAL1min * kAL1min * fMapDtypeSize, C0_SIZE) * innerBatch;
 
    uint64_t minL1LoadSize = biasUsedL1Size + scaleUsedL1Size + fmapUsedL1Size + weightUsedL1Size;
    if (minL1LoadSize > platformInfo.l1Size) {
        TILING_LOG_DEBUG("KernelSplitMinL1LoadSize > L1size, current L1size: %lu, maxL1Size: %lu",
            minL1LoadSize, platformInfo.l1Size);
        return false;
    }
    return true;
}

bool Conv2dTiling::CheckInstructionLimits()
{
    if (!CheckLoad3DLimits() || !CheckL1SizeLimitsKernelFullLoad()) {
        if (!CheckDmaLimits()) {
            return false;
        }
        this->isDmaFlag = true;
    }
 
    if (!CheckDataCopyLimits()) {
        return false;
    }
 
    if (!CheckFixpipeLimits()) {
        return false;
    }
 
    return true;
}

bool Conv2dTiling::CheckL1SizeLimit()
{
    uint32_t fMapDtypeSize = DTYPE_SIZE_TAB.at(descInfo.fMapType.dtype);
    uint32_t weightDtypeSize = DTYPE_SIZE_TAB.at(descInfo.weightType.dtype);
    // ensure 32-byte alignment
    uint64_t minBiasSize = hasBias ? AlignB(cubeInfo.n0 * DTYPE_SIZE_TAB.at(descInfo.biasType.dtype), C0_SIZE) : 0;
    uint64_t minScaleSize = 0;
    if (hasQuantScale) {
        minScaleSize = shapeInfo.channelWiseCoeff *
                       cubeInfo.n0 * DTYPE_SIZE_TAB.at(descInfo.quantScaleType.dtype);
    }
    TILING_LOG_DEBUG("minScaleSize:%lu, hasQuantScale:%d, channelWiseCoeff:%f", minScaleSize, hasQuantScale,
                    shapeInfo.channelWiseCoeff);
    uint64_t minBL1Size = shapeInfo.orgkH * shapeInfo.orgkW * cubeInfo.n0 * cubeInfo.k0 * weightDtypeSize;
    uint64_t minAL1Size = 0;
    uint64_t minHoAL1 = min(MIN_M_L1_SIZE / shapeInfo.orgWo + CONST_VALUE_2, shapeInfo.orgHo);
    uint64_t khd = attrInfo.dilationH * (shapeInfo.orgkH - 1) + 1; // d means dilation
    uint64_t minHiAL1 = (minHoAL1 - 1) * attrInfo.strideH + khd;
    minHiAL1 = (minHiAL1 < static_cast<uint64_t>(shapeInfo.orgHi)) ? minHiAL1 : static_cast<uint64_t>(shapeInfo.orgHi);
    minAL1Size = minHiAL1 * shapeInfo.orgWi * cubeInfo.k0 * fMapDtypeSize * innerBatch;
    uint64_t minL1LoadSize = minAL1Size + minBL1Size + minBiasSize + minScaleSize;
    if (minL1LoadSize > platformInfo.l1Size) {
        TILING_LOG_DEBUG("CheckL1SizeLimit, current L1size: %lu, maxL1Size: %lu",
            minL1LoadSize, platformInfo.l1Size);
        return false;
    }
    return true;
}

void Conv2dTiling::PrintConv2DBasicBlockInfoPhase1(UNUSED const Conv2DBasicBlockInfo& conv2DBasicBlockInfo) const
{
    TILING_LOG_DEBUG("[Phase1]");
    TILING_LOG_DEBUG("Conv2DBasicBlock fDim is: %u", conv2DBasicBlockInfo.fDim);
    TILING_LOG_DEBUG("Conv2DBasicBlock nDim is: %u", conv2DBasicBlockInfo.nDim);
    TILING_LOG_DEBUG("Conv2DBasicBlock groupDim is: %u", conv2DBasicBlockInfo.groupDim);
    TILING_LOG_DEBUG("Conv2DBasicBlock aicoreNum is: %u", conv2DBasicBlockInfo.aicoreNum);
    TILING_LOG_DEBUG("Conv2DBasicBlock mIn is: %lu", conv2DBasicBlockInfo.mIn);
    TILING_LOG_DEBUG("Conv2DBasicBlock mTile is: %u", conv2DBasicBlockInfo.mTile);
    TILING_LOG_DEBUG("Conv2DBasicBlock nTile is: %u", conv2DBasicBlockInfo.nTile);
    TILING_LOG_DEBUG("Conv2DBasicBlock mCut is: %u", conv2DBasicBlockInfo.mCut);
    TILING_LOG_DEBUG("Conv2DBasicBlock nCut is: %u", conv2DBasicBlockInfo.nCut);
    TILING_LOG_DEBUG("Conv2DBasicBlock batch is: %u", conv2DBasicBlockInfo.batch);
}

bool Conv2dTiling::CheckConv2DBasicBlockInfoPhase1(Conv2DBasicBlockInfo& conv2DBasicBlockInfo) const
{
    PrintConv2DBasicBlockInfoPhase1(conv2DBasicBlockInfo);
    if (conv2DBasicBlockInfo.fDim == 0) {
        TILING_LOG_ERROR("fDim is illegal which is: %u", conv2DBasicBlockInfo.fDim);
        return false;
    }
    if (conv2DBasicBlockInfo.nDim == 0) {
        TILING_LOG_ERROR("nDim is illegal which is: %u", conv2DBasicBlockInfo.nDim);
        return false;
    }
    if (conv2DBasicBlockInfo.groupDim == 0) {
        TILING_LOG_ERROR("groupDim is illegal which is: %u", conv2DBasicBlockInfo.groupDim);
        return false;
    }
    if (conv2DBasicBlockInfo.aicoreNum == 0) {
        TILING_LOG_ERROR("aicoreNum is illegal which is: %u", conv2DBasicBlockInfo.aicoreNum);
        return false;
    }
    if (conv2DBasicBlockInfo.mIn == 0) {
        TILING_LOG_ERROR("mIn is illegal which is: %lu", conv2DBasicBlockInfo.mIn);
        return false;
    }
    if (conv2DBasicBlockInfo.mTile == 0) {
        TILING_LOG_ERROR("mTile is illegal which is: %u", conv2DBasicBlockInfo.mTile);
        return false;
    }
    if (conv2DBasicBlockInfo.nTile == 0) {
        TILING_LOG_ERROR("nTile is illegal which is: %u", conv2DBasicBlockInfo.nTile);
        return false;
    }
    if (conv2DBasicBlockInfo.mCut == 0) {
        TILING_LOG_ERROR("mCut is illegal which is: %u", conv2DBasicBlockInfo.mCut);
        return false;
    }
    if (conv2DBasicBlockInfo.nCut == 0) {
        TILING_LOG_ERROR("nCut is illegal which is: %u", conv2DBasicBlockInfo.nCut);
        return false;
    }
    if (conv2DBasicBlockInfo.batch == 0) {
        TILING_LOG_ERROR("batch is illegal which is: %u", conv2DBasicBlockInfo.batch);
        return false;
    }
    return true;
}

void Conv2dTiling::PrintConv2DBasicBlockInfoPhase2(UNUSED const Conv2DBasicBlockInfo& conv2DBasicBlockInfo) const
{
    TILING_LOG_DEBUG("[Phase2]");
    TILING_LOG_DEBUG("Conv2DBasicBlock batchDim is: %u", conv2DBasicBlockInfo.batchDim);
    TILING_LOG_DEBUG("Conv2DBasicBlock mDim is: %u", conv2DBasicBlockInfo.mDim);
    TILING_LOG_DEBUG("Conv2DBasicBlock nDim is: %u", conv2DBasicBlockInfo.nDim);
    TILING_LOG_DEBUG("Conv2DBasicBlock groupDim is: %u", conv2DBasicBlockInfo.groupDim);
    TILING_LOG_DEBUG("Conv2DBasicBlock mCut is: %u", conv2DBasicBlockInfo.mCut);
    TILING_LOG_DEBUG("Conv2DBasicBlock nCut is: %u", conv2DBasicBlockInfo.nCut);
    TILING_LOG_DEBUG("Conv2DBasicBlock mTile is: %u", conv2DBasicBlockInfo.mTile);
    TILING_LOG_DEBUG("Conv2DBasicBlock nTile is: %u", conv2DBasicBlockInfo.nTile);
    TILING_LOG_DEBUG("Conv2DBasicBlock mIn is: %lu", conv2DBasicBlockInfo.mIn);
}

bool Conv2dTiling::CheckConv2DBasicBlockInfoPhase2(Conv2DBasicBlockInfo& conv2DBasicBlockInfo) const
{
    PrintConv2DBasicBlockInfoPhase2(conv2DBasicBlockInfo);
    if (conv2DBasicBlockInfo.batchDim == 0) {
        TILING_LOG_ERROR("batchDim is illegal which is: %u", conv2DBasicBlockInfo.batchDim);
        return false;
    }
    if (conv2DBasicBlockInfo.mDim == 0) {
        TILING_LOG_ERROR("mDim is illegal which is: %u", conv2DBasicBlockInfo.mDim);
        return false;
    }
    if (conv2DBasicBlockInfo.nDim == 0) {
        TILING_LOG_ERROR("nDim is illegal which is: %u", conv2DBasicBlockInfo.nDim);
        return false;
    }
    if (conv2DBasicBlockInfo.mTile == 0) {
        TILING_LOG_ERROR("mTile is illegal which is: %u", conv2DBasicBlockInfo.mTile);
        return false;
    }
    if (conv2DBasicBlockInfo.nTile == 0) {
        TILING_LOG_ERROR("nTile is illegal which is: %u", conv2DBasicBlockInfo.nTile);
        return false;
    }
    if (conv2DBasicBlockInfo.mIn == 0) {
        TILING_LOG_ERROR("mIn is illegal which is: %lu", conv2DBasicBlockInfo.mIn);
        return false;
    }
    if (conv2DBasicBlockInfo.iterateMNOrder == IterateMNOrder::INVALID) {
        TILING_LOG_ERROR("iterateMNOrder is illegal which is: %u (INVALID)",
                         static_cast<uint8_t>(conv2DBasicBlockInfo.iterateMNOrder));
        return false;
    }
    return true;
}

bool Conv2dTiling::CheckTilingAlgorithmType(Conv2DBasicBlockInfo& conv2DBasicBlockInfo, int8_t phase)
{
    if (phase == BB_PHASE_1) {
        if (!CheckParamsBBModePhase1()) { // General Check
            TILING_LOG_ERROR("conv2d api tiling general params check failed in phase 1.");
            return false;
        }
    } else {
        if (!CheckParams()) {
            TILING_LOG_ERROR("conv2d api tiling general params check failed in phase 2.");
            return false;
        }
    }
    InferCubeInfo();
    if (!CheckL1SizeLimit()) { // L1size check
        TILING_LOG_ERROR("conv2d api tiling L1size check failed.");
        return false;
    }

    if (phase == BB_PHASE_1) {
        if (!CheckConv2DBasicBlockInfoPhase1(conv2DBasicBlockInfo)) {
            TILING_LOG_ERROR("conv2d basic block info check failed in phase 1.");
            return false;
        }
    } else {
        if (!CheckConv2DBasicBlockInfoPhase2(conv2DBasicBlockInfo)) {
            TILING_LOG_ERROR("conv2d basic block info check failed in phase 2.");
            return false;
        }
    }
    if (CheckBBmodeLimits(conv2DBasicBlockInfo)) { // Basic Block Algorithm Check
        tilingAlgorithmType = static_cast<int8_t>(TilingAlgorithmType::BASIC_BLOCK);
    }
    switch (tilingAlgorithmType) {
        case static_cast<int8_t>(TilingAlgorithmType::BASIC_BLOCK):
            return true;
        case static_cast<int8_t>(TilingAlgorithmType::FORMULAS):
        default:
            TILING_LOG_ERROR("Unsupported tiling algorithm type %d, only support BASIC_BLOCK(%d) and FORMULAS(%d).",
                tilingAlgorithmType, static_cast<int8_t>(TilingAlgorithmType::BASIC_BLOCK),
                static_cast<int8_t>(TilingAlgorithmType::FORMULAS));
            return false;
    }
    return true;
}

bool Conv2dTiling::GetCoreBindingDecisionFactor(Conv2DBasicBlockInfo& conv2DBasicBlockInfo)
{
    if (!CheckTilingAlgorithmType(conv2DBasicBlockInfo, BB_PHASE_1)) {
        return false;
    }

    Infer5hdShape();
    std::shared_ptr<ConvTilingAlgorithmBBmode> algoBBPtr = std::make_shared<ConvTilingAlgorithmBBmode>(this,
        conv2DBasicBlockInfo);
    algoBBPtr->AdjustM();
    algoBBPtr->AdjustN();
    algoBBPtr->CalcBestL1LoadStratgy();
    algoBBPtr->CalcCoreUtilization();
    return true;
}
} // namespace conv_tiling