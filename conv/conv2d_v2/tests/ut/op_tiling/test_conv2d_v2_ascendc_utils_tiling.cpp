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
 * \file test_conv2d_v2_ascendc_utils_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include "platform/platform_info.h"
#include "test_conv2d_v2_ascendc_utils_tiling.h"
#include <platform/soc_spec.h>

namespace conv_api_tiling_test {

using namespace std;
using namespace conv_tiling;
using namespace conv_tiling_algo_m;
using namespace conv_tiling_algo_hw;

uint64_t InferHo(uint64_t inputHi, uint64_t kH, uint64_t padTop, uint64_t padBottom, uint64_t dilationH, uint64_t strideH)
{
    if (strideH == 0) {
        return 0;
    }
    return (inputHi + padTop + padBottom - dilationH * (kH - 1) - 1) / strideH + 1;
}

uint64_t InferWo(uint64_t inputWi, uint64_t kW, uint64_t padLeft, uint64_t padRight, uint64_t dilationW, uint64_t strideW)
{
    if (strideW == 0) {
        return 0;
    }
    return (inputWi + padLeft + padRight - dilationW * (kW - 1) - 1) / strideW + 1;
}

uint64_t InferHi(uint64_t inputHo, uint64_t kH, uint64_t padTop, uint64_t padBottom, uint64_t dilationH, uint64_t strideH)
{
    return (inputHo - 1) * strideH - (padTop + padBottom - dilationH * (kH - 1) - 1);
}

uint64_t InferWi(uint64_t inputWo, uint64_t kW, uint64_t padLeft, uint64_t padRight, uint64_t dilationW, uint64_t strideW)
{
    return (inputWo - 1) * strideW - (padLeft + padRight - dilationW * (kW - 1) - 1);
}

int64_t InferHiL1(uint64_t inputHoL1, int64_t hi, uint64_t singlekH, uint64_t dilationH, uint64_t strideH)
{
    int64_t khDilated = (singlekH - 1) * dilationH + 1;
    int64_t tmpHiL1 = (inputHoL1 - 1) * strideH + khDilated;
    if (tmpHiL1 > hi) {
        tmpHiL1 = hi;
    }

    return tmpHiL1;
}

int64_t InferWiL1(uint64_t inputWoL1, int64_t wi, uint64_t singlekW, uint64_t dilationW, uint64_t strideW)
{
    int64_t kwDilated = (singlekW - 1) * dilationW + 1;
    int64_t tmpWiL1 = (inputWoL1 - 1) * strideW + kwDilated;
    if (tmpWiL1 > wi) {
        tmpWiL1 = wi;
    }

    return tmpWiL1;
}

uint64_t ConvCeilDiv(uint64_t a, uint64_t b)
{
    return (a + b - 1) / b;
}

uint64_t ConvGcd(uint64_t a, uint64_t b) {
    while (b != 0) {
        uint64_t temp = a % b;
        a = b;
        b = temp;
    }
    return a;
}

uint64_t CalcUsdL1Size(optiling::TConv2DTiling &tilingData,
                       Conv2dTiling &tiling,
                       int8_t pbAL1,
                       int8_t pbBL1)
{
    uint32_t weightDtyeSize = DTYPE_SIZE_TAB.at(tiling.descInfo.weightType.dtype);
    uint32_t featuremapDtyeSize = DTYPE_SIZE_TAB.at(tiling.descInfo.fMapType.dtype);
    uint32_t biasDtyeSize = tiling.hasBias ? DTYPE_SIZE_TAB.at(tiling.descInfo.biasType.dtype) : 0;
    uint32_t scaleDtyeSize = tiling.hasScale ? DTYPE_SIZE_TAB.at(tiling.descInfo.scaleType.dtype) : 0;
    uint64_t curl1Size = 0;
    uint64_t al1Size = 0;
    uint64_t bl1Size = 0;
    uint64_t biasL1Size = 0;
    uint64_t scaleL1Size = 0;

    auto n0 = tiling.cubeInfo.n0;

    if (tiling.outputOrder == 1) {
        uint64_t hoAL1Tmp = min(tilingData.get_hoL1() / tilingData.get_orgWo() + 2,  tilingData.get_orgHo());
        uint64_t hiL1Tmp = min((hoAL1Tmp - 1) * tilingData.get_strideH() + (tilingData.get_kernelH() - 1) / tilingData.get_dilationH() + 1, tilingData.get_orgHi());
        uint64_t al1Cin = tiling.isC04Flag ? 4 : (tilingData.get_kAL1() / (tilingData.get_kernelH() * tilingData.get_kernelW()));
        al1Size = hiL1Tmp * tilingData.get_orgWi() * al1Cin * (pbAL1 + 1) * featuremapDtyeSize;
        bl1Size = tilingData.get_nBL1() * tilingData.get_kBL1() * (pbBL1 + 1) * weightDtyeSize;
        if (tiling.hasBias) {
            if (tilingData.get_biasFullLoadFlag() == 1) {
                biasL1Size = CeilDiv(tilingData.get_singleCoreCo(), n0) * n0 * biasDtyeSize;
            } else if (tilingData.get_biasFullLoadFlag() == 1) {
                biasL1Size = tilingData.get_nL0() * biasDtyeSize;
            }
        } else {
            biasL1Size = 0;
        }
        curl1Size = al1Size + bl1Size + biasL1Size;
    } else {
        uint64_t hiL1 = InferHiL1(tilingData.get_hoL1(), tilingData.get_orgHi(), tilingData.get_kernelH(), tilingData.get_dilationH(), tilingData.get_strideH());
        uint64_t wiL1 = InferWiL1(tilingData.get_woL1(), tilingData.get_orgWi(), tilingData.get_kernelW(), tilingData.get_dilationW(), tilingData.get_strideW());
        uint64_t al1Cin = tiling.isC04Flag ? 4 : (tilingData.get_kAL1() / (tilingData.get_kernelH() * tilingData.get_kernelW()));
        al1Size = hiL1 * wiL1 * al1Cin * (pbAL1 + 1) * featuremapDtyeSize;
        bl1Size = tilingData.get_nBL1() * tilingData.get_kBL1() * weightDtyeSize;
        if (tiling.hasBias) {
            if (tilingData.get_biasFullLoadFlag()) {
                biasL1Size = ConvCeilDiv(tilingData.get_singleCoreCo(), n0) * n0 * biasDtyeSize;
            } else {
                biasL1Size = tilingData.get_nBL1() * biasDtyeSize;
            }
        }

        if (tiling.hasScale) {
            if (tilingData.get_fixpParamsFullLoadFlag()) {
                scaleL1Size = ConvCeilDiv(tilingData.get_singleCoreCo(), n0) * n0 * scaleDtyeSize;
            } else {
                scaleL1Size = tilingData.get_nBL1() * scaleDtyeSize;
            }
        }
        curl1Size = al1Size + bl1Size + biasL1Size + scaleL1Size;
    }

    return curl1Size;
}

uint64_t CalcUsdL0ASize(optiling::TConv2DTiling &tilingData,
                       Conv2dTiling &tiling,
                       int8_t pbAL0)
{
    uint32_t featuremapDtyeSize = DTYPE_SIZE_TAB.at(tiling.descInfo.fMapType.dtype);
    uint64_t curl0aSize = 0;
    if (tiling.outputOrder == 0) {
        curl0aSize = tilingData.get_hoL0() * tilingData.get_woL0() * tilingData.get_kL0() * (pbAL0 + 1) * featuremapDtyeSize;
    } else {
        curl0aSize = tilingData.get_hoL0() * tilingData.get_kL0() * (pbAL0 + 1) * featuremapDtyeSize;
    }
    return curl0aSize;
}

uint64_t CalcUsdL0BSize(optiling::TConv2DTiling &tilingData,
                       Conv2dTiling &tiling,
                       int8_t pbBL0)
{
    uint32_t weightDtyeSize = DTYPE_SIZE_TAB.at(tiling.descInfo.weightType.dtype);
    return tilingData.get_nL0() * tilingData.get_kL0() * (pbBL0 + 1) * weightDtyeSize;
}

uint64_t CalcUsdL0CSize(optiling::TConv2DTiling &tilingData,
                       Conv2dTiling &tiling,
                       int8_t pbCL0)
{
    uint64_t curl0cSize = 0;
    uint32_t mmadDtypeSize = DTYPE_SIZE_TAB.at(tiling.cubeInfo.madType);
    if (tiling.outputOrder == 0) {
        curl0cSize = tilingData.get_hoL0() * tilingData.get_woL0() * tilingData.get_nL0() * (pbCL0 + 1) * mmadDtypeSize;
    } else {
        curl0cSize = tilingData.get_hoL0() * tilingData.get_nL0() * (pbCL0 + 1) * mmadDtypeSize;
    }
    return curl0cSize;
}

void CheckValidTilingData(optiling::TConv2DTiling &tilingData,
                          Conv2dTiling &tiling)
{
    uint32_t weightDtyeSize = DTYPE_SIZE_TAB.at(tiling.descInfo.weightType.dtype);
    uint32_t featuremapDtyeSize = DTYPE_SIZE_TAB.at(tiling.descInfo.fMapType.dtype);
    uint32_t biasDtyeSize = tiling.hasBias ? DTYPE_SIZE_TAB.at(tiling.descInfo.biasType.dtype) : 0;

    uint32_t mmadDtypeSize = DTYPE_SIZE_TAB.at(tiling.cubeInfo.madType);
    auto l0aSize = tiling.platformInfo.l0ASize;
    auto l0bSize = tiling.platformInfo.l0BSize;
    auto l0cSize = tiling.platformInfo.l0CSize;
    auto l1Size = tiling.platformInfo.l1Size;
    auto m0 = tiling.cubeInfo.m0;
    auto k0 = tiling.cubeInfo.k0;
    auto n0 = tiling.cubeInfo.n0;

    uint64_t pBuffer = tilingData.get_pBufferFlag();
    int8_t pbAL0 = pBuffer & 0x01;
    int8_t pbBL0 = (pBuffer & 0x02) >> 1;
    int8_t pbCL0 = (pBuffer & 0x04) >> 2;
    int8_t pbAL1 = (pBuffer & 0x08) >> 3;
    int8_t pbBL1 = (pBuffer & 0x10) >> 4;

    EXPECT_GT(tilingData.get_kAL1(), 0);
    EXPECT_GT(tilingData.get_kBL1(), 0);
    EXPECT_GT(tilingData.get_hoL1(), 0);
    EXPECT_GT(tilingData.get_nBL1(), 0);
    EXPECT_GT(tilingData.get_hoL0(), 0);
    EXPECT_GT(tilingData.get_kL0(), 0);
    EXPECT_GT(tilingData.get_nL0(), 0);

    EXPECT_EQ(tilingData.get_kAL1() % k0, 0);
    EXPECT_EQ(tilingData.get_nBL1() % n0, 0);
    EXPECT_EQ(tilingData.get_nL0() % n0, 0);
    EXPECT_EQ(tilingData.get_kL0() % k0, 0);
    EXPECT_EQ(tilingData.get_kAL1() % tilingData.get_kL0(), 0);
    EXPECT_EQ(tilingData.get_kBL1() % tilingData.get_kL0(), 0);

    EXPECT_LE(CalcUsdL1Size(tilingData, tiling, pbAL1, pbBL1), l1Size);
    EXPECT_LE(CalcUsdL0ASize(tilingData, tiling, pbAL0), l0aSize);
    EXPECT_LE(CalcUsdL0BSize(tilingData, tiling, pbBL0), l0bSize);
    EXPECT_LE(CalcUsdL0CSize(tilingData, tiling, pbCL0), l0cSize);

    if (tiling.outputOrder == 1) {
        EXPECT_LE(tilingData.get_nBL1(), ConvCeilDiv(ConvCeilDiv(tilingData.get_singleCoreCo(), n0) * n0, tilingData.get_nL0()) * tilingData.get_nL0());
        EXPECT_LE(tilingData.get_hoL1(), ConvCeilDiv(ConvCeilDiv(tilingData.get_singleCoreHo(), m0) * m0, tilingData.get_hoL0()) * tilingData.get_hoL0());
        EXPECT_LE(tilingData.get_kAL1(), tilingData.get_kernelH() * tilingData.get_kernelW() * ConvCeilDiv(tilingData.get_orgCi(), k0) * k0);
        EXPECT_LE(tilingData.get_kBL1(), tilingData.get_kernelH() * tilingData.get_kernelW() * ConvCeilDiv(tilingData.get_orgCi(), k0) * k0);
        EXPECT_LE(tilingData.get_hoL0(), ConvCeilDiv(std::min(l0aSize / (k0 * (pbAL0 + 1) * featuremapDtyeSize), l0cSize / (n0 * (pbCL0 + 1) * mmadDtypeSize)), m0) * m0);
        EXPECT_LE(tilingData.get_nL0(), ConvCeilDiv(std::min(l0bSize / (k0 * (pbBL0 + 1) * weightDtyeSize), l0cSize / (m0 * (pbCL0 + 1) * mmadDtypeSize)), n0) * n0);
        EXPECT_LE(tilingData.get_kL0(), ConvGcd(ConvCeilDiv(tilingData.get_kAL1(), k0), ConvCeilDiv(tilingData.get_kBL1(), k0)) * k0);

        EXPECT_EQ(tilingData.get_woL1(), 0);
        EXPECT_EQ(tilingData.get_woL0(), 0);
        EXPECT_EQ(tilingData.get_singleCoreWo(), 0);
        EXPECT_EQ(tilingData.get_hoL1() % m0, 0);
        if (!tiling.l1TilingInfo.bl1FullLoad) {
            EXPECT_EQ(tilingData.get_nBL1() % tilingData.get_nL0(), 0);
        }
        if (!tiling.l1TilingInfo.al1FullLoad) {
            EXPECT_EQ(tilingData.get_hoL1() % tilingData.get_hoL0(), 0);
        }
    }


    if (tiling.outputOrder == 0) {
        // K direction check
        EXPECT_GE(tilingData.get_kL0(), k0);
        EXPECT_LE(tilingData.get_kL0(), std::min(tilingData.get_kAL1(), tilingData.get_kBL1()));
        EXPECT_EQ(tilingData.get_kL0() % k0, 0);
        
        // N direction check
        EXPECT_GE(tilingData.get_nL0(), n0);
        EXPECT_GE(tilingData.get_nBL1(), tilingData.get_nL0());
        EXPECT_LE(tilingData.get_nBL1(), ConvCeilDiv(ConvCeilDiv(tilingData.get_singleCoreCo(), n0) * n0, tilingData.get_nL0()) * tilingData.get_nL0());
        EXPECT_EQ(tilingData.get_nL0() % n0, 0);
        uint32_t nBL1DivCheck = 0;
        if (tilingData.get_nBL1() % tilingData.get_nL0() == 0 ||
            tilingData.get_nBL1() == ConvCeilDiv(tilingData.get_singleCoreCo(), n0) * n0) {
            nBL1DivCheck = 1;
        }
        EXPECT_EQ(nBL1DivCheck, 1);
        
        // W direction check
        EXPECT_GE(tilingData.get_woL0(), m0);
        EXPECT_GE(tilingData.get_woL1(), tilingData.get_woL0());
        EXPECT_LE(tilingData.get_woL1(), 
            ConvCeilDiv(ConvCeilDiv(tilingData.get_singleCoreWo(), m0) * m0, tilingData.get_woL0()) * tilingData.get_woL0());
        EXPECT_EQ(tilingData.get_woL0() % m0, 0);
        if (tilingData.get_woL0() < ConvCeilDiv(tilingData.get_orgWo(), m0) * m0) {
            // woL0 does not reach the upper limit, thus hoL0 must be 1.
            EXPECT_EQ(tilingData.get_hoL0(), 1);
        }
        if (tilingData.get_hoL0() > 1) {
            EXPECT_EQ(tilingData.get_woL0(), ConvCeilDiv(tilingData.get_orgWo(), m0) * m0);
            EXPECT_EQ(tilingData.get_woL1(), ConvCeilDiv(tilingData.get_orgWo(), m0) * m0);
        }

        // H direction check
        uint32_t hoL1DivCheck = 0;
        EXPECT_GE(tilingData.get_hoL0(), 1);
        EXPECT_GE(tilingData.get_hoL1(), tilingData.get_hoL0());
        EXPECT_LE(tilingData.get_hoL1(), tilingData.get_singleCoreHo());
        uint32_t hoL1Check = 0;
        if (tilingData.get_hoL1() % tilingData.get_hoL0() == 0 || tilingData.get_hoL1() == tilingData.get_singleCoreHo()) {
            hoL1Check = 1;
        }
        EXPECT_EQ(hoL1Check, 1);

        if (tiling.isC04Flag) {
            EXPECT_EQ(tilingData.get_kAL1(), ConvCeilDiv(4 * tilingData.get_kernelH() * tilingData.get_kernelW(), k0) * k0);
            EXPECT_EQ(tilingData.get_kBL1(), ConvCeilDiv(4 * tilingData.get_kernelH() * tilingData.get_kernelW(), k0) * k0);
            if (tilingData.get_orgHi() > 1) {
                // w fullload in L1
                EXPECT_EQ(tilingData.get_woL1(), ConvCeilDiv(tilingData.get_orgWo(), m0) * m0);
            }
            // if tilingData.get_orgHi() == 1, process is same as NO_C04_SITUATION
            // fmap fullload in L1, woL1 == AlignB(tilingIns_->shapeInfo.singleWo, tilingIns_->cubeInfo.m0)
            // woL1 may not be able to divide woL0 exactly
        } else {
            EXPECT_LE(tilingData.get_kAL1(), ConvCeilDiv(tilingData.get_singleCoreCi(), k0) * k0 * tilingData.get_kernelH() * tilingData.get_kernelW());
            EXPECT_LE(tilingData.get_kBL1(), ConvCeilDiv(tilingData.get_singleCoreCi(), k0) * k0 * tilingData.get_kernelH() * tilingData.get_kernelW());
            EXPECT_EQ(tilingData.get_kAL1() % (k0 * tilingData.get_kernelH() * tilingData.get_kernelW()), 0);
            EXPECT_EQ(tilingData.get_kBL1() % (k0 * tilingData.get_kernelH() * tilingData.get_kernelW()), 0);
            uint32_t kAL1DivCheck = 0;
            if (tilingData.get_kAL1() % tilingData.get_kL0() == 0 ||
                tilingData.get_kAL1() == ConvCeilDiv(tilingData.get_singleCoreCi(), k0) * k0 * tilingData.get_kernelH() * tilingData.get_kernelW()) {
                kAL1DivCheck = 1;
            }
            EXPECT_EQ(kAL1DivCheck, 1);
            uint32_t kBL1DivCheck = false;
            if (tilingData.get_kBL1() % tilingData.get_kL0() == 0 ||
                tilingData.get_kBL1() == ConvCeilDiv(tilingData.get_singleCoreCi(), k0) * k0 * tilingData.get_kernelH() * tilingData.get_kernelW()) {
                kBL1DivCheck = true;
            }
            EXPECT_EQ(kBL1DivCheck, 1);

            // No woL1 % woL0 check
            // In fmap fullload situation, woL1 is not obtained by magnifying, thus woL1 may not be able to divide woL0 exactly
            // In fmap fullload situation, since woL1 needs to align to m0, thus woL1 may be not equal to singleCoreWo
        }
    }
}

uint64_t ConvAlignB(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return 0;
    }
    return ((a + b - 1) / b) * b;
}

void Conv2DCase(vector<uint64_t> fmShape, vector<uint64_t> weightShape,
                vector<uint32_t> pads, vector<uint32_t> strides, vector<uint32_t> dilations,
                vector<uint32_t> NumBlocks, std::vector<ConvDtype> dtypes,
                std::vector<bool> flags,
                std::vector<uint8_t> modes,
                uint32_t groups)
{
    bool hasBias = flags[0];
    bool hasScale = flags[1];
    bool isC04Flag = flags[2];

    uint8_t outputOrder = modes[0];
    uint8_t roundMode = modes[1];
    uint8_t offsetX = modes[2];

    uint64_t orgCi = fmShape[0];
    uint64_t orgHi = fmShape[1];
    uint64_t orgWi = fmShape[2];
    uint64_t orgCo = weightShape[0];
    uint64_t orgKh = weightShape[1];
    uint64_t orgKw = weightShape[2];
    uint64_t orgCi_weight = orgCi / groups;

    uint32_t hoDim = 0;
    uint32_t moDim = 0;
    if (outputOrder == 1) {
        moDim = NumBlocks[0];
    } else {
        hoDim = NumBlocks[0];
    }
    uint32_t nDim = NumBlocks[1];

    uint32_t padTop = pads[0];
    uint32_t padBottom = pads[1];
    uint32_t padLeft = pads[2];
    uint32_t padRight = pads[3];
    uint32_t dilationH = dilations[0];
    uint32_t dilationW = dilations[1];
    uint32_t strideH = strides[0];
    uint32_t strideW = strides[1];

    uint64_t orgHo = InferHo(orgHi, orgKh, padTop, padBottom, dilationH, strideH);
    uint64_t orgWo = InferWo(orgWi, orgKw, padLeft, padRight, dilationW, strideW);
    EXPECT_GT(orgHo, 0);
    EXPECT_GT(orgWo, 0);
    int64_t singlekH = orgKh;
    int64_t singlekW = orgKw;
    int64_t singleCi = orgCi;
    int64_t singleCi_weight = orgCi_weight;
    uint64_t n0 = 16;
    int64_t singleCo = ConvAlignB(ConvCeilDiv(ConvAlignB(orgCo, n0), nDim), n0);
    int64_t singleWo = -1;
    int64_t singleHo = -1;
    int64_t singleM = -1;
    if (outputOrder == 1) {
        singleM = ConvCeilDiv(orgHo * orgWo, moDim);
    } else {
        singleWo = orgWo;
        singleHo = ConvCeilDiv(orgHo, hoDim);
    }

    ConvDtype featuremapDtype = dtypes[0];
    ConvDtype weightDtype = dtypes[1];
    ConvDtype biasDtype;
    ConvDtype outputDtype;
    if (hasBias) {
        biasDtype = dtypes[2];
        outputDtype = dtypes[3];
    } else {
        outputDtype = dtypes[2];
    }

    PlatformInfo platformInfo;
    platformInfo.npuArch = NpuArch::DAV_3510;
    platformInfo.l1Size = 524288;
    platformInfo.l0ASize = 65536;
    platformInfo.l0BSize = 65536;
    platformInfo.l0CSize = 262144;
    platformInfo.ubSize = 262144;
    platformInfo.btSize = 4096;
    platformInfo.fbSize = 4096;
    platformInfo.aivPerAic = 2;

    Conv2dTiling testTiling(platformInfo);
    testTiling.SetOrgWeightShape(orgCo, orgKh, orgKw);
    testTiling.SetOrgFmapShape(orgCi, orgHi, orgWi);
    testTiling.SetSingleWeightShape(singleCi_weight, singlekH, singlekW);

    if (outputOrder == 1) {
        testTiling.SetSingleOutputShape(singleCo, singleM, 1);
        testTiling.SetOutputOrder(1);
    } else {
        testTiling.SetSingleOutputShape(singleCo, singleHo, singleWo, 1);
        testTiling.SetOutputOrder(0);
    }
    if (hasScale) {
        testTiling.SetQuantScale(hasScale);
        testTiling.SetRoundMode(roundMode);
        testTiling.SetOffsetx(offsetX);
    }
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, featuremapDtype);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, weightDtype);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, outputDtype);
    if (hasBias) {
        testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, biasDtype);
    }
    testTiling.SetPadding(padTop, padBottom, padLeft, padRight);
    testTiling.SetDilation(dilationH, dilationW);
    testTiling.SetStride(strideH, strideW);
    testTiling.SetC04Flag(isC04Flag);
    testTiling.SetGroups(groups);
    optiling::TConv2DTiling tilingData;
    int64_t ret = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret, 0);

    CheckValidTilingData(tilingData, testTiling);
}

}