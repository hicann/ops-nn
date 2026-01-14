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
 * \file test_conv3d_v2_ascendc_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "log/log.h"
#include "array_ops.h"
#include "tests/ut/common/ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "tests/ut/common/kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "conv/conv3d_v2/op_host/op_tiling/conv3d_base_tiling.h"
#include "conv/common/op_host/op_tiling/arch35/conv_base_utils.h"

using namespace std;
using namespace ge;
using namespace ut_util;

namespace {
class Conv3dv2Tiling : public testing::Test {
    protected:
      static void SetUpTestCase() {}
      static void TearDownTestCase() {}
};

uint64_t InferDoForConv3dV2(uint64_t inputDi, uint64_t singlekD, uint64_t padHead, uint64_t padTail, uint64_t dilationD, uint64_t strideD)
{
    return (inputDi + padHead + padTail - dilationD * (singlekD - 1) - 1) / strideD + 1;
}

uint64_t InferHoForConv3dV2(uint64_t inputHiL1, uint64_t singlekH, uint64_t padTop, uint64_t padBottom, uint64_t dilationH, uint64_t strideH)
{
    return (inputHiL1 + padTop + padBottom - dilationH * (singlekH - 1) - 1) / strideH + 1;
}

uint64_t InferWoForConv3dV2(uint64_t inputWiL1, uint64_t singlekW, uint64_t padLeft, uint64_t padRight, uint64_t dilationW, uint64_t strideW)
{
    return (inputWiL1 + padLeft + padRight - dilationW * (singlekW - 1) - 1) / strideW + 1;
}

int64_t InferHiL1ForConv3dV2(uint64_t inputHoL1, int64_t hi, uint64_t singlekH, uint64_t dilationH, uint64_t strideH)
{
    int64_t khDilated = (singlekH - 1) * dilationH + 1;
    int64_t tmpHiL1 = (inputHoL1 - 1) * strideH + khDilated;
    if (tmpHiL1 > hi) {
        tmpHiL1 = hi;
    }

    return tmpHiL1;
}

int64_t InferWiL1ForConv3dV2(uint64_t inputWoL1, int64_t wi, uint64_t singlekW, uint64_t dilationW, uint64_t strideW)
{
    int64_t kwDilated = (singlekW - 1) * dilationW + 1;
    int64_t tmpWiL1 = (inputWoL1 - 1) * strideW + kwDilated;
    if (tmpWiL1 > wi) {
        tmpWiL1 = wi;
    }

    return tmpWiL1;
}

uint64_t CeilDivForConv3dV2(uint64_t a, uint64_t b)
{
    return (a + b - 1) / b;
}

uint64_t gcd(uint64_t a, uint64_t b) { // Get the greatest common divisor of a and b
    while (b != 0) {
        uint64_t temp = a % b;
        a = b;
        b = temp;
    }
    return a;
}

struct TilingParam {
    // api tilingdata
    uint64_t orgDo;
    uint64_t orgHo;
    uint64_t orgWo;
    uint64_t orgDi;
    uint64_t orgHi;
    uint64_t orgWi;
    uint64_t singleCoreBatch;
    uint64_t singleCoreDo;
    uint64_t singleCoreM;
    uint64_t singleCoreWo;
    uint64_t singleCoreHo;
    uint64_t kL0xorgCoAlignN0;
    uint64_t kernelHxkernelW;
    uint64_t cin1xOriHixOriWixk0;
    uint64_t oriHixOriWixk0;
    uint64_t oriWixk0;
    uint64_t orgHixWi;
    uint64_t orgHoxWo;
    uint32_t orgCi;
    uint32_t kernelD;
    uint32_t kernelH;
    uint32_t kernelW;
    uint32_t singleCoreCo;
    uint32_t orgCo;
    uint32_t singleCoreCi;
    uint32_t singleCoreGroups;
    uint32_t singleCoreGroupOpt;
    uint32_t groups_api;
    uint32_t enlarge_api;
    uint32_t strideH_api;
    uint32_t strideW_api;
    uint32_t strideD_api;
    uint32_t dilationH_api;
    uint32_t dilationW_api;
    uint32_t dilationD_api;
    uint32_t padHead_api;
    uint32_t padTail_api;
    uint32_t padTop_api;
    uint32_t padBottom_api;
    uint32_t padLeft_api;
    uint32_t padRight_api;
    uint32_t mL0;
    uint32_t woL0;
    uint32_t kL0;
    uint32_t nL0;
    uint32_t kAL1;
    uint32_t kAL1Tail;
    uint32_t kBL1;
    uint32_t kBL1Tail;
    uint32_t nBL1;
    uint32_t mAL1;
    uint32_t woL1;
    uint32_t hoL0;
    uint32_t hoL1;
    uint32_t KBL1Divk0;
    uint32_t KBL1TailDivk0;
    uint32_t nBL1DivnL0;
    uint32_t mAL1DivmL0;
    uint32_t fmapKStride;
    uint32_t weightKStride;
    uint32_t cinOffsetBlockInGM;
    uint32_t coutOffsetBlock;
    uint32_t nL1DivBlockSize;
    uint32_t cin1InAL1;
    uint32_t cin1InAL1Tail;
    uint32_t cinBInCore;
    uint32_t cinBTailInCore;
    uint32_t cinAInCore;
    uint32_t cinATailInCore;
    uint32_t nL0xk0;
    uint32_t mStep;
    uint32_t kStep;
    uint32_t nStep;
    uint32_t aL1SpaceSize;
    uint32_t multiNBL1;
    uint32_t pBufferFlag;
    uint32_t groupOpt_api;
    uint32_t cinOpt_api;
    uint32_t coutOpt_api;
    uint32_t mUB;
    uint32_t nUB;
    uint32_t scaleAndBiasLoadType;
    uint32_t workspaceSize;
    uint32_t kernelHxkernelWxkernelD;
    int8_t offsetx;
    int8_t roundMode;
    uint8_t hasBias_api;
    uint8_t hasScale;
    uint8_t bl1FullLoad;
    uint8_t al1FullLoad;
    uint8_t bl1BypassFlag;
    uint8_t iterateMNOrder;
    uint8_t biasFullLoadFlag;
    uint8_t fixpParamsFullLoadFlag;
    uint8_t hf32Enable;
    uint8_t hf32TransMode;
    uint8_t quantType;
    uint8_t resvered1;
    uint8_t resvered2;
    uint8_t resvered3;
    // ops tilingdata
    uint32_t batch;
    uint32_t cin;
    uint32_t din;
    uint64_t hin;
    uint64_t win;
    uint32_t cout;
    uint32_t kd;
    uint32_t kh;
    uint32_t kw;
    uint32_t dout;
    uint64_t hout;
    uint64_t wout;
    uint32_t batchDim;
    uint32_t doDim;
    uint32_t mDim;
    uint32_t wDim;
    uint32_t nDim;
    uint32_t groupDim;
    uint32_t hoDim;
    uint32_t strideH;
    uint32_t strideW;
    uint32_t strideD;
    uint32_t dilationH;
    uint32_t dilationW;
    uint32_t dilationD;
    uint32_t padHead;
    uint32_t padTail;
    uint32_t padTop;
    uint32_t padBottom;
    uint32_t padLeft;
    uint32_t padRight;
    uint32_t groups;
    uint32_t enlarge;
    uint32_t cinOpt;
    uint32_t coutOpt;
    uint32_t groupOpt;
    uint8_t hasBias;
};

uint64_t CalcConv3dUsdL1Size(TilingParam &tilingData,
                       uint32_t featuremapDtyeSize,
                       uint32_t weightDtyeSize,
                       uint64_t n0,
                       uint32_t outputOrder,
                       int8_t pbAL1,
                       int8_t pbBL1,
                       int64_t padCompensationValue,
                       bool hasBias,
                       bool hasQuantScale,
                       bool isC04Flag=false)
{
    uint32_t biasDtyeSize = 0;
    uint32_t scaleDtyeSize = 0;
	if(hasBias) {
		if(hasQuantScale) {
      biasDtyeSize = 4; // biasdetype is int32 for int8; biasdetype is fp32 for hif8/fp8
		}
		else {
			biasDtyeSize = featuremapDtyeSize; // biasdtype is same as fmdtype for fp32/hf32/fp16/bf16
		}
	}
	if(hasQuantScale) {
		scaleDtyeSize = 8; // scaleDtye is int64/uint64 for int8/hif8/fp8
	}
    uint64_t curl1Size = 0;
    uint64_t al1Size = 0;
    uint64_t bl1Size = 0;
    uint64_t biasL1Size = 0;
    uint64_t scaleL1Size = 0;
    if (outputOrder == 1) { // Mmode
        uint64_t hoAL1Tmp = tilingData.hoL1 / tilingData.orgWo + 2;
        uint64_t hiL1Tmp = min((hoAL1Tmp - 1) * tilingData.strideH + (tilingData.kernelH - 1) / tilingData.dilationH + 1, tilingData.orgHi);
        al1Size = hiL1Tmp * tilingData.orgWi * (tilingData.kAL1 / (tilingData.kernelH * tilingData.kernelW)) * (pbAL1 + 1) * featuremapDtyeSize;
        bl1Size = tilingData.nBL1 * tilingData.kBL1 * (pbBL1 + 1) * weightDtyeSize;
        if (hasBias) {
            if (tilingData.biasFullLoadFlag == 1) {
                biasL1Size = CeilDivForConv3dV2(tilingData.singleCoreCo, n0) * n0 * biasDtyeSize;
            } else { // tilingData.biasFullLoadFlag == 0
                biasL1Size = tilingData.nL0 * biasDtyeSize;
            }
        }
        if (hasQuantScale) {
            if (tilingData.fixpParamsFullLoadFlag) {
                scaleL1Size = CeilDivForConv3dV2(tilingData.singleCoreCo, n0) * n0 * scaleDtyeSize;
            } else {
                scaleL1Size = tilingData.nL0 * scaleDtyeSize;
            }
        }
        curl1Size = al1Size + bl1Size + biasL1Size + scaleL1Size;
    } else { // HWmode
        uint64_t hiL1 = InferWiL1ForConv3dV2(tilingData.hoL1, tilingData.orgHi, tilingData.kernelH, tilingData.dilationH, tilingData.strideH);
        uint64_t wiL1 = InferWiL1ForConv3dV2(tilingData.woL1, tilingData.orgWi, tilingData.kernelW, tilingData.dilationW_api, tilingData.strideW_api);
        al1Size = hiL1 * wiL1 * (tilingData.kAL1 / (tilingData.kernelH * tilingData.kernelW)) * (pbAL1 + 1) * featuremapDtyeSize;
        if (isC04Flag) {
            al1Size = CeilDivForConv3dV2(hiL1 * wiL1 * 4 * featuremapDtyeSize, 32) * 32 * (pbAL1 + 1);
        }
        bl1Size = tilingData.nBL1 * tilingData.kBL1 * weightDtyeSize;
        if (hasBias) {
            if (tilingData.biasFullLoadFlag) {
                biasL1Size = CeilDivForConv3dV2(tilingData.singleCoreCo, n0) * n0 * biasDtyeSize;
            } else {
                biasL1Size = tilingData.nBL1 * biasDtyeSize;
            }
        }
        if (hasQuantScale) {
            if (tilingData.fixpParamsFullLoadFlag) {
                scaleL1Size = CeilDivForConv3dV2(tilingData.singleCoreCo, n0) * n0 * scaleDtyeSize;
            } else {
                scaleL1Size = tilingData.nBL1 * scaleDtyeSize;
            }
        }
        curl1Size = al1Size + bl1Size + biasL1Size + scaleL1Size;
    }

    return curl1Size;
}

uint64_t CalcConv3dUsdL0ASize(TilingParam &tilingData,
                        uint32_t outputOrder,
                        uint32_t featuremapDtyeSize,
                        int8_t pbAL0)
{
    uint64_t curl0aSize = 0;
    if (outputOrder == 0) {
        curl0aSize = tilingData.hoL0 * tilingData.woL0 * tilingData.kL0 * (pbAL0 + 1) * featuremapDtyeSize;
    } else {
        curl0aSize = tilingData.hoL0 * tilingData.kL0 * (pbAL0 + 1) * featuremapDtyeSize;
    }
    return curl0aSize;
}

uint64_t CalcConv3dUsdL0BSize(TilingParam &tilingData,
                        uint32_t weightDtyeSize,
                        int8_t pbBL0)
{
    return tilingData.nL0 * tilingData.kL0 * (pbBL0 + 1) * weightDtyeSize;
}

uint64_t CalcConv3dUsdL0CSize(TilingParam &tilingData,
					              uint32_t outputOrder,
                        int8_t pbCL0)
{
    uint64_t curl0cSize = 0;
    uint32_t mmadDtypeSize = 4;
    if (outputOrder == 0) {
        curl0cSize = tilingData.hoL0 * tilingData.woL0 * tilingData.nL0 * (pbCL0 + 1) * mmadDtypeSize;
    } else {
        curl0cSize = tilingData.hoL0 * tilingData.nL0 * (pbCL0 + 1) * mmadDtypeSize;
    }
    return curl0cSize;
}

void CheckHWModeTilingDataValidForConv3dV2(TilingParam &tilingData, uint64_t m0, uint64_t n0, uint64_t k0, bool isC04Flag)
{
        // K direction check
        EXPECT_GE(tilingData.kL0, k0);
        EXPECT_LE(tilingData.kL0, std::min(tilingData.kAL1, tilingData.kBL1));
        EXPECT_EQ(tilingData.kL0 % k0, 0);

        // N direction check
        EXPECT_GE(tilingData.nL0, n0);
        EXPECT_GE(tilingData.nBL1, tilingData.nL0);
        EXPECT_LE(tilingData.nBL1, CeilDivForConv3dV2(CeilDivForConv3dV2(tilingData.singleCoreCo, n0) * n0, tilingData.nL0) * tilingData.nL0);
        EXPECT_EQ(tilingData.nL0 % n0, 0);
        uint32_t nBL1DivCheck = 0;
        if (tilingData.nBL1 % tilingData.nL0 == 0 ||
            tilingData.nBL1 == CeilDivForConv3dV2(tilingData.singleCoreCo, n0) * n0) {
            nBL1DivCheck = 1;
        }
        EXPECT_EQ(nBL1DivCheck, 1);

        // W direction check
        EXPECT_GE(tilingData.woL0, m0);
        EXPECT_GE(tilingData.woL1, tilingData.woL0);
        EXPECT_LE(tilingData.woL1,
            CeilDivForConv3dV2(CeilDivForConv3dV2(tilingData.singleCoreWo, m0) * m0, tilingData.woL0) * tilingData.woL0);
        EXPECT_EQ(tilingData.woL0 % m0, 0);
        if (tilingData.woL0 < CeilDivForConv3dV2(tilingData.orgWo, m0) * m0) {
            // woL0 does not reach the upper limit, thus hoL0 must be 1.
            EXPECT_EQ(tilingData.hoL0, 1);
        }
        if (tilingData.hoL0 > 1) {
            EXPECT_EQ(tilingData.woL0, CeilDivForConv3dV2(tilingData.orgWo, m0) * m0);
            EXPECT_EQ(tilingData.woL1, CeilDivForConv3dV2(tilingData.orgWo, m0) * m0);
        }

        // H direction check
        uint32_t hoL1DivCheck = 0;
        EXPECT_GE(tilingData.hoL0, 1);
        EXPECT_GE(tilingData.hoL1, tilingData.hoL0);
        EXPECT_LE(tilingData.hoL1, tilingData.singleCoreHo);
        uint32_t hoL1Check = 0;
        if (tilingData.hoL1 % tilingData.hoL0 == 0 || tilingData.hoL1 == tilingData.singleCoreHo) {
            hoL1Check = 1;
        }
        EXPECT_EQ(hoL1Check, 1);

        if (isC04Flag) {
            EXPECT_EQ(tilingData.kAL1, CeilDivForConv3dV2(4 * tilingData.kernelH * tilingData.kernelW, k0) * k0);
            EXPECT_EQ(tilingData.kBL1, CeilDivForConv3dV2(4 * tilingData.kernelH * tilingData.kernelW, k0) * k0);
            if (tilingData.orgHi > 1) {
                // w fullload in L1
                EXPECT_EQ(tilingData.woL1, CeilDivForConv3dV2(tilingData.orgWo, m0) * m0);
            }
            // if tilingData.orgHi == 1, process is same as NO_C04_SITUATION
            // fmap fullload in L1, woL1 == AlignB(tilingIns_->shapeInfo.singleWo, tilingIns_->cubeInfo.m0)
            // woL1 may not be able to divide woL0 exactly
        } else {
            EXPECT_LE(tilingData.kAL1, CeilDivForConv3dV2(tilingData.singleCoreCi, k0) * k0 * tilingData.kernelH * tilingData.kernelW * tilingData.kernelD);
            EXPECT_LE(tilingData.kBL1, CeilDivForConv3dV2(tilingData.singleCoreCi, k0) * k0 * tilingData.kernelH * tilingData.kernelW * tilingData.kernelD);
            EXPECT_EQ(tilingData.kAL1 % (k0 * tilingData.kernelH * tilingData.kernelW), 0);
            EXPECT_EQ(tilingData.kBL1 % (k0 * tilingData.kernelH * tilingData.kernelW), 0);
            uint32_t kAL1DivCheck = 0;
            if (tilingData.kAL1 % tilingData.kL0 == 0 ||
                tilingData.kAL1 == CeilDivForConv3dV2(tilingData.singleCoreCi, k0) * k0 * tilingData.kernelH * tilingData.kernelW) {
                kAL1DivCheck = 1;
            }
            EXPECT_EQ(kAL1DivCheck, 1);
            uint32_t kBL1DivCheck = false;
            if (tilingData.kBL1 % tilingData.kL0 == 0 ||
                tilingData.kBL1 == CeilDivForConv3dV2(tilingData.singleCoreCi, k0) * k0 * tilingData.kernelH * tilingData.kernelW) {
                kBL1DivCheck = true;
            }
            EXPECT_EQ(kBL1DivCheck, 1);

            // No woL1 % woL0 check
            // In fmap fullload situation, woL1 is not obtained by magnifying, thus woL1 may not be able to divide woL0 exactly
            // In fmap fullload situation, since woL1 needs to align to m0, thus woL1 may be not equal to singleCoreWo
        }
}

void CheckValidTilingData(TilingParam &tilingData,
                          uint64_t m0,
                          uint64_t n0,
                          uint64_t k0,
                          uint32_t weightDtypeSize,
                          uint32_t featuremapDtypeSize,
						  uint32_t biasDtypeSize,
						  uint32_t mmadDtypeSize,
                          uint64_t l1Size,
                          uint64_t l0aSize,
                          uint64_t l0bSize,
                          uint64_t l0cSize,
                          bool hasBias,
                          uint64_t tilingKey,
                          bool isConv3dDequant)
{
    int32_t outputOrder = tilingData.singleCoreWo == 0 && tilingData.woL1 == 0;
    if (isConv3dDequant) {
      ASSERT_GT(tilingData.mUB, 0);
      ASSERT_GT(tilingData.nUB, 0);
      EXPECT_EQ(tilingData.nUB % n0, 0);
    }
    // check size
    uint64_t pBuffer = tilingData.pBufferFlag;
    int8_t pbAL0 = pBuffer & 0x01;
    int8_t pbBL0 = (pBuffer & 0x02) >> 1;
    int8_t pbCL0 = (pBuffer & 0x04) >> 2;
    int8_t pbAL1 = (pBuffer & 0x08) >> 3;
    int8_t pbBL1 = (pBuffer & 0x10) >> 4;
    uint64_t multi_nBL1max = CeilDivForConv3dV2(CeilDivForConv3dV2(tilingData.singleCoreCo, n0) * n0, tilingData.nL0);
    uint64_t multi_mAL1max = CeilDivForConv3dV2(CeilDivForConv3dV2(tilingData.singleCoreHo, m0) * m0, tilingData.hoL0);
    uint64_t mL0max = min(l0aSize / (k0 * (pbAL0 + 1) * featuremapDtypeSize), l0cSize / (n0 * (pbCL0 + 1) * mmadDtypeSize));
    uint64_t nL0max = min(l0bSize / (k0 * (pbBL0 + 1) * weightDtypeSize), l0cSize / (m0 * (pbCL0 + 1) * mmadDtypeSize));


    if (outputOrder == 1) {
      ASSERT_GT(tilingData.kAL1, 0);
      ASSERT_GT(tilingData.kBL1, 0);
      ASSERT_GT(tilingData.hoL1, 0);
      ASSERT_GT(tilingData.nBL1, 0);
      ASSERT_GT(tilingData.nL0, 0);
      ASSERT_GT(tilingData.hoL0, 0);
      ASSERT_GT(tilingData.kL0, 0);
      EXPECT_LE(tilingData.nBL1, multi_nBL1max * tilingData.nL0);
      EXPECT_LE(tilingData.hoL1, multi_mAL1max * tilingData.hoL0);
      EXPECT_LE(tilingData.kAL1, tilingData.kernelD * tilingData.kernelH * tilingData.kernelW * CeilDivForConv3dV2(tilingData.orgCi, k0) * k0);
      EXPECT_LE(tilingData.kBL1, tilingData.kernelD * tilingData.kernelH * tilingData.kernelW * CeilDivForConv3dV2(tilingData.orgCi, k0) * k0);
      EXPECT_LE(tilingData.hoL0, CeilDivForConv3dV2(mL0max, m0) * m0);
      EXPECT_LE(tilingData.nL0, CeilDivForConv3dV2(nL0max, n0) * n0);
      EXPECT_LE(tilingData.kL0, gcd(CeilDivForConv3dV2(tilingData.kAL1, k0), CeilDivForConv3dV2(tilingData.kBL1, k0)) * k0);

      EXPECT_EQ(tilingData.kAL1 % k0, 0);
      EXPECT_EQ(tilingData.kBL1 % k0, 0);
      EXPECT_EQ(tilingData.nBL1 % n0, 0);
      EXPECT_EQ(tilingData.woL1 % m0, 0);
      EXPECT_EQ(tilingData.nL0 % n0, 0);
      EXPECT_EQ(tilingData.kL0 % k0, 0);
      EXPECT_EQ(tilingData.woL0 % m0, 0);

      EXPECT_EQ(tilingData.kAL1 % tilingData.kL0, 0);
      EXPECT_EQ(tilingData.kBL1 % tilingData.kL0, 0);
    } else if (outputOrder == 0) {
      bool isC04Flag = false;
      CheckHWModeTilingDataValidForConv3dV2(tilingData, m0, n0, k0, isC04Flag);
    }

    EXPECT_LE(CalcConv3dUsdL1Size(tilingData, featuremapDtypeSize, weightDtypeSize, n0, outputOrder, pbAL1, pbBL1, 0, hasBias, false, false), l1Size);
    EXPECT_LE(CalcConv3dUsdL0ASize(tilingData, outputOrder, featuremapDtypeSize, pbAL0), l0aSize);
    EXPECT_LE(CalcConv3dUsdL0BSize(tilingData, weightDtypeSize, pbBL0), l0bSize);
    EXPECT_LE(CalcConv3dUsdL0CSize(tilingData, outputOrder, pbCL0), l0cSize);
}

void GetOriPadFromPadMode(const string& padMode, uint32_t& padh, uint32_t& padt, uint32_t& padu, uint32_t& padd,
  uint32_t& padl, uint32_t& padr, uint32_t strideD, uint32_t strideH, uint32_t strideW,
  uint32_t dilationD, uint32_t dilationH, uint32_t dilationW, int64_t batch, int64_t cin, int64_t di, int64_t hi, int64_t wi,
  int64_t cout, int64_t kD, int64_t kH, int64_t kW)
{
    if (padMode == "SPECIFIC") {
        return;
    }

    if (padMode == "VALID") {
        padh = 0;
        padt = 0;
        padu = 0;
        padd = 0;
        padl = 0;
        padr = 0;
        return;
    } else {
        auto padD = (CeilDivForConv3dV2(di, strideD) - 1) * strideD + dilationD * (kD - 1) - di + 1;
        auto padH = (CeilDivForConv3dV2(hi, strideH) - 1) * strideH + dilationH * (kH - 1) - hi + 1;
        auto padW = (CeilDivForConv3dV2(wi, strideW) - 1) * strideW + dilationW * (kW - 1) - wi + 1;
        if (padMode == "SAME" || padMode == "SAME_UPPER") {
            padt = CeilDivForConv3dV2(padD, 2);
            padh = padD - padt;
            padd = CeilDivForConv3dV2(padH, 2);
            padu = padH - padd;
            padr = CeilDivForConv3dV2(padW, 2);
            padl = padW - padr;
        } else {
            // padMode is "SAME_LOWER"
            padh = CeilDivForConv3dV2(padD, 2);
            padt = padD - padh;
            padu = CeilDivForConv3dV2(padH, 2);
            padd = padH - padu;
            padl = CeilDivForConv3dV2(padW, 2);
            padr = padW - padl;
        }
    }
    return;
}

void Conv3DV2TestCase(vector<int64_t> fmShape, vector<int64_t> weightShape,
                    vector<uint32_t> pads, vector<uint32_t> strides, vector<uint32_t> dilations, ge::DataType dtype,
                    uint32_t isHasBias = 1, uint32_t groups = 1, int64_t fixBatcho = 0, int64_t fixDo = 0, int64_t fixHo = 0,
                    int64_t fixWo = 0, bool isErrorCaseFlag = false, string padMode = "SPECIFIC", bool enableHf32 = false,
                    string shortSocVersion = "Ascend910_95", ge::Format format = ge::Format::FORMAT_NCDHW,
                    ge::Format outformat = ge::Format::FORMAT_NCDHW,
                    bool isConv3dDequant=false, bool hasScale = false, ge::DataType biasDtypeIn = ge::DT_FLOAT16,
                    ge::DataType outputDtypeIn = ge::DT_FLOAT16, ge::DataType scaleDtypeIn = ge::DT_FLOAT16) {
	uint32_t  mmadDtypesize = 4;//mmadDtype is FLOAT32 in develop4

  bool hasBias = (isHasBias == 1);
	uint32_t padh = pads[0];
  uint32_t padt = pads[1];
  uint32_t padu = pads[2];
  uint32_t padd = pads[3];
  uint32_t padl = pads[4];
  uint32_t padr = pads[5];
  uint32_t strideD = strides[0];
  uint32_t strideH = strides[1];
  uint32_t strideW = strides[2];
  uint32_t dilationD = dilations[0];
  uint32_t dilationH = dilations[1];
  uint32_t dilationW = dilations[2];
  int64_t cout = weightShape[0];
  int64_t kD = weightShape[1];
  int64_t kH = weightShape[2];
  int64_t kW = weightShape[3];
  int64_t batch = fmShape[0];
  int64_t cin = fmShape[1];
  int64_t di = fmShape[2];
  int64_t hi = fmShape[3];
  int64_t wi = fmShape[4];
  GetOriPadFromPadMode(padMode, padh, padt, padu, padd, padl, padr, strideD, strideH, strideW,
    dilationD, dilationH, dilationW, batch, cin, di, hi, wi, cout, kD, kH, kW);
  int64_t Do = InferDoForConv3dV2(di, kD, padh, padt, dilationD, strideD);
  int64_t ho = InferHoForConv3dV2(hi, kH, padu, padd, dilationH, strideH);
  int64_t wo = InferWoForConv3dV2(wi, kW, padl, padr, dilationW, strideW);
  int64_t batcho = batch;
  if (fixBatcho != 0) {
    batcho = fixBatcho;
  }
  if (fixDo != 0) {
    Do = fixDo;
  }
  if (fixHo != 0) {
    ho = fixHo;
  }
  if (fixWo != 0) {
    wo = fixWo;
  }
  ge::Format fmapFormat = ge::FORMAT_NCDHW;
  ge::Format weightFormat = ge::FORMAT_NCDHW;
  ge::Format outputFormat = ge::FORMAT_NCDHW;
  gert::StorageShape featuremap = {{batch, cin, di, hi, wi}, {batch, cin, di, hi, wi}};
  gert::StorageShape weight = {{cout, cin / groups, kD, kH, kW}, {cout, cin / groups, kD, kH, kW}};
  gert::StorageShape bias = {{cout}, {cout}};
  gert::StorageShape scale = {{cout}, {cout}};
  gert::StorageShape offset_w;
  gert::StorageShape output = {{batcho, cout, Do, ho, wo}, {batcho, cout, Do, ho, wo}};
  if (format == ge::FORMAT_NDHWC) {
    fmapFormat = ge::FORMAT_NDHWC;
    weightFormat = ge::FORMAT_DHWCN;
    outputFormat = ge::FORMAT_NDHWC;
    featuremap = {{batch, di, hi, wi, cin}, {batch, di, hi, wi, cin}};
    weight = {{kD, kH, kW, cin / groups, cout}, {kD, kH, kW, cin / groups, cout}};
    output = {{batcho, Do, ho, wo, cout}, {batch, Do, ho, wo, cout}};
  }
  if (outformat == ge::FORMAT_NDHWC) {
    outputFormat = ge::FORMAT_NDHWC;
    output = {{batcho, Do, ho, wo, cout}, {batch, Do, ho, wo, cout}};
  }
  // 对于可选输入，不传时用nullptr占位
  std::vector<void*> input_shape_ref;
	if(hasBias) {
		input_shape_ref = {&featuremap, &weight, &bias};
	} else {
		input_shape_ref = {&featuremap, &weight, nullptr};
	}

	if(hasScale) {
		input_shape_ref.push_back(&scale);
	} else {
    input_shape_ref.push_back(nullptr);
  }

    std::vector<void*> output_shapes_ref = {&output};
    std::vector<int64_t> strides_ref = {1, 1, strideD, strideH, strideW};
    std::vector<int64_t> pads_ref = {padh, padt, padu, padd, padl, padr};
    std::vector<int64_t> dilations_ref = {1, 1, dilationD, dilationH, dilationW};
    if (format == ge::FORMAT_NDHWC) {
      strides_ref = {1, strideD, strideH, strideW, 1};
      dilations_ref = {1, dilationD, dilationH, dilationW, 1};
    }

    std::string op_type = "Conv3DV2";
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

	uint64_t L1_SIZE = 524288;
    uint64_t L0a_SIZE = 65536;
    uint64_t L0b_SIZE = 65536;
    uint64_t L0c_SIZE = 262144;
    uint64_t aicoreNum = 32;
    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144,
                            "CORE_NUM": 32}
                            })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    map<string, string> soc_version_infos = {{"Short_SoC_version", shortSocVersion}};
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::conv_ops_tiling::ConvTilingParseInfo compile_info;
    compile_info.tilingType = op_type;
    compile_info.aicoreNum = aicoreNum;
    compile_info.shortSocVersion = shortSocVersion;
    auto tilingDataPtr = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(tilingDataPtr, nullptr);

    std::map<ge::DataType, uint32_t> k0Map = {{ge::DT_FLOAT16, 16}, {ge::DT_FLOAT, 8}, {ge::DT_INT8, 32}, {ge::DT_BF16, 16}, {ge::DT_HIFLOAT8, 32}};
    std::map<ge::DataType, uint32_t> dtypesizeMap = {{ge::DT_FLOAT16, 2}, {ge::DT_FLOAT, 4}, {ge::DT_INT8, 1}, {ge::DT_BF16, 2}, {ge::DT_HIFLOAT8, 1}};
    uint64_t m0 = 16;
    uint64_t k0 = k0Map.at(dtype);
    uint64_t n0 = 16;

    uint32_t featuremapDtypeSize = dtypesizeMap.at(dtype);
	  uint32_t weightDtypeSize = dtypesizeMap.at(dtype);
	  uint32_t biasDtypeSize = dtypesizeMap.at(dtype);
	  ge::DataType fmapDype = dtype;
    ge::DataType weightDtype = dtype;
    ge::DataType biasDtype = dtype == DT_HIFLOAT8 ? ge::DT_FLOAT : dtype;
    ge::DataType outputDtype = dtype;
    ge::DataType scaleDtype = dtype;
    if (isConv3dDequant) {
      biasDtype = biasDtypeIn;
      outputDtype = outputDtypeIn;
      scaleDtype = scaleDtypeIn;
      outputFormat = outformat;
    }
    auto holder = gert::TilingContextFaker().SetOpType(op_type)
                                            .NodeIoNum(4, 1)
                                            .IrInstanceNum({1, 1, 1, 1})
                                            .InputShapes(input_shape_ref)
                                            .OutputShapes(output_shapes_ref)
                                            .CompileInfo(&compile_info)
                                            .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                                            .NodeInputTd(0, fmapDype, fmapFormat, fmapFormat)
                                            .NodeInputTd(1, weightDtype, weightFormat, weightFormat)
                                            .NodeInputTd(2, biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                                            .NodeInputTd(3, scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                                            .NodeOutputTd(0, outputDtype, outputFormat, outputFormat)
                                            .NodeAttrs({
                                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides_ref)},
                                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads_ref)},
                                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations_ref)},
                                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>(padMode)},
                                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(enableHf32)}
                                                })
                                            .TilingData(tilingDataPtr.get())
                                            .Workspace(ws_size)
                                            .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    if (isErrorCaseFlag == false) {
      EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
	    auto buf = (TilingParam*)tiling_context->GetRawTilingData()->GetData();
      TilingParam tilingParam = *buf;
      uint64_t tilingKey = tiling_context->GetTilingKey();
      EXPECT_LE(tilingParam.batchDim * tilingParam.doDim * tilingParam.hoDim * tilingParam.nDim, compile_info.aicoreNum);
      EXPECT_GE(tilingParam.batchDim, 1);
      EXPECT_GE(tilingParam.doDim, 1);
      EXPECT_GE(tilingParam.hoDim, 1);
	    EXPECT_GE(tilingParam.nDim, 1);
	    if((tilingParam.batchDim >= 1) && (tilingParam.doDim >= 1) && (tilingParam.hoDim >= 1) && (tilingParam.nDim >= 1)) {
		    CheckValidTilingData(tilingParam, m0, n0, k0, weightDtypeSize, featuremapDtypeSize, biasDtypeSize,
          mmadDtypesize, L1_SIZE, L0a_SIZE, L0b_SIZE, L0c_SIZE, hasBias, tilingKey, isConv3dDequant);
	    }
    } else {
        EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    }
}
} // namespace

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_cache_init) {
  Conv3DV2TestCase({2,1,1,256,1}, {1,1,255,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_cache_0) {
  Conv3DV2TestCase({1,1,1,256,1}, {1,1,255,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_cache_1) {
  for (int i = 0; i < 10; i++) {
    Conv3DV2TestCase({1,1,1,256,1}, {1,1,255,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
  }
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_7) {
  Conv3DV2TestCase({1,1,1,256,1}, {1,1,255,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_valid) {
  Conv3DV2TestCase({1,1,256,256,256}, {1,8,3,4}, {9999,9999,9999,9999,9999,9999}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "VALID");
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_same) {
  Conv3DV2TestCase({1,1,256,256,256}, {1,8,3,4}, {9999,9999,9999,9999,9999,9999}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SAME");
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_sameUpper) {
  Conv3DV2TestCase({1,1,256,256,256}, {1,8,3,4}, {9999,9999,9999,9999,9999,9999}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SAME_UPPER");
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_sameLower) {
  Conv3DV2TestCase({1,1,256,256,256}, {1,8,3,4}, {9999,9999,9999,9999,9999,9999}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SAME_LOWER");
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_hf32) {
  Conv3DV2TestCase({1,1,256,256,256}, {1,8,3,4}, {9999,9999,9999,9999,9999,9999}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SAME_LOWER", true);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_8) {
  Conv3DV2TestCase({1,1,1,1,256}, {1,1,1,255}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_12) {
  Conv3DV2TestCase({1,3,1,300,4}, {3,1,5,3}, {1,1,1,1,1,1}, {1,63,1}, {1,3,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_13) {
  Conv3DV2TestCase({1,2,1,300,7}, {3,1,5,3}, {1,1,1,1,1,1}, {1,1,63}, {1,3,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_14) {
  Conv3DV2TestCase({1,3,1,300,6}, {3,1,1,3}, {0,0,0,0,1,1}, {1,2,1}, {1,255,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_15) {
  Conv3DV2TestCase({1,3,2,18,770}, {3,1,1,2}, {0,0,0,0,1,1}, {1,2,3}, {1,1,255}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_16) {
  Conv3DV2TestCase({1,1,1,256,1}, {1,1,255,1}, {0,0,254,254,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_17) {
  Conv3DV2TestCase({1,1,1,1,256}, {1,1,1,255}, {0,0,0,0,254,254}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_18) {
  Conv3DV2TestCase({1,4,6,75,293}, {13,3,15,6}, {0,0,0,0,0,0}, {1,18,27}, {1,1,1}, ge::DT_BF16, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_20) {
  Conv3DV2TestCase({1,3,1,16,23}, {16,3,3,3}, {3,3,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_24) {
  Conv3DV2TestCase({1,3,9,200,123}, {18,1,4,7}, {0,0,1,1,2,2}, {1,32,18}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_25) {
  Conv3DV2TestCase({1,4,1,1,873}, {11,1,1,6}, {0,0,0,0,1,2}, {1,1,18}, {1,1,1}, ge::DT_BF16, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_27) {
  Conv3DV2TestCase({1,2,11,242,10}, {4,1,1,1}, {0,0,0,0,0,0}, {1,62,7}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_28) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_29) {
  Conv3DV2TestCase({1,4,120,32,32}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_30) {
  Conv3DV2TestCase({3,4,16,26,36}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_31) {
  Conv3DV2TestCase({8,240,4,32,32}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_32) {
  Conv3DV2TestCase({8,3,18,130,130}, {240,4,4,4}, {0,0,0,0,0,0}, {2,2,2}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_33) {
  Conv3DV2TestCase({8,240,10,66,66}, {240,4,4,4}, {0,0,0,0,0,0}, {2,2,2}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_34) {
  Conv3DV2TestCase({8,240,6,34,34}, {240,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_35) {
  Conv3DV2TestCase({8,240,6,34,34}, {120,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_36) {
  Conv3DV2TestCase({8,120,4,32,32}, {240,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_37) {
  Conv3DV2TestCase({4,3,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_38) {
  Conv3DV2TestCase({4,128,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_39) {
  Conv3DV2TestCase({4,128,17,257,257}, {128,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_40) {
  Conv3DV2TestCase({4,128,11,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_41) {
  Conv3DV2TestCase({4,256,11,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_42) {
  Conv3DV2TestCase({4,128,9,128,128}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_43) {
  Conv3DV2TestCase({4,256,9,129,129}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_44) {
  Conv3DV2TestCase({4,256,7,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_45) {
  Conv3DV2TestCase({4,512,7,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_46) {
  Conv3DV2TestCase({4,256,5,64,64}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_47) {
  Conv3DV2TestCase({4,512,5,65,65}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_48) {
  Conv3DV2TestCase({4,512,7,32,32}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_49) {
  Conv3DV2TestCase({4,512,5,32,32}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_50) {
  Conv3DV2TestCase({4,512,7,32,32}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_51) {
  Conv3DV2TestCase({4,8,5,32,32}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_52) {
  Conv3DV2TestCase({16,3,3,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_53) {
  Conv3DV2TestCase({16,128,3,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_54) {
  Conv3DV2TestCase({16,128,1,257,257}, {128,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_55) {
  Conv3DV2TestCase({16,128,3,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_56) {
  Conv3DV2TestCase({16,256,3,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_57) {
  Conv3DV2TestCase({16,128,1,128,128}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_58) {
  Conv3DV2TestCase({16,256,1,129,129}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_59) {
  Conv3DV2TestCase({16,256,3,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_60) {
  Conv3DV2TestCase({16,512,3,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_61) {
  Conv3DV2TestCase({16,256,1,64,64}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_62) {
  Conv3DV2TestCase({16,512,1,65,65}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_63) {
  Conv3DV2TestCase({16,512,3,32,32}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_64) {
  Conv3DV2TestCase({16,512,1,32,32}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_65) {
  Conv3DV2TestCase({16,512,3,32,32}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_66) {
  Conv3DV2TestCase({16,8,1,32,32}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_67) {
  Conv3DV2TestCase({16,3,26,134,134}, {64,7,7,7}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_68) {
  Conv3DV2TestCase({16,64,22,130,130}, {64,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_69) {
  Conv3DV2TestCase({16,64,20,128,128}, {64,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_70) {
  Conv3DV2TestCase({16,128,22,66,66}, {128,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_71) {
  Conv3DV2TestCase({16,128,20,64,64}, {128,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_72) {
  Conv3DV2TestCase({16,256,22,34,34}, {256,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_73) {
  Conv3DV2TestCase({16,256,20,32,32}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_74) {
  Conv3DV2TestCase({16,256,20,32,32}, {1364,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_75) {
  Conv3DV2TestCase({16,682,20,32,32}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_76) {
  Conv3DV2TestCase({16,512,22,18,18}, {512,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_77) {
  Conv3DV2TestCase({16,512,20,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_78) {
  Conv3DV2TestCase({16,512,20,16,16}, {2730,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_79) {
  Conv3DV2TestCase({16,1365,20,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_80) {
  Conv3DV2TestCase({16,512,12,18,18}, {512,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_81) {
  Conv3DV2TestCase({16,512,10,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_82) {
  Conv3DV2TestCase({16,512,7,18,18}, {512,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_83) {
  Conv3DV2TestCase({16,512,5,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_84) {
  Conv3DV2TestCase({16,512,5,16,16}, {2730,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_85) {
  Conv3DV2TestCase({16,1365,5,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_86) {
  Conv3DV2TestCase({16,64,22,130,130}, {3,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_87) {
  Conv3DV2TestCase({1,320,25,40,64}, {320,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_88) {
  Conv3DV2TestCase({1,640,25,20,32}, {640,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_89) {
  Conv3DV2TestCase({1,1280,25,10,16}, {1280,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_90) {
  Conv3DV2TestCase({1,1280,25,5,8}, {1280,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_91) {
  Conv3DV2TestCase({2,320,25,40,64}, {320,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_92) {
  Conv3DV2TestCase({2,640,25,20,32}, {640,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_93) {
  Conv3DV2TestCase({2,1280,25,10,16}, {1280,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_94) {
  Conv3DV2TestCase({2,1280,25,5,8}, {1280,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_95) {
  Conv3DV2TestCase({1,512,8,40,64}, {512,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_96) {
  Conv3DV2TestCase({1,512,8,80,128}, {512,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_97) {
  Conv3DV2TestCase({1,256,8,160,256}, {256,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_98) {
  Conv3DV2TestCase({1,128,8,320,512}, {128,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_99) {
  Conv3DV2TestCase({1,3,8,320,512}, {3,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_100) {
  Conv3DV2TestCase({1,512,1,40,64}, {512,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_101) {
  Conv3DV2TestCase({1,512,1,80,128}, {512,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_102) {
  Conv3DV2TestCase({1,256,1,160,256}, {256,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_103) {
  Conv3DV2TestCase({1,128,1,320,512}, {128,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_104) {
  Conv3DV2TestCase({1,3,1,320,512}, {3,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_105) {
  Conv3DV2TestCase({1,3,16,224,224}, {768,2,16,16}, {0,0,0,0,0,0}, {2,16,16}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_106) {
  Conv3DV2TestCase({2,3,35,192,192}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_107) {
  Conv3DV2TestCase({2,128,35,192,192}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_108) {
  Conv3DV2TestCase({2,128,33,193,193}, {128,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_109) {
  Conv3DV2TestCase({2,128,35,96,96}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_110) {
  Conv3DV2TestCase({2,256,35,96,96}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_111) {
  Conv3DV2TestCase({2,128,33,96,96}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_112) {
  Conv3DV2TestCase({2,256,33,97,97}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_113) {
  Conv3DV2TestCase({2,256,19,48,48}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_114) {
  Conv3DV2TestCase({2,512,19,48,48}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_115) {
  Conv3DV2TestCase({2,256,17,48,48}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_116) {
  Conv3DV2TestCase({2,512,17,49,49}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_117) {
  Conv3DV2TestCase({2,512,11,24,24}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_118) {
  Conv3DV2TestCase({2,512,9,24,24}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_119) {
  Conv3DV2TestCase({2,512,11,24,24}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_120) {
  Conv3DV2TestCase({2,8,9,24,24}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_121) {
  Conv3DV2TestCase({2,4,9,24,24}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_122) {
  Conv3DV2TestCase({2,3,19,320,320}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_123) {
  Conv3DV2TestCase({2,128,19,320,320}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_124) {
  Conv3DV2TestCase({2,128,17,321,321}, {128,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_125) {
  Conv3DV2TestCase({2,128,19,160,160}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_126) {
  Conv3DV2TestCase({2,256,19,160,160}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_127) {
  Conv3DV2TestCase({2,128,17,160,160}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_128) {
  Conv3DV2TestCase({2,256,17,161,161}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_129) {
  Conv3DV2TestCase({2,256,11,80,80}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_130) {
  Conv3DV2TestCase({2,512,11,80,80}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_131) {
  Conv3DV2TestCase({2,256,9,80,80}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_132) {
  Conv3DV2TestCase({2,512,9,81,81}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_133) {
  Conv3DV2TestCase({2,512,7,40,40}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_134) {
  Conv3DV2TestCase({2,512,5,40,40}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_135) {
  Conv3DV2TestCase({2,512,7,40,40}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_136) {
  Conv3DV2TestCase({2,8,5,40,40}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_137) {
  Conv3DV2TestCase({2,4,5,40,40}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_138) {
  Conv3DV2TestCase({1,3,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_139) {
  Conv3DV2TestCase({1,128,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_140) {
  Conv3DV2TestCase({1,128,17,257,257}, {128,1,3,3}, {0,0,0,0,1,1}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_141) {
  Conv3DV2TestCase({1,128,11,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_142) {
  Conv3DV2TestCase({1,256,11,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_143) {
  Conv3DV2TestCase({1,128,9,128,128}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_144) {
  Conv3DV2TestCase({1,256,9,129,129}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_145) {
  Conv3DV2TestCase({1,256,7,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_146) {
  Conv3DV2TestCase({1,512,7,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_147) {
  Conv3DV2TestCase({1,256,5,64,64}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_148) {
  Conv3DV2TestCase({1,512,5,65,65}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_149) {
  Conv3DV2TestCase({1,512,7,32,32}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_150) {
  Conv3DV2TestCase({1,512,5,32,32}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_151) {
  Conv3DV2TestCase({1,8,5,32,32}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_152) {
  Conv3DV2TestCase({1,4,5,32,32}, {4,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_153) {
  Conv3DV2TestCase({1,4,7,32,32}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_154) {
  Conv3DV2TestCase({1,512,5,64,64}, {512,1,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_155) {
  Conv3DV2TestCase({1,512,11,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_156) {
  Conv3DV2TestCase({1,512,9,128,128}, {512,1,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_157) {
  Conv3DV2TestCase({1,512,19,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_158) {
  Conv3DV2TestCase({1,256,19,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_159) {
  Conv3DV2TestCase({1,512,17,128,128}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_160) {
  Conv3DV2TestCase({1,256,17,256,256}, {256,1,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_161) {
  Conv3DV2TestCase({1,256,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_162) {
  Conv3DV2TestCase({1,256,17,256,256}, {128,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_163) {
  Conv3DV2TestCase({1,128,19,256,256}, {3,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_164) {
  Conv3DV2TestCase({1,512,7,32,32}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_165) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_166) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_167) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_168) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_169) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_170) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_171) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_172) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_173) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_174) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_175) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_176) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_177) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_178) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_179) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_180) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_181) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_182) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_183) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_184) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_185) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_186) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_187) {
  Conv3DV2TestCase({1,8,16,16,16}, {1152,16,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_188) {
  Conv3DV2TestCase({1,12,16,16,16}, {1152,16,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_189) {
  Conv3DV2TestCase({16,1,1,1,2}, {17,5,2,4}, {3,3,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_191) {
  Conv3DV2TestCase({2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_192) {
  Conv3DV2TestCase({2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_193) {
  Conv3DV2TestCase({2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_194) {
  Conv3DV2TestCase({2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_195) {
  Conv3DV2TestCase({2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_196) {
  Conv3DV2TestCase({2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_197) {
  Conv3DV2TestCase({2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_198) {
  Conv3DV2TestCase({2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_199) {
  Conv3DV2TestCase({1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_200) {
  Conv3DV2TestCase({1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_201) {
  Conv3DV2TestCase({1,4,16,32,64}, {17,3,5,2}, {1,1,2,2,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_202) {
  Conv3DV2TestCase({1,31,24,48,56}, {8,6,9,5}, {2,2,4,4,3,3}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_203) {
  Conv3DV2TestCase({1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_204) {
  Conv3DV2TestCase({1,4,16,32,64}, {17,3,5,2}, {1,1,2,2,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_205) {
  Conv3DV2TestCase({1,31,24,48,56}, {8,6,9,5}, {2,2,4,4,3,3}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_206) {
  Conv3DV2TestCase({1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_207) {
  Conv3DV2TestCase({1,4,16,32,64}, {17,3,5,2}, {1,1,2,2,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_208) {
  Conv3DV2TestCase({1,31,24,48,56}, {8,6,9,5}, {2,2,4,4,3,3}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_209) {
  Conv3DV2TestCase({1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_210) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_211) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_213) {
  Conv3DV2TestCase({1,16,15,16,16}, {4,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_214) {
  Conv3DV2TestCase({1,16,15,16,16},{2,1,1,1},{0,0,0,0,0,0},{1,1,1},{1,1,1}, ge::DT_BF16, 0);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_215) {
  Conv3DV2TestCase({2,15,16,16,16},{15,2,2,2},{1,1,0,0,0,0},{2,1,1},{1,1,2}, ge::DT_BF16, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_216) {
  Conv3DV2TestCase({1,16,17,16,16},{16,3,3,3},{1,1,1,1,0,0},{1,2,1},{2,1,1}, ge::DT_BF16, 0);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_217) {
  Conv3DV2TestCase({1,16,16,17,16},{20,4,3,3},{1,1,1,1,2,2},{1,1,2},{1,1,2}, ge::DT_BF16, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_218) {
  Conv3DV2TestCase({1,16,16,15,17},{4,3,4,3},{2,2,1,1,1,1},{2,1,1},{2,1,1}, ge::DT_BF16, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_219) {
  Conv3DV2TestCase({1,18,16,16,15},{9,3,3,4},{1,1,2,2,1,1},{1,2,1},{2,1,1}, ge::DT_BF16, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_220) {
  Conv3DV2TestCase({1,16,16,16,4096},{1,4,2,40},{1,1,0,0,1,1},{10,3,2},{2,2,2}, ge::DT_BF16, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_221) {
  Conv3DV2TestCase({1,16,64,16,2560},{1,64,3,3},{1,1,1,1,1,1},{1,63,1},{1,3,1}, ge::DT_BF16, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_7_fp16) {
  Conv3DV2TestCase({1,1,1,256,1}, {1,1,255,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_8_fp16) {
  Conv3DV2TestCase({1,1,1,1,256}, {1,1,1,255}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_12_fp16) {
  Conv3DV2TestCase({1,3,1,300,4}, {3,1,5,3}, {1,1,1,1,1,1}, {1,63,1}, {1,3,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_13_fp16) {
  Conv3DV2TestCase({1,2,1,300,7}, {3,1,5,3}, {1,1,1,1,1,1}, {1,1,63}, {1,3,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_14_fp16) {
  Conv3DV2TestCase({1,3,1,300,6}, {3,1,1,3}, {0,0,0,0,1,1}, {1,2,1}, {1,255,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_15_fp16) {
  Conv3DV2TestCase({1,3,2,18,770}, {3,1,1,2}, {0,0,0,0,1,1}, {1,2,3}, {1,1,255}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_16_fp16) {
  Conv3DV2TestCase({1,1,1,256,1}, {1,1,255,1}, {0,0,254,254,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_17_fp16) {
  Conv3DV2TestCase({1,1,1,1,256}, {1,1,1,255}, {0,0,0,0,254,254}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_18_fp16) {
  Conv3DV2TestCase({1,4,6,75,293}, {13,3,15,6}, {0,0,0,0,0,0}, {1,18,27}, {1,1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_20_fp16) {
  Conv3DV2TestCase({1,3,1,16,23}, {16,3,3,3}, {3,3,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_24_fp16) {
  Conv3DV2TestCase({1,3,9,200,123}, {18,1,4,7}, {0,0,1,1,2,2}, {1,32,18}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_25_fp16) {
  Conv3DV2TestCase({1,4,1,1,873}, {11,1,1,6}, {0,0,0,0,1,2}, {1,1,18}, {1,1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_27_fp16) {
  Conv3DV2TestCase({1,2,11,242,10}, {4,1,1,1}, {0,0,0,0,0,0}, {1,62,7}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_28_fp16) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_29_fp16) {
  Conv3DV2TestCase({1,4,120,32,32}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_30_fp16) {
  Conv3DV2TestCase({3,4,16,26,36}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_31_fp16) {
  Conv3DV2TestCase({8,240,4,32,32}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_32_fp16) {
  Conv3DV2TestCase({8,3,18,130,130}, {240,4,4,4}, {0,0,0,0,0,0}, {2,2,2}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_33_fp16) {
  Conv3DV2TestCase({8,240,10,66,66}, {240,4,4,4}, {0,0,0,0,0,0}, {2,2,2}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_34_fp16) {
  Conv3DV2TestCase({8,240,6,34,34}, {240,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_35_fp16) {
  Conv3DV2TestCase({8,240,6,34,34}, {120,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_36_fp16) {
  Conv3DV2TestCase({8,120,4,32,32}, {240,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_37_fp16) {
  Conv3DV2TestCase({4,3,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_38_fp16) {
  Conv3DV2TestCase({4,128,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_39_fp16) {
  Conv3DV2TestCase({4,128,17,257,257}, {128,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_40_fp16) {
  Conv3DV2TestCase({4,128,11,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_41_fp16) {
  Conv3DV2TestCase({4,256,11,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_42_fp16) {
  Conv3DV2TestCase({4,128,9,128,128}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_43_fp16) {
  Conv3DV2TestCase({4,256,9,129,129}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_44_fp16) {
  Conv3DV2TestCase({4,256,7,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_45_fp16) {
  Conv3DV2TestCase({4,512,7,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_46_fp16) {
  Conv3DV2TestCase({4,256,5,64,64}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_47_fp16) {
  Conv3DV2TestCase({4,512,5,65,65}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_48_fp16) {
  Conv3DV2TestCase({4,512,7,32,32}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_49_fp16) {
  Conv3DV2TestCase({4,512,5,32,32}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_50_fp16) {
  Conv3DV2TestCase({4,512,7,32,32}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_51_fp16) {
  Conv3DV2TestCase({4,8,5,32,32}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_52_fp16) {
  Conv3DV2TestCase({16,3,3,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_53_fp16) {
  Conv3DV2TestCase({16,128,3,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_54_fp16) {
  Conv3DV2TestCase({16,128,1,257,257}, {128,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_55_fp16) {
  Conv3DV2TestCase({16,128,3,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_56_fp16) {
  Conv3DV2TestCase({16,256,3,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_57_fp16) {
  Conv3DV2TestCase({16,128,1,128,128}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_58_fp16) {
  Conv3DV2TestCase({16,256,1,129,129}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_59_fp16) {
  Conv3DV2TestCase({16,256,3,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_60_fp16) {
  Conv3DV2TestCase({16,512,3,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_61_fp16) {
  Conv3DV2TestCase({16,256,1,64,64}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_62_fp16) {
  Conv3DV2TestCase({16,512,1,65,65}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_63_fp16) {
  Conv3DV2TestCase({16,512,3,32,32}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_64_fp16) {
  Conv3DV2TestCase({16,512,1,32,32}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_65_fp16) {
  Conv3DV2TestCase({16,512,3,32,32}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_66_fp16) {
  Conv3DV2TestCase({16,8,1,32,32}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_67_fp16) {
  Conv3DV2TestCase({16,3,26,134,134}, {64,7,7,7}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_68_fp16) {
  Conv3DV2TestCase({16,64,22,130,130}, {64,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_69_fp16) {
  Conv3DV2TestCase({16,64,20,128,128}, {64,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_70_fp16) {
  Conv3DV2TestCase({16,128,22,66,66}, {128,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_71_fp16) {
  Conv3DV2TestCase({16,128,20,64,64}, {128,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_72_fp16) {
  Conv3DV2TestCase({16,256,22,34,34}, {256,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_73_fp16) {
  Conv3DV2TestCase({16,256,20,32,32}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_74_fp16) {
  Conv3DV2TestCase({16,256,20,32,32}, {1364,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_75_fp16) {
  Conv3DV2TestCase({16,682,20,32,32}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_76_fp16) {
  Conv3DV2TestCase({16,512,22,18,18}, {512,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_77_fp16) {
  Conv3DV2TestCase({16,512,20,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_78_fp16) {
  Conv3DV2TestCase({16,512,20,16,16}, {2730,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_79_fp16) {
  Conv3DV2TestCase({16,1365,20,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_80_fp16) {
  Conv3DV2TestCase({16,512,12,18,18}, {512,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_81_fp16) {
  Conv3DV2TestCase({16,512,10,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_82_fp16) {
  Conv3DV2TestCase({16,512,7,18,18}, {512,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_83_fp16) {
  Conv3DV2TestCase({16,512,5,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_84_fp16) {
  Conv3DV2TestCase({16,512,5,16,16}, {2730,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_85_fp16) {
  Conv3DV2TestCase({16,1365,5,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_86_fp16) {
  Conv3DV2TestCase({16,64,22,130,130}, {3,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_87_fp16) {
  Conv3DV2TestCase({1,320,25,40,64}, {320,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_88_fp16) {
  Conv3DV2TestCase({1,640,25,20,32}, {640,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_89_fp16) {
  Conv3DV2TestCase({1,1280,25,10,16}, {1280,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_90_fp16) {
  Conv3DV2TestCase({1,1280,25,5,8}, {1280,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_91_fp16) {
  Conv3DV2TestCase({2,320,25,40,64}, {320,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_92_fp16) {
  Conv3DV2TestCase({2,640,25,20,32}, {640,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_93_fp16) {
  Conv3DV2TestCase({2,1280,25,10,16}, {1280,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_94_fp16) {
  Conv3DV2TestCase({2,1280,25,5,8}, {1280,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_95_fp16) {
  Conv3DV2TestCase({1,512,8,40,64}, {512,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_96_fp16) {
  Conv3DV2TestCase({1,512,8,80,128}, {512,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_97_fp16) {
  Conv3DV2TestCase({1,256,8,160,256}, {256,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_98_fp16) {
  Conv3DV2TestCase({1,128,8,320,512}, {128,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_99_fp16) {
  Conv3DV2TestCase({1,3,8,320,512}, {3,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_100_fp16) {
  Conv3DV2TestCase({1,512,1,40,64}, {512,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_101_fp16) {
  Conv3DV2TestCase({1,512,1,80,128}, {512,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_102_fp16) {
  Conv3DV2TestCase({1,256,1,160,256}, {256,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_103_fp16) {
  Conv3DV2TestCase({1,128,1,320,512}, {128,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_104_fp16) {
  Conv3DV2TestCase({1,3,1,320,512}, {3,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_105_fp16) {
  Conv3DV2TestCase({1,3,16,224,224}, {768,2,16,16}, {0,0,0,0,0,0}, {2,16,16}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_106_fp16) {
  Conv3DV2TestCase({2,3,35,192,192}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_107_fp16) {
  Conv3DV2TestCase({2,128,35,192,192}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_108_fp16) {
  Conv3DV2TestCase({2,128,33,193,193}, {128,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_109_fp16) {
  Conv3DV2TestCase({2,128,35,96,96}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_110_fp16) {
  Conv3DV2TestCase({2,256,35,96,96}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_111_fp16) {
  Conv3DV2TestCase({2,128,33,96,96}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_112_fp16) {
  Conv3DV2TestCase({2,256,33,97,97}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_113_fp16) {
  Conv3DV2TestCase({2,256,19,48,48}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_114_fp16) {
  Conv3DV2TestCase({2,512,19,48,48}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_115_fp16) {
  Conv3DV2TestCase({2,256,17,48,48}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_116_fp16) {
  Conv3DV2TestCase({2,512,17,49,49}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_117_fp16) {
  Conv3DV2TestCase({2,512,11,24,24}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_118_fp16) {
  Conv3DV2TestCase({2,512,9,24,24}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_119_fp16) {
  Conv3DV2TestCase({2,512,11,24,24}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_120_fp16) {
  Conv3DV2TestCase({2,8,9,24,24}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_121_fp16) {
  Conv3DV2TestCase({2,4,9,24,24}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_122_fp16) {
  Conv3DV2TestCase({2,3,19,320,320}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_123_fp16) {
  Conv3DV2TestCase({2,128,19,320,320}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_124_fp16) {
  Conv3DV2TestCase({2,128,17,321,321}, {128,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_125_fp16) {
  Conv3DV2TestCase({2,128,19,160,160}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_126_fp16) {
  Conv3DV2TestCase({2,256,19,160,160}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_127_fp16) {
  Conv3DV2TestCase({2,128,17,160,160}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_128_fp16) {
  Conv3DV2TestCase({2,256,17,161,161}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_129_fp16) {
  Conv3DV2TestCase({2,256,11,80,80}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_130_fp16) {
  Conv3DV2TestCase({2,512,11,80,80}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_131_fp16) {
  Conv3DV2TestCase({2,256,9,80,80}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_132_fp16) {
  Conv3DV2TestCase({2,512,9,81,81}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_133_fp16) {
  Conv3DV2TestCase({2,512,7,40,40}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_134_fp16) {
  Conv3DV2TestCase({2,512,5,40,40}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_135_fp16) {
  Conv3DV2TestCase({2,512,7,40,40}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_136_fp16) {
  Conv3DV2TestCase({2,8,5,40,40}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_137_fp16) {
  Conv3DV2TestCase({2,4,5,40,40}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_138_fp16) {
  Conv3DV2TestCase({1,3,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_139_fp16) {
  Conv3DV2TestCase({1,128,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_140_fp16) {
  Conv3DV2TestCase({1,128,17,257,257}, {128,1,3,3}, {0,0,0,0,1,1}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_141_fp16) {
  Conv3DV2TestCase({1,128,11,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_142_fp16) {
  Conv3DV2TestCase({1,256,11,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_143_fp16) {
  Conv3DV2TestCase({1,128,9,128,128}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_144_fp16) {
  Conv3DV2TestCase({1,256,9,129,129}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_145_fp16) {
  Conv3DV2TestCase({1,256,7,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_146_fp16) {
  Conv3DV2TestCase({1,512,7,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_147_fp16) {
  Conv3DV2TestCase({1,256,5,64,64}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_148_fp16) {
  Conv3DV2TestCase({1,512,5,65,65}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_149_fp16) {
  Conv3DV2TestCase({1,512,7,32,32}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_150_fp16) {
  Conv3DV2TestCase({1,512,5,32,32}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_151_fp16) {
  Conv3DV2TestCase({1,8,5,32,32}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_152_fp16) {
  Conv3DV2TestCase({1,4,5,32,32}, {4,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_153_fp16) {
  Conv3DV2TestCase({1,4,7,32,32}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_154_fp16) {
  Conv3DV2TestCase({1,512,5,64,64}, {512,1,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_155_fp16) {
  Conv3DV2TestCase({1,512,11,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_156_fp16) {
  Conv3DV2TestCase({1,512,9,128,128}, {512,1,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_157_fp16) {
  Conv3DV2TestCase({1,512,19,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_158_fp16) {
  Conv3DV2TestCase({1,256,19,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_159_fp16) {
  Conv3DV2TestCase({1,512,17,128,128}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_160_fp16) {
  Conv3DV2TestCase({1,256,17,256,256}, {256,1,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_161_fp16) {
  Conv3DV2TestCase({1,256,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_162_fp16) {
  Conv3DV2TestCase({1,256,17,256,256}, {128,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_163_fp16) {
  Conv3DV2TestCase({1,128,19,256,256}, {3,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_164_fp16) {
  Conv3DV2TestCase({1,512,7,32,32}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_165_fp16) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_166_fp16) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_167_fp16) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_168_fp16) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_169_fp16) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_170_fp16) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_171_fp16) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_172_fp16) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_173_fp16) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_174_fp16) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_175_fp16) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_176_fp16) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_177_fp16) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_178_fp16) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_179_fp16) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_180_fp16) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_181_fp16) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_182_fp16) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_183_fp16) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_184_fp16) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_185_fp16) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_186_fp16) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_187_fp16) {
  Conv3DV2TestCase({1,8,16,16,16}, {1152,16,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_188_fp16) {
  Conv3DV2TestCase({1,12,16,16,16}, {1152,16,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_189_fp16) {
  Conv3DV2TestCase({16,1,1,1,2}, {17,5,2,4}, {3,3,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_191_fp16) {
  Conv3DV2TestCase({2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_192_fp16) {
  Conv3DV2TestCase({2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_193_fp16) {
  Conv3DV2TestCase({2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_194_fp16) {
  Conv3DV2TestCase({2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_195_fp16) {
  Conv3DV2TestCase({2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_196_fp16) {
  Conv3DV2TestCase({2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_197_fp16) {
  Conv3DV2TestCase({2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_198_fp16) {
  Conv3DV2TestCase({2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_199_fp16) {
  Conv3DV2TestCase({1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_200_fp16) {
  Conv3DV2TestCase({1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_201_fp16) {
  Conv3DV2TestCase({1,4,16,32,64}, {17,3,5,2}, {1,1,2,2,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_202_fp16) {
  Conv3DV2TestCase({1,31,24,48,56}, {8,6,9,5}, {2,2,4,4,3,3}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_203_fp16) {
  Conv3DV2TestCase({1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_204_fp16) {
  Conv3DV2TestCase({1,4,16,32,64}, {17,3,5,2}, {1,1,2,2,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_205_fp16) {
  Conv3DV2TestCase({1,31,24,48,56}, {8,6,9,5}, {2,2,4,4,3,3}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_206_fp16) {
  Conv3DV2TestCase({1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_207_fp16) {
  Conv3DV2TestCase({1,4,16,32,64}, {17,3,5,2}, {1,1,2,2,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_208_fp16) {
  Conv3DV2TestCase({1,31,24,48,56}, {8,6,9,5}, {2,2,4,4,3,3}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_209_fp16) {
  Conv3DV2TestCase({1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_210_fp16) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_211_fp16) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_213_fp16) {
  Conv3DV2TestCase({1,16,15,16,16}, {4,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_214_fp16) {
  Conv3DV2TestCase({1,16,15,16,16},{2,1,1,1},{0,0,0,0,0,0},{1,1,1},{1,1,1}, ge::DT_FLOAT16, 0);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_215_fp16) {
  Conv3DV2TestCase({2,15,16,16,16},{15,2,2,2},{1,1,0,0,0,0},{2,1,1},{1,1,2}, ge::DT_FLOAT16, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_216_fp16) {
  Conv3DV2TestCase({1,16,17,16,16},{16,3,3,3},{1,1,1,1,0,0},{1,2,1},{2,1,1}, ge::DT_FLOAT16, 0);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_217_fp16) {
  Conv3DV2TestCase({1,16,16,17,16},{20,4,3,3},{1,1,1,1,2,2},{1,1,2},{1,1,2}, ge::DT_FLOAT16, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_218_fp16) {
  Conv3DV2TestCase({1,16,16,15,17},{4,3,4,3},{2,2,1,1,1,1},{2,1,1},{2,1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_219_fp16) {
  Conv3DV2TestCase({1,18,16,16,15},{9,3,3,4},{1,1,2,2,1,1},{1,2,1},{2,1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_220_fp16) {
  Conv3DV2TestCase({1,16,16,16,4096},{1,4,2,40},{1,1,0,0,1,1},{10,3,2},{2,2,2}, ge::DT_FLOAT16, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_221_fp16) {
  Conv3DV2TestCase({1,16,64,16,2560},{1,64,3,3},{1,1,1,1,1,1},{1,63,1},{1,3,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_7_fp32) {
  Conv3DV2TestCase({1,1,1,256,1}, {1,1,255,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_8_fp32) {
  Conv3DV2TestCase({1,1,1,1,256}, {1,1,1,255}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_12_fp32) {
  Conv3DV2TestCase({1,3,1,300,4}, {3,1,5,3}, {1,1,1,1,1,1}, {1,63,1}, {1,3,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_13_fp32) {
  Conv3DV2TestCase({1,2,1,300,7}, {3,1,5,3}, {1,1,1,1,1,1}, {1,1,63}, {1,3,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_14_fp32) {
  Conv3DV2TestCase({1,3,1,300,6}, {3,1,1,3}, {0,0,0,0,1,1}, {1,2,1}, {1,255,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_15_fp32) {
  Conv3DV2TestCase({1,3,2,18,770}, {3,1,1,2}, {0,0,0,0,1,1}, {1,2,3}, {1,1,255}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_16_fp32) {
  Conv3DV2TestCase({1,1,1,256,1}, {1,1,255,1}, {0,0,254,254,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_17_fp32) {
  Conv3DV2TestCase({1,1,1,1,256}, {1,1,1,255}, {0,0,0,0,254,254}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_18_fp32) {
  Conv3DV2TestCase({1,4,6,75,293}, {13,3,15,6}, {0,0,0,0,0,0}, {1,18,27}, {1,1,1}, ge::DT_FLOAT, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_20_fp32) {
  Conv3DV2TestCase({1,3,1,16,23}, {16,3,3,3}, {3,3,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_24_fp32) {
  Conv3DV2TestCase({1,3,9,200,123}, {18,1,4,7}, {0,0,1,1,2,2}, {1,32,18}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_25_fp32) {
  Conv3DV2TestCase({1,4,1,1,873}, {11,1,1,6}, {0,0,0,0,1,2}, {1,1,18}, {1,1,1}, ge::DT_FLOAT, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_27_fp32) {
  Conv3DV2TestCase({1,2,11,242,10}, {4,1,1,1}, {0,0,0,0,0,0}, {1,62,7}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_28_fp32) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_29_fp32) {
  Conv3DV2TestCase({1,4,120,32,32}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_30_fp32) {
  Conv3DV2TestCase({3,4,16,26,36}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_31_fp32) {
  Conv3DV2TestCase({8,240,4,32,32}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_32_fp32) {
  Conv3DV2TestCase({8,3,18,130,130}, {240,4,4,4}, {0,0,0,0,0,0}, {2,2,2}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_33_fp32) {
  Conv3DV2TestCase({8,240,10,66,66}, {240,4,4,4}, {0,0,0,0,0,0}, {2,2,2}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_34_fp32) {
  Conv3DV2TestCase({8,240,6,34,34}, {240,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_35_fp32) {
  Conv3DV2TestCase({8,240,6,34,34}, {120,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_36_fp32) {
  Conv3DV2TestCase({8,120,4,32,32}, {240,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_37_fp32) {
  Conv3DV2TestCase({4,3,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_38_fp32) {
  Conv3DV2TestCase({4,128,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_39_fp32) {
  Conv3DV2TestCase({4,128,17,257,257}, {128,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_40_fp32) {
  Conv3DV2TestCase({4,128,11,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_41_fp32) {
  Conv3DV2TestCase({4,256,11,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_42_fp32) {
  Conv3DV2TestCase({4,128,9,128,128}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_43_fp32) {
  Conv3DV2TestCase({4,256,9,129,129}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_44_fp32) {
  Conv3DV2TestCase({4,256,7,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_45_fp32) {
  Conv3DV2TestCase({4,512,7,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_46_fp32) {
  Conv3DV2TestCase({4,256,5,64,64}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_47_fp32) {
  Conv3DV2TestCase({4,512,5,65,65}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_48_fp32) {
  Conv3DV2TestCase({4,512,7,32,32}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_49_fp32) {
  Conv3DV2TestCase({4,512,5,32,32}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_50_fp32) {
  Conv3DV2TestCase({4,512,7,32,32}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_51_fp32) {
  Conv3DV2TestCase({4,8,5,32,32}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_52_fp32) {
  Conv3DV2TestCase({16,3,3,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_53_fp32) {
  Conv3DV2TestCase({16,128,3,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_54_fp32) {
  Conv3DV2TestCase({16,128,1,257,257}, {128,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_55_fp32) {
  Conv3DV2TestCase({16,128,3,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_56_fp32) {
  Conv3DV2TestCase({16,256,3,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_57_fp32) {
  Conv3DV2TestCase({16,128,1,128,128}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_58_fp32) {
  Conv3DV2TestCase({16,256,1,129,129}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_59_fp32) {
  Conv3DV2TestCase({16,256,3,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_60_fp32) {
  Conv3DV2TestCase({16,512,3,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_61_fp32) {
  Conv3DV2TestCase({16,256,1,64,64}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_62_fp32) {
  Conv3DV2TestCase({16,512,1,65,65}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_63_fp32) {
  Conv3DV2TestCase({16,512,3,32,32}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_64_fp32) {
  Conv3DV2TestCase({16,512,1,32,32}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_65_fp32) {
  Conv3DV2TestCase({16,512,3,32,32}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_66_fp32) {
  Conv3DV2TestCase({16,8,1,32,32}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_67_fp32) {
  Conv3DV2TestCase({16,3,26,134,134}, {64,7,7,7}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_68_fp32) {
  Conv3DV2TestCase({16,64,22,130,130}, {64,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_69_fp32) {
  Conv3DV2TestCase({16,64,20,128,128}, {64,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_70_fp32) {
  Conv3DV2TestCase({16,128,22,66,66}, {128,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_71_fp32) {
  Conv3DV2TestCase({16,128,20,64,64}, {128,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_72_fp32) {
  Conv3DV2TestCase({16,256,22,34,34}, {256,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_73_fp32) {
  Conv3DV2TestCase({16,256,20,32,32}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_74_fp32) {
  Conv3DV2TestCase({16,256,20,32,32}, {1364,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_75_fp32) {
  Conv3DV2TestCase({16,682,20,32,32}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_76_fp32) {
  Conv3DV2TestCase({16,512,22,18,18}, {512,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_77_fp32) {
  Conv3DV2TestCase({16,512,20,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_78_fp32) {
  Conv3DV2TestCase({16,512,20,16,16}, {2730,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_79_fp32) {
  Conv3DV2TestCase({16,1365,20,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_80_fp32) {
  Conv3DV2TestCase({16,512,12,18,18}, {512,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_81_fp32) {
  Conv3DV2TestCase({16,512,10,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_82_fp32) {
  Conv3DV2TestCase({16,512,7,18,18}, {512,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_83_fp32) {
  Conv3DV2TestCase({16,512,5,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_84_fp32) {
  Conv3DV2TestCase({16,512,5,16,16}, {2730,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_85_fp32) {
  Conv3DV2TestCase({16,1365,5,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_86_fp32) {
  Conv3DV2TestCase({16,64,22,130,130}, {3,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_87_fp32) {
  Conv3DV2TestCase({1,320,25,40,64}, {320,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_88_fp32) {
  Conv3DV2TestCase({1,640,25,20,32}, {640,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_89_fp32) {
  Conv3DV2TestCase({1,1280,25,10,16}, {1280,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_90_fp32) {
  Conv3DV2TestCase({1,1280,25,5,8}, {1280,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_91_fp32) {
  Conv3DV2TestCase({2,320,25,40,64}, {320,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_92_fp32) {
  Conv3DV2TestCase({2,640,25,20,32}, {640,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_93_fp32) {
  Conv3DV2TestCase({2,1280,25,10,16}, {1280,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_94_fp32) {
  Conv3DV2TestCase({2,1280,25,5,8}, {1280,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_95_fp32) {
  Conv3DV2TestCase({1,512,8,40,64}, {512,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_96_fp32) {
  Conv3DV2TestCase({1,512,8,80,128}, {512,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_97_fp32) {
  Conv3DV2TestCase({1,256,8,160,256}, {256,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_98_fp32) {
  Conv3DV2TestCase({1,128,8,320,512}, {128,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_99_fp32) {
  Conv3DV2TestCase({1,3,8,320,512}, {3,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_100_fp32) {
  Conv3DV2TestCase({1,512,1,40,64}, {512,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_101_fp32) {
  Conv3DV2TestCase({1,512,1,80,128}, {512,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_102_fp32) {
  Conv3DV2TestCase({1,256,1,160,256}, {256,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_103_fp32) {
  Conv3DV2TestCase({1,128,1,320,512}, {128,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_104_fp32) {
  Conv3DV2TestCase({1,3,1,320,512}, {3,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_105_fp32) {
  Conv3DV2TestCase({1,3,16,224,224}, {768,2,16,16}, {0,0,0,0,0,0}, {2,16,16}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_106_fp32) {
  Conv3DV2TestCase({2,3,35,192,192}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_107_fp32) {
  Conv3DV2TestCase({2,128,35,192,192}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_108_fp32) {
  Conv3DV2TestCase({2,128,33,193,193}, {128,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_109_fp32) {
  Conv3DV2TestCase({2,128,35,96,96}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_110_fp32) {
  Conv3DV2TestCase({2,256,35,96,96}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_111_fp32) {
  Conv3DV2TestCase({2,128,33,96,96}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_112_fp32) {
  Conv3DV2TestCase({2,256,33,97,97}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_113_fp32) {
  Conv3DV2TestCase({2,256,19,48,48}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_114_fp32) {
  Conv3DV2TestCase({2,512,19,48,48}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_115_fp32) {
  Conv3DV2TestCase({2,256,17,48,48}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_116_fp32) {
  Conv3DV2TestCase({2,512,17,49,49}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_117_fp32) {
  Conv3DV2TestCase({2,512,11,24,24}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_118_fp32) {
  Conv3DV2TestCase({2,512,9,24,24}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_119_fp32) {
  Conv3DV2TestCase({2,512,11,24,24}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_120_fp32) {
  Conv3DV2TestCase({2,8,9,24,24}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_121_fp32) {
  Conv3DV2TestCase({2,4,9,24,24}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_122_fp32) {
  Conv3DV2TestCase({2,3,19,320,320}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_123_fp32) {
  Conv3DV2TestCase({2,128,19,320,320}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_124_fp32) {
  Conv3DV2TestCase({2,128,17,321,321}, {128,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_125_fp32) {
  Conv3DV2TestCase({2,128,19,160,160}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_126_fp32) {
  Conv3DV2TestCase({2,256,19,160,160}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_127_fp32) {
  Conv3DV2TestCase({2,128,17,160,160}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_128_fp32) {
  Conv3DV2TestCase({2,256,17,161,161}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_129_fp32) {
  Conv3DV2TestCase({2,256,11,80,80}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_130_fp32) {
  Conv3DV2TestCase({2,512,11,80,80}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_131_fp32) {
  Conv3DV2TestCase({2,256,9,80,80}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_132_fp32) {
  Conv3DV2TestCase({2,512,9,81,81}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_133_fp32) {
  Conv3DV2TestCase({2,512,7,40,40}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_134_fp32) {
  Conv3DV2TestCase({2,512,5,40,40}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_135_fp32) {
  Conv3DV2TestCase({2,512,7,40,40}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_136_fp32) {
  Conv3DV2TestCase({2,8,5,40,40}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_137_fp32) {
  Conv3DV2TestCase({2,4,5,40,40}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_138_fp32) {
  Conv3DV2TestCase({1,3,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_139_fp32) {
  Conv3DV2TestCase({1,128,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_140_fp32) {
  Conv3DV2TestCase({1,128,17,257,257}, {128,1,3,3}, {0,0,0,0,1,1}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_141_fp32) {
  Conv3DV2TestCase({1,128,11,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_142_fp32) {
  Conv3DV2TestCase({1,256,11,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_143_fp32) {
  Conv3DV2TestCase({1,128,9,128,128}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_144_fp32) {
  Conv3DV2TestCase({1,256,9,129,129}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_145_fp32) {
  Conv3DV2TestCase({1,256,7,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_146_fp32) {
  Conv3DV2TestCase({1,512,7,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_147_fp32) {
  Conv3DV2TestCase({1,256,5,64,64}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_148_fp32) {
  Conv3DV2TestCase({1,512,5,65,65}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_149_fp32) {
  Conv3DV2TestCase({1,512,7,32,32}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_150_fp32) {
  Conv3DV2TestCase({1,512,5,32,32}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_151_fp32) {
  Conv3DV2TestCase({1,8,5,32,32}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_152_fp32) {
  Conv3DV2TestCase({1,4,5,32,32}, {4,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_153_fp32) {
  Conv3DV2TestCase({1,4,7,32,32}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_154_fp32) {
  Conv3DV2TestCase({1,512,5,64,64}, {512,1,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_155_fp32) {
  Conv3DV2TestCase({1,512,11,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_156_fp32) {
  Conv3DV2TestCase({1,512,9,128,128}, {512,1,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_157_fp32) {
  Conv3DV2TestCase({1,512,19,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_158_fp32) {
  Conv3DV2TestCase({1,256,19,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_159_fp32) {
  Conv3DV2TestCase({1,512,17,128,128}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_160_fp32) {
  Conv3DV2TestCase({1,256,17,256,256}, {256,1,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_161_fp32) {
  Conv3DV2TestCase({1,256,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_162_fp32) {
  Conv3DV2TestCase({1,256,17,256,256}, {128,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_163_fp32) {
  Conv3DV2TestCase({1,128,19,256,256}, {3,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_164_fp32) {
  Conv3DV2TestCase({1,512,7,32,32}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_165_fp32) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_166_fp32) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_167_fp32) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_168_fp32) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_169_fp32) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_170_fp32) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_171_fp32) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_172_fp32) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_173_fp32) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_174_fp32) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_175_fp32) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_176_fp32) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_177_fp32) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_178_fp32) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_179_fp32) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_180_fp32) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_181_fp32) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_182_fp32) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_183_fp32) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_184_fp32) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_185_fp32) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_186_fp32) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_187_fp32) {
  Conv3DV2TestCase({1,8,16,16,16}, {1152,16,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_188_fp32) {
  Conv3DV2TestCase({1,12,16,16,16}, {1152,16,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_189_fp32) {
  Conv3DV2TestCase({16,1,1,1,2}, {17,5,2,4}, {3,3,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_191_fp32) {
  Conv3DV2TestCase({2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_192_fp32) {
  Conv3DV2TestCase({2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_193_fp32) {
  Conv3DV2TestCase({2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_194_fp32) {
  Conv3DV2TestCase({2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_195_fp32) {
  Conv3DV2TestCase({2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_196_fp32) {
  Conv3DV2TestCase({2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_197_fp32) {
  Conv3DV2TestCase({2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_198_fp32) {
  Conv3DV2TestCase({2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_199_fp32) {
  Conv3DV2TestCase({1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_200_fp32) {
  Conv3DV2TestCase({1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_201_fp32) {
  Conv3DV2TestCase({1,4,16,32,64}, {17,3,5,2}, {1,1,2,2,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_202_fp32) {
  Conv3DV2TestCase({1,31,24,48,56}, {8,6,9,5}, {2,2,4,4,3,3}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_203_fp32) {
  Conv3DV2TestCase({1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_204_fp32) {
  Conv3DV2TestCase({1,4,16,32,64}, {17,3,5,2}, {1,1,2,2,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_205_fp32) {
  Conv3DV2TestCase({1,31,24,48,56}, {8,6,9,5}, {2,2,4,4,3,3}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_206_fp32) {
  Conv3DV2TestCase({1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_207_fp32) {
  Conv3DV2TestCase({1,4,16,32,64}, {17,3,5,2}, {1,1,2,2,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_208_fp32) {
  Conv3DV2TestCase({1,31,24,48,56}, {8,6,9,5}, {2,2,4,4,3,3}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_209_fp32) {
  Conv3DV2TestCase({1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_210_fp32) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}
TEST_F(Conv3dv2Tiling, run_conv3dv2_case_211_fp32) {
  Conv3DV2TestCase({1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_213_fp32) {
  Conv3DV2TestCase({1,16,15,16,16}, {4,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_214_fp32) {
  Conv3DV2TestCase({1,16,15,16,16},{2,1,1,1},{0,0,0,0,0,0},{1,1,1},{1,1,1}, ge::DT_FLOAT, 0);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_215_fp32) {
  Conv3DV2TestCase({2,15,16,16,16},{15,2,2,2},{1,1,0,0,0,0},{2,1,1},{1,1,2}, ge::DT_FLOAT, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_216_fp32) {
  Conv3DV2TestCase({1,16,17,16,16},{16,3,3,3},{1,1,1,1,0,0},{1,2,1},{2,1,1}, ge::DT_FLOAT, 0);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_217_fp32) {
  Conv3DV2TestCase({1,16,16,17,16},{20,4,3,3},{1,1,1,1,2,2},{1,1,2},{1,1,2}, ge::DT_FLOAT, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_218_fp32) {
  Conv3DV2TestCase({1,16,16,15,17},{4,3,4,3},{2,2,1,1,1,1},{2,1,1},{2,1,1}, ge::DT_FLOAT, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_219_fp32) {
  Conv3DV2TestCase({1,18,16,16,15},{9,3,3,4},{1,1,2,2,1,1},{1,2,1},{2,1,1}, ge::DT_FLOAT, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_220_fp32) {
  Conv3DV2TestCase({1,16,16,16,4096},{1,4,2,40},{1,1,0,0,1,1},{10,3,2},{2,2,2}, ge::DT_FLOAT, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_221_fp32) {
  Conv3DV2TestCase({1,16,64,16,2560},{1,64,3,3},{1,1,1,1,1,1},{1,63,1},{1,3,1}, ge::DT_FLOAT, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_222_fp32) {
  Conv3DV2TestCase({1,6,87,3400,39},{45,3,18,2},{0,0,11,11,1,1},{4,15,52},{6,9,6}, ge::DT_FLOAT, 0);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_1_hif8) {
  Conv3DV2TestCase({1,1,1,1,1}, {1,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_HIFLOAT8, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_exception_1) { // batch fail
  Conv3DV2TestCase({1,1,1,256,1}, {1,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 2, 1, 256, 1, true);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_exception_2) { // D fail
  Conv3DV2TestCase({1,1,1,256,1}, {1,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 1, 2, 256, 1, true);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_exception_3) { // H fail
  Conv3DV2TestCase({1,1,1,256,1}, {1,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 1, 1, 257, 1, true);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_exception_4) { // W fail
  Conv3DV2TestCase({1,1,1,256,1}, {1,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 1, 1, 256, 2, true);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_group_case1) {
  Conv3DV2TestCase({1,12,1,1,1}, {2,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 2);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_group_case2) {
  Conv3DV2TestCase({2,15,1,10,10}, {6,1,2,2}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 3);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_group_case3) {
  Conv3DV2TestCase({1,16,1,11,10}, {12,1,3,4}, {0,0,0,0,1,1}, {1,1,2}, {1,1,1}, ge::DT_FLOAT16, 0, 4);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_group_case4) {
  Conv3DV2TestCase({2,15,1,10,11}, {20,1,4,3}, {0,0,1,1,0,0}, {1,1,1}, {1,1,2}, ge::DT_FLOAT16, 1, 5);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_group_case5) {
  Conv3DV2TestCase({1,12,1,11,11}, {30,1,6,6}, {0,0,1,2,3,4}, {1,2,1}, {1,1,1}, ge::DT_FLOAT16, 0, 6);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_group_case6) {
  Conv3DV2TestCase({2,7,1,11,20}, {42,1,6,6}, {0,0,2,1,4,3}, {1,1,1}, {1,2,1}, ge::DT_FLOAT16, 1, 7);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_group_case7) {
  Conv3DV2TestCase({1,32,1,64,64}, {32,1,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_group_case8) {
  Conv3DV2TestCase({1,20,123,3984,16}, {22,3,23,2}, {0,0,14,14,0,0}, {54,47,15}, {20,10,6}, ge::DT_FLOAT16, 0, 2);
}

TEST_F(Conv3dv2Tiling, run_solomon_conv3dv2_group_not_equal_one_case1) {
  Conv3DV2TestCase({1,20,123,3984,16}, {22,3,23,2}, {0,0,14,14,0,0}, {54,47,15}, {20,10,6}, ge::DT_FLOAT16, 0, 2, 0, 0, 0, 0, true, "VALID", false, "Ascend910_55");
}

TEST_F(Conv3dv2Tiling, run_solomon_conv3dv2_group_not_equal_one_case2) {
  Conv3DV2TestCase({1,20,123,3984,16}, {22,3,23,2}, {0,0,14,14,0,0}, {54,47,15}, {20,10,6}, ge::DT_BF16, 0, 2, 0, 0, 0, 0, true, "VALID", false, "Ascend910_55");
}

TEST_F(Conv3dv2Tiling, run_solomon_conv3dv2_group_not_equal_one_case3) {
  Conv3DV2TestCase({1,20,123,3984,16}, {22,3,23,2}, {0,0,14,14,0,0}, {54,47,15}, {20,10,6}, ge::DT_FLOAT, 0, 2, 0, 0, 0, 0, true, "VALID", false, "Ascend910_55");
}

TEST_F(Conv3dv2Tiling, run_solomon_conv3dv2_group_not_equal_one_case4) {
  Conv3DV2TestCase({1,20,123,3984,16}, {22,3,23,2}, {0,0,14,14,0,0}, {54,47,15}, {20,10,6}, ge::DT_HIFLOAT8, 0, 2, 0, 0, 0, 0, true, "VALID", false, "Ascend910_55");
}

TEST_F(Conv3dv2Tiling, run_solomen_conv3dv2_NDHWC_case1) {
  Conv3DV2TestCase({1,20,123,3984,16}, {22,3,23,2}, {0,0,14,14,0,0}, {54,47,15}, {20,10,6}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0,  true, "SPECIFIC",false, "Ascend910_55",ge::FORMAT_NDHWC);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_NDHWC_case1) {
  Conv3DV2TestCase({1,20,123,3984,16}, {22,3,23,2}, {0,0,14,14,0,0}, {54,47,15}, {20,10,6}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0,  false, "SPECIFIC",false, "Ascend910_95",ge::FORMAT_NDHWC);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_NDHWC_case2) {
  Conv3DV2TestCase({1,20,123,3984,16}, {22,3,23,2}, {0,0,14,14,0,0}, {54,47,15}, {20,10,6}, ge::DT_BF16, 0, 1, 0, 0, 0, 0,  false, "SPECIFIC",false, "Ascend910_95",ge::FORMAT_NDHWC);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_NDHWC_case3) {
  Conv3DV2TestCase({1,20,123,3984,16}, {22,3,23,2}, {0,0,14,14,0,0}, {54,47,15}, {20,10,6}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0,  false, "SPECIFIC",false, "Ascend910_95",ge::FORMAT_NDHWC);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_NDHWC_case4) {
  Conv3DV2TestCase({1,20,123,3984,16}, {22,3,23,2}, {0,0,14,14,0,0}, {54,47,15}, {20,10,6}, ge::DT_HIFLOAT8, 0, 1, 0, 0, 0, 0,  true, "SPECIFIC",false, "Ascend910_95",ge::FORMAT_NDHWC);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_NDHWC_group_not_equal_one_case5) {
  Conv3DV2TestCase({1,20,123,3984,16}, {22,3,23,2}, {0,0,14,14,0,0}, {54,47,15}, {20,10,6}, ge::DT_FLOAT16, 0, 2, 0, 0, 0, 0,  false, "SPECIFIC",false, "Ascend910_95",ge::FORMAT_NDHWC);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_NDHWC_case6) {
  Conv3DV2TestCase({1,20,123,3984,16}, {22,3,23,2}, {0,0,14,14,0,0}, {54,47,15}, {20,10,6}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0,  false, "SPECIFIC",true, "Ascend910_95",ge::FORMAT_NDHWC);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_NDHWC_case7) {
  Conv3DV2TestCase({1,20,16,16,1000000}, {1000000,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0,  true, "SPECIFIC",false, "Ascend910_95",ge::FORMAT_NDHWC);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_NDHWC_case8) {
  Conv3DV2TestCase({1,1000000,16,1000000,512}, {16,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0,  false, "SPECIFIC",false, "Ascend910_95",ge::FORMAT_NDHWC);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_pad_ge_kernel_case_1) {
  Conv3DV2TestCase({2,256,19,160,6160}, {256,3,3,3}, {10,10,31,31,31,31}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_pad_ge_kernel_case_2) {
  Conv3DV2TestCase({2,256,19,160,160}, {256,3,3,3}, {10,10,31,31,31,31}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_pad_ge_kernel_case_3) {
  Conv3DV2TestCase({2,256,19,160,160}, {256,3,3,3}, {10,10,31,31,31,31}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_pad_ge_kernel_case_4) {
  Conv3DV2TestCase({2,256,19,160,160}, {256,3,3,3}, {10,10,31,31,31,31}, {1,1,1}, {1,1,1}, ge::DT_HIFLOAT8, 1);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_pad_ge_kernel_ndhwc_case_1) {
  Conv3DV2TestCase({2,256,19,160,160}, {256,3,3,3}, {10,10,31,31,31,31}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, "Ascend910_95", ge::FORMAT_NDHWC);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_pad_ge_kernel_ndhwc_case_2) {
  Conv3DV2TestCase({2,256,19,160,160}, {256,3,3,3}, {10,10,31,31,31,31}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, "Ascend910_95", ge::FORMAT_NDHWC);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_pad_ge_kernel_ndhwc_case_3) {
  Conv3DV2TestCase({2,256,19,160,160}, {256,3,3,3}, {10,10,31,31,31,31}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, "Ascend910_95", ge::FORMAT_NDHWC);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_pad_ge_kernel_valid) {
  Conv3DV2TestCase({2,256,19,160,160}, {256,3,3,3}, {310,310,310,310,310,310}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, true, "SPECIFIC", false, "Ascend910_95");
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_pad_ge_kernel_ndhwc_valid) {
  Conv3DV2TestCase({2,256,19,160,160}, {256,3,3,3}, {310,310,310,310,310,310}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, true, "SPECIFIC", false, "Ascend910_95", ge::FORMAT_NDHWC);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_dequant_outputfp16_biasfp16_Mmode) {
  Conv3DV2TestCase({1,16,16,17,16},{20,4,3,3},{1,1,1,1,2,2},{1,1,2},{1,1,2}, ge::DT_INT8, 1, 1, 0, 0, 0, 0,
    false, "SPECIFIC", false, "Ascend910_95", ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, true, true, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_dequant_outputfp16_biasfp32_Mmode) {
  Conv3DV2TestCase({1,16,16,17,16},{20,4,3,3},{1,1,1,1,2,2},{1,1,2},{1,1,2}, ge::DT_INT8, 1, 1, 0, 0, 0, 0,
    false, "SPECIFIC", false, "Ascend910_95", ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, true, true, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_dequant_outputbf16_biasbf16_Mmode) {
  Conv3DV2TestCase({1,16,16,17,16},{20,4,3,3},{1,1,1,1,2,2},{1,1,2},{1,1,2}, ge::DT_INT8, 1, 1, 0, 0, 0, 0,
    false, "SPECIFIC", false, "Ascend910_95", ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, true, true, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_dequant_outputbf16_biasfp32_Mmode) {
  Conv3DV2TestCase({1,16,16,17,16},{20,4,3,3},{1,1,1,1,2,2},{1,1,2},{1,1,2}, ge::DT_INT8, 1, 1, 0, 0, 0, 0,
    false, "SPECIFIC", false, "Ascend910_95", ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, true, true, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_dequant_outputfp16_biasfp16) {
  Conv3DV2TestCase({1,6,87,3400,39},{45,3,18,2},{0,0,11,11,1,1},{4,15,52},{6,9,6}, ge::DT_INT8, 1, 1, 0, 0, 0, 0,
    false, "SPECIFIC", false, "Ascend910_95", ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, true, true, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_dequant_outputfp16_biasfp32) {
  Conv3DV2TestCase({1,6,87,3400,39},{45,3,18,2},{0,0,11,11,1,1},{4,15,52},{6,9,6}, ge::DT_INT8, 1, 1, 0, 0, 0, 0,
    false, "SPECIFIC", false, "Ascend910_95", ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, true, true, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_dequant_outputbf16_biasbf16) {
  Conv3DV2TestCase({1,6,87,3400,39},{45,3,18,2},{0,0,11,11,1,1},{4,15,52},{6,9,6}, ge::DT_INT8, 1, 1, 0, 0, 0, 0,
    false, "SPECIFIC", false, "Ascend910_95", ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, true, true, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_dequant_outputbf16_biasfp32) {
  Conv3DV2TestCase({1,6,87,3400,39},{45,3,18,2},{0,0,11,11,1,1},{4,15,52},{6,9,6}, ge::DT_INT8, 1, 1, 0, 0, 0, 0,
    false, "SPECIFIC", false, "Ascend910_95", ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, true, true, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_dequant_error_scaledtype) {
  Conv3DV2TestCase({1,6,87,3400,39},{45,3,18,2},{0,0,11,11,1,1},{4,15,52},{6,9,6}, ge::DT_INT8, 1, 1, 0, 0, 0, 0,
    true, "SPECIFIC", false, "Ascend910_95", ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, true, true, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_dequant_error_dtype_group) {
  Conv3DV2TestCase({1,6,87,3400,39},{45,3,18,2},{0,0,11,11,1,1},{4,15,52},{6,9,6}, ge::DT_INT8, 1, 1, 0, 0, 0, 0,
    true, "SPECIFIC", false, "Ascend910_95", ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, true, true, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_dequant_error_format_group) {
  Conv3DV2TestCase({1,6,87,3400,39},{45,3,18,2},{0,0,11,11,1,1},{4,15,52},{6,9,6}, ge::DT_INT8, 1, 1, 0, 0, 0, 0,
    true, "SPECIFIC", false, "Ascend910_95", ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, true, true, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_dequant_error_groups) {
  Conv3DV2TestCase({1,6,87,3400,39},{45,3,18,2},{0,0,11,11,1,1},{4,15,52},{6,9,6}, ge::DT_INT8, 1, 2, 0, 0, 0, 0,
    true, "SPECIFIC", false, "Ascend910_95", ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, true, true, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_dequant_error_bias_null) {
  Conv3DV2TestCase({1,6,87,3400,39},{45,3,18,2},{0,0,11,11,1,1},{4,15,52},{6,9,6}, ge::DT_INT8, 0, 1, 0, 0, 0, 0,
    true, "SPECIFIC", false, "Ascend910_95", ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, true, true, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT);
}

TEST_F(Conv3dv2Tiling, run_conv3dv2_dequant_error_sclae_null) {
  Conv3DV2TestCase({1,6,87,3400,39},{45,3,18,2},{0,0,11,11,1,1},{4,15,52},{6,9,6}, ge::DT_INT8, 1, 1, 0, 0, 0, 0,
    true, "SPECIFIC", false, "Ascend910_95", ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, true, false, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT);
}