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
 * \file test_conv2d_ascendc_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <stdio.h>

#include "log/log.h"
#include "array_ops.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_base_utils.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_base.h"

using namespace std;
using namespace ge;
using namespace ut_util;

uint64_t L1_Size = 524288;
uint64_t n0 = 16;
uint64_t mAL1_min = 16;
uint64_t m0 = 16;
uint64_t C0_SIZE = 32;

class Tiling : public testing::Test {
    protected:
      static void SetUpTestCase() {
          std::cout << "Conv2d ascendc ops tiling testParam setup" << std::endl;
      }
      static void TearDownTestCase() {
          std::cout << "Conv2d ascendc ops tiling testParam tearDown" << std::endl;
      }
};

uint64_t InferHo(uint64_t inputHiL1, uint64_t singlekH, uint64_t padTop, uint64_t padBottom, uint64_t dilationH, uint64_t strideH)
{
    return (inputHiL1 + padTop + padBottom - dilationH * (singlekH - 1) - 1) / strideH + 1;
}

uint64_t InferWo(uint64_t inputWiL1, uint64_t singlekW, uint64_t padLeft, uint64_t padRight, uint64_t dilationW, uint64_t strideW)
{
    return (inputWiL1 + padLeft + padRight - dilationW * (singlekW - 1) - 1) / strideW + 1;
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

uint64_t ConvAlignB(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return 0;
    }
    return ((a + b - 1) / b) * b;
}

struct TilingParam {
    // api tilingdata
    uint64_t orgHi;
    uint64_t orgWi;
    uint64_t orgHo;
    uint64_t orgWo;
    uint64_t oriHinxWin;
    uint64_t singleCoreBatch;
    uint64_t singleCoreHo;
    uint64_t singleCoreWo;
    uint32_t orgCi;
    uint32_t orgCo;
    uint32_t singleCoreCi;
    uint32_t singleCoreCo;
    uint32_t hoL1;
    uint32_t woL1;
    uint32_t kAL1;
    uint32_t kBL1;
    uint32_t khL1;
    uint32_t kwL1;
    uint32_t nBL1;
    uint32_t hoL0;
    uint32_t woL0;
    uint32_t kL0;
    uint32_t nL0;
    uint32_t pBufferFlag;
    uint32_t groups_api;
    uint32_t enlarge_api;
    uint32_t singleCoreGroups;
    uint32_t singleCoreGroupOpt;
    uint32_t bUbNStep;
    uint32_t bUbKStep;
    uint32_t khUb;
    uint32_t kwUb;
    uint32_t kernelHxkernelW;
    uint32_t kernelHxkernelWxkernelD;
    uint32_t aL1SpaceSize;
    uint32_t multiNBL1;
    uint32_t cinAInCore;
    uint32_t cinATailInCore;
    uint32_t cinBInCore;
    uint32_t cinBTailInCore;
    uint32_t mStep;
    uint32_t kStep;
    uint32_t nStep;
    uint32_t fmapKStride;
    uint32_t weightKStride;
    uint32_t cinOffsetBlockInGM;
    uint32_t coutOffsetBlock;
    uint32_t nL1DivBlockSize;
    uint32_t kernelH;
    uint32_t kernelW;
    uint32_t strideH_api;
    uint32_t strideW_api;
    uint32_t dilationH_api;
    uint32_t dilationW_api;
    uint32_t padTop_api;
    uint32_t padBottom;
    uint32_t padLeft_api;
    uint32_t padRight;
    uint32_t innerBatch;
    uint8_t iterateMNOrder;
    uint8_t biasFullLoadFlag;
    uint8_t fixpParamsFullLoadFlag;
    uint8_t hf32Enable;
    uint8_t hf32TransMode;
    uint8_t hasBias_api;
    uint8_t hasScale;
    uint8_t dualOutput;
    uint8_t quantMode0;
    uint8_t reluMode0;
    uint8_t clipMode0;
    uint8_t quantMode1;
    uint8_t reluMode1;
    uint8_t clipMode1;
    int8_t offsetx;
    int8_t roundMode;
    // ops tilingdata
    uint64_t hin;
    uint64_t win;
    uint64_t hout;
    uint64_t wout;
    uint32_t batch;
    uint32_t cin;
    uint32_t cout;
    uint32_t kh;
    uint32_t kw;
    uint32_t batchDim;
    uint32_t groupDim;
    uint32_t nDim;
    uint32_t hoDim;
    uint32_t woDim;
    uint32_t strideH;
    uint32_t strideW;
    uint32_t dilationH;
    uint32_t dilationW;
    uint32_t padTop;
    uint32_t padLeft;
    uint32_t groups;
    uint32_t enlarge;
    uint32_t cinOpt;
    uint32_t coutOpt;
    uint32_t groupOpt;
    uint8_t hasBias;
};

//=========================
bool isConv1dFlag(uint64_t Hi, uint64_t Hk, uint64_t dilationH, uint64_t strideH, int64_t padTop, int64_t padBottom, bool hasScale, bool isC04Mode)
{
    if (isC04Mode) {
        return false;
    }
    if (Hi == 1 && Hk == 1 && strideH == 1 && dilationH == 1 &&
        padTop == 0 && padBottom == 0) {
        return true;
    }
    return false;
}

uint64_t CalcMinUsedL1SizeInMsplitMode(TilingParam &tilingData, uint32_t kAL1min, uint32_t kBL1min,
  uint32_t fMapDtypeSize, uint32_t biasDtypeSize, uint32_t weightDtypeSize, uint32_t scaleDtypeSize, bool hasScale)
{
    uint64_t nBL1min = n0;
    uint64_t biasUsedL1Size = tilingData.hasBias_api ? ConvAlignB(nBL1min * biasDtypeSize, 32) : 0;
    uint64_t scaleUsedL1Size = hasScale ? ConvAlignB(nBL1min * scaleDtypeSize, 32) : 0;
    uint64_t weightUsedL1Size = ConvAlignB(kBL1min * nBL1min * weightDtypeSize, 32);
    uint64_t hoAL1min = std::min(m0 / tilingData.orgWo + 2, tilingData.orgHo);
    uint64_t hiAL1min = InferHiL1(hoAL1min, tilingData.orgHi, tilingData.kernelH, tilingData.dilationH_api, tilingData.strideH_api);
    uint64_t fmapUsedL1Size = ConvAlignB(hiAL1min * tilingData.orgWi * kAL1min * fMapDtypeSize, 32);
    uint64_t minL1LoadSize = biasUsedL1Size + fmapUsedL1Size + weightUsedL1Size + scaleUsedL1Size;
    return minL1LoadSize;
}

uint64_t CalcMinUsedL1SizeInHWsplitMode(TilingParam &tilingData, uint32_t kAL1min, uint32_t kBL1min, uint32_t wiAL1min,
  uint32_t fMapDtypeSize, uint32_t biasDtypeSize, uint32_t weightDtypeSize, uint32_t scaleDtypeSize, bool hasScale)
{
    uint64_t nBL1min = n0;
    uint64_t biasUsedL1Size = tilingData.hasBias_api ? ConvAlignB(nBL1min * biasDtypeSize, 32) : 0;
    uint64_t scaleUsedL1Size = hasScale ? ConvAlignB(nBL1min * scaleDtypeSize, 32) : 0;
    uint64_t weightUsedL1Size = ConvAlignB(kBL1min * nBL1min * weightDtypeSize, 32);
    uint64_t hoAL1min = tilingData.orgWo < m0 ? ConvCeilDiv(m0, tilingData.orgWo) : 1;
    uint64_t hiAL1min = InferHiL1(hoAL1min, tilingData.orgHi, tilingData.kernelH, tilingData.dilationH_api, tilingData.strideH_api);
    uint64_t fmapUsedL1Size = ConvAlignB(hiAL1min * wiAL1min * kAL1min * fMapDtypeSize, 32);

    uint64_t minL1LoadSize = biasUsedL1Size + scaleUsedL1Size + fmapUsedL1Size + weightUsedL1Size;
    return minL1LoadSize;
}

bool CheckL1SizeLimitsInHWsplitMode(TilingParam &tilingData,
  uint32_t fMapDtypeSize, uint32_t biasDtypeSize, uint32_t weightDtypeSize, uint32_t scaleDtypeSize, bool hasScale)
{
    // require hiL1 * wiL1 >= m0
    uint64_t woAL1min = m0;
    uint32_t k0 = 32 / fMapDtypeSize;
    uint64_t wiAL1min = InferWiL1(woAL1min, tilingData.orgWi, tilingData.kernelW, tilingData.dilationW, tilingData.strideW);
    uint64_t usdL1SizeUnderMinHWtiling = CalcMinUsedL1SizeInHWsplitMode(tilingData, k0, tilingData.kernelH * tilingData.kernelW * k0, wiAL1min,
      fMapDtypeSize, biasDtypeSize, weightDtypeSize, scaleDtypeSize, hasScale);
    if (usdL1SizeUnderMinHWtiling > 524288) {
        return false;
    }
    return true;
}

bool CheckL1SizeLimitsInMsplitMode(TilingParam &tilingData,
  uint32_t fMapDtypeSize, uint32_t biasDtypeSize, uint32_t weightDtypeSize, uint32_t scaleDtypeSize, bool hasScale)
{
    uint32_t k0 = 32 / fMapDtypeSize;
    uint64_t usdL1SizeUnderMinMtiling = CalcMinUsedL1SizeInMsplitMode(tilingData, k0, tilingData.kernelH * tilingData.kernelW * k0,
      fMapDtypeSize, biasDtypeSize, weightDtypeSize, scaleDtypeSize, hasScale);
    if (usdL1SizeUnderMinMtiling > 524288) {
        return false;
    }
    return true;
}

bool CheckC04L1SizeLimitsInHWsplitMode(TilingParam &tilingData,
  uint32_t fMapDtypeSize, uint32_t biasDtypeSize, uint32_t weightDtypeSize, uint32_t scaleDtypeSize, bool hasScale)
{
    // c04 require wi fulload L1
    uint32_t k0 = 32 / fMapDtypeSize;
    uint64_t usdL1SizeUnderMinHWtiling = CalcMinUsedL1SizeInHWsplitMode(tilingData, 4, ConvAlignB(
        4 * tilingData.kernelH * tilingData.kernelW, k0), tilingData.orgWi,
        fMapDtypeSize, biasDtypeSize, weightDtypeSize, scaleDtypeSize, hasScale);
    if (usdL1SizeUnderMinHWtiling > 524288) {
        return false;
    }
    return true;
}

bool CheckC04L1SizeLimitsInMsplitMode(TilingParam &tilingData,
  uint32_t fMapDtypeSize, uint32_t biasDtypeSize, uint32_t weightDtypeSize, uint32_t scaleDtypeSize, bool hasScale)
{
    uint32_t k0 = 32 / fMapDtypeSize;
    uint64_t c04UsdL1SizeUnderMinMtiling = CalcMinUsedL1SizeInMsplitMode(tilingData, 4, ConvAlignB(4 * tilingData.kernelH * tilingData.kernelW, k0),
      fMapDtypeSize, biasDtypeSize, weightDtypeSize, scaleDtypeSize, hasScale);
    if (c04UsdL1SizeUnderMinMtiling > 524288) {
        return false;
    }
    return true;
}

// M split mode return 1, HW split mode retun 0, M and HW split mode both fail return -1
int32_t GetSplitMode(TilingParam &tilingData, uint32_t featuremapDtyeSize, uint32_t weightDtypeSize, bool hasScale, bool isC04Mode)
{
	uint32_t biasDtyeSize = 0;
  uint32_t scaleDtyeSize = 0;
	if (tilingData.hasBias) {
		if (hasScale) {
			biasDtyeSize = 4; // biasdetype is int32 for int8; biasdetype is fp32 for hif8/fp8
		}
		else {
			biasDtyeSize = featuremapDtyeSize; // biasdtype is same as fmdtype for fp32/hf32/fp16/bf16
		}
	}
	if (hasScale) {
		scaleDtyeSize = 8; // scaleDtye is int64/uint64 for int8/hif8/fp8
	}
  bool MsplitModeL1LimitCheckRes = false;
  bool HWsplitModeL1LimitCheckRes = false;
  if (isC04Mode) {
    MsplitModeL1LimitCheckRes = CheckC04L1SizeLimitsInMsplitMode(tilingData, featuremapDtyeSize, biasDtyeSize, weightDtypeSize, scaleDtyeSize, hasScale);
    HWsplitModeL1LimitCheckRes = CheckC04L1SizeLimitsInHWsplitMode(tilingData, featuremapDtyeSize, biasDtyeSize, weightDtypeSize, scaleDtyeSize, hasScale);
  } else {
    MsplitModeL1LimitCheckRes = CheckL1SizeLimitsInMsplitMode(tilingData, featuremapDtyeSize, biasDtyeSize, weightDtypeSize, scaleDtyeSize, hasScale);
    HWsplitModeL1LimitCheckRes = CheckL1SizeLimitsInHWsplitMode(tilingData, featuremapDtyeSize, biasDtyeSize, weightDtypeSize, scaleDtyeSize, hasScale);
  }
  if (isConv1dFlag(tilingData.orgHi, tilingData.kernelH, tilingData.dilationH, tilingData.strideH, tilingData.padTop, tilingData.padBottom, hasScale, false)) {
      MsplitModeL1LimitCheckRes = false; // only hw split mode in conv1d
  }

	if(MsplitModeL1LimitCheckRes && HWsplitModeL1LimitCheckRes) {
		return 1;
	}
	if(!MsplitModeL1LimitCheckRes && HWsplitModeL1LimitCheckRes) {
		return 0;
	}
	if(MsplitModeL1LimitCheckRes && !HWsplitModeL1LimitCheckRes) {
		return 1;
	}
	if(!MsplitModeL1LimitCheckRes && !HWsplitModeL1LimitCheckRes) {
		return -1;
	}
}

uint64_t CalcUsdL1Size(TilingParam &tilingData,
					   uint32_t featuremapDtyeSize,
					   uint32_t weightDtyeSize,
					   uint64_t n0,
					   uint32_t outputOrder,
                       int8_t pbAL1,
                       int8_t pbBL1,
                       int64_t padCompensationValue,
					   bool hasBias,
					   bool hasQuantScale,
					   bool isC04Flag)
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
        uint64_t mL1Max = tilingData.hoL1 < tilingData.singleCoreHo ? tilingData.hoL1 : tilingData.singleCoreHo;
        uint64_t hoAL1Tmp = min(mL1Max / tilingData.orgWo + 2, tilingData.orgHo);
        uint64_t hiL1Tmp = min((hoAL1Tmp - 1) * tilingData.strideH + (tilingData.kernelH - 1) / tilingData.dilationH + 1, tilingData.orgHi);
        al1Size = hiL1Tmp * tilingData.orgWi * (tilingData.kAL1 / (tilingData.kernelH * tilingData.kernelW)) * (pbAL1 + 1) * featuremapDtyeSize;
        bl1Size = tilingData.nBL1 * tilingData.kBL1 * (pbBL1 + 1) * weightDtyeSize;
        if (hasBias) {
            if (tilingData.biasFullLoadFlag == 1) {
                biasL1Size = ConvCeilDiv(tilingData.singleCoreCo, n0) * n0 * biasDtyeSize;
            } else { // tilingData.biasFullLoadFlag == 0
                biasL1Size = tilingData.nL0 * biasDtyeSize;
            }
        }
        if (hasQuantScale) {
            if (tilingData.fixpParamsFullLoadFlag) {
                scaleL1Size = ConvCeilDiv(tilingData.singleCoreCo, n0) * n0 * scaleDtyeSize;
            } else {
                scaleL1Size = tilingData.nL0 * scaleDtyeSize;
            }
        }
        curl1Size = al1Size + bl1Size + biasL1Size + scaleL1Size;
    } else { // HWmode
        uint64_t hiL1 = InferHiL1(tilingData.hoL1, tilingData.orgHi, tilingData.kernelH, tilingData.dilationH, tilingData.strideH);
        uint64_t wiL1 = InferWiL1(tilingData.woL1, tilingData.orgWi, tilingData.kernelW, tilingData.dilationW, tilingData.strideW);
        al1Size = hiL1 * wiL1 * (tilingData.kAL1 / (tilingData.kernelH * tilingData.kernelW)) * (pbAL1 + 1) * featuremapDtyeSize;
        if (isC04Flag) {
            al1Size = ConvCeilDiv(hiL1 * wiL1 * 4 * featuremapDtyeSize, 32) * 32 * (pbAL1 + 1);
        }
        bl1Size = tilingData.nBL1 * tilingData.kBL1 * weightDtyeSize;
        if (hasBias) {
            if (tilingData.biasFullLoadFlag) {
                biasL1Size = ConvCeilDiv(tilingData.singleCoreCo, n0) * n0 * biasDtyeSize;
            } else {
                biasL1Size = tilingData.nBL1 * biasDtyeSize;
            }
        }
        if (hasQuantScale) {
            if (tilingData.fixpParamsFullLoadFlag) {
                scaleL1Size = ConvCeilDiv(tilingData.singleCoreCo, n0) * n0 * scaleDtyeSize;
            } else {
                scaleL1Size = tilingData.nBL1 * scaleDtyeSize;
            }
        }
        curl1Size = al1Size + bl1Size + biasL1Size + scaleL1Size;
    }

    return curl1Size;
}

uint64_t CalcUsdL0ASize(TilingParam &tilingData,
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

uint64_t CalcUsdL0BSize(TilingParam &tilingData,
					   uint32_t weightDtyeSize,
                       int8_t pbBL0)
{
    return tilingData.nL0 * tilingData.kL0 * (pbBL0 + 1) * weightDtyeSize;
}

uint64_t CalcUsdL0CSize(TilingParam &tilingData,
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
//===============================================================================
void GetInitBasicBlockMN(TilingParam &tilingData, uint64_t& basicBlockM, uint64_t& basicBlockN)
{
    constexpr uint32_t BASICBLOCK_BOUNDARY_VALUE_64 = 64;
    constexpr uint32_t BASICBLOCK_BOUNDARY_VALUE_128 = 128;
    constexpr uint32_t BASICBLOCK_INIT_VALUE_64 = 64;
    constexpr uint32_t BASICBLOCK_INIT_VALUE_128 = 128;
    constexpr uint32_t BASICBLOCK_INIT_VALUE_256 = 256;
    constexpr uint32_t BASICBLOCK_INIT_VALUE_512 = 512;
    constexpr uint32_t BASICBLOCK_INIT_VALUE_1024 = 1024;

    uint64_t howo = tilingData.hout * tilingData.wout;
    if (tilingData.cout <= BASICBLOCK_BOUNDARY_VALUE_64) {
        basicBlockM = BASICBLOCK_INIT_VALUE_1024;
        basicBlockN = BASICBLOCK_INIT_VALUE_64;
    } else if (tilingData.cout > BASICBLOCK_BOUNDARY_VALUE_64
        && tilingData.cout <= BASICBLOCK_BOUNDARY_VALUE_128) {
        basicBlockM = BASICBLOCK_INIT_VALUE_512;
        basicBlockN = BASICBLOCK_INIT_VALUE_128;
    } else if (howo <= BASICBLOCK_BOUNDARY_VALUE_64) {
        basicBlockM = BASICBLOCK_INIT_VALUE_64;
        basicBlockN = BASICBLOCK_INIT_VALUE_1024;
    } else if (howo > BASICBLOCK_BOUNDARY_VALUE_64
        && howo <= BASICBLOCK_BOUNDARY_VALUE_128) {
        basicBlockM = BASICBLOCK_INIT_VALUE_128;
        basicBlockN = BASICBLOCK_INIT_VALUE_512;
    } else {
        basicBlockM = BASICBLOCK_INIT_VALUE_256;
        basicBlockN = BASICBLOCK_INIT_VALUE_256;
    }
}

bool CmpFirstAdjustMnTile(TilingParam &tilingData, int64_t& mTile, int64_t& nTile,
                            int64_t availableL1Size, uint64_t basicBlockM, uint64_t basicBlockN,
                            int64_t fMapDtypeSize, int64_t weightDtypeSize, vector<int64_t> pads)
{
    int64_t k0 = 32 / fMapDtypeSize;
    int64_t maxHiWiL1 = availableL1Size / fMapDtypeSize / k0 / 2;
    int64_t padTop = pads[0];
    int64_t padBottom = pads[1];
    if (maxHiWiL1 <= 0) {
        return false;
    }
    int64_t maxhiL1 = maxHiWiL1 / static_cast<int64_t>(tilingData.win);
    if (maxhiL1 <= 2) {
        return false;
    }
    int64_t hoMax = 0;
    int64_t padCompensationValue = 0;
	if (basicBlockM >= tilingData.wout * tilingData.hout) { // L1 can full load M direction
        padCompensationValue = padTop + padBottom;
        hoMax = (maxhiL1 + padCompensationValue -
            static_cast<int64_t>(tilingData.dilationH) *
            (static_cast<int64_t>(tilingData.kh) - 1) - 1) /
            static_cast<int64_t>(tilingData.strideH) + 1;
    } else {
        padCompensationValue = max(padTop, padBottom);
        hoMax = (maxhiL1 + padCompensationValue - static_cast<int64_t>(tilingData.dilationH) *
		    (static_cast<int64_t>(tilingData.kh) - 1) - 1) /
            static_cast<int64_t>(tilingData.strideH) + 1;
    }
	if (hoMax <= 0) {
        return false;
	}
    int64_t maxHoWoL1 = hoMax * static_cast<int64_t>(tilingData.wout);
    int64_t cmpM = tilingData.hout * tilingData.wout;
    int64_t cmpN = availableL1Size / weightDtypeSize / 2 / k0 / tilingData.kh / tilingData.kw;
    mTile = min(min(cmpM, maxHoWoL1), static_cast<int64_t>(basicBlockM));
    nTile = min(min(static_cast<int64_t>(tilingData.cout), cmpN), static_cast<int64_t>(basicBlockN));
    if (tilingData.groupOpt == 0) {
        nTile = ConvCeilDiv(nTile, tilingData.groups);
    } else if (tilingData.groupOpt != 0) {
        nTile = ConvCeilDiv(nTile, tilingData.groups) * tilingData.enlarge;
    }
    int64_t m0 = 16;
    int64_t n0 = 16;
    mTile = mTile / m0 * m0;
    nTile = nTile / n0 * n0;
    if (mTile < m0 || nTile < n0) {
        return false;
    }
    return true;
}

void SelectMmodeAlgorithm(TilingParam &tilingData, bool& mBasicBlockModeFlag, uint64_t aicoreNum, uint32_t featuremapDtyeSize,
                        uint32_t weightDtyeSize, vector<int64_t> pads, bool hasBias, bool hasScale)
{
    mBasicBlockModeFlag = false;
    uint64_t basicBlockM = 0;
    uint64_t basicBlockN = 0;
    GetInitBasicBlockMN(tilingData, basicBlockM, basicBlockN);
    uint64_t mCut = ConvCeilDiv(tilingData.wout * tilingData.hout, basicBlockM);
    uint64_t nCut = ConvCeilDiv(tilingData.cout, basicBlockN);
    uint64_t group = tilingData.groups;
    if (tilingData.groupOpt != 0) {
        group = tilingData.groupOpt;
    }
    if (mCut * nCut * tilingData.batch * group <= aicoreNum) {
        return;
    }
    int64_t biasSize = 0;
    int64_t scaleSize = 0;
    if (hasBias) {
        biasSize = featuremapDtyeSize; // for fp32/hf32/fp16/bf16
        if (hasScale) {
            biasSize = 4; // bias dtype is fp32 for int8/fp8/hif8
        }
    }
    if (hasScale) {
        scaleSize = 8; // scale dtype is uint64/int64 for int8/fp8/hif8
    }
    int64_t availableL1Size = L1_Size - biasSize - scaleSize;
    int64_t mTile = 0;
    int64_t nTile = 0;
    if (!CmpFirstAdjustMnTile(tilingData, mTile, nTile, availableL1Size,basicBlockM,
                            basicBlockN, featuremapDtyeSize, weightDtyeSize, pads)) {
        return;
    }
    mBasicBlockModeFlag = true;
    return;
}

void CheckHWModeTilingDataValidForConv2d(TilingParam &tilingData, uint64_t m0, uint64_t n0, uint64_t k0, bool isC04Flag)
{
        // K direction check
        EXPECT_GE(tilingData.kL0, k0);
        EXPECT_LE(tilingData.kL0, std::min(tilingData.kAL1, tilingData.kBL1));
        EXPECT_EQ(tilingData.kL0 % k0, 0);

        // N direction check
        EXPECT_GE(tilingData.nL0, n0);
        EXPECT_GE(tilingData.nBL1, tilingData.nL0);
        EXPECT_LE(tilingData.nBL1, ConvCeilDiv(ConvCeilDiv(tilingData.singleCoreCo, n0) * n0, tilingData.nL0) * tilingData.nL0);
        EXPECT_EQ(tilingData.nL0 % n0, 0);
        uint32_t nBL1DivCheck = 0;
        if (tilingData.nBL1 % tilingData.nL0 == 0 ||
            tilingData.nBL1 == ConvCeilDiv(tilingData.singleCoreCo, n0) * n0) {
            nBL1DivCheck = 1;
        }
        EXPECT_EQ(nBL1DivCheck, 1);

        // W direction check
        EXPECT_GE(tilingData.woL0, m0);
        EXPECT_GE(tilingData.woL1, tilingData.woL0);
        EXPECT_LE(tilingData.woL1,
            ConvCeilDiv(ConvCeilDiv(tilingData.singleCoreWo, m0) * m0, tilingData.woL0) * tilingData.woL0);
        EXPECT_EQ(tilingData.woL0 % m0, 0);
        if (tilingData.woL0 < ConvCeilDiv(tilingData.orgWo, m0) * m0) {
            // woL0 does not reach the upper limit, thus hoL0 must be 1.
            EXPECT_EQ(tilingData.hoL0, 1);
        }
        if (tilingData.hoL0 > 1) {
            EXPECT_EQ(tilingData.woL0, ConvCeilDiv(tilingData.orgWo, m0) * m0);
            EXPECT_EQ(tilingData.woL1, ConvCeilDiv(tilingData.orgWo, m0) * m0);
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
            EXPECT_EQ(tilingData.kAL1, ConvCeilDiv(4 * tilingData.kernelH * tilingData.kernelW, k0) * k0);
            EXPECT_EQ(tilingData.kBL1, ConvCeilDiv(4 * tilingData.kernelH * tilingData.kernelW, k0) * k0);
            if (tilingData.orgHi > 1) {
                // w fullload in L1
                EXPECT_EQ(tilingData.woL1, ConvCeilDiv(tilingData.orgWo, m0) * m0);
            }
            // if tilingData.orgHi == 1, process is same as NO_C04_SITUATION
            // fmap fullload in L1, woL1 == AlignB(tilingIns_->shapeInfo.singleWo, tilingIns_->cubeInfo.m0)
            // woL1 may not be able to divide woL0 exactly
        } else {
            EXPECT_LE(tilingData.kAL1, ConvCeilDiv(tilingData.singleCoreCi, k0) * k0 * tilingData.kernelH * tilingData.kernelW);
            // Mmode KBL1 iter when Cin is Max, will multi kd in some case. This is a loose validation condition.
            EXPECT_LE(tilingData.kBL1, ConvCeilDiv(tilingData.singleCoreCi, k0) * k0 * tilingData.kernelHxkernelWxkernelD);
            EXPECT_EQ(tilingData.kAL1 % (k0 * tilingData.kernelH * tilingData.kernelW), 0);
            EXPECT_EQ(tilingData.kBL1 % (k0 * tilingData.kernelH * tilingData.kernelW), 0);
            uint32_t kAL1DivCheck = 0;
            if (tilingData.kAL1 % tilingData.kL0 == 0 ||
                tilingData.kAL1 == ConvCeilDiv(tilingData.singleCoreCi, k0) * k0 * tilingData.kernelH * tilingData.kernelW) {
                kAL1DivCheck = 1;
            }
            EXPECT_EQ(kAL1DivCheck, 1);
            uint32_t kBL1DivCheck = false;
            if (tilingData.kBL1 % tilingData.kL0 == 0 ||
                tilingData.kBL1 == ConvCeilDiv(tilingData.singleCoreCi, k0) * k0 * tilingData.kernelH * tilingData.kernelW ||
                tilingData.kBL1 == ConvCeilDiv(tilingData.singleCoreCi, k0) * k0 * tilingData.kernelHxkernelWxkernelD) {
                kBL1DivCheck = true;
            }
            EXPECT_EQ(kBL1DivCheck, 1);

            // No woL1 % woL0 check
            // In fmap fullload situation, woL1 is not obtained by magnifying, thus woL1 may not be able to divide woL0 exactly
            // In fmap fullload situation, since woL1 needs to align to m0, thus woL1 may be not equal to singleCoreWo
        }
}

void CheckGroupsTiling(TilingParam &tilingData, uint64_t n0, uint64_t tilingKey)
{
    int32_t groupMode = tilingData.enlarge > 0 ? 2 : 1;
    uint64_t realCo = 0;
    if (groupMode == 1) {
        realCo = tilingData.cout / tilingData.groups;
    } else if (groupMode == 2) {
        realCo = tilingData.coutOpt;

    }
    EXPECT_LE(tilingData.nDim, ConvCeilDiv(realCo, n0));
}

void CheckValidTilingData(TilingParam &tilingData,
                          uint64_t m0,
                          uint64_t n0,
                          uint64_t k0,
                          uint32_t weightDtyeSize,
                          uint32_t featuremapDtyeSize,
                          std::vector<int64_t> pads,
                          uint64_t l1Size,
                          uint64_t l0aSize,
                          uint64_t l0bSize,
                          uint64_t l0cSize,
                          bool hasBias,
                          bool hasScale,
						  uint64_t tilingKey,
                          uint64_t aicoreNum)
{
    bool isC04Flag = (tilingData.bUbNStep > 0 && tilingData.bUbKStep == 0) ? true : false;
    if (tilingData.groups > 1) {
        CheckGroupsTiling(tilingData, n0, tilingKey);
    }
    uint64_t pBuffer = tilingData.pBufferFlag;
    int8_t pbAL0 = pBuffer & 0x01;
    int8_t pbBL0 = (pBuffer & 0x02) >> 1;
    int8_t pbCL0 = (pBuffer & 0x04) >> 2;
    int8_t pbAL1 = (pBuffer & 0x08) >> 3;
    int8_t pbBL1 = (pBuffer & 0x10) >> 4;
	 uint64_t nBL1 = tilingData.multiNBL1 * tilingData.nL0;
	 int32_t outputOrder = tilingData.singleCoreWo == 0 && tilingData.woL1 == 0;
    bool mBasicBlockModeFlag = false;
    int64_t padCompensationValue = 0;
    int64_t padTop = pads[0];
    int64_t padBottom = pads[1];
    int32_t splitModeFromCmp = GetSplitMode(tilingData, featuremapDtyeSize, weightDtyeSize, hasScale, isC04Flag);
   //  EXPECT_EQ(splitModeFromCmp, outputOrder);
    if (outputOrder == 1) {
        SelectMmodeAlgorithm(tilingData, mBasicBlockModeFlag, aicoreNum,
        featuremapDtyeSize, weightDtyeSize, pads, hasBias, hasScale); // determine m-mode formulation algorithm or basic block algorithm
    }
    if (mBasicBlockModeFlag) {
        if (tilingData.hoL0 >= tilingData.wout * tilingData.hout) {
            padCompensationValue = padTop + padBottom;
        } else {
            padCompensationValue = max(padTop, padBottom);
        }
    }

    EXPECT_LE(CalcUsdL1Size(tilingData, featuremapDtyeSize, weightDtyeSize, n0, outputOrder, pbAL1, pbBL1, padCompensationValue, hasBias, hasScale, isC04Flag), l1Size);
    EXPECT_LE(CalcUsdL0ASize(tilingData, outputOrder, featuremapDtyeSize, pbAL0), l0aSize);
    EXPECT_LE(CalcUsdL0BSize(tilingData, weightDtyeSize, pbBL0), l0bSize);
    EXPECT_LE(CalcUsdL0CSize(tilingData, outputOrder, pbCL0), l0cSize);

    if (outputOrder == 1) {
        EXPECT_GT(tilingData.kAL1, 0);
        EXPECT_GT(tilingData.kBL1, 0);
        EXPECT_GT(tilingData.hoL1, 0);
        EXPECT_GT(nBL1, 0);
        EXPECT_GT(tilingData.hoL0, 0);
        EXPECT_GT(tilingData.kL0, 0);
        EXPECT_GT(tilingData.nL0, 0);

        EXPECT_EQ(tilingData.kAL1 % k0, 0);
        EXPECT_EQ(tilingData.nBL1 % n0, 0);
        EXPECT_EQ(tilingData.nL0 % n0, 0);
        EXPECT_EQ(tilingData.kL0 % k0, 0);
        if (!mBasicBlockModeFlag) { // only check m-mode/hw-mode formulation algorithm
            EXPECT_EQ(tilingData.kAL1 % tilingData.kL0, 0);
            EXPECT_EQ(tilingData.kBL1 % tilingData.kL0, 0);
        }
		    uint32_t mmadDtypeSize = 4; // if bf16 fp16 fp32 hf32 in cube is 4, int8 in cube is int32, hif8 / fp8 in cube is fp32
        EXPECT_LE(tilingData.nBL1, ConvCeilDiv(ConvCeilDiv(tilingData.singleCoreCo, n0) * n0, tilingData.nL0) * tilingData.nL0);
        EXPECT_LE(tilingData.hoL1, ConvCeilDiv(ConvCeilDiv(tilingData.singleCoreHo, m0) * m0, tilingData.hoL0) * tilingData.hoL0);
        EXPECT_LE(tilingData.kAL1, tilingData.kernelH * tilingData.kernelW * ConvCeilDiv(tilingData.cin, k0) * k0);
        EXPECT_LE(tilingData.kBL1, tilingData.kernelH * tilingData.kernelW * ConvCeilDiv(tilingData.cin, k0) * k0);
        EXPECT_LE(tilingData.hoL0, ConvCeilDiv(std::min(l0aSize / (k0 * (pbAL0 + 1) * featuremapDtyeSize), l0cSize / (n0 * (pbCL0 + 1) * mmadDtypeSize)), m0) * m0);
        EXPECT_LE(tilingData.nL0, ConvCeilDiv(std::min(l0bSize / (k0 * (pbBL0 + 1) * weightDtyeSize), l0cSize / (m0 * (pbCL0 + 1) * mmadDtypeSize)), n0) * n0);
        if (!mBasicBlockModeFlag) { // only check m-mode formulation algorithm
            EXPECT_LE(tilingData.kL0, ConvGcd(ConvCeilDiv(tilingData.kAL1, k0), ConvCeilDiv(tilingData.kBL1, k0)) * k0);
        }
        EXPECT_EQ(tilingData.woL1, 0);
        EXPECT_EQ(tilingData.woL0, 0);
        EXPECT_EQ(tilingData.singleCoreWo, 0);
        EXPECT_EQ(tilingData.hoL1 % m0, 0);
        /*
		if (!tiling.l1TilingInfo.bl1FullLoad) {
            EXPECT_EQ(tilingData.nBL1 % tilingData.nL0, 0);
        }
        if (!tiling.l1TilingInfo.al1FullLoad) {
            EXPECT_EQ(tilingData.hoL1 % tilingData.hoL0, 0);
        }
		*/
    }

    if (outputOrder == 0) {
        CheckHWModeTilingDataValidForConv2d(tilingData, m0, n0, k0, isC04Flag);
        /*
        EXPECT_GT(tilingData.woL1, 0);
        EXPECT_GT(tilingData.woL0, 0);

        EXPECT_LE(tilingData.nBL1, ConvCeilDiv(tilingData.singleCoreCo, n0) * n0);
        EXPECT_LE(tilingData.hoL1, tilingData.singleCoreHo);
        EXPECT_LE(tilingData.woL1, ConvCeilDiv(tilingData.singleCoreWo, m0) * m0);
        EXPECT_LE(tilingData.hoL0, tilingData.hoL1);
        EXPECT_LE(tilingData.woL0, tilingData.woL1);
        EXPECT_LE(tilingData.kL0, std::min(tilingData.kAL1, tilingData.kBL1));

        EXPECT_EQ(tilingData.hoL1 % tilingData.hoL0, 0);
        EXPECT_EQ(tilingData.singleCoreHo % tilingData.hoL1, 0);
        EXPECT_EQ(ConvCeilDiv(tilingData.singleCoreCo, n0) * n0 % tilingData.nBL1, 0);
        if (isC04Flag) {
            EXPECT_EQ(tilingData.kAL1, ConvCeilDiv(4 * tilingData.kernelH * tilingData.kernelW, k0) * k0);
            EXPECT_EQ(tilingData.kBL1, ConvCeilDiv(4 * tilingData.kernelH * tilingData.kernelW, k0) * k0);
            EXPECT_EQ(tilingData.kBL1 % (ConvCeilDiv(4 * tilingData.kernelH * tilingData.kernelW, k0)), 0);
            EXPECT_EQ(tilingData.kBL1 % (ConvCeilDiv(4 * tilingData.kernelH * tilingData.kernelW, k0)), 0);
        } else {
            EXPECT_EQ(tilingData.woL0 % m0, 0);
            EXPECT_EQ(tilingData.kBL1 % k0, 0);
            EXPECT_EQ(tilingData.woL1 % tilingData.woL0, 0);
            EXPECT_EQ(ConvCeilDiv(tilingData.singleCoreCi, k0) * k0 * tilingData.kernelH * tilingData.kernelW % tilingData.kAL1, 0);
            EXPECT_EQ(ConvCeilDiv(tilingData.singleCoreCi, k0) * k0 * tilingData.kernelH * tilingData.kernelW % tilingData.kBL1, 0);
            EXPECT_EQ(ConvCeilDiv(tilingData.singleCoreWo, m0) * m0 % tilingData.woL1, 0);
            EXPECT_EQ(tilingData.kBL1 % k0, 0);
            if (tilingData.woL1 != tilingData.singleCoreWo) {
                EXPECT_EQ(tilingData.woL1 % m0, 0);
            }

            EXPECT_LE(tilingData.kAL1, ConvCeilDiv(tilingData.singleCoreCi, k0) * k0 * tilingData.kernelH * tilingData.kernelW);
            EXPECT_LE(tilingData.kBL1, ConvCeilDiv(tilingData.singleCoreCi, k0) * k0 * tilingData.kernelH * tilingData.kernelW);
        }
        */
    }
}

void GetOriPadFromPadModeConv2D(const string& padMode, uint32_t& padu, uint32_t& padd,
  uint32_t& padl, uint32_t& padr, uint32_t strideH, uint32_t strideW,
  uint32_t dilationH, uint32_t dilationW, int64_t batch, int64_t cin, int64_t hi, int64_t wi,
  int64_t cout, int64_t kH, int64_t kW)
{
    if (padMode == "SPECIFIC") {
        return;
    }

    if (padMode == "VALID") {
        padu = 0;
        padd = 0;
        padl = 0;
        padr = 0;
        return;
    } else {
        auto padH = (ConvCeilDiv(hi, strideH) - 1) * strideH + dilationH * (kH - 1) - hi + 1;
        auto padW = (ConvCeilDiv(wi, strideW) - 1) * strideW + dilationW * (kW - 1) - wi + 1;
        if (padMode == "SAME" || padMode == "SAME_UPPER") {
            padd = ConvCeilDiv(padH, 2);
            padu = padH - padd;
            padr = ConvCeilDiv(padW, 2);
            padl = padW - padr;
        } else {
            // padMode is "SAME_LOWER"
            padu = ConvCeilDiv(padH, 2);
            padd = padH - padu;
            padl = ConvCeilDiv(padW, 2);
            padr = padW - padl;
        }
    }
    return;
}

//======================================================================================================

void Conv2DTestCase(vector<int64_t> fmShape, vector<int64_t> weightShape,
                    vector<uint32_t> pads, vector<uint32_t> strides, vector<uint32_t> dilations, ge::DataType dtype,
                    uint32_t isHasBias = 1, uint32_t isHasScale = 0, bool enableHf32Mode = false, uint32_t groups = 1,
                    string padMode = "SPECIFIC", string socVersion = "Ascend910_9589", string shortSocVersion = "Ascend910_95",
                    bool isErrorCaseFlag = false, string format = "NCHW") {
    bool hasBias = isHasBias == 1;
    bool hasScale = isHasScale == 1;

    uint32_t padu = pads[0];
    uint32_t padd = pads[1];
    uint32_t padl = pads[2];
    uint32_t padr = pads[3];
    uint32_t strideH = strides[0];
    uint32_t strideW = strides[1];
    uint32_t dilationH = dilations[0];
    uint32_t dilationW = dilations[1];
    int64_t cout = weightShape[0];
    int64_t kH = weightShape[1];
    int64_t kW = weightShape[2];
    int64_t batch = fmShape[0];
    int64_t cin = fmShape[1];
    int64_t hi = fmShape[2];
    int64_t wi = fmShape[3];
    GetOriPadFromPadModeConv2D(padMode, padu, padd, padl, padr, strideH, strideW,
        dilationH, dilationW, batch, cin, hi, wi, cout, kH, kW);
    int64_t ho = InferHo(hi, kH, padu, padd, dilationH, strideH);
    int64_t wo = InferWo(wi, kW, padl, padr, dilationW, strideW);
    EXPECT_GE(ho, 1);
    EXPECT_GE(wo, 1);

    ge::Format fmapFormat = ge::FORMAT_NCHW;
    ge::Format weightFormat = ge::FORMAT_NCHW;
    ge::Format outputFormat = ge::FORMAT_NCHW;

    gert::StorageShape featuremap = {{batch, cin, hi, wi}, {batch, cin, hi, wi}};
    gert::StorageShape weight = {{cout, cin / groups, kH, kW}, {cout, cin / groups, kH, kW}};
    gert::StorageShape bias = {{cout}, {cout}};
    gert::StorageShape quantScale = {{cout}, {cout}};
    gert::StorageShape offset_w;
    gert::StorageShape output = {{batch, cout, ho, wo}, {batch, cout, ho, wo}};

    if (format == "NHWC") {
      fmapFormat = ge::FORMAT_NHWC;
      weightFormat = ge::FORMAT_HWCN;
      outputFormat = ge::FORMAT_NHWC;

      featuremap = {{batch, hi, wi, cin}, {batch, hi, wi, cin}};
      weight = {{kH, kW, cin / groups, cout}, {kH, kW, cin / groups, cout}};
      output = {{batch, ho, wo, cout}, {batch, ho, wo, cout}};
    }

    // 对于可选输入，不传时用nullptr占位
    std::vector<void*> input_shape_ref;

    if (hasScale) {
      if (hasBias) {
          input_shape_ref = {&featuremap, &weight, &quantScale, &bias, nullptr};
      } else {
          input_shape_ref = {&featuremap, &weight, &quantScale, nullptr, nullptr};
      }
    } else {
      if (hasBias) {
          input_shape_ref = {&featuremap, &weight, &bias, nullptr};
      } else {
          input_shape_ref = {&featuremap, &weight, nullptr, nullptr};
      }
    }
    std::vector<void*> output_shapes_ref = {&output};
    std::vector<int64_t> strides = {1, 1, strideH, strideW};
    std::vector<int64_t> pads = {padu, padd, padl, padr};
    std::vector<int64_t> dilations = {1, 1, dilationH, dilationW};

    if (format == "NHWC") {
      strides = {1, strideH, strideW, 1};
      dilations = {1, dilationH, dilationW, 1};
    }

    std::string op_type = hasScale ? "QuantConv2D" : "Conv2DV2";
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    uint64_t L1_SIZE = 524288;
    uint64_t L0a_SIZE = 65536;
    uint64_t L0b_SIZE = 65536;
    uint64_t L0c_SIZE = 262144;
    uint64_t bt_SIZE = 4096;
    uint64_t fb_SIZE = 4096;
    uint64_t aicoreNum = 32;

    string compile_info_string = R"({"hardware_info":
      {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
       "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 196608,
       "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "FB_SIZE": 4096, "BT_SIZE": 4096,"L0C_SIZE": 262144, "CORE_NUM": 32}})";
    if (shortSocVersion == "Ascend910_55") {
      aicoreNum = 24;
      compile_info_string = R"({"hardware_info":
      {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
       "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 196608,
       "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 24}})";
    }
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    map<string, string> soc_version_infos = {{"Short_SoC_version", shortSocVersion}};
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::conv_ops_tiling::ConvTilingParseInfo compile_info;
    compile_info.aicoreNum = aicoreNum;
    compile_info.socVersion = socVersion;
    compile_info.shortSocVersion = shortSocVersion;

    optiling::Conv2DTilingParseInfo quant_compile_info;
    quant_compile_info.opType = op_type;
    quant_compile_info.socVersion = socVersion;
    quant_compile_info.shortSocVersion = shortSocVersion;
    auto tilingDataPtr = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(tilingDataPtr, nullptr);

    std::map<ge::DataType, uint32_t> k0Map = {{ge::DT_FLOAT16, 16}, {ge::DT_FLOAT, 8}, {ge::DT_INT8, 32}, {ge::DT_BF16, 16}, {ge::DT_HIFLOAT8, 32}, {ge::DT_FLOAT8_E4M3FN, 32}};
    std::map<ge::DataType, uint32_t> dtypesizeMap = {{ge::DT_FLOAT16, 2}, {ge::DT_FLOAT, 4}, {ge::DT_INT8, 1}, {ge::DT_BF16, 2}, {ge::DT_HIFLOAT8, 1}, {ge::DT_FLOAT8_E4M3FN, 1}};
    uint64_t m0 = 16;
    uint64_t k0 = k0Map.at(dtype);
    uint64_t n0 = 16;
    uint32_t weightDtyeSize = dtypesizeMap.at(dtype);
    uint32_t featuremapDtyeSize = dtypesizeMap.at(dtype);
    ge::DataType bias_dtype = (!hasScale && dtype == ge::DT_HIFLOAT8) ? ge::DT_FLOAT : dtype;

    auto holder = hasScale ?
                      gert::TilingContextFaker().SetOpType(op_type)
                                                .NodeIoNum(5, 1)
                                                .IrInstanceNum({1, 1, 1, 1, 1})
                                                .InputShapes(input_shape_ref)
                                                .OutputShapes(output_shapes_ref)
                                                .CompileInfo(&quant_compile_info)
                                                .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                                                .NodeInputTd(0, ge::DT_INT8, fmapFormat, fmapFormat)
                                                .NodeInputTd(1, ge::DT_INT8, weightFormat, weightFormat)
                                                .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                                                .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                                .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                                .NodeOutputTd(0, ge::DT_FLOAT16, outputFormat, outputFormat)
                                                .NodeAttrs({
                                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                                  {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                                                  {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(format)},
                                                  {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                                  {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")}
                                                  })
                                                .TilingData(tilingDataPtr.get())
                                                .Workspace(ws_size)
                                                .Build() :
                      gert::TilingContextFaker().SetOpType(op_type)
                                                .NodeIoNum(4, 1)
                                                .IrInstanceNum({1, 1, 1, 1})
                                                .InputShapes(input_shape_ref)
                                                .OutputShapes(output_shapes_ref)
                                                .CompileInfo(&compile_info)
                                                .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                                                .NodeInputTd(0, dtype, fmapFormat, fmapFormat)
                                                .NodeInputTd(1, dtype, weightFormat, weightFormat)
                                                .NodeInputTd(2, bias_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                                                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                                .NodeOutputTd(0, dtype, outputFormat, outputFormat)
                                                .NodeAttrs({
                                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                                  {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                                                  {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(format)},
                                                  {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                                  {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(enableHf32Mode)}
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
      // printf("tilingKey is equal to %lu\n", tilingKey);
      EXPECT_LE(tilingParam.batchDim * tilingParam.hoDim * tilingParam.nDim * tilingParam.groupDim, compile_info.aicoreNum);
      EXPECT_GE(tilingParam.batchDim, 1);
      EXPECT_GE(tilingParam.hoDim, 1);
      EXPECT_GE(tilingParam.nDim, 1);
      EXPECT_GE(tilingParam.groupDim, 1);
      if (tilingParam.batchDim > 0 && tilingParam.hoDim > 0 && tilingParam.nDim > 0 && tilingParam.groupDim > 0) {
        CheckValidTilingData(tilingParam, m0, n0, k0, weightDtyeSize, featuremapDtyeSize, pads, L1_SIZE, L0a_SIZE, L0b_SIZE, L0c_SIZE, hasBias, hasScale, tilingKey, compile_info.aicoreNum);
      }
    } else {
      EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    }

}

void ExtendConv2DTestCase(vector<int64_t> fmShape, vector<int64_t> weightShape,
                    vector<uint32_t> pads, vector<uint32_t> strides, vector<uint32_t> dilations,
                    ge::DataType inDataType, ge::DataType out0DataType, ge::DataType out1DataType,
                    bool isHasBias = true, bool isHasScale0 = false, bool isHasScale1 = false,
                    bool isHasReluWeight0 = false, bool isHasReluWeight1 = false,
                    bool isHasClipValue0 = false, bool isHasClipValue1 = false,
                    bool enableRelu0 = false, bool enableRelu1 = false, bool dualOutput = false,
                    bool enableHf32Mode = false, uint32_t groups = 1,
                    string padMode = "SPECIFIC",  string socVersion = "Ascend910_9589", string shortSocVersion = "Ascend910_95",
                    uint8_t errorCaseStatus = 0, string format = "NCHW", string round_mode = "rint") {
    bool hasBias = isHasBias == 1;
    bool hasScale = isHasScale0 == 1;
    bool isErrorCaseFlag = errorCaseStatus == 0 ? false : true;

    uint32_t padu = pads[0];
    uint32_t padd = pads[1];
    uint32_t padl = pads[2];
    uint32_t padr = pads[3];
    uint32_t strideH = strides[0];
    uint32_t strideW = strides[1];
    uint32_t dilationH = dilations[0];
    uint32_t dilationW = dilations[1];
    int64_t cout = weightShape[0];
    int64_t kH = weightShape[1];
    int64_t kW = weightShape[2];
    int64_t batch = fmShape[0];
    int64_t cin = fmShape[1];
    int64_t hi = fmShape[2];
    int64_t wi = fmShape[3];
    GetOriPadFromPadModeConv2D(padMode, padu, padd, padl, padr, strideH, strideW,
        dilationH, dilationW, batch, cin, hi, wi, cout, kH, kW);
    int64_t ho = InferHo(hi, kH, padu, padd, dilationH, strideH);
    int64_t wo = InferWo(wi, kW, padl, padr, dilationW, strideW);
    EXPECT_GE(ho, 1);
    EXPECT_GE(wo, 1);

    ge::Format fmapFormat = ge::FORMAT_NCHW;
    ge::Format weightFormat = ge::FORMAT_NCHW;
    ge::Format outputFormat = ge::FORMAT_NCHW;

    gert::StorageShape featuremap = {{batch, cin, hi, wi}, {batch, cin, hi, wi}};
    gert::StorageShape weight = {{cout, cin / groups, kH, kW}, {cout, cin / groups, kH, kW}};
    gert::StorageShape bias = {{cout}, {cout}};
    gert::StorageShape Scale0 = {{cout}, {cout}};
    gert::StorageShape Scale1 = {{cout}, {cout}};
    gert::StorageShape offset_w;
    gert::StorageShape output0 = {{batch, cout, ho, wo}, {batch, cout, ho, wo}};
    gert::StorageShape output1 = {{batch, cout, ho, wo}, {batch, cout, ho, wo}};

    if (format == "NHWC") {
      fmapFormat = ge::FORMAT_NHWC;
      weightFormat = ge::FORMAT_HWCN;
      outputFormat = ge::FORMAT_NHWC;

      featuremap = {{batch, hi, wi, cin}, {batch, hi, wi, cin}};
      weight = {{kH, kW, cin / groups, cout}, {kH, kW, cin / groups, cout}};
      output0 = {{batch, ho, wo, cout}, {batch, ho, wo, cout}};
      output1 = {{batch, ho, wo, cout}, {batch, ho, wo, cout}};
    }

    // 对于可选输入，不传时用nullptr占位
    std::vector<void*> input_shape_ref;
   //  if (hasScale) {
   //    if (hasBias) {
   //        input_shape_ref = {&featuremap, &weight, &bias, nullptr, &Scale0};
   //    } else {
   //        input_shape_ref = {&featuremap, &weight, nullptr, nullptr, &Scale0};
   //    }
   //  }
   if (hasBias) {
      if (isHasScale1) {
         input_shape_ref = {&featuremap, &weight, &bias, &Scale0, &Scale1};
      } else {
         input_shape_ref = {&featuremap, &weight, &bias, &Scale0};
      }
   } else {
      if (isHasScale1) {
         input_shape_ref = {&featuremap, &weight, nullptr, nullptr, &Scale0, nullptr, nullptr, &Scale1};
      } else {
         input_shape_ref = {&featuremap, &weight, nullptr, nullptr, &Scale0};
      }
   }
    std::vector<void*> output_shapes_ref = {&output0, &output1};
    ge::DataType fmapDataType = inDataType;
    ge::DataType weightDataType = inDataType;
    ge::DataType biasDataType = (inDataType == ge::DT_INT8) ? ge::DT_INT32 : (inDataType == ge::DT_HIFLOAT8 || inDataType == ge::DT_FLOAT8_E4M3FN) ? ge::DT_FLOAT : inDataType;
    ge::DataType offsetWDataType = ge::DT_INT8; // format is ND
    ge::DataType scale0DType = ge::DT_INT64;
    ge::DataType scale1DType = ge::DT_INT64;
    ge::DataType reluWeight0DType = ge::DT_FLOAT;
    ge::DataType clipValue0DType = ge::DT_FLOAT16;
    ge::DataType outputDataType = out0DataType;

    std::vector<int64_t> strides = {1, 1, strideH, strideW};
    std::vector<int64_t> pads = {padu, padd, padl, padr};
    std::vector<int64_t> dilations = {1, 1, dilationH, dilationW};

    if (format == "NHWC") {
      strides = {1, strideH, strideW, 1};
      dilations = {1, dilationH, dilationW, 1};
    }
    std::string op_type = "ExtendConv2D";
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    uint64_t L1_SIZE = 524288;
    uint64_t L0a_SIZE = 65536;
    uint64_t L0b_SIZE = 65536;
    uint64_t L0c_SIZE = 262144;
    uint64_t bt_SIZE = 4096;
    uint64_t fb_SIZE = 4096;
    uint64_t aicoreNum = 32;
    string compile_info_string = R"({"hardware_info":
      {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
       "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 196608,
       "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "FB_SIZE": 4096, "BT_SIZE": 4096, "L0C_SIZE": 262144, "CORE_NUM": 32}})";
    if (shortSocVersion == "Ascend910_55") {
      aicoreNum = 24;
      compile_info_string = R"({"hardware_info":
      {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
       "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 196608,
       "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 24}})";
    }
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    map<string, string> soc_version_infos = {{"Short_SoC_version", shortSocVersion}};
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::conv_ops_tiling::ConvTilingParseInfo compile_info;
    compile_info.aicoreNum = aicoreNum;
    compile_info.socVersion = socVersion;
    compile_info.shortSocVersion = shortSocVersion;

    auto tilingDataPtr = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(tilingDataPtr, nullptr);

    std::map<ge::DataType, uint32_t> k0Map = {{ge::DT_FLOAT16, 16}, {ge::DT_FLOAT, 8}, {ge::DT_INT8, 32}, {ge::DT_BF16, 16}, {ge::DT_HIFLOAT8, 32}, {ge::DT_FLOAT8_E4M3FN, 32}};
    std::map<ge::DataType, uint32_t> dtypesizeMap = {{ge::DT_FLOAT16, 2}, {ge::DT_FLOAT, 4}, {ge::DT_INT8, 1}, {ge::DT_BF16, 2}, {ge::DT_HIFLOAT8, 1}, {ge::DT_FLOAT8_E4M3FN, 1}};
    uint64_t m0 = 16;
    uint64_t k0 = k0Map.at(fmapDataType);
    uint64_t n0 = 16;
    uint32_t weightDtyeSize = dtypesizeMap.at(fmapDataType);
    uint32_t featuremapDtyeSize = dtypesizeMap.at(fmapDataType);
    ge::DataType bias_dtype = (!hasScale && fmapDataType == ge::DT_HIFLOAT8) ? ge::DT_FLOAT : fmapDataType;


    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    // generate TilingParseContext to do TilingPrepareForConv2DV2
    auto kernel_holder = gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    // 新增
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    //新增
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    fe::PlatFormInfos platform_info1;
    platform_info1.Init();
    auto holder = isHasScale1 ?
                     gert::TilingContextFaker().SetOpType(op_type)
                                             .NodeIoNum(5, 2)
                                             .IrInstanceNum({1, 1, 1, 0, 1, 0, 0, 1}) // 控制算子原型对应位置的可选输入，是否存在
                                             .InputShapes(input_shape_ref)
                                             .OutputShapes(output_shapes_ref)
                                             .CompileInfo(&compile_info)
                                             .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                                             .NodeInputTd(0, fmapDataType, fmapFormat, fmapFormat)
                                             .NodeInputTd(1, weightDataType, weightFormat, weightFormat)
                                             .NodeInputTd(2, biasDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                                             .NodeInputTd(3, scale0DType, ge::FORMAT_ND, ge::FORMAT_ND)
                                             .NodeInputTd(4, scale1DType, ge::FORMAT_ND, ge::FORMAT_ND)
                                             .NodeOutputTd(0, out0DataType, outputFormat, outputFormat)
                                             .NodeOutputTd(1, out1DataType, outputFormat, outputFormat)
                                             .NodeAttrs({
                                             {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                             {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                             {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                                             {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                                             {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                             {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                             {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>(round_mode)},
                                             {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                                             {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(enableHf32Mode)},
                                             {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(enableRelu0)},
                                             {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(enableRelu1)},
                                             {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(dualOutput)},
                                             {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                             {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)}
                                             })
                                             .TilingData(tilingDataPtr.get())
                                             .Workspace(ws_size)
                                             .Build() :
                     gert::TilingContextFaker().SetOpType(op_type)
                                             .NodeIoNum(4, 2)
                                             .IrInstanceNum({1, 1, 1, 0, 1}) // 控制算子原型对应位置的可选输入，是否存在
                                             .InputShapes(input_shape_ref)
                                             .OutputShapes(output_shapes_ref)
                                             .CompileInfo(&compile_info)
                                             .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                                             .NodeInputTd(0, fmapDataType, fmapFormat, fmapFormat)
                                             .NodeInputTd(1, weightDataType, weightFormat, weightFormat)
                                             .NodeInputTd(2, biasDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                                             .NodeInputTd(3, scale0DType, ge::FORMAT_ND, ge::FORMAT_ND)
                                             .NodeOutputTd(0, outputDataType, outputFormat, outputFormat)
                                             .NodeOutputTd(1, outputDataType, outputFormat, outputFormat)
                                             .NodeAttrs({
                                             {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                             {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                             {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                                             {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                                             {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                             {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                             {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>(round_mode)},
                                             {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                                             {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(enableHf32Mode)},
                                             {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(enableRelu0)},
                                             {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(enableRelu1)},
                                             {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(dualOutput)},
                                             {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                             {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)}
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
      // printf("tilingKey is equal to %lu\n", tilingKey);
      EXPECT_LE(tilingParam.batchDim * tilingParam.hoDim * tilingParam.nDim * tilingParam.groupDim, compile_info.aicoreNum);
      EXPECT_GE(tilingParam.batchDim, 1);
      EXPECT_GE(tilingParam.hoDim, 1);
      EXPECT_GE(tilingParam.nDim, 1);
      EXPECT_GE(tilingParam.groupDim, 1);
      if (tilingParam.batchDim > 0 && tilingParam.hoDim > 0 && tilingParam.nDim > 0 && tilingParam.groupDim > 0) {
        // CheckValidTilingData(tilingParam, m0, n0, k0, weightDtyeSize, featuremapDtyeSize, pads, L1_SIZE, L0a_SIZE, L0b_SIZE, L0c_SIZE, hasBias, hasScale, tilingKey, compile_info.aicoreNum);
      }
    } else {
      EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    }
}

struct QuantConv2DDtype {
  ge::DataType inputDtype = ge::DT_INT8;
  ge::DataType filterDtype = ge::DT_INT8;
  ge::DataType scaleDtype = ge::DT_INT64;
  ge::DataType biasDtype = ge::DT_INT32;
  ge::DataType offsetDtype = ge::DT_FLOAT;
  ge::DataType yDtype = ge::DT_FLOAT16;
};

static QuantConv2DDtype quantConv2DDtype;

void QuantConv2DWithoutPlatformInfoTestCase(vector<int64_t> fmShape, vector<int64_t> weightShape,
											vector<uint32_t> pads, vector<uint32_t> strides, vector<uint32_t> dilations, ge::DataType dtype,
											uint32_t isHasBias = 1, uint32_t isHasScale = 0, bool compileHf32Mode = false, uint32_t groups = 1,
                      string socVersion = "Ascend910_9589", string shortSocVersion = "Ascend910_95", bool isErrorCaseFlag = false,
                      QuantConv2DDtype quantConv2DDtype = quantConv2DDtype) {
    bool hasBias = isHasBias == 1;
    bool hasScale = isHasScale == 1;

    uint32_t padu = pads[0];
    uint32_t padd = pads[1];
    uint32_t padl = pads[2];
    uint32_t padr = pads[3];
    uint32_t strideH = strides[0];
    uint32_t strideW = strides[1];
    uint32_t dilationH = dilations[0];
    uint32_t dilationW = dilations[1];
    int64_t cout = weightShape[0];
    int64_t kH = weightShape[1];
    int64_t kW = weightShape[2];
    int64_t batch = fmShape[0];
    int64_t cin = fmShape[1];
    int64_t hi = fmShape[2];
    int64_t wi = fmShape[3];
    int64_t ho = InferHo(hi, kH, padu, padd, dilationH, strideH);
    int64_t wo = InferWo(wi, kW, padl, padr, dilationW, strideW);
    EXPECT_GE(ho, 1);
    EXPECT_GE(wo, 1);

    gert::StorageShape featuremap = {{batch, cin, hi, wi}, {batch, cin, hi, wi}};
    gert::StorageShape weight = {{cout, cin / groups, kH, kW}, {cout, cin / groups, kH, kW}};
    gert::StorageShape bias = {{cout}, {cout}};
    gert::StorageShape quantScale = {{cout}, {cout}};
    gert::StorageShape offset_w;
    gert::StorageShape output = {{batch, cout, ho, wo}, {batch, cout, ho, wo}};
    // 对于可选输入，不传时用nullptr占位
    std::vector<void*> input_shape_ref;

    if (hasScale) {
      if (hasBias) {
          input_shape_ref = {&featuremap, &weight, &quantScale, &bias, nullptr};
      } else {
          input_shape_ref = {&featuremap, &weight, &quantScale, nullptr, nullptr};
      }
    } else {
      if (hasBias) {
          input_shape_ref = {&featuremap, &weight, &bias, nullptr};
      } else {
          input_shape_ref = {&featuremap, &weight, nullptr, nullptr};
      }
    }
    std::vector<void*> output_shapes_ref = {&output};
    std::vector<int64_t> strides = {1, 1, strideH, strideW};
    std::vector<int64_t> pads = {padu, padd, padl, padr};
    std::vector<int64_t> dilations = {1, 1, dilationH, dilationW};

    std::string op_type = hasScale ? "QuantConv2D" : "Conv2DV2";
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    uint64_t L1_SIZE = 524288;
    uint64_t L0a_SIZE = 65536;
    uint64_t L0b_SIZE = 65536;
    uint64_t L0c_SIZE = 262144;
    uint64_t bt_SIZE = 4096;
    uint64_t fb_SIZE = 4096;
    uint64_t aicoreNum = 32;
    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144,
                            "FB_SIZE": 4096, "BT_SIZE": 4096,
                            "CORE_NUM": 32}
                            })";
    if (shortSocVersion == "Ascend910_55") {
      aicoreNum = 24;
      compile_info_string = R"({"hardware_info":
      {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
       "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 196608,
       "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 24}})";
    }
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    map<string, string> soc_version_infos = {{"Short_SoC_version", shortSocVersion}};
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::Conv2DTilingParseInfo quant_compile_info;
    quant_compile_info.aicoreNum = aicoreNum;
    quant_compile_info.l2Size = 33554432;
    quant_compile_info.l1Size = L1_SIZE;
    quant_compile_info.l0aSize = L0a_SIZE;
    quant_compile_info.l0bSize = L0b_SIZE;
    quant_compile_info.l0cSize = L0c_SIZE;
    quant_compile_info.btSize = bt_SIZE;
    quant_compile_info.ubSize = 196608;
    quant_compile_info.btSize = 4096;
    quant_compile_info.l2Rate = 100;
    quant_compile_info.socVersion = socVersion;
    quant_compile_info.shortSocVersion = shortSocVersion;

    auto tilingDataPtr = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(tilingDataPtr, nullptr);

    std::map<ge::DataType, uint32_t> k0Map = {{ge::DT_FLOAT16, 16}, {ge::DT_FLOAT, 8}, {ge::DT_INT8, 32}, {ge::DT_BF16, 16}, {ge::DT_HIFLOAT8, 32}, {ge::DT_FLOAT8_E4M3FN, 32}};
    std::map<ge::DataType, uint32_t> dtypesizeMap = {{ge::DT_FLOAT16, 2}, {ge::DT_FLOAT, 4}, {ge::DT_INT8, 1}, {ge::DT_BF16, 2}, {ge::DT_HIFLOAT8, 1}, {ge::DT_FLOAT8_E4M3FN, 1}};
    uint64_t m0 = 16;
    uint64_t k0 = k0Map.at(dtype);
    uint64_t n0 = 16;
    uint32_t weightDtyeSize = dtypesizeMap.at(dtype);
    uint32_t featuremapDtyeSize = dtypesizeMap.at(dtype);

    auto holder = gert::TilingContextFaker().SetOpType(op_type)
											.NodeIoNum(5, 1)
											.IrInstanceNum({1, 1, 1, 1, 1})
											.InputShapes(input_shape_ref)
											.OutputShapes(output_shapes_ref)
											.CompileInfo(&quant_compile_info)
											.PlatformInfo(nullptr)
											.NodeInputTd(0, quantConv2DDtype.inputDtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
											.NodeInputTd(1, quantConv2DDtype.filterDtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
											.NodeInputTd(2, quantConv2DDtype.scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
											.NodeInputTd(3, quantConv2DDtype.biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
											.NodeInputTd(4, quantConv2DDtype.offsetDtype, ge::FORMAT_ND, ge::FORMAT_ND)
											.NodeOutputTd(0, quantConv2DDtype.yDtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
											.NodeAttrs({
											  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
											  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
											  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
											  {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
											  {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
											  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
											  {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
											  {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")}
											  })
											.TilingData(tilingDataPtr.get())
											.Workspace(ws_size)
											.Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();

    if (isErrorCaseFlag == false) {
      EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
      auto buf = (TilingParam*)tiling_context->GetRawTilingData()->GetData();
      TilingParam tilingParam = *buf;
      uint64_t tilingKey = tiling_context->GetTilingKey();
      // printf("tilingKey is equal to %lu\n", tilingKey);
      EXPECT_LE(tilingParam.batchDim * tilingParam.hoDim * tilingParam.nDim * tilingParam.groupDim, quant_compile_info.aicoreNum);
      EXPECT_GE(tilingParam.batchDim, 1);
      EXPECT_GE(tilingParam.hoDim, 1);
      EXPECT_GE(tilingParam.nDim, 1);
      EXPECT_GE(tilingParam.groupDim, 1);
      if (tilingParam.batchDim > 0 && tilingParam.hoDim > 0 && tilingParam.nDim > 0 && tilingParam.groupDim > 0) {
        CheckValidTilingData(tilingParam, m0, n0, k0, weightDtyeSize, featuremapDtyeSize, pads, L1_SIZE, L0a_SIZE, L0b_SIZE, L0c_SIZE, hasBias, hasScale, tilingKey, quant_compile_info.aicoreNum);
      }
    } else {
      EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    }

}
//TEST_F(Tiling, run_conv2d_case_cache_init) {
  //Conv2DTestCase({2,640,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
//}

TEST_F(Tiling, run_conv2d_case_cache_0) {
  Conv2DTestCase({1,1,1,16}, {16,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_cache_1) {
  for  (int i = 0; i < 10; i++) {
  Conv2DTestCase({1,1,1,16}, {16,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
  }
}

TEST_F(Tiling, run_conv2d_case_0) {
  Conv2DTestCase({1,1,1,16}, {16,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_1) {
  Conv2DTestCase({2,1280,36,28}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_2) {
  Conv2DTestCase({2,640,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_3) {
  Conv2DTestCase({2,640,28,36}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_4) {
  Conv2DTestCase({2,320,144,112}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_5) {
  Conv2DTestCase({2,320,152,104}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_6) {
  Conv2DTestCase({2,1920,72,56}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_7) {
  Conv2DTestCase({2,640,20,48}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_8) {
  Conv2DTestCase({2,2560,38,26}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_9) {
  Conv2DTestCase({2,2560,38,26}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_10) {
  Conv2DTestCase({2,1920,36,28}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_11) {
  Conv2DTestCase({2,1280,72,56}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_12) {
  Conv2DTestCase({2,960,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_13) {
  Conv2DTestCase({2,640,144,112}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_14) {
  Conv2DTestCase({2,640,144,112}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_15) {
  Conv2DTestCase({2,1920,26,38}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_16) {
  Conv2DTestCase({2,640,76,52}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_17) {
  Conv2DTestCase({2,960,52,76}, {640,3,3}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_18) {
  Conv2DTestCase({2,640,72,56}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_19) {
  Conv2DTestCase({2,640,72,56}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_20) {
  Conv2DTestCase({2,640,26,38}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_21) {
  Conv2DTestCase({2,640,80,192}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_22) {
  Conv2DTestCase({2,640,38,26}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_23) {
  Conv2DTestCase({2,2560,28,36}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_24) {
  Conv2DTestCase({2,960,112,144}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_25) {
  Conv2DTestCase({2,1280,20,48}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_26) {
  Conv2DTestCase({2,1280,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_27) {
  Conv2DTestCase({2,640,112,144}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_28) {
  Conv2DTestCase({2,320,104,152}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_29) {
  Conv2DTestCase({2,640,144,112}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_30) {
  Conv2DTestCase({2,960,76,52}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_31) {
  Conv2DTestCase({2,2560,36,28}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_32) {
  Conv2DTestCase({2,320,76,52}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_33) {
  Conv2DTestCase({2,640,152,104}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_34) {
  Conv2DTestCase({2,1280,72,56}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_35) {
  Conv2DTestCase({2,1280,72,56}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_36) {
  Conv2DTestCase({2,640,80,192}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_37) {
  Conv2DTestCase({2,2560,26,38}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_38) {
  Conv2DTestCase({2,2560,26,38}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_39) {
  Conv2DTestCase({2,1280,26,38}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_40) {
  Conv2DTestCase({2,320,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_41) {
  Conv2DTestCase({2,320,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_42) {
  Conv2DTestCase({2,1280,52,76}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_43) {
  Conv2DTestCase({2,640,104,152}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_44) {
  Conv2DTestCase({2,1280,28,36}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_45) {
  Conv2DTestCase({2,1280,56,72}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_46) {
  Conv2DTestCase({2,960,104,152}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_47) {
  Conv2DTestCase({2,320,104,152}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_48) {
  Conv2DTestCase({2,1920,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_49) {
  Conv2DTestCase({2,2560,38,26}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_50) {
  Conv2DTestCase({2,960,80,192}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_51) {
  Conv2DTestCase({2,1920,72,56}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_52) {
  Conv2DTestCase({2,320,144,112}, {4,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_53) {
  Conv2DTestCase({2,1920,76,52}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_54) {
  Conv2DTestCase({2,960,112,144}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_55) {
  Conv2DTestCase({2,960,144,112}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_56) {
  Conv2DTestCase({2,960,72,56}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_57) {
  Conv2DTestCase({2,1920,28,36}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_58) {
  Conv2DTestCase({2,960,76,52}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_59) {
  Conv2DTestCase({2,1920,52,76}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_60) {
  Conv2DTestCase({2,1280,38,26}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_61) {
  Conv2DTestCase({2,960,56,72}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_62) {
  Conv2DTestCase({2,960,152,104}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_63) {
  Conv2DTestCase({2,640,104,152}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_64) {
  Conv2DTestCase({2,2560,36,28}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_65) {
  Conv2DTestCase({2,1280,40,96}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_66) {
  Conv2DTestCase({2,1920,28,36}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_67) {
  Conv2DTestCase({2,320,80,192}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_68) {
  Conv2DTestCase({2,640,104,152}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_69) {
  Conv2DTestCase({2,1920,20,48}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_70) {
  Conv2DTestCase({2,320,112,144}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_71) {
  Conv2DTestCase({2,1920,36,28}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_72) {
  Conv2DTestCase({2,960,40,96}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_73) {
  Conv2DTestCase({2,960,56,72}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_74) {
  Conv2DTestCase({2,320,112,144}, {4,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_75) {
  Conv2DTestCase({2,960,72,56}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_76) {
  Conv2DTestCase({2,2560,20,48}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_77) {
  Conv2DTestCase({2,960,152,104}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_78) {
  Conv2DTestCase({4,3,1152,896}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_79) {
  Conv2DTestCase({2,960,144,112}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_80) {
  Conv2DTestCase({2,1920,76,52}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_81) {
  Conv2DTestCase({2,320,152,104}, {4,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_82) {
  Conv2DTestCase({4,3,1024,1024}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_83) {
  Conv2DTestCase({2,1920,26,38}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_84) {
  Conv2DTestCase({2,2560,28,36}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_86) {
  Conv2DTestCase({4,128,1152,896}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_87) {
  Conv2DTestCase({2,1280,52,76}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_88) {
  Conv2DTestCase({4,128,897,1153}, {128,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_89) {
  Conv2DTestCase({2,1280,56,72}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_90) {
  Conv2DTestCase({2,640,56,72}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_91) {
  Conv2DTestCase({2,640,56,72}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_92) {
  Conv2DTestCase({2,960,104,152}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_93) {
  Conv2DTestCase({2,960,40,96}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_94) {
  Conv2DTestCase({4,128,448,576}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_95) {
  Conv2DTestCase({2,1920,56,72}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_97) {
  Conv2DTestCase({4,128,1153,897}, {128,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_98) {
  Conv2DTestCase({2,2560,20,48}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_99) {
  Conv2DTestCase({2,1920,38,26}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_100) {
  Conv2DTestCase({2,640,112,144}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_101) {
  Conv2DTestCase({4,128,1025,1025}, {128,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_102) {
  Conv2DTestCase({2,1280,40,96}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_103) {
  Conv2DTestCase({2,1280,76,52}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_104) {
  Conv2DTestCase({4,3,896,1152}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_105) {
  Conv2DTestCase({2,640,152,104}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_106) {
  Conv2DTestCase({2,960,80,192}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_107) {
  Conv2DTestCase({4,128,576,448}, {256,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_108) {
  Conv2DTestCase({2,1280,56,72}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_109) {
  Conv2DTestCase({2,640,152,104}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_110) {
  Conv2DTestCase({4,256,512,512}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_112) {
  Conv2DTestCase({2,640,36,28}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_113) {
  Conv2DTestCase({2,640,40,96}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_114) {
  Conv2DTestCase({4,3,832,1216}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_115) {
  Conv2DTestCase({4,128,544,480}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_116) {
  Conv2DTestCase({4,128,544,480}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_117) {
  Conv2DTestCase({2,320,72,56}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_118) {
  Conv2DTestCase({2,320,56,72}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_119) {
  Conv2DTestCase({4,3,1088,960}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_120) {
  Conv2DTestCase({4,128,544,480}, {256,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_121) {
  Conv2DTestCase({2,1280,40,96}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_122) {
  Conv2DTestCase({2,1920,20,48}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_123) {
  Conv2DTestCase({4,128,1216,832}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_124) {
  Conv2DTestCase({4,256,208,304}, {512,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_125) {
  Conv2DTestCase({4,128,833,1217}, {128,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_126) {
  Conv2DTestCase({2,1280,76,52}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_127) {
  Conv2DTestCase({4,256,609,417}, {256,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_128) {
  Conv2DTestCase({4,3,896,1152}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_129) {
  Conv2DTestCase({4,256,416,608}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_130) {
  Conv2DTestCase({2,1280,76,52}, {640,3,3}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_131) {
  Conv2DTestCase({4,128,896,1152}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_132) {
  Conv2DTestCase({4,256,288,224}, {512,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_133) {
  Conv2DTestCase({4,256,288,224}, {512,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_134) {
  Conv2DTestCase({4,128,608,416}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_135) {
  Conv2DTestCase({2,1920,56,72}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_136) {
  Conv2DTestCase({4,512,304,208}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_137) {
  Conv2DTestCase({4,128,1024,1024}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_138) {
  Conv2DTestCase({4,256,608,416}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_139) {
  Conv2DTestCase({2,640,112,144}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_140) {
  Conv2DTestCase({4,128,416,608}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_141) {
  Conv2DTestCase({4,256,256,256}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_142) {
  Conv2DTestCase({4,256,417,609}, {256,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_143) {
  Conv2DTestCase({2,1920,40,96}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_144) {
  Conv2DTestCase({4,128,448,576}, {256,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_145) {
  Conv2DTestCase({4,512,112,144}, {8,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_146) {
  Conv2DTestCase({4,256,449,577}, {256,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_147) {
  Conv2DTestCase({4,128,1217,833}, {128,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_148) {
  Conv2DTestCase({2,1920,40,96}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_149) {
  Conv2DTestCase({4,512,144,112}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_150) {
  Conv2DTestCase({4,256,224,288}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_151) {
  Conv2DTestCase({2,640,80,192}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_152) {
  Conv2DTestCase({4,256,576,448}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_153) {
  Conv2DTestCase({4,512,152,104}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_154) {
  Conv2DTestCase({4,512,144,112}, {8,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_155) {
  Conv2DTestCase({4,256,288,224}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_156) {
  Conv2DTestCase({4,3,1216,832}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_157) {
  Conv2DTestCase({4,128,832,1216}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_158) {
  Conv2DTestCase({4,256,544,480}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_159) {
  Conv2DTestCase({4,512,128,128}, {8,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_160) {
  Conv2DTestCase({4,128,1088,960}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_161) {
  Conv2DTestCase({4,4,112,144}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_162) {
  Conv2DTestCase({4,256,448,576}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_163) {
  Conv2DTestCase({4,512,104,152}, {8,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_164) {
  Conv2DTestCase({4,256,304,208}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_165) {
  Conv2DTestCase({4,4,104,152}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_166) {
  Conv2DTestCase({4,128,576,448}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_167) {
  Conv2DTestCase({4,512,289,225}, {512,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_168) {
  Conv2DTestCase({4,512,209,305}, {512,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_169) {
  Conv2DTestCase({4,320,144,112}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_170) {
  Conv2DTestCase({4,512,112,144}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_171) {
  Conv2DTestCase({4,320,112,144}, {320,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_172) {
  Conv2DTestCase({4,128,512,512}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_173) {
  Conv2DTestCase({4,512,305,209}, {512,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_174) {
  Conv2DTestCase({4,256,545,481}, {256,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_175) {
  Conv2DTestCase({4,128,1089,961}, {128,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_176) {
  Conv2DTestCase({4,320,128,128}, {320,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_177) {
  Conv2DTestCase({4,512,128,128}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_178) {
  Conv2DTestCase({4,512,256,256}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_179) {
  Conv2DTestCase({4,256,208,304}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_180) {
  Conv2DTestCase({4,320,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_181) {
  Conv2DTestCase({4,512,136,120}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_182) {
  Conv2DTestCase({4,512,272,240}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_183) {
  Conv2DTestCase({4,512,208,304}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_184) {
  Conv2DTestCase({4,640,56,72}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_185) {
  Conv2DTestCase({4,512,136,120}, {8,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_186) {
  Conv2DTestCase({4,512,224,288}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_187) {
  Conv2DTestCase({4,640,72,56}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_188) {
  Conv2DTestCase({4,640,72,56}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_189) {
  Conv2DTestCase({4,640,72,56}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_190) {
  Conv2DTestCase({4,4,128,128}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_191) {
  Conv2DTestCase({4,512,152,104}, {8,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_192) {
  Conv2DTestCase({4,512,288,224}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_193) {
  Conv2DTestCase({4,320,64,64}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_194) {
  Conv2DTestCase({4,320,136,120}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_195) {
  Conv2DTestCase({4,512,104,152}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_196) {
  Conv2DTestCase({4,640,76,52}, {640,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_197) {
  Conv2DTestCase({4,320,128,128}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_198) {
  Conv2DTestCase({4,4,136,120}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_199) {
  Conv2DTestCase({4,640,52,76}, {640,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_200) {
  Conv2DTestCase({4,320,152,104}, {320,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_201) {
  Conv2DTestCase({4,4,144,112}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_202) {
  Conv2DTestCase({4,640,32,32}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_203) {
  Conv2DTestCase({4,320,56,72}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_204) {
  Conv2DTestCase({4,320,104,152}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_205) {
  Conv2DTestCase({4,256,272,240}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_206) {
  Conv2DTestCase({4,1280,28,36}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_207) {
  Conv2DTestCase({4,320,64,64}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_208) {
  Conv2DTestCase({4,320,136,120}, {320,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_209) {
  Conv2DTestCase({4,8,112,144}, {8,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_210) {
  Conv2DTestCase({4,640,68,60}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_211) {
  Conv2DTestCase({4,320,104,152}, {320,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_212) {
  Conv2DTestCase({4,320,72,56}, {640,3,3}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_213) {
  Conv2DTestCase({4,640,56,72}, {640,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_214) {
  Conv2DTestCase({4,640,76,52}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_215) {
  Conv2DTestCase({4,8,136,120}, {8,3,3}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_216) {
  Conv2DTestCase({4,640,72,56}, {640,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_217) {
  Conv2DTestCase({4,640,64,64}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_218) {
  Conv2DTestCase({4,4,152,104}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_219) {
  Conv2DTestCase({4,640,34,30}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_220) {
  Conv2DTestCase({4,320,112,144}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_221) {
  Conv2DTestCase({4,1280,38,26}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_222) {
  Conv2DTestCase({4,640,28,36}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_223) {
   Conv2DTestCase({4,320,152,104}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_224) {
   Conv2DTestCase({4,640,36,28}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_225) {
   Conv2DTestCase({4,320,68,60}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_226) {
   Conv2DTestCase({4,320,76,52}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_227) {
   Conv2DTestCase({4,640,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_228) {
   Conv2DTestCase({4,640,68,60}, {640,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_229) {
   Conv2DTestCase({4,320,144,112}, {320,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_230) {
   Conv2DTestCase({4,640,64,64}, {640,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_231) {
   Conv2DTestCase({4,640,38,26}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_232) {
   Conv2DTestCase({4,640,26,38}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_233) {
   Conv2DTestCase({4,640,28,36}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_234) {
   Conv2DTestCase({4,256,224,288}, {512,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_235) {
   Conv2DTestCase({4,128,608,416}, {256,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_236) {
   Conv2DTestCase({4,128,512,512}, {256,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_237) {
   Conv2DTestCase({4,8,152,104}, {8,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_238) {
   Conv2DTestCase({4,512,225,289}, {512,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_239) {
   Conv2DTestCase({4,256,513,513}, {256,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_240) {
   Conv2DTestCase({4,256,577,449}, {256,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_241) {
   Conv2DTestCase({4,128,416,608}, {256,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_242) {
   Conv2DTestCase({4,256,256,256}, {512,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_243) {
   Conv2DTestCase({4,8,144,112}, {8,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_244) {
   Conv2DTestCase({4,512,288,224}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_245) {
   Conv2DTestCase({4,8,128,128}, {8,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_246) {
   Conv2DTestCase({4,256,304,208}, {512,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_247) {
   Conv2DTestCase({4,8,104,152}, {8,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_248) {
   Conv2DTestCase({4,256,272,240}, {512,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_249) {
   Conv2DTestCase({4,320,72,56}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_250) {
   Conv2DTestCase({4,512,257,257}, {512,1,1}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_251) {
   Conv2DTestCase({4,512,273,241}, {512,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_252) {
   Conv2DTestCase({4,320,52,76}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_253) {
   Conv2DTestCase({4,320,56,72}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_254) {
   Conv2DTestCase({4,640,38,26}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_255) {
   Conv2DTestCase({4,320,68,60}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_case_256) {
   Conv2DTestCase({4,320,76,52}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_gencase_1) {
   Conv2DTestCase({2,8,1,100}, {2,1,12}, {0,0,3,4}, {1,63}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_gencase_5) {
   Conv2DTestCase({3,3,114,376}, {8,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_gencase_6) {
   Conv2DTestCase({4,6,234,216}, {9,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_gencase_7) {
   Conv2DTestCase({5,9,456,234}, {10,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_gencase_8) {
   Conv2DTestCase({6,12,246,342}, {11,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_gencase_9) {
   Conv2DTestCase({10,21,53,923}, {14,1,3}, {0,0,0,0}, {6,7}, {1,255}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_gencase_10) {
   Conv2DTestCase({11,24,57,821}, {15,1,2}, {0,0,0,0}, {8,9}, {1,255}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_gencase_11) {
   Conv2DTestCase({12,27,24,714}, {16,1,1}, {0,0,0,0}, {10,11}, {1,255}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_gencase_12) {
   Conv2DTestCase({13,30,42,523}, {17,1,2}, {0,0,0,0}, {12,13}, {1,255}, ge::DT_FLOAT16, 1);
}
// TEST_F(Tiling, run_conv2d_gencase_13) {
//    Conv2DTestCase({19,39,553,1}, {20,255,1}, {255,18,0,0}, {18,1}, {1,1}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_14) {
//    Conv2DTestCase({20,42,341,1}, {21,255,1}, {19,255,0,0}, {20,1}, {1,1}, ge::DT_FLOAT16, 1);
// }
TEST_F(Tiling, run_conv2d_gencase_15) {
   Conv2DTestCase({21,45,443,1}, {22,255,1}, {21,22,0,0}, {22,1}, {1,1}, ge::DT_FLOAT16, 1);
}
// TEST_F(Tiling, run_conv2d_gencase_16) {
//    Conv2DTestCase({22,48,323,1}, {23,255,1}, {255,24,0,0}, {24,1}, {1,1}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_20) {
//    Conv2DTestCase({28,66,9,12766}, {29,5,43}, {37,38,39,40}, {7,8}, {11,1}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_58) {
//    Conv2DTestCase({134217712,1,1,1}, {1,1,1}, {1,1,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_59) {
//    Conv2DTestCase({1,1,134217712,1}, {1,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_60) {
//    Conv2DTestCase({1,1,1,134217712}, {1,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_61) {
//    Conv2DTestCase({134217712,1,1,1}, {1,1,1}, {1,1,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_62) {
//    Conv2DTestCase({1,1,134217712,1}, {1,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_63) {
//    Conv2DTestCase({1,1,1,134217712}, {1,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_64) {
//    Conv2DTestCase({67108832,1,1,1}, {1,1,1}, {1,1,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_65) {
//    Conv2DTestCase({1,1,67108832,1}, {1,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_66) {
//    Conv2DTestCase({1,1,1,67108832}, {1,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_67) {
//    Conv2DTestCase({67108832,1,1,1}, {1,1,1}, {1,1,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_68) {
//    Conv2DTestCase({1,1,67108832,1}, {1,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_69) {
//    Conv2DTestCase({1,1,1,67108832}, {1,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
// }
TEST_F(Tiling, run_conv2d_gencase_70) {
   Conv2DTestCase({15,14,55,88}, {13,11,9}, {4,5,3,2}, {12,8}, {4,3}, ge::DT_FLOAT16, 1);
}
// TEST_F(Tiling, run_conv2d_gencase_71) {
//    Conv2DTestCase({14,13,99,94}, {30,10,7}, {6,7,8,9}, {8,7}, {5,7}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_72) {
//    Conv2DTestCase({13,12,65,57}, {60,5,2}, {1,0,11,12}, {5,3}, {2,6}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_73) {
//    Conv2DTestCase({16,8,44,75}, {87,3,7}, {5,11,12,15}, {6,4}, {3,2}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_74) {
//    Conv2DTestCase({12,11,76,65}, {246,8,6}, {3,2,7,6}, {7,6}, {1,4}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_75) {
//    Conv2DTestCase({11,10,58,69}, {889,4,3}, {2,3,5,7}, {4,9}, {6,3}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_76) {
//    Conv2DTestCase({10,9,32,45}, {5440,1,11}, {3,6,9,2}, {2,2}, {2,2}, ge::DT_FLOAT16, 1);
// }
TEST_F(Tiling, run_conv2d_gencase_77) {
   Conv2DTestCase({15,22,55,88}, {13,11,9}, {4,5,3,2}, {8,8}, {4,3}, ge::DT_FLOAT16, 1);
}
// TEST_F(Tiling, run_conv2d_gencase_78) {
//    Conv2DTestCase({16,61,43,76}, {11,6,7}, {5,11,12,15}, {6,4}, {3,2}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_79) {
//    Conv2DTestCase({11,125,46,77}, {13,3,5}, {5,11,8,15}, {6,4}, {3,2}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_80) {
//    Conv2DTestCase({8,222,68,79}, {15,4,3}, {2,3,5,7}, {4,9}, {6,3}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_81) {
//    Conv2DTestCase({7,889,78,59}, {6,4,3}, {2,3,5,7}, {2,4}, {5,4}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_82) {
//    Conv2DTestCase({5,4097,46,35}, {9,7,2}, {3,2,2,4}, {4,3}, {1,2}, ge::DT_FLOAT16, 1);
// }
TEST_F(Tiling, run_conv2d_gencase_83) {
   Conv2DTestCase({19,6,55,88}, {13,11,9}, {4,5,3,2}, {12,8}, {4,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_gencase_84) {
   Conv2DTestCase({45,3,54,86}, {12,7,5}, {3,4,1,2}, {3,8}, {2,3}, ge::DT_FLOAT16, 1);
}
// TEST_F(Tiling, run_conv2d_gencase_85) {
//    Conv2DTestCase({84,8,53,84}, {11,3,1}, {2,3,0,2}, {2,5}, {1,2}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_87) {
//    Conv2DTestCase({15,4,99,99}, {13,3,2}, {4,5,3,2}, {1,1}, {18,17}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_88) {
//    Conv2DTestCase({12,8,86,83}, {11,2,2}, {4,9,8,2}, {1,2}, {34,35}, ge::DT_FLOAT16, 1);
// }
TEST_F(Tiling, run_conv2d_gencase_94) {
   Conv2DTestCase({7,8,94,96}, {12,5,22}, {1,2,3,4}, {10,11}, {1,2}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_gencase_95) {
   Conv2DTestCase({8,9,95,97}, {13,32,6}, {1,2,3,4}, {12,13}, {3,2}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_gencase_106) {
   Conv2DTestCase({11,14,55,88}, {13,11,9}, {4,5,3,2}, {12,8}, {4,3}, ge::DT_FLOAT16, 1);
}
// TEST_F(Tiling, run_conv2d_gencase_107) {
//    Conv2DTestCase({12,13,99,94}, {30,10,7}, {6,7,8,9}, {8,7}, {5,7}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_108) {
//    Conv2DTestCase({13,12,65,57}, {60,5,2}, {1,0,11,12}, {5,3}, {2,6}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_109) {
//    Conv2DTestCase({14,8,44,75}, {87,3,7}, {5,11,12,15}, {6,4}, {3,2}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_110) {
//    Conv2DTestCase({6,11,76,65}, {246,8,6}, {3,2,7,6}, {7,6}, {1,4}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_111) {
//    Conv2DTestCase({7,10,58,69}, {889,4,3}, {2,3,5,7}, {4,9}, {6,3}, ge::DT_FLOAT16, 1);
// }
TEST_F(Tiling, run_conv2d_gencase_113) {
   Conv2DTestCase({9,22,55,88}, {13,11,9}, {4,5,3,2}, {8,8}, {4,3}, ge::DT_FLOAT16, 1);
}
// TEST_F(Tiling, run_conv2d_gencase_114) {
//    Conv2DTestCase({10,61,43,76}, {11,6,7}, {5,11,12,15}, {6,4}, {3,2}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_115) {
//    Conv2DTestCase({11,125,46,77}, {13,3,5}, {5,11,8,15}, {6,4}, {3,2}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_116) {
//    Conv2DTestCase({12,222,68,79}, {15,4,3}, {2,3,5,7}, {4,9}, {6,3}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_117) {
//    Conv2DTestCase({13,889,78,59}, {6,4,3}, {2,3,5,7}, {2,4}, {5,4}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_118) {
//    Conv2DTestCase({14,4097,46,35}, {9,7,2}, {3,2,2,4}, {4,3}, {1,2}, ge::DT_FLOAT16, 1);
// }
TEST_F(Tiling, run_conv2d_gencase_119) {
   Conv2DTestCase({24,6,55,88}, {13,11,9}, {4,5,3,2}, {12,8}, {4,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_gencase_120) {
   Conv2DTestCase({52,3,54,86}, {12,7,5}, {3,4,1,2}, {3,8}, {2,3}, ge::DT_FLOAT16, 1);
}
// TEST_F(Tiling, run_conv2d_gencase_121) {
//    Conv2DTestCase({72,8,53,84}, {11,3,1}, {2,3,0,2}, {2,5}, {1,2}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_123) {
//    Conv2DTestCase({15,4,99,99}, {13,3,2}, {4,5,3,2}, {1,1}, {18,17}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_124) {
//    Conv2DTestCase({12,8,86,83}, {11,2,2}, {4,9,8,2}, {1,2}, {34,35}, ge::DT_FLOAT16, 1);
// }
TEST_F(Tiling, run_conv2d_gencase_130) {
   Conv2DTestCase({12,12,86,96}, {12,5,22}, {1,2,3,4}, {10,11}, {1,2}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_gencase_142) {
   Conv2DTestCase({11,14,55,88}, {13,11,9}, {4,5,3,2}, {12,8}, {2,3}, ge::DT_FLOAT16, 1);
}
// TEST_F(Tiling, run_conv2d_gencase_143) {
//    Conv2DTestCase({12,13,99,94}, {30,10,7}, {6,7,8,9}, {8,7}, {5,6}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_144) {
//    Conv2DTestCase({13,12,65,57}, {60,5,2}, {1,0,11,12}, {5,3}, {2,5}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_145) {
//    Conv2DTestCase({14,8,44,75}, {87,3,7}, {5,11,12,15}, {6,4}, {3,1}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_146) {
//    Conv2DTestCase({6,11,76,65}, {246,8,6}, {3,2,7,6}, {7,6}, {1,3}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_147) {
//    Conv2DTestCase({7,10,57,69}, {889,4,3}, {2,3,5,7}, {4,9}, {5,3}, ge::DT_FLOAT16, 1);
// }
TEST_F(Tiling, run_conv2d_gencase_149) {
   Conv2DTestCase({9,22,59,88}, {13,11,9}, {4,5,3,2}, {7,8}, {3,3}, ge::DT_FLOAT16, 1);
}
// TEST_F(Tiling, run_conv2d_gencase_150) {
//    Conv2DTestCase({10,61,45,76}, {11,6,7}, {5,11,11,15}, {6,4}, {2,2}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_151) {
//    Conv2DTestCase({11,125,48,77}, {13,3,5}, {5,11,8,15}, {6,4}, {3,2}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_152) {
//    Conv2DTestCase({12,222,68,79}, {15,4,3}, {2,3,6,7}, {4,8}, {6,4}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_153) {
//    Conv2DTestCase({13,889,78,59}, {6,4,3}, {2,3,5,7}, {2,4}, {4,4}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_154) {
//    Conv2DTestCase({14,4097,46,35}, {9,7,2}, {3,2,2,4}, {4,3}, {1,1}, ge::DT_FLOAT16, 1);
// }
TEST_F(Tiling, run_conv2d_gencase_155) {
   Conv2DTestCase({24,6,55,88}, {13,11,9}, {4,5,3,2}, {12,8}, {3,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_gencase_156) {
   Conv2DTestCase({52,3,58,86}, {12,7,5}, {3,6,1,2}, {3,8}, {2,2}, ge::DT_FLOAT16, 1);
}
// TEST_F(Tiling, run_conv2d_gencase_157) {
//    Conv2DTestCase({72,8,59,84}, {11,3,1}, {2,3,0,2}, {2,5}, {1,2}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_159) {
//    Conv2DTestCase({15,4,99,99}, {13,3,2}, {4,5,3,2}, {1,1}, {18,17}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_160) {
//    Conv2DTestCase({12,8,86,88}, {11,2,2}, {4,8,8,2}, {1,2}, {34,35}, ge::DT_FLOAT16, 1);
// }
TEST_F(Tiling, run_conv2d_gencase_166) {
   Conv2DTestCase({12,12,86,96}, {12,5,22}, {1,2,3,4}, {10,9}, {1,2}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_gencase_178) {
   Conv2DTestCase({3,1,55,88}, {13,11,9}, {4,5,3,2}, {12,8}, {4,3}, ge::DT_FLOAT16, 1);
}
// TEST_F(Tiling, run_conv2d_gencase_179) {
//    Conv2DTestCase({4,2,99,94}, {30,10,7}, {6,7,8,9}, {8,7}, {5,7}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_180) {
//    Conv2DTestCase({5,3,65,57}, {60,5,2}, {1,0,11,12}, {5,3}, {2,6}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_181) {
//    Conv2DTestCase({6,4,44,75}, {87,3,7}, {5,11,12,15}, {6,4}, {3,2}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_182) {
//    Conv2DTestCase({7,5,76,65}, {246,8,6}, {3,2,7,6}, {7,6}, {1,4}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_183) {
//    Conv2DTestCase({8,6,58,69}, {889,4,3}, {2,3,5,7}, {4,9}, {6,3}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_184) {
//    Conv2DTestCase({9,7,32,45}, {5440,1,11}, {3,6,9,2}, {2,2}, {2,2}, ge::DT_FLOAT16, 1);
// }
TEST_F(Tiling, run_conv2d_gencase_185) {
   Conv2DTestCase({10,24,55,88}, {13,11,9}, {4,5,3,2}, {8,8}, {4,3}, ge::DT_FLOAT16, 1);
}
// TEST_F(Tiling, run_conv2d_gencase_186) {
//    Conv2DTestCase({11,65,43,76}, {11,6,7}, {5,11,12,15}, {6,4}, {3,2}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_187) {
//    Conv2DTestCase({12,128,49,77}, {13,3,5}, {5,11,8,15}, {6,4}, {3,2}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_188) {
//    Conv2DTestCase({13,220,68,79}, {15,4,3}, {2,3,5,7}, {4,9}, {6,3}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_189) {
//    Conv2DTestCase({14,899,78,59}, {6,4,3}, {2,3,5,7}, {2,4}, {5,4}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_190) {
//    Conv2DTestCase({15,4097,56,35}, {9,7,2}, {3,2,2,4}, {4,3}, {1,2}, ge::DT_FLOAT16, 1);
// }
TEST_F(Tiling, run_conv2d_gencase_191) {
   Conv2DTestCase({19,7,55,88}, {13,11,9}, {4,5,3,2}, {12,8}, {4,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_gencase_192) {
   Conv2DTestCase({45,6,58,86}, {12,7,5}, {3,4,1,2}, {3,8}, {2,3}, ge::DT_FLOAT16, 1);
}
// TEST_F(Tiling, run_conv2d_gencase_193) {
//    Conv2DTestCase({81,9,53,84}, {11,3,1}, {2,3,0,2}, {2,5}, {1,2}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_195) {
//    Conv2DTestCase({15,5,99,99}, {13,3,2}, {4,5,3,2}, {1,1}, {18,17}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_196) {
//    Conv2DTestCase({12,7,86,83}, {11,2,2}, {4,9,8,2}, {1,2}, {34,35}, ge::DT_FLOAT16, 1);
// }
TEST_F(Tiling, run_conv2d_gencase_202) {
   Conv2DTestCase({7,12,96,96}, {12,5,22}, {1,2,3,4}, {10,11}, {1,2}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_gencase_203) {
   Conv2DTestCase({8,11,97,97}, {13,32,6}, {1,2,3,4}, {12,13}, {3,2}, ge::DT_FLOAT16, 1);
}
// TEST_F(Tiling, run_conv2d_gencase_214) {
//    Conv2DTestCase({2,512,18,18}, {2048,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 0);
// }
TEST_F(Tiling, run_conv2d_gencase_216) {
   Conv2DTestCase({1,64,30,30}, {1,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_gencase_217) {
   Conv2DTestCase({1,15,15,16}, {1,2,3}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_gencase_219) {
   Conv2DTestCase({1,3,4,4}, {1000,2,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_gencase_220) {
   Conv2DTestCase({8,768,16,16}, {768,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 0);
}
TEST_F(Tiling, run_conv2d_gencase_221) {
   Conv2DTestCase({4,291,480,480}, {291,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}
TEST_F(Tiling, run_conv2d_gencase_222) {
   Conv2DTestCase({32,512,105,105}, {1024,1,1}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 0);
}
// TEST_F(Tiling, run_conv2d_gencase_226) {
//    Conv2DTestCase({134217712,1,1,1}, {1,1,1}, {1,1,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_227) {
//    Conv2DTestCase({1,1,134217712,1}, {1,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
// }
// TEST_F(Tiling, run_conv2d_gencase_228) {
//    Conv2DTestCase({1,1,1,134217712}, {1,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
// }
//  TEST_F(Tiling, run_quantConv2d_gencase_3) {
//    Conv2DTestCase({7,15,723,23}, {12,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
//  }
// TEST_F(Tiling, run_quantConv2d_gencase_4) {
//    Conv2DTestCase({8,18,523,98}, {13,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_5) {
//    Conv2DTestCase({14,33,34,442}, {18,1,1}, {0,0,0,0}, {14,15}, {1,255}, ge::DT_INT8, 1, 1);
// }
// // TEST_F(Tiling, run_quantConv2d_gencase_6) {
// //    Conv2DTestCase({17,36,43,631}, {19,1,3}, {0,0,0,0}, {16,17}, {1,255}, ge::DT_INT8, 1, 1);
// // }
// // TEST_F(Tiling, run_quantConv2d_gencase_13) {
// //    Conv2DTestCase({512,512,1,1}, {64,1,1}, {1,0,1,0}, {1,1}, {1,1}, ge::DT_INT8, 0, 1);
// // }
// // TEST_F(Tiling, run_quantConv2d_gencase_15) {
// //    Conv2DTestCase({558,1,1,108}, {256,1,57}, {1,0,27,19}, {1,46}, {1,1}, ge::DT_INT8, 1, 1);
// // }
// TEST_F(Tiling, run_quantConv2d_gencase_16) {
//    Conv2DTestCase({32,64,2,13}, {256,27,20}, {26,0,14,0}, {1,1}, {1,1}, ge::DT_INT8, 0, 1);
// }
// // TEST_F(Tiling, run_quantConv2d_gencase_17) {
// //    Conv2DTestCase({128,256,1,2}, {196,1,2}, {1,0,1,0}, {1,1}, {1,1}, ge::DT_INT8, 0, 1);
// // }
// TEST_F(Tiling, run_quantConv2d_gencase_18) {
//    Conv2DTestCase({32,256,2,2}, {512,210,1}, {209,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// // TEST_F(Tiling, run_quantConv2d_gencase_19) {
// //    Conv2DTestCase({256,74,1,2}, {65,1,2}, {1,0,1,0}, {1,1}, {1,1}, ge::DT_INT8, 0, 1);
// // }
// // TEST_F(Tiling, run_quantConv2d_gencase_20) {
// //    Conv2DTestCase({133,32,3,5}, {166,2,1}, {0,2,0,1}, {3,5}, {1,1}, ge::DT_INT8, 1, 1);
// // }
// // TEST_F(Tiling, run_quantConv2d_gencase_21) {
// //    Conv2DTestCase({156,128,3,1}, {1,3,1}, {2,1,1,0}, {2,1}, {1,1}, ge::DT_INT8, 0, 1);
// // }
// // TEST_F(Tiling, run_quantConv2d_gencase_22) {
// //    Conv2DTestCase({133,64,1,5}, {256,1,2}, {0,1,1,1}, {1,2}, {1,2}, ge::DT_INT8, 0, 1);
// // }
// TEST_F(Tiling, run_quantConv2d_gencase_23) {
//    Conv2DTestCase({1212,1,1,49}, {184,1,7}, {0,0,1,6}, {1,31}, {1,1}, ge::DT_INT8, 0, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_24) {
//    Conv2DTestCase({129,1,4,127}, {9,4,84}, {0,3,39,21}, {4,35}, {1,1}, ge::DT_INT8, 0, 1);
// }
// // TEST_F(Tiling, run_quantConv2d_gencase_25) {
// //    Conv2DTestCase({169,357,1,1}, {256,1,1}, {0,1,1,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// // }
// // TEST_F(Tiling, run_quantConv2d_gencase_44) {
// //    Conv2DTestCase({67108832,1,1,1}, {1,1,1}, {1,1,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// // }
// // TEST_F(Tiling, run_quantConv2d_gencase_45) {
// //    Conv2DTestCase({1,1,67108832,1}, {1,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// // }
// // TEST_F(Tiling, run_quantConv2d_gencase_46) {
// //    Conv2DTestCase({1,1,1,67108832}, {1,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// // }
// TEST_F(Tiling, run_quantConv2d_gencase_47) {
//    Conv2DTestCase({3,14,55,88}, {13,11,9}, {4,5,3,2}, {12,8}, {4,3}, ge::DT_INT8, 1, 1);
// }
// // TEST_F(Tiling, run_quantConv2d_gencase_48) {
// //    Conv2DTestCase({4,13,99,94}, {30,10,7}, {6,7,8,9}, {8,7}, {5,7}, ge::DT_INT8, 1, 1);
// // }
// // TEST_F(Tiling, run_quantConv2d_gencase_49) {
// //    Conv2DTestCase({7,12,66,57}, {60,5,2}, {1,0,11,12}, {5,3}, {2,6}, ge::DT_INT8, 1, 1);
// // }
// // TEST_F(Tiling, run_quantConv2d_gencase_50) {
// //    Conv2DTestCase({16,8,45,75}, {87,3,7}, {5,11,12,15}, {6,4}, {3,2}, ge::DT_INT8, 1, 1);
// // }
// // TEST_F(Tiling, run_quantConv2d_gencase_51) {
// //    Conv2DTestCase({9,11,77,65}, {246,8,6}, {3,2,7,6}, {7,6}, {1,4}, ge::DT_INT8, 1, 1);
// // }
// // TEST_F(Tiling, run_quantConv2d_gencase_52) {
// //    Conv2DTestCase({3,10,59,69}, {889,4,3}, {2,3,5,7}, {4,9}, {6,3}, ge::DT_INT8, 1, 1);
// // }
// TEST_F(Tiling, run_quantConv2d_gencase_54) {
//    Conv2DTestCase({7,22,56,88}, {13,11,9}, {4,5,3,2}, {8,8}, {4,3}, ge::DT_INT8, 1, 1);
// }
// // TEST_F(Tiling, run_quantConv2d_gencase_55) {
// //    Conv2DTestCase({8,61,44,76}, {11,6,7}, {5,11,12,15}, {6,4}, {3,2}, ge::DT_INT8, 1, 1);
// // }
// // TEST_F(Tiling, run_quantConv2d_gencase_56) {
// //    Conv2DTestCase({11,125,47,77}, {13,3,5}, {5,11,8,15}, {6,4}, {3,2}, ge::DT_INT8, 1, 1);
// // }
// // TEST_F(Tiling, run_quantConv2d_gencase_57) {
// //    Conv2DTestCase({16,222,69,79}, {15,4,3}, {2,3,5,7}, {4,9}, {6,3}, ge::DT_INT8, 1, 1);
// // }
// // TEST_F(Tiling, run_quantConv2d_gencase_58) {
// //    Conv2DTestCase({14,889,79,59}, {6,4,3}, {2,3,5,7}, {2,4}, {5,4}, ge::DT_INT8, 1, 1);
// // }
// // TEST_F(Tiling, run_quantConv2d_gencase_59) {
// //    Conv2DTestCase({15,4097,47,35}, {9,7,2}, {3,2,2,4}, {4,3}, {1,2}, ge::DT_INT8, 1, 1);
// // }
// TEST_F(Tiling, run_quantConv2d_gencase_60) {
//    Conv2DTestCase({27,6,56,88}, {13,11,9}, {4,5,3,2}, {12,8}, {4,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_61) {
//    Conv2DTestCase({51,3,55,86}, {12,7,5}, {3,4,1,2}, {3,8}, {2,3}, ge::DT_INT8, 1, 1);
// }
// // TEST_F(Tiling, run_quantConv2d_gencase_64) {
// //    Conv2DTestCase({15,4,98,99}, {13,3,2}, {4,5,3,2}, {1,1}, {18,17}, ge::DT_INT8, 1, 1);
// // }
// // TEST_F(Tiling, run_quantConv2d_gencase_65) {
// //    Conv2DTestCase({12,8,87,83}, {11,2,2}, {4,9,8,2}, {1,2}, {34,35}, ge::DT_INT8, 1, 1);
// // }
// TEST_F(Tiling, run_quantConv2d_gencase_72) {
//    Conv2DTestCase({7,8,83,96}, {12,5,22}, {1,2,3,4}, {10,11}, {1,2}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_89) {
//    Conv2DTestCase({4,3,896,1152}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_90) {
//    Conv2DTestCase({4,3,1152,896}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_91) {
//    Conv2DTestCase({4,3,832,1216}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_92) {
//    Conv2DTestCase({4,3,1216,832}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_93) {
//    Conv2DTestCase({4,128,896,1152}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_94) {
//    Conv2DTestCase({4,3,1024,1024}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_95) {
//    Conv2DTestCase({4,3,1088,960}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_96) {
//    Conv2DTestCase({4,128,832,1216}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_97) {
//    Conv2DTestCase({4,128,1152,896}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_98) {
//    Conv2DTestCase({4,128,1216,832}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_99) {
//    Conv2DTestCase({4,128,1024,1024}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_100) {
//    Conv2DTestCase({4,128,1088,960}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_101) {
//    Conv2DTestCase({4,128,897,1153}, {128,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_102) {
//    Conv2DTestCase({4,128,448,576}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_103) {
//    Conv2DTestCase({4,128,833,1217}, {128,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_104) {
//    Conv2DTestCase({4,128,416,608}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_105) {
//    Conv2DTestCase({4,256,448,576}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_106) {
//    Conv2DTestCase({4,128,1153,897}, {128,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_107) {
//    Conv2DTestCase({4,128,448,576}, {256,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_108) {
//    Conv2DTestCase({4,256,416,608}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_109) {
//    Conv2DTestCase({4,128,576,448}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_110) {
//    Conv2DTestCase({4,128,1217,833}, {128,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_111) {
//    Conv2DTestCase({4,128,416,608}, {256,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_112) {
//    Conv2DTestCase({4,128,1025,1025}, {128,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_113) {
//    Conv2DTestCase({4,128,608,416}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_114) {
//    Conv2DTestCase({4,128,512,512}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_115) {
//    Conv2DTestCase({4,256,576,448}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_116) {
//    Conv2DTestCase({4,256,608,416}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_117) {
//    Conv2DTestCase({4,128,576,448}, {256,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_118) {
//    Conv2DTestCase({4,128,1089,961}, {128,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_119) {
//    Conv2DTestCase({4,128,608,416}, {256,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_120) {
//    Conv2DTestCase({4,256,512,512}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_121) {
//    Conv2DTestCase({4,256,417,609}, {256,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_122) {
//    Conv2DTestCase({4,128,544,480}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_123) {
//    Conv2DTestCase({4,128,512,512}, {256,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_124) {
//    Conv2DTestCase({4,256,208,304}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_125) {
//    Conv2DTestCase({4,256,449,577}, {256,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_126) {
//    Conv2DTestCase({4,256,544,480}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_127) {
//    Conv2DTestCase({4,256,224,288}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_128) {
//    Conv2DTestCase({4,512,208,304}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_129) {
//    Conv2DTestCase({4,128,544,480}, {256,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_130) {
//    Conv2DTestCase({4,256,208,304}, {512,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_131) {
//    Conv2DTestCase({4,512,224,288}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_132) {
//    Conv2DTestCase({4,256,577,449}, {256,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_133) {
//    Conv2DTestCase({4,256,224,288}, {512,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_134) {
//    Conv2DTestCase({4,256,288,224}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_135) {
//    Conv2DTestCase({4,256,609,417}, {256,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_136) {
//    Conv2DTestCase({4,512,288,224}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_137) {
//    Conv2DTestCase({4,256,304,208}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_138) {
//    Conv2DTestCase({4,512,209,305}, {512,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_139) {
//    Conv2DTestCase({4,256,288,224}, {512,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_140) {
//    Conv2DTestCase({4,512,104,152}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_141) {
//    Conv2DTestCase({4,512,225,289}, {512,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_142) {
//    Conv2DTestCase({4,512,304,208}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_143) {
//    Conv2DTestCase({4,512,112,144}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_144) {
//    Conv2DTestCase({4,256,513,513}, {256,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_145) {
//    Conv2DTestCase({4,256,304,208}, {512,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_146) {
//    Conv2DTestCase({4,256,256,256}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_147) {
//    Conv2DTestCase({4,512,104,152}, {8,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_148) {
//    Conv2DTestCase({4,256,545,481}, {256,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_149) {
//    Conv2DTestCase({4,8,104,152}, {8,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_150) {
//    Conv2DTestCase({4,256,272,240}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_151) {
//    Conv2DTestCase({4,512,289,225}, {512,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_152) {
//    Conv2DTestCase({4,512,112,144}, {8,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_153) {
//    Conv2DTestCase({4,512,256,256}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_154) {
//    Conv2DTestCase({4,256,256,256}, {512,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_155) {
//    Conv2DTestCase({4,8,112,144}, {8,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_156) {
//    Conv2DTestCase({4,512,144,112}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_157) {
//    Conv2DTestCase({4,512,272,240}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_158) {
//    Conv2DTestCase({4,256,272,240}, {512,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_159) {
//    Conv2DTestCase({4,512,305,209}, {512,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_160) {
//    Conv2DTestCase({4,512,152,104}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_161) {
//    Conv2DTestCase({4,512,257,257}, {512,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_162) {
//    Conv2DTestCase({4,512,144,112}, {8,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_163) {
//    Conv2DTestCase({4,512,128,128}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_164) {
//    Conv2DTestCase({4,8,144,112}, {8,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_165) {
//    Conv2DTestCase({4,512,273,241}, {512,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_166) {
//    Conv2DTestCase({4,512,136,120}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_167) {
//    Conv2DTestCase({4,512,152,104}, {8,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_168) {
//    Conv2DTestCase({4,8,152,104}, {8,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_169) {
//    Conv2DTestCase({4,512,128,128}, {8,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_170) {
//    Conv2DTestCase({4,8,128,128}, {8,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_171) {
//    Conv2DTestCase({4,512,136,120}, {8,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_172) {
//    Conv2DTestCase({4,8,136,120}, {8,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_173) {
//    Conv2DTestCase({4,4,136,120}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_174) {
//    Conv2DTestCase({4,4,112,144}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_175) {
//    Conv2DTestCase({4,4,152,104}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_176) {
//    Conv2DTestCase({4,4,128,128}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_177) {
//    Conv2DTestCase({4,4,144,112}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_178) {
//    Conv2DTestCase({4,4,104,152}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_179) {
//    Conv2DTestCase({4,320,136,120}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_180) {
//    Conv2DTestCase({4,320,112,144}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_181) {
//    Conv2DTestCase({4,320,152,104}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_182) {
//    Conv2DTestCase({4,320,128,128}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_183) {
//    Conv2DTestCase({4,320,144,112}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_184) {
//    Conv2DTestCase({4,320,104,152}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_185) {
//    Conv2DTestCase({4,320,112,144}, {320,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_186) {
//    Conv2DTestCase({4,320,136,120}, {320,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_187) {
//    Conv2DTestCase({4,320,152,104}, {320,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_188) {
//    Conv2DTestCase({4,320,144,112}, {320,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_189) {
//    Conv2DTestCase({4,320,128,128}, {320,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_190) {
//    Conv2DTestCase({4,320,104,152}, {320,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_191) {
//    Conv2DTestCase({4,320,56,72}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_192) {
//    Conv2DTestCase({4,320,68,60}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_193) {
//    Conv2DTestCase({4,320,76,52}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_194) {
//    Conv2DTestCase({4,320,72,56}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_195) {
//    Conv2DTestCase({4,320,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_196) {
//    Conv2DTestCase({4,320,64,64}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_197) {
//    Conv2DTestCase({4,640,56,72}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_198) {
//    Conv2DTestCase({4,640,68,60}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_199) {
//    Conv2DTestCase({4,320,56,72}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_200) {
//    Conv2DTestCase({4,320,68,60}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_201) {
//    Conv2DTestCase({4,640,76,52}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_202) {
//    Conv2DTestCase({4,320,76,52}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_203) {
//    Conv2DTestCase({4,640,72,56}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_204) {
//    Conv2DTestCase({4,320,72,56}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_205) {
//    Conv2DTestCase({4,640,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_206) {
//    Conv2DTestCase({4,320,52,76}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_207) {
//    Conv2DTestCase({4,640,64,64}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_208) {
//    Conv2DTestCase({4,320,64,64}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_209) {
//    Conv2DTestCase({4,640,56,72}, {640,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_210) {
//    Conv2DTestCase({4,640,76,52}, {640,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_211) {
//    Conv2DTestCase({4,640,68,60}, {640,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_212) {
//    Conv2DTestCase({4,640,64,64}, {640,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_213) {
//    Conv2DTestCase({4,640,72,56}, {640,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_214) {
//    Conv2DTestCase({4,640,52,76}, {640,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_215) {
//    Conv2DTestCase({4,640,28,36}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_216) {
//    Conv2DTestCase({4,640,38,26}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_217) {
//    Conv2DTestCase({4,640,34,30}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_218) {
//    Conv2DTestCase({4,640,32,32}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_219) {
//    Conv2DTestCase({4,640,36,28}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_220) {
//    Conv2DTestCase({4,640,26,38}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_221) {
//    Conv2DTestCase({4,1280,38,26}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_222) {
//    Conv2DTestCase({4,1280,28,36}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_223) {
//    Conv2DTestCase({4,640,38,26}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_224) {
//    Conv2DTestCase({4,640,28,36}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_225) {
//    Conv2DTestCase({2,640,26,38}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_226) {
//    Conv2DTestCase({2,1280,26,38}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_227) {
//    Conv2DTestCase({2,1920,36,28}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_228) {
//    Conv2DTestCase({2,1280,36,28}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_229) {
//    Conv2DTestCase({2,2560,28,36}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_230) {
//    Conv2DTestCase({2,1280,28,36}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_231) {
//    Conv2DTestCase({2,960,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_232) {
//    Conv2DTestCase({2,640,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_233) {
//    Conv2DTestCase({2,1280,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_234) {
//    Conv2DTestCase({2,1920,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_235) {
//    Conv2DTestCase({2,640,28,36}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_236) {
//    Conv2DTestCase({2,640,144,112}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_237) {
//    Conv2DTestCase({2,320,144,112}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_238) {
//    Conv2DTestCase({2,640,144,112}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_239) {
//    Conv2DTestCase({2,320,144,112}, {4,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_240) {
//    Conv2DTestCase({2,640,76,52}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_241) {
//    Conv2DTestCase({2,320,76,52}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_242) {
//    Conv2DTestCase({2,320,152,104}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_243) {
//    Conv2DTestCase({2,960,72,56}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_244) {
//    Conv2DTestCase({2,640,72,56}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_245) {
//    Conv2DTestCase({2,1280,72,56}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_246) {
//    Conv2DTestCase({2,1920,72,56}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_247) {
//    Conv2DTestCase({2,1280,38,26}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_248) {
//    Conv2DTestCase({2,640,38,26}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_249) {
//    Conv2DTestCase({2,2560,26,38}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_250) {
//    Conv2DTestCase({2,2560,36,28}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_251) {
//    Conv2DTestCase({2,640,20,48}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_252) {
//    Conv2DTestCase({2,1280,20,48}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_253) {
//    Conv2DTestCase({2,2560,38,26}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_254) {
//    Conv2DTestCase({2,640,104,152}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_255) {
//    Conv2DTestCase({2,320,104,152}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_256) {
//    Conv2DTestCase({2,640,104,152}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_257) {
//    Conv2DTestCase({2,320,104,152}, {4,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_258) {
//    Conv2DTestCase({2,2560,36,28}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_259) {
//    Conv2DTestCase({2,1920,36,28}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_260) {
//    Conv2DTestCase({2,1280,72,56}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_261) {
//    Conv2DTestCase({2,1920,72,56}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_262) {
//    Conv2DTestCase({2,1280,72,56}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_263) {
//    Conv2DTestCase({2,960,72,56}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_264) {
//    Conv2DTestCase({2,640,144,112}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_265) {
//    Conv2DTestCase({2,960,144,112}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_266) {
//    Conv2DTestCase({2,960,144,112}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_267) {
//    Conv2DTestCase({2,2560,26,38}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_268) {
//    Conv2DTestCase({2,1920,26,38}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_269) {
//    Conv2DTestCase({2,1920,26,38}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_270) {
//    Conv2DTestCase({2,1280,52,76}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_271) {
//    Conv2DTestCase({2,1920,52,76}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_272) {
//    Conv2DTestCase({2,1280,52,76}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_273) {
//    Conv2DTestCase({2,960,52,76}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_274) {
//    Conv2DTestCase({2,640,104,152}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_275) {
//    Conv2DTestCase({2,960,104,152}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_276) {
//    Conv2DTestCase({2,960,104,152}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_277) {
//    Conv2DTestCase({2,640,80,192}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_278) {
//    Conv2DTestCase({2,320,80,192}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_279) {
//    Conv2DTestCase({2,960,80,192}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_280) {
//    Conv2DTestCase({2,1920,38,26}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_281) {
//    Conv2DTestCase({2,960,112,144}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_282) {
//    Conv2DTestCase({2,320,112,144}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_283) {
//    Conv2DTestCase({2,960,112,144}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_284) {
//    Conv2DTestCase({2,640,112,144}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_285) {
//    Conv2DTestCase({2,640,112,144}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_286) {
//    Conv2DTestCase({2,320,112,144}, {4,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_287) {
//    Conv2DTestCase({2,960,76,52}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_288) {
//    Conv2DTestCase({2,960,76,52}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_289) {
//    Conv2DTestCase({2,640,152,104}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_290) {
//    Conv2DTestCase({2,960,152,104}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_291) {
//    Conv2DTestCase({2,960,152,104}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_292) {
//    Conv2DTestCase({2,640,152,104}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_293) {
//    Conv2DTestCase({2,640,152,104}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_294) {
//    Conv2DTestCase({2,320,152,104}, {4,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_295) {
//    Conv2DTestCase({2,1920,28,36}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_296) {
//    Conv2DTestCase({2,640,80,192}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_297) {
//    Conv2DTestCase({2,320,80,192}, {4,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_298) {
//    Conv2DTestCase({2,640,40,96}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_299) {
//    Conv2DTestCase({2,320,40,96}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_300) {
//    Conv2DTestCase({2,320,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_301) {
//    Conv2DTestCase({2,640,56,72}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_302) {
//    Conv2DTestCase({2,320,56,72}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_303) {
//    Conv2DTestCase({2,960,56,72}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_304) {
//    Conv2DTestCase({2,1280,56,72}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_305) {
//    Conv2DTestCase({2,1920,56,72}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_306) {
//    Conv2DTestCase({2,1920,20,48}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_307) {
//    Conv2DTestCase({2,2560,20,48}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_308) {
//    Conv2DTestCase({2,2560,38,26}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_309) {
//    Conv2DTestCase({2,1920,38,26}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_310) {
//    Conv2DTestCase({2,1280,76,52}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_311) {
//    Conv2DTestCase({2,1920,76,52}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_312) {
//    Conv2DTestCase({2,1920,76,52}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_313) {
//    Conv2DTestCase({2,1280,76,52}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_314) {
//    Conv2DTestCase({2,1280,76,52}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_315) {
//    Conv2DTestCase({2,2560,28,36}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_316) {
//    Conv2DTestCase({2,1920,28,36}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_317) {
//    Conv2DTestCase({2,1280,56,72}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_318) {
//    Conv2DTestCase({2,1920,56,72}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_319) {
//    Conv2DTestCase({2,1280,56,72}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_320) {
//    Conv2DTestCase({2,960,56,72}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_321) {
//    Conv2DTestCase({2,640,112,144}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_322) {
//    Conv2DTestCase({2,640,36,28}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_323) {
//    Conv2DTestCase({2,960,40,96}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_324) {
//    Conv2DTestCase({2,1280,40,96}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_325) {
//    Conv2DTestCase({2,1920,40,96}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_326) {
//    Conv2DTestCase({2,320,72,56}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_327) {
//    Conv2DTestCase({2,2560,20,48}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_328) {
//    Conv2DTestCase({2,1920,20,48}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_329) {
//    Conv2DTestCase({2,1280,40,96}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_330) {
//    Conv2DTestCase({2,1920,40,96}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_331) {
//    Conv2DTestCase({2,1280,40,96}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_332) {
//    Conv2DTestCase({2,960,40,96}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_333) {
//    Conv2DTestCase({2,640,80,192}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_334) {
//    Conv2DTestCase({2,960,80,192}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_335) {
//    Conv2DTestCase({1,16,128,128}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_336) {
//    Conv2DTestCase({1,1280,32,32}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_337) {
//    Conv2DTestCase({1,640,32,32}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_338) {
//    Conv2DTestCase({1,2560,32,32}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_339) {
//    Conv2DTestCase({1,2560,32,32}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_340) {
//    Conv2DTestCase({1,1920,32,32}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_341) {
//    Conv2DTestCase({1,1920,32,32}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_342) {
//    Conv2DTestCase({1,1280,64,64}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_343) {
//    Conv2DTestCase({1,1920,64,64}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_344) {
//    Conv2DTestCase({1,1920,64,64}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_345) {
//    Conv2DTestCase({1,1280,64,64}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_346) {
//    Conv2DTestCase({1,1280,64,64}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_347) {
//    Conv2DTestCase({1,960,64,64}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_348) {
//    Conv2DTestCase({1,960,64,64}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_349) {
//    Conv2DTestCase({1,640,128,128}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_350) {
//    Conv2DTestCase({1,960,128,128}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_351) {
//    Conv2DTestCase({1,320,128,128}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_352) {
//    Conv2DTestCase({1,960,128,128}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_353) {
//    Conv2DTestCase({1,640,128,128}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_354) {
//    Conv2DTestCase({1,640,128,128}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_355) {
//    Conv2DTestCase({1,320,128,128}, {4,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_356) {
//    Conv2DTestCase({1,16,128,128}, {384,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_357) {
//    Conv2DTestCase({1,384,128,128}, {384,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_358) {
//    Conv2DTestCase({1,384,128,128}, {384,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_359) {
//    Conv2DTestCase({1,384,64,64}, {768,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_360) {
//    Conv2DTestCase({1,768,64,64}, {768,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_361) {
//    Conv2DTestCase({1,384,64,64}, {768,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_362) {
//    Conv2DTestCase({1,768,64,64}, {768,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_363) {
//    Conv2DTestCase({1,768,32,32}, {1536,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_364) {
//    Conv2DTestCase({1,1536,32,32}, {1536,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_365) {
//    Conv2DTestCase({1,768,32,32}, {1536,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_366) {
//    Conv2DTestCase({1,1536,32,32}, {1536,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_367) {
//    Conv2DTestCase({1,1536,16,16}, {1536,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_368) {
//    Conv2DTestCase({1,3072,16,16}, {1536,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_369) {
//    Conv2DTestCase({1,3072,16,16}, {1536,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_370) {
//    Conv2DTestCase({1,3072,32,32}, {1536,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_371) {
//    Conv2DTestCase({1,3072,32,32}, {1536,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_372) {
//    Conv2DTestCase({1,2304,32,32}, {1536,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_373) {
//    Conv2DTestCase({1,2304,32,32}, {1536,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_374) {
//    Conv2DTestCase({1,2304,64,64}, {768,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_375) {
//    Conv2DTestCase({1,2304,64,64}, {768,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_376) {
//    Conv2DTestCase({1,1536,64,64}, {768,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_377) {
//    Conv2DTestCase({1,1536,64,64}, {768,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_378) {
//    Conv2DTestCase({1,1152,64,64}, {768,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_379) {
//    Conv2DTestCase({1,1152,64,64}, {768,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_380) {
//    Conv2DTestCase({1,768,128,128}, {768,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_381) {
//    Conv2DTestCase({1,1152,128,128}, {384,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_382) {
//    Conv2DTestCase({1,1152,128,128}, {384,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_383) {
//    Conv2DTestCase({1,768,128,128}, {384,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_384) {
//    Conv2DTestCase({1,768,128,128}, {384,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_385) {
//    Conv2DTestCase({1,384,128,128}, {4,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_386) {
//    Conv2DTestCase({1,640,64,64}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_387) {
//    Conv2DTestCase({1,4,128,128}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_388) {
//    Conv2DTestCase({1,4,128,128}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_389) {
//    Conv2DTestCase({1,4,128,128}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_390) {
//    Conv2DTestCase({1,4,128,128}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_391) {
//    Conv2DTestCase({1,4,128,128}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_392) {
//    Conv2DTestCase({2,640,26,38}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_393) {
//    Conv2DTestCase({2,1280,52,76}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_394) {
//    Conv2DTestCase({2,640,20,48}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_395) {
//    Conv2DTestCase({1,3072,16,16}, {1536,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_396) {
//    Conv2DTestCase({1,1152,128,128}, {384,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }
// TEST_F(Tiling, run_quantConv2d_gencase_397) {
//    Conv2DTestCase({1,1280,32,32}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }

// //TEST_F(Tiling, run_conv_bound_N_b16) {
// //  Conv2DTestCase({134217712,1280,36,28}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
// //}
// //TEST_F(Tiling, run_conv_bound_N_b8) {
// //  Conv2DTestCase({67108832,1280,32,32}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// //}
// //TEST_F(Tiling, run_conv_bound_N_b32) {
// //  Conv2DTestCase({268435448,1280,36,28}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT, 1);
// //}
// //TEST_F(Tiling, run_conv_bound_CI_b16) {
// //  Conv2DTestCase({2,8388592,36,28}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
// //}
// //TEST_F(Tiling, run_conv_bound_CI_b8) {
// //  Conv2DTestCase({1,33554424,32,32}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// //}
// //TEST_F(Tiling, run_conv_bound_CI_b32) {
// //  Conv2DTestCase({2,134217696,36,28}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT, 1);
// //}
// //TEST_F(Tiling, run_conv_bound_H_b16) {
// //  Conv2DTestCase({2,1280,134217712,28}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
// //}
// //TEST_F(Tiling, run_conv_bound_H_b8) {
// //  Conv2DTestCase({1,1280,67108832,32}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// //}
// //TEST_F(Tiling, run_conv_bound_W_b32) {
// //  Conv2DTestCase({1,1280,32,67108832}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT, 1);
// //}
// //TEST_F(Tiling, run_conv_bound_CO_b16) {
// //  Conv2DTestCase({2,1280,36,28}, {134217712,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
// //}
// //TEST_F(Tiling, run_conv_bound_CO_b8) {
// //  Conv2DTestCase({1,1280,32,32}, {67108832,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// //}
// //TEST_F(Tiling, run_conv_bound_CO_b32) {
// //  Conv2DTestCase({2,1280,36,28}, {268435448,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT, 1);
// //}

TEST_F(Tiling, run_conv2d_MorHWsplit_case1) {
   Conv2DTestCase({16,16,448,576}, {16,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT, 1);
}

TEST_F(Tiling, run_conv2d_MorHWsplit_case2) {
   Conv2DTestCase({16,16,4480,5760}, {16,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT, 1);
}

TEST_F(Tiling, run_conv2d_MorHWsplit_case3) {
   Conv2DTestCase({16,16,1089,961}, {16,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_MorHWsplit_case4) {
   Conv2DTestCase({16,16,1089,961}, {16,30,30}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_MorHWsplit_case5) {
   Conv2DTestCase({16,16,1089,961}, {16,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_BF16, 1);
}

TEST_F(Tiling, run_conv2d_MorHWsplit_case6) {
   Conv2DTestCase({16,16,1089,961}, {16,30,30}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_BF16, 1);
}

TEST_F(Tiling, run_conv2d_case_fp32) {
   Conv2DTestCase({1,1,1,16}, {16,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT, 1, 0, false);
}
TEST_F(Tiling, run_conv2d_case_hf32) {
   Conv2DTestCase({1,1,1,16}, {16,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT, 1, 0, true);
}

TEST_F(Tiling, run_conv2d_group_case1) {
   Conv2DTestCase({1,12,1,1}, {2,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 0, 0, false, 2);
}

TEST_F(Tiling, run_conv2d_group_case2) {
   Conv2DTestCase({2,15,10,10}, {6,2,2}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 3);
}

TEST_F(Tiling, run_conv2d_group_case3) {
   Conv2DTestCase({1,16,11,10}, {12,3,4}, {0,0,1,1}, {1,2}, {1,1}, ge::DT_FLOAT16, 0, 0, false, 4);
}

TEST_F(Tiling, run_conv2d_group_case4) {
   Conv2DTestCase({2,15,10,11}, {20,4,3}, {1,1,0,0}, {1,1}, {1,2}, ge::DT_FLOAT16, 1, 0, false, 5);
}

TEST_F(Tiling, run_conv2d_group_case5) {
   Conv2DTestCase({1,12,11,11}, {30,6,6}, {1,2,3,4}, {2,1}, {1,1}, ge::DT_FLOAT16, 0, 0, false, 6);
}

TEST_F(Tiling, run_conv2d_group_case6) {
   Conv2DTestCase({2,7,11,20}, {42,6,6}, {2,1,4,3}, {1,1}, {2,1}, ge::DT_FLOAT16, 1, 0, false, 7);
}

TEST_F(Tiling, run_conv2d_group_case7) {
   Conv2DTestCase({1,32,64,64}, {32,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_0) { // bb choice 0
   Conv2DTestCase({1,32,1024,1024}, {64,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_1) { // bb choice 1
   Conv2DTestCase({36,32,32,32}, {64,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_2) { // bb choice 2
   Conv2DTestCase({1,32,512,512}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_3) { // bb choice 3
   Conv2DTestCase({36,32,8,8}, {1024,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_4) { // bb choice 4
   Conv2DTestCase({36,32,8,16}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_5) { // bb choice 5
   Conv2DTestCase({36,32,16,16}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_6) { // return in (mCut * nCut * batch * group <= aicoreNum)
   Conv2DTestCase({1,32,32,32}, {32,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_7) { //return in (maxHiWiL1 <= 0)
   Conv2DTestCase({1,32,64,64}, {524288,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_8) { //return in (mTile < m0 || nTile < n0)
   Conv2DTestCase({36,32,3,3}, {32,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_9) { //return in (mTile < m0 || nTile < n0)
   Conv2DTestCase({36,32,32,32}, {3,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_10) { //return in (maxhiL1 <= CONST_VALUE_2)
   Conv2DTestCase({1,32,64,64}, {261952,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_11) { // YYF STC1
   Conv2DTestCase({4,32,64,64}, {32,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_12) { // YYF STC2
   Conv2DTestCase({4,32,64,64}, {32,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_13) { // YYF STC3
   Conv2DTestCase({4,64,64,64}, {32,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 2);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_14) { // case from testTeam
   Conv2DTestCase({36,8,26,92},{85,10,9},{9,9,8,8},{13,5},{1,9}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_15) {
   Conv2DTestCase({1,32,64,64}, {32,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_perf_21072) {
   Conv2DTestCase({8,192,16,16},{1,1,1},{0,0,0,0},{1,1},{1,1}, ge::DT_FLOAT16, 1);
}

// TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_perf_21067) {
//    Conv2DTestCase({2,512,18,18},{2048,1,1},{0,0,0,0},{1,1},{1,1}, ge::DT_FLOAT16, 0);
// }

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_perf_21076) {
   Conv2DTestCase({1,1280,32,32},{1280,3,3},{1,1,1,1},{1,1},{1,1}, ge::DT_FLOAT16, 1);
}

//TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_perf_21077) { // Normal interception
//  Conv2DTestCase({1,320,128,128},{320,1,1},{1,1,1,1},{1,1},{1,1}, ge::DT_FLOAT16, 1);
//}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_perf_21000) {
   Conv2DTestCase({1,1280,64,64},{640,1,1},{0,0,0,0},{1,1},{1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_perf_14809) {
   Conv2DTestCase({1,128,32,32},{256,3,3},{1,1,1,1},{1,1},{1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_perf_19758) {
   Conv2DTestCase({1,256,33,33},{256,3,3},{0,0,0,0},{2,2},{1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_perf_15882) {
   Conv2DTestCase({1,256,16,16},{512,3,3},{1,1,1,1},{1,1},{1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_perf_18101) {
   Conv2DTestCase({1,128,512,512},{256,3,3},{1,1,1,1},{1,1},{1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_perf_12384) {
   Conv2DTestCase({1,256,512,512},{256,3,3},{1,1,1,1},{1,1},{1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_perf_17366) {
   Conv2DTestCase({1,256,513,513},{256,3,3},{0,0,0,0},{2,2},{1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_perf_18013) {
   Conv2DTestCase({1,256,256,256},{512,3,3},{1,1,1,1},{1,1},{1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_perf_11958) {
   Conv2DTestCase({1,512,256,256},{512,3,3},{1,1,1,1},{1,1},{1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_perf_11594) {
   Conv2DTestCase({1,512,257,257},{512,3,3},{0,0,0,0},{2,2},{1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_perf_18357) {
   Conv2DTestCase({1,512,128,128},{512,3,3},{1,1,1,1},{1,1},{1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_perf_12753) {
   Conv2DTestCase({1,320,128,128},{320,3,3},{1,1,1,1},{1,1},{1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_perf_10356) {
   Conv2DTestCase({1,320,128,128},{320,3,3},{1,1,1,1},{2,2},{1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_perf_16404) {
   Conv2DTestCase({1,640,128,128},{640,3,3},{1,1,1,1},{1,1},{1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_perf_15795) {
   Conv2DTestCase({1,960,128,128},{320,3,3},{1,1,1,1},{1,1},{1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_perf_15600) {
   Conv2DTestCase({1,640,128,128},{320,3,3},{1,1,1,1},{1,1},{1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_perf_21065) {
   Conv2DTestCase({2,640,26,38},{1280,3,3},{1,1,1,1},{1,1},{1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_perf_21001) {
   Conv2DTestCase({2,256,18,18},{2048,3,3},{1,1,1,1},{1,1},{1,1}, ge::DT_FLOAT16, 1);
}

// TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_preload_case1) {
//    Conv2DTestCase({2,640,26,38}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 1);
// }

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_preload_case2) {
   Conv2DTestCase({1,1280,32,32},{1280,3,3},{1,1,1,1},{1,1},{1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_preload_case3) {
   Conv2DTestCase({2,256,18,18},{2048,3,3},{1,1,1,1},{1,1},{1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_debug1) {
   Conv2DTestCase({6,8,231,221},{119,12,11},{11,11,2,2},{2,6},{5,5}, ge::DT_FLOAT16, 0);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_debug2) {
   Conv2DTestCase({12,12,211,132},{87,11,14},{0,0,8,8},{3,3},{7,3}, ge::DT_FLOAT16, 0);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_group) {
   Conv2DTestCase({8,224,38,38},{1344,1,1},{0,0,0,0},{1,1},{1,1}, ge::DT_FLOAT16, 0, 0, false, 2);
}

TEST_F(Tiling, run_conv2d_case_BasicBlockAlgo_group_opt) {
   Conv2DTestCase({36,32,16,16}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 4);
}

TEST_F(Tiling, run_Conv2d_case_Hif8NoScale1) {
   Conv2DTestCase({1,1,1,1}, {1,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_HIFLOAT8, 0, 0);
}

TEST_F(Tiling, run_conv1d_HWsplit_fp16_case1) {
   Conv2DTestCase({16,16,1,1000000}, {16,1,3}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv1d_HWsplit_fp16_case2) {
   Conv2DTestCase({2,16,1,1000000}, {16,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv1d_HWsplit_fp16_case3) {
   Conv2DTestCase({1,16,1,1000000}, {1,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv1d_HWsplit_fp16_case4) {
   Conv2DTestCase({1,16,1,1000000}, {2,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv1d_HWsplit_fp16_case5) {
   Conv2DTestCase({1,16,1,200}, {2,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv1d_HWsplit_fp16_case6) {
   Conv2DTestCase({1,16,1,8}, {2,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv1d_HWsplit_fp16_not_conv1d_case1) {
   Conv2DTestCase({1,16,8,8}, {2,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv1d_HWsplit_fp16_not_conv1d_case2) {
   Conv2DTestCase({1,16,1,8}, {2,4,1}, {2,2,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1);
}

TEST_F(Tiling, run_conv1d_HWsplit_fp16_group_case1) {
   Conv2DTestCase({1,16,1,1000000}, {16,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 2);
}

TEST_F(Tiling, run_conv1d_HWsplit_fp16_group_case2) {
   Conv2DTestCase({1,20,1,1000000}, {20,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 5);
}

TEST_F(Tiling, run_conv1d_HWsplit_fp32_case1) {
   Conv2DTestCase({16,16,1,1000000}, {16,1,3}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT, 1);
}

TEST_F(Tiling, run_conv1d_HWsplit_fp32_case2) {
   Conv2DTestCase({2,16,1,1000000}, {16,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT, 1);
}

TEST_F(Tiling, run_conv1d_HWsplit_fp32_case3) {
   Conv2DTestCase({1,16,1,1000000}, {1,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT, 1);
}

TEST_F(Tiling, run_conv1d_HWsplit_fp32_case4) {
   Conv2DTestCase({1,16,1,1000000}, {2,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT, 1);
}

TEST_F(Tiling, run_conv1d_HWsplit_fp32_case5) {
   Conv2DTestCase({1,16,1,200}, {2,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT, 1);
}

TEST_F(Tiling, run_conv1d_HWsplit_fp32_case6) {
   Conv2DTestCase({1,16,1,8}, {2,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT, 1);
}

TEST_F(Tiling, run_conv1d_HWsplit_fp32_group_case1) {
   Conv2DTestCase({1,16,1,1000000}, {16,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT, 1, 0, false, 2);
}

TEST_F(Tiling, run_conv1d_HWsplit_fp32_group_case2) {
   Conv2DTestCase({1,20,1,1000000}, {20,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT, 1, 0, false, 5);
}

TEST_F(Tiling, run_conv1d_HWsplit_bf16_case1) {
   Conv2DTestCase({16,16,1,1000000}, {16,1,3}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_BF16, 1);
}

TEST_F(Tiling, run_conv1d_HWsplit_bf16_case2) {
   Conv2DTestCase({2,16,1,1000000}, {16,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_BF16, 1);
}

TEST_F(Tiling, run_conv1d_HWsplit_bf16_case3) {
   Conv2DTestCase({1,16,1,1000000}, {1,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_BF16, 1);
}

TEST_F(Tiling, run_conv1d_HWsplit_bf16_case4) {
   Conv2DTestCase({1,16,1,1000000}, {2,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_BF16, 1);
}

TEST_F(Tiling, run_conv1d_HWsplit_bf16_case5) {
   Conv2DTestCase({1,16,1,200}, {2,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_BF16, 1);
}

TEST_F(Tiling, run_conv1d_HWsplit_bf16_case6) {
   Conv2DTestCase({1,16,1,8}, {2,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_BF16, 1);
}

TEST_F(Tiling, run_conv1d_HWsplit_bf16_group_case1) {
   Conv2DTestCase({1,16,1,1000000}, {16,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_BF16, 1, 0, false, 2);
}

TEST_F(Tiling, run_conv1d_HWsplit_bf16_group_case2) {
   Conv2DTestCase({1,20,1,1000000}, {20,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_BF16, 1, 0, false, 5);
}

TEST_F(Tiling, run_conv1d_HWsplit_hif8_case1) {
   Conv2DTestCase({16,16,1,1000000}, {16,1,3}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_HIFLOAT8, 1);
}

TEST_F(Tiling, run_conv1d_HWsplit_hif8_case2) {
   Conv2DTestCase({2,16,1,1000000}, {16,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_HIFLOAT8, 1);
}

TEST_F(Tiling, run_conv1d_HWsplit_hif8_case3) {
   Conv2DTestCase({1,16,1,1000000}, {1,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_HIFLOAT8, 1);
}

TEST_F(Tiling, run_conv1d_HWsplit_hif8_case4) {
   Conv2DTestCase({1,16,1,1000000}, {2,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_HIFLOAT8, 1);
}

TEST_F(Tiling, run_conv1d_HWsplit_hif8_case5) {
   Conv2DTestCase({1,16,1,200}, {2,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_HIFLOAT8, 1);
}

TEST_F(Tiling, run_conv1d_HWsplit_hif8_case6) {
   Conv2DTestCase({1,16,1,8}, {2,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_HIFLOAT8, 1);
}

TEST_F(Tiling, run_conv1d_HWsplit_hif8_group_case1) {
   Conv2DTestCase({1,16,1,1000000}, {16,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_HIFLOAT8, 1, 0, false, 2);
}

TEST_F(Tiling, run_conv1d_HWsplit_hif8_group_case2) {
   Conv2DTestCase({1,20,1,1000000}, {20,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_HIFLOAT8, 1, 0, false, 5);
}

TEST_F(Tiling, run_conv2d_case_padMode_valid) {
   Conv2DTestCase({1,1,256,256}, {1,3,4}, {9999,9999,9999,9999}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "VALID");
}

TEST_F(Tiling, run_conv2d_case_padMode_same) {
   Conv2DTestCase({1,1,256,256}, {1,3,4}, {9999,9999,9999,9999}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SAME");
}

TEST_F(Tiling, run_conv2d_case_padMode_sameUpper) {
   Conv2DTestCase({1,1,256,256}, {1,3,4}, {9999,9999,9999,9999}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SAME_UPPER");
}

TEST_F(Tiling, run_conv2d_case_padMode_sameLower) {
   Conv2DTestCase({1,1,256,256}, {1,3,4}, {9999,9999,9999,9999}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SAME_LOWER");
}

// TEST_F(Tiling, run_solomon_conv2d_case_1) {
//    Conv2DTestCase({2,640,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_BF16, 0, 0, false, 1, "", "Ascend910_5591", "Ascend910_55");
// }

// TEST_F(Tiling, run_solomon_conv2d_case_2) {
//    Conv2DTestCase({2,640,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 0, 0, false, 1, "", "Ascend910_5591", "Ascend910_55");
// }

// TEST_F(Tiling, run_solomon_conv2d_case_3) {
//    Conv2DTestCase({2,640,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT, 0, 0, false, 1, "", "Ascend910_5591", "Ascend910_55");
// }

// TEST_F(Tiling, run_solomon_conv2d_case_4) {
//    Conv2DTestCase({2,640,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_HIFLOAT8, 0, 0, false, 1, "", "Ascend910_5591", "Ascend910_55");
// }

TEST_F(Tiling, run_solomon_conv2d_group_not_equal_one_case1) {
   Conv2DTestCase({1,12,1,1}, {2,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_BF16, 0, 0, false, 2, "", "Ascend910_5591", "Ascend910_55", true);
}

TEST_F(Tiling, run_solomon_conv2d_group_not_equal_one_case2) {
   Conv2DTestCase({1,12,1,1}, {2,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 0, 0, false, 2, "", "Ascend910_5591", "Ascend910_55", true);
}

TEST_F(Tiling, run_solomon_conv2d_group_not_equal_one_case3) {
   Conv2DTestCase({1,12,1,1}, {2,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT, 0, 0, false, 2, "", "Ascend910_5591", "Ascend910_55", true);
}

TEST_F(Tiling, run_solomon_conv2d_group_not_equal_one_case4) {
   Conv2DTestCase({1,12,1,1}, {2,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_HIFLOAT8, 0, 0, false, 2, "", "Ascend910_5591", "Ascend910_55", true);
}

// TEST_F(Tiling, run_quantConv2d_without_platfrom_info_gencase_16) {
//   QuantConv2DWithoutPlatformInfoTestCase({32,64,2,13}, {256,27,20}, {26,0,14,0}, {1,1}, {1,1}, ge::DT_INT8, 0, 1);
// }

// TEST_F(Tiling, run_solomon_quantConv2d_normal_case_1) {
//   QuantConv2DWithoutPlatformInfoTestCase({1,32,16,16}, {2,3,3}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 0, 1, false, 1, "Ascend910_5591", "Ascend910_55");
//}

// TEST_F(Tiling, run_solomon_quantConv2d_normal_case_2) {
//   QuantConv2DWithoutPlatformInfoTestCase({1,32,16,16}, {2,3,3}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT8_E4M3FN, 0, 1, false, 1, "Ascend910_5591", "Ascend910_55");
//}

// TEST_F(Tiling, run_solomon_quantConv2d_normal_case_3) {
//   QuantConv2DWithoutPlatformInfoTestCase({1,32,16,16}, {2,3,3}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_HIFLOAT8, 0, 1, false, 1, "Ascend910_5591", "Ascend910_55");
// }

// TEST_F(Tiling, run_solomon_quantConv2d_groups_not_equal_1) {
// //   QuantConv2DWithoutPlatformInfoTestCase({1,32,16,16}, {2,3,3}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_INT8, 0, 1, false, 2, "Ascend910_5591", "Ascend910_55", true);
// }

// static QuantConv2DDtype quantConv2DDtype1 = {
//   ge::DT_INT8, ge::DT_INT8, ge::DT_INT32,
//   ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16
// };
// TEST_F(Tiling, run_solomon_quantConv2d_scale_dtype_illegal) {
//    QuantConv2DWithoutPlatformInfoTestCase({1,32,16,16}, {2,3,3}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_HIFLOAT8, 0, 1, false, 1, "Ascend910_5591", "Ascend910_55", true, quantConv2DDtype1);
// }

TEST_F(Tiling, run_conv2d_case_NHWC_fp16) {
   Conv2DTestCase({1,2,3,16}, {16,2,7}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", "Ascend910_9589", "Ascend910_95", false, "NHWC");
}

TEST_F(Tiling, run_conv2d_case_NHWC_bf16) {
   Conv2DTestCase({1,2,3,16}, {16,2,7}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_BF16, 1, 0, false, 1, "SPECIFIC", "Ascend910_9589", "Ascend910_95", false, "NHWC");
}

TEST_F(Tiling, run_conv2d_case_NHWC_fp32) {
   Conv2DTestCase({1,2,3,16}, {16,2,7}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT, 1, 0, false, 1, "SPECIFIC", "Ascend910_9589", "Ascend910_95", false, "NHWC");
}

TEST_F(Tiling, run_conv2d_case_NHWC_hf32) {
   Conv2DTestCase({1,2,3,16}, {16,2,7}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT, 1, 0, true, 1, "SPECIFIC", "Ascend910_9589", "Ascend910_95", false, "NHWC");
}

TEST_F(Tiling, run_conv2d_case_NHWC_soc_unsupport) {
   Conv2DTestCase({1,2,3,16}, {16,2,7}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", "Ascend910_5591", "Ascend910_55", true, "NHWC");
}

TEST_F(Tiling, run_conv2d_case_NHWC_group_support) {
   Conv2DTestCase({4,64,64,64}, {32,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 2, "SPECIFIC", "Ascend910_9589", "Ascend910_95", false, "NHWC");
}

TEST_F(Tiling, run_conv2d_case_NHWC_C04_not_support) {
   Conv2DTestCase({4,3,64,64}, {32,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", "Ascend910_9589", "Ascend910_95", false, "NHWC");
}

TEST_F(Tiling, run_conv2d_NHWC_hif8_not_support) {
   Conv2DTestCase({1,20,3,1000}, {20,3,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_HIFLOAT8, 1, 0, false, 1, "SPECIFIC", "Ascend910_9589", "Ascend910_95", true, "NHWC");
}

TEST_F(Tiling, run_conv1d_NHWC_hif8_not_support) {
   Conv2DTestCase({1,20,1,1000000}, {20,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_HIFLOAT8, 1, 0, false, 1, "SPECIFIC", "Ascend910_9589", "Ascend910_95", true, "NHWC");
}

TEST_F(Tiling, run_conv2d_NHWC_case_padMode_valid) {
   Conv2DTestCase({1,1,256,256}, {1,3,4}, {9999,9999,9999,9999}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "VALID", "Ascend910_9589", "Ascend910_95", false, "NHWC");
}

TEST_F(Tiling, run_conv2d_NHWC_case_padMode_same) {
   Conv2DTestCase({1,1,256,256}, {1,3,4}, {9999,9999,9999,9999}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SAME", "Ascend910_9589", "Ascend910_95", false, "NHWC");
}

TEST_F(Tiling, run_conv2d_NHWC_case_padMode_sameUpper) {
   Conv2DTestCase({1,1,256,256}, {1,3,4}, {9999,9999,9999,9999}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SAME_UPPER", "Ascend910_9589", "Ascend910_95", false, "NHWC");
}

TEST_F(Tiling, run_conv2d_NHWC_case_padMode_sameLower) {
   Conv2DTestCase({1,1,256,256}, {1,3,4}, {9999,9999,9999,9999}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SAME_LOWER", "Ascend910_9589", "Ascend910_95", false, "NHWC");
}

TEST_F(Tiling, run_conv2d_case_NHWC_exceed_fixpipe_instr_limit) {
   Conv2DTestCase({1,2,3,180000}, {160000,2,7}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", "Ascend910_9589", "Ascend910_95", true, "NHWC");
}

TEST_F(Tiling, run_ExtendConv2D_case_001) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      0, "NCHW"); // errorCaseStatus, format
}

TEST_F(Tiling, run_ExtendConv2D_case_002) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      true, 1, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      1, "NCHW"); // errorCaseStatus, format
}

TEST_F(Tiling, run_ExtendConv2D_case_003) {
  ExtendConv2DTestCase({1,256,64,64}, {256,3,3},{0,0,0,0}, {1,1}, {1,1},
                      ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      0, "NCHW", "round"); // errorCaseStatus, format, round_mode
}

TEST_F(Tiling, run_ExtendConv2D_case_004) {
  ExtendConv2DTestCase({1,256,64,64}, {256,3,3},{0,0,0,0}, {1,1}, {1,1},
                      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      0, "NCHW", "rint"); // errorCaseStatus, format, round_mode
}

TEST_F(Tiling, run_ExtendConv2D_case_005) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      0, "NHWC"); // errorCaseStatus, format
}

TEST_F(Tiling, run_ExtendConv2D_case_006) { //extendconv2d cache tiling
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      0, "NCHW"); // errorCaseStatus, format
}

TEST_F(Tiling, run_ExtendConv2D_case_007) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      true, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      0, "NCHW"); // errorCaseStatus, format
}

TEST_F(Tiling, run_ExtendConv2D_case_008) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, true, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      1, "NCHW"); // errorCaseStatus, format
}

TEST_F(Tiling, run_ExtendConv2D_case_009) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, true, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      0, "NCHW"); // errorCaseStatus, format
}

TEST_F(Tiling, run_ExtendConv2D_case_010) {
  ExtendConv2DTestCase({1,256,64,64}, {256,3,3},{0,0,0,0}, {1,1}, {1,1},
                      ge::DT_HIFLOAT8, ge::DT_FLOAT, ge::DT_FLOAT, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      0, "NCHW", "round"); // errorCaseStatus, format, round_mode
}

TEST_F(Tiling, run_ExtendConv2D_case_011) {
  ExtendConv2DTestCase({1,256,64,64}, {256,3,3},{0,0,0,0}, {1,1}, {1,1},
                      ge::DT_HIFLOAT8, ge::DT_BF16, ge::DT_BF16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      0, "NCHW", "round"); // errorCaseStatus, format, round_mode
}

TEST_F(Tiling, run_ExtendConv2D_case_012) {
  ExtendConv2DTestCase({1,256,64,64}, {256,3,3},{0,0,0,0}, {1,1}, {1,1},
                      ge::DT_HIFLOAT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      0, "NCHW", "round"); // errorCaseStatus, format, round_mode
}

TEST_F(Tiling, run_ExtendConv2D_case_013) {
  ExtendConv2DTestCase({1,256,64,64}, {256,3,3},{0,0,0,0}, {1,1}, {1,1},
                      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT, ge::DT_FLOAT, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      0, "NCHW", "rint"); // errorCaseStatus, format, round_mode
}

TEST_F(Tiling, run_ExtendConv2D_case_014) {
  ExtendConv2DTestCase({1,256,64,64}, {256,3,3},{0,0,0,0}, {1,1}, {1,1},
                      ge::DT_FLOAT8_E4M3FN, ge::DT_BF16, ge::DT_BF16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      0, "NCHW", "rint"); // errorCaseStatus, format, round_mode
}

TEST_F(Tiling, run_ExtendConv2D_case_015) {
  ExtendConv2DTestCase({1,256,64,64}, {256,3,3},{0,0,0,0}, {1,1}, {1,1},
                      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      0, "NCHW", "rint"); // errorCaseStatus, format, round_mode
}

// TEST_F(Tiling, run_quantConv2d_BasicBlockAlgo_singleCoreCo_fix) {
//    Conv2DTestCase({16,6,92395,21}, {386,16,14}, {4,0,5,0}, {53,3}, {6,1}, ge::DT_HIFLOAT8, 1, 1, false, 2);
// }

TEST_F(Tiling, run_ExtendConv2D_singleCoreCo_fix_case1) {
  ExtendConv2DTestCase({4,4,43765,5}, {1024,165,4},{146,10,1,1}, {13,2}, {4,1},
                      ge::DT_FLOAT8_E4M3FN, ge::DT_BF16, ge::DT_BF16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 4, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      0, "NCHW", "rint"); // errorCaseStatus, format, round_mode
}

TEST_F(Tiling, run_ExtendConv2D_case_nchw_int8_in_fp16_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      0, "NCHW"); // errorCaseStatus, format
}

TEST_F(Tiling, run_ExtendConv2D_case_nhwc_int8_in_int8_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      0, "NHWC"); // errorCaseStatus, format
}

TEST_F(Tiling, run_ExtendConv2D_case_nhwc_int8_in_fp16_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      0, "NHWC"); // errorCaseStatus, format
}

TEST_F(Tiling, run_ExtendConv2D_case_nchw_int8_in_int8_fp16_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, true, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, true, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      0, "NCHW"); // errorCaseStatus, format
}

TEST_F(Tiling, run_ExtendConv2D_case_nchw_int8_in_fp16_int8_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8, // inDataType, out0DataType, out1DataType
                      true, true, true, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, true, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      0, "NCHW"); // errorCaseStatus, format
}

TEST_F(Tiling, run_ExtendConv2D_case_nhwc_int8_in_int8_fp16_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, true, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, true, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      0, "NHWC"); // errorCaseStatus, format
}

TEST_F(Tiling, run_ExtendConv2D_case_nhwc_int8_in_fp16_int8_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8, // inDataType, out0DataType, out1DataType
                      true, true, true, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, true, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      0, "NHWC"); // errorCaseStatus, format
}

TEST_F(Tiling, run_ExtendConv2D_case_nchw_int8_in_int8_fp16_out_case_cache_tiling) {
   for (int i = 0; i < 2; i++) {
      ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                           ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                           true, true, true, //isHasBias, isHasScale0, isHasScale1,
                           false, false, // isHasReluWeight0, isHasReluWeight1
                           false, false, //isHasClipValue0, isHasClipValue1
                           false, false, true, // enableRelu0, enableRelu1, dualOutput
                           false, 1, // enableHf32Mode, groups
                           "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                           0, "NHWC"); // errorCaseStatus, format
   }
}

TEST_F(Tiling, run_ExtendConv2D_case_nchw_fp8_in_int8_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_FLOAT8_E4M3FN, ge::DT_INT8, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, true, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      1, "NCHW"); // errorCaseStatus, format
}

TEST_F(Tiling, run_ExtendConv2D_case_nhwc_fp8_in_fp8_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      1, "NHWC"); // errorCaseStatus, format
}

TEST_F(Tiling, run_ExtendConv2D_case_nhwc_hif8_in_hif8_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      1, "NHWC"); // errorCaseStatus, format
}

TEST_F(Tiling, run_ExtendConv2D_case_nhwc_int8_in_int8_out_groups2_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 2, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      0, "NHWC"); // errorCaseStatus, format
}

TEST_F(Tiling, run_ExtendConv2D_case_nchw_int8_in_fp16_fp16_out_case) {
  ExtendConv2DTestCase({2,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, true, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, true, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      0, "NCHW"); // errorCaseStatus, format
}

TEST_F(Tiling, run_ExtendConv2D_case_nhwc_int8_in_fp16_fp16_out_case) {
  ExtendConv2DTestCase({2,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, true, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, true, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC", "Ascend910_9589", "Ascend910_95", // padMode, socVersion, shortSocVersion
                      0, "NHWC"); // errorCaseStatus, format
}
