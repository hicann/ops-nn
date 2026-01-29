/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_extend_conv2d_tiling.cpp
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

namespace {
uint64_t L1_Size = 524288;
uint64_t n0 = 16;
uint64_t mAL1_min = 16;
uint64_t m0 = 16;
uint64_t C0_SIZE = 32;

class ExtendConv2dTiling : public testing::Test {
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
    }

    if (outputOrder == 0) {
        CheckHWModeTilingDataValidForConv2d(tilingData, m0, n0, k0, isC04Flag);
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

void ExtendConv2DTestCase(vector<int64_t> fmShape, vector<int64_t> weightShape,
                    vector<uint32_t> pads, vector<uint32_t> strides, vector<uint32_t> dilations,
                    ge::DataType inDataType, ge::DataType out0DataType, ge::DataType out1DataType,
                    bool isHasBias = true, bool isHasScale0 = false, bool isHasScale1 = false,
                    bool isHasReluWeight0 = false, bool isHasReluWeight1 = false,
                    bool isHasClipValue0 = false, bool isHasClipValue1 = false,
                    bool enableRelu0 = false, bool enableRelu1 = false, bool dualOutput = false,
                    bool enableHf32Mode = false, uint32_t groups = 1,
                    string padMode = "SPECIFIC",
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
    gert::StorageShape ReluWeight0 = {{cout}, {cout}};
    gert::StorageShape ReluWeight1 = {{cout}, {cout}};
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
    ge::DataType reluWeight1DType = ge::DT_FLOAT;
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
      {"BT_SIZE": 4096, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
       "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 253952,
       "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "FB_SIZE": 4096,
       "BT_SIZE": 4096, "L0C_SIZE": 262144, "CORE_NUM": 32, "cube_core_cnt": 32, "vector_core_cnt": 64,
       "core_type_list": "CubeCore,VectorCore"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    map<string, string> soc_version_infos = {{"NpuArch", "3510"}};
    aicore_spec.insert({"fb0_size", "4096"});
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::conv_ops_tiling::ConvTilingParseInfo compile_info;

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
      EXPECT_LE(tilingParam.batchDim * tilingParam.hoDim * tilingParam.nDim * tilingParam.groupDim, aicoreNum);
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
} // namespace

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_001) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW"); // errorCaseStatus, format
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_002) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      true, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      1, "NCHW"); // errorCaseStatus, format
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_003) {
  ExtendConv2DTestCase({1,256,64,64}, {256,3,3},{0,0,0,0}, {1,1}, {1,1},
                      ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW", "round"); // errorCaseStatus, format, round_mode
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_004) {
  ExtendConv2DTestCase({1,256,64,64}, {256,3,3},{0,0,0,0}, {1,1}, {1,1},
                      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW", "rint"); // errorCaseStatus, format, round_mode
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_005) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NHWC"); // errorCaseStatus, format
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_006) { //extendconv2d cache tiling
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW"); // errorCaseStatus, format
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_007) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      true, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW"); // errorCaseStatus, format
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_008) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, true, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      1, "NCHW"); // errorCaseStatus, format
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_009) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, true, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW"); // errorCaseStatus, format
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_010) {
  ExtendConv2DTestCase({1,256,64,64}, {256,3,3},{0,0,0,0}, {1,1}, {1,1},
                      ge::DT_HIFLOAT8, ge::DT_FLOAT, ge::DT_FLOAT, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW", "round"); // errorCaseStatus, format, round_mode
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_011) {
  ExtendConv2DTestCase({1,256,64,64}, {256,3,3},{0,0,0,0}, {1,1}, {1,1},
                      ge::DT_HIFLOAT8, ge::DT_BF16, ge::DT_BF16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW", "round"); // errorCaseStatus, format, round_mode
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_012) {
  ExtendConv2DTestCase({1,256,64,64}, {256,3,3},{0,0,0,0}, {1,1}, {1,1},
                      ge::DT_HIFLOAT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW", "round"); // errorCaseStatus, format, round_mode
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_013) {
  ExtendConv2DTestCase({1,256,64,64}, {256,3,3},{0,0,0,0}, {1,1}, {1,1},
                      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT, ge::DT_FLOAT, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW", "rint"); // errorCaseStatus, format, round_mode
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_014) {
  ExtendConv2DTestCase({1,256,64,64}, {256,3,3},{0,0,0,0}, {1,1}, {1,1},
                      ge::DT_FLOAT8_E4M3FN, ge::DT_BF16, ge::DT_BF16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW", "rint"); // errorCaseStatus, format, round_mode
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_015) {
  ExtendConv2DTestCase({1,256,64,64}, {256,3,3},{0,0,0,0}, {1,1}, {1,1},
                      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW", "rint"); // errorCaseStatus, format, round_mode
}

// TEST_F(ExtendConv2dTiling, run_quantConv2d_BasicBlockAlgo_singleCoreCo_fix) {
//    Conv2DTestCase({16,6,92395,21}, {386,16,14}, {4,0,5,0}, {53,3}, {6,1}, ge::DT_HIFLOAT8, 1, 1, false, 2);
// }

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_singleCoreCo_fix_case1) {
  ExtendConv2DTestCase({4,4,43765,5}, {1024,165,4},{146,10,1,1}, {13,2}, {4,1},
                      ge::DT_FLOAT8_E4M3FN, ge::DT_BF16, ge::DT_BF16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 4, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW", "rint"); // errorCaseStatus, format, round_mode
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nchw_int8_in_fp16_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW"); // errorCaseStatus, format
}
 
TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nhwc_int8_in_int8_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NHWC"); // errorCaseStatus, format
}
 
TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nhwc_int8_in_fp16_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NHWC"); // errorCaseStatus, format
}
 
TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nchw_int8_in_int8_fp16_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, true, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, true, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW"); // errorCaseStatus, format
}
 
TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nchw_int8_in_fp16_int8_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8, // inDataType, out0DataType, out1DataType
                      true, true, true, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, true, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW"); // errorCaseStatus, format
}
 
TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nhwc_int8_in_int8_fp16_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, true, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, true, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NHWC"); // errorCaseStatus, format
}
 
TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nhwc_int8_in_fp16_int8_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8, // inDataType, out0DataType, out1DataType
                      true, true, true, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, true, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NHWC"); // errorCaseStatus, format
}
 
TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nchw_int8_in_int8_fp16_out_case_cache_tiling) {
   for (int i = 0; i < 2; i++) {
      ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                           ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                           true, true, true, //isHasBias, isHasScale0, isHasScale1,
                           false, false, // isHasReluWeight0, isHasReluWeight1
                           false, false, //isHasClipValue0, isHasClipValue1
                           false, false, true, // enableRelu0, enableRelu1, dualOutput
                           false, 1, // enableHf32Mode, groups
                           "SPECIFIC",
                           0, "NHWC"); // errorCaseStatus, format
   }
}
 
TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nchw_fp8_in_int8_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_FLOAT8_E4M3FN, ge::DT_INT8, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, true, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      1, "NCHW"); // errorCaseStatus, format
}
 
TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nhwc_fp8_in_fp8_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      1, "NHWC"); // errorCaseStatus, format
}
 
TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nhwc_hif8_in_hif8_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      1, "NHWC"); // errorCaseStatus, format
}
 
TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nhwc_int8_in_int8_out_groups2_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 2, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NHWC"); // errorCaseStatus, format
}
 
TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nchw_int8_in_fp16_fp16_out_case) {
  ExtendConv2DTestCase({2,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, true, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, true, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW"); // errorCaseStatus, format
}
 
TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nhwc_int8_in_fp16_fp16_out_case) {
  ExtendConv2DTestCase({2,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, true, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, true, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NHWC"); // errorCaseStatus, format
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nchw_conv_leakyrelu_case) {
  ExtendConv2DTestCase({2,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, false, false, //isHasBias, isHasScale0, isHasScale1,
                      true, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      true, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW"); // errorCaseStatus, format
}
