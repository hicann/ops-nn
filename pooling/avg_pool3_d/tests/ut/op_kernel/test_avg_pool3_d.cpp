/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <cstdint>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "avg_pool3_d_tiling_def.h"

using namespace std;

extern "C" void avg_pool3_d(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

struct AvgPool3DTestParam {
  string caseName;

  size_t dataTypeSize;

  uint32_t tilingKey;
  AvgPool3DTilingData tiling;
};

class AvgPool3DTest : public testing::TestWithParam<AvgPool3DTestParam> {
protected:
  static void SetUpTestCase() {
    std::cout << "AvgPool3DTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AvgPool3DTest TearDown" << std::endl;
  }
};

TEST_P(AvgPool3DTest, test_case_avg_pool3d) {
  AvgPool3DTestParam param = GetParam();

  uint32_t tilingKey = param.tilingKey;
  AvgPool3DTilingData tilingParam = param.tiling;

  int64_t inputShapeSize = tilingParam.inN * tilingParam.inC * tilingParam.inD * tilingParam.inH *tilingParam.inW;
  int64_t outputShapeSize = tilingParam.inN * tilingParam.inC * tilingParam.outD * tilingParam.outH *tilingParam.outW;
  int64_t inputXByteSize = inputShapeSize * param.dataTypeSize;
  int64_t outputYByteSize = outputShapeSize * param.dataTypeSize;

  int64_t workspaceSize = 32;
  int64_t tilingSize = sizeof(AvgPool3DTilingData);

  uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize);
  uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

  AvgPool3DTilingData* tilingData = reinterpret_cast<AvgPool3DTilingData*>(tiling);
  tilingData->inN = tilingParam.inN;
  tilingData->inC = tilingParam.inC;
  tilingData->tileC = tilingParam.tileC;
  tilingData->inD = tilingParam.inD;
  tilingData->inH = tilingParam.inH;
  tilingData->inW = tilingParam.inW;
  tilingData->outD = tilingParam.outD;
  tilingData->outH = tilingParam.outH;
  tilingData->outW = tilingParam.outW;
  tilingData->kD = tilingParam.kD;
  tilingData->kH = tilingParam.kH;
  tilingData->kW = tilingParam.kW;
  tilingData->dD = tilingParam.dD;
  tilingData->dH = tilingParam.dH;
  tilingData->dW = tilingParam.dW;
  tilingData->pD = tilingParam.pD;
  tilingData->pH = tilingParam.pH;
  tilingData->pW = tilingParam.pW;
  tilingData->divisorOverride = tilingParam.divisorOverride;
  tilingData->countIncludePad = tilingParam.countIncludePad;
  tilingData->ceilMode = tilingParam.ceilMode;
  tilingData->formerLength = tilingParam.formerLength;
  tilingData->formerNum = tilingParam.formerNum;
  tilingData->tailLength = tilingParam.tailLength;
  tilingData->tailNum = tilingParam.tailNum;
  tilingData->indexBufLen = tilingParam.indexBufLen;
  tilingData->windowWNum = tilingParam.windowWNum;
  tilingData->tileInput = tilingParam.tileInput;
  tilingData->tileHW = tilingParam.tileHW;
  tilingData->atomicAddNum = tilingParam.atomicAddNum;

  uint32_t blockDim = tilingParam.formerNum + tilingParam.tailNum;
  ICPU_SET_TILING_KEY(tilingKey);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(avg_pool3_d, blockDim, x, y, workspace, tiling);

  AscendC::GmFree((void *)x);
  AscendC::GmFree((void *)y);
  AscendC::GmFree((void *)workspace);
  AscendC::GmFree((void *)tiling);
}

static AvgPool3DTestParam cases[] = {
  {
    "test_case_split_c_float32",
    sizeof(float),
    10,
    {
      2, 256, 256,
      8, 16, 16,
      8, 7, 7,
      1, 3, 3,
      1, 2, 2,
      0, 0, 0,
      0, 1, 0,
      17, 16, 16, 32,
      128, 0, 0, 0
    }
  },
  {
    "test_case_split_c_float16",
    sizeof(half),
    11,
    {
      2, 256, 256,
      8, 16, 16,
      8, 7, 7,
      1, 3, 3,
      1, 2, 2,
      0, 0, 0,
      0, 1, 0,
      17, 16, 16, 32,
      128, 0, 0, 0
    }
  },
  {
    "test_case_split_c_bfloat16",
    sizeof(bfloat16_t),
    12,
    {
      2, 256, 256,
      8, 16, 16,
      8, 7, 7,
      1, 3, 3,
      1, 2, 2,
      0, 0, 0,
      0, 1, 0,
      17, 16, 16, 32,
      128, 0, 0, 0
    }
  },
  {
    "test_case_split_w_float32",
    sizeof(float),
    20,
    {
      2, 128, 0,
      8, 16, 16,
      8, 7, 7,
      1, 4, 4,
      1, 2, 2,
      0, 0, 0,
      0, 1, 0,
      17, 16, 16, 32,
      128, 0, 3, 0
    }
  },
  {
    "test_case_split_w_float16",
    sizeof(half),
    21,
    {
      2, 128, 0,
      8, 16, 16,
      8, 7, 7,
      1, 4, 4,
      1, 2, 2,
      0, 0, 0,
      0, 1, 0,
      17, 16, 16, 32,
      128, 0, 3, 0
    }
  },
  {
    "test_case_split_w_bfloat16",
    sizeof(bfloat16_t),
    22,
    {
      2, 128, 0,
      8, 16, 16,
      8, 7, 7,
      1, 4, 4,
      1, 2, 2,
      0, 0, 0,
      0, 1, 0,
      17, 16, 16, 32,
      128, 0, 3, 0
    }
  },
  {
    "test_case_multi_w_float32",
    sizeof(float),
    30,
    {
      2, 128, 0,
      8, 16, 16,
      8, 7, 7,
      1, 3, 3,
      1, 2, 2,
      0, 0, 0,
      0, 1, 0,
      17, 16, 16, 32,
      128, 4, 0, 0
    }
  },
  {
    "test_case_multi_w_float16",
    sizeof(half),
    31,
    {
      2, 128, 0,
      8, 16, 16,
      8, 7, 7,
      1, 3, 3,
      1, 2, 2,
      0, 0, 0,
      0, 1, 0,
      17, 16, 16, 32,
      128, 4, 0, 0
    }
  },
  {
    "test_case_multi_w_bfloat16",
    sizeof(bfloat16_t),
    32,
    {
      2, 128, 0,
      8, 16, 16,
      8, 7, 7,
      1, 3, 3,
      1, 2, 2,
      0, 0, 0,
      0, 1, 0,
      17, 16, 16, 32,
      128, 4, 0, 0
    }
  },
  {
    "test_case_reduce_d_float32",
    sizeof(float),
    40,
    {
      2, 128, 0,
      8, 16, 16,
      3, 16, 16,
      3, 1, 1,
      2, 1, 1,
      0, 0, 0,
      0, 1, 0,
      43, 32, 42, 16,
      128, 0, 0, 256
    }
  },
  {
    "test_case_reduce_d_float16",
    sizeof(half),
    41,
    {
      2, 128, 0,
      8, 16, 16,
      3, 16, 16,
      3, 1, 1,
      2, 1, 1,
      0, 0, 0,
      0, 1, 0,
      43, 32, 42, 16,
      128, 0, 0, 256
    }
  },
  {
    "test_case_reduce_d_bfloat16",
    sizeof(bfloat16_t),
    42,
    {
      2, 128, 0,
      8, 16, 16,
      3, 16, 16,
      3, 1, 1,
      2, 1, 1,
      0, 0, 0,
      0, 1, 0,
      43, 32, 42, 16,
      128, 0, 0, 256
    }
  },
  {
    "test_case_normal_float32",
    sizeof(float),
    50,
    {
      2, 128, 0,
      8, 16, 16,
      3, 16, 16,
      3, 1, 1,
      2, 1, 1,
      0, 0, 0,
      0, 1, 0,
      43, 32, 42, 16,
      128, 0, 0, 256
    }
  },
  {
    "test_case_normal_float16",
    sizeof(half),
    51,
    {
      2, 128, 0,
      8, 16, 16,
      3, 16, 16,
      3, 1, 1,
      2, 1, 1,
      0, 0, 0,
      0, 1, 0,
      43, 32, 42, 16,
      128, 0, 0, 256
    }
  },
  {
    "test_case_normal_bfloat16",
    sizeof(bfloat16_t),
    52,
    {
      2, 128, 0,
      8, 16, 16,
      3, 16, 16,
      3, 1, 1,
      2, 1, 1,
      0, 0, 0,
      0, 1, 0,
      43, 32, 42, 16,
      128, 0, 0, 256
    }
  },
};

INSTANTIATE_TEST_CASE_P(AvgPool3D, AvgPool3DTest, testing::ValuesIn(cases));
