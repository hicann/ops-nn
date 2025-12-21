/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

#include "adaptive_max_pool3d_grad_tiling_def.h"

using namespace std;

extern "C" void adaptive_max_pool3d_grad(
    GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

struct AdaptiveMaxPool3DGradTestParam {
    string case_name;

    int64_t N = 0;
    int64_t C = 0;

    int64_t inD = 0;
    int64_t inH = 0;
    int64_t inW = 0;

    int64_t outD = 0;
    int64_t outH = 0;
    int64_t outW = 0;

    int64_t blockDim = 0;

    size_t dataTypeSize;

    uint32_t tilingKey;
    AdaptiveMaxPool3DGradTilingData tiling;
};

class AdaptiveMaxPool3DGradTest : public testing::TestWithParam<AdaptiveMaxPool3DGradTestParam>
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "AdaptiveMaxPool3DGradTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AdaptiveMaxPool3DGradTest TearDown" << std::endl;
    }
};

TEST_P(AdaptiveMaxPool3DGradTest, test_case_adaptive_max_pool3d_grad)
{
    AdaptiveMaxPool3DGradTestParam param = GetParam();

    int64_t N = param.N;
    int64_t C = param.C;
    int64_t inD = param.inD;
    int64_t inH = param.inH;
    int64_t inW = param.inW;
    int64_t outD = param.outD;
    int64_t outH = param.outH;
    int64_t outW = param.outW;
    int64_t blockDim = param.blockDim;

    uint32_t tilingKey = param.tilingKey;
    AdaptiveMaxPool3DGradTilingData tilingParam = param.tiling;

    int64_t inputShapeSize = N * C * inD * inH * inW;
    int64_t outputShapeSize = N * C * outD * outH * outW;
    int64_t xByteSize = inputShapeSize * param.dataTypeSize;
    int64_t gradByteSize = outputShapeSize * param.dataTypeSize;
    int64_t argmaxByteSize = outputShapeSize * sizeof(int32_t);
    int64_t dxByteSize = inputShapeSize * param.dataTypeSize;

    int64_t workspaceSize = 0;
    int64_t tilingDataSize = sizeof(AdaptiveMaxPool3DGradTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* argmax = (uint8_t*)AscendC::GmAlloc(argmaxByteSize);
    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    AdaptiveMaxPool3DGradTilingData* tilingDatafromBin = reinterpret_cast<AdaptiveMaxPool3DGradTilingData*>(tiling);

    tilingDatafromBin->ncDim = N * C;
    tilingDatafromBin->diDim = inD;
    tilingDatafromBin->hiDim = inH;
    tilingDatafromBin->wiDim = inW;
    tilingDatafromBin->doDim = outD;
    tilingDatafromBin->hoDim = outH;
    tilingDatafromBin->woDim = outW;
    tilingDatafromBin->kdMax = tilingParam.kdMax;
    tilingDatafromBin->khMax = tilingParam.khMax;
    tilingDatafromBin->kwMax = tilingParam.kwMax;
    tilingDatafromBin->baseNc = tilingParam.baseNc;
    tilingDatafromBin->baseDo = tilingParam.baseDo;
    tilingDatafromBin->baseHo = tilingParam.baseHo;
    tilingDatafromBin->baseWo = tilingParam.baseWo;
    tilingDatafromBin->singleCoreNc = tilingParam.singleCoreNc;
    tilingDatafromBin->singleCoreDo = tilingParam.singleCoreDo;
    tilingDatafromBin->singleCoreHo = tilingParam.singleCoreHo;
    tilingDatafromBin->singleCoreWo = tilingParam.singleCoreWo;
    tilingDatafromBin->ncTail = tilingParam.ncTail;
    tilingDatafromBin->doTail = tilingParam.doTail;
    tilingDatafromBin->hoTail = tilingParam.hoTail;
    tilingDatafromBin->woTail = tilingParam.woTail;
    tilingDatafromBin->ncCnt = tilingParam.ncCnt;
    tilingDatafromBin->doCnt = tilingParam.doCnt;
    tilingDatafromBin->hoCnt = tilingParam.hoCnt;
    tilingDatafromBin->woCnt = tilingParam.woCnt;
    tilingDatafromBin->totalCnt = tilingParam.totalCnt;
    tilingDatafromBin->usedCoreNum = tilingParam.usedCoreNum;
    tilingDatafromBin->totalUBSize = tilingParam.totalUBSize;

    tilingDatafromBin->ncRound = tilingParam.ncRound;
    tilingDatafromBin->ncRoundTail = tilingParam.ncRoundTail;
    tilingDatafromBin->totalRound = tilingParam.totalRound;
    tilingDatafromBin->preCoreNum = tilingParam.preCoreNum;

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(adaptive_max_pool3d_grad, blockDim, x, grad, argmax, dy, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(grad);
    AscendC::GmFree(argmax);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

static AdaptiveMaxPool3DGradTestParam cases[] = {
    {"test_case_adaptive_max_pool3d_grad_scatter",
     5,
     640,
     1,
     64,
     64,
     1,
     1,
     1,
     sizeof(float),
     40,
     2,
     {3200, 1,  64, 64, 1, 1,  1, 1, 64, 64, 80, 1,      1, 1, 3200, 1, 1,
      1,    80, 1,  1,  1, 40, 1, 1, 1,  40, 40, 196352, 1, 0, 1,    0}}
};

INSTANTIATE_TEST_CASE_P(AdaptiveMaxPool3DGrad, AdaptiveMaxPool3DGradTest, testing::ValuesIn(cases));