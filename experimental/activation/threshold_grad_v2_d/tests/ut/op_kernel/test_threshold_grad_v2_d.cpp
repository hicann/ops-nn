/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif
#include "../../../op_kernel/threshold_grad_v2_d.cpp"
#include "../../../op_kernel/threshold_grad_v2_d_tiling_data.h"
#include <cstdint>

using namespace std;

class threshold_grad_v2_d_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "threshold_grad_v2_d_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "threshold_grad_v2_d_test TearDown\n" << endl;
    }
};

TEST_F(threshold_grad_v2_d_test, test_case_0)
{
    size_t input_gradientByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t input_featureByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t output_backpropsByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t tiling_data_size = sizeof(ThresholdGradV2DTilingData);
    uint32_t blockDim = 1;

    uint8_t* input_gradient = (uint8_t*)AscendC::GmAlloc(input_gradientByteSize);
    uint8_t* input_feature = (uint8_t*)AscendC::GmAlloc(input_featureByteSize);
    uint8_t* output_backprops = (uint8_t*)AscendC::GmAlloc(output_backpropsByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    ThresholdGradV2DTilingData* tilingDatafromBin = reinterpret_cast<ThresholdGradV2DTilingData*>(tiling);

    tilingDatafromBin->smallCoreDataNum = 2048;
    tilingDatafromBin->bigCoreDataNum = 2112;
    tilingDatafromBin->tileDataNum = 4032;
    tilingDatafromBin->smallTailDataNum = 2048;
    tilingDatafromBin->bigTailDataNum = 2112;
    tilingDatafromBin->finalSmallTileNum = 1;
    tilingDatafromBin->finalBigTileNum = 1;
    tilingDatafromBin->tailBlockNum = 0;
    tilingDatafromBin->threshold = 1.0;

    auto ThresholdGradV2DKernel = [](GM_ADDR input_gradient, GM_ADDR input_feature, GM_ADDR output_backprops, GM_ADDR workspace, GM_ADDR tiling) {
        ::threshold_grad_v2_d<0>(input_gradient, input_feature, output_backprops, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(ThresholdGradV2DKernel, blockDim, input_gradient, input_feature, output_backprops, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(input_gradient);
    AscendC::GmFree(input_feature);
    AscendC::GmFree(output_backprops);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}