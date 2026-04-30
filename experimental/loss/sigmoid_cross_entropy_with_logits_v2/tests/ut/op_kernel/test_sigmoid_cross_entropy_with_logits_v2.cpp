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
#include "../../../op_kernel/sigmoid_cross_entropy_with_logits_v2_tiling_data.h"
#include "../../../op_kernel/sigmoid_cross_entropy_with_logits_v2_tiling_key.h"
#include <cstdint>

using namespace std;

using TilingDataType = SigmoidCrossEntropyWithLogitsV2TilingData;

extern "C" __global__ __aicore__ void sigmoid_cross_entropy_with_logits_v2(
    GM_ADDR predict, GM_ADDR target, GM_ADDR weight, GM_ADDR posWeight, GM_ADDR loss, GM_ADDR workspace,
    GM_ADDR tiling);

class sigmoid_cross_entropy_with_logits_v2_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "sigmoid_cross_entropy_with_logits_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "sigmoid_cross_entropy_with_logits_v2_test TearDown\n" << endl;
    }
};

TEST_F(sigmoid_cross_entropy_with_logits_v2_test, test_case_0)
{
    size_t inputPredictByteSize = 128 * sizeof(float);
    size_t inputTargetByteSize = 128 * sizeof(float);
    size_t inputWeightByteSize = 128 * sizeof(float);
    size_t inputPosWeightByteSize = 128 * sizeof(float);
    size_t outputLossByteSize = 128 * sizeof(float);
    size_t tilingDataSize = sizeof(TilingDataType);

    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(inputPredictByteSize);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(inputTargetByteSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(inputWeightByteSize);
    uint8_t* posWeight = (uint8_t*)AscendC::GmAlloc(inputPosWeightByteSize);

    uint8_t* loss = (uint8_t*)AscendC::GmAlloc(outputLossByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 1;

    TilingDataType* tilingDatafromBin = reinterpret_cast<TilingDataType*>(tiling);

    tilingDatafromBin->totalLength = 128;
    tilingDatafromBin->tileNum = 1;
    tilingDatafromBin->tileLength = 128;
    tilingDatafromBin->has_weight = 1;
    tilingDatafromBin->has_pos_weight = 1;
    tilingDatafromBin->dtype_enum = TILING_KEY_FLOAT;

    ICPU_SET_TILING_KEY(TILING_KEY_FLOAT);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        sigmoid_cross_entropy_with_logits_v2, blockDim, predict, target, weight, posWeight, loss, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(predict);
    AscendC::GmFree(target);
    AscendC::GmFree(weight);
    AscendC::GmFree(posWeight);
    AscendC::GmFree(loss);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}