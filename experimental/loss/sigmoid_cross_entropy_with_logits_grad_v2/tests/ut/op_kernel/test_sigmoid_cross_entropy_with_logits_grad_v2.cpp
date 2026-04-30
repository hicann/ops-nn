/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <iostream>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#endif

#include "../../../op_kernel/sigmoid_cross_entropy_with_logits_grad_v2_tiling_data.h"
#include "../../../op_kernel/sigmoid_cross_entropy_with_logits_grad_v2_tiling_key.h"

using TilingDataType = SigmoidCrossEntropyWithLogitsGradV2TilingData;

extern "C" __global__ __aicore__ void sigmoid_cross_entropy_with_logits_grad_v2(
    GM_ADDR predict, GM_ADDR target, GM_ADDR dout, GM_ADDR weight, GM_ADDR pos_weight, GM_ADDR gradient,
    GM_ADDR workspace, GM_ADDR tiling);

class sigmoid_cross_entropy_with_logits_grad_v2_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "sigmoid_cross_entropy_with_logits_grad_v2_test SetUp" << std::endl; }
    static void TearDownTestCase()
    {
        std::cout << "sigmoid_cross_entropy_with_logits_grad_v2_test TearDown" << std::endl;
    }
};

TEST_F(sigmoid_cross_entropy_with_logits_grad_v2_test, test_case_fp32_smoke)
{
    size_t tensor_size = 128 * sizeof(float);
    size_t tiling_data_size = sizeof(TilingDataType);

    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(tensor_size);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(tensor_size);
    uint8_t* dout = (uint8_t*)AscendC::GmAlloc(tensor_size);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(tensor_size);
    uint8_t* pos_weight = (uint8_t*)AscendC::GmAlloc(tensor_size);
    uint8_t* gradient = (uint8_t*)AscendC::GmAlloc(tensor_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t block_dim = 1;

    TilingDataType* tiling_data_from_bin = reinterpret_cast<TilingDataType*>(tiling);
    tiling_data_from_bin->smallCoreDataNum = 128;
    tiling_data_from_bin->bigCoreDataNum = 128;
    tiling_data_from_bin->finalBigTileNum = 1;
    tiling_data_from_bin->finalSmallTileNum = 1;
    tiling_data_from_bin->tileDataNum = 128;
    tiling_data_from_bin->smallTailDataNum = 128;
    tiling_data_from_bin->bigTailDataNum = 128;
    tiling_data_from_bin->tailBlockNum = 0;
    tiling_data_from_bin->globalScale = 1.0f;
    tiling_data_from_bin->has_weight = 0;
    tiling_data_from_bin->has_pos_weight = 0;
    tiling_data_from_bin->fp32_lite_mode = 1;

    ICPU_SET_TILING_KEY(TILING_KEY_FLOAT);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        sigmoid_cross_entropy_with_logits_grad_v2, block_dim, predict, target, dout, weight, pos_weight, gradient,
        workspace, (uint8_t*)(tiling_data_from_bin));

    AscendC::GmFree(predict);
    AscendC::GmFree(target);
    AscendC::GmFree(dout);
    AscendC::GmFree(weight);
    AscendC::GmFree(pos_weight);
    AscendC::GmFree(gradient);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
