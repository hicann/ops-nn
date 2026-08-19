/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include "kernel_utils.h"
#include "data_utils.h"

using namespace std;

extern "C" __global__ __aicore__ void dequant_situ_quant(GM_ADDR x, GM_ADDR weight_scale, GM_ADDR activation_scale,
                                                         GM_ADDR bias, GM_ADDR quant_scale, GM_ADDR quant_offset,
                                                         GM_ADDR group_index, GM_ADDR y, GM_ADDR scale,
                                                         GM_ADDR workspace, GM_ADDR tiling);

class DequantSituQuantKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DequantSituQuantKernelTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "DequantSituQuantKernelTest TearDown" << std::endl; }
};

TEST_F(DequantSituQuantKernelTest, test_static_quant_scalar_no_bias)
{
    int64_t rowLen = 16;
    int64_t inDimy = 64;
    int64_t outDimy = 32;

    auto x_size = rowLen * inDimy;
    auto weight_scale_size = inDimy;
    auto y_size = rowLen * outDimy;
    auto scale_size = rowLen;

    std::vector<int8_t> x(x_size, 0);
    std::vector<float> weight_scale(weight_scale_size, 0.01f);
    std::vector<int8_t> y(y_size, 0);
    std::vector<float> scale_out(scale_size, 0.0f);
    std::vector<float> quant_scale(1, 0.1f);
    std::vector<float> quant_offset(1, 0.0f);

    for (int64_t i = 0; i < x_size; i++) {
        x[i] = static_cast<int8_t>(i % 127);
    }

    void* x_dev = nullptr;
    void* ws_dev = nullptr;
    void* qs_dev = nullptr;
    void* qo_dev = nullptr;
    void* y_dev = nullptr;
    void* scale_dev = nullptr;

    aclrtMalloc(&x_dev, x_size * sizeof(int8_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&ws_dev, weight_scale_size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&qs_dev, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&qo_dev, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&y_dev, y_size * sizeof(int8_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&scale_dev, scale_size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(x_dev, x_size * sizeof(int8_t), x.data(), x_size * sizeof(int8_t), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(ws_dev, weight_scale_size * sizeof(float), weight_scale.data(), weight_scale_size * sizeof(float),
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(qs_dev, sizeof(float), quant_scale.data(), sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(qo_dev, sizeof(float), quant_offset.data(), sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

    aclrtFree(x_dev);
    aclrtFree(ws_dev);
    aclrtFree(qs_dev);
    aclrtFree(qo_dev);
    aclrtFree(y_dev);
    aclrtFree(scale_dev);
}
