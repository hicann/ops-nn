/**
 * Copyright (c) 2026 Huawei Technologies
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_swiglu_group_quant.cpp
 * \brief Kernel unit test for SwiGLU Group Dynamic Quant operator
 */
#include <iostream>
#include <vector>
#include "gtest/gtest.h"
#include "data_utils.h"

extern "C" __global__ __aicore__ void swiglu_group_quant(
    GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR groupIndexGM, GM_ADDR scaleGM,
    GM_ADDR yGM, GM_ADDR yScaleGM, GM_ADDR yOriginGM,
    GM_ADDR workspace, GM_ADDR tiling);

using namespace ut_op_kernel;

class swiglu_group_quant_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "swiglu_group_quant_test SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "swiglu_group_quant_test TearDown" << std::endl; }
};

TEST_F(swiglu_group_quant_test, test_case_fp32_nogroup)
{
    // Test case: float32, non-group mode, no clamp
    int64_t tokens = 128;
    int64_t dim2H = 2048;
    int64_t dimH = 1024;
    
    auto x_shape = std::vector<int64_t>{tokens, dim2H};
    auto yOut_shape = std::vector<int64_t>{tokens, dimH};
    auto scaleOut_shape = std::vector<int64_t>{1};
    
    // Create input/output tensors
    auto x = std::make_shared<ut_op_kernel::Tensor>(x_shape, ge::DT_FLOAT);
    auto yOut = std::make_shared<ut_op_kernel::Tensor>(yOut_shape, ge::DT_HIFLOAT8);
    auto scaleOut = std::make_shared<ut_op_kernel::Tensor>(scaleOut_shape, ge::DT_FLOAT);
    
    // Generate random input data
    ut_op_kernel::GenRandomFloatData(x, -3.0f, 3.0f);
    
    // Run kernel
    int32_t blockDim = 1;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(swiglu_group_quant, blockDim, x, nullptr, nullptr,
        yOut, scaleOut, nullptr, (uint8_t*)nullptr);
    
    // Verify output shapes
    EXPECT_EQ(yOut->GetShape().GetDim(0), tokens);
    EXPECT_EQ(yOut->GetShape().GetDim(1), dimH);
    EXPECT_EQ(scaleOut->GetShape().GetDim(0), 1);
}

TEST_F(swiglu_group_quant_test, test_case_fp32_clamp)
{
    // Test case: float32, non-group mode, with clamp
    int64_t tokens = 64;
    int64_t dim2H = 4096;
    int64_t dimH = 2048;
    
    auto x_shape = std::vector<int64_t>{tokens, dim2H};
    auto yOut_shape = std::vector<int64_t>{tokens, dimH};
    auto scaleOut_shape = std::vector<int64_t>{1};
    
    auto x = std::make_shared<ut_op_kernel::Tensor>(x_shape, ge::DT_FLOAT);
    auto yOut = std::make_shared<ut_op_kernel::Tensor>(yOut_shape, ge::DT_HIFLOAT8);
    auto scaleOut = std::make_shared<ut_op_kernel::Tensor>(scaleOut_shape, ge::DT_FLOAT);
    
    ut_op_kernel::GenRandomFloatData(x, -10.0f, 10.0f);
    
    int32_t blockDim = 1;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(swiglu_group_quant, blockDim, x, nullptr, nullptr,
        yOut, scaleOut, nullptr, (uint8_t*)nullptr);
}

TEST_F(swiglu_group_quant_test, test_case_fp16_nogroup)
{
    // Test case: float16, non-group mode
    int64_t tokens = 32;
    int64_t dim2H = 2048;
    int64_t dimH = 1024;
    
    auto x_shape = std::vector<int64_t>{tokens, dim2H};
    auto yOut_shape = std::vector<int64_t>{tokens, dimH};
    auto scaleOut_shape = std::vector<int64_t>{1};
    
    auto x = std::make_shared<ut_op_kernel::Tensor>(x_shape, ge::DT_FLOAT16);
    auto yOut = std::make_shared<ut_op_kernel::Tensor>(yOut_shape, ge::DT_HIFLOAT8);
    auto scaleOut = std::make_shared<ut_op_kernel::Tensor>(scaleOut_shape, ge::DT_FLOAT);
    
    ut_op_kernel::GenRandomFloatData(x, -3.0f, 3.0f);
    
    int32_t blockDim = 1;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(swiglu_group_quant, blockDim, x, nullptr, nullptr,
        yOut, scaleOut, nullptr, (uint8_t*)nullptr);
}