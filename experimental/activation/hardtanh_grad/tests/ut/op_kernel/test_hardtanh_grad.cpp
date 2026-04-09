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
#include <random>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif
#include "../../../op_kernel/hardtanh_grad.cpp"
#include "../../../op_kernel/hardtanh_grad_tiling_data.h"
#include <cstdint>

using namespace std;

class hardtanh_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "hardtanh_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "hardtanh_grad_test TearDown\n" << endl;
    }

    void GenerateHardTanhGradGoldenData(
        float* input_x1,
        float* input_x2,
        float* golden,
        int data_length,
        int seed = 42,
        float x1_min = -10.0f,
        float x1_max = 0.0f,
        float x2_min = 0.0f,
        float x2_max = 10.0f,
        float threshold_min = -1.0f,
        float threshold_max = 1.0f
    ) {
        // 创建随机数生成器
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist_x1(x1_min, x1_max);
        std::uniform_real_distribution<float> dist_x2(x2_min, x2_max);
        
        // 生成随机数据
        for (int i = 0; i < data_length; ++i) {
            input_x1[i] = dist_x1(rng);
            input_x2[i] = dist_x2(rng);
            
            // 计算hardtanh_grad标杆数据
            if (input_x1[i] >= threshold_min && input_x1[i] <= threshold_max) {
                golden[i] = input_x2[i];  // 在范围内，保留梯度
            } else {
                golden[i] = 0.0f;         // 范围外，梯度为0
            }
        }
    }    

    bool verify_vectors(float* output, float* golden, int size, 
                    float error_tol = 1e-4f, float rtol = 1e-4f, float atol = 1e-4f) {
        int error_count = 0;
        
        for (int i = 0; i < size; i++) {
            float abs_error = fabsf(output[i] - golden[i]);
            float threshold = atol + rtol * fabsf(golden[i]);
            
            if (abs_error > threshold) {
                error_count++;
                
                if (error_count <= 100) {  // 最多显示100个错误
                    float rel_error = 0.0f;
                    if (golden[i] != 0.0f) {
                        rel_error = abs_error / fabsf(golden[i]);
                    } else if (output[i] != 0.0f) {
                        rel_error = abs_error / fabsf(output[i]);
                    }
                    
                    std::cout << "data index: " << std::setw(6) << std::setfill('0') << i
                            << ", expected: " << std::fixed << std::setprecision(9) << golden[i]
                            << ", actual: " << output[i]
                            << ", abs_err: " << abs_error
                            << ", rel_err: " << rel_error << std::endl;
                }
            }
        }
        
        float error_ratio = static_cast<float>(error_count) / size;
        std::cout << "error ratio: " << std::fixed << std::setprecision(4) << error_ratio
                << ", tolerance: " << error_tol << std::endl;
        
        return error_ratio <= error_tol;
    }

};

TEST_F(hardtanh_grad_test, test_case_0)
{
    size_t inputNum = 128;
    size_t gradOutputByteSize = inputNum * sizeof(float);
    size_t selfByteSize = inputNum * sizeof(float);
    size_t outByteSize = inputNum * sizeof(float);
    size_t tiling_data_size = sizeof(HardtanhGradTilingData);
    uint32_t blockDim = 1;

    uint8_t* gradOutput = (uint8_t*)AscendC::GmAlloc(gradOutputByteSize);
    uint8_t* self = (uint8_t*)AscendC::GmAlloc(selfByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);

    uint8_t* golden = (uint8_t*)AscendC::GmAlloc(outByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    float min_val = -1.0;
    float max_val = 1.0;

    // 2. 生成标杆数据
    float* gradOutput_float = reinterpret_cast<float*>(gradOutput);
    float* self_float = reinterpret_cast<float*>(self);
    float* golden_float = reinterpret_cast<float*>(golden);

    GenerateHardTanhGradGoldenData(gradOutput_float, self_float, golden_float, inputNum, 42, -10.0f, 10.0f, -2.0f, 2.0f, min_val, max_val);

    HardtanhGradTilingData* tilingDatafromBin = reinterpret_cast<HardtanhGradTilingData*>(tiling);

    tilingDatafromBin->smallCoreDataNum = 0;
    tilingDatafromBin->bigCoreDataNum = 0;
    tilingDatafromBin->finalBigTileNum = 0;
    tilingDatafromBin->finalSmallTileNum = 0;

    tilingDatafromBin->tileDataNum = 128;
    tilingDatafromBin->smallTailDataNum = 128;
    tilingDatafromBin->bigTailDataNum = 0;
    tilingDatafromBin->tailBlockNum = 0;
    tilingDatafromBin->maskTileDataNum = 256;
    tilingDatafromBin->max = max_val;
    tilingDatafromBin->min = min_val;

    auto HardtanhGradKernel = [](GM_ADDR gradOutput, GM_ADDR self, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
        ::hardtanh_grad<0>(gradOutput, self, out, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(HardtanhGradKernel,
        blockDim,
        gradOutput,
        self,
        out,
        workspace,
        (uint8_t *)(tilingDatafromBin));

    // 计算
    // bool ret = SomeFunction();
    float* out_float = reinterpret_cast<float*>(out);
    bool passed = verify_vectors(golden_float, out_float, inputNum, 1e-4f);

    AscendC::GmFree(gradOutput);
    AscendC::GmFree(self);
    AscendC::GmFree(out);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);


    EXPECT_TRUE(passed) << "精度测试失败";
    // EXPECT_EQ(ret, true);
}