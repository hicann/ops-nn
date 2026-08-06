/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_apply_adam_w.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

#include "../../../op_kernel/apply_adam_w.cpp"

using namespace std;

extern "C" __global__ __aicore__ void apply_adam_w(GM_ADDR var, GM_ADDR m, GM_ADDR v, GM_ADDR beta1_power,
                                                   GM_ADDR beta2_power, GM_ADDR lr, GM_ADDR weight_decay, GM_ADDR beta1,
                                                   GM_ADDR beta2, GM_ADDR epsilon, GM_ADDR grad, GM_ADDR max_grad_norm,
                                                   GM_ADDR var_out, GM_ADDR m_out, GM_ADDR v_out, GM_ADDR workspace,
                                                   GM_ADDR tiling);

class ApplyAdamWTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "apply_adam_w_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./apply_adam_w_data/");
    }
    static void TearDownTestCase() { std::cout << "apply_adam_w_test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string ApplyAdamWTest::rootPath = "../../../../experimental/";
const std::string ApplyAdamWTest::dataPath = rootPath + "optim/apply_adam_w/tests/ut/op_kernel/apply_adam_w_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

TEST_F(ApplyAdamWTest, test_case_float32_1)
{
    uint32_t blockDim = 1;
    // 1. 调用 Python 脚本生成输入和 Golden 数据
    system("cd ./apply_adam_w_data/ && python3 golden.py '2, 2' 'float32'");

    // shape是(2,2)，所以数据总数是 4
    uint32_t dataCount = 4;
    size_t inputByteSize = dataCount * sizeof(float);
    size_t scalarByteSize = 1 * sizeof(float);

    // 2. 准备各种变量的 bin 路径
    std::string var_fileName = "./apply_adam_w_data/input_var_float32_apply_adam_w.bin";
    std::string m_fileName = "./apply_adam_w_data/input_m_float32_apply_adam_w.bin";
    std::string v_fileName = "./apply_adam_w_data/input_v_float32_apply_adam_w.bin";
    std::string grad_fileName = "./apply_adam_w_data/input_grad_float32_apply_adam_w.bin";
    std::string max_grad_norm_fileName = "./apply_adam_w_data/input_max_grad_norm_float32_apply_adam_w.bin";

    std::string beta1_power_fileName = "./apply_adam_w_data/input_beta1_power_float32_apply_adam_w.bin";
    std::string beta2_power_fileName = "./apply_adam_w_data/input_beta2_power_float32_apply_adam_w.bin";
    std::string lr_fileName = "./apply_adam_w_data/input_lr_float32_apply_adam_w.bin";
    std::string weight_decay_fileName = "./apply_adam_w_data/input_weight_decay_float32_apply_adam_w.bin";
    std::string beta1_fileName = "./apply_adam_w_data/input_beta1_float32_apply_adam_w.bin";
    std::string beta2_fileName = "./apply_adam_w_data/input_beta2_float32_apply_adam_w.bin";
    std::string eps_fileName = "./apply_adam_w_data/input_eps_float32_apply_adam_w.bin";

    // 3. 为所有入参分配 Device (Global Memory) 空间并读取数据
    uint8_t* var = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 256));
    ReadFile(var_fileName, inputByteSize, var, inputByteSize);
    uint8_t* m = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 256));
    ReadFile(m_fileName, inputByteSize, m, inputByteSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 256));
    ReadFile(v_fileName, inputByteSize, v, inputByteSize);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 256));
    ReadFile(grad_fileName, inputByteSize, grad, inputByteSize);
    uint8_t* max_grad_norm = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 256));
    ReadFile(max_grad_norm_fileName, inputByteSize, max_grad_norm, inputByteSize);

    uint8_t* beta1_power = (uint8_t*)AscendC::GmAlloc(CeilAlign(scalarByteSize, 256));
    ReadFile(beta1_power_fileName, scalarByteSize, beta1_power, scalarByteSize);
    uint8_t* beta2_power = (uint8_t*)AscendC::GmAlloc(CeilAlign(scalarByteSize, 256));
    ReadFile(beta2_power_fileName, scalarByteSize, beta2_power, scalarByteSize);
    uint8_t* lr = (uint8_t*)AscendC::GmAlloc(CeilAlign(scalarByteSize, 256));
    ReadFile(lr_fileName, scalarByteSize, lr, scalarByteSize);
    uint8_t* weight_decay = (uint8_t*)AscendC::GmAlloc(CeilAlign(scalarByteSize, 256));
    ReadFile(weight_decay_fileName, scalarByteSize, weight_decay, scalarByteSize);
    uint8_t* beta1 = (uint8_t*)AscendC::GmAlloc(CeilAlign(scalarByteSize, 256));
    ReadFile(beta1_fileName, scalarByteSize, beta1, scalarByteSize);
    uint8_t* beta2 = (uint8_t*)AscendC::GmAlloc(CeilAlign(scalarByteSize, 256));
    ReadFile(beta2_fileName, scalarByteSize, beta2, scalarByteSize);
    uint8_t* epsilon = (uint8_t*)AscendC::GmAlloc(CeilAlign(scalarByteSize, 256));
    ReadFile(eps_fileName, scalarByteSize, epsilon, scalarByteSize);

    // 4. 为出参分配空间
    size_t outputByteSize = dataCount * sizeof(float);
    uint8_t* var_out = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 256));
    uint8_t* m_out = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 256));
    uint8_t* v_out = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 256));

    // 5. 分配 Workspace 和 Tiling
    size_t workspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ApplyAdamWTilingData));

    ApplyAdamWTilingData* tilingData = reinterpret_cast<ApplyAdamWTilingData*>(tiling);

    // 设置切分 Tiling 数据
    tilingData->smallCoreDataNum = 64;
    tilingData->bigCoreDataNum = 128;
    tilingData->tileDataNum = 1024;
    tilingData->smallTailDataNum = 64;
    tilingData->bigTailDataNum = 128;
    tilingData->finalSmallTileNum = 1;
    tilingData->finalBigTileNum = 1;
    tilingData->tailBlockNum = 0;

    tilingData->amsgrad = 1;
    tilingData->maximize = 1;

    // 6. 运行仿真并下发 Kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    // 这里的参数列表顺序严格对应你 kernel_operator.h 中声明的 apply_adam_w
    ICPU_RUN_KF(apply_adam_w<0>, blockDim, var, m, v, beta1_power, beta2_power, lr, weight_decay, beta1, beta2, epsilon,
                grad, max_grad_norm, var_out, m_out, v_out, workspace, (uint8_t*)(tilingData));

    // 7. 保存真实的计算结果 (Host)
    std::string var_out_fileName = "./apply_adam_w_data/output_var_float32_apply_adam_w.bin";
    WriteFile(var_out_fileName, var_out, outputByteSize);
    std::string m_out_fileName = "./apply_adam_w_data/output_m_float32_apply_adam_w.bin";
    WriteFile(m_out_fileName, m_out, outputByteSize);
    std::string v_out_fileName = "./apply_adam_w_data/output_v_float32_apply_adam_w.bin";
    WriteFile(v_out_fileName, v_out, outputByteSize);

    // 8. 释放全部内存资源，防止内存泄露 (Memory Leak)
    AscendC::GmFree((void*)var);
    AscendC::GmFree((void*)m);
    AscendC::GmFree((void*)v);
    AscendC::GmFree((void*)grad);
    AscendC::GmFree((void*)max_grad_norm);
    AscendC::GmFree((void*)beta1_power);
    AscendC::GmFree((void*)beta2_power);
    AscendC::GmFree((void*)lr);
    AscendC::GmFree((void*)weight_decay);
    AscendC::GmFree((void*)beta1);
    AscendC::GmFree((void*)beta2);
    AscendC::GmFree((void*)epsilon);
    AscendC::GmFree((void*)var_out);
    AscendC::GmFree((void*)m_out);
    AscendC::GmFree((void*)v_out);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./apply_adam_w_data/ && python3 compare_data.py 'float32'");
}
