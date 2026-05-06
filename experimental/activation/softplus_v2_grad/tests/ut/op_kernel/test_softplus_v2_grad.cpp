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
 * \file test_softplus_v2_grad.cpp
 * \brief
 */

#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "data_utils.h"
#include "gtest/gtest.h"
#include "tikicpulib.h"

// 引入SoftplusV2Grad核函数实现
#include "../../../op_kernel/softplus_v2_grad.cpp"

using namespace std;

// 1. 严格匹配kernel侧的Tiling数据结构（完整字段）
struct SoftplusV2GradTilingData {
    uint32_t smallCoreDataNum;
    uint32_t bigCoreDataNum;
    uint32_t finalBigTileNum;
    uint32_t finalSmallTileNum;
    uint32_t tileDataNum;
    uint32_t smallTailDataNum;
    uint32_t bigTailDataNum;
    uint32_t tailBlockNum;
    uint32_t bigprocessDataNum_computes;
    uint32_t smallprocessDataNum_computes;
    uint32_t tailbigprocessDataNum_computes;
    uint32_t tailsmallprocessDataNum_computes;
    uint32_t dataType;    // 0=float16, 1=float32, 2=bfloat16
    uint32_t typeLength;  // 类型字节长度：float16=2, float32=4, bfloat16=2
    float beta;           // Softplus beta参数
    float threshold;      // Softplus阈值参数
};

// 2. 核函数声明（适配Tiling结构体，移除独立的beta/threshold GM参数）
extern "C" __global__ __aicore__ void softplus_v2_grad(GM_ADDR gradOutput,  // 反向输入梯度（GM地址）
                                                       GM_ADDR inputX,      // Softplus输入x（GM地址）
                                                       GM_ADDR y,  // 输出梯度（GM地址，用于和golden对比）
                                                       GM_ADDR workspace,  // 工作空间（GM地址）
                                                       GM_ADDR tiling  // 分片数据（包含beta/threshold/类型等信息）
);

// 测试类（修正拼写+路径）
class SoftplusV2GradTest : public testing::Test {
   protected:
    static void SetUpTestCase() {
        std::cout << "softplus_v2_grad_test SetUp" << std::endl;
        // 拷贝测试数据目录
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./softplus_v2_grad_data/");
    }
    static void TearDownTestCase() {
        std::cout << "softplus_v2_grad TearDown" << std::endl;
        // 清理生成的二进制文件
        system("rm -rf ./softplus_v2_grad_data/*.bin");
    }

   private:
    // 静态路径配置
    const static std::string rootPath;
    const static std::string dataPath;
};

// 静态成员初始化（修正类名+路径）
const std::string SoftplusV2GradTest::rootPath = "../../../../";
const std::string SoftplusV2GradTest::dataPath =
    rootPath + "activation/softplus_v2_grad/tests/ut/op_kernel/softplus_v2_grad_data";

// 通用对齐函数（保留）
template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b) {
    return (a + b - 1) / b * b;
}

// 数据类型映射工具函数（匹配Tiling的dataType字段）
uint32_t GetDataTypeEnum(const std::string& dtype) {
    if (dtype == "float16") return 0;
    if (dtype == "float32") return 1;
    if (dtype == "bfloat16") return 2;
    throw runtime_error("Unsupported data type: " + dtype);
}

// 类型长度获取函数（匹配Tiling的typeLength字段）
uint32_t GetTypeLength(const std::string& dtype) {
    if (dtype == "float16" || dtype == "bfloat16") return 2;
    if (dtype == "float32") return 4;
    throw runtime_error("Unsupported data type: " + dtype);
}

// 核心测试用例（适配float16 + 指定Tiling结构体）
TEST_F(SoftplusV2GradTest, test_case_float16_1) {
    // ========== 1. 基础参数配置 ==========
    const std::string dtype = "float16";
    const float beta_val = 2.0f;             // 对应Python脚本的beta参数
    const float threshold_val = 10.0f;       // 对应Python脚本的threshold参数
    const uint32_t blockDim = 1;             // 核函数Block维度（按需调整）
    const uint32_t dim0 = 128, dim1 = 64;    // 数据形状
    const uint32_t dataCount = dim0 * dim1;  // 总数据量

    // ========== 2. 生成测试数据（调用Python脚本） ==========
    std::string gen_cmd = "cd ./softplus_v2_grad_data/ && python3 gen_data.py '(" + to_string(dim0) + "," +
                          to_string(dim1) + ")' '" + dtype + "' " + to_string(beta_val) + " " +
                          to_string(threshold_val);
    system(gen_cmd.c_str());

    // ========== 3. 文件路径配置 ==========
    std::string gradOutputFile = "./softplus_v2_grad_data/" + dtype + "_gradOutput_softplus_backward.bin";
    std::string inputXFile = "./softplus_v2_grad_data/" + dtype + "_input_x_softplus_backward.bin";
    std::string outputFile = "./softplus_v2_grad_data/" + dtype + "_output_x_grad_softplus_backward.bin";
    std::string goldenFile = "./softplus_v2_grad_data/" + dtype + "_golden_x_grad_softplus_backward.bin";

    // ========== 4. 内存大小计算（按32字节对齐） ==========
    const uint32_t type_len = GetTypeLength(dtype);
    size_t gradOutputByteSize = dataCount * type_len;
    size_t inputXByteSize = dataCount * type_len;
    size_t outputByteSize = dataCount * type_len;

    // 对齐后的内存大小（昇腾GM内存要求32字节对齐）
    size_t aligned_gradOutputSize = CeilAlign(gradOutputByteSize, 32);
    size_t aligned_inputXSize = CeilAlign(inputXByteSize, 32);
    size_t aligned_outputSize = CeilAlign(outputByteSize, 32);
    size_t workspaceSize = 32 * 1024 * 1024;  // 32MB工作空间（按需调整）

    // ========== 5. GM内存分配 ==========
    uint8_t* gradOutput = (uint8_t*)AscendC::GmAlloc(aligned_gradOutputSize);
    uint8_t* inputX = (uint8_t*)AscendC::GmAlloc(aligned_inputXSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(aligned_outputSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling_buf = (uint8_t*)AscendC::GmAlloc(sizeof(SoftplusV2GradTilingData));

    // 内存分配校验
    if (!gradOutput || !inputX || !y || !workspace || !tiling_buf) {
        FAIL() << "GM memory allocation failed!";
        return;
    }

    // ========== 6. 读取输入数据到GM内存 ==========
    ReadFile(gradOutputFile, gradOutputByteSize, gradOutput, gradOutputByteSize);
    ReadFile(inputXFile, inputXByteSize, inputX, inputXByteSize);

    // ========== 7. 配置Tiling数据（完整填充所有字段） ==========
    SoftplusV2GradTilingData* tilingData = reinterpret_cast<SoftplusV2GradTilingData*>(tiling_buf);
    // 核心分片参数（示例值，需和kernel侧逻辑匹配）
    tilingData->smallCoreDataNum = 8192;               // 小核处理数据量
    tilingData->bigCoreDataNum = 8208;                 // 大核处理数据量
    tilingData->finalBigTileNum = 1;                   // 最终大核Tile数
    tilingData->finalSmallTileNum = 1;                 // 最终小核Tile数
    tilingData->tileDataNum = 11760;                   // 每个Tile的数据量
    tilingData->smallTailDataNum = 8192;               // 小核尾数据量
    tilingData->bigTailDataNum = 8208;                 // 大核尾数据量
    tilingData->tailBlockNum = 0;                      // 尾Block数
    tilingData->bigprocessDataNum_computes = 8208;     // 大核计算数据量
    tilingData->smallprocessDataNum_computes = 8192;   // 小核计算数据量
    tilingData->tailbigprocessDataNum_computes = 0;    // 大核尾计算数据量
    tilingData->tailsmallprocessDataNum_computes = 0;  // 小核尾计算数据量
    // 类型+参数配置
    tilingData->dataType = GetDataTypeEnum(dtype);  // float16=0
    tilingData->typeLength = type_len;              // float16=2
    tilingData->beta = beta_val;                    // 2.0
    tilingData->threshold = threshold_val;          // 10.0

    // ========== 8. 核函数执行 ==========
    AscendC::SetKernelMode(KernelMode::AIV_MODE);  // 设置AI Core运行模式
    // 调用核函数（参数顺序：gradOutput, inputX, y, workspace, tiling）
    ICPU_RUN_KF(softplus_v2_grad, blockDim, gradOutput, inputX, y, workspace, tiling_buf);

    // ========== 9. 输出结果写入文件（用于对比） ==========
    WriteFile(outputFile, y, outputByteSize);

    // ========== 10. 内存释放 ==========
    AscendC::GmFree(gradOutput);
    AscendC::GmFree(inputX);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling_buf);

    // ========== 11. 调用对比脚本验证结果 ==========
    std::string cmp_cmd = "cd ./softplus_v2_grad_data/ && python3 compare_data.py '" + dtype + "'";
    int cmp_ret = system(cmp_cmd.c_str());
    EXPECT_EQ(cmp_ret, 0) << "Golden data comparison failed!";
}