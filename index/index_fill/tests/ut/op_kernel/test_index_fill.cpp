/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <array>
#include <vector>
#include <iostream>
#include <string>

#include "gtest/gtest.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "index_fill_tiling_def.h"
#endif

using namespace std;

extern "C" __global__ __aicore__ void index_fill(GM_ADDR x, GM_ADDR indices, GM_ADDR value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

// 结构化测试参数，仅包含变化项
struct IndexFillTestParams {
    int P;                  // 维度P
    int N;                  // 维度N
    int Q;                  // 维度Q（0表示二维场景）
    int indicesNum;         // 索引数量
    int indicesProcessMode; // 索引处理模式
    // Tiling相关参数
    int frontCoreNumTaskIndices;
    int tailCoreNumTaskIndices;
    int frontCoreDataTaskIndices;
    int tailCoreDataTaskIndices;
};

class index_fill_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "index_fill_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "index_fill_test TearDown\n" << endl; }

    // 通用测试执行函数（提取所有重复逻辑）
    static void RunTest(const IndexFillTestParams& params, int tilingKey) {
        // 固定常量定义
        const size_t kSysWorkspaceSize = 16 * 1024 * 1024 + 4096;
        const size_t kUbSize = 196352;
        const int kCoreNum = 48;
        const size_t kValueSize = 1 * sizeof(int32_t);

        // 计算数据尺寸（Q=0时为二维：P*N，否则为三维：P*N*Q）
        size_t dataDim3 = (params.Q == 0) ? 1 : params.Q;
        size_t dataSize = params.P * params.N * dataDim3 * sizeof(float);
        size_t indicesSize = params.indicesNum * sizeof(int32_t);

        // 设备内存分配
        uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(kSysWorkspaceSize);
        uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(IndexFillTilingData));
        uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(dataSize);
        uint8_t* indicesGM = (uint8_t*)AscendC::GmAlloc(indicesSize);
        uint8_t* valueGM = (uint8_t*)AscendC::GmAlloc(kValueSize);
        uint8_t* yGM = (uint8_t*)AscendC::GmAlloc(dataSize);

        // 内存初始化
        memset(xGM, 0, dataSize);
        memset(indicesGM, 0, indicesSize);
        memset(valueGM, 0, kValueSize);
        memset(yGM, 0, dataSize);

        // 配置Tiling参数
        IndexFillTilingData* tilingData = reinterpret_cast<IndexFillTilingData*>(tiling);
        tilingData->coreNum = kCoreNum;
        tilingData->N = params.N;
        tilingData->indicesNum = params.indicesNum;
        tilingData->indicesProcessMode = params.indicesProcessMode;
        tilingData->frontCoreNumTaskIndices = params.frontCoreNumTaskIndices;
        tilingData->tailCoreNumTaskIndices = params.tailCoreNumTaskIndices;
        tilingData->frontCoreDataTaskIndices = params.frontCoreDataTaskIndices;
        tilingData->tailCoreDataTaskIndices = params.tailCoreDataTaskIndices;
        tilingData->ubSize = kUbSize;
        tilingData->P = params.P;
        tilingData->Q = params.Q;

        // 执行Kernel
        ICPU_SET_TILING_KEY(tilingKey);
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ICPU_RUN_KF(index_fill, kCoreNum, xGM, indicesGM, valueGM, yGM, workspace, tiling);

        // 内存释放（顺序与分配相反）
        AscendC::GmFree(yGM);
        AscendC::GmFree(valueGM);
        AscendC::GmFree(indicesGM);
        AscendC::GmFree(xGM);
        AscendC::GmFree(tiling);
        AscendC::GmFree(workspace);
    }
};

// 测试用例简化为：仅配置参数+调用通用函数
TEST_F(index_fill_test, index_fill_kernel_test_key_0) {
    IndexFillTestParams params = {
        .P = 100, .N = 7, .Q = 89567,
        .indicesNum = 18, .indicesProcessMode = 0,
        .frontCoreNumTaskIndices = 18, .tailCoreNumTaskIndices = 0,
        .frontCoreDataTaskIndices = 1, .tailCoreDataTaskIndices = 0
    };
    RunTest(params, 0);
}

TEST_F(index_fill_test, index_fill_kernel_test_key_1) {
    IndexFillTestParams params = {
        .P = 12, .N = 640, .Q = 128,
        .indicesNum = 600, .indicesProcessMode = 0,
        .frontCoreNumTaskIndices = 24, .tailCoreNumTaskIndices = 24,
        .frontCoreDataTaskIndices = 13, .tailCoreDataTaskIndices = 12
    };
    RunTest(params, 1);
}

TEST_F(index_fill_test, index_fill_kernel_test_key_2) {
    IndexFillTestParams params = {
        .P = 1000, .N = 32, .Q = 32,
        .indicesNum = 40, .indicesProcessMode = 0,
        .frontCoreNumTaskIndices = 40, .tailCoreNumTaskIndices = 0,
        .frontCoreDataTaskIndices = 1, .tailCoreDataTaskIndices = 0
    };
    RunTest(params, 2);
}

TEST_F(index_fill_test, index_fill_kernel_test_key_3) {
    IndexFillTestParams params = {
        .P = 12, .N = 122880, .Q = 0,
        .indicesNum = 12280, .indicesProcessMode = 1,
        .frontCoreNumTaskIndices = 48, .tailCoreNumTaskIndices = 0,
        .frontCoreDataTaskIndices = 2560, .tailCoreDataTaskIndices = 0
    };
    RunTest(params, 3);
}

TEST_F(index_fill_test, index_fill_kernel_test_key_4) {
    IndexFillTestParams params = {
        .P = 1200, .N = 128, .Q = 0,
        .indicesNum = 128, .indicesProcessMode = 0,
        .frontCoreNumTaskIndices = 32, .tailCoreNumTaskIndices = 16,
        .frontCoreDataTaskIndices = 3, .tailCoreDataTaskIndices = 2
    };
    RunTest(params, 4);
}